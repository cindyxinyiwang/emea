# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
from .integrations import (  # isort: split
    default_hp_search_backend,
    hp_params,
    is_azureml_available,
    is_comet_available,
    is_mlflow_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from .trainer import Trainer
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook, is_torch_tpu_available
from .modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
)
from .training_args import TrainingArguments
from .utils import logging
from torch.distributions.beta import Beta

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from .file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from .integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from .integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from .integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_mlflow_available():
    from .integrations import MLflowCallback

    DEFAULT_CALLBACKS.append(MLflowCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

if is_azureml_available():
    from .integrations import AzureMLCallback

    DEFAULT_CALLBACKS.append(AzureMLCallback)

logger = logging.get_logger(__name__)

def get_lambda_(batch_size, tau):
    dist = Beta(tau, tau)
    lambda_ = dist.sample(sample_shape=[batch_size])
    lambda_ = torch.max(lambda_, 1-lambda_)
    return lambda_

class TrainerMixup(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        meta_train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        do_save_full_model: bool = True,
        do_save_adapters: bool = False,
        do_save_adapter_fusion: bool = False,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        augment_data_collator: Optional[DataCollator] = None,
        mixup_tau = 0.1, 
        **kwargs,
    ):
        Trainer.__init__(self, model, args, data_collator, train_dataset, meta_train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, do_save_full_model, do_save_adapters, do_save_adapter_fusion, adapter_names, optimizers, **kwargs)
        self.augment_data_collator = augment_data_collator
        self.train_data_collator = train_data_collator
        self.mixup_tau = mixup_tau

    def _get_meta_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.meta_train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.meta_train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.meta_train_dataset)
        else:
            return (
                RandomSampler(self.meta_train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.meta_train_dataset)
            )

    def get_meta_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.meta_train_dataset is None:
            raise ValueError("Trainer: training requires a meta_train_dataset.")
        train_sampler = self._get_meta_train_sampler()

        return DataLoader(
            self.meta_train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], meta_inputs: Dict[str, Union[torch.Tensor, Any]] = None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        train_inputs = {"input_ids": inputs["input_ids"], "labels": inputs["labels"]}
        meta_inputs = {"input_ids": inputs["meta_input_ids"], "labels": inputs["meta_labels"]}
        meta_outputs = model(**meta_inputs)
        #train_outputs = model(**train_inputs)
       
        emb = model.get_input_embeddings()
        train_emb = emb(train_inputs["input_ids"])
        meta_emb = emb(meta_inputs["input_ids"])
        bsize = train_emb.size(0)
        lambda_ = get_lambda_(bsize, self.mixup_tau).to(meta_emb.device).view(-1, 1, 1)
        mix_inputs = {"inputs_embeds": lambda_*meta_emb + (1-lambda_)*train_emb}
        mix_outputs = model(**mix_inputs)
        mix_pred_score = mix_outputs[0].view(-1, model.config.vocab_size) 

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # -100 index = padding token
        mix_loss = loss_fct(mix_pred_score, train_inputs["labels"].view(-1)).view(bsize, -1)
        mix_loss_mask = (train_inputs["labels"] != -100).float()
        mix_loss = mix_loss*mix_loss_mask*(1-lambda_.squeeze(-1))
        mix_loss = mix_loss.sum() / mix_loss_mask.sum()

        meta_mix_loss = loss_fct(mix_pred_score, meta_inputs["labels"].view(-1)).view(bsize, -1)
        meta_mix_loss_mask = (meta_inputs["labels"] != -100).float()
        meta_mix_loss = meta_mix_loss*meta_mix_loss_mask*(lambda_.squeeze(-1))
        meta_mix_loss = meta_mix_loss.sum() / meta_mix_loss_mask.sum()


        #loss = train_outputs[0] + meta_outputs[0] + mix_loss
        loss = 0.5*meta_outputs[0] + 0.5*mix_loss
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return loss



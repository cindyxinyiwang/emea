# Efficient Test Time Adapter Ensembling for Low-resource Language Varieties

This repository contains the implementation for [our paper](https://arxiv.org/abs/2109.04877).

``
Efficient Test Time Adapter Ensembling for Low-resource Language Varieties

Xinyi Wang, Yulia Tsvetkov,Sebastian Ruder, Graham Neubig

EMNLP 2021 Findings
``

Our code is based on the [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers/tree/master/src/transformers/adapters) codebase and the [XTREME](https://github.com/google-research/xtreme) benchmark

# Introduction
We find that specialized language adapters might not be robust to unseen language variations, and that utilization of multiple existing pretrained language adapters alleviates this issue. We propose an algorithm named EMEA(Entropy Minimized Ensemble of Language Adapters), which optimizes the ensemble weights of a group of related language adapters at test time for each test input.

# Main method implementation
The main function for optimizing the adapter weighting using EMEA is [here](https://github.com/cindyxinyiwang/emea/blob/main/third_party/run_tag.py#L245).

# Download the data
We simply use the data downloading instruction from the official XTREME repo. We also provide the processed data for NER in data/.

# Installation
To install the dependencies:
``
pip install --editable .
``

## Decoding scripts
EMEA is a test time decoding algorithm. You need to train a task adapter before testing out the different decoding strategies. Here we provide a pretrained NER task adapter in outputs/ner/.

Baseline
``
bash job_scripts/test_panx_adapter.sh
``

Ensemble
``
bash test_panx_adapter_ensemble.sh
``

EMEA-s1
``
bash test_panx_adapter_emea_s1.sh
``

EMEA-s10
``
bash test_panx_adapter_emea_s10.sh
``

## Citation

Please cite our paper as:

```
@inproceedings{wang2021emea,
    title={Efficient Test Time Adapter Ensembling for Low-resource Language Varieties},
    author={Wang, Xinyi and
            Tsvetkov, Yulia and
            Ruder, Sebastian and
            Neubig, Graham},
    booktitle={EMNLP: Findings},
    year={2021}
}
```

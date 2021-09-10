import numpy as np
import torch

def mask_tokens(inputs: torch.Tensor, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, masked_indices


def mask_tokens_sde(inputs: torch.Tensor, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels_ngram = inputs.clone()
    labels = inputs[:, :, 0].clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    pad_mask  = inputs.eq(tokenizer.pad_token_id)
    labels_ngram[~masked_indices] = -100  # We only compute loss on masked tokens
    labels_ngram[pad_mask] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    ## 10% of the time, we replace masked input tokens with random word
    #indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    #inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels_ngram

def mlm_switch_tokens(tokens, tokenizer, pretrained_model, p=0.1, topk=0, return_corrupts=False):
    # get MLM embs
    mlm_inputs_ids, mlm_labels, masked_indices = mask_tokens(tokens, tokenizer, p)
    model_device = next(pretrained_model.parameters()).device
    tokens = tokens.to(model_device)
    mlm_inputs_ids = mlm_inputs_ids.to(model_device)
    mlm_labels = mlm_labels.to(model_device)
    masked_indices = masked_indices.to(model_device)
    #attention_mask = attention_mask.to(model_device)

    mlm_inputs = {"input_ids": mlm_inputs_ids, "labels": mlm_labels}
    #print(mlm_inputs)
    mlm_outputs = pretrained_model.forward(**mlm_inputs)
    mlm_loss = mlm_outputs[0]
    logits =  mlm_outputs[1]
    if topk > 0:
        indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
        logits[indices_to_remove] = -float("inf")

    mlm_probs = torch.softmax(logits, dim=-1)
    sampled_words_indices = torch.distributions.Categorical(mlm_probs).sample()
    sampled_tokens = tokens * (~masked_indices).long() + sampled_words_indices * masked_indices.long()
    if return_corrupts:
        mlm_labels[masked_indices] = sampled_words_indices[masked_indices]
        return sampled_tokens, mlm_inputs_ids, mlm_labels
    return sampled_tokens

def switchout(tokens, mask, tau, tokenizer, vocab_dist=None, vocabs=None):
    # first sample the number of words to corrupt
    unk_token_id = tokenizer.unk_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    max_len = tokens.size(1)

    pad_mask = (tokens == pad_token_id)
    cls_mask = (tokens == cls_token_id)
    sep_mask = (tokens == sep_token_id)
    m_mask = (tokens == mask_token_id)
    sample_mask = ~((~pad_mask) & (~cls_mask) & (~sep_mask) & (~m_mask))

    logits = torch.arange(max_len).float().to(tokens.device)
    lengths = mask.long().sum(dim=-1)
    # 1 for padding, 0 for tokens
    mask = (1-mask).bool()
    logits = logits.mul_(-1).unsqueeze(0).expand_as(tokens).contiguous().masked_fill_(mask, -float('inf'))
    probs = torch.softmax(logits.mul_(tau), dim=-1)
    num_words = torch.distributions.Categorical(probs).sample().float()
    lengths = lengths.float()

    # sample the indices to corrupt
    corrupt_pos = num_words.div_(lengths).unsqueeze(1).expand_as(tokens).contiguous().masked_fill_(sample_mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte().bool()
    total_words = int(corrupt_pos.sum())
    if total_words == 0:
        return tokens
    # sample the corrupts
    if vocab_dist is not None:
      assert vocabs is not None
      # vocab: tensor of allowed vocab [1, vocab_size]
      # vocab_dist: dict of numpy values of sampling probabilities 
      corrupt_words = tokens[corrupt_pos].view(-1)
      corrupt_words_dist = []
      for w in corrupt_words.cpu().numpy():
        if w in vocab_dist:
          corrupt_words_dist.append(vocab_dist[w])
        else:
          corrupt_words_dist.append(vocab_dist[unk_token_id])
      corrupt_words_dist = torch.FloatTensor(corrupt_words_dist).to(tokens.device)
      # num_words x 1
      sampled_words_indices = torch.distributions.Categorical(corrupt_words_dist).sample().view(-1, 1)
      corrupt_val = torch.gather(vocabs.expand(corrupt_words.size(0), -1).to(tokens.device), 1, sampled_words_indices)
      sampled_tokens = tokens.masked_scatter_(corrupt_pos, corrupt_val)
    else:
      corrupt_val = torch.LongTensor(total_words).to(tokens.device)
      corrupts = torch.zeros_like(tokens).long().to(tokens.device)
      corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
      sampled_tokens = tokens.add(corrupts).remainder_(vocab_size).masked_fill_(pad_mask, pad_token_id)
    return sampled_tokens




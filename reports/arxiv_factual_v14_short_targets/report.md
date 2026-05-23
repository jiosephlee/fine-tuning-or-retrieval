# Arxiv Factual v14 Short Targets

Conservative suffix-only target shortening for arxiv factual v13 rows with targets longer than 8 words.

## Summary

domain,reviewed,accepted,rejected_or_invalid
1_58,2,1,1
BOFT,7,6,1
ByteLatent,19,18,1
DPO,9,6,3
FeatLLM,13,12,1
GRPO,8,8,0
GSPO,10,5,5
LongRoPE,8,6,2
OFT,3,3,0
QLoRA,9,9,0
fa3,5,4,1
xLSTM,9,8,1


## Accepted Examples

### 1_58 row 25

- old target words: 12
- new target words: 6
- reason: Leaves a meaningful destination suffix.

Old probe:
```text
According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", during inference, an expense in addition to computation is
```
Old target:
```text
 transferring model parameters from DRAM to the memory of an on-chip accelerator.
```
New probe:
```text
According to the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", during inference, an expense in addition to computation is transferring model parameters from DRAM to
```
New target:
```text
 the memory of an on-chip accelerator.
```

### BOFT row 18

- old target words: 11
- new target words: 8
- reason: Leaves a coherent verb phrase.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", broad progress in the field requires
```
Old target:
```text
 the ability to adapt such models without retraining them from scratch.
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", broad progress in the field requires the ability to
```
New target:
```text
 adapt such models without retraining them from scratch.
```

### BOFT row 107

- old target words: 13
- new target words: 10
- reason: Leaves a meaningful product complement.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", OFT reparameterizes the new weight matrix when finetuning the pretrained weight matrix as
```
Old target:
```text
 the product of a learnable orthogonal matrix and the frozen pretrained weight matrix
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", OFT reparameterizes the new weight matrix when finetuning the pretrained weight matrix as the product of
```
New target:
```text
 a learnable orthogonal matrix and the frozen pretrained weight matrix
```

### BOFT row 188

- old target words: 9
- new target words: 5
- reason: Leaves the referenced base pattern.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", for butterfly components $\tilde{\bm{B}}^b(d,k)$ with $k>2$, the non-zero pattern relative to $\tilde{\bm{B}}^b(d,2)$ is
```
Old target:
```text
 the block-wise permutation of the non-zero pattern of $\tilde{\bm{B}}^b(d,2)$
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", for butterfly components $\tilde{\bm{B}}^b(d,k)$ with $k>2$, the non-zero pattern relative to $\tilde{\bm{B}}^b(d,2)$ is the block-wise permutation of
```
New target:
```text
 the non-zero pattern of $\tilde{\bm{B}}^b(d,2)$
```

### BOFT row 220

- old target words: 12
- new target words: 4
- reason: Leaves a coherent object of approximation.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", although the butterfly structure along with permutations can perfectly recover many classic fast linear transforms such as the fast Fourier transform and Hadamard transform, it is still unknown
```
Old target:
```text
 how well our orthogonal butterfly matrix can approximate a general orthogonal matrix
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", although the butterfly structure along with permutations can perfectly recover many classic fast linear transforms such as the fast Fourier transform and Hadamard transform, it is still unknown how well our orthogonal butterfly matrix can approximate
```
New target:
```text
 a general orthogonal matrix
```

### BOFT row 270

- old target words: 12
- new target words: 7
- reason: Leaves the unknown property.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", what remains unknown about the butterfly network in the authors' information transmission framework is
```
Old target:
```text
 whether the butterfly network is the most efficient way to transmit information
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", what remains unknown about the butterfly network in the authors' information transmission framework is whether the butterfly network is
```
New target:
```text
 the most efficient way to transmit information
```

### BOFT row 276

- old target words: 9
- new target words: 7
- reason: Leaves a coherent verb phrase.

Old probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", finetuning methods usually start with the exact pretrained model such that the finetuned model
```
Old target:
```text
 will not deviate too much from the pretrained one.
```
New probe:
```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", finetuning methods usually start with the exact pretrained model such that the finetuned model will not
```
New target:
```text
 deviate too much from the pretrained one.
```

### ByteLatent row 42

- old target words: 9
- new target words: 6
- reason: Leaves a meaningful determinant.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the computational cost of processing data for both token-based and patch-based models is primarily determined by
```
Old target:
```text
 the number of steps executed by the main Transformer
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the computational cost of processing data for both token-based and patch-based models is primarily determined by the number of
```
New target:
```text
 steps executed by the main Transformer
```

### ByteLatent row 75

- old target words: 9
- new target words: 5
- reason: Leaves the two sides of the tradeoff.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", compared with tokenization-based models, BLT crucially redefines
```
Old target:
```text
 the trade off between the vocabulary size and compute.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", compared with tokenization-based models, BLT crucially redefines the trade off between
```
New target:
```text
 the vocabulary size and compute.
```

### ByteLatent row 76

- old target words: 9
- new target words: 5
- reason: Leaves the two sides of the tradeoff.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", BLT’s crucial improvement over tokenization-based models is said to redefine
```
Old target:
```text
 the trade off between the vocabulary size and compute.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", BLT’s crucial improvement over tokenization-based models is said to redefine the trade off between
```
New target:
```text
 the vocabulary size and compute.
```

### ByteLatent row 89

- old target words: 12
- new target words: 3
- reason: Leaves the dependency source.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the discussion of incremental patching for byte sequences, BPE is not considered an incremental patching scheme, given the property $f_p(\pmb{x}_{<i}) = f_p(\pmb{x})_{<i}$, because
```
Old target:
```text
 the same prefix can be tokenized differently depending on the continuation sequence
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the discussion of incremental patching for byte sequences, BPE is not considered an incremental patching scheme, given the property $f_p(\pmb{x}_{<i}) = f_p(\pmb{x})_{<i}$, because the same prefix can be tokenized differently depending on
```
New target:
```text
 the continuation sequence
```

### ByteLatent row 139

- old target words: 9
- new target words: 3
- reason: Leaves a coherent source representation.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the cross-attention layer in the decoder is applied before the transformer layer in order
```
Old target:
```text
 to first create byte representations from the patch representations
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the cross-attention layer in the decoder is applied before the transformer layer in order to first create byte representations from
```
New target:
```text
 the patch representations
```

### ByteLatent row 178

- old target words: 10
- new target words: 5
- reason: Leaves the cause of memory spikes.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors pad and possibly truncate byte sequences to 12k and 24k bytes respectively for the Llama 2 and BLT-1T datasets during training
```
Old target:
```text
 to avoid memory spikes from sequences with unusually large patches
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors pad and possibly truncate byte sequences to 12k and 24k bytes respectively for the Llama 2 and BLT-1T datasets during training to avoid memory spikes from
```
New target:
```text
 sequences with unusually large patches
```

### ByteLatent row 197

- old target words: 11
- new target words: 5
- reason: Leaves the sequence type distinction.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", whether attention uses a different context length, $m$, when computing attention FLOPs is determined by
```
Old target:
```text
 whether a module is applied on the byte or patch sequence.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", whether attention uses a different context length, $m$, when computing attention FLOPs is determined by whether a module is applied on
```
New target:
```text
 the byte or patch sequence.
```

### ByteLatent row 236

- old target words: 9
- new target words: 7
- reason: Leaves a coherent ratio phrase.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", Llama 3 determines the ratio used for the bpe models as
```
Old target:
```text
 the optimal ratio of model parameters to training data
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", Llama 3 determines the ratio used for the bpe models as the optimal
```
New target:
```text
 ratio of model parameters to training data
```

### ByteLatent row 295

- old target words: 9
- new target words: 5
- reason: Leaves the shifted endpoint.

Old probe:
```text
In the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", at the crossover point for larger flop-class models where {\textsc{BLT}} improves over token-based models, this shifts
```
Old target:
```text
 from 3x down to 2.5x the compute optimal budget.
```
New probe:
```text
In the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", at the crossover point for larger flop-class models where {\textsc{BLT}} improves over token-based models, this shifts from 3x down to
```
New target:
```text
 2.5x the compute optimal budget.
```

### ByteLatent row 349

- old target words: 12
- new target words: 3
- reason: Leaves a coherent capability.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors ensure that all models see the same number of bytes in each sequence during training and inference in expectation in order
```
Old target:
```text
 to prevent any confounding factors from being able to model larger contexts.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors ensure that all models see the same number of bytes in each sequence during training and inference in expectation in order to prevent any confounding factors from being able to
```
New target:
```text
 model larger contexts.
```

### ByteLatent row 360

- old target words: 10
- new target words: 8
- reason: Leaves the pooled representation.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the encoder cross-attention query initialization at the given encoder layer uses
```
Old target:
```text
 pooling of the encoder hidden representation of the patch bytes
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the encoder cross-attention query initialization at the given encoder layer uses pooling of
```
New target:
```text
 the encoder hidden representation of the patch bytes
```

### ByteLatent row 413

- old target words: 11
- new target words: 7
- reason: Leaves the scale dimensions.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", many patching-based approaches demonstrated initial successes at
```
Old target:
```text
 a small scale of model size and number of training bytes.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", many patching-based approaches demonstrated initial successes at a small scale of
```
New target:
```text
 model size and number of training bytes.
```

### ByteLatent row 461

- old target words: 15
- new target words: 3
- reason: Leaves the reason for high cost.

Old probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", tokenization has previously been essential because
```
Old target:
```text
 directly training llms on bytes is prohibitively costly at scale due to long sequence lengths.
```
New probe:
```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", tokenization has previously been essential because directly training llms on bytes is prohibitively costly at scale due to
```
New target:
```text
 long sequence lengths.
```

## Rejection Reasons

- `1_58` row `84`: No safe coherent suffix-only shortening from review. (rejected)
- `BOFT` row `129`: No safe coherent suffix-only shortening from review. (rejected)
- `ByteLatent` row `192`: No safe coherent suffix-only shortening from review. (rejected)
- `DPO` row `106`: No safe coherent suffix-only shortening from review. (rejected)
- `DPO` row `140`: No safe coherent suffix-only shortening from review. (rejected)
- `DPO` row `281`: No safe coherent suffix-only shortening from review. (rejected)
- `FeatLLM` row `37`: No safe coherent suffix-only shortening from review. (rejected)
- `GSPO` row `29`: No safe coherent suffix-only shortening from review. (rejected)
- `GSPO` row `34`: Formula suffix did not pass verbatim source validation. (rejected)
- `GSPO` row `65`: No safe coherent suffix-only shortening from review. (rejected)
- `GSPO` row `89`: No safe coherent suffix-only shortening from review. (rejected)
- `GSPO` row `93`: No safe coherent suffix-only shortening from review. (rejected)
- `LongRoPE` row `152`: No safe coherent suffix-only shortening from review. (rejected)
- `LongRoPE` row `228`: No safe coherent suffix-only shortening from review. (rejected)
- `fa3` row `246`: No safe coherent suffix-only shortening from review. (rejected)
- `xLSTM` row `66`: No safe coherent suffix-only shortening from review. (rejected)
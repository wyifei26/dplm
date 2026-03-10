# DPLM-2 Architecture Documentation

> This document provides an in-depth explanation of the DPLM-2 architecture, answering:
> 1. What is the overall architectural design of DPLM-2?
> 2. What is the LFQ (Lookup-Free Quantization) principle used to tokenize protein structure information?
> 3. Are structure tokens and sequence tokens concatenated together for diffusion generation?

---

## Table of Contents

- [Overview](#overview)
- [1. Overall Architecture Design](#1-overall-architecture-design)
  - [1.1 High-Level Data Flow](#11-high-level-data-flow)
  - [1.2 Vocabulary and Token Space](#12-vocabulary-and-token-space)
  - [1.3 Multi-Modal Training Strategy](#13-multi-modal-training-strategy)
- [2. LFQ: Lookup-Free Quantization for Structure Tokenization](#2-lfq-lookup-free-quantization-for-structure-tokenization)
  - [2.1 Motivation](#21-motivation)
  - [2.2 StructOK Architecture (structok_lfq.py)](#22-structok-architecture-structok_lfqpy)
  - [2.3 LFQ Quantization Principle (lfq.py)](#23-lfq-quantization-principle-lfqpy)
  - [2.4 Loss Functions](#24-loss-functions)
  - [2.5 Tokenize and Detokenize APIs](#25-tokenize-and-detokenize-apis)
- [3. Token Concatenation Strategy for Diffusion Generation](#3-token-concatenation-strategy-for-diffusion-generation)
  - [3.1 Input Sequence Layout](#31-input-sequence-layout)
  - [3.2 Modality Type Detection](#32-modality-type-detection)
  - [3.3 Separate Diffusion Noise per Modality](#33-separate-diffusion-noise-per-modality)
  - [3.4 Joint Forward Pass and Logit Splitting](#34-joint-forward-pass-and-logit-splitting)
  - [3.5 Bimodal Rotary Position Embedding](#35-bimodal-rotary-position-embedding)
- [Key Source Files](#key-source-files)

---

## Overview

DPLM-2 (Multimodal Diffusion Protein Language Model) is a **discrete diffusion generative model** that simultaneously models protein sequences and 3D structures. It extends the sequence-only DPLM by introducing a second modality — discretized structure tokens produced by a structure tokenizer based on **Lookup-Free Quantization (LFQ)**.

The model is described in:
- [DPLM-2: A Multimodal Diffusion Protein Language Model (ICLR 2025)](https://arxiv.org/abs/2410.13782)
- [Elucidating the Design Space of Multimodal Protein Language Models (ICML 2025 Spotlight)](https://arxiv.org/abs/2504.11454)

---

## 1. Overall Architecture Design

### 1.1 High-Level Data Flow

```
                     ┌─────────────────────────────────────────────┐
                     │            Structure Tokenizer (StructOK)    │
   3D Coordinates ──►│  GVP Encoder → pre_quant → LFQ Quantizer   │──► struct_tokens (int ∈ [0, 8191])
                     │              → post_quant → ESMFold Decoder │
                     └─────────────────────────────────────────────┘

  struct_tokens ──► [<cls_struct>, s₁, s₂, …, sₙ, <eos_struct>]   (structure half)
  aa_tokens     ──► [<cls_aa>,    a₁, a₂, …, aₙ, <eos_aa>]        (sequence half)
                                        │
                               torch.concat(dim=1)
                                        │
                                        ▼
              combined_tokens: [<cls_struct>, s₁, …, sₙ, <eos_struct>,
                                 <cls_aa>,    a₁, …, aₙ, <eos_aa>]
                                        │
                        ┌───────────────┴───────────────┐
                        │   Discrete Diffusion (DPLM-2) │
                        │   ESM-based Transformer with   │
                        │   Bimodal Rotary Embeddings    │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │   struct_logits (first half)  │
                        │   aa_logits    (second half)  │
                        └───────────────────────────────┘
```

### 1.2 Vocabulary and Token Space

DPLM-2 uses a **unified token vocabulary** that covers both modalities (defined in `dplm2.py` and `tokenized_protein.py`):

| Range | Content | Count |
|-------|---------|-------|
| `[0, 32]` | Amino acid tokens (standard ESM alphabet) | 33 |
| `[33, 8224]` | Structure tokens (LFQ codebook indices) | 8192 |
| `[8225, 8228]` | Special structure tokens (`<cls_struct>`, `<eos_struct>`, `<mask_struct>`, `<unk_struct>`) | 4 |

The model uses **token ID < 33** as a fast runtime check to distinguish amino acid positions from structure positions (see `get_modality_type` in `dplm2.py`).

Special tokens per modality:

| Token | Modality | Role |
|-------|----------|------|
| `<cls_aa>` | Sequence | BOS for sequence half |
| `<eos_aa>` | Sequence | EOS for sequence half |
| `<mask_aa>` | Sequence | Mask token during diffusion |
| `<unk_aa>` | Sequence | Unknown amino acid |
| `<cls_struct>` | Structure | BOS for structure half |
| `<eos_struct>` | Structure | EOS for structure half |
| `<mask_struct>` | Structure | Mask token during diffusion |
| `<unk_struct>` | Structure | Unknown structure token |
| `<pad>` | Both | Padding |

### 1.3 Multi-Modal Training Strategy

DPLM-2 uses a **warm-up training strategy** initialized from the pre-trained DPLM checkpoint. LoRA adapters prevent excessive drift from the original evolutionary knowledge. Each training batch is randomly split into four tasks controlled by ratio hyperparameters (configured in `configs/experiment/dplm2/dplm2_650m.yaml`):

| Task | Struct noise | Seq noise | Learns |
|------|-------------|-----------|--------|
| `folding_loss_ratio` (default 0.25) | ✓ (noised) | ✗ (t=0, clear) | Sequence → Structure (folding) |
| `inverse_folding_loss_ratio` (default 0.25) | ✗ (t=0, clear) | ✓ (noised) | Structure → Sequence (inverse folding) |
| `joint_loss_ratio` (default 0.25) | ✓ (same t) | ✓ (same t) | Joint structure-sequence generation |
| `single_modality_ratio` (default 0.25) | either | either | Single-modality modeling with cross-attention masked |
| `independent_loss_ratio` (default 0.0) | ✓ (independent t) | ✓ (independent t) | Independent per-modality generation |

The ratio constraints must sum to 1.0:
```
single_modality_ratio + folding_loss_ratio + inverse_folding_loss_ratio
  + joint_loss_ratio + independent_loss_ratio == 1.0
```

---

## 2. LFQ: Lookup-Free Quantization for Structure Tokenization

### 2.1 Motivation

To bridge continuous 3D structure information with the discrete diffusion framework used for sequences, DPLM-2 needs a **structure tokenizer** that converts atomic coordinates into a finite set of discrete indices. Traditional VQ-VAE codebooks require an explicit embedding table lookup and suffer from codebook collapse. LFQ avoids both problems.

### 2.2 StructOK Architecture (`structok_lfq.py`)

The structure tokenizer (`VQModel`, registered as `"structok_lfq"`) has three stages:

```
atom37 coordinates (B, L, 37, 3)
         │
         ▼
  ┌──────────────────┐
  │   GVP Encoder    │   Graph Vector Perceptron transformer over backbone geometry
  │ (GVPTransformer) │   Produces per-residue continuous features
  └────────┬─────────┘
           │ encoder_feats  (B, L, encoder_dim)
           ▼
  ┌──────────────────┐
  │    pre_quant     │   LayerNorm → Linear → ReLU → Linear
  │  (2-layer MLP)   │   Projects to codebook dimension (13 dims for 8192 tokens)
  └────────┬─────────┘
           │ pre_quant  (B, L, 13)
           ▼
  ┌──────────────────┐
  │   LFQ Quantizer  │   Sign-based quantization, no lookup table needed
  │  (codebook=8192) │   Returns: quantized features + struct_tokens (integer indices)
  └────────┬─────────┘
           │ quant  (B, L, 13)   struct_tokens  (B, L)  ∈ [0, 8191]
           ▼
  ┌──────────────────┐
  │   post_quant     │   LayerNorm → Linear → ReLU → Linear → TransformerEncoder
  │ (MLP + Attn.)    │   Contextualizes quantized features
  └────────┬─────────┘
           │ decoder_input  (B, L, decoder_dim)
           ▼
  ┌──────────────────┐
  │  ESMFold Decoder │   Lightweight ESMFold structure module
  │  (4 IPA blocks)  │   Reconstructs 3D backbone coordinates
  └──────────────────┘
```

Only the structure tokens (integer indices from LFQ) are passed to DPLM-2; the GVP encoder is frozen during DPLM-2 training.

### 2.3 LFQ Quantization Principle (`lfq.py`)

LFQ (proposed in [Language Model Beats Diffusion, 2024](https://arxiv.org/abs/2310.05737)) replaces the traditional codebook lookup with a **direct binary quantization** of the continuous feature vector. The core idea is:

#### Step 1 — Represent as Binary Code

For a codebook of size `C = 2^D` (here `C = 8192`, so `D = 13`), each continuous feature vector is projected to `D` dimensions. Each dimension is independently quantized to `{-1, +1}` using the sign function:

```python
# From lfq.py, line 293-296
codebook_value = torch.Tensor([1.0])
quantized = torch.where(x > 0, codebook_value, -codebook_value)
# x > 0  →  +1
# x ≤ 0  →  -1
```

This produces a `D`-bit binary code per residue (big-endian convention).

#### Step 2 — Convert Binary Code to Integer Index

The binary code `{0, 1}^D` (where `+1` maps to `1`, `-1` maps to `0`) is converted to an integer index via:

```python
# From lfq.py, line 312-317
# self.mask = [2^(D-1), 2^(D-2), ..., 2^0]  (big-endian weights)
indices = reduce(
    (quantized > 0).int() * self.mask.int(),
    'b n c d -> b n c', 'sum'
)
# e.g., bits [1,0,1,1,...] → index = 2^12 + 0 + 2^10 + 2^9 + ...
```

#### Step 3 — Straight-Through Gradient Estimator

Since the sign function has zero gradient almost everywhere, LFQ uses the **straight-through estimator** to allow gradients to flow through quantization during training:

```python
# From lfq.py, line 363
quantized = x + (quantized - x).detach()
# Forward pass: uses the quantized (binary) value
# Backward pass: gradient flows as if quantization did not exist
```

#### No Lookup Table Required

The key insight of LFQ: the "codebook" is implicitly defined by the binary representation itself. There is no learned embedding table to look up. To reconstruct the continuous representation from an index during decoding:

```python
# From lfq.py, method get_codebook_entry / decode
bits = (index.unsqueeze(-1) & mask) != 0   # index → bits
x = bits.float() * 2.0 - 1.0              # bits → {-1, +1}
```

This makes LFQ:
- **Memory-efficient**: no large embedding table
- **Collapse-resistant**: all `2^D` codes are always reachable
- **Gradient-friendly**: the straight-through estimator is unambiguous

### 2.4 Loss Functions

During structure tokenizer training, three loss terms are combined:

#### Commitment Loss

Pulls the continuous pre-quantization features toward the nearest binary code:

```python
# From lfq.py, line 352-357
commit_loss = F.mse_loss(x, quantized.detach(), reduction='none')
```

Weighted by `commitment_loss_weight` (default: 0.25).

#### Entropy Loss (Codebook Utilization)

Encourages all `C = 8192` codes to be used roughly equally. Implemented as a two-term objective from [MAGVIT-v2](https://github.com/google-research/magvit):

```python
# From lfq.py, entropy_loss function (lines 95-130)
# Per-sample entropy: minimize to make individual assignments confident
sample_entropy = -Σ p(c|x) log p(c|x)

# Batch-average entropy: maximize to spread usage across codebook
avg_entropy = -Σ p̄(c) log p̄(c)

# Combined: minimize sample entropy, maximize batch entropy
loss = sample_minimization_weight * sample_entropy
     - batch_maximization_weight  * avg_entropy
```

Affinities between `x` and the implicit binary codebook vectors are computed as:
```python
logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
```

Note: `self.codebook` here is **not a learned embedding table** — it is a pre-computed, fixed buffer that lists all `2^D` binary code vectors (`{-1, +1}^D`) registered via `register_buffer`. It is used only to compute affinities for the entropy loss; the actual quantization never looks up this table.

Weighted by `entropy_loss_weight` (default: 0.1).

#### Structure Reconstruction Loss

The ESMFold decoder reconstructs the backbone coordinates. The reconstruction loss (e.g., FAPE, distogram loss) is computed at the decoder level and drives the overall quality of the quantization.

### 2.5 Tokenize and Detokenize APIs

```python
# Tokenize: 3D coordinates → integer token indices
struct_tokens = struct_tokenizer.tokenize(
    atom_positions,   # (B, L, 37, 3)
    res_mask,         # (B, L)
    seq_length,       # (B,)
)
# struct_tokens: (B, L), dtype=int64, values ∈ [0, 8191]

# Detokenize: integer token indices → 3D coordinates
decoder_out = struct_tokenizer.detokenize(
    struct_tokens,    # (B, L)
    res_mask,         # (B, L)
)
# decoder_out["atom37_positions"]: (B, L, 37, 3)
```

---

## 3. Token Concatenation Strategy for Diffusion Generation

**Yes — structure tokens and sequence tokens are concatenated along the sequence dimension** before being fed into the transformer. The two modalities share one ESM-based transformer backbone.

### 3.1 Input Sequence Layout

```
Position:  0            1 … N    N+1          N+2         N+3 … 2N+2   2N+3
Token:     <cls_struct>  s₁…sₙ  <eos_struct>  <cls_aa>    a₁…aₙ        <eos_aa>
Modality:  struct        struct   struct        aa          aa           aa
```

Total sequence length = `2 * protein_length + 4` (for 4 special tokens).

During **generation** (`generate_dplm2.py`), this is constructed as:
```python
input_tokens = torch.concat(
    [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
)
```

During **training** (`dplm2.py`, `compute_loss`):
```python
x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
```

### 3.2 Modality Type Detection

At runtime, DPLM-2 does not need a separate modality embedding; it recovers the modality type by inspecting the token ID range (an intentional design choice in the tokenizer vocabulary):

```python
# From dplm2.py, get_modality_type()
def get_modality_type(self, input_ids):
    input_mask = input_ids.ne(self.pad_id)
    # All amino acid token IDs < 33; all structure token IDs >= 33
    modality_type = ((input_ids < 33) & input_mask).int()
    # 0 = struct, 1 = aa, 2 = padding
    modality_type[~input_mask] = self.pad_type
    return modality_type
```

### 3.3 Separate Diffusion Noise per Modality

Even though structure and sequence tokens are concatenated into a single sequence, **independent timesteps** `t_struct` and `t_aa` are sampled for each modality during training. This allows the model to learn asymmetric conditional generation tasks (folding, inverse folding) in addition to joint generation:

```python
# From dplm2.py, construct_x_t()
struct_t = torch.randint(1, num_diffusion_timesteps + 1, (bsz,))
aatype_t = torch.randint(1, num_diffusion_timesteps + 1, (bsz,))

# For folding task: sequence is unmasked (t_aa = 0), only structure is noised
aatype_t = aatype_t.masked_fill(folding_index, 0)

# For inverse folding task: structure is unmasked (t_struct = 0), only sequence is noised
struct_t = struct_t.masked_fill(inverse_folding_index, 0)

# For joint generation: same timestep t for both modalities
aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])
```

Each modality uses its own mask token when corrupted:
```python
# From dplm2.py, q_sample()
x_t = x_0.masked_fill(t_mask & aa_position,     self.aa_mask_id)
x_t = x_t.masked_fill(t_mask & struct_position, self.struct_mask_id)
```

### 3.4 Joint Forward Pass and Logit Splitting

The concatenated noisy token sequence is passed through the shared ESM transformer in a single forward pass:

```python
# Combined input: [struct_half | aa_half]
x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)

# One forward pass for both modalities
model_outputs = self.forward(input_ids=x_t)

# Split logits back into per-modality halves
struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)
```

During inference, the model prevents cross-modality token predictions by masking out invalid vocabulary ranges per position:

```python
# From dplm2.py, forward_decoder()
# Amino acid positions: disable all structure token logits (IDs >= 33)
logits[indices_aa[0], indices_aa[1], 33:] = -math.inf

# Structure positions: disable all amino acid token logits (IDs < 33)
logits[indices_struct[0], indices_struct[1], :33] = -math.inf
```

### 3.5 Bimodal Rotary Position Embedding

A key modification enables the shared transformer to understand that the first and second halves of the sequence are **parallel representations of the same protein** rather than a longer sequence. The `ModifiedRotaryEmbedding` (`dplm2_modeling_esm.py`) applies rotary embeddings **independently** to each modality half:

```python
# From dplm2_modeling_esm.py, ModifiedRotaryEmbedding.forward()
if self.aa_type in type_ids and self.struct_type in type_ids:
    # Both modalities present: split queries/keys, apply RoPE independently
    q_1, q_2 = q.chunk(2, dim=-2)   # q_1 = struct half, q_2 = aa half
    k_1, k_2 = k.chunk(2, dim=-2)

    q_1 = apply_rotary_pos_emb(q_1, cos_cached, sin_cached)  # positions 0…N-1
    q_2 = apply_rotary_pos_emb(q_2, cos_cached, sin_cached)  # positions 0…N-1 (reused!)
    # ...
    q = torch.cat((q_1, q_2), dim=-2)
```

This means residue `i` in the structure half and residue `i` in the amino acid half get the **same rotary position index**, making their inter-modality attention naturally position-aligned.

For single-modality tasks (folding / inverse folding), cross-attention between the two halves can be selectively masked to prevent information leakage:

```python
# From dplm2.py, forward()
if "single_modality" in kwargs:
    # Block struct positions from attending to aa positions (and vice versa)
    struct_attention_bias[single_modality_index, :, :, L // 2:] = -math.inf
    aa_attention_bias   [single_modality_index, :, :, :L // 2] = -math.inf
```

---

## Key Source Files

| File | Role |
|------|------|
| `src/byprot/models/structok/modules/lfq.py` | LFQ quantizer implementation (sign-based quantization, entropy loss, straight-through estimator) |
| `src/byprot/models/structok/structok_lfq.py` | Structure tokenizer (GVP encoder → LFQ → ESMFold decoder) |
| `src/byprot/models/dplm2/dplm2.py` | DPLM-2 main model (diffusion forward/backward, token concatenation, loss computation) |
| `src/byprot/models/dplm2/modules/dplm2_modeling_esm.py` | Modified ESM transformer (bimodal rotary position embedding) |
| `src/byprot/datamodules/dataset/tokenized_protein.py` | `DPLM2Tokenizer` (unified vocabulary for sequence + structure tokens) |
| `generate_dplm2.py` | Inference script (co-generation, folding, inverse folding, motif scaffolding) |
| `configs/experiment/dplm2/dplm2_650m.yaml` | DPLM-2 training configuration (modality ratios, model size) |
| `configs/experiment/structok/structok_lfq_8k_pdb_swissprot_c512.yaml` | Structure tokenizer configuration (codebook size, encoder/decoder dims) |

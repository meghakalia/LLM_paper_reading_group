# LLaMA 1 — Paper Notes

## Core Claim

Scaling laws (Hoffmann et al., Chinchilla) optimize for **training compute** — they ask: given a fixed FLOP budget, what is the best model size and number of tokens? Their answer: train a larger model for fewer tokens.

LLaMA's insight: this ignores **inference cost**. At deployment scale, you run the model millions of times. A smaller model trained for longer is:
- Cheaper per inference call (fewer parameters to load)
- Ultimately more practical, even if it required more training tokens

**Key claim:** A 7B model trained on 1T tokens can match or outperform a 10x larger model trained for fewer tokens on many benchmarks.

- Trained only on publicly available, open-source datasets (unlike GPT-3/PaLM which used proprietary data).
- Model sizes released: **7B, 13B, 33B, 65B** parameters.

---

## Training Data

| Source                  | Weight | Notes |
|-------------------------|--------|-------|
| English CommonCrawl     | 67%    | Filtered with CCNet pipeline |
| C4                      | 15%    | Cleaned web text |
| GitHub                  | 4.5%   | Available via Google BigQuery |
| Wikipedia               | 4.5%   | 20 languages |
| Gutenberg + Books3      | 4.5%   | Open-source books; Books3 is from ThePile |
| ArXiv                   | 2.5%   | LaTeX source; removed preamble, bibliography, comments, expanded macros for consistency |
| Stack Exchange          | 2%     | 28 major sites; Q&A pairs; answers sorted high-to-low score; HTML tags removed |

**Total: ~1.4 trillion tokens.** Most data is seen only once during training; Wikipedia and Books were used for **2 epochs**.

### Data Processing — CommonCrawl (CCNet Pipeline)
1. **Deduplication** at the line level
2. **Language identification** using a fastText linear classifier (keep English)
3. **Quality filtering** using an n-gram language model trained on Wikipedia; pages with low perplexity (close to Wikipedia-like text) are kept

---

## Tokenizer

- **Byte-Pair Encoding (BPE)** tokenizer (same family as GPT-2/3)
- All numbers split into **individual digits** (e.g., `2024` → `2`, `0`, `2`, `4`) — prevents arithmetic inconsistencies
- **Byte-level fallback** for unknown UTF-8 characters — the vocabulary never encounters unknown tokens

---

## Architecture: Key Modifications vs. GPT-3

LLaMA uses a standard Transformer decoder but with three important changes: **Pre-Norm with RMSNorm**, **SwiGLU activations**, and **Rotary Positional Embeddings (RoPE)**.

---

### 1. Pre-Normalization (from GPT-3)

Normalizing the *input* to each sub-layer (instead of the output) leads to more stable training gradients.

**Post-Norm** (original Transformer):
```
h₁ = Norm(x + Attn(x))
h₂ = Norm(h₁ + MLP(h₁))
```

**Pre-Norm** (LLaMA):
```
h₁ = x + Attn(Norm(x))
h₂ = h₁ + MLP(Norm(h₁))
```

The residual stream `x` bypasses the norm — the raw signal is always preserved, which helps gradients flow unchanged through many layers.

---

### 2. RMSNorm vs. LayerNorm

**LayerNorm** normalizes by subtracting the mean and dividing by the standard deviation, then applies learnable scale (γ) and shift (β):

$$y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i$$

where $\mu = \frac{1}{d}\sum_j x_j$ and $\sigma^2 = \frac{1}{d}\sum_j (x_j - \mu)^2$.

**RMSNorm** drops the mean subtraction entirely. The hypothesis is that *re-centering* (mean subtraction) adds little benefit, while *re-scaling* is what matters for stability:

$$y_i = \gamma_i \cdot \frac{x_i}{\text{RMS}(\mathbf{x})}, \quad \text{where} \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}$$

- $x_i$: individual elements of the input vector
- $d$: number of elements (hidden dimension)
- $\epsilon$: small constant for numerical stability (avoids division by zero)
- $\gamma_i$: learnable per-element scale parameter

**Comparison:**

| Component        | LayerNorm     | RMSNorm              |
|------------------|---------------|----------------------|
| Mean subtraction | Yes           | No                   |
| Variance scaling | Yes           | No (uses RMS instead)|
| γ (scale)        | Learned       | Learned              |
| β (shift)        | Learned       | Removed              |

RMSNorm is **simpler and faster** — fewer operations, fewer parameters, empirically similar or better training stability.

---

### 3. SwiGLU Activation (from PaLM)

LLaMA replaces the standard ReLU/GeLU FFN with **SwiGLU**, a gated activation.

**Swish** (the base activation):

$$\text{Swish}(z) = z \cdot \sigma(z)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid. Unlike ReLU, Swish is smooth and allows small negative values to pass through.

**SwiGLU** (gated variant):

$$\text{SwiGLU}(x) = (x W_1) \odot \text{Swish}(x W_2)$$

- $W_1$: learns the **main signal** (what features to produce)
- $W_2$: learns the **gate** (which features to allow or suppress)
- $\odot$: element-wise (Hadamard) product

The gate $\text{Swish}(xW_2)$ dynamically scales each feature of the main branch. This gives the FFN layer a form of input-dependent routing.

> Note: Because SwiGLU uses two weight matrices instead of one, the hidden dimension of the FFN is reduced by $\frac{2}{3}$ to keep parameter count comparable to a standard FFN.

---

### 4. Rotary Positional Embeddings — RoPE (from GPTNeo)

Instead of adding absolute position embeddings to token vectors, **RoPE encodes position directly into the attention mechanism** by rotating Query and Key vectors.

**Core idea:** For position $m$, rotate the query (and key) vector by an angle proportional to $m$ in each 2D subspace of the embedding. When you then compute $q_m^\top k_n$, the dot product depends only on the **relative position** $m - n$, not the absolute positions.

For each pair of dimensions $(2i, 2i+1)$ in a $d$-dimensional vector, apply a 2D rotation:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

where the base frequencies are $\theta_i = 10000^{-2i/d}$ (same design as sinusoidal embeddings, but applied as a rotation).

**Why this is better than absolute positional embeddings:**
- Dot products between $q_m$ and $k_n$ naturally **decay with distance** $|m - n|$ — distant tokens attend less
- **Extrapolates** to longer sequences not seen during training
- No additional parameters; position is baked into Q/K before the attention score

---

## Engineering Optimizations

### Efficient Causal Attention
Instead of storing the full $N \times N$ attention weight matrix (memory: $O(N^2)$), LLaMA uses **FlashAttention** — the attention weights are computed on-the-fly during the backward pass using tiling, without materializing the full matrix.

### Gradient Checkpointing
During a standard forward pass, all intermediate activations are stored for use in backpropagation — memory scales with depth. **Gradient checkpointing** trades memory for compute:
- Only a subset of activations ("checkpoints") are saved during the forward pass
- During backprop, missing activations are **recomputed** by re-running the forward pass from the nearest checkpoint

This significantly reduces peak memory, allowing larger batch sizes or longer sequences.

---

## Glossary

- **CommonCrawl**: A massive public web crawl dataset; raw and noisy, requires heavy filtering
- **CCNet Pipeline**: Meta's pipeline for cleaning CommonCrawl — deduplication, language ID (fastText), quality filtering (n-gram LM perplexity)
- **C4 (Colossal Cleaned Crawled Corpus)**: A cleaner version of CommonCrawl used in T5; filtered using heuristics
- **BPE (Byte-Pair Encoding)**: Subword tokenization algorithm that merges frequent character pairs iteratively
- **ThePile**: A large open-source dataset by EleutherAI that includes Books3, among other sources
- **FlashAttention**: Memory-efficient exact attention using tiling; avoids materializing the $O(N^2)$ attention matrix
- **Gradient Checkpointing**: Re-compute activations during backward pass instead of storing all of them; trades FLOP for memory

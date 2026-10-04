# TurboQuant: a note for a mathematician

## Context

We are building an approximate nearest-neighbour (ANN) search over large vector collections. Instead of storing full float32 vectors we **quantize** each vector to a compact code of a few bits per coordinate, then estimate distances from those codes.

The algorithm follows **TurboQuant** (ICLR 2026, arXiv:2504.19874).

---

## Algorithm, step by step

### Input

A vector $x \in \mathbb{R}^d$. In our case $d = 128$, and the coordinates are integers from 0 to 255, so the source data is 8-bit.

### Step 1: normalization

$$\hat{x} = \frac{x}{\|x\|_2}$$

$\|x\|_2$ is stored on its own as float32. $\hat{x}$ then lies on the unit sphere $S^{d-1}$.

### Step 2: randomized Hadamard transform

$$r = \text{WHT}(D \cdot \hat{x})$$

$D$ is a diagonal matrix of random signs $\pm 1$. $\text{WHT}$ is the normalized Walsh–Hadamard transform, the orthogonal matrix $H / \sqrt{d}$.

**Why:** after this transform the coordinates $r_i$ are approximately i.i.d. and close to $\mathcal{N}(0, \sigma^2)$. By the central limit theorem each coordinate is a linear combination of $d$ source coordinates with random signs. One scalar quantizer can then be used for every coordinate.

### Step 3: one global scale $\sigma$

$$\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} r_i^2}$$

One scalar for the whole vector, not one per dimension. Stored as float32.

### Step 4: scalar quantization (Lloyd-Max for a Gaussian)

Normalize the coordinates: $z_i = r_i / \sigma$.

Assume $z_i \sim \mathcal{N}(0, 1)$ and apply the **Lloyd-Max scalar quantizer** for that distribution.

Of a budget of $b$ bits per coordinate, the algorithm gives $(b-1)$ bits to the quantizer and 1 bit to the QJL correction below. So:

| bits_per_coord | quantization levels | QJL bits |
|:-:|:-:|:-:|
| 4 | $2^3 = 8$ | 1 |
| 8 | $2^7 = 128$ | 1 |

Lloyd-Max iteratively finds the boundaries $\{b_j\}$ and centroids $\{c_j\}$ for $\mathcal{N}(0,1)$ that minimize the MSE:

$$\text{MSE} = \mathbb{E}\left[(Z - Q(Z))^2\right], \quad Z \sim \mathcal{N}(0,1)$$

Each coordinate becomes an index $q_i \in \{0, \ldots, 2^{b-1}-1\}$. The reconstruction is $\tilde{r}_i = c_{q_i} \cdot \sigma$.

### Step 5: QJL correction (quantized Johnson–Lindenstrauss)

The residual is

$$e_i = r_i - \tilde{r}_i = r_i - c_{q_i} \cdot \sigma$$

A second randomized Hadamard transform is applied to $e$, and only the **sign bit** of the result is kept (1 bit per coordinate):

$$s_i = \text{sign}\left(\text{WHT}(D' \cdot e)\right)_i \in \{-1, +1\}$$

$\gamma = \|e\|_2$ is also stored as float32.

**Stored per vector:**
- $d$ bytes: $(q_i \ll 1) \mid s_i$, the packed index plus the sign bit
- 3 float32 values: $\|x\|_2$, $\gamma$, $\sigma$

---

## Distance

### Asymmetric (query $\leftrightarrow$ code)

For a query $q$ and an encoded vector $x$:

$$\langle q, x \rangle \approx \|q\|_2 \cdot \|x\|_2 \cdot \left(\text{ip\_mse} + \text{correction}\right)$$

where

$$\text{ip\_mse} = \sigma \sum_{i=1}^{d} \hat{q}^{\text{rot}}_i \cdot c_{q_i}$$

$$\text{correction} = \underbrace{\sqrt{\frac{\pi}{2d}}}_{\text{scale}} \cdot \gamma \cdot \sum_{i=1}^{d} \hat{q}^{\text{qjl}}_i \cdot s_i$$

$\hat{q}^{\text{rot}}$ is the query after normalization and the Hadamard with the same seed.
$\hat{q}^{\text{qjl}}$ is the query after the second Hadamard (the QJL seed).

The final squared L2 distance is

$$d(q, x) = \max\left(0,\ \|q\|^2 + \|x\|^2 - 2\langle q, x \rangle\right)$$

### Symmetric (code $\leftrightarrow$ code)

A comparison of two encoded vectors uses **only the MSE part**. The QJL bit is ignored:

$$\langle x_a, x_b \rangle \approx \|x_a\| \cdot \|x_b\| \cdot \sum_{i=1}^d (c_{q_i^a} \cdot \sigma_a)(c_{q_i^b} \cdot \sigma_b)$$

---

## Problem

The **SIFT1M** dataset has 1 million vectors of dimension 128. Coordinates are `uint8` (0..255). The source data carries exactly 8 bits of information per coordinate.

With `bits_per_coord = 8` we expected **recall = 1.0**, a perfect recovery of neighbour order, because the source itself is 8-bit.

**Measured:** recall@1 $\approx$ 0.91, recall@10 $\approx$ 0.94, recall@100 $\approx$ 0.97.

---

## Where we think accuracy is lost

### Hypothesis 1: 7 bits of quantization instead of 8

At $b = 8$ bits per coordinate, $(b-1) = 7$ bits go to Lloyd-Max (128 levels) and 1 bit goes to QJL. The source has 256 levels. At 8 bits the QJL correction may not pay for the lost MSE bit.

**Question:** How should $b$ bits be split between MSE quantization and the QJL correction? Is there an analytic error bound in terms of that split?

### Hypothesis 2: one global $\sigma$ is not optimal

One $\sigma$ for the whole vector assumes the same variance on every coordinate. After the Hadamard that is approximately true, but not exact (finite dimension, padding). Would a per-dimension scale help?

**Question:** What is the expected variation of $\text{Var}(r_i)$ across coordinates after a randomized Hadamard at finite $d = 128$? How much does that move the quantization MSE?

### Hypothesis 3: Lloyd-Max for $\mathcal{N}(0,1)$ does not match the real distribution

After the Hadamard the coordinates are approximately Gaussian, not exactly. Tails and higher moments can differ. At 128 quantization levels those differences can matter.

**Question:** Can the extra MSE from the gap between the real distribution and $\mathcal{N}(0,1)$ be bounded? How fast does that gap shrink as $d$ grows?

### Hypothesis 4: the QJL correction formula

The factor $\sqrt{\pi / 2d}$ is $\mathbb{E}[|Z|]$ for $Z \sim \mathcal{N}(0, 1/d)$. It approximates an inner product from sign bits (the same idea as a sign random projection).

**Question:** Does the formula still hold when the residual $e$ is no longer Gaussian, because it is a quantization error with bounded support? Is the scaling order right?

### Hypothesis 5: the symmetric distance ignores QJL

Code-to-code comparison does not use the QJL bit. One of the 8 bits then takes no part in the distance. That is 7-bit quantization in practice.

**Question:** Is there a correct code-to-code formula that uses the QJL bits of both vectors?

### Hypothesis 6: normalization drops information

$x / \|x\|$ maps the discrete lattice $\{0, \ldots, 255\}^{128}$ onto a continuous set on the sphere. The inverse map, through $\sigma$, the centroids, and $\|x\|$, does not recover that lattice exactly.

**Question:** Is there an information-theoretic lower bound on the reconstruction error of "normalize → rotate → scalar-quantize → denormalize" for data on an integer lattice?

---

## What would help

1. Which hypothesis is the **dominant** error source
2. An analytic estimate, at least in order of magnitude, of the MSE or the inner-product error for each source
3. Which formula changes would buy the most accuracy

---

## References

- Paper: [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- Implementation: `include/turboquant/space_turboquant.h` (encoding and distances), `include/turboquant/turboquant.h` (code layout), `include/turboquant/srht.h` (Hadamard and signs)

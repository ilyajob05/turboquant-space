# TurboQuant: why recall is below 1 at 8 bits on SIFT1M

## Problem

SIFT1M stores uint8 descriptors (8 bits per coordinate). Quantizing with `bits_per_coord=8` suggests recall of 1.0. The measured values are recall@1 ≈ 0.91, recall@10 ≈ 0.94, and recall@100 ≈ 0.97.

## Causes, largest first

### 1. Normalization destroys the original discrete grid

Pipeline: `uint8 → float / ||x|| → Hadamard → / σ → Lloyd-Max`. After the norm and the Hadamard rotation the coordinates are continuous floats. The original 8-bit lattice is gone. The rank correlation of distances before and after normalization is **0.85** (Spearman). This is the main loss. The algorithm still needs that split: the norm and the direction are quantized separately.

### 2. 7 bits of MSE plus 1 QJL bit, instead of 8 bits of MSE

At `bits_per_coord=8` the algorithm gives 7 bits to Lloyd-Max (128 levels) and 1 bit to the QJL sign of the residual. The QJL bit adds +3.3% to recall@10, but 128 levels are fewer than the 256 levels of the source data.

### 3. The Gaussian Lloyd-Max model is the right model here

A data-driven Lloyd-Max, trained on the real distribution of rotated coordinates, was compared with the Gaussian one. At 7 bits (128 levels) the Gaussian model is **better**. After the Hadamard the distribution is close to N(0, σ²). Data-driven centroids help only at a small number of levels (3 bits: +7%).

### 4. float32 arithmetic does not explain the gap

float32 and float64 pipelines differ by 0.00%. Compute precision does not account for the lost recall.

## Tests

| Test | What was checked | Result |
|------|------------------|--------|
| QJL bit value | recall with and without the QJL correction (asymmetric vs symmetric) | QJL adds +3.3% at 8 bits and +18% at 4 bits |
| Normalization loss | Spearman correlation of L2 distances before and after normalize+Hadamard | ρ=0.85. The Hadamard is orthogonal, so ρ is unchanged by the rotation itself |
| float32 vs float64 | quantization RMSE in both pipelines | 0% difference |
| Gaussian vs empirical LM | quantization MSE: analytic vs data-driven Lloyd-Max | 3 bits: empirical is 7% better. 7 bits: Gaussian is better |
| Distribution normality | kurtosis and skewness of rotated coordinates | kurtosis=0.34, skew=−0.41. Close to Gaussian |
| Symmetric QJL correction | three symmetric distances: original, light, full | Full: recall 0.908→0.937 (8 bits), RMSE −39% |

## Side result: a better symmetric distance

The full symmetric distance uses the QJL correction and all four inner-product terms: `<r̃_a, r̃_b> + <r̃_a, e_b> + <e_a, r̃_b> + <e_a, e_b>`. It runs a Hadamard per pair, so the cost is O(d log d) rather than O(d). Recall then nearly matches the asymmetric distance:

| bits | Original (MSE only) | Full (MSE+QJL) | Asymmetric (reference) |
|------|--------------------:|---------------:|-----------------------:|
| 4 | 0.198 | **0.351** | 0.379 |
| 8 | 0.908 | **0.937** | 0.941 |

## Conclusion

The implementation **matches the algorithm**. Recall below 1 is a property of the normalize → Hadamard → quantize pipeline. The largest term is normalization (projection onto the sphere). The next term is 7 bits of MSE instead of 8. One possible change, for data that is already 8-bit, is a mode with 8 MSE bits and no QJL sign bit (256 levels).

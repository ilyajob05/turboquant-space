# Energy proxy validation (powermetrics)

Default model: uncalibrated: arm64-apple placeholder (p_core=4.0 W, p_base=1.5 W, e_byte=0.1 nJ).

Spearman rho(proxy, measured) over 28 runs: **0.982**
Fitted: p_core_w=2.1, p_base_w=2.32, e_byte_nj=0.0759

| config | n | fewest threads mJ (proxy / measured) | most threads mJ (proxy / measured) |
|---|---:|---|---|
| rq4-flat | 20000 | 15.5 / 12.6 (1 t) | 13.8 / 12.9 (4 t) |
| rq4-flat | 90000 | 69.3 / 56.5 (1 t) | 62.9 / 50.1 (4 t) |
| rq8-flat | 20000 | 22.3 / 17.4 (1 t) | 19.7 / 14.2 (4 t) |
| rq8-flat | 90000 | 100 / 73.9 (1 t) | 90.8 / 63.7 (4 t) |
| tq16-flat | 20000 | 24.5 / 17.6 (1 t) | 23 / 16.9 (4 t) |
| tq16-flat | 90000 | 109 / 81.3 (1 t) | 107 / 70.2 (4 t) |
| tq4-flat | 20000 | 29.6 / 24.7 (1 t) | 26.2 / 21.2 (4 t) |
| tq4-flat | 90000 | 131 / 104 (1 t) | 137 / 61.7 (4 t) |
| tq8-flat | 20000 | 41.7 / 39.7 (1 t) | 38.3 / 28.2 (4 t) |
| tq8-flat | 90000 | 186 / 157 (1 t) | 177 / 124 (4 t) |

# Autotune validation 2026-10-06

Data: `dbpedia_openai_100K_vectors.npy`. n_sample=20000, n_queries=200, k=10, time_budget_s=60.0, vsq 0.2.0.

## Recall drift (sample -> full n)

| n_full | profile | seed | chosen | recall@n_sample | recall@n_full | drift | eps_r | within eps_r |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 50000 | speed | 0 | tq4-fs | 0.9815 | 0.9760 | +0.0055 | 0.0060 | yes |
| 50000 | speed | 1 | tq4-fs | 0.9820 | 0.9760 | +0.0060 | 0.0059 | **no** |
| 50000 | speed | 2 | tq4-fs | 0.9850 | 0.9820 | +0.0030 | 0.0054 | yes |
| 50000 | accuracy | 0 | rq8-fs | 0.9980 | 0.9975 | +0.0005 | 0.0020 | yes |
| 50000 | accuracy | 1 | tq8-flat | 0.9985 | 0.9990 | -0.0005 | 0.0020 | yes |
| 50000 | accuracy | 2 | rq8-flat | 0.9980 | 0.9985 | -0.0005 | 0.0020 | yes |
| 100000 | speed | 0 | tq4-fs | 0.9845 | 0.9815 | +0.0030 | 0.0055 | yes |
| 100000 | speed | 1 | tq4-fs | 0.9795 | 0.9785 | +0.0010 | 0.0063 | yes |
| 100000 | speed | 2 | tq4-fs | 0.9775 | 0.9805 | -0.0030 | 0.0066 | yes |
| 100000 | accuracy | 0 | tq8-flat | 0.9995 | 0.9995 | +0.0000 | 0.0020 | yes |
| 100000 | accuracy | 1 | tq8-flat | 0.9985 | 0.9995 | -0.0010 | 0.0020 | yes |
| 100000 | accuracy | 2 | tq8-flat | 0.9995 | 0.9975 | +0.0020 | 0.0020 | **no** |

## Latency extrapolation

| n_full | profile | seed | chosen | kind | predicted ms | measured ms | error |
|---:|---|---:|---|---|---:|---:|---:|
| 50000 | speed | 0 | tq4-fs | direct | 1.73 | 1.77 | -2.5% |
| 50000 | speed | 1 | tq4-fs | direct | 1.74 | 1.75 | -0.4% |
| 50000 | speed | 2 | tq4-fs | direct | 1.72 | 1.74 | -0.8% |
| 50000 | accuracy | 0 | rq8-fs | sample-fit | 2.89 | 3.1 | -6.5% |
| 50000 | accuracy | 1 | tq8-flat | direct | 17.2 | 17.1 | +0.5% |
| 50000 | accuracy | 2 | rq8-flat | direct | 8.75 | 8.65 | +1.2% |
| 100000 | speed | 0 | tq4-fs | direct | 3.34 | 3.3 | +1.2% |
| 100000 | speed | 1 | tq4-fs | direct | 3.41 | 3.35 | +1.8% |
| 100000 | speed | 2 | tq4-fs | direct | 3.31 | 3.31 | +0.2% |
| 100000 | accuracy | 0 | tq8-flat | direct | 34.4 | 34.7 | -1.0% |
| 100000 | accuracy | 1 | tq8-flat | direct | 35 | 35.7 | -2.1% |
| 100000 | accuracy | 2 | tq8-flat | direct | 35.2 | 35.1 | +0.3% |

## Choice stability across seeds

| n_full | profile | choices |
|---:|---|---|
| 50000 | accuracy | rq8-fs, tq8-flat, rq8-flat (**varies**) |
| 50000 | speed | tq4-fs, tq4-fs, tq4-fs (stable) |
| 100000 | accuracy | tq8-flat, tq8-flat, tq8-flat (stable) |
| 100000 | speed | tq4-fs, tq4-fs, tq4-fs (stable) |

## Verdict

- recall drift beyond eps_r: up to +0.0001
- arena-fit latency error: no arena-fit choice in this run

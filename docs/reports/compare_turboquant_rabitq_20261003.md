# Quantizer comparison 20261003T134329Z

Host `Darwin arm64`, Python 3.13.7, turboquant 0.1.0.

Shared float32 N(0, 1) draw per dimension. Accuracy uses n_base=2000, n_query=40, k=10. Throughput uses n_speed=400 and the fastest of the timed repeats. recall is overlap with exact squared L2. rel_mae is the mean relative error of the asymmetric distance. Both search rates are `distance_1_to_n`: the query is prepared once, then every slot is scored in one C++ pass.

| method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search pairs/s | search api |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| turboquant | 4 | 128 | 76 | 0.8250 | 0.8350 | 0.0098 | 1,259,347 | 31,578,102 | distance_1_to_n |
| turboquant | 8 | 128 | 140 | 1.0000 | 0.9900 | 0.0007 | 877,353 | 25,736,716 | distance_1_to_n |
| rabitq | 1 | 128 | 24 | 0.2500 | 0.3500 | 0.0535 | 2,011,738 | 65,306,246 | distance_1_to_n |
| rabitq | 4 | 128 | 72 | 0.8750 | 0.8875 | 0.0071 | 44,728 | 66,203,097 | distance_1_to_n |
| rabitq | 8 | 128 | 136 | 1.0000 | 0.9950 | 0.0004 | 2,858 | 76,190,318 | distance_1_to_n |
| turboquant | 4 | 1024 | 524 | 0.7250 | 0.8600 | 0.0035 | 174,647 | 4,013,365 | distance_1_to_n |
| turboquant | 8 | 1024 | 1036 | 1.0000 | 0.9975 | 0.0003 | 111,701 | 3,676,742 | distance_1_to_n |
| rabitq | 1 | 1024 | 136 | 0.1000 | 0.3850 | 0.0188 | 202,267 | 8,290,157 | distance_1_to_n |
| rabitq | 4 | 1024 | 520 | 0.7500 | 0.9050 | 0.0027 | 6,117 | 12,467,656 | distance_1_to_n |
| rabitq | 8 | 1024 | 1032 | 1.0000 | 0.9975 | 0.0002 | 332 | 9,125,335 | distance_1_to_n |

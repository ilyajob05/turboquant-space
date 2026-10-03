# Quantizer comparison 20261003T124202Z

Host `Darwin arm64`, Python 3.13.7, turboquant 0.1.0.

Shared float32 N(0, 1) draw per dimension. Accuracy uses n_base=2000, n_query=40, k=10. Throughput uses n_speed=400 and the fastest of the timed repeats. recall is overlap with exact squared L2. rel_mae is the mean relative error of the asymmetric distance. Both search rates are `distance_1_to_n`: the query is prepared once, then every slot is scored in one C++ pass.

| method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search pairs/s | search api |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| turboquant | 4 | 128 | 76 | 0.8250 | 0.8350 | 0.0098 | 748,072 | 16,931,213 | distance_1_to_n |
| turboquant | 8 | 128 | 140 | 1.0000 | 0.9900 | 0.0007 | 485,290 | 14,999,815 | distance_1_to_n |
| rabitq | 1 | 128 | 24 | 0.2500 | 0.3500 | 0.0535 | 503,092 | 35,819,845 | distance_1_to_n |
| rabitq | 4 | 128 | 72 | 0.8750 | 0.8875 | 0.0071 | 20,262 | 41,025,696 | distance_1_to_n |
| rabitq | 8 | 128 | 136 | 1.0000 | 0.9950 | 0.0004 | 1,050 | 40,853,869 | distance_1_to_n |
| turboquant | 4 | 1024 | 524 | 0.7250 | 0.8600 | 0.0035 | 100,484 | 2,348,907 | distance_1_to_n |
| turboquant | 8 | 1024 | 1036 | 1.0000 | 0.9975 | 0.0003 | 62,600 | 2,099,738 | distance_1_to_n |
| rabitq | 1 | 1024 | 136 | 0.1000 | 0.3850 | 0.0188 | 104,376 | 4,764,229 | distance_1_to_n |
| rabitq | 4 | 1024 | 520 | 0.7500 | 0.9050 | 0.0027 | 1,940 | 6,694,562 | distance_1_to_n |
| rabitq | 8 | 1024 | 1032 | 1.0000 | 0.9975 | 0.0002 | 102 | 5,280,526 | distance_1_to_n |

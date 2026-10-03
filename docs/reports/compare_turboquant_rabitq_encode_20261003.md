# Quantizer comparison 20261003T194028Z

Host `Darwin arm64`, Python 3.13.7, turboquant 0.1.0.

Shared float32 N(0, 1) draw per dimension. Accuracy uses n_base=2000, n_query=40, k=10. Throughput uses n_speed=400 and the fastest of the timed repeats. recall is overlap with exact squared L2. rel_mae is the mean relative error of the asymmetric distance. Both search rates are `distance_1_to_n`: the query is prepared once, then every slot is scored in one C++ pass.

Method names below were aligned with the API after this run. The measured codes and rates are unchanged. `rabitq` at 1 bit is the sign code. `rabitq-algorithm1` is the Extended RaBitQ sweep (these rows were labeled `rabitq` at 4 and 8 bits). `rabitq-fixed-scale` is one frozen scale for the space (the raw log called it `rabitq-const`); it is now the 4/8-bit default. `rabitq-windowed-scale` is one scale per vector inside the tight window (the raw log called it `rabitq-window`), the scalar next-event heap. The morning file `compare_turboquant_rabitq_20261003` still labels that sweep as `rabitq`.

| method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search pairs/s | search api |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| turboquant | 4 | 128 | 76 | 0.8250 | 0.8350 | 0.0098 | 747,140 | 16,991,623 | distance_1_to_n |
| turboquant | 8 | 128 | 140 | 1.0000 | 0.9900 | 0.0007 | 472,836 | 15,262,516 | distance_1_to_n |
| rabitq | 1 | 128 | 24 | 0.2500 | 0.3500 | 0.0535 | 1,040,761 | 35,426,428 | distance_1_to_n |
| rabitq-algorithm1 | 4 | 128 | 72 | 0.8750 | 0.8875 | 0.0071 | 25,328 | 40,675,168 | distance_1_to_n |
| rabitq-algorithm1 | 8 | 128 | 136 | 1.0000 | 0.9950 | 0.0004 | 1,679 | 41,025,574 | distance_1_to_n |
| rabitq-fixed-scale | 4 | 128 | 72 | 0.9250 | 0.8850 | 0.0075 | 730,206 | 41,025,696 | distance_1_to_n |
| rabitq-fixed-scale | 8 | 128 | 136 | 1.0000 | 0.9725 | 0.0011 | 1,270,011 | 41,919,885 | distance_1_to_n |
| rabitq-windowed-scale | 4 | 128 | 72 | 0.8500 | 0.8750 | 0.0073 | 66,938 | 40,000,045 | distance_1_to_n |
| rabitq-windowed-scale | 8 | 128 | 136 | 1.0000 | 0.9950 | 0.0004 | 15,618 | 41,740,630 | distance_1_to_n |
| turboquant | 4 | 1024 | 524 | 0.7250 | 0.8600 | 0.0035 | 100,488 | 2,348,920 | distance_1_to_n |
| turboquant | 8 | 1024 | 1036 | 1.0000 | 0.9975 | 0.0003 | 61,762 | 2,120,610 | distance_1_to_n |
| rabitq | 1 | 1024 | 136 | 0.1000 | 0.3850 | 0.0188 | 119,008 | 4,417,888 | distance_1_to_n |
| rabitq-algorithm1 | 4 | 1024 | 520 | 0.7500 | 0.9050 | 0.0027 | 3,329 | 6,736,844 | distance_1_to_n |
| rabitq-algorithm1 | 8 | 1024 | 1032 | 1.0000 | 0.9975 | 0.0002 | 186 | 5,330,347 | distance_1_to_n |
| rabitq-fixed-scale | 4 | 1024 | 520 | 0.7250 | 0.9100 | 0.0027 | 91,028 | 6,755,845 | distance_1_to_n |
| rabitq-fixed-scale | 8 | 1024 | 1032 | 0.9750 | 0.9925 | 0.0002 | 178,194 | 5,292,126 | distance_1_to_n |
| rabitq-windowed-scale | 4 | 1024 | 520 | 0.7500 | 0.9050 | 0.0027 | 6,275 | 6,722,690 | distance_1_to_n |
| rabitq-windowed-scale | 8 | 1024 | 1032 | 1.0000 | 0.9975 | 0.0002 | 1,638 | 5,286,321 | distance_1_to_n |

# Quantizer comparison 20261005T084158Z

Host `Darwin arm64`, Python 3.13.7, vsq 0.2.0.

Shared float32 N(0, 1) draw per dimension. Accuracy uses n_base=20000, n_query=200, k=10. Throughput uses n_speed=20000 and the fastest of the timed repeats. recall is overlap with exact squared L2. rel_mae is the mean relative error of the asymmetric distance. Both search rates are `distance_1_to_n`: the query is prepared once, then every slot is scored in one C++ pass.

| method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search pairs/s | search api |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| turboquant | 4 | 128 | 78 | 0.8050 | 0.8685 | 0.0068 | 209,271 | 47,430,793 | distance_1_to_n |
| turboquant | 8 | 128 | 142 | 0.9750 | 0.9885 | 0.0004 | 191,746 | 34,737,299 | distance_1_to_n |
| rabitq | 1 | 128 | 24 | 0.1700 | 0.2455 | 0.0548 | 1,217,937 | 117,704,526 | distance_1_to_n |
| rabitq | 4 | 128 | 72 | 0.8150 | 0.8595 | 0.0075 | 1,013,484 | 88,675,280 | distance_1_to_n |
| rabitq | 8 | 128 | 136 | 0.9350 | 0.9750 | 0.0012 | 1,647,181 | 89,769,830 | distance_1_to_n |
| rabitq-algorithm1 | 4 | 128 | 72 | 0.8050 | 0.8645 | 0.0070 | 46,188 | 95,294,818 | distance_1_to_n |
| rabitq-algorithm1 | 8 | 128 | 136 | 0.9950 | 0.9895 | 0.0004 | 5,890 | 89,635,854 | distance_1_to_n |
| rabitq-windowed-scale | 4 | 128 | 72 | 0.8050 | 0.8520 | 0.0072 | 113,678 | 88,757,528 | distance_1_to_n |
| rabitq-windowed-scale | 8 | 128 | 136 | 0.9950 | 0.9895 | 0.0004 | 27,408 | 89,904,611 | distance_1_to_n |
| turboquant | 4 | 768 | 398 | 0.8400 | 0.8690 | 0.0028 | 37,661 | 8,281,718 | distance_1_to_n |
| turboquant | 8 | 768 | 782 | 0.9800 | 0.9900 | 0.0002 | 34,406 | 6,106,637 | distance_1_to_n |
| rabitq | 1 | 768 | 104 | 0.1600 | 0.2605 | 0.0226 | 176,557 | 44,113,593 | distance_1_to_n |
| rabitq | 4 | 768 | 392 | 0.8500 | 0.8515 | 0.0031 | 147,355 | 16,462,599 | distance_1_to_n |
| rabitq | 8 | 768 | 776 | 0.9800 | 0.9850 | 0.0003 | 227,072 | 13,535,994 | distance_1_to_n |
| rabitq-algorithm1 | 4 | 768 | 392 | 0.8300 | 0.8495 | 0.0031 | 14,785 | 16,467,682 | distance_1_to_n |
| rabitq-algorithm1 | 8 | 768 | 776 | 0.9900 | 0.9890 | 0.0002 | 969 | 13,527,224 | distance_1_to_n |
| rabitq-windowed-scale | 4 | 768 | 392 | 0.8300 | 0.8515 | 0.0031 | 14,413 | 16,462,599 | distance_1_to_n |
| rabitq-windowed-scale | 8 | 768 | 776 | 0.9900 | 0.9890 | 0.0002 | 3,911 | 13,634,808 | distance_1_to_n |
| turboquant | 4 | 1024 | 526 | 0.8900 | 0.8720 | 0.0024 | 29,038 | 6,087,122 | distance_1_to_n |
| turboquant | 8 | 1024 | 1038 | 0.9950 | 0.9875 | 0.0002 | 26,510 | 4,376,168 | distance_1_to_n |
| rabitq | 1 | 1024 | 136 | 0.1400 | 0.2785 | 0.0196 | 149,221 | 33,862,434 | distance_1_to_n |
| rabitq | 4 | 1024 | 520 | 0.8350 | 0.8520 | 0.0027 | 121,883 | 12,383,579 | distance_1_to_n |
| rabitq | 8 | 1024 | 1032 | 0.9750 | 0.9795 | 0.0003 | 198,831 | 9,680,739 | distance_1_to_n |
| rabitq-algorithm1 | 4 | 1024 | 520 | 0.8550 | 0.8590 | 0.0027 | 11,266 | 12,434,911 | distance_1_to_n |
| rabitq-algorithm1 | 8 | 1024 | 1032 | 0.9850 | 0.9875 | 0.0002 | 723 | 9,684,255 | distance_1_to_n |
| rabitq-windowed-scale | 4 | 1024 | 520 | 0.8550 | 0.8590 | 0.0027 | 10,608 | 12,430,398 | distance_1_to_n |
| rabitq-windowed-scale | 8 | 1024 | 1032 | 0.9850 | 0.9875 | 0.0002 | 2,798 | 9,662,027 | distance_1_to_n |

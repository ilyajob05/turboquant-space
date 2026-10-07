# Quantizer comparison 20261006T094012Z

Host `Darwin arm64`, Python 3.13.7, vsq 0.2.0, num_threads=1.

Datasets: `gauss128` (dim 128), `gauss768` (dim 768), `gauss1024` (dim 1024), `dbpedia_openai_100K_vectors` (dim 1536). `gaussN` is a float32 N(0, 1) draw; a file name is real vectors (base = first rows, queries = last rows, held out). Accuracy uses n_base=20000, n_query=500, k=10; throughput uses n_speed=20000 codes and the fastest of 5 repeats.

Configurations: TurboQuant uses its library defaults (IVF centering, 256 k-means centroids fitted on the base, corrected estimator, no QJL). RaBitQ uses centroid=mean (`train()` on the base), float queries for 1-bit codes, and bare `rabitq` keeps the library default encode mode (`windowed_scale` at 4/8 bits). `turboquant-fastscan` re-scores k·4 candidates; `rabitq-fastscan` uses eps0=1.9.

recall is overlap with exact squared L2. rel_mae is the mean relative error of the asymmetric distance. Flat rows rank every `distance_1_to_n` estimate; FastScan rows use the index's own top-k `search` (rel_mae is undefined there). search codes/s is n_speed divided by the time of one query against n_speed codes. `refined` is the mean share of the base re-scored with the full code.

| dataset | method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search codes/s | search api | refined |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| gauss128 | turboquant | 4 | 128 | 78 | 0.8000 | 0.8662 | 0.0068 | 212,995 | 47,412,015 | distance_1_to_n | 100.0% |
| gauss128 | turboquant | 8 | 128 | 142 | 0.9820 | 0.9880 | 0.0004 | 198,424 | 37,330,845 | distance_1_to_n | 100.0% |
| gauss128 | turboquant-fastscan | 4 | 128 | 78 | 0.8000 | 0.8662 | — | 215,402 | 209,150,330 | TurboQuantFastScan.search (rerank=4) | 0.2% |
| gauss128 | rabitq | 1 | 128 | 24 | 0.1880 | 0.2580 | 0.0533 | 1,312,950 | 80,794,697 | distance_1_to_n | 100.0% |
| gauss128 | rabitq | 4 | 128 | 72 | 0.8100 | 0.8586 | 0.0072 | 118,927 | 95,503,686 | distance_1_to_n | 100.0% |
| gauss128 | rabitq | 8 | 128 | 136 | 0.9860 | 0.9898 | 0.0004 | 28,490 | 90,737,102 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-fixed-scale | 4 | 128 | 72 | 0.8180 | 0.8586 | 0.0074 | 1,080,903 | 95,106,305 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-fixed-scale | 8 | 128 | 136 | 0.9560 | 0.9740 | 0.0012 | 1,669,809 | 94,117,647 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-trained-scale | 4 | 128 | 72 | 0.8180 | 0.8584 | 0.0074 | 1,088,004 | 94,955,490 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-trained-scale | 8 | 128 | 136 | 0.9580 | 0.9748 | 0.0012 | 1,634,382 | 96,308,031 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-algorithm1 | 4 | 128 | 72 | 0.7860 | 0.8652 | 0.0070 | 46,232 | 93,403,386 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-algorithm1 | 8 | 128 | 136 | 0.9860 | 0.9898 | 0.0004 | 5,985 | 85,929,108 | distance_1_to_n | 100.0% |
| gauss128 | rabitq-fastscan | 1 | 128 | 24 | 0.1880 | 0.2580 | — | 648,121 | 23,330,417 | RaBitQFastScan.search (eps0=1.9) | 1.6% |
| gauss128 | rabitq-fastscan | 4 | 128 | 72 | 0.8060 | 0.8526 | — | 108,659 | 22,041,604 | RaBitQFastScan.search (eps0=1.9) | 5.2% |
| gauss128 | rabitq-fastscan | 8 | 128 | 136 | 0.9820 | 0.9804 | — | 27,909 | 22,917,173 | RaBitQFastScan.search (eps0=1.9) | 5.3% |
| gauss768 | turboquant | 4 | 768 | 398 | 0.8400 | 0.8652 | 0.0028 | 37,636 | 8,258,065 | distance_1_to_n | 100.0% |
| gauss768 | turboquant | 8 | 768 | 782 | 0.9920 | 0.9890 | 0.0002 | 34,436 | 6,067,501 | distance_1_to_n | 100.0% |
| gauss768 | turboquant-fastscan | 4 | 768 | 398 | 0.8400 | 0.8652 | — | 37,096 | 48,524,021 | TurboQuantFastScan.search (rerank=4) | 0.2% |
| gauss768 | rabitq | 1 | 768 | 104 | 0.1860 | 0.2774 | 0.0217 | 187,497 | 14,724,830 | distance_1_to_n | 100.0% |
| gauss768 | rabitq | 4 | 768 | 392 | 0.8320 | 0.8566 | 0.0031 | 15,338 | 16,395,687 | distance_1_to_n | 100.0% |
| gauss768 | rabitq | 8 | 768 | 776 | 0.9940 | 0.9854 | 0.0002 | 4,100 | 13,549,364 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-fixed-scale | 4 | 768 | 392 | 0.8200 | 0.8614 | 0.0031 | 159,947 | 16,450,751 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-fixed-scale | 8 | 768 | 776 | 0.9820 | 0.9814 | 0.0003 | 240,796 | 14,595,877 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-trained-scale | 4 | 768 | 392 | 0.8280 | 0.8594 | 0.0031 | 162,485 | 16,458,657 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-trained-scale | 8 | 768 | 776 | 0.9780 | 0.9816 | 0.0003 | 243,149 | 14,485,752 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-algorithm1 | 4 | 768 | 392 | 0.8340 | 0.8564 | 0.0031 | 16,226 | 16,491,445 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-algorithm1 | 8 | 768 | 776 | 0.9940 | 0.9854 | 0.0002 | 984 | 13,563,543 | distance_1_to_n | 100.0% |
| gauss768 | rabitq-fastscan | 1 | 768 | 104 | 0.1860 | 0.2774 | — | 91,514 | 21,002,888 | RaBitQFastScan.search (eps0=1.9) | 1.3% |
| gauss768 | rabitq-fastscan | 4 | 768 | 392 | 0.8300 | 0.8520 | — | 14,010 | 18,924,466 | RaBitQFastScan.search (eps0=1.9) | 4.2% |
| gauss768 | rabitq-fastscan | 8 | 768 | 776 | 0.9920 | 0.9748 | — | 3,887 | 18,425,402 | RaBitQFastScan.search (eps0=1.9) | 4.3% |
| gauss1024 | turboquant | 4 | 1024 | 526 | 0.8740 | 0.8724 | 0.0024 | 28,928 | 6,073,720 | distance_1_to_n | 100.0% |
| gauss1024 | turboquant | 8 | 1024 | 1038 | 0.9920 | 0.9902 | 0.0002 | 26,486 | 4,375,850 | distance_1_to_n | 100.0% |
| gauss1024 | turboquant-fastscan | 4 | 1024 | 526 | 0.8740 | 0.8724 | — | 28,505 | 40,268,429 | TurboQuantFastScan.search (rerank=4) | 0.2% |
| gauss1024 | rabitq | 1 | 1024 | 136 | 0.1720 | 0.2966 | 0.0188 | 153,890 | 10,854,569 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq | 4 | 1024 | 520 | 0.8380 | 0.8596 | 0.0027 | 11,176 | 12,456,853 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq | 8 | 1024 | 1032 | 0.9900 | 0.9872 | 0.0002 | 2,837 | 9,610,376 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-fixed-scale | 4 | 1024 | 520 | 0.8140 | 0.8606 | 0.0027 | 127,727 | 12,427,509 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-fixed-scale | 8 | 1024 | 1032 | 0.9880 | 0.9846 | 0.0003 | 200,569 | 10,006,464 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-trained-scale | 4 | 1024 | 520 | 0.8120 | 0.8614 | 0.0027 | 133,405 | 12,287,533 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-trained-scale | 8 | 1024 | 1032 | 0.9860 | 0.9822 | 0.0003 | 205,760 | 9,650,954 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-algorithm1 | 4 | 1024 | 520 | 0.8380 | 0.8588 | 0.0027 | 11,336 | 12,493,823 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-algorithm1 | 8 | 1024 | 1032 | 0.9900 | 0.9872 | 0.0002 | 716 | 9,522,866 | distance_1_to_n | 100.0% |
| gauss1024 | rabitq-fastscan | 1 | 1024 | 136 | 0.1720 | 0.2966 | — | 74,074 | 19,121,227 | RaBitQFastScan.search (eps0=1.9) | 1.3% |
| gauss1024 | rabitq-fastscan | 4 | 1024 | 520 | 0.8380 | 0.8568 | — | 10,038 | 18,036,965 | RaBitQFastScan.search (eps0=1.9) | 4.1% |
| gauss1024 | rabitq-fastscan | 8 | 1024 | 1032 | 0.9900 | 0.9780 | — | 2,739 | 18,107,053 | RaBitQFastScan.search (eps0=1.9) | 4.2% |
| dbpedia_openai_100K_vectors | turboquant | 4 | 1536 | 782 | 0.9680 | 0.9742 | 0.0018 | 18,741 | 4,088,168 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | turboquant | 8 | 1536 | 1550 | 1.0000 | 0.9988 | 0.0001 | 17,246 | 2,938,530 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | turboquant-fastscan | 4 | 1536 | 782 | 0.9680 | 0.9742 | — | 18,538 | 27,137,042 | TurboQuantFastScan.search (rerank=4) | 0.2% |
| dbpedia_openai_100K_vectors | rabitq | 1 | 1536 | 200 | 0.8140 | 0.8304 | 0.0154 | 87,533 | 7,444,054 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq | 4 | 1536 | 776 | 0.9640 | 0.9684 | 0.0022 | 6,635 | 8,342,458 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq | 8 | 1536 | 1544 | 1.0000 | 0.9990 | 0.0002 | 1,799 | 5,960,438 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-fixed-scale | 4 | 1536 | 776 | 0.9620 | 0.9694 | 0.0022 | 74,445 | 8,331,310 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-fixed-scale | 8 | 1536 | 1544 | 0.9980 | 0.9984 | 0.0002 | 110,467 | 5,940,080 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-trained-scale | 4 | 1536 | 776 | 0.9600 | 0.9694 | 0.0022 | 74,609 | 8,363,826 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-trained-scale | 8 | 1536 | 1544 | 0.9980 | 0.9972 | 0.0002 | 109,801 | 5,913,369 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-algorithm1 | 4 | 1536 | 776 | 0.9640 | 0.9686 | 0.0022 | 7,029 | 7,931,787 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-algorithm1 | 8 | 1536 | 1544 | 1.0000 | 0.9990 | 0.0002 | 471 | 5,958,588 | distance_1_to_n | 100.0% |
| dbpedia_openai_100K_vectors | rabitq-fastscan | 1 | 1536 | 200 | 0.8140 | 0.8304 | — | 43,092 | 19,214,603 | RaBitQFastScan.search (eps0=1.9) | 0.1% |
| dbpedia_openai_100K_vectors | rabitq-fastscan | 4 | 1536 | 776 | 0.9640 | 0.9664 | — | 6,275 | 19,125,030 | RaBitQFastScan.search (eps0=1.9) | 0.1% |
| dbpedia_openai_100K_vectors | rabitq-fastscan | 8 | 1536 | 1544 | 1.0000 | 0.9946 | — | 1,769 | 18,707,621 | RaBitQFastScan.search (eps0=1.9) | 0.1% |

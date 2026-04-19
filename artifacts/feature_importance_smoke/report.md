# HRV Feature Importance Benchmark Report

## Configuration
- Data source: synthetic fallback
- CV folds requested: 3
- Random state: 42
- RF estimators: 100
- Permutation repeats: 5
- Window seconds: 120
- Min event gap seconds: 120

## Dataset summary
- Samples: 16
- Features: 8
- Positive class count: 8
- Negative class count: 8

## Plots
![Bar charts](feature_importance_bars.png)

![Importance heatmap](feature_importance_heatmap.png)

![Rank heatmap](feature_rank_heatmap.png)

## Ranked feature lists

### RFE

| Rank | Feature | Mean Importance | Std |
|---:|---|---:|---:|
| 1 | hf_power | 0.791667 | 0.155902 |
| 2 | pnn50 | 0.750000 | 0.176777 |
| 3 | lf_power | 0.583333 | 0.294628 |
| 4 | rmssd | 0.583333 | 0.155902 |
| 5 | mean_rr | 0.541667 | 0.256851 |
| 6 | hr_mean | 0.458333 | 0.256851 |
| 7 | lf_hf_ratio | 0.458333 | 0.386401 |
| 8 | sdnn | 0.333333 | 0.212459 |

### MI

| Rank | Feature | Mean Importance | Std |
|---:|---|---:|---:|
| 1 | lf_power | 0.102044 | 0.144312 |
| 2 | hf_power | 0.096020 | 0.135792 |
| 3 | lf_hf_ratio | 0.013372 | 0.018911 |
| 4 | mean_rr | 0.005327 | 0.007534 |
| 5 | pnn50 | 0.002730 | 0.003860 |
| 6 | sdnn | 0.000000 | 0.000000 |
| 7 | rmssd | 0.000000 | 0.000000 |
| 8 | hr_mean | 0.000000 | 0.000000 |

### RF

| Rank | Feature | Mean Importance | Std |
|---:|---|---:|---:|
| 1 | hf_power | 0.169552 | 0.025868 |
| 2 | lf_hf_ratio | 0.161260 | 0.020174 |
| 3 | mean_rr | 0.141158 | 0.028511 |
| 4 | pnn50 | 0.135960 | 0.008781 |
| 5 | hr_mean | 0.110929 | 0.017708 |
| 6 | rmssd | 0.110541 | 0.017754 |
| 7 | sdnn | 0.102949 | 0.014358 |
| 8 | lf_power | 0.067651 | 0.023383 |

### PI

| Rank | Feature | Mean Importance | Std |
|---:|---|---:|---:|
| 1 | hf_power | 0.240000 | 0.121228 |
| 2 | sdnn | 0.140000 | 0.197990 |
| 3 | lf_hf_ratio | 0.066667 | 0.094281 |
| 4 | lf_power | 0.053333 | 0.075425 |
| 5 | pnn50 | 0.035556 | 0.050283 |
| 6 | rmssd | 0.033333 | 0.047140 |
| 7 | hr_mean | 0.028571 | 0.040406 |
| 8 | mean_rr | 0.012698 | 0.017958 |
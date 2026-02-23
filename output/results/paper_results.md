# Global vs Local Deep Learning for Multi-City PM2.5 Forecasting

## Experimental Setup
- Seed: 42
- Device: cpu
- Pollutants: pm25, pm10, o3, no2 (pm25 target)
- Window/Horizon: 24h / +6h
- Global split: Train=['Delhi'], Val=Hong Kong, Test=Tsuen China (zero-shot)

## Comparison Table
| City        |   Local LSTM MAE |   Global CNN-LSTM MAE |   Improvement |   Local LSTM RMSE |   Global CNN-LSTM RMSE |   Local LSTM R2 |   Global CNN-LSTM R2 | Winner     |
|:------------|-----------------:|----------------------:|--------------:|------------------:|-----------------------:|----------------:|---------------------:|:-----------|
| Delhi       |          58.8902 |               44.7699 |       23.9772 |           68.8415 |                55.4902 |         -0.0001 |               0.3502 | ⭐ WIN     |
| Hong Kong   |           2.6181 |                3.8726 |      -47.9192 |            3.1202 |                 4.2065 |         -1.5395 |              -3.6157 | Local edge |
| Tsuen China |           7.5327 |                8.7202 |      -15.7646 |            9.8906 |                11.3801 |          0.3414 |               0.1282 | Local edge |

## Zero-shot Finding
- Tsuen China: Local MAE=7.533, Global MAE=8.720, Improvement=-15.76%

## Generated Files
- output/processed_dataset.pt
- output/models/global_cnn_lstm.pt
- output/results/comparison_table.csv
- output/results/loss_curves.png
- output/results/pred_actual_scatter.png
- output/results/paper_results.md
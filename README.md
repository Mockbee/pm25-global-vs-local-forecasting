# Global vs Local Deep Learning for Multi-City PM2.5 Forecasting

End-to-end PyTorch Lightning pipeline for comparing:
- **Global CNN-LSTM** (multi-city training, zero-shot transfer)
- **Local LSTM** (per-city baseline)

## Main script
- `pm25_global_vs_local_pipeline.py`

## Install
```bash
pip install -U numpy pandas matplotlib torch lightning wandb tabulate
```

## Run
```bash
python3 pm25_global_vs_local_pipeline.py
```

## Expected input
Place OpenAQ city files under either:
- `./city_data/**/*.csv`, or
- city folders in project root (current script supports recursive discovery)

Required semantic fields (auto-mapped for OpenAQ exports):
- city, station, date.utc, parameter, value, latitude, longitude

## Outputs
Generated under `./output/`:
- `processed_dataset.pt`
- `models/global_cnn_lstm.pt`
- `results/comparison_table.csv`
- `results/loss_curves.png`
- `results/pred_actual_scatter.png`
- `results/paper_results.md`

At completion, script prints:
`PIPELINE COMPLETE - PAPER READY`

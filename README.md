# Household Energy Demand Forecasting

**Multi-step forecasting of household power consumption: classical ML vs. neural networks and LSTMs.**

This project forecasts the next **2 hours (120 steps)** of a household's global active power from the **previous 30 minutes (30 steps)** of minute-level readings. It compares how the error of eight model families grows as the forecast horizon extends. The setup follows the parallel multi-step framework of Bonetto & Rossi (2017), [*Machine Learning Approaches to Energy Consumption Forecasting in Households*](https://arxiv.org/abs/1706.09648).

📄 [Project report](docs/report.docx) · 📚 [Base paper](docs/base_paper_bonetto_rossi_2017.pdf)

## Dataset

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) contains about 2 million one-minute readings from one household over four years. The target is `Global_active_power`.

## Approach

- **Sliding window:** 30 past readings → 120 future readings
- **Parallel multi-step forecasting:** one sub-model per forecast step (e.g. `MultiOutputRegressor`), so the full 120-step vector is predicted in one pass
- **Training size:** 5,000 samples for the classical models, matching the base paper; 100,000 for the LSTM
- **Metrics:** MAE, MSE and error variance at each step of the horizon

| Model family | Notebook |
|---|---|
| Linear Regression (Ridge), Decision Tree, AdaBoost, XGBoost | [`xgb_dt_lr_adaboost.ipynb`](notebooks/xgb_dt_lr_adaboost.ipynb) |
| SVM (SVR), Random Forest | [`svm_random_forest.ipynb`](notebooks/svm_random_forest.ipynb), [`scripts/execute_svm_rf.py`](scripts/execute_svm_rf.py) |
| Feed-forward neural network (MLP) | [`neural_network.ipynb`](notebooks/neural_network.ipynb), [`neural_network_v2.ipynb`](notebooks/neural_network_v2.ipynb) |
| LSTM | [`lstm.ipynb`](notebooks/lstm.ipynb), [`lstm_v2.ipynb`](notebooks/lstm_v2.ipynb) |

## Findings

- **Linear Regression** handles very short horizons, but its MAE, MSE and error variance climb steeply as the horizon grows, because it can't capture non-linear temporal structure.
- **Decision Tree** and **AdaBoost** model non-linearity better, but their error still rises steadily with the horizon. Trees also overfit, which inflates the error variance.
- **XGBoost** gives the lowest early-horizon error of the tree models, but it degrades over long horizons because it has no explicit temporal modeling.
- **LSTM** is designed for exactly that gap. Its gated memory captures longer-range dependencies in the consumption sequence, which is why the project focuses on neural and LSTM models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- The classical-ML notebooks and `scripts/execute_svm_rf.py` download the dataset automatically through `ucimlrepo`.
- The neural-network and LSTM notebooks were run on Google Colab and read `/content/household_power_consumption.txt`. Download the dataset from UCI and update that path if you run them elsewhere.

```bash
python scripts/execute_svm_rf.py
```

## License

MIT. See [LICENSE](LICENSE).

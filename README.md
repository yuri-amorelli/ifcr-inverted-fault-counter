# IFCR — Inverted Fault Count Regression

**A novel event-driven labeling algorithm for Remaining Useful Life (RUL) estimation in predictive maintenance.**

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue)](https://doi.org/10.1109/MetroLivEnv64961.2025.11107070)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is IFCR?

<img width="1150" height="471" alt="Screenshot 2026-07-06 021327" src="https://github.com/user-attachments/assets/d4394fe6-210f-4119-908c-8778dee6a757" />

IFCR solves a fundamental problem in predictive maintenance: **how do you train a supervised model for RUL estimation when your dataset has no labels — only raw fault logs?**

The state of the art typically addresses this with deep learning architectures (LSTM + self-attention), survival analysis with complex probabilistic assumptions, or AutoML frameworks that sacrifice interpretability. IFCR takes a different approach: **reconstruct RUL regression targets directly from fault counters**, enabling supervised learning on previously unlabelable datasets while maintaining full interpretability.

Validated with hold-out testing on unseen data: 0.02%–0.63% prediction error in two out of three configurations. Cross-cycle generalization to entirely new failure events remains an open challenge, documented transparently in Part 5.

<!-- IMAGE: inserire qui il grafico principale (scatter actual-vs-predicted o dashboard semaforica) -->
<!-- ![IFCR results](docs/img/NOME_IMMAGINE.png) -->

---

## Publication

> Amorelli, Y. et al. (2025). *Predictive Maintenance for Water Supply Networks: Advanced Expert System Models for Enhanced Water Resource Management and Monitoring.*  
> IEEE International Workshop on Metrology for Living Environment (MetroLivEnv), Venice, 2025. **First Author.**  
> DOI: [https://doi.org/10.1109/MetroLivEnv64961.2025.11107070](https://doi.org/10.1109/MetroLivEnv64961.2025.11107070)

---

## How It Works

IFCR works in three steps:

1. **INDEX column** — Create an incremental counter that resets to 0 at each fault event
2. **Interval Reversal** — Invert each interval between consecutive faults, so the counter counts *down* to the next fault
3. **RUL column** — The inverted array becomes the regression target: each row reports how many recordings remain before the next fault

This transformation makes the temporal relationship between sensor readings and fault occurrence **explicit and learnable** by any supervised regression algorithm.

```
Raw data:     [0, 1, 2, 3, FAULT, 0, 1, 2, FAULT, ...]
INDEX:        [0, 1, 2, 3,   0,   0, 1, 2,   0,   ...]
RUL (IFCR):   [3, 2, 1, 0,   0,   2, 1, 0,   0,   ...]
```

---

## Results

Validated on a real industrial dataset: water pump monitored by **52 sensors** over **one year**, generating **220,000+ minute-by-minute readings** with **7 documented fault events**.

### Train/Test Split

| Model | RMSE | MAE | R² |
|---|---|---|---|
| DecisionTreeRegressor | 1294.0 | 186.0 | **0.99** |
| XGBRegressor | 823.91 | 196.19 | **0.99** |

> **Caveat:** random-split metrics on minute-by-minute data are optimistic due to temporal adjacency between train and test rows. The hold-out validation below — entire future segments withheld — is the honest benchmark. See Part 5 of the notebook for the full temporal-leakage discussion.

### K-Fold Cross Validation (MAE)

| Model | K=5 | K=10 |
|---|---|---|
| DecisionTreeRegressor | ~187 | ~186 |
| XGBRegressor | ~205 | ~201 |

### Hold-Out Validation

| Prediction Rows | Real RUL | DT Predicted | XGB Predicted | DT Error |
|---|---|---|---|---|
| [70000:70500] | 7,539 | ~7,540 | ~7,546 | **0.02%** |
| [30000:30500] | 39,317 | ~39,068 | ~39,073 | **0.63%** |
| [18000:18500] | 6,090 | ~7,235 | ~7,025 | ~18% |

> The third configuration shows higher deviation (~18%). See Part 5 of IFCR_demonstration.ipynb for a full methodological analysis of generalization limits.

---

## Dataset

The demonstration uses the publicly available **Pump Sensor Data** from Kaggle:  
[https://www.kaggle.com/datasets/nphantawee/pump-sensor-data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

Download `sensor.csv` and place it in the same directory as the notebook before running.

---

## Installation

```bash
git clone https://github.com/yuri-amorelli/ifcr-inverted-fault-counter
cd ifcr-inverted-fault-counter
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
jupyter
```

---

## Usage

### Run the demonstration notebook

```bash
jupyter notebook IFCR_demonstration.ipynb
```

The notebook walks through the complete pipeline:
1. Loading and preprocessing the dataset
2. Applying IFCR to generate RUL labels
3. Training DecisionTree and XGBoost regressors
4. Evaluating with K-fold cross-validation
5. Hold-out validation replicating publication results
6. Operational traffic light dashboard

### Use IFCR on your own dataset

Your dataset needs:
- At least one column with fault status (e.g., `machine_status`)
- At least one documented fault event

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('your_dataset.csv')

# Encode fault column as binary (1 = fault, 0 = normal)
df['fault'] = (df['machine_status'] == 'BROKEN').astype(int)

# Apply IFCR (vectorized):
# each fault starts a new segment; within each segment, count down to the next fault
segment = df['fault'].cumsum()
df['RUL'] = df.groupby(segment).cumcount(ascending=False)

# Rows after the last fault have no future fault event:
# their RUL is undefined and they should be excluded from training
last_fault_idx = df.index[df['fault'] == 1].max()
train_df = df.loc[:last_fault_idx]



# Train your model
X = train_df.drop(columns=['RUL', 'fault', 'machine_status'])
y = train_df['RUL']
model = DecisionTreeRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=7)
model.fit(X, y)
```

The vectorized labeling runs in milliseconds on 220K+ rows and correctly handles edge cases (consecutive fault rows) that a naive row-by-row implementation mislabels.

---

## Key Advantages

- **No probabilistic assumptions** — unlike survival analysis
- **No deep learning required** — simple tree-based models achieve 0.02%–0.63% hold-out error on unseen data within a known failure cycle
- **Fully interpretable** — feature importance reveals which sensors drive predictions
- **Broadly applicable** — works on any dataset with a fault status column and at least one documented fault event (cross-cycle generalization is dataset-dependent, see Part 5)
- **Operational ready** — predictions map directly to actionable maintenance states

For a full methodological analysis — including the temporal-leakage discussion and cross-cycle generalization limits — see Part 5 of the notebook.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Yuri Amorelli**  
Machine Learning Engineer — Predictive Maintenance & RUL Estimation  
[GitHub](https://github.com/yuri-amorelli) · [IEEE Publication](https://doi.org/10.1109/MetroLivEnv64961.2025.11107070)

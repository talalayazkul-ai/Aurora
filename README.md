# Aurora — ML Regression Pipeline

An end-to-end machine learning pipeline that predicts student math scores based on demographic and academic features. Built to demonstrate production-grade ML engineering practices: modular architecture, automated model selection, and a live prediction interface.

## Architecture

```
src/
├── components/
│   ├── data_ingestion.py       # CSV ingestion + train/test split
│   ├── data_transformation.py  # Preprocessing pipelines (numerical + categorical)
│   └── model_trainer.py        # Multi-model training & evaluation
├── pipeline/
│   ├── train_pipeline.py       # Orchestrates full training workflow
│   └── predict_pipeline.py     # Loads artifacts, serves predictions
├── exception.py                # Custom exception with traceback detail
├── logger.py                   # Timestamped file logging
└── utils.py                    # Object serialisation (dill)
```

## Models Compared

Nine regression algorithms evaluated and compared automatically by R² score:

| Model | Type |
|---|---|
| Linear Regression | Baseline linear |
| Ridge / Lasso | Regularised linear |
| K-Nearest Neighbours | Instance-based |
| Decision Tree | Non-parametric |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| AdaBoost | Ensemble (boosting) |
| SVR | Kernel-based |

Best model is selected automatically and serialised to `artifacts/model.pkl`. Any model scoring below R² = 0.6 raises a custom exception.

## Preprocessing Pipeline

- **Numerical features** (`reading_score`, `writing_score`): median imputation → standard scaling
- **Categorical features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`): mode imputation → one-hot encoding → standard scaling
- Pipeline built with `sklearn.ColumnTransformer` and serialised alongside the model

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train — evaluates all 9 models, saves best to artifacts/
python main.py

# Run Flask web app
python application.py
# → http://localhost:5000
```

## Tech Stack

Python · scikit-learn · pandas · NumPy · Flask · dill · seaborn

## Key Design Decisions

- **Modular components** allow each stage (ingestion, transformation, training) to be tested and replaced independently
- **Custom exception class** captures file name and line number for rapid debugging
- **Automatic model selection** removes manual tuning bias — the pipeline picks the best performer on held-out test data
- **dill serialisation** handles complex sklearn pipeline objects that pickle cannot serialise

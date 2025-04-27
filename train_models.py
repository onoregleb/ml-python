import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# синтетические данные для регрессии
X, y = make_regression(n_samples=500, n_features=5, noise=0.2, random_state=42)

# Обучаем модели
models = {
    'lr': LinearRegression(),
    'rf': RandomForestRegressor(n_estimators=50, random_state=42),
    'cb': CatBoostRegressor(verbose=0, random_state=42)
}

for name, model in models.items():
    model.fit(X, y)
    joblib.dump(model, f"model_{name}.joblib")

print("Модели обучены и сохранены: model_lr.joblib, model_rf.joblib, model_cb.joblib")

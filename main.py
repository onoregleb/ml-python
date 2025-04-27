from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from core.dto.CreateAccountDto import CreateAccountDto
from core.dto.CreatePredictionDto import CreatePredictionDto
from core.db import (
    init_db, create_user, get_user_by_username, get_account_by_user_id,
    add_balance, subtract_balance, get_models, create_model, get_model_by_id,
    create_prediction, get_prediction_by_id, get_predictions_by_user
)
import asyncio
import joblib
import numpy as np
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBasic()

# Глобальный словарь для хранения моделей
LOADED_MODELS = {}

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up...")
    await init_db()
    # Загружаем модели из файлов
    global LOADED_MODELS
    try:
        LOADED_MODELS = {
            "lr": joblib.load("model_lr.joblib"),
            "rf": joblib.load("model_rf.joblib"),
            "cb": joblib.load("model_cb.joblib")
        }
        logger.info(f"Models loaded: {list(LOADED_MODELS.keys())}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    # Инициализация моделей в БД, если нужно
    models = await get_models()
    logger.info(f"Existing models in DB: {len(models)}")
    
    if not models:
        logger.info("No models found in DB, creating new ones...")
        await create_model("Risk Assessment Model", 10.0, "Predicts investment risks", "lr")
        await create_model("Return Prediction Model", 15.0, "Predicts investment returns", "rf")
        await create_model("Premium Analysis Model", 20.0, "Advanced investment analysis", "cb")
        logger.info("Models created in DB")
    else:
        logger.info(f"Models in DB: {[m['name'] for m in models]}")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    async def _get():
        user = await get_user_by_username(credentials.username)
        if not user or user["password"] != credentials.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return user
    return asyncio.run(_get())

@app.post("/users")
async def create_user_endpoint(user_data: CreateAccountDto):
    existing = await get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user_id = await create_user(user_data.firstname, user_data.username, user_data.password)
    return {"user_id": user_id, "message": "User created successfully"}

@app.get("/users/me")
async def get_current_user_info(user = Depends(get_current_user)):
    account = await get_account_by_user_id(user["id"])
    return {
        "user_id": user["id"],
        "firstname": user["firstname"],
        "username": user["username"],
        "balance": account["balance"] if account else 0.0
    }

@app.get("/models")
async def get_models_endpoint():
    models = await get_models()
    logger.info(f"GET /models: returning {len(models)} models")
    return [{
        "id": m["id"],
        "name": m["name"],
        "cost": m["cost"],
        "description": m["description"]
    } for m in models]

@app.post("/predict")
async def create_prediction_endpoint(prediction_data: CreatePredictionDto, user = Depends(get_current_user)):
    model = await get_model_by_id(prediction_data.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    account = await get_account_by_user_id(user["id"])
    if not account or account["balance"] < model["cost"]:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    await subtract_balance(user["id"], model["cost"])
    # --- Предсказание ---
    model_key = model["file_name"] if "file_name" in model.keys() else model["id"]
    loaded_model = LOADED_MODELS.get(model_key)
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")
    # input_data должен быть dict с фичами, преобразуем в DataFrame
    try:
        X_pred = pd.DataFrame([prediction_data.input_data])
        y_pred = loaded_model.predict(X_pred)
        output_data = y_pred.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    prediction_id = await create_prediction(prediction_data.model_id, user["id"], prediction_data.input_data)
    # Можно добавить сохранение output_data в БД, если нужно
    return {"prediction_id": prediction_id, "output": output_data, "message": "Prediction created successfully"}

@app.get("/predict/{prediction_id}")
async def get_prediction_endpoint(prediction_id: str, user = Depends(get_current_user)):
    prediction = await get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if prediction["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction")
    return {
        "id": prediction["id"],
        "model_id": prediction["model_id"],
        "status": prediction["status"],
        "input_data": prediction["input_data"],
        "output_data": prediction["output_data"],
        "created_at": prediction["created_at"],
        "updated_at": prediction["updated_at"]
    }

class TopUpDto(BaseModel):
    amount: float

@app.post("/account/topup")
async def topup_account(data: TopUpDto, user = Depends(get_current_user)):
    amount = data.amount
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    await add_balance(user["id"], amount)
    account = await get_account_by_user_id(user["id"])
    return {"message": f"Account topped up successfully. New balance: {account['balance']}"}

@app.get("/predictions")
async def get_user_predictions_endpoint(user = Depends(get_current_user)):
    predictions = await get_predictions_by_user(user["id"])
    return [{
        "id": p["id"],
        "model_id": p["model_id"],
        "status": p["status"],
        "created_at": p["created_at"]
    } for p in predictions]
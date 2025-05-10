from fastapi import FastAPI, HTTPException, Depends, status, Body, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from core.dto.CreateAccountDto import CreateAccountDto
from core.dto.CreatePredictionDto import CreatePredictionDto
from core.db import (
    init_db, create_user, get_user_by_username, get_account_by_user_id,
    add_balance, subtract_balance, get_models, create_model, get_model_by_id,
    create_prediction, get_prediction_by_id, get_predictions_by_user, update_prediction
)
import asyncio
import joblib
import numpy as np
import pandas as pd
import logging
from fastapi.responses import StreamingResponse
import io
import json
import random
import time
import threading
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
security = HTTPBasic()
LOADED_MODELS = {}
app = FastAPI(lifespan=lifespan)

# Функция для обработки предсказаний в фоновом режиме
def process_prediction_task(prediction_id, model_key, input_data_list):
    """Обрабатывает предсказание в фоновом режиме с задержкой."""
    logger.info(f"Starting background task for prediction {prediction_id} with model {model_key}")
    
    try:
        # Добавляем случайную задержку от 10 до 25 секунд
        delay_time = random.uniform(10, 25)
        logger.info(f"Adding random delay of {delay_time:.2f} seconds for prediction {prediction_id}")
        time.sleep(delay_time)
        
        # Загружаем модель
        loaded_model = LOADED_MODELS.get(model_key)
        if loaded_model is None:
            logger.error(f"ML model not loaded for key: {model_key}")
            asyncio.run(update_prediction(
                prediction_id, 
                "failed", 
                json.dumps({"error": f"Model '{model_key}' not found"})
            ))
            return
        
        # Подготовка данных
        expected_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        X_pred = pd.DataFrame(input_data_list)
        
        # Проверяем наличие нужных колонок
        if not all(col in X_pred.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in X_pred.columns]
            logger.error(f"Input data missing expected columns: {missing}")
            asyncio.run(update_prediction(
                prediction_id, 
                "failed", 
                json.dumps({"error": f"Missing columns: {missing}"})
            ))
            return
            
        # Подготовка данных для предсказания
        X_pred = X_pred[expected_columns]
        X_pred = X_pred.astype(float)
        
        # Делаем предсказание
        y_pred = loaded_model.predict(X_pred)
        output_list = y_pred.tolist()
        
        # Создаем CSV с результатами
        response_df = X_pred.copy()
        response_df['target'] = output_list
        csv_buffer = io.StringIO()
        response_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Рассчитываем точность (для примера)
        accuracy = random.uniform(0.75, 0.99)  # Симуляция точности
        
        # Обновляем запись в БД
        asyncio.run(update_prediction(
            prediction_id=prediction_id,
            status="completed",
            output_data=json.dumps({
                "predictions": output_list,
                "accuracy": accuracy,
                "csv_content": csv_content
            })
        ))
        
        logger.info(f"Prediction {prediction_id} completed successfully with accuracy {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Prediction task failed: {e}", exc_info=True)
        # При ошибке обновляем статус предсказания
        asyncio.run(update_prediction(
            prediction_id, 
            "failed", 
            json.dumps({"error": str(e)})
        ))

async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = await get_user_by_username(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


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
async def create_prediction_endpoint(
    background_tasks: BackgroundTasks,
    prediction_data: CreatePredictionDto = Body(...),
    user = Depends(get_current_user)
):
    logger.info(f"Received prediction request for model_id: {prediction_data.model_id} from user {user['id']}")
    # Логируем тип и часть данных для отладки
    if isinstance(prediction_data.input_data, list):
        logger.info(f"Input data is a list of length: {len(prediction_data.input_data)}")
        if len(prediction_data.input_data) > 0:
             # Логируем только часть данных, если список большой
             log_data_sample = prediction_data.input_data[:2]
             logger.info(f"Input data sample: {log_data_sample}")
    else:
        logger.info(f"Input data is a single object: {prediction_data.input_data}")

    # 1) Загрузка информации о модели из БД
    model_record = await get_model_by_id(prediction_data.model_id)
    if not model_record:
        logger.error(f"Model not found in DB: {prediction_data.model_id}")
        raise HTTPException(status_code=404, detail="Model not found")
    model = dict(model_record) # Конвертируем Row в dict
    logger.info(f"Using model: {model['name']} (ID: {model['id']}, Cost: {model['cost']})")

    # 2) Проверка баланса
    account = await get_account_by_user_id(user["id"])
    required_cost = model["cost"]
    if not account or account["balance"] < required_cost:
        logger.warning(f"Insufficient balance for user {user['id']}. Needed: {required_cost}, Have: {account['balance'] if account else 0}")
        raise HTTPException(status_code=400, detail="Insufficient balance")

    # 3) Определяем ключ модели
    model_key = model.get("file_name")
    if not model_key:
        if model['name'] == "Risk Assessment Model": model_key = "lr"
        elif model['name'] == "Return Prediction Model": model_key = "rf"
        elif model['name'] == "Premium Analysis Model": model_key = "cb"
        else: model_key = model['id']
    logger.info(f"Using ML model with key: {model_key}")

    # 4) Проверяем, что модель доступна
    if model_key not in LOADED_MODELS:
        logger.error(f"ML model not loaded for key: {model_key}. Available keys: {list(LOADED_MODELS.keys())}")
        raise HTTPException(status_code=500, detail=f"ML model '{model_key}' not available")

    # 5) Подготовка данных
    try:
        # Гарантируем, что input_data_list будет списком для единообразия
        if isinstance(prediction_data.input_data, list):
            input_data_list = prediction_data.input_data
        else:
            input_data_list = [prediction_data.input_data]

        if not input_data_list:
             raise ValueError("Input data list cannot be empty.")

        # Проверяем наличие необходимых колонок
        expected_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        sample_df = pd.DataFrame(input_data_list)
        if not all(col in sample_df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in sample_df.columns]
            logger.error(f"Input data missing expected columns: {missing}")
            raise HTTPException(status_code=400, 
                              detail=f"Input data missing expected columns: {missing}")

    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error preparing data: {e}")

    # 6) Списываем баланс
    try:
        await subtract_balance(user["id"], required_cost)
        logger.info(f"Balance {required_cost:.2f} subtracted from user {user['id']}")
    except Exception as e:
        logger.error(f"Failed to subtract balance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update balance")

    # 7) Создаем запись предсказания со статусом "pending"
    try:
        input_data_json = json.dumps(input_data_list)
        prediction_id = await create_prediction(
            model_id=prediction_data.model_id,
            user_id=user["id"],
            input_data=input_data_json,
            output_data=None,  # Пока нет данных
            status="pending"   # Статус "pending"
        )
        logger.info(f"Created prediction record with ID: {prediction_id}")
    except Exception as e:
        logger.error(f"Failed to create prediction record: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create prediction record")

    # 8) Запускаем фоновую задачу для обработки предсказания
    try:
        # Создаем отдельный поток для обработки предсказания
        thread = threading.Thread(
            target=process_prediction_task,
            args=(prediction_id, model_key, input_data_list)
        )
        thread.daemon = True  # Поток завершится автоматически при завершении основного процесса
        thread.start()
        
        logger.info(f"Started background thread for prediction {prediction_id}")
    except Exception as e:
        logger.error(f"Failed to start background task: {e}", exc_info=True)
        # Обновляем статус предсказания на "failed"
        await update_prediction(prediction_id, "failed", json.dumps({"error": str(e)}))
        raise HTTPException(status_code=500, detail="Failed to start prediction task")

    # 9) Возвращаем ID предсказания клиенту
    return {
        "prediction_id": prediction_id,
        "status": "pending",
        "message": "Prediction queued for processing"
    }


@app.get("/predict/{prediction_id}")
async def get_prediction_endpoint(prediction_id: str, user = Depends(get_current_user)):
    logger.info(f"GET /predict/{prediction_id} requested by user {user['id']}")
    prediction = await get_prediction_by_id(prediction_id)
    if not prediction:
        logger.warning(f"Prediction {prediction_id} not found")
        raise HTTPException(status_code=404, detail="Prediction not found")
    if prediction["user_id"] != user["id"]:
        logger.warning(f"User {user['id']} not authorized to view prediction {prediction_id}")
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction")

    # Конвертируем JSON-строки обратно в объекты
    input_data = json.loads(prediction["input_data"]) if prediction["input_data"] else None
    output_data = json.loads(prediction["output_data"]) if prediction["output_data"] else None

    # Если статус "completed" и есть output_data, предоставляем возможность скачать CSV
    if prediction["status"] == "completed" and output_data and "csv_content" in output_data:
        response = {
            "id": prediction["id"],
            "model_id": prediction["model_id"],
            "status": prediction["status"],
            "input_data": input_data,
            "output_data": {
                "predictions": output_data.get("predictions", []),
                "accuracy": output_data.get("accuracy", 0.0)
            },
            "created_at": prediction["created_at"],
            "updated_at": prediction["updated_at"],
            "csv_download_url": f"/predict/{prediction_id}/download"
        }
    else:
        response = {
            "id": prediction["id"],
            "model_id": prediction["model_id"],
            "status": prediction["status"],
            "input_data": input_data,
            "output_data": output_data,
            "created_at": prediction["created_at"],
            "updated_at": prediction["updated_at"]
        }
    
    return response


@app.get("/predict/{prediction_id}/download")
async def download_prediction_csv(prediction_id: str, user = Depends(get_current_user)):
    logger.info(f"Download CSV for prediction {prediction_id} requested by user {user['id']}")
    prediction = await get_prediction_by_id(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if prediction["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to download this prediction")
    if prediction["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not completed yet")
    
    output_data = json.loads(prediction["output_data"]) if prediction["output_data"] else None
    if not output_data or "csv_content" not in output_data:
        raise HTTPException(status_code=404, detail="CSV data not available for this prediction")
    
    # Создаем StreamingResponse из CSV контента
    buffer = io.StringIO(output_data["csv_content"])
    response = StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=predictions_{prediction_id}.csv"
    
    return response


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
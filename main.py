from fastapi import FastAPI, HTTPException, Depends, status, Body
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
from fastapi.responses import StreamingResponse
import io
import json
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

    # 3) Загрузка ML модели в память
    # Исправлено: используем file_name, если он есть, иначе model['name'] или model['id'] как ключ
    model_key = model.get("file_name") # .get() безопаснее
    if not model_key:
        # Пытаемся найти по имени или ID, если file_name нет
        if model['name'] == "Risk Assessment Model": model_key = "lr"
        elif model['name'] == "Return Prediction Model": model_key = "rf"
        elif model['name'] == "Premium Analysis Model": model_key = "cb"
        else: model_key = model['id'] # Fallback
    logger.info(f"Attempting to load ML model with key: {model_key}")

    loaded_model = LOADED_MODELS.get(model_key)
    if loaded_model is None:
        logger.error(f"ML model not loaded for key: {model_key}. Available keys: {list(LOADED_MODELS.keys())}")
        # Дополнительная проверка: существует ли модель в БД, но не загружена?
        db_models = await get_models()
        db_model_keys = [m.get('file_name', m['id']) for m in db_models]
        logger.error(f"Model keys found in DB: {db_model_keys}")
        raise HTTPException(status_code=500, detail=f"ML model associated with key '{model_key}' not loaded on server.")

    # 4) Подготовка данных
    expected_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    try:
        # Гарантируем, что input_data_list будет списком для единообразия
        if isinstance(prediction_data.input_data, list):
            input_data_list = prediction_data.input_data
        else:
            input_data_list = [prediction_data.input_data] # Оборачиваем одиночный объект в список

        if not input_data_list: # Проверка на пустой список
             raise ValueError("Input data list cannot be empty.")

        X_pred = pd.DataFrame(input_data_list)

        # Проверка наличия колонок
        if not all(col in X_pred.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in X_pred.columns]
            logger.error(f"Input data missing expected columns: {missing}. Available columns: {list(X_pred.columns)}")
            raise HTTPException(status_code=400,
                                detail=f"Input data missing expected columns: {missing}. Required: {expected_columns}")

        # Убедимся, что колонки идут в нужном порядке и имеют правильный тип
        X_pred = X_pred[expected_columns]
        logger.info(f"DataFrame shape for prediction: {X_pred.shape}")
        # Попытка конвертации в float с логированием проблем
        try:
            X_pred = X_pred.astype(float)
        except ValueError as e:
             logger.error(f"Error converting input data to float: {e}. Check input data format.")
             # Попробуем найти проблемную колонку/значение (упрощенно)
             for col in expected_columns:
                 try:
                     X_pred[col].astype(float)
                 except ValueError:
                     logger.error(f"Problem likely in column '{col}'. First few values: {X_pred[col].head().tolist()}")
                     break
             raise HTTPException(status_code=400, detail=f"Invalid numeric format in input data: {e}")

    except KeyError as e:
        logger.error(f"Input data structure error, missing key: {e}")
        raise HTTPException(status_code=400, detail=f"Input data structure error, missing key: {e}. Required keys in each object: {expected_columns}")
    except ValueError as e: # Ловим ошибки типа пустых данных или неверной структуры
         logger.error(f"Data preparation error: {e}")
         raise HTTPException(status_code=400, detail=f"Error preparing data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error preparing DataFrame: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error preparing data for prediction: {e}")

    # 5) Предсказание
    try:
        y_pred = loaded_model.predict(X_pred)
        output_list = y_pred.tolist() # Конвертируем numpy array в Python list
        logger.info(f"Prediction successful. Number of predictions: {len(output_list)}")
        # Логируем пример вывода
        if output_list:
            logger.info(f"Prediction output sample: {output_list[:5]}")
    except Exception as e:
        logger.error(f"Prediction failed during model.predict(): {e}", exc_info=True)
        # Возможно, стоит вернуть деньги или не списывать их, если предсказание упало?
        # Пока что просто возвращаем ошибку.
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") # 500 т.к. проблема на сервере

    # 6) Вычитаем баланс *после* успешного предсказания
    try:
        await subtract_balance(user["id"], required_cost)
        logger.info(f"Balance {required_cost:.2f} subtracted from user {user['id']}")
    except Exception as e:
        # Что делать если списание не удалось? Предсказание уже сделано.
        # Можно попробовать "откатить" или просто залогировать серьезную ошибку.
        logger.error(f"CRITICAL: Failed to subtract balance for user {user['id']} after successful prediction! Error: {e}", exc_info=True)
        # Решаем не прерывать процесс, но логируем
        # raise HTTPException(status_code=500, detail="Failed to update balance after prediction.")

    # 7) Сохраняем ОДНУ запись предсказания в БД со ВСЕМИ входами и выходами
    try:
        # Сериализуем весь список входов и весь список выходов в JSON строки
        input_data_json = json.dumps(input_data_list)
        output_data_json = json.dumps(output_list)

        # Вызываем функцию создания предсказания ОДИН РАЗ
        prediction_id = await create_prediction(
            model_id=prediction_data.model_id,
            user_id=user["id"],
            input_data=input_data_json,    # Сохраняем JSON строку списка входов
            output_data=output_data_json,  # Сохраняем JSON строку списка выходов
            # Статус можно установить здесь или внутри create_prediction
            status="completed" # Явно указываем статус
        )
        logger.info(f"Successfully saved prediction record with ID: {prediction_id} containing {len(input_data_list)} inputs/outputs.")

    except Exception as db_err:
        logger.error(f"Database error while saving prediction bundle: {db_err}", exc_info=True)
        # Если запись в БД не удалась, баланс уже списан. Это проблема.
        # Возможно, стоит добавить логику компенсации или пометки записи как ошибочной.
        raise HTTPException(status_code=500, detail="Failed to save prediction results to database")

    # 8) Подготовка CSV файла для ответа (остается без изменений)
    try:
        # Убедимся, что длины совпадают перед созданием DataFrame для CSV
        if len(X_pred) != len(output_list):
            logger.error(f"CRITICAL: Mismatch between input count ({len(X_pred)}) and output count ({len(output_list)}) before CSV generation.")
            # Эта ошибка не должна произойти, если predict не вызвал исключение, но проверим
            raise HTTPException(status_code=500, detail="Internal error: Input and output count mismatch.")

        # Используем исходный DataFrame X_pred (он содержит все входные фичи)
        # Копируем, чтобы не изменять X_pred, если он еще где-то нужен
        response_df = X_pred.copy()
        response_df['target'] = output_list # Добавляем колонку с предсказаниями

        buffer = io.StringIO()
        response_df.to_csv(buffer, index=False)
        buffer.seek(0)
        logger.info(f"Successfully prepared CSV response buffer for prediction {prediction_id}.")

        response = StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=predictions_{prediction_id}.csv" # Добавим ID в имя файла
        # Можно добавить ID предсказания в заголовок ответа для удобства клиента
        response.headers["X-Prediction-ID"] = str(prediction_id)

        return response

    except Exception as csv_prep_err:
        logger.error(f"Error preparing CSV response for prediction {prediction_id}: {csv_prep_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV response: {csv_prep_err}")


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

    return {
        "id": prediction["id"],
        "model_id": prediction["model_id"],
        "status": prediction["status"],
        "input_data": input_data,
        "output_data": output_data,
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
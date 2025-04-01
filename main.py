from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List, Optional
import uuid
from datetime import datetime

from core.model.User import User
from core.model.Account import Account
from core.model.MLModel import MLModel
from core.model.Prediction import Prediction
from core.dto.CreateAccountDto import CreateAccountDto
from core.dto.CreatePredictionDto import CreatePredictionDto

app = FastAPI()
security = HTTPBasic()

# Mock database
users_db = {}
accounts_db = {}
models_db = {}
predictions_db = {}


# Initialize with some data
def init_mock_data():
    # Add admin user
    admin_id = str(uuid.uuid4())
    users_db[admin_id] = User(admin_id, "Admin", "admin", "admin123")
    accounts_db[admin_id] = Account(str(uuid.uuid4()), admin_id, 1000.0)

    # Add some models
    model1_id = str(uuid.uuid4())
    models_db[model1_id] = MLModel(model1_id, "Risk Assessment Model", 10.0, "Predicts investment risks")

    model2_id = str(uuid.uuid4())
    models_db[model2_id] = MLModel(model2_id, "Return Prediction Model", 15.0, "Predicts investment returns")


init_mock_data()


# Auth functions
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = None
    for u in users_db.values():
        if u.get_username() == credentials.username and u.verify_password(credentials.password):
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


# API Endpoints
@app.post("/users")
async def create_user(user_data: CreateAccountDto):
    user_id = str(uuid.uuid4())
    new_user = User(user_id, user_data.firstname, user_data.username, user_data.password)
    users_db[user_id] = new_user

    account_id = str(uuid.uuid4())
    new_account = Account(account_id, user_id, 0.0)
    accounts_db[user_id] = new_account

    return {"user_id": user_id, "message": "User created successfully"}


@app.get("/users/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    account = accounts_db.get(user.get_id())
    return {
        "user_id": user.get_id(),
        "firstname": user.get_firstname(),
        "username": user.get_username(),
        "balance": account.get_balance() if account else 0.0
    }


@app.get("/models")
async def get_models():
    return [{
        "id": model.get_id(),
        "name": model.get_name(),
        "cost": model.get_cost(),
        "description": model.get_description()
    } for model in models_db.values()]


@app.post("/predict")
async def create_prediction(prediction_data: CreatePredictionDto, user: User = Depends(get_current_user)):
    # Check if model exists
    model = models_db.get(prediction_data.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check user balance
    account = accounts_db.get(user.get_id())
    if not account or account.get_balance() < model.get_cost():
        raise HTTPException(status_code=400, detail="Insufficient balance")

    # Create prediction
    prediction_id = str(uuid.uuid4())
    new_prediction = Prediction(
        prediction_id,
        prediction_data.model_id,
        user.get_id(),
        prediction_data.input_data
    )
    predictions_db[prediction_id] = new_prediction

    # Deduct balance (in real app this should be in transaction)
    account.subtract_balance(model.get_cost())

    # In real app, here you would queue the prediction task
    # For demo, we'll just simulate it
    return {"prediction_id": prediction_id, "message": "Prediction created successfully"}


@app.get("/predict/{prediction_id}")
async def get_prediction(prediction_id: str, user: User = Depends(get_current_user)):
    prediction = predictions_db.get(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.get_user_id() != user.get_id():
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction")

    return {
        "id": prediction.get_id(),
        "model_id": prediction.get_model_id(),
        "status": prediction.get_status(),
        "input_data": prediction.get_input_data(),
        "output_data": prediction.get_output_data(),
        "created_at": prediction.created_at,
        "updated_at": prediction.updated_at
    }


@app.post("/account/topup")
async def topup_account(amount: float, user: User = Depends(get_current_user)):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    account = accounts_db.get(user.get_id())
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    account.add_balance(amount)
    return {"message": f"Account topped up successfully. New balance: {account.get_balance()}"}


@app.get("/predictions")
async def get_user_predictions(user: User = Depends(get_current_user)):
    user_predictions = [p for p in predictions_db.values() if p.get_user_id() == user.get_id()]
    return [{
        "id": p.get_id(),
        "model_id": p.get_model_id(),
        "status": p.get_status(),
        "created_at": p.created_at
    } for p in user_predictions]
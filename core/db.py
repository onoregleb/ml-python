import aiosqlite
from config.settings import DB_PATH
from datetime import datetime
import uuid
import json

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            firstname TEXT,
            username TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            user_id TEXT UNIQUE,
            balance REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT,
            cost REAL,
            description TEXT,
            file_name TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            user_id TEXT,
            input_data TEXT,
            output_data TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(model_id) REFERENCES models(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
        await db.commit()

async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db

async def create_user(firstname, username, password):
    db = await get_db()
    user_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    try:
        await db.execute(
            "INSERT INTO users (id, firstname, username, password, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, firstname, username, password, created_at)
        )
        account_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO accounts (id, user_id, balance) VALUES (?, ?, ?)",
            (account_id, user_id, 0.0)
        )
        await db.commit()
    finally:
        await db.close()
    return user_id

async def get_user_by_username(username):
    db = await get_db()
    cur = await db.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = await cur.fetchone()
    await db.close()
    return user

async def get_user_by_id(user_id):
    db = await get_db()
    cur = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = await cur.fetchone()
    await db.close()
    return user

async def get_account_by_user_id(user_id):
    db = await get_db()
    cur = await db.execute("SELECT * FROM accounts WHERE user_id = ?", (user_id,))
    account = await cur.fetchone()
    await db.close()
    return account

async def add_balance(user_id, amount):
    db = await get_db()
    await db.execute("UPDATE accounts SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
    await db.commit()
    await db.close()

async def subtract_balance(user_id, amount):
    db = await get_db()
    await db.execute("UPDATE accounts SET balance = balance - ? WHERE user_id = ?", (amount, user_id))
    await db.commit()
    await db.close()

async def get_models():
    db = await get_db()
    cur = await db.execute("SELECT * FROM models")
    models = await cur.fetchall()
    await db.close()
    return models

async def create_model(name, cost, description, file_name):
    db = await get_db()
    model_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO models (id, name, cost, description, file_name) VALUES (?, ?, ?, ?, ?)",
        (model_id, name, cost, description, file_name)
    )
    await db.commit()
    await db.close()
    return model_id

async def get_model_by_id(model_id):
    db = await get_db()
    cur = await db.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    model = await cur.fetchone()
    await db.close()
    return model

async def create_prediction(model_id, user_id, input_data):
    db = await get_db()
    prediction_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    await db.execute(
        "INSERT INTO predictions (id, model_id, user_id, input_data, output_data, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (prediction_id, model_id, user_id, json.dumps(input_data), None, "pending", now, now)
    )
    await db.commit()
    await db.close()
    return prediction_id

async def get_prediction_by_id(prediction_id):
    db = await get_db()
    cur = await db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    prediction = await cur.fetchone()
    await db.close()
    return prediction

async def get_predictions_by_user(user_id):
    db = await get_db()
    cur = await db.execute("SELECT * FROM predictions WHERE user_id = ?", (user_id,))
    predictions = await cur.fetchall()
    await db.close()
    return predictions

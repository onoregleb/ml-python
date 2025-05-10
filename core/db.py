"""
Модуль для взаимодействия с базой данных SQLite.

Данный модуль предоставляет функции для создания и управления базой данных приложения.
Включает функции для работы с пользователями, аккаунтами, моделями и предсказаниями.
Все функции являются асинхронными для эффективной работы с FastAPI.
"""

import aiosqlite
from config.settings import DB_PATH
from datetime import datetime
import uuid
import json

async def init_db():
    """
    Инициализирует базу данных, создавая необходимые таблицы, если они еще не существуют.
    
    Создает таблицы:
    - users: для хранения данных пользователей
    - accounts: для хранения баланса пользователей
    - models: для хранения информации о моделях машинного обучения
    - predictions: для хранения данных о предсказаниях
    
    Returns:
        None
    """
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
            file_name TEXT -- Поле для связи с файлом модели (lr, rf, cb)
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            user_id TEXT,
            input_data TEXT,    -- Будет хранить JSON-строку (список объектов)
            output_data TEXT,   -- Будет хранить JSON-строку (список результатов)
            status TEXT,        -- Статус выполнения (e.g., pending, completed, failed)
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(model_id) REFERENCES models(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
        await db.commit()


async def get_db():
    """
    Создает и возвращает соединение с базой данных.
    
    Устанавливает row_factory, чтобы результаты запросов возвращались в виде словарей.
    
    Returns:
        aiosqlite.Connection: Соединение с базой данных
    """
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row  # чтобы получать строки в виде dict
    return db


async def create_user(firstname, username, password):
    """
    Создает нового пользователя в базе данных и связанный с ним аккаунт.
    
    Создает запись в таблице users и связанную запись в таблице accounts с нулевым балансом.
    
    Args:
        firstname (str): Имя пользователя
        username (str): Логин пользователя (unique)
        password (str): Пароль пользователя (в открытом виде)
    
    Returns:
        str: Идентификатор созданного пользователя
    """
    db = await get_db()
    try:
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
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
    """
    Получает информацию о пользователе по логину.
    
    Args:
        username (str): Логин пользователя
    
    Returns:
        dict or None: Данные пользователя или None, если пользователь не найден
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = await cur.fetchone()
        return user
    finally:
        await db.close()


async def get_user_by_id(user_id):
    """
    Получает информацию о пользователе по идентификатору.
    
    Args:
        user_id (str): Идентификатор пользователя
    
    Returns:
        dict or None: Данные пользователя или None, если пользователь не найден
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = await cur.fetchone()
        return user
    finally:
        await db.close()


async def get_account_by_user_id(user_id):
    """
    Получает информацию об аккаунте пользователя по идентификатору пользователя.
    
    Args:
        user_id (str): Идентификатор пользователя
    
    Returns:
        dict or None: Данные аккаунта или None, если аккаунт не найден
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM accounts WHERE user_id = ?", (user_id,))
        account = await cur.fetchone()
        return account
    finally:
        await db.close()


async def add_balance(user_id, amount):
    """
    Увеличивает баланс пользователя на указанную сумму.
    
    Args:
        user_id (str): Идентификатор пользователя
        amount (float): Сумма, на которую увеличивается баланс
    
    Returns:
        None
    """
    db = await get_db()
    try:
        await db.execute("UPDATE accounts SET balance = balance + ? WHERE user_id = ?", (amount, user_id))
        await db.commit()
    finally:
        await db.close()


async def subtract_balance(user_id, amount):
    """
    Уменьшает баланс пользователя на указанную сумму.
    
    Args:
        user_id (str): Идентификатор пользователя
        amount (float): Сумма, на которую уменьшается баланс
    
    Returns:
        None
    
    Note:
        Функция не проверяет достаточность средств на балансе, эту проверку необходимо 
        выполнять перед вызовом функции.
    """
    db = await get_db()
    try:
        await db.execute("UPDATE accounts SET balance = balance - ? WHERE user_id = ?", (amount, user_id))
        await db.commit()
    finally:
        await db.close()


async def get_models():
    """
    Получает список всех доступных моделей машинного обучения.
    
    Returns:
        list: Список доступных моделей с их метаданными
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM models")
        models = await cur.fetchall()
        return models
    finally:
        await db.close()


async def create_model(name, cost, description, file_name):
    """
    Создает запись о новой модели в базе данных.
    
    Args:
        name (str): Название модели
        cost (float): Стоимость использования модели
        description (str): Описание модели
        file_name (str): Имя файла модели или ключ для доступа к модели (lr, rf, cb)
    
    Returns:
        str: Идентификатор созданной модели
    """
    db = await get_db()
    try:
        model_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO models (id, name, cost, description, file_name) VALUES (?, ?, ?, ?, ?)",
            (model_id, name, cost, description, file_name)
        )
        await db.commit()
        return model_id
    finally:
        await db.close()


async def get_model_by_id(model_id):
    """
    Получает информацию о модели по ее идентификатору.
    
    Args:
        model_id (str): Идентификатор модели
    
    Returns:
        dict or None: Данные модели или None, если модель не найдена
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        model = await cur.fetchone()
        return model
    finally:
        await db.close()


async def create_prediction(model_id: str, user_id: str, input_data: str, output_data: str, status: str):
    """
    Сохраняет запись предсказания в БД.
    
    Создает новую запись о предсказании с входными данными, результатами и указанным статусом.
    Ожидает получить input_data и output_data как готовые JSON строки.
    
    Args:
        model_id (str): Идентификатор модели, используемой для предсказания
        user_id (str): Идентификатор пользователя, создающего предсказание
        input_data (str): Входные данные для предсказания в формате JSON-строки
        output_data (str): Результаты предсказания в формате JSON-строки (может быть пустым для 'pending')
        status (str): Статус предсказания ('pending', 'completed', 'failed')
    
    Returns:
        str: Идентификатор созданного предсказания
    """
    db = await get_db()
    try:
        prediction_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()


        await db.execute(
            """
            INSERT INTO predictions
              (id, model_id, user_id, input_data, output_data, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (prediction_id, model_id, user_id, input_data, output_data, status, now, now)
        )
        await db.commit()
        return prediction_id
    finally:
        await db.close()


async def get_prediction_by_id(prediction_id):
    """
    Получает информацию о предсказании по его идентификатору.
    
    Args:
        prediction_id (str): Идентификатор предсказания
    
    Returns:
        dict or None: Данные предсказания или None, если предсказание не найдено
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
        prediction = await cur.fetchone()
        return prediction
    finally:
        await db.close()


async def get_predictions_by_user(user_id):
    """
    Получает список всех предсказаний, созданных пользователем.
    
    Args:
        user_id (str): Идентификатор пользователя
    
    Returns:
        list: Список предсказаний пользователя
    """
    db = await get_db()
    try:
        cur = await db.execute("SELECT * FROM predictions WHERE user_id = ?", (user_id,))
        predictions = await cur.fetchall()
        return predictions
    finally:
        await db.close()


async def update_prediction(prediction_id: str, status: str, output_data=None):
    """
    Обновляет статус и результаты предсказания.
    
    Args:
        prediction_id: Идентификатор предсказания
        status: Новый статус (completed, failed, etc.)
        output_data: JSON строка с результатами предсказания или None
    
    Returns:
        bool: True если обновление прошло успешно, иначе False
    """
    db = await get_db()
    try:
        now = datetime.utcnow().isoformat()
        
        if output_data is not None:
            # Обновляем и статус и результаты
            await db.execute(
                """
                UPDATE predictions 
                SET status = ?, output_data = ?, updated_at = ? 
                WHERE id = ?
                """,
                (status, output_data, now, prediction_id)
            )
        else:
            # Обновляем только статус
            await db.execute(
                """
                UPDATE predictions 
                SET status = ?, updated_at = ? 
                WHERE id = ?
                """,
                (status, now, prediction_id)
            )
            
        await db.commit()
        return True
    except Exception as e:
        print(f"Error updating prediction: {e}")
        return False
    finally:
        await db.close()
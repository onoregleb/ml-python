import streamlit as st
import requests
import json
import joblib
import io
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import pandas as pd

# Config
API_BASE_URL = "http://127.0.0.1:8000"  # Какой порт слушаем

# Инициализация состояний сессии
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'user' not in st.session_state:
    st.session_state.user = None
if 'password' not in st.session_state:
    st.session_state.password = ""

# Вспомогательная функция
def make_request(method, endpoint, data=None, auth=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, auth=auth)
        elif method == "POST":
            response = requests.post(url, json=data, auth=auth)
        else:
            return None, "Invalid method"

        if response.status_code == 200:
            return response.json(), None
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = response.text
            return None, error_detail
    except Exception as e:
        return None, str(e)


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.password = ""
    st.session_state.page = "login"
    st.rerun()


def check_balance(required_amount):
    """Проверка на достаточность баланса"""
    if not st.session_state.user:
        return False
    return st.session_state.user['balance'] >= required_amount


# Pages
def login_page():
    st.title("ML Service Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if not username or not password:
                st.error("Please enter both username and password")
                return

            with st.spinner("Logging in..."):
                response, error = make_request("GET", "/users/me", auth=(username, password))

                if error:
                    st.error(f"Login failed: {error}")
                else:
                    st.session_state.user = response
                    st.session_state.password = password
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
    with col2:
        if st.button("Go to Register"):
            st.session_state.page = "register"
            st.rerun()


def register_page():
    st.title("Register New Account")

    firstname = st.text_input("First Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            if not all([firstname, username, password]):
                st.error("Please fill all fields")
                return

            data = {
                "firstname": firstname,
                "username": username,
                "password": password
            }

            with st.spinner("Creating account..."):
                response, error = make_request("POST", "/users", data=data)

                if error:
                    st.error(f"Registration failed: {error}")
                else:
                    st.success("Registration successful! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
    with col2:
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()


def dashboard_page():
    st.title("ML Service Dashboard")

    # Проверка успешности авторизации
    if not st.session_state.user or not st.session_state.password:
        st.error("Authentication error. Please login again.")
        logout()
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"Welcome, {st.session_state.user['firstname']}!")
        st.write(f"Your balance: ${st.session_state.user['balance']:.2f}")
    with col2:
        if st.button("Logout"):
            logout()

    # Пополнение баланса
    with st.expander("💳 Top Up Balance", expanded=False):
        amount = st.number_input("Amount to add", min_value=1.0, step=1.0, value=10.0)
        if st.button("Top Up Now"):
            with st.spinner("Processing payment..."):
                _, error = make_request("POST", "/account/topup",
                                        data={"amount": amount},
                                        auth=(st.session_state.user['username'],
                                              st.session_state.password))
                if error:
                    st.error(f"Top up failed: {error}")
                else:
                    # Refresh user data
                    response, _ = make_request("GET", "/users/me",
                                               auth=(st.session_state.user['username'],
                                                     st.session_state.password))
                    st.session_state.user = response
                    st.success("Balance topped up successfully!")
                    st.rerun()

    # Получение доступных моделей
    models_response, error = make_request(
        "GET",
        "/models",
        auth=(st.session_state.user['username'], st.session_state.password)
    )
    models_info = [
        {"id": m["id"], "name": m["name"], "price": m["cost"], "description": m["description"]}
        for m in models_response
    ]

    # Секция с предсказаниями
    st.header("📊 Make a Prediction")
    
    # Кнопка для скачивания примера CSV (вне формы)
    if st.button("Download Sample CSV"):
        try:
            with open('sample_data.csv', 'rb') as f:
                st.download_button(
                    label="Download Sample CSV",
                    data=f,
                    file_name="sample_data.csv",
                    mime="text/csv"
                )
        except:
            st.warning("Sample CSV file not found. Please run model_training.py first.")

    with st.form("prediction_form"):
        model_id = st.selectbox(
            "Select Model",
            options=[m['id'] for m in models_info],
            format_func=lambda
                x: f"{next(m['name'] for m in models_info if m['id'] == x)} (${next(m['price'] for m in models_info if m['id'] == x)})"
        )

        selected_model = next(m for m in models_info if m['id'] == model_id)

        # Описание модели и цена
        st.write(f"**Description:** {selected_model['description']}")
        st.write(f"**Cost:** ${selected_model['price']}")

        # Проверка баланса
        if not check_balance(selected_model['price']):
            st.error(f"Insufficient balance. You need ${selected_model['price']} for this prediction.")

        # Добавляем выбор способа ввода данных
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload CSV"]
        )

        if input_method == "Manual Input":
            input_data = st.text_area(
                "Input Data (JSON format)",
                value='{"feature_0": 0.0, "feature_1": 0.0, "feature_2": 0.0, "feature_3": 0.0, "feature_4": 0.0}',
                height=150
            )
        else:
            # Загрузка CSV файла
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                help="CSV file should contain columns: feature_0, feature_1, feature_2, feature_3, feature_4"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
                    
                    # Проверяем наличие всех необходимых колонок
                    if all(col in df.columns for col in required_columns):
                        # Показываем предпросмотр данных
                        st.write("Preview of your data:")
                        st.dataframe(df[required_columns].head())
                        
                        # Конвертируем в словарь
                        input_data = df[required_columns].iloc[0].to_dict()
                        st.success("CSV file loaded successfully!")
                    else:
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.info("Required columns: feature_0, feature_1, feature_2, feature_3, feature_4")
                        input_data = None
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    input_data = None

        submitted = st.form_submit_button("Submit Prediction", disabled=not check_balance(selected_model['price']))

        if submitted and input_data:
            try:
                # Если это ручной ввод, парсим JSON
                if input_method == "Manual Input":
                    json_data = json.loads(input_data)
                else:
                    # Если это CSV, у нас уже есть словарь
                    json_data = input_data

                with st.spinner("Processing prediction..."):
                    data = {
                        "model_id": model_id,
                        "input_data": json_data
                    }
                    response, error = make_request("POST", "/predict",
                                                   data=data,
                                                   auth=(st.session_state.user['username'],
                                                         st.session_state.password))
                    if error:
                        st.error(f"Prediction failed: {error}")
                    else:
                        # Списываем со счета
                        st.session_state.user['balance'] -= selected_model['price']
                        st.success(f"Prediction submitted! ID: {response['prediction_id']}")
                        st.rerun()

            except json.JSONDecodeError:
                st.error("Invalid JSON format")
                return

    # История предсказаний
    st.header("🕒 Prediction History")
    with st.spinner("Loading your predictions..."):
        predictions, error = make_request("GET", "/predictions",
                                          auth=(st.session_state.user['username'],
                                                st.session_state.password))

    if error:
        st.error(f"Failed to load predictions: {error}")
    elif not predictions:
        st.info("You haven't made any predictions yet.")
    else:
        for pred in predictions:
            with st.expander(f"🔮 {pred['id']} - {pred['status'].capitalize()}", expanded=False):
                model_name = next((m['name'] for m in models_info if m['id'] == pred['model_id']), "Unknown Model")
                st.write(f"**Model:** {model_name}")
                st.write(f"**Created:** {datetime.fromisoformat(pred['created_at']).strftime('%Y-%m-%d %H:%M')}")

                if st.button("View Details", key=f"view_{pred['id']}"):
                    with st.spinner("Loading details..."):
                        full_pred, _ = make_request("GET", f"/predict/{pred['id']}",
                                                    auth=(st.session_state.user['username'],
                                                          st.session_state.password))
                        if full_pred:
                            st.json(full_pred)


def main():
    if st.session_state.logged_in and st.session_state.user:
        dashboard_page()
    else:
        if st.session_state.page == "login":
            login_page()
        else:
            register_page()


if __name__ == "__main__":
    main()
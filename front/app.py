import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import io

# Config
API_BASE_URL = "http://127.0.0.1:8000"

# --- Инициализация состояний сессии ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'user' not in st.session_state:
    st.session_state.user = None
if 'password' not in st.session_state:
    st.session_state.password = ""
if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual Input" # Значение по умолчанию
if 'prediction_status_check' not in st.session_state:
    st.session_state.prediction_status_check = {}
if 'last_status_check_time' not in st.session_state:
    st.session_state.last_status_check_time = datetime.now()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True

# --- Вспомогательная функция ---
def make_request(method, endpoint, data=None, auth=None, files=None):
    """Отправляет запрос к API и обрабатывает ответ."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        headers = {}
        if data is not None and files is None:
            # Отправляем JSON только если нет файлов
            headers["Content-Type"] = "application/json"

        response = None
        if method == "GET":
            response = requests.get(url, auth=auth, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, auth=auth, headers=headers, files=files)
        else:
            return None, "Invalid HTTP method specified"

        # Обработка кодов состояния
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    return response.json(), None
                except json.JSONDecodeError:
                    return None, "Invalid JSON received from server"
            else:
                return {"raw_text": response.text}, None
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                error_detail = response.text  # Если ответ не JSON
            st.error(f"API Error ({response.status_code}): {error_detail}")
            return None, error_detail

    except requests.exceptions.ConnectionError:
        error_msg = f"Connection Error: Could not connect to API at {url}. Is the backend running?"
        st.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        st.error(error_msg)
        return None, error_msg


# --- Функции управления сессией ---
def logout():
    """Сбрасывает состояние сессии для выхода пользователя."""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.password = ""
    st.session_state.page = "login"
    st.session_state.input_method = "Manual Input" # Сброс при выходе
    st.session_state.prediction_status_check = {}
    st.session_state.last_status_check_time = datetime.now()
    st.session_state.auto_refresh_enabled = True
    st.rerun()

def check_balance(required_amount):
    """Проверяет, достаточно ли средств на балансе пользователя."""
    if not st.session_state.user or 'balance' not in st.session_state.user:
        return False
    try:
        current_balance = float(st.session_state.user['balance'])
        return current_balance >= float(required_amount)
    except (ValueError, TypeError):
        st.error("Invalid balance format received from API.")
        return False

def check_pending_predictions(auth):
    """Проверяет статус предсказаний, которые находятся в состоянии 'pending'."""
    # Получаем текущее время
    current_time = datetime.now()
    
    # Проверяем, прошло ли достаточно времени с последней проверки (10 секунд)
    time_since_last_check = (current_time - st.session_state.last_status_check_time).total_seconds()
    if time_since_last_check < 10:
        return False  # Еще не прошло 10 секунд с последней проверки
    
    # Обновляем время последней проверки
    st.session_state.last_status_check_time = current_time
    
    # Получаем историю предсказаний
    predictions_response, error = make_request("GET", "/predictions", auth=auth)
    if error or not predictions_response:
        return False
    
    # Проверяем, есть ли предсказания в состоянии 'pending'
    has_pending = False
    status_changed = False
    
    for pred in predictions_response:
        if pred.get('status') == 'pending':
            has_pending = True
            pred_id = pred.get('id')
            
            # Проверяем статус для этого предсказания
            pred_detail, detail_error = make_request("GET", f"/predict/{pred_id}", auth=auth)
            if not detail_error and pred_detail:
                current_status = pred_detail.get('status')
                if current_status != 'pending':
                    status_changed = True  # Статус изменился
                    
    return has_pending and status_changed

# --- Страницы ---
def login_page():
    """Страница входа пользователя."""
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
                    pass
                elif response:
                    st.session_state.user = response
                    st.session_state.password = password # Сохраняем пароль для будущих запросов
                    st.session_state.logged_in = True
                    st.session_state.page = "dashboard" # Перенаправляем на дашборд
                    st.success("Login successful!")
                    st.rerun() # Перезапускаем для отображения новой страницы
                else:
                    st.error("Login failed. Unknown error.")
    with col2:
        if st.button("Go to Register"):
            st.session_state.page = "register"
            st.rerun()

def register_page():
    """Страница регистрации нового пользователя."""
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
                     pass
                elif response:
                    st.success("Registration successful! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Registration failed. Unknown error.")

    with col2:
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()

def dashboard_page():
    """Главная страница приложения после входа."""
    st.title("ML Service Dashboard")

    if not st.session_state.user or not st.session_state.password or 'username' not in st.session_state.user or 'balance' not in st.session_state.user:
        st.error("Authentication error or missing user data. Please login again.")
        logout() # Выходим, если данные некорректны
        return

    auth = (st.session_state.user['username'], st.session_state.password)
    
    # Автоматическая проверка статуса предсказаний
    if st.session_state.auto_refresh_enabled:
        if check_pending_predictions(auth):
            # Если статус изменился, перезагружаем страницу
            st.rerun()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"Welcome, {st.session_state.user.get('firstname', 'User')}!")
        try:
            balance_float = float(st.session_state.user['balance'])
            st.write(f"Your balance: ${balance_float:.2f}")
        except (ValueError, TypeError):
             st.write(f"Your balance: Invalid format ({st.session_state.user['balance']})")

    with col2:
        # Добавляем переключатель для автоматического обновления
        auto_refresh = st.checkbox("Auto refresh", value=st.session_state.auto_refresh_enabled,
                                  help="Automatically check prediction status every 10 seconds")
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh
            st.session_state.last_status_check_time = datetime.now()
    
    with col3:
        if st.button("Logout"):
            logout()
            return

    # Пополнение баланса
    with st.expander("💳 Top Up Balance", expanded=False):
        amount = st.number_input("Amount to add", min_value=1.0, step=1.0, value=10.0, key="topup_amount")
        if st.button("Top Up Now"):
            if amount <= 0:
                st.warning("Please enter a positive amount to top up.")
            else:
                with st.spinner("Processing payment..."):
                    _, error = make_request("POST", "/account/topup",
                                            data={"amount": amount},
                                            auth=auth)
                    if error:
                        pass
                    else:
                        response, fetch_error = make_request("GET", "/users/me", auth=auth)
                        if fetch_error:
                             st.warning("Could not refresh user data after top-up.")
                        elif response:
                            st.session_state.user = response
                            st.success("Balance topped up successfully!")
                            st.rerun()

    # Получение доступных моделей
    st.subheader("Available Models")
    models_info = []
    with st.spinner("Loading available models..."):
        models_response, error = make_request("GET", "/models", auth=auth)

    if error:
        st.error("Failed to load models list.")
    elif not models_response:
         st.warning("No models available at the moment.")
    else:
        models_info = [
            {"id": m.get("id"), "name": m.get("name", "Unnamed Model"),
             "cost": m.get("cost", 0.0), "description": m.get("description", "No description")}
            for m in models_response if m.get("id") is not None
        ]
        if not models_info:
             st.warning("Received model data, but could not parse it correctly or no valid models found.")


    # --- Секция с предсказаниями ---
    st.header("📊 Make a Prediction")

    try:
        with open('sample_data.csv', 'rb') as f:
            st.download_button(
                label="Download Sample CSV for Features",
                data=f,
                file_name="sample_data.csv",
                mime="text/csv",
                key="download_sample"
            )
    except FileNotFoundError:
        st.warning("Sample CSV file ('sample_data.csv') not found.")
    except Exception as e:
         st.error(f"Could not provide sample CSV: {e}")

    # Выбор способа ввода данных
    st.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV"],
        key="input_method"
    )

    # --- Сама форма предсказаний ---
    with st.form("prediction_form"):
        if not models_info:
             st.error("Cannot make predictions: No models are available.")
             st.form_submit_button("Submit Prediction", disabled=True)
             return

        model_options = {m['id']: f"{m['name']} (${m['cost']:.2f})" for m in models_info}
        if not model_options:
            st.error("Model options could not be created.")
            st.form_submit_button("Submit Prediction", disabled=True)
            return

        model_id = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options.get(x, "Invalid Model ID"),
            key="selected_model_id"
        )

        selected_model = next((m for m in models_info if m['id'] == model_id), None)

        if not selected_model:
            st.error("Selected model details not found. Please try again.")
            st.form_submit_button("Submit Prediction", disabled=True)
            return

        st.write(f"**Description:** {selected_model['description']}")
        st.write(f"**Cost per prediction:** ${selected_model['cost']:.2f}")

        input_data_for_submission = None
        num_predictions_to_process = 0
        total_cost = 0.0

        if st.session_state.input_method == "Manual Input":
            input_data_str = st.text_area(
                "Input Data (JSON format, single prediction)",
                value='{"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3, "feature_3": 0.4, "feature_4": 0.5}',
                height=150,
                key="manual_input_area",
                help="Enter the features for a single prediction as a JSON object."
            )
            # Расчет стоимости придет после Submit

        else:
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=["csv"],
                key="csv_uploader",
                help="Upload a CSV file. It should contain columns: feature_0, feature_1, ..., feature_4. All data rows will be sent as one prediction request priced at the model's cost."
            )

            input_data_from_csv_list = None

            if uploaded_file is not None:
                try:
                    csv_bytes = uploaded_file.getvalue()
                    csv_io = io.BytesIO(csv_bytes)

                    required_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
                    df = pd.read_csv(csv_io)

                    if not all(col in df.columns for col in required_columns):
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        st.error(f"Missing required columns in CSV: {', '.join(missing_columns)}")
                        st.info(f"Required columns: {', '.join(required_columns)}")
                        input_data_from_csv_list = None
                    elif df.empty:
                        st.error("The uploaded CSV file is empty or contains no data rows.")
                        input_data_from_csv_list = None
                    else:
                        # Используем только требуемые колонки
                        input_data_from_csv_list = df[required_columns].to_dict('records')
                        num_rows_in_csv = len(input_data_from_csv_list)
                        st.success(f"CSV file loaded successfully! Found {num_rows_in_csv} data rows.")
                        if num_rows_in_csv > 0:
                             st.write("Preview of the data to be used:")
                             st.dataframe(df[required_columns].head())
                        else:
                             st.warning("CSV file contains headers but no data rows.")


                except pd.errors.EmptyDataError:
                    st.error("The uploaded CSV file is empty or invalid.")
                    input_data_from_csv_list = None
                except Exception as e:
                    st.error(f"Error reading or processing CSV file: {str(e)}")
                    input_data_from_csv_list = None

                input_data_for_submission = input_data_from_csv_list
                if input_data_for_submission is not None and len(input_data_for_submission) > 0:
                     num_predictions_to_process = 1
                else:
                     num_predictions_to_process = 0


        can_afford_one = check_balance(selected_model['cost'])
        if not can_afford_one:
             st.error(f"Insufficient balance. You need at least ${selected_model['cost']:.2f} to submit a request. Please top up.")


        # --- Кнопка отправки формы ---
        submitted = st.form_submit_button(
            "Submit Prediction",
            disabled=not can_afford_one # Отключаем, если даже на 1 предсказание не хватает
        )

        # --- Логика обработки после нажатия Submit ---
        if submitted:
            current_balance_float = float(st.session_state.user['balance'])

            if st.session_state.input_method == "Manual Input":
                try:
                    parsed_data = json.loads(input_data_str)
                    if not isinstance(parsed_data, dict):
                         st.error("Invalid format: Manual input must be a JSON object (dictionary).")
                         input_data_for_submission = None
                         num_predictions_to_process = 0
                    else:
                         input_data_for_submission = parsed_data
                         num_predictions_to_process = 1

                except json.JSONDecodeError:
                    st.error("Invalid JSON format in manual input. Please check your syntax.")
                    input_data_for_submission = None
                    num_predictions_to_process = 0
                except Exception as e:
                     st.error(f"Error processing manual input: {e}")
                     input_data_for_submission = None
                     num_predictions_to_process = 0

                total_cost = selected_model['cost'] * num_predictions_to_process

            else:
                if input_data_for_submission is not None and num_predictions_to_process == 1:
                     total_cost = selected_model['cost'] * num_predictions_to_process # Умножение на 1
                elif uploaded_file is None:
                    st.error("Please upload a CSV file.")
                    input_data_for_submission = None
                    num_predictions_to_process = 0
                    total_cost = 0.0
                else:
                     st.warning("Cannot submit. Please fix the issues with the uploaded CSV file.")
                     input_data_for_submission = None
                     num_predictions_to_process = 0
                     total_cost = 0.0

            #Отправка запроса на предсказание, если данные готовы и баланс достаточен
            if input_data_for_submission is not None and num_predictions_to_process > 0:
                if current_balance_float >= total_cost:
                    with st.spinner(f"Processing request (cost: ${total_cost:.2f})..."):
                        payload = {
                            "model_id": model_id,
                            "input_data": input_data_for_submission  # dict или list of dicts
                        }
                        response, error = make_request("POST", "/predict", data=payload, auth=auth)

                        if error:
                            st.error("Prediction request failed.")
                        elif response:
                            try:
                                # Пытаемся обновить баланс
                                st.session_state.user['balance'] = current_balance_float - total_cost
                            except (ValueError, TypeError, KeyError):
                                st.warning("Could not update balance display locally.")

                            # Получаем ID предсказания
                            prediction_id = response.get("prediction_id")
                            if prediction_id:
                                # Добавляем в список для автоматической проверки статуса
                                st.session_state.prediction_status_check[prediction_id] = {
                                    "checked": False,
                                    "model_id": model_id,
                                    "model_name": selected_model['name']
                                }
                                st.success(f"Prediction request submitted successfully! ID: {prediction_id}")
                                st.info("Your prediction is being processed. You can check the status in the Prediction History section.")
                            else:
                                st.warning("Prediction request accepted but no ID was returned.")

                            st.rerun()  # Обновить баланс и историю
                        else:
                            st.error("Prediction request completed, but response is empty.")
                else:
                    st.error(
                        f"Insufficient balance. You need ${total_cost:.2f} for this request. Your current balance is ${current_balance_float:.2f}. Please top up.")
            elif input_data_for_submission is None or num_predictions_to_process == 0:
                st.warning(
                    "Cannot submit prediction. Please ensure your input data is correct (valid JSON or properly formatted CSV with data).")

    # --- История предсказаний ---
    st.header("🕒 Prediction History")
    with st.spinner("Loading your prediction history..."):
        predictions_response, error = make_request("GET", "/predictions", auth=auth)

    if error:
        st.error(f"Failed to load prediction history.")
    elif not predictions_response:
        st.info("You haven't made any predictions yet.")
    else:
        predictions = predictions_response

        sorted_predictions = sorted(predictions, key=lambda x: x.get('created_at', ''), reverse=True)

        if not sorted_predictions:
             st.info("No prediction records found.")
        else:
            # Проверяем наличие предсказаний в статусе pending
            pending_count = sum(1 for p in sorted_predictions if p.get('status') == 'pending')
            if pending_count > 0:
                st.info(f"You have {pending_count} pending prediction(s). {'Auto-refreshing is enabled.' if st.session_state.auto_refresh_enabled else 'Enable auto-refresh for automatic updates.'}")
            
            # Добавляем кнопку для ручного обновления
            if st.button("Refresh Predictions"):
                st.session_state.last_status_check_time = datetime.now() - datetime.timedelta(seconds=11)  # Форсируем проверку
                st.rerun()
            
            for pred in sorted_predictions[:10]:
                pred_id = pred.get('id', 'N/A')
                status = pred.get('status', 'Unknown').capitalize()
                model_id_hist = pred.get('model_id')
                created_at_iso = pred.get('created_at')

                model_name = next((m['name'] for m in models_info if m.get('id') == model_id_hist), "Unknown Model")

                try:
                    created_at_str = datetime.fromisoformat(created_at_iso.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S') if created_at_iso else "N/A"
                except ValueError:
                    created_at_str = str(created_at_iso)

                status_color = "blue"
                if status == "Completed":
                    status_color = "green"
                elif status == "Failed":
                    status_color = "red"
                elif status == "Pending":
                    status_color = "orange"

                expander_title = f"🔮 ID: {pred_id} | Status: :{status_color}[{status}] | Model: {model_name} | Time: {created_at_str}"

                with st.expander(expander_title, expanded=False):
                    if status == "Pending":
                        st.info("This prediction is still being processed. Please wait or refresh the page to check for updates.")
                        st.progress(0.5, text="Processing...")
                        if st.button("Check Status", key=f"check_status_{pred_id}"):
                            with st.spinner(f"Checking status for prediction {pred_id}..."):
                                pred_detail, detail_error = make_request("GET", f"/predict/{pred_id}", auth=auth)
                                if detail_error:
                                    st.error(f"Could not check status for prediction {pred_id}.")
                                elif pred_detail:
                                    updated_status = pred_detail.get('status', 'Unknown')
                                    if updated_status != 'pending':
                                        st.success(f"Status updated to: {updated_status.capitalize()}")
                                        st.rerun()  # Обновляем страницу для отображения нового статуса
                                    else:
                                        st.info("Prediction is still being processed.")
                                else:
                                    st.warning("Received no details for this prediction.")
                    
                    if st.button("View Details", key=f"view_detail_{pred_id}"):
                        with st.spinner(f"Loading details for prediction {pred_id}..."):
                            full_pred, detail_error = make_request("GET", f"/predict/{pred_id}", auth=auth)
                            if detail_error:
                                 st.error(f"Could not load details for prediction {pred_id}.")
                            elif full_pred:
                                st.write("**Input Data:**")
                                input_data_display = full_pred.get("input_data", "Not available")
                                if isinstance(input_data_display, list) and len(input_data_display) > 5:
                                    st.json(input_data_display[:5])
                                    st.write(f"... and {len(input_data_display) - 5} more items. Full data available in raw JSON.")
                                else:
                                    st.json(input_data_display)


                                st.write("**Output Data:**")
                                output_data_display = full_pred.get("output_data", "Not available or prediction not finished")
                                
                                # Проверяем, завершено ли предсказание и есть ли accuracy
                                if full_pred.get('status') == 'completed' and isinstance(output_data_display, dict):
                                    # Проверяем, есть ли информация о точности
                                    if 'accuracy' in output_data_display:
                                        accuracy = output_data_display.get('accuracy', 0)
                                        st.metric("Prediction Accuracy", f"{accuracy:.2%}")
                                    
                                    # Отображаем предсказания
                                    if 'predictions' in output_data_display:
                                        predictions = output_data_display.get('predictions', [])
                                        if predictions:
                                            st.write("**Prediction Results:**")
                                            if len(predictions) > 5:
                                                st.write(f"Showing first 5 of {len(predictions)} predictions:")
                                                st.write(predictions[:5])
                                                st.write("...")
                                            else:
                                                st.write(predictions)
                                    
                                    # Проверяем, есть ли ссылка на скачивание CSV
                                    if 'csv_download_url' in full_pred:
                                        download_url = full_pred.get('csv_download_url')
                                        csv_url = f"{API_BASE_URL}{download_url}"
                                        
                                        # Создаем кнопку для скачивания через API
                                        if st.button("Download Prediction Results CSV", key=f"download_csv_{pred_id}"):
                                            with st.spinner("Preparing CSV download..."):
                                                # Выполняем запрос для получения CSV файла
                                                response = requests.get(csv_url, auth=auth)
                                                if response.status_code == 200:
                                                    # Подготавливаем данные для скачивания
                                                    csv_data = response.content
                                                    st.download_button(
                                                        label="Save CSV File",
                                                        data=csv_data,
                                                        file_name=f"prediction_{pred_id}.csv",
                                                        mime="text/csv",
                                                        key=f"save_csv_{pred_id}"
                                                    )
                                                else:
                                                    st.error(f"Failed to download CSV: {response.status_code}")
                                
                                # Отображаем вывод в JSON формате
                                st.write("**Raw Output Data:**")
                                st.json(output_data_display)
                                
                                st.write("**Full Details (Raw JSON):**")
                                st.json(full_pred)
                            else:
                                st.warning("Received no details for this prediction.")


# --- Основная логика приложения ---
def main():
    """Определяет, какую страницу показать."""
    if st.session_state.logged_in and st.session_state.user:
        dashboard_page()
    else:
        if st.session_state.page == "register":
            register_page()
        else:
            login_page()

if __name__ == "__main__":
    main()
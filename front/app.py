import streamlit as st
import requests
import json
from datetime import datetime

# Config
API_BASE_URL = "http://127.0.0.1:8000"  # Update if your API is hosted elsewhere

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'user' not in st.session_state:
    st.session_state.user = None
if 'password' not in st.session_state:
    st.session_state.password = ""

# Model prices (should match your backend)
MODEL_PRICES = {
    "risk_model": 5,
    "return_model": 10,
    "premium_model": 20
}


# Helper functions
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
    """Check if user has enough balance"""
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

    # Check if user is properly authenticated
    if not st.session_state.user or not st.session_state.password:
        st.error("Authentication error. Please login again.")
        logout()
        return

    # User info header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"Welcome, {st.session_state.user['firstname']}!")
        st.write(f"Your balance: ${st.session_state.user['balance']:.2f}")
    with col2:
        if st.button("Logout"):
            logout()

    # Top up balance
    with st.expander("💳 Top Up Balance", expanded=False):
        amount = st.number_input("Amount to add", min_value=1.0, step=1.0, value=10.0)
        if st.button("Top Up Now"):
            with st.spinner("Processing payment..."):
                _, error = make_request("POST", "/account/topup",
                                        data=amount,
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

    # Available models with prices
    models_response, error = make_request(
        "GET",
        "/models",
        auth=(st.session_state.user['username'], st.session_state.password)
    )
    models_info = [
        {"id": m["id"], "name": m["name"], "price": m["cost"], "description": m["description"]}
        for m in models_response
    ]

    # Prediction section
    st.header("📊 Make a Prediction")
    with st.form("prediction_form"):
        # Model selection with prices
        model_id = st.selectbox(
            "Select Model",
            options=[m['id'] for m in models_info],
            format_func=lambda
                x: f"{next(m['name'] for m in models_info if m['id'] == x)} (${next(m['price'] for m in models_info if m['id'] == x)})"
        )

        selected_model = next(m for m in models_info if m['id'] == model_id)

        # Show model description and price
        st.write(f"**Description:** {selected_model['description']}")
        st.write(f"**Cost:** ${selected_model['price']}")

        # Check balance before allowing prediction
        if not check_balance(selected_model['price']):
            st.error(f"Insufficient balance. You need ${selected_model['price']} for this prediction.")

        input_data = st.text_area(
            "Input Data (JSON format)",
            value='{"data": "your input here"}',
            height=150
        )

        submitted = st.form_submit_button("Submit Prediction", disabled=not check_balance(selected_model['price']))

        if submitted:
            try:
                json_data = json.loads(input_data)

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
                        # Immediately deduct from balance in UI
                        st.session_state.user['balance'] -= selected_model['price']
                        st.success(f"Prediction submitted! ID: {response['prediction_id']}")
                        st.rerun()

            except json.JSONDecodeError:
                st.error("Invalid JSON format")
                return

    # Prediction history
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


# Main app
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
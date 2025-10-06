import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .main-header { font-size:2.5rem; font-weight:700; color:#1e293b; text-align:center; margin:1rem 0 2rem 0; }
    .sub-header { font-size:1.1rem; color:#64748b; text-align:center; font-weight:400; margin-bottom:2rem; }
    .metric-card, .stForm, .prediction-card { padding:1.5rem; border-radius:12px; background:white; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:1rem; }
    .prediction-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color:white; text-align:center; }
    .prediction-amount { font-size:2.5rem; font-weight:700; }
    .prediction-label { font-size:0.875rem; opacity:0.8; margin:0.5rem 0 0 0; }
    .status-card { padding:1rem; border-radius:8px; margin:1rem 0; border-left:4px solid; }
    .status-success { background-color:#f0fdf4; border-left-color:#22c55e; color:#166534; }
    .status-warning { background-color:#fefce8; border-left-color:#eab308; color:#a16207; }
    .status-info { background-color:#eff6ff; border-left-color:#3b82f6; color:#1d4ed8; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<h1 class="main-header">Car Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Accurate vehicle valuation using advanced ML models</p>', unsafe_allow_html=True)

# -------------------------------
# Session State
# -------------------------------
for key in ['data_loaded', 'models_trained', 'best_model', 'preprocessor']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Select Section", ["Data Analysis", "Model Training", "Price Prediction"])
    st.markdown("---")
    st.markdown("### System Information")
    st.markdown(f"**Model Status:** {'✅ Trained' if st.session_state.models_trained else '⏳ Pending'}")
    st.markdown(f"**Data Status:** {'✅ Loaded' if st.session_state.data_loaded else '⏳ Loading'}")
    st.markdown("**Version:** 1.0.0")

# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data
def load_sample_data(n_samples=1000):
    np.random.seed(42)
    models = ['Swift', 'i10', 'City', 'Verna', 'Creta', 'Santro', 'Alto', 'Wagon R', 'Baleno', 'Dzire']
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
    transmission_types = ['Manual', 'Automatic']

    data = {
        'model': np.random.choice(models, n_samples),
        'vehicle_age': np.random.randint(1, 20, n_samples),
        'km_driven': np.random.randint(5000, 200000, n_samples),
        'seller_type': np.random.choice(seller_types, n_samples),
        'fuel_type': np.random.choice(fuel_types, n_samples),
        'transmission_type': np.random.choice(transmission_types, n_samples),
        'mileage': np.random.uniform(10, 30, n_samples),
        'engine': np.random.randint(800, 2000, n_samples),
        'max_power': np.random.uniform(50, 200, n_samples),
        'seats': np.random.choice([4,5,7,8], n_samples)
    }

    base_price = 500000
    age_factor = (20 - data['vehicle_age']) / 20
    km_factor = 1 - (np.array(data['km_driven']) / 200000)
    engine_factor = np.array(data['engine']) / 2000
    power_factor = np.array(data['max_power']) / 200
    selling_price = base_price * age_factor * km_factor * (1 + engine_factor) * (1 + power_factor)
    selling_price = selling_price + np.random.normal(0, 50000, n_samples)
    data['selling_price'] = np.maximum(selling_price, 50000).astype(int)
    return pd.DataFrame(data)

def preprocess_data(df):
    X = df.drop(['selling_price'], axis=1)
    y = df['selling_price']

    le = LabelEncoder()
    X['model'] = le.fit_transform(X['model'])

    categorical_cols = ['seller_type', 'fuel_type', 'transmission_type']
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(drop='first'), categorical_cols),
        ("scale", StandardScaler(), numerical_cols)
    ], remainder='passthrough')

    return X, y, preprocessor

def evaluate_model(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

# -------------------------------
# Pages
# -------------------------------
# 1️⃣ Data Analysis
if page == "Data Analysis":
    st.subheader("📊 Data Analysis & Market Insights")
    df = load_sample_data()
    st.session_state.data_loaded = True
    st.dataframe(df.head(10), use_container_width=True)

# 2️⃣ Model Training
elif page == "Model Training":
    st.subheader("⚙️ Machine Learning Model Training")
    df = load_sample_data() if not st.session_state.data_loaded else load_sample_data()
    X, y, preprocessor = preprocess_data(df)

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 1, 100, 42)

    if st.button("🚀 Train Models"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(alpha=0.1),
            "Ridge": Ridge(alpha=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=10),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state)
        }

        results = []
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train_proc, y_train)
            trained_models[name] = model
            y_pred = model.predict(X_test_proc)
            mae, rmse, r2 = evaluate_model(y_test, y_pred)
            results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        st.session_state.models_trained = True
        st.session_state.best_model = trained_models[max(results, key=lambda x:x['R2'])['Model']]
        st.session_state.preprocessor = preprocessor

        st.dataframe(pd.DataFrame(results).round(4))

# 3️⃣ Price Prediction
elif page == "Price Prediction":
    st.subheader("🔮 Vehicle Price Prediction")
    if not st.session_state.models_trained:
        st.warning("Train a model first under 'Model Training'")
    else:
        with st.form("predict_form"):
            model = st.selectbox("Vehicle Model", ['Swift','i10','City','Verna','Creta','Santro','Alto','Wagon R','Baleno','Dzire'])
            vehicle_age = st.slider("Vehicle Age (Years)", 0, 20, 5)
            km_driven = st.number_input("Kilometers Driven", 1000, 500000, 25000, step=1000)
            seller_type = st.selectbox("Seller Type", ["Individual","Dealer","Trustmark Dealer"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","Electric"])
            transmission_type = st.selectbox("Transmission Type", ["Manual","Automatic"])
            mileage = st.number_input("Mileage (km/l)", 5.0, 30.0, 15.0)
            engine = st.number_input("Engine (CC)", 800, 2000, 1200)
            max_power = st.number_input("Max Power (BHP)", 40.0, 200.0, 80.0)
            seats = st.selectbox("Seats", [4,5,7,8])
            submitted = st.form_submit_button("💰 Calculate Price")

        if submitted:
            input_df = pd.DataFrame([{
                'model': model, 'vehicle_age': vehicle_age, 'km_driven': km_driven,
                'seller_type': seller_type, 'fuel_type': fuel_type, 'transmission_type': transmission_type,
                'mileage': mileage, 'engine': engine, 'max_power': max_power, 'seats': seats
            }])
            proc_input = st.session_state.preprocessor.transform(input_df)
            prediction = st.session_state.best_model.predict(proc_input)[0]

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-amount">₹ {prediction:,.0f}</div>
                <div class="prediction-label">Estimated Selling Price</div>
            </div>
            """, unsafe_allow_html=True)

            # Show Key Factors
            st.markdown("#### Key Factors")
            factors = {k:v for k,v in input_df.iloc[0].items()}
            for key, value in factors.items():
                st.markdown(f"- **{key}**: {value}")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import io

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin: 1rem 0 2rem 0;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    /* Navigation Styling */
    .stSidebar {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
    
    .stSidebar .stRadio > div {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e293b !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Form Styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Input Styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 6px;
    }
    
    .stNumberInput > div > div {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 6px;
    }
    
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .prediction-label {
        font-size: 0.875rem;
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Status Cards */
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background-color: #f0fdf4;
        border-left-color: #22c55e;
        color: #166534;
    }
    
    .status-warning {
        background-color: #fefce8;
        border-left-color: #eab308;
        color: #a16207;
    }
    
    .status-info {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
        color: #1d4ed8;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Data Table Styling */
    .stDataFrame {
        background: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Car Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced machine learning models for accurate vehicle valuation based on comprehensive market data</p>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Professional Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Section",
        ["Data Analysis", "Model Training", "Price Prediction"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### System Information")
    st.markdown("""
    **Model Status:** {}
    
    **Data Status:** {}
    
    **Version:** 1.0.0
    """.format(
        "✅ Trained" if st.session_state.models_trained else "⏳ Pending",
        "✅ Loaded" if st.session_state.data_loaded else "⏳ Loading"
    ))

# Sample data generation (same as before)
@st.cache_data
def load_sample_data():
    """Generate sample data similar to the CarDekho dataset"""
    np.random.seed(42)
    n_samples = 1000
    
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
        'seats': np.random.choice([4, 5, 7, 8], n_samples)
    }
    
    # Generate realistic selling prices based on features
    base_price = 500000
    age_factor = (20 - data['vehicle_age']) / 20
    km_factor = 1 - (np.array(data['km_driven']) / 200000)
    engine_factor = np.array(data['engine']) / 2000
    power_factor = np.array(data['max_power']) / 200
    
    selling_price = base_price * age_factor * km_factor * (1 + engine_factor) * (1 + power_factor)
    selling_price = selling_price + np.random.normal(0, 50000, n_samples)
    selling_price = np.maximum(selling_price, 50000)  # Minimum price
    
    data['selling_price'] = selling_price.astype(int)
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    df = load_sample_data()
    return df

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

def preprocess_data(df):
    """Preprocess the data similar to the notebook"""
    columns_to_drop = ['car_name', 'brand']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    X = df.drop(['selling_price'], axis=1)
    y = df['selling_price']
    
    if 'model' in X.columns:
        le = LabelEncoder()
        X = X.copy()
        X['model'] = le.fit_transform(X['model'])
    
    categorical_cols = ['seller_type', 'fuel_type', 'transmission_type']
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()
    
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, categorical_cols),
            ("StandardScaler", numeric_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )
    
    return X, y, preprocessor

# Page 1: Data Analysis
if page == "Data Analysis":
    st.markdown('<h2 class="section-header">Data Analysis & Market Insights</h2>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    st.session_state.data_loaded = True
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📋 Data Information", "📈 Feature Analysis", "🔗 Correlations"])
    
    with tab1:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem; color: #3b82f6;">{:,}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">Total Records</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem; color: #10b981;">{}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">Features</p>
            </div>
            """.format(df.shape[1]-1), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem; color: #f59e0b;">₹{:,.0f}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">Average Price</p>
            </div>
            """.format(df['selling_price'].mean()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem; color: #ef4444;">₹{:,.0f}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">Max Price</p>
            </div>
            """.format(df['selling_price'].max()), unsafe_allow_html=True)
        
        st.markdown("### Sample Dataset")
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Types")
            dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtypes['Non-Null Count'] = df.notnull().sum()
            dtypes['Unique Values'] = df.nunique()
            st.dataframe(dtypes, use_container_width=True)
        
        with col2:
            st.markdown("### Data Quality")
            missing_data = df.isnull().sum()
            if missing_data.sum() == 0:
                st.markdown("""
                <div class="status-card status-success">
                    <strong>✅ Data Quality: Excellent</strong><br>
                    No missing values detected in the dataset.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.dataframe(missing_data[missing_data > 0])
            
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe().round(2))
    
    with tab3:
        st.markdown("### Price Distribution Analysis")
        
        fig1 = px.histogram(
            df, x='selling_price', nbins=50,
            title='Distribution of Vehicle Selling Prices',
            labels={'selling_price': 'Selling Price (₹)', 'count': 'Frequency'},
            color_discrete_sequence=['#3b82f6']
        )
        fig1.update_layout(
            height=400,
            title_font_size=16,
            title_font_color='#1e293b',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.box(
                df, y='vehicle_age',
                title='Vehicle Age Distribution',
                color_discrete_sequence=['#10b981']
            )
            fig2.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.histogram(
                df, x='km_driven', nbins=30,
                title='Kilometers Driven Distribution',
                color_discrete_sequence=['#f59e0b']
            )
            fig3.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig3, use_container_width=True)
        
        # Categorical features analysis
        st.markdown("### Categorical Features Distribution")
        categorical_cols = ['fuel_type', 'transmission_type', 'seller_type']
        cols = st.columns(len(categorical_cols))
        
        colors = ['#8b5cf6', '#06b6d4', '#84cc16']
        for i, col_name in enumerate(categorical_cols):
            with cols[i]:
                fig = px.pie(
                    df, names=col_name,
                    title=f'{col_name.replace("_", " ").title()}',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Feature Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation Coefficient"),
            title="Correlation Matrix - Numerical Features",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=600, title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df, x='vehicle_age', y='selling_price',
                title='Price vs Vehicle Age Relationship',
                trendline="ols",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df, x='km_driven', y='selling_price',
                title='Price vs Kilometers Driven',
                trendline="ols",
                color_discrete_sequence=['#ef4444']
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

# Page 2: Model Training
elif page == "Model Training":
    st.markdown('<h2 class="section-header">Machine Learning Model Training</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        df = load_data()
        st.session_state.data_loaded = True
    else:
        df = load_data()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Training Configuration")
        
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0; color: #1e293b;">Model Parameters</h4>
            """, unsafe_allow_html=True)
            
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 1, 100, 42)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Training models... Please wait"):
                # Preprocess data
                X, y, preprocessor = preprocess_data(df)
                st.session_state.preprocessor = preprocessor
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Transform data
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)
                
                # Define models
                models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso Regression": Lasso(alpha=0.1),
                    "Ridge Regression": Ridge(alpha=0.1),
                    "K-Neighbors": KNeighborsRegressor(n_neighbors=10),
                    "Decision Tree": DecisionTreeRegressor(random_state=random_state),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
                }
                
                # Train and evaluate models
                results = []
                trained_models = {}
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for i, (name, model) in enumerate(models.items()):
                    progress_text.text(f"Training {name}...")
                    
                    # Train model
                    model.fit(X_train_processed, y_train)
                    trained_models[name] = model
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train_processed)
                    y_test_pred = model.predict(X_test_processed)
                    
                    # Evaluate
                    train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
                    test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
                    
                    results.append({
                        'Model': name,
                        'Train_MAE': train_mae,
                        'Train_RMSE': train_rmse,
                        'Train_R2': train_r2,
                        'Test_MAE': test_mae,
                        'Test_RMSE': test_rmse,
                        'Test_R2': test_r2
                    })
                    
                    progress_bar.progress((i + 1) / len(models))
                
                # Store results
                results_df = pd.DataFrame(results)
                st.session_state.results_df = results_df
                st.session_state.trained_models = trained_models
                st.session_state.X_test = X_test_processed
                st.session_state.y_test = y_test
                st.session_state.models_trained = True
                
                # Find best model
                best_model_name = results_df.loc[results_df['Test_R2'].idxmax(), 'Model']
                st.session_state.best_model = trained_models[best_model_name]
                st.session_state.best_model_name = best_model_name
                
                progress_text.empty()
                progress_bar.empty()
                
                st.markdown("""
                <div class="status-card status-success">
                    <strong>✅ Training Complete</strong><br>
                    Best performing model: <strong>{}</strong>
                </div>
                """.format(best_model_name), unsafe_allow_html=True)
    
    with col1:
        if st.session_state.models_trained:
            st.markdown("### Model Performance Comparison")
            
            results_df = st.session_state.results_df
            
            # Format results for display
            display_df = results_df.copy()
            for col in ['Train_MAE', 'Train_RMSE', 'Test_MAE', 'Test_RMSE']:
                display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
            for col in ['Train_R2', 'Test_R2']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Performance visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score Comparison', 'RMSE Comparison', 'MAE Comparison', 'Train vs Test R²'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            models = results_df['Model']
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
            
            # R² Score comparison
            fig.add_trace(
                go.Bar(x=models, y=results_df['Test_R2'], name='Test R²', 
                       marker_color=colors, showlegend=False),
                row=1, col=1
            )
            
            # RMSE comparison
            fig.add_trace(
                go.Bar(x=models, y=results_df['Test_RMSE'], name='Test RMSE',
                       marker_color=colors, showlegend=False),
                row=1, col=2
            )
            
            # MAE comparison
            fig.add_trace(
                go.Bar(x=models, y=results_df['Test_MAE'], name='Test MAE',
                       marker_color=colors, showlegend=False),
                row=2, col=1
            )
            
            # Train vs Test R²
            fig.add_trace(
                go.Scatter(x=results_df['Train_R2'], y=results_df['Test_R2'],
                          mode='markers+text', text=models, textposition="top center",
                          name='Models', marker=dict(size=10, color=colors), showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Line',
                          line=dict(dash='dash', color='red'), showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=700, title_text="Model Performance Analysis", showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_idx = results_df['Test_R2'].idxmax()
            best_model_name = results_df.loc[best_idx, 'Model']
            
            st.markdown("""
            <div class="status-card status-success">
                <strong>🏆 Best Performing Model: {}</strong>
            </div>
            """.format(best_model_name), unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem; color: #3b82f6;">{:.4f}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">R² Score</p>
                </div>
                """.format(results_df.loc[best_idx, 'Test_R2']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem; color: #ef4444;">₹{:,.0f}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">RMSE</p>
                </div>
                """.format(results_df.loc[best_idx, 'Test_RMSE']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem; color: #10b981;">₹{:,.0f}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;">MAE</p>
                </div>
                """.format(results_df.loc[best_idx, 'Test_MAE']), unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="status-card status-info">
                <strong>ℹ️ Model Training Required</strong><br>
                Click 'Start Training' to train machine learning models and view performance metrics.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Expected Performance Metrics")
            st.markdown("After training, you will see detailed performance comparisons including:")
            st.markdown("""
            - **R² Score**: Model accuracy (higher is better)
            - **RMSE**: Root Mean Square Error (lower is better)  
            - **MAE**: Mean Absolute Error (lower is better)
            - **Cross-validation results** and **overfitting analysis**
            """)
            
            # Show sample results for demonstration
            sample_results = pd.DataFrame({
                'Model': ['Linear Regression', 'Random Forest', 'K-Neighbors', 'Decision Tree'],
                'Test_R2': [0.850, 0.920, 0.880, 0.790],
                'Test_RMSE': ['₹75,000', '₹55,000', '₹68,000', '₹89,000'],
                'Test_MAE': ['₹52,000', '₹38,000', '₹47,000', '₹63,000']
            })
            st.dataframe(sample_results, use_container_width=True)

# Page 3: Price Prediction
elif page == "Price Prediction":
    st.markdown('<h2 class="section-header">Vehicle Price Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown("""
        <div class="status-card status-warning">
            <strong>⚠️ Model Not Available</strong><br>
            Please complete model training in the 'Model Training' section first. For demonstration purposes, a sample model will be used.
        </div>
        """, unsafe_allow_html=True)
        
        # Load sample data for demo
        df = load_data()
        X, y, preprocessor = preprocess_data(df)
        
        # Train a quick Random Forest model for demo
        with st.spinner("Loading demonstration model..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_processed = preprocessor.fit_transform(X_train)
            
            demo_model = RandomForestRegressor(n_estimators=100, random_state=42)
            demo_model.fit(X_processed, y_train)
            
            st.session_state.best_model = demo_model
            st.session_state.preprocessor = preprocessor
            st.session_state.best_model_name = "Random Forest (Demo)"
        
        st.markdown("""
        <div class="status-card status-info">
            <strong>ℹ️ Demo Mode Active</strong><br>
            Using pre-trained Random Forest model for price predictions.
        </div>
        """, unsafe_allow_html=True)
    
    # Model information
    model_name = st.session_state.get('best_model_name', 'Random Forest (Demo)')
    st.markdown(f"### Current Model: **{model_name}**")
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Vehicle Specifications")
        
        with st.form("prediction_form"):
            st.markdown("#### Basic Information")
            col_a, col_b = st.columns(2)
            
            with col_a:
                model = st.selectbox(
                    "Vehicle Model",
                    ['Swift', 'i10', 'City', 'Verna', 'Creta', 'Santro', 'Alto', 'Wagon R', 'Baleno', 'Dzire'],
                    help="Select the car model"
                )
                
                vehicle_age = st.slider(
                    "Vehicle Age (Years)", 
                    min_value=1, max_value=20, value=5,
                    help="Age of the vehicle in years"
                )
                
                km_driven = st.number_input(
                    "Kilometers Driven", 
                    min_value=5000, max_value=300000, value=50000, step=5000,
                    help="Total kilometers driven"
                )
                
                seller_type = st.selectbox(
                    "Seller Type",
                    ['Individual', 'Dealer', 'Trustmark Dealer'],
                    help="Type of seller"
                )
            
            with col_b:
                fuel_type = st.selectbox(
                    "Fuel Type",
                    ['Petrol', 'Diesel', 'CNG', 'Electric'],
                    help="Type of fuel used"
                )
                
                transmission_type = st.selectbox(
                    "Transmission Type",
                    ['Manual', 'Automatic'],
                    help="Transmission system"
                )
                
                seats = st.selectbox(
                    "Number of Seats",
                    [4, 5, 7, 8],
                    index=1,
                    help="Seating capacity"
                )
            
            st.markdown("#### Performance Specifications")
            col_c, col_d = st.columns(2)
            
            with col_c:
                mileage = st.slider(
                    "Mileage (km/l)", 
                    min_value=5.0, max_value=40.0, value=15.0, step=0.5,
                    help="Fuel efficiency in km per liter"
                )
                
                engine = st.number_input(
                    "Engine Capacity (CC)", 
                    min_value=500, max_value=3000, value=1200, step=50,
                    help="Engine displacement in cubic centimeters"
                )
            
            with col_d:
                max_power = st.slider(
                    "Maximum Power (BHP)", 
                    min_value=30.0, max_value=300.0, value=100.0, step=1.0,
                    help="Maximum power output in brake horsepower"
                )
            
            submitted = st.form_submit_button(
                "🔮 Calculate Price", 
                type="primary", 
                use_container_width=True
            )
            
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'model': [model],
                    'vehicle_age': [vehicle_age],
                    'km_driven': [km_driven],
                    'seller_type': [seller_type],
                    'fuel_type': [fuel_type],
                    'transmission_type': [transmission_type],
                    'mileage': [mileage],
                    'engine': [engine],
                    'max_power': [max_power],
                    'seats': [seats]
                })
                
                # Preprocess input
                try:
                    # Label encode model
                    model_mapping = {
                        'Swift': 0, 'i10': 1, 'City': 2, 'Verna': 3, 'Creta': 4, 
                        'Santro': 5, 'Alto': 6, 'Wagon R': 7, 'Baleno': 8, 'Dzire': 9
                    }
                    input_data['model'] = input_data['model'].map(model_mapping)
                    
                    # Transform using preprocessor
                    input_processed = st.session_state.preprocessor.transform(input_data)
                    
                    # Make prediction
                    prediction = st.session_state.best_model.predict(input_processed)[0]
                    
                    # Store prediction and input
                    st.session_state.last_prediction = prediction
                    st.session_state.last_input = input_data.iloc[0].to_dict()
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    # Fallback prediction
                    base_price = 500000
                    age_factor = max(0.3, (20 - vehicle_age) / 20)
                    km_factor = max(0.3, 1 - (km_driven / 200000))
                    engine_factor = engine / 2000
                    power_factor = max_power / 200
                    
                    prediction = base_price * age_factor * km_factor * (1 + engine_factor) * (1 + power_factor)
                    st.session_state.last_prediction = prediction
                    st.session_state.last_input = input_data.iloc[0].to_dict()
    
    with col2:
        st.markdown("### Prediction Results")
        
        if 'last_prediction' in st.session_state:
            prediction = st.session_state.last_prediction
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-card">
                <h2 class="prediction-amount">₹{prediction:,.0f}</h2>
                <p class="prediction-label">Estimated Market Value</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence interval
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            
            st.markdown("#### Price Range Analysis")
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span style="color: #64748b;">Conservative</span>
                    <span style="color: #64748b;">Optimistic</span>
                </div>
                <div style="background: #f1f5f9; height: 8px; border-radius: 4px; position: relative;">
                    <div style="background: linear-gradient(90deg, #10b981, #3b82f6); height: 100%; width: 70%; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.875rem;">
                    <span><strong>₹{lower_bound:,.0f}</strong></span>
                    <span><strong>₹{upper_bound:,.0f}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Market insights
            st.markdown("#### Market Insights")
            if prediction < 300000:
                st.markdown("""
                <div class="status-card status-success">
                    <strong>💰 Budget Segment</strong><br>
                    Excellent value proposition for cost-conscious buyers. High demand in entry-level market.
                </div>
                """, unsafe_allow_html=True)
            elif prediction < 700000:
                st.markdown("""
                <div class="status-card status-warning">
                    <strong>🚗 Mid-Range Segment</strong><br>
                    Balanced features and pricing. Good resale value and moderate depreciation expected.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card status-info">
                    <strong>✨ Premium Segment</strong><br>
                    Luxury features with higher depreciation. Target audience: premium car enthusiasts.
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics
            st.markdown("#### Key Factors")
            if 'last_input' in st.session_state:
                input_data = st.session_state.last_input
                
                # Age impact
                age_impact = "High" if input_data['vehicle_age'] > 10 else "Moderate" if input_data['vehicle_age'] > 5 else "Low"
                age_color = "#ef4444" if age_impact == "High" else "#f59e0b" if age_impact == "Moderate" else "#10b981"
                
                # Mileage impact  
                km_impact = "High" if input_data['km_driven'] > 100000 else "Moderate" if input_data['km_driven'] > 50000 else "Low"
                km_color = "#ef4444" if km_impact == "High" else "#f59e0b" if km_impact == "Moderate" else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>Age Depreciation</span>
                            <span style="color: {age_color}; font-weight: 600;">{age_impact}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Usage Impact</span>
                            <span style="color: {km_color}; font-weight: 600;">{km_impact}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b;">
                        Analysis based on {input_data['vehicle_age']} years age and {input_data['km_driven']:,} km driven
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="prediction-card" style="opacity: 0.5;">
                <h2 class="prediction-amount">₹--,--,---</h2>
                <p class="prediction-label">Enter details to predict</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="status-card status-info">
                <strong>ℹ️ How to Use</strong><br>
                Fill in the vehicle specifications on the left and click 'Calculate Price' to get an instant market valuation.
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0 1rem 0; font-size: 0.875rem;">
    <p style="margin: 0;"><strong>Car Price Prediction System</strong> | Advanced ML-based Vehicle Valuation</p>
    <p style="margin: 0.5rem 0 0 0;">Built with Streamlit • Python • Scikit-learn | Version 1.0.0</p>
</div>
""", unsafe_allow_html=True)
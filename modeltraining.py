"""
Used Car Price Prediction Model Training Script
Professional implementation with proper structure and error handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

warnings.filterwarnings("ignore")

class CarPricePredictionModel:
    """
    A comprehensive class for training and evaluating used car price prediction models
    """
    
    def __init__(self, data_path="./data/cardekho_imputated.csv"):
        """
        Initialize the model with data loading
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.best_model = None
        self.label_encoder = None
        
    def load_data(self):
        """Load and perform initial data inspection"""
        try:
            self.df = pd.read_csv(self.data_path, index_col=[0])
            print(f"Dataset loaded successfully with shape: {self.df.shape}")
            print("\nDataset Info:")
            print(self.df.info())
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found!")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Perform data cleaning operations"""
        print("\n=== Data Cleaning ===")
        
        # Check for missing values
        print("Missing values per column:")
        print(self.df.isnull().sum())
        
        # Remove unnecessary columns
        columns_to_drop = ['car_name', 'brand']
        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        
        if existing_columns:
            self.df.drop(existing_columns, axis=1, inplace=True)
            print(f"Dropped columns: {existing_columns}")
        
        # Display basic statistics
        print(f"\nDataset shape after cleaning: {self.df.shape}")
        print(f"Unique models: {len(self.df['model'].unique())}")
        
    def feature_analysis(self):
        """Analyze and categorize features"""
        print("\n=== Feature Analysis ===")
        
        # Categorize features
        num_features = [feature for feature in self.df.columns if self.df[feature].dtype != 'O']
        cat_features = [feature for feature in self.df.columns if self.df[feature].dtype == 'O']
        discrete_features = [feature for feature in num_features if len(self.df[feature].unique()) <= 25]
        continuous_features = [feature for feature in num_features if feature not in discrete_features]
        
        print(f'Numerical Features ({len(num_features)}): {num_features}')
        print(f'Categorical Features ({len(cat_features)}): {cat_features}')
        print(f'Discrete Features ({len(discrete_features)}): {discrete_features}')
        print(f'Continuous Features ({len(continuous_features)}): {continuous_features}')
        
        return num_features, cat_features, discrete_features, continuous_features
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("\n=== Feature Preparation ===")
        
        # Separate features and target
        self.X = self.df.drop(['selling_price'], axis=1)
        self.y = self.df['selling_price']
        
        # Handle model column with Label Encoding
        self.label_encoder = LabelEncoder()
        self.X['model'] = self.label_encoder.fit_transform(self.X['model'])
        
        # Define categorical columns for one-hot encoding
        onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
        num_features = self.X.select_dtypes(exclude="object").columns
        
        # Create preprocessor
        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("OneHotEncoder", oh_transformer, onehot_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ],
            remainder='passthrough'
        )
        
        # Transform features
        self.X = self.preprocessor.fit_transform(self.X)
        print(f"Features prepared with shape: {self.X.shape}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"\nData split completed:")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
    
    @staticmethod
    def evaluate_model(y_true, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            tuple: MAE, RMSE, R2 Score
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2_square = r2_score(y_true, y_pred)
        return mae, rmse, r2_square
    
    def train_baseline_models(self):
        """Train and evaluate baseline models"""
        print("\n=== Baseline Model Training ===")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
        }
        
        model_results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Evaluate performance
            train_mae, train_rmse, train_r2 = self.evaluate_model(self.y_train, y_train_pred)
            test_mae, test_rmse, test_r2 = self.evaluate_model(self.y_test, y_test_pred)
            
            model_results[name] = {
                'model': model,
                'train_metrics': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
                'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
            }
            
            print(f"\n{name}")
            print('Training Performance:')
            print(f"- RMSE: {train_rmse:.4f}")
            print(f"- MAE: {train_mae:.4f}")
            print(f"- R² Score: {train_r2:.4f}")
            
            print('Test Performance:')
            print(f"- RMSE: {test_rmse:.4f}")
            print(f"- MAE: {test_mae:.4f}")
            print(f"- R² Score: {test_r2:.4f}")
            print('=' * 50)
        
        return model_results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        print("\n=== Hyperparameter Tuning ===")
        
        # Define parameter grids
        knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}
        rf_params = {
            "max_depth": [5, 8, 15, None, 10],
            "max_features": [5, 7, "auto", 8],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators": [100, 200, 500, 1000]
        }
        
        models_to_tune = [
            ('KNN', KNeighborsRegressor(), knn_params),
            ("RF", RandomForestRegressor(), rf_params)
        ]
        
        best_params = {}
        tuned_models = {}
        
        for name, model, params in models_to_tune:
            print(f"\nTuning {name}...")
            
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=50,  # Reduced for faster execution
                cv=3,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(self.X_train, self.y_train)
            best_params[name] = random_search.best_params_
            tuned_models[name] = random_search.best_estimator_
            
            print(f"Best parameters for {name}: {random_search.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return best_params, tuned_models
    
    def train_final_models(self, tuned_models):
        """Train final models with best parameters"""
        print("\n=== Final Model Training ===")
        
        final_results = {}
        
        for name, model in tuned_models.items():
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Evaluate performance
            train_mae, train_rmse, train_r2 = self.evaluate_model(self.y_train, y_train_pred)
            test_mae, test_rmse, test_r2 = self.evaluate_model(self.y_test, y_test_pred)
            
            final_results[name] = {
                'model': model,
                'train_metrics': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
                'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
            }
            
            print(f"\n{name} - Final Performance:")
            print('Training Performance:')
            print(f"- RMSE: {train_rmse:.4f}")
            print(f"- MAE: {train_mae:.4f}")
            print(f"- R² Score: {train_r2:.4f}")
            
            print('Test Performance:')
            print(f"- RMSE: {test_rmse:.4f}")
            print(f"- MAE: {test_mae:.4f}")
            print(f"- R² Score: {test_r2:.4f}")
            print('=' * 50)
        
        return final_results
    
    def select_best_model(self, final_results):
        """Select the best performing model based on test R² score"""
        best_score = -float('inf')
        best_model_name = None
        
        for name, results in final_results.items():
            test_r2 = results['test_metrics']['r2']
            if test_r2 > best_score:
                best_score = test_r2
                best_model_name = name
        
        self.best_model = final_results[best_model_name]['model']
        print(f"\nBest Model Selected: {best_model_name}")
        print(f"Test R² Score: {best_score:.4f}")
        
        return best_model_name, self.best_model
    
    def save_model_artifacts(self, model_name="best_model"):
        """Save trained model and preprocessors"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save the best model
        joblib.dump(self.best_model, f'models/{model_name}.pkl')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        # Save feature names for reference
        feature_info = {
            'categorical_columns': ['seller_type', 'fuel_type', 'transmission_type'],
            'numerical_columns': ['model', 'year', 'km_driven', 'owner']
        }
        joblib.dump(feature_info, 'models/feature_info.pkl')
        
        print(f"\nModel artifacts saved successfully in 'models/' directory")
    
    def generate_model_report(self, final_results):
        """Generate a comprehensive model performance report"""
        print("\n" + "="*60)
        print("USED CAR PRICE PREDICTION - MODEL PERFORMANCE REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in final_results.items():
            comparison_data.append({
                'Model': name,
                'Train_R2': results['train_metrics']['r2'],
                'Test_R2': results['test_metrics']['r2'],
                'Train_RMSE': results['train_metrics']['rmse'],
                'Test_RMSE': results['test_metrics']['rmse'],
                'Train_MAE': results['train_metrics']['mae'],
                'Test_MAE': results['test_metrics']['mae'],
                'Overfitting': results['train_metrics']['r2'] - results['test_metrics']['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.round(4))
        
        # Best model summary
        best_model_row = comparison_df.iloc[0]
        print(f"\nBest Model: {best_model_row['Model']}")
        print(f"Test R² Score: {best_model_row['Test_R2']:.4f}")
        print(f"Test RMSE: {best_model_row['Test_RMSE']:.4f}")
        print(f"Overfitting Score: {best_model_row['Overfitting']:.4f}")
        
        return comparison_df
    
    def run_complete_pipeline(self):
        """Execute the complete model training pipeline"""
        print("Starting Used Car Price Prediction Model Training Pipeline...")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Feature analysis
        self.feature_analysis()
        
        # Step 4: Prepare features
        self.prepare_features()
        
        # Step 5: Split data
        self.split_data()
        
        # Step 6: Train baseline models
        baseline_results = self.train_baseline_models()
        
        # Step 7: Hyperparameter tuning
        best_params, tuned_models = self.hyperparameter_tuning()
        
        # Step 8: Train final models
        final_results = self.train_final_models(tuned_models)
        
        # Step 9: Select best model
        best_model_name, _ = self.select_best_model(final_results)
        
        # Step 10: Generate report
        comparison_df = self.generate_model_report(final_results)
        
        # Step 11: Save model artifacts
        self.save_model_artifacts()
        
        print("\n" + "="*60)
        print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True


def main():
    """Main execution function"""
    # Initialize and run the model training pipeline
    model_trainer = CarPricePredictionModel()
    success = model_trainer.run_complete_pipeline()
    
    if success:
        print("\nModel training completed successfully!")
        print("You can now run the Streamlit app using: streamlit run app.py")
    else:
        print("Model training failed. Please check the data file and try again.")


if __name__ == "__main__":
    main()
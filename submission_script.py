# Environment: This code was developed and tested on Kaggle Notebooks
# Hardware: 2 CPU cores, 16GB RAM, no GPU
# Date: April 1, 2025

# Import all required libraries at the top
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
SEED = 42

# Data sources (from competition page)
TRAIN_PATH = '/kaggle/input/international-womens-day-challengedataset/Train.csv'
TEST_PATH = '/kaggle/input/international-womens-day-challengedataset/Test.csv'
SUBMISSION_PATH = '/kaggle/input/international-womens-day-challengedataset/SampleSubmission.csv'

def load_and_prepare_data():
    """Load and align training and test datasets."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    
    target = train['target']
    train, test = train.align(test, join='inner', axis=1)
    
    # Add separator and concatenate
    train['separator'] = 0
    test['separator'] = 1
    combined = pd.concat([train, test])
    
    # Split back and add target
    train = combined[combined.separator == 0].drop('separator', axis=1)
    test = combined[combined.separator == 1].drop('separator', axis=1)
    train['target'] = target
    
    X = train.drop(['ward', 'ADM4_PCODE', 'target'], axis=1)
    y = target
    X_test = test.drop(['ward', 'ADM4_PCODE'], axis=1)
    
    return X, y, X_test, submission

def optimize_catboost(trial, X, y):
    """Objective function for CatBoost hyperparameter optimization."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'random_seed': SEED
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = CatBoostRegressor(**params, logging_level='Silent')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return mean_squared_error(y_val, y_pred)

def train_base_models(X, y, X_test, best_params):
    """Train base models with different random states."""
    # CatBoost with optimized parameters
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=29)
    catboost_pred = CatBoostRegressor(**best_params, logging_level='Silent', random_seed=SEED).fit(X_train, y_train).predict(X_test)
    
    # StackingRegressor with random_state=65
    estimators = [
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=SEED)),
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=SEED)),
        ('lgb', LGBMRegressor(random_state=SEED)),
        ('svr', SVR()),
        ('lasso', Lasso(random_state=SEED)),
        ('kneiba', KNeighborsRegressor()),
        ('cat', CatBoostRegressor(logging_level='Silent', random_seed=SEED))
    ]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=65)
    stack_65_pred = StackingRegressor(estimators=estimators, 
                                    final_estimator=CatBoostRegressor(logging_level='Silent', random_seed=SEED)).fit(X_train, y_train).predict(X_test)
    
    # StackingRegressor with random_state=27
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=27)
    stack_27_pred = StackingRegressor(estimators=estimators, 
                                    final_estimator=CatBoostRegressor(logging_level='Silent', random_seed=SEED)).fit(X_train, y_train).predict(X_test)
    
    return catboost_pred, stack_65_pred, stack_27_pred

def blend_predictions(catboost_pred, stack_65_pred, stack_27_pred, X_test):
    """Blend predictions from different models."""
    stack = [x * 0.5 + y * 0.5 for x, y in zip(stack_65_pred, stack_27_pred)]
    stack_2 = [x * 0.5 + y * 0.5 for x, y in zip(stack, catboost_pred)]
    
    # Final blending with Ridge and CatBoost
    X = X_test.copy()
    y = stack_2
    
    ridge = Ridge(random_state=SEED)
    ridge.fit(X, y)
    ridge_pred = ridge.predict(X)
    
    cat = CatBoostRegressor(verbose=False, random_seed=SEED)
    cat.fit(X, y)
    cat_pred = cat.predict(X)
    
    return [x * 0.5 + y * 0.5 for x, y in zip(ridge_pred, cat_pred)]

def main():
    # Load and prepare data
    X, y, X_test, submission = load_and_prepare_data()
    
    # Optimize CatBoost hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optimize_catboost(trial, X, y), n_trials=50)
    print(f"Best CatBoost parameters: {study.best_params}")
    print(f"Best MSE: {study.best_value}")
    
    # Train base models
    catboost_pred, stack_65_pred, stack_27_pred = train_base_models(X, y, X_test, study.best_params)
    
    # Generate final predictions
    final_predictions = blend_predictions(catboost_pred, stack_65_pred, stack_27_pred, X_test)
    
    # Create submission
    test = pd.read_csv(TEST_PATH)
    test['target'] = final_predictions
    test[['ward', 'target']].to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")

if __name__ == "__main__":
    main()


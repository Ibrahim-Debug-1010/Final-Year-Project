import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb

def load_data(file_path='datasets/NIR_Glucose_Data.csv'):
    df = pd.read_csv(file_path)
    df = df.dropna()
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_nir_data(df, scaler_type='standard', pca_components=3):
    df['NIR_Reading_Smooth'] = savgol_filter(df['NIR_Reading'], window_length=7, polyorder=2)
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(df[['NIR_Reading_Smooth']])
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca, scaler, pca

def extract_features(df, X_pca):
    stats = np.array([
        np.mean(df['NIR_Reading_Smooth']),
        skew(df['NIR_Reading_Smooth']),
        kurtosis(df['NIR_Reading_Smooth'])
    ]).reshape(1, -1)
    stats_repeated = np.repeat(stats, X_pca.shape[0], axis=0)
    X_features = np.hstack([X_pca, stats_repeated])
    return X_features

def preprocess_data(df):
    X_scaled, X_pca, scaler, pca = preprocess_nir_data(df)
    X_features = extract_features(df, X_pca)
    y = df['GLUCOSE_LEVEL'].values
    X = X_features  # shape: (samples, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, pca

def hypertune_models(X_train, y_train):
    param_grids = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10]
        },
        'knn': {
            'n_neighbors': [3, 5, 7]
        },
        'svr': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        },
        'lgbm': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    base_models = {
        'rf': RandomForestRegressor(random_state=42),
        'knn': KNeighborsRegressor(),
        'svr': SVR(),
        'lgbm': lgb.LGBMRegressor(random_state=42)
    }
    best_models = {}
    for name, model in base_models.items():
        print(f"Hypertuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    return best_models

def blend_predict(models, X):
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return np.mean(preds, axis=1)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f} mg/dL")
    print(f"Root Mean Squared Error: {rmse:.2f} mg/dL")
    print(f"R² Score: {r2:.4f}")
    return mae, rmse, r2, y_pred

def evaluate_blend(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f} mg/dL")
    print(f"Root Mean Squared Error: {rmse:.2f} mg/dL")
    print(f"R² Score: {r2:.4f}")
    return mae, rmse, r2

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p2, p1], [p2, p1], 'r--')
    plt.xlabel('Actual Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.title('Actual vs Predicted Glucose Levels')
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/model_performance.png')
    plt.close()

def save_model(model, scaler, pca, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    model.save(f'{output_dir}/deep_model.h5')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(pca, f'{output_dir}/pca.pkl')
    print(f"Model, scaler, and PCA saved to {output_dir}")

def save_blend_models(models, scaler, pca, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f'{output_dir}/{name}.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(pca, f'{output_dir}/pca.pkl')
    print(f"Models, scaler, and PCA saved to {output_dir}")

def create_error_analysis(y_test, y_pred):
    error = y_pred - y_test
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.hist(error, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error (mg/dL)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, error, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Glucose (mg/dL)')
    plt.ylabel('Error (mg/dL)')
    plt.title('Error vs Actual Value')
    result_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': error,
        'AbsError': np.abs(error)
    })
    result_df['GlucoseRange'] = pd.cut(
        result_df['Actual'], 
        bins=[0, 70, 100, 126, 200, float('inf')],
        labels=['Hypoglycemia', 'Normal', 'Prediabetes', 'Diabetes', 'Severe']
    )
    range_metrics = result_df.groupby('GlucoseRange', observed=False)['AbsError'].mean()
    plt.subplot(2, 2, 3)
    range_metrics.plot(kind='bar')
    plt.ylabel('Mean Absolute Error (mg/dL)')
    plt.title('Error by Glucose Range')
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, y_pred, alpha=0.5)
    max_val = max(max(y_test), max(y_pred)) * 1.1
    plt.fill_between([0, 70], [0, 70*1.2], [0, 70*0.8], alpha=0.1, color='green')
    plt.fill_between([70, 290], [70*1.2, 290*1.2], [70*0.8, 290*0.8], alpha=0.1, color='green')
    plt.fill_between([290, max_val], [290*1.2, max_val], [290*0.8, max_val*0.8], alpha=0.1, color='green')
    plt.plot([0, max_val], [0, max_val], 'k--')
    plt.xlabel('Reference Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.title('Clarke Error Grid Analysis')
    plt.tight_layout()
    plt.savefig('images/error_analysis.png')
    plt.close()
    return range_metrics

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(df)
    models = hypertune_models(X_train, y_train)
    y_pred = blend_predict(models, X_test)
    mae, rmse, r2 = evaluate_blend(y_test, y_pred)
    plot_results(y_test, y_pred)
    error_by_range = create_error_analysis(y_test, y_pred)
    save_blend_models(models, scaler, pca)
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'error_by_range': error_by_range
    }

if __name__ == "__main__":
    main()
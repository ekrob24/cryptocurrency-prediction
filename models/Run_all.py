import os
import pandas as pd
from sklearn.metrics import r2_score
import tensorflow as tf
import joblib
import torch

# Define model training and prediction functions
from lstm import train_lstm, predict_lstm
from gru import train_gru, predict_gru
from svr import train_svr, predict_svr
from rf import train_rf, predict_rf
from knn import train_knn, predict_knn
from arima import train_arima, predict_arima
from tft import train_tft, predict_tft
from tcn import train_tcn, predict_tcn
from voting_regressor import train_voting_regressor, predict_voting_regressor
from lstm_gru_hybrid import train_lstm_gru_hybrid, predict_lstm_gru_hybrid

# Define the calculate_r2 function
def calculate_r2(predictions_file, actuals_file):
    # Read the predictions and actual values from the CSV files
    predictions = pd.read_csv(predictions_file)['close']
    actuals = pd.read_csv(actuals_file)['close']
    
    # Calculate the R-squared value
    r2 = r2_score(actuals, predictions)
    
    return r2

# List of cryptocurrency data files
crypto_files = [
    'crypto_task_btc.csv',
    'crypto_task_eth.csv',
    'crypto_task_ltc.csv',
    'crypto_task_xmr.csv',
    'crypto_task_xrp.csv'
]

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

# Define a function to train and predict with each model for a given cryptocurrency
def train_and_predict(model_name, train_func, predict_func, train_data, train_labels, test_data, test_labels, crypto_name):
    model = train_func(train_data, train_labels)
    predictions = predict_func(model, test_data)
    
    # Save model
    model_path = os.path.join('models', f'{crypto_name}_{model_name}_model')
    if model_name in ['lstm', 'gru', 'tcn', 'tft']:
        model.save(model_path)
    else:
        joblib.dump(model, f'{model_path}.joblib')
    
    # Save predictions
    predictions_path = os.path.join('predictions', f'{crypto_name}_{model_name}_predictions.csv')
    pd.DataFrame(predictions, columns=['close']).to_csv(predictions_path, index=False)
    
    # Evaluate model
    r2 = calculate_r2(predictions_path, f'{crypto_name}_actuals.csv')
    print(f'R-squared for {model_name} on {crypto_name}: {r2}')
    return r2

# Loop through each cryptocurrency and run all models
results = []

for crypto_file in crypto_files:
    crypto_name = crypto_file.split('_')[-1].split('.')[0]  # Extract the cryptocurrency name
    data = pd.read_csv(crypto_file)
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
    train_labels = train_data.pop('close').values
    test_labels = test_data.pop('close').values
    
    results.append({
        'crypto': crypto_name,
        'lstm': train_and_predict('lstm', train_lstm, predict_lstm, train_data, train_labels, test_data, test_labels, crypto_name),
        'gru': train_and_predict('gru', train_gru, predict_gru, train_data, train_labels, test_data, test_labels, crypto_name),
        'svr': train_and_predict('svr', train_svr, predict_svr, train_data, train_labels, test_data, test_labels, crypto_name),
        'rf': train_and_predict('rf', train_rf, predict_rf, train_data, train_labels, test_data, test_labels, crypto_name),
        'knn': train_and_predict('knn', train_knn, predict_knn, train_data, train_labels, test_data, test_labels, crypto_name),
        'arima': train_and_predict('arima', train_arima, predict_arima, train_data['close'], test_data['close'], train_data['close'], test_data['close'], crypto_name),
        'tft': train_and_predict('tft', train_tft, predict_tft, train_data, train_labels, test_data, test_labels, crypto_name),
        'tcn': train_and_predict('tcn', train_tcn, predict_tcn, train_data, train_labels, test_data, test_labels, crypto_name),
        'voting_regressor': train_and_predict('voting_regressor', train_voting_regressor, predict_voting_regressor, train_data, train_labels, test_data, test_labels, crypto_name),
        'lstm_gru_hybrid': train_and_predict('lstm_gru_hybrid', train_lstm_gru_hybrid, predict_lstm_gru_hybrid, train_data, train_labels, test_data, test_labels, crypto_name)
    })

# Save the results
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("All models trained and evaluated successfully on all cryptocurrencies. Results saved to 'model_evaluation_results.csv'")
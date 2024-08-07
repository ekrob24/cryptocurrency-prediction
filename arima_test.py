from models.stats.arima import STATS_ARIMA
from util import plot_training, save_results, dataset_binance, r2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import time
import psutil
import os

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    h = 0
    targets = ['close']
    cryptos = ['btc', 'eth', 'ltc', 'xmr', 'xrp']
    retrain = [0, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30]
    outputs = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]
    scaling = ['minmax']
    tuned = 0
    window = 30

    # List to store all information
    results_list = []

    for t in targets:
        for c in cryptos:
            add_split_value = 0
            mse, rmse, mape, r2_score, mae = [], [], [], [], []
            all_predictions, all_labels = [], []

            for index, r in enumerate(retrain):
                output = outputs[index]
                
                experiment_name = 'arima-' + c + '-' + t + '-w' + str(window) + '-h' + str(h) + '-' + str(len(outputs)) + 'm'
                ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=window, output_window=1,
                                                    horizon=h, training_features=['close'],
                                                    target_name=['close'], train_split_factor=0.8)
                df, diff_df = ds.differenced_dataset()
                ds.df = diff_df
                if index > 0:
                    add_split_value += r
                
                ds.add_split_value = add_split_value
                
                if tuned:
                    parameters = pd.read_csv("param/p_arima-" + c + "-h0.csv").iloc[0] 
                    p = {'p': parameters['p'],
                         'd': parameters['d'],
                         'q': parameters['q'],
                         'P': parameters['P'],
                         'Q': parameters['Q'],
                         'D': parameters['D'],
                         'S': parameters['S'],
                         'selection': False,
                         'loop': parameters['loop'],
                         'horizon': parameters['horizon'],
                         'sliding_window': parameters['sliding_window']}
                else:
                    p = {'p': 1,
                         'd': 0,
                         'q': 2,
                         'P': 2,
                         'Q': 0,
                         'D': 0,
                         'S': 12,
                         'loop': 0,
                         'horizon': 0,
                         'sliding_window': 0}

                model = STATS_ARIMA(experiment_name)
                model.ds = ds 
                ds.dataset_creation(df=True, detrended=True)
                ds.dataset_normalization(scaling)
                ds.data_summary()
                
                # Start time and resource monitoring
                start_time = time.time()
                process = psutil.Process()
                cpu_start = process.cpu_percent(interval=None)
                memory_start = process.memory_info().rss
                
                yhat, prediction_std, train_model = model.training(p=p, X_test=ds.X_test_array[:output])
                
                # End time and resource monitoring
                end_time = time.time()
                cpu_end = process.cpu_percent(interval=None)
                memory_end = process.memory_info().rss
                
                if p['horizon'] > 0:
                    yhat = yhat[:-p['horizon']]
                preds = np.array(yhat).reshape(-1, 1)
                np_preds = ds.inverse_transform_predictions(preds=preds)
                inversed_preds = ds.inverse_differenced_dataset(diff_vals=np_preds, df=df, l=(len(ds.y_test_array)))
                ds.df = df
                ds.dataset_creation(df=True)
                labels = ds.y_test_array[h:(len(inversed_preds) + h)].reshape(-1, 1)

                ds.add_split_value = 0
                ds.df = df
                ds.dataset_creation(df=True)
                ds.dataset_normalization(scaling)
                n_preds = ds.scale_predictions(preds=inversed_preds)
                n_labels = ds.scale_predictions(preds=labels)

                mse_value = mean_squared_error(n_labels, n_preds)
                rmse_value = np.sqrt(mean_squared_error(n_labels, n_preds))
                mae_value = mean_absolute_error(n_labels, n_preds)
                mape_value = mean_absolute_percentage_error(n_labels, n_preds)
                r2_value = r2.r_squared(n_labels, n_preds)
                
                mse.append(mse_value)
                rmse.append(rmse_value)
                mae.append(mae_value)
                mape.append(mape_value)
                r2_score.append(r2_value)

                print("MSE", mse_value)
                print("MAE", mae_value)
                print("MAPE", mape_value)
                print("RMSE", rmse_value)
                print("R2", r2_value)
                
                # Calculate timing and resource usage information
                duration = end_time - start_time
                cpu_usage = cpu_end - cpu_start
                memory_usage = (memory_end - memory_start) / (1024 * 1024)  # Convert to MB
                
                # Save information to list
                results_list.append({
                    'Experiment': experiment_name,
                    'Crypto': c,
                    'Target': t,
                    'Training Time (s)': duration,
                    'CPU Usage (%)': cpu_usage,
                    'Memory Usage (MB)': memory_usage,
                    'Parameters': p,
                    'MSE': mse_value,
                    'MAE': mae_value,
                    'MAPE': mape_value,
                    'RMSE': rmse_value,
                    'R2': r2_value
                })
                
                n_experiment_name = experiment_name + '_N'
                all_predictions.extend(n_preds)
                all_labels.extend(n_labels)

                # Ensure directories exist before saving plots
                ensure_dir_exists("img/loss")
                ensure_dir_exists("img/accuracy")
                ensure_dir_exists("img/preds")
                
                # Plot predicted vs actual series
                plot_training.plot_series(np.arange(len(n_labels)), n_labels, n_preds, label1='Actual', label2='Predicted', title=experiment_name)

            # Plot loss and accuracy
            plot_training.plot_loss(mse, mae, title=experiment_name)
            plot_training.plot_accuracy(r2_score, title=experiment_name)

            if not tuned:
                save_results.save_params_csv(model.p, model.name)
               
            save_results.save_output_csv(preds=all_predictions, labels=all_labels, feature=t, filename=n_experiment_name, bivariate=len(ds.target_name) > 1)
            save_results.save_metrics_csv(mses=mse, maes=mae, rmses=rmse, mapes=mape, filename=experiment_name, r2=r2_score)
    
    # Convert results list to DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('training_resources_and_metrics.csv', index=False)

if __name__ == "__main__":
    main()

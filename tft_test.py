from models.dl.tft import TFT
from util import plot_training, save_results, dataset_binance, r2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

def main():
    h = 0
    targets = ['close']
    cryptos = ['btc', 'eth', 'ltc', 'xmr', 'xrp']
    retrain = [0, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30]
    outputs = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31] 
    scaling = ['minmax']
    tuned = 0
    window = 30

    for t in targets:
        for c in cryptos:
            add_split_value = 0
            mse, rmse, mape, r2_score, mae = [], [], [], [], []
            all_predictions, all_labels, train_time, inference_time = [], [], [], []
            for index, r in enumerate(retrain):
                output = outputs[index]
                experiment_name = f'tft-{c}-{t}-w{window}-h{h}-{len(outputs)}m'
                ds = dataset_binance.BinanceDataset(
                    filename=f'crypto_task_{c}.csv', 
                    input_window=window, 
                    output_window=1,
                    horizon=h, 
                    training_features=['close'],
                    target_name=['close'], 
                    train_split_factor=0.8
                )
                df, diff_df = ds.differenced_dataset()
                ds.df = diff_df
                if index > 0:
                    add_split_value += r
                
                ds.add_split_value = add_split_value
                
                if tuned:
                    parameters = pd.read_csv(f"param/p_TFT-{t}-{c}-w30-h0.csv").iloc[0] 
                    p = {
                        'input_chunk_length': parameters['input_chunk_length'],
                        'hidden_layer_dim': parameters['hidden_layer_dim'],
                        'num_lstm_layers': parameters['num_lstm_layers'],
                        'num_attention_heads': parameters['num_attention_heads'],
                        'dropout_rate': parameters['dropout_rate'],
                        'batch_size': parameters['batch_size'],
                        'output_chunk_length': parameters['output_chunk_length'],
                        'patience': parameters['patience'],
                        'lr': parameters['lr'],
                        'optimizer': parameters['optimizer'],
                        'feed_forward': parameters['feed_forward'],
                        'epochs': parameters['epochs'],
                    }
                else:
                    p = {   
                        'input_chunk_length': 30,
                        'hidden_layer_dim': 64,
                        'num_lstm_layers': 3,
                        'num_attention_heads': 7,
                        'dropout_rate': 0.05,
                        'batch_size': 256,
                        'output_chunk_length': 1,
                        'patience': 50,
                        'lr': 1e-3,
                        'optimizer': 'adam',
                        'feed_forward': 'GatedResidualNetwork',
                        'epochs': 200,
                    }

                model = TFT(experiment_name)
                model.ds = ds
                ds.dataset_creation(df=True, detrended=True)
                ds.dataset_normalization(scaling)
                ds.data_summary()

                to_predict = ds.X_test[:output]
                yhat, train_model = model.training(p, X_test=to_predict)

                if hasattr(yhat, 'values'):
                    yhat_values = yhat.values
                else:
                    raise TypeError("Expected yhat to have 'values' attribute or be a list/array.")

                try:
                    preds = np.array(yhat_values).reshape(-1, 1)
                except ValueError as e:
                    print(f"Error reshaping yhat_values: {e}")
                    yhat_flat = np.array(yhat_values).flatten()
                    preds = yhat_flat.reshape(-1, 1)
                    print(f"Reshaped preds shape: {preds.shape}")

                try:
                    np_preds = ds.inverse_transform_predictions(preds=preds)
                except Exception as e:
                    print(f"Error in inverse_transform_predictions: {e}")
                    print(f"Type of preds: {type(preds)}")
                    print(f"Contents of preds: {preds}")
                    np_preds = preds  # Handle the situation as needed

                try:
                    inversed_preds = ds.inverse_differenced_dataset(
                        diff_vals=np_preds, df=df, l=len(ds.y_test_array))
                except Exception as e:
                    print(f"Error in inverse_differenced_dataset: {e}")
                    inversed_preds = np_preds  # Handle as needed

                ds.df = df
                ds.dataset_creation(df=True)
                ds.dataset_normalization(scaling)
                labels = ds.y_test_array[h:(len(inversed_preds)+h)].reshape(-1, 1)
                ds.add_split_value = 0
                ds.df = df
                ds.dataset_creation(df=True)
                ds.dataset_normalization(scaling)

                try:
                    # Ensuring correct format for labels and predictions
                    if not isinstance(labels, np.ndarray):
                        print(f"labels is not an np.ndarray. Type: {type(labels)}")
                        labels = np.array(labels)

                    if not isinstance(inversed_preds, np.ndarray):
                        print(f"inversed_preds is not an np.ndarray. Type: {type(inversed_preds)}")
                        inversed_preds = np.array(inversed_preds)

                    # Ensure both are 2D arrays
                    if labels.ndim == 1:
                        labels = labels.reshape(-1, 1)
                    if inversed_preds.ndim == 1:
                        inversed_preds = inversed_preds.reshape(-1, 1)

                    # Calculate the metrics
                    mse_value = mean_squared_error(labels, inversed_preds)
                    mae_value = mean_absolute_error(labels, inversed_preds)
                    mape_value = mean_absolute_percentage_error(labels, inversed_preds)
                    rmse_value = np.sqrt(mse_value)
                    r2_value = r2.r_squared(labels, inversed_preds)

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
                except Exception as e:
                    print(f"Error in calculating metrics: {e}")
                    print(f"labels: {labels}")
                    print(f"inversed_preds: {inversed_preds}")

                all_predictions.extend(inversed_preds)
                all_labels.extend(labels)
                train_time.append(model.train_time)
                inference_time.append(model.inference_time)

            if not tuned:
                save_results.save_params_csv(model.p, model.name)

            save_results.save_output_csv(preds=all_predictions, labels=all_labels, feature=t, filename=experiment_name, bivariate=len(ds.target_name) > 1)
            save_results.save_metrics_csv(mses=mse, maes=mae, rmses=rmse, mapes=mape, filename=experiment_name, r2=r2_score)
            inference_name = experiment_name + '-inf_time'
            save_results.save_timing(times=inference_time, filename=inference_name)
            train_name = experiment_name + '-train_time'
            save_results.save_timing(times=train_time, filename=train_name)

if __name__ == "__main__":
    main()

from util.dataset import DatasetInterface
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from binance.client import Client
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

class BinanceDataset(DatasetInterface):
    def __init__(self, filename="", input_window=10, output_window=1, horizon=0, training_features=[], target_name=[],
                 train_split_factor=0.8, apiKey="", apiSecurity=""):
        """
        Call to the parent constructor [DatasetInterface] and passing the required parameters.
        """
        super().__init__(filename, input_window, output_window, horizon, training_features, target_name, train_split_factor)
        self.apiKey = apiKey
        self.apiSecurity = apiSecurity

    def create_frame(self, data):
        """
        Builds DataFrame to structure Data.
        """
        df = pd.DataFrame(data)
        df = df.iloc[:, 0:9]
        df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumTrades']
        df.Open = df.Open.astype(float)
        df.Close = df.Close.astype(float)
        df.High = df.High.astype(float)
        df.Low = df.Low.astype(float)
        df.Volume = df.Volume.astype(float)
        df.QuoteAssetVolume = df.QuoteAssetVolume.astype(float)
        df.NumTrades = df.NumTrades.astype(int)
        df.OpenTime = pd.to_datetime(df.OpenTime, unit='ms')
        df.CloseTime = pd.to_datetime(df.CloseTime, unit='ms')
        return df

    def get_binance_data(self, sym, start_date, end_date):
        """
        Retrieves Data from Binance API.
        """
        try:
            client = Client(self.apiKey, self.apiSecurity)
            print("Logged in")
            interval = Client.KLINE_INTERVAL_1DAY
            klines = client.get_historical_klines(sym, interval, start_date, end_date)
            df = self.create_frame(klines)
            df_b = df.iloc[:, 0:5]
            return df_b
        except Exception as e:
            print(f'Error in getting Binance data: {e}')
            return None

    def __save_crypto_df_to_csv(self, df):
        """
        Saves DataFrame to .csv file within the saved_data folder.
        """
        df.columns = ['date', 'open', 'high', 'low', 'close', 'timestamp']
        save_name = self.data_path + self.data_file
        df.to_csv(save_name)

    def build_crypto_dataset(self, name, start_year, end_year, sym, start_date, end_date):
        """
        Combines Data from Investing.com with Binance API Data.
        """
        path = f'https://github.com/katemurraay/TFT_Data/blob/main/{name}_{start_year}_{end_year}.csv?raw=true'
        df_git = pd.read_csv(path)
        df_git.Date = pd.to_datetime(df_git.Date, format='%d/%m/%Y')

        df_binance = self.get_binance_data(sym, start_date, end_date)
        if df_binance is None:
            print("Failed to retrieve Binance data.")
            return

        column_names = ["Date", "Open", "High", "Low", "Close"]
        df_git = df_git.reindex(columns=column_names)
        df_binance = df_binance.rename(columns={'OpenTime': 'Date'})

        i = df_git.index[df_git['Date'] == df_binance['Date'].iloc[0]].tolist()
        if not i:
            print("No matching start date found between datasets.")
            return

        df_a = df_git[:i[0]]
        df_b = df_binance.iloc[:, 0:5]
        list_combine = df_a.values.tolist() + df_b.values.tolist()

        df_combined = pd.DataFrame(list_combine, columns=['date', 'open', 'high', 'low', 'close'])
        df_combined['timestamp'] = pd.to_datetime(df_combined['date']).view(int) // 10**9
        self.__save_crypto_df_to_csv(df=df_combined)

    def inverse_transform_predictions(self, preds, X=0, method="minmax", scale_range=(0, 1)):             
        """
        Inverts Scaling from the Data.
        """
        if isinstance(X, int):
            inverse_preds = self.y_scalers[0].inverse_transform(preds)
        else: 
            scale_method = MinMaxScaler(feature_range=scale_range) if method == 'minmax' else StandardScaler()
            scaler = Scaler(scaler=scale_method)
            to_invert = TimeSeries.from_values(preds)
            scaler.fit(X)
            inv_preds = scaler.inverse_transform(to_invert)
            inverse_preds = inv_preds.pd_dataframe().values.reshape(-1, 1)
        return inverse_preds

    def differenced_dataset(self, interval=1):
        """
        Builds Difference Dataset based on Interval.
        """
        df = pd.read_csv(self.data_path + self.data_file)
        df.date = pd.to_datetime(df.date, format="%d/%m/%Y")
        df = df.set_index('date')
        target = self.target_name[0]
        diff = [df[target][i] - df[target][i - 1] for i in range(interval, len(df))]
        time_steps = df['timestamp'].iloc[interval:].tolist()
        diff_df = pd.DataFrame(diff, columns=[target])
        diff_df['timestamp'] = time_steps
        return df, diff_df

    def inverse_differenced_dataset(self, df, diff_vals, l=0, df_start=0):
        """
        Inverses the Difference on a Dataset.
        """
        invert = []
        target = self.target_name[0]
        if df_start == 0:
            df_start = len(df) - len(diff_vals) - 1 if l == 0 else len(df) - l - 1

        for i in range(len(diff_vals)):
            value = diff_vals[i] + df[target][df_start + i]
            invert.append(value)
        inverted_values = np.array(invert)
        return inverted_values

    def scale_predictions(self, preds, X=0, method="minmax", scale_range=(0, 1)):             
        """
        Scale Predictions.
        """
        if isinstance(preds, np.ndarray):
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            try:
                scaled_preds = self.y_scalers[0].transform(preds)
            except Exception as e:
                print(f"Error during scaling: {e}")
                print(f"Type of preds: {type(preds)}, Shape: {preds.shape}")
                scaled_preds = preds  # Fallback in case of an error
        else:
            try:
                scale_method = MinMaxScaler(feature_range=scale_range) if method == 'minmax' else StandardScaler()
                scaler = Scaler(scaler=scale_method)
                scaler.fit(X)
                scaled_preds = scaler.transform(preds)
            except Exception as e:
                print(f"Error during scaling with Darts: {e}")
                print(f"Type of preds: {type(preds)}, preds: {preds}")
                scaled_preds = preds  # Fallback in case of an error

        return scaled_preds

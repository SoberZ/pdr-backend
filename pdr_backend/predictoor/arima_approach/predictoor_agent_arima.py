from typing import Tuple
import numpy as np

from enforce_typing import enforce_types
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import ta

import xgboost as xgb
import scipy.stats as sta

from pdr_backend.data_eng.data_factory import DataFactory
from pdr_backend.data_eng.data_pp import DataPP
from pdr_backend.model_eng.model_factory import ModelFactory

from pdr_backend.predictoor.base_predictoor_agent import BasePredictoorAgent
from pdr_backend.predictoor.arima_approach.predictoor_config_arima import PredictoorConfigArima
from pdr_backend.util.timeutil import timestr_to_ut

def create_lagged_features(df, n_lags):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['binance:BTC:close'].shift(lag)
    return df

def add_rolling_window_features(df, window_size):
    df[f'rolling_mean_{window_size}'] = df['binance:BTC:close'].rolling(window=window_size).mean()
    df[f'rolling_std_{window_size}'] = df['binance:BTC:close'].rolling(window=window_size).std()
    return df

def add_ARIMA_features(df, p, d, q):
    original_index = df.index

    df_reset = df.reset_index()

    model = ARIMA(df_reset['binance:BTC:close'], order=(p, d, q)).fit()
    df_reset[f'ARIMA_{p}_{d}_{q}'] = model.predict(start=0, end=len(df_reset) - 1)
    df_reset[f'ARIMA_error_{p}_{d}_{q}'] = df_reset['binance:BTC:close'] - df_reset[f'ARIMA_{p}_{d}_{q}']

    df_reset.set_index(original_index, inplace=True)

    return df_reset

def add_RSI_feature(df, column='binance:BTC:close', period=14):
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    return df

def add_MACD_feature(df, column='binance:BTC:close', short_period=12, long_period=26, signal_period=9):
    short_ema = df[column].ewm(span=short_period, adjust=False).mean()
    long_ema = df[column].ewm(span=long_period, adjust=False).mean()

    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

    return df

def add_volatility_features(df, column='binance:BTC:close', period=14):
    df[f'Volatility_{period}'] = df[column].rolling(window=period).std()
    return df


def add_ema_features(df, column='binance:BTC:close', period=14):
    df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()

    return df

def add_bollinger_bands(df, column='binance:BTC:close', period=20, std=2):
    df[f'BB_{period}'] = df[column].rolling(window=period).mean()
    df[f'BB_Upper_{period}'] = df[f'BB_{period}'] + (df[column].rolling(window=period).std() * std)
    df[f'BB_Lower_{period}'] = df[f'BB_{period}'] - (df[column].rolling(window=period).std() * std)
    return df

def add_ATR_feature(df, period=14):
    df['H-L'] = abs(df['binance:BTC:high'] - df['binance:BTC:low'])
    df['H-PC'] = abs(df['binance:BTC:high'] - df['binance:BTC:close'].shift(1))
    df['L-PC'] = abs(df['binance:BTC:low'] - df['binance:BTC:close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(period).mean()
    df.drop(['H-L', 'H-PC', 'L-PC'], inplace=True, axis=1)
    return df

def add_VWAP_feature(df, exchange="binance"):
    df['TP'] = (df[f'{exchange}:BTC:high'] + df[f'{exchange}:BTC:low'] + df[f'{exchange}:BTC:close']) / 3
    df['TradedValue'] = df['TP'] * df[f'{exchange}:BTC:volume']
    df['CumulativeTradedValue'] = df['TradedValue'].cumsum()
    df['CumulativeVolume'] = df[f'{exchange}:BTC:volume'].cumsum()
    df['VWAP'] = df['CumulativeTradedValue'] / df['CumulativeVolume']
    df.drop(['TP', 'TradedValue', 'CumulativeTradedValue', 'CumulativeVolume'], inplace=True, axis=1)
    return df

def add_TWAP_feature(df, exchange="binance"):
    df['TP'] = (df[f'{exchange}:BTC:high'] + df[f'{exchange}:BTC:low'] + df[f'{exchange}:BTC:close']) / 3
    df['TradedValue'] = df['TP'] * df[f'{exchange}:BTC:volume']
    df['CumulativeTradedValue'] = df['TradedValue'].cumsum()
    df['TWAP'] = df['CumulativeTradedValue'] / df[f'{exchange}:BTC:volume'].cumsum()
    df.drop(['TP', 'TradedValue', 'CumulativeTradedValue'], inplace=True, axis=1)
    return df

def add_on_balance_volume_feature(df, column='binance:BTC:close', exchange="binance"):
    df['OBV'] = np.where(df[column] > df[column].shift(1), df[f'{exchange}:BTC:volume'], np.where(df[column] < df[column].shift(1), -df[f'{exchange}:BTC:volume'], 0)).cumsum()
    return df

def add_datetime_features(df):
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['week'] = df['datetime'].dt.isocalendar().week
    df.drop('datetime', axis=1, inplace=True)
    return df

def add_model_features(df):
    df = add_datetime_features(df)
    # Add multiple rolling window features
    for i in range(14, 140, 20):
        df = add_rolling_window_features(df, i)

    # Add multiple ARIMA features
    df = add_ARIMA_features(df, 0, 1, 4)
    df = create_lagged_features(df, 5)
    df = add_MACD_feature(df)
    df = add_VWAP_feature(df, exchange="binance")
    df = add_VWAP_feature(df, exchange="binanceus")
    df = add_on_balance_volume_feature(df, exchange="binance")


    for i in range(14, 200, 20):
        df = add_RSI_feature(df, period=i)
        df = add_ATR_feature(df, period=i)

    df['price_up'] = df['binance:BTC:close'].shift(-1) > df['binance:BTC:close']

    # df.dropna(inplace=True)
    return df

# Define LSTM model in a function
def create_lstm_model(timesteps, input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(timesteps, input_dim)),  # Adjust 'timesteps' and 'input_dim'
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



@enforce_types
class PredictoorAgentArima(BasePredictoorAgent):
    predictoor_config_class = PredictoorConfigArima

    def __init__(self, config: PredictoorConfigArima):
        super().__init__(config)
        self.config: PredictoorConfigArima = config

        # Define KNN Classifiers
        self.knearest1 = KNeighborsClassifier(n_neighbors=6)
        self.knearest2 = KNeighborsClassifier(n_neighbors=8)

        # Define MLP
        self.mlp1 = MLPClassifier(random_state=31, max_iter=300)
        self.mlp2 = MLPClassifier(random_state=98, max_iter=300)

        # Define LSTM
        self.lstm1 = KerasClassifier(build_fn=create_lstm_model, timesteps=1, input_dim=1, epochs=10, batch_size=32, verbose=0)

    def get_prediction(
        self, addr: str, timestamp: int  # pylint: disable=unused-argument
    ) -> Tuple[bool, float]:
        """
        @description
          Given a feed, let's predict for a given timestamp.

        @arguments
          addr -- str -- address of the trading pair. Info in self.feeds[addr]
          timestamp -- int -- when to make prediction for (unix time)

        @return
          predval -- bool -- if True, it's predicting 'up'. If False, 'down'
          stake -- int -- amount to stake, in units of Eth
        """
        feed = self.feeds[addr]

        # user-uncontrollable params, at data-eng level
        data_pp = DataPP(
            timeframe=feed.timeframe,  # eg "5m"
            yval_exchange_id=feed.source,  # eg "binance"
            yval_coin=feed.base,  # eg "BTC"
            usdcoin=feed.quote,  # eg "USDT"
            yval_signal="close",  # pdr feed setup is always "close"
            N_test=1,  # N/A for this context
        )

        # user-controllable params, at data-eng level
        data_ss = self.config.data_ss.copy_with_yval(data_pp)
        data_ss.fin_timestamp = timestr_to_ut("now")

        #  user-controllable params, at model-eng level
        model_ss = self.config.model_ss

        # do work...
        data_factory = DataFactory(data_pp, data_ss)

        # Compute X/y
        hist_df = data_factory.get_hist_df()
        hist_df = add_model_features(hist_df)
        hist_df = ta.add_all_ta_features(hist_df, "binance:BTC:open", "binance:BTC:high", "binance:BTC:low", "binance:BTC:close", "binance:BTC:volume", fillna=True)
        hist_df['price_up'] = hist_df['binance:BTC:close'].shift(-1) > hist_df['binance:BTC:close']

        X = hist_df.drop(['price_up'], axis=1)
        y = hist_df['price_up']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # voting_model = VotingClassifier(estimators=[('knn1', self.knearest1), ('knn2', self.knearest2), ('mlp1', self.mlp1), ('mlp2', self.mlp2), ('lstm1', self.lstm1)], voting='hard')
        # # Fit the model
        # voting_model.fit(X_train, y_train)
        # # Obtain prediction
        # predval = voting_model.predict(X_test)

        # Get feature importance from votingclassifier
        # print(f"Feature Importances: {voting_model.feature_importances_}")

        # Specify validations set to watch performance
        model = xgb.XGBClassifier(objective="binary:logistic")
        model.fit(X_train, y_train)
        # print(model.get_booster().get_score(importance_type="gain"))
        # print(f"Feature Importances: {model.feature_importances_}")

        pyplot.figure(figsize=(10, 14))
        xgb.plot_importance(model, max_num_features=25)
        pyplot.tight_layout()
        pyplot.savefig("feature_importance.png")

        # Predicting with the best model
        predval = model.predict(X_test)

        # y_pred_prob = model.predict_proba(X_test)
        # print(f"LAST PREDICTION PROBABILITY: {y_pred_prob[-1]}")
        # print(f"Log loss: {log_loss(y_test, y_pred_prob)}")
        # print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob[:,1])}")
        # print(classification_report(y_test, predval))

        # Stake what was set via envvar STAKE_AMOUNT
        stake = self.config.stake_amount

        return (bool(predval[-1]), stake)

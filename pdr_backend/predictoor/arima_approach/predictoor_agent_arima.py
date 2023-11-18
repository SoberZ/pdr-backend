from typing import Tuple
import numpy as np

from enforce_typing import enforce_types
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import RandomizedSearchCV
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
    """
    Adds the Relative Strength Index (RSI) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating RSI.

    Returns:
    pd.DataFrame: DataFrame with the RSI feature added.
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def add_MACD_feature(df, column='binance:BTC:close', short_period=12, long_period=26, signal_period=9):
    """
    Adds the Moving Average Convergence Divergence (MACD) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    short_period (int): Short term period for the MACD.
    long_period (int): Long term period for the MACD.
    signal_period (int): Signal line period for the MACD.

    Returns:
    pd.DataFrame: DataFrame with MACD and MACD signal line features added.
    """
    short_ema = df[column].ewm(span=short_period, adjust=False).mean()
    long_ema = df[column].ewm(span=long_period, adjust=False).mean()

    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

    return df


def add_model_features(df):
    # Add multiple rolling window features
    for i in range(1, 250, 10):
        df = add_rolling_window_features(df, i)

    # Add multiple ARIMA features
    df = add_ARIMA_features(df, 0, 1, 4)

    # Add multiple RSI features
    for i in range(1, 20, 2):
        df = add_RSI_feature(df, period=i)

    # Add multiple MACD features
    df = create_lagged_features(df, 5)
    df = add_MACD_feature(df)
    # df.dropna(inplace=True)
    return df


@enforce_types
class PredictoorAgentArima(BasePredictoorAgent):
    predictoor_config_class = PredictoorConfigArima

    def __init__(self, config: PredictoorConfigArima):
        super().__init__(config)
        self.config: PredictoorConfigArima = config

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
        X, y, _ = data_factory.create_xy(hist_df, testshift=0)

        # Split X/y into train & test datacoin
        st, fin = 0, X.shape[0] - 1
        X_train, X_test = X[st:fin, :], X[fin : fin + 1]
        y_train, _ = y[st:fin], y[fin : fin + 1]

        # Define XGBoost model parameters
        param_dist = {
            'max_depth': sta.randint(3, 10),
            'min_child_weight': sta.randint(1, 10),
            'gamma': sta.uniform(0, 10),
            'subsample': sta.uniform(0.5, 0.5),
            'colsample_bytree': sta.uniform(0.5, 0.5),
            'learning_rate': sta.uniform(0.05, 0.3)
        }

        # Specify validations set to watch performance
        xgb_model = xgb.XGBRegressor(tree_method='hist')

        # Randomized Search for Hyperparameter Tuning
        random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=25, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=3)
        random_search.fit(X_train, y_train)

        # Best Model from Randomized Search
        best_model = random_search.best_estimator_

        # Predicting with the best model
        predprice = best_model.predict(X_test)
        curprice = y[-1]
        predval = predprice > curprice

        # Stake what was set via envvar STAKE_AMOUNT
        stake = self.config.stake_amount

        return (bool(predval), stake)


# $env:PRIVATE_KEY = "be0712fe7ba53118a491218f8edd1460bb6b81b3eb1b55a4127575ffe6f1c3db"

# $env:ADDRESS_FILE = "${env:HOME}\.ocean\ocean-contracts\artifacts\address.json"
# $env:PAIR_FILTER = "BTC/USDT"
# $env:TIMEFRAME_FILTER = "5m"
# $env:SOURCE_FILTER = "binance"

# $env:RPC_URL = "https://testnet.sapphire.oasis.dev"
# $env:SUBGRAPH_URL = "https://v4.subgraph.sapphire-testnet.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph"
# $env:STAKE_TOKEN = "0x973e69303259B0c2543a38665122b773D28405fB" # (fake) OCEAN token address
# $env:OWNER_ADDRS = "0xe02a421dfc549336d47efee85699bd0a3da7d6ff" # OPF deployer address

import os
import time
from typing import List

import scipy.stats as sta
from enforce_typing import enforce_types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from pdr_backend.data_eng.data_factory import DataFactory
from pdr_backend.data_eng.data_pp import DataPP
from pdr_backend.data_eng.data_ss import DataSS
from pdr_backend.model_eng.model_factory import ModelFactory
from pdr_backend.model_eng.model_ss import ModelSS
from pdr_backend.simulation.sim_ss import SimSS
from pdr_backend.simulation.trade_ss import TradeSS
from pdr_backend.simulation.trade_pp import TradePP
from pdr_backend.util.mathutil import nmse
from pdr_backend.util.timeutil import current_ut, pretty_timestr

FONTSIZE = 12


def create_lagged_features(df, n_lags):
    d = {}
    for lag in range(1, n_lags + 1):
        d[f'lag_{lag}'] = df['binance:BTC:close'].shift(lag)
    return pd.concat([df, pd.DataFrame(d)], axis=1)

def add_rolling_window_features(df, window_size):
    new_cols = {
        f'rolling_mean_{window_size}': df['binance:BTC:close'].rolling(window=window_size).mean(),
        f'rolling_std_{window_size}': df['binance:BTC:close'].rolling(window=window_size).std()
    }
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

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
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

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


def add_ema_features(df, column='binance:BTC:close', period=14):
    """
    Adds the Exponential Moving Average (EMA) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating EMA.

    Returns:
    pd.DataFrame: DataFrame with the EMA feature added.
    """
    df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()

    return df

def add_bollinger_bands(df, column='binance:BTC:close', period=20, std=2):
    """
    Adds the Bollinger Bands (BB) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating BB.
    std (int): Standard deviation for calculating BB.

    Returns:
    pd.DataFrame: DataFrame with the BB feature added.
    """
    df[f'BB_{period}'] = df[column].rolling(window=period).mean()
    df[f'BB_Upper_{period}'] = df[f'BB_{period}'] + (df[column].rolling(window=period).std() * std)
    df[f'BB_Lower_{period}'] = df[f'BB_{period}'] - (df[column].rolling(window=period).std() * std)
    return df

def add_ATR_feature(df, column='binance:BTC:close', period=14):
    """
    Adds the Average True Range (ATR) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating ATR.

    Returns:
    pd.DataFrame: DataFrame with the ATR feature added.
    """

    df['H-L'] = df['binance:BTC:high'] - df['binance:BTC:low']
    df['H-PC'] = abs(df['binance:BTC:high'] - df['binance:BTC:close'].shift(1))
    df['L-PC'] = abs(df['binance:BTC:low'] - df['binance:BTC:close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(period).mean()

    return df

def add_ADX_feature(df, column='binance:BTC:close', period=14):
    """
    Adds the Average Directional Index (ADX) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating ADX.

    Returns:
    pd.DataFrame: DataFrame with the ADX feature added.
    """
    df['H-PC'] = df['binance:BTC:high'] - df['binance:BTC:high'].shift(1)
    df['PC-L'] = df['binance:BTC:low'].shift(1) - df['binance:BTC:low']
    df['+DM'] = np.where((df['H-PC'] > 0) & (df['H-PC'] > df['PC-L']), df['H-PC'], 0)
    df['-DM'] = np.where((df['PC-L'] > 0) & (df['PC-L'] > df['H-PC']), df['PC-L'], 0)
    df['TRn'] = np.where(df['TR'] == 0, 0, df['TR'])
    df['+DMn'] = np.where(df['+DM'] == 0, 0, df['+DM'])
    df['-DMn'] = np.where(df['-DM'] == 0, 0, df['-DM'])

    df['TRn-1'] = df['TRn'].shift(1)
    df['+DMn-1'] = df['+DMn'].shift(1)
    df['-DMn-1'] = df['-DMn'].shift(1)
    df['TRn-1'] = np.where(df['TRn-1'] == 0, df['TR'], df['TRn-1'])

    df['+DI'] = 100 * (df['+DMn'] / df['TRn-1']).ewm(span=period, adjust=False).mean()
    df['-DI'] = 100 * (df['-DMn'] / df['TRn-1']).ewm(span=period, adjust=False).mean()
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).ewm(span=period, adjust=False).mean()
    df['ADX'] = df['DX'].ewm(span=period, adjust=False).mean()


def add_volume_features(df, column='binance:BTC:volume', period=14):
    """
    Adds the Volume (VOL) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating VOL.

    Returns:
    pd.DataFrame: DataFrame with the VOL feature added.
    """
    d = {}
    d[f'Volume_{period}'] = df[column].rolling(window=period).mean()

    return pd.concat([df, pd.DataFrame(d)], axis=1)

def add_VWAP_feature(df, column='binance:BTC:close', period=14):
    """
    Adds the Volume Weighted Average Price (VWAP) to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame with price data.
    column (str): Column name of price data.
    period (int): Period for calculating VWAP.

    Returns:
    pd.DataFrame: DataFrame with the VWAP feature added.
    """
    df['TP'] = (df['binance:BTC:high'] + df['binance:BTC:low'] + df['binance:BTC:close']) / 3
    df['TradedValue'] = df['TP'] * df['binance:BTC:volume']
    df['CumVol'] = df['binance:BTC:volume'].cumsum()
    df['CumTradedValue'] = df['TradedValue'].cumsum()
    df[f'VWAP_{period}'] = df['CumTradedValue'] / df['CumVol']

    return df

def add_model_features(df):
    # Add multiple rolling window features
    for i in range(10, 250, 10):
        df = add_rolling_window_features(df, i)
        df = add_ema_features(df, period=i)

    # Add multiple ARIMA features
    df = add_ARIMA_features(df, 0, 1, 4)

    # Add multiple RSI features
    for i in range(14, 50, 4):
        df = add_RSI_feature(df, period=i)

    # Add multiple MACD features
    df = create_lagged_features(df, 8)
    # df = add_MACD_feature(df)

    # Add volatility features
    for i in range(10, 100, 5):
        # df = add_bollinger_bands(df, period=i)
        df = add_ATR_feature(df, period=i)
        # df = add_ADX_feature(df, period=i)
        df = add_volume_features(df, column="binance:BTC:volume", period=i)
        df = add_VWAP_feature(df, period=i)

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
class PlotState:
    def __init__(self):
        self.fig, (self.ax0, self.ax1) = plt.subplots(2)
        plt.ioff()
        # plt.show()


# pylint: disable=too-many-instance-attributes
class TradeEngine:
    @enforce_types
    def __init__(
        self,
        data_pp: DataPP,
        data_ss: DataSS,
        model_ss: ModelSS,
        trade_pp: TradePP,
        trade_ss: TradeSS,
        sim_ss: SimSS,
    ):
        """
        @arguments
          data_pp -- user-uncontrollable params, at data level
          data_ss -- user-controllable params, at data level
          model_ss -- user-controllable params, at model level
          trade_pp -- user-uncontrollable params, at trading level
          trade_ss -- user-controllable params, at trading level
          sim_ss -- user-controllable params, at sim level
        """
        # ensure training data has the target yval
        assert data_pp.yval_exchange_id in data_ss.exchs_dict
        assert data_pp.yval_signal in data_ss.signals
        assert data_pp.yval_coin in data_ss.coins

        # pp & ss values
        self.data_pp = data_pp
        self.data_ss = data_ss
        self.model_ss = model_ss
        self.trade_pp = trade_pp
        self.trade_ss = trade_ss
        self.sim_ss = sim_ss

        # state
        self.holdings = self.trade_pp.init_holdings
        self.tot_profit_usd = 0.0
        self.nmses_train: List[float] = []
        self.ys_test: List[float] = []
        self.ys_testhat: List[float] = []
        self.corrects: List[bool] = []
        self.profit_usds: List[float] = []
        self.tot_profit_usds: List[float] = []

        self.data_factory = DataFactory(self.data_pp, self.data_ss)

        self.logfile = ""

        self.plot_state = None
        if self.sim_ss.do_plot:
            self.plot_state = PlotState()

    @property
    def usdcoin(self) -> str:
        return self.data_pp.usdcoin

    @property
    def tokcoin(self) -> str:
        return self.data_pp.yval_coin

    @enforce_types
    def _init_loop_attributes(self):
        filebase = f"out_{current_ut()}.txt"
        self.logfile = os.path.join(self.sim_ss.logpath, filebase)
        with open(self.logfile, "w") as f:
            f.write("\n")

        self.tot_profit_usd = 0.0
        self.nmses_train, self.ys_test, self.ys_testhat, self.corrects = [], [], [], []
        self.profit_usds, self.tot_profit_usds = [], []

    @enforce_types
    def run(self):
        self._init_loop_attributes()
        log = self._log
        log("Start run")
        # main loop!
        hist_df = self.data_factory.get_hist_df()
        for test_i in range(self.data_pp.N_test):
            self.run_one_iter(test_i, hist_df)
            self._plot(test_i, self.data_pp.N_test)

        log("Done all iters.")

        nmse_train = np.average(self.nmses_train)
        nmse_test = nmse(self.ys_testhat, self.ys_test)
        log(f"Final nmse_train={nmse_train:.5f}, nmse_test={nmse_test:.5f}")

    @enforce_types
    def run_one_iter(self, test_i: int, hist_df: pd.DataFrame):
        log = self._log

        # Apply the functions to create features
        hist_df = add_model_features(hist_df)

        testshift = self.data_pp.N_test - test_i - 1  # eg [99, 98, .., 2, 1, 0]
        X, y, _ = self.data_factory.create_xy(hist_df, testshift)

        st, fin = 0, X.shape[0] - 1
        X_train, X_test = X[st:fin, :], X[fin : fin + 1]
        y_train, y_test = y[st:fin], y[fin : fin + 1]

        # y_trainhat = model.predict(X_train)  # eg yhat=zhat[y-5]
        # Check if the model to use is ARIMA
        if self.model_ss.model_approach == "ARIMA":
            y_train = y_train.ravel() if y_train.ndim > 1 else y_train
            # For ARIMA, you might need the entire series as input
            # Adjust the order (p, d, q) based on your data and requirements
            p, d, q = 0, 1, 4  # Example parameters
            model = ARIMA(y_train, order=(p, d, q)).fit()
            predprice = model.forecast(steps=1)[0]
        elif self.model_ss.model_approach == "HYBRID":
            y_train = y_train.ravel() if y_train.ndim > 1 else y_train
            # For ARIMA, you might need the entire series as input
            # Adjust the order (p, d, q) based on your data and requirements
            p, d, q = 0, 1, 4  # Example parameters
            model = ARIMA(y_train, order=(p, d, q)).fit()
            predprice = model.forecast(steps=1)[0]

            # Calculate residuals
            residuals = y_train - model.predict()

            residuals_2d = residuals.reshape(-1, 1)

            # Prepare data for XGBoost
            dtrain = xgb.DMatrix(data=residuals_2d, label=y_train)

            # Define XGBoost model parameters
            xgb_params = {
                'max_depth': 3,  # Example parameters, tune these
                'eta': 0.1,
                'objective': 'reg:squarederror'
            }

            # Train XGBoost model
            xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)

            # Use XGBoost to predict the residuals
            xgb_pred = xgb_model.predict(dtrain)

            # Final prediction is ARIMA prediction plus XGBoost prediction
            final_pred = predprice + xgb_pred[-1]  # Assuming last value is the forecast
            predprice = final_pred
        elif self.model_ss.model_approach == "XGBOOST":
            # Create XGBoost model with logistic regression for binary price prediction
            # dtrain = xgb.DMatrix(X_train, label=y_train)
            # dtest = xgb.DMatrix(X_test, label=y_test)

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
        elif self.model_ss.model_approach == "ENSEMBLE":
            # Define KNN Classifiers
            knearest1 = KNeighborsClassifier(n_neighbors=6)
            knearest2 = KNeighborsClassifier(n_neighbors=8)

            # Define MLP
            mlp1 = MLPClassifier(random_state=31, max_iter=300)
            mlp2 = MLPClassifier(random_state=98, max_iter=300)

            # Define LSTM
            lstm1 = KerasClassifier(build_fn=create_lstm_model, timesteps=1, input_dim=1, epochs=10, batch_size=32, verbose=0)

            voting_model = VotingClassifier(estimators=[('knn1', knearest1), ('knn2', knearest2), ('mlp1', mlp1), ('mlp2', mlp2), ('lstm1', lstm1)], voting='hard')

            # Train the model
            voting_model.fit(X_train, y_train)

            # Predict the price
            predprice = voting_model.predict(X_test)
        else:
            # Existing logic for other model types
            model_factory = ModelFactory(self.model_ss)
            model = model_factory.build(X_train, y_train)
            predprice = model.predict(X_test)[0]
            y_trainhat = model.predict(X_train)  # eg yhat=zhat[y-5]
            nmse_train = nmse(y_train, y_trainhat, min(y), max(y))
            self.nmses_train.append(nmse_train)

        # current time
        ut = int(hist_df.index.values[-1]) - testshift * self.data_pp.timeframe_ms

        # current price
        curprice = y_train[-1]

        self.ys_testhat.append(predprice)

        # simulate buy. Buy 'amt_usd' worth of TOK if we think price going up
        usdcoin_holdings_before = self.holdings[self.usdcoin]
        if self._do_buy(predprice, curprice):
            self._buy(curprice, self.trade_ss.buy_amt_usd)

        # observe true price
        trueprice = y_test[0]
        self.ys_test.append(trueprice)

        # simulate sell. Update tot_profit_usd
        tokcoin_amt_sell = self.holdings[self.tokcoin]
        if tokcoin_amt_sell > 0:
            self._sell(trueprice, tokcoin_amt_sell)
        usdcoin_holdings_after = self.holdings[self.usdcoin]

        profit_usd = usdcoin_holdings_after - usdcoin_holdings_before

        self.tot_profit_usd += profit_usd
        self.profit_usds.append(profit_usd)
        self.tot_profit_usds.append(self.tot_profit_usd)

        # err = abs(predprice - trueprice)
        pred_dir = "UP" if predprice > curprice else "DN"
        true_dir = "UP" if trueprice > curprice else "DN"
        correct = pred_dir == true_dir
        correct_s = "Y" if correct else "N"
        self.corrects.append(correct)
        acc = float(sum(self.corrects)) / len(self.corrects) * 100
        log(
            f"Iter #{test_i+1:3}/{self.data_pp.N_test}: "
            f" ut{pretty_timestr(ut)[9:][:-9]}"
            # f". Predval|true|err {predprice:.2f}|{trueprice:.2f}|{err:6.2f}"
            f". Preddir|true|correct = {pred_dir}|{true_dir}|{correct_s}"
            f". Total correct {sum(self.corrects):3}/{len(self.corrects):3}"
            f" ({acc:.1f}%)"
            # f". Spent ${amt_usdcoin_sell:9.2f}, recd ${amt_usdcoin_recd:9.2f}"
            f", profit ${profit_usd:7.2f}"
            f", tot_profit ${self.tot_profit_usd:9.2f}"
        )

    def _do_buy(self, predprice: float, curprice: float) -> bool:
        """
        @arguments
          predprice -- predicted price (5 min from now)
          curprice -- current price (now)

        @return
          bool -- buy y/n?
        """
        return predprice > curprice

    def _buy(self, price: float, usdcoin_amt_spend: float):
        """
        @description
          Buy tokcoin with usdcoin

        @arguments
          price -- amt of usdcoin per token
          usdcoin_amt_spend -- amount to spend, in usdcoin; spend less if have less
        """
        # simulate buy
        usdcoin_amt_sent = min(usdcoin_amt_spend, self.holdings[self.usdcoin])
        self.holdings[self.usdcoin] -= usdcoin_amt_sent

        p = self.trade_pp.fee_percent
        usdcoin_amt_fee = p * usdcoin_amt_sent
        tokcoin_amt_recd = (1 - p) * usdcoin_amt_sent / price
        self.holdings[self.tokcoin] += tokcoin_amt_recd

        self._log(
            f"  TX: BUY : send {usdcoin_amt_sent:8.2f} {self.usdcoin:4}"
            f", receive {tokcoin_amt_recd:8.2f} {self.tokcoin:4}"
            f", fee = {usdcoin_amt_fee:8.4f} {self.usdcoin:4}"
        )

    def _sell(self, price: float, tokcoin_amt_sell: float):
        """
        @description
          Sell tokcoin for usdcoin

        @arguments
          price -- amt of usdcoin per token
          tokcoin_amt_sell -- how much of coin to sell, in tokcoin
        """
        tokcoin_amt_sent = tokcoin_amt_sell
        self.holdings[self.tokcoin] -= tokcoin_amt_sent

        p = self.trade_pp.fee_percent
        usdcoin_amt_fee = p * tokcoin_amt_sent * price
        usdcoin_amt_recd = (1 - p) * tokcoin_amt_sent * price
        self.holdings[self.usdcoin] += usdcoin_amt_recd

        self._log(
            f"  TX: SELL: send {tokcoin_amt_sent:8.2f} {self.tokcoin:4}"
            f", receive {usdcoin_amt_recd:8.2f} {self.usdcoin:4}"
            f", fee = {usdcoin_amt_fee:8.4f} {self.usdcoin:4}"
        )

    @enforce_types
    def _plot(self, i, N):
        if not self.sim_ss.do_plot:
            return

        # don't plot first 5 iters -> not interesting
        # then plot the next 5 -> "stuff's happening!"
        # then plot every 5th iter, to balance "stuff's happening" w/ speed
        do_update = i >= 5 and (i < 10 or i % 5 == 0 or (i + 1) == N)
        if not do_update:
            return

        fig, ax0, ax1 = self.plot_state.fig, self.plot_state.ax0, self.plot_state.ax1

        y0 = self.tot_profit_usds
        N = len(y0)
        x = list(range(0, N))
        ax0.plot(x, y0, "g-")
        ax0.set_title("Trading profit vs time", fontsize=FONTSIZE, fontweight="bold")
        ax0.set_xlabel("time", fontsize=FONTSIZE)
        ax0.set_ylabel("trading profit (USD)", fontsize=FONTSIZE)

        y1_est, y1_l, y1_u = [], [], []  # est, 95% confidence intervals
        for i_ in range(N):
            n_correct = sum(self.corrects[: i_ + 1])
            n_trials = len(self.corrects[: i_ + 1])
            l, u = proportion_confint(count=n_correct, nobs=n_trials)
            y1_est.append(n_correct / n_trials * 100)
            y1_l.append(l * 100)
            y1_u.append(u * 100)

        ax1.cla()
        ax1.plot(x, y1_est, "b")
        ax1.fill_between(x, y1_l, y1_u, color="b", alpha=0.15)
        now_s = f"{y1_est[-1]:.2f}% [{y1_l[-1]:.2f}%, {y1_u[-1]:.2f}%]"
        ax1.set_title(
            f"% correct vs time. {now_s}", fontsize=FONTSIZE, fontweight="bold"
        )
        ax1.set_xlabel("time", fontsize=FONTSIZE)
        ax1.set_ylabel("% correct", fontsize=FONTSIZE)

        HEIGHT = 8  # magic number
        WIDTH = HEIGHT * 2  # magic number
        fig.set_size_inches(WIDTH, HEIGHT)
        fig.tight_layout(pad=1.0)  # add space between plots
        plt.pause(0.001)
        plt.savefig("simulation_plot.png")

    @enforce_types
    def _log(self, s: str):
        """Log to both stdout and to file"""
        print(s)
        with open(self.logfile, "a") as f:
            f.write(s + "\n")

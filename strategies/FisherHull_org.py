import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


def _wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(series, period):
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = _wma(series, half_period)
    wma2 = _wma(series, period)
    hull = 2 * wma1 - wma2
    return _wma(hull, sqrt_period)


class FisherHull_org(IStrategy):
    """
    Original FisherHull strategy from mikedigriz/freqtrade-strategy-mikedigriz
    Autor original: https://github.com/werkkrew/freqtrade-strategies
    Ported to modern freqtrade API (>= 2023.x), long-only, no futures changes.
    """

    buy_params = {}
    sell_params = {}

    minimal_roi = {'0': 1000}

    stoploss = -0.27654

    trailing_stop = True
    trailing_stop_positive = 0.32606
    trailing_stop_positive_offset = 0.33314
    trailing_only_offset_is_reached = True

    timeframe = '1m'
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hma'] = _hma(dataframe['close'], 14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['hma'] < dataframe['hma'].shift()) &
                (dataframe['cci'] <= -50.0) &
                (dataframe['fisher_rsi'] < -0.5) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['hma'] > dataframe['hma'].shift()) &
                (dataframe['cci'] >= 100.0) &
                (dataframe['fisher_rsi'] > 0.5) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1
        return dataframe

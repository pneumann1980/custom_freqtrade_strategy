import numpy as np
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


class EasyInEasyOut_org(IStrategy):
    """
    Original EasyInEasyOut strategy from mikedigriz/freqtrade-strategy-mikedigriz
    Ported to modern freqtrade API (>= 2023.x), long-only, no futures changes.
    """

    buy_params = {}
    sell_params = {}

    minimal_roi = {
        "0": 0.02,
        "60": 0.03,
        "120": 0.02,
        "900": 0.01
    }

    stoploss = -1
    timeframe = '1m'
    exit_profit_only = True
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hma_20'] = _hma(dataframe['close'], 20)
        dataframe['close_prev'] = dataframe['close'].shift(2)
        dataframe['hma_20_prev'] = dataframe['hma_20'].shift(2)
        dataframe['close_curr'] = dataframe['close'].shift(1)
        dataframe['hma_20_current'] = dataframe['hma_20'].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close_curr'] > dataframe['hma_20_current']) &
                (dataframe['close_prev'] < dataframe['hma_20_prev'])
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

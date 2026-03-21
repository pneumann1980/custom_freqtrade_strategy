import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter


def _chaikin_money_flow(dataframe: DataFrame, period: int = 20) -> "DataFrame.Series":
    """
    Chaikin Money Flow (CMF) – inline implementation (no 'technical' package needed).
    """
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    volume = dataframe['volume']

    hl_diff = high - low
    hl_diff = hl_diff.replace(0, np.nan)

    mfm = ((close - low) - (high - close)) / hl_diff
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume

    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf


class SmartMoneyStrategy_org(IStrategy):
    """
    Original SmartMoneyStrategy from mikedigriz/freqtrade-strategy-mikedigriz
    Ported to modern freqtrade API (>= 2023.x), long-only, no futures changes.
    chaikin_money_flow implemented inline (no 'technical' package dependency).
    """

    minimal_roi = {
        "0": 10
    }

    stoploss = -1
    timeframe = '30m'
    exit_profit_only = True
    exit_profit_offset = 0.01
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cmf'] = _chaikin_money_flow(dataframe, period=20)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['mfi'] < 35) &
                (dataframe['cmf'] < -0.07)
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['mfi'] > 70) &
                (dataframe['cmf'] > 0.20)
            ),
            'exit_long'
        ] = 1
        return dataframe


class SmartMoneyStrategyHyperopt_org(IStrategy):
    """
    Original SmartMoneyStrategyHyperopt from mikedigriz/freqtrade-strategy-mikedigriz
    Ported to modern freqtrade API (>= 2023.x), long-only.
    """

    minimal_roi = {
        "0": 10
    }

    stoploss = -1
    timeframe = '1h'
    exit_profit_only = True
    exit_profit_offset = 0.01
    can_short = False

    buy_mfi = IntParameter(20, 60, default=35, space="buy")
    buy_cmf = DecimalParameter(-0.4, -0.01, decimals=2, default=-0.07, space="buy")

    sell_mfi = IntParameter(50, 95, default=70, space="sell")
    sell_cmf = DecimalParameter(0.1, 0.6, decimals=2, default=0.2, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cmf'] = _chaikin_money_flow(dataframe, period=20)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['mfi'] < self.buy_mfi.value) &
                (dataframe['cmf'] < self.buy_cmf.value)
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['mfi'] > self.sell_mfi.value) &
                (dataframe['cmf'] > self.sell_cmf.value)
            ),
            'exit_long'
        ] = 1
        return dataframe

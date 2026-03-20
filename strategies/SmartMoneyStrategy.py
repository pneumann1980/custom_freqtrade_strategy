"""
SmartMoney Strategy - Futures Version
Original: https://github.com/mikedigriz/freqtrade-strategy-mikedigriz
Updated for freqtrade >= 2023.x (new API) + Futures/Short support

Logic: Buy deep (below EMA200, oversold MFI/CMF), sell high (above EMA200, overbought MFI/CMF)
For futures shorts: enter short when overbought above EMA200, exit when oversold

Backtesting:
  freqtrade backtesting -s SmartMoneyStrategy --timerange 20210601- -i 30m -p BTC/USDT:USDT

Plot:
  freqtrade plot-dataframe -s SmartMoneyStrategy --timerange 20210601- -i 30m \
    -p BTC/USDT:USDT --indicators1 ema_200 --indicators2 cmf mfi
"""

import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

try:
    from technical.indicators import chaikin_money_flow
except ImportError:
    # Fallback: simple CMF implementation if 'technical' package not available
    def chaikin_money_flow(dataframe: DataFrame, period: int = 20) -> "pd.Series":
        import pandas as pd

        clv = (
            (dataframe["close"] - dataframe["low"])
            - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"]).replace(0, 1)
        mf = clv * dataframe["volume"]
        cmf = mf.rolling(period).sum() / dataframe["volume"].rolling(period).sum()
        return cmf


class SmartMoneyStrategy(IStrategy):
    """
    Smart money dip-buy / peak-short strategy.
    Long:  close < EMA200, MFI < 35, CMF < -0.07  (institutional accumulation dip)
    Short: close > EMA200, MFI > 70, CMF >  0.20  (institutional distribution peak)
    """

    # --- Futures ---
    can_short = True

    # --- ROI ---
    minimal_roi = {"0": 10}

    # --- Stoploss ---
    stoploss = -0.10

    timeframe = "30m"

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["cmf"] = chaikin_money_flow(dataframe, period=20)
        dataframe["mfi"] = ta.MFI(dataframe)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: dip below EMA200 with bearish money flow (smart money accumulating)
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_200"])
                & (dataframe["mfi"] < 35)
                & (dataframe["cmf"] < -0.07)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: peak above EMA200 with bullish money flow (smart money distributing)
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["mfi"] > 70)
                & (dataframe["cmf"] > 0.20)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: price back above EMA200 with distribution signals
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["mfi"] > 70)
                & (dataframe["cmf"] > 0.20)
            ),
            "exit_long",
        ] = 1

        # Exit short: price back below EMA200 with accumulation signals
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_200"])
                & (dataframe["mfi"] < 35)
                & (dataframe["cmf"] < -0.07)
            ),
            "exit_short",
        ] = 1

        return dataframe


class SmartMoneyStrategyHyperopt(IStrategy):
    """
    Hyperopt-optimizable version of SmartMoneyStrategy for futures.
    """

    # --- Futures ---
    can_short = True

    minimal_roi = {"0": 10}
    stoploss = -0.10
    timeframe = "1h"

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01

    # Buy params
    buy_mfi = IntParameter(20, 60, default=35, space="buy")
    buy_cmf = DecimalParameter(-0.4, -0.01, decimals=2, default=-0.07, space="buy")

    # Sell params
    sell_mfi = IntParameter(50, 95, default=70, space="sell")
    sell_cmf = DecimalParameter(0.1, 0.6, decimals=2, default=0.2, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["cmf"] = chaikin_money_flow(dataframe, period=20)
        dataframe["mfi"] = ta.MFI(dataframe)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_200"])
                & (dataframe["mfi"] < self.buy_mfi.value)
                & (dataframe["cmf"] < self.buy_cmf.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["mfi"] > self.sell_mfi.value)
                & (dataframe["cmf"] > self.sell_cmf.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["mfi"] > self.sell_mfi.value)
                & (dataframe["cmf"] > self.sell_cmf.value)
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_200"])
                & (dataframe["mfi"] < self.buy_mfi.value)
                & (dataframe["cmf"] < self.buy_cmf.value)
            ),
            "exit_short",
        ] = 1

        return dataframe

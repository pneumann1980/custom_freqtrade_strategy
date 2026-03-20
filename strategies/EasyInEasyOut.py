"""
EasyInEasyOut Strategy - Futures Version
Original: https://github.com/mikedigriz/freqtrade-strategy-mikedigriz
Updated for freqtrade >= 2023.x (new API) + Futures/Short support
"""
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


def hull_moving_average(series, window: int):
    """HMA implementation without external 'technical' package dependency."""
    half = int(window / 2)
    sqrt_w = int(np.sqrt(window))
    wma_half = ta.WMA(series, timeperiod=half)
    wma_full = ta.WMA(series, timeperiod=window)
    raw = 2 * wma_half - wma_full
    return ta.WMA(raw, timeperiod=sqrt_w)


class EasyInEasyOut(IStrategy):
    """
    HMA crossover scalping strategy for futures on 1m.
    Long:  price crosses above HMA(20)
    Short: price crosses below HMA(20)
    """

    # --- Futures ---
    can_short = True

    minimal_roi = {
        "0": 0.02,
        "60": 0.03,
        "120": 0.02,
        "900": 0.01,
    }

    stoploss = -0.03
    timeframe = "1m"

    use_exit_signal = True
    exit_profit_only = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hma_20"] = hull_moving_average(dataframe["close"], 20)
        dataframe["close_prev"] = dataframe["close"].shift(2)
        dataframe["hma_20_prev"] = dataframe["hma_20"].shift(2)
        dataframe["close_curr"] = dataframe["close"].shift(1)
        dataframe["hma_20_current"] = dataframe["hma_20"].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: price crosses above HMA
        dataframe.loc[
            (
                (dataframe["close_curr"] > dataframe["hma_20_current"])
                & (dataframe["close_prev"] < dataframe["hma_20_prev"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: price crosses below HMA
        dataframe.loc[
            (
                (dataframe["close_curr"] < dataframe["hma_20_current"])
                & (dataframe["close_prev"] > dataframe["hma_20_prev"])
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long on downward HMA cross
        dataframe.loc[
            (
                (dataframe["close_curr"] < dataframe["hma_20_current"])
                & (dataframe["close_prev"] > dataframe["hma_20_prev"])
            ),
            "exit_long",
        ] = 1

        # Exit short on upward HMA cross
        dataframe.loc[
            (
                (dataframe["close_curr"] > dataframe["hma_20_current"])
                & (dataframe["close_prev"] < dataframe["hma_20_prev"])
            ),
            "exit_short",
        ] = 1

        return dataframe

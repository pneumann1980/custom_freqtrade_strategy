"""
FisherHull Strategy - Futures Version
Original: https://github.com/mikedigriz/freqtrade-strategy-mikedigriz
Based on: https://github.com/werkkrew/freqtrade-strategies
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


class FisherHull(IStrategy):
    """
    Fisher RSI + Hull MA + CCI strategy for futures.
    Long:  HMA falling, CCI <= -50, Fisher RSI < -0.5  (oversold dip)
    Short: HMA rising,  CCI >= 100, Fisher RSI >  0.5  (overbought peak)
    """

    # --- Futures ---
    can_short = True

    # --- ROI ---
    minimal_roi = {"0": 1000}

    # --- Stoploss ---
    stoploss = -0.27654

    # --- Trailing stop ---
    trailing_stop = True
    trailing_stop_positive = 0.32606
    trailing_stop_positive_offset = 0.33314
    trailing_only_offset_is_reached = True

    timeframe = "1m"

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hma"] = hull_moving_average(dataframe["close"], 14)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        rsi_scaled = 0.1 * (dataframe["rsi"] - 50)
        dataframe["fisher_rsi"] = (np.exp(2 * rsi_scaled) - 1) / (
            np.exp(2 * rsi_scaled) + 1
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: HMA turning down (oversold), enter long on dip
        dataframe.loc[
            (
                (dataframe["hma"] < dataframe["hma"].shift(1))
                & (dataframe["cci"] <= -50.0)
                & (dataframe["fisher_rsi"] < -0.5)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: HMA turning up (overbought), enter short on peak
        dataframe.loc[
            (
                (dataframe["hma"] > dataframe["hma"].shift(1))
                & (dataframe["cci"] >= 100.0)
                & (dataframe["fisher_rsi"] > 0.5)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when HMA turns back up + overbought
        dataframe.loc[
            (
                (dataframe["hma"] > dataframe["hma"].shift(1))
                & (dataframe["cci"] >= 100.0)
                & (dataframe["fisher_rsi"] > 0.5)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        # Exit short when HMA turns back down + oversold
        dataframe.loc[
            (
                (dataframe["hma"] < dataframe["hma"].shift(1))
                & (dataframe["cci"] <= -50.0)
                & (dataframe["fisher_rsi"] < -0.5)
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe

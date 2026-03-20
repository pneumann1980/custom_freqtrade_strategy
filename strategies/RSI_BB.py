"""
RSI + Bollinger Bands Strategy - Futures Version
Original: https://github.com/mikedigriz/freqtrade-strategy-mikedigriz
Updated for freqtrade >= 2023.x (new API) + Futures/Short support
"""
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter

try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    import qtpylib.indicators as qtpylib


class RSI_BB(IStrategy):
    """
    RSI + Bollinger Bands mean-reversion strategy for futures.
    Long:  price closes below BB lower band 1σ  (oversold)
    Short: price closes above BB upper band 3σ + RSI > 56  (overbought)
    """

    # --- Futures ---
    can_short = True

    # --- ROI ---
    minimal_roi = {
        "0": 0.85,
        "11343": 0.407,
        "23766": 0.16,
        "41495": 0,
    }

    # --- Stoploss ---
    stoploss = -0.10

    timeframe = "15m"

    use_exit_signal = True
    exit_profit_only = False

    # Hyperopt parameters
    sell_rsi = IntParameter(50, 75, default=56, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)

        bollinger1 = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=1
        )
        dataframe["bb_lowerband1"] = bollinger1["lower"]
        dataframe["bb_upperband1"] = bollinger1["upper"]

        bollinger3 = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=3
        )
        dataframe["bb_upperband3"] = bollinger3["upper"]
        dataframe["bb_lowerband3"] = bollinger3["lower"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: price below 1σ lower band (oversold, mean-reversion buy)
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["bb_lowerband1"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: RSI overbought + price above 3σ upper band
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["close"] > dataframe["bb_upperband3"])
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: RSI overbought + price above 3σ upper band
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["close"] > dataframe["bb_upperband3"])
            ),
            "exit_long",
        ] = 1

        # Exit short: price back below 1σ lower band (oversold = cover)
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lowerband1"]),
            "exit_short",
        ] = 1

        return dataframe

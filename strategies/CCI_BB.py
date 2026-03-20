"""
CCI + Bollinger Bands Strategy - Futures Version
Original: https://github.com/mikedigriz/freqtrade-strategy-mikedigriz
Updated for freqtrade >= 2023.x (new API) + Futures/Short support
"""
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    import qtpylib.indicators as qtpylib


class CCI_BB(IStrategy):
    """
    CCI + Bollinger Bands strategy for futures.
    Long:  CCI <= -134 AND close < BB lower band
    Short: CCI >=  134 AND close > BB upper band
    """

    # --- Futures ---
    can_short = True

    minimal_roi = {
        "0": 0.02,
        "60": 0.04,
        "120": 0.02,
    }

    stoploss = -0.05
    timeframe = "5m"

    use_exit_signal = True
    exit_profit_only = False

    # Hyperopt parameters
    buy_cci = IntParameter(-200, -50, default=-134, space="buy")
    sell_cci = IntParameter(50, 200, default=134, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["cci"] = ta.CCI(dataframe)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_upperband"] = bollinger["upper"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: oversold CCI + price below lower BB
        dataframe.loc[
            (
                (dataframe["cci"] <= self.buy_cci.value)
                & (dataframe["close"] < dataframe["bb_lowerband"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: overbought CCI + price above upper BB
        dataframe.loc[
            (
                (dataframe["cci"] >= self.sell_cci.value)
                & (dataframe["close"] > dataframe["bb_upperband"])
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when CCI recovers above 0
        dataframe.loc[
            (dataframe["cci"] > 0),
            "exit_long",
        ] = 1

        # Exit short when CCI recovers below 0
        dataframe.loc[
            (dataframe["cci"] < 0),
            "exit_short",
        ] = 1

        return dataframe

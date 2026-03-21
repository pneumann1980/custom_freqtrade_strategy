import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame

try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    qtpylib = None


class CCI_BB_org(IStrategy):
    """
    Original CCI_BB strategy from mikedigriz/freqtrade-strategy-mikedigriz
    Ported to modern freqtrade API (>= 2023.x), long-only, no futures changes.
    """

    buy_params = {}
    sell_params = {}

    minimal_roi = {
        "0": 0.02,
        "60": 0.04,
        "120": 0.02,
    }

    stoploss = -1
    timeframe = '5m'
    exit_profit_only = False
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe)
        if qtpylib:
            bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        else:
            typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            std = typical_price.rolling(20).std()
            mid = typical_price.rolling(20).mean()
            bollinger1 = {'lower': mid - 2 * std}
        dataframe['bb_lowerband1'] = bollinger1['lower']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['cci'] <= -134) &
                (dataframe['close'] < dataframe['bb_lowerband1'])
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

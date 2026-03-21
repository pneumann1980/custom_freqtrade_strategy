import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy

try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    qtpylib = None


class RSI_BB_org(IStrategy):
    """
    Original RSI_BB strategy from mikedigriz/freqtrade-strategy-mikedigriz
    Ported to modern freqtrade API (>= 2023.x), long-only, no futures changes.
    ticker_interval removed (deprecated).
    """

    timeframe = '15m'

    minimal_roi = {
        "0": 0.85,
        "11343": 0.407,
        "23766": 0.16,
        "41495": 0
    }

    stoploss = -1
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        if qtpylib:
            bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
            bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        else:
            typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            std = typical_price.rolling(20).std()
            mid = typical_price.rolling(20).mean()
            bollinger1 = {'lower': mid - 1 * std}
            bollinger3 = {'upper': mid + 3 * std}
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_upperband3'] = bollinger3['upper']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['close'] < dataframe['bb_lowerband1']
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 56) &
                (dataframe['close'] > dataframe['bb_upperband3'])
            ),
            'exit_long'
        ] = 1
        return dataframe

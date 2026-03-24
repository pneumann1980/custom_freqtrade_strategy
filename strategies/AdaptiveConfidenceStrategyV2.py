# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    stoploss_from_open,
    stoploss_from_absolute,
    merge_informative_pair,
)
import talib.abstract as ta
from freqtrade.persistence import Trade

# ============================================================
#  STRATEGY: ADAPTIVE CONFIDENCE DAYTRADING V2
#  Timeframe: 15m (Entry) + 1h (Trend-Filter)
#  Assets: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
#
#  V2 improvements over V1:
#   - explicit pair param in custom_stake_amount (Freqtrade 2023+)
#   - ATR absolute stoploss via stoploss_from_absolute()
#   - vectorized confidence score (faster backtests)
#   - explicit pullback detection (recent RSI dip / EMA touch)
#   - stabilized 1h regime filter with hysteresis-like neutral zone
#   - conservative leverage cap (MAX 8x)
#   - removed unreliable daily wallet-loss guard
#   - reduced score/entry-filter redundancy
#   - NaN/Inf cleanup before score generation
#   - dead indicators removed
# ============================================================


class AdaptiveConfidenceStrategyV2(IStrategy):
    """
    Adaptive Confidence Daytrading Strategy V2

    Entry philosophy: Trend (1h) + Pullback + Re-Acceleration (15m).
    Enter when the larger trend is intact AND a short-term pullback
    has occurred AND momentum is resuming — not at momentum peaks.
    """

    INTERFACE_VERSION = 3

    # ── Timeframes ─────────────────────────────────────────
    timeframe = "15m"
    informative_timeframe = "1h"

    # ── Strategy flags ─────────────────────────────────────
    can_short = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    process_only_new_candles = True

    # Wide static fallback — real exits handled by custom_stoploss / custom_exit
    minimal_roi = {"0": 10.0}
    stoploss = -0.35
    trailing_stop = False

    # EMA200 on 1h needs 200 candles; extra buffer for rolling pullback window
    startup_candle_count = 240

    # ── Protections ─────────────────────────────────────────
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,   # 12h on 15m
                "trade_limit": 3,
                "stop_duration_candles": 4,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,   # 24h on 15m
                "trade_limit": 1,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.15,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 96,
                "trade_limit": 2,
                "stop_duration_candles": 24,
                "required_profit": 0.0,
            },
        ]

    # ── Hyperopt parameters ──────────────────────────────────
    rsi_period    = IntParameter(10, 20, default=14, space="buy", optimize=False)
    adx_period    = IntParameter(10, 20, default=14, space="buy", optimize=False)
    atr_period    = IntParameter(10, 20, default=14, space="buy", optimize=False)
    ema_fast      = IntParameter(15, 30, default=21, space="buy", optimize=False)
    ema_slow      = IntParameter(40, 80, default=55, space="buy", optimize=False)
    volume_ma     = IntParameter(15, 30, default=20, space="buy", optimize=False)

    min_confidence   = IntParameter(45, 75, default=56, space="buy", optimize=True)
    atr_stop_mult    = DecimalParameter(1.2, 2.5, default=1.8, decimals=1, space="sell", optimize=True)
    atr_tp_mult      = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True)
    base_risk_pct    = DecimalParameter(0.005, 0.02, default=0.01, decimals=3, space="buy", optimize=False)

    adx_min_15m      = IntParameter(15, 30, default=18, space="buy", optimize=True)
    vol_confirm_ratio = DecimalParameter(1.0, 1.8, default=1.1, decimals=1, space="buy", optimize=True)
    pullback_lookback = IntParameter(2, 6, default=3, space="buy", optimize=True)

    # 1h regime ADX thresholds (not hyperopt by default — change if needed)
    neutral_adx_1h = IntParameter(18, 24, default=20, space="buy", optimize=False)
    trend_adx_1h   = IntParameter(22, 30, default=25, space="buy", optimize=False)

    # ── Constants ────────────────────────────────────────────
    MAX_LEVERAGE     = 8
    MIN_LEVERAGE     = 2
    MAX_OPEN_POSITIONS = 2
    NO_TRADE_HOURS   = {0, 1}   # 00:00–02:00 UTC (low liquidity)

    # ── Informative pairs ─────────────────────────────────────
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    # ── Helper: replace Inf/NaN in numeric columns ───────────
    @staticmethod
    def _sanitize(df: DataFrame, columns: list) -> DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        return df

    # ── Vectorized confidence score ───────────────────────────
    def _build_confidence(self, df: DataFrame, direction: str) -> pd.Series:
        """
        Four-component score (0–100):
          1. HTF regime alignment      (0–25)
          2. LTF trend structure       (0–20)
          3. Pullback quality          (0–25)
          4. Momentum re-acceleration  (0–20)
          5. Volume / ADX participation (0–10)
        """
        score = pd.Series(0.0, index=df.index)

        if direction == "long":
            # 1) HTF alignment
            score += np.where(df["regime_1h"] == "TRENDING_BULL", 25, 0)

            # 2) LTF structure: fast EMA above slow, fast EMA rising
            score += np.where(df["ema_fast_val"] > df["ema_slow_val"], 12, 0)
            score += np.where(df["ema_fast_slope"] > 0, 8, 0)

            # 3) Pullback quality: recent dip then recovery
            score += np.where((df["rsi"] >= 42) & (df["rsi"] <= 58), 12, 0)
            score += np.where(df["pullback_long"], 8, 0)
            # Price between fast and slow EMA → pulled back to support
            score += np.where(
                (df["close"] > df["ema_slow_val"]) & (df["close"] < df["ema_fast_val"]),
                5, 0,
            )

            # 4) Re-acceleration
            score += np.where(df["macd_hist"] > df["macd_hist_prev"], 8, 0)
            score += np.where(df["macd_cross_up"], 7, 0)
            score += np.where(df["rsi"] > df["rsi_prev"], 5, 0)

            # 5) Participation
            score += np.where(df["vol_ratio"] >= 1.5, 7,
                     np.where(df["vol_ratio"] >= 1.1, 4, 0))
            score += np.where(df["adx"] >= 25, 3,
                     np.where(df["adx"] >= 20, 1, 0))

            # Penalties
            score -= np.where(df["rsi"] > 68, 8, 0)        # overbought
            score -= np.where(df["close"] < df["ema_slow_val"], 10, 0)  # below trend EMA

        else:  # short
            # 1) HTF alignment
            score += np.where(df["regime_1h"] == "TRENDING_BEAR", 25, 0)

            # 2) LTF structure: fast EMA below slow, fast EMA falling
            score += np.where(df["ema_fast_val"] < df["ema_slow_val"], 12, 0)
            score += np.where(df["ema_fast_slope"] < 0, 8, 0)

            # 3) Pullback quality: recent bounce then rollover
            score += np.where((df["rsi"] >= 42) & (df["rsi"] <= 58), 12, 0)
            score += np.where(df["pullback_short"], 8, 0)
            # Price between fast and slow EMA → bounced to resistance
            score += np.where(
                (df["close"] < df["ema_slow_val"]) & (df["close"] > df["ema_fast_val"]),
                5, 0,
            )

            # 4) Re-acceleration (downward)
            score += np.where(df["macd_hist"] < df["macd_hist_prev"], 8, 0)
            score += np.where(df["macd_cross_down"], 7, 0)
            score += np.where(df["rsi"] < df["rsi_prev"], 5, 0)

            # 5) Participation
            score += np.where(df["vol_ratio"] >= 1.5, 7,
                     np.where(df["vol_ratio"] >= 1.1, 4, 0))
            score += np.where(df["adx"] >= 25, 3,
                     np.where(df["adx"] >= 20, 1, 0))

            # Penalties
            score -= np.where(df["rsi"] < 32, 8, 0)        # oversold
            score -= np.where(df["close"] > df["ema_slow_val"], 10, 0)  # above trend EMA

        return pd.Series(np.clip(score, 0, 100).astype(int), index=df.index)

    # ── Indicator calculation ─────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ── 15m base indicators ──────────────────────────────
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        macd_raw, macdsig, macdhist = ta.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd"]          = macd_raw
        dataframe["macd_signal"]   = macdsig
        dataframe["macd_hist"]     = macdhist

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Use column names that don't clash with the hyperopt IntParameter attributes
        dataframe["ema_fast_val"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow_val"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        dataframe["volume_ma"] = dataframe["volume"].rolling(self.volume_ma.value).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        # ── Derived / shifted series ──────────────────────────
        dataframe["rsi_prev"]          = dataframe["rsi"].shift(1)
        dataframe["macd_hist_prev"]    = dataframe["macd_hist"].shift(1)
        macd_prev                      = dataframe["macd"].shift(1)
        macd_signal_prev               = dataframe["macd_signal"].shift(1)

        dataframe["macd_cross_up"] = (
            (dataframe["macd"] > dataframe["macd_signal"]) &
            (macd_prev <= macd_signal_prev)
        )
        dataframe["macd_cross_down"] = (
            (dataframe["macd"] < dataframe["macd_signal"]) &
            (macd_prev >= macd_signal_prev)
        )

        # Fast EMA slope over 3 candles (direction indicator)
        dataframe["ema_fast_slope"] = dataframe["ema_fast_val"] - dataframe["ema_fast_val"].shift(3)

        # ── Pullback detection ────────────────────────────────
        lookback = int(self.pullback_lookback.value)
        recent_rsi_min  = dataframe["rsi"].rolling(lookback).min()
        recent_close_min = dataframe["close"].rolling(lookback).min()
        recent_rsi_max  = dataframe["rsi"].rolling(lookback).max()
        recent_close_max = dataframe["close"].rolling(lookback).max()

        # Long pullback: in recent candles RSI dipped OR price touched below fast EMA,
        # and now RSI is recovering
        dataframe["pullback_long"] = (
            (
                (recent_rsi_min < 45) |
                (recent_close_min < dataframe["ema_fast_val"])
            ) &
            (dataframe["rsi"] > dataframe["rsi_prev"])
        )

        # Short pullback: in recent candles RSI peaked OR price touched above fast EMA,
        # and now RSI is declining
        dataframe["pullback_short"] = (
            (
                (recent_rsi_max > 55) |
                (recent_close_max > dataframe["ema_fast_val"])
            ) &
            (dataframe["rsi"] < dataframe["rsi_prev"])
        )

        # ── 1h informative indicators ─────────────────────────
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        ).copy()

        informative["ema200"]   = ta.EMA(informative, timeperiod=200)
        informative["adx_1h_i"] = ta.ADX(informative, timeperiod=14)

        trend_adx   = int(self.trend_adx_1h.value)
        neutral_adx = int(self.neutral_adx_1h.value)

        informative["regime"] = "NEUTRAL"
        informative.loc[
            (informative["close"] > informative["ema200"]) &
            (informative["adx_1h_i"] >= trend_adx),
            "regime",
        ] = "TRENDING_BULL"
        informative.loc[
            (informative["close"] < informative["ema200"]) &
            (informative["adx_1h_i"] >= trend_adx),
            "regime",
        ] = "TRENDING_BEAR"
        informative.loc[
            informative["adx_1h_i"] <= neutral_adx,
            "regime",
        ] = "RANGING"

        dataframe = merge_informative_pair(
            dataframe,
            informative[["date", "ema200", "adx_1h_i", "regime"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        # ── NaN/Inf cleanup ────────────────────────────────────
        dataframe = self._sanitize(dataframe, [
            "rsi", "rsi_prev", "macd", "macd_signal", "macd_hist",
            "macd_hist_prev", "adx", "atr", "ema_fast_val", "ema_slow_val",
            "volume_ma", "vol_ratio", "ema_fast_slope",
        ])

        # ── Confidence scores (vectorized) ────────────────────
        dataframe["confidence_long"]  = self._build_confidence(dataframe, "long")
        dataframe["confidence_short"] = self._build_confidence(dataframe, "short")

        return dataframe

    # ── Dynamic leverage ─────────────────────────────────────
    def _get_leverage(self, confidence: int, atr: float, price: float,
                      max_leverage: float) -> int:
        if confidence >= 80:
            base_lev = 8
        elif confidence >= 70:
            base_lev = 6
        elif confidence >= 60:
            base_lev = 4
        elif confidence >= 50:
            base_lev = 3
        else:
            return 0   # below min_confidence → no trade

        atr_pct = (atr / price) * 100 if price > 0 else 2.0
        if atr_pct > 3.0:
            base_lev -= 2
        elif atr_pct > 2.0:
            base_lev -= 1
        elif atr_pct < 1.0:
            base_lev += 1

        return int(np.clip(base_lev, self.MIN_LEVERAGE, min(self.MAX_LEVERAGE, int(max_leverage))))

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return float(self.MIN_LEVERAGE)

        last       = dataframe.iloc[-1]
        confidence = int(last["confidence_long"] if side == "long" else last["confidence_short"])
        lev        = self._get_leverage(confidence, float(last["atr"]), float(last["close"]), max_leverage)

        return float(max(lev, self.MIN_LEVERAGE))

    # ── Entry logic ──────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"]  = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"]   = ""

        min_conf = int(self.min_confidence.value)
        adx_min  = int(self.adx_min_15m.value)
        vol_min  = float(self.vol_confirm_ratio.value)

        long_conditions = [
            dataframe["regime_1h"] == "TRENDING_BULL",
            dataframe["confidence_long"] >= min_conf,
            dataframe["adx"] >= adx_min,
            dataframe["close"] > dataframe["ema_slow_val"],          # 15m uptrend structure
            dataframe["pullback_long"],                               # genuine pullback happened
            dataframe["macd_hist"] > dataframe["macd_hist_prev"],    # momentum recovering
            dataframe["rsi"] > dataframe["rsi_prev"],                 # RSI recovering
            dataframe["rsi"] < 62,                                    # not overbought
            dataframe["vol_ratio"] >= vol_min,
            dataframe["volume"] > 0,
        ]

        short_conditions = [
            dataframe["regime_1h"] == "TRENDING_BEAR",
            dataframe["confidence_short"] >= min_conf,
            dataframe["adx"] >= adx_min,
            dataframe["close"] < dataframe["ema_slow_val"],          # 15m downtrend structure
            dataframe["pullback_short"],                              # genuine bounce happened
            dataframe["macd_hist"] < dataframe["macd_hist_prev"],    # momentum weakening
            dataframe["rsi"] < dataframe["rsi_prev"],                 # RSI declining
            dataframe["rsi"] > 38,                                    # not oversold
            dataframe["vol_ratio"] >= vol_min,
            dataframe["volume"] > 0,
        ]

        long_mask  = reduce(lambda a, b: a & b, long_conditions)
        short_mask = reduce(lambda a, b: a & b, short_conditions)

        dataframe.loc[long_mask,  "enter_long"]  = 1
        dataframe.loc[short_mask, "enter_short"] = 1

        dataframe.loc[long_mask,  "enter_tag"] = (
            "long_" + dataframe.loc[long_mask, "confidence_long"].astype(str)
        )
        dataframe.loc[short_mask, "enter_tag"] = (
            "short_" + dataframe.loc[short_mask, "confidence_short"].astype(str)
        )

        # Avoid low-liquidity hours (00:00–02:00 UTC)
        hour_mask = dataframe["date"].dt.hour.isin(self.NO_TRADE_HOURS)
        dataframe.loc[hour_mask, "enter_long"]  = 0
        dataframe.loc[hour_mask, "enter_short"] = 0

        return dataframe

    # ── Exit signal ──────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0

        # Exit only on genuine regime flip — not on every neutral patch (V1 bug)
        dataframe.loc[
            dataframe["regime_1h"].isin(["TRENDING_BEAR", "RANGING"]),
            "exit_long",
        ] = 1
        dataframe.loc[
            dataframe["regime_1h"].isin(["TRENDING_BULL", "RANGING"]),
            "exit_short",
        ] = 1

        return dataframe

    # ── Custom stoploss (ATR absolute) ───────────────────────
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return None   # keep current stop

        last = dataframe.iloc[-1]
        atr  = float(last["atr"])
        if atr <= 0 or trade.open_rate <= 0:
            return None

        atr_mult = float(self.atr_stop_mult.value)

        # ── Initial ATR stop from entry price ─────────────────
        if trade.is_short:
            stop_price = trade.open_rate + atr * atr_mult
        else:
            stop_price = trade.open_rate - atr * atr_mult

        # ── Break-even: once +1.2 ATR in profit, move stop to open ─
        be_trigger = (atr / trade.open_rate) * 1.2 * trade.leverage
        if current_profit >= be_trigger:
            stop_price = trade.open_rate

        # ── After partial exit: lock in 0.3 ATR of profit ────
        if trade.nr_of_successful_exits > 0:
            if trade.is_short:
                # For shorts: stop below open rate locks in profit
                stop_price = min(stop_price, trade.open_rate - atr * 0.3)
            else:
                # For longs: stop above open rate locks in profit
                stop_price = max(stop_price, trade.open_rate + atr * 0.3)

        return stoploss_from_absolute(
            stop_rate=stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    # ── Custom exit (ATR TP + momentum rollover) ─────────────
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        atr  = float(last["atr"])
        if atr <= 0 or trade.open_rate <= 0:
            return None

        atr_pct = atr / trade.open_rate
        full_tp  = atr_pct * float(self.atr_tp_mult.value) * trade.leverage

        # Full ATR take-profit
        if current_profit >= full_tp:
            return "atr_take_profit"

        regime = last.get("regime_1h", "NEUTRAL")

        # HTF regime flip → exit immediately
        if not trade.is_short and regime == "TRENDING_BEAR":
            return "htf_regime_flip"
        if trade.is_short and regime == "TRENDING_BULL":
            return "htf_regime_flip"

        # Momentum rollover while in profit → protect gains
        if current_profit > 0:
            if not trade.is_short:
                if (last["macd_hist"] < last["macd_hist_prev"]) and (last["rsi"] < last["rsi_prev"]):
                    return "momentum_rollover"
            else:
                if (last["macd_hist"] > last["macd_hist_prev"]) and (last["rsi"] > last["rsi_prev"]):
                    return "momentum_rollover"

        return None

    # ── Partial close at 50% of TP ───────────────────────────
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        atr  = float(last["atr"])
        if atr <= 0 or trade.open_rate <= 0:
            return None

        half_tp = (atr / trade.open_rate) * float(self.atr_tp_mult.value) * 0.5 * trade.leverage

        # One-time 50% partial close at half-TP
        if current_profit >= half_tp and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount * 0.5)

        return None

    # ── Position sizing ──────────────────────────────────────
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return proposed_stake

        # Hard cap: no more than MAX_OPEN_POSITIONS at once
        if len(Trade.get_open_trades()) >= self.MAX_OPEN_POSITIONS:
            return 0

        last = dataframe.iloc[-1]
        atr  = float(last["atr"])
        if atr <= 0 or current_rate <= 0 or leverage <= 0:
            return proposed_stake

        wallet      = self.wallets.get_total_stake_amount()
        risk_amount = wallet * float(self.base_risk_pct.value)

        # Stake = risk / (stop_pct * leverage)
        # stop_pct = ATR_stop_distance / price
        stop_dist = atr * float(self.atr_stop_mult.value)
        stop_pct  = stop_dist / current_rate
        if stop_pct <= 0:
            return proposed_stake

        stake = risk_amount / (stop_pct * leverage)

        if min_stake is not None:
            stake = max(stake, min_stake)
        stake = min(stake, max_stake)

        return float(stake)

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    stoploss_from_open,
)
import talib.abstract as ta
from freqtrade.persistence import Trade

# ================================================================
#  STRATEGY: CONFIDENCE-THRESHOLD FRAMEWORK
#  Basierend auf: Kuznetsov et al. (2025), Appl. Sci. 15, 11145
#
#  SCHICHT 1: Feature Engineering (12 Macro + 8 Orderbook-Proxies)
#  SCHICHT 2: Direction Classifier (Weighted Score → Sigmoid → P)
#  SCHICHT 3: Confidence-Threshold Execution Engine (θ-Gate)
#  SCHICHT 4: Walk-Forward θ-Kalibrierung (bot_start)
# ================================================================


class ConfidenceThresholdStrategy(IStrategy):
    """
    ML-freies Confidence-Threshold Framework nach Kuznetsov et al. (2025).

    Kernidee: Neural-Net-Classifier durch normierte Multi-Signal-Scores
    ersetzt. Softmax-Wahrscheinlichkeit → Sigmoid-Transformation.
    Trade nur wenn confidence > θ (kalibriert auf Precision ≥ 60%).
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True
    use_custom_stoploss = True
    position_adjustment_enable = True

    # Emergency exit only; real TP managed by custom_exit (ATR-based)
    minimal_roi = {"0": 10.0}
    stoploss = -0.35           # Hard fallback – must be wider than atr_pct*1.5*max_leverage
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 210  # EMA200 needs 200 candles

    # ── Protections (active in live + backtest --enable-protections) ─
    @property
    def protections(self):
        return [
            {
                # Stop trading after 3 consecutive stoploss hits in 12h
                "method": "StoplossGuard",
                "lookback_period_candles": 48,   # 48 × 15min = 12h
                "trade_limit": 3,
                "stop_duration_candles": 4,
                "only_per_pair": False,
            },
            {
                # Pause when drawdown exceeds 15% in 24h window
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,   # 96 × 15min = 24h
                "trade_limit": 1,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.15,
            },
            {
                # Stop trading a pair that lost money in last 24h
                "method": "LowProfitPairs",
                "lookback_period_candles": 96,
                "trade_limit": 2,
                "stop_duration_candles": 24,
                "required_profit": 0.0,
            },
        ]

    # ── Hyperopt-tunable confidence thresholds ──────────────────
    theta_execute = DecimalParameter(0.55, 0.75, default=0.65, decimals=2,
                                     space="buy", optimize=True)
    theta_high = DecimalParameter(0.75, 0.90, default=0.80, decimals=2,
                                  space="buy", optimize=False)
    theta_very_high = DecimalParameter(0.85, 0.95, default=0.90, decimals=2,
                                       space="buy", optimize=False)

    # ── ATR multipliers ─────────────────────────────────────────
    atr_stop_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=1,
                                     space="sell", optimize=True)
    atr_tp_mult = DecimalParameter(2.0, 5.0, default=3.0, decimals=1,
                                   space="sell", optimize=True)

    # ── Deadband ────────────────────────────────────────────────
    deadband_bps = IntParameter(5, 20, default=10, space="buy", optimize=False)

    # ── Hard confluence filters ──────────────────────────────────
    adx_min_entry = IntParameter(15, 35, default=22, space="buy", optimize=True)
    vol_confirm_ratio = DecimalParameter(1.05, 1.60, default=1.15, decimals=2,
                                         space="buy", optimize=True)

    # ── Constants ───────────────────────────────────────────────
    PREDICTION_HORIZON_MIN = 600    # Paper: 600-min prediction horizon
    CANDLE_TF_MIN = 15
    CALIB_WINDOW = 500              # Candles used for walk-forward calibration
    LEV_MIN = 2
    LEV_MAX = 8
    RISK_PER_TRADE = 0.01
    MAX_OPEN_POSITIONS = 2
    NO_TRADE_HOURS = {0, 1}         # 00:00–02:00 UTC

    # Per-pair walk-forward calibrated θ (updated in bot_start)
    _pair_theta: Dict[str, float] = {}

    # ── Feature weights (from paper feature-importance analysis) ─
    # Positive = bullish contribution, negative = bearish/noise
    # Sum ≈ 1.0 for stable sigmoid input scaling
    WEIGHTS = {
        # Macro features
        "ret_1":            0.04,
        "ret_5":            0.06,
        "ret_20":           0.05,
        "ema_spread_short": 0.08,
        "ema_spread_trend": 0.07,
        "rsi_norm":         0.09,   # Paper: RSI = top indicator
        "macd_norm":        0.06,
        "macd_hist_norm":   0.07,
        "adx_norm":         0.05,
        "volume_ratio_c":   0.04,   # volume_ratio centered at 0
        "obv_slope":        0.05,
        "vol_ratio_neg":    0.04,   # high vol → negative contribution
        # Orderbook proxies
        "buy_pressure":     0.09,   # Paper: imbalance = strongest proxy
        "buy_pressure_ma":  0.07,
        "vwap_deviation":   0.06,
        "absorption":       0.05,
        "micro_momentum":   0.06,
        "tick_imbalance":   0.07,
    }
    # Total = 1.10 — intentional: sigmoid scaling absorbs this

    # ================================================================
    #  SCHICHT 1: FEATURE ENGINEERING
    # ================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ── Macro: Return series ─────────────────────────────────
        dataframe["ret_1"] = dataframe["close"].pct_change(1)
        dataframe["ret_5"] = dataframe["close"].pct_change(5)
        dataframe["ret_20"] = dataframe["close"].pct_change(20)

        # ── Macro: EMAs & spreads ────────────────────────────────
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # Spreads normalized by price → direction signal
        dataframe["ema_spread_short"] = (
            (dataframe["ema_12"] - dataframe["ema_26"]) / dataframe["close"]
        )
        dataframe["ema_spread_trend"] = (
            (dataframe["close"] - dataframe["ema_200"]) / dataframe["close"]
        )

        # ── Macro: Volatility regime ─────────────────────────────
        dataframe["vol_5"] = dataframe["ret_1"].rolling(5).std()
        dataframe["vol_30"] = dataframe["ret_1"].rolling(30).std()
        dataframe["vol_ratio"] = (
            dataframe["vol_5"] / dataframe["vol_30"].replace(0, np.nan)
        ).fillna(1.0)

        # ── Macro: RSI (normalized to [-1, +1]) ──────────────────
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_norm"] = (dataframe["rsi_14"] - 50.0) / 50.0

        # ── Macro: MACD (normalized by price) ────────────────────
        macd, macdsig, macdhist = ta.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd"] = macd
        dataframe["macd_signal"] = macdsig
        dataframe["macd_hist"] = macdhist
        dataframe["macd_norm"] = dataframe["macd"] / dataframe["close"]
        dataframe["macd_hist_norm"] = dataframe["macd_hist"] / dataframe["close"]

        # ── Macro: ADX (normalized to [0, 1]) ────────────────────
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["adx_norm"] = dataframe["adx"] / 100.0

        # ── Macro: ATR ───────────────────────────────────────────
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        # ── Macro: Volume features ───────────────────────────────
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_ma"].replace(0, np.nan)
        ).fillna(1.0)
        # Centered: >0 = above average, <0 = below
        dataframe["volume_ratio_c"] = dataframe["volume_ratio"] - 1.0

        # OBV slope (10-candle momentum, clipped for stability)
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_slope"] = (
            (dataframe["obv"] - dataframe["obv"].shift(10))
            / dataframe["obv"].shift(10).replace(0, np.nan)
        ).fillna(0.0).clip(-1.0, 1.0)

        # ── Orderbook Proxy: Spread ──────────────────────────────
        dataframe["spread_proxy"] = (
            (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        )

        # ── Orderbook Proxy: Buy pressure ────────────────────────
        # +1 = close at high (buyers dominate), -1 = close at low
        candle_range = dataframe["high"] - dataframe["low"]
        dataframe["buy_pressure"] = np.where(
            candle_range > 0,
            ((dataframe["close"] - dataframe["low"]) / candle_range) * 2.0 - 1.0,
            0.0,
        )
        dataframe["buy_pressure_ma"] = dataframe["buy_pressure"].rolling(5).mean()

        # ── Orderbook Proxy: VWAP deviation (14-period) ──────────
        tp = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        vwap_14 = (
            (tp * dataframe["volume"]).rolling(14).sum()
            / dataframe["volume"].rolling(14).sum()
        )
        dataframe["vwap_deviation"] = (
            (dataframe["close"] - vwap_14) / vwap_14.replace(0, np.nan)
        ).fillna(0.0)

        # ── Orderbook Proxy: Absorption ──────────────────────────
        # Large candle body × buy_pressure = absorbed selling/buying
        body = (dataframe["close"] - dataframe["open"]).abs()
        avg_body = body.rolling(20).mean()
        dataframe["absorption"] = (
            (body / avg_body.replace(0, np.nan)).fillna(1.0) * dataframe["buy_pressure"]
        ).clip(-3.0, 3.0)

        # ── Orderbook Proxy: Micro momentum ──────────────────────
        dataframe["micro_momentum"] = (
            dataframe["buy_pressure"].rolling(3).mean()
            - dataframe["buy_pressure"].rolling(10).mean()
        )

        # ── Orderbook Proxy: Tick imbalance ──────────────────────
        # 5-candle green count → [-1, +1]
        is_green = (dataframe["close"] > dataframe["open"]).astype(float)
        green_5 = is_green.rolling(5).sum()
        dataframe["tick_imbalance"] = (green_5 - 2.5) / 2.5

        # ── Orderbook Proxy: Liquidity score (informational) ─────
        dataframe["liquidity_score"] = dataframe["volume_ratio"] / (
            1.0 + dataframe["spread_proxy"] * 100.0
        )

        # ================================================================
        #  SCHICHT 2: DIRECTION CLASSIFIER (vectorized)
        # ================================================================

        # Negative vol_ratio contribution: high vol → lower confidence
        vol_ratio_neg = -(dataframe["vol_ratio"] - 1.0).clip(0.0, 1.0)

        # Helper: clip each feature to [-1, +1] before weighting
        def c(s: pd.Series, scale: float = 1.0) -> pd.Series:
            return (s * scale).clip(-1.0, 1.0)

        # Weighted raw score (direction + magnitude)
        raw_score = (
            self.WEIGHTS["ret_1"]            * c(dataframe["ret_1"],            100.0)
          + self.WEIGHTS["ret_5"]            * c(dataframe["ret_5"],             20.0)
          + self.WEIGHTS["ret_20"]           * c(dataframe["ret_20"],            10.0)
          + self.WEIGHTS["ema_spread_short"] * c(dataframe["ema_spread_short"], 100.0)
          + self.WEIGHTS["ema_spread_trend"] * c(dataframe["ema_spread_trend"],  20.0)
          + self.WEIGHTS["rsi_norm"]         * c(dataframe["rsi_norm"])
          + self.WEIGHTS["macd_norm"]        * c(dataframe["macd_norm"],       1000.0)
          + self.WEIGHTS["macd_hist_norm"]   * c(dataframe["macd_hist_norm"],  1000.0)
          + self.WEIGHTS["adx_norm"]         * c(dataframe["adx_norm"])
          + self.WEIGHTS["volume_ratio_c"]   * c(dataframe["volume_ratio_c"])
          + self.WEIGHTS["obv_slope"]        * c(dataframe["obv_slope"])
          + self.WEIGHTS["vol_ratio_neg"]    * c(vol_ratio_neg)
          + self.WEIGHTS["buy_pressure"]     * c(dataframe["buy_pressure"])
          + self.WEIGHTS["buy_pressure_ma"]  * c(dataframe["buy_pressure_ma"])
          + self.WEIGHTS["vwap_deviation"]   * c(dataframe["vwap_deviation"],  100.0)
          + self.WEIGHTS["absorption"]       * c(dataframe["absorption"] / 3.0)
          + self.WEIGHTS["micro_momentum"]   * c(dataframe["micro_momentum"])
          + self.WEIGHTS["tick_imbalance"]   * c(dataframe["tick_imbalance"])
        )
        dataframe["raw_score"] = raw_score

        # Deadband filter: expected move in basis points
        expected_move_bps = raw_score.abs() * dataframe["atr_pct"] * 10000.0
        above_deadband = expected_move_bps >= self.deadband_bps.value

        # Sigmoid transformation: scale by 5 → steep curve around 0
        # raw_prob ∈ [0.5, 1.0] — maps to confidence in UP/DOWN direction
        scaled = raw_score * 5.0
        raw_prob = 1.0 / (1.0 + np.exp(-scaled.abs()))

        # Volatility penalty (high vol → lower confidence)
        vol_penalty = (dataframe["vol_ratio"] - 1.0).clip(0.0, 0.5) * 0.2

        # ADX bonus: clear trend → slight confidence boost
        adx_bonus = np.where(dataframe["adx"] > 30, 0.03, 0.0)

        # Weak trend penalty: ADX < 18 = choppy/no trend → reduce confidence
        weak_trend_penalty = np.where(dataframe["adx"] < 18, 0.05, 0.0)

        confidence = (raw_prob - vol_penalty + adx_bonus - weak_trend_penalty).clip(0.5, 1.0)

        # Apply deadband mask (no signal = 0.5)
        dataframe["confidence"] = np.where(above_deadband, confidence, 0.5)

        # ── Signal strength tier ─────────────────────────────────
        theta_very_high = float(self.theta_very_high.value)
        theta_high = float(self.theta_high.value)
        theta_exec = float(self.theta_execute.value)
        dataframe["signal_strength"] = np.select(
            [
                dataframe["confidence"] >= theta_very_high,
                dataframe["confidence"] >= theta_high,
                dataframe["confidence"] >= theta_exec,
            ],
            ["VERY_HIGH", "HIGH", "NORMAL"],
            default="LOW",
        )

        # ── Pullback-timing helpers ──────────────────────────────
        # Used in populate_entry_trend to filter out overbought/oversold entries
        dataframe["macd_hist_prev"] = dataframe["macd_hist"].shift(1)
        dataframe["rsi_prev"] = dataframe["rsi_14"].shift(1)

        # Signal direction: UP (long), DOWN (short), NONE (no trade)
        dataframe["signal_direction"] = np.where(
            above_deadband,
            np.where(raw_score > 0, "UP", "DOWN"),
            "NONE",
        )

        return dataframe

    # ── Per-pair θ lookup ────────────────────────────────────────
    def _get_theta_for_pair(self, pair: Optional[str]) -> float:
        """Return calibrated θ for pair, or global default if not calibrated."""
        if pair and pair in self._pair_theta:
            return float(self._pair_theta[pair])
        return float(self.theta_execute.value)

    # ================================================================
    #  SCHICHT 3: EXECUTION ENGINE – ENTRY / EXIT SIGNALS
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""

        pair = metadata.get("pair")
        theta = self._get_theta_for_pair(pair)
        adx_min = self.adx_min_entry.value
        vol_min = float(self.vol_confirm_ratio.value)

        # ── Long: UP signal above θ + EMA stacking + microstructure + pullback ─
        long_cond = (
            (df["signal_direction"] == "UP")
            & (df["confidence"] >= theta)
            & (df["close"] > df["ema_200"])
            # EMA stacking: confirms uptrend alignment across all timeframes
            & (df["ema_12"] > df["ema_26"])
            & (df["ema_26"] > df["ema_50"])
            & (df["adx"] >= adx_min)
            & (df["volume_ratio"] >= vol_min)
            # Pullback timing
            & (df["rsi_14"] < 58)                              # Not overbought
            & (df["rsi_14"] > df["rsi_prev"])                  # RSI recovering
            & (df["macd_hist"] > 0)                            # MACD hist positive
            & (df["macd_hist"] > df["macd_hist_prev"])         # MACD hist improving
            # Microstructure confirmation
            & (df["buy_pressure"] > 0.10)
            & (df["buy_pressure_ma"] > 0.05)
            & (df["micro_momentum"] > 0.0)
            & (df["liquidity_score"] > 0.50)
            & (~df["date"].dt.hour.isin(self.NO_TRADE_HOURS))
            & (df["volume"] > 0)
        )
        df.loc[long_cond, "enter_long"] = 1
        df.loc[long_cond, "enter_tag"] = (
            "conf_" + (df["confidence"] * 100).round(0).astype(int).astype(str)
        )

        # ── Short: DOWN signal above θ + EMA stacking + microstructure + pullback ─
        short_cond = (
            (df["signal_direction"] == "DOWN")
            & (df["confidence"] >= theta)
            & (df["close"] < df["ema_200"])
            # EMA stacking: confirms downtrend alignment
            & (df["ema_12"] < df["ema_26"])
            & (df["ema_26"] < df["ema_50"])
            & (df["adx"] >= adx_min)
            & (df["volume_ratio"] >= vol_min)
            # Pullback timing
            & (df["rsi_14"] > 42)                              # Not oversold
            & (df["rsi_14"] < df["rsi_prev"])                  # RSI declining
            & (df["macd_hist"] < 0)                            # MACD hist negative
            & (df["macd_hist"] < df["macd_hist_prev"])         # MACD hist worsening
            # Microstructure confirmation
            & (df["buy_pressure"] < -0.10)
            & (df["buy_pressure_ma"] < -0.05)
            & (df["micro_momentum"] < 0.0)
            & (df["liquidity_score"] > 0.50)
            & (~df["date"].dt.hour.isin(self.NO_TRADE_HOURS))
            & (df["volume"] > 0)
        )
        df.loc[short_cond, "enter_short"] = 1
        df.loc[short_cond, "enter_tag"] = (
            "conf_" + (df["confidence"] * 100).round(0).astype(int).astype(str)
        )

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        theta_high = float(self.theta_high.value)

        # Exit long on high-confidence reversal signal (DOWN)
        dataframe.loc[
            (dataframe["signal_direction"] == "DOWN")
            & (dataframe["confidence"] >= theta_high),
            "exit_long",
        ] = 1

        # Exit short on high-confidence reversal signal (UP)
        dataframe.loc[
            (dataframe["signal_direction"] == "UP")
            & (dataframe["confidence"] >= theta_high),
            "exit_short",
        ] = 1

        return dataframe

    # ── Custom stoploss: ATR-based + break-even after 1x ATR ────
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        atr = last["atr"]
        entry = trade.open_rate
        if entry <= 0:
            return self.stoploss

        atr_pct = atr / entry
        leverage = trade.leverage

        # ATR-based stop — multiply by leverage because current_profit in
        # futures is (price_change / open_rate) * leverage
        sl_distance = atr_pct * float(self.atr_stop_mult.value) * leverage

        # Two-tier break-even: 2 ATR MUST be checked before 1 ATR (order matters!)
        # +2 ATR → lock in +0.8% above entry
        if current_profit >= atr_pct * 2.0 * leverage:
            return stoploss_from_open(0.008, current_profit, is_short=trade.is_short,
                                      leverage=leverage)
        # +1 ATR → break-even + 0.2%
        if current_profit >= atr_pct * leverage:
            return stoploss_from_open(0.002, current_profit, is_short=trade.is_short,
                                      leverage=leverage)

        return -sl_distance

    # ── Custom exit: ATR TP + signal reversal exit ───────────────
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        atr = last["atr"]
        entry = trade.open_rate
        if entry <= 0:
            return None

        atr_pct = atr / entry
        tp_pct = atr_pct * float(self.atr_tp_mult.value) * trade.leverage

        # Full take-profit
        if current_profit >= tp_pct:
            return "atr_take_profit"

        # High-confidence signal reversal → close immediately
        direction = last["signal_direction"]
        confidence = last["confidence"]
        theta_high = float(self.theta_high.value)

        if not trade.is_short and direction == "DOWN" and confidence >= theta_high:
            return "signal_reversal_exit"
        if trade.is_short and direction == "UP" and confidence >= theta_high:
            return "signal_reversal_exit"

        return None

    # ── Partial close: 50% at half-TP; 30% on confidence drop ───
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
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        atr = last["atr"]
        entry = trade.open_rate
        if entry <= 0:
            return None

        atr_pct = atr / entry
        half_tp_pct = atr_pct * float(self.atr_tp_mult.value) * 0.5 * trade.leverage
        # 50% partial close at half-TP target (only once)
        if current_profit >= half_tp_pct and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 2.0)

        return None

    # ── Dynamic leverage from confidence tier ────────────────────
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return float(self.LEV_MIN)

        last = dataframe.iloc[-1]
        confidence = last["confidence"]
        atr_pct = last["atr_pct"] * 100.0

        # Confidence tier → base leverage
        if confidence >= float(self.theta_very_high.value):
            base_lev = 15
        elif confidence >= float(self.theta_high.value):
            base_lev = 10
        elif confidence >= 0.72:
            base_lev = 6
        else:
            base_lev = 3

        # ATR volatility scaling
        if atr_pct > 3.0:
            base_lev = int(base_lev * 0.5)
        elif atr_pct > 2.0:
            base_lev = int(base_lev * 0.7)

        return float(int(np.clip(base_lev, self.LEV_MIN, self.LEV_MAX)))

    # ── ATR-based position sizing ────────────────────────────────
    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        pair: str = "",
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        # Max open positions guard
        if len(Trade.get_open_trades()) >= self.MAX_OPEN_POSITIONS:
            return 0.0

        last = dataframe.iloc[-1]
        atr = last["atr"]
        price = current_rate

        wallet = self.wallets.get_total_stake_amount()
        risk_amount = wallet * self.RISK_PER_TRADE
        stop_dist = atr * float(self.atr_stop_mult.value)

        if stop_dist <= 0 or price <= 0:
            return proposed_stake

        # Formula: stake = risk_amount / (stop_pct * leverage)
        stop_pct = stop_dist / price
        stake = risk_amount / (stop_pct * leverage)

        if min_stake and stake < min_stake:
            stake = min_stake
        if stake > max_stake:
            stake = max_stake

        return stake

    # ================================================================
    #  SCHICHT 4: WALK-FORWARD θ KALIBRIERUNG
    # ================================================================

    def bot_start(self, **kwargs) -> None:
        """
        Called once at strategy load. Calibrates θ per-pair on the most recent
        500 candles. Precision target: ≥ 60% over PREDICTION_HORIZON_MIN window.
        """
        try:
            pairs = self.dp.current_whitelist()
            if not pairs:
                return

            horizon_candles = self.PREDICTION_HORIZON_MIN // self.CANDLE_TF_MIN
            min_candles = 500 + horizon_candles + self.startup_candle_count

            for pair in pairs:
                try:
                    df = self.dp.get_pair_dataframe(pair, self.timeframe)
                    if df is None or len(df) < min_candles:
                        continue

                    df_calc = df.tail(700 + horizon_candles).copy().reset_index(drop=True)
                    df_calc = self.populate_indicators(df_calc, {"pair": pair})

                    calibrated = self._calibrate_theta(df_calc, window=self.CALIB_WINDOW)
                    if calibrated is not None:
                        self._pair_theta[pair] = calibrated
                        print(
                            f"[ConfidenceThreshold] {pair}: walk-forward θ = {calibrated:.2f}"
                        )
                except Exception as pair_exc:
                    print(f"[ConfidenceThreshold] {pair}: calibration skipped: {pair_exc}")
        except Exception as e:
            print(f"[ConfidenceThreshold] bot_start error: {e}")

    def _calibrate_theta(
        self, df: DataFrame, window: int = 500
    ) -> Optional[float]:
        """
        Walk-forward θ search.
        For each candidate θ, measure directional precision over
        PREDICTION_HORIZON_MIN minutes ahead.
        Returns θ that maximises trade count with precision ≥ 60%.
        Tie-break: higher precision first, then lower θ (more opportunities).
        """
        horizon = self.PREDICTION_HORIZON_MIN // self.CANDLE_TF_MIN
        deadband = self.deadband_bps.value / 10000.0

        if len(df) < window + horizon:
            return None

        df_sub = df.tail(window + horizon).reset_index(drop=True)
        theta_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        best_theta: Optional[float] = None
        best_count = 0
        best_precision = 0.0

        for theta in theta_candidates:
            correct = 0
            total = 0

            for i in range(len(df_sub) - horizon):
                row = df_sub.iloc[i]
                confidence = row.get("confidence", 0.5)
                direction = row.get("signal_direction", "NONE")

                if confidence < theta or direction == "NONE":
                    continue

                future_close = df_sub.iloc[i + horizon]["close"]
                current_close = row["close"]
                if current_close <= 0:
                    continue

                future_ret = (future_close - current_close) / current_close
                if abs(future_ret) <= deadband:
                    continue  # Inside deadband → no label

                actual = "UP" if future_ret > 0 else "DOWN"
                if actual == direction:
                    correct += 1
                total += 1

            if total == 0:
                continue

            precision = correct / total

            if precision >= 0.60:
                # Primary: max trades; tie-break: higher precision, then lower θ
                if (total > best_count
                        or (total == best_count and precision > best_precision)):
                    best_theta = theta
                    best_count = total
                    best_precision = precision
                print(
                    f"[ConfidenceThreshold] θ={theta:.2f} | "
                    f"Precision={precision:.2%} | Trades={total}"
                )

        return best_theta if best_theta is not None else float(self.theta_execute.value)

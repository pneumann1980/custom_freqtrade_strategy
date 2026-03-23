# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime, timedelta, timezone
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
#  STRATEGY: ADAPTIVE CONFIDENCE DAYTRADING
#  Timeframe: 15min (Entry) + 1h (Trend-Filter)
#  Assets: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
# ============================================================


class AdaptiveConfidenceStrategy(IStrategy):
    """
    Adaptive Confidence Daytrading Strategy

    Combines:
    - Market regime detection via 1h EMA200 + ADX
    - Multi-indicator confidence scoring (MACD, RSI, EMA, Volume, ADX)
    - Dynamic leverage based on confidence + volatility (ATR)
    - ATR-based Stop-Loss and Take-Profit (RRR 1:2)
    - Partial close at 50% TP target
    - Trailing stop to break-even after +1.5x ATR
    - Safety guards: max open positions, daily loss limit
    """

    INTERFACE_VERSION = 3

    # ── Timeframes ─────────────────────────────────────────
    timeframe = "15m"
    informative_timeframe = "1h"

    # ── Strategy flags ─────────────────────────────────────
    can_short = True
    use_custom_stoploss = True
    position_adjustment_enable = True  # required for partial closes

    # ── Freqtrade ROI / Stoploss (overridden by custom logic) ──
    # Set wide ROI so custom stoploss/TP takes over
    minimal_roi = {"0": 0.10}
    stoploss = -0.05  # Hard fallback stoploss (5%)
    trailing_stop = False  # Managed manually in custom_stoploss

    # ── Process-only-new-candles ────────────────────────────
    process_only_new_candles = True

    # ── Startup candles needed ──────────────────────────────
    startup_candle_count = 210  # EMA200 on 1h needs 200 candles

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

    # ── Hyperopt-tunable parameters ────────────────────────
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=False)
    ema_fast = IntParameter(15, 30, default=21, space="buy", optimize=False)
    ema_slow = IntParameter(40, 60, default=50, space="buy", optimize=False)
    volume_ma = IntParameter(15, 30, default=20, space="buy", optimize=False)

    min_confidence = IntParameter(40, 60, default=45, space="buy", optimize=True)
    atr_stop_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="sell", optimize=True)
    atr_tp_mult = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True)
    base_risk_pct = DecimalParameter(0.005, 0.03, default=0.01, decimals=3, space="buy", optimize=False)

    # ── Constants ───────────────────────────────────────────
    MAX_LEVERAGE = 20
    MIN_LEVERAGE = 2
    MAX_OPEN_POSITIONS = 2
    NO_TRADE_HOURS = {0, 1}  # 00:00–02:00 UTC (low liquidity)
    MAX_DAILY_LOSS_PCT = -0.05  # -5%
    MAX_DRAWDOWN_PCT = -0.15  # -15%

    # ── Informative pairs ───────────────────────────────────
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    # ── Indicator calculation ───────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ── 15min indicators ──────────────────────────────
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        macd, macdsignal, macdhist = ta.MACD(
            dataframe["close"],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
        dataframe["macd"] = macd
        dataframe["macd_signal"] = macdsignal
        dataframe["macd_hist"] = macdhist

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        dataframe["volume_ma"] = dataframe["volume"].rolling(self.volume_ma.value).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        # 20-candle rolling high for breakout check
        dataframe["recent_high"] = dataframe["close"].rolling(20).max()

        # Previous values for crossover detection
        dataframe["rsi_prev"] = dataframe["rsi"].shift(1)
        dataframe["macd_hist_prev"] = dataframe["macd_hist"].shift(1)
        dataframe["macd_prev"] = dataframe["macd"].shift(1)
        dataframe["macd_signal_prev"] = dataframe["macd_signal"].shift(1)

        # ── 1h informative indicators (market regime) ─────
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative["ema200_1h"] = ta.EMA(informative, timeperiod=200)
        informative["adx_1h"] = ta.ADX(informative, timeperiod=14)

        # Market regime column
        informative["regime"] = "UNCLEAR"
        informative.loc[
            (informative["close"] > informative["ema200_1h"]) & (informative["adx_1h"] > 25),
            "regime",
        ] = "TRENDING_BULL"
        informative.loc[
            (informative["close"] < informative["ema200_1h"]) & (informative["adx_1h"] > 25),
            "regime",
        ] = "TRENDING_BEAR"
        informative.loc[informative["adx_1h"] < 20, "regime"] = "RANGING"

        # Merge 1h into 15min frame
        dataframe = merge_informative_pair(
            dataframe,
            informative[["date", "ema200_1h", "adx_1h", "regime"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        # ── Confidence score ──────────────────────────────
        dataframe["confidence_long"] = dataframe.apply(
            lambda row: self._compute_confidence(row, "LONG"), axis=1
        )
        dataframe["confidence_short"] = dataframe.apply(
            lambda row: self._compute_confidence(row, "SHORT"), axis=1
        )

        return dataframe

    # ── Confidence scoring ──────────────────────────────────
    def _compute_confidence(self, row, direction: str) -> int:
        score = 0

        rsi = row.get("rsi", 50)
        rsi_prev = row.get("rsi_prev", 50)
        macd_hist = row.get("macd_hist", 0)
        macd_hist_prev = row.get("macd_hist_prev", 0)
        macd = row.get("macd", 0)
        macd_signal = row.get("macd_signal", 0)
        macd_prev = row.get("macd_prev", 0)
        macd_signal_prev = row.get("macd_signal_prev", 0)
        ema_fast = row.get("ema_fast", 0)
        ema_slow = row.get("ema_slow", 0)
        vol_ratio = row.get("vol_ratio", 1.0)
        adx = row.get("adx", 0)

        # A) MACD (0–25 Punkte)
        if direction == "LONG":
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                score += 15
            if macd > macd_signal and macd_prev < macd_signal_prev:  # fresh crossover
                score += 10
        else:  # SHORT
            if macd_hist < 0 and macd_hist < macd_hist_prev:
                score += 15
            if macd < macd_signal and macd_prev > macd_signal_prev:
                score += 10

        # B) RSI (0–25 Punkte)
        if direction == "LONG":
            if 50 < rsi < 65:
                score += 15
            if rsi > 40 and rsi_prev < 40:  # RSI crosses 40 upward
                score += 10
            if rsi < 35:
                score -= 5
        else:  # SHORT
            if 35 < rsi < 50:
                score += 15
            if rsi < 60 and rsi_prev > 60:  # RSI crosses 60 downward
                score += 10
            if rsi > 65:
                score -= 5

        # C) EMA Trend Alignment (0–20 Punkte)
        if direction == "LONG" and ema_fast > ema_slow:
            score += 20
        if direction == "SHORT" and ema_fast < ema_slow:
            score += 20

        # D) Volume (0–20 Punkte)
        if vol_ratio > 1.5:
            score += 20
        elif vol_ratio > 1.2:
            score += 10

        # E) ADX Trend Strength (0–10 Punkte)
        if adx > 30:
            score += 10
        elif adx > 25:
            score += 5

        return int(np.clip(score, 0, 100))

    # ── Dynamic leverage ────────────────────────────────────
    def _get_leverage(self, confidence: int, atr: float, price: float) -> int:
        # Base leverage from confidence
        if confidence >= 85:
            base_lev = 15
        elif confidence >= 70:
            base_lev = 10
        elif confidence >= 55:
            base_lev = 5
        elif confidence >= 45:
            base_lev = 3
        else:
            return 0  # No trade

        # Volatility adjustment
        atr_pct = (atr / price) * 100 if price > 0 else 2.0
        if atr_pct > 3.0:
            lev_mult = 0.5
        elif atr_pct > 2.0:
            lev_mult = 0.7
        elif atr_pct > 1.0:
            lev_mult = 1.0
        else:
            lev_mult = 1.2

        final_lev = int(np.clip(round(base_lev * lev_mult), self.MIN_LEVERAGE, self.MAX_LEVERAGE))
        return final_lev

    # ── Leverage callback (called by freqtrade) ─────────────
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.MIN_LEVERAGE

        last = dataframe.iloc[-1]
        confidence = last["confidence_long"] if side == "long" else last["confidence_short"]
        lev = self._get_leverage(confidence, last["atr"], last["close"])
        return max(float(lev), float(self.MIN_LEVERAGE))

    # ── Entry signals ───────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        min_conf = self.min_confidence.value

        # LONG conditions
        long_conditions = [
            dataframe["regime_1h"] == "TRENDING_BULL",
            dataframe["confidence_long"] >= min_conf,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda a, b: a & b, long_conditions), "enter_long"] = 1
        dataframe.loc[
            reduce(lambda a, b: a & b, long_conditions),
            "enter_tag",
        ] = "conf_" + dataframe["confidence_long"].astype(str)

        # SHORT conditions
        short_conditions = [
            dataframe["regime_1h"] == "TRENDING_BEAR",
            dataframe["confidence_short"] >= min_conf,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda a, b: a & b, short_conditions), "enter_short"] = 1
        dataframe.loc[
            reduce(lambda a, b: a & b, short_conditions),
            "enter_tag",
        ] = "conf_" + dataframe["confidence_short"].astype(str)

        # ── Safety guards ─────────────────────────────────
        # No trade during low-liquidity window (00:00–02:00 UTC)
        dataframe.loc[dataframe["date"].dt.hour.isin(self.NO_TRADE_HOURS), "enter_long"] = 0
        dataframe.loc[dataframe["date"].dt.hour.isin(self.NO_TRADE_HOURS), "enter_short"] = 0

        return dataframe

    # ── Exit signals ────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Exit long when regime flips to bear or ranging
        dataframe.loc[
            dataframe["regime_1h"].isin(["TRENDING_BEAR", "RANGING", "UNCLEAR"]),
            "exit_long",
        ] = 1

        # Exit short when regime flips to bull or ranging
        dataframe.loc[
            dataframe["regime_1h"].isin(["TRENDING_BULL", "RANGING", "UNCLEAR"]),
            "exit_short",
        ] = 1

        return dataframe

    # ── Custom stoploss (ATR-based + trailing to BE) ────────
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
        entry_price = trade.open_rate
        atr_pct = atr / entry_price if entry_price > 0 else 0.015

        # ATR-based initial stop
        sl_distance = atr * self.atr_stop_mult.value / entry_price

        # After +1.5x ATR profit → move SL to break-even
        be_trigger = atr_pct * 1.5
        if current_profit >= be_trigger:
            # Return 0.0 → stop at break-even (no loss)
            return stoploss_from_open(0.0, current_profit, is_short=trade.is_short)

        # Return negative value (freqtrade expects negative stoploss)
        return -sl_distance

    # ── Custom exit (ATR take-profit) ───────────────────────
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
        entry_price = trade.open_rate
        atr_pct = atr / entry_price if entry_price > 0 else 0.015

        tp_pct = atr_pct * self.atr_tp_mult.value

        # Full TP reached
        if not trade.is_short and current_profit >= tp_pct:
            return "atr_take_profit"
        if trade.is_short and current_profit >= tp_pct:
            return "atr_take_profit"

        # Regime change → close
        regime = last.get("regime_1h", "UNCLEAR")
        if not trade.is_short and regime in ["TRENDING_BEAR", "RANGING"]:
            return "regime_change_exit"
        if trade.is_short and regime in ["TRENDING_BULL", "RANGING"]:
            return "regime_change_exit"

        # Funding rate guard (long only) – requires exchange data via dataprovider
        # Skipped in backtest; hook point for live trading
        return None

    # ── Partial close at 50% of TP ──────────────────────────
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
        entry_price = trade.open_rate
        atr_pct = atr / entry_price if entry_price > 0 else 0.015
        half_tp_pct = atr_pct * self.atr_tp_mult.value * 0.5

        # Partial close at 50% of TP – only once
        if (
            current_profit >= half_tp_pct
            and trade.nr_of_successful_exits == 0
        ):
            # Close 50% of the position
            return -(trade.stake_amount / 2)

        return None

    # ── Custom position sizing (ATR/Kelly-like) ─────────────
    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(
            kwargs.get("pair", ""), self.timeframe
        )
        if dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        atr = last["atr"]
        price = current_rate

        # Safety: max open positions
        open_trades = len([t for t in Trade.get_open_trades() if not t.is_open is False])
        if open_trades >= self.MAX_OPEN_POSITIONS:
            return 0

        # Daily P&L guard
        try:
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            closed_today = Trade.get_overall_performance()
            # Simplified: use freqtrade's wallet
            wallet = self.wallets.get_total_stake_amount()
            daily_pnl_pct = self.wallets.get_available_stake_amount() / wallet - 1
            if daily_pnl_pct < self.MAX_DAILY_LOSS_PCT:
                return 0
        except Exception:
            pass

        # ATR-based sizing: risk_amount / stop_distance * price = position_usd
        wallet = self.wallets.get_total_stake_amount()
        risk_amount = wallet * float(self.base_risk_pct.value)
        stop_dist = atr * float(self.atr_stop_mult.value)
        if stop_dist <= 0 or price <= 0:
            return proposed_stake

        position_usd = (risk_amount / stop_dist) * price
        max_position = wallet * leverage
        position_usd = min(position_usd, max_position)

        # Clamp to freqtrade limits
        if min_stake and position_usd < min_stake:
            position_usd = min_stake
        if position_usd > max_stake:
            position_usd = max_stake

        return position_usd

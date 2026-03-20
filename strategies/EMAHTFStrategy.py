# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
"""
EMAHTFStrategy – EMA-Cross mit HTF-Trendfilter und ATR-basiertem Stop/TP
Entwickelt für OKX Futures (Isolated Margin) mit Freqtrade >= 2023.x

Strategie-Übersicht:
    - Entry: EMA9/EMA21 Kreuzung auf 15m + 1h-Trendfilter (EMA50/EMA200)
    - Exit:  ATR-basierter Stop-Loss (1.5x ATR) und Take-Profit (3x ATR)
    - Trailing Stop ab 2x ATR Gewinn
    - Dynamischer Hebel per leverage_callback (max. 5x)
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


class EMAHTFStrategy(IStrategy):
    """
    EMA-Cross Strategie mit Higher-Timeframe Filter und ATR-Stop.

    Entry-Bedingungen (15m):
      - Long:  EMA9 kreuzt EMA21 von unten  UND  1h bullisch (EMA50 > EMA200)  UND  Volumen > Durchschnitt
      - Short: EMA9 kreuzt EMA21 von oben   UND  1h bearisch (EMA50 < EMA200)  UND  Volumen > Durchschnitt

    Exit-Bedingungen:
      - Stop-Loss:     1.5x ATR(14) unter/über Entry-Preis
      - Take-Profit:   3.0x ATR(14) über/unter Entry-Preis  → Risk-Reward 1:2
      - Trailing Stop: Aktiviert ab +2x ATR Gewinn (schützt laufende Gewinne)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Basis-Konfiguration
    # ──────────────────────────────────────────────────────────────────────────

    # Freqtrade-Versionsanforderung
    INTERFACE_VERSION = 3

    # Long und Short erlaubt (Futures)
    can_short: bool = True

    # Haupt-Timeframe (Einstiegssignale)
    timeframe = "15m"

    # HTF für Trendfilter
    informative_timeframe = "1h"

    # Wieviele historische Candles für Berechnungen benötigt werden
    startup_candle_count: int = 220  # EMA200 auf 1h braucht min. 200 Candles

    # Trailing Stop global deaktiviert – wird per custom_stoploss() gesteuert
    trailing_stop = False

    # Signal-Exits deaktiviert: Alle Exits laufen über custom_stoploss() (ATR Stop/TP/Trail)
    # EMA-Kreuzungs-Exits auf 15m erzeugen Whipsaws → massive Verluste
    use_exit_signal = False

    # Freqtrade stoploss als harter Fallback (sehr weit, da custom_stoploss aktiv)
    stoploss = -0.10

    # ROI-Tabelle – wird durch custom_stoploss / TP-Signal ersetzt
    # Hohe Werte damit ROI nicht vorzeitig auslöst
    minimal_roi = {
        "0": 0.30
    }

    # Prozessiere nur neue Candles (spart Rechenzeit)
    process_only_new_candles = True

    # ──────────────────────────────────────────────────────────────────────────
    # Optimierbare Parameter (Hyperopt-fähig)
    # ──────────────────────────────────────────────────────────────────────────

    # EMA-Perioden (15m)
    ema_fast = IntParameter(5, 20, default=9, space="buy", optimize=True)
    ema_slow = IntParameter(15, 50, default=21, space="buy", optimize=True)

    # ATR-Periode für Stop und TP
    atr_period = IntParameter(10, 20, default=14, space="sell", optimize=False)

    # Stop-Loss Multiplikator (x ATR)
    sl_atr_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True)

    # Take-Profit Multiplikator (x ATR) → RR 1:2 bei default
    tp_atr_mult = DecimalParameter(2.0, 6.0, default=3.0, decimals=1, space="sell", optimize=True)

    # Trailing-Stop Aktivierung (x ATR Gewinn bevor Trail aktiviert)
    trail_atr_mult = DecimalParameter(1.5, 4.0, default=2.0, decimals=1, space="sell", optimize=True)

    # Volumen-Filter: aktuelles Volumen muss > X-faches des Durchschnitts sein
    volume_factor = DecimalParameter(1.0, 2.5, default=1.2, decimals=1, space="buy", optimize=True)

    # HTF EMA-Perioden (1h Trendfilter)
    htf_ema_fast = IntParameter(30, 70, default=50, space="buy", optimize=False)
    htf_ema_slow = IntParameter(150, 250, default=200, space="buy", optimize=False)

    # ──────────────────────────────────────────────────────────────────────────
    # HTF-Daten: informative_pairs()
    # ──────────────────────────────────────────────────────────────────────────

    def informative_pairs(self):
        """
        Gibt alle Paare zurück, für die zusätzlich HTF-Daten (1h) geladen werden.
        Freqtrade lädt diese Daten automatisch und stellt sie in populate_indicators() bereit.
        """
        pairs = self.dp.current_whitelist()
        # Für jedes Paar aus der Whitelist 1h-Daten laden
        informative = [(pair, self.informative_timeframe) for pair in pairs]
        return informative

    # ──────────────────────────────────────────────────────────────────────────
    # Indikatoren
    # ──────────────────────────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Berechnet alle technischen Indikatoren für das Haupt-Timeframe (15m)
        und merged die HTF-Daten (1h) per merge_informative_pair().
        """

        # ── 1h HTF-Daten laden und Indikatoren berechnen ──────────────────────
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"],
            timeframe=self.informative_timeframe,
        )

        # EMA50 und EMA200 auf 1h für Trendfilter
        informative[f"ema{self.htf_ema_fast.value}"] = pta.ema(
            informative["close"], length=self.htf_ema_fast.value
        )
        informative[f"ema{self.htf_ema_slow.value}"] = pta.ema(
            informative["close"], length=self.htf_ema_slow.value
        )

        # HTF-Trendsignal: True = bullisch, False = bearisch
        informative["htf_bull"] = (
            informative[f"ema{self.htf_ema_fast.value}"]
            > informative[f"ema{self.htf_ema_slow.value}"]
        )
        informative["htf_bear"] = (
            informative[f"ema{self.htf_ema_fast.value}"]
            < informative[f"ema{self.htf_ema_slow.value}"]
        )

        # HTF-Daten in 15m-Dataframe mergen (forward-fill, kein Look-Ahead)
        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        # ── 15m Indikatoren ───────────────────────────────────────────────────

        # Schnelle und langsame EMA für Kreuzungssignal
        dataframe["ema_fast"] = pta.ema(dataframe["close"], length=self.ema_fast.value)
        dataframe["ema_slow"] = pta.ema(dataframe["close"], length=self.ema_slow.value)

        # ATR(14) für Stop-Loss und Take-Profit Berechnung
        atr_result = pta.atr(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            length=self.atr_period.value,
        )
        dataframe["atr"] = atr_result

        # Volumen-Durchschnitt (20 Perioden) für Volumen-Filter
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()

        # EMA-Kreuzung: Bullisch (golden cross) – EMA9 kreuzt EMA21 von unten
        dataframe["ema_cross_bull"] = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
        )

        # EMA-Kreuzung: Bearisch (death cross) – EMA9 kreuzt EMA21 von oben
        dataframe["ema_cross_bear"] = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
        )

        # Volumen-Bedingung: aktuelles Volumen übertrifft gleitenden Durchschnitt
        dataframe["volume_ok"] = dataframe["volume"] > (
            dataframe["volume_mean"] * self.volume_factor.value
        )

        return dataframe

    # ──────────────────────────────────────────────────────────────────────────
    # Entry-Signale
    # ──────────────────────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Definiert Long- und Short-Einstiegssignale.

        Long-Bedingungen:
          1. EMA9 kreuzt EMA21 von unten (bullisches Kreuzungssignal)
          2. 1h-Trend bullisch (EMA50 > EMA200 auf 1h)
          3. Volumen über Durchschnitt (Bestätigung der Bewegung)
          4. ATR vorhanden (Daten vollständig)

        Short-Bedingungen (gespiegelt):
          1. EMA9 kreuzt EMA21 von oben (bearisches Kreuzungssignal)
          2. 1h-Trend bearisch (EMA50 < EMA200 auf 1h)
          3. Volumen über Durchschnitt
          4. ATR vorhanden
        """

        # ── Long-Entry ────────────────────────────────────────────────────────
        dataframe.loc[
            (
                dataframe["ema_cross_bull"]              # EMA-Kreuzung bullisch
                & dataframe[f"htf_bull_{self.informative_timeframe}"]  # 1h-Trend bullisch
                & dataframe["volume_ok"]                 # Volumen-Filter
                & dataframe["atr"].notna()               # ATR-Daten vorhanden
                & (dataframe["volume"] > 0)              # Valide Candle
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ema_cross_bull_htf")

        # ── Short-Entry ───────────────────────────────────────────────────────
        dataframe.loc[
            (
                dataframe["ema_cross_bear"]              # EMA-Kreuzung bearisch
                & dataframe[f"htf_bear_{self.informative_timeframe}"]  # 1h-Trend bearisch
                & dataframe["volume_ok"]                 # Volumen-Filter
                & dataframe["atr"].notna()               # ATR-Daten vorhanden
                & (dataframe["volume"] > 0)              # Valide Candle
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "ema_cross_bear_htf")

        return dataframe

    # ──────────────────────────────────────────────────────────────────────────
    # Exit-Signale (Signal-basiert)
    # ──────────────────────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Signal-Exits deaktiviert (use_exit_signal = False).

        EMA-Kreuzungs-Exits auf 15m erzeugten massive Whipsaw-Verluste:
        670 Signal-Exits à -3.3% = -2246 USDT im Backtest.
        Alle Exits werden jetzt ausschließlich über custom_stoploss() gesteuert:
          - Stop-Loss:    1.5x ATR unter/über Entry
          - Take-Profit:  3.0x ATR über/unter Entry (RR 1:2)
          - Trailing Stop: Aktiviert ab 2x ATR Gewinn
        """
        return dataframe

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamischer Stop-Loss via custom_stoploss()
    # ──────────────────────────────────────────────────────────────────────────

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
        """
        ATR-basierter dynamischer Stop-Loss mit Trailing-Funktion.

        Logik:
          1. ATR des aktuellen 15m-Candles laden
          2. Stop-Loss = 1.5x ATR unter/über Entry-Preis (fester Basis-Stop)
          3. Take-Profit = 3.0x ATR über/unter Entry-Preis
          4. Trailing Stop: Sobald Gewinn >= 2x ATR → Stop zieht nach

        Rückgabewert ist IMMER negativ (Freqtrade-Konvention: relativer Verlust).
        Beispiel: -0.05 bedeutet Stop bei 5% unter aktuellem Preis.
        """

        # Aktuelle Candle-Daten laden
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            # Fallback: harter Stop aus stoploss-Konfiguration
            return self.stoploss

        # Letzte vollständige Candle
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr", None)

        if atr is None or np.isnan(atr) or atr <= 0:
            # Kein valider ATR → Fallback
            return self.stoploss

        # ── Stop-Loss Berechnung ──────────────────────────────────────────────
        entry_price = trade.open_rate

        # ATR-basierter Stop (absoluter Preisabstand vom Entry)
        sl_distance = self.sl_atr_mult.value * atr

        # ATR-basierter Take-Profit (absoluter Preisabstand vom Entry)
        tp_distance = self.tp_atr_mult.value * atr

        # ── Take-Profit prüfen ────────────────────────────────────────────────
        if trade.is_short:
            # Short: TP wenn Preis unter Entry - tp_distance gefallen
            tp_price = entry_price - tp_distance
            if current_rate <= tp_price:
                # Sofortiger Exit: minimaler Stop-Loss (0.01% unter aktuellem Kurs)
                return -0.0001
        else:
            # Long: TP wenn Preis über Entry + tp_distance gestiegen
            tp_price = entry_price + tp_distance
            if current_rate >= tp_price:
                # Sofortiger Exit: minimaler Stop-Loss (0.01% unter aktuellem Kurs)
                return -0.0001

        # ── Trailing Stop (aktiviert ab trail_atr_mult * ATR Gewinn) ──────────
        trail_activation_distance = self.trail_atr_mult.value * atr

        if trade.is_short:
            # Short Trade
            trail_activation_price = entry_price - trail_activation_distance
            if current_rate <= trail_activation_price:
                # Trail aktiv: Stop folgt 1x ATR über dem aktuellen Tief
                trail_stop_price = current_rate + (1.0 * atr)
                trail_stop_relative = (trail_stop_price / current_rate) - 1.0
                # Wert muss negativ sein (Abstand nach oben für Short)
                return -abs(trail_stop_relative)
        else:
            # Long Trade
            trail_activation_price = entry_price + trail_activation_distance
            if current_rate >= trail_activation_price:
                # Trail aktiv: Stop folgt 1x ATR unter dem aktuellen Hoch
                trail_stop_price = current_rate - (1.0 * atr)
                trail_stop_relative = 1.0 - (trail_stop_price / current_rate)
                return -abs(trail_stop_relative)

        # ── Basis Stop-Loss (noch kein Trail aktiv) ───────────────────────────
        if trade.is_short:
            # Short: Stop über Entry-Preis
            sl_price = entry_price + sl_distance
            sl_relative = (sl_price / current_rate) - 1.0
        else:
            # Long: Stop unter Entry-Preis
            sl_price = entry_price - sl_distance
            sl_relative = 1.0 - (sl_price / current_rate)

        # Sicherheit: Minimalwert -0.01 (1%), Maximalwert stoploss (-20%)
        result = -abs(sl_relative)
        return max(result, self.stoploss)

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamischer Hebel via leverage_callback()
    # ──────────────────────────────────────────────────────────────────────────

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
        """
        Dynamische Hebelsteuerung basierend auf dem Entry-Signal.

        Logik:
          - Standard: 3x Hebel (konservativ, für die meisten Entries)
          - Starkes Signal (ema_cross_bull_htf / ema_cross_bear_htf): 5x Hebel
          - Maximaler Hebel: 5x (OKX Isolated Margin Limit)

        Der tatsächliche Hebel wird durch min(gewünschter_hebel, max_leverage)
        nach oben begrenzt.
        """

        # Maximaler erlaubter Hebel aus der Exchange-Konfiguration
        max_allowed = min(5.0, max_leverage)

        # Starkes Signal → höherer Hebel
        if entry_tag in ("ema_cross_bull_htf", "ema_cross_bear_htf"):
            desired_leverage = 5.0
        else:
            # Standard-Hebel für andere Entries (z.B. manuelle oder Hyperopt-Entries)
            desired_leverage = 3.0

        return min(desired_leverage, max_allowed)

    # ──────────────────────────────────────────────────────────────────────────
    # Positionsgröße (optional: konstante Stake)
    # ──────────────────────────────────────────────────────────────────────────

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
        **kwargs,
    ) -> float:
        """
        Positionsgröße: Standardmäßig wird der von Freqtrade vorgeschlagene
        Stake-Betrag (basierend auf max_open_trades und verfügbarem Kapital)
        verwendet. Hier kann eine risiko-basierte Berechnung ergänzt werden.
        """
        # Einfache Implementierung: vorgeschlagenen Stake verwenden
        # (Freqtrade teilt das Kapital gleichmäßig auf max_open_trades auf)
        return proposed_stake

# EMAHTFStrategy – Freqtrade OKX Futures

Eine vollständige Freqtrade-Strategie für OKX Futures (Isolated Margin) mit:
- EMA9/EMA21 Kreuzungssignal auf 15m
- 1h Higher-Timeframe Trendfilter (EMA50/EMA200)
- ATR-basiertem Stop-Loss, Take-Profit und Trailing Stop
- Dynamischem Hebel bis 5x

## Strategie-Übersicht

| Parameter | Wert |
|---|---|
| Exchange | OKX Futures |
| Margin-Modus | Isolated |
| Haupt-Timeframe | 15m |
| HTF-Filter | 1h |
| Long + Short | Ja |
| Max. Hebel | 5x |
| Stop-Loss | 1.5x ATR(14) |
| Take-Profit | 3.0x ATR(14) |
| Trailing Stop | ab 2.0x ATR Gewinn |

---

## Voraussetzungen

- Docker + Docker Compose
- Freqtrade >= 2023.1
- OKX-Account (für Live-Trading: API-Key mit Futures-Berechtigung)

---

## Setup

### 1. Repository klonen

```bash
git clone <repo-url> freqtrade-okx-futures
cd freqtrade-okx-futures
```

### 2. Freqtrade Docker-Image herunterladen

```bash
docker pull freqtradeorg/freqtrade:stable
```

### 3. Verzeichnisstruktur anlegen

```bash
mkdir -p user_data/{logs,data,backtest_results,hyperopts}
```

---

## Dry-Run starten (Paper Trading)

Startet den Bot im Simulationsmodus mit 1000 USDT virtuellem Kapital:

```bash
docker run -d \
  --name freqtrade-ema-htf \
  -v $(pwd)/strategies:/freqtrade/user_data/strategies \
  -v $(pwd)/config_okx_futures.json:/freqtrade/config.json \
  -v $(pwd)/user_data:/freqtrade/user_data \
  -p 8080:8080 \
  freqtradeorg/freqtrade:stable \
  trade \
  --config /freqtrade/config.json \
  --strategy EMAHTFStrategy \
  --logfile /freqtrade/user_data/logs/freqtrade.log
```

Bot stoppen:
```bash
docker stop freqtrade-ema-htf && docker rm freqtrade-ema-htf
```

Logs live beobachten:
```bash
docker logs -f freqtrade-ema-htf
```

---

## Historische Daten herunterladen

Vor dem Backtesting müssen OHLCV-Daten heruntergeladen werden.

```bash
docker run --rm \
  -v $(pwd)/strategies:/freqtrade/user_data/strategies \
  -v $(pwd)/config_okx_futures.json:/freqtrade/config.json \
  -v $(pwd)/user_data:/freqtrade/user_data \
  freqtradeorg/freqtrade:stable \
  download-data \
  --config /freqtrade/config.json \
  --timeframes 15m 1h \
  --days 365
```

---

## Backtesting

Backtest über die letzten 365 Tage:

```bash
docker run --rm \
  -v $(pwd)/strategies:/freqtrade/user_data/strategies \
  -v $(pwd)/config_okx_futures.json:/freqtrade/config.json \
  -v $(pwd)/user_data:/freqtrade/user_data \
  freqtradeorg/freqtrade:stable \
  backtesting \
  --config /freqtrade/config.json \
  --strategy EMAHTFStrategy \
  --timerange 20240101- \
  --export trades \
  --export-filename user_data/backtest_results/backtest_result.json
```

Backtest-Ergebnisse anzeigen:

```bash
docker run --rm \
  -v $(pwd)/user_data:/freqtrade/user_data \
  freqtradeorg/freqtrade:stable \
  backtesting-show \
  --export-filename user_data/backtest_results/backtest_result.json
```

---

## Hyperopt (Parameter-Optimierung)

Optimiert Entry- und Exit-Parameter mit 100 Iterationen:

```bash
docker run --rm \
  -v $(pwd)/strategies:/freqtrade/user_data/strategies \
  -v $(pwd)/config_okx_futures.json:/freqtrade/config.json \
  -v $(pwd)/user_data:/freqtrade/user_data \
  freqtradeorg/freqtrade:stable \
  hyperopt \
  --config /freqtrade/config.json \
  --strategy EMAHTFStrategy \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell \
  --epochs 100 \
  --timerange 20240101-
```

---

## Live-Trading auf OKX einrichten

> **Warnung:** Erst nach erfolgreichem Dry-Run und Backtesting auf Live umstellen!

### 1. API-Keys in config_okx_futures.json eintragen

```json
"exchange": {
  "name": "okx",
  "key": "DEIN_API_KEY",
  "secret": "DEIN_API_SECRET",
  "password": "DEIN_API_PASSPHRASE"
}
```

### 2. dry_run auf false setzen

```json
"dry_run": false,
"dry_run_wallet": 1000
```

### 3. OKX API-Berechtigungen

Die OKX API benötigt folgende Berechtigungen:
- **Read** – Kontoinformationen lesen
- **Trade** – Orders erstellen/stornieren
- **Futures** – Futures-Handel erlaubt

IP-Whitelist in OKX eintragen (empfohlen für Sicherheit).

---

## Konfiguration anpassen

### Handelspaar-Liste ändern

In `config_okx_futures.json` unter `pair_whitelist`:

```json
"pair_whitelist": [
  "BTC/USDT:USDT",
  "ETH/USDT:USDT",
  "SOL/USDT:USDT"
]
```

Das `:USDT` Suffix ist für Futures-Perpetuals auf OKX erforderlich.

### Stake-Betrag pro Trade

```json
"stake_amount": 50,
"max_open_trades": 5
```

Mit diesen Einstellungen werden max. 5 Trades à 50 USDT geöffnet.

### Telegram-Benachrichtigungen aktivieren

```json
"telegram": {
  "enabled": true,
  "token": "DEIN_BOT_TOKEN",
  "chat_id": "DEINE_CHAT_ID"
}
```

---

## FreqUI (Web-Interface)

Der Bot stellt eine REST-API unter Port 8080 bereit.
FreqUI kann lokal gestartet werden:

```bash
docker run -d \
  --name frequi \
  -p 3000:3000 \
  freqtradeorg/frequi:stable
```

Dann FreqUI unter `http://localhost:3000` aufrufen und mit der Bot-API unter `http://localhost:8080` verbinden.

Zugangsdaten aus `config_okx_futures.json`:
- Username: `freqtrade`
- Password: `aendere-dieses-passwort`

---

## Projektstruktur

```
.
├── strategies/
│   └── EMAHTFStrategy.py      # Hauptstrategie
├── config_okx_futures.json    # Exchange- und Bot-Konfiguration
├── user_data/
│   ├── data/                  # Heruntergeladene OHLCV-Daten
│   ├── logs/                  # Bot-Logs
│   └── backtest_results/      # Backtest-Ergebnisse
└── README.md
```

---

## Sicherheitshinweise

- **API-Keys niemals in Git eincommiten** – `.gitignore` für `config_okx_futures.json` oder Keys als Umgebungsvariablen übergeben
- Immer erst **Dry-Run testen**, bevor echtes Kapital eingesetzt wird
- **Backtesting ≠ Echtgeld-Performance** – vergangene Ergebnisse sind keine Garantie
- Maximalen Hebel und Stake-Betrag dem eigenen Risikoprofil anpassen

---

## Lizenz

MIT

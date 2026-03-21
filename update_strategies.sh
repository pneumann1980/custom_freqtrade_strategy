#!/bin/bash
# ============================================================
#  update_strategies.sh
#  Freqtrade auf Unraid – Strategien aus Git aktualisieren
#  Dieses Skript auf dem Unraid-Host ausführen (nicht im Docker)
# ============================================================

# --- ANPASSEN falls dein Pfad abweicht ---
REPO_DIR="/mnt/user/appdata/freqtrade/custom_freqtrade_strategy"
TARGET_DIR="/mnt/user/appdata/freqtrade/user_data/strategies"
# -----------------------------------------

# 1. Git Pull
cd "$REPO_DIR" && git pull origin claude/freqtrade-okx-futures-strategy-GA2T3

# 2. Strategien kopieren
cp -v "$REPO_DIR"/strategies/*.py "$TARGET_DIR/"

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

echo "=== Freqtrade Strategy Updater ==="
echo ""

# 1. Prüfe ob Repo-Verzeichnis existiert
if [ ! -d "$REPO_DIR" ]; then
  echo "[INFO] Repo nicht gefunden – klone es..."
  git clone https://github.com/pneumann1980/custom_freqtrade_strategy "$REPO_DIR"
  if [ $? -ne 0 ]; then
    echo "[FEHLER] Git clone fehlgeschlagen!"
    exit 1
  fi
else
  echo "[INFO] Repo gefunden: $REPO_DIR"
  echo "[INFO] Führe git pull aus..."
  cd "$REPO_DIR" && git pull origin claude/freqtrade-okx-futures-strategy-GA2T3
  if [ $? -ne 0 ]; then
    echo "[FEHLER] Git pull fehlgeschlagen!"
    exit 1
  fi
fi

# 2. Zielverzeichnis sicherstellen
mkdir -p "$TARGET_DIR"

# 3. Strategien kopieren
echo ""
echo "[INFO] Kopiere Strategien nach $TARGET_DIR ..."
cp -v "$REPO_DIR"/strategies/*.py "$TARGET_DIR/"

if [ $? -eq 0 ]; then
  echo ""
  echo "=== Fertig! Folgende Strategien sind jetzt aktuell: ==="
  ls "$TARGET_DIR"/*.py | xargs -I{} basename {}
else
  echo "[FEHLER] Kopieren fehlgeschlagen!"
  exit 1
fi

echo ""
echo "[HINWEIS] Falls der Bot läuft, muss er neu gestartet werden:"
echo "  docker restart freqtrade"

#!/usr/bin/env bash
# Start the viewer (macOS / Linux). The first run creates a private Python environment.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up for the first time..."
  python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
exec .venv/bin/python viewer.py "$@"

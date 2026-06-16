#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
/Users/alizhanaskarov/miniconda3/bin/python app.py --port "${1:-8765}"

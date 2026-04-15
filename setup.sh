#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  LLM Eval Framework — One-command setup
#  Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

echo ""
echo "══════════════════════════════════════════════"
echo "  LLM Eval Framework — Setup"
echo "══════════════════════════════════════════════"
echo ""

# ── 1. Python version check ───────────────────────────────────
PYTHON=$(which python3 || which python)
PY_VER=$($PYTHON --version 2>&1)
echo "✓ Python: $PY_VER"

# ── 2. Install dependencies ───────────────────────────────────
echo ""
echo "Installing dependencies..."
$PYTHON -m pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# ── 3. Check / create .env ────────────────────────────────────
echo ""
if [ ! -f ".env" ]; then
    if [ -n "$OPENROUTER_API_KEY" ]; then
        echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" > .env
        echo "✓ .env created from environment variable"
    else
        cp .env.example .env
        echo ""
        echo "┌─────────────────────────────────────────────┐"
        echo "│  ACTION REQUIRED                            │"
        echo "│                                             │"
        echo "│  Open .env and paste your OpenRouter key:  │"
        echo "│  OPENROUTER_API_KEY=sk-or-...              │"
        echo "│                                             │"
        echo "│  Get a key free at: openrouter.ai/keys     │"
        echo "└─────────────────────────────────────────────┘"
        echo ""
        read -p "Press Enter once you've added your key to .env..."
    fi
else
    echo "✓ .env already exists"
fi

# ── 4. Validate key ───────────────────────────────────────────
echo ""
echo "Validating OpenRouter API key..."
$PYTHON -c "
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path('.env'))
except ImportError:
    pass

key = os.getenv('OPENROUTER_API_KEY', '')
if not key or key == 'sk-or-paste-your-key-here':
    print('✗ OPENROUTER_API_KEY is not set in .env')
    exit(1)

import requests
r = requests.get(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {key}'},
    timeout=8
)
if r.status_code == 200:
    n = len(r.json().get('data', []))
    print(f'✓ API key valid — {n} models available')
elif r.status_code == 401:
    print('✗ Invalid API key — check your .env file')
    exit(1)
else:
    print(f'⚠ Unexpected status {r.status_code} — continuing anyway')
"

# ── 5. Generate demo data ─────────────────────────────────────
echo ""
echo "Generating demo data (no API calls needed)..."
$PYTHON generate_demo_data.py
echo "✓ Demo data ready"

# ── 6. Launch dashboard ───────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete! Launching dashboard..."
echo "  Press Ctrl+C to stop"
echo "══════════════════════════════════════════════"
echo ""
$PYTHON -m streamlit run dashboard.py

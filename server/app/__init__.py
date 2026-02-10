"""Application package bootstrap hooks."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


_ROOT_DIR = Path(__file__).resolve().parent.parent  # server/
_REPO_ROOT = _ROOT_DIR.parent

# Load repo root .env first, then server/.env, then server/.env.local overrides.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_ROOT_DIR / ".env.local", override=True)

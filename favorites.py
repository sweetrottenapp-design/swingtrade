"""
favorites.py — Watchlist / Favorites Module
============================================
Persists to favorites.json in the same directory.
No database needed — plain JSON file.
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

FAVORITES_FILE = os.path.join(os.path.dirname(__file__), "favorites.json")

DEFAULT_FAVORITES = ["NVDA", "TSLA", "AAPL", "PLTR", "AMD", "META", "MSFT", "CRWD"]


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _load() -> dict:
    """Load the full favorites store from disk."""
    if not os.path.exists(FAVORITES_FILE):
        store = {
            "symbols": DEFAULT_FAVORITES,
            "folders": {"預設清單": DEFAULT_FAVORITES},
            "updated_at": datetime.utcnow().isoformat(),
        }
        _save(store)
        return store
    try:
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"[favorites] Load error: {e}")
        return {"symbols": DEFAULT_FAVORITES, "folders": {}, "updated_at": ""}


def _save(store: dict) -> None:
    """Persist the favorites store to disk."""
    store["updated_at"] = datetime.utcnow().isoformat()
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"[favorites] Save error: {e}")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_favorites() -> dict:
    """Return full favorites store."""
    return _load()


def get_symbols(folder: Optional[str] = None) -> list[str]:
    """Return list of symbols. If folder given, return that folder's symbols."""
    store = _load()
    if folder:
        return store.get("folders", {}).get(folder, [])
    return store.get("symbols", DEFAULT_FAVORITES)


def add_symbol(symbol: str, folder: Optional[str] = None) -> dict:
    """Add a symbol to favorites (and optionally to a folder)."""
    sym = symbol.upper().strip()
    store = _load()

    # Add to main list
    if sym not in store["symbols"]:
        store["symbols"].append(sym)
        log.info(f"[favorites] Added {sym}")

    # Add to folder
    if folder:
        folders = store.setdefault("folders", {})
        if folder not in folders:
            folders[folder] = []
        if sym not in folders[folder]:
            folders[folder].append(sym)

    _save(store)
    return {"ok": True, "symbol": sym, "symbols": store["symbols"]}


def remove_symbol(symbol: str, folder: Optional[str] = None) -> dict:
    """Remove a symbol from favorites (or just from a folder)."""
    sym = symbol.upper().strip()
    store = _load()

    if folder:
        # Remove only from folder
        folders = store.get("folders", {})
        if folder in folders and sym in folders[folder]:
            folders[folder].remove(sym)
    else:
        # Remove from everything
        if sym in store["symbols"]:
            store["symbols"].remove(sym)
        for folder_syms in store.get("folders", {}).values():
            if sym in folder_syms:
                folder_syms.remove(sym)

    _save(store)
    return {"ok": True, "symbol": sym, "symbols": store["symbols"]}


def create_folder(name: str, symbols: list[str] = None) -> dict:
    """Create a named watchlist folder."""
    store = _load()
    store.setdefault("folders", {})[name] = [s.upper() for s in (symbols or [])]
    _save(store)
    return {"ok": True, "folder": name}


def get_folders() -> dict:
    """Return all folders."""
    return _load().get("folders", {})


def is_favorite(symbol: str) -> bool:
    return symbol.upper() in get_symbols()

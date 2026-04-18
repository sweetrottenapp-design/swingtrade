"""
SwingTrader Pro v2 — Backend Server
=====================================
New in v2:
  • Favorites / Watchlist folders  (favorites.py)
  • News integration               (news.py)
  • News sentiment embedded in /api/watchlist & /api/snapshot

Install:
  pip install fastapi uvicorn yfinance pandas numpy

Run:
  python3 server.py

Open: http://localhost:8000
"""

import math, time, datetime, logging, os
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

import yfinance as yf
import pandas as pd
import numpy as np

# ── Our modules ────────────────────────────────────────────────
import favorites as fav_module
import news as news_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="SwingTrader Pro v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Simple cache ───────────────────────────────────────────────
_cache: dict = {}

def cache_get(key, ttl=60):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None

def cache_set(key, val):
    _cache[key] = (time.time(), val)
    return val


# ══════════════════════════════════════════════════════════════
# INDICATORS — pure pandas/numpy, NO pandas-ta needed
# ══════════════════════════════════════════════════════════════

def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return round(float(v), 2) if not math.isnan(v) else 50.0

def calc_macd(closes: pd.Series, fast=12, slow=26, signal=9):
    ema_f = closes.ewm(span=fast,   adjust=False).mean()
    ema_s = closes.ewm(span=slow,   adjust=False).mean()
    ml    = ema_f - ema_s
    sl    = ml.ewm(span=signal, adjust=False).mean()
    return (round(float(ml.iloc[-1]), 4),
            round(float(sl.iloc[-1]), 4),
            round(float((ml - sl).iloc[-1]), 4))

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr   = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 4) if not math.isnan(atr) else 0.0

def calc_rs(sym_c: pd.Series, spy_c: pd.Series):
    df = pd.DataFrame({"s": sym_c, "b": spy_c}).dropna()
    if len(df) < 63:
        return 50.0, 0.0, 0.0
    n   = len(df)
    s63 = (df["s"].iloc[-1] / df["s"].iloc[max(0,n-64)] - 1) * 100
    b63 = (df["b"].iloc[-1] / df["b"].iloc[max(0,n-64)] - 1) * 100
    s21 = (df["s"].iloc[-1] / df["s"].iloc[max(0,n-22)] - 1) * 100
    b21 = (df["b"].iloc[-1] / df["b"].iloc[max(0,n-22)] - 1) * 100
    comp = 0.6*(s63-b63) + 0.4*(s21-b21)
    rank = max(0.0, min(100.0, 50 + 50*math.tanh(comp/20)))
    return round(rank,1), round(s63-b63,2), round(s21-b21,2)


# ══════════════════════════════════════════════════════════════
# STRATEGY ENGINE
# ══════════════════════════════════════════════════════════════

def score_trend(p, ma20, ma50, ma200):
    s, n = 0, []
    if ma200:
        d = (p-ma200)/ma200*100
        if p > ma200: s += 10 if d>5 else 7; n.append(f"站上 200MA (+{d:.1f}%) ✓")
        else: n.append("低於 200MA ✗")
    if ma50:
        if p > ma50: s += 8; n.append(f"站上 50MA (+{(p-ma50)/ma50*100:.1f}%) ✓")
        else: n.append("低於 50MA ✗")
    if ma20:
        if p > ma20: s += 5; n.append("站上 20MA ✓")
        else: n.append(f"低於 20MA -{(ma20-p)/ma20*100:.1f}% ✗")
    if ma20 and ma50 and ma200:
        if ma20 > ma50 > ma200: s += 7; n.append("均線多頭排列完整 ✓✓")
        elif ma50 > ma200:      s += 4; n.append("中期多頭排列 ✓")
    return min(s, 100), n

def score_momentum(rsi, macd, ms, mh, p, ma10):
    s, n = 0, []
    if   rsi >= 78: s += 5;  n.append(f"RSI {rsi:.0f} — 過熱 ⚠")
    elif rsi >= 60: s += 10; n.append(f"RSI {rsi:.0f} — 強勁動能 ✓✓")
    elif rsi >= 50: s += 7;  n.append(f"RSI {rsi:.0f} — 多頭區間 ✓")
    elif rsi >= 40: s += 3;  n.append(f"RSI {rsi:.0f} — 偏弱")
    else:           n.append(f"RSI {rsi:.0f} — 超賣 ✗")
    if macd > ms: s += 3; n.append("MACD 在信號線上方 ✓")
    if mh > 0:    s += 2; n.append("MACD 柱狀圖正值 ✓")
    if macd > 0:  s += 1; n.append("MACD 在零軸上方 ✓")
    if ma10 and p > ma10:
        d = (p-ma10)/ma10*100
        s += 4 if d<5 else 2
        n.append(f"高於 MA10 ({d:.1f}%) {'✓' if d<5 else '— 注意延伸'}")
    return min(s, 100), n

def score_rs(rs_rank, rs63, rs21):
    n = [f"3月超額報酬: {rs63:+.1f}%", f"1月超額報酬: {rs21:+.1f}%"]
    if   rs_rank >= 90: sc=100; n.insert(0,f"RS {rs_rank:.0f} — 頂尖領導股 ✓✓✓")
    elif rs_rank >= 80: sc=85;  n.insert(0,f"RS {rs_rank:.0f} — 強勢領導股 ✓✓")
    elif rs_rank >= 70: sc=70;  n.insert(0,f"RS {rs_rank:.0f} — 相對強勢 ✓")
    elif rs_rank >= 60: sc=50;  n.insert(0,f"RS {rs_rank:.0f} — 略優於大盤")
    elif rs_rank >= 50: sc=35;  n.insert(0,f"RS {rs_rank:.0f} — 與大盤同步")
    else:               sc=10;  n.insert(0,f"RS {rs_rank:.0f} — 落後大盤 ✗")
    return sc, n

def score_volume(vol, avg_vol, chg_pct):
    vr = vol/avg_vol if avg_vol else 1.0
    s, n = 0, []
    if   vr >= 2.0: s += 6; n.append(f"量達均量 {vr:.1f}x — 強勢爆量 ✓✓")
    elif vr >= 1.5: s += 4; n.append(f"量達均量 {vr:.1f}x — 量能確認 ✓")
    elif vr >= 1.0: s += 2; n.append(f"量達均量 {vr:.1f}x — 正常")
    else:           n.append(f"量低於均量 ({vr:.1f}x) — 動能不足")
    if chg_pct>0 and vr>=1:   s += 5; n.append("上漲帶量 — 積累訊號 ✓")
    elif chg_pct<0 and vr<1:  s += 3; n.append("下跌縮量 — 健康回調 ✓")
    elif chg_pct<0 and vr>=1.5: s -= 2; n.append("下跌帶量 — 派發訊號 ✗")
    am = avg_vol/1e6
    if   am >= 5:   s += 4; n.append(f"流動性充裕 ({am:.1f}M/日) ✓")
    elif am >= 1:   s += 2; n.append(f"流動性正常 ({am:.1f}M/日)")
    elif am >= 0.5: s += 1; n.append(f"流動性偏低 ({am:.2f}M/日) ⚠")
    else:           s -= 2; n.append(f"流動性不足 ({am:.2f}M/日) ✗")
    return max(0, min(s, 100)), n

def score_breakout(p, h20, h50, wk_h):
    s, n = 0, []
    if h20:
        d = (h20-p)/h20*100
        if   d <= 1: s += 5; n.append("正在突破 20 日高點 ✓✓")
        elif d <= 3: s += 4; n.append(f"接近 20 日高點 ({d:.1f}%) ✓")
        elif d <= 8: s += 2; n.append(f"距 20 日高點 {d:.1f}%")
        else:        n.append(f"距 20 日高點較遠 ({d:.1f}%) ✗")
    if h50 and (h50-p)/h50*100 <= 3: s += 3; n.append("接近 50 日高點 ✓")
    if wk_h and (wk_h-p)/wk_h*100 <= 5: s += 2; n.append("接近 52W 高點 ✓")
    return min(s, 100), n

def score_risk(p, ma20, earn_days):
    s, n = 100, []
    if earn_days is not None:
        if   earn_days <= 7:  s -= 40; n.append(f"⚠ 財報 {earn_days} 天後 — 高風險")
        elif earn_days <= 14: s -= 20; n.append(f"⚠ 財報 {earn_days} 天後 — 風險偏高")
        else: n.append(f"財報 {earn_days} 天後 — 安全")
    else:
        n.append("財報日期未知")
    if ma20:
        ext = (p-ma20)/ma20*100
        if   ext > 15: s -= 40; n.append(f"高於 20MA {ext:.1f}% — 嚴重延伸 ✗")
        elif ext > 8:  s -= 20; n.append(f"高於 20MA {ext:.1f}% — 追高風險 ⚠")
        else: n.append(f"高於 20MA {ext:.1f}% — 正常")
    return max(0, s), n

def classify_setup(p, ma20, ma50, ma200, h20, vr, earn_days):
    above50  = ma50  and p > ma50
    above200 = ma200 and p > ma200
    ext20    = (p-ma20)/ma20*100 if ma20 else 0
    near20h  = h20 and (h20-p)/h20*100 <= 3
    close20  = ma20 and abs(p-ma20)/ma20*100 < 3
    disq = []
    if ma50 and p < ma50*0.95:
        disq.append("低於 50MA — 趨勢未確立")
    if not above50 and not above200: return "Broken",     disq
    if ext20 > 8:                    return "Extended",   disq
    if near20h and vr >= 1.5:        return "Breakout",   disq
    if close20 and above50:          return "Pullback",   disq
    if above50 and above200:         return "Base",       disq
    return "Early Trend", disq

def run_strategy(snap: dict) -> dict:
    p    = snap["price"]
    ma20 = snap.get("ma20",0) or 0; ma50 = snap.get("ma50",0) or 0
    ma200= snap.get("ma200",0) or 0; ma10 = snap.get("ma10",0) or 0
    rsi  = snap.get("rsi",50) or 50
    macd = snap.get("macd",0) or 0; ms = snap.get("macd_signal",0) or 0; mh = snap.get("macd_hist",0) or 0
    atr  = snap.get("atr",0) or 0
    vol  = snap.get("volume",0) or 1; avgv = snap.get("avg_volume",1) or 1
    vr   = vol/avgv
    rs   = snap.get("rs_rank",50) or 50; rs63 = snap.get("rs_63d",0) or 0; rs21 = snap.get("rs_21d",0) or 0
    chg  = snap.get("change_pct",0) or 0
    h20  = snap.get("high_20d",0) or 0; h50 = snap.get("high_50d",0) or 0
    wkh  = snap.get("week_high_52",0) or 0; earn = snap.get("earnings_days",None)

    ts,tn = score_trend(p,ma20,ma50,ma200)
    ms2,mn= score_momentum(rsi,macd,ms,mh,p,ma10)
    rss,rn= score_rs(rs,rs63,rs21)
    vs,vn = score_volume(vol,avgv,chg)
    bs,bn = score_breakout(p,h20,h50,wkh)
    rks,kn= score_risk(p,ma20,earn)

    raw   = ts*.25 + ms2*.20 + rss*.20 + vs*.15 + bs*.10 + rks*.10
    final = max(0, min(100, round(raw)))
    rating= "A+" if final>=90 else "A" if final>=80 else "B+" if final>=70 else "B" if final>=60 else "C" if final>=50 else "D"
    setup, disq = classify_setup(p,ma20,ma50,ma200,h20,vr,earn)

    atr_v = atr if atr>0 else p*0.05
    if   setup=="Breakout": eL,eH = p*.995,p*1.01
    elif setup=="Pullback": base=max(ma20,ma50); eL,eH = base*.99,base*1.02
    elif setup=="Base":     eL,eH = (h20 or p)*.99,(h20 or p)*1.01
    else:                   eL,eH = None,None

    entry = eL or p
    stop  = round(entry - 2.5*atr_v, 2)
    stop_p= round((entry-stop)/entry*100, 2)
    target= round(entry + 6*atr_v, 2)
    rr    = round((target-entry)/(entry-stop),2) if stop<entry else 0

    flags = []
    if earn and earn<=14:                 flags.append(f"⚠ 財報 {earn} 天後")
    if ma20 and (p-ma20)/ma20*100>8:      flags.append("距 20MA 延伸較多")
    if rsi>78:                            flags.append(f"RSI {rsi:.0f} 過熱")
    if rr>0 and rr<2:                     flags.append(f"風報比 {rr:.1f}x 偏低")

    qualifies = final>=55 and not disq and rr>=1.5 and (ma50==0 or p>ma50)

    parts = []
    if   setup=="Breakout": parts.append(f"以帶量突破關鍵阻力（量達均量 {vr:.1f}x），站上所有主要均線。")
    elif setup=="Pullback": parts.append(f"拉回至 50MA (${ma50:.1f}) 附近縮量整理，主要趨勢完整。")
    elif setup=="Base":     parts.append("在均線上方形成底部整理，成交量收縮良好。")
    elif setup=="Extended": parts.append("目前過度延伸，建議等待回測後進場。")
    else:                   parts.append("型態未達條件，建議觀察等待。")
    parts.append(f"RS {rs:.0f} vs SPY | RSI {rsi:.0f} | 量比 {vr:.2f}x")
    if eL: parts.append(f"進場 ${entry:.2f} → 停損 ${stop} → 目標 ${target} (風報比 {rr:.1f}x)")
    if earn and earn<=14: parts.append(f"⚠ 財報 {earn} 天後，注意風險。")

    return {
        "setup_type":   setup, "score": final, "rating": rating, "qualifies": qualifies,
        "entry_low":    round(eL,2) if eL else None,
        "entry_high":   round(eH,2) if eH else None,
        "stop_loss":    stop, "stop_pct": stop_p, "target": target, "rr_ratio": rr,
        "flags": flags, "disqualifiers": disq, "why": " | ".join(parts),
        "dimensions": [
            {"name":"趨勢結構","score":ts, "weight":25,"notes":tn},
            {"name":"動能",   "score":ms2,"weight":20,"notes":mn},
            {"name":"相對強度","score":rss,"weight":20,"notes":rn},
            {"name":"成交量", "score":vs, "weight":15,"notes":vn},
            {"name":"突破形態","score":bs, "weight":10,"notes":bn},
            {"name":"風險調整","score":rks,"weight":10,"notes":kn},
        ],
    }


# ══════════════════════════════════════════════════════════════
# YFINANCE DATA FETCH
# ══════════════════════════════════════════════════════════════

_spy_closes: Optional[pd.Series] = None

def get_spy():
    global _spy_closes
    cached = cache_get("spy_closes", ttl=300)
    if cached is not None:
        _spy_closes = cached
        return cached
    try:
        df = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            s = df["Close"].squeeze()
            _spy_closes = s
            return cache_set("spy_closes", s)
    except Exception as e:
        log.error(f"SPY fetch: {e}")
    return _spy_closes


def fetch_snapshot(symbol: str, include_news: bool = False) -> Optional[dict]:
    cached = cache_get(f"snap:{symbol}", ttl=60)
    if cached and not include_news:
        return cached

    try:
        log.info(f"Downloading {symbol}...")
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        closes  = df["Close"].squeeze()
        highs   = df["High"].squeeze()
        lows    = df["Low"].squeeze()
        volumes = df["Volume"].squeeze()
        n       = len(closes)

        price      = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if n>1 else price
        change     = round(price - prev_close, 2)
        change_pct = round((change/prev_close)*100, 2) if prev_close else 0.0

        ma10  = round(float(closes.rolling(10).mean().iloc[-1]), 2)
        ma20  = round(float(closes.rolling(20).mean().iloc[-1]), 2)
        ma50  = round(float(closes.rolling(50).mean().iloc[-1]), 2)
        ma200 = round(float(closes.rolling(min(200,n)).mean().iloc[-1]), 2)

        rsi              = calc_rsi(closes)
        macd_v, msig, mh = calc_macd(closes)
        atr              = calc_atr(df)

        volume   = int(volumes.iloc[-1])
        avg_vol  = int(volumes.rolling(20).mean().iloc[-1])
        vol_ratio= round(volume/avg_vol, 2) if avg_vol else 1.0

        wk_high  = round(float(highs.rolling(min(252,n)).max().iloc[-1]), 2)
        wk_low   = round(float(lows.rolling(min(252,n)).min().iloc[-1]),  2)
        high_20d = round(float(highs.rolling(20).max().iloc[-1]),          2)
        high_50d = round(float(highs.rolling(min(50,n)).max().iloc[-1]),   2)

        spy = get_spy()
        rs_rank, rs63, rs21 = calc_rs(closes, spy) if spy is not None else (50.0,0.0,0.0)

        earn_days = None
        name, sector = symbol, "—"
        try:
            info = yf.Ticker(symbol).info
            et = info.get("earningsTimestamp") or info.get("earningsDate")
            if et:
                if isinstance(et, (list,tuple)) and et: et = et[0]
                if isinstance(et, (int,float)):
                    ed   = datetime.datetime.fromtimestamp(et).date()
                    days = (ed - datetime.date.today()).days
                    if 0 <= days < 180: earn_days = days
            name   = info.get("longName", symbol)
            sector = info.get("sector", "—")
        except Exception:
            pass

        ohlc = []
        for ts_idx, row in df.tail(130).iterrows():
            ohlc.append({
                "date":   str(ts_idx.date()),
                "open":   round(float(row.get("Open",  price)), 2),
                "high":   round(float(row.get("High",  price)), 2),
                "low":    round(float(row.get("Low",   price)), 2),
                "close":  round(float(row.get("Close", price)), 2),
                "volume": int(row.get("Volume", 0)),
            })

        snap = {
            "symbol": symbol, "name": name, "sector": sector,
            "price": round(price,2), "change": change, "change_pct": change_pct,
            "prev_close": round(prev_close,2),
            "volume": volume, "avg_volume": avg_vol, "volume_ratio": vol_ratio,
            "ma10": ma10, "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "rsi": rsi, "macd": macd_v, "macd_signal": msig, "macd_hist": mh,
            "atr": round(atr,2), "atr_pct": round(atr/price*100,2) if price else 0,
            "week_high_52": wk_high, "week_low_52": wk_low,
            "high_20d": high_20d, "high_50d": high_50d,
            "rs_rank": rs_rank, "rs_63d": rs63, "rs_21d": rs21,
            "earnings_days": earn_days, "ohlc": ohlc,
            "is_favorite": fav_module.is_favorite(symbol),
            "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        # Attach news summary if requested
        if include_news:
            snap["news_summary"] = news_module.get_news_summary(symbol, name)

        cache_set(f"snap:{symbol}", snap)
        log.info(f"{symbol}: ${price} | RSI={rsi} | RS={rs_rank} | ⭐={snap['is_favorite']}")
        return snap

    except Exception as e:
        log.error(f"fetch_snapshot({symbol}): {e}")
        return None


# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class FavoriteAdd(BaseModel):
    symbol: str
    folder: Optional[str] = None

class FolderCreate(BaseModel):
    name: str
    symbols: Optional[list[str]] = None


# ══════════════════════════════════════════════════════════════
# API ROUTES — HEALTH
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok", "version": "2.0.0",
        "yfinance": True, "favorites": True, "news": True,
        "time": datetime.datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# API ROUTES — FAVORITES ⭐
# ══════════════════════════════════════════════════════════════

@app.get("/api/favorites")
def get_favorites(folder: Optional[str] = Query(default=None)):
    """Get all favorites, or just one folder."""
    store = fav_module.get_favorites()
    symbols = fav_module.get_symbols(folder)
    return {
        "symbols":    symbols,
        "count":      len(symbols),
        "folders":    store.get("folders", {}),
        "updated_at": store.get("updated_at", ""),
        "folder":     folder,
    }

@app.post("/api/favorites")
def add_favorite(body: FavoriteAdd):
    """Add a symbol to favorites."""
    sym = body.symbol.upper().strip()
    if not sym:
        raise HTTPException(400, "Symbol required")
    result = fav_module.add_symbol(sym, body.folder)
    log.info(f"[favorites] ⭐ Added {sym}")
    return result

@app.delete("/api/favorites/{symbol}")
def remove_favorite(symbol: str, folder: Optional[str] = Query(default=None)):
    """Remove a symbol from favorites (or just from a folder)."""
    result = fav_module.remove_symbol(symbol, folder)
    log.info(f"[favorites] ❌ Removed {symbol}")
    return result

@app.get("/api/favorites/folders")
def get_folders():
    """List all watchlist folders."""
    return {"folders": fav_module.get_folders()}

@app.post("/api/favorites/folders")
def create_folder(body: FolderCreate):
    """Create a named watchlist folder."""
    return fav_module.create_folder(body.name, body.symbols)


# ══════════════════════════════════════════════════════════════
# API ROUTES — NEWS 📰
# ══════════════════════════════════════════════════════════════

@app.get("/api/news/{symbol}")
def get_news(symbol: str, limit: int = Query(default=10, le=20)):
    """
    Get news for a specific symbol.
    Sources: yfinance + Yahoo RSS + Google News RSS
    Cache TTL: 5 minutes
    """
    sym  = symbol.upper()
    snap = cache_get(f"snap:{sym}", ttl=3600)
    name = snap.get("name", sym) if snap else sym
    items = news_module.get_news(sym, name, max_items=limit)
    return {
        "symbol":    sym,
        "count":     len(items),
        "cached_ttl": 300,
        "items":     items,
    }

@app.get("/api/news")
def get_market_news(limit: int = Query(default=10, le=20)):
    """General market news (CNBC RSS)."""
    items = news_module.get_market_news(max_items=limit)
    return {"count": len(items), "items": items}


# ══════════════════════════════════════════════════════════════
# API ROUTES — MARKET DATA
# ══════════════════════════════════════════════════════════════

@app.get("/api/snapshot/{symbol}")
def api_snapshot(
    symbol: str,
    news: bool = Query(default=True, description="Include news summary")
):
    """
    Full snapshot + strategy analysis for one symbol.
    Now includes: is_favorite flag + news_summary
    """
    snap = fetch_snapshot(symbol.upper(), include_news=news)
    if not snap:
        return JSONResponse({"error": f"No data for {symbol}"}, status_code=404)
    result = run_strategy(snap)
    return {**snap, **result}


@app.get("/api/watchlist")
def api_watchlist(
    symbols:       str  = Query(default=""),
    favorites_only:bool = Query(default=False, description="Show only favorited symbols"),
    news:          bool = Query(default=False, description="Include news summaries (slower)"),
    folder:        Optional[str] = Query(default=None),
):
    """
    Watchlist endpoint — now integrated with favorites.

    Parameters:
      symbols        — comma-separated list (overrides favorites if provided)
      favorites_only — true: use favorites list
      folder         — use a specific folder
      news           — attach news summary to each stock
    """
    # Determine symbol list
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
    elif favorites_only or folder:
        sym_list = fav_module.get_symbols(folder)
    else:
        sym_list = fav_module.get_symbols()   # default = all favorites

    if not sym_list:
        return {"count": 0, "data": [], "symbols": [], "favorites_only": favorites_only}

    get_spy()  # preload SPY
    results = []
    for sym in sym_list:
        snap = fetch_snapshot(sym, include_news=news)
        if snap:
            r = run_strategy(snap)
            row = {**snap, **r}
            # Attach minimal news if requested
            if news and "news_summary" not in row:
                row["news_summary"] = news_module.get_news_summary(sym, snap.get("name",""))
            results.append(row)

    results.sort(key=lambda x: -x.get("score", 0))
    fav_syms = fav_module.get_symbols()

    return {
        "count":          len(results),
        "symbols":        sym_list,
        "favorites":      fav_syms,
        "folders":        fav_module.get_folders(),
        "favorites_only": favorites_only,
        "data":           results,
    }


# ══════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def serve_index():
    for fname in ["index.html", "swingtrader_live.html", "swingtrader_pro.html"]:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                return f.read()
    return """<html><body style='background:#060810;color:#00e8a2;font-family:monospace;padding:40px;line-height:2'>
    <h2>✅ SwingTrader Pro v2 is Running</h2>
    <p>Put <b>index.html</b> in the same folder as server.py, then open <a href='http://localhost:8000' style='color:#00e8a2'>http://localhost:8000</a></p>
    <hr style='border-color:#1a2438;margin:20px 0'>
    <p>📡 API Endpoints:</p>
    <ul style='color:#7a8fa8'>
      <li><a href='/api/health' style='color:#00e8a2'>GET /api/health</a></li>
      <li><a href='/api/favorites' style='color:#00e8a2'>GET /api/favorites</a></li>
      <li><a href='/api/watchlist' style='color:#00e8a2'>GET /api/watchlist</a></li>
      <li><a href='/api/watchlist?favorites_only=true' style='color:#00e8a2'>GET /api/watchlist?favorites_only=true</a></li>
      <li><a href='/api/snapshot/NVDA' style='color:#00e8a2'>GET /api/snapshot/NVDA</a></li>
      <li><a href='/api/news/NVDA' style='color:#00e8a2'>GET /api/news/NVDA</a></li>
      <li><a href='/api/news' style='color:#00e8a2'>GET /api/news (market news)</a></li>
      <li>POST /api/favorites  {"symbol": "AAPL"}</li>
      <li>DELETE /api/favorites/AAPL</li>
    </ul>
    </body></html>"""


if __name__ == "__main__":
    print("\n" + "="*58)
    print("  SwingTrader Pro v2 — Decision Support Dashboard")
    print("="*58)
    print("  ✅ Favorites / Watchlist folders")
    print("  ✅ News integration (yfinance + RSS)")
    print("  ✅ No pandas-ta — pure pandas/numpy")
    print("  📦 pip install fastapi uvicorn yfinance pandas numpy")
    print("  🌐 http://localhost:8000")
    print("  📡 /api/favorites  /api/news/{sym}  /api/watchlist")
    print("="*58 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

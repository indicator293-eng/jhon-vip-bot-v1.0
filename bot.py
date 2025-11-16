#!/usr/bin/env python3
# bot.py — Merged single-file BD Trader Bot (UPDATED v2 - Web Service Compatible)
# Keep this file UTF-8 encoded.
from __future__ import annotations
import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread

import pandas as pd
import websockets
from dotenv import load_dotenv

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler,
)

# Flask for web service (Render requirement)
try:
    from flask import Flask, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available - will run in polling mode only")

    # +++ এই কোডটুকু যোগ করুন +++
if FLASK_AVAILABLE:
    app = Flask(__name__)
else:
    app = None

if app:
    @app.route('/')
    def health_check():
        """Render health check endpoint."""
        return jsonify({"status": "healthy", "message": "JHON VIP BOT is running!"}), 200
    #END=======

# -------------------------
# Config — edit or set .env
# -------------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "7972983089:AAEN2kvyMR6kLAqqU-c_h1xAMSugbd5KqTA").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
OWNER_CONTACT = os.getenv("OWNER_CONTACT", "@emonjohn744").strip()
raw_admins = os.getenv("ADMIN_IDS", "7529660852").strip()
ADMIN_IDS = set()
for part in raw_admins.split(","):
    part = part.strip()
    if part.isdigit():
        ADMIN_IDS.add(int(part))

# Authentication credentials
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "@emonjohn744").strip()
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "jhonsignalvip").strip()
AUTH_LICENSE = os.getenv("AUTH_LICENSE", "jhonvipbot").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set. Please set BOT_TOKEN in environment or edit this file.")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bdtraderbot")

# -------------------------
# Storage / DB (sqlite)
# -------------------------
# Render.com compatible path
import sys
if sys.platform == "linux":
    # For Render/Linux servers
    DB_PATH = Path("./bdtrader.db")
else:
    # For Android/local development
    DB_PATH = Path("/storage/emulated/0/bdtrader/bdtrader.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def upgrade_db():
    con = _connect()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE users ADD COLUMN username TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE users ADD COLUMN bonus_signals INTEGER DEFAULT 0")
        con.commit()
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE users ADD COLUMN authenticated INTEGER DEFAULT 0")
        con.commit()
    except Exception:
        pass
    con.close()

def init_db():
    DB_PATH.touch(exist_ok=True)
    con = _connect()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id      INTEGER PRIMARY KEY,
        first_seen   TEXT NOT NULL,
        username     TEXT,
        country      TEXT,
        tz           TEXT,
        is_vip       INTEGER DEFAULT 0,
        vip_expiry   TEXT,
        bonus_signals INTEGER DEFAULT 0,
        referred_by  INTEGER,
        referrals_count INTEGER DEFAULT 0,
        authenticated INTEGER DEFAULT 0
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usage_logs(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        used_at     TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        pair        TEXT,
        timeframe   TEXT,
        sent_at     TEXT NOT NULL,
        confidence  INTEGER,
        risky       INTEGER,
        message     TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS outcomes(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id   INTEGER NOT NULL,
        result      TEXT,
        note        TEXT
    );
    """)
    con.commit()
    con.close()

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def get_or_create_user(user_id: int, username: Optional[str] = None):
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (user_id, first_seen, username) VALUES (?, ?, ?)",
                    (user_id, date.today().isoformat(), username))
    else:
        if username:
            cur.execute("UPDATE users SET username=? WHERE user_id=?", (username, user_id))
    con.commit()
    con.close()

def set_country(user_id: int, country: str, tz: Optional[str] = None):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE users SET country=?, tz=? WHERE user_id=?", (country, tz, user_id))
    con.commit(); con.close()

def is_authenticated(user_id: int) -> bool:
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT authenticated FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    return bool(row["authenticated"])

def set_authenticated(user_id: int, status: bool = True):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE users SET authenticated=? WHERE user_id=?", (1 if status else 0, user_id))
    con.commit(); con.close()

def is_vip(user_id: int) -> bool:
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT is_vip, vip_expiry FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    isvip = row["is_vip"]
    expiry = row["vip_expiry"]
    if not isvip:
        return False
    if expiry:
        try:
            exp = datetime.fromisoformat(expiry)
            if datetime.now(timezone.utc) > exp:
                con = _connect(); cur = con.cursor()
                cur.execute("UPDATE users SET is_vip=0, vip_expiry=NULL WHERE user_id=?", (user_id,))
                con.commit(); con.close()
                return False
        except Exception:
            return True
    return True

def set_vip(user_id: int, days: int):
    con = _connect(); cur = con.cursor()
    exp = datetime.now(timezone.utc) + timedelta(days=max(1, days))
    cur.execute("""
        INSERT INTO users(user_id, first_seen, is_vip, vip_expiry)
        VALUES (?,?,1,?)
        ON CONFLICT(user_id) DO UPDATE SET is_vip=1, vip_expiry=excluded.vip_expiry
    """, (user_id, now_utc_iso(), exp.isoformat()))
    con.commit(); con.close()

def remove_vip(user_id: int):
    con = _connect(); cur = con.cursor()
    cur.execute("UPDATE users SET is_vip=0, vip_expiry=NULL WHERE user_id=?", (user_id,))
    con.commit(); con.close()

def record_usage(user_id: int):
    con = _connect(); cur = con.cursor()
    cur.execute("INSERT INTO usage_logs(user_id, used_at) VALUES (?,?)", (user_id, now_utc_iso()))
    con.commit(); con.close()

def count_usage_today(user_id: int) -> int:
    today = datetime.utcnow().date().isoformat()
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM usage_logs WHERE user_id=? AND substr(used_at,1,10)=?", (user_id, today))
    n = cur.fetchone()[0]; con.close(); return int(n)

def get_first_seen_date(user_id: int):
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT first_seen FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone(); con.close()
    if not row or not row[0]:
        return None
    try:
        dt = datetime.fromisoformat(row[0])
        return dt.date()
    except Exception:
        return None

def get_referrals_count(user_id: int) -> int:
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT referrals_count FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone(); con.close()
    return int(row["referrals_count"] if row and row["referrals_count"] is not None else 0)

def log_signal(user_id: int, pair: str, tf: str, confidence: int, risky: bool, message: str):
    con = _connect(); cur = con.cursor()
    cur.execute("INSERT INTO signals(user_id, pair, timeframe, sent_at, confidence, risky, message) VALUES (?,?,?,?,?,?,?)",
                (user_id, pair, tf, now_utc_iso(), int(confidence), 1 if risky else 0, message))
    con.commit(); con.close()

def get_bonus_signals(user_id: int) -> int:
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT bonus_signals FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone(); con.close()
    return int(row["bonus_signals"] if row and row["bonus_signals"] is not None else 0)

def use_bonus_signal(user_id: int) -> bool:
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT bonus_signals FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        con.close(); return False
    current = int(row["bonus_signals"] or 0)
    if current <= 0:
        con.close(); return False
    cur.execute("UPDATE users SET bonus_signals = bonus_signals - 1 WHERE user_id=?", (user_id,))
    con.commit(); con.close()
    return True

def credit_referral(referrer_id: int, new_user_id: int, bonus_per_ref: int = 2) -> bool:
    if referrer_id == new_user_id:
        return False
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT referred_by FROM users WHERE user_id=?", (new_user_id,))
    row = cur.fetchone()
    if not row:
        con.close(); return False
    if row["referred_by"]:
        con.close(); return False
    cur.execute("UPDATE users SET referred_by=? WHERE user_id=?", (referrer_id, new_user_id))
    cur.execute("""
        UPDATE users
        SET referrals_count = COALESCE(referrals_count,0) + 1,
            bonus_signals = COALESCE(bonus_signals,0) + ?
        WHERE user_id=?
    """, (bonus_per_ref, referrer_id))
    con.commit(); con.close()
    return True

def list_vip():
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT user_id, vip_expiry FROM users WHERE is_vip=1")
    rows = cur.fetchall(); con.close()
    return rows

def vip_stats():
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE is_vip=1"); vip_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM users"); total = cur.fetchone()[0]; con.close()
    return {"vip": vip_count, "total": total}

def get_all_users():
    con = _connect(); cur = con.cursor()
    cur.execute("SELECT user_id FROM users"); rows = cur.fetchall(); con.close()
    return [row[0] for row in rows]


# -------------------------
# Pairs / validators
# -------------------------
USER_TO_DERIV = {
    "EURUSD": "frxEURUSD", "USDJPY": "frxUSDJPY", "GBPUSD": "frxGBPUSD", "AUDUSD": "frxAUDUSD",
    "USDCAD": "frxUSDCAD", "USDCHF": "frxUSDCHF", "EURJPY": "frxEURJPY", "EURGBP": "frxEURGBP",
    "GBPJPY": "frxGBPJPY", "AUDJPY": "frxAUDJPY", "NZDUSD": "frxNZDUSD", "EURCHF": "frxEURCHF",
    "USDSGD": "frxUSDSGD", "USDHKD": "frxUSDHKD", "XAUUSD": "frxXAUUSD",
    "BTCUSD": "cryBTCUSD", "ETHUSD": "cryETHUSD", "LTCUSD": "cryLTCUSD", "BCHUSD": "cryBCHUSD",
}
NORMALIZE = {"BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "LTCUSDT": "LTCUSD", "BCHUSDT": "BCHUSD"}
TF_TO_GRAN = {"M1": 60, "M5": 300, "M10": 600, "M15": 900}

def normalize_symbol(sym):
    if not sym: return None
    s = str(sym).strip().upper(); return NORMALIZE.get(s, s)

def to_deriv_symbol(sym):
    s = normalize_symbol(sym); return USER_TO_DERIV.get(s)

def is_valid_pair(sym: str) -> bool:
    s = normalize_symbol(sym); return bool(s and s in USER_TO_DERIV)

def is_otc_pair(sym: str) -> bool:
    return "OTC" in str(sym or "").upper()

def is_supported_tf(tf) -> bool:
    if not tf: return False
    return str(tf).strip().upper() in TF_TO_GRAN

def granularity(tf) -> int:
    tf_str = str(tf).strip().upper()
    return TF_TO_GRAN.get(tf_str)

def supported_user_symbols() -> list[str]:
    return list(USER_TO_DERIV.keys())

def supported_timeframes():
    return list(TF_TO_GRAN.keys())


# -------------------------
# Deriv Websocket client
# -------------------------
class DerivClient:
    def __init__(self):
        self.url = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

    async def _send(self, ws, data: dict):
        await ws.send(json.dumps(data))

    async def get_candles(self, symbol: str, granularity: int, count: int = 250) -> List[Dict[str, Any]]:
        async with websockets.connect(self.url) as ws:
            req = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "granularity": granularity,
                "style": "candles"
            }
            await self._send(ws, req)
            resp = await ws.recv()
            data = json.loads(resp)
            if "candles" not in data:
                raise ValueError(f"Failed to get candles: {data}")
            candles = []
            for c in data["candles"]:
                candles.append({
                    "epoch": c["epoch"],
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                })
            candles.sort(key=lambda x: x["epoch"])
            return candles

_client_instance = DerivClient()

async def get_candles(symbol: str, granularity: int, count: int = 250):
    return await _client_instance.get_candles(symbol, granularity, count)


# -------------------------
# Indicators & signals (ENHANCED)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_indicators(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    if df.empty or len(df) < 100:
        raise ValueError("Not enough candle data (need >= 100).")
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi_wilder(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"], 12, 26, 9)
    df["macd"] = macd_line; df["macd_signal"] = signal_line; df["macd_hist"] = hist
    df["atr"] = atr(df, 14)
    return df

def vote_and_confidence(row: pd.Series) -> Tuple[str | None, int, Dict[str, str]]:
    price = float(row["close"])
    rsi = float(row["rsi14"])
    ema9 = float(row["ema9"])
    ema21 = float(row["ema21"])
    ema50 = float(row["ema50"])
    macd_v = float(row["macd"])
    macd_sig = float(row["macd_signal"])
    hist = float(row["macd_hist"])
    atr_val = float(row["atr"])
    
    bullish = 0
    bearish = 0
    rationale: Dict[str, str] = {}

    # Check ATR for volatility filter
    atr_pct = atr_val / max(price, 1e-9)
    if atr_pct < 0.0005:
        rationale["volatility"] = f"⚠️ ATR too low ({atr_val:.5f}) - Low liquidity detected"
        return None, 50, rationale

    # RSI
    if rsi < 30:
        bullish += 1; rationale["rsi"] = f"RSI at {rsi:.0f} (Oversold - Strong Buy Signal)"
    elif rsi > 70:
        bearish += 1; rationale["rsi"] = f"RSI at {rsi:.0f} (Overbought - Strong Sell Signal)"
    else:
        rationale["rsi"] = f"RSI at {rsi:.0f}"

    # EMA Trend Filter (9, 21, 50 alignment)
    ema_trend_bullish = (ema9 > ema21) and (ema21 > ema50)
    ema_trend_bearish = (ema9 < ema21) and (ema21 < ema50)
    
    if ema_trend_bullish:
        bullish += 2; rationale["ema"] = "✅ EMA(9>21>50) - Strong Bullish Trend Confirmed"
    elif ema_trend_bearish:
        bearish += 2; rationale["ema"] = "✅ EMA(9<21<50) - Strong Bearish Trend Confirmed"
    elif ema9 > ema21:
        bullish += 1; rationale["ema"] = "EMA(9) above EMA(21) (Bullish momentum)"
    elif ema9 < ema21:
        bearish += 1; rationale["ema"] = "EMA(9) below EMA(21) (Bearish momentum)"
    else:
        rationale["ema"] = "EMA(9) equals EMA(21)"

    # MACD
    if hist > 0 and macd_v > macd_sig:
        bullish += 1; rationale["macd"] = "MACD histogram positive (Bullish momentum)"
    elif hist < 0 and macd_v < macd_sig:
        bearish += 1; rationale["macd"] = "MACD histogram negative (Bearish momentum)"
    else:
        rationale["macd"] = "MACD mixed"

    # Signal determination
    signal = None
    if bullish >= 3 and bullish > bearish:
        signal = "CALL"
    elif bearish >= 3 and bearish > bullish:
        signal = "PUT"

    # Enhanced confidence calculation
    base = 65 if ((bullish == 3) or (bearish == 3)) else 75 if ((bullish >= 4) or (bearish >= 4)) else 55
    bonus = 0
    
    if rsi < 25 or rsi > 75:
        bonus += 8
    
    ema_dist = abs(ema9 - ema21) / max(price, 1e-9)
    bonus += min(int(ema_dist * 1200), 15)
    
    macd_strength = abs(hist) / max(price, 1e-9)
    bonus += min(int(macd_strength * 4000), 12)
    
    if atr_pct > 0.002:
        bonus += 5
    
    confidence = max(50, min(base + bonus, 95))
    return signal, int(confidence), rationale

def build_signal_text(pair: str, tf: str, row: pd.Series, signal: str | None, confidence: int, rationale: Dict[str, str]) -> str:
    """Enhanced signal format with better visual presentation"""
    last_price = float(row.get("close", 0.0))
    stype = signal if signal else "⚠️ NO CLEAR SIGNAL"
    
    # Signal type emoji
    signal_emoji = "🟢" if signal == "CALL" else "🔴" if signal == "PUT" else "⚪"
    
    # Confidence bar
    conf_bars = int(confidence / 10)
    conf_display = "█" * conf_bars + "░" * (10 - conf_bars)
    
    # Get indicator values
    rsi_val = float(row.get("rsi14", 0.0))
    ema9 = float(row.get("ema9", 0.0))
    ema21 = float(row.get("ema21", 0.0))
    ema50 = float(row.get("ema50", 0.0))
    macd_hist = float(row.get("macd_hist", 0.0))
    atr_val = float(row.get("atr", 0.0))
    
    # Build message
    lines = [
        "═══════════════════════════",
        "⚡ JHON VIP BOT SIGNAL ⚡",
        "═══════════════════════════",
        "",
        f"{signal_emoji} SIGNAL: {stype}",
        f"📊 PAIR: {pair} | ⏰ TF: {tf}",
        f"💰 PRICE: {last_price:.5f}",
        "",
        f"🎯 CONFIDENCE: {confidence}%",
        f"{conf_display}",
        "",
        "──────────────────────────",
        "📈 TECHNICAL ANALYSIS",
        "──────────────────────────",
        ""
    ]
    
    # RSI Analysis
    if rsi_val < 30:
        lines.append(f"🔵 RSI: {rsi_val:.1f} → OVERSOLD (Bullish)")
    elif rsi_val > 70:
        lines.append(f"🔴 RSI: {rsi_val:.1f} → OVERBOUGHT (Bearish)")
    elif rsi_val < 50:
        lines.append(f"⚪ RSI: {rsi_val:.1f} → Neutral-Bearish")
    else:
        lines.append(f"⚪ RSI: {rsi_val:.1f} → Neutral-Bullish")
    
    # EMA Trend
    if ema9 > ema21 and ema21 > ema50:
        lines.append(f"🟢 EMA: 9>21>50 → STRONG UPTREND ⭐")
    elif ema9 < ema21 and ema21 < ema50:
        lines.append(f"🔴 EMA: 9<21<50 → STRONG DOWNTREND ⭐")
    elif ema9 > ema21:
        lines.append(f"🟡 EMA: 9>21 → Bullish Crossover")
    elif ema9 < ema21:
        lines.append(f"🟡 EMA: 9<21 → Bearish Crossover")
    else:
        lines.append(f"⚪ EMA: Neutral Position")
    
    # MACD
    if macd_hist > 0:
        lines.append(f"🟢 MACD: Positive → Upward Momentum")
    elif macd_hist < 0:
        lines.append(f"🔴 MACD: Negative → Downward Momentum")
    else:
        lines.append(f"⚪ MACD: Neutral")
    
    # ATR Volatility
    atr_pct = (atr_val / max(last_price, 1e-9)) * 100
    if atr_pct > 0.2:
        lines.append(f"🔥 VOLATILITY: HIGH ({atr_pct:.2f}%)")
    else:
        lines.append(f"📊 VOLATILITY: NORMAL ({atr_pct:.2f}%)")
    
    lines.extend([
        "",
        "──────────────────────────",
        "💡 TRADING TIPS",
        "──────────────────────────",
        "✅ Enter on NEXT candle open",
        "✅ Max Martingale: 1-2 steps",
        "✅ Use proper risk management",
        "✅ Avoid trading during news",
        "",
        "═══════════════════════════",
        "🚀 Trade Smart, Trade Safe! 💰",
        f"📢 Channel: @quotexx_non_mtg",
        f"💬 Support: @emonjohn744",
        "═══════════════════════════"
    ])
    
    return "\n".join(lines)

def analyze_and_signal(df: pd.DataFrame, pair: str, tf: str) -> Dict[str, Any]:
    row = df.iloc[-1]
    signal, confidence, rationale = vote_and_confidence(row)
    msg = build_signal_text(pair, tf, row, signal, confidence, rationale)
    risky = (confidence < 75) or (signal is None)
    return {"message": msg, "risky": risky, "confidence": confidence, "signal": signal or "NONE", "price": float(row["close"])}


# -------------------------
# Bot handlers & commands
# -------------------------
# States for both conversations
ASK_USERNAME, ASK_PASSWORD, ASK_LICENSE = range(3)
ASK_PAIR, ASK_TF, ASK_COUNTRY = range(3, 6)

TF_KEYBOARD = ReplyKeyboardMarkup([["M1", "M5", "M10", "M15"]], one_time_keyboard=True, resize_keyboard=True)

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

CHANNEL_USERNAME = "@quotexx_non_mtg"

# Decorator that checks both channel membership AND authentication
def require_auth_and_channel(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if not user:
            return await func(update, context, *args, **kwargs)
        
        # Admins bypass all checks
        if is_admin(user.id):
            return await func(update, context, *args, **kwargs)
        
        # Check channel membership FIRST
        try:
            member = await context.bot.get_chat_member(CHANNEL_USERNAME, user.id)
            if member.status not in ["member", "administrator", "creator"]:
                keyboard = [
                    [InlineKeyboardButton("📢 Join Channel", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
                    [InlineKeyboardButton("✅ Verify", callback_data="verify_join")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    f"⚠️ Channel Membership Required!\n\n"
                    f"You must join {CHANNEL_USERNAME} to use this bot.\n\n"
                    f"📌 Steps:\n"
                    f"1️⃣ Click 'Join Channel'\n"
                    f"2️⃣ Join the channel\n"
                    f"3️⃣ Click '✅ Verify'\n\n"
                    f"Then use /start to continue!",
                    reply_markup=reply_markup
                )
                return
        except Exception as e:
            logger.error(f"Channel check failed in decorator: {e}")
            await update.message.reply_text(
                f"❌ Cannot verify channel membership!\n\n"
                f"Please ensure:\n"
                f"• You joined {CHANNEL_USERNAME}\n"
                f"• Bot is admin in channel\n\n"
                f"Contact: {OWNER_CONTACT}"
            )
            return
        
        # Check authentication SECOND
        if not is_authenticated(user.id):
            await update.message.reply_text(
                "🔐 Authentication Required!\n\n"
                "Please use /start to authenticate first.\n\n"
                f"💬 Contact {OWNER_CONTACT} for credentials."
            )
            return
        
        return await func(update, context, *args, **kwargs)
    wrapper.__name__ = getattr(func, "__name__", "wrapper")
    return wrapper


# Welcome message
async def send_welcome_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    msg = (
        "═══════════════════════════\n"
        "🌟 WELCOME TO JHON VIP BOT 🌟\n"
        "═══════════════════════════\n\n"
        "⚡ Your Elite Trading Companion! ⚡\n\n"
        "🎯 FEATURES:\n"
        "✅ AI-Powered Signal Generation\n"
        "✅ Advanced Technical Analysis\n"
        "✅ Multi-Timeframe Support (M1-M15)\n"
        "✅ Real-Time Market Insights\n"
        "✅ Referral Rewards Program\n\n"
        "──────────────────────────\n"
        "📋 QUICK COMMANDS:\n"
        "──────────────────────────\n"
        "🚀 /getsignal - Generate premium signals\n"
        "📊 /pairs - View supported pairs\n"
        "💰 /referral - Earn bonus signals\n"
        "ℹ️ /help - Full command list\n"
        "👤 /myinfo - Account status\n\n"
        "──────────────────────────\n"
        "⚠️ IMPORTANT NOTES:\n"
        "──────────────────────────\n"
        "❌ No OTC pairs supported\n"
        "🧪 Test on Demo first\n"
        "📊 Use proper risk management\n"
        "⏰ Avoid high-impact news times\n\n"
        "──────────────────────────\n"
        "💎 GET MORE SIGNALS:\n"
        "──────────────────────────\n"
        "• Refer 10+ friends → 5 signals/day\n"
        "• Below 10 referrals → 3 signals/day\n"
        "• VIP members → Unlimited! 🔥\n\n"
        "═══════════════════════════\n"
        f"📢 Channel: {CHANNEL_USERNAME}\n"
        f"💬 Support: {OWNER_CONTACT}\n"
        "═══════════════════════════\n\n"
        "🚀 Ready to dominate the markets? 💰\n"
        "Start with /getsignal now! 🎯"
    )
    await context.bot.send_message(chat_id=chat_id, text=msg)


# START command with integrated authentication
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    # Create user record
    get_or_create_user(user_id, user.username)
    
    # Handle referral
    if context.args:
        arg = (context.args[0] or "").strip()
        if arg.startswith("ref_"):
            try:
                referrer_id = int(arg.split("ref_")[1])
                if referrer_id != user.id:
                    credited = credit_referral(referrer_id, user.id, bonus_per_ref=2)
                    if credited:
                        try:
                            await context.bot.send_message(
                                chat_id=referrer_id,
                                text=(
                                    f"🎉 New Referral Alert!\n\n"
                                    f"User {user.first_name} joined via your link!\n"
                                    f"✅ +2 bonus signals credited\n\n"
                                    f"Current bonus: {get_bonus_signals(referrer_id)} signals"
                                )
                            )
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Check if admin
    if is_admin(user_id):
        await send_welcome_message(context, user_id)
        return ConversationHandler.END
    
    # Check channel membership STRICTLY
    try:
        member = await context.bot.get_chat_member(CHANNEL_USERNAME, user_id)
        if member.status not in ["member", "administrator", "creator"]:
            keyboard = [
                [InlineKeyboardButton("📢 Join Channel", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
                [InlineKeyboardButton("✅ Verify", callback_data="verify_join")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                f"👋 Welcome {user.first_name}!\n\n"
                f"⚠️ You must join our channel first!\n\n"
                f"👉 Channel: {CHANNEL_USERNAME}\n\n"
                f"📌 Steps:\n"
                f"1️⃣ Click 'Join Channel' button\n"
                f"2️⃣ Join the channel\n"
                f"3️⃣ Come back and click '✅ Verify'\n\n"
                f"🔒 Channel membership is required to use this bot!",
                reply_markup=reply_markup
            )
            return ConversationHandler.END
    except Exception as e:
        logger.error(f"Channel check error: {e}")
        # If there's an error checking (bot not admin), inform user clearly
        await update.message.reply_text(
            f"❌ ERROR: Cannot verify channel membership!\n\n"
            f"⚠️ Please make sure:\n"
            f"1. You have joined {CHANNEL_USERNAME}\n"
            f"2. The channel is public\n"
            f"3. Bot is admin in the channel\n\n"
            f"💬 Contact admin if issue persists:\n{OWNER_CONTACT}"
        )
        return ConversationHandler.END
    
    # Check authentication
    if is_authenticated(user_id):
        await send_welcome_message(context, user_id)
        return ConversationHandler.END
    
    # Start authentication flow
    await update.message.reply_text(
        "🔐 AUTHENTICATION REQUIRED\n\n"
        "To use this bot, you need credentials.\n\n"
        f"Contact {OWNER_CONTACT} if you don't have them.\n\n"
        "Please enter your USERNAME:"
    )
    return ASK_USERNAME


async def verify_join(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    
    await query.answer()  # Acknowledge the button click
    
    user = query.from_user
    user_id = user.id
    
    try:
        member = await context.bot.get_chat_member(CHANNEL_USERNAME, user_id)
        
        if member.status in ["member", "administrator", "creator"]:
            # ✅ User has joined the channel
            await query.answer("✅ Verified! Welcome!", show_alert=True)
            
            # Delete the join prompt message
            try:
                await query.message.delete()
            except Exception:
                pass
            
            # Check if authenticated
            if is_authenticated(user_id):
                # Already authenticated - send welcome directly
                await send_welcome_message(context, user_id)
            else:
                # Not authenticated - ask to use /start
                await context.bot.send_message(
                    chat_id=user_id,
                    text=(
                        "✅ Channel verified successfully!\n\n"
                        "🔐 Now you need to authenticate.\n\n"
                        "👉 Please use /start command to begin authentication.\n\n"
                        f"💬 Contact {OWNER_CONTACT} if you need credentials."
                    )
                )
        else:
            # ❌ User still not a member
            await query.answer(
                f"❌ You haven't joined {CHANNEL_USERNAME} yet!\n\n"
                f"Please join the channel first, then click Verify again.",
                show_alert=True
            )
    except Exception as e:
        logger.error(f"Verify join error: {e}")
        await query.answer(
            f"❌ Verification failed!\n\n"
            f"Possible reasons:\n"
            f"• Bot is not admin in channel\n"
            f"• Channel username is wrong\n\n"
            f"Contact: {OWNER_CONTACT}",
            show_alert=True
        )


# Authentication conversation handlers
async def ask_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.message.text.strip()
    context.user_data["auth_username"] = username
    await update.message.reply_text("🔑 Now enter your PASSWORD:")
    return ASK_PASSWORD


async def ask_license(update: Update, context: ContextTypes.DEFAULT_TYPE):
    password = update.message.text.strip()
    context.user_data["auth_password"] = password
    await update.message.reply_text("🎫 Finally, enter your LICENSE KEY:")
    return ASK_LICENSE


async def verify_credentials(update: Update, context: ContextTypes.DEFAULT_TYPE):
    license_key = update.message.text.strip()
    username = context.user_data.get("auth_username", "")
    password = context.user_data.get("auth_password", "")
    user = update.effective_user
    
    # Verify credentials
    if username == AUTH_USERNAME and password == AUTH_PASSWORD and license_key == AUTH_LICENSE:
        set_authenticated(user.id, True)
        await update.message.reply_text(
            "✅ AUTHENTICATION SUCCESSFUL!\n\n"
            "🎉 Welcome to JHON VIP BOT!\n\n"
            "You now have full access to:\n"
            "✅ Premium signal generation\n"
            "✅ Advanced market analysis\n"
            "✅ Referral rewards program\n\n"
            "──────────────────────────\n"
            "Ready to start? Use /getsignal! 🚀"
        )
        
        # Send welcome message
        await send_welcome_message(context, user.id)
    else:
        await update.message.reply_text(
            "❌ AUTHENTICATION FAILED\n\n"
            "Credentials are incorrect!\n\n"
            "Please get valid credentials from:\n"
            f"👉 {OWNER_CONTACT}\n\n"
            "Try again with /start"
        )
    
    # Clear credentials
    context.user_data.pop("auth_username", None)
    context.user_data.pop("auth_password", None)
    
    return ConversationHandler.END


async def auth_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❌ Authentication cancelled.\n\n"
        "Use /start when ready to authenticate."
    )
    context.user_data.pop("auth_username", None)
    context.user_data.pop("auth_password", None)
    return ConversationHandler.END


# Signal generation flow
@require_auth_and_channel
async def getsignal_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if not is_vip(user.id):
        first_seen = get_first_seen_date(user.id)
        if first_seen and first_seen < date.today():
            used = count_usage_today(user.id)
            referrals = get_referrals_count(user.id)
            daily_limit = 5 if referrals >= 10 else 3
            
            if used >= daily_limit:
                if use_bonus_signal(user.id):
                    await update.message.reply_text(
                        f"🎁 Bonus signal used!\n\n"
                        f"Remaining bonus: {get_bonus_signals(user.id)} signals"
                    )
                else:
                    link = f"https://t.me/{context.bot.username}?start=ref_{user.id}"
                    await update.message.reply_text(
                        f"⚠️ DAILY LIMIT REACHED\n\n"
                        f"📊 Your status:\n"
                        f"• Used today: {used}/{daily_limit}\n"
                        f"• Referrals: {referrals}/10\n"
                        f"• Bonus signals: {get_bonus_signals(user.id)}\n\n"
                        f"──────────────────────────\n"
                        f"💡 GET MORE SIGNALS:\n"
                        f"──────────────────────────\n"
                        f"🎁 Refer friends: +2 signals/referral\n"
                        f"🎯 Reach 10 refs → 5 daily signals\n"
                        f"👑 VIP membership → Unlimited!\n\n"
                        f"🔗 Your referral link:\n{link}\n\n"
                        f"💎 Upgrade to VIP: {OWNER_CONTACT}"
                    )
                    return ConversationHandler.END
    
    await update.message.reply_text(
        "📊 SIGNAL GENERATOR\n\n"
        "Enter a trading pair:\n"
        "Examples: EURUSD, GBPUSD, XAUUSD, BTCUSD\n\n"
        "⚠️ No OTC pairs allowed",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASK_PAIR


@require_auth_and_channel
async def ask_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = (update.message.text or "").strip().upper()
    
    if is_otc_pair(pair):
        await update.message.reply_text(
            "❌ OTC pairs not supported!\n\n"
            "Please enter a valid pair:\n"
            "EURUSD, GBPUSD, XAUUSD, etc."
        )
        return ASK_PAIR
    
    if not is_valid_pair(pair):
        await update.message.reply_text(
            "❌ Invalid pair!\n\n"
            "Supported pairs:\n"
            "• Forex: EURUSD, GBPUSD, USDJPY, etc.\n"
            "• Gold: XAUUSD\n"
            "• Crypto: BTCUSD, ETHUSD, etc.\n\n"
            "Try again:"
        )
        return ASK_PAIR
    
    context.user_data["pair_user"] = pair
    context.user_data["pair_deriv"] = to_deriv_symbol(pair)
    
    await update.message.reply_text(
        f"✅ Pair selected: {pair}\n\n"
        "Now select TIMEFRAME:",
        reply_markup=TF_KEYBOARD
    )
    return ASK_TF


@require_auth_and_channel
async def ask_country(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tf = (update.message.text or "").strip().upper()
    
    if not is_supported_tf(tf):
        await update.message.reply_text(
            "❌ Invalid timeframe!\n\n"
            "Please select:",
            reply_markup=TF_KEYBOARD
        )
        return ASK_TF
    
    context.user_data["tf"] = tf
    await update.message.reply_text(
        f"✅ Timeframe: {tf}\n\n"
        "Enter your COUNTRY name:\n"
        "(For timezone adjustments)",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASK_COUNTRY


@require_auth_and_channel
async def do_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    country = (update.message.text or "").strip().title()
    user = update.effective_user
    
    if country:
        set_country(user.id, country, None)
    
    pair_user = context.user_data.get("pair_user")
    pair_deriv = context.user_data.get("pair_deriv")
    tf = context.user_data.get("tf")
    gran = granularity(tf)
    
    status = await update.message.reply_text(
        f"🔍 ANALYZING...\n\n"
        f"Pair: {pair_user}\n"
        f"Timeframe: {tf}\n\n"
        f"⏳ Please wait..."
    )
    
    try:
        candles = await get_candles(pair_deriv, gran, count=250)
        df = compute_indicators(candles)
        result = analyze_and_signal(df, pair_user, tf)
        
        # Send signal
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=result["message"]
        )
        
        # Record usage
        record_usage(user.id)
        log_signal(user.id, pair_user, tf, result["confidence"], result["risky"], result["message"])
        
        # Feedback prompt
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=(
                "✅ Signal generated successfully!\n\n"
                "💬 Share your feedback:\n"
                "/feedback <your message>\n\n"
                "🔄 Generate another: /getsignal"
            )
        )
        
    except Exception as e:
        logger.exception("Error in do_analyze")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ ERROR\n\nFailed to generate signal: {str(e)}\n\nTry again with /getsignal"
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass
    
    return ConversationHandler.END


# Other commands
@require_auth_and_channel
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "═══════════════════════════\n"
        "📖 JHON VIP BOT - HELP MENU\n"
        "═══════════════════════════\n\n"
        "🎯 MAIN COMMANDS:\n"
        "──────────────────────────\n"
        "/start - Start bot & authenticate\n"
        "/getsignal - Generate trading signal\n"
        "/help - Show this help menu\n"
        "/myinfo - View account info\n"
        "/pairs - List supported pairs\n"
        "/referral - Get referral link\n"
        "/feedback <text> - Send feedback\n"
        "/support - Contact support\n"
        "/about - About this bot\n\n"
        "📊 SIGNAL GENERATION:\n"
        "──────────────────────────\n"
        "1. Use /getsignal\n"
        "2. Enter pair (e.g., EURUSD)\n"
        "3. Select timeframe (M1-M15)\n"
        "4. Enter your country\n"
        "5. Get premium signal! 🎯\n\n"
        "💎 DAILY LIMITS:\n"
        "──────────────────────────\n"
        "• < 10 referrals: 3 signals/day\n"
        "• ≥ 10 referrals: 5 signals/day\n"
        "• VIP members: Unlimited! 👑\n\n"
        "⚠️ IMPORTANT:\n"
        "──────────────────────────\n"
        "❌ No OTC pairs\n"
        "🧪 Test on demo first\n"
        "📊 Use risk management\n"
        "⏰ Avoid news times\n\n"
        "═══════════════════════════\n"
        f"💬 Support: {OWNER_CONTACT}\n"
        f"📢 Channel: {CHANNEL_USERNAME}\n"
        "═══════════════════════════"
    )
    await update.message.reply_text(msg)


@require_auth_and_channel
async def about_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "═══════════════════════════\n"
        "ℹ️ ABOUT JHON VIP BOT BOT\n"
        "═══════════════════════════\n\n"
        "🤖 TECHNOLOGY:\n"
        "✅ AI-Powered Signal Engine\n"
        "✅ Advanced Technical Analysis\n"
        "✅ Multi-Indicator System\n"
        "✅ Real-Time Market Data\n\n"
        "📊 INDICATORS USED:\n"
        "• RSI (14) - Momentum\n"
        "• EMA (9, 21, 50) - Trend\n"
        "• MACD - Momentum & Direction\n"
        "• ATR - Volatility Filter\n\n"
        "🎯 MARKETS SUPPORTED:\n"
        "• Forex Majors & Minors\n"
        "• Gold (XAUUSD)\n"
        "• Cryptocurrencies\n\n"
        "⏰ TIMEFRAMES:\n"
        "M1, M5, M10, M15\n\n"
        "═══════════════════════════\n"
        "⚡ Developed for serious traders\n"
        "who demand accuracy & consistency!\n\n"
        f"💬 Contact: {OWNER_CONTACT}\n"
        "═══════════════════════════"
    )
    await update.message.reply_text(msg)


@require_auth_and_channel
async def myinfo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    first_seen = get_first_seen_date(user.id)
    vip_status = "👑 VIP MEMBER" if is_vip(user.id) else "👤 FREE USER"
    usage_today = count_usage_today(user.id)
    referrals = get_referrals_count(user.id)
    bonus = get_bonus_signals(user.id)
    daily_limit = 5 if referrals >= 10 else 3
    
    msg = (
        "═══════════════════════════\n"
        "👤 YOUR ACCOUNT INFO\n"
        "═══════════════════════════\n\n"
        f"🆔 User ID: {user.id}\n"
        f"📛 Username: @{user.username or 'N/A'}\n"
        f"📅 Member since: {first_seen}\n\n"
        "──────────────────────────\n"
        "📊 STATUS\n"
        "──────────────────────────\n"
        f"{vip_status}\n"
        f"🔐 Authentication: ✅ Verified\n\n"
        "──────────────────────────\n"
        "📈 USAGE STATS\n"
        "──────────────────────────\n"
        f"Today: {usage_today}/{daily_limit} signals\n"
        f"Bonus: {bonus} signals\n"
        f"Referrals: {referrals}/10\n\n"
        "──────────────────────────\n"
        "💡 TIPS\n"
        "──────────────────────────\n"
    )
    
    if referrals < 10:
        msg += f"🎯 Refer {10-referrals} more friends\nto unlock 5 daily signals!\n\n"
    else:
        msg += "🎉 You've unlocked 5 daily signals!\n\n"
    
    msg += (
        f"Use /referral to get your link!\n\n"
        "═══════════════════════════"
    )
    
    await update.message.reply_text(msg)


@require_auth_and_channel
async def pairs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = list(USER_TO_DERIV.keys())
    forex = [p for p in pairs if p not in ["XAUUSD", "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"]]
    crypto = [p for p in pairs if p in ["BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"]]
    
    msg = (
        "═══════════════════════════\n"
        "📊 SUPPORTED TRADING PAIRS\n"
        "═══════════════════════════\n\n"
        "💱 FOREX PAIRS:\n"
        "──────────────────────────\n"
    )
    msg += ", ".join(forex[:8]) + ",\n"
    msg += ", ".join(forex[8:]) + "\n\n"
    msg += (
        "🏆 GOLD:\n"
        "──────────────────────────\n"
        "XAUUSD\n\n"
        "₿ CRYPTOCURRENCIES:\n"
        "──────────────────────────\n"
    )
    msg += ", ".join(crypto) + "\n\n"
    msg += (
        "⏰ TIMEFRAMES:\n"
        "──────────────────────────\n"
        "M1, M5, M10, M15\n\n"
        "═══════════════════════════\n"
        "❌ OTC pairs NOT supported\n"
        "═══════════════════════════"
    )
    
    await update.message.reply_text(msg)


@require_auth_and_channel
async def referral_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    me = await context.bot.get_me()
    link = f"https://t.me/{me.username}?start=ref_{user.id}"
    bonus = get_bonus_signals(user.id)
    referrals = get_referrals_count(user.id)
    
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Share Link", url=f"https://t.me/share/url?url={link}")
    ]])
    
    msg = (
        "═══════════════════════════\n"
        "🎁 REFERRAL PROGRAM\n"
        "═══════════════════════════\n\n"
        "📊 YOUR STATS:\n"
        f"• Referrals: {referrals}/10\n"
        f"• Bonus signals: {bonus}\n\n"
        "──────────────────────────\n"
        "💰 REWARDS:\n"
        "──────────────────────────\n"
        "✅ +2 signals per referral\n"
        "✅ 10+ refs = 5 daily signals\n"
        "✅ < 10 refs = 3 daily signals\n\n"
        "──────────────────────────\n"
        "🔗 YOUR LINK:\n"
        "──────────────────────────\n"
        f"{link}\n\n"
        "═══════════════════════════\n"
        "Share with friends & earn! 🚀\n"
        "═══════════════════════════"
    )
    
    await update.message.reply_text(msg, reply_markup=kb)


async def feedback_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if not context.args:
        await update.message.reply_text(
            "📝 SEND FEEDBACK\n\n"
            "Usage:\n"
            "/feedback <your message>\n\n"
            "Example:\n"
            "/feedback Great bot! Signals are accurate!"
        )
        return
    
    feedback_text = " ".join(context.args)
    
    # Get admin IDs
    owner_ids = list(ADMIN_IDS)
    if not owner_ids:
        await update.message.reply_text("✅ Feedback recorded! (Admin forwarding not configured)")
        return
    
    sent = 0
    for oid in owner_ids:
        try:
            await context.bot.send_message(
                chat_id=oid,
                text=(
                    "═══════════════════════════\n"
                    "📩 NEW FEEDBACK RECEIVED\n"
                    "═══════════════════════════\n\n"
                    f"👤 From: {user.full_name}\n"
                    f"🆔 ID: {user.id}\n"
                    f"📛 Username: @{user.username or 'N/A'}\n\n"
                    "──────────────────────────\n"
                    "💬 MESSAGE:\n"
                    "──────────────────────────\n"
                    f"{feedback_text}\n\n"
                    "═══════════════════════════"
                )
            )
            sent += 1
        except Exception as e:
            logger.info(f"Could not forward to {oid}: {e}")
    
    if sent > 0:
        await update.message.reply_text(
            "✅ Feedback sent successfully!\n\n"
            "Thank you for your input! 🙏"
        )
    else:
        await update.message.reply_text("⚠️ Could not send feedback. Contact support.")


async def support_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "═══════════════════════════\n"
        "💬 SUPPORT & CONTACT\n"
        "═══════════════════════════\n\n"
        "Need help? Contact us:\n\n"
        f"👤 Admin: {OWNER_CONTACT}\n"
        f"📢 Channel: {CHANNEL_USERNAME}\n\n"
        "──────────────────────────\n"
        "📝 For support requests:\n"
        "──────────────────────────\n"
        "• VIP membership inquiries\n"
        "• Technical issues\n"
        "• Account problems\n"
        "• General questions\n\n"
        "═══════════════════════════\n"
        "We're here to help! 🚀\n"
        "═══════════════════════════"
    )
    await update.message.reply_text(msg)


# Admin commands
async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    users = get_all_users()
    await update.message.reply_text(f"👥 Total users: {len(users)}")


async def setvip_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /setvip <user_id> <days>")
        return
    try:
        target = int(context.args[0])
        days = int(context.args[1])
        set_vip(target, days)
        await update.message.reply_text(f"✅ VIP set for user {target} for {days} days.")
        try:
            await context.bot.send_message(
                chat_id=target,
                text=(
                    "═══════════════════════════\n"
                    "🎉 VIP UPGRADE SUCCESSFUL!\n"
                    "═══════════════════════════\n\n"
                    f"Congratulations! You've been upgraded to VIP for {days} days!\n\n"
                    "🌟 VIP BENEFITS:\n"
                    "✅ Unlimited daily signals\n"
                    "✅ Priority support\n"
                    "✅ Exclusive features\n"
                    "✅ No daily limits\n\n"
                    "Use /myinfo to check your status!\n\n"
                    "═══════════════════════════\n"
                    "Enjoy your VIP access! 👑\n"
                    "═══════════════════════════"
                )
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ Could not notify user: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def removevip_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /removevip <user_id>")
        return
    try:
        target = int(context.args[0])
        remove_vip(target)
        await update.message.reply_text(f"✅ VIP removed for user {target}.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def resetauth_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /resetauth <user_id>")
        return
    try:
        target = int(context.args[0])
        set_authenticated(target, False)
        await update.message.reply_text(f"✅ Authentication reset for user {target}.")
        try:
            await context.bot.send_message(
                chat_id=target,
                text=(
                    "🔐 Your authentication has been reset by admin.\n\n"
                    "Please use /start to authenticate again."
                )
            )
        except Exception:
            pass
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def viplist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    rows = list_vip()
    if not rows:
        await update.message.reply_text("No active VIP users.")
        return
    lines = ["👑 Active VIP Users:\n"]
    for uid, exp in rows:
        lines.append(f"• {uid} (expires: {exp})")
    await update.message.reply_text("\n".join(lines))


async def vipstats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    st = vip_stats()
    await update.message.reply_text(f"📊 VIP: {st['vip']} / Total: {st['total']}")


async def broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    replied_message = update.message.reply_to_message
    if not replied_message:
        await update.message.reply_text("⚠️ Please reply to a message to broadcast it.")
        return
    users = get_all_users()
    sent = failed = 0
    await update.message.reply_text(f"📢 Broadcasting to {len(users)} users...")
    for uid in users:
        try:
            await replied_message.copy(chat_id=uid)
            sent += 1
        except Exception:
            failed += 1
    await update.message.reply_text(f"✅ Broadcast complete!\n\nSent: {sent}\nFailed: {failed}")


async def userlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT user_id, username, first_seen, is_vip, authenticated FROM users")
    rows = cur.fetchall()
    con.close()
    if not rows:
        await update.message.reply_text("No users found.")
        return
    lines = []
    for r in rows:
        uname = r["username"] or "N/A"
        vip = "👑" if r["is_vip"] else "👤"
        auth = "✅" if r["authenticated"] else "❌"
        lines.append(f"{vip} @{uname} | ID: {r['user_id']} | Auth: {auth}")
    msg = "\n".join(lines)
    if len(msg) > 4000:
        for i in range(0, len(msg), 4000):
            await update.message.reply_text(msg[i:i+4000])
    else:
        await update.message.reply_text(msg)


# -------------------------
# Main app
# -------------------------
def main():
    init_db()
    upgrade_db()
    app = Application.builder().token(BOT_TOKEN).build()

    # Start conversation with authentication integrated
    start_conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_USERNAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_password)],
            ASK_PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_license)],
            ASK_LICENSE: [MessageHandler(filters.TEXT & ~filters.COMMAND, verify_credentials)],
        },
        fallbacks=[CommandHandler("cancel", auth_cancel)],
    )

    # Signal generation conversation
    signal_conv = ConversationHandler(
        entry_points=[CommandHandler("getsignal", getsignal_entry)],
        states={
            ASK_PAIR: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_tf)],
            ASK_TF: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_country)],
            ASK_COUNTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, do_analyze)],
        },
        fallbacks=[CommandHandler("cancel", auth_cancel)],
    )

    # Handlers
    app.add_handler(start_conv)
    app.add_handler(CallbackQueryHandler(verify_join, pattern="^verify_join$"))
    app.add_handler(signal_conv)

    # User commands
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("about", about_cmd))
    app.add_handler(CommandHandler("myinfo", myinfo_cmd))
    app.add_handler(CommandHandler("pairs", pairs_cmd))
    app.add_handler(CommandHandler("referral", referral_cmd))
    app.add_handler(CommandHandler("feedback", feedback_cmd))
    app.add_handler(CommandHandler("support", support_cmd))

    # Admin commands
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CommandHandler("setvip", setvip_cmd))
    app.add_handler(CommandHandler("removevip", removevip_cmd))
    app.add_handler(CommandHandler("resetauth", resetauth_cmd))
    app.add_handler(CommandHandler("viplist", viplist_cmd))
    app.add_handler(CommandHandler("vipstats", vipstats_cmd))
    app.add_handler(CommandHandler("broadcast", broadcast_cmd))
    app.add_handler(CommandHandler("userlist", userlist_cmd))

    logger.info("BD Trader Auto Bot is running...")
    app.run_polling(close_loop=False)


# ... (existing code before if __name__ == "__main__":)

# +++ নিচের সম্পূর্ণ ব্লকটি দিয়ে পুরনোটি Replace করুন +++
if __name__ == "__main__":
    
    # Start the web server in a separate thread
    if FLASK_AVAILABLE and app:
        logger.info("Starting Flask web server for health checks...")
        # Render এর জন্য PORT সেট করা
        port = int(os.environ.get("PORT", 10000))
        
        def run_web_server():
            try:
                # Flask এর নিজস্ব সার্ভার ব্যবহার করা
                app.run(host="0.0.0.0", port=port, debug=False)
            except Exception as e:
                logger.error(f"Flask server failed: {e}")

        # ওয়েব সার্ভারটি একটি আলাদা থ্রেডে চালু করা
        web_thread = Thread(target=run_web_server, daemon=True)
        web_thread.start()
    
    # এটি আপনার বটের মূল লুপ, এটি আগের মতোই থাকবে
    while True:
        try:
            main() # main() ফাংশনটি আপনার বটটি চালু করবে
        except Exception as e:
            logger.exception("Bot crashed — restarting in 5s...")
            time.sleep(5)
# +++ এই পর্যন্ত +++
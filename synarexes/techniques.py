import os
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import pytz
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from datetime import datetime
import calculateprices
import time
import threading
import traceback
from datetime import timedelta
import traceback
import shutil
from datetime import datetime
import re
import placeorders
import insiders_server
import timeorders


def load_brokers_dictionary():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    """Load brokers config from JSON file with error handling and fallback."""
    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Convert numeric strings back to int where needed
        for broker_name, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        print(f"Brokers config loaded successfully → {len(data)} broker(s)", "SUCCESS")
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in brokersdictionary.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load brokersdictionary.json: {e}", "CRITICAL")
        return {}
brokersdictionary = load_brokers_dictionary()


BASE_ERROR_FOLDER = r"C:\xampp\htdocs\chronedge\synarex\chart\debugs"
TIMEFRAME_MAP = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4
}
ERROR_JSON_PATH = os.path.join(BASE_ERROR_FOLDER, "chart_errors.json")
           
def log_and_print(message, level="INFO"):
    """Log and print messages in a structured format."""
    timestamp = datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level:8} | {message}")

def save_errors(error_log):
    """Save error log to JSON file."""
    try:
        os.makedirs(BASE_ERROR_FOLDER, exist_ok=True)
        with open(ERROR_JSON_PATH, 'w') as f:
            json.dump(error_log, f, indent=4)
        log_and_print("Error log saved", "ERROR")
    except Exception as e:
        log_and_print(f"Failed to save error log: {str(e)}", "ERROR")

def initialize_mt5(terminal_path, login_id, password, server):
    """Initialize MetaTrader 5 terminal for a specific broker."""
    error_log = []
    if not os.path.exists(terminal_path):
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"MT5 terminal executable not found: {terminal_path}",
            "broker": server
        })
        save_errors(error_log)
        log_and_print(f"MT5 terminal executable not found: {terminal_path}", "ERROR")
        return False, error_log

    try:
        if not mt5.initialize(
            path=terminal_path,
            login=int(login_id),
            server=server,
            password=password,
            timeout=30000
        ):
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to initialize MT5: {mt5.last_error()}",
                "broker": server
            })
            save_errors(error_log)
            log_and_print(f"Failed to initialize MT5: {mt5.last_error()}", "ERROR")
            return False, error_log

        if not mt5.login(login=int(login_id), server=server, password=password):
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to login to MT5: {mt5.last_error()}",
                "broker": server
            })
            save_errors(error_log)
            log_and_print(f"Failed to login to MT5: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            return False, error_log

        log_and_print(f"MT5 initialized and logged in successfully (loginid={login_id}, server={server})", "SUCCESS")
        return True, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Unexpected error in initialize_mt5: {str(e)}",
            "broker": server
        })
        save_errors(error_log)
        log_and_print(f"Unexpected error in initialize_mt5: {str(e)}", "ERROR")
        return False, error_log

def get_symbols():
    """Retrieve all available symbols from MT5."""
    error_log = []
    symbols = mt5.symbols_get()
    if not symbols:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve symbols: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to retrieve symbols: {mt5.last_error()}", "ERROR")
        return [], error_log

    available_symbols = [s.name for s in symbols]
    log_and_print(f"Retrieved {len(available_symbols)} symbols", "INFO")
    return available_symbols, error_log

def fetch_ohlcv_data(symbol, mt5_timeframe, bars):
    """Fetch OHLCV data for a given symbol and timeframe."""
    error_log = []
    if not mt5.symbol_select(symbol, True):
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to select symbol {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to select symbol {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve rates for {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to retrieve rates for {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "tick_volume": float, "spread": int, "real_volume": float
    })
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    log_and_print(f"OHLCV data fetched for {symbol}", "INFO")
    return df, error_log

def define_candle_types(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        candles = json.load(f)

    def o(c): return c['open']
    def h(c): return c['high']
    def l(c): return c['low']
    def c(c): return c['close']

    # Reset all labels
    for candle in candles:
        candle["candle_type"] = "Regular Candle"
        candle["is_fvg_middle"] = False

    for i, candle in enumerate(candles):
        open_p = o(candle)
        high_p = h(candle)
        low_p = l(candle)
        close_p = c(candle)

        range_p = high_p - low_p + 1e-10
        body = abs(close_p - open_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        is_bull = close_p >= open_p

        body_pct = body / range_p
        upper_pct = upper_wick / range_p
        lower_pct = lower_wick / range_p

        types = []

        # ────────────────────── SINGLE CANDLE PATTERNS ──────────────────────
        if body_pct <= 0.08:
            if lower_pct >= 0.60 and upper_pct <= 0.15:
                types.append("Dragonfly Doji")
            elif upper_pct >= 0.60 and lower_pct <= 0.15:
                types.append("Gravestone Doji")
            elif upper_pct >= 0.30 and lower_pct >= 0.30:
                types.append("Long-Legged Doji")
            else:
                types.append("Doji")

        elif lower_pct >= 0.60 and upper_pct <= 0.20 and body_pct <= 0.40:
            types.append("Hammer" if is_bull else "Hanging Man")
        elif upper_pct >= 0.60 and lower_pct <= 0.20 and body_pct <= 0.40:
            types.append("Shooting Star" if not is_bull else "Inverted Hammer")

        elif upper_pct <= 0.05 and lower_pct <= 0.05:
            types.append("Marubozu Bullish" if is_bull else "Marubozu Bearish")

        elif body_pct >= 0.80:
            types.append("Displacement")

        elif body_pct <= 0.35 and upper_pct >= 0.30 and lower_pct >= 0.30:
            types.append("Spinning Top")

        # ────────────────────── TWO CANDLE PATTERNS ──────────────────────
        if i > 0:
            prev = candles[i-1]
            # Engulfing
            if c(prev) < o(prev) and is_bull and open_p <= c(prev) and close_p >= o(prev):
                types.append("Bullish Engulfing")
            elif c(prev) > o(prev) and not is_bull and open_p >= c(prev) and close_p <= o(prev):
                types.append("Bearish Engulfing")
            # Harami
            if c(prev) < o(prev) and is_bull and open_p > c(prev) and close_p < o(prev):
                types.append("Bullish Harami")
            elif c(prev) > o(prev) and not is_bull and open_p < c(prev) and close_p > o(prev):
                types.append("Bearish Harami")
            # Inside / Outside
            if high_p <= h(prev) and low_p >= l(prev):
                types.append("Inside Bar")
            if high_p > h(prev) + 0.0005 and low_p < l(prev) - 0.0005:
                types.append("Outside Bar")

        # ────────────────────── FINAL & CORRECT FVG DETECTION (ICT/SMC) ──────────────────────
        if i >= 2:
            c1 = candles[i-2]   # oldest
            c2 = candles[i-1]   # ← MIDDLE candle (gets labeled)
            c3 = candle         # newest

            min_gap_pips = 0.0003   # ~3 pips minimum → eliminates all noise

            # BULLISH FVG → price gapped UP → inefficiency BELOW
            if h(c1) < l(c3) and (l(c3) - h(c1)) >= min_gap_pips:
                c2["candle_type"] = "FVG Middle (Bullish)"
                c2["is_fvg_middle"] = True

            # BEARISH FVG → price gapped DOWN → inefficiency ABOVE
            elif l(c1) > h(c3) and (l(c1) - h(c3)) >= min_gap_pips:
                c2["candle_type"] = "FVG Middle (Bearish)"
                c2["is_fvg_middle"] = True

        # ────────────────────── FINAL LABEL ASSIGNMENT ──────────────────────
        current_label = candle["candle_type"]

        if "FVG Middle" in current_label:
            # FVG has highest priority — optionally append other patterns
            if types:
                candle["candle_type"] = current_label + " | " + " | ".join(types)
        elif current_label == "Regular Candle" and types:
            candle["candle_type"] = " | ".join(types)

    # Save fixed file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(candles, f, indent=4, ensure_ascii=False)

    print(f"Processed {len(candles)} candles → 100% correct FVG detection guaranteed")

def save_candle_data(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels, ch_labels, cl_labels):
    """
    Save all candles + highlight the SECOND-MOST-RECENT candle (candle_number = 1)
    as 'x' in previouslatestcandle.json with age in days/hours.
    Now includes Child Highs (CH) and Child Lows (CL).
    """
    error_log = []
    all_json_path = os.path.join(timeframe_folder, "all_candles.json")
    latest_json_path = os.path.join(timeframe_folder, "previouslatestcandle.json")
    
    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    try:
        if len(df) < 2:
            error_msg = f"Not enough data for {symbol} ({timeframe_str})"
            log_and_print(error_msg, "ERROR")
            error_log.append({"error": error_msg, "timestamp": now.isoformat()})
            save_errors(error_log)
            return error_log

        # === Prepare PH/PL/CH/CL lookup dictionaries ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}
        ch_dict = {t: label for label, _, t in ch_labels}
        cl_dict = {t: label for label, _, t in cl_labels}

        # === Save ALL candles (oldest = 0) ===
        all_candles = []
        for i, (ts, row) in enumerate(df[::-1].iterrows()):  # oldest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL',
                "is_ch": ch_dict.get(ts, None) == 'CH',
                "is_cl": cl_dict.get(ts, None) == 'CL'
            })
            all_candles.append(candle)

        # Write all candles
        with open(all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_candles, f, indent=4)

        # === ENRICH WITH CANDLE PATTERN NAMES ===
        define_candle_types(all_json_path)

        # === Save CANDLE #1 as "x" with age ===
        if len(all_candles) < 2:
            raise ValueError("Expected at least 2 candles")

        previous_latest_candle = all_candles[1].copy()
        candle_time_str = previous_latest_candle["time"]
        candle_time = datetime.strptime(candle_time_str, '%Y-%m-%d %H:%M:%S')
        candle_time = lagos_tz.localize(candle_time)

        delta = now - candle_time
        total_hours = delta.total_seconds() / 3600
        age_str = f"{int(total_hours)} hour{'s' if int(total_hours) != 1 else ''} old" if total_hours <= 24 else \
                  f"{int(total_hours // 24)} day{'s' if total_hours // 24 != 1 else ''} old"

        previous_latest_candle["age"] = age_str
        previous_latest_candle["id"] = "x"
        previous_latest_candle.pop("candle_number", None)

        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(previous_latest_candle, f, indent=4)

        log_and_print(
            f"SAVED {symbol} {timeframe_str}: all_candles.json ({len(all_candles)} candles with PH/PL/CH/CL + types) + previouslatestcandle.json ({age_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_candle_data failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({"error": err, "timestamp": now.isoformat()})
        save_errors(error_log)

    return error_log

def save_next_candles(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels, ch_labels, cl_labels):
    """
    Save candles that appear **after** the previous-latest candle ('x')
    into <timeframe_folder>/nextcandles.json
    Now includes CH/CL flags.
    """
    error_log = []
    next_json_path = os.path.join(timeframe_folder, "nextcandles.json")

    try:
        if len(df) < 3:
            return error_log  # need at least: old, previous-latest (x), and one new

        # === Build lookup dicts ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}
        ch_dict = {t: label for label, _, t in ch_labels}
        cl_dict = {t: label for label, _, t in cl_labels}

        # === Build full ordered list: oldest → newest ===
        ordered_candles = []
        for i, (ts, row) in enumerate(df[::-1].iterrows()):  # oldest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL',
                "is_ch": ch_dict.get(ts, None) == 'CH',
                "is_cl": cl_dict.get(ts, None) == 'CL'
            })
            ordered_candles.append(candle)

        # === Find timestamp of 'x' candle (candle_number == 1) ===
        if len(ordered_candles) < 2:
            return error_log

        x_candle_time_str = ordered_candles[1]["time"]
        x_time = datetime.strptime(x_candle_time_str, '%Y-%m-%d %H:%M:%S')

        # === Collect candles newer than 'x' ===
        next_candles = []
        for candle in ordered_candles:
            candle_time = datetime.strptime(candle["time"], '%Y-%m-%d %H:%M:%S')
            if candle_time > x_time:
                next_candles.append(candle)

        if not next_candles:
            return error_log

        # === Save ===
        with open(next_json_path, 'w', encoding='utf-8') as f:
            json.dump(next_candles, f, indent=4)

        log_and_print(
            f"SAVED {symbol} {timeframe_str}: nextcandles.json "
            f"({len(next_candles)} candles after {x_candle_time_str}, with PH/PL/CH/CL)",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_next_candles failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).isoformat()
        })
        save_errors(error_log)

    return error_log
    
def identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right):
    """Identify Parent Highs (PH) and Parent Lows (PL) based on neighbor candles."""
    error_log = []
    ph_indices = []
    pl_indices = []
    ph_labels = []
    pl_labels = []

    try:
        for i in range(len(df)):
            if i >= len(df) - neighborcandles_right:
                continue

            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            right_highs = df.iloc[i + 1:i + neighborcandles_right + 1]['high']
            right_lows = df.iloc[i + 1:i + neighborcandles_right + 1]['low']
            left_highs = df.iloc[max(0, i - neighborcandles_left):i]['high']
            left_lows = df.iloc[max(0, i - neighborcandles_left):i]['low']

            if len(right_highs) == neighborcandles_right:
                is_ph = True
                if len(left_highs) > 0:
                    is_ph = current_high > left_highs.max()
                is_ph = is_ph and current_high > right_highs.max()
                if is_ph:
                    ph_indices.append(df.index[i])
                    ph_labels.append(('PH', current_high, df.index[i]))

            if len(right_lows) == neighborcandles_right:
                is_pl = True
                if len(left_lows) > 0:
                    is_pl = current_low < left_lows.min()
                is_pl = is_pl and current_low < right_lows.min()
                if is_pl:
                    pl_indices.append(df.index[i])
                    pl_labels.append(('PL', current_low, df.index[i]))

        log_and_print(f"Identified {len(ph_indices)} PH and {len(pl_indices)} PL for {df['symbol'].iloc[0]}", "INFO")
        return ph_labels, pl_labels, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to identify PH/PL: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to identify PH/PL: {str(e)}", "ERROR")
        return [], [], error_log

def identify_child_highs_and_lows(df, neighborcandles_left, neighborcandles_right):
    """Identify Child Highs (CH) and Child Lows (CL) based on neighbor candles."""
    error_log = []
    ch_indices = []
    cl_indices = []
    ch_labels = []
    cl_labels = []

    try:
        for i in range(len(df)):
            if i >= len(df) - neighborcandles_right:
                continue

            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            right_highs = df.iloc[i + 1:i + neighborcandles_right + 1]['high']
            right_lows = df.iloc[i + 1:i + neighborcandles_right + 1]['low']
            left_highs = df.iloc[max(0, i - neighborcandles_left):i]['high']
            left_lows = df.iloc[max(0, i - neighborcandles_left):i]['low']

            if len(right_highs) == neighborcandles_right:
                is_ch = True
                if len(left_highs) > 0:
                    is_ch = current_high > left_highs.max()
                is_ch = is_ch and current_high > right_highs.max()
                if is_ch:
                    ch_indices.append(df.index[i])
                    ch_labels.append(('CH', current_high, df.index[i]))

            if len(right_lows) == neighborcandles_right:
                is_cl = True
                if len(left_lows) > 0:
                    is_cl = current_low < left_lows.min()
                is_cl = is_cl and current_low < right_lows.min()
                if is_cl:
                    cl_indices.append(df.index[i])
                    cl_labels.append(('CL', current_low, df.index[i]))

        log_and_print(f"Identified {len(ch_indices)} CH and {len(cl_indices)} CL for {df['symbol'].iloc[0]}", "INFO")
        return ch_labels, cl_labels, error_log

    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to identify CH/CL: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to identify CH/CL: {str(e)}", "ERROR")
        return [], [], error_log
    
def generate_and_save_chart(df, symbol, timeframe_str, timeframe_folder,
                            parent_left, parent_right, child_left, child_right):
    """
    Generate chart.png + chartanalysed.png
    PH/PL colors = YOUR ORIGINAL (blue ^ / purple v)
    CH/CL = NEW (cyan small ^ / orange small v)
    """
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    trendline_log = []

    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green", down="red",
                edge="inherit",
                wick={"up": "green", "down": "red"},
                volume="gray"
            )
        )

        # === 1. Basic clean chart ===
        fig, axlist = mpf.plot(df, type='candle', style=custom_style, volume=False,
                               title=f"{symbol} ({timeframe_str})", returnfig=True,
                               warn_too_much_data=5000)
        for ax in axlist:
            for line in ax.get_lines():
                if not line.get_label():
                    line.set_linewidth(0.5)
        fig.set_size_inches(25, fig.get_size_inches()[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        # === 2. Detect Parent Highs/Lows (PH/PL) ===
        ph_labels, pl_labels, phpl_errors = identifyparenthighsandlows(df, parent_left, parent_right)
        error_log.extend(phpl_errors)

        # === 3. Detect Child Highs/Lows (CH/CL) ===
        ch_labels, cl_labels, chcl_errors = identify_child_highs_and_lows(df, child_left, child_right)
        error_log.extend(chcl_errors)

        # ----------------------------------------------------
        # === 4. Label Prioritization (PH/PL over CH/CL) ===
        # ----------------------------------------------------
        
        # Get indices of all identified Parent Highs and Parent Lows
        parent_high_indices = {t for _, _, t in ph_labels}
        parent_low_indices = {t for _, _, t in pl_labels}

        # Filter out Child Highs that are also Parent Highs
        # If a candle is a PH, it cannot be a CH (Parent dictates authority)
        ch_labels_filtered = []
        for label, price, t in ch_labels:
            if t not in parent_high_indices:
                ch_labels_filtered.append((label, price, t))
        ch_labels = ch_labels_filtered

        # Filter out Child Lows that are also Parent Lows
        # If a candle is a PL, it cannot be a CL
        cl_labels_filtered = []
        for label, price, t in cl_labels:
            if t not in parent_low_indices:
                cl_labels_filtered.append((label, price, t))
        cl_labels = cl_labels_filtered
        
        # === 5. Add plots — EXACT SAME COLORS AS BEFORE FOR PH/PL ===
        apds = []

        # PH → Blue large triangle (YOUR ORIGINAL)
        if ph_labels:
            ph_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ph_labels:
                ph_series.loc[t] = price
            apds.append(mpf.make_addplot(ph_series, type='scatter', markersize=100, marker='^', color='blue'))

        # PL → Purple large triangle (YOUR ORIGINAL)
        if pl_labels:
            pl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in pl_labels:
                pl_series.loc[t] = price
            apds.append(mpf.make_addplot(pl_series, type='scatter', markersize=100, marker='v', color='purple'))

        # CH → Cyan small triangle (NEW)
        if ch_labels:
            ch_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ch_labels:
                ch_series.loc[t] = price
            # Note: ch_labels now contains only true Child Highs
            apds.append(mpf.make_addplot(ch_series, type='scatter', markersize=70, marker='^', color='cyan'))

        # CL → Orange small triangle (NEW)
        if cl_labels:
            cl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in cl_labels:
                cl_series.loc[t] = price
            # Note: cl_labels now contains only true Child Lows
            apds.append(mpf.make_addplot(cl_series, type='scatter', markersize=70, marker='v', color='orange'))

        # === 6. Log (Update step number) ===
        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "status": "info",
            "reason": f"PH:{len(ph_labels)} PL:{len(pl_labels)} CH:{len(ch_labels)} CL:{len(cl_labels)}"
        })
        with open(trendline_log_json_path, 'w', encoding='utf-8') as f:
            json.dump(trendline_log, f, indent=4)

        # === 7. Final analysed chart (Update step number) ===
        fig, axlist = mpf.plot(
            df, type='candle', style=custom_style, volume=False,
            title=f"{symbol} ({timeframe_str}) - PH/PL + CH/CL",
            addplot=apds, returnfig=True
        )
        for ax in axlist:
            for line in ax.get_lines():
                if not line.get_label():
                    line.set_linewidth(0.5)
        fig.set_size_inches(25, fig.get_size_inches()[1])
        axlist[0].grid(True, linestyle='--', alpha=0.6)
        fig.savefig(chart_analysed_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        log_and_print(f"Analysed chart saved with original PH=blue PL=purple + new CH/CL", "SUCCESS")

        return chart_path, error_log, ph_labels, pl_labels, ch_labels, cl_labels

    except Exception as e:
        error_log.append({"error": str(e), "symbol": symbol, "timeframe": timeframe_str})
        save_errors(error_log)
        log_and_print(f"Chart generation failed: {e}", "ERROR")
        return None, error_log, [], [], [], []     

def detect_candle_contours(chart_path, symbol, timeframe_str, timeframe_folder, candleafterintersector=2, minbreakoutcandleposition=5, startOBsearchFrom=0, minOBleftneighbor=1, minOBrightneighbor=1, reversal_leftcandle=0, reversal_rightcandle=0):
    error_log = []
    contour_json_path = os.path.join(timeframe_folder, "chart_contours.json")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    ob_none_oi_json_path = os.path.join(timeframe_folder, "ob_none_oi_data.json")
    output_image_path = os.path.join(timeframe_folder, "chart_with_contours.png")
    candle_json_path = os.path.join(timeframe_folder, "all_candles.json")
    trendline_log = []
    ob_none_oi_data = []
    team_counter = 1  # Counter for naming teams (team1, team2, ...)

    def draw_chevron_arrow(img, x, y, direction, color=(0, 0, 255), line_length=15, chevron_size=8):
        """
        Draw a chevron-style arrow on the image at position (x, y).
        direction: 'up' for upward chevron (base at y, chevron '^' at y - line_length),
                  'down' for downward chevron (base at y, chevron 'v' at y + line_length).
        color: BGR tuple, default red (0, 0, 255).
        line_length: Length of the vertical line in pixels.
        chevron_size: Width of the chevron head (distance from center to each wing tip) in pixels.
        """
        if direction == 'up':
            top_y = y - line_length
            cv2.line(img, (x, y), (x, top_y), color, thickness=1)
            cv2.line(img, (x - chevron_size // 2, top_y + chevron_size // 2), (x, top_y), color, thickness=1)
            cv2.line(img, (x + chevron_size // 2, top_y + chevron_size // 2), (x, top_y), color, thickness=1)
        else:  # direction == 'down'
            top_y = y + line_length
            cv2.line(img, (x, y), (x, top_y), color, thickness=1)
            cv2.line(img, (x - chevron_size // 2, top_y - chevron_size // 2), (x, top_y), color, thickness=1)
            cv2.line(img, (x + chevron_size // 2, top_y - chevron_size // 2), (x, top_y), color, thickness=1)

    def draw_right_arrow(img, x, y, oi_x=None, color=(255, 0, 0), line_length=15, arrow_size=8):
        """
        Draw a right-facing arrow on the image at position (x, y).
        If oi_x is provided, extend the arrow to touch the 'oi' candle's body at x=oi_x with red color.
        If oi_x is None, extend the arrow to the right edge of the image.
        color: BGR tuple, default red (255, 0, 0) when oi_x is provided.
        line_length: Length of the horizontal line in pixels (default 15 if oi_x is None and not extending to edge).
        arrow_size: Size of the arrowhead (distance from center to each wing tip) in pixels.
        """
        img_height, img_width = img.shape[:2]
        if oi_x is not None:
            line_length = max(10, oi_x - x)
            end_x = oi_x
            color = (255, 0, 0)
        else:
            end_x = img_width - 5
            color = (0, 255, 0)
        cv2.line(img, (x, y), (end_x, y), color, thickness=1)
        cv2.line(img, (end_x - arrow_size // 2, y - arrow_size // 2), (end_x, y), color, thickness=1)
        cv2.line(img, (end_x - arrow_size // 2, y + arrow_size // 2), (end_x, y), color, thickness=1)

    def draw_oi_marker(img, x, y, color=(0, 255, 0)):
        """
        Draw an 'oi' marker (e.g., a green circle with a different radius) at position (x, y).
        """
        cv2.circle(img, (x, y), 7, color, thickness=1)

    def find_reversal_candle(start_idx, is_ph):
        """
        Find the first candle after start_idx where:
        - For PL team: low is lower than specified left and right neighbors.
        - For PH team: high is higher than specified left and right neighbors.
        Returns (index, x, y) or None if no such candle is found.
        """
        for idx in range(start_idx - 1, -1, -1):
            if idx not in candle_bounds:
                continue
            left_neighbors = []
            right_neighbors = []
            if reversal_leftcandle in [0, 1]:
                if idx - 1 in candle_bounds:
                    left_neighbors.append(idx - 1)
            elif reversal_leftcandle == 2:
                if idx - 1 in candle_bounds and idx - 2 in candle_bounds:
                    left_neighbors.extend([idx - 1, idx - 2])
            if reversal_rightcandle in [0, 1]:
                if idx + 1 in candle_bounds:
                    right_neighbors.append(idx + 1)
            elif reversal_rightcandle == 2:
                if idx + 1 in candle_bounds and idx + 2 in candle_bounds:
                    right_neighbors.extend([idx + 1, idx + 2])
            if len(left_neighbors) < max(1, reversal_leftcandle) or len(right_neighbors) < max(1, reversal_rightcandle):
                continue
            current_candle = candle_bounds[idx]
            all_neighbors_valid = True
            if is_ph:
                for neighbor_idx in left_neighbors + right_neighbors:
                    neighbor_candle = candle_bounds[neighbor_idx]
                    if current_candle["high"] <= neighbor_candle["high"]:
                        all_neighbors_valid = False
                        break
                if all_neighbors_valid:
                    x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                    y = current_candle["high_y"] - 10
                    return idx, x, y
            else:
                for neighbor_idx in left_neighbors + right_neighbors:
                    neighbor_candle = candle_bounds[neighbor_idx]
                    if current_candle["low"] >= neighbor_candle["low"]:
                        all_neighbors_valid = False
                        break
                if all_neighbors_valid:
                    x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                    y = current_candle["low_y"] + 10
                    return idx, x, y
        return None

    try:
        img = cv2.imread(chart_path)
        if img is None:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load chart image {chart_path} for contour detection",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to load chart image {chart_path} for contour detection", "ERROR")
            return error_log

        img_height, img_width = img.shape[:2]

        try:
            with open(candle_json_path, 'r') as f:
                candle_data = json.load(f)
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load {candle_json_path} for PH/PL data: {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to load {candle_json_path} for PH/PL data: {str(e)}", "ERROR")
            return error_log

        ph_candles = [c for c in candle_data if c.get("is_ph", False)]
        pl_candles = [c for c in candle_data if c.get("is_pl", False)]
        ph_indices = {int(c["candle_number"]): c for c in ph_candles}
        pl_indices = {int(c["candle_number"]): c for c in pl_candles}

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(img_hsv, red_lower1, red_upper1) | cv2.inRange(img_hsv, red_lower2, red_upper2)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours = sorted(green_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        red_contours = sorted(red_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        green_count = len(green_contours)
        red_count = len(red_contours)
        total_count = green_count + red_count
        all_contours = green_contours + red_contours
        all_contours = sorted(all_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)

        contour_positions = {}
        candle_bounds = {}
        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            contour_positions[i] = {"x": x + w // 2, "y": y, "width": w, "height": h}
            candle = candle_data[i]
            candle_bounds[i] = {
                "high_y": y,
                "low_y": y + h,
                "body_top_y": y + min(h // 4, 10),
                "body_bottom_y": y + h - min(h // 4, 10),
                "x_left": x,
                "x_right": x + w,
                "high": float(candle["high"]),
                "low": float(candle["low"])
            }

        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.pointPolygonTest(contour, (x + w // 2, y + h // 2), False) >= 0:
                if green_mask[y + h // 2, x + w // 2] > 0:
                    cv2.drawContours(img, [contour], -1, (0, 128, 0), 1)
                elif red_mask[y + h // 2, x + w // 2] > 0:
                    cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)
            if i in ph_indices:
                points = np.array([
                    [x + w // 2, y - 10],
                    [x + w // 2 - 10, y + 5],
                    [x + w // 2 + 10, y + 5]
                ])
                cv2.fillPoly(img, [points], color=(255, 0, 0))
            if i in pl_indices:
                points = np.array([
                    [x + w // 2, y + h + 10],
                    [x + w // 2 - 10, y + h - 5],
                    [x + w // 2 + 10, y + h - 5]
                ])
                cv2.fillPoly(img, [points], color=(128, 0, 128))

        def find_intersectors(sender_idx, receiver_idx, sender_x, sender_y, is_ph, receiver_price):
            intersectors = []
            receiver_pos = contour_positions.get(receiver_idx)
            receiver_y = receiver_pos["y"] if is_ph else receiver_pos["y"] + receiver_pos["height"]
            dx = receiver_pos["x"] - sender_x
            dy = receiver_y - sender_y
            if dx == 0:
                slope = float('inf')
            else:
                slope = dy / dx
            i = receiver_idx - 1
            found_first = False
            first_intersector_price = None
            previous_intersector_price = None
            while i >= 0:
                if i not in candle_bounds or i == sender_idx or i == receiver_idx:
                    i -= 1
                    continue
                bounds = candle_bounds[i]
                x = bounds["x_left"]
                if slope == float('inf'):
                    y = sender_y
                else:
                    y = sender_y + slope * (x - sender_x)
                if not found_first:
                    price = bounds["high"] if is_ph else bounds["low"]
                    price_range = max(c["high"] for c in candle_data) - min(c["low"] for c in candle_data)
                    if price_range == 0:
                        y_price = y
                    else:
                        min_y = min(b["high_y"] for b in candle_bounds.values())
                        max_y = max(b["low_y"] for b in candle_bounds.values())
                        price_min = min(c["low"] for c in candle_data)
                        y_price = max_y - ((price - price_min) / price_range) * (max_y - min_y)
                        y_price = int(y_price)
                    if abs(y - y_price) <= 10:
                        check_idx = i - candleafterintersector
                        is_trendbreaker = False
                        if check_idx >= 0 and check_idx in candle_bounds:
                            check_candle = candle_bounds[check_idx]
                            check_price = check_candle["high"] if is_ph else check_candle["low"]
                            if (is_ph and check_price > receiver_price) or (not is_ph and check_price < receiver_price):
                                is_trendbreaker = True
                        intersectors.append((i, x, y_price, price, True, bounds["high_y"], bounds["low_y"], is_trendbreaker))
                        found_first = True
                        first_intersector_price = price
                        previous_intersector_price = price
                        if is_trendbreaker:
                            break
                        i -= 1
                        continue
                if found_first and bounds["body_top_y"] <= y <= bounds["body_bottom_y"]:
                    current_price = bounds["high"] if is_ph else bounds["low"]
                    is_trendbreaker = False
                    check_idx = i - candleafterintersector
                    if check_idx >= 0 and check_idx in candle_bounds:
                        check_candle = candle_bounds[check_idx]
                        check_price = check_candle["high"] if is_ph else check_candle["low"]
                        if (is_ph and check_price > previous_intersector_price) or (not is_ph and check_price < previous_intersector_price):
                            is_trendbreaker = True
                        elif is_ph and current_price > first_intersector_price:
                            is_trendbreaker = True
                        elif not is_ph and current_price < first_intersector_price:
                            is_trendbreaker = True
                    intersectors.append((i, x, int(y), None, False, bounds["high_y"], bounds["low_y"], is_trendbreaker))
                    previous_intersector_price = current_price
                    if is_trendbreaker:
                        break
                i -= 1
            return intersectors, slope

        def find_intruder(sender_idx, receiver_idx, sender_price, is_ph):
            start_idx = min(sender_idx, receiver_idx) + 1
            end_idx = max(sender_idx, receiver_idx) - 1
            for i in range(start_idx, end_idx + 1):
                if i in candle_bounds:
                    candle = candle_bounds[i]
                    price = candle["high"] if is_ph else candle["low"]
                    if (is_ph and price > sender_price) or (not is_ph and price < sender_price):
                        return i
            return None

        def find_OB(start_idx, receiver_idx, is_ph):
            """
            Find the first candle from start_idx + startOBsearchFrom to receiver_idx where its high (for PH) is higher than
            specified left and right neighbors, or its low (for PL) is lower than specified neighbors.
            Returns (index, x, y) or None if no such candle is found.
            """
            start_search = start_idx + max(1, startOBsearchFrom)
            end_search = receiver_idx
            for idx in range(start_search, end_search + 1):
                if idx not in candle_bounds:
                    continue
                left_neighbors = []
                right_neighbors = []
                if minOBleftneighbor in [0, 1]:
                    if idx - 1 in candle_bounds:
                        left_neighbors.append(idx - 1)
                elif minOBleftneighbor == 2:
                    if idx - 1 in candle_bounds and idx - 2 in candle_bounds:
                        left_neighbors.extend([idx - 1, idx - 2])
                if minOBrightneighbor in [0, 1]:
                    if idx + 1 in candle_bounds:
                        right_neighbors.append(idx + 1)
                elif minOBrightneighbor == 2:
                    if idx + 1 in candle_bounds and idx + 2 in candle_bounds:
                        right_neighbors.extend([idx + 1, idx + 2])
                if len(left_neighbors) < max(1, minOBleftneighbor) or len(right_neighbors) < max(1, minOBrightneighbor):
                    continue
                current_candle = candle_bounds[idx]
                all_neighbors_valid = True
                if is_ph:
                    for neighbor_idx in left_neighbors + right_neighbors:
                        neighbor_candle = candle_bounds[neighbor_idx]
                        if current_candle["high"] <= neighbor_candle["high"]:
                            all_neighbors_valid = False
                            break
                    if all_neighbors_valid:
                        x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                        y = current_candle["high_y"]
                        return idx, x, y
                else:
                    for neighbor_idx in left_neighbors + right_neighbors:
                        neighbor_candle = candle_bounds[neighbor_idx]
                        if current_candle["low"] >= neighbor_candle["low"]:
                            all_neighbors_valid = False
                            break
                    if all_neighbors_valid:
                        x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                        y = current_candle["low_y"]
                        return idx, x, y
            return None

        def find_oi_candle(reversal_idx, ob_idx, is_ph):
            """
            Find the first candle after reversal_idx where:
            - For PH team: low is lower than the high of the OB candle at ob_idx.
            - For PL team: high is higher than the low of the OB candle at ob_idx.
            Returns (index, x, y) or None if no such candle is found.
            """
            if ob_idx is None or ob_idx not in candle_bounds or reversal_idx is None or reversal_idx not in candle_bounds:
                return None
            reference_candle = candle_bounds[ob_idx]
            reference_price = reference_candle["high"] if is_ph else reference_candle["low"]
            for idx in range(reversal_idx - 1, -1, -1):
                if idx not in candle_bounds:
                    continue
                current_candle = candle_bounds[idx]
                if is_ph:
                    if current_candle["low"] < reference_price:
                        x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                        y = current_candle["low_y"] + 10
                        return idx, x, y
                else:
                    if current_candle["high"] > reference_price:
                        x = current_candle["x_left"] + (current_candle["x_right"] - current_candle["x_left"]) // 2
                        y = current_candle["low_y"] + 10
                        return idx, x, y
            return None

        ph_teams = []
        pl_teams = []
        ph_additional_trendlines = []
        pl_additional_trendlines = []
        sorted_ph = sorted(ph_indices.items(), key=lambda x: x[0], reverse=True)
        sorted_pl = sorted(pl_indices.items(), key=lambda x: x[0], reverse=True)

        # Process PH-to-PH trendlines
        i = 0
        while i < len(sorted_ph) - 1:
            sender_idx, sender_data = sorted_ph[i]
            sender_high = float(sender_data["high"])
            if i + 1 < len(sorted_ph):
                next_idx, next_data = sorted_ph[i + 1]
                next_high = float(next_data["high"])
                if next_high > sender_high:
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "skipped",
                        "reason": f"Immediate intruder PH found at candle {next_idx} (high {next_high}) higher than sender high {sender_high} (candle {sender_idx}), setting intruder as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
                    continue
            best_receiver_idx = None
            best_receiver_high = float('-inf')
            j = i + 1
            while j < len(sorted_ph):
                candidate_idx, candidate_data = sorted_ph[j]
                candidate_high = float(candidate_data["high"])
                if sender_high > candidate_high > best_receiver_high:
                    best_receiver_idx = candidate_idx
                    best_receiver_high = candidate_high
                j += 1
            if best_receiver_idx is not None:
                intruder_idx = find_intruder(sender_idx, best_receiver_idx, sender_high, is_ph=True)
                if intruder_idx is not None:
                    intruder_high = candle_bounds[intruder_idx]["high"]
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "skipped",
                        "reason": f"Intruder candle found at candle {intruder_idx} (high {intruder_high}) higher than sender high {sender_high} (candle {sender_idx}) between sender and receiver (candle {best_receiver_idx}), setting receiver as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i = next((k for k, (idx, _) in enumerate(sorted_ph) if idx == best_receiver_idx), i + 1)
                    continue
                sender_pos = contour_positions.get(sender_idx)
                receiver_pos = contour_positions.get(best_receiver_idx)
                if sender_pos and receiver_pos:
                    sender_x = sender_pos["x"]
                    sender_y = sender_pos["y"]
                    receiver_x = receiver_pos["x"]
                    receiver_y = receiver_pos["y"]
                    intersectors, slope = find_intersectors(sender_idx, best_receiver_idx, sender_x, sender_y, is_ph=True, receiver_price=best_receiver_high)
                    team_data = {
                        "sender": {"candle_number": sender_idx, "high": sender_high, "x": sender_x, "y": sender_y},
                        "receiver": {"candle_number": best_receiver_idx, "high": best_receiver_high, "x": receiver_x, "y": receiver_y},
                        "intersectors": [],
                        "trendlines": []
                    }
                    reason = ""
                    selected_idx = best_receiver_idx
                    selected_x = receiver_x
                    selected_y = receiver_y
                    selected_high = best_receiver_high
                    selected_type = "receiver"
                    selected_is_first = False
                    selected_marker = None
                    min_high = best_receiver_high
                    for idx, x, y, price, is_first, high_y, low_y, is_trendbreaker in intersectors:
                        marker = "star" if is_first else "circle"
                        team_data["intersectors"].append({
                            "candle_number": idx,
                            "high": price if is_first else None,
                            "x": x,
                            "y": high_y,
                            "is_first": is_first,
                            "is_trendbreaker": is_trendbreaker,
                            "marker": marker
                        })
                        if is_trendbreaker:
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, high {price if is_first else candle_bounds[idx]['high']}, x={x}, y={high_y}, marker={marker}, trendbreaker=True), skipped trendline generation"
                        else:
                            current_high = price if is_first else candle_bounds[idx]["high"]
                            if current_high < min_high:
                                min_high = current_high
                                selected_idx = idx
                                selected_x = x
                                selected_y = high_y
                                selected_high = current_high
                                selected_type = "intersector"
                                selected_is_first = is_first
                                selected_marker = "star" if is_first else "circle"
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, high {price if is_first else candle_bounds[idx]['high']}, x={x}, y={high_y}, marker={marker}, trendbreaker=False)"
                    if selected_type == "receiver" and not intersectors:
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                    next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                    first_breakout_idx = check_idx
                                    break
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(255, 0, 0), thickness=1)
                        team_data["trendlines"].append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y
                        })
                        ph_additional_trendlines.append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y
                        })
                        reason = f"Drew PH-to-PH trendline from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) to receiver (candle {selected_idx}, high {selected_high}, x={end_x}, y={end_y}) as no intersectors found" + reason
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                        next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["high_y"] + 10
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'up', color=(255, 0, 0))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (high_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for blue chevron arrow at center x-axis with offset from high"
                                        ob_result = find_OB(selected_idx, best_receiver_idx, is_ph=True)
                                        if ob_result:
                                            ob_idx, ob_x, ob_y = ob_result
                                            reversal_result = find_reversal_candle(target_idx, is_ph=True)
                                            oi_result = None
                                            if reversal_result:
                                                reversal_idx, _, _ = reversal_result
                                                oi_result = find_oi_candle(reversal_idx, ob_idx, is_ph=True)
                                            if oi_result:
                                                oi_idx, oi_x, oi_y = oi_result
                                                draw_right_arrow(img, ob_x, ob_y, oi_x=oi_x)
                                                draw_oi_marker(img, oi_x, oi_y)
                                                reason += f"; for PH intersector at candle {selected_idx}, found OB candle {ob_idx} with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with red right arrow at x={ob_x}, y={ob_y} extended to 'oi' candle {oi_idx} at x={oi_x}"
                                                reason += f"; for PH team at candle {selected_idx}, found 'oi' candle {oi_idx} with low ({candle_bounds[oi_idx]['low']}) lower than high ({candle_bounds[ob_idx]['high']}) of OB candle {ob_idx} after reversal candle {reversal_idx}, marked with green circle (radius=7) at x={oi_x}, y={oi_y} below candle"
                                            else:
                                                draw_right_arrow(img, ob_x, ob_y)
                                                ob_candle = next((c for c in candle_data if int(c["candle_number"]) == ob_idx), None)
                                                ob_timestamp = ob_candle["time"] if ob_candle else datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00')
                                                ob_none_oi_data.append({
                                                    f"team{team_counter}": {
                                                        "timestamp": ob_timestamp,
                                                        "team_type": "PH-to-PH",
                                                        "none_oi_x_OB_high_price": candle_bounds[ob_idx]["high"],
                                                        "none_oi_x_OB_low_price": candle_bounds[ob_idx]["low"]
                                                    }
                                                })
                                                team_counter += 1
                                                reason += f"; for PH intersector at candle {selected_idx}, found OB candle {ob_idx} with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with green right arrow at x={ob_x}, y={ob_y} extended to right edge of image"
                                                reason += f"; for PH team at candle {selected_idx}, no 'oi' candle found with low lower than high ({candle_bounds[ob_idx]['high']}) of OB candle {ob_idx} after reversal candle {reversal_idx if reversal_result else 'None'}"
                                        else:
                                            reason += f"; for PH intersector at candle {selected_idx}, no OB candle found with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}"
                                        reversal_result = find_reversal_candle(target_idx, is_ph=True)
                                        if reversal_result:
                                            reversal_idx, reversal_x, reversal_y = reversal_result
                                            draw_chevron_arrow(img, reversal_x, reversal_y, 'down', color=(0, 128, 0))
                                            reason += f"; for PH team at candle {selected_idx}, found reversal candle {reversal_idx} with high ({candle_bounds[reversal_idx]['high']}) higher than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}, marked with dim green downward chevron arrow at x={reversal_x}, y={reversal_y} above candle"
                                        else:
                                            reason += f"; for PH team at candle {selected_idx}, no reversal candle found with high higher than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}"
                                        break
                    elif selected_type == "intersector":
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                    next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                    first_breakout_idx = check_idx
                                    break
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(255, 0, 0), thickness=1)
                        if selected_is_first:
                            star_points = [
                                [selected_x, selected_y - 15], [selected_x + 4, selected_y - 5], [selected_x + 14, selected_y - 5],
                                [selected_x + 5, selected_y + 2], [selected_x + 10, selected_y + 12], [selected_x, selected_y + 7],
                                [selected_x - 10, selected_y + 12], [selected_x - 5, selected_y + 2], [selected_x - 14, selected_y - 5],
                                [selected_x - 4, selected_y - 5]
                            ]
                            cv2.fillPoly(img, [np.array(star_points)], color=(255, 0, 0))
                        else:
                            cv2.circle(img, (selected_x, selected_y), 5, color=(255, 0, 0), thickness=-1)
                        team_data["trendlines"].append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        ph_additional_trendlines.append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        reason = f"Drew PH-to-PH trendline from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) to intersector (candle {selected_idx}, high {selected_high}, x={end_x}, y={end_y}, marker={selected_marker}, trendbreaker=False) with lowest high" + reason
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                        next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["high_y"] + 10
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'up', color=(255, 0, 0))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (high_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for blue chevron arrow at center x-axis with offset from high"
                                        ob_result = find_OB(selected_idx, best_receiver_idx, is_ph=True)
                                        if ob_result:
                                            ob_idx, ob_x, ob_y = ob_result
                                            reversal_result = find_reversal_candle(target_idx, is_ph=True)
                                            oi_result = None
                                            if reversal_result:
                                                reversal_idx, _, _ = reversal_result
                                                oi_result = find_oi_candle(reversal_idx, ob_idx, is_ph=True)
                                            if oi_result:
                                                oi_idx, oi_x, oi_y = oi_result
                                                draw_right_arrow(img, ob_x, ob_y, oi_x=oi_x)
                                                draw_oi_marker(img, oi_x, oi_y)
                                                reason += f"; for PH intersector at candle {selected_idx}, found OB candle {ob_idx} with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with red right arrow at x={ob_x}, y={ob_y} extended to 'oi' candle {oi_idx} at x={oi_x}"
                                                reason += f"; for PH team at candle {selected_idx}, found 'oi' candle {oi_idx} with low ({candle_bounds[oi_idx]['low']}) lower than high ({candle_bounds[ob_idx]['high']}) of OB candle {ob_idx} after reversal candle {reversal_idx}, marked with green circle (radius=7) at x={oi_x}, y={oi_y} below candle"
                                            else:
                                                draw_right_arrow(img, ob_x, ob_y)
                                                ob_candle = next((c for c in candle_data if int(c["candle_number"]) == ob_idx), None)
                                                ob_timestamp = ob_candle["time"] if ob_candle else datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00')
                                                ob_none_oi_data.append({
                                                    f"team{team_counter}": {
                                                        "timestamp": ob_timestamp,
                                                        "team_type": "PH-to-PH",
                                                        "none_oi_x_OB_high_price": candle_bounds[ob_idx]["high"],
                                                        "none_oi_x_OB_low_price": candle_bounds[ob_idx]["low"]
                                                    }
                                                })
                                                team_counter += 1
                                                reason += f"; for PH intersector at candle {selected_idx}, found OB candle {ob_idx} with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with green right arrow at x={ob_x}, y={ob_y} extended to right edge of image"
                                                reason += f"; for PH team at candle {selected_idx}, no 'oi' candle found with low lower than high ({candle_bounds[ob_idx]['high']}) of OB candle {ob_idx} after reversal candle {reversal_idx if reversal_result else 'None'}"
                                        else:
                                            reason += f"; for PH intersector at candle {selected_idx}, no OB candle found with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}"
                                        reversal_result = find_reversal_candle(target_idx, is_ph=True)
                                        if reversal_result:
                                            reversal_idx, reversal_x, reversal_y = reversal_result
                                            draw_chevron_arrow(img, reversal_x, reversal_y, 'down', color=(0, 128, 0))
                                            reason += f"; for PH team at candle {selected_idx}, found reversal candle {reversal_idx} with high ({candle_bounds[reversal_idx]['high']}) higher than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}, marked with dim green downward chevron arrow at x={reversal_x}, y={reversal_y} above candle"
                                        else:
                                            reason += f"; for PH team at candle {selected_idx}, no reversal candle found with high higher than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}"
                                        break
                    else:
                        reason = f"No PH-to-PH trendline drawn from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) as first intersector is trendbreaker" + reason
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "success" if team_data["trendlines"] else "skipped",
                        "reason": reason,
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    ph_teams.append(team_data)
                    i = next((k for k, (idx, _) in enumerate(sorted_ph) if idx == best_receiver_idx), i + 1) + 1
                else:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Missing contour positions for PH sender (candle {sender_idx}) or receiver (candle {best_receiver_idx})",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
            else:
                trendline_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "symbol": symbol,
                    "timeframe": timeframe_str,
                    "team_type": "PH-to-PH",
                    "status": "skipped",
                    "reason": f"No valid PH receiver found for sender high {sender_high} (candle {sender_idx})",
                    "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                })
                i += 1

        # Process PL-to-PL trendlines
        i = 0
        while i < len(sorted_pl) - 1:
            sender_idx, sender_data = sorted_pl[i]
            sender_low = float(sender_data["low"])
            if i + 1 < len(sorted_pl):
                next_idx, next_data = sorted_pl[i + 1]
                next_low = float(next_data["low"])
                if next_low < sender_low:
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "skipped",
                        "reason": f"Immediate intruder PL found at candle {next_idx} (low {next_low}) lower than sender low {sender_low} (candle {sender_idx}), setting intruder as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
                    continue
            best_receiver_idx = None
            best_receiver_low = float('inf')
            j = i + 1
            while j < len(sorted_pl):
                candidate_idx, candidate_data = sorted_pl[j]
                candidate_low = float(candidate_data["low"])
                if sender_low < candidate_low < best_receiver_low:
                    best_receiver_idx = candidate_idx
                    best_receiver_low = candidate_low
                j += 1
            if best_receiver_idx is not None:
                intruder_idx = find_intruder(sender_idx, best_receiver_idx, sender_low, is_ph=False)
                if intruder_idx is not None:
                    intruder_low = candle_bounds[intruder_idx]["low"]
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "skipped",
                        "reason": f"Intruder candle found at candle {intruder_idx} (low {intruder_low}) lower than sender low {sender_low} (candle {sender_idx}) between sender and receiver (candle {best_receiver_idx}), setting receiver as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i = next((k for k, (idx, _) in enumerate(sorted_pl) if idx == best_receiver_idx), i + 1)
                    continue
                sender_pos = contour_positions.get(sender_idx)
                receiver_pos = contour_positions.get(best_receiver_idx)
                if sender_pos and receiver_pos:
                    sender_x = sender_pos["x"]
                    sender_y = sender_pos["y"] + sender_pos["height"]
                    receiver_x = receiver_pos["x"]
                    receiver_y = receiver_pos["y"] + receiver_pos["height"]
                    intersectors, slope = find_intersectors(sender_idx, best_receiver_idx, sender_x, sender_y, is_ph=False, receiver_price=best_receiver_low)
                    team_data = {
                        "sender": {"candle_number": sender_idx, "low": sender_low, "x": sender_x, "y": sender_y},
                        "receiver": {"candle_number": best_receiver_idx, "low": best_receiver_low, "x": receiver_x, "y": receiver_y},
                        "intersectors": [],
                        "trendlines": []
                    }
                    reason = ""
                    selected_idx = best_receiver_idx
                    selected_x = receiver_x
                    selected_y = receiver_y
                    selected_low = best_receiver_low
                    selected_type = "receiver"
                    selected_is_first = False
                    selected_marker = None
                    max_low = best_receiver_low
                    for idx, x, y, price, is_first, high_y, low_y, is_trendbreaker in intersectors:
                        marker = "star" if is_first else "circle"
                        team_data["intersectors"].append({
                            "candle_number": idx,
                            "low": price if is_first else None,
                            "x": x,
                            "y": low_y,
                            "is_first": is_first,
                            "is_trendbreaker": is_trendbreaker,
                            "marker": marker
                        })
                        if is_trendbreaker:
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, low {price if is_first else candle_bounds[idx]['low']}, x={x}, y={low_y}, marker={marker}, trendbreaker=True), skipped trendline generation"
                        else:
                            current_low = price if is_first else candle_bounds[idx]["low"]
                            if current_low > max_low:
                                max_low = current_low
                                selected_idx = idx
                                selected_x = x
                                selected_y = low_y
                                selected_low = current_low
                                selected_type = "intersector"
                                selected_is_first = is_first
                                selected_marker = "star" if is_first else "circle"
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, low {price if is_first else candle_bounds[idx]['low']}, x={x}, y={low_y}, marker={marker}, trendbreaker=False)"
                    if selected_type == "receiver" and not intersectors:
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                    next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                    first_breakout_idx = check_idx
                                    break
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(0, 255, 255), thickness=1)
                        team_data["trendlines"].append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y
                        })
                        pl_additional_trendlines.append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y
                        })
                        reason = f"Drew PL-to-PL trendline from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) to receiver (candle {selected_idx}, low={selected_low}, x={end_x}, y={end_y}) as no intersectors found" + reason
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                        next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["low_y"] - 10
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'down', color=(128, 0, 128))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (low_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for purple chevron arrow at center x-axis with offset from low"
                                        ob_result = find_OB(selected_idx, best_receiver_idx, is_ph=False)
                                        if ob_result:
                                            ob_idx, ob_x, ob_y = ob_result
                                            reversal_result = find_reversal_candle(target_idx, is_ph=False)
                                            oi_result = None
                                            if reversal_result:
                                                reversal_idx, _, _ = reversal_result
                                                oi_result = find_oi_candle(reversal_idx, ob_idx, is_ph=False)
                                            if oi_result:
                                                oi_idx, oi_x, oi_y = oi_result
                                                draw_right_arrow(img, ob_x, ob_y, oi_x=oi_x)
                                                draw_oi_marker(img, oi_x, oi_y)
                                                reason += f"; for PL intersector at candle {selected_idx}, found OB candle {ob_idx} with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with red right arrow at x={ob_x}, y={ob_y} extended to 'oi' candle {oi_idx} at x={oi_x}"
                                                reason += f"; for PL team at candle {selected_idx}, found 'oi' candle {oi_idx} with high ({candle_bounds[oi_idx]['high']}) higher than low ({candle_bounds[ob_idx]['low']}) of OB candle {ob_idx} after reversal candle {reversal_idx}, marked with green circle (radius=7) at x={oi_x}, y={oi_y} below candle"
                                            else:
                                                draw_right_arrow(img, ob_x, ob_y)
                                                ob_candle = next((c for c in candle_data if int(c["candle_number"]) == ob_idx), None)
                                                ob_timestamp = ob_candle["time"] if ob_candle else datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00')
                                                ob_none_oi_data.append({
                                                    f"team{team_counter}": {
                                                        "timestamp": ob_timestamp,
                                                        "team_type": "PL-to-PL",
                                                        "none_oi_x_OB_high_price": candle_bounds[ob_idx]["high"],
                                                        "none_oi_x_OB_low_price": candle_bounds[ob_idx]["low"]
                                                    }
                                                })
                                                team_counter += 1
                                                reason += f"; for PL intersector at candle {selected_idx}, found OB candle {ob_idx} with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with green right arrow at x={ob_x}, y={ob_y} extended to right edge of image"
                                                reason += f"; for PL team at candle {selected_idx}, no 'oi' candle found with high higher than low ({candle_bounds[ob_idx]['low']}) of OB candle {ob_idx} after reversal candle {reversal_idx if reversal_result else 'None'}"
                                        else:
                                            reason += f"; for PL intersector at candle {selected_idx}, no OB candle found with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}"
                                        reversal_result = find_reversal_candle(target_idx, is_ph=False)
                                        if reversal_result:
                                            reversal_idx, reversal_x, reversal_y = reversal_result
                                            draw_chevron_arrow(img, reversal_x, reversal_y, 'up', color=(0, 0, 255))
                                            reason += f"; for PL team at candle {selected_idx}, found reversal candle {reversal_idx} with low ({candle_bounds[reversal_idx]['low']}) lower than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}, marked with red upward chevron arrow at x={reversal_x}, y={reversal_y} below candle"
                                        else:
                                            reason += f"; for PL team at candle {selected_idx}, no reversal candle found with low lower than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}"
                                        break
                    elif selected_type == "intersector":
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                    next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                    first_breakout_idx = check_idx
                                    break
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(0, 255, 255), thickness=1)
                        if selected_is_first:
                            star_points = [
                                [selected_x, selected_y - 15], [selected_x + 4, selected_y - 5], [selected_x + 14, selected_y - 5],
                                [selected_x + 5, selected_y + 2], [selected_x + 10, selected_y + 12], [selected_x, selected_y + 7],
                                [selected_x - 10, selected_y + 12], [selected_x - 5, selected_y + 2], [selected_x - 14, selected_y - 5],
                                [selected_x - 4, selected_y - 5]
                            ]
                            cv2.fillPoly(img, [np.array(star_points)], color=(0, 255, 255))
                        else:
                            cv2.circle(img, (selected_x, selected_y), 5, color=(0, 255, 255), thickness=-1)
                        team_data["trendlines"].append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        pl_additional_trendlines.append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        reason = f"Drew PL-to-PL trendline from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) to intersector (candle {selected_idx}, low={selected_low}, x={end_x}, y={end_y}, marker={selected_marker}, trendbreaker=False) with highest low" + reason
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                        next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["low_y"] - 10
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'down', color=(128, 0, 128))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (low_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for purple chevron arrow at center x-axis with offset from low"
                                        ob_result = find_OB(selected_idx, best_receiver_idx, is_ph=False)
                                        if ob_result:
                                            ob_idx, ob_x, ob_y = ob_result
                                            reversal_result = find_reversal_candle(target_idx, is_ph=False)
                                            oi_result = None
                                            if reversal_result:
                                                reversal_idx, _, _ = reversal_result
                                                oi_result = find_oi_candle(reversal_idx, ob_idx, is_ph=False)
                                            if oi_result:
                                                oi_idx, oi_x, oi_y = oi_result
                                                draw_right_arrow(img, ob_x, ob_y, oi_x=oi_x)
                                                draw_oi_marker(img, oi_x, oi_y)
                                                reason += f"; for PL intersector at candle {selected_idx}, found OB candle {ob_idx} with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with red right arrow at x={ob_x}, y={ob_y} extended to 'oi' candle {oi_idx} at x={oi_x}"
                                                reason += f"; for PL team at candle {selected_idx}, found 'oi' candle {oi_idx} with high ({candle_bounds[oi_idx]['high']}) higher than low ({candle_bounds[ob_idx]['low']}) of OB candle {ob_idx} after reversal candle {reversal_idx}, marked with green circle (radius=7) at x={oi_x}, y={oi_y} below candle"
                                            else:
                                                draw_right_arrow(img, ob_x, ob_y)
                                                ob_candle = next((c for c in candle_data if int(c["candle_number"]) == ob_idx), None)
                                                ob_timestamp = ob_candle["time"] if ob_candle else datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00')
                                                ob_none_oi_data.append({
                                                    f"team{team_counter}": {
                                                        "timestamp": ob_timestamp,
                                                        "team_type": "PL-to-PL",
                                                        "none_oi_x_OB_high_price": candle_bounds[ob_idx]["high"],
                                                        "none_oi_x_OB_low_price": candle_bounds[ob_idx]["low"]
                                                    }
                                                })
                                                team_counter += 1
                                                reason += f"; for PL intersector at candle {selected_idx}, found OB candle {ob_idx} with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}, marked with green right arrow at x={ob_x}, y={ob_y} extended to right edge of image"
                                                reason += f"; for PL team at candle {selected_idx}, no 'oi' candle found with high higher than low ({candle_bounds[ob_idx]['low']}) of OB candle {ob_idx} after reversal candle {reversal_idx if reversal_result else 'None'}"
                                        else:
                                            reason += f"; for PL intersector at candle {selected_idx}, no OB candle found with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {selected_idx + max(1, startOBsearchFrom)} to receiver {best_receiver_idx}"
                                        reversal_result = find_reversal_candle(target_idx, is_ph=False)
                                        if reversal_result:
                                            reversal_idx, reversal_x, reversal_y = reversal_result
                                            draw_chevron_arrow(img, reversal_x, reversal_y, 'up', color=(0, 0, 255))
                                            reason += f"; for PL team at candle {selected_idx}, found reversal candle {reversal_idx} with low ({candle_bounds[reversal_idx]['low']}) lower than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}, marked with red upward chevron arrow at x={reversal_x}, y={reversal_y} below candle"
                                        else:
                                            reason += f"; for PL team at candle {selected_idx}, no reversal candle found with low lower than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors after target candle {target_idx}"
                                        break
                    else:
                        reason = f"No PL-to-PL trendline drawn from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) as first intersector is trendbreaker" + reason
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "success" if team_data["trendlines"] else "skipped",
                        "reason": reason,
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    pl_teams.append(team_data)
                    i = next((k for k, (idx, _) in enumerate(sorted_pl) if idx == best_receiver_idx), i + 1) + 1
                else:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Missing contour positions for PL sender (candle {sender_idx}) or receiver (candle {best_receiver_idx})",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
            else:
                trendline_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "symbol": symbol,
                    "timeframe": timeframe_str,
                    "team_type": "PL-to-PL",
                    "status": "skipped",
                    "reason": f"No valid PL receiver found for sender low {sender_low} (candle {sender_idx})",
                    "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                })
                i += 1

        cv2.imwrite(output_image_path, img)
        log_and_print(
            f"Chart with colored contours (dim green for up, red for down), "
            f"PH/PL markers (blue for PH, purple for PL), "
            f"single trendlines (blue for PH-to-PH to lowest high intersector or receiver extended to first breakout candle, yellow for PL-to-PL to highest low intersector or receiver extended to first breakout candle), "
            f"intersector markers (blue star/circle for PH selected intersector, yellow star/circle for PL selected intersector), "
            f"blue upward chevron arrow on first PH breakout candle after skipping {minbreakoutcandleposition} candles from initial breakout with higher low and higher high for last PH trendline at center x-axis with offset from high, "
            f"purple downward chevron arrow on first PL breakout candle after skipping {minbreakoutcandleposition} candles from initial breakout with lower high and lower low for last PL trendline at center x-axis with offset from low, "
            f"red right arrow for PH intersector on first candle with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {startOBsearchFrom} behind intersector to receiver at high price, extended to 'oi' candle body if found, "
            f"green right arrow for PH intersector on first candle with high higher than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {startOBsearchFrom} behind intersector to receiver at high price, extended to right edge of image if no 'oi' found, "
            f"red right arrow for PL intersector on first candle with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {startOBsearchFrom} behind intersector to receiver at low price, extended to 'oi' candle body if found, "
            f"green right arrow for PL intersector on first candle with low lower than {minOBleftneighbor} left and {minOBrightneighbor} right neighbors from candle {startOBsearchFrom} behind intersector to receiver at low price, extended to right edge of image if no 'oi' found, "
            f"green circle (radius=7) for PH team 'oi' candle with low lower than high of OB candle, placed below candle, "
            f"green circle (radius=7) for PL team 'oi' candle with high higher than low of OB candle, placed below candle, "
            f"dim green downward chevron arrow for PH team reversal candle with high higher than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors, placed above candle, "
            f"red upward chevron arrow for PL team reversal candle with low lower than {reversal_leftcandle} left and {reversal_rightcandle} right neighbors, placed below candle, "
            f"saved for {symbol} ({timeframe_str}) at {output_image_path}",
            "SUCCESS"
        )

        contour_data = {
            "total_count": total_count,
            "green_candle_count": green_count,
            "red_candle_count": red_count,
            "candle_contours": [],
            "ph_teams": ph_teams,
            "pl_teams": pl_teams,
            "ph_additional_trendlines": ph_additional_trendlines,
            "pl_additional_trendlines": pl_additional_trendlines
        }
        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            candle_type = "green" if green_mask[y + h // 2, x + w // 2] > 0 else "red" if red_mask[y + h // 2, x + w // 2] > 0 else "unknown"
            contour_data["candle_contours"].append({
                "candle_number": i,
                "type": candle_type,
                "x": x + w // 2,
                "y": y,
                "width": w,
                "height": h,
                "is_ph": i in ph_indices,
                "is_pl": i in pl_indices
            })
        try:
            with open(contour_json_path, 'w') as f:
                json.dump(contour_data, f, indent=4)
            log_and_print(
                f"Contour count and trendline data saved for {symbol} ({timeframe_str}) at {contour_json_path} "
                f"with total_count={total_count} (green={green_count}, red={red_count}, PH={len(ph_indices)}, PL={len(pl_indices)}, "
                f"PH_teams={len(ph_teams)}, PL_teams={len(pl_teams)}, "
                f"PH_trendlines={len(ph_additional_trendlines)}, PL_trendlines={len(pl_additional_trendlines)})",
                "SUCCESS"
            )
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save contour count for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to save contour count for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        try:
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(
                f"Trendline log saved for {symbol} ({timeframe_str}) at {trendline_log_json_path} "
                f"with {len(trendline_log)} entries",
                "SUCCESS"
            )
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        try:
            with open(ob_none_oi_json_path, 'w') as f:
                json.dump(ob_none_oi_data, f, indent=4)
            log_and_print(
                f"OB none oi_x data saved for {symbol} ({timeframe_str}) at {ob_none_oi_json_path} "
                f"with {len(ob_none_oi_data)} entries",
                "SUCCESS"
            )
        except Exception as e:
            error_log.append({
                                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save OB none oi_x data for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to save OB none oi_x data for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        return error_log

    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Unexpected error in detect_candle_contours for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Unexpected error in detect_candle_contours for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        return error_log


def collect_ob_none_oi_data(symbol, symbol_folder, broker_name, base_folder, all_symbols):
    """Collect and convert ob_none_oi_data.json from each timeframe for a symbol, save to market folder as alltimeframes_ob_none_oi_data.json,
    update allmarkets_limitorders.json, allnoordermarkets.json, and save to market-type-specific JSONs based on allsymbolsvolumesandrisk.json.
    If symbol not found directly, use symbolsmatch.json to map via broker-specific list to a main symbol and retrieve risk/volume."""

    error_log = []
    all_timeframes_data = {tf: [] for tf in TIMEFRAME_MAP.keys()}
    allmarkets_json_path = os.path.join(base_folder, "allmarkets_limitorders.json")
    allnoordermarkets_json_path = os.path.join(base_folder, "allnoordermarkets.json")

    # === CORRECTED PATHS ===
    allsymbols_json_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    symbols_match_json_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\symbolsmatch.json"  # Fixed name + path

    # Paths for market-type-specific JSONs (unchanged - these are in root)
    market_type_paths = {
        "forex": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\forexvolumesandrisk.json",
        "stocks": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\stocksvolumesandrisk.json",
        "indices": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\indicesvolumesandrisk.json",
        "synthetics": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\syntheticsvolumesandrisk.json",
        "commodities": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\commoditiesvolumesandrisk.json",
        "crypto": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\cryptovolumesandrisk.json",
        "equities": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\equitiesvolumesandrisk.json",
        "energies": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\energiesvolumesandrisk.json",
        "etfs": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\etfsvolumesandrisk.json",
        "basket_indices": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\basketindicesvolumesandrisk.json",
        "metals": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\metalsvolumesandrisk.json"
    }

    # Initialize or load existing allmarkets_limitorders.json
    if os.path.exists(allmarkets_json_path):
        try:
            with open(allmarkets_json_path, 'r') as f:
                allmarkets_data = json.load(f)
            markets_limitorders_count = allmarkets_data.get("markets_limitorders", 0)
            markets_nolimitorders_count = allmarkets_data.get("markets_nolimitorders", 0)
            limitorders = allmarkets_data.get("limitorders", {})
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load {allmarkets_json_path} for {broker_name}: {str(e)}",
                "broker": broker_name
            })
            log_and_print(f"Failed to load {allmarkets_json_path} for {broker_name}: {str(e)}", "ERROR")
            markets_limitorders_count = 0
            markets_nolimitorders_count = 0
            limitorders = {}
    else:
        markets_limitorders_count = 0
        markets_nolimitorders_count = 0
        limitorders = {}

    # Initialize or load existing allnoordermarkets.json
    if os.path.exists(allnoordermarkets_json_path):
        try:
            with open(allnoordermarkets_json_path, 'r') as f:
                allnoordermarkets_data = json.load(f)
            noorder_markets = allnoordermarkets_data.get("noorder_markets", [])
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load {allnoordermarkets_json_path} for {broker_name}: {str(e)}",
                "broker": broker_name
            })
            log_and_print(f"Failed to load {allnoordermarkets_json_path} for {broker_name}: {str(e)}", "ERROR")
            noorder_markets = []
    else:
        noorder_markets = []

    # === ENHANCED: FIND SYMBOL IN allsymbols OR VIA symbolsmatch.json ===
    market_type = None
    risk_volume_map = {}  # {risk_amount: volume}
    mapped_main_symbol = None

    def find_symbol_in_allsymbols(target_symbol):
        """Search allsymbolsvolumesandrisk.json for target_symbol and return market_type + risk_volume_map"""
        try:
            if not os.path.exists(allsymbols_json_path):
                log_and_print(f"allsymbolsvolumesandrisk.json not found at {allsymbols_json_path}", "ERROR")
                return None, {}

            with open(allsymbols_json_path, 'r') as f:
                allsymbols_data = json.load(f)

            for risk_key, markets in allsymbols_data.items():
                try:
                    risk_amount = float(risk_key.split(": ")[1])
                    for mkt_type in market_type_paths.keys():
                        for item in markets.get(mkt_type, []):
                            if item["symbol"] == target_symbol:
                                return mkt_type, {risk_amount: item["volume"]}
                except Exception as parse_err:
                    continue
            return None, {}
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to search allsymbols for {target_symbol}: {str(e)}",
                "broker": broker_name
            })
            return None, {}

    # Step 1: Try direct match
    market_type, risk_volume_map = find_symbol_in_allsymbols(symbol)
    if market_type:
        log_and_print(f"Direct match: Symbol {symbol} → market_type: {market_type}, risks: {sorted(risk_volume_map.keys())}", "INFO")
    else:
        # Step 2: Fallback to symbolsmatch.json
        log_and_print(f"Direct match failed for {symbol}. Trying symbolsmatch.json...", "INFO")
        try:
            if os.path.exists(symbols_match_json_path):
                with open(symbols_match_json_path, 'r') as f:
                    symbols_match_data = json.load(f).get("main_symbols", [])

                broker_key = broker_name.lower()
                if broker_key not in ["deriv", "bybit", "exness"]:
                    broker_key = "deriv"  # fallback

                for entry in symbols_match_data:
                    broker_symbols = entry.get(broker_key, [])
                    if symbol in broker_symbols:
                        mapped_main_symbol = entry["symbol"]
                        log_and_print(f"Mapped {symbol} ({broker_name}) → main symbol: {mapped_main_symbol}", "INFO")
                        market_type, risk_volume_map = find_symbol_in_allsymbols(mapped_main_symbol)
                        if market_type:
                            log_and_print(f"Using risk/volume from main symbol {mapped_main_symbol}: {sorted(risk_volume_map.keys())}", "INFO")
                        break
            else:
                error_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "error": f"symbolsmatch.json not found at {symbols_match_json_path}",
                    "broker": broker_name
                })
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to process symbolsmatch.json: {str(e)}",
                "broker": broker_name
            })

    if not market_type or not risk_volume_map:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Symbol {symbol} (mapped: {mapped_main_symbol}) not found in any risk level after fallback",
            "broker": broker_name
        })
        log_and_print(f"Symbol {symbol} not found in risk data even after mapping", "ERROR")
    else:
        log_and_print(f"Final → {symbol} (broker: {broker_name}) → market_type: {market_type}, risks: {sorted(risk_volume_map.keys())}", "INFO")

    # Process the current symbol's timeframes
    symbol_limitorders = {tf: [] for tf in TIMEFRAME_MAP.keys()}
    has_limit_orders = False
    market_type_orders = []

    # Get current bid price + symbol info
    current_bid_price = None
    tick_size = None
    tick_value = None
    try:
        config = brokersdictionary.get(broker_name)
        if not config:
            raise Exception(f"No configuration found for broker {broker_name}")
        success, init_errors = initialize_mt5(
            config["TERMINAL_PATH"],
            config["LOGIN_ID"],
            config["PASSWORD"],
            config["SERVER"]
        )
        error_log.extend(init_errors)
        if not success:
            raise Exception(f"MT5 initialization failed for {broker_name}")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Exception(f"Failed to retrieve tick data for {symbol}")
        current_bid_price = tick.bid

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise Exception(f"Failed to retrieve symbol info for {symbol}")
        tick_size = sym_info.point
        tick_value = sym_info.trade_tick_value

        log_and_print(f"Retrieved current bid price {current_bid_price} for {symbol} ({broker_name})", "INFO")
        log_and_print(f"Tick Size: {tick_size}, Tick Value: {tick_value}", "INFO")
        
        mt5.shutdown()
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve current bid price or symbol info for {symbol} ({broker_name}): {str(e)}",
            "broker": broker_name
        })
        log_and_print(f"Failed to retrieve current bid price or symbol info for {symbol} ({broker_name}): {str(e)}", "ERROR")
        current_bid_price = None
        tick_size = None
        tick_value = None

    try:
        for timeframe_str in TIMEFRAME_MAP.keys():
            timeframe_folder = os.path.join(symbol_folder, timeframe_str)
            ob_none_oi_json_path = os.path.join(timeframe_folder, "ob_none_oi_data.json")
            
            if os.path.exists(ob_none_oi_json_path):
                try:
                    with open(ob_none_oi_json_path, 'r') as f:
                        data = json.load(f)
                        converted_data = []
                        for item in data:
                            for team_key, team_data in item.items():
                                converted_team = {
                                    "timestamp": team_data["timestamp"],
                                    "limit_order": "",
                                    "entry_price": 0.0
                                }
                                if team_data["team_type"] == "PH-to-PH":
                                    converted_team["limit_order"] = "buy_limit"
                                    converted_team["entry_price"] = team_data["none_oi_x_OB_high_price"]
                                elif team_data["team_type"] == "PL-to-PL":
                                    converted_team["limit_order"] = "sell_limit"
                                    converted_team["entry_price"] = team_data["none_oi_x_OB_low_price"]
                                else:
                                    error_log.append({
                                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                                        "error": f"Unknown team_type '{team_data['team_type']}' in ob_none_oi_data for {symbol} ({timeframe_str})",
                                        "broker": broker_name
                                    })
                                    log_and_print(f"Unknown team_type '{team_data['team_type']}' in ob_none_oi_data for {symbol} ({timeframe_str})", "ERROR")
                                    continue
                                
                                # Filter based on current bid price
                                if current_bid_price is not None:
                                    if converted_team["limit_order"] == "sell_limit" and current_bid_price >= converted_team["entry_price"]:
                                        log_and_print(
                                            f"Skipped sell limit order for {symbol} ({timeframe_str}) at entry_price {converted_team['entry_price']} "
                                            f"as current bid price {current_bid_price} is >= entry price",
                                            "INFO"
                                        )
                                        continue
                                    if converted_team["limit_order"] == "buy_limit" and current_bid_price <= converted_team["entry_price"]:
                                        log_and_print(
                                            f"Skipped buy limit order for {symbol} ({timeframe_str}) at entry_price {converted_team['entry_price']} "
                                            f"as current bid price {current_bid_price} is <= entry price",
                                            "INFO"
                                        )
                                        continue
                                
                                converted_data.append({"team1": converted_team})

                                # === ADD ONE ORDER PER RISK LEVEL (using mapped or direct symbol) ===
                                if market_type and risk_volume_map:
                                    for risk_amount, volume in risk_volume_map.items():
                                        order_entry = {
                                            "market": symbol,  # Use broker's actual symbol
                                            "limit_order": converted_team["limit_order"],
                                            "timeframe": timeframe_str,
                                            "entry_price": converted_team["entry_price"],
                                            "volume": volume,
                                            "riskusd_amount": risk_amount,
                                            "tick_size": tick_size,
                                            "tick_value": tick_value,
                                            "broker": broker_name
                                        }
                                        market_type_orders.append(order_entry)
                        
                        all_timeframes_data[timeframe_str] = converted_data
                        if converted_data:
                            symbol_limitorders[timeframe_str] = converted_data
                            has_limit_orders = True
                    log_and_print(f"Collected and converted ob_none_oi_data for {symbol} ({timeframe_str}) from {ob_none_oi_json_path}", "INFO")
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Failed to read ob_none_oi_data.json for {symbol} ({timeframe_str}): {str(e)}",
                        "broker": broker_name
                    })
                    log_and_print(f"Failed to read ob_none_oi_data.json for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
            else:
                error_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "error": f"ob_none_oi_data.json not found for {symbol} ({timeframe_str}) at {ob_none_oi_json_path}",
                    "broker": broker_name
                })
                log_and_print(f"ob_none_oi_data.json not found for {symbol} ({timeframe_str})", "WARNING")

        # === SAVE alltimeframes_ob_none_oi_data.json ===
        output_json_path = os.path.join(symbol_folder, "alltimeframes_ob_none_oi_data.json")
        output_data = {"market": symbol, **all_timeframes_data}
        try:
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            log_and_print(f"Saved all timeframes ob_none_oi_data for {symbol} to {output_json_path}", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save alltimeframes_ob_none_oi_data.json for {symbol}: {str(e)}",
                "broker": broker_name
            })
            log_and_print(f"Failed to save alltimeframes_ob_none_oi_data.json for {symbol}: {str(e)}", "ERROR")

        # Update counts
        if has_limit_orders:
            markets_limitorders_count += 1
            limitorders[symbol] = symbol_limitorders
            if symbol in noorder_markets:
                noorder_markets.remove(symbol)
        else:
            markets_nolimitorders_count += 1
            if symbol not in noorder_markets:
                noorder_markets.append(symbol)

        # Save allmarkets_limitorders.json
        allmarkets_output_data = {
            "markets_limitorders": markets_limitorders_count,
            "markets_nolimitorders": markets_nolimitorders_count,
            "limitorders": limitorders
        }
        try:
            with open(allmarkets_json_path, 'w') as f:
                json.dump(allmarkets_output_data, f, indent=4)
            log_and_print(f"Updated allmarkets_limitorders.json", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save allmarkets_limitorders.json: {str(e)}",
                "broker": broker_name
            })
            log_and_print(f"Failed to save allmarkets_limitorders.json: {str(e)}", "ERROR")

        # Save allnoordermarkets.json
        allnoordermarkets_output_data = {
            "markets_nolimitorders": markets_nolimitorders_count,
            "noorder_markets": noorder_markets
        }
        try:
            with open(allnoordermarkets_json_path, 'w') as f:
                json.dump(allnoordermarkets_output_data, f, indent=4)
            log_and_print(f"Updated allnoordermarkets.json", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save allnoordermarkets.json: {str(e)}",
                "broker": broker_name
            })
            log_and_print(f"Failed to save allnoordermarkets.json: {str(e)}", "ERROR")

        # === SAVE TO MARKET-TYPE JSON (per broker, no deduplication) ===
        if market_type and market_type_orders:
            market_json_path = market_type_paths.get(market_type)
            if market_json_path:
                try:
                    existing_data = {}
                    if os.path.exists(market_json_path):
                        with open(market_json_path, 'r') as f:
                            existing_data = json.load(f)

                    if market_type == "forex":
                        if not existing_data or not isinstance(existing_data, dict):
                            existing_data = {
                                "xxxchf": [], "xxxjpy": [], "xxxnzd": [], "xxxusd": [],
                                "usdxxx": [], "xxxaud": [], "xxxcad": [], "other": []
                            }
                        symbol_lower = symbol.lower()
                        group = "other"
                        if symbol_lower.endswith('chf'): group = "xxxchf"
                        elif symbol_lower.endswith('jpy'): group = "xxxjpy"
                        elif symbol_lower.endswith('nzd'): group = "xxxnzd"
                        elif symbol_lower.endswith('usd'): group = "xxxusd"
                        elif symbol_lower.startswith('usd'): group = "usdxxx"
                        elif symbol_lower.endswith('aud'): group = "xxxaud"
                        elif symbol_lower.endswith('cad'): group = "xxxcad"
                        existing_data[group].extend(market_type_orders)
                    else:
                        if not isinstance(existing_data, list):
                            existing_data = []
                        existing_data.extend(market_type_orders)

                    with open(market_json_path, 'w') as f:
                        json.dump(existing_data, f, indent=4)
                    log_and_print(f"[{broker_name}] Saved {len(market_type_orders)} orders to {market_json_path}", "SUCCESS")
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Failed to save {market_json_path}: {str(e)}",
                        "broker": broker_name
                    })
                    log_and_print(f"Failed to save {market_json_path}: {str(e)}", "ERROR")
            else:
                log_and_print(f"No JSON path for market type {market_type}", "ERROR")
        else:
            log_and_print(f"No orders or market type for {symbol} ({broker_name})", "WARNING")

    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Unexpected error in collect_ob_none_oi_data for {symbol}: {str(e)}",
            "broker": broker_name
        })
        log_and_print(f"Unexpected error: {str(e)}", "ERROR")

    if error_log:
        save_errors(error_log)
    return error_log
  

def delete_all_category_jsons():
    """
    Delete (empty) every market-type JSON file that collect_ob_none_oi_data writes to.
    - Resets the files to an empty structure (list or dict) so that the next run
      starts from a clean slate.
    - Logs every action and collects any errors in the same format as the rest
      of the module.
    Returns the list of error dictionaries (empty if everything went fine).
    """
    error_log = []

    # ------------------------------------------------------------------ #
    # 1. Exact same paths you already use in collect_ob_none_oi_data
    # ------------------------------------------------------------------ #
    market_type_paths = {
        "forex": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\forexvolumesandrisk.json",
        "stocks": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\stocksvolumesandrisk.json",
        "indices": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\indicesvolumesandrisk.json",
        "synthetics": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\syntheticsvolumesandrisk.json",
        "commodities": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\commoditiesvolumesandrisk.json",
        "crypto": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\cryptovolumesandrisk.json",
        "equities": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\equitiesvolumesandrisk.json",
        "energies": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\energiesvolumesandrisk.json",
        "etfs": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\etfsvolumesandrisk.json",
        "basket_indices": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\basketindicesvolumesandrisk.json",
        "metals": r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\metalsvolumesandrisk.json"
    }

    # ------------------------------------------------------------------ #
    # 2. Helper: what an *empty* file should contain for each type
    # ------------------------------------------------------------------ #
    def empty_structure(mkt_type: str):
        """Return the correct empty JSON structure for a given market type."""
        if mkt_type == "forex":
            return {
                "xxxchf": [], "xxxjpy": [], "xxxnzd": [], "xxxusd": [],
                "usdxxx": [], "xxxaud": [], "xxxcad": [], "other": []
            }
        # All other categories are simple lists
        return []

    # ------------------------------------------------------------------ #
    # 3. Iterate over every file and wipe it
    # ------------------------------------------------------------------ #
    for mkt_type, json_path in market_type_paths.items():
        empty_data = empty_structure(mkt_type)
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f, indent=4)
            #log_and_print(f"[{mkt_type.upper()}] Emptied JSON → {json_path}","SUCCESS")
        except Exception as e:
            err = {
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos'))
                             .strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to empty {json_path}: {str(e)}",
                "broker": "N/A"          # no broker context here
            }
            error_log.append(err)
            log_and_print(
                f"Failed to empty {json_path}: {str(e)}",
                "ERROR"
            )

    # ------------------------------------------------------------------ #
    # 4. Persist any errors (same helper you already use elsewhere)
    # ------------------------------------------------------------------ #
    if error_log:
        save_errors(error_log)

    return error_log

def crop_chart(chart_path, symbol, timeframe_str, timeframe_folder):
    """Crop the saved chart.png and chartanalysed.png images, then detect candle contours only for chart.png."""
    error_log = []
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")

    try:
        # Crop chart.png
        with Image.open(chart_path) as img:
            right = 20
            left = 130
            top = 150
            bottom = 180
            crop_box = (left, top, img.width - right, img.height - bottom)
            cropped_img = img.crop(crop_box)
            cropped_img.save(chart_path, "PNG")
            log_and_print(f"Chart cropped for {symbol} ({timeframe_str}) at {chart_path}", "SUCCESS")

        # Detect contours for chart.png only
        contour_errors = detect_candle_contours(chart_path, symbol, timeframe_str, timeframe_folder)
        error_log.extend(contour_errors)

        # Crop chartanalysed.png if it exists
        if os.path.exists(chart_analysed_path):
            with Image.open(chart_analysed_path) as img:
                crop_box = (left, top, img.width - right, img.height - bottom)
                cropped_img = img.crop(crop_box)
                cropped_img.save(chart_analysed_path, "PNG")
                log_and_print(f"Analysed chart cropped for {symbol} ({timeframe_str}) at {chart_analysed_path}", "SUCCESS")
        else:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"chartanalysed.png not found for {symbol} ({timeframe_str})",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"chartanalysed.png not found for {symbol} ({timeframe_str})", "WARNING")

    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to crop charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to crop charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

    return error_log

def delete_all_calculated_risk_jsons():
    """Run the updateorders script for M5 timeframe."""
    try:
        calculateprices.delete_all_calculated_risk_jsons()
        print("symbols prices calculated ")
    except Exception as e:
        print(f"Error when calculating symbols prices: {e}")

def calculate_symbols_sl_tp_prices():
    """Run the updateorders script for M5 timeframe."""
    try:
        calculateprices.main()
        print("symbols prices calculated ")
    except Exception as e:
        print(f"Error when calculating symbols prices: {e}")

def delete_issue_jsons():
    """
    Deletes all issue / report JSON files for **ALL** brokers (real, demo, test, …)
    before `place_real_orders()` runs – guarantees a clean slate.

    Returns:
        dict: Summary of deleted files per broker (and global schedule)
    """
    
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
    REPORT_SUFFIX = "forex_order_report.json"
    ISSUES_FILE   = "ordersissues.json"

    # Add any other JSON files you want to wipe out here
    EXTRA_FILES_TO_DELETE = [
        # "debug_orders.json",
        # "temp_pending.json"
    ]

    deleted_summary = {}

    # --------------------------------------------------------------
    # 1. Walk through every broker folder (no account-type filter)
    # --------------------------------------------------------------
    base_path = Path(BASE_INPUT_DIR)
    if not base_path.exists():
        print(f"[CLEAN] Base directory not found: {base_path}")
        return deleted_summary

    for broker_dir in base_path.iterdir():
        if not broker_dir.is_dir():
            continue

        broker_name = broker_dir.name
        deleted_files = []
        risk_folders = [p.name for p in broker_dir.iterdir()
                        if p.is_dir() and p.name.startswith("risk_")]

        for risk_folder in risk_folders:
            risk_path = broker_dir / risk_folder

            for file_name in [ISSUES_FILE, REPORT_SUFFIX] + EXTRA_FILES_TO_DELETE:
                file_path = risk_path / file_name
                if file_path.exists():
                    try:
                        file_path.unlink()          # atomic delete
                        deleted_files.append(str(file_path))
                        print(f"[CLEAN] Deleted: {file_path}")
                    except Exception as e:
                        print(f"[CLEAN] Failed to delete {file_path}: {e}")

        deleted_summary[broker_name] = deleted_files or "No issue/report files found"

    # --------------------------------------------------------------
    # 2. Delete the global schedule file (if it exists)
    # --------------------------------------------------------------
    global_schedule = r"C:\xampp\htdocs\chronedge\synarex\fullordersschedules.json"
    if Path(global_schedule).exists():
        try:
            Path(global_schedule).unlink()
            print(f"[CLEAN] Deleted global: {global_schedule}")
            deleted_summary["global_schedule"] = global_schedule
        except Exception as e:
            print(f"[CLEAN] Failed to delete global schedule: {e}")

    # --------------------------------------------------------------
    # 3. Final summary
    # --------------------------------------------------------------
    total_deleted = sum(len(v) if isinstance(v, list) else 0
                        for v in deleted_summary.values())
    print(f"[CLEAN] Pre-order cleanup complete – {total_deleted} file(s) removed.")
    return deleted_summary

def backup_brokers_dictionary():
    main_path = Path(r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json")
    backup_path = Path(r"C:\xampp\htdocs\chronedge\synarex\brokersdictionarybackup.json")
    
    main_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Main file   : {main_path}")
    print(f"Backup file : {backup_path}")

    def read_json_safe(path: Path) -> dict | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # {} is considered EMPTY, not valid data
            if data == {}:
                return None
            return data
        except json.JSONDecodeError:
            return None

    def write_json(path: Path, data: dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Step 1: Check main file
    main_data = read_json_safe(main_path)
    
    if main_data is not None:
        # Main has real data → copy to backup (and make backup pretty)
        print("Main has valid data → syncing to backup")
        write_json(backup_path, main_data)
        print(f"Copied valid data: {main_path} → {backup_path}")
        return

    # Step 2: Main is empty {} or invalid → check backup
    print("Main is empty or invalid → checking backup")
    backup_data = read_json_safe(backup_path)

    if backup_data is not None:
        # Backup has real data → restore to main
        print("Backup has valid data → restoring to main")
        write_json(main_path, backup_data)
        print(f"Restored: {backup_path} → {main_path}")
        return

    # Step 3: Both are empty or invalid → create clean empty files
    print("Both files empty or corrupted → initializing clean empty state")
    empty_dict = {}
    write_json(main_path, empty_dict)
    write_json(backup_path, empty_dict)
    print("Created fresh empty brokersdictionary.json and backup") 

def placeallorders():
    """Run the updateorders script for M5 timeframe."""
    try:
        placeorders.main()
        print("all orders placed")
    except Exception as e:
        print(f"Error placing all orders: {e}")

def delete_news_sensitive_orders_during_news():
    """
    Deletes all open positions + pending orders on news-sensitive markets:
    - forex
    - basket_indices
    - energies (e.g. US Oil, UK Oil)
    - metals (XAUUSD, XAGUSD, etc.)
    from 20 minutes BEFORE until 1 minute AFTER the news time.
    Fully compatible with MT5 TradeOrder/TradePosition objects.
    """
    REQUIREMENTS_JSON = r"C:\xampp\htdocs\chronedge\synarex\requirements.json"
    
    # === NEWS-SENSITIVE CATEGORIES (updated) ===
    NEWS_SENSITIVE_CATEGORIES = {"forex", "basket_indices", "energies", "metals"}

    WINDOW_BEFORE_MINUTES = 20
    WINDOW_AFTER_MINUTES  = 1

    lagos_tz = pytz.timezone("Africa/Lagos")
    now_dt = datetime.now(lagos_tz)

    # === Load and parse news time ===
    try:
        with open(REQUIREMENTS_JSON, "r", encoding="utf-8") as f:
            req = json.load(f)
        news_raw = req.get("news", "").strip()
        if not news_raw:
            log_and_print("No news time set → skip protection", "INFO")
            return
    except Exception as e:
        log_and_print(f"Failed to read requirements.json: {e}", "WARNING")
        return

    try:
        news_dt = datetime.strptime(news_raw, "%Y-%m-%d %I:%M %p")
        news_dt = lagos_tz.localize(news_dt)
    except ValueError:
        try:
            # Fallback format if user uses 24-hour
            news_dt = datetime.strptime(news_raw, "%Y-%m-%d %H:%M")
            news_dt = lagos_tz.localize(news_dt)
        except:
            log_and_print(f"Invalid news format '{news_raw}' → use 'YYYY-MM-DD h:mm AM/PM' or 'YYYY-MM-DD HH:MM'", "ERROR")
            return

    time_to_news = (news_dt - now_dt).total_seconds()
    time_since_news = (now_dt - news_dt).total_seconds()

    if time_to_news > (WINDOW_BEFORE_MINUTES * 60):
        log_and_print(f"Too early → {time_to_news/60:.1f} min to news", "INFO")
        return
    if time_since_news > (WINDOW_AFTER_MINUTES * 60):
        log_and_print(f"News passed → protection off", "INFO")
        return

    log_and_print(f"NEWS PROTECTION ACTIVE | News: {news_raw} | "
                  f"Delta: {'-' if time_to_news < 0 else ''}{abs(time_to_news)/60:.1f} min", "CRITICAL")

    # === Load symbol → category mapping ===
    allsymbols_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    symbol_to_category = {}

    if os.path.exists(allsymbols_path):
        try:
            with open(allsymbols_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for risk_key, markets in data.items():
                if not isinstance(markets, dict):
                    continue
                for category, items in markets.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        symbol = item.get("symbol") or item.get("Symbol")
                        if symbol:
                            symbol_to_category[symbol.strip()] = category.lower()
        except Exception as e:
            log_and_print(f"Failed to load symbol categories: {e}", "ERROR")
    else:
        log_and_print(f"Symbol mapping file not found: {allsymbols_path}", "ERROR")

    # Add common Deriv-specific symbol aliases (very important!)
    extra_mappings = {
        "US Oil": "energies",
        "UK Oil": "energies",
        "Oil/USD": "energies",
        "XAUUSD": "metals",
        "XAGUSD": "metals",
        "Gold": "metals",
        "Silver": "metals",
    }
    symbol_to_category.update(extra_mappings)

    total_deleted = 0

    for broker_name, cfg in brokersdictionary.items():
        log_and_print(f"Connecting to {broker_name.upper()} for news protection...", "INFO")

        if not mt5.initialize(path=cfg["TERMINAL_PATH"], login=int(cfg["LOGIN_ID"]),
                              password=cfg["PASSWORD"], server=cfg["SERVER"], timeout=30000):
            log_and_print(f"{broker_name}: init failed", "ERROR")
            continue
        if not mt5.login(int(cfg["LOGIN_ID"]), cfg["PASSWORD"], cfg["SERVER"]):
            log_and_print(f"{broker_name}: login failed", "ERROR")
            mt5.shutdown()
            continue

        positions = mt5.positions_get() or []
        pending = mt5.orders_get() or []

        # Log positions
        if positions:
            log_and_print(f"{broker_name} — OPEN POSITIONS:", "INFO")
            for p in positions:
                cat = symbol_to_category.get(p.symbol, "unknown")
                sensitive = "NEWS-SENSITIVE" if cat in NEWS_SENSITIVE_CATEGORIES else "safe"
                log_and_print(f"  → {p.symbol} | Vol:{p.volume:.2f} | {'BUY' if p.type==0 else 'SELL'} | "
                              f"Entry:{p.price_open:.5f} | Ticket:{p.ticket} | Cat:{cat} [{sensitive}]", "INFO")

        # Log pending orders
        if pending:
            log_and_print(f"{broker_name} — PENDING ORDERS:", "INFO")
            order_types = ["BUY_LIMIT","SELL_LIMIT","BUY_STOP","SELL_STOP","BUY_STOP_LIMIT","SELL_STOP_LIMIT"]
            for o in pending:
                if o.type > 5: continue
                cat = symbol_to_category.get(o.symbol, "unknown")
                sensitive = "NEWS-SENSITIVE" if cat in NEWS_SENSITIVE_CATEGORIES else "safe"
                log_and_print(f"  → {o.symbol} | Vol:{o.volume_current:.2f} | {order_types[o.type]} | "
                              f"Price:{o.price_open:.5f} | Ticket:{o.ticket} | Cat:{cat} [{sensitive}]", "INFO")

        deleted = 0

        # === Close sensitive positions ===
        for p in positions:
            cat = symbol_to_category.get(p.symbol, "").lower()
            if cat not in NEWS_SENSITIVE_CATEGORIES:
                continue

            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                log_and_print(f"Cannot get tick for {p.symbol}, skipping close", "WARNING")
                continue

            close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": close_price,
                "deviation": 100,
                "magic": 999999,
                "comment": "NEWS_PROTECTION_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log_and_print(f"{broker_name}: CLOSED {p.symbol} ({cat}) - News Protection", "WARNING")
                deleted += 1
            else:
                log_and_print(f"{broker_name}: FAILED to close {p.symbol} | Retcode: {result.retcode if result else 'None'}", "ERROR")

        # === Cancel sensitive pending orders ===
        for o in pending:
            if o.type > 5:
                continue
            cat = symbol_to_category.get(o.symbol, "").lower()
            if cat not in NEWS_SENSITIVE_CATEGORIES:
                continue

            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log_and_print(f"{broker_name}: CANCELED pending {o.symbol} ({cat}) #{o.ticket}", "WARNING")
                deleted += 1
            else:
                log_and_print(f"{broker_name}: FAILED to cancel order {o.ticket} | Retcode: {result.retcode if result else 'None'}", "ERROR")

        total_deleted += deleted
        log_and_print(f"{broker_name}: {deleted} sensitive trades removed", "SUCCESS" if deleted > 0 else "INFO")
        mt5.shutdown()

    log_and_print(f"NEWS PROTECTION COMPLETE → {total_deleted} sensitive trades deleted", "CRITICAL")

def BreakevenRunningPositions():
    backup_brokers_dictionary()
    delete_news_sensitive_orders_during_news()
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
    BREAKEVEN_REPORT = "breakeven_report.json"
    ISSUES_FILE = "ordersissues.json"

    # === BREAKEVEN STAGES ===
    BE_STAGE_1 = 0.25   # SL moves here at ratio 1
    BE_STAGE_2 = 0.50   # SL moves here at ratio 2
    RATIO_1 = 1.0
    RATIO_2 = 2.0

    # === Helper: Round to symbol digits ===
    def _round_price(price, symbol):
        digits = mt5.symbol_info(symbol).digits
        return round(price, digits)

    # === Helper: Price at ratio ===
    def _ratio_price(entry, sl, tp, ratio, is_buy):
        risk = abs(entry - sl) or 1e-9
        return entry + risk * ratio * (1 if is_buy else -1)

    # === Helper: Modify SL ===
    def _modify_sl(pos, new_sl_raw):
        new_sl = _round_price(new_sl_raw, pos.symbol)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp,
            "magic": pos.magic,
            "comment": pos.comment
        }
        return mt5.order_send(req)

    # === Helper: Print block ===
    def _log_block(lines):
        log_and_print("\n".join(lines), "INFO")

    # === Helper: Safe JSON read (handles corrupted/multi-object files) ===
    def _safe_read_json(path):
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                # Handle multiple JSON objects by parsing line-by-line
                objs = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            objs.extend(obj)
                        elif isinstance(obj, dict):
                            objs.append(obj)
                    except json.JSONDecodeError:
                        continue
                return objs
        except Exception as e:
            log_and_print(f"Failed to read {path.name}: {e}. Starting fresh.", "WARNING")
            return []

    # === Helper: Safe JSON write ===
    def _safe_write_json(path, data):
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Ensure file ends cleanly
            return True
        except Exception as e:
            log_and_print(f"Failed to write {path.name}: {e}", "ERROR")
            return False

    # ------------------------------------------------------------------ #
    for broker_name, cfg in brokersdictionary.items():
        # ---- MT5 Connection ------------------------------------------------
        if not mt5.initialize(path=cfg["TERMINAL_PATH"], login=int(cfg["LOGIN_ID"]),
                              password=cfg["PASSWORD"], server=cfg["SERVER"], timeout=30000):
            log_and_print(f"{broker_name}: MT5 init failed", "ERROR")
            continue
        if not mt5.login(int(cfg["LOGIN_ID"]), cfg["PASSWORD"], cfg["SERVER"]):
            log_and_print(f"{broker_name}: MT5 login failed", "ERROR")
            mt5.shutdown()
            continue

        broker_dir = Path(BASE_INPUT_DIR) / broker_name
        report_path = broker_dir / BREAKEVEN_REPORT
        issues_path = broker_dir / ISSUES_FILE

        # Load existing report (unchanged)
        existing_report = []
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    existing_report = json.load(f)
            except Exception as e:
                log_and_print(f"{broker_name}: Failed to load breakeven_report.json – {e}", "WARNING")

        issues = []
        now = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z")
        now = f"{now[:-2]}:{now[-2:]}"  # Format +01:00 properly
        updated = pending_info = 0

        positions = mt5.positions_get() or []
        pending   = mt5.orders_get()   or []

        # ---- Group pending orders by symbol ----
        pending_by_sym = {}
        for o in pending:
            if o.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue
            pending_by_sym.setdefault(o.symbol, {})[o.type] = {
                "price": o.price_open, "sl": o.sl, "tp": o.tp
            }

        # ==================================================================
        # === PROCESS RUNNING POSITIONS ===
        # ==================================================================
        for pos in positions:
            if pos.sl == 0 or pos.tp == 0:
                continue

            sym = pos.symbol
            tick = mt5.symbol_info_tick(sym)
            info = mt5.symbol_info(sym)
            if not tick or not info:
                continue

            cur_price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            typ = "BUY" if is_buy else "SELL"

            # Key levels
            r1_price = _ratio_price(pos.price_open, pos.sl, pos.tp, RATIO_1, is_buy)
            r2_price = _ratio_price(pos.price_open, pos.sl, pos.tp, RATIO_2, is_buy)
            be_025   = _ratio_price(pos.price_open, pos.sl, pos.tp, BE_STAGE_1, is_buy)
            be_050   = _ratio_price(pos.price_open, pos.sl, pos.tp, BE_STAGE_2, is_buy)

            stage1 = (cur_price >= r1_price) if is_buy else (cur_price <= r1_price)
            stage2 = (cur_price >= r2_price) if is_buy else (cur_price <= r2_price)

            # Base block
            block = [
                f"┌─ {broker_name} ─ {sym} ─ {typ} (ticket {pos.ticket})",
                f"│ Entry : {pos.price_open:.{info.digits}f}   SL : {pos.sl:.{info.digits}f}   TP : {pos.tp:.{info.digits}f}",
                f"│ Now   : {cur_price:.{info.digits}f}"
            ]

            # === STAGE 2: SL to 0.50 ===
            if stage2 and abs(pos.sl - be_050) > info.point:
                res = _modify_sl(pos, be_050)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    block += [
                        f"│ BE @ 0.25 → {be_025:.{info.digits}f}",
                        f"│ BE @ 0.50 → {be_050:.{info.digits}f}  ← SL MOVED",
                        f"└─ All left to market"
                    ]
                    updated += 1
                else:
                    issues.append({"symbol": sym, "diagnosed_reason": "SL modify failed (stage 2)"})
                    block.append(f"└─ SL move FAILED")
                _log_block(block)
                continue

            # === STAGE 1: SL to 0.25 ===
            if stage1 and abs(pos.sl - be_025) > info.point:
                res = _modify_sl(pos, be_025)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    block += [
                        f"│ BE @ 0.25 → {be_025:.{info.digits}f}  ← SL MOVED",
                        f"│ Waiting ratio 2 @ {r2_price:.{info.digits}f} → BE @ 0.50 → {be_050:.{info.digits}f}"
                    ]
                    updated += 1
                else:
                    issues.append({"symbol": sym, "diagnosed_reason": "SL modify failed (stage 1)"})
                    block.append(f"└─ SL move FAILED")
                _log_block(block)
                continue

            # === STAGE 1 REACHED, WAITING STAGE 2 ===
            if stage1:
                block += [
                    f"│ BE @ 0.25 → {be_025:.{info.digits}f}",
                    f"│ Waiting ratio 2 @ {r2_price:.{info.digits}f} → BE @ 0.50 → {be_050:.{info.digits}f}"
                ]
            # === WAITING STAGE 1 ===
            else:
                block += [
                    f"│ Waiting ratio 1 @ {r1_price:.{info.digits}f} → BE @ 0.25 → {be_025:.{info.digits}f}"
                ]

            block.append("")
            _log_block(block)

        # ==================================================================
        # === PROCESS PENDING ORDERS (INFO ONLY) ===
        # ==================================================================
        for sym, orders in pending_by_sym.items():
            for otype, o in orders.items():
                if o["sl"] == 0 or o["tp"] == 0:
                    continue
                info = mt5.symbol_info(sym)
                if not info:
                    continue
                is_buy = otype == mt5.ORDER_TYPE_BUY_LIMIT
                typ = "BUY_LIMIT" if is_buy else "SELL_LIMIT"

                r1_price = _ratio_price(o["price"], o["sl"], o["tp"], RATIO_1, is_buy)
                r2_price = _ratio_price(o["price"], o["sl"], o["tp"], RATIO_2, is_buy)
                be_025   = _ratio_price(o["price"], o["sl"], o["tp"], BE_STAGE_1, is_buy)
                be_050   = _ratio_price(o["price"], o["sl"], o["tp"], BE_STAGE_2, is_buy)

                block = [
                    f"┌─ {broker_name} ─ {sym} ─ PENDING {typ}",
                    f"│ Entry : {o['price']:.{info.digits}f}   SL : {o['sl']:.{info.digits}f}   TP : {o['tp']:.{info.digits}f}",
                    f"│ Target 1 → {r1_price:.{info.digits}f}  |  BE @ 0.25 → {be_025:.{info.digits}f}",
                    f"│ Target 2 → {r2_price:.{info.digits}f}  |  BE @ 0.50 → {be_050:.{info.digits}f}",
                    f"└─ Order not running – waiting…"
                ]
                _log_block(block)
                pending_info += 1

        # === SAVE BREAKEVEN REPORT (unchanged) ===
        _safe_write_json(report_path, existing_report)

        # === SAVE ISSUES – ROBUST MERGE ===
        current_issues = _safe_read_json(issues_path)
        all_issues = current_issues + issues
        _safe_write_json(issues_path, all_issues)

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Breakeven done – SL Updated: {updated} | Pending Info: {pending_info}",
            "SUCCESS"
        )

    log_and_print("All brokers breakeven processed.", "SUCCESS")
    
def verifying_brokers():
    # --- CONFIGURATION ---
    BROKERS_JSON = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    USERS_JSON   = r"C:\xampp\htdocs\chronedge\synarex\updatedusersdictionary.json"
    REQUIREMENTS_JSON = r"C:\xampp\htdocs\chronedge\synarex\requirements.json"

    # Load requirements
    if not os.path.exists(REQUIREMENTS_JSON):
        print(f"CRITICAL: {REQUIREMENTS_JSON} not found!", "CRITICAL")
        return

    try:
        with open(REQUIREMENTS_JSON, "r", encoding="utf-8") as f:
            req_data = json.load(f)
        
        BALANCE_REQUIRED = float(req_data.get("minimum_deposit", 0))
        CONTRACT_DURATION_DAYS = int(req_data.get("contract_duration", 30))
        
        if BALANCE_REQUIRED <= 0 or CONTRACT_DURATION_DAYS <= 0:
            print(f"CRITICAL: Invalid values in {REQUIREMENTS_JSON}", "CRITICAL")
            return
            
        print(f"Requirements loaded → Min Deposit: ${BALANCE_REQUIRED:.2f} | Duration: {CONTRACT_DURATION_DAYS} days", "INFO")
    except Exception as e:
        print(f"CRITICAL: Failed to load {REQUIREMENTS_JSON}: {e}", "CRITICAL")
        return

    if not os.path.exists(BROKERS_JSON):
        print(f"CRITICAL: {BROKERS_JSON} not found!", "CRITICAL")
        return

    # Load active brokers
    try:
        with open(BROKERS_JSON, "r", encoding="utf-8") as f:
            brokers_dict = json.load(f)
    except Exception as e:
        print(f"Failed to load brokers JSON: {e}", "CRITICAL")
        return

    # Load historical/expired users
    users_dict = {}
    if os.path.exists(USERS_JSON):
        try:
            with open(USERS_JSON, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    users_dict = loaded
        except Exception as e:
            print(f"Warning: Could not load users JSON: {e}. Starting fresh.", "WARNING")

    move_list = []  # Brokers to move (expired or low balance)
    updated_any = False
    lagos_tz = pytz.timezone("Africa/Lagos")
    now_dt = datetime.now(lagos_tz)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    now_ts = int(now_dt.timestamp())

    def parse_start_date(date_val):
        if not date_val or str(date_val).strip().lower() in ("", "none", "null", "unknown"):
            return None
        date_str = str(date_val).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return lagos_tz.localize(dt) if dt.tzinfo is None else dt
            except ValueError:
                continue
        return None

    # --- PHASE 1: Check contract time & mark for move if ≤5 days left ---
    for broker_name, cfg in list(brokers_dict.items()):
        current_start_str = cfg.get("EXECUTION_START_DATE")
        current_start_dt = parse_start_date(current_start_str)

        # Fix missing start date from history
        history_str = cfg.get("EXECUTION_DATES_HISTORY", "")
        if not current_start_dt and history_str:
            parts = [p.strip() for p in history_str.split(",") if p.strip() and p.lower() not in ("none", "unknown")]
            if parts:
                restored = parse_start_date(parts[0])
                if restored:
                    current_start_dt = restored
                    cfg["EXECUTION_START_DATE"] = restored.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"**{broker_name}**: Start date restored from history", "INFO")
                    updated_any = True

        # Calculate days left with precision
        if current_start_dt is None:
            days_left = CONTRACT_DURATION_DAYS
            cfg["EXECUTION_START_DATE"] = now_str
            current_start_dt = now_dt
            updated_any = True
        else:
            delta = now_dt - current_start_dt
            days_passed = delta.days + (delta.seconds / 86400.0)
            days_left = max(0, CONTRACT_DURATION_DAYS - days_passed)

        days_left_int = int(days_left)
        cfg["CONTRACT_DAYS_LEFT"] = days_left_int
        updated_any = True

        # If 5 or fewer days remain → EXPIRE & MOVE
        if days_left_int <= 5:
            reason = "Contract Expired (≤5 days left)"
            print(f"**{broker_name}**: {reason} → {days_left_int} days remaining → Will MOVE", "WARNING")
            move_list.append((broker_name, reason))

    # --- PHASE 2: MT5 Live Update (only for brokers NOT being moved yet) ---
    brokers_to_remove_due_to_balance = []

    for broker_name, cfg in list(brokers_dict.items()):
        # Skip if already marked for move due to time
        if any(broker_name == b[0] for b in move_list):
            continue

        print(f"**{broker_name}**: Connecting to MT5 for live update...", "INFO")

        terminal_path = cfg.get("TERMINAL_PATH")
        login_id = cfg.get("LOGIN_ID")
        password = cfg.get("PASSWORD")
        server = cfg.get("SERVER")

        if not all([terminal_path, login_id, password, server]):
            print(f"**{broker_name}**: Missing credentials → Will move", "ERROR")
            move_list.append((broker_name, "Missing credentials"))
            continue

        try:
            if not mt5.initialize(path=terminal_path, timeout=30000):
                print(f"**{broker_name}**: MT5 init failed → Will move", "ERROR")
                move_list.append((broker_name, "MT5 init failed"))
                continue

            if not mt5.login(int(login_id), password=password, server=server):
                print(f"**{broker_name}**: Login failed → Will move", "ERROR")
                mt5.shutdown()
                move_list.append((broker_name, "Login failed"))
                continue

            account_info = mt5.account_info()
            if not account_info:
                print(f"**{broker_name}**: No account info → Will move", "ERROR")
                mt5.shutdown()
                move_list.append((broker_name, "No account info"))
                continue

            current_balance = round(account_info.balance, 2)
            start_dt = parse_start_date(cfg.get("EXECUTION_START_DATE"))
            from_ts = int(start_dt.timestamp()) if start_dt else now_ts

            # Fetch realized P&L since start date
            deals = mt5.history_deals_get(from_ts, now_ts + 120)
            realized_pnl = 0.0
            total_trades = won_trades = lost_trades = 0
            wins_by_symbol = {}
            losses_by_symbol = {}

            if deals:
                for deal in deals:
                    if deal.entry not in (1, 3):
                        continue
                    profit = deal.profit
                    symbol = deal.symbol or "Unknown"
                    realized_pnl += profit
                    if profit > 0:
                        won_trades += 1
                        wins_by_symbol[symbol] = wins_by_symbol.get(symbol, 0) + profit
                    elif profit < 0:
                        lost_trades += 1
                        losses_by_symbol[symbol] = losses_by_symbol.get(symbol, 0) + profit
                    else:
                        won_trades += 1
                total_trades = won_trades + lost_trades

            realized_pnl = round(realized_pnl, 2)
            wins_list = [f"{s}:{p:.2f}" for s, p in sorted(wins_by_symbol.items(), key=lambda x: -x[1])]
            losses_list = [f"{s}:{p:.2f}" for s, p in sorted(losses_by_symbol.items(), key=lambda x: x[1])]

            trades_summary = (
                f"{total_trades}:Trades, {won_trades}:Won, {lost_trades}:Lost, "
                f"symbolsthatlost:({', '.join(losses_list) if losses_list else 'None'}), "
                f"symbolsthatwon:({', '.join(wins_list) if wins_list else 'None'})"
            )

            # Update live stats
            cfg["BROKER_BALANCE"] = current_balance
            cfg["PROFITANDLOSS"] = realized_pnl
            cfg["TRADES"] = trades_summary

            print(f"**{broker_name}**: Live → Bal: ${current_balance:.2f} | P&L: {realized_pnl:+.2f} | Days Left: {cfg['CONTRACT_DAYS_LEFT']}", "INFO")

            # Check minimum balance
            if current_balance < BALANCE_REQUIRED:
                print(f"**{broker_name}**: Balance ${current_balance:.2f} < ${BALANCE_REQUIRED:.2f} → Will move", "CRITICAL")
                brokers_to_remove_due_to_balance.append(broker_name)

            mt5.shutdown()
            updated_any = True

        except Exception as e:
            print(f"**{broker_name}**: MT5 error: {e} → Will move", "ERROR")
            move_list.append((broker_name, "MT5 error"))
            if 'mt5' in locals():
                mt5.shutdown()
            continue

    # Add low-balance brokers to move list
    for b in brokers_to_remove_due_to_balance:
        if not any(b == x[0] for x in move_list):
            move_list.append((b, "Balance too low"))

    # --- PHASE 3: MOVE all flagged brokers ---
    moved_count = 0
    for broker_name, reason in move_list:
        if broker_name not in brokers_dict:
            continue
        broker_data = brokers_dict.pop(broker_name)

        # Clean sensitive data
        for field in ("TERMINAL_PATH", "BASE_FOLDER", "RESET_EXECUTION_DATE_AND_BROKER_BALANCE"):
            broker_data.pop(field, None)

        broker_data["MOVED_REASON"] = reason
        broker_data["MOVED_TIMESTAMP"] = now_str

        users_dict[broker_name] = broker_data
        print(f"**{broker_name}**: MOVED → {reason} | Final Bal: ${broker_data.get('BROKER_BALANCE', 0):.2f}", "SUCCESS")
        moved_count += 1
        updated_any = True

    # --- SAVE ---
    try:
        if moved_count == 0:
            print("No brokers moved → Saving active snapshot", "INFO")
            for name, data in brokers_dict.items():
                clean = data.copy()
                for f in ("TERMINAL_PATH", "BASE_FOLDER", "RESET_EXECUTION_DATE_AND_BROKER_BALANCE"):
                    clean.pop(f, None)
                clean["LAST_SEEN_ACTIVE"] = now_str
                users_dict[name] = clean

        with open(BROKERS_JSON, "w", encoding="utf-8") as f:
            json.dump(brokers_dict, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print("brokersdictionary.json → Saved (only active)", "SUCCESS")

        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump(users_dict, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print(f"updatedusersdictionary.json → Saved ({len(users_dict)} total records)", "SUCCESS")

    except Exception as e:
        print(f"CRITICAL: Failed to save files: {e}", "CRITICAL")       
        
def updating_database_record():
    try:
        timeorders.current_time()
        print("updated")
    except Exception as e:
        print(f"Error updating {e}")

def calc_and_placeorders():  
    verifying_brokers()
    updating_database_record()
    calculate_symbols_sl_tp_prices() 
    placeallorders()

def clear_chart_folder(base_folder: str):
    """Delete ONLY symbols that have NO valid OB-none-OI record on 15m-4h."""
    error_log = []
    IMPORTANT_TFS = {"15m", "30m", "1h", "4h"}

    if not os.path.exists(base_folder):
        log_and_print(f"Chart folder {base_folder} does not exist – nothing to clear.", "INFO")
        return True, error_log

    deleted = 0
    kept    = 0

    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if not os.path.isdir(item_path):
            continue                                 # skip stray files

        # --------------------------------------------------
        # Look for ob_none_oi_data.json inside any timeframe folder
        # --------------------------------------------------
        keep_symbol = False
        for tf in IMPORTANT_TFS:
            json_path = os.path.join(item_path, tf, "ob_none_oi_data.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # file exists → assume it contains at least one team entry
                keep_symbol = True
                break
            except Exception:
                pass                                 # corrupted → treat as “missing”

        # --------------------------------------------------
        # Delete or keep
        # --------------------------------------------------
        try:
            if keep_symbol:
                kept += 1
                log_and_print(f"KEEP   {item_path} (has 15m-4h OB-none-OI)", "INFO")
            else:
                shutil.rmtree(item_path)
                deleted += 1
                log_and_print(f"DELETE {item_path} (no 15m-4h record)", "INFO")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime(
                    '%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to handle {item_path}: {str(e)}",
                "broker": base_folder
            })
            log_and_print(f"Failed to handle {item_path}: {str(e)}", "ERROR")

    log_and_print(
        f"Smart clean finished → {deleted} folders deleted, {kept} folders kept.",
        "SUCCESS")
    return True, error_log

def clear_unknown_broker():
    base_path = r"C:\xampp\htdocs\chronedge\synarex\chart"
    
    if not os.path.exists(base_path):
        print(f"ERROR: Base directory does not exist:\n    {base_path}")
        return
    
    if not brokersdictionary:
        print("No brokers found in brokersdictionary.")
        return

    print("Configured Brokers & Folder Check (Human-readable folders):")
    print("=" * 90)
    
    configured_displays = set()
    known_broker_bases = set()
    broker_details = []
    existing = 0
    missing = 0
    
    def format_broker_name(name):
        name = name.strip()
        match = re.match(r"([a-zA-Z_]+)(\d*)$", name, re.IGNORECASE)
        if not match:
            return name.capitalize()
        base, num = match.groups()
        base_clean = base.capitalize()
        if num:
            known_broker_bases.add(base_clean)
            return f"{base_clean} {int(num)}"
        known_broker_bases.add(base_clean)
        return base_clean

    # ——— Scan configured brokers ———
    for broker_name in brokersdictionary.keys():
        original = broker_name.strip()
        display_name = format_broker_name(original)
        lower_display = display_name.lower()
        
        configured_displays.add(lower_display)
        
        folder_path = os.path.join(base_path, display_name)
        exists = os.path.isdir(folder_path)
        
        marker = "Success" if exists else "Error"
        status = "EXISTS" if exists else "MISSING"
        
        print(f"{marker} {original.ljust(25)} → {display_name.ljust(20)} → {status}")
        print(f"    Path: {folder_path}\n")
        
        broker_details.append({
            'original': original,
            'display': display_name,
            'lower': lower_display,
            'path': folder_path,
            'exists': exists
        })
        
        if exists: existing += 1
        else: missing += 1
    
    print("=" * 90)
    print(f"Total configured: {len(brokersdictionary)} broker(s) | {existing} folder(s) exist | {missing} missing")

    # ——— Unique broker types ———
    print("\nUnique Configured Broker Types:")
    print("-" * 60)
    for base in sorted(known_broker_bases):
        instances = [b['display'] for b in broker_details if b['display'].startswith(base)]
        print(f"• {base.ljust(15)} → {len(instances)} account(s): {', '.join(instances)}")
    print("-" * 60)
    print(f"Unique broker types: {len(known_broker_bases)}")

    # ——— AUTO-DELETE ORPHANED FOLDERS (NO CONFIRMATION) ———
    print("\nCleaning Orphaned Broker Folders (AUTO-DELETE enabled)...")
    print("-" * 70)
    
    if not os.path.isdir(base_path):
        print("Base path not accessible.")
    else:
        orphaned = []
        all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        
        for folder in all_folders:
            folder_lower = folder.lower()
            full_path = os.path.join(base_path, folder)
            
            if folder_lower in configured_displays:
                continue
                
            suspected_base = None
            for base in known_broker_bases:
                if folder_lower.startswith(base.lower()):
                    suspected_base = base
                    break
            
            if suspected_base:
                orphaned.append((folder, full_path, suspected_base))
        
        if orphaned:
            print(f"Deleting {len(orphaned)} orphaned broker folder(s):")
            deleted_count = 0
            for folder, full_path, base in orphaned:
                try:
                    shutil.rmtree(full_path)
                    print(f"  Deleted: {folder}  (was {base})")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Failed to delete {folder}: {e}")
            print(f"\nAuto-clean complete: {deleted_count}/{len(orphaned)} orphaned folders removed.")
        else:
            print("No orphaned broker folders found. Directory is clean!")

    print("-" * 70)
    
    if missing > 0:
        print(f"\nReminder: {missing} configured broker(s) missing their folder!")
        print("   Expected format: Deriv 2, Bybit 6, Exness 1, etc.")

def fetch_charts_all_brokers():
    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    backup_brokers_dictionary()
    delete_all_category_jsons()
    delete_all_calculated_risk_jsons()
    delete_issue_jsons()
    clear_unknown_broker()
    required_allowed_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    fallback_allowed_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    allsymbols_path       = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    match_path            = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\symbolsmatch.json"
    brokers_report_path   = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\brokerslimitorders.json"

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    IMPORTANT_TFS = {"15m", "30m", "1h", "4h"}

    def normalize_broker_key(name: str) -> str:
        return re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', '', name.strip())).lower()

    def clean_folder_name(name: str) -> str:
        # NOTE: This function is kept for parts of the original script that might
        # rely on a 'cleaned' name (like broker report loading or dictionary keys),
        # but it will be explicitly avoided in load_technique_config.
        cleaned = re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', ' ', name.strip()))
        return cleaned.strip().title()

    def normalize_symbol(s: str) -> str:
        return re.sub(r'[\/\s\-_]+', '', s.strip()).upper() if s else ""

    def load_technique_config(broker_name: str) -> dict:
        """
        Load technique.json with full support for parent & child highs/lows.
        FIX: Uses the raw broker_name for folder creation, not a cleaned one.
        """
        # --- FIX APPLIED HERE: Use raw broker_name for dev_folder ---
        dev_folder = fr"C:\xampp\htdocs\chronedge\synarex\chart\developers\{broker_name}"
        config_path = os.path.join(dev_folder, "technique.json")
        
        default_config = {
            "BARS": 201,
            "parenthighsandlows": {
                "NEIGHBOR_LEFT": 10,
                "NEIGHBOR_RIGHT": 15
            },
            "childhighsandlows": {
                "NEIGHBOR_LEFT": 5,
                "NEIGHBOR_RIGHT": 7
            }
        }
        
        if not os.path.exists(config_path):
            os.makedirs(dev_folder, exist_ok=True)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4)
                log_and_print(f"CREATED default technique.json → {config_path}", "INFO")
            except Exception as e:
                log_and_print(f"FAILED to create technique.json for {broker_name}: {e}", "ERROR")
                return default_config.copy()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Ensure BARS
            bars = int(config.get("BARS", default_config["BARS"]))

            # Ensure parenthighsandlows
            phl = config.get("parenthighsandlows", {})
            if not isinstance(phl, dict):
                phl = default_config["parenthighsandlows"]
            parent_left  = int(phl.get("NEIGHBOR_LEFT", default_config["parenthighsandlows"]["NEIGHBOR_LEFT"]))
            parent_right = int(phl.get("NEIGHBOR_RIGHT", default_config["parenthighsandlows"]["NEIGHBOR_RIGHT"]))

            # Ensure childhighsandlows
            chl = config.get("childhighsandlows", {})
            if not isinstance(chl, dict):
                chl = default_config["childhighsandlows"]
            child_left  = int(chl.get("NEIGHBOR_LEFT", default_config["childhighsandlows"]["NEIGHBOR_LEFT"]))
            child_right = int(chl.get("NEIGHBOR_RIGHT", default_config["childhighsandlows"]["NEIGHBOR_RIGHT"]))

            result = {
                "BARS": bars,
                "PARENT_NEIGHBOR_LEFT": parent_left,
                "PARENT_NEIGHBOR_RIGHT": parent_right,
                "CHILD_NEIGHBOR_LEFT": child_left,
                "CHILD_NEIGHBOR_RIGHT": child_right
            }

            log_and_print(f"Loaded technique for {broker_name}: BARS={bars}, "
                          f"Parent(L={parent_left},R={parent_right}), "
                          f"Child(L={child_left},R={child_right})", "INFO")

            return result

        except Exception as e:
            log_and_print(f"FAILED to load technique.json for {broker_name}, using default: {e}", "WARNING")
            return default_config.copy()


    def delete_symbol_folder(symbol: str, base_folder: str, reason: str = ""):
        sym_folder = os.path.join(base_folder, symbol.replace(" ", "_"))
        if os.path.exists(sym_folder):
            try:
                shutil.rmtree(sym_folder)
                log_and_print(f"DELETED {sym_folder} {reason}", "INFO")
            except Exception as e:
                log_and_print(f"FAILED to delete {sym_folder}: {e}", "ERROR")
        os.makedirs(base_folder, exist_ok=True)

    def delete_all_non_blocked_symbol_folders(base_folder: str, blocked_symbols: set):
        if not os.path.exists(base_folder):
            return
        deleted = 0
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if not os.path.isdir(item_path):
                continue
            symbol = item.replace("_", " ")
            if symbol in blocked_symbols:
                log_and_print(f"KEEPING folder {item} → {symbol} is BLOCKED", "INFO")
                continue
            try:
                shutil.rmtree(item_path)
                deleted += 1
            except Exception as e:
                log_and_print(f"FAILED to delete {item_path}: {e}", "ERROR")
        log_and_print(f"CLEANED {deleted} non-blocked symbol folders in {base_folder}", "SUCCESS")

    def breakeven_worker():
        while True:
            try:
                BreakevenRunningPositions()
            except Exception as e:
                log_and_print(f"BREAKEVEN ERROR: {e}", "CRITICAL")
            time.sleep(10)

    threading.Thread(target=breakeven_worker, daemon=True).start()
    log_and_print("Breakeven thread started", "SUCCESS")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    while True:
        error_log = []
        log_and_print("\n=== NEW FULL CYCLE STARTED (DEVELOPER BROKERS ONLY) ===", "INFO")

        try:
            # ------------------------------------------------------------------
            # 0. LOAD BLOCKED SYMBOLS FROM brokerslimitorders.json
            # ------------------------------------------------------------------
            normalized_blocked_symbols = {}

            if os.path.exists(brokers_report_path):
                try:
                    with open(brokers_report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)

                    for section in ["pending_orders", "open_positions", "history_orders"]:
                        items = report.get(section, [])
                        for item in items:
                            broker_raw = item.get("broker", "")
                            symbol = item.get("symbol", "")
                            if not broker_raw or not symbol:
                                continue

                            clean_broker = clean_folder_name(broker_raw) # Use clean name for report matching
                            normalized_blocked_symbols.setdefault(clean_broker, set())

                            if section == "history_orders":
                                age_str = item.get("age", "")
                                if "d" in age_str:
                                    days = int(age_str.split("d")[0])
                                    if days >= 5:
                                        continue
                                elif not any(t in age_str for t in ["h", "m", "s"]):
                                    continue

                            normalized_blocked_symbols[clean_broker].add(symbol)
                except Exception as e:
                    log_and_print(f"FAILED to load brokerslimitorders.json: {e}", "ERROR")

            # ------------------------------------------------------------------
            # 1. FILTER ONLY DEVELOPER BROKERS + LOAD THEIR TECHNIQUE CONFIG
            # ------------------------------------------------------------------
            developer_brokers = {}
            for original_key, cfg in brokersdictionary.items():
                position = cfg.get("POSITION", "").lower()
                if position != "developer":
                    continue

                broker_name = cfg.get("original_name", original_key)
                clean_name = clean_folder_name(broker_name)
                
                # --- FIX APPLIED HERE: Pass the raw broker_name to load_technique_config ---
                technique = load_technique_config(broker_name) 
                
                cfg_copy = cfg.copy()
                cfg_copy["technique"] = technique
                cfg_copy["clean_name"] = clean_name # Retaining clean_name for other uses (like blocked symbols)
                cfg_copy["raw_name"] = broker_name # Storing the raw name for folder creation in step 6
                developer_brokers[clean_name] = cfg_copy # Keying by clean name remains to align with normalized_blocked_symbols

            if not developer_brokers:
                log_and_print("NO DEVELOPER BROKERS FOUND → sleeping 30 min", "WARNING")
                time.sleep(1800)
                continue

            log_and_print(f"FOUND {len(developer_brokers)} DEVELOPER BROKER(S): {list(developer_brokers.keys())}", "SUCCESS")

            # ------------------------------------------------------------------
            # 1.5 CLEAN NON-BLOCKED FOLDERS
            # ------------------------------------------------------------------
            for clean_name, cfg in developer_brokers.items():
                blocked = normalized_blocked_symbols.get(clean_name, set())
                delete_all_non_blocked_symbol_folders(cfg["BASE_FOLDER"], blocked)

            # ------------------------------------------------------------------
            # 2–4. Load allowed markets, symbol mapping, etc.
            # ------------------------------------------------------------------
            if not os.path.exists(required_allowed_path):
                if os.path.exists(fallback_allowed_path):
                    os.makedirs(os.path.dirname(required_allowed_path), exist_ok=True)
                    shutil.copy2(fallback_allowed_path, required_allowed_path)
                    log_and_print("AUTO-COPIED allowedmarkets.json", "INFO")
                else:
                    log_and_print("CRITICAL: allowedmarkets.json missing!", "CRITICAL")
                    time.sleep(600); continue

            with open(required_allowed_path, "r", encoding="utf-8") as f:
                allowed_config = json.load(f)

            normalized_allowed = {
                cat: {normalize_symbol(s) for s in cfg.get("allowed", [])}
                for cat, cfg in allowed_config.items()
            }

            if not os.path.exists(allsymbols_path):
                log_and_print(f"Missing {allsymbols_path}", "CRITICAL")
                time.sleep(600); continue

            symbol_to_category = {}
            with open(allsymbols_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for markets in data.values():
                for cat in markets:
                    for item in markets.get(cat, []):
                        if sym := item.get("symbol"):
                            norm = normalize_symbol(sym)
                            symbol_to_category[norm] = cat
                            symbol_to_category[sym] = cat

            if not os.path.exists(match_path):
                log_and_print(f"Missing {match_path}", "CRITICAL")
                time.sleep(600); continue
            with open(match_path, "r", encoding="utf-8") as f:
                symbolsmatch_data = json.load(f)

            # ------------------------------------------------------------------
            # 5. Build candidate list
            # ------------------------------------------------------------------
            all_cats = ["stocks","forex","crypto","synthetics","indices","commodities","equities","energies","etfs","basket_indices","metals"]
            candidates = {name: {c: [] for c in all_cats} for name in developer_brokers}
            total_to_do = 0

            for clean_name, cfg in developer_brokers.items():
                blocked = normalized_blocked_symbols.get(clean_name, set())
                broker_symbols_raw = cfg.get("SYMBOLS", "").strip()
                broker_allowed_symbols = None
                if broker_symbols_raw and broker_symbols_raw.lower() != "all":
                    broker_allowed_symbols = {normalize_symbol(s) for s in broker_symbols_raw.split(",") if s.strip()}

                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if not ok:
                    mt5.shutdown()
                    continue
                avail, _ = get_symbols()
                mt5.shutdown()

                for entry in symbolsmatch_data.get("main_symbols", []):
                    canonical = entry.get("symbol")
                    if not canonical:
                        continue
                    norm_canonical = normalize_symbol(canonical)

                    broker_symbols_list = []
                    # Match against the cleaned name, as this is used for dictionary keys
                    for possible_key in [clean_name.lower(), clean_name.title(), clean_name.upper()]:
                        if possible_key in entry:
                            broker_symbols_list = entry.get(possible_key, [])
                            break
                    if not broker_symbols_list:
                        continue

                    for sym_mt5 in broker_symbols_list:
                        if sym_mt5 not in avail or sym_mt5 in blocked:
                            continue

                        cat = symbol_to_category.get(norm_canonical) or symbol_to_category.get(canonical)
                        if not cat or cat not in all_cats:
                            continue

                        if allowed_config.get(cat, {}).get("limited", False):
                            if norm_canonical not in normalized_allowed.get(cat, set()):
                                continue

                        if broker_allowed_symbols is not None and norm_canonical not in broker_allowed_symbols:
                            continue

                        delete_symbol_folder(sym_mt5, cfg["BASE_FOLDER"], "(pre-process cleanup)")
                        candidates[clean_name][cat].append(sym_mt5)

                for cat in all_cats:
                    cnt = len(candidates[clean_name][cat])
                    if cnt:
                        log_and_print(f"{clean_name.upper()} → {cat.upper():10} : {cnt:3} queued", "INFO")
                        total_to_do += cnt

            if total_to_do == 0:
                log_and_print("No symbols to process – sleeping 30 min", "WARNING")
                time.sleep(1800)
                continue

            log_and_print(f"TOTAL TO PROCESS: {total_to_do}", "SUCCESS")

            # ------------------------------------------------------------------
            # 6. PROCESS ALL DEVELOPER BROKERS
            # ------------------------------------------------------------------
            for clean_name, cfg in developer_brokers.items():
                tech = cfg["technique"]
                bars = tech["BARS"]
                parent_left = tech["PARENT_NEIGHBOR_LEFT"]
                parent_right = tech["PARENT_NEIGHBOR_RIGHT"]
                child_left = tech["CHILD_NEIGHBOR_LEFT"]
                child_right = tech["CHILD_NEIGHBOR_RIGHT"]

                log_and_print(f"STARTING PROCESSING FOR {clean_name.upper()} | "
                              f"BARS: {bars} | Parent(L={parent_left},R={parent_right}) | "
                              f"Child(L={child_left},R={child_right})", "INFO")

                for cat in all_cats:
                    for symbol in candidates[clean_name][cat]:
                        for run in (1, 2):
                            ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                            error_log.extend(errs)
                            if not ok:
                                log_and_print(f"MT5 INIT FAILED → {clean_name}/{symbol} (run {run})", "ERROR")
                                mt5.shutdown()
                                continue

                            log_and_print(f"RUN {run} – PROCESSING {symbol} ({cat}) on {clean_name.upper()}", "INFO")

                            # --- FIX APPLIED HERE: Use the raw_name to construct the BASE_FOLDER ---
                            # Note: The original code used cfg["BASE_FOLDER"], which is fine if it was initialized with the raw name.
                            # Since BASE_FOLDER is likely defined in brokersdictionary using the original key, we can continue to rely on it 
                            # if it was set up correctly, but for maximum safety in subsequent logic, we'll confirm
                            # we're using the raw name's folder structure if needed. For now, we trust BASE_FOLDER 
                            # is correct for the broker. The fix was primarily in load_technique_config.
                            sym_folder = os.path.join(cfg["BASE_FOLDER"], symbol.replace(" ", "_"))
                            os.makedirs(sym_folder, exist_ok=True)

                            def process_symbol_timeframes():
                                for tf_str, mt5_tf in TIMEFRAME_MAP.items():
                                    tf_folder = os.path.join(sym_folder, tf_str)
                                    os.makedirs(tf_folder, exist_ok=True)

                                    df, errs = fetch_ohlcv_data(symbol, mt5_tf, bars)
                                    error_log.extend(errs)
                                    if df is None or len(df) < 50:
                                        log_and_print(f"Insufficient data for {symbol} {tf_str}", "WARNING")
                                        continue

                                    df["symbol"] = symbol

                                    # Now passing all 4 neighbor params + returning CH/CL
                                    chart_path, ch_errs, ph, pl, ch, cl = generate_and_save_chart(
                                        df, symbol, tf_str, tf_folder,
                                        parent_left, parent_right,
                                        child_left, child_right
                                    )
                                    error_log.extend(ch_errs)

                                    # Save candles with full PH/PL/CH/CL labels
                                    save_candle_data(df, symbol, tf_str, tf_folder, ph, pl, ch, cl)
                                    
                                    next_errs = save_next_candles(df, symbol, tf_str, tf_folder, ph, pl, ch, cl)
                                    error_log.extend(next_errs)

                                    if chart_path and os.path.exists(chart_path):
                                        detect_errors = detect_candle_contours(
                                            chart_path, symbol, tf_str, tf_folder,
                                            candleafterintersector=2,
                                            minbreakoutcandleposition=5,
                                            startOBsearchFrom=0,
                                            minOBleftneighbor=1,
                                            minOBrightneighbor=1,
                                            reversal_leftcandle=0,
                                            reversal_rightcandle=0
                                        )
                                        error_log.extend(detect_errors)

                                mt5.shutdown()
                            
                            process_symbol_timeframes()

            save_errors(error_log)
            log_and_print("CYCLE 100% COMPLETED (ALL DEVELOPER BROKERS)", "SUCCESS")
            log_and_print("Sleeping 30 minutes before next cycle...", "INFO")
            time.sleep(1800)

        except Exception as e:
            log_and_print(f"MAIN LOOP CRASH: {e}\n{traceback.format_exc()}", "CRITICAL")
            time.sleep(600)




def custom_chart():
    import cv2
    import numpy as np
    import os
    import json
    from datetime import datetime
    import pytz

    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ==================================================================
    # PROFESSIONAL COLORS & HELPERS
    # ==================================================================
    COLOR_MAP = {
        "ph": (255, 100, 0),       # Orange
        "pl": (200, 0, 200),       # Magenta
        "ch": (255, 200, 0),       # Cyan
        "cl": (0, 140, 255),       # Warm Orange
        "fvg_middle_(bullish)": (0, 255, 0),
        "fvg_middle_(bearish)": (60, 20, 220),
    }

    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))

    def is_level_match(candle, key):
        if key == "ph": return candle.get("is_ph")
        if key == "pl": return candle.get("is_pl")
        if key == "ch": return candle.get("is_ch")
        if key == "cl": return candle.get("is_cl")
        if key == "fvg_middle_(bullish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bullish"
        if key == "fvg_middle_(bearish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bearish"
        return False

    def get_y_position(positions, cnum, key):
        pos = positions[cnum]
        return pos["high_y"] if key in ["ph", "ch", "fvg_middle_(bullish)"] else pos["low_y"]

    # ==================================================================
    # Detect candle bodies → get X/Y positions (right = newest)
    # ==================================================================
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None:
            return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)  # right → left
        raw = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            raw[idx] = {"x": x + w // 2, "high_y": y, "low_y": y + h}
        return img.copy(), raw

    # ==================================================================
    # MAIN LOOP – DEVELOPER BROKERS
    # ==================================================================
    developer_brokers = {
        k: v for k, v in brokersdictionary.items()
        if v.get("POSITION", "").lower() == "developer"
    }

    if not developer_brokers:
        log("No developer brokers found!", "ERROR")
        return

    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "technique.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "technique.json")
        if not os.path.exists(technique_path):
            log(f"technique.json missing → {broker_raw_name}", "WARNING")
            continue

        try:
            with open(technique_path, 'r', encoding='utf-8') as f:
                tech = json.load(f)
        except Exception as e:
            log(f"Failed loading technique.json: {e}", "ERROR")
            continue

        draw_trend = str(tech.get("drawings_switch", {}).get("trendline", "no")).strip().lower() == "yes"
        draw_horiz = str(tech.get("drawings_switch", {}).get("horizontal_line", "no")).strip().lower() == "yes"

        if not draw_trend and not draw_horiz:
            log(f"Both trendline & horizontal_line disabled → {broker_raw_name}", "INFO")
            continue

        # Load configurations
        trend_list = []
        if draw_trend:
            for key in sorted([k for k in tech.get("trendline", {}).keys() if str(k).isdigit()]):
                conf = tech["trendline"][key]
                if not isinstance(conf, dict): continue
                fr = conf.get("FROM", "").strip().lower().replace(" ", "_")
                to = conf.get("TO", "ray").strip().lower()
                if fr:
                    trend_list.append({"id": key, "FROM": fr, "TO": to})

        horiz_list = []
        if draw_horiz:
            for key in sorted([k for k in tech.get("horizontal_line", {}).keys() if str(k).isdigit()]):
                conf = tech["horizontal_line"][key]
                if not isinstance(conf, dict): continue
                fr = conf.get("FROM", "").strip().lower().replace(" ", "_")
                to = conf.get("TO", "").strip().lower().replace(" ", "_")
                if fr and to:
                    horiz_list.append({"id": key, "FROM": fr, "TO": to})

        log(f"Processing {broker_raw_name} → Trendlines: {len(trend_list)} | Horizontal: {len(horiz_list)}")

        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue

            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue

                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")

                if not os.path.exists(chart_path) or not os.path.exists(json_path):
                    continue

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        candles = json.load(f)  # index 0 = newest
                except:
                    continue

                img, raw_positions = get_candle_positions(chart_path)
                if img is None:
                    continue

                # Map: candle_number → position (correct for your data)
                positions = {}
                for i, candle in enumerate(candles):
                    if i >= len(raw_positions):
                        break
                    cnum = candle["candle_number"]
                    pos = raw_positions[i]
                    positions[cnum] = {"x": pos["x"], "high_y": pos["high_y"], "low_y": pos["low_y"]}

                anything_drawn = False

                # ====================== DRAW MARKERS (oldest → newest) ======================
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy = positions[cnum]["high_y"]
                    ly = positions[cnum]["low_y"]

                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                        anything_drawn = True
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                        anything_drawn = True
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                        anything_drawn = True
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                        anything_drawn = True
                    if candle.get("is_fvg_middle"):
                        dir_key = "fvg_middle_(bullish)" if candle.get("fvg_direction", "").lower() == "bullish" else "fvg_middle_(bearish)"
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP.get(dir_key, (180,180,180)), -1)
                        anything_drawn = True

                # ====================== DRAW TRENDLINES ======================
                for conf in trend_list:
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    line_id = f"T{conf['id']}"
                    color = get_color(from_key)

                    from_candle = None
                    to_candle = None

                    for candle in reversed(candles):
                        if is_level_match(candle, from_key):
                            from_candle = candle
                            break
                    if not from_candle or from_candle["candle_number"] not in positions:
                        continue

                    fx = positions[from_candle["candle_number"]]["x"]
                    fy = get_y_position(positions, from_candle["candle_number"], from_key)

                    if to_key not in ["ray", "none", "full"]:
                        found = False
                        for candle in reversed(candles):
                            if candle["candle_number"] == from_candle["candle_number"]:
                                found = True
                                continue
                            if found and is_level_match(candle, to_key):
                                to_candle = candle
                                break

                    tx = positions[to_candle["candle_number"]]["x"] if to_candle else img.shape[1] - 30
                    ty = get_y_position(positions, to_candle["candle_number"], to_key) if to_candle else fy

                    cv2.line(img, (fx, fy), (tx, ty), color, 2)
                    label_x = tx + 10 if tx > fx else fx + 10
                    label_y = ty - 15 if fy < ty else ty + 20
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    anything_drawn = True

                # ====================== DRAW HORIZONTAL LINES (stops at TO) ======================
                for conf in horiz_list:
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    line_id = f"H{conf['id']}"
                    color = get_color(from_key)

                    from_candle = None
                    to_candle = None

                    for candle in reversed(candles):
                        if is_level_match(candle, from_key):
                            from_candle = candle
                            break
                    if not from_candle or from_candle["candle_number"] not in positions:
                        continue

                    fx = positions[from_candle["candle_number"]]["x"]
                    fy = get_y_position(positions, from_candle["candle_number"], from_key)

                    found_from = False
                    for candle in reversed(candles):
                        if candle["candle_number"] == from_candle["candle_number"]:
                            found_from = True
                            continue
                        if found_from and is_level_match(candle, to_key):
                            to_candle = candle
                            break

                    if not to_candle:
                        continue

                    tx = positions[to_candle["candle_number"]]["x"]
                    cv2.line(img, (fx, fy), (tx, fy), color, 2)
                    cv2.putText(img, line_id, (tx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    anything_drawn = True

                # ====================== SAVE RESULT ======================
                if anything_drawn:
                    cv2.imwrite(output_path, img)
                    log(f"CUSTOM CHART DRAWN → {symbol_folder}/{tf_folder}", "SUCCESS")
                else:
                    log(f"Nothing to draw → {symbol_folder}/{tf_folder}", "INFO")

    log("=== CUSTOM_CHART() – MARKERS + TRENDLINES + HORIZONTAL LINES – COMPLETE ====", "SUCCESS")

def custom_trendline():
    import cv2
    import numpy as np
    import os
    import json
    from datetime import datetime
    import pytz
    import re

    # --- INITIAL SETUP ---
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ==================================================================
    # LEVEL FAMILY CLASSIFICATION (UNCHANGED CORE LOGIC)
    # ==================================================================
    BEARISH_FAMILY = {"ph", "ch"}
    BULLISH_FAMILY = {"pl", "cl"}
    
    # NEW: Dynamic family mapping based on input key
    def get_level_family(key):
        if key in BEARISH_FAMILY: return "bearish"
        if key in BULLISH_FAMILY: return "bullish"
        return None

    # This map is used to find opposite levels (e.g., ph -> {pl, cl})
    OPPOSITE_MAP = {
        "ph": {"pl", "cl"}, "ch": {"pl", "cl"},
        "pl": {"ph", "ch"}, "cl": {"ph", "ch"}
    }
    
    def is_bearish_level(key): return key in BEARISH_FAMILY
    def is_bullish_level(key): return key in BULLISH_FAMILY
    def is_level_match(candle, key): return candle.get(f"is_{key}", False)
    
    PARENT_LEVELS = {"ph", "pl"}
    CHILD_LEVELS = {"ch", "cl"}
    def is_parent_level(key): return key in PARENT_LEVELS
    def is_child_level(key): return key in CHILD_LEVELS
    
    COLOR_MAP = {
        "ph": (255, 100, 0), # Orange
        "pl": (200, 0, 200), # Purple
        "ch": (255, 200, 0), # Yellow
        "cl": (0, 140, 255), # Blue
        "fvg_middle": (0, 255, 0),
    }
    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))
        
    def get_y_position(positions, candle_num, key):
        pos = positions[candle_num]
        return pos["high_y"] if key in ["ph", "ch", "fvg_middle"] else pos["low_y"]

    # ------------------------------------------------------------------
    # MARKERS (unchanged)
    # ------------------------------------------------------------------
    def mark_extreme_interceptor(img, x, body_y, color):
        cv2.circle(img, (x, body_y), 18, color, 4)
        cv2.circle(img, (x, body_y), 14, (0, 255, 255), 2)
        arrow_start_x = x - 70
        arrow_end_x = x - 20
        cv2.arrowedLine(img, (arrow_start_x, body_y), (arrow_end_x, body_y), color, thickness=4, tipLength=0.3)

    def mark_breakout_candle(img, x, body_y, color):
        label_x = x - 38
        label_y = body_y + 6
        cv2.putText(img, "B", (label_x + 1, label_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, "B", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "B", (label_x - 1, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def draw_opposition_arrow(img, x, y, color, direction_up=True):
        size = 20
        thickness = 3
        shaft_length = 40
        if direction_up:
            cv2.line(img, (x, y + size), (x, y + size + shaft_length), color, thickness)
            pts = np.array([[x, y + size], [x - 12, y + size + 12], [x + 12, y + size + 12]], np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.line(img, (x, y - size), (x, y - size - shaft_length), color, thickness)
            pts = np.array([[x, y - size], [x - 12, y - size - 12], [x + 12, y - size - 12]], np.int32)
            cv2.fillPoly(img, [pts], color)

    def draw_double_retest_arrow(img, x, y_price, color, direction_up=True):
        offset = 12
        draw_opposition_arrow(img, x - offset, y_price, color, direction_up=direction_up)
        draw_opposition_arrow(img, x + offset, y_price, color, direction_up=direction_up)

    def draw_target_zone_marker(img, x, y_price, color, size=10):
        cv2.rectangle(img, (x - size, y_price - size), (x + size, y_price + size), color, -1)

    # ------------------------------------------------------------------
    # Candle positions from chart.png (unchanged)
    # ------------------------------------------------------------------
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None: return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        bounds = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            bounds[idx] = {"x": center_x, "high_y": y, "low_y": y + h, "width": w}
        return img.copy(), bounds

    # ------------------------------------------------------------------
    # Line–rectangle intersection (unchanged)
    # ------------------------------------------------------------------
    def line_intersects_rect(x1, y1, x2, y2, rect_left, rect_top, rect_right, rect_bottom):
        expand = 6
        rect_left -= expand; rect_right += expand; rect_top -= expand; rect_bottom += expand
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10: return 0
            return 1 if val > 0 else 2
        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2); o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1); o4 = orientation(p2, q2, q1)
            if o1 != o2 and o3 != o4: return True
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True
            return False
        edges = [((rect_left, rect_top), (rect_right, rect_top)),
                 ((rect_right, rect_top), (rect_right, rect_bottom)),
                 ((rect_right, rect_bottom), (rect_left, rect_bottom)),
                 ((rect_left, rect_bottom), (rect_left, rect_top))]
        for p2, q2 in edges:
            if do_intersect((x1, y1), (x2, y2), p2, q2):
                return True
        return False
        
    # ------------------------------------------------------------------
    # MAIN LOOP (with dynamic family and key loading)
    # ------------------------------------------------------------------
    developer_brokers = {k: v for k, v in globals().get("brokersdictionary", {}).items() if v.get("POSITION", "").lower() == "developer"}
    if not developer_brokers:
        log("No developer brokers found!", "ERROR")
        return
    
    global_tech_data = {}
    
    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "technique.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "technique.json")
        if not os.path.exists(technique_path):
            log(f"technique.json missing → {broker_raw_name}", "WARNING")
            continue
        
        with open(technique_path, 'r', encoding='utf-8') as f:
            tech = json.load(f)
            global_tech_data = tech # Store for later access
            
        if str(tech.get("drawings_switch", {}).get("trendline", "no")).strip().lower() != "yes":
            continue
        
        trend_configs = tech.get("trendline", {})
        trend_list = []
        for key in sorted([k for k in trend_configs.keys() if str(k).isdigit()]):
            conf = trend_configs[key]
            if not isinstance(conf, dict): continue
            
            # --- DYNAMIC CONFIGURATION LOADING ---
            fr = conf.get("FROM", "").strip().lower()
            direction = conf.get("DIRECTION", "").strip().lower()
            trend_family = conf.get("TREND", "").strip().lower() # NEW: Load the trend family
            
            rules = conf.get("rules", {})
            breakout_cond = rules.get("breakout_condition", "").strip().lower()
            
            # Load the dynamic point keys
            define_points = rules.get("define_trend_points", {})
            sender_key_name = define_points.get("sender_candle", "").strip().lower()
            receiver_key_name = define_points.get("receiver_candle", "").strip().lower()
            opposition_key_name = define_points.get("opposition_candle", "").strip().lower()
            retest_key_name = define_points.get("retest_candle", "").strip().lower()
            
            seq_count = 0
            if breakout_cond and "_sequence_candle" in breakout_cond:
                match = re.search(r"(\d+)_sequence_candle", breakout_cond)
                if match:
                    seq_count = int(match.group(1))

            if fr:
                trend_list.append({
                    "id": key,
                    "FROM": fr,
                    "TO": conf.get("TO", "ray").strip().lower(),
                    "rule": rules.get("extreme_intruder", "continue").strip().lower(),
                    "sender_condition": rules.get("sender_condition", "none").strip().lower(),
                    "interceptor_enabled": str(conf.get("INTERCEPTOR", "no")).strip().lower() == "yes",
                    "direction": direction,
                    "breakout_sequence_count": seq_count if direction == "breakout" and seq_count > 0 else 0,
                    # NEW DYNAMIC KEYS
                    "trend_family": trend_family,
                    "point_keys": {
                        "sender": sender_key_name,
                        "receiver": receiver_key_name,
                        "opposition": opposition_key_name,
                        "retest": retest_key_name,
                    }
                })

        if not trend_list:
            continue
            
        # Get Neighbor Right settings
        parent_neighbor_right = global_tech_data.get("parenthighsandlows", {}).get("NEIGHBOR_RIGHT", 15)
        child_neighbor_right = global_tech_data.get("childhighsandlows", {}).get("NEIGHBOR_RIGHT", 7)
        
        log(f"Processing {broker_raw_name} → {len(trend_list)} institutional trendlines (Parent NR={parent_neighbor_right}, Child NR={child_neighbor_right})")
        
        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue
            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue
                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")
                report_path = os.path.join(tf_path, "custom_levels.json")
                if not all(os.path.exists(p) for p in [chart_path, json_path]):
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    candles = json.load(f)
                img, raw_positions = get_candle_positions(chart_path)
                if img is None: continue
                positions = {}
                for idx, data in sorted(raw_positions.items(), key=lambda x: x[1]["x"], reverse=True):
                    if idx < len(candles):
                        cnum = candles[idx]["candle_number"]
                        positions[cnum] = data
                
                # Draw level markers (unchanged)
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy, ly = positions[cnum]["high_y"], positions[cnum]["low_y"]
                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                    if candle.get("is_fvg_middle"):
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP["fvg_middle"], -1)
                
                final_teams = {}
                final_trendlines_for_redraw = []
                
                def draw_trendline(line_id, fx, fy, tx, ty, color, extreme_cnum=None, extreme_y=None):
                    cv2.line(img, (fx, fy), (tx, ty), color, 3)
                    label_x = tx + 15 if tx > fx else fx + 15
                    label_y = ty - 20 if fy < ty else ty + 25
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    if extreme_cnum and extreme_y is not None:
                        ex_x = positions[extreme_cnum]["x"]
                        pts = np.array([[ex_x, extreme_y - 15], [ex_x - 10, extreme_y], [ex_x + 10, extreme_y]], np.int32)
                        cv2.fillPoly(img, [pts], color)
                        
                def validate_sender_condition(sender_cnum, receiver_cnum, key, condition):
                    if condition == "none": return True
                    if sender_cnum not in positions or receiver_cnum not in positions: return False
                    sender_candle = next(c for c in candles if c["candle_number"] == sender_cnum)
                    receiver_candle = next(c for c in candles if c["candle_number"] == receiver_cnum)
                    if is_bearish_level(key):
                        return sender_candle["high"] >= receiver_candle["high"] if condition == "beyond" else sender_candle["high"] <= receiver_candle["high"]
                    elif is_bullish_level(key):
                        return sender_candle["low"] <= receiver_candle["low"] if condition == "beyond" else sender_candle["low"] >= receiver_candle["low"]
                    return True
                    
                def process_trendline(conf, depth=0, max_depth=50):
                    if depth > max_depth:
                        log(f"Max recursion depth for T{conf['id']}", "WARNING")
                        return False
                    
                    line_id = f"T{conf['id']}"
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    rule = conf["rule"]
                    sender_condition = conf["sender_condition"]
                    interceptor_enabled = conf["interceptor_enabled"]
                    direction = conf["direction"]
                    breakout_seq_count = conf["breakout_sequence_count"]
                    color = get_color(from_key)
                    
                    # --- CORE: FINDING FROM/SENDER POINT ---
                    from_candle = next((c for c in reversed(candles) if is_level_match(c, from_key)), None)
                    if not from_candle or from_candle["candle_number"] not in positions:
                        return False
                        
                    from_cnum = from_candle["candle_number"]
                    fx = positions[from_cnum]["x"]
                    fy = get_y_position(positions, from_cnum, from_key)
                    
                    # --- CORE: FINDING TO/RECEIVER POINT ---
                    to_cnum = None
                    tx, ty = img.shape[1] - 30, fy # Default ray endpoint
                    
                    found_from = False
                    for c in reversed(candles):
                        if c["candle_number"] == from_cnum:
                            found_from = True
                            continue
                        if found_from and (to_key == "ray" or is_level_match(c, to_key)):
                            to_cnum = c["candle_number"]
                            if to_cnum in positions:
                                tx = positions[to_cnum]["x"]
                                ty = get_y_position(positions, to_cnum, to_key)
                            break
                            
                    is_ray = (to_cnum is None)

                    # --- INTERMEDIATE TOUCHES FOR EXTREME INTRUDER RULE ---
                    touches = []
                    min_c = min(from_cnum, to_cnum or from_cnum + 99999)
                    max_c = max(from_cnum, to_cnum or from_cnum + 99999)
                    
                    for c in candles:
                        cn = c["candle_number"]
                        if cn in [from_cnum, to_cnum] or cn not in positions: continue
                        if not (min_c <= cn <= max_c): continue
                        
                        pos = positions[cn]
                        if line_intersects_rect(fx, fy, tx, ty,
                                               pos["x"] - pos["width"]//2, pos["high_y"],
                                               pos["x"] + pos["width"]//2, pos["low_y"]):
                            touches.append(cn)
                            
                    extreme_cnum = extreme_y = None
                    if touches:
                        s_min, s_max = min(touches), max(touches)
                        if is_bearish_level(from_key):
                            best = max((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["high"], default=None)
                        else:
                            best = min((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["low"], default=None)
                            
                        if best:
                            extreme_cnum = best["candle_number"]
                            extreme_y = positions[extreme_cnum]["high_y"] if is_bearish_level(from_key) else positions[extreme_cnum]["low_y"]
                            
                    # --- APPLYING EXTREME INTRUDER RULE (new_from/new_to) ---
                    final_fx, final_fy = fx, fy
                    final_tx, final_ty = tx, ty
                    final_from_cnum = from_cnum
                    final_to_cnum = to_cnum
                    applied_rule = "continue"
                    
                    if rule == "new_from" and extreme_cnum:
                        final_fx = positions[extreme_cnum]["x"]
                        final_fy = extreme_y
                        final_from_cnum = extreme_cnum
                        applied_rule = "new_from"
                    elif rule == "new_to" and extreme_cnum:
                        final_tx = positions[extreme_cnum]["x"]
                        final_ty = extreme_y
                        final_to_cnum = extreme_cnum
                        applied_rule = "new_to"
                        
                    # --- VALIDATE SENDER CONDITION ---
                    sender_cnum = final_from_cnum
                    receiver_cnum = final_to_cnum if not is_ray else final_from_cnum # For condition, if ray, receiver is sender
                    if not validate_sender_condition(sender_cnum, receiver_cnum, from_key, sender_condition):
                        return False
                        
                    # --- DRAW INITIAL LINE AND STORE FOR FINAL PROCESSING ---
                    draw_trendline(line_id, int(final_fx), int(final_fy), int(final_tx), int(final_ty), color,
                                    extreme_cnum, extreme_y if rule in ["new_from", "new_to"] else None)
                                    
                    final_trendlines_for_redraw.append({
                        "line_id": line_id,
                        "from_x": int(final_fx),
                        "from_y": int(final_fy),
                        "to_x": int(final_tx),
                        "to_y": int(final_ty),
                        "receiver_cnum": final_to_cnum if final_to_cnum else final_from_cnum, # Use the actual 'TO' or 'FROM' if ray
                        "from_key": from_key,
                        "color": color,
                        "interceptor_enabled": interceptor_enabled,
                        "direction": direction,
                        "breakout_sequence_count": breakout_seq_count,
                        "trend_family": conf["trend_family"], # NEW: Store trend family
                        "point_keys": conf["point_keys"],     # NEW: Store dynamic point keys
                    })
                    
                    final_teams[line_id] = {"team": {
                        "trendline_info": {
                            "line_id": line_id,
                            "from_candle": final_from_cnum,
                            "to_candle": final_to_cnum,
                            "receiver_candle": final_to_cnum if final_to_cnum else final_from_cnum,
                            "is_ray": is_ray,
                            "intermediate_touches": len(touches),
                            "touched_candles": touches,
                            "extreme_intruder_candle": extreme_cnum,
                            "rule_applied": applied_rule,
                            "color": list(map(int, color)),
                            "interceptors": [],
                            "opposition_candle": None,
                            "extreme_interceptor_candle": None,
                            "breakout_sequence_candles": [],
                            "retest_candle": None,
                            "target_zone_candle": None 
                        }
                    }}
                    return True
                    
                for conf in trend_list:
                    process_trendline(conf)
                    
                # ==================================================================
                # FINAL PROCESSING — INTERCEPTORS, OPPOSITION, RETEST, & TARGET ZONE
                # ==================================================================
                for trend in final_trendlines_for_redraw:
                    fx, fy = trend["from_x"], trend["from_y"]
                    tx, ty = trend["to_x"], trend["to_y"]
                    color = trend["color"]
                    line_id = trend["line_id"]
                    receiver_cnum = trend["receiver_cnum"]
                    from_key = trend["from_key"]
                    direction = trend["direction"]
                    seq_count = trend["breakout_sequence_count"]
                    
                    # NEW: Get dynamic key for the opposition candle
                    opposition_key_name = trend["point_keys"]["opposition"]
                    retest_key_name = trend["point_keys"]["retest"]
                    
                    if tx - fx == 0:
                        continue
                        
                    # 1. Redraw Ray
                    slope = (ty - fy) / (tx - fx)
                    extend_x = img.shape[1] - 10
                    extend_y = int(fy + slope * (extend_x - fx))
                    cv2.line(img, (fx, fy), (extend_x, extend_y), color, 3)
                    cv2.putText(img, line_id, (fx + 20, fy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    
                    # Mark Receiver Candle
                    if receiver_cnum and receiver_cnum in positions:
                        rx = positions[receiver_cnum]["x"]
                        ry = get_y_position(positions, receiver_cnum, from_key)
                        cv2.circle(img, (rx, ry), 12, (0, 0, 0), -1)
                        cv2.putText(img, "R", (rx - 8, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    # 2. Interceptors
                    interceptors = []
                    receiver_x = positions[receiver_cnum]["x"] if receiver_cnum in positions else -99999
                    for c in candles:
                        cn = c["candle_number"]
                        if cn not in positions: continue
                        pos = positions[cn]
                        if pos["x"] <= receiver_x: continue
                        if line_intersects_rect(fx, fy, extend_x, extend_y,
                                                pos["x"] - pos["width"]//2, pos["high_y"],
                                                pos["x"] + pos["width"]//2, pos["low_y"]):
                            body_y = (pos["high_y"] + pos["low_y"]) // 2
                            cv2.circle(img, (pos["x"], body_y), 14, color, 4)
                            interceptors.append({
                                "candle_number": cn,
                                "x": pos["x"],
                                "y": body_y,
                                "high": c["high"],
                                "low": c["low"],
                                "close": c["close"]
                            })
                            
                    # 3. Opposition Candle
                    opposition_cnum = None
                    if interceptors and receiver_cnum:
                        oldest_int_cnum = min(i["candle_number"] for i in interceptors)
                        
                        # Find the actual level of the receiver (ph, pl, ch, cl)
                        receiver_candle = next(c for c in candles if c["candle_number"] == receiver_cnum)
                        receiver_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                        
                        if receiver_level and opposition_key_name: # Check for the dynamic key name
                            # The target levels for opposition are the *opposite* family, but must be the *type* defined in the JSON.
                            # The JSON gives the family (e.g., "bullish"), we need the actual level keys (e.g., "pl", "cl").
                            
                            target_levels = []
                            if opposition_key_name == "bullish":
                                target_levels = BULLISH_FAMILY
                            elif opposition_key_name == "bearish":
                                target_levels = BEARISH_FAMILY

                            # Enforce: Parent receivers ignore child opposition (Parent/Child Hierarchy)
                            if is_parent_level(receiver_level):
                                target_levels = {lvl for lvl in target_levels if is_parent_level(lvl)}
                                
                            for cn in range(receiver_cnum - 1, oldest_int_cnum, -1):
                                if cn <= 0: break
                                candle = next((c for c in candles if c["candle_number"] == cn), None)
                                if candle and any(candle.get(f"is_{lvl}") for lvl in target_levels):
                                    opposition_cnum = cn
                                    break
                                    
                    if opposition_cnum and opposition_cnum in positions:
                        opp_x = positions[opposition_cnum]["x"]
                        opp_candle = next(c for c in candles if c["candle_number"] == opposition_cnum)
                        # Find the actual level of the opposition candle
                        opp_level = next(k for k in ["ph","ch","pl","cl"] if opp_candle.get(f"is_{k}"))
                        direction_up = opp_level in {"pl", "cl"}
                        arrow_y = positions[opposition_cnum]["low_y"] if direction_up else positions[opposition_cnum]["high_y"]
                        draw_opposition_arrow(img, opp_x, arrow_y, color, direction_up=direction_up)
                        
                    # 4. Extreme Interceptor
                    extreme_interceptor_cnum = None
                    if opposition_cnum and interceptors:
                        pre_opposition = [i for i in interceptors if i["candle_number"] < opposition_cnum]
                        if pre_opposition:
                            extreme_interceptor_cnum = max(i["candle_number"] for i in pre_opposition)
                            
                    if extreme_interceptor_cnum and extreme_interceptor_cnum in positions:
                        ex_int = next(i for i in interceptors if i["candle_number"] == extreme_interceptor_cnum)
                        mark_extreme_interceptor(img, ex_int["x"], ex_int["y"], color)
                        
                    # 5. BREAKOUT SEQUENCE
                    breakout_sequence_cnums = []
                    if direction == "breakout" and seq_count > 0 and extreme_interceptor_cnum:
                        ext_candle = next(c for c in candles if c["candle_number"] == extreme_interceptor_cnum)
                        ext_high = ext_candle["high"]
                        ext_low = ext_candle["low"]
                        younger_candles = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                        younger_candles.sort(key=lambda x: x["candle_number"], reverse=True)
                        
                        for i in range(len(younger_candles) - seq_count + 1):
                            seq = younger_candles[i:i + seq_count]
                            if is_bullish_level(from_key): # Bullish trendline (support) -> look for break below (Bearish breakout)
                                if all(c["high"] < ext_high and c["low"] < ext_low for c in seq):
                                    breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                    break
                            else: # Bearish trendline (resistance) -> look for break above (Bullish breakout)
                                if all(c["high"] > ext_high and c["low"] > ext_low for c in seq):
                                    breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                    break
                                
                    for cnum in breakout_sequence_cnums:
                        if cnum in positions:
                            body_y = (positions[cnum]["high_y"] + positions[cnum]["low_y"]) // 2
                            mark_breakout_candle(img, positions[cnum]["x"], body_y, color)
                            
                    # 6. RETEST CANDLE
                    retest_cnum = None
                    if breakout_sequence_cnums and extreme_interceptor_cnum and retest_key_name:
                        younger_candles = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                        younger_candles.sort(key=lambda x: x["candle_number"], reverse=True)
                        
                        receiver_candle = next(c for c in candles if c["candle_number"] == receiver_cnum)
                        receiver_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                        
                        if receiver_level:
                            # The retest is an opposite move back to the trendline. 
                            # The JSON 'retest_candle' specifies the FAMILY of the retest level.
                            
                            allowed_levels = set()
                            if retest_key_name == "bullish":
                                allowed_levels = BULLISH_FAMILY
                            elif retest_key_name == "bearish":
                                allowed_levels = BEARISH_FAMILY
                                
                            # Enforce: Parent receivers ignore child retests (Parent/Child Hierarchy)
                            if is_parent_level(receiver_level):
                                allowed_levels = {lvl for lvl in allowed_levels if is_parent_level(lvl)}

                            for c in younger_candles:
                                if c["candle_number"] in breakout_sequence_cnums: continue
                                # Start searching from the youngest candle
                                if any(c.get(f"is_{lvl}") for lvl in allowed_levels):
                                    retest_cnum = c["candle_number"]
                                    break
                                    
                    if retest_cnum and retest_cnum in positions:
                        retest_candle = next(c for c in candles if c["candle_number"] == retest_cnum)
                        level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                        is_bullish_retest = level in {"cl", "pl"}
                        y_price = positions[retest_cnum]["low_y"] if is_bullish_retest else positions[retest_cnum]["high_y"]
                        draw_double_retest_arrow(img, positions[retest_cnum]["x"], y_price, color, direction_up=is_bullish_retest)
                        
                    # 7. TARGET ZONE CANDLE LOGIC
                    target_zone_cnum = None
                    neighbor_right = 0 # Initialize for reporting
                    
                    if retest_cnum and retest_cnum in positions:
                        retest_candle = next(c for c in candles if c["candle_number"] == retest_cnum)
                        level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                        
                        if level in PARENT_LEVELS:
                            neighbor_right = parent_neighbor_right
                        elif level in CHILD_LEVELS:
                            neighbor_right = child_neighbor_right
                            
                        if neighbor_right > 0:
                            # Target is 'neighbor_right' candles *after* the retest candle.
                            # Decreasing candle numbers means forward in time.
                            target_cnum = retest_cnum - neighbor_right
                            
                            if target_cnum in positions:
                                target_zone_cnum = target_cnum
                                target_x = positions[target_cnum]["x"]
                                
                                # Determine the Y position for the marker
                                if is_bullish_level(from_key): # Bullish Trendline (Entry is Long) -> Target is above
                                    target_y = positions[target_cnum]["high_y"] - 20 
                                else: # Bearish Trendline (Entry is Short) -> Target is below
                                    target_y = positions[target_cnum]["low_y"] + 20
                                    
                                draw_target_zone_marker(img, target_x, target_y, color)

                    # 8. Save to JSON
                    if line_id in final_teams:
                        final_teams[line_id]["team"]["trendline_info"]["interceptors"] = interceptors
                        
                        if opposition_cnum:
                            opp_candle = next(c for c in candles if c["candle_number"] == opposition_cnum)
                            level = next(k for k in ["ph","ch","pl","cl"] if opp_candle.get(f"is_{k}"))
                            final_teams[line_id]["team"]["trendline_info"]["opposition_candle"] = {
                                "candle_number": opposition_cnum,
                                "x": positions[opposition_cnum]["x"],
                                "y": get_y_position(positions, opposition_cnum, level),
                                "level": level
                            }
                            
                        if extreme_interceptor_cnum:
                            final_teams[line_id]["team"]["trendline_info"]["extreme_interceptor_candle"] = {
                                "candle_number": extreme_interceptor_cnum,
                                "x": positions[extreme_interceptor_cnum]["x"],
                                "y": (positions[extreme_interceptor_cnum]["high_y"] + positions[extreme_interceptor_cnum]["low_y"]) // 2
                            }
                            
                        if breakout_sequence_cnums:
                            final_teams[line_id]["team"]["trendline_info"]["breakout_sequence_candles"] = [
                                {"candle_number": cnum, "x": positions[cnum]["x"], "y": (positions[cnum]["high_y"] + positions[cnum]["low_y"]) // 2}
                                for cnum in breakout_sequence_cnums
                            ]
                            
                        if retest_cnum:
                            retest_candle = next(c for c in candles if c["candle_number"] == retest_cnum)
                            level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                            final_teams[line_id]["team"]["trendline_info"]["retest_candle"] = {
                                "candle_number": retest_cnum,
                                "x": positions[retest_cnum]["x"],
                                "y": get_y_position(positions, retest_cnum, level),
                                "level": level
                            }
                            
                        if target_zone_cnum: # Save new target zone info
                            final_teams[line_id]["team"]["trendline_info"]["target_zone_candle"] = {
                                "candle_number": target_zone_cnum,
                                "x": positions[target_zone_cnum]["x"],
                                "level_type": level, # This is the level type of the *Retest* candle (ph/pl/ch/cl) which determined NR
                                "neighbor_right": neighbor_right
                            }

                cv2.imwrite(output_path, img)
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(final_teams, f, indent=2, ensure_ascii=False)
                log(f"→ {symbol_folder}/{tf_folder} | {len(final_teams)} Trendlines processed", "SUCCESS")
                
    log("=== INSTITUTIONAL TRENDLINE ENGINE v10.1 — DYNAMIC CONFIGURATION & TARGET ZONE ADDED ===", "SUCCESS")
       
def trendline_interceptor():
    import cv2
    import json
    import os
    from datetime import datetime
    import pytz

    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    developer_brokers = {k: v for k, v in brokersdictionary.items() if v.get("POSITION", "").lower() == "developer"}
    if not developer_brokers:
        log("No developer brokers found!", "ERROR")
        return

    processed_count = 0

    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]

        # Check if trendlines are enabled
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "technique.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "technique.json")
        if not os.path.exists(technique_path):
            continue

        with open(technique_path, 'r', encoding='utf-8') as f:
            tech = json.load(f)

        if str(tech.get("drawings_switch", {}).get("trendline", "no")).strip().lower() != "yes":
            continue

        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path):
                continue

            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path):
                    continue

                chart_path = os.path.join(tf_path, "chart_custom.png")
                report_path = os.path.join(tf_path, "custom_levels.json")

                if not os.path.exists(chart_path) or not os.path.exists(report_path):
                    continue

                img = cv2.imread(chart_path)
                if img is None:
                    continue

                with open(report_path, 'r', encoding='utf-8') as f:
                    final_teams = json.load(f)

                if not final_teams:
                    continue

                updated = False
                img_height, img_width = img.shape[:2]

                for line_id, data in final_teams.items():
                    info = data["team"]["trendline_info"]

                    fx = info.get("from_x")
                    fy = info.get("from_y")
                    tx = info.get("to_x")
                    ty = info.get("to_y")
                    color = tuple(info["color"])

                    # Skip if pixel coordinates are missing (shouldn't happen after custom_trendline fix)
                    if None in (fx, fy, tx, ty):
                        continue

                    fx, fy, tx, ty = int(fx), int(fy), int(tx), int(ty)

                    # Avoid division by zero
                    dx = tx - fx
                    if abs(dx) < 5:
                        dx = 5 if dx >= 0 else -5

                    slope = (ty - fy) / dx
                    extend_x = img_width - 20
                    extend_y = int(fy + slope * (extend_x - fx))

                    # Draw infinite ray
                    cv2.line(img, (fx, fy), (extend_x, extend_y), color, 3)

                    # Label near the starting point (clean and always visible)
                    cv2.putText(img, line_id, (fx + 15, fy - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                    # Mark receiver candle with "R" if exists
                    if info.get("to_candle") is not None:
                        rx, ry = tx, ty
                        cv2.circle(img, (rx, ry), 11, (0, 0, 0), -1)        # black circle
                        cv2.putText(img, "R", (rx - 9, ry + 9),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    updated = True

                if updated:
                    cv2.imwrite(chart_path, img)
                    processed_count += 1

    log(f"EXTENDED {processed_count} CHARTS → INFINITE INSTITUTIONAL TRENDLINES + 'R' MARKERS", "SUCCESS")
    

def custom_horizontal_line():
    import cv2
    import numpy as np
    import os
    import json
    from datetime import datetime
    import pytz

    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ------------------------------------------------------------------
    # PROFESSIONAL COLORS (same as trendline & markers)
    # ------------------------------------------------------------------
    COLOR_MAP = {
        "ph": (255, 100, 0),       # Orange
        "pl": (200, 0, 200),       # Magenta
        "ch": (255, 200, 0),       # Cyan
        "cl": (0, 140, 255),       # Warm Orange
        "fvg_middle_(bullish)": (0, 255, 0),
        "fvg_middle_(bearish)": (60, 20, 220),
    }

    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))

    def is_level_match(candle, key):
        if key == "ph": return candle.get("is_ph")
        if key == "pl": return candle.get("is_pl")
        if key == "ch": return candle.get("is_ch")
        if key == "cl": return candle.get("is_cl")
        if key == "fvg_middle_(bullish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bullish"
        if key == "fvg_middle_(bearish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bearish"
        return False

    def get_y_position(positions, candle_num, key):
        pos = positions[candle_num]
        if key in ["ph", "ch", "fvg_middle_(bullish)"]:
            return pos["high_y"]
        else:
            return pos["low_y"]

    # ------------------------------------------------------------------
    # Get candle positions (right = newest = candle_number 0)
    # ------------------------------------------------------------------
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None:
            return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        raw_positions = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            raw_positions[idx] = {"x": x + w // 2, "high_y": y, "low_y": y + h}
        return img.copy(), raw_positions

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    developer_brokers = {
        k: v for k, v in brokersdictionary.items()
        if v.get("POSITION", "").lower() == "developer"
    }

    if not developer_brokers:
        log("No developer brokers found!", "ERROR")
        return

    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "technique.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "technique.json")
        if not os.path.exists(technique_path):
            log(f"technique.json missing → {broker_raw_name}", "WARNING")
            continue

        try:
            with open(technique_path, 'r', encoding='utf-8') as f:
                tech = json.load(f)
        except Exception as e:
            log(f"Failed loading technique.json: {e}", "ERROR")
            continue

        if str(tech.get("drawings_switch", {}).get("horizontal_line", "no")).strip().lower() != "yes":
            log(f"Horizontal lines disabled for {broker_raw_name}", "INFO")
            continue

        horiz_configs = tech.get("horizontal_line", {})
        horiz_list = []
        for key in sorted([k for k in horiz_configs.keys() if str(k).isdigit()]):
            conf = horiz_configs[key]
            if not isinstance(conf, dict): continue
            fr = conf.get("FROM", "").strip().lower().replace(" ", "_")
            to = conf.get("TO", "").strip().lower().replace(" ", "_")
            if fr and to:
                horiz_list.append({"id": key, "FROM": fr, "TO": to})

        if not horiz_list:
            log(f"No horizontal lines defined → {broker_raw_name}", "INFO")
            continue

        log(f"Processing {broker_raw_name} → {len(horiz_list)} horizontal lines")

        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue

            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue

                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")

                if not os.path.exists(chart_path) or not os.path.exists(json_path):
                    continue

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        candles = json.load(f)  # index 0 = newest → last = oldest
                except:
                    continue

                img, raw_positions = get_candle_positions(chart_path)
                if img is None:
                    continue

                # Build correct positions: candle_number → coordinates
                positions = {}
                for i, candle in enumerate(candles):
                    if i >= len(raw_positions):
                        break
                    cnum = candle["candle_number"]
                    pos = raw_positions[i]
                    positions[cnum] = {
                        "x": pos["x"],
                        "high_y": pos["high_y"],
                        "low_y": pos["low_y"]
                    }

                anything_drawn = False

                # DRAW MARKERS (same as before)
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy = positions[cnum]["high_y"]
                    ly = positions[cnum]["low_y"]

                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                        anything_drawn = True
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                        anything_drawn = True
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                        anything_drawn = True
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                        anything_drawn = True
                    if candle.get("is_fvg_middle"):
                        dir_key = "fvg_middle_(bullish)" if candle.get("fvg_direction", "").lower() == "bullish" else "fvg_middle_(bearish)"
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP.get(dir_key, (180,180,180)), -1)
                        anything_drawn = True

                # DRAW HORIZONTAL LINES — stops exactly at TO candle
                for conf in horiz_list:
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    line_id = f"H{conf['id']}"
                    color = get_color(from_key)

                    from_candle = None
                    to_candle = None

                    # Scan from oldest to newest
                    for candle in reversed(candles):
                        if is_level_match(candle, from_key):
                            from_candle = candle
                            break

                    if not from_candle or from_candle["candle_number"] not in positions:
                        continue

                    fx = positions[from_candle["candle_number"]]["x"]
                    fy = get_y_position(positions, from_candle["candle_number"], from_key)

                    # Find first TO candle AFTER the FROM candle
                    found_from = False
                    for candle in reversed(candles):
                        if candle["candle_number"] == from_candle["candle_number"]:
                            found_from = True
                            continue
                        if found_from and is_level_match(candle, to_key):
                            to_candle = candle
                            break

                    if not to_candle or to_candle["candle_number"] not in positions:
                        continue  # No valid TO found

                    tx = positions[to_candle["candle_number"]]["x"]
                    ty = get_y_position(positions, to_candle["candle_number"], to_key)

                    # Draw perfectly horizontal line from FROM.x to TO.x at FROM.y
                    cv2.line(img, (fx, fy), (tx, fy), color, 2)

                    # Label on the right side
                    label_x = tx + 10
                    label_y = fy
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    anything_drawn = True

                if anything_drawn:
                    cv2.imwrite(output_path, img)
                    log(f"HORIZONTAL LINES + MARKERS → {symbol_folder}/{tf_folder}", "SUCCESS")
                else:
                    log(f"Nothing drawn → {symbol_folder}/{tf_folder}", "INFO")

    log("=== CUSTOM HORIZONTAL LINE (STOPS AT TO) COMPLETED ===", "SUCCESS")

if __name__ == "__main__":
    custom_trendline()


        
        
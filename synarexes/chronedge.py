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


def save_oldest_newest(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save all candles + highlight the SECOND-MOST-RECENT candle (candle_number = 1)
    as 'x' in previous latest candle with age in days/hours.
    """
    error_log = []
    all_json_path = os.path.join(timeframe_folder, "all_oldest_newest_candles.json")
    latest_json_path = os.path.join(timeframe_folder, "previous_oldest_newest_latestcandle.json")
    
    # Use Lagos timezone consistently
    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    try:
        if len(df) < 2:
            error_msg = f"Not enough data for {symbol} ({timeframe_str})"
            log_and_print(error_msg, "ERROR")
            error_log.append({
                "error": error_msg,
                "timestamp": now.isoformat()
            })
            save_errors(error_log)
            return error_log

        # === 1. Prepare PH/PL lookup ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}

        # === 2. Save ALL candles (oldest = 0) ===
        all_candles = []
        for i, (ts, row) in enumerate(df[::-1].iterrows()):  # newest first → reverse → oldest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,  # 0 = oldest, 1 = second-most-recent, ..., N-1 = most recent
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL'
            })
            all_candles.append(candle)

        # Write all candles
        with open(all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_candles, f, indent=4)

        # === 3. Save CANDLE #1 (second-most-recent) as id: "x" with age ===
        if len(all_candles) < 2:
            raise ValueError("Expected at least 2 candles to extract candle_number 1")

        previous_latest_candle = all_candles[1].copy()  # candle_number == 1
        candle_time_str = previous_latest_candle["time"]
        candle_time = datetime.strptime(candle_time_str, '%Y-%m-%d %H:%M:%S')
        candle_time = lagos_tz.localize(candle_time)  # Make timezone-aware

        # Calculate age
        delta = now - candle_time
        total_hours = delta.total_seconds() / 3600

        if total_hours <= 24:
            age_str = f"{int(total_hours)} hour{'s' if int(total_hours) != 1 else ''} old"
        else:
            days = int(total_hours // 24)
            age_str = f"{days} day{'s' if days != 1 else ''} old"

        # Add age field
        previous_latest_candle["age"] = age_str
        previous_latest_candle["id"] = "x"

        # Remove candle_number as per original logic
        if "candle_number" in previous_latest_candle:
            del previous_latest_candle["candle_number"]

        # Write the highlighted candle
        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(previous_latest_candle, f, indent=4)

        # === 4. LOG SUCCESS ===
        log_and_print(
            f"SAVED {symbol} {timeframe_str}: "
            f"all_oldest_newest_candles.json ({len(all_candles)} candles) + "
            f"previouslatestcandle.json (candle_number=1 → id='x', {age_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_oldest_newest failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": now.isoformat()
        })
        save_errors(error_log)

    return error_log

def save_newest_oldest(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save all candles + highlight the SECOND-MOST-RECENT candle (candle_number = 1)
    as 'x' in previous latest candle  with age in days/hours.
    Candles are now numbered with 0 = newest (most recent), 1 = previous, etc.
    """
    error_log = []
    all_json_path = os.path.join(timeframe_folder, "all_newest_oldest_candles.json")
    latest_json_path = os.path.join(timeframe_folder, "previous_newest_oldest_latestcandle.json")
    
    # Use Lagos timezone consistently
    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    try:
        if len(df) < 2:
            error_msg = f"Not enough data for {symbol} ({timeframe_str})"
            log_and_print(error_msg, "ERROR")
            error_log.append({
                "error": error_msg,
                "timestamp": now.isoformat()
            })
            save_errors(error_log)
            return error_log

        # === 1. Prepare PH/PL lookup ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}

        # === 2. Save ALL candles (newest = 0) ===
        all_candles = []
        # Iterate from newest to oldest (no reverse needed)
        for i, (ts, row) in enumerate(df.iterrows()):  # newest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,  # 0 = newest, 1 = previous, 2 = two ago, ...
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL'
            })
            all_candles.append(candle)

        # Write all candles
        with open(all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_candles, f, indent=4)

        # === 3. Save CANDLE #1 (second-most-recent, i.e., one candle ago) as id: "x" with age ===
        if len(all_candles) < 2:
            raise ValueError("Expected at least 2 candles to extract candle_number 1")

        previous_latest_candle = all_candles[1].copy()  # candle_number == 1 (previous candle)
        candle_time_str = previous_latest_candle["time"]
        candle_time = datetime.strptime(candle_time_str, '%Y-%m-%d %H:%M:%S')
        candle_time = lagos_tz.localize(candle_time)  # Make timezone-aware

        # Calculate age
        delta = now - candle_time
        total_hours = delta.total_seconds() / 3600

        if total_hours <= 24:
            age_str = f"{int(total_hours)} hour{'s' if int(total_hours) != 1 else ''} old"
        else:
            days = int(total_hours // 24)
            age_str = f"{days} day{'s' if days != 1 else ''} old"

        # Add age field
        previous_latest_candle["age"] = age_str
        previous_latest_candle["id"] = "x"

        # Remove candle_number as per original logic
        if "candle_number" in previous_latest_candle:
            del previous_latest_candle["candle_number"]

        # Write the highlighted candle
        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(previous_latest_candle, f, indent=4)

        # === 4. LOG SUCCESS ===
        log_and_print(
            f"SAVED {symbol} {timeframe_str}: "
            f"all_newest_oldest_candles.json ({len(all_candles)} candles, 0=newest) + "
            f"previouslatestcandle.json (candle_number=1 → id='x', {age_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save newest to oldest failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": now.isoformat()
        })
        save_errors(error_log)

    return error_log

def save_next_oldest_newest_candles(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save candles that appear **after** the previous-latest candle (the one saved as 'x')
    i.e. candles with timestamp > timestamp of 'x' candle
    into <timeframe_folder>/next_oldest_newest_candles.json
    """
    error_log = []
    next_json_path = os.path.join(timeframe_folder, "next_oldest_newest_candles.json")

    try:
        if len(df) < 3:
            return error_log  # need at least: old, previous-latest, and one new

        # === 1. Build full ordered list: oldest → newest (candle_number 0 = oldest) ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}

        ordered_candles = []
        for i, (ts, row) in enumerate(df[::-1].iterrows()):  # newest first → reverse → oldest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,  # 0 = oldest, 1 = second-most-recent (x), ..., N-1 = newest
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL'
            })
            ordered_candles.append(candle)

        # === 2. Find the timestamp of the 'x' candle (candle_number == 1) ===
        if len(ordered_candles) < 2:
            return error_log

        x_candle_time_str = ordered_candles[1]["time"]  # candle_number 1
        x_time = datetime.strptime(x_candle_time_str, '%Y-%m-%d %H:%M:%S')

        # === 3. Collect all candles with timestamp > x_time ===
        next_candles = []
        for candle in ordered_candles:
            candle_time = datetime.strptime(candle["time"], '%Y-%m-%d %H:%M:%S')
            if candle_time > x_time:
                # Keep original candle_number (from full history context)
                next_candles.append(candle)

        if not next_candles:
            return error_log  # No newer candles

        # === 4. Write ===
        with open(next_json_path, 'w', encoding='utf-8') as f:
            json.dump(next_candles, f, indent=4)

        log_and_print(
            f"SAVED {symbol} {timeframe_str}: next_oldest_newest_candles.json "
            f"({len(next_candles)} candles after {x_candle_time_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_next_oldest_newest_candles failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).isoformat()
        })
        save_errors(error_log)

    return error_log

def save_next_newest_oldest_candles(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save candles that appear **after** the previous-latest candle (the one saved as 'x')
    i.e. candles with timestamp > timestamp of 'x' candle
    into <timeframe_folder>/next_newest_oldest_candles.json
    """
    error_log = []
    next_json_path = os.path.join(timeframe_folder, "next_newest_oldest_candles.json")

    try:
        if len(df) < 3:
            return error_log  # need at least: old, previous-latest, and one new

        # === 1. Build full ordered list: oldest → newest (candle_number 0 = oldest) ===
        ph_dict = {t: label for label, _, t in ph_labels}
        pl_dict = {t: label for label, _, t in pl_labels}

        ordered_candles = []
        for i, (ts, row) in enumerate(df[::-1].iterrows()):  # newest first → reverse → oldest first
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,  # 0 = oldest, 1 = second-most-recent (x), ..., N-1 = newest
                "symbol": symbol,
                "timeframe": timeframe_str,
                "is_ph": ph_dict.get(ts, None) == 'PH',
                "is_pl": pl_dict.get(ts, None) == 'PL'
            })
            ordered_candles.append(candle)

        # === 2. Find the timestamp of the 'x' candle (candle_number == 1) ===
        if len(ordered_candles) < 2:
            return error_log

        x_candle_time_str = ordered_candles[1]["time"]  # candle_number 1
        x_time = datetime.strptime(x_candle_time_str, '%Y-%m-%d %H:%M:%S')

        # === 3. Collect all candles with timestamp > x_time ===
        next_candles = []
        for candle in ordered_candles:
            candle_time = datetime.strptime(candle["time"], '%Y-%m-%d %H:%M:%S')
            if candle_time > x_time:
                # Keep original candle_number (from full history context)
                next_candles.append(candle)

        if not next_candles:
            return error_log  # No newer candles

        # === 4. Write ===
        with open(next_json_path, 'w', encoding='utf-8') as f:
            json.dump(next_candles, f, indent=4)

        log_and_print(
            f"SAVED {symbol} {timeframe_str}: next_newest_oldest_candles.json "
            f"({len(next_candles)} candles after {x_candle_time_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_next newest_candles failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).isoformat()
        })
        save_errors(error_log)

    return error_log

def generate_and_save_oldest_newest_chart(df, symbol, timeframe_str, timeframe_folder, neighborcandles_left, neighborcandles_right):
    """Generate and save a basic candlestick chart as chart.png, then identify PH/PL and save as chartanalysed.png with markers."""
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "oldest_newest.png")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    trendline_log = []

    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green",
                down="red",
                edge="inherit",
                wick={"up": "green", "down": "red"},
                volume="gray"
            )
        )

        # Step 1: Save basic candlestick chart as chart.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str})",
            returnfig=True,
            warn_too_much_data=5000  # Add this line
        )

        # Adjust wick thickness for basic chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        log_and_print(f"Basic chart saved for {symbol} ({timeframe_str}) as {chart_path}", "SUCCESS")

        # Step 2: Identify PH/PL
        ph_labels, pl_labels, phpl_errors = identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right)
        error_log.extend(phpl_errors)

        # Step 3: Prepare annotations for analyzed chart with PH/PL markers
        apds = []
        if ph_labels:
            ph_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ph_labels:
                ph_series.loc[t] = price
            apds.append(mpf.make_addplot(
                ph_series,
                type='scatter',
                markersize=100,
                marker='^',
                color='blue'
            ))
        if pl_labels:
            pl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in pl_labels:
                pl_series.loc[t] = price
            apds.append(mpf.make_addplot(
                pl_series,
                type='scatter',
                markersize=100,
                marker='v',
                color='purple'
            ))

        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "team_type": "initial",
            "status": "info",
            "reason": f"Found {len(ph_labels)} PH points and {len(pl_labels)} PL points",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })

        # Save Trendline Log (only PH/PL info, no trendlines)
        try:
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(f"Trendline log saved for {symbol} ({timeframe_str})", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        # Step 4: Save analyzed chart with PH/PL markers as chartanalysed.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str}) - Analysed",
            addplot=apds if apds else None,
            returnfig=True
        )

        # Adjust wick thickness for analyzed chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(True, linestyle='--')
        fig.savefig(chart_analysed_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        log_and_print(f"Analysed chart saved for {symbol} ({timeframe_str}) as {chart_analysed_path}", "SUCCESS")

        return chart_path, error_log, ph_labels, pl_labels
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "status": "failed",
            "reason": f"Chart generation failed: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        with open(trendline_log_json_path, 'w') as f:
            json.dump(trendline_log, f, indent=4)
        save_errors(error_log)
        log_and_print(f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        return chart_path if os.path.exists(chart_path) else None, error_log, [], []

def generate_and_save_newest_oldest_chart(df, symbol, timeframe_str, timeframe_folder, neighborcandles_left, neighborcandles_right):
    """Generate and save a basic candlestick chart as chart.png, then identify PH/PL and save as chartanalysed.png with markers."""
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "newest_oldest.png")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    trendline_log = []

    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green",
                down="red",
                edge="inherit",
                wick={"up": "green", "down": "red"},
                volume="gray"
            )
        )

        # Step 1: Save basic candlestick chart as chart.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str})",
            returnfig=True,
            warn_too_much_data=5000  # Add this line
        )

        # Adjust wick thickness for basic chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        log_and_print(f"Basic chart saved for {symbol} ({timeframe_str}) as {chart_path}", "SUCCESS")

        # Step 2: Identify PH/PL
        ph_labels, pl_labels, phpl_errors = identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right)
        error_log.extend(phpl_errors)

        # Step 3: Prepare annotations for analyzed chart with PH/PL markers
        apds = []
        if ph_labels:
            ph_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ph_labels:
                ph_series.loc[t] = price
            apds.append(mpf.make_addplot(
                ph_series,
                type='scatter',
                markersize=100,
                marker='^',
                color='blue'
            ))
        if pl_labels:
            pl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in pl_labels:
                pl_series.loc[t] = price
            apds.append(mpf.make_addplot(
                pl_series,
                type='scatter',
                markersize=100,
                marker='v',
                color='purple'
            ))

        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "team_type": "initial",
            "status": "info",
            "reason": f"Found {len(ph_labels)} PH points and {len(pl_labels)} PL points",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })

        # Save Trendline Log (only PH/PL info, no trendlines)
        try:
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(f"Trendline log saved for {symbol} ({timeframe_str})", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        # Step 4: Save analyzed chart with PH/PL markers as chartanalysed.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str}) - Analysed",
            addplot=apds if apds else None,
            returnfig=True
        )

        # Adjust wick thickness for analyzed chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(True, linestyle='--')
        fig.savefig(chart_analysed_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        log_and_print(f"Analysed chart saved for {symbol} ({timeframe_str}) as {chart_analysed_path}", "SUCCESS")

        return chart_path, error_log, ph_labels, pl_labels
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "status": "failed",
            "reason": f"Chart generation failed: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        with open(trendline_log_json_path, 'w') as f:
            json.dump(trendline_log, f, indent=4)
        save_errors(error_log)
        log_and_print(f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        return chart_path if os.path.exists(chart_path) else None, error_log, [], []

def ticks_value(symbol, symbol_folder, broker_name, base_folder, all_symbols):
    error_log = []
    
    # Output file path
    output_json_path = os.path.join(symbol_folder, "ticksvalue.json")
    
    # Default values in case of failure
    tick_size = None
    tick_value = None
    
    try:
        # Get broker config and initialize MT5
        config = brokersdictionary.get(broker_name)
        if not config:
            raise Exception(f"No configuration found for broker '{broker_name}' in brokersdictionary")
        
        success, init_errors = initialize_mt5(
            config["TERMINAL_PATH"],
            config["LOGIN_ID"],
            config["PASSWORD"],
            config["SERVER"]
        )
        error_log.extend(init_errors)
        
        if not success:
            raise Exception("MT5 initialization failed")
        
        # Retrieve symbol info
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise Exception(f"Symbol '{symbol}' not found or not available in MT5 terminal")
        
        tick_size = sym_info.point               # Minimum price increment (e.g., 0.00001 for EURUSD)
        tick_value = sym_info.trade_tick_value   # Value of one tick per standard lot
        
        log_and_print(
            f"[{broker_name}] Retrieved for {symbol}: tick_size={tick_size}, tick_value={tick_value}",
            "SUCCESS"
        )
        
        # Always shutdown MT5 connection
        mt5.shutdown()
        
    except Exception as e:
        error_msg = f"Failed to retrieve tick info for {symbol} ({broker_name}): {str(e)}"
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": error_msg,
            "broker": broker_name
        })
        log_and_print(error_msg, "ERROR")
        # tick_size and tick_value remain None
    
    # Prepare data to save
    output_data = {
        "market": symbol,
        "broker": broker_name,
        "tick_size": tick_size,
        "tick_value": tick_value
    }
    
    # Save to alltimeframes_ob_none_oi_data.json
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        log_and_print(f"Saved tick info to {output_json_path}", "SUCCESS")
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to write {output_json_path}: {str(e)}",
            "broker": broker_name
        })
        log_and_print(f"Failed to save tick info JSON: {str(e)}", "ERROR")
    
    # Save errors if any
    if error_log:
        save_errors(error_log)
    
    return error_log
    
def delete_all_category_jsons():
    """
    Delete (empty) every market-type JSON file that ticks_value writes to.
    - Resets the files to an empty structure (list or dict) so that the next run
      starts from a clean slate.
    - Logs every action and collects any errors in the same format as the rest
      of the module.
    Returns the list of error dictionaries (empty if everything went fine).
    """
    error_log = []

    # ------------------------------------------------------------------ #
    # 1. Exact same paths you already use in ticks_value
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
            right = 0
            left = 0
            top = 0
            bottom = 0
            crop_box = (left, top, img.width - right, img.height - bottom)
            cropped_img = img.crop(crop_box)
            cropped_img.save(chart_path, "PNG")
            log_and_print(f"Chart cropped for {symbol} ({timeframe_str}) at {chart_path}", "SUCCESS")

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

def fetch_charts_all_brokers(
    bars,
    neighborcandles_left,
    neighborcandles_right
):
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
        """Normalize broker name: remove digits, spaces, case-insensitive"""
        return re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', '', name.strip())).lower()

    def clean_folder_name(name: str) -> str:
        """Convert 'Deriv 2', 'deriv6', 'Bybit 10' → 'Deriv', 'Bybit' (Title case)"""
        cleaned = re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', ' ', name.strip()))
        return cleaned.strip().title()

    def normalize_symbol(s: str) -> str:
        return re.sub(r'[\/\s\-_]+', '', s.strip()).upper() if s else ""

    def symbol_needs_processing(symbol: str, base_folder: str) -> bool:
        log_and_print(f"QUEUED {symbol} → will be processed", "INFO")
        return True

    def delete_symbol_folder(symbol: str, base_folder: str, reason: str = ""):
        sym_folder = os.path.join(base_folder, symbol.replace(" ", "_"))
        if os.path.exists(sym_folder):
            try:
                shutil.rmtree(sym_folder)
                log_and_print(f"DELETED {sym_folder} {reason}", "INFO")
            except Exception as e:
                log_and_print(f"FAILED to delete {sym_folder}: {e}", "ERROR")
        os.makedirs(base_folder, exist_ok=True)

    def delete_all_non_blocked_symbol_folders(broker_cfg: dict, blocked_symbols: set):
        base_folder = broker_cfg["BASE_FOLDER"]
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

    def mark_chosen_broker(original_broker_key: str, broker_name: str, balance: float):
        """Create chosenbroker.json in symbols_calculated_prices\<original_key>\chosenbroker.json"""
        target_dir = fr"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices\{original_broker_key}"
        os.makedirs(target_dir, exist_ok=True)
        chosen_path = os.path.join(target_dir, "chosenbroker.json")
        
        chosen_data = {
            "chosen": True,
            "broker_display_name": broker_name,
            "original_key": original_broker_key,
            "balance": round(balance, 2),
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "Highest balance among same broker type"
        }
        
        try:
            with open(chosen_path, "w", encoding="utf-8") as f:
                json.dump(chosen_data, f, indent=4)
            log_and_print(f"MARKED AS CHOSEN → {chosen_path} (Balance: {balance})", "SUCCESS")
        except Exception as e:
            log_and_print(f"FAILED to write chosenbroker.json for {original_broker_key}: {e}", "ERROR")

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
        log_and_print("\n=== NEW FULL CYCLE STARTED ===", "INFO")

        try:
            # ------------------------------------------------------------------
            # 0. LOAD AND AGGREGATE BLOCKED SYMBOLS BY NORMALIZED BROKER KEY
            # ------------------------------------------------------------------
            normalized_blocked_symbols = {}  # normalized_broker -> set of blocked symbols

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

                            norm_broker = normalize_broker_key(broker_raw)
                            normalized_blocked_symbols.setdefault(norm_broker, set())

                            if section == "history_orders":
                                age_str = item.get("age", "")
                                if "d" in age_str:
                                    days = int(age_str.split("d")[0])
                                    if days >= 5:
                                        continue
                                elif not any(t in age_str for t in ["h", "m", "s"]):
                                    continue

                            normalized_blocked_symbols[norm_broker].add(symbol)

                    log_and_print(f"Loaded & merged blocked symbols from {len(normalized_blocked_symbols)} unique brokers", "INFO")
                except Exception as e:
                    log_and_print(f"FAILED to load brokerslimitorders.json: {e}", "ERROR")

            # ------------------------------------------------------------------
            # 1. SELECT ONLY ONE BROKER PER UNIQUE TYPE (HIGHEST BALANCE) + MARK CHOSEN
            # ------------------------------------------------------------------
            selected_brokers = {}  # normalized_key -> (original_name, config, balance, original_dict_key)

            for original_key, cfg in brokersdictionary.items():  # original_key = "deriv2", "bybit10", etc.
                broker_name = cfg.get("original_name", original_key)  # fallback if not set
                norm_key = normalize_broker_key(broker_name)

                balance = 0.0
                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if ok:
                    try:
                        account_info = mt5.account_info()
                        if account_info:
                            balance = account_info.balance
                    except:
                        pass
                    mt5.shutdown()

                current = selected_brokers.get(norm_key)
                if current is None or balance > current[2]:
                    cfg_copy = cfg.copy()
                    cfg_copy["balance"] = balance
                    cfg_copy["original_name"] = broker_name
                    selected_brokers[norm_key] = (broker_name, cfg_copy, balance, original_key)

            # Now mark all selected brokers as "chosen" with their original dictionary key
            unique_brokers = {}
            for norm_key, (broker_name, cfg, balance, original_key) in selected_brokers.items():
                unique_brokers[broker_name] = cfg
                mark_chosen_broker(original_key, broker_name, balance)  # <-- THIS IS THE NEW FEATURE

            log_and_print(f"Selected & MARKED {len(unique_brokers)} unique brokers (highest balance): {list(unique_brokers.keys())}", "SUCCESS")

            # ------------------------------------------------------------------
            # 0.5 DELETE NON-BLOCKED FOLDERS FOR SELECTED BROKERS
            # ------------------------------------------------------------------
            for bn, cfg in unique_brokers.items():
                norm_key = normalize_broker_key(bn)
                blocked = normalized_blocked_symbols.get(norm_key, set())
                delete_all_non_blocked_symbol_folders(cfg, blocked)

            # ------------------------------------------------------------------
            # 1. Load allowed markets
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

            # ------------------------------------------------------------------
            # 2. Symbol → category map
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 3. Load symbolsmatch
            # ------------------------------------------------------------------
            if not os.path.exists(match_path):
                log_and_print(f"Missing {match_path}", "CRITICAL")
                time.sleep(600); continue
            with open(match_path, "r", encoding="utf-8") as f:
                symbolsmatch_data = json.load(f)

            # ------------------------------------------------------------------
            # 5. Build candidate list — ONLY UNIQUE BROKERS
            # ------------------------------------------------------------------
            all_cats = ["stocks","forex","crypto","synthetics","indices","commodities","equities","energies","etfs","basket_indices","metals"]
            candidates = {}
            total_to_do = 0

            for broker_name, cfg in unique_brokers.items():
                norm_key = normalize_broker_key(broker_name)
                blocked = normalized_blocked_symbols.get(norm_key, set())
                candidates[broker_name] = {c: [] for c in all_cats}

                broker_symbols_raw = cfg.get("SYMBOLS", "").strip()
                broker_allowed_symbols = None
                if broker_symbols_raw and broker_symbols_raw.lower() != "all":
                    broker_allowed_symbols = {normalize_symbol(s) for s in broker_symbols_raw.split(",") if s.strip()}

                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if not ok:
                    mt5.shutdown(); continue
                avail, _ = get_symbols()
                mt5.shutdown()

                for entry in symbolsmatch_data.get("main_symbols", []):
                    canonical = entry.get("symbol")
                    if not canonical:
                        continue
                    norm_canonical = normalize_symbol(canonical)

                    found = False
                    broker_symbols_list = []
                    for possible_key in [norm_key, norm_key.title(), norm_key.upper()]:
                        if possible_key in entry:
                            broker_symbols_list = entry.get(possible_key, [])
                            found = True
                            break
                    if not found:
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

                        if symbol_needs_processing(sym_mt5, cfg["BASE_FOLDER"]):
                            delete_symbol_folder(sym_mt5, cfg["BASE_FOLDER"], "(pre-process cleanup)")
                            candidates[broker_name][cat].append(sym_mt5)

                for cat in all_cats:
                    cnt = len(candidates[broker_name][cat])
                    if cnt:
                        log_and_print(f"{broker_name.upper()} → {cat.upper():10} : {cnt:3} queued", "INFO")
                        total_to_do += cnt

            if total_to_do == 0:
                log_and_print("No symbols to process – sleeping 30 min", "WARNING")
                time.sleep(1800)
                continue

            log_and_print(f"TOTAL TO PROCESS: {total_to_do}", "SUCCESS")

            # ------------------------------------------------------------------
            # 6. ROUND-ROBIN PROCESSING ACROSS UNIQUE BROKERS ONLY
            # ------------------------------------------------------------------
            remaining = {b: {c: candidates[b][c][:] for c in all_cats} for b in unique_brokers}
            indices   = {b: {c: 0 for c in all_cats} for b in unique_brokers}

            round_no = 1
            while any(any(remaining[b][c]) for b in unique_brokers for c in all_cats):
                log_and_print(f"\n--- ROUND {round_no} ---", "INFO")

                for cat in all_cats:
                    for bn, cfg in unique_brokers.items():
                        if not remaining[bn][cat]:
                            continue

                        idx = indices[bn][cat]
                        if idx >= len(remaining[bn][cat]):
                            remaining[bn][cat] = []
                            continue

                        symbol = remaining[bn][cat][idx]
                        norm_key = normalize_broker_key(bn)
                        if symbol in normalized_blocked_symbols.get(norm_key, set()):
                            indices[bn][cat] += 1
                            continue

                        for run in (1, 2):
                            ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                            error_log.extend(errs)
                            if not ok:
                                log_and_print(f"MT5 INIT FAILED → {bn}/{symbol} (run {run})", "ERROR")
                                mt5.shutdown()
                                continue

                            log_and_print(f"RUN {run} – PROCESSING {symbol} ({cat}) on {bn.upper()}", "INFO")

                            sym_folder = os.path.join(cfg["BASE_FOLDER"], symbol.replace(" ", "_"))
                            os.makedirs(sym_folder, exist_ok=True)

                            def roundgoblin():
                                for tf_str, mt5_tf in TIMEFRAME_MAP.items():
                                    tf_folder = os.path.join(sym_folder, tf_str)
                                    os.makedirs(tf_folder, exist_ok=True)

                                    df, errs = fetch_ohlcv_data(symbol, mt5_tf, bars)
                                    error_log.extend(errs)
                                    if df is None:
                                        log_and_print(f"NO DATA for {symbol} {tf_str}", "WARNING")
                                        continue

                                    df["symbol"] = symbol
                                    chart_path, ch_errs, ph, pl = generate_and_save_oldest_newest_chart(
                                        df, symbol, tf_str, tf_folder,
                                        neighborcandles_left, neighborcandles_right
                                    )
                                    error_log.extend(ch_errs)

                                    generate_and_save_newest_oldest_chart(
                                        df, symbol, tf_str, tf_folder,
                                        neighborcandles_left, neighborcandles_right
                                    )
                                    error_log.extend(ch_errs)

                                    save_oldest_newest(df, symbol, tf_str, tf_folder, ph, pl)
                                    next_errs = save_next_oldest_newest_candles(df, symbol, tf_str, tf_folder, ph, pl)
                                    error_log.extend(next_errs)

                                    save_newest_oldest(df, symbol, tf_str, tf_folder, ph, pl)
                                    next_errs = save_next_newest_oldest_candles(df, symbol, tf_str, tf_folder, ph, pl)
                                    error_log.extend(next_errs)

                                    if chart_path:
                                        crop_chart(chart_path, symbol, tf_str, tf_folder)

                                mt5.shutdown()

                            roundgoblin()
                            ticks_value(symbol, sym_folder, bn, cfg["BASE_FOLDER"], candidates[bn][cat])
                            calc_and_placeorders()

                        indices[bn][cat] += 1

                round_no += 1

            save_errors(error_log)
            calc_and_placeorders()
            log_and_print("CYCLE 100% COMPLETED (UNIQUE BROKERS ONLY)", "SUCCESS")

            # ------------------------------------------------------------------
            # FINAL STEP: RENAME BASE_FOLDERS TO REMOVE NUMBERS (AFTER CYCLE)
            # ------------------------------------------------------------------
            log_and_print("Starting post-cycle BASE_FOLDER renaming (removing numbers)...", "INFO")
            renamed = 0
            for original_name, cfg in unique_brokers.items():
                old_path = cfg["BASE_FOLDER"]
                if not os.path.exists(old_path):
                    continue

                parent_dir = os.path.dirname(old_path)
                new_name = clean_folder_name(original_name)
                new_path = os.path.join(parent_dir, new_name)

                if old_path == new_path:
                    continue

                if os.path.exists(new_path):
                    log_and_print(f"Target already exists: {new_path} — skipping rename from {old_path}", "WARNING")
                    continue

                try:
                    os.rename(old_path, new_path)
                    cfg["BASE_FOLDER"] = new_path
                    log_and_print(f"RENAMED FOLDER: {old_path} → {new_path}", "SUCCESS")
                    renamed += 1
                except Exception as e:
                    log_and_print(f"FAILED RENAME {old_path} → {new_path}: {e}", "ERROR")

            log_and_print(f"Folder renaming complete. {renamed} folder(s) cleaned.", "SUCCESS" if renamed > 0 else "INFO")

            log_and_print("Sleeping 30 minutes before next cycle...", "INFO")
            time.sleep(1800)

        except Exception as e:
            log_and_print(f"MAIN LOOP CRASH: {e}\n{traceback.format_exc()}", "CRITICAL")
            time.sleep(600)


if __name__ == "__main__":
    success = fetch_charts_all_brokers(
        bars=201,
        neighborcandles_left=10,
        neighborcandles_right=15
    )
    if success:
        log_and_print("Chart generation, cropping, arrow detection, PH/PL analysis, and candle data saving completed successfully for all brokers!", "SUCCESS")
    else:
        log_and_print("Process failed. Check error log for details.", "ERROR")


        
        
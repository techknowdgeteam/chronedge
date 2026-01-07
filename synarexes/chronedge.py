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
import concurrent.futures
import multiprocessing



def load_developers_dictionary():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json"
    """Load brokers config from JSON file with error handling and fallback."""
    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Convert numeric strings back to int where needed
        for user_brokerid, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in developersdictionary.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load developersdictionary.json: {e}", "CRITICAL")
        return {}
developersdictionary = load_developers_dictionary()


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


def fetch_ohlcv_data(symbol, mt5_timeframe, bars):
    """
    Fetch OHLCV data for a given symbol and timeframe with detailed diagnostics.
    
    Returns:
        df (pd.DataFrame or None), error_log (list of dicts)
    """
    error_log = []
    lagos_tz = pytz.timezone('Africa/Lagos')
    timestamp = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S.%f%z')

    broker_name = mt5.terminal_info().name if mt5.terminal_info() else "unknown"

    # --- Step 1: Ensure symbol is selected (with retry) ---
    selected = False
    for attempt in range(3):
        if mt5.symbol_select(symbol, True):
            selected = True
            break
        time.sleep(0.5)  # small delay before retry

    if not selected:
        last_err = mt5.last_error()
        err_msg = f"FAILED symbol_select('{symbol}') after 3 attempts: {last_err}"
        log_and_print(err_msg, "ERROR")
        error_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": mt5_timeframe,
            "requested_bars": bars,
            "error": err_msg,
            "broker": broker_name
        })
        save_errors(error_log)
        return None, error_log

    # --- Step 2: Try to copy rates ---
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

    if rates is None:
        last_err = mt5.last_error()
        err_msg = f"copy_rates_from_pos returned None for {symbol} (TF: {mt5_timeframe}): {last_err}"
        log_and_print(err_msg, "ERROR")
        error_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": mt5_timeframe,
            "requested_bars": bars,
            "error": err_msg,
            "broker": broker_name
        })
        save_errors(error_log)
        return None, error_log

    available_bars = len(rates)
    if available_bars == 0:
        err_msg = f"NO historical data available for {symbol} on this timeframe (requested {bars} bars, got 0)"
        log_and_print(err_msg, "WARNING")
        error_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "timeframe": mt5_timeframe,
            "requested_bars": bars,
            "available_bars": 0,
            "error": "No bars returned (likely broker limitation on higher timeframes)",
            "broker": broker_name
        })
        save_errors(error_log)
        return None, error_log

    # --- Success path ---
    if available_bars < bars:
        log_msg = (f"Partial data: {symbol} → requested {bars} bars, "
                   f"but only {available_bars} available (common on higher TFs like 1h/4h)")
        log_and_print(log_msg, "WARNING")
    else:
        log_and_print(f"Fetched {available_bars} bars for {symbol}", "INFO")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")

    # Clean and standardize dtypes
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "tick_volume": float,
        "spread": int,
        "real_volume": float
    })
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    return df, error_log

def save_newest_oldest_df(df, symbol, timeframe_str, timeframe_folder):
    """Save candles: oldest → newest, candle_number 0 = oldest. Fixed filenames."""
    error_log = []
    
    target_subfolder = os.path.join(timeframe_folder, "candlesdetails")
    os.makedirs(target_subfolder, exist_ok=True)
    
    all_json_path = os.path.join(target_subfolder, "newest_oldest.json")
    latest_json_path = os.path.join(target_subfolder, "latest_completed_candle.json")
    
    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    try:
        if len(df) < 2:
            error_msg = f"Not enough data for {symbol} ({timeframe_str})"
            log_and_print(error_msg, "ERROR")
            error_log.append({"error": error_msg, "timestamp": now.isoformat()})
            save_errors(error_log)
            return error_log

        all_candles = []
        for i, (ts, row) in enumerate(df.iterrows()):
            candle = row.to_dict()
            candle.update({
                "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
                "candle_number": i,
                "symbol": symbol,
                "timeframe": timeframe_str
            })
            all_candles.append(candle)

        with open(all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_candles, f, indent=4)

        # Latest completed candle: second from end (-2)
        previous_latest_candle = all_candles[-2].copy()
        candle_time = lagos_tz.localize(datetime.strptime(previous_latest_candle["time"], '%Y-%m-%d %H:%M:%S'))
        delta = now - candle_time
        total_hours = delta.total_seconds() / 3600
        age_str = f"{int(total_hours)}h old" if total_hours <= 24 else f"{int(total_hours // 24)}d old"

        previous_latest_candle.update({"age": age_str, "id": "x"})
        if "candle_number" in previous_latest_candle:
            del previous_latest_candle["candle_number"]

        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(previous_latest_candle, f, indent=4)

        log_and_print(f"SAVED: newest_oldest.json for {symbol} {timeframe_str}", "SUCCESS")

    except Exception as e:
        err = f"save_newest_oldest_df failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({"error": err, "timestamp": now.isoformat()})
        save_errors(error_log)

    return error_log

def generate_and_save_chart_df(df, symbol, timeframe_str, timeframe_folder):
    """Generate and save only the basic full chart. Sliced charts have been removed."""
    error_log = []
    
    chart_path = os.path.join(timeframe_folder, "chart.png")
    
    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green", down="red", edge="inherit",
                wick={"up": "green", "down": "red"}, volume="gray"
            )
        )

        # Generate and save only the full chart
        fig, axlist = mpf.plot(
            df, 
            type='candle', 
            style=custom_style, 
            volume=False,
            title=f"{symbol} ({timeframe_str})", 
            returnfig=True,
            warn_too_much_data=5000
        )
        
        fig.set_size_inches(25, 10)
        for ax in axlist:
            ax.grid(False)
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        fig.savefig(chart_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        log_and_print(f"SAVED: chart.png for {symbol} {timeframe_str}", "SUCCESS")

        return chart_path, error_log

    except Exception as e:
        log_and_print(f"Error in chart generation: {e}", "ERROR")
        error_log.append(str(e))
        return None, error_log
        
def generate_and_save_chart(symbol, timeframe_str, timeframe_folder):
    """Generate sliced charts + return list of slice counts actually generated"""
    error_log = []

    target_subfolder = os.path.join(timeframe_folder, "candlesdetails")
    json_path = os.path.join(target_subfolder, "newest_oldest.json")

    candle_slices = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 121, 131, 141, 151, 161, 171, 181, 191, 201, 221, 231, 241, 251, 261, 271, 281, 291, 301]

    generated_slice_counts = []  # To pass to JSON slicers

    try:
        if not os.path.exists(json_path):
            err = f"JSON file not found: {json_path}"
            log_and_print(err, "ERROR")
            error_log.append({"error": err})
            return [], error_log

        with open(json_path, 'r', encoding='utf-8') as f:
            all_candles = json.load(f)

        if len(all_candles) < 11:
            err = f"Not enough candles in JSON (need at least 11) for {symbol} {timeframe_str}"
            log_and_print(err, "WARNING")
            return [], error_log

        df = pd.DataFrame(all_candles)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.astype(float)
        df = df.sort_index()

        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green", down="red", edge="inherit",
                wick={"up": "green", "down": "red"}, volume="gray"
            )
        )

        generated_slices = 0
        for count in candle_slices:
            if len(df) >= count:
                df_slice = df.iloc[-count:]
                slice_path = os.path.join(timeframe_folder, f"chart_{count}.png")

                fig, axlist = mpf.plot(
                    df_slice,
                    type='candle',
                    style=custom_style,
                    title=f"{symbol} ({timeframe_str}) - Last {count}",
                    returnfig=True,
                    warn_too_much_data=5000
                )

                fig.set_size_inches(25, 10)
                for ax in axlist:
                    ax.grid(False)
                    for line in ax.get_lines():
                        if line.get_label() == '':
                            line.set_linewidth(0.5)

                fig.savefig(slice_path, bbox_inches="tight", dpi=100)
                plt.close(fig)

                generated_slice_counts.append(count)
                generated_slices += 1

        log_and_print(f"SAVED: {generated_slices} sliced charts (from JSON) for {symbol} {timeframe_str}", "SUCCESS")
        return generated_slice_counts, error_log

    except Exception as e:
        log_and_print(f"Error in sliced chart generation (from JSON): {e}", "ERROR")
        error_log.append({"error": str(e)})
        return [], error_log
    
def save_sliced_newest_oldest_json(symbol, timeframe_str, timeframe_folder, slice_counts):
    """Save sliced versions: oldest → newest (candle_number 0 = oldest) from full newest_oldest.json"""
    error_log = []

    target_subfolder = os.path.join(timeframe_folder, "candlesdetails")
    full_json_path = os.path.join(target_subfolder, "newest_oldest.json")

    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    try:
        if not os.path.exists(full_json_path):
            err = f"Full JSON not found for slicing: {full_json_path}"
            log_and_print(err, "ERROR")
            error_log.append({"error": err})
            return error_log

        with open(full_json_path, 'r', encoding='utf-8') as f:
            all_candles = json.load(f)

        if len(all_candles) < 11:
            return error_log  # Not enough for smallest slice

        generated = 0
        for count in slice_counts:
            if len(all_candles) < count:
                continue

            # Slice: last `count` candles → most recent
            sliced_candles = all_candles[-count:]

            # Full sliced JSON: oldest → newest in this slice
            reordered = []
            for i, candle in enumerate(sliced_candles):
                c = candle.copy()
                c["candle_number"] = i  # 0 = oldest in slice
                reordered.append(c)

            slice_json_path = os.path.join(target_subfolder, f"new_old_{count}.json")
            with open(slice_json_path, 'w', encoding='utf-8') as f:
                json.dump(reordered, f, indent=4)

            # Latest completed candle in this slice: second from end (-2)
            if len(reordered) >= 2:
                prev_candle = reordered[-2].copy()
                candle_time = lagos_tz.localize(datetime.strptime(prev_candle["time"], '%Y-%m-%d %H:%M:%S'))
                delta = now - candle_time
                total_hours = delta.total_seconds() / 3600
                age_str = f"{int(total_hours)}h old" if total_hours <= 24 else f"{int(total_hours // 24)}d old"
                prev_candle.update({"age": age_str, "id": "x"})
                if "candle_number" in prev_candle:
                    del prev_candle["candle_number"]

                latest_slice_path = os.path.join(target_subfolder, f"latest_completed_candle.json")
                with open(latest_slice_path, 'w', encoding='utf-8') as f:
                    json.dump(prev_candle, f, indent=4)

            generated += 1

        if generated > 0:
            log_and_print(f"SAVED: {generated} sliced new_old_*.json + latest_completed for {symbol} {timeframe_str}", "SUCCESS")

    except Exception as e:
        err = f"save_sliced_newest_oldest_json failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({"error": err, "timestamp": now.isoformat()})

    return error_log

def ticks_value(symbol, symbol_folder, user_brokerid, base_folder, all_symbols):
    error_log = []
    
    # Strip numbers from broker ID (e.g., "deriv6" -> "deriv")
    cleaned_broker = ''.join([char for char in user_brokerid if not char.isdigit()])
    
    # Individual file path
    safe_symbol = symbol.replace('/', '_').replace(' ', '_').upper()  # Safer: also handle spaces
    output_json_filename = f"{safe_symbol}_ticks.json"
    output_json_path = os.path.join(symbol_folder, output_json_filename)
    
    # Combined file path
    combined_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbolstick\symbolstick.json"
    
    # Default values
    tick_size = None
    tick_value = None
    
    try:
        # Get broker config and initialize MT5
        config = developersdictionary.get(user_brokerid)
        if not config:
            raise Exception(f"No configuration found for broker '{user_brokerid}' in developersdictionary")
        
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
        
        tick_size = sym_info.point               # Minimum price increment
        tick_value = sym_info.trade_tick_value   # Value of one tick per standard lot
        
        log_and_print(
            f"[{user_brokerid}] Retrieved for {symbol}: tick_size={tick_size}, tick_value={tick_value}",
            "SUCCESS"
        )
        
        # Always shutdown MT5 connection
        mt5.shutdown()
        
    except Exception as e:
        error_msg = f"Failed to retrieve tick info for {symbol} ({user_brokerid}): {str(e)}"
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": error_msg,
            "broker": user_brokerid
        })
        log_and_print(error_msg, "ERROR")
    
    # Data to save - using cleaned broker name (e.g., "deriv" instead of "deriv6")
    output_data = {
        "market": symbol,
        "broker": cleaned_broker,        # <-- Cleaned version here
        "tick_size": tick_size,
        "tick_value": tick_value
    }
    
    # 1. Save individual symbol JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        log_and_print(f"Saved tick info to {output_json_path}", "SUCCESS")
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to write {output_json_path}: {str(e)}",
            "broker": user_brokerid
        })
        log_and_print(f"Failed to save individual JSON: {str(e)}", "ERROR")
    
    # 2. Update the combined symbolstick.json
    combined_data = {}
    file_exists = os.path.exists(combined_path)
    
    if file_exists:
        try:
            with open(combined_path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to read combined JSON: {str(e)}",
                "broker": user_brokerid
            })
            log_and_print(f"Failed to read combined JSON: {str(e)}", "ERROR")
            combined_data = {}
    
    # Create the new entry with cleaned broker
    entry = {
        "market": symbol,
        "broker": cleaned_broker,
        "tick_size": tick_size,
        "tick_value": tick_value
    }
    
    previous_entry = combined_data.get(safe_symbol)
    
    # Only write if new symbol or values have changed
    if previous_entry != entry:
        combined_data[safe_symbol] = entry
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=4)
            
            action = "Updated" if previous_entry is not None else "Added"
            log_and_print(f"{action} {safe_symbol} (broker: {cleaned_broker}) in combined symbolstick.json", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to write combined JSON: {str(e)}",
                "broker": user_brokerid
            })
            log_and_print(f"Failed to save combined JSON: {str(e)}", "ERROR")
    
    # Save any errors
    if error_log:
        save_errors(error_log)
    
    return error_log

def crop_chart(chart_path, symbol, timeframe_str, timeframe_folder):
    """Crop all charts in the folder including slices (chart_XX.png) and analyzed versions."""
    error_log = []
    
    # List of all images to crop: the main ones and all the slices
    images_to_crop = [f for f in os.listdir(timeframe_folder) if f.endswith(".png") and "chart" in f]

    try:
        for filename in images_to_crop:
            full_path = os.path.join(timeframe_folder, filename)
            with Image.open(full_path) as img:
                # Set your crop margins here if needed (currently 0)
                left, top, right, bottom = 0, 0, 0, 0 
                crop_box = (left, top, img.width - right, img.height - bottom)
                cropped_img = img.crop(crop_box)
                cropped_img.save(full_path, "PNG")
        
        log_and_print(f"All {len(images_to_crop)} charts cropped for {symbol} ({timeframe_str})", "SUCCESS")

    except Exception as e:
        err_msg = f"Failed to crop charts: {str(e)}"
        log_and_print(err_msg, "ERROR")
        error_log.append({"error": err_msg})

    return error_log

def backup_developers_dictionary():
    main_path = Path(r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json")
    backup_path = Path(r"C:\xampp\htdocs\chronedge\synarex\developersdictionarybackup.json")
    
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
    print("Created fresh empty developersdictionary.json and backup") 

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
    
    if not developersdictionary:
        print("No brokers found in developersdictionary.")
        return

    print("Configured Brokers & Folder Check (Human-readable folders):")
    print("=" * 90)
    
    configured_displays = set()
    known_broker_bases = set()
    broker_details = []
    existing = 0
    missing = 0
    
    def format_user_brokerid(name):
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
    for user_brokerid in developersdictionary.keys():
        original = user_brokerid.strip()
        display_name = format_user_brokerid(original)
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
    print(f"Total configured: {len(developersdictionary)} broker(s) | {existing} folder(s) exist | {missing} missing")

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
    bars
):
    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    backup_developers_dictionary()
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

    def mark_chosen_broker(original_broker_key: str, user_brokerid: str, balance: float):
        """Create chosenbroker.json in symbols_calculated_prices\<original_key>\chosenbroker.json"""
        target_dir = fr"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices\{original_broker_key}"
        os.makedirs(target_dir, exist_ok=True)
        chosen_path = os.path.join(target_dir, "chosenbroker.json")
        
        chosen_data = {
            "chosen": True,
            "broker_display_name": user_brokerid,
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

            for original_key, cfg in developersdictionary.items():  # original_key = "deriv2", "bybit10", etc.
                user_brokerid = cfg.get("original_name", original_key)  # fallback if not set
                norm_key = normalize_broker_key(user_brokerid)

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
                    cfg_copy["original_name"] = user_brokerid
                    selected_brokers[norm_key] = (user_brokerid, cfg_copy, balance, original_key)

            # Now mark all selected brokers as "chosen" with their original dictionary key
            unique_brokers = {}
            for norm_key, (user_brokerid, cfg, balance, original_key) in selected_brokers.items():
                unique_brokers[user_brokerid] = cfg
                mark_chosen_broker(original_key, user_brokerid, balance)  # <-- THIS IS THE NEW FEATURE

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

            for user_brokerid, cfg in unique_brokers.items():
                norm_key = normalize_broker_key(user_brokerid)
                blocked = normalized_blocked_symbols.get(norm_key, set())
                candidates[user_brokerid] = {c: [] for c in all_cats}

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
                            candidates[user_brokerid][cat].append(sym_mt5)

                for cat in all_cats:
                    cnt = len(candidates[user_brokerid][cat])
                    if cnt:
                        log_and_print(f"{user_brokerid.upper()} → {cat.upper():10} : {cnt:3} queued", "INFO")
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

                        ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                        error_log.extend(errs)
                        if not ok:
                            log_and_print(f"MT5 INIT FAILED → {bn}/{symbol}", "ERROR")
                            mt5.shutdown()
                            indices[bn][cat] += 1
                            continue

                        log_and_print(f"PROCESSING {symbol} ({cat}) on {bn.upper()}", "INFO")

                        sym_folder = os.path.join(cfg["BASE_FOLDER"], symbol.replace(" ", "_"))
                        os.makedirs(sym_folder, exist_ok=True)
                        def roundgoblin():
                            for tf_str, mt5_tf in TIMEFRAME_MAP.items():
                                if not mt5.initialize():
                                    log_and_print(f"MT5 initialize() failed for {tf_str}, error code = {mt5.last_error()}", "ERROR")
                                    error_log.append(f"MT5 init failed for {tf_str}: {mt5.last_error()}")
                                    continue
                                
                                tf_folder = os.path.join(sym_folder, tf_str)
                                os.makedirs(tf_folder, exist_ok=True)
                                
                                df, errs = fetch_ohlcv_data(symbol, mt5_tf, bars)
                                error_log.extend(errs)
                                
                                if df is None or len(df) == 0:
                                    log_and_print(f"NO DATA for {symbol} {tf_str}", "WARNING")
                                    mt5.shutdown()
                                    continue
                                
                                df["symbol"] = symbol
                                
                                
                                save_newest_oldest_df(df, symbol, tf_str, tf_folder)

                                chart_path, ch_errs = generate_and_save_chart_df(df, symbol, tf_str, tf_folder)
                                error_log.extend(ch_errs or [])
                                slice_counts, ch_errs = generate_and_save_chart(symbol, tf_str, tf_folder)
                                error_log.extend(ch_errs or [])

                                if slice_counts:
                                    error_log.extend(save_sliced_newest_oldest_json(symbol, tf_str, tf_folder, slice_counts))
                                
                                if chart_path:
                                    crop_chart(chart_path, symbol, tf_str, tf_folder)
                                
                                mt5.shutdown()
                        
                        roundgoblin()
                        ticks_value(symbol, sym_folder, bn, cfg["BASE_FOLDER"], candidates[bn][cat])

                        indices[bn][cat] += 1

                round_no += 1

            save_errors(error_log)
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
        bars=2001
    )

    if success:
        log_and_print("Chart generation, cropping, arrow detection, PH/PL analysis, and candle data saving completed successfully for all brokers!", "SUCCESS")
    else:
        log_and_print("Process failed. Check error log for details.", "ERROR")


        
        
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
import insiderfetch


def load_brokers_dictionary():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\brokersdictionary.json"
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


BASE_ERROR_FOLDER = r"C:\xampp\htdocs\chronedge\chart\debugs"
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


def save_candle_data(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save all candles + highlight the SECOND-MOST-RECENT candle (candle_number = 1)
    as 'x' in previouslatestcandle.json with age in days/hours.
    """
    error_log = []
    all_json_path = os.path.join(timeframe_folder, "all_candles.json")
    latest_json_path = os.path.join(timeframe_folder, "previouslatestcandle.json")
    
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
            f"all_candles.json ({len(all_candles)} candles) + "
            f"previouslatestcandle.json (candle_number=1 → id='x', {age_str})",
            "SUCCESS"
        )

    except Exception as e:
        err = f"save_candle_data failed: {str(e)}"
        log_and_print(err, "ERROR")
        error_log.append({
            "error": err,
            "timestamp": now.isoformat()
        })
        save_errors(error_log)

    return error_log

def save_next_candles(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """
    Save candles that appear **after** the previous-latest candle (the one saved as 'x')
    i.e. candles with timestamp > timestamp of 'x' candle
    into <timeframe_folder>/nextcandles.json
    """
    error_log = []
    next_json_path = os.path.join(timeframe_folder, "nextcandles.json")

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
            f"SAVED {symbol} {timeframe_str}: nextcandles.json "
            f"({len(next_candles)} candles after {x_candle_time_str})",
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

def generate_and_save_chart(df, symbol, timeframe_str, timeframe_folder, neighborcandles_left, neighborcandles_right):
    """Generate and save a basic candlestick chart as chart.png, then identify PH/PL and save as chartanalysed.png with markers."""
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")
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
    allsymbols_json_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    symbols_match_json_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\symbolsmatch.json"  # Fixed name + path

    # Paths for market-type-specific JSONs (unchanged - these are in root)
    market_type_paths = {
        "forex": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\forexvolumesandrisk.json",
        "stocks": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\stocksvolumesandrisk.json",
        "indices": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\indicesvolumesandrisk.json",
        "synthetics": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\syntheticsvolumesandrisk.json",
        "commodities": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\commoditiesvolumesandrisk.json",
        "crypto": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\cryptovolumesandrisk.json",
        "equities": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\equitiesvolumesandrisk.json",
        "energies": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\energiesvolumesandrisk.json",
        "etfs": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\etfsvolumesandrisk.json",
        "basket_indices": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\basketindicesvolumesandrisk.json",
        "metals": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\metalsvolumesandrisk.json"
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
  
def collect_all_calculated_prices_to_json():
    """
    FINAL @teamxtech APPROVED
    → Converts flat calculated prices → YOUR EXACT limitorders.json format
    → Saves TWO files:
        1. <BASE_FOLDER>/allmarkets_limitorderscalculatedprices.json → ALL markets combined
        2. <BASE_FOLDER>/<market>/<timeframe>/limitorderscalculatedprices.json → Per-market/timeframe
    → Structure: markets_limitorders + limitorders[market][timeframe][team1]
    → Call once at the end
    """
    global brokersdictionary
    CALCULATED_ROOT = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices")
    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd",
        3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"
    }
    TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]
    error_log = []
    print("\n" + "═" * 95)
    print(" CONSOLIDATING → allmarkets_limitorderscalculatedprices.json + PER-MARKET/TF JSONS ".center(95))
    print(" STRUCTURE: markets_limitorders + limitorders[market][tf][team1] ".center(95))
    print("═" * 95 + "\n")

    # Per-market/timeframe storage
    per_market_tf_data = {}  # {market: {tf: [team1, ...]}}

    for broker_name, config in brokersdictionary.items():
        BASE_FOLDER = config.get("BASE_FOLDER")
        if not BASE_FOLDER:
            log_and_print(f"BASE_FOLDER missing for {broker_name}", "ERROR")
            continue
        base_path = Path(BASE_FOLDER)
        if not base_path.exists():
            log_and_print(f"BASE_FOLDER not found: {BASE_FOLDER}", "WARNING")
            continue
        broker_calc_dir = CALCULATED_ROOT / broker_name
        if not broker_calc_dir.is_dir():
            log_and_print(f"No data for {broker_name}", "INFO")
            continue
        print(f"[{broker_name.upper()}] → {BASE_FOLDER}")

        # Master structure for all markets
        limitorders = {}
        markets_with_orders = set()

        # Walk all risk folders
        for risk_val, folder in RISK_FOLDERS.items():
            risk_dir = broker_calc_dir / folder
            if not risk_dir.is_dir():
                continue
            for calc_file in risk_dir.glob("*calculatedprices.json"):
                try:
                    with open(calc_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else sum(data.values(), [])
                    for entry in entries:
                        market = entry.get("market")
                        tf = entry.get("timeframe", "30m")
                        if not market or tf not in TIMEFRAMES:
                            continue

                        # Build team1
                        team1 = {
                            "timestamp": entry.get("calculated_at", datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S')),
                            "limit_order": entry.get("limit_order"),
                            "entry_price": entry.get("entry_price"),
                            "volume": entry.get("volume"),
                            "riskusd_amount": entry.get("riskusd_amount"),
                            "sl_price": entry.get("sl_price"),
                            "sl_pips": entry.get("sl_pips"),
                            "tp_price": entry.get("tp_price"),
                            "tp_pips": entry.get("tp_pips"),
                            "rr_ratio": entry.get("rr_ratio"),
                            "calculated_at": entry.get("calculated_at"),
                            "selection_criteria": entry.get("selection_criteria"),
                            "broker": broker_name
                        }

                        # === 1. Add to ALL-MARKETS structure ===
                        if market not in limitorders:
                            limitorders[market] = {tf: [] for tf in TIMEFRAMES}
                        if tf not in limitorders[market]:
                            limitorders[market][tf] = []
                        limitorders[market][tf].append({"team1": team1})
                        markets_with_orders.add(market)

                        # === 2. Add to PER-MARKET/TF structure ===
                        if market not in per_market_tf_data:
                            per_market_tf_data[market] = {tf: [] for tf in TIMEFRAMES}
                        if tf not in per_market_tf_data[market]:
                            per_market_tf_data[market][tf] = []
                        per_market_tf_data[market][tf].append(team1)  # Only team1 dict, no wrapper

                except Exception as e:
                    ts = datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S')
                    error_log.append({
                        "timestamp": ts,
                        "error": f"Failed: {calc_file}",
                        "details": str(e),
                        "broker": broker_name
                    })

        # === SAVE 1: ALL MARKETS COMBINED ===
        final_data = {
            "markets_limitorders": len(markets_with_orders),
            "limitorders": limitorders
        }
        output_file = base_path / "allmarkets_limitorderscalculatedprices.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4)
            print(f" SUCCESS: {len(markets_with_orders)} markets → {output_file.name}")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "error": f"Write failed: {output_file}",
                "details": str(e)
            })
            print(f" FAILED: {e}")

        # === SAVE 2: PER MARKET/TIMEFRAME JSONS ===
        saved_count = 0
        for market, tf_dict in per_market_tf_data.items():
            market_clean = market.replace(" ", "_").replace("/", "_")
            market_folder = base_path / market_clean
            market_folder.mkdir(exist_ok=True)
            for tf, team1_list in tf_dict.items():
                if not team1_list:
                    continue
                tf_folder = market_folder / tf
                tf_folder.mkdir(exist_ok=True)
                per_tf_data = {
                    "market": market,
                    "timeframe": tf,
                    "broker": broker_name,
                    "orders": team1_list  # List of team1 dicts
                }
                per_tf_file = tf_folder / "limitorderscalculatedprices.json"
                try:
                    with open(per_tf_file, 'w', encoding='utf-8') as f:
                        json.dump(per_tf_data, f, indent=4)
                    saved_count += 1
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "error": f"Per-TF write failed: {per_tf_file}",
                        "details": str(e)
                    })
        print(f" SUCCESS: {saved_count} per-market/timeframe JSONs saved under broker folder")
        print()

    # Final Report
    print("═" * 95)
    print(" CONSOLIDATION COMPLETE ")
    print(" 1. allmarkets_limitorderscalculatedprices.json → All markets ")
    print(" 2. <market>/<tf>/limitorderscalculatedprices.json → Per market/timeframe ")
    print(" Ready for your dashboard ")
    print("═" * 95 + "\n")

    if error_log:
        save_errors(error_log)
    return error_log

def redraw_contours_from_json(
    chart_path,
    symbol,
    timeframe_str,
    timeframe_folder
):
    error_log = []
    lagos_tz = pytz.timezone('Africa/Lagos')
    now = datetime.now(lagos_tz)

    # === PATHS ===
    candle_json_path         = os.path.join(timeframe_folder, "all_candles.json")
    contour_json_path        = os.path.join(timeframe_folder, "chart_contours.json")
    ob_none_oi_json_path     = os.path.join(timeframe_folder, "ob_none_oi_data.json")
    limitorders_json_path    = os.path.join(timeframe_folder, "limitorderscalculatedprices.json")
    next_candles_path        = os.path.join(timeframe_folder, "nextcandles.json")
    previouslatest_path      = os.path.join(timeframe_folder, "previouslatestcandle.json")
    output_image_path        = os.path.join(timeframe_folder, "chart_with_contours_redrawn.png")
    output_cropped_image_path = os.path.join(timeframe_folder, "chart_entry_redrawn.png")  # NEW CROPPED IMAGE
    output_redraw_json_path  = os.path.join(timeframe_folder, "redrawn oi ob data.json")
    all_timeframes_json_path = os.path.join(os.path.dirname(timeframe_folder), "alltimeframeslimitorders.json")

    # === LOAD REQUIRED DATA ===
    try:
        with open(candle_json_path, 'r') as f:
            candle_data = json.load(f)
        with open(contour_json_path, 'r') as f:
            contour_data = json.load(f)
    except Exception as e:
        error_msg = f"Missing JSON files: {e}"
        error_log.append({"error": error_msg, "timestamp": now.isoformat()})
        save_errors(error_log)
        return error_log

    # === LOAD nextcandles.json ===
    next_candles = []
    if os.path.exists(next_candles_path):
        try:
            with open(next_candles_path, 'r', encoding='utf-8') as f:
                next_candles = json.load(f)
            log_and_print(f"Loaded {len(next_candles)} next candles", "INFO")
        except Exception as e:
            log_and_print(f"Failed to load nextcandles.json: {e}", "WARNING")
    else:
        log_and_print(f"nextcandles.json not found", "INFO")

    # === LOAD SL/TP ===
    calculated_orders = {}
    if os.path.exists(limitorders_json_path):
        try:
            with open(limitorders_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for order in data.get("orders", []):
                entry = order.get("entry_price")
                if entry is not None:
                    calculated_orders[entry] = {
                        "sl_price": order.get("sl_price"),
                        "tp_price": order.get("tp_price")
                    }
        except Exception as e:
            log_and_print(f"Failed to load limitorderscalculatedprices.json: {e}", "WARNING")

    # === LOAD OB-None-OI entries ===
    ob_none_oi_entries = []
    try:
        if os.path.exists(ob_none_oi_json_path):
            with open(ob_none_oi_json_path, 'r') as f:
                raw_data = json.load(f)
            for item in raw_data:
                for team_key, team_data in item.items():
                    limit_order = ""
                    entry_price = 0.0
                    if team_data["team_type"] == "PH-to-PH":
                        limit_order = "buy_limit"
                        entry_price = team_data["none_oi_x_OB_high_price"]
                    elif team_data["team_type"] == "PL-to-PL":
                        limit_order = "sell_limit"
                        entry_price = team_data["none_oi_x_OB_low_price"]
                    else:
                        continue
                    ob_none_oi_entries.append({
                        "limit_order": limit_order,
                        "entry_price": entry_price,
                        "team_type": team_data["team_type"]
                    })
        else:
            log_and_print(f"ob_none_oi_data.json not found", "WARNING")
    except Exception as e:
        error_log.append({"error": f"Failed to load ob_none_oi_data.json: {str(e)}", "timestamp": now.isoformat()})

    img = cv2.imread(chart_path)
    if img is None:
        error_log.append({"error": "Failed to load chart image", "timestamp": now.isoformat()})
        save_errors(error_log)
        return error_log

    # === HELPERS ===
    def draw_right_arrow(img, x, y, oi_x=None):
        end_x = oi_x if oi_x else img.shape[1] - 5
        color = (255, 0, 0) if oi_x else (0, 255, 0)
        cv2.line(img, (x, y), (end_x, y), color, 1)
        cv2.line(img, (end_x-4, y-4), (end_x, y), color, 1)
        cv2.line(img, (end_x-4, y+4), (end_x, y), color, 1)

    def draw_oi_marker(img, x, y):
        cv2.circle(img, (x, y), 7, (0, 255, 0), 1)

    # === BUILD CANDLE BOUNDS & PRICE MAPPING ===
    candle_bounds = {}
    all_prices = []
    for c in contour_data["candle_contours"]:
        i = c["candle_number"]
        cd = candle_data[i]
        high = float(cd["high"])
        low = float(cd["low"])
        all_prices.extend([high, low])
        x = c["x"] - c["width"]//2
        y = c["y"]
        w = c["width"]
        h = c["height"]
        candle_bounds[i] = {
            "high_y": y,
            "low_y": y + h,
            "x_left": x,
            "x_right": x + w,
            "high": high,
            "low": low,
            "center_x": x + w // 2
        }

    if not all_prices:
        return error_log

    global_high = max(all_prices)
    global_low = min(all_prices)

    # Find chart area bounds
    xs = [b["x_left"] for b in candle_bounds.values()]
    ys = [b["high_y"] for b in candle_bounds.values()] + [b["low_y"] for b in candle_bounds.values()]
    chart_left = min(xs)
    chart_right = max(b["x_right"] for b in candle_bounds.values())
    chart_top = min(ys)
    chart_bottom = max(ys)
    chart_height = chart_bottom - chart_top

    # === PRICE → Y-PIXEL ===
    def price_to_y(price):
        if global_high == global_low:
            return chart_top + chart_height // 2
        ratio = (global_high - price) / (global_high - global_low)
        return int(chart_top + ratio * chart_height)

    # === FIND FIRST NEXT CANDLE X-POSITION ===
    first_next_x_right = None
    if next_candles:
        next_times = {c["time"] for c in next_candles}
        for i, cd in enumerate(candle_data):
            if cd["time"] in next_times:
                if i in candle_bounds:
                    first_next_x_right = candle_bounds[i]["x_right"]
                    break

    # === 1. PH/PL TRIANGLES ===
    for c in contour_data["candle_contours"]:
        cx, cy, h = c["x"], c["y"], c["height"]
        if c["is_ph"]:
            cv2.fillPoly(img, [np.array([[cx, cy-10], [cx-10, cy+5], [cx+10, cy+5]])], (255, 0, 0))
        if c["is_pl"]:
            cv2.fillPoly(img, [np.array([[cx, cy+h+10], [cx-10, cy+h-5], [cx+10, cy+h-5]])], (128, 0, 128))

    # === 2. TRENDLINES ===
    for team in contour_data.get("ph_teams", []):
        s = team["sender"]
        for tl in team["trendlines"]:
            cv2.line(img, (s["x"], s["y"]), (tl["x"], tl["y"]), (255, 0, 0), 1)
            if tl.get("is_first"):
                star = np.array([[tl["x"], tl["y"]-15], [tl["x"]+4, tl["y"]-5], [tl["x"]+14, tl["y"]-5],
                                [tl["x"]+5, tl["y"]+2], [tl["x"]+10, tl["y"]+12], [tl["x"], tl["y"]+7],
                                [tl["x"]-10, tl["y"]+12], [tl["x"]-5, tl["y"]+2], [tl["x"]-14, tl["y"]-5],
                                [tl["x"]-4, tl["y"]-5]], np.int32)
                cv2.fillPoly(img, [star], (255, 0, 0))
            else:
                cv2.circle(img, (tl["x"], tl["y"]), 5, (255, 0, 0), -1)

    for team in contour_data.get("pl_teams", []):
        s = team["sender"]
        for tl in team["trendlines"]:
            cv2.line(img, (s["x"], s["y"]), (tl["x"], tl["y"]), (0, 255, 255), 1)
            if tl.get("is_first"):
                star = np.array([[tl["x"], tl["y"]-15], [tl["x"]+4, tl["y"]-5], [tl["x"]+14, tl["y"]-5],
                                [tl["x"]+5, tl["y"]+2], [tl["x"]+10, tl["y"]+12], [tl["x"], tl["y"]+7],
                                [tl["x"]-10, tl["y"]+12], [tl["x"]-5, tl["y"]+2], [tl["x"]-14, tl["y"]-5],
                                [tl["x"]-4, tl["y"]-5]], np.int32)
                cv2.fillPoly(img, [star], (0, 255, 255))
            else:
                cv2.circle(img, (tl["x"], tl["y"]), 5, (0, 255, 255), -1)

    # === 3. PROCESS ORDERS + HIT DETECTION + BOX DRAWING ===
    redrawn_data = {"marketname": symbol, "timeframe": timeframe_str, "orders": []}
    overlay = img.copy()

    # Track the leftmost OB candle for cropping
    crop_start_x = img.shape[1]  # fallback to full width

    for entry in ob_none_oi_entries:
        target_price = entry["entry_price"]
        order_type = entry["limit_order"]
        sl_price = calculated_orders.get(target_price, {}).get("sl_price")
        tp_price = calculated_orders.get(target_price, {}).get("tp_price")

        # Find OB candle position
        found = False
        cx, cy = 0, 0
        ob_candle_x_right = 0
        ob_candle_x_left = 0
        for i, cd in enumerate(candle_data):
            high = float(cd["high"])
            low = float(cd["low"])
            if abs(high - target_price) < 1e-6 or abs(low - target_price) < 1e-6:
                b = candle_bounds[i]
                cx = b["center_x"]
                cy = b["high_y"] if order_type == "buy_limit" else b["low_y"]
                ob_candle_x_right = b["x_right"]
                ob_candle_x_left = b["x_left"]
                found = True
                break
        if not found:
            continue

        # Update crop start: use the LEFT edge of this candle
        crop_start_x = min(crop_start_x, ob_candle_x_left)

        # === HIT DETECTION ===
        hit_info = {
            "entry_hit": False, "entry_hit_candle_highprice": None, "entry_hit_candle_lowprice": None,
            "sl_hit": False, "sl_hit_candle_highprice": None, "sl_hit_candle_lowprice": None,
            "tp_hit": False, "tp_hit_candle_highprice": None, "tp_hit_candle_lowprice": None
        }

        if next_candles and sl_price is not None and tp_price is not None:
            if order_type == "buy_limit":
                for c in next_candles:
                    low = float(c["low"])
                    high = float(c["high"])
                    if low <= target_price and not hit_info["entry_hit"]:
                        hit_info["entry_hit"] = True
                        hit_info["entry_hit_candle_lowprice"] = low
                        hit_info["entry_hit_candle_highprice"] = high
                if hit_info["entry_hit"]:
                    for c in next_candles:
                        low = float(c["low"])
                        high = float(c["high"])
                        if low <= target_price:
                            if low <= sl_price and not hit_info["sl_hit"] and not hit_info["tp_hit"]:
                                hit_info["sl_hit"] = True
                                hit_info["sl_hit_candle_lowprice"] = low
                                hit_info["sl_hit_candle_highprice"] = high
                            if high >= tp_price and not hit_info["tp_hit"] and not hit_info["sl_hit"]:
                                hit_info["tp_hit"] = True
                                hit_info["tp_hit_candle_highprice"] = high
                                hit_info["tp_hit_candle_lowprice"] = low
            elif order_type == "sell_limit":
                for c in next_candles:
                    high = float(c["high"])
                    low = float(c["low"])
                    if high >= target_price and not hit_info["entry_hit"]:
                        hit_info["entry_hit"] = True
                        hit_info["entry_hit_candle_highprice"] = high
                        hit_info["entry_hit_candle_lowprice"] = low
                if hit_info["entry_hit"]:
                    for c in next_candles:
                        high = float(c["high"])
                        low = float(c["low"])
                        if high >= target_price:
                            if high >= sl_price and not hit_info["sl_hit"] and not hit_info["tp_hit"]:
                                hit_info["sl_hit"] = True
                                hit_info["sl_hit_candle_highprice"] = high
                                hit_info["sl_hit_candle_lowprice"] = low
                            if low <= tp_price and not hit_info["tp_hit"] and not hit_info["sl_hit"]:
                                hit_info["tp_hit"] = True
                                hit_info["tp_hit_candle_highprice"] = high
                                hit_info["tp_hit_candle_lowprice"] = low

        # === DRAW ARROW + MARKER ===
        draw_right_arrow(img, cx, cy)
        draw_oi_marker(img, cx, cy)

        # === DRAW TP/SL BOXES (only from first next candle or full right) ===
        if sl_price is not None and tp_price is not None and first_next_x_right is not None:
            entry_y = price_to_y(target_price)
            sl_y = price_to_y(sl_price)
            tp_y = price_to_y(tp_price)
            box_left = max(first_next_x_right, ob_candle_x_left)  # Start from next candle or entry candle
            box_right = img.shape[1]

            if order_type == "buy_limit":
                if tp_y < entry_y:
                    cv2.rectangle(overlay, (box_left, tp_y), (box_right, entry_y), (0, 255, 0), -1)
                if sl_y > entry_y:
                    cv2.rectangle(overlay, (box_left, entry_y), (box_right, sl_y), (0, 0, 255), -1)
            elif order_type == "sell_limit":
                if tp_y > entry_y:
                    cv2.rectangle(overlay, (box_left, entry_y), (box_right, tp_y), (0, 255, 0), -1)
                if sl_y < entry_y:
                    cv2.rectangle(overlay, (box_left, sl_y), (box_right, entry_y), (0, 0, 255), -1)

        # === APPEND TO DATA ===
        redrawn_data["orders"].append({
            "ordertype": order_type,
            "entry_price": round(target_price, 6),
            "sl_price": round(sl_price, 6) if sl_price is not None else None,
            "tp_price": round(tp_price, 6) if tp_price is not None else None,
            **hit_info
        })

    # === APPLY TRANSPARENCY ===
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # === 4. SAVE FULL IMAGE ===
    cv2.imwrite(output_image_path, img)
    log_and_print(f"REDRAWN chart with correct TP/SL boxes → {output_image_path}", "SUCCESS")

    # === 5. CROP TO FIRST OB CANDLE & SAVE CROPPED VERSION ===
    if ob_none_oi_entries and crop_start_x < img.shape[1]:
        # Crop from crop_start_x to right edge, full height
        cropped_img = img[:, crop_start_x:]
        cv2.imwrite(output_cropped_image_path, cropped_img)
        log_and_print(f"CROPPED chart to first OB entry candle → {output_cropped_image_path}", "SUCCESS")
    else:
        # No OB entries → copy full image as cropped (or skip)
        cv2.imwrite(output_cropped_image_path, img)
        log_and_print(f"No OB entries; saved full chart as cropped → {output_cropped_image_path}", "INFO")

    # === 6. SAVE PER-TIMEFRAME JSON ===
    try:
        with open(output_redraw_json_path, 'w', encoding='utf-8') as f:
            json.dump(redrawn_data, f, indent=4)
        log_and_print(f"SAVED per-timeframe JSON", "SUCCESS")
    except Exception as e:
        error_log.append({"error": f"Save JSON failed: {str(e)}", "timestamp": now.isoformat()})

    # === 7. SCAN ALL TIMEFRAMES FOR OLDEST previouslatestcandle.json ===
    oldest_age_str = ""
    oldest_hours = -1

    base_dir = os.path.dirname(timeframe_folder)
    timeframes = ["5m", "15m", "30m", "1h", "4h"]

    # === 8. COLLECT ALL ORDERS FROM ALL TIMEFRAMES ===
    all_timeframes_data = {
        "oldestage_acrosstimeframe": oldest_age_str,
        "market": symbol,
        "timeframes": {tf: [] for tf in timeframes}
    }

    # Load existing alltimeframes data (if any)
    if os.path.exists(all_timeframes_json_path):
        try:
            with open(all_timeframes_json_path, 'r') as f:
                loaded = json.load(f)
            if loaded.get("market") == symbol:
                all_timeframes_data["oldestage_acrosstimeframe"] = loaded.get("oldestage_acrosstimeframe", "")
                all_timeframes_data["timeframes"] = loaded.get("timeframes", all_timeframes_data["timeframes"])
        except Exception as e:
            log_and_print(f"Load alltimeframes failed: {e}", "WARNING")

    # === Update current timeframe (without flags yet) ===
    current_tf_orders = [
        {k: o[k] for k in o if k not in ["team_type"]} for o in redrawn_data["orders"]
    ]
    all_timeframes_data["timeframes"][timeframe_str] = current_tf_orders

    # === Find oldest age ===
    for tf in timeframes:
        tf_folder = os.path.join(base_dir, tf)
        plc_path = os.path.join(tf_folder, "previouslatestcandle.json")
        if not os.path.exists(plc_path):
            continue
        try:
            with open(plc_path, 'r') as f:
                plc = json.load(f)
            age = plc.get("age", "")
            if not age:
                continue
            if "hour" in age:
                hours = int(age.split()[0])
            elif "day" in age:
                days = int(age.split()[0])
                hours = days * 24
            else:
                continue
            if hours > oldest_hours:
                oldest_hours = hours
                oldest_age_str = age
        except Exception as e:
            log_and_print(f"Failed to read {plc_path}: {e}", "WARNING")

    all_timeframes_data["oldestage_acrosstimeframe"] = oldest_age_str

    # === APPLY BUY/SELL FLAGS ACROSS ALL TIMEFRAMES ===
    all_buys = []
    all_sells = []

    for tf, orders in all_timeframes_data["timeframes"].items():
        for order in orders:
            if order["ordertype"] == "buy_limit":
                all_buys.append((order["entry_price"], tf, order))
            elif order["ordertype"] == "sell_limit":
                all_sells.append((order["entry_price"], tf, order))

    # Sort buys and sells
    all_buys.sort(key=lambda x: x[0])   # ascending price
    all_sells.sort(key=lambda x: x[0], reverse=True)  # descending price

    # === Mark Buy Flags ===
    if all_buys:
        lowest_buy_price = all_buys[0][0]
        highest_buy_price = all_buys[-1][0]
        is_single_buy = len(all_buys) == 1

        for price, tf, order in all_buys:
            order.update({
                "is_lowest_buy": price == lowest_buy_price and not is_single_buy,
                "is_highest_buy": price == highest_buy_price and not is_single_buy,
                "is_single_buy": is_single_buy
            })

    # === Mark Sell Flags ===
    if all_sells:
        highest_sell_price = all_sells[0][0]
        lowest_sell_price = all_sells[-1][0]
        is_single_sell = len(all_sells) == 1

        for price, tf, order in all_sells:
            order.update({
                "is_highest_sell": price == highest_sell_price and not is_single_sell,
                "is_lowest_sell": price == lowest_sell_price and not is_single_sell,
                "is_single_sell": is_single_sell
            })

    # === 9. SAVE FINAL ALLTIMEFRAMES JSON WITH FLAGS ===
    try:
        with open(all_timeframes_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_timeframes_data, f, indent=4)
        log_and_print(f"UPDATED alltimeframeslimitorders.json (oldest: {oldest_age_str}) with buy/sell flags", "SUCCESS")
    except Exception as e:
        error_log.append({"error": f"Save alltimeframes failed: {str(e)}", "timestamp": now.isoformat()})

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
        "forex": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\forexvolumesandrisk.json",
        "stocks": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\stocksvolumesandrisk.json",
        "indices": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\indicesvolumesandrisk.json",
        "synthetics": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\syntheticsvolumesandrisk.json",
        "commodities": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\commoditiesvolumesandrisk.json",
        "crypto": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\cryptovolumesandrisk.json",
        "equities": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\equitiesvolumesandrisk.json",
        "energies": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\energiesvolumesandrisk.json",
        "etfs": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\etfsvolumesandrisk.json",
        "basket_indices": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\basketindicesvolumesandrisk.json",
        "metals": r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\metalsvolumesandrisk.json"
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
    
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
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
    global_schedule = r"C:\xampp\htdocs\chronedge\fullordersschedules.json"
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
    
def placeallorders():
    """Run the updateorders script for M5 timeframe."""
    try:
        placeorders.main()
        print("all orders placed")
    except Exception as e:
        print(f"Error placing all orders: {e}")

def BreakevenRunningPositions():
    r"""
    Staged Breakeven:
      • Ratio 1 → SL to 0.25 (actual price shown)
      • Ratio 2 → SL to 0.50 (actual price shown)
    Clean logs, full precision, MT5-safe.
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
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


def calc_and_placeorders():  
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

def fetch_brokersdetails():
    """Run the updateorders script for M5 timeframe."""
    try:
        insiderfetch.fetch_insiders_server_columns()
        print("fetched insiders servers")
    except Exception as e:
        print(f"Error fetching insiders servers: {e}")

def fetch_charts_all_brokers(
    bars,
    neighborcandles_left,
    neighborcandles_right
):
    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    delete_all_category_jsons()
    delete_all_calculated_risk_jsons()
    delete_issue_jsons()
    required_allowed_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    fallback_allowed_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    allsymbols_path       = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    match_path            = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\symbolsmatch.json"
    brokers_report_path   = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allowedmarkets\brokerslimitorders.json"

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    IMPORTANT_TFS = {"15m", "30m", "1h", "4h"}

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
        log_and_print("\n=== NEW FULL CYCLE STARTED ===", "INFO")

        try:
            # ------------------------------------------------------------------
            # 0. RE-LOAD BLOCKED SYMBOLS
            # ------------------------------------------------------------------
            blocked_symbols_per_broker = {bn: set() for bn in brokersdictionary.keys()}
            if os.path.exists(brokers_report_path):
                try:
                    with open(brokers_report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    for order in report.get("pending_orders", []):
                        sym, broker = order.get("symbol"), order.get("broker")
                        if broker in blocked_symbols_per_broker:
                            blocked_symbols_per_broker[broker].add(sym)
                    for pos in report.get("open_positions", []):
                        sym, broker = pos.get("symbol"), pos.get("broker")
                        if broker in blocked_symbols_per_broker:
                            blocked_symbols_per_broker[broker].add(sym)
                    for hist in report.get("history_orders", []):
                        sym, broker = hist.get("symbol"), hist.get("broker")
                        age_str = hist.get("age", "")
                        if "h" in age_str or "m" in age_str or "s" in age_str:
                            if any(p in age_str for p in ["m", "s"]) or ("h" in age_str and int(age_str.split("h")[0]) < 5):
                                if broker in blocked_symbols_per_broker:
                                    blocked_symbols_per_broker[broker].add(sym)
                    log_and_print(f"Loaded blocked symbols from {brokers_report_path}", "INFO")
                    for bn, syms in blocked_symbols_per_broker.items():
                        if syms:
                            log_and_print(f"  → {bn.upper()}: {len(syms)} blocked", "INFO")
                except Exception as e:
                    log_and_print(f"FAILED to load brokerslimitorders.json: {e}", "ERROR")
            else:
                log_and_print("brokerslimitorders.json not found – no blocks", "WARNING")

            # ------------------------------------------------------------------
            # 0.5 DELETE NON-BLOCKED FOLDERS
            # ------------------------------------------------------------------
            for bn, cfg in brokersdictionary.items():
                blocked = blocked_symbols_per_broker.get(bn, set())
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
            # 5. Build candidate list WITH FINAL CASE-INSENSITIVE SYMBOLS CHECK
            # ------------------------------------------------------------------
            broker_name_mapping = {}
            for broker_name in brokersdictionary.keys():
                # Extract the base broker name by removing trailing digits
                base_name = re.sub(r'\d+$', '', broker_name)
                broker_name_mapping[broker_name] = base_name

            all_cats = ["stocks","forex","crypto","synthetics","indices","commodities","equities","energies","etfs","basket_indices","metals"]

            candidates = {}
            total_to_do = 0

            for broker_name, cfg in brokersdictionary.items():
                mapped = broker_name_mapping.get(broker_name, broker_name)
                candidates[broker_name] = {c: [] for c in all_cats}
                blocked = blocked_symbols_per_broker.get(broker_name, set())

                # Parse broker's SYMBOLS — case-insensitive
                broker_symbols_raw = cfg.get("SYMBOLS", "").strip()
                if broker_symbols_raw.lower() == "all":
                    broker_allowed_symbols = None  # Allow all
                else:
                    # Normalize and store as uppercase set
                    broker_allowed_symbols = {
                        normalize_symbol(s) for s in broker_symbols_raw.split(",") if s.strip()
                    }

                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if not ok:
                    mt5.shutdown(); continue
                avail, _ = get_symbols()
                mt5.shutdown()

                for entry in symbolsmatch_data.get("main_symbols", []):
                    for sym in entry.get(mapped, []):
                        if sym not in avail:
                            continue
                        if sym in blocked:
                            continue

                        cat = symbol_to_category.get(sym)
                        if not cat or cat not in all_cats:
                            continue
                        if allowed_config.get(cat, {}).get("limited", False):
                            if normalize_symbol(sym) not in normalized_allowed.get(cat, set()):
                                continue

                        # FINAL VERIFICATION: Broker-specific SYMBOLS check (case-insensitive)
                        if broker_allowed_symbols is not None:
                            if normalize_symbol(sym) not in broker_allowed_symbols:
                                continue  # Skip if not in broker's SYMBOLS list

                        if symbol_needs_processing(sym, cfg["BASE_FOLDER"]):
                            delete_symbol_folder(sym, cfg["BASE_FOLDER"], "(pre-process cleanup)")
                            candidates[broker_name][cat].append(sym)

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
            # 6. ROUND-ROBIN PROCESSING
            # ------------------------------------------------------------------
            remaining = {b: {c: candidates[b][c][:] for c in all_cats} for b in brokersdictionary}
            indices   = {b: {c: 0 for c in all_cats} for b in brokersdictionary}

            round_no = 1
            while any(any(remaining[b][c]) for b in brokersdictionary for c in all_cats):
                log_and_print(f"\n--- ROUND {round_no} ---", "INFO")

                for cat in all_cats:
                    for bn, cfg in brokersdictionary.items():
                        if not remaining[bn][cat]:
                            continue

                        idx = indices[bn][cat]
                        if idx >= len(remaining[bn][cat]):
                            remaining[bn][cat] = []
                            continue

                        symbol = remaining[bn][cat][idx]

                        if symbol in blocked_symbols_per_broker.get(bn, set()):
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
                                    chart_path, ch_errs, ph, pl = generate_and_save_chart(
                                        df, symbol, tf_str, tf_folder,
                                        neighborcandles_left, neighborcandles_right
                                    )
                                    error_log.extend(ch_errs)

                                    save_candle_data(df, symbol, tf_str, tf_folder, ph, pl)
                                    next_errs = save_next_candles(df, symbol, tf_str, tf_folder, ph, pl)
                                    error_log.extend(next_errs)

                                    if chart_path:
                                        crop_chart(chart_path, symbol, tf_str, tf_folder)
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
                                        redraw_errors = redraw_contours_from_json(chart_path, symbol, tf_str, tf_folder)
                                        error_log.extend(redraw_errors)

                                mt5.shutdown()

                            roundgoblin()
                            collect_ob_none_oi_data(symbol, sym_folder, bn, cfg["BASE_FOLDER"], candidates[bn][cat])
                            calc_and_placeorders()

                        indices[bn][cat] += 1

                round_no += 1

            save_errors(error_log)
            calc_and_placeorders()
            log_and_print("CYCLE 100% COMPLETED", "SUCCESS")
            log_and_print("Sleeping 30 minutes...", "INFO")
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

        
        
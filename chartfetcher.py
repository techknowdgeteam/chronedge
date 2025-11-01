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
import timeorders

# Brokers configuration
brokersdictionary = {
    "deriv": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 deriv\terminal64.exe",
        "LOGIN_ID": "101347351",
        "PASSWORD": "@Techknowdge12#",
        "SERVER": "DerivSVG-Server-02",
        "ACCOUNT": "demo",
        "STRATEGY": "both",
        "BASE_FOLDER": r"C:\xampp\htdocs\chronedge\chart\deriv\derivsymbols"
    },
    "bybit1": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 bybit 1\terminal64.exe",
        "LOGIN_ID": "4836528",
        "PASSWORD": "@Techknowdge12#",
        "SERVER": "Bybit-Live",
        "ACCOUNT": "real",
        "STRATEGY": "lowtohigh",
        "BASE_FOLDER": r"C:\xampp\htdocs\chronedge\chart\bybit 1\bybit1symbols"
    }
}

BASE_ERROR_FOLDER = r"C:\xampp\htdocs\chronedge\chart\debugs"
TIMEFRAME_MAP = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4
}
ERROR_JSON_PATH = os.path.join(BASE_ERROR_FOLDER, "chart_errors.json")
           
def clear_chart_folder(base_folder):
    """Clear all contents of the chart folder to ensure fresh data is saved."""
    error_log = []
    try:
        if not os.path.exists(base_folder):
            log_and_print(f"Chart folder {base_folder} does not exist, no need to clear.", "INFO")
            return True, error_log

        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                log_and_print(f"Deleted {item_path}", "INFO")
            except Exception as e:
                error_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "error": f"Failed to delete {item_path}: {str(e)}",
                    "broker": base_folder
                })
                log_and_print(f"Failed to delete {item_path}: {str(e)}", "ERROR")

        log_and_print(f"Chart folder {base_folder} cleared successfully", "SUCCESS")
        return True, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to clear chart folder {base_folder}: {str(e)}",
            "broker": base_folder
        })
        save_errors(error_log)
        log_and_print(f"Failed to clear chart folder {base_folder}: {str(e)}", "ERROR")
        return False, error_log

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
    """Save all candle data with numbering and PH/PL labels."""
    error_log = []
    candle_json_path = os.path.join(timeframe_folder, "all_candles.json")
    try:
        if len(df) >= 2:
            candles = []
            ph_dict = {t: label for label, _, t in ph_labels}
            pl_dict = {t: label for label, _, t in pl_labels}

            for i, (index, row) in enumerate(df[::-1].iterrows()):
                candle = row.to_dict()
                candle["time"] = index.strftime('%Y-%m-%d %H:%M:%S')
                candle["candle_number"] = i
                candle["symbol"] = symbol
                candle["timeframe"] = timeframe_str
                candle["is_ph"] = ph_dict.get(index, None) == 'PH'
                candle["is_pl"] = pl_dict.get(index, None) == 'PL'
                candles.append(candle)
            with open(candle_json_path, 'w') as f:
                json.dump(candles, f, indent=4)
            log_and_print(f"Candle data saved for {symbol} ({timeframe_str})", "SUCCESS")
        else:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Not enough data to save candles for {symbol} ({timeframe_str})",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Not enough data to save candles for {symbol} ({timeframe_str})", "ERROR")
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save candles for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to save candles for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
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
            returnfig=True
        )

        # Adjust wick thickness for basic chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=100)
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
    If symbol not found directly, use symbols_match.json to map via broker-specific list to a main symbol and retrieve risk/volume."""

    error_log = []
    all_timeframes_data = {tf: [] for tf in TIMEFRAME_MAP.keys()}
    allmarkets_json_path = os.path.join(base_folder, "allmarkets_limitorders.json")
    allnoordermarkets_json_path = os.path.join(base_folder, "allnoordermarkets.json")
    allsymbols_json_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allsymbolsvolumesandrisk.json"
    symbols_match_json_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\symbols_match.json"  # <-- NEW

    # Paths for market-type-specific JSONs
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

    # === ENHANCED: FIND SYMBOL IN allsymbols OR VIA symbols_match.json ===
    market_type = None
    risk_volume_map = {}  # {risk_amount: volume}
    mapped_main_symbol = None

    def find_symbol_in_allsymbols(target_symbol):
        """Search allsymbolsvolumesandrisk.json for target_symbol and return market_type + risk_volume_map"""
        try:
            with open(allsymbols_json_path, 'r') as f:
                allsymbols_data = json.load(f)

            for risk_key, markets in allsymbols_data.items():
                try:
                    risk_amount = float(risk_key.split(": ")[1])
                    for mkt_type in market_type_paths.keys():
                        for item in markets.get(mkt_type, []):
                            if item["symbol"] == target_symbol:
                                return mkt_type, {risk_amount: item["volume"]}
                except:
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
        # Step 2: Fallback to symbols_match.json
        log_and_print(f"Direct match failed for {symbol}. Trying symbols_match.json...", "INFO")
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
                    "error": f"symbols_match.json not found at {symbols_match_json_path}",
                    "broker": broker_name
                })
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to process symbols_match.json: {str(e)}",
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
                                            "broker": broker_name  # Optional: tag with broker
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
    
def crop_chart(chart_path, symbol, timeframe_str, timeframe_folder):
    """Crop the saved chart.png and chartanalysed.png images, then detect candle contours only for chart.png."""
    error_log = []
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")

    try:
        # Crop chart.png
        with Image.open(chart_path) as img:
            right = 8
            left = 80
            top = 80
            bottom = 70
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

def calculate_symbols_sl_tp_prices():
    """Run the updateorders script for M5 timeframe."""
    try:
        calculateprices.main()
        print("symbols prices calculated ")
    except Exception as e:
        print(f"Error when calculating symbols prices: {e}")

def place_demo_orders():
    r"""
    Place demo pending orders with full diagnostics and auto-corrections.
    - Only 1 BUY_LIMIT and 1 SELL_LIMIT per symbol
    - Failed → ordersissues.json + report JSON (NO LOG)
    - Skipped (price past / duplicate / running position) → logged + reported
    - No new pending if same-direction position is running
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    REPORT_SUFFIX = "forex_order_report.json"
    ISSUES_FILE = "ordersissues.json"

    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd",
        3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd",
    }
    STRATEGY_FILES = ["hightolow.json", "lowtohigh.json"]

    # ------------------------------------------------------------------ #
    def _send_with_fallback(request, symbol_info):
        supported = symbol_info.filling_mode
        FILLING_FOK, FILLING_IOC, FILLING_RETURN, FILLING_GTC = 1, 2, 4, 8
        modes = []
        if supported & FILLING_FOK:     modes.append(FILLING_FOK)
        if supported & FILLING_IOC:     modes.append(FILLING_IOC)
        if supported & FILLING_RETURN:  modes.append(FILLING_RETURN)
        if supported & FILLING_GTC:     modes.append(FILLING_GTC)
        priority = [FILLING_IOC, FILLING_RETURN, FILLING_FOK, FILLING_GTC]
        for mode in priority:
            if mode in modes:
                request["type_filling"] = mode
                result = mt5.order_send(request)
                if result is None:
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    return result, mode
                if "unsupported filling" in result.comment.lower():
                    continue
                return result, mode
        return None, None

    # ------------------------------------------------------------------ #
    for broker_name, broker_cfg in brokersdictionary.items():
        broker_account = broker_cfg.get("ACCOUNT", "").lower()
        if broker_account != "demo":
            log_and_print(f"Skipping {broker_name} (account type: {broker_account})", "INFO")
            continue

        TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
        LOGIN_ID = broker_cfg["LOGIN_ID"]
        PASSWORD = broker_cfg["PASSWORD"]
        SERVER = broker_cfg["SERVER"]

        log_and_print(f"Processing demo broker: {broker_name}", "INFO")

        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"{broker_name}: Terminal path not found ({TERMINAL_PATH})", "ERROR")
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                              server=SERVER, timeout=30000):
            err = f"{broker_name}: MT5 init failed: {mt5.last_error()}"
            log_and_print(err, "ERROR")
            _write_global_error_report(os.path.join(BASE_INPUT_DIR, broker_name), RISK_FOLDERS, err)
            continue

        if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            err = f"{broker_name}: MT5 login failed: {mt5.last_error()}"
            log_and_print(err, "ERROR")
            mt5.shutdown()
            _write_global_error_report(os.path.join(BASE_INPUT_DIR, broker_name), RISK_FOLDERS, err)
            continue

        log_and_print(f"{broker_name} connected successfully.", "SUCCESS")

        total_placed = total_failed = total_skipped = 0
        issues_list = []

        # Track existing pending orders and running positions per symbol
        existing_pending = {}  # (symbol, type) → ticket
        running_positions = {}  # symbol → direction: 1=buy, -1=sell, 0=none

        pending = mt5.orders_get()
        for order in pending or []:
            if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                key = (order.symbol, order.type)
                existing_pending[key] = order.ticket

        positions = mt5.positions_get()
        for pos in positions or []:
            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            running_positions[pos.symbol] = direction

        for risk_usd, risk_folder in RISK_FOLDERS.items():
            for strat_file in STRATEGY_FILES:
                calc_file = Path(BASE_INPUT_DIR) / broker_name / risk_folder / strat_file
                if not calc_file.exists():
                    log_and_print(f"{broker_name}: Missing {calc_file}", "WARNING")
                    continue

                try:
                    with calc_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        entries = data.get("entries", [])
                except Exception as e:
                    log_and_print(f"{broker_name}: Cannot read {calc_file}: {e}", "ERROR")
                    continue

                if not entries:
                    log_and_print(f"{broker_name}: Empty {calc_file}", "WARNING")
                    continue

                report_file = Path(BASE_INPUT_DIR) / broker_name / risk_folder / REPORT_SUFFIX
                existing_reports = []
                if report_file.exists():
                    try:
                        with report_file.open("r", encoding="utf-8") as f:
                            existing_reports = json.load(f)
                    except:
                        existing_reports = []

                for entry in entries:
                    try:
                        symbol = entry["market"]
                        raw_volume = float(entry["volume"])
                        price = float(entry["entry_price"])
                        sl_price = float(entry["sl_price"])
                        tp_price = float(entry["tp_price"])
                        order_type_str = entry["limit_order"]
                        order_type = (
                            mt5.ORDER_TYPE_BUY_LIMIT
                            if order_type_str == "buy_limit"
                            else mt5.ORDER_TYPE_SELL_LIMIT
                        )

                        symbol_info = mt5.symbol_info(symbol)
                        tick = mt5.symbol_info_tick(symbol)
                        now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime(
                            "%Y-%m-%d %H:%M:%S.%f+01:00")

                        can_place = True
                        reason = ""

                        # === DUPLICATE PENDING CHECK ===
                        dup_key = (symbol, order_type)
                        if dup_key in existing_pending:
                            skip_reason = f"Duplicate {order_type_str.upper()} already exists (ticket: {existing_pending[dup_key]})"
                            total_skipped += 1
                            log_and_print(
                                f"{broker_name} | Risk ${risk_usd}: {symbol} {order_type_str} @ {price} → {skip_reason} → SKIPPED",
                                "INFO"
                            )
                            report_entry = {
                                "symbol": symbol,
                                "order_type": order_type_str,
                                "price": price,
                                "volume": raw_volume,
                                "sl": sl_price,
                                "tp": tp_price,
                                "risk_usd": entry["riskusd_amount"],
                                "ticket": None,
                                "success": False,
                                "error_code": None,
                                "error_msg": f"SKIPPED: {skip_reason}",
                                "timestamp": now_str,
                            }
                            existing_reports.append(report_entry)
                            try:
                                with report_file.open("w", encoding="utf-8") as f:
                                    json.dump(existing_reports, f, indent=2)
                            except Exception as e:
                                log_and_print(f"{broker_name}: Failed to write report {symbol}: {e}", "ERROR")
                            continue

                        # === RUNNING POSITION CHECK (same direction) ===
                        current_dir = running_positions.get(symbol, 0)
                        new_dir = 1 if order_type == mt5.ORDER_TYPE_BUY_LIMIT else -1
                        if current_dir == new_dir:
                            skip_reason = f"Running {'BUY' if new_dir == 1 else 'SELL'} position already open"
                            total_skipped += 1
                            log_and_print(
                                f"{broker_name} | Risk ${risk_usd}: {symbol} {order_type_str} @ {price} → {skip_reason} → SKIPPED",
                                "INFO"
                            )
                            report_entry = {
                                "symbol": symbol,
                                "order_type": order_type_str,
                                "price": price,
                                "volume": raw_volume,
                                "sl": sl_price,
                                "tp": tp_price,
                                "risk_usd": entry["riskusd_amount"],
                                "ticket": None,
                                "success": False,
                                "error_code": None,
                                "error_msg": f"SKIPPED: {skip_reason}",
                                "timestamp": now_str,
                            }
                            existing_reports.append(report_entry)
                            try:
                                with report_file.open("w", encoding="utf-8") as f:
                                    json.dump(existing_reports, f, indent=2)
                            except Exception as e:
                                log_and_print(f"{broker_name}: Failed to write report {symbol}: {e}", "ERROR")
                            continue

                        # 1. Symbol checks
                        if not symbol_info:
                            reason = "Symbol not found on server"
                            can_place = False
                        elif not symbol_info.visible:
                            reason = "Symbol not enabled"
                            can_place = False
                        elif symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                            reason = f"Trade restricted: {symbol_info.trade_mode}"
                            can_place = False
                        elif tick is None:
                            reason = "No tick data (market closed?)"
                            can_place = False

                        # 2. Volume auto-fix
                        volume = raw_volume
                        if can_place and symbol_info:
                            lot_step = symbol_info.volume_step
                            min_lot = symbol_info.volume_min
                            max_lot = symbol_info.volume_max
                            volume = max(min_lot, round(raw_volume / lot_step) * lot_step)
                            volume = min(volume, max_lot)
                            if abs(volume - raw_volume) > 1e-8:
                                reason = f"Volume auto-fixed: {raw_volume} → {volume}"

                        # 3. Price distance (SKIP if already past market)
                        skip_reason = None
                        if can_place and tick:
                            bid, ask = tick.bid, tick.ask
                            point = symbol_info.point
                            if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                                if price >= ask:
                                    skip_reason = f"BUY_LIMIT {price} >= ask {ask} (already above market)"
                                elif (ask - price) < point * 10:
                                    reason = f"BUY_LIMIT too close to ask ({(ask - price)/point:.1f} pts)"
                                    can_place = False
                            else:
                                if price <= bid:
                                    skip_reason = f"SELL_LIMIT {price} <= bid {bid} (already below market)"
                                elif (price - bid) < point * 10:
                                    reason = f"SELL_LIMIT too close to bid ({(price - bid)/point:.1f} pts)"
                                    can_place = False

                        if skip_reason:
                            total_skipped += 1
                            log_and_print(
                                f"{broker_name} | Risk ${risk_usd}: {symbol} {order_type_str} @ {price} → {skip_reason} → SKIPPED",
                                "INFO"
                            )
                            report_entry = {
                                "symbol": symbol,
                                "order_type": order_type_str,
                                "price": price,
                                "volume": volume,
                                "sl": sl_price,
                                "tp": tp_price,
                                "risk_usd": entry["riskusd_amount"],
                                "ticket": None,
                                "success": False,
                                "error_code": None,
                                "error_msg": f"SKIPPED: {skip_reason}",
                                "timestamp": now_str,
                            }
                            existing_reports.append(report_entry)
                            try:
                                with report_file.open("w", encoding="utf-8") as f:
                                    json.dump(existing_reports, f, indent=2)
                            except Exception as e:
                                log_and_print(f"{broker_name}: Failed to write report {symbol}: {e}", "ERROR")
                            continue

                        # 4. SL/TP logic
                        if can_place:
                            if sl_price <= 0 or tp_price <= 0:
                                reason = "SL/TP ≤ 0"
                                can_place = False
                            elif order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                                if sl_price >= price or tp_price <= price:
                                    reason = "Invalid SL/TP for BUY_LIMIT"
                                    can_place = False
                            else:
                                if sl_price <= price or tp_price >= price:
                                    reason = "Invalid SL/TP for SELL_LIMIT"
                                    can_place = False

                        # ---- Build request ----
                        request = {
                            "action": mt5.TRADE_ACTION_PENDING,
                            "symbol": symbol,
                            "volume": volume,
                            "type": order_type,
                            "price": price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "deviation": 10,
                            "magic": 123456,
                            "comment": f"AutoSLTP_{entry['riskusd_amount']}USD",
                            "type_time": mt5.ORDER_TIME_GTC,
                        }

                        # ---- SEND WITH FALLBACK ----
                        result = None
                        used_filling = None
                        if can_place:
                            result, used_filling = _send_with_fallback(request, symbol_info)
                        else:
                            result = type('obj', (), {
                                'retcode': 10000,
                                'comment': reason,
                                'order': None
                            })()
                            used_filling = "N/A"

                        # === CRITICAL FIX: Ensure result is never None ===
                        if result is None:
                            status = "FAILED"
                            final_reason = "order_send returned None (possible connection issue)"
                            result = type('obj', (), {
                                'retcode': mt5.TRADE_RETCODE_TIMEOUT,
                                'comment': final_reason,
                                'order': None
                            })()
                        else:
                            status = "SUCCESS" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED"
                            final_reason = reason or result.comment

                        # Update existing_pending on success
                        if status == "SUCCESS" and result.order:
                            existing_pending[(symbol, order_type)] = result.order

                        report_entry = {
                            "symbol": symbol,
                            "order_type": order_type_str,
                            "price": price,
                            "volume": volume,
                            "sl": sl_price,
                            "tp": tp_price,
                            "risk_usd": entry["riskusd_amount"],
                            "ticket": result.order if status == "SUCCESS" else None,
                            "success": status == "SUCCESS",
                            "error_code": result.retcode if status == "FAILED" else None,
                            "error_msg": result.comment if status == "FAILED" else None,
                            "timestamp": now_str,
                            "filling_used": used_filling,
                        }
                        existing_reports.append(report_entry)
                        try:
                            with report_file.open("w", encoding="utf-8") as f:
                                json.dump(existing_reports, f, indent=2)
                        except Exception as e:
                            log_and_print(f"{broker_name}: Failed to write report {symbol}: {e}", "ERROR")

                        # === LOGGING: SUCCESS & SKIPPED → LOGGED, FAILED → SILENT (only in JSON) ===
                        if status == "SUCCESS":
                            log_msg = f"{broker_name} | Risk ${risk_usd}: {symbol} {order_type_str} @ {price} → SL {sl_price} | TP {tp_price} | SUCCESS"
                            if used_filling and used_filling != "N/A":
                                log_msg += f" (filling: {used_filling})"
                            log_and_print(log_msg, "INFO")
                            total_placed += 1
                        else:  # FAILED
                            total_failed += 1
                            issues_list.append({"symbol": symbol, "diagnosed_reason": final_reason})
                            # NO log_and_print() for FAILED orders

                    except Exception as e:
                        log_and_print(f"{broker_name}: Exception placing {symbol} → {e}", "ERROR")
                        total_failed += 1
                        issues_list.append({"symbol": symbol, "diagnosed_reason": f"Exception: {e}"})

        # ---- Save issues (only failed orders) ----
        issues_file = Path(BASE_INPUT_DIR) / broker_name / ISSUES_FILE
        try:
            existing_issues = json.load(issues_file.open("r", encoding="utf-8")) if issues_file.exists() else []
        except:
            existing_issues = []
        all_issues = existing_issues + issues_list
        try:
            with issues_file.open("w", encoding="utf-8") as f:
                json.dump(all_issues, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to write {ISSUES_FILE}: {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Demo orders completed – Placed: {total_placed}, Failed: {total_failed}, Skipped: {total_skipped}",
            "SUCCESS"
        )

    log_and_print("All demo brokers processed successfully.", "SUCCESS")
    
def place_real_orders():
    r"""
    Place real account pending orders with balance-based risk selection.
    - Only 1 BUY_LIMIT and 1 SELL_LIMIT per symbol
    - Strategy-aware price optimisation (lowtohigh / hightolow)
    - No new pending if same-direction position is running
    - FAILED orders → silent (only in JSON + issues file)
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    REPORT_SUFFIX = "forex_order_report.json"
    ISSUES_FILE = "ordersissues.json"

    BALANCE_TO_RISK = [
        (0.5, 3.8, "risk_0_50cent_usd"), (3.8, 8.0, "risk_1_usd"), (8.0, 12.0, "risk_2_usd"),
        (12.0, 16.0, "risk_3_usd"), (16.0, 96.0, "risk_4_usd"), (96.0, 150.0, "risk_8_usd"),
        (150.0, float('inf'), "risk_1_usd")
    ]
    STRATEGY_FILE_MAP = {"lowtohigh": "lowtohigh.json", "hightolow": "hightolow.json"}

    # ------------------------------------------------------------------ #
    def _send_with_fallback(request, symbol_info):
        supported = symbol_info.filling_mode
        FILLING_FOK, FILLING_IOC, FILLING_RETURN, FILLING_GTC = 1, 2, 4, 8
        modes = []
        if supported & FILLING_FOK:     modes.append(FILLING_FOK)
        if supported & FILLING_IOC:     modes.append(FILLING_IOC)
        if supported & FILLING_RETURN:  modes.append(FILLING_RETURN)
        if supported & FILLING_GTC:     modes.append(FILLING_GTC)
        priority = [FILLING_IOC, FILLING_RETURN, FILLING_FOK, FILLING_GTC]
        for mode in priority:
            if mode in modes:
                request["type_filling"] = mode
                result = mt5.order_send(request)
                if result is None:
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    return result, mode
                if "unsupported filling" in result.comment.lower():
                    continue
                return result, mode
        return None, None

    # ------------------------------------------------------------------ #
    for broker_name, broker_cfg in brokersdictionary.items():
        broker_account = broker_cfg.get("ACCOUNT", "").lower()
        if broker_account != "real":
            log_and_print(f"Skipping {broker_name} (account type: {broker_account})", "INFO")
            continue

        strategy_key = broker_cfg.get("STRATEGY", "").lower()
        if strategy_key not in STRATEGY_FILE_MAP:
            log_and_print(f"{broker_name}: Invalid STRATEGY '{strategy_key}'", "ERROR")
            continue

        # --------------------------------------------------------------
        # Helper: decide whether to replace an existing pending order
        # --------------------------------------------------------------
        def should_replace(existing_price, new_price, order_type, strategy):
            is_buy = order_type == mt5.ORDER_TYPE_BUY_LIMIT
            if strategy == "lowtohigh":
                if is_buy:
                    return (new_price > existing_price,
                            f"New BUY entry {new_price} > existing {existing_price} → replace")
                else:
                    return (new_price < existing_price,
                            f"New SELL entry {new_price} < existing {existing_price} → replace")
            else:  # hightolow
                if is_buy:
                    return (new_price < existing_price,
                            f"New BUY entry {new_price} < existing {existing_price} → replace")
                else:
                    return (new_price > existing_price,
                            f"New SELL entry {new_price} > existing {existing_price} → replace")

        # --------------------------------------------------------------
        strategy_file = STRATEGY_FILE_MAP[strategy_key]
        TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
        LOGIN_ID = broker_cfg["LOGIN_ID"]
        PASSWORD = broker_cfg["PASSWORD"]
        SERVER = broker_cfg["SERVER"]

        log_and_print(f"Processing real broker: {broker_name} | Strategy: {strategy_key}", "INFO")

        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"{broker_name}: Terminal path not found ({TERMINAL_PATH})", "ERROR")
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                              server=SERVER, timeout=30000):
            err = f"{broker_name}: MT5 init failed: {mt5.last_error()}"
            log_and_print(err, "ERROR")
            _write_global_error_report(os.path.join(BASE_INPUT_DIR, broker_name), {}, err)
            continue

        if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            err = f"{broker_name}: MT5 login failed: {mt5.last_error()}"
            log_and_print(err, "ERROR")
            mt5.shutdown()
            _write_global_error_report(os.path.join(BASE_INPUT_DIR, broker_name), {}, err)
            continue

        account_info = mt5.account_info()
        if not account_info:
            err = f"{broker_name}: Failed to get account info: {mt5.last_error()}"
            log_and_print(err, "ERROR")
            mt5.shutdown()
            continue

        balance = account_info.balance
        log_and_print(f"{broker_name}: Account balance = ${balance:.2f}", "INFO")

        selected_risk_folder = next(
            (folder for min_bal, max_bal, folder in BALANCE_TO_RISK if min_bal <= balance < max_bal),
            "risk_1_usd"
        )
        risk_usd = float(selected_risk_folder.split("_")[1].replace("usd", "").replace("cent", "0.5"))
        log_and_print(f"{broker_name}: Selected risk folder → {selected_risk_folder} (${risk_usd} risk)", "INFO")

        calc_file = Path(BASE_INPUT_DIR) / broker_name / selected_risk_folder / strategy_file
        if not calc_file.exists():
            log_and_print(f"{broker_name}: Missing file {calc_file}", "WARNING")
            mt5.shutdown()
            continue

        try:
            with calc_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                entries = data.get("entries", [])
        except Exception as e:
            log_and_print(f"{broker_name}: Cannot read {calc_file}: {e}", "ERROR")
            mt5.shutdown()
            continue

        if not entries:
            log_and_print(f"{broker_name}: Empty {calc_file}", "INFO")
            mt5.shutdown()
            continue

        report_file = calc_file.parent / REPORT_SUFFIX
        existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []

        total_placed = total_failed = total_skipped = 0
        issues_list = []
        now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")

        # === Load existing pending orders and running positions ===
        existing_orders = {}          # (symbol, type) → {'ticket': int, 'price': float}
        running_positions = {}        # symbol → direction: 1=buy, -1=sell

        pending = mt5.orders_get()
        for order in pending or []:
            if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                existing_orders[(order.symbol, order.type)] = {'ticket': order.ticket, 'price': order.price_open}

        positions = mt5.positions_get()
        for pos in positions or []:
            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            running_positions[pos.symbol] = direction

        for entry in entries:
            try:
                symbol = entry["market"]
                raw_volume = float(entry["volume"])
                price = float(entry["entry_price"])
                sl_price = float(entry["sl_price"])
                tp_price = float(entry["tp_price"])
                order_type_str = entry["limit_order"]
                order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                symbol_info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)

                can_place = True
                reason = ""

                # === RUNNING POSITION CHECK (same direction) ===
                current_dir = running_positions.get(symbol, 0)
                new_dir = 1 if order_type == mt5.ORDER_TYPE_BUY_LIMIT else -1
                if current_dir == new_dir:
                    skip_reason = f"Running {'BUY' if new_dir == 1 else 'SELL'} position already open"
                    total_skipped += 1
                    log_and_print(f"{broker_name} | ${risk_usd} | {symbol} {order_type_str} @ {price} → {skip_reason} → SKIPPED", "INFO")
                    report_entry = {
                        "symbol": symbol, "order_type": order_type_str, "price": price,
                        "volume": raw_volume, "sl": sl_price, "tp": tp_price,
                        "risk_usd": entry["riskusd_amount"], "ticket": None,
                        "success": False, "error_code": None,
                        "error_msg": f"SKIPPED: {skip_reason}", "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except: pass
                    continue

                # === DUPLICATE / PRICE-OPTIMISATION CHECK ===
                dup_key = (symbol, order_type)
                if dup_key in existing_orders:
                    old = existing_orders[dup_key]
                    replace, replace_reason = should_replace(old['price'], price, order_type, strategy_key)

                    if replace:
                        del_req = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": old['ticket']
                        }
                        del_res = mt5.order_send(del_req)
                        if del_res is None:
                            log_and_print(
                                f"{broker_name} | ${risk_usd} | {symbol} order_send returned None on delete",
                                "ERROR"
                            )
                            total_skipped += 1
                            report_entry = {
                                "symbol": symbol, "order_type": order_type_str, "price": price,
                                "volume": raw_volume, "sl": sl_price, "tp": tp_price,
                                "risk_usd": entry["riskusd_amount"], "ticket": None,
                                "success": False, "error_code": mt5.TRADE_RETCODE_TIMEOUT,
                                "error_msg": "DELETE FAILED: order_send returned None",
                                "timestamp": now_str
                            }
                            existing_reports.append(report_entry)
                            try:
                                with report_file.open("w", encoding="utf-8") as f:
                                    json.dump(existing_reports, f, indent=2)
                            except: pass
                            continue
                        if del_res.retcode != mt5.TRADE_RETCODE_DONE:
                            log_and_print(
                                f"{broker_name} | ${risk_usd} | {symbol} FAILED to delete old {order_type_str} ticket {old['ticket']} → {del_res.comment}",
                                "WARNING"
                            )
                            total_skipped += 1
                            report_entry = {
                                "symbol": symbol, "order_type": order_type_str, "price": price,
                                "volume": raw_volume, "sl": sl_price, "tp": tp_price,
                                "risk_usd": entry["riskusd_amount"], "ticket": None,
                                "success": False, "error_code": del_res.retcode,
                                "error_msg": f"DELETE FAILED: {del_res.comment}",
                                "timestamp": now_str
                            }
                            existing_reports.append(report_entry)
                            try:
                                with report_file.open("w", encoding="utf-8") as f:
                                    json.dump(existing_reports, f, indent=2)
                            except: pass
                            continue

                        log_and_print(
                            f"{broker_name} | ${risk_usd} | {symbol} DELETED old {order_type_str} @ {old['price']} (ticket {old['ticket']}) → {replace_reason}",
                            "INFO"
                        )
                        del existing_orders[dup_key]
                    else:
                        total_skipped += 1
                        skip_msg = f"New {order_type_str} @ {price} not better than existing @ {old['price']} → SKIPPED"
                        log_and_print(f"{broker_name} | ${risk_usd} | {symbol} {skip_msg}", "INFO")
                        report_entry = {
                            "symbol": symbol, "order_type": order_type_str, "price": price,
                            "volume": raw_volume, "sl": sl_price, "tp": tp_price,
                            "risk_usd": entry["riskusd_amount"], "ticket": None,
                            "success": False, "error_code": None,
                            "error_msg": f"SKIPPED: {skip_msg}", "timestamp": now_str
                        }
                        existing_reports.append(report_entry)
                        try:
                            with report_file.open("w", encoding="utf-8") as f:
                                json.dump(existing_reports, f, indent=2)
                        except: pass
                        continue

                # === Symbol / market checks ===
                if not symbol_info: reason, can_place = "Symbol not found", False
                elif not symbol_info.visible: reason, can_place = "Symbol not enabled", False
                elif symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL: reason, can_place = f"Trade restricted: {symbol_info.trade_mode}", False
                elif tick is None: reason, can_place = "No tick data", False

                # === Volume auto-fix ===
                volume = raw_volume
                if can_place and symbol_info:
                    volume = max(symbol_info.volume_min,
                                 round(raw_volume / symbol_info.volume_step) * symbol_info.volume_step)
                    volume = min(volume, symbol_info.volume_max)
                    if abs(volume - raw_volume) > 1e-8:
                        reason = f"Volume auto-fixed: {raw_volume} → {volume}"

                # === Price-distance (skip if already past market) ===
                skip_reason = None
                if can_place and tick:
                    bid, ask = tick.bid, tick.ask
                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= ask: skip_reason = f"BUY_LIMIT {price} >= ask {ask}"
                        elif (ask - price) < point * 10: reason, can_place = "BUY_LIMIT too close to ask", False
                    else:
                        if price <= bid: skip_reason = f"SELL_LIMIT {price} <= bid {bid}"
                        elif (price - bid) < point * 10: reason, can_place = "SELL_LIMIT too close to bid", False

                if skip_reason:
                    total_skipped += 1
                    log_and_print(f"{broker_name} | ${risk_usd} | {symbol} {order_type_str} @ {price} → {skip_reason} → SKIPPED", "INFO")
                    report_entry = {
                        "symbol": symbol, "order_type": order_type_str, "price": price,
                        "volume": volume, "sl": sl_price, "tp": tp_price,
                        "risk_usd": entry["riskusd_amount"], "ticket": None,
                        "success": False, "error_msg": f"SKIPPED: {skip_reason}",
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except: pass
                    continue

                # === SL/TP sanity ===
                if can_place:
                    if sl_price <= 0 or tp_price <= 0: reason, can_place = "SL/TP ≤ 0", False
                    elif order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if sl_price >= price or tp_price <= price: reason, can_place = "Invalid SL/TP for BUY_LIMIT", False
                    else:
                        if sl_price <= price or tp_price >= price: reason, can_place = "Invalid SL/TP for SELL_LIMIT", False

                # === Build request & send ===
                request = {
                    "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol, "volume": volume,
                    "type": order_type, "price": price, "sl": sl_price, "tp": tp_price,
                    "deviation": 10, "magic": 123456,
                    "comment": f"AutoSLTP_{entry['riskusd_amount']}USD",
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                result = None
                used_filling = None
                if can_place:
                    result, used_filling = _send_with_fallback(request, symbol_info)
                else:
                    result = type('obj', (), {
                        'retcode': 10000,
                        'comment': reason,
                        'order': None
                    })()
                    used_filling = "N/A"

                # === CRITICAL FIX: Handle None result from order_send ===
                if result is None:
                    status = "FAILED"
                    final_reason = "order_send returned None (possible connection or server issue)"
                    result = type('obj', (), {
                        'retcode': mt5.TRADE_RETCODE_TIMEOUT,
                        'comment': final_reason,
                        'order': None
                    })()
                else:
                    status = "SUCCESS" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED"
                    final_reason = reason or result.comment

                if status == "SUCCESS" and result.order:
                    existing_orders[dup_key] = {'ticket': result.order, 'price': price}

                # === Reporting ===
                report_entry = {
                    "symbol": symbol, "order_type": order_type_str, "price": price,
                    "volume": volume, "sl": sl_price, "tp": tp_price,
                    "risk_usd": entry["riskusd_amount"],
                    "ticket": result.order if status == "SUCCESS" else None,
                    "success": status == "SUCCESS",
                    "error_code": result.retcode if status == "FAILED" else None,
                    "error_msg": result.comment if status == "FAILED" else None,
                    "timestamp": now_str, "filling_used": used_filling
                }
                existing_reports.append(report_entry)
                try:
                    with report_file.open("w", encoding="utf-8") as f:
                        json.dump(existing_reports, f, indent=2)
                except Exception as e:
                    log_and_print(f"{broker_name}: Failed to write report {symbol}: {e}", "ERROR")

                # === LOGGING: SUCCESS → LOGGED, FAILED → SILENT (only JSON) ===
                if status == "SUCCESS":
                    log_msg = f"{broker_name} | ${risk_usd} | {symbol} {order_type_str} @ {price} → SL {sl_price} | TP {tp_price} | SUCCESS"
                    if used_filling and used_filling != "N/A":
                        log_msg += f" (filling: {used_filling})"
                    log_and_print(log_msg, "INFO")
                    total_placed += 1
                else:  # FAILED
                    total_failed += 1
                    issues_list.append({"symbol": symbol, "diagnosed_reason": final_reason})
                    # NO log_and_print() for FAILED orders

            except Exception as e:
                log_and_print(f"{broker_name}: Exception placing {symbol} → {e}", "ERROR")
                total_failed += 1
                issues_list.append({"symbol": symbol, "diagnosed_reason": f"Exception: {e}"})

        # === Save issues (only failed orders) ===
        issues_file = calc_file.parent / ISSUES_FILE
        try:
            existing_issues = json.load(issues_file.open("r", encoding="utf-8")) if issues_file.exists() else []
            with issues_file.open("w", encoding="utf-8") as f:
                json.dump(existing_issues + issues_list, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to write {ISSUES_FILE}: {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Real orders completed – Placed: {total_placed}, Failed: {total_failed}, Skipped: {total_skipped}",
            "SUCCESS"
        )

    log_and_print("All real brokers processed.", "SUCCESS")
    
def deduplicate_pending_orders():
    r"""
    Deduplicate pending BUY_LIMIT / SELL_LIMIT orders.
    Rules:
      1. Only ONE pending BUY_LIMIT per symbol
      2. Only ONE pending SELL_LIMIT per symbol
      3. If a BUY position is open → delete ALL pending BUY_LIMIT on that symbol
      4. If a SELL position is open → delete ALL pending SELL_LIMIT on that symbol
      5. When multiple pendings exist → use STRATEGY (lowtohigh/hightolow) to keep best price
         or keep oldest (lowest ticket) if no strategy.
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    DEDUP_REPORT = "dedup_report.json"
    ISSUES_FILE = "ordersissues.json"

    # ------------------------------------------------------------------ #
    def _order_type_str(mt5_type):
        return "BUY_LIMIT" if mt5_type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL_LIMIT"

    def _decide_winner(existing, candidate, order_type, strategy):
        """Return (keep_existing, reason)"""
        is_buy = order_type == mt5.ORDER_TYPE_BUY_LIMIT

        if strategy == "lowtohigh":
            if is_buy:
                better = candidate["price"] > existing["price"]
                reason = f"lowtohigh → new {candidate['price']} > old {existing['price']}"
            else:
                better = candidate["price"] < existing["price"]
                reason = f"lowtohigh → new {candidate['price']} < old {existing['price']}"
        elif strategy == "hightolow":
            if is_buy:
                better = candidate["price"] < existing["price"]
                reason = f"hightolow → new {candidate['price']} < old {existing['price']}"
            else:
                better = candidate["price"] > existing["price"]
                reason = f"hightolow → new {candidate['price']} > old {existing['price']}"
        else:
            better = candidate["ticket"] < existing["ticket"]
            reason = f"no strategy → keep oldest ticket {candidate['ticket']} < {existing['ticket']}"

        return (not better, reason)  # True → keep existing

    # ------------------------------------------------------------------ #
    for broker_name, broker_cfg in brokersdictionary.items():
        account_type = broker_cfg.get("ACCOUNT", "").lower()
        if account_type not in ("demo", "real"):
            log_and_print(f"Skipping {broker_name} (account type: {account_type})", "INFO")
            continue

        strategy_key = broker_cfg.get("STRATEGY", "").lower()
        if strategy_key and strategy_key not in ("lowtohigh", "hightolow"):
            log_and_print(f"{broker_name}: Unknown STRATEGY '{strategy_key}' – using oldest ticket", "WARNING")
            strategy_key = ""

        TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
        LOGIN_ID      = broker_cfg["LOGIN_ID"]
        PASSWORD      = broker_cfg["PASSWORD"]
        SERVER        = broker_cfg["SERVER"]

        log_and_print(f"Deduplicating pending orders for {broker_name} ({account_type})", "INFO")

        # ------------------- MT5 connection -------------------
        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"{broker_name}: Terminal path missing", "ERROR")
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                              server=SERVER, timeout=30000):
            log_and_print(f"{broker_name}: MT5 init failed: {mt5.last_error()}", "ERROR")
            continue

        if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            log_and_print(f"{broker_name}: MT5 login failed: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            continue

        # ------------------- Get running positions -------------------
        running_positions = {}  # symbol → direction: 1=buy, -1=sell
        positions = mt5.positions_get()
        for pos in (positions or []):
            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            running_positions[pos.symbol] = direction

        # ------------------- Get pending orders -------------------
        pending = mt5.orders_get()
        pending_by_key = {}  # (symbol, type) → list of {'ticket':, 'price':}
        for order in (pending or []):
            if order.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue
            key = (order.symbol, order.type)
            pending_by_key.setdefault(key, []).append({
                "ticket": order.ticket,
                "price":  order.price_open
            })

        # ------------------- Deduplication -------------------
        total_deleted = total_kept = 0
        dedup_report = []
        issues_list   = []
        now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime(
            "%Y-%m-%d %H:%M:%S.%f+01:00")

        for (symbol, otype), orders in pending_by_key.items():
            new_dir = 1 if otype == mt5.ORDER_TYPE_BUY_LIMIT else -1
            type_str = _order_type_str(otype)

            # === RULE: If same-direction position is running → delete ALL pending of this type ===
            if symbol in running_positions and running_positions[symbol] == new_dir:
                for order in orders:
                    del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order["ticket"]}
                    del_res = mt5.order_send(del_req)

                    status = "DELETED"
                    err_msg = None
                    if del_res is None:
                        status = "DELETE FAILED (None)"
                        err_msg = "order_send returned None"
                    elif del_res.retcode != mt5.TRADE_RETCODE_DONE:
                        status = f"DELETE FAILED ({del_res.retcode})"
                        err_msg = del_res.comment

                    log_and_print(
                        f"{broker_name} | {symbol} {type_str} "
                        f"ticket {order['ticket']} @ {order['price']} → {status} "
                        f"(running { 'BUY' if new_dir==1 else 'SELL' } position)",
                        "INFO" if status == "DELETED" else "WARNING"
                    )

                    dedup_report.append({
                        "symbol": symbol,
                        "order_type": type_str,
                        "ticket": order["ticket"],
                        "price": order["price"],
                        "action": status.split()[0],
                        "reason": "Deleted: same-direction position already running",
                        "error_msg": err_msg,
                        "timestamp": now_str
                    })

                    if status == "DELETED":
                        total_deleted += 1
                    else:
                        issues_list.append({"symbol": symbol, "diagnosed_reason": f"Delete failed: {err_msg}"})
                continue  # skip to next symbol

            # === RULE: Only one pending per type → deduplicate if >1 ===
            if len(orders) <= 1:
                total_kept += 1
                continue

            # Sort by ticket (oldest first) for fallback
            orders.sort(key=lambda x: x["ticket"])

            keep = orders[0]
            for cand in orders[1:]:
                keep_it, reason = _decide_winner(keep, cand, otype, strategy_key)
                to_delete = cand if keep_it else keep

                del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": to_delete["ticket"]}
                del_res = mt5.order_send(del_req)

                status = "DELETED"
                err_msg = None
                if del_res is None:
                    status = "DELETE FAILED (None)"
                    err_msg = "order_send returned None"
                elif del_res.retcode != mt5.TRADE_RETCODE_DONE:
                    status = f"DELETE FAILED ({del_res.retcode})"
                    err_msg = del_res.comment

                log_and_print(
                    f"{broker_name} | {symbol} {type_str} "
                    f"ticket {to_delete['ticket']} @ {to_delete['price']} → {status} | {reason}",
                    "INFO" if status == "DELETED" else "WARNING"
                )

                dedup_report.append({
                    "symbol": symbol,
                    "order_type": type_str,
                    "ticket": to_delete["ticket"],
                    "price": to_delete["price"],
                    "action": status.split()[0],
                    "reason": reason,
                    "error_msg": err_msg,
                    "timestamp": now_str
                })

                if status == "DELETED":
                    total_deleted += 1
                    if not keep_it:
                        keep = cand  # promote winner
                else:
                    issues_list.append({"symbol": symbol, "diagnosed_reason": f"Delete failed: {err_msg}"})

            total_kept += 1  # one survivor

        # ------------------- Save reports -------------------
        broker_dir = Path(BASE_INPUT_DIR) / broker_name
        dedup_file = broker_dir / DEDUP_REPORT
        try:
            existing = json.load(dedup_file.open("r", encoding="utf-8")) if dedup_file.exists() else []
        except:
            existing = []
        all_report = existing + dedup_report
        try:
            with dedup_file.open("w", encoding="utf-8") as f:
                json.dump(all_report, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to write {DEDUP_REPORT}: {e}", "ERROR")

        issues_path = broker_dir / ISSUES_FILE
        try:
            existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
            with issues_path.open("w", encoding="utf-8") as f:
                json.dump(existing_issues + issues_list, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to update {ISSUES_FILE}: {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Deduplication complete – Kept: {total_kept}, Deleted: {total_deleted}",
            "SUCCESS"
        )

    log_and_print("All brokers deduplicated successfully.", "SUCCESS")

def filterout_overleveragetrades(demo):
    r"""
    Enforce risk rules:
      - If balance < 3.8 USD → NO trades allowed → close ALL positions + cancel ALL pending orders
      - For each canceled/closed trade → calculate and log risk_usd
      - Else → close any position with risk > allowed_risk_usd

    Parameters
    ----------
    demo : bool
        True  → include demo accounts
        False → skip demo accounts
    """
    if not isinstance(demo, bool):
        log_and_print(f"Invalid 'demo' value: {demo}. Expected bool. Using True.", "WARNING")
        demo = True

    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    ISSUES_FILE = "ordersissues.json"
    OVERLEVERAGE_REPORT = "overleverage_report.json"

    MIN_BALANCE_FOR_TRADING = 3.8

    BALANCE_TO_RISK = [
        (0.0, 3.8, "risk_0_50cent_usd", 0.5),
        (3.8, 8.0, "risk_1_usd", 1.0),
        (8.0, 12.0, "risk_2_usd", 2.0),
        (12.0, 16.0, "risk_3_usd", 3.0),
        (16.0, 96.0, "risk_4_usd", 4.0),
        (96.0, 150.0, "risk_8_usd", 8.0),
        (150.0, float('inf'), "risk_16_usd", 16.0),
    ]

    def _get_allowed_risk(balance):
        for min_bal, max_bal, folder, risk_usd in BALANCE_TO_RISK:
            if min_bal <= balance < max_bal:
                return risk_usd, folder
        return 16.0, "risk_16_usd"

    def _calculate_pending_risk(order, symbol_info):
        """Calculate risk for pending BUY_LIMIT / SELL_LIMIT order."""
        try:
            price = order.price_open
            sl = order.sl
            volume = order.volume
            if not all([price, sl, volume]) or sl == 0:
                return None
            point = symbol_info.point
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            if not all([point, tick_value, tick_size]):
                return None

            price_diff = abs(price - sl)
            points = price_diff / point
            contract_size = tick_value / tick_size * point
            risk = points * volume * contract_size
            return round(risk, 2)
        except:
            return None

    def _calculate_position_risk(position, symbol_info):
        try:
            price_open = position.price_open
            sl = position.sl
            volume = position.volume
            if not all([price_open, sl, volume]) or sl == 0:
                return None
            point = symbol_info.point
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            if not all([point, tick_value, tick_size]):
                return None

            price_diff = abs(price_open - sl)
            points = price_diff / point
            contract_size = tick_value / tick_size * point
            risk = points * volume * contract_size
            return round(risk, 2)
        except:
            return None

    # ------------------------------------------------------------------ #
    for broker_name, broker_cfg in brokersdictionary.items():
        account_type = broker_cfg.get("ACCOUNT", "").lower()

        if account_type == "demo" and not demo:
            log_and_print(f"Skipping demo broker {broker_name} (demo=False)", "INFO")
            continue

        if account_type not in ("demo", "real"):
            log_and_print(f"Skipping {broker_name} (account type: {account_type})", "INFO")
            continue

        TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
        LOGIN_ID      = broker_cfg["LOGIN_ID"]
        PASSWORD      = broker_cfg["PASSWORD"]
        SERVER        = broker_cfg["SERVER"]

        log_and_print(f"Checking over-leverage for {broker_name} ({account_type})", "INFO")

        # ------------------- MT5 connection -------------------
        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"{broker_name}: Terminal path missing", "ERROR")
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                              server=SERVER, timeout=30000):
            log_and_print(f"{broker_name}: MT5 init failed: {mt5.last_error()}", "ERROR")
            continue

        if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            log_and_print(f"{broker_name}: MT5 login failed: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            continue

        account_info = mt5.account_info()
        if not account_info:
            log_and_print(f"{broker_name}: Failed to get account info", "ERROR")
            mt5.shutdown()
            continue

        balance = account_info.balance
        log_and_print(f"{broker_name}: Balance = ${balance:.2f}", "INFO")

        report_entries = []
        issues_list = []
        now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")

        # ======================================================
        # CASE 1: Balance < 3.8 → ZERO TOLERANCE
        # ======================================================
        if balance < MIN_BALANCE_FOR_TRADING:
            log_and_print(f"{broker_name}: Balance ${balance:.2f} < ${MIN_BALANCE_FOR_TRADING} → ZERO TRADES ALLOWED", "WARNING")

            # --- Cancel ALL pending orders (with risk calculation) ---
            pending = mt5.orders_get()
            for order in (pending or []):
                if order.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    continue

                symbol_info = mt5.symbol_info(order.symbol)
                if not symbol_info:
                    log_and_print(f"{broker_name}: No symbol info for pending {order.symbol}", "WARNING")
                    continue

                risk_usd = _calculate_pending_risk(order, symbol_info)
                order_type_str = "BUY_LIMIT" if order.type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL_LIMIT"

                del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
                result = mt5.order_send(del_req)

                status = "CANCELED"
                err_msg = None
                if result is None:
                    status = "CANCEL FAILED (None)"
                    err_msg = "order_send returned None"
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    status = f"CANCEL FAILED ({result.retcode})"
                    err_msg = result.comment

                log_msg = (
                    f"{broker_name} | {order.symbol} PENDING {order_type_str} "
                    f"ticket {order.ticket} @ {order.price_open}"
                )
                if risk_usd is not None:
                    log_msg += f" risk ${risk_usd}"
                log_msg += f" → {status} (low balance)"

                log_and_print(log_msg, "INFO" if status == "CANCELED" else "WARNING")

                report_entries.append({
                    "symbol": order.symbol,
                    "ticket": order.ticket,
                    "type": "PENDING",
                    "order_type": order_type_str,
                    "price": order.price_open,
                    "volume": order.volume,
                    "risk_usd": risk_usd,
                    "allowed_usd": 0.0,
                    "action": status.split()[0],
                    "reason": "Low balance < $3.8",
                    "error_msg": err_msg,
                    "timestamp": now_str
                })

                if status != "CANCELED":
                    issues_list.append({"symbol": order.symbol, "diagnosed_reason": f"Cancel failed: {err_msg}"})

            # --- Close ALL open positions (with risk) ---
            positions = mt5.positions_get()
            for pos in (positions or []):
                symbol_info = mt5.symbol_info(pos.symbol)
                tick = mt5.symbol_info_tick(pos.symbol)
                if not symbol_info or not tick:
                    log_and_print(f"{broker_name}: Missing info for {pos.symbol} ticket {pos.ticket}", "WARNING")
                    continue

                risk_usd = _calculate_position_risk(pos, symbol_info)
                direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

                close_req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": 123457,
                    "comment": "CloseLowBalance",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(close_req)
                status = "CLOSED"
                err_msg = None
                if result is None:
                    status = "CLOSE FAILED (None)"
                    err_msg = "order_send returned None"
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    status = f"CLOSE FAILED ({result.retcode})"
                    err_msg = result.comment

                log_msg = (
                    f"{broker_name} | {pos.symbol} {direction} ticket {pos.ticket} "
                    f"@ {pos.price_open} volume {pos.volume}"
                )
                if risk_usd is not None:
                    log_msg += f" risk ${risk_usd}"
                log_msg += f" → {status} (low balance)"

                log_and_print(log_msg, "INFO" if status == "CLOSED" else "WARNING")

                report_entries.append({
                    "symbol": pos.symbol,
                    "ticket": pos.ticket,
                    "type": "POSITION",
                    "direction": direction,
                    "volume": pos.volume,
                    "risk_usd": risk_usd,
                    "allowed_usd": 0.0,
                    "action": status.split()[0],
                    "reason": "Low balance < $3.8",
                    "error_msg": err_msg,
                    "timestamp": now_str
                })

                if status != "CLOSED":
                    issues_list.append({"symbol": pos.symbol, "diagnosed_reason": f"Close failed: {err_msg}"})

            mt5.shutdown()
            log_and_print(f"{broker_name}: Zero-tolerance cleanup complete.", "SUCCESS")
            continue

        # ======================================================
        # CASE 2: Normal risk enforcement
        # ======================================================
        allowed_risk_usd, _ = _get_allowed_risk(balance)
        log_and_print(f"{broker_name}: Max risk per trade = ${allowed_risk_usd}", "INFO")

        positions = mt5.positions_get()
        total_closed = total_kept = 0

        for pos in (positions or []):
            symbol_info = mt5.symbol_info(pos.symbol)
            if not symbol_info:
                log_and_print(f"{broker_name}: No symbol info for {pos.symbol}", "WARNING")
                total_kept += 1
                continue

            risk_usd = _calculate_position_risk(pos, symbol_info)
            if risk_usd is None:
                log_and_print(f"{broker_name}: Cannot calculate risk for {pos.symbol} ticket {pos.ticket}", "WARNING")
                total_kept += 1
                continue

            if risk_usd > allowed_risk_usd + 1e-6:
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    log_and_print(f"{broker_name}: No tick for {pos.symbol}", "WARNING")
                    total_kept += 1
                    continue

                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

                close_req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": 123457,
                    "comment": "CloseOverLeverage",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(close_req)
                status = "CLOSED"
                err_msg = None
                if result is None:
                    status = "CLOSE FAILED (None)"
                    err_msg = "order_send returned None"
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    status = f"CLOSE FAILED ({result.retcode})"
                    err_msg = result.comment

                direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                log_and_print(
                    f"{broker_name} | {pos.symbol} {direction} ticket {pos.ticket} "
                    f"risk ${risk_usd} > ${allowed_risk_usd} → {status}",
                    "INFO" if status == "CLOSED" else "WARNING"
                )

                report_entries.append({
                    "symbol": pos.symbol,
                    "ticket": pos.ticket,
                    "type": "POSITION",
                    "direction": direction,
                    "volume": pos.volume,
                    "risk_usd": risk_usd,
                    "allowed_usd": allowed_risk_usd,
                    "action": status.split()[0],
                    "error_msg": err_msg,
                    "timestamp": now_str
                })

                if status == "CLOSED":
                    total_closed += 1
                else:
                    issues_list.append({"symbol": pos.symbol, "diagnosed_reason": f"Close failed: {err_msg}"})
            else:
                total_kept += 1

        # ------------------- Save reports -------------------
        broker_dir = Path(BASE_INPUT_DIR) / broker_name
        report_file = broker_dir / OVERLEVERAGE_REPORT
        try:
            existing = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
        except:
            existing = []
        all_report = existing + report_entries
        try:
            with report_file.open("w", encoding="utf-8") as f:
                json.dump(all_report, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to write {OVERLEVERAGE_REPORT}: {e}", "ERROR")

        issues_path = broker_dir / ISSUES_FILE
        try:
            existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
            with issues_path.open("w", encoding="utf-8") as f:
                json.dump(existing_issues + issues_list, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to update {ISSUES_FILE}: {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Risk filter complete – Kept: {total_kept}, Closed: {total_closed}",
            "SUCCESS"
        )

    log_and_print("All brokers filtered for over-leverage and low balance.", "SUCCESS")

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
        existing_report = json.load(report_path.open("r", encoding="utf-8")) if report_path.exists() else []

        issues = []
        now = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
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

        # === SAVE REPORT & ISSUES ===
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(existing_report, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: report write error – {e}", "ERROR")

        issues_path = broker_dir / ISSUES_FILE
        try:
            cur_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
            with issues_path.open("w", encoding="utf-8") as f:
                json.dump(cur_issues + issues, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: issues write error – {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Breakeven done – SL Updated: {updated} | Pending Info: {pending_info}",
            "SUCCESS"
        )

    log_and_print("All brokers breakeven processed.", "SUCCESS")  

def _write_global_error_report(base_dir, risk_folders, error_msg):
    for folder_name in risk_folders.values():
        folder = Path(base_dir) / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        report_file = folder / "forex_order_report.json"
        try:
            with report_file.open("w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "symbol": "GLOBAL",
                            "error_msg": error_msg,
                            "timestamp": datetime.now(pytz.timezone("Africa/Lagos")).strftime(
                                "%Y-%m-%d %H:%M:%S.%f+01:00"
                            ),
                        }
                    ],
                    f,
                    indent=2,
                )
        except Exception as e:
            log_and_print(f"Failed to write global error report: {e}", "ERROR")         

def calc_and_placeorders():
    calculate_symbols_sl_tp_prices()
    log_and_print("Starting order placement …", "INFO")
    place_demo_orders()
    place_real_orders()
    deduplicate_pending_orders()
    filterout_overleveragetrades(demo=False) 
    BreakevenRunningPositions()

def time_orders():
    """Run the updateorders script for M5 timeframe."""
    try:
        timeorders.main()
        print("updated time orders")
    except Exception as e:
        print(f"Error updating time orders {e}")

def current_time():
    """Run the updateorders script for M5 timeframe."""
    try:
        timeorders.current_time()
        print("current time checked")
    except Exception as e:
        print(f"Error checking current time {e}")


def fetch_charts_all_brokers(
    bars,
    neighborcandles_left,
    neighborcandles_right
):
    """
    Main function to fetch OHLCV data, save charts, crop them, apply arrow detection,
    save candle details, and collect ob_none_oi_data for symbols in symbolsmatch.json.

    NEW BEHAVIOR:
    - Infinite loop
    - First execution: Run full fetch + time_orders() unconditionally
    - Every loop: Run current_time()
    - Only run time_orders() + full fetch when current time >= next_schedule
    - BREAKEVEN runs EVERY 10 SECONDS in background thread (independent)
    """
    import time
    import json
    import os
    from datetime import datetime
    import pytz
    import threading
    import traceback

    # File paths
    current_time_path = r"C:\xampp\htdocs\chronedge\current_time.json"
    fullorders_path = r"C:\xampp\htdocs\chronedge\fullordersschedules.json"

    # =============================================
    # BREAKEVEN THREAD: Runs every 10 seconds (NON-STOP)
    # =============================================
    def breakeven_worker():
        while True:
            try:
                log_and_print("Breakeven check (every 10s)...", "INFO")
                BreakevenRunningPositions()
                log_and_print("Breakeven check completed.", "INFO")
            except Exception as e:
                log_and_print(f"BREAKEVEN THREAD ERROR: {e}\n{traceback.format_exc()}", "CRITICAL")
            time.sleep(10)  # Every 10 seconds

    # Start breakeven thread immediately
    breakeven_thread = threading.Thread(target=breakeven_worker, daemon=True)
    breakeven_thread.start()
    log_and_print("Breakeven background thread started (every 10s)", "SUCCESS")

    # =============================================
    # MAIN LOOP (10-minute schedule waits)
    # =============================================
    is_first_execution = True

    while True:
        error_log = []
        log_and_print("Starting new cycle of fetch_charts_all_brokers...", "INFO")

        try:
            # === STEP 1: ALWAYS RUN current_time() ===
            try:
                current_time()
                log_and_print("current time checked", "INFO")
            except Exception as e:
                log_and_print(f"Error in current_time(): {e}", "ERROR")

            # === STEP 2: LOAD CURRENT TIME ===
            current_24h = "00:00"
            if os.path.exists(current_time_path):
                try:
                    with open(current_time_path, 'r') as f:
                        current_data = json.load(f)
                    current_24h = current_data.get("time_24hour", "00:00")
                    log_and_print(f"Current time: {current_24h}", "INFO")
                except Exception as e:
                    log_and_print(f"Failed to read current_time.json: {e}", "WARNING")
                    current_24h = "00:00"
            else:
                log_and_print("current_time.json not found", "WARNING")

            # === STEP 3: LOAD NEXT SCHEDULE ===
            schedule_time = "00:00"
            schedule_date = "N/A"
            if os.path.exists(fullorders_path):
                try:
                    with open(fullorders_path, 'r') as f:
                        schedule_data = json.load(f)
                    next_schedule = schedule_data.get("next_schedule", {})
                    schedule_time = next_schedule.get("time_24hour", "00:00")
                    schedule_date = next_schedule.get("date", "N/A")
                    log_and_print(f"Next order schedule: {schedule_time} on {schedule_date}", "INFO")
                except Exception as e:
                    log_and_print(f"Failed to read fullordersschedules.json: {e}", "WARNING")
            else:
                log_and_print("fullordersschedules.json not found", "WARNING")

            # === HELPER: Convert HH:MM to minutes ===
            def time_to_minutes(t):
                try:
                    h, m = map(int, t.split(':'))
                    return h * 60 + m
                except:
                    return 0

            current_minutes = time_to_minutes(current_24h)
            schedule_minutes = time_to_minutes(schedule_time)

            # === STEP 4: DECIDE WHETHER TO RUN FULL PROCESS ===
            should_run_full = False
            if is_first_execution:
                log_and_print("Executing on no rules on first loop", "INFO")
                should_run_full = True
                try:
                    time_orders()
                    log_and_print("updated time orders", "INFO")
                except Exception as e:
                    log_and_print(f"Error in time_orders(): {e}", "ERROR")
                is_first_execution = False

            elif current_minutes >= schedule_minutes:
                log_and_print(f"Current time {current_24h} >= Schedule {schedule_time}. Triggering full run + time_orders()", "INFO")
                should_run_full = True
                try:
                    time_orders()
                    log_and_print("updated time orders", "INFO")
                except Exception as e:
                    log_and_print(f"Error in time_orders(): {e}", "ERROR")
            else:
                # NOT TIME YET → just wait
                time_diff = schedule_minutes - current_minutes
                hours_left = time_diff // 60
                mins_left = time_diff % 60
                log_and_print(f"check current time: {current_24h}", "INFO")
                log_and_print(f"order time: {schedule_time}", "INFO")
                log_and_print(f"time_left: {hours_left}h {mins_left}m", "INFO")
                log_and_print("will check in the next 10 mins", "INFO")
                time.sleep(600)
                continue  # Skip full fetch

            # === STEP 5: FULL CHART FETCH PROCESS (ONLY IF SHOULD RUN) ===
            if should_run_full:
                log_and_print("Starting chart generation process for all brokers with symbols from symbolsmatch.json, processing one symbol per market category per round until all symbols are exhausted", "INFO")

                # Load symbolsmatch.json
                symbolsmatch_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\symbolsmatch.json"
                try:
                    with open(symbolsmatch_path, 'r') as f:
                        symbolsmatch_data = json.load(f)
                    log_and_print(f"Loaded symbolsmatch.json from {symbolsmatch_path}", "INFO")
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Failed to load symbolsmatch.json: {str(e)}",
                        "broker": "none"
                    })
                    save_errors(error_log)
                    log_and_print(f"Failed to load symbolsmatch.json: {str(e)}", "ERROR")
                    time.sleep(600)
                    continue

                # Load allsymbolsvolumesandrisk.json to map symbols to market categories
                allsymbols_json_path = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\allsymbolsvolumesandrisk.json"
                symbol_to_category = {}
                try:
                    with open(allsymbols_json_path, 'r') as f:
                        allsymbols_data = json.load(f)
                    for risk_key, markets in allsymbols_data.items():
                        for mkt_type in ["stocks", "indices", "commodities", "synthetics", "forex", "crypto", "equities", "energies", "etfs", "basket_indices", "metals"]:
                            for item in markets.get(mkt_type, []):
                                symbol_to_category[item["symbol"]] = mkt_type
                    log_and_print(f"Loaded symbol-to-category mapping from {allsymbols_json_path} with {len(symbol_to_category)} symbols", "INFO")
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Failed to load {allsymbols_json_path}: {str(e)}",
                        "broker": "none"
                    })
                    save_errors(error_log)
                    log_and_print(f"Failed to load {allsymbols_json_path}: {str(e)}", "ERROR")
                    time.sleep(600)
                    continue

                # Create a mapping for broker names (remove digits)
                broker_name_mapping = {
                    "deriv": "deriv",
                    "deriv1": "deriv",
                    "deriv2": "deriv",
                    "bybit1": "bybit",
                    "exness1": "exness"
                }

                # Get symbols for each broker from symbolsmatch.json and group by market category
                broker_category_symbols = {}
                all_categories = ["stocks", "forex", "crypto", "synthetics", "indices", "commodities", "equities", "energies", "etfs", "basket_indices", "metals"]
                for broker_name, config in brokersdictionary.items():
                    mapped_broker = broker_name_mapping.get(broker_name, broker_name)
                    broker_category_symbols[broker_name] = {cat: [] for cat in all_categories}
                    
                    # Initialize MT5 to get available symbols
                    log_and_print(f"Initializing MT5 for broker: {broker_name}", "INFO")
                    success, init_errors = initialize_mt5(
                        config["TERMINAL_PATH"],
                        config["LOGIN_ID"],
                        config["PASSWORD"],
                        config["SERVER"]
                    )
                    error_log.extend(init_errors)
                    if not success:
                        log_and_print(f"MT5 initialization failed for {broker_name}, skipping", "ERROR")
                        mt5.shutdown()
                        continue

                    all_symbols, sym_errors = get_symbols()
                    error_log.extend(sym_errors)
                    mt5.shutdown()

                    # Filter symbols and assign to categories
                    for symbol_entry in symbolsmatch_data.get("main_symbols", []):
                        broker_specific_symbols = symbol_entry.get(mapped_broker, [])
                        for symbol in broker_specific_symbols:
                            if symbol in all_symbols:
                                category = symbol_to_category.get(symbol)
                                if category and category in all_categories:
                                    if symbol not in broker_category_symbols[broker_name][category]:
                                        broker_category_symbols[broker_name][category].append(symbol)
                    
                    # Log the number of symbols per category
                    for category in all_categories:
                        log_and_print(f"Broker {broker_name} has {len(broker_category_symbols[broker_name][category])} symbols in {category}: {broker_category_symbols[broker_name][category]}", "INFO")

                # Check if any symbols were found
                if not any(any(symbols) for broker, categories in broker_category_symbols.items() for symbols in categories.values()):
                    log_and_print("No matched symbols found for any broker in symbolsmatch.json, aborting", "ERROR")
                    save_errors(error_log)
                    time.sleep(600)
                    continue

                # Clear chart folders for all brokers
                for broker_name, config in brokersdictionary.items():
                    log_and_print(f"Clearing chart folder for broker: {broker_name}", "INFO")
                    success_clear, clear_errors = clear_chart_folder(config["BASE_FOLDER"])
                    error_log.extend(clear_errors)
                    if not success_clear:
                        log_and_print(f"Failed to clear chart folder for {broker_name}, continuing", "ERROR")

                # Initialize remaining symbols for each broker and category
                remaining_symbols = {
                    broker: {cat: symbols.copy() for cat, symbols in categories.items()}
                    for broker, categories in broker_category_symbols.items()
                }
                broker_category_indices = {
                    broker: {cat: 0 for cat in all_categories}
                    for broker in broker_category_symbols.keys()
                }

                # Process one symbol from each category across all brokers until all symbols are exhausted
                round_number = 1
                while any(any(remaining_symbols[broker][cat] for cat in all_categories) for broker in broker_category_symbols.keys()):
                    log_and_print(f"Starting round {round_number} of symbol processing", "INFO")
                    for category in all_categories:
                        for broker_name in broker_category_symbols.keys():
                            if not remaining_symbols[broker_name][category]:
                                continue

                            current_index = broker_category_indices[broker_name][category]
                            if current_index >= len(remaining_symbols[broker_name][category]):
                                remaining_symbols[broker_name][category] = []  # Mark category as exhausted
                                continue

                            symbol = remaining_symbols[broker_name][category][current_index]

                            config = brokersdictionary[broker_name]
                            success, init_errors = initialize_mt5(
                                config["TERMINAL_PATH"],
                                config["LOGIN_ID"],
                                config["PASSWORD"],
                                config["SERVER"]
                            )
                            error_log.extend(init_errors)
                            if not success:
                                log_and_print(f"MT5 initialization failed for {broker_name} while processing {symbol} ({category}), skipping", "ERROR")
                                mt5.shutdown()
                                continue

                            log_and_print(f"Processing symbol {symbol} ({category}) for broker: {broker_name} in round {round_number}", "INFO")
                            symbol_folder = os.path.join(config["BASE_FOLDER"], symbol.replace(' ', '_'))
                            os.makedirs(symbol_folder, exist_ok=True)

                            for timeframe_str, mt5_timeframe in TIMEFRAME_MAP.items():
                                log_and_print(f"Processing timeframe: {timeframe_str} for {symbol} ({category})", "INFO")
                                timeframe_folder = os.path.join(symbol_folder, timeframe_str)
                                os.makedirs(timeframe_folder, exist_ok=True)

                                df, data_errors = fetch_ohlcv_data(symbol, mt5_timeframe, bars)
                                error_log.extend(data_errors)
                                if df is None:
                                    continue

                                df['symbol'] = symbol
                                # Generate chart and get PH/PL labels
                                chart_path, chart_errors, ph_labels, pl_labels = generate_and_save_chart(
                                    df, symbol, timeframe_str, timeframe_folder,
                                    neighborcandles_left, neighborcandles_right
                                )
                                error_log.extend(chart_errors)

                                # Save candle data with PH/PL labels
                                candle_errors = save_candle_data(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels)
                                error_log.extend(candle_errors)

                                if chart_path:
                                    crop_errors = crop_chart(chart_path, symbol, timeframe_str, timeframe_folder)
                                    error_log.extend(crop_errors)
                                log_and_print("", "INFO")

                            # Collect ob_none_oi_data for all timeframes of the current symbol
                            collect_errors = collect_ob_none_oi_data(symbol, symbol_folder, broker_name, config["BASE_FOLDER"], broker_category_symbols[broker_name][category])
                            error_log.extend(collect_errors)

                            # === RUN SL/TP CALCULATOR AFTER EVERY SYMBOL ===
                            log_and_print(f"Running calc_and_placeorders() for symbol: {symbol} ({category})", "INFO")
                            try:
                                calc_and_placeorders()
                                log_and_print(f"SL/TP calculator finished for {symbol}", "SUCCESS")
                            except Exception as e:
                                error_log.append({
                                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                                    "error": f"calc_and_placeorders() failed for {symbol}: {str(e)}",
                                    "broker": broker_name
                                })
                                log_and_print(f"SL/TP calculator failed for {symbol}: {str(e)}", "ERROR")

                            mt5.shutdown()
                            log_and_print("", "INFO")
                            broker_category_indices[broker_name][category] += 1

                    round_number += 1

                # === FINAL RUN: CALCULATE SL/TP FOR ALL MARKETS ===
                log_and_print("Running FINAL calc_and_placeorders() for ALL markets...", "INFO")
                try:
                    calc_and_placeorders()
                    log_and_print("FINAL SL/TP calculator completed!", "SUCCESS")
                except Exception as e:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Final calc_and_placeorders() failed: {str(e)}",
                        "broker": "all"
                    })
                    log_and_print(f"Final calc_and_placeorders() failed: {str(e)}", "ERROR")

                save_errors(error_log)
                log_and_print("Chart generation, cropping, arrow detection, PH/PL analysis, candle data saving, and allmarkets_limitorders collection completed for all brokers!", "SUCCESS")

                # === AFTER SUCCESS: REPORT STATUS ===
                log_and_print("after successfully processing all brokers", "SUCCESS")
                log_and_print(f"check current time: {current_24h}", "INFO")
                log_and_print(f"order time: {schedule_time}", "INFO")
                time_diff = schedule_minutes - current_minutes
                if time_diff > 0:
                    hours_left = time_diff // 60
                    mins_left = time_diff % 60
                    log_and_print(f"time_left: {hours_left}h {mins_left}m", "INFO")
                else:
                    log_and_print("time_left: 0m (past schedule)", "INFO")
                log_and_print("will check in the next 10 mins", "INFO")

                # Wait 10 minutes before next full cycle
                time.sleep(600)

        except Exception as e:
            log_and_print(f"CRITICAL ERROR in main loop: {e}", "CRITICAL")
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Main loop crash: {str(e)}",
                "broker": "system"
            })
            save_errors(error_log)
            time.sleep(600)  # Prevent rapid crash   

if __name__ == "__main__":
    success = fetch_charts_all_brokers(
        bars=251,
        neighborcandles_left=10,
        neighborcandles_right=15
    )
    if success:
        log_and_print("Chart generation, cropping, arrow detection, PH/PL analysis, and candle data saving completed successfully for all brokers!", "SUCCESS")
    else:
        log_and_print("Process failed. Check error log for details.", "ERROR")
        
        
        
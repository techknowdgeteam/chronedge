import os
import json
import re
import cv2
import numpy as np
import pytz
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import glob
import os
import json
from datetime import datetime
import pytz
import shutil



def load_developers_dictionary():
    path = r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def get_account_management(broker_name):
    path = os.path.join(r"C:\xampp\htdocs\chronedge\synarex\chart\developers", broker_name, "accountmanagement.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_analysis_paths(
    base_folder,
    broker_name,
    sym,
    tf,
    direction,
    bars,
    output_filename_base,
    receiver_tf=None,
    target=None
):
    """
    Generate all relevant file paths for analysis, including the new standardized
    config.json location in the developers folder.

    Returns a dictionary containing all important paths.
    """
    # Root of developer outputs
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))

    # Source files (where raw/previous data lives)
    source_json = os.path.join(base_folder, sym, tf, "candlesdetails", f"{direction}_{bars}.json")
    source_chart = os.path.join(base_folder, sym, tf, f"chart_{bars}.png")
    
    # Full bars reference (often useful)
    full_bars_source = os.path.join(base_folder, sym, tf, "candlesdetails", "newest_oldest.json")

    # Output directory & files (inside developers/broker/sym/tf/)
    output_dir = os.path.join(dev_output_base, sym, tf)
    
    # Main output files for this particular analysis
    output_json = os.path.join(output_dir, output_filename_base)
    output_chart = os.path.join(output_dir, output_filename_base.replace(".json", ".png"))

    # Standardized config.json — same for all analyses in this symbol/timeframe
    config_json = os.path.join(output_dir, "config.json")

    # Communication paths (when used in receiver/forwarding logic)
    comm_paths = {}
    if receiver_tf and target:
        base_name = output_filename_base.replace(".json", "")
        # Format example: 15m_highers_contourmaker_5m
        comm_filename_base = f"{receiver_tf}_{base_name}_{target}_{tf}"
        comm_paths = {
            "json": os.path.join(output_dir, f"{comm_filename_base}.json"),
            "png": os.path.join(output_dir, f"{comm_filename_base}.png"),
            "base_name": comm_filename_base
        }

    return {
        "dev_output_base": dev_output_base,
        "source_json": source_json,
        "source_chart": source_chart,
        "full_bars_source": full_bars_source,
        "output_dir": output_dir,
        "output_json": output_json,
        "output_chart": output_chart,
        "config_json": config_json,           # ← NEW: standardized config location
        "comm_paths": comm_paths
    }
    
def label_objects_and_text(
    img,
    cx,
    y_rect,
    h_rect,
    c_num=None,                 
    custom_text=None,           
    object_type="arrow",        
    is_bullish_arrow=True,      
    is_marked=False,
    double_arrow=False,
    arrow_color=(0, 255, 0),
    object_color=(0, 255, 0),
    font_scale=0.55,
    text_thickness=2,
    label_position="auto",
    end_x=None  # New parameter for horizontal line stopping point
):
    color = object_color if object_color != (0, 255, 0) else arrow_color

    # Dimensions for markers
    shaft_length = 26
    head_size = 9
    thickness = 2
    wing_size = 7 if double_arrow else 6

    # 1. Determine Vertical Placement (Anchor Point)
    if label_position == "auto":
        place_at_high = not is_bullish_arrow  # HH/LH → top, HL/LL → bottom
    else:
        place_at_high = (label_position.lower() == "high")

    # tip_y is the exact pixel of the wick tip (top or bottom)
    tip_y = y_rect if place_at_high else (y_rect + h_rect)

    # 2. Draw Objects (Arrows/Shapes/Lines) ONLY if is_marked is True
    if is_marked:
        if object_type in ["arrow", "reverse_arrow"]:
            def draw_single_arrow_logic(center_x: int, is_reverse=False):
                if not is_reverse:
                    if place_at_high: # Tip at top, shaft goes UP
                        shaft_start_y = tip_y - head_size
                        cv2.line(img, (center_x, shaft_start_y), (center_x, shaft_start_y - shaft_length), arrow_color, thickness)
                        pts = np.array([[center_x, tip_y - 2], [center_x - wing_size, tip_y - head_size], [center_x + wing_size, tip_y - head_size]], np.int32)
                    else: # Tip at bottom, shaft goes DOWN
                        shaft_start_y = tip_y + head_size
                        cv2.line(img, (center_x, shaft_start_y), (center_x, shaft_start_y + shaft_length), arrow_color, thickness)
                        pts = np.array([[center_x, tip_y + 2], [center_x - wing_size, tip_y + head_size], [center_x + wing_size, tip_y + head_size]], np.int32)
                else:
                    # Pointing AWAY from candle
                    base_y = tip_y - 5 if place_at_high else tip_y + 5
                    end_y = base_y - shaft_length if place_at_high else base_y + shaft_length
                    cv2.line(img, (center_x, base_y), (center_x, end_y), arrow_color, thickness)
                    tip_offset = -head_size if place_at_high else head_size
                    pts = np.array([[center_x, end_y + tip_offset], [center_x - wing_size, end_y], [center_x + wing_size, end_y]], np.int32)
                cv2.fillPoly(img, [pts], arrow_color)

            is_rev = (object_type == "reverse_arrow")
            if double_arrow:
                draw_single_arrow_logic(cx - 5, is_reverse=is_rev)
                draw_single_arrow_logic(cx + 5, is_reverse=is_rev)
            else:
                draw_single_arrow_logic(cx, is_reverse=is_rev)

        elif object_type == "rightarrow":
            base_x, tip_x = cx - 30, cx - 10
            cv2.line(img, (base_x, tip_y), (tip_x, tip_y), color, thickness)
            pts = np.array([[tip_x, tip_y], [tip_x - head_size, tip_y - wing_size], [tip_x - head_size, tip_y + wing_size]], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif object_type == "leftarrow":
            base_x, tip_x = cx + 30, cx + 10
            cv2.line(img, (base_x, tip_y), (tip_x, tip_y), color, thickness)
            pts = np.array([[tip_x, tip_y], [tip_x + head_size, tip_y - wing_size], [tip_x + head_size, tip_y + wing_size]], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif object_type == "hline":
            # Horizontal line starting at candle center
            # If end_x is None, extend to the right edge of the image
            stop_x = end_x if end_x is not None else img.shape[1]
            cv2.line(img, (cx, tip_y), (int(stop_x), tip_y), color, thickness)

        else:
            shape_y = tip_y - 12 if place_at_high else tip_y + 12
            if object_type == "circle":
                cv2.circle(img, (cx, shape_y), 6, color, thickness=thickness)
            elif object_type == "dot":
                cv2.circle(img, (cx, shape_y), 6, color, thickness=-1)
            elif object_type == "pentagon":
                radius = 8
                pts = np.array([[cx, shape_y - radius], [cx + int(radius * 0.95), shape_y - int(radius * 0.31)], [cx + int(radius * 0.58), shape_y + int(radius * 0.81)], [cx - int(radius * 0.58), shape_y + int(radius * 0.81)], [cx - int(radius * 0.95), shape_y - int(radius * 0.31)]], np.int32)
                cv2.fillPoly(img, [pts], color)
            elif object_type == "star":
                outer_rad, inner_rad = 11, 5
                pts = []
                for i in range(10):
                    angle = i * (np.pi / 5) - (np.pi / 2)
                    r = outer_rad if i % 2 == 0 else inner_rad
                    pts.append([cx + int(np.cos(angle) * r), shape_y + int(np.sin(angle) * r)])
                cv2.fillPoly(img, [np.array(pts, np.int32)], color)

    # 3. Text Placement Logic (FIXED DISTANCE)
    if not (custom_text or c_num is not None):
        return

    # Determine vertical reach for text
    if is_marked:
        is_vertical_obj = object_type in ["arrow", "reverse_arrow"]
        # hline doesn't add height, so we treat it like other shapes
        reach = (shaft_length + head_size + 4) if is_vertical_obj else 14
    else:
        reach = 4

    if place_at_high:
        base_text_y = tip_y - reach
    else:
        base_text_y = tip_y + reach + 10

    # Draw the Higher/Lower text (HH, LL, etc.)
    if custom_text:
        (tw, th), _ = cv2.getTextSize(custom_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.putText(img, custom_text, (cx - tw // 2, int(base_text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, arrow_color, text_thickness)
        c_num_y = (base_text_y - 15) if place_at_high else (base_text_y + 15)
    else:
        c_num_y = base_text_y

    # Draw the candle number (c_num)
    if c_num is not None:
        cv2.putText(img, str(c_num), (cx - 8, int(c_num_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2) # Shadow
        cv2.putText(img, str(c_num), (cx - 8, int(c_num_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1) # White

def lower_highs_lower_lows(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data: return f"[{broker_name}] Error: accountmanagement.json missing."
    
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    keyword = "lowerhighsandlowerlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]

    if not matching_configs:
        return f"[{broker_name}] Error: No configuration found for '{keyword}'."

    total_marked_all, processed_charts_all = 0, 0

    def resolve_marker(raw):
        if not raw: return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
        if raw in ["doublearrow", "doublearrows"]: return "arrow", True
        if raw in ["reverse_arrow", "reversearrow"]: return "reverse_arrow", False
        if raw in ["reverse_doublearrow", "reverse_doublearrows"]: return "reverse_arrow", True
        if raw in ["rightarrow", "right_arrow"]: return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]: return "leftarrow", False
        if "dot" in raw: return "dot", False
        return raw, False

    for config_key, lhll_cfg in matching_configs:
        bars = lhll_cfg.get("BARS", 101)
        output_filename_base = lhll_cfg.get("filename", "lowers.json")
        direction = lhll_cfg.get("read_candles_from", "new_old")
        
        neighbor_left = lhll_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = lhll_cfg.get("NEIGHBOR_RIGHT", 5)

        label_cfg = lhll_cfg.get("label", {})
        lh_text = label_cfg.get("lowerhighs_text", "LH")
        ll_text = label_cfg.get("lowerlows_text", "LL")
        cm_text = label_cfg.get("contourmaker_text", "m")

        label_at = label_cfg.get("label_at", {})
        lh_pos = label_at.get("lower_highs", "high").lower()
        ll_pos = label_at.get("lower_lows", "low").lower()

        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        lh_col = color_map.get(label_at.get("lower_highs_color", "red").lower(), (255, 0, 0))
        ll_col = color_map.get(label_at.get("lower_lows_color", "green").lower(), (0, 255, 0))

        lh_obj, lh_dbl = resolve_marker(label_at.get("lower_highs_marker", "arrow"))
        ll_obj, ll_dbl = resolve_marker(label_at.get("lower_lows_marker", "arrow"))
        lh_cm_obj, lh_cm_dbl = resolve_marker(label_at.get("lower_highs_contourmaker_marker", ""))
        ll_cm_obj, ll_cm_dbl = resolve_marker(label_at.get("lower_lows_contourmaker_marker", ""))

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                    continue

                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue

                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 0: continue

                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

                    # Sync data and contours
                    if len(data) != len(contours):
                        min_len = min(len(data), len(contours))
                        data = data[:min_len]
                        contours = contours[:min_len]

                    # Record coordinates for EVERY candle
                    for idx, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        data[idx].update({
                            "candle_x": x + (w // 2),
                            "candle_y": y,
                            "candle_width": w,
                            "candle_height": h,
                            "candle_left": x,
                            "candle_right": x + w,
                            "candle_top": y,
                            "candle_bottom": y + h
                        })

                    n = len(data)
                    swing_count_in_chart = 0
                    for i in range(neighbor_left, n - neighbor_right):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        l_h = [d['high'] for d in data[i-neighbor_left:i]]
                        l_l = [d['low'] for d in data[i-neighbor_left:i]]
                        r_h = [d['high'] for d in data[i+1:i+neighbor_right+1]]
                        r_l = [d['low'] for d in data[i+1:i+neighbor_right+1]]

                        is_peak = curr_h > max(l_h) and curr_h > max(r_h)
                        is_valley = curr_l < min(l_l) and curr_l < min(r_l)

                        if is_peak or is_valley:
                            swing_count_in_chart += 1
                            is_bull = is_valley
                            active_color = ll_col if is_bull else lh_col
                            custom_text = ll_text if is_bull else lh_text
                            obj_type = ll_obj if is_bull else lh_obj
                            dbl_arrow = ll_dbl if is_bull else lh_dbl
                            position = ll_pos if is_bull else lh_pos

                            # Draw LH/LL
                            label_objects_and_text(
                                img, data[i]["candle_x"], data[i]["candle_y"], data[i]["candle_height"], 
                                c_num=data[i]['candle_number'],
                                custom_text=custom_text,
                                object_type=obj_type,
                                is_bullish_arrow=is_bull,
                                is_marked=True,
                                double_arrow=dbl_arrow,
                                arrow_color=active_color,
                                label_position=position
                            )

                            m_idx = i + neighbor_right
                            contour_maker_entry = None
                            if m_idx < n:
                                cm_obj = ll_cm_obj if is_bull else lh_cm_obj
                                cm_dbl = ll_cm_dbl if is_bull else lh_cm_dbl
                                
                                label_objects_and_text(
                                    img, data[m_idx]["candle_x"], data[m_idx]["candle_y"], data[m_idx]["candle_height"], 
                                    custom_text=cm_text,
                                    object_type=cm_obj,
                                    is_bullish_arrow=is_bull,
                                    is_marked=True,
                                    double_arrow=cm_dbl,
                                    arrow_color=active_color,
                                    label_position=position
                                )
                                contour_maker_entry = data[m_idx].copy()
                                contour_maker_entry.update({
                                    "draw_x": data[m_idx]["candle_x"],
                                    "draw_y": data[m_idx]["candle_y"],
                                    "draw_w": data[m_idx]["candle_width"],
                                    "draw_h": data[m_idx]["candle_height"],
                                    "draw_left": data[m_idx]["candle_left"],
                                    "draw_right": data[m_idx]["candle_right"],
                                    "draw_top": data[m_idx]["candle_top"],
                                    "draw_bottom": data[m_idx]["candle_bottom"]
                                })

                            data[i].update({
                                "swing_type": "lower_low" if is_bull else "lower_high",
                                "is_swing": True,
                                "active_color": active_color,
                                "draw_x": data[i]["candle_x"],
                                "draw_y": data[i]["candle_y"],
                                "draw_w": data[i]["candle_width"],
                                "draw_h": data[i]["candle_height"],
                                "draw_left": data[i]["candle_left"],
                                "draw_right": data[i]["candle_right"],
                                "draw_top": data[i]["candle_top"],
                                "draw_bottom": data[i]["candle_bottom"],
                                "contour_maker": contour_maker_entry,
                                "m_idx": m_idx if m_idx < n else None
                            })

                    # Save full chart data (all candles)
                    os.makedirs(paths["output_dir"], exist_ok=True)
                    cv2.imwrite(paths["output_chart"], img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)

                    # --- OVERWRITE SECTION: SYNC TO CONFIG.JSON ---
                    config_path = os.path.join(paths["output_dir"], "config.json")
                    try:
                        config_content = {}
                        if os.path.exists(config_path):
                            with open(config_path, 'r', encoding='utf-8') as f:
                                try:
                                    config_content = json.load(f)
                                    if not isinstance(config_content, dict):
                                        config_content = {}
                                except:
                                    config_content = {}

                        # Exact copy/paste to the specific config key
                        config_content[config_key] = data

                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_content, f, indent=4)
                    except Exception as e:
                        log(f"Config sync failed for {sym}/{tf}: {e}", "WARN")
                    # --- END OF OVERWRITE SECTION ---
                    
                    processed_charts_all += 1
                    total_marked_all += swing_count_in_chart
                    
                    if processed_charts_all % 10 == 0:
                        log(f"Processed {processed_charts_all} charts for {broker_name}...")

                except Exception as e:
                    log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Identify Done. Swings: {total_marked_all} | Charts: {processed_charts_all}"   

def higher_highs_higher_lows(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")
    
    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    keyword = "higherhighsandhigherlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]
    if not matching_configs:
        return f"[{broker_name}] Error: No configuration found for '{keyword}'."
    
    total_marked_all, processed_charts_all = 0, 0

    def resolve_marker(raw):
        if not raw: return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
        if raw in ["doublearrow", "doublearrows"]: return "arrow", True
        if raw in ["reverse_arrow", "reversearrow"]: return "reverse_arrow", False
        if raw in ["reverse_doublearrow", "reverse_doublearrows"]: return "reverse_arrow", True
        if raw in ["rightarrow", "right_arrow"]: return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]: return "leftarrow", False
        if "dot" in raw: return "dot", False
        return raw, False

    for config_key, hhhl_cfg in matching_configs:
        bars = hhhl_cfg.get("BARS", 101)
        output_filename_base = hhhl_cfg.get("filename", "highers.json")
        direction = hhhl_cfg.get("read_candles_from", "new_old")
        
        neighbor_left = hhhl_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = hhhl_cfg.get("NEIGHBOR_RIGHT", 5)
        label_cfg = hhhl_cfg.get("label", {})
        hh_text = label_cfg.get("higherhighs_text", "HH")
        hl_text = label_cfg.get("higherlows_text", "HL")
        cm_text = label_cfg.get("contourmaker_text", "m")
        label_at = label_cfg.get("label_at", {})
        hh_pos = label_at.get("higher_highs", "high").lower()
        hl_pos = label_at.get("higher_lows", "low").lower()
        
        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red").lower(), (255, 0, 0))
        hl_col = color_map.get(label_at.get("higher_lows_color", "green").lower(), (0, 255, 0))
        
        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        hl_obj, hl_dbl = resolve_marker(label_at.get("higher_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        hl_cm_obj, hl_cm_dbl = resolve_marker(label_at.get("higher_lows_contourmaker_marker", ""))

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            
            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                    continue
                
                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) == 0: continue
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                    
                    if len(data) != len(contours):
                        min_len = min(len(data), len(contours))
                        data = data[:min_len]
                        contours = contours[:min_len]

                    # STEP 1: Record coordinates for EVERY candle
                    for idx, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        data[idx].update({
                            "candle_x": x + (w // 2),
                            "candle_y": y,
                            "candle_width": w,
                            "candle_height": h,
                            "candle_left": x,
                            "candle_right": x + w,
                            "candle_top": y,
                            "candle_bottom": y + h
                        })

                    n = len(data)
                    swing_count_in_chart = 0
                    start_idx = neighbor_left
                    end_idx = n - neighbor_right

                    for i in range(start_idx, end_idx):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        
                        l_h = [d['high'] for d in data[i - neighbor_left:i]]
                        l_l = [d['low'] for d in data[i - neighbor_left:i]]
                        r_h = [d['high'] for d in data[i + 1:i + 1 + neighbor_right]]
                        r_l = [d['low'] for d in data[i + 1:i + 1 + neighbor_right]]
                        
                        is_hh = curr_h > max(l_h) and curr_h > max(r_h)
                        is_hl = curr_l < min(l_l) and curr_l < min(r_l)
                        
                        if not (is_hh or is_hl):
                            continue
                        
                        swing_count_in_chart += 1
                        is_bull = is_hl
                        active_color = hl_col if is_bull else hh_col
                        custom_text = hl_text if is_bull else hh_text
                        obj_type = hl_obj if is_bull else hh_obj
                        dbl_arrow = hl_dbl if is_bull else hh_dbl
                        position = hl_pos if is_bull else hh_pos

                        # Draw swing marker
                        label_objects_and_text(
                            img, data[i]["candle_x"], data[i]["candle_y"], data[i]["candle_height"],
                            c_num=data[i]['candle_number'],
                            custom_text=custom_text,
                            object_type=obj_type,
                            is_bullish_arrow=is_bull,
                            is_marked=True,
                            double_arrow=dbl_arrow,
                            arrow_color=active_color,
                            label_position=position
                        )

                        # Contour maker logic
                        m_idx = i + neighbor_right
                        contour_maker_entry = None
                        if m_idx < n:
                            cm_obj = hl_cm_obj if is_bull else hh_cm_obj
                            cm_dbl = hl_cm_dbl if is_bull else hh_cm_dbl
                            
                            label_objects_and_text(
                                img, data[m_idx]["candle_x"], data[m_idx]["candle_y"], data[m_idx]["candle_height"],
                                custom_text=cm_text,
                                object_type=cm_obj,
                                is_bullish_arrow=is_bull,
                                is_marked=True,
                                double_arrow=cm_dbl,
                                arrow_color=active_color,
                                label_position=position
                            )

                            # Mark the candle in the main list
                            data[m_idx]["is_contour_maker"] = True

                            # Prepare nested contour_maker entry (also with the flag)
                            contour_maker_entry = data[m_idx].copy()
                            contour_maker_entry.update({
                                "draw_x": data[m_idx]["candle_x"],
                                "draw_y": data[m_idx]["candle_y"],
                                "draw_w": data[m_idx]["candle_width"],
                                "draw_h": data[m_idx]["candle_height"],
                                "draw_left": data[m_idx]["candle_left"],
                                "draw_right": data[m_idx]["candle_right"],
                                "draw_top": data[m_idx]["candle_top"],
                                "draw_bottom": data[m_idx]["candle_bottom"],
                                "is_contour_maker": True   # redundant but explicit as requested
                            })

                        # Update the swing candle record
                        data[i].update({
                            "swing_type": "higher_low" if is_bull else "higher_high",
                            "is_swing": True,
                            "active_color": active_color,
                            "draw_x": data[i]["candle_x"],
                            "draw_y": data[i]["candle_y"],
                            "draw_w": data[i]["candle_width"],
                            "draw_h": data[i]["candle_height"],
                            "draw_left": data[i]["candle_left"],
                            "draw_right": data[i]["candle_right"],
                            "draw_top": data[i]["candle_top"],
                            "draw_bottom": data[i]["candle_bottom"],
                            "contour_maker": contour_maker_entry,
                            "m_idx": m_idx if m_idx < n else None
                        })

                    # Save full chart data
                    os.makedirs(paths["output_dir"], exist_ok=True)
                    cv2.imwrite(paths["output_chart"], img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)

                    # --- Consolidate to config.json ---
                    config_path = os.path.join(paths["output_dir"], "config.json")
                    try:
                        if os.path.exists(config_path):
                            with open(config_path, 'r', encoding='utf-8') as f:
                                try:
                                    config_json = json.load(f)
                                    if not isinstance(config_json, dict):
                                        config_json = {}
                                except:
                                    config_json = {}
                        else:
                            config_json = {}
                        
                        config_json[config_key] = data
                        
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_json, f, indent=4)
                    except Exception as e:
                        log(f"Config sync failed for {sym}/{tf}: {e}", "WARN")
                    # --- END config sync ---

                    processed_charts_all += 1
                    total_marked_all += swing_count_in_chart
                    
                    if processed_charts_all % 10 == 0:
                        log(f"Processed {processed_charts_all} charts for {broker_name}...")

                except Exception as e:
                    log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Identify Done. Swings: {total_marked_all} | Charts: {processed_charts_all}" 
    
def fair_value_gaps(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")
    
    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    # === DYNAMIC SEARCH BY KEYWORD "fvg" IN SECTION NAME ===
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    if not define_candles:
        return f"[{broker_name}] Error: 'define_candles' section missing in accountmanagement.json."

    keyword = "fvg"
    matching_configs = []

    for key, section in define_candles.items():
        if isinstance(section, dict) and keyword in key.lower():
            matching_configs.append((key, section))

    if not matching_configs:
        return f"[{broker_name}] Error: No section key containing '{keyword}' found in 'define_candles'."

    log(f"Found {len(matching_configs)} FVG configuration(s) matching keyword '{keyword}': {[k for k, _ in matching_configs]}")

    total_marked_all = 0
    processed_charts_all = 0

    for config_key, fvg_cfg in matching_configs:
        log(f"Processing FVG configuration: '{config_key}'")

        # Config extraction
        bars = fvg_cfg.get("BARS", 101)
        output_filename_base = fvg_cfg.get("filename", "fvg.json")
        direction = fvg_cfg.get("read_candles_from", "new_old")
        number_mode = fvg_cfg.get("number_candles", "all").lower()
        validate_filter = fvg_cfg.get("validate_my_condition", False)
        
        # Condition extraction
        conditions = fvg_cfg.get("condition", {})
        do_c1_check = (conditions.get("strong_c1") == "greater")
        do_strength_check = conditions.get("strong_fvg") in ["taller_body", "tallest_body"]
        do_c3_check = (conditions.get("strong_c3") == "greater")
        
        check_c1_type = (conditions.get("c1_candle_type") == "same_with_fvg")
        check_c3_type = (conditions.get("c3_candle_type") == "same_with_fvg")
        c3_closing_cfg = conditions.get("c3_closing")
        fvg_body_size_cfg = str(conditions.get("fvg_body_size", "")).strip().lower()
        
        # Body > Wick requirement toggle
        body_vs_wick_mode = str(conditions.get("c1_c3_higher_body_than_wicks", "")).strip().lower()
        apply_body_vs_wick_rule = (body_vs_wick_mode == "apply")
        
        # Lookback Logic
        raw_lookback = conditions.get("c1_lookback")
        if raw_lookback is None or str(raw_lookback).strip().lower() in ["", "0", "null", "none"]:
            c1_lookback = 0
        else:
            try:
                c1_lookback = min(int(raw_lookback), 5)
            except (ValueError, TypeError):
                c1_lookback = 0
        
        # Label & Color Logic
        label_cfg = fvg_cfg.get("label", {})
        bull_text = label_cfg.get("bullish_text", "+fvg")
        bear_text = label_cfg.get("bearish_text", "-fvg")
        label_at = label_cfg.get("label_at", {})
        
        color_map = {"green": (0, 255, 0), "red": (255, 0, 0)}  # BGR
        bullish_color = color_map.get(label_at.get("bullish_color", "green").lower(), (0, 255, 0))
        bearish_color = color_map.get(label_at.get("bearish_color", "red").lower(), (255, 0, 0))
        
        def resolve_marker(raw):
            raw = str(raw).lower().strip()
            if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
            if raw in ["doublearrow", "doublearrows"]: return "arrow", True
            if raw in ["reverse_arrow", "reversearrow"]: return "reverse_arrow", False
            if raw in ["reverse_doublearrow", "reverse_doublearrows"]: return "reverse_arrow", True
            return raw, False
        
        bull_obj, bull_double = resolve_marker(label_at.get("bullish_marker", "arrow"))
        bear_obj, bear_double = resolve_marker(label_at.get("bearish_marker", "arrow"))
        
        number_all = (number_mode == "all")
        number_only_marked = number_mode in ["define_candles", "define_candle", "definecandle"]
        
        total_marked = 0
        processed_charts = 0

        log(f"Starting FVG Analysis with config '{config_key}' | Mode: Left-to-Right Only")

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            
            for tf in sorted(os.listdir(sym_p)):
                tf_path = os.path.join(sym_p, tf)
                if not os.path.isdir(tf_path): continue

                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)

                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                    continue
                    
                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    min_required = 3 + c1_lookback
                    if len(data) < min_required: continue
                    
                    data = sorted(data, key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=False)
                    
                    marked_count = 0
                    potential_fvgs_map = {} # Maps candle_number to its enriched FVG data
                    
                    # --- Identify Potential FVGs ---
                    for i in range(1 + c1_lookback, len(data) - 1):
                        if i >= len(contours): break
                        
                        c_idx_1 = i - 1 - c1_lookback
                        c1, c2, c3 = data[c_idx_1], data[i], data[i+1]
                        
                        c1_u_wick = round(c1['high'] - max(c1['open'], c1['close']), 5)
                        c1_l_wick = round(min(c1['open'], c1['close']) - c1['low'], 5)
                        c1_total_wick = round(c1_u_wick + c1_l_wick, 5)
                        
                        c3_u_wick = round(c3['high'] - max(c3['open'], c3['close']), 5)
                        c3_l_wick = round(min(c3['open'], c3['close']) - c3['low'], 5)
                        c3_total_wick = round(c3_u_wick + c3_l_wick, 5)
                        
                        body1 = round(abs(c1['close'] - c1['open']), 5)
                        body2 = round(abs(c2['close'] - c2['open']), 5)
                        body3 = round(abs(c3['close'] - c3['open']), 5)
                        
                        height1 = round(c1['high'] - c1['low'], 5)
                        height2 = round(c2['high'] - c2['low'], 5)
                        height3 = round(c3['high'] - c3['low'], 5)
                        
                        fvg_type = None
                        gap_top, gap_bottom = 0, 0
                        
                        if c1['low'] > c3['high']:
                            fvg_type = "bearish"
                            gap_top, gap_bottom = c1['low'], c3['high']
                        elif c1['high'] < c3['low']:
                            fvg_type = "bullish"
                            gap_top, gap_bottom = c3['low'], c1['high']
                        
                        if fvg_type:
                            gap_size = round(abs(gap_top - gap_bottom), 5)
                            is_bullish_fvg_c = c2['close'] > c2['open']
                            
                            c1_type_match = ((fvg_type == "bullish" and c1['close'] > c1['open']) or 
                                            (fvg_type == "bearish" and c1['close'] < c1['open'])) if check_c1_type else True
                            c3_type_match = ((fvg_type == "bullish" and c3['close'] > c3['open']) or 
                                            (fvg_type == "bearish" and c3['close'] < c3['open'])) if check_c3_type else True
                            
                            c3_beyond_wick = (fvg_type == "bearish" and c3['close'] < c2['low']) or \
                                             (fvg_type == "bullish" and c3['close'] > c2['high'])
                            c3_beyond_close = (fvg_type == "bearish" and c3['close'] < c2['close']) or \
                                              (fvg_type == "bullish" and c3['close'] > c2['close'])
                            
                            c3_closing_match = True
                            if c3_closing_cfg == "beyond_wick":    c3_closing_match = c3_beyond_wick
                            elif c3_closing_cfg == "beyond_close": c3_closing_match = c3_beyond_close
                            
                            is_strong_c1 = ((fvg_type == "bearish" and c1['high'] > c2['high']) or 
                                           (fvg_type == "bullish" and c1['low'] < c2['low'])) if do_c1_check else True
                            is_strong_c3 = ((fvg_type == "bearish" and c3['low'] < c2['low']) or 
                                           (fvg_type == "bullish" and c3['high'] > c2['high'])) if do_c3_check else True
                            
                            excess_c1 = round(body2 - body1, 5)
                            excess_c3 = round(body2 - body3, 5)
                            is_strong_fvg = (excess_c1 > 0 and excess_c3 > 0) if do_strength_check else True
                            
                            fvg_durability = True
                            if fvg_body_size_cfg == "sum_c1_c3_body":
                                fvg_durability = (body1 + body3) <= body2
                            elif fvg_body_size_cfg == "sum_c1_c3_height":
                                fvg_durability = (height1 + height3) <= height2
                            elif fvg_body_size_cfg == "multiply_c1_body_by_2":
                                fvg_durability = (body1 * 2) <= body2
                            elif fvg_body_size_cfg == "multiply_c3_body_by_2":
                                fvg_durability = (body3 * 2) <= body2
                            elif fvg_body_size_cfg == "multiply_c1_height_by_2":
                                fvg_durability = (height1 * 2) <= body2
                            elif fvg_body_size_cfg == "multiply_c3_height_by_2":
                                fvg_durability = (height3 * 2) <= body2
                            
                            meets_all = all([
                                c1_type_match, c3_type_match, c3_closing_match,
                                is_strong_c1, is_strong_c3, is_strong_fvg, fvg_durability
                            ])
                            
                            if not validate_filter or (validate_filter and meets_all):
                                enriched = c2.copy()
                                enriched.update({
                                    "fvg_type": fvg_type,
                                    "fvg_gap_size": gap_size,
                                    "fvg_gap_top": round(gap_top, 5),
                                    "fvg_gap_bottom": round(gap_bottom, 5),
                                    "c1_lookback_used": c1_lookback,
                                    "c1_data": c1,
                                    "c3_data": c3,
                                    "c1_body_size": body1,
                                    "c2_body_size": body2,
                                    "c3_body_size": body3,
                                    "c1_height": height1,
                                    "c2_height": height2,
                                    "c3_height": height3,
                                    "c1_upper_and_lower_wick": c1_total_wick,
                                    "c1_upper_and_lower_wick_against_body": round(body1 - c1_total_wick, 5),
                                    "c1_upper_and_lower_wick_higher": c1_total_wick > body1,
                                    "c1_body_higher": body1 >= c1_total_wick,
                                    "c3_upper_and_lower_wick": c3_total_wick,
                                    "c3_upper_and_lower_wick_against_body": round(body3 - c3_total_wick, 5),
                                    "c3_upper_and_lower_wick_higher": c3_total_wick > body3,
                                    "c3_body_higher": body3 >= c3_total_wick,
                                    "fvg_body_size_durability": fvg_durability,
                                    "c1_candletype_withfvg": c1_type_match,
                                    "c3_candletype_withfvg": c3_type_match,
                                    "c3_beyond_wick": c3_beyond_wick,
                                    "is_strong_c1": is_strong_c1,
                                    "is_strong_c3": is_strong_c3,
                                    "is_strong_fvg": is_strong_fvg,
                                    "c2_body_excess_against_c1": excess_c1,
                                    "c2_body_excess_against_c3": excess_c3,
                                    "meets_all_conditions": meets_all,
                                    "_contour_idx": i,
                                    "_is_bull_c2": is_bullish_fvg_c
                                })
                                potential_fvgs_map[c2.get('candle_number')] = enriched

                    # --- Build Final JSON (All Candles) ---
                    fvg_results = []
                    max_gap_found = max([p["fvg_gap_size"] for p in potential_fvgs_map.values()], default=0)
                    
                    for idx, candle in enumerate(data):
                        # Extract coordinates for every candle based on the contours
                        if idx < len(contours):
                            x_c, y_c, w_c, h_c = cv2.boundingRect(contours[idx])
                            candle.update({
                                "candle_x": x_c + w_c // 2,
                                "candle_y": y_c,
                                "candle_width": w_c,
                                "candle_height": h_c,
                                "candle_left": x_c,
                                "candle_right": x_c + w_c,
                                "candle_top": y_c,
                                "candle_bottom": y_c + h_c,
                                "draw_x": x_c + w_c // 2,
                                "draw_y": y_c,
                                "draw_w": w_c,
                                "draw_h": h_c,
                                "draw_left": x_c,
                                "draw_right": x_c + w_c,
                                "draw_top": y_c,
                                "draw_bottom": y_c + h_c
                            })

                        c_num = candle.get('candle_number')
                        if c_num in potential_fvgs_map:
                            entry = potential_fvgs_map[c_num]
                            
                            coord_keys = [
                                "candle_x", "candle_y", "candle_width", "candle_height", 
                                "candle_left", "candle_right", "candle_top", "candle_bottom",
                                "draw_x", "draw_y", "draw_w", "draw_h", 
                                "draw_left", "draw_right", "draw_top", "draw_bottom"
                            ]
                            entry.update({k: candle[k] for k in coord_keys if k in candle})

                            is_tallest = (entry["fvg_gap_size"] == max_gap_found)
                            wick_compromised = (entry["c1_upper_and_lower_wick_higher"] or 
                                              entry["c3_upper_and_lower_wick_higher"])
                            
                            entry["constestant_fvg_chosed"] = is_tallest and (wick_compromised if apply_body_vs_wick_rule else True)
                            
                            c1_good = entry["c1_body_higher"]
                            c3_good = entry["c3_body_higher"]
                            is_contestant = entry["constestant_fvg_chosed"]
                            
                            if apply_body_vs_wick_rule:
                                body_condition_ok = (c1_good and c3_good) or is_contestant
                            else:
                                body_condition_ok = True
                            
                            if body_condition_ok:
                                # Mark on chart
                                c_idx = entry["_contour_idx"]
                                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[c_idx])
                                cx = x_rect + w_rect // 2
                                should_num = number_all or number_only_marked
                                
                                label_objects_and_text(
                                    img=img, cx=cx, y_rect=y_rect, h_rect=h_rect,
                                    c_num=c_num if should_num else None,
                                    custom_text=bull_text if entry["fvg_type"] == "bullish" else bear_text,
                                    object_type=bull_obj if entry["fvg_type"] == "bullish" else bear_obj,
                                    is_bullish_arrow=entry["_is_bull_c2"],
                                    is_marked=True,
                                    double_arrow=bull_double if entry["fvg_type"] == "bullish" else bear_double,
                                    arrow_color=bullish_color if entry["_is_bull_c2"] else bearish_color,
                                    label_position="low" if entry["_is_bull_c2"] else "high"
                                )
                                final_entry = {k: v for k, v in entry.items() 
                                              if not k.startswith("_") and k != "fvg_gap_size"}
                                fvg_results.append(final_entry)
                                marked_count += 1
                                continue 

                        fvg_results.append(candle)
                    
                    if marked_count > 0 or (number_all and len(data) > 0):
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w', encoding='utf-8') as f:
                            json.dump(fvg_results, f, indent=4)

                        # --- OVERWRITE SECTION: SYNC TO CONFIG.JSON ---
                        config_path = os.path.join(paths["output_dir"], "config.json")
                        try:
                            config_content = {}
                            if os.path.exists(config_path):
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    try:
                                        config_content = json.load(f)
                                        if not isinstance(config_content, dict):
                                            config_content = {}
                                    except:
                                        config_content = {}
                            
                            # Update specific key with the full fvg_results
                            config_content[config_key] = fvg_results

                            with open(config_path, 'w', encoding='utf-8') as f:
                                json.dump(config_content, f, indent=4)
                        except Exception as e:
                            log(f"Config sync failed for {sym}/{tf}: {e}", "WARN")
                        # --- END OVERWRITE SECTION ---

                        processed_charts += 1
                        total_marked += marked_count
                        
                except Exception as e:
                    log(f"Error processing {sym}/{tf} with config '{config_key}': {e}", "ERROR")

        log(f"Completed config '{config_key}': FVGs marked: {total_marked} | Charts: {processed_charts}")
        total_marked_all += total_marked
        processed_charts_all += processed_charts

    return f"Done (all FVG configs). Total FVGs: {total_marked_all} | Total Charts: {processed_charts_all}"

def fvg_higherhighsandhigherlows(broker_name):
    """
    Runs after fair_value_gaps().
    Takes the FVG-processed JSON + chart,
    detects HH/HL swings on that same data using parameters from FVG config,
    adds swing information into the JSON and draws swing markers on the existing chart.
    """
    from datetime import datetime
    import pytz
    import os
    import json
    import cv2

    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."

    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."

    # Find FVG configuration section(s)
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    fvg_configs = [(k, v) for k, v in define_candles.items() if "fvg" in k.lower()]
    
    if not fvg_configs:
        return f"[{broker_name}] Warning: No FVG config found → cannot determine filename pattern."

    # For now we process using the **first** matching FVG config
    # (you can later extend to loop through all if needed)
    config_key, fvg_cfg = fvg_configs[0]
    log(f"Using FVG config section: {config_key} for swing detection parameters")

    # Get output filename pattern and reading direction from FVG config
    output_filename_base = fvg_cfg.get("filename", "fvg.json")
    direction = fvg_cfg.get("read_candles_from", "new_old")
    bars = fvg_cfg.get("BARS", 301)

    # ── Get swing detection parameters from the FVG config itself ─────────────
    NEIGHBOR_LEFT = fvg_cfg.get("NEIGHBOR_LEFT", 5)     # fallback 5
    NEIGHBOR_RIGHT = fvg_cfg.get("NEIGHBOR_RIGHT", 5)   # fallback 5

    log(f"Swing detection neighbors → Left: {NEIGHBOR_LEFT} | Right: {NEIGHBOR_RIGHT}")

    # Visual / label settings (can be later moved to config too)
    HH_TEXT = "HH"
    HL_TEXT = "HL"
    color_map = {"green": (0, 255, 0), "red": (255, 0, 0)}
    HH_COLOR = color_map["red"]     # classic HH red
    HL_COLOR = color_map["green"]   # classic HL green

    total_swings_added = 0
    processed_charts = 0

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p):
            continue

        for tf in sorted(os.listdir(sym_p)):
            tf_path = os.path.join(sym_p, tf)
            if not os.path.isdir(tf_path):
                continue

            paths = get_analysis_paths(
                base_folder, broker_name, sym, tf,
                direction, bars, output_filename_base
            )

            # Expect files already created by fair_value_gaps()
            if not os.path.exists(paths["output_json"]) or not os.path.exists(paths["output_chart"]):
                continue

            try:
                # Load FVG-enriched candle data
                with open(paths["output_json"], 'r', encoding='utf-8') as f:
                    data = json.load(f)

                data = sorted(data, key=lambda x: x.get('candle_number', 0))

                # Load existing chart with FVG markings
                img = cv2.imread(paths["output_chart"])
                if img is None:
                    log(f"Cannot read output chart: {paths['output_chart']}", "ERROR")
                    continue

                n = len(data)
                min_required = NEIGHBOR_LEFT + NEIGHBOR_RIGHT + 1
                if n < min_required:
                    log(f"Too few candles ({n} < {min_required}) → {sym}/{tf}", "SKIP")
                    continue

                swing_count = 0
                start_idx = NEIGHBOR_LEFT
                end_idx = n - NEIGHBOR_RIGHT

                for i in range(start_idx, end_idx):
                    curr_h = data[i]['high']
                    curr_l = data[i]['low']

                    left_highs = [d['high'] for d in data[i - NEIGHBOR_LEFT:i]]
                    left_lows  = [d['low']  for d in data[i - NEIGHBOR_LEFT:i]]
                    right_highs = [d['high'] for d in data[i + 1:i + 1 + NEIGHBOR_RIGHT]]
                    right_lows  = [d['low']  for d in data[i + 1:i + 1 + NEIGHBOR_RIGHT]]

                    is_hh = curr_h > max(left_highs) and curr_h > max(right_highs)
                    is_hl = curr_l < min(left_lows)  and curr_l < min(right_lows)

                    if not (is_hh or is_hl):
                        continue

                    swing_count += 1
                    is_bullish_swing = is_hl
                    swing_color = HL_COLOR if is_bullish_swing else HH_COLOR
                    label_text = HL_TEXT if is_bullish_swing else HH_TEXT

                    # Use coordinates already prepared by FVG step
                    x = data[i].get("candle_x")
                    y = data[i].get("candle_y")
                    h = data[i].get("candle_height")

                    if not all(v is not None for v in [x, y, h]):
                        continue

                    # Draw swing marker on top of existing FVG drawing
                    label_objects_and_text(
                        img=img,
                        cx=x,
                        y_rect=y,
                        h_rect=h,
                        c_num=data[i].get('candle_number'),
                        custom_text=label_text,
                        object_type="arrow",
                        is_bullish_arrow=is_bullish_swing,
                        is_marked=True,
                        double_arrow=False,
                        arrow_color=swing_color,
                        label_position="low" if is_bullish_swing else "high"
                    )

                    # Enrich candle with swing information
                    data[i].update({
                        "swing_type": "higher_low" if is_bullish_swing else "higher_high",
                        "is_swing": True,
                        "swing_color_bgr": [int(c) for c in swing_color],
                        "swing_neighbor_left": NEIGHBOR_LEFT,
                        "swing_neighbor_right": NEIGHBOR_RIGHT,
                        "swing_detected_on": "fvg_processed_data",
                        "swing_config_key": config_key  # optional: remember which config was used
                    })

                # If we added any swings → save updated files
                if swing_count > 0:
                    cv2.imwrite(paths["output_chart"], img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)

                    processed_charts += 1
                    total_swings_added += swing_count

                    log(f"Added {swing_count} swings → {sym}/{tf}")

            except Exception as e:
                log(f"Error in fvg+swings processing {sym}/{tf}: {e}", "ERROR")

    msg = f"fvg + HH/HL Done | Swings added: {total_swings_added} | Charts updated: {processed_charts}"
    log(msg)
    return msg
    
def directional_bias(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def get_base_type(bias_direction):
        if not bias_direction: return None
        return "support" if bias_direction == "upward" else "resistance"

    def resolve_marker(raw):
        raw = str(raw or "").lower().strip()
        if not raw: return None, False
        if "double" in raw: return "arrow", True
        if "arrow"  in raw: return "arrow", False
        if "dot" in raw or "circle" in raw: return "dot", False
        if "pentagon" in raw: return "pentagon", False
        return raw, False

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    db_section = define_candles.get("directional_bias_candles", {})
    
    if not db_section:
        return f"[{broker_name}] Error: 'directional_bias_candles' section missing."

    total_db_marked = 0
    total_liq_marked = 0

    self_apprehend_cfg = db_section.get("apprehend_directional_bias_candles", {})
    self_label_cfg = self_apprehend_cfg.get("label", {}) if self_apprehend_cfg else {}
    self_db_text = self_label_cfg.get("directional_bias_candles_text", "DB2")
    self_label_at = self_label_cfg.get("label_at", {})
    
    self_up_obj, self_up_dbl = resolve_marker(self_label_at.get("upward_directional_bias_marker"))
    self_dn_obj, self_dn_dbl = resolve_marker(self_label_at.get("downward_directional_bias_marker"))
    self_up_pos = self_label_at.get("upward_directional_bias", "high").lower()
    self_dn_pos = self_label_at.get("downward_directional_bias", "high").lower()
    has_self_apprehend = bool(self_apprehend_cfg)

    for apprehend_key, apprehend_cfg in db_section.items():
        if not isinstance(apprehend_cfg, dict) or apprehend_key == "apprehend_directional_bias_candles":
            continue 

        log(f"Processing directional bias apprehend: '{apprehend_key}'")

        target_type = apprehend_cfg.get("target", "").lower()
        label_cfg = apprehend_cfg.get("label", {})
        db_text   = label_cfg.get("directional_bias_candles_text", "DB")
        label_at  = label_cfg.get("label_at", {})
        up_obj, up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker"))
        dn_obj, dn_dbl = resolve_marker(label_at.get("downward_directional_bias_marker"))
        up_pos = label_at.get("upward_directional_bias", "high").lower()
        dn_pos = label_at.get("downward_directional_bias", "high").lower()

        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        if not source_config: continue

        bars = source_config.get("BARS", 101)
        filename = source_config.get("filename", "output.json")
        
        is_hhhl = "higherhighsandhigherlows" in source_config_name.lower()
        is_lhll = "lowerhighsandlowerlows"   in source_config_name.lower()

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in sorted(os.listdir(sym_p)):
                dev_output_dir = os.path.join(os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)), sym, tf)
                config_json_path = os.path.join(dev_output_dir, "config.json")
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, "new_old", bars, filename)
                
                if not all(os.path.exists(paths.get(p)) for p in ["source_json", "source_chart", "output_json", "output_chart"]):
                    continue

                try:
                    load_path = paths["output_json"] if os.path.exists(paths["output_json"]) else paths["source_json"]
                    with open(load_path, 'r', encoding='utf-8') as f:
                        full_data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    with open(paths["output_json"], 'r', encoding='utf-8') as f:
                        input_structures = json.load(f)

                    if isinstance(input_structures, dict) and "structures" in input_structures:
                        input_structures = input_structures["structures"]

                    clean_img  = cv2.imread(paths["source_chart"])
                    marked_img = cv2.imread(paths["output_chart"])
                    if clean_img is None or marked_img is None: continue

                    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(raw_contours, key=lambda c: cv2.boundingRect(c)[0])
                    n_candles = len(full_data)

                    final_flat_list = []
                    for structure in input_structures:
                        # Extract the base Marker candle data (HH/HL/LH/LL)
                        # We remove nested keys to keep the list flat
                        marker_candle = {k: v for k, v in structure.items() if k not in ["contour_maker", "directional_bias"]}
                        final_flat_list.append(marker_candle)

                        reference_idx = None
                        reference_high = None
                        reference_low = None
                        active_color = tuple(structure.get("active_color", [0,255,0]))

                        # 1. Process Contour Maker
                        if (is_hhhl or is_lhll) and target_type == "contourmaker":
                            cm_data = structure.get("contour_maker")
                            if cm_data:
                                # Extract CM candle without its nested liq
                                cm_only = {k: v for k, v in cm_data.items() if k != "contour_maker_liquidity_candle"}
                                cm_only["is_contour_maker"] = True
                                final_flat_list.append(cm_only)

                                reference_idx = structure.get("m_idx")
                                reference_high, reference_low = cm_data["high"], cm_data["low"]

                        if reference_idx is None or reference_idx >= n_candles:
                            continue

                        # 2. Process Level 1 Directional Bias
                        first_db_info = None
                        for k in range(reference_idx + 1, n_candles):
                            candle = full_data[k]
                            if candle['high'] < reference_low:
                                first_db_info = {**candle, "idx": k, "type": "downward", "level": 1, "is_directional_bias": True}
                                break
                            if candle['low'] > reference_high:
                                first_db_info = {**candle, "idx": k, "type": "upward", "level": 1, "is_directional_bias": True}
                                break

                        if first_db_info:
                            db_idx = first_db_info["idx"]
                            base_type = get_base_type(first_db_info["type"])
                            first_db_info["base_type"] = base_type
                            
                            # Add Level 1 DB to list
                            final_flat_list.append(first_db_info)

                            # 3. Process Contour Maker Liquidity Sweep
                            for l_idx in range(db_idx + 1, n_candles):
                                l_candle = full_data[l_idx]
                                if (base_type == "support" and l_candle['low'] < reference_low) or \
                                   (base_type == "resistance" and l_candle['high'] > reference_high):
                                    liq_obj = {**l_candle, "idx": l_idx, "is_contour_maker_liquidity": True}
                                    final_flat_list.append(liq_obj)
                                    total_liq_marked += 1
                                    break

                            # Visual Marking for DB1
                            x, y, w, h = cv2.boundingRect(contours[db_idx])
                            is_up = first_db_info["type"] == "upward"
                            label_objects_and_text(
                                img=marked_img, cx=x + w // 2, y_rect=y, h_rect=h,
                                custom_text=db_text, object_type=up_obj if is_up else dn_obj,
                                is_bullish_arrow=is_up, is_marked=True,
                                double_arrow=up_dbl if is_up else dn_dbl,
                                arrow_color=active_color, label_position=up_pos if is_up else dn_pos
                            )
                            total_db_marked += 1

                            # 4. Process Level 2 Bias (Self Apprehend)
                            if has_self_apprehend and db_idx + 1 < n_candles:
                                s_ref_h, s_ref_l = first_db_info["high"], first_db_info["low"]
                                second_db_info = None
                                for m in range(db_idx + 1, n_candles):
                                    c2 = full_data[m]
                                    if c2['high'] < s_ref_l:
                                        second_db_info = {**c2, "idx": m, "type": "downward", "level": 2, "is_next_bias_candle": True}
                                        break
                                    if c2['low'] > s_ref_h:
                                        second_db_info = {**c2, "idx": m, "type": "upward", "level": 2, "is_next_bias_candle": True}
                                        break
                                
                                if second_db_info:
                                    s_idx = second_db_info["idx"]
                                    next_base_type = get_base_type(second_db_info["type"])
                                    final_flat_list.append(second_db_info)

                                    # 5. Process Directional Bias Liquidity Sweep
                                    for dl_idx in range(s_idx + 1, n_candles):
                                        dl_candle = full_data[dl_idx]
                                        if (next_base_type == "support" and dl_candle['low'] < s_ref_l) or \
                                           (next_base_type == "resistance" and dl_candle['high'] > s_ref_h):
                                            final_flat_list.append({**dl_candle, "idx": dl_idx, "directional_bias_liquidity_candle": True})
                                            total_liq_marked += 1
                                            break

                                    # Visual Marking for DB2
                                    sx, sy, sw, sh = cv2.boundingRect(contours[s_idx])
                                    s_is_up = second_db_info["type"] == "upward"
                                    label_objects_and_text(
                                        img=marked_img, cx=sx + sw // 2, y_rect=sy, h_rect=sh,
                                        custom_text=self_db_text, object_type=self_up_obj if s_is_up else self_dn_obj,
                                        is_bullish_arrow=s_is_up, is_marked=True,
                                        double_arrow=self_up_dbl if s_is_up else self_dn_dbl,
                                        arrow_color=active_color, label_position=self_up_pos if s_is_up else self_dn_pos
                                    )
                                    total_db_marked += 1

                    # FINAL SAVE
                    cv2.imwrite(paths["output_chart"], marked_img)
                    
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(final_flat_list, f, indent=4)

                    # Update config.json (preserves candle list and structure)
                    local_config = {}
                    if os.path.exists(config_json_path):
                        try:
                            with open(config_json_path, 'r', encoding='utf-8') as f:
                                local_config = json.load(f)
                        except: local_config = {}
                    
                    local_config[source_config_name] = final_flat_list
                    local_config[f"{source_config_name}_candle_list"] = full_data
                    
                    os.makedirs(dev_output_dir, exist_ok=True)
                    with open(config_json_path, 'w', encoding='utf-8') as f:
                        json.dump(local_config, f, indent=4)
                    
                    log(f"Successfully finalized {sym} {tf}")

                except Exception as e:
                    log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Directional Bias Done. DB Markers: {total_db_marked}, Liq Sweeps: {total_liq_marked}"    

def directional_bias_fvg(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def resolve_marker(raw):
        raw = str(raw or "").lower().strip()
        if not raw: return None, False
        if "double" in raw: return "arrow", True
        if "arrow"  in raw: return "arrow", False
        if "dot" in raw or "circle" in raw: return "dot", False
        return raw, False

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: return f"[{broker_name}] Error: Broker missing."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    db_section = define_candles.get("directional_bias_candles", {})
    
    total_db_marked = 0

    self_apprehend_cfg = db_section.get("apprehend_directional_bias_candles", {})
    self_label_cfg = self_apprehend_cfg.get("label", {}) if self_apprehend_cfg else {}
    self_db_text = self_label_cfg.get("directional_bias_candles_text", "DB2")
    self_label_at = self_label_cfg.get("label_at", {})
    self_up_obj, self_up_dbl = resolve_marker(self_label_at.get("upward_directional_bias_marker"))
    self_dn_obj, self_dn_dbl = resolve_marker(self_label_at.get("downward_directional_bias_marker"))
    has_self_apprehend = bool(self_apprehend_cfg)

    for apprehend_key, apprehend_cfg in db_section.items():
        if not isinstance(apprehend_cfg, dict) or apprehend_key == "apprehend_directional_bias_candles":
            continue 

        source_config_name = apprehend_key.replace("apprehend_", "")
        if "fvg" not in source_config_name.lower():
            continue

        log(f"Processing Fresh FVG Identification: '{apprehend_key}'")

        target_type = apprehend_cfg.get("target", "").lower()
        label_cfg = apprehend_cfg.get("label", {})
        db_text   = label_cfg.get("directional_bias_candles_text", "DB")
        label_at  = label_cfg.get("label_at", {})
        up_obj, up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker"))
        dn_obj, dn_dbl = resolve_marker(label_at.get("downward_directional_bias_marker"))
        
        source_config = define_candles.get(source_config_name)
        if not source_config: continue

        bars = source_config.get("BARS", 101)
        filename = source_config.get("filename", "output.json")

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in sorted(os.listdir(sym_p)):
                dev_output_dir = os.path.join(os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)), sym, tf)
                config_json_path = os.path.join(dev_output_dir, "config.json")
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, "new_old", bars, filename)
                
                if not all(os.path.exists(paths.get(p)) for p in ["source_json", "source_chart", "output_json"]):
                    continue

                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        full_data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    with open(paths["output_json"], 'r', encoding='utf-8') as f:
                        original_json_data = json.load(f)
                    
                    is_nested = isinstance(original_json_data, dict) and "structures" in original_json_data
                    structures = original_json_data["structures"] if is_nested else original_json_data

                    marked_img = cv2.imread(paths["output_chart"])
                    clean_img  = cv2.imread(paths["source_chart"])
                    
                    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(raw_contours, key=lambda c: cv2.boundingRect(c)[0])
                    n_candles = len(full_data)

                    flattened_output = []

                    for structure in structures:
                        c1 = structure.pop("c1_data", None)
                        c2 = structure  # Main FVG Candle
                        c3 = structure.pop("c3_data", None)
                        
                        ref_candle = c1 if target_type == "fvg_c1" else c2 if target_type == "fvg_c2" else c3 if target_type == "fvg_c3" else None
                        
                        if not ref_candle:
                            flattened_output.append(c2); continue
                        
                        ref_num = ref_candle.get("candle_number")
                        reference_idx = next((i for i, d in enumerate(full_data) if d.get("candle_number") == ref_num), None)
                        
                        if reference_idx is None:
                            flattened_output.append(c2); continue

                        ref_high, ref_low = ref_candle["high"], ref_candle["low"]
                        active_color = (0, 255, 0) if "bullish" in structure.get("fvg_type", "").lower() else (255, 0, 0)

                        # --- FIND DIRECTIONAL BIAS 1 ---
                        first_db_info = None
                        for k in range(reference_idx + 1, n_candles):
                            candle = full_data[k]
                            if candle['high'] < ref_low:
                                first_db_info = {**candle, "idx": k, "type": "downward", "level": 1, "fvg_directional_bias": True}
                                break
                            if candle['low'] > ref_high:
                                first_db_info = {**candle, "idx": k, "type": "upward", "level": 1, "fvg_directional_bias": True}
                                break

                        if first_db_info:
                            db_idx = first_db_info["idx"]
                            x, y, w, h = cv2.boundingRect(contours[db_idx])
                            
                            # Inject coordinate details for DB1
                            first_db_info.update({
                                "candle_x": x + w // 2, "candle_y": y, "candle_width": w, "candle_height": h,
                                "candle_left": x, "candle_right": x + w, "candle_top": y, "candle_bottom": y + h,
                                "draw_x": x + w // 2, "draw_y": y, "draw_w": w, "draw_h": h,
                                "draw_left": x, "draw_right": x + w, "draw_top": y, "draw_bottom": y + h
                            })

                            is_up = first_db_info["type"] == "upward"
                            label_objects_and_text(
                                img=marked_img, cx=x + w // 2, y_rect=y, h_rect=h,
                                custom_text=db_text, object_type=up_obj if is_up else dn_obj,
                                is_bullish_arrow=is_up, is_marked=True,
                                double_arrow=up_dbl if is_up else dn_dbl,
                                arrow_color=active_color, label_position="high"
                            )
                            total_db_marked += 1

                            # --- LIQUIDATION ---
                            for t_key, t_data in [("fvg_c1", c1), ("fvg_c3", c3)]:
                                if not t_data: continue
                                liq_candle_num = None
                                for m in range(db_idx + 1, n_candles):
                                    chk = full_data[m]
                                    if first_db_info["type"] == "upward" and chk["low"] < t_data.get("high"):
                                        liq_candle_num = chk["candle_number"]; break
                                    elif first_db_info["type"] == "downward" and chk["high"] > t_data.get("low"):
                                        liq_candle_num = chk["candle_number"]; break
                                if liq_candle_num:
                                    t_data[f"{t_key}_liquidated"] = True
                                    t_data[f"{t_key}_liquidated_by_candle_number"] = liq_candle_num
                                else:
                                    t_data[f"{t_key}_is_not_liquidated"] = True

                            # --- FIND DIRECTIONAL BIAS 2 ---
                            second_db_info = None
                            if has_self_apprehend and db_idx + 1 < n_candles:
                                s_ref_h, s_ref_l = first_db_info["high"], first_db_info["low"]
                                for m in range(db_idx + 1, n_candles):
                                    c_next = full_data[m]
                                    if c_next['high'] < s_ref_l:
                                        second_db_info = {**c_next, "idx": m, "type": "downward", "level": 2, "fvg_next_bias": True}; break
                                    if c_next['low'] > s_ref_h:
                                        second_db_info = {**c_next, "idx": m, "type": "upward", "level": 2, "fvg_next_bias": True}; break
                                
                                if second_db_info:
                                    sx, sy, sw, sh = cv2.boundingRect(contours[second_db_info["idx"]])
                                    
                                    # Inject coordinate details for DB2
                                    second_db_info.update({
                                        "candle_x": sx + sw // 2, "candle_y": sy, "candle_width": sw, "candle_height": sh,
                                        "candle_left": sx, "candle_right": sx + sw, "candle_top": sy, "candle_bottom": sy + sh,
                                        "draw_x": sx + sw // 2, "draw_y": sy, "draw_w": sw, "draw_h": sh,
                                        "draw_left": sx, "draw_right": sx + sw, "draw_top": sy, "draw_bottom": sy + sh
                                    })

                                    s_up = second_db_info["type"] == "upward"
                                    label_objects_and_text(
                                        img=marked_img, cx=sx + sw // 2, y_rect=sy, h_rect=sh,
                                        custom_text=self_db_text, object_type=self_up_obj if s_up else self_dn_obj,
                                        is_bullish_arrow=s_up, is_marked=True,
                                        double_arrow=self_up_dbl if s_up else self_dn_dbl,
                                        arrow_color=active_color, label_position="high"
                                    )
                                    total_db_marked += 1

                            # Sequential Flattening
                            if c1:
                                c1["c1_data"] = True
                                flattened_output.append(c1)
                            
                            flattened_output.append(c2)
                            
                            if c3:
                                c3["c3_data"] = True
                                flattened_output.append(c3)
                                
                            if first_db_info: flattened_output.append(first_db_info)
                            if second_db_info: flattened_output.append(second_db_info)

                    # Update Flags in full_data
                    for item in flattened_output:
                        if item.get("fvg_directional_bias") or item.get("fvg_next_bias"):
                            for candle in full_data:
                                if candle.get("candle_number") == item.get("candle_number"):
                                    if item.get("fvg_directional_bias"): candle["fvg_directional_bias"] = True
                                    if item.get("fvg_next_bias"): candle["fvg_next_bias"] = True

                    cv2.imwrite(paths["output_chart"], marked_img)
                    
                    if is_nested:
                        original_json_data["structures"] = flattened_output
                    else:
                        original_json_data = flattened_output
                    
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(original_json_data, f, indent=4)

                    local_config = {}
                    if os.path.exists(config_json_path):
                        with open(config_json_path, 'r', encoding='utf-8') as f: 
                            local_config = json.load(f)
                    
                    local_config[source_config_name] = flattened_output
                    local_config[f"{source_config_name}_candle_list"] = full_data
                    
                    os.makedirs(dev_output_dir, exist_ok=True)
                    with open(config_json_path, 'w', encoding='utf-8') as f: 
                        json.dump(local_config, f, indent=4)
                    
                    log(f"Finalized Flattened Records for {sym} {tf}")

                except Exception as e:
                    log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Done. Markers: {total_db_marked}"

def timeframes_communication(broker_name):
    """
    Updated version - reads structures from config.json in developers folder
    Saves communication files DIRECTLY in the sender timeframe folder (no 'communications' subfolder)
    """
    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."

    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    tf_comm_section = define_candles.get("timeframes_communication", {})

    if not tf_comm_section:
        log("No 'timeframes_communication' section found.", "WARN")
        return f"[{broker_name}] No timeframes_communication section."

    total_marked_all = 0
    processed_charts_all = 0

    tf_normalize = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h",
        "m1": "1m", "m5": "5m", "m15": "15m", "m30": "30m", "h1": "1h", "h4": "4h"
    }

    custom_style = mpf.make_mpf_style(
        base_mpl_style="default",
        marketcolors=mpf.make_marketcolors(
            up="green", down="red", edge="inherit",
            wick={"up": "green", "down": "red"}
        ),
        gridstyle="",
        gridcolor="none",
        rc={
            'axes.grid': False,
            'axes.labelsize': 12,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        }
    )

    for apprehend_key, comm_cfg in tf_comm_section.items():
        if not isinstance(comm_cfg, dict) or not apprehend_key.startswith("apprehend_"):
            continue

        log(f"Processing timeframes communication: '{apprehend_key}'")

        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        if not source_config:
            log(f"Source config '{source_config_name}' not found.", "ERROR")
            continue

        sender_raw = comm_cfg.get("timeframe_sender", "").strip()
        receiver_raw = comm_cfg.get("timeframe_receiver", "").strip()

        if not sender_raw or not receiver_raw:
            log(f"Missing timeframe_sender or timeframe_receiver in {apprehend_key}", "WARN")
            continue

        sender_list = [s.strip().lower() for s in sender_raw.split(",") if s.strip()]
        receiver_list = [r.strip().lower() for r in receiver_raw.split(",") if r.strip()]

        if len(sender_list) != len(receiver_list):
            log(f"Length mismatch in {apprehend_key}", "ERROR")
            continue

        sender_tfs = [tf_normalize.get(tf, tf) for tf in sender_list]
        receiver_tfs = [tf_normalize.get(tf, tf) for tf in receiver_list]

        log(f"Paired communications: {list(zip(sender_tfs, receiver_tfs))}")

        raw_targets = comm_cfg.get("target", "")
        if not raw_targets:
            log(f"No target specified in {apprehend_key}", "WARN")
            continue

        targets = [t.strip().lower() for t in raw_targets.split(",") if t.strip()]
        if not targets:
            log(f"Invalid target format in {apprehend_key}", "WARN")
            continue

        source_filename = source_config.get("filename", "output.json")
        base_output_name = source_filename.replace(".json", "")
        bars = source_config.get("BARS", 101)
        is_fvg_source = "fvg" in source_config_name.lower()

        for sym in sorted(os.listdir(base_folder)):
            sym_path = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_path):
                continue

            for sender_tf, receiver_tf in zip(sender_tfs, receiver_tfs):
                log(f"{sym}: Processing {sender_tf} → {receiver_tf} (max {bars} forward candles)")

                sender_tf_path = os.path.join(sym_path, sender_tf)
                receiver_tf_path = os.path.join(sym_path, receiver_tf)

                if not os.path.isdir(sender_tf_path) or not os.path.isdir(receiver_tf_path):
                    log(f"{sym}: Missing folder(s) for {sender_tf} or {receiver_tf} → Skipping", "WARN")
                    continue

                # Read structures from config.json in developers folder
                dev_output_dir = os.path.join(
                    os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)),
                    sym, sender_tf
                )
                config_json_path = os.path.join(dev_output_dir, "config.json")

                if not os.path.exists(config_json_path):
                    log(f"{sym}/{sender_tf}: config.json NOT FOUND in developers folder", "ERROR")
                    continue

                try:
                    with open(config_json_path, 'r', encoding='utf-8') as f:
                        local_config = json.load(f)

                    structures = local_config.get(source_config_name, [])
                    
                    if not structures:
                        log(f"{sym}/{sender_tf}: No structures found under key '{source_config_name}' in config.json", "WARN")
                        continue

                    log(f"{sym}/{sender_tf}: Loaded {len(structures)} structures from config.json")

                    # Receiver full candle history
                    receiver_full_json = os.path.join(receiver_tf_path, "candlesdetails", "newest_oldest.json")
                    if not os.path.exists(receiver_full_json):
                        log(f"{sym}/{receiver_tf}: newest_oldest.json NOT FOUND", "ERROR")
                        continue

                    with open(receiver_full_json, 'r', encoding='utf-8') as f:
                        all_receiver_candles = json.load(f)

                    if len(all_receiver_candles) < 10:
                        log(f"{sym}/{receiver_tf}: Not enough candles", "WARN")
                        continue

                    df_full = pd.DataFrame(all_receiver_candles)
                    df_full["time"] = pd.to_datetime(df_full["time"])
                    df_full = df_full.set_index("time")
                    df_full = df_full[["open", "high", "low", "close"]].astype(float)
                    df_full = df_full.sort_index()

                    candle_index_map = {ts: idx for idx, ts in enumerate(df_full.index)}

                    for target in targets:
                        log(f"{sym}: → Processing target '{target}' from {sender_tf} → {receiver_tf}")

                        matched_times = []

                        for structure in structures:
                            ref_time = None

                            if target == "contourmaker":
                                cm = structure.get("contour_maker")
                                if cm and "time" in cm:
                                    ref_time = cm["time"]

                            elif target in ["c1", "c2", "c3"]:
                                c_data = structure.get(f"{target}_data")
                                if c_data and "time" in c_data:
                                    ref_time = c_data["time"]

                            elif is_fvg_source and target in ["fvg_c1", "fvg_c2", "fvg_c3"]:
                                if target == "fvg_c1":
                                    c_data = structure.get("c1_data")
                                elif target == "fvg_c2":
                                    c_data = structure.get("c2_data") or structure
                                elif target == "fvg_c3":
                                    c_data = structure.get("c3_data")
                                if c_data and ("time" in c_data or "candle_number" in c_data):
                                    ref_time = c_data.get("time") or c_data.get("candle_number")

                            elif target in ["directional_bias", "next_bias"]:
                                db_info = structure.get("directional_bias", {})
                                if target == "next_bias":
                                    db_info = db_info.get("next_bias", {}) if db_info else {}
                                if db_info and "time" in db_info:
                                    ref_time = db_info["time"]

                            if not ref_time:
                                continue

                            if isinstance(ref_time, str) and "-" in ref_time:
                                try:
                                    ref_time_dt = pd.to_datetime(ref_time)
                                except:
                                    continue
                            else:
                                try:
                                    ref_time_dt = df_full.index[int(ref_time)]
                                except:
                                    continue

                            if ref_time_dt not in df_full.index:
                                continue

                            matched_times.append(ref_time_dt)

                        if not matched_times:
                            log(f"{sym}: No matches for target '{target}' → skipping")
                            continue

                        matched_times_sorted = sorted(set(matched_times))

                        used_suffixes = {}

                        for signal_time in matched_times_sorted:
                            if signal_time not in df_full.index:
                                continue

                            candle_idx = candle_index_map[signal_time]
                            df_chart = df_full.loc[signal_time:].iloc[:bars]

                            if len(df_chart) < 5:
                                log(f"{sym}: Too few candles (only {len(df_chart)}) from signal #{candle_idx} → skipping", "WARN")
                                continue

                            base_name = f"{receiver_tf}_{base_output_name}_{target}_{sender_tf}_{candle_idx}"

                            suffix = ""
                            if base_name in used_suffixes:
                                suffix_num = used_suffixes[base_name] + 1
                                used_suffixes[base_name] = suffix_num
                                suffix = f"_{chr(96 + suffix_num)}"
                            else:
                                used_suffixes[base_name] = 0

                            final_output_base = base_name + suffix

                            # Yellow marker
                            scatter_data = pd.Series([float('nan')] * len(df_chart), index=df_chart.index)
                            high_val = df_chart.iloc[0]["high"]
                            scatter_data.iloc[0] = high_val * 1.001

                            addplots = [
                                mpf.make_addplot(
                                    scatter_data,
                                    type='scatter',
                                    markersize=300,
                                    marker='o',
                                    color='yellow',
                                    alpha=0.9
                                )
                            ]

                            fig, axlist = mpf.plot(
                                df_chart,
                                type='candle',
                                style=custom_style,
                                addplot=addplots,
                                returnfig=True,
                                figsize=(28, 10),
                                volume=False,
                                show_nontrading=False,
                                warn_too_much_data=5000,
                                tight_layout=False,
                                panel_ratios=(1,)
                            )

                            for ax in axlist:
                                ax.grid(False)
                                ax.set_axisbelow(True)

                            plt.subplots_adjust(left=0.06, right=0.92, top=0.88, bottom=0.12)
                            fig.suptitle(
                                f"{sym} ({receiver_tf}) ← Signal #{candle_idx}{suffix} from {sender_tf} ({target}) | "
                                f"{len(df_chart)}/{bars} Forward Candles | Start: {signal_time.strftime('%Y-%m-%d %H:%M')}",
                                fontsize=16, fontweight='bold', y=0.95
                            )

                            # SAVE DIRECTLY IN dev_output_dir (no subfolder)
                            os.makedirs(dev_output_dir, exist_ok=True)

                            output_json_path = os.path.join(dev_output_dir, f"{final_output_base}.json")
                            output_png_path = os.path.join(dev_output_dir, f"{final_output_base}.png")

                            fig.savefig(output_png_path, bbox_inches="tight", dpi=120, facecolor='white')
                            plt.close(fig)

                            forward_candles = [
                                {
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "time": row.name.strftime('%Y-%m-%d %H:%M:%S'),
                                    "candle_index": candle_index_map[row.name],
                                    "communicated_from": sender_tf,
                                    "communicated_to": receiver_tf,
                                    "is_signal_candle": idx == 0
                                }
                                for idx, (_, row) in enumerate(df_chart.iterrows())
                            ]

                            with open(output_json_path, 'w', encoding='utf-8') as f:
                                json.dump(forward_candles, f, indent=4)

                            processed_charts_all += 1
                            total_marked_all += 1
                            log(f"Generated: {final_output_base}.{{json,png}} → {len(df_chart)} candles")

                except Exception as e:
                    log(f"ERROR {sym} ({sender_tf}→{receiver_tf}): {str(e)}", "ERROR")

    return f"Timeframes Communication Done. Total Signals Charted: {total_marked_all} | Charts Generated: {processed_charts_all}"

def enrich_receiver_comm_paths(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: 
        return "Error: Broker missing."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return "Error: accountmanagement.json missing."
    
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    tf_comm_section = define_candles.get("timeframes_communication", {})

    total_synced_all = 0
    processed_files_all = 0
    errors = []

    # Iterate through communication configurations
    for apprehend_key, comm_cfg in tf_comm_section.items():
        if not apprehend_key.startswith("apprehend_"): 
            continue
        
        source_config_name = apprehend_key.replace("apprehend_", "")
        hhhl_cfg = define_candles.get(source_config_name, {})
        
        direction = hhhl_cfg.get("read_candles_from", "new_old")
        bars = hhhl_cfg.get("BARS", 101)
        source_filename = hhhl_cfg.get("filename", "highers.json")
        
        sender_tfs_raw = comm_cfg.get("timeframe_sender", "")
        receiver_tfs_raw = comm_cfg.get("timeframe_receiver", "")
        targets_raw = comm_cfg.get("target", "")

        sender_tfs = [s.strip().lower() for s in sender_tfs_raw.split(",") if s.strip()]
        receiver_tfs = [r.strip().lower() for r in receiver_tfs_raw.split(",") if r.strip()]
        targets = [t.strip().lower() for t in targets_raw.split(",") if t.strip()]

        if len(sender_tfs) != len(receiver_tfs):
            log(f"Length mismatch between sender and receiver timeframes in {apprehend_key}", "ERROR")
            continue

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): 
                continue

            for s_tf, r_tf in zip(sender_tfs, receiver_tfs):
                # Get base paths using existing helper
                paths = get_analysis_paths(
                    base_folder, broker_name, sym, s_tf, direction, bars, 
                    source_filename, receiver_tf=r_tf, target=None  # target=None to avoid specific comm path
                )
                
                output_dir = paths["output_dir"]
                
                # Pattern to match all files generated by timeframes_communication
                # Example: 15m_higherhighsandhigherlows_contourmaker_4h_*.json
                base_pattern = f"{r_tf}_{source_filename.replace('.json','')}_*_{s_tf}_*"
                pattern = os.path.join(output_dir, base_pattern + ".json")
                
                comm_files = glob.glob(pattern)
                
                if not comm_files:
                    log(f"No communication files found for {sym} | {r_tf} ← {s_tf}", "DEBUG")
                    continue

                log(f"Found {len(comm_files)} communication files for {sym} | {r_tf} ← {s_tf}")

                for comm_path in comm_files:
                    receiver_source_path = os.path.join(base_folder, sym, r_tf, "candlesdetails", "newest_oldest.json")
                    
                    if not os.path.exists(receiver_source_path):
                        log(f"Receiver source missing: {receiver_source_path}", "WARN")
                        continue

                    try:
                        with open(comm_path, 'r', encoding='utf-8') as f: 
                            comm_data = json.load(f)
                            
                        with open(receiver_source_path, 'r', encoding='utf-8') as f: 
                            receiver_truth_data = json.load(f)

                        # Create lookup: time → candle_number
                        truth_map = {}
                        for c in receiver_truth_data:
                            if "time" in c and c["time"]:
                                truth_map[c["time"]] = c.get("candle_number")

                        sync_count = 0
                        for candle in comm_data:
                            c_time = candle.get("time")
                            if c_time in truth_map:
                                candle["candle_number"] = truth_map[c_time]
                                sync_count += 1
                            
                            # Clean up unnecessary fields if they exist
                            candle.pop("type", None)
                            candle.pop("label_text", None)
                            # Add any other cleanup you might need

                        # Save back enriched data
                        with open(comm_path, 'w', encoding='utf-8') as f:
                            json.dump(comm_data, f, indent=4)
                        
                        total_synced_all += sync_count
                        processed_files_all += 1
                        
                        log(f"Enriched: {os.path.basename(comm_path)} | Synced {sync_count} candles", "INFO")

                    except Exception as e:
                        err_msg = f"{sym} ({r_tf}) {os.path.basename(comm_path)}: {str(e)}"
                        errors.append(err_msg)
                        log(err_msg, "ERROR")

    # Final summary
    log("──────────────────────────────────────────", "INFO")
    log(f"ENRICHMENT SUMMARY FOR {broker_name.upper()}", "INFO")
    log(f"Processed files: {processed_files_all}", "INFO")
    log(f"Total candles enriched with candle_number: {total_synced_all}", "INFO")
    if errors:
        log("First few errors:", "WARNING")
        for err in errors[:3]:
            log(f"  • {err}", "WARNING")
        if len(errors) > 3:
            log(f"  ... and {len(errors)-3} more", "WARNING")
    log("──────────────────────────────────────────", "INFO")

    return f"Done: {processed_files_all} files updated | {total_synced_all} candles enriched"

def receiver_comm_higher_highs_higher_lows(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    tf_comm_section = define_candles.get("timeframes_communication", {})
    
    total_marked_all = 0
    processed_charts_all = 0

    def resolve_marker(raw):
        if not raw: return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
        if raw in ["doublearrow", "doublearrows"]: return "arrow", True
        if raw in ["reverse_arrow", "reversearrow"]: return "reverse_arrow", False
        if raw in ["reverse_doublearrow", "reverse_doublearrows"]: return "reverse_arrow", True
        if raw in ["rightarrow", "right_arrow"]: return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]: return "leftarrow", False
        if "dot" in raw: return "dot", False
        return raw, False

    for apprehend_key, comm_cfg in tf_comm_section.items():
        if not apprehend_key.startswith("apprehend_"): 
            continue
        
        source_config_name = apprehend_key.replace("apprehend_", "")
        hhhl_cfg = define_candles.get(source_config_name, {})
        if not hhhl_cfg: 
            continue

        neighbor_left = hhhl_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = hhhl_cfg.get("NEIGHBOR_RIGHT", 5)
        source_filename = hhhl_cfg.get("filename", "highers.json")
        bars = hhhl_cfg.get("BARS", 101)
        direction = hhhl_cfg.get("read_candles_from", "new_old")

        label_cfg = hhhl_cfg.get("label", {})
        hh_text = label_cfg.get("higherhighs_text", "HH")
        hl_text = label_cfg.get("higherlows_text", "HL")
        cm_text = label_cfg.get("contourmaker_text", "m")

        label_at = label_cfg.get("label_at", {})
        hh_pos = label_at.get("higher_highs", "high").lower()
        hl_pos = label_at.get("higher_lows", "low").lower()

        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red").lower(), (255, 0, 0))
        hl_col = color_map.get(label_at.get("higher_lows_color", "green").lower(), (0, 255, 0))

        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        hl_obj, hl_dbl = resolve_marker(label_at.get("higher_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        hl_cm_obj, hl_cm_dbl = resolve_marker(label_at.get("higher_lows_contourmaker_marker", ""))

        sender_tfs_raw = comm_cfg.get("timeframe_sender", "")
        receiver_tfs_raw = comm_cfg.get("timeframe_receiver", "")
        targets_raw = comm_cfg.get("target", "")

        sender_tfs = [s.strip().lower() for s in sender_tfs_raw.split(",") if s.strip()]
        receiver_tfs = [r.strip().lower() for r in receiver_tfs_raw.split(",") if r.strip()]
        targets = [t.strip().lower() for t in targets_raw.split(",") if t.strip()]

        if len(sender_tfs) != len(receiver_tfs):
            log(f"Length mismatch between sender and receiver in {apprehend_key}", "ERROR")
            continue

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): 
                continue

            for s_tf, r_tf in zip(sender_tfs, receiver_tfs):
                paths = get_analysis_paths(
                    base_folder, broker_name, sym, s_tf, direction, bars, 
                    source_filename, receiver_tf=r_tf, target=None
                )
                
                output_dir = paths["output_dir"]
                config_path = os.path.join(output_dir, "config.json")
                
                base_pattern = f"{r_tf}_{source_filename.replace('.json','')}_*_{s_tf}_*"
                json_pattern = os.path.join(output_dir, base_pattern + ".json")
                
                json_files = glob.glob(json_pattern)
                
                if not json_files:
                    log(f"No communication files found for {sym} | {r_tf} ← {s_tf}", "DEBUG")
                    continue

                log(f"Found {len(json_files)} communication files for {sym} | {r_tf} ← {s_tf}")

                for json_path in json_files:
                    png_path = json_path.replace(".json", ".png")
                    
                    if not os.path.exists(png_path):
                        log(f"Missing PNG for {os.path.basename(json_path)}", "WARN")
                        continue

                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            raw_data = json.load(f)
                        
                        # Sort ascending (oldest → newest)
                        data = sorted(raw_data, key=lambda x: x.get('candle_number', 0), reverse=False)
                        
                        img = cv2.imread(png_path)
                        if img is None:
                            log(f"Failed to load image: {os.path.basename(png_path)}", "WARN")
                            continue

                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | \
                               cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours:
                            log(f"No contours found in {os.path.basename(png_path)}", "WARN")
                            continue

                        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

                        json_candles_count = len(data)
                        contour_count = len(contours)

                        min_len = min(json_candles_count, contour_count)
                        data = data[:min_len]
                        contours = contours[:min_len]

                        log(f"Processing {os.path.basename(json_path)}", "DEBUG")

                        # ── 1. Record coordinates for EVERY candle ───────────────────────────────
                        for i in range(min_len):
                            x, y, w, h = cv2.boundingRect(contours[i])
                            
                            candle_info = {
                                "candle_x": int(x + w // 2),
                                "candle_y": int(y),
                                "candle_width": int(w),
                                "candle_height": int(h),
                                "candle_left": int(x),
                                "candle_right": int(x + w),
                                "candle_top": int(y),
                                "candle_bottom": int(y + h)
                            }
                            
                            data[i].update(candle_info)

                        # Summary at index 0
                        highs_summary = {
                            "highs_summary": {
                                "json_candles_count": json_candles_count,
                                "contour_candles_count": contour_count,
                                "processed_candles_with_coords": min_len
                            }
                        }
                        data.insert(0, highs_summary)

                        # ── 2. Swing detection & drawing (original logic unchanged) ──────────────
                        swing_results = []
                        modified = False
                        n = len(data)

                        for i in range(neighbor_left + 1, n - neighbor_right):
                            real_idx = i - 1   # because we inserted summary at 0
                            curr_h, curr_l = data[i]['high'], data[i]['low']
                            
                            l_h = [d['high'] for d in data[i-neighbor_left : i] if 'high' in d]
                            r_h = [d['high'] for d in data[i+1 : i+1+neighbor_right] if 'high' in d]
                            l_l = [d['low'] for d in data[i-neighbor_left : i] if 'low' in d]
                            r_l = [d['low'] for d in data[i+1 : i+1+neighbor_right] if 'low' in d]

                            is_hh = curr_h > max(l_h) and curr_h > max(r_h) if l_h and r_h else False
                            is_hl = curr_l < min(l_l) and curr_l < min(r_l) if l_l and r_l else False

                            if not (is_hh or is_hl):
                                continue

                            is_bull = is_hl
                            active_color = hl_col if is_bull else hh_col
                            label_text = hl_text if is_bull else hh_text
                            obj_type = hl_obj if is_bull else hh_obj
                            dbl_arrow = hl_dbl if is_bull else hh_dbl
                            pos = hl_pos if is_bull else hh_pos

                            x, y, w, h = cv2.boundingRect(contours[real_idx])
                            center_x = x + w // 2
                            center_y = y   # top of candle

                            label_objects_and_text(
                                img, center_x, center_y, h,
                                c_num=data[i]['candle_number'],
                                custom_text=label_text,
                                object_type=obj_type,
                                is_bullish_arrow=is_bull,
                                is_marked=True,
                                double_arrow=dbl_arrow,
                                arrow_color=active_color,
                                label_position=pos
                            )

                            swing_dict = data[i]
                            swing_dict["draw_x"] = center_x
                            swing_dict["draw_y"] = center_y
                            swing_dict["draw_w"] = w
                            swing_dict["draw_h"] = h
                            swing_dict["draw_left"]   = int(x)
                            swing_dict["draw_right"]  = int(x + w)
                            swing_dict["draw_top"]    = int(y)
                            swing_dict["draw_bottom"] = int(y + h)

                            # ── CONTOUR MAKER (original logic kept unchanged) ───────────────────
                            m_idx = i + neighbor_right
                            contour_maker_entry = None
                            if m_idx < n and 'high' in data[m_idx]:
                                real_m_idx = m_idx - 1
                                mx, my, mw, mh = cv2.boundingRect(contours[real_m_idx])
                                cm_center_x = mx + mw // 2
                                cm_center_y = my

                                c_m_obj = hl_cm_obj if is_bull else hh_cm_obj
                                c_m_dbl = hl_cm_dbl if is_bull else hh_cm_dbl

                                label_objects_and_text(
                                    img, cm_center_x, cm_center_y, mh,
                                    custom_text=cm_text,
                                    object_type=c_m_obj,
                                    is_bullish_arrow=is_bull,
                                    is_marked=True,
                                    double_arrow=c_m_dbl,
                                    arrow_color=active_color,
                                    label_position=pos
                                )

                                contour_maker_entry = data[m_idx].copy()
                                contour_maker_entry["draw_x"] = cm_center_x
                                contour_maker_entry["draw_y"] = cm_center_y
                                contour_maker_entry["draw_w"] = mw
                                contour_maker_entry["draw_h"] = mh
                                contour_maker_entry["draw_left"]   = int(mx)
                                contour_maker_entry["draw_right"]  = int(mx + mw)
                                contour_maker_entry["draw_top"]    = int(my)
                                contour_maker_entry["draw_bottom"] = int(my + mh)

                            if "swing_type" not in swing_dict:
                                swing_dict["swing_type"] = "higher_low" if is_bull else "higher_high"
                            if "active_color" not in swing_dict:
                                swing_dict["active_color"] = active_color

                            swing_dict["contour_maker"] = contour_maker_entry

                            modified = True
                            swing_results.append(swing_dict)

                        # ── Save chart + json (original behavior) ───────────────────────────────
                        if modified:
                            cv2.imwrite(png_path, img)
                            
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=4)
                            
                            processed_charts_all += 1
                            total_marked_all += len(swing_results)
                            
                            log(f"Processed {os.path.basename(json_path)}", "INFO")

                        # ────────────────────────────────────────────────────────────────
                        #               Optional: Update config.json
                        #               (new isolated section - does not affect original logic)
                        # ────────────────────────────────────────────────────────────────
                        if modified:  # only if we actually made changes
                            file_key = os.path.basename(json_path).replace('.json', '')
                            
                            config_data = {}
                            try:
                                if os.path.exists(config_path):
                                    with open(config_path, 'r', encoding='utf-8') as f:
                                        config_data = json.load(f)
                            except Exception as e:
                                log(f"Error reading config.json (will overwrite): {e}", "WARN")
                                config_data = {}

                            # Store under its own key (safe, isolated)
                            config_data[file_key] = data  # the full enriched data list

                            try:
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config_data, f, indent=4)
                                log(f"config.json updated → key: {file_key}", "INFO")
                            except Exception as e:
                                log(f"Failed to save config.json: {e}", "ERROR")

                    except Exception as e:
                        log(f"Error processing {sym} | {os.path.basename(json_path)}: {e}", "ERROR")

    return f"Identify Receiver Comm Done. Processed {processed_charts_all} files with total {total_marked_all} swings marked/enriched."

def receiver_directional_bias(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def resolve_marker(raw):
        raw = str(raw or "").lower().strip()
        if not raw: return None, False
        if "double" in raw: return "arrow", True
        if "arrow"  in raw: return "arrow", False
        if "dot" in raw or "circle" in raw: return "dot", False
        if "pentagon" in raw: return "pentagon", False
        return raw, False

    def is_valid_candle(item):
        return isinstance(item, dict) and item.get("candle_number") is not None

    def get_base_type(bias_type):
        if not bias_type: return None
        return "support" if bias_type == "upward" else "resistance"

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    db_section = define_candles.get("directional_bias_candles", {})
    tf_comm_section = define_candles.get("timeframes_communication", {})

    if not db_section:
        return f"[{broker_name}] Error: 'directional_bias_candles' section missing."

    total_db_marked = 0
    total_liq_marked = 0
    total_files_processed = 0

    self_apprehend_cfg = db_section.get("apprehend_directional_bias_candles", {})
    self_label_cfg = self_apprehend_cfg.get("label", {}) if self_apprehend_cfg else {}
    self_db_text = self_label_cfg.get("directional_bias_candles_text", "DB2")
    self_label_at = self_label_cfg.get("label_at", {})
    self_up_pos = self_label_at.get("upward_directional_bias", "high").lower()
    self_dn_pos = self_label_at.get("downward_directional_bias", "high").lower()
    self_up_obj, self_up_dbl = resolve_marker(self_label_at.get("upward_directional_bias_marker"))
    self_dn_obj, self_dn_dbl = resolve_marker(self_label_at.get("downward_directional_bias_marker"))
    has_self_apprehend = bool(self_apprehend_cfg)

    for apprehend_key, apprehend_cfg in db_section.items():
        if not apprehend_key.startswith("apprehend_"): continue
        if apprehend_key == "apprehend_directional_bias_candles": continue

        log(f"Processing Receiver Directional Bias: '{apprehend_key}'")
        target_type = apprehend_cfg.get("target", "").lower()
        if not target_type: continue

        label_cfg = apprehend_cfg.get("label", {})
        db_text = label_cfg.get("directional_bias_candles_text", "DB")
        label_at = label_cfg.get("label_at", {})
        up_pos, dn_pos = label_at.get("upward_directional_bias", "high").lower(), label_at.get("downward_directional_bias", "high").lower()
        up_obj, up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker"))
        dn_obj, dn_dbl = resolve_marker(label_at.get("downward_directional_bias_marker"))

        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        comm_cfg = tf_comm_section.get(apprehend_key)
        if not source_config or not comm_cfg: continue

        source_filename = source_config.get("filename", "output.json")
        sender_tfs = [s.strip().lower() for s in comm_cfg.get("timeframe_sender", "").split(",") if s.strip()]
        receiver_tfs = [r.strip().lower() for r in comm_cfg.get("timeframe_receiver", "").split(",") if r.strip()]

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for s_tf, r_tf in zip(sender_tfs, receiver_tfs):
                paths = get_analysis_paths(base_folder, broker_name, sym, s_tf, "new_old", 101, source_filename, receiver_tf=r_tf)
                output_dir = paths.get("output_dir")
                config_path = os.path.join(output_dir, "config.json")
                pattern = os.path.join(output_dir, f"{r_tf}_{source_filename.replace('.json', '')}_*_{s_tf}_*.json")
                
                for json_path in glob.glob(pattern):
                    img_path = json_path.replace(".json", ".png")
                    if not os.path.exists(img_path): continue

                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            raw_data = json.load(f)

                        existing_summaries = [item for item in raw_data if not is_valid_candle(item)]
                        actual_candles = sorted([item for item in raw_data if is_valid_candle(item)], key=lambda x: x.get('candle_number', 0))

                        img = cv2.imread(img_path)
                        if img is None: continue
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        contour_list = []
                        for cnt in cnts:
                            x, y, w, h = cv2.boundingRect(cnt)
                            contour_list.append({"cnt": cnt, "cx": x + w/2, "cy": y + h/2})

                        candle_to_contour_map = {}
                        used_contour_indices = set()
                        for i, candle in enumerate(actual_candles):
                            t_cx = candle.get("candle_left", 0) + (candle.get("candle_width", 0) / 2)
                            t_cy = candle.get("candle_top", 0) + (candle.get("candle_height", 0) / 2)
                            best_dist, best_idx = 15, None
                            for idx, c_info in enumerate(contour_list):
                                if idx in used_contour_indices: continue
                                dist = ((t_cx - c_info["cx"])**2 + (t_cy - c_info["cy"])**2)**0.5
                                if dist < best_dist: best_dist, best_idx = dist, idx
                            if best_idx is not None:
                                candle_to_contour_map[i] = contour_list[best_idx]["cnt"]
                                used_contour_indices.add(best_idx)

                        matched_indices = sorted(candle_to_contour_map.keys())
                        working_data = [actual_candles[i] for i in matched_indices]
                        ordered_contours = [candle_to_contour_map[i] for i in matched_indices]
                        num_working = len(working_data)

                        modified = False
                        marked_indices = set()

                        # --- Processing Loop (Logic Logic) ---
                        for i, candle in enumerate(working_data):
                            # Temporary extraction of Nested Data for logic processing
                            cm_data = candle.get("contour_maker")
                            if not isinstance(cm_data, dict): continue
                            
                            ref_num = cm_data.get("candle_number")
                            ref_idx = next((j for j, c in enumerate(working_data) if c.get("candle_number") == ref_num), None)
                            if ref_idx is None: continue

                            ref_h, ref_l = cm_data.get("high"), cm_data.get("low")
                            active_color = tuple(candle.get("active_color", [0, 255, 0]))
                            
                            first_db_candle_idx = None
                            db_type = None

                            for k in range(ref_idx + 1, num_working):
                                check_c = working_data[k]
                                if check_c['low'] > ref_h:
                                    first_db_candle_idx, db_type = k, "upward"
                                    break
                                if check_c['high'] < ref_l:
                                    first_db_candle_idx, db_type = k, "downward"
                                    break

                            if first_db_candle_idx is not None and first_db_candle_idx not in marked_indices:
                                db_candle = working_data[first_db_candle_idx]
                                base_type = get_base_type(db_type)
                                
                                # 1. Mark the primary Directional Bias Candle
                                db_candle["is_directional_bias"] = True
                                db_candle["bias_type"] = db_type
                                db_candle["bias_level"] = 1
                                
                                # 2. Mark if Contour Maker was liquidated
                                found_cm_liq = False
                                for l_idx in range(first_db_candle_idx + 1, num_working):
                                    l_candle = working_data[l_idx]
                                    if (base_type == "support" and l_candle['low'] < ref_l) or \
                                       (base_type == "resistance" and l_candle['high'] > ref_h):
                                        
                                        # Update the CM candle (working_data[ref_idx])
                                        working_data[ref_idx]["is_liquidated"] = True
                                        working_data[ref_idx]["liquidated_by_candle_number"] = l_candle["candle_number"]
                                        
                                        # Update the Liquidator candle
                                        l_candle["liquidated_contour_maker"] = True
                                        l_candle["liquidated_contour_maker_number"] = ref_num
                                        
                                        total_liq_marked += 1
                                        found_cm_liq = True
                                        break
                                if not found_cm_liq: 
                                    working_data[ref_idx]["is_not_liquidated"] = True

                                # Image Drawing
                                x, y, w, h = cv2.boundingRect(ordered_contours[first_db_candle_idx])
                                is_up = db_type == "upward"
                                label_objects_and_text(
                                    img=img, cx=x+w//2, y_rect=y, h_rect=h, custom_text=db_text,
                                    object_type=up_obj if is_up else dn_obj, is_bullish_arrow=is_up,
                                    is_marked=True, double_arrow=up_dbl if is_up else dn_dbl,
                                    arrow_color=active_color, label_position=up_pos if is_up else dn_pos
                                )

                                marked_indices.add(first_db_candle_idx)
                                modified = True
                                total_db_marked += 1

                                # Self Apprehend Logic (Level 2)
                                if has_self_apprehend:
                                    s_ref_h, s_ref_l = db_candle["high"], db_candle["low"]
                                    for m in range(first_db_candle_idx + 1, num_working):
                                        c2 = working_data[m]
                                        if (c2['high'] < s_ref_l or c2['low'] > s_ref_h) and m not in marked_indices:
                                            s_up = c2['low'] > s_ref_h
                                            
                                            # Mark Level 2 Bias
                                            c2["is_directional_bias"] = True
                                            c2["is_next_bias_candle"] = True
                                            c2["bias_level"] = 2
                                            
                                            next_base_type = get_base_type("upward" if s_up else "downward")
                                            
                                            # Check if Level 1 Bias was liquidated
                                            found_db_liq = False
                                            for dl_idx in range(m + 1, num_working):
                                                dl_c = working_data[dl_idx]
                                                if (next_base_type == "support" and dl_c['low'] < s_ref_l) or \
                                                   (next_base_type == "resistance" and dl_c['high'] > s_ref_h):
                                                    
                                                    db_candle["is_liquidated"] = True
                                                    db_candle["liquidated_by_candle_number"] = dl_c["candle_number"]
                                                    
                                                    dl_c["liquidated_directional_bias"] = True
                                                    dl_c["liquidated_directional_bias_number"] = db_candle["candle_number"]
                                                    
                                                    total_liq_marked += 1
                                                    found_db_liq = True
                                                    break
                                            if not found_db_liq: db_candle["is_not_liquidated"] = True

                                            # Drawing Level 2
                                            sx, sy, sw, sh = cv2.boundingRect(ordered_contours[m])
                                            label_objects_and_text(
                                                img=img, cx=sx+sw//2, y_rect=sy, h_rect=sh, custom_text=self_db_text,
                                                object_type=self_up_obj if s_up else self_dn_obj, is_bullish_arrow=s_up,
                                                is_marked=True, double_arrow=self_up_dbl if s_up else self_dn_dbl,
                                                arrow_color=active_color, label_position=self_up_pos if s_up else self_dn_pos
                                            )
                                            
                                            marked_indices.add(m)
                                            total_db_marked += 1
                                            break

                        # --- Cleanup Pass: Remove internal dictionaries ---
                        if modified:
                            for c in working_data:
                                c.pop("contour_maker", None)
                                c.pop("directional_bias", None)
                                # Flag the original contour makers found in logic
                                if c.get("candle_number") == ref_num:
                                    c["is_contour_maker"] = True

                            final_output = existing_summaries + working_data
                            cv2.imwrite(img_path, img)
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(final_output, f, indent=4)
                            
                            file_key = os.path.basename(json_path).replace('.json', '')
                            config_data = {}
                            if os.path.exists(config_path):
                                try:
                                    with open(config_path, 'r', encoding='utf-8') as f:
                                        config_data = json.load(f)
                                except: config_data = {}
                            
                            config_data[file_key] = final_output
                            temp_config_path = config_path + ".tmp"
                            with open(temp_config_path, 'w', encoding='utf-8') as f:
                                json.dump(config_data, f, indent=4)
                            shutil.move(temp_config_path, config_path)
                            
                            total_files_processed += 1
                            log(f"✓ Processed {os.path.basename(json_path)}")

                    except Exception as e:
                        log(f"Error processing {json_path}: {str(e)}", "ERROR")

    return f"Done. Processed {total_files_processed} files. DB: {total_db_marked}, Liq: {total_liq_marked}"   

def enrich_base_types_via_config(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def get_base_type(bias_type):
        if not bias_type:
            return None
        bias_type = str(bias_type).lower().strip()
        if bias_type == "upward":
            return "support"
        if bias_type == "downward":
            return "resistance"
        return None

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"Error: Broker '{broker_name}' not found."

    base_folder = cfg.get("BASE_FOLDER")
    if not base_folder:
        return "Error: BASE_FOLDER not defined."

    am_data = get_account_management(broker_name)
    if not am_data:
        return "Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    tf_comm_section = define_candles.get("timeframes_communication", {})

    total_configs_processed = 0
    total_files_updated = 0
    total_base_types_fixed = 0

    # We only need to know which symbols and sender tfs exist
    for sym in sorted(os.listdir(base_folder)):
        sym_path = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_path):
            continue

        for apprehend_key, comm_cfg in tf_comm_section.items():
            if not apprehend_key.startswith("apprehend_"):
                continue

            source_config_name = apprehend_key.replace("apprehend_", "")
            source_cfg = define_candles.get(source_config_name)
            if not source_cfg:
                continue

            source_filename = source_cfg.get("filename", "output.json")

            sender_tfs = [s.strip().lower() for s in comm_cfg.get("timeframe_sender", "").split(",") if s.strip()]
            receiver_tfs = [r.strip().lower() for r in comm_cfg.get("timeframe_receiver", "").split(",") if r.strip()]

            for s_tf in sender_tfs:  # we try each possible sender tf
                for r_tf in receiver_tfs:
                    paths = get_analysis_paths(
                        base_folder=base_folder,
                        broker_name=broker_name,
                        sym=sym,
                        tf=s_tf,                     # sender tf
                        direction="new_old",
                        bars=101,
                        output_filename_base=source_filename,
                        receiver_tf=r_tf
                    )

                    config_path = paths["config_json"]
                    if not os.path.isfile(config_path):
                        continue

                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)

                        if not isinstance(config_data, dict):
                            continue

                        modified_config = False

                        # Every key in config.json is a receiver file
                        for file_key, data in config_data.items():
                            if not isinstance(data, list):
                                continue

                            file_modified = False

                            for item in data:
                                if not isinstance(item, dict):
                                    continue

                                # 1. directional_bias → base_type from next_bias (priority) or own type
                                db = item.get("directional_bias")
                                if isinstance(db, dict):
                                    expected = None

                                    # Priority: next_bias
                                    next_b = db.get("next_bias")
                                    if isinstance(next_b, dict):
                                        expected = get_base_type(next_b.get("type"))

                                    # Fallback: own type
                                    if expected is None:
                                        expected = get_base_type(db.get("type"))

                                    current = db.get("base_type")
                                    if expected and (current != expected or "base_type" not in db):
                                        db["base_type"] = expected
                                        total_base_types_fixed += 1
                                        file_modified = True

                                # 2. contour_maker → base_type from directional_bias.type
                                cm = item.get("contour_maker")
                                if isinstance(cm, dict) and isinstance(db, dict):
                                    expected_cm = get_base_type(db.get("type"))
                                    current_cm = cm.get("base_type")

                                    if expected_cm and (current_cm != expected_cm or "base_type" not in cm):
                                        cm["base_type"] = expected_cm
                                        total_base_types_fixed += 1
                                        file_modified = True

                            if file_modified:
                                total_files_updated += 1
                                modified_config = True

                        if modified_config:
                            try:
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config_data, f, indent=4)
                                total_configs_processed += 1
                            except Exception as e:
                                log(f"Failed to save {config_path}: {e}", "ERROR")

                    except Exception as e:
                        log(f"Error reading config {config_path}: {e}", "ERROR")

    return (
        f"Base type enrichment via config.json finished:\n"
        f"• config.json files processed: {total_configs_processed}\n"
        f"• Individual receiver files updated inside configs: {total_files_updated}\n"
        f"• base_type fields corrected/added: {total_base_types_fixed}"
    )

def liquidity_candles(broker_name):
    from datetime import datetime
    import pytz
    import os
    import json
    import cv2

    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def resolve_marker(raw):
        if not raw:
            return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]:
            return "arrow", False
        if raw in ["doublearrow", "doublearrows"]:
            return "arrow", True
        if raw in ["rightarrow", "right_arrow"]:
            return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]:
            return "leftarrow", False
        if "dot" in raw:
            return "dot", False
        return raw, False

    log(f"--- STARTING DUAL-TARGET LIQUIDITY ANALYSIS (ALL TFs): {broker_name} ---")

    # 1. Setup Environment
    dev_dict = load_developers_dictionary()           # assumed external function
    cfg = dev_dict.get(broker_name)
    if not cfg:
        log(f"Broker '{broker_name}' not found.", "ERROR")
        return f"Error: Broker {broker_name} not in dictionary."

    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)     # assumed external function
    if not am_data:
        log(f"accountmanagement.json missing for {broker_name}", "ERROR")
        return "Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    liq_root = define_candles.get("liquidity_candle", {})

    total_configs_updated = 0
    total_liq_found = 0

    # 2. Iterate through apprehend_* definitions
    for apprehend_key, liq_cfg in liq_root.items():
        if not apprehend_key.startswith("apprehend_"):
            continue

        source_def_name = apprehend_key.replace("apprehend_", "")
        source_def = define_candles.get(source_def_name, {})

        if not source_def:
            log(f"No definition found for '{source_def_name}'. Skipping.", "WARNING")
            continue

        raw_filename = source_def.get("filename", "")
        if not raw_filename:
            continue

        target_file_filter = raw_filename.replace(".json", "").lower()
        primary_png_name = raw_filename.replace(".json", ".png")

        log(f"Processing: {apprehend_key} | Source: {source_def_name} | Filter: {target_file_filter}")

        # ── Determine swing direction type ───────────────────────────────────────
        is_bullish_structure = "higherhighsandhigherlows" in apprehend_key
        is_bearish_structure = "lowerhighsandlowerlows" in apprehend_key

        if not (is_bullish_structure or is_bearish_structure):
            log(f"Unknown structure type in key: {apprehend_key}", "WARNING")
            continue

        swing_prefix = "higher" if is_bullish_structure else "lower"

        log(f"[{apprehend_key}] Detected structure: {'BULLISH' if is_bullish_structure else 'BEARISH'} "
            f"→ looking for {swing_prefix}_high & {swing_prefix}_low", "INFO")

        # ── CONFIG READING ───────────────────────────────────────────────────────────────
        apprentice_section = liq_cfg.get("liquidity_apprentice_candle", {})
        apprentice_cfg = apprentice_section.get("swing_types", {})

        sweeper_section = liq_cfg.get("liquidity_sweeper_candle", {})
        liq_label_at = sweeper_section.get("label_at", {})

        hh_txt = liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_text", "Liq")
        hl_txt = liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_text", "Liq")

        markers = {
            "liq_hh": resolve_marker(liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_marker")),
            "liq_hl": resolve_marker(liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_marker")),
            "liq_hh_txt": hh_txt,
            "liq_hl_txt": hl_txt,
            "app_hh": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_high_marker")),
            "app_hl": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_low_marker"))
        }

        # ── Iterate through Symbols and ALL Timeframes ─────────────────────────────
        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p):
                continue

            timeframes = [d for d in os.listdir(sym_p) if os.path.isdir(os.path.join(sym_p, d))]

            for tf in timeframes:
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, "new_old", 101, raw_filename)
                output_dir = paths["output_dir"]
                config_path = os.path.join(output_dir, "config.json")

                if not os.path.exists(config_path):
                    continue

                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    config_modified = False

                    for file_key, candles in config_data.items():
                        is_primary_key = (file_key == source_def_name)
                        is_related_file = (target_file_filter in file_key.lower())

                        if not (is_primary_key or is_related_file):
                            continue

                        if not isinstance(candles, list):
                            continue

                        current_png = primary_png_name if is_primary_key else f"{file_key}.png"
                        png_path = os.path.join(output_dir, current_png)
                        img = cv2.imread(png_path) if os.path.exists(png_path) else None
                        key_modified = False

                        # ── SWEEP DETECTION ────────────────────────────────────────────────
                        for i, swing_c in enumerate(candles):
                            stype = str(swing_c.get("swing_type", "")).lower()
                            if not stype:
                                continue

                            if is_bullish_structure:
                                is_target_high = "higher_high" in stype
                                is_target_low  = "higher_low"  in stype
                            else:
                                is_target_high = "lower_high" in stype
                                is_target_low  = "lower_low"  in stype

                            if not (is_target_high or is_target_low):
                                continue

                            ref_price = swing_c.get("high") if is_target_high else swing_c.get("low")
                            if ref_price is None:
                                continue

                            for j in range(i + 1, len(candles)):
                                target_c = candles[j]
                                grabbed = False
                                txt = "Liq"

                                if is_target_high and target_c.get("high", 0) >= ref_price:
                                    grabbed = True
                                    obj, dbl = markers["liq_hh"]
                                    txt = markers["liq_hh_txt"]
                                    swept_pos = "high"
                                    liq_pos = "high"
                                    app_obj, app_dbl = markers["app_hh"]

                                elif is_target_low and target_c.get("low", 999999) <= ref_price:
                                    grabbed = True
                                    obj, dbl = markers["liq_hl"]
                                    txt = markers["liq_hl_txt"]
                                    swept_pos = "low"
                                    liq_pos = "low"
                                    app_obj, app_dbl = markers["app_hl"]

                                if grabbed:
                                    # ── LIQUIDITY CANDLE (sweeper) ──
                                    target_c["is_liquidity"] = True
                                    target_c["from_swing_index"] = swing_c.get("candle_number")
                                    target_c["swing_type_swept"] = stype
                                    target_c["swept_candle_number"] = target_c.get("candle_number")
                                    target_c["price_level"] = ref_price
                                    target_c["draw_coords"] = {
                                        "x": int(target_c.get("candle_left", 0) + (target_c.get("candle_width", 0) // 2)),
                                        "y": int(target_c.get("candle_top", 0)),
                                        "h": int(target_c.get("candle_height", 0)),
                                        "text": txt
                                    }

                                    # ── SWEPT SWING / APPRENTICE ──
                                    swing_c["swept_by_liquidity"] = True
                                    swing_c["swept_by_candle"] = target_c.get("candle_number")
                                    swing_c["draw_coords"] = {
                                        "x": int(swing_c.get("candle_left", 0) + (swing_c.get("candle_width", 0) // 2)),
                                        "y": int(swing_c.get("candle_top", 0)),
                                        "marker": app_obj
                                    }

                                    if img is not None:
                                        label_objects_and_text(
                                            img,
                                            target_c["draw_coords"]["x"],
                                            target_c["draw_coords"]["y"],
                                            target_c["draw_coords"]["h"],
                                            custom_text=txt,
                                            object_type=obj,
                                            is_bullish_arrow=(not is_target_high),
                                            is_marked=True,
                                            double_arrow=dbl,
                                            arrow_color=(0, 255, 255),
                                            label_position=liq_pos
                                        )

                                        label_objects_and_text(
                                            img,
                                            swing_c["draw_coords"]["x"],
                                            swing_c["draw_coords"]["y"],
                                            int(swing_c.get("candle_height", 0)),
                                            custom_text="",
                                            object_type=app_obj,
                                            is_bullish_arrow=(not is_target_high),
                                            is_marked=True,
                                            double_arrow=app_dbl,
                                            arrow_color=(255, 165, 0),
                                            label_position=swept_pos
                                        )

                                    key_modified = True
                                    config_modified = True
                                    total_liq_found += 1
                                    break

                        if key_modified and img is not None:
                            cv2.imwrite(png_path, img)
                            log(f"[CHART UPDATED] {sym} | {tf} → {current_png}")

                    if config_modified:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=4)
                        total_configs_updated += 1
                        log(f"Config saved: {sym} [{tf}]")

                except Exception as e:
                    log(f"Error processing {sym} ({tf}): {e}", "ERROR")

    log(f"--- ANALYSIS COMPLETE ---")
    log(f"Total liquidity sweeps found: {total_liq_found}")
    log(f"Config files updated: {total_configs_updated}")

    return f"Completed: {total_liq_found} sweeps across {total_configs_updated} files."

def liquidity_candle(broker_name):
    from datetime import datetime
    import pytz
    import os
    import json
    import cv2

    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def resolve_marker(raw):
        if not raw:
            return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]:
            return "arrow", False
        if raw in ["doublearrow", "doublearrows"]:
            return "arrow", True
        if raw in ["rightarrow", "right_arrow"]:
            return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]:
            return "leftarrow", False
        if "dot" in raw:
            return "dot", False
        return raw, False

    log(f"--- STARTING DUAL-TARGET LIQUIDITY ANALYSIS (ALL TFs): {broker_name} ---")

    dev_dict = load_developers_dictionary() 
    cfg = dev_dict.get(broker_name)
    if not cfg:
        log(f"Broker '{broker_name}' not found.", "ERROR")
        return f"Error: Broker {broker_name} not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        log(f"accountmanagement.json missing for {broker_name}", "ERROR")
        return "Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    liq_root = define_candles.get("liquidity_candle", {})
    
    total_configs_updated = 0
    total_liq_found = 0

    for apprehend_key, liq_cfg in liq_root.items():
        if not apprehend_key.startswith("apprehend_"):
            continue
        
        source_def_name = apprehend_key.replace("apprehend_", "")
        source_def = define_candles.get(source_def_name, {})
        
        if not source_def:
            log(f"No definition found for '{source_def_name}'. Skipping.", "WARNING")
            continue

        raw_filename = source_def.get("filename", "")
        if not raw_filename:
            continue
        
        target_file_filter = raw_filename.replace(".json", "").lower()
        primary_png_name = raw_filename.replace(".json", ".png")

        apprentice_section = liq_cfg.get("liquidity_apprentice_candle", {})
        apprentice_cfg = apprentice_section.get("swing_types", {})
        
        is_bullish_structure = any("higher" in k for k in apprentice_cfg.keys())
        is_bearish_structure = any("lower" in k for k in apprentice_cfg.keys())
        
        if not (is_bullish_structure or is_bearish_structure):
            continue

        swing_prefix = "higher" if is_bullish_structure else "lower"
        log(f"Processing: {apprehend_key} | Source: {source_def_name} | Filter: {target_file_filter}")

        sweeper_section = liq_cfg.get("liquidity_sweeper_candle", {})
        liq_label_at = sweeper_section.get("label_at", {})

        markers = {
            "liq_hh": resolve_marker(liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_marker")),
            "liq_hl": resolve_marker(liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_marker")),
            "liq_hh_txt": liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_text", "Liq"),
            "liq_hl_txt": liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_text", "Liq"),
            "app_hh": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_high_marker")),
            "app_hl": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_low_marker"))
        }

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in os.listdir(sym_p):
                tf_p = os.path.join(sym_p, tf)
                if not os.path.isdir(tf_p): continue
                
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, "new_old", 101, raw_filename)
                output_dir = paths["output_dir"]
                config_path = os.path.join(output_dir, "config.json")

                if not os.path.exists(config_path): continue

                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    config_modified = False
                    
                    for file_key, candles in config_data.items():
                        # --- FVG SPECIFIC KEY MATCHING ---
                        is_primary = (file_key.lower() == source_def_name.lower())
                        is_filter_match = (target_file_filter in file_key.lower())
                        
                        if not (is_primary or is_filter_match) or not isinstance(candles, list):
                            continue
                        
                        # LOG: Found a matching key
                        if "fvg" in file_key.lower():
                            log(f"[FVG DEBUG] Match found for key: {file_key} in {tf}")

                        # Check for image with fallback names
                        current_png = f"{file_key}.png"
                        png_path = os.path.join(output_dir, current_png)
                        
                        if not os.path.exists(png_path):
                            png_path = os.path.join(output_dir, primary_png_name)

                        img = cv2.imread(png_path) if os.path.exists(png_path) else None
                        
                        if img is None and "fvg" in file_key.lower():
                            log(f"[FVG DEBUG] Image NOT FOUND for {file_key}. Looked for {current_png} or {primary_png_name}", "WARNING")
                            continue

                        key_modified = False

                        for i, swing_c in enumerate(candles):
                            stype = str(swing_c.get("swing_type", "")).lower()
                            is_target_high = f"{swing_prefix}_high" in stype
                            is_target_low  = f"{swing_prefix}_low" in stype

                            if not (is_target_high or is_target_low): continue

                            ref_price = swing_c.get("high") if is_target_high else swing_c.get("low")
                            if ref_price is None: continue

                            for j in range(i + 1, len(candles)):
                                target_c = candles[j]
                                grabbed = False

                                if is_target_high and target_c.get("high", 0) >= ref_price:
                                    grabbed, pos, m_key, a_key = True, "high", "liq_hh", "app_hh"
                                elif is_target_low and target_c.get("low", 999999) <= ref_price:
                                    grabbed, pos, m_key, a_key = True, "low", "liq_hl", "app_hl"

                                if grabbed:
                                    obj, dbl = markers[m_key]
                                    app_obj, app_dbl = markers[a_key]
                                    txt = markers[f"{m_key}_txt"]

                                    # Update Data
                                    target_c.update({"is_liquidity": True, "price_level": ref_price, 
                                                   "draw_coords": {"x": int(target_c.get("candle_left", 0) + (target_c.get("candle_width", 0) // 2)),
                                                                  "y": int(target_c.get("candle_top", 0)), "h": int(target_c.get("candle_height", 0)), "text": txt}})
                                    
                                    swing_c.update({"swept_by_liquidity": True, 
                                                   "draw_coords": {"x": int(swing_c.get("candle_left", 0) + (swing_c.get("candle_width", 0) // 2)),
                                                                  "y": int(swing_c.get("candle_top", 0)), "marker": app_obj}})

                                    if img is not None:
                                        label_objects_and_text(img, target_c["draw_coords"]["x"], target_c["draw_coords"]["y"], target_c["draw_coords"]["h"],
                                                               custom_text=txt, object_type=obj, is_bullish_arrow=(not is_target_high),
                                                               is_marked=True, double_arrow=dbl, arrow_color=(0, 255, 255), label_position=pos)
                                        
                                        label_objects_and_text(img, swing_c["draw_coords"]["x"], swing_c["draw_coords"]["y"], int(swing_c.get("candle_height", 0)),
                                                               custom_text="", object_type=app_obj, is_bullish_arrow=(not is_target_high),
                                                               is_marked=True, double_arrow=app_dbl, arrow_color=(255, 165, 0), label_position=pos)

                                    key_modified = config_modified = True
                                    total_liq_found += 1
                                    break 

                        if key_modified and img is not None:
                            cv2.imwrite(png_path, img)
                            log(f"[CHART UPDATED] {sym} | {tf} → {os.path.basename(png_path)}")

                    if config_modified:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=4)
                        total_configs_updated += 1
                        log(f"Config saved: {sym} [{tf}]")

                except Exception as e:
                    log(f"Error processing {sym} ({tf}): {e}", "ERROR")

    log(f"--- ANALYSIS COMPLETE --- Found: {total_liq_found}")
    return f"Completed: {total_liq_found} sweeps."  

def single():
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return


    broker_names = sorted(dev_dict.keys()) 
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:

        # STEP 2: Higher Highs & Higher Lows
        hh_hl_results = pool.map(liquidity_candle, broker_names)
        for r in hh_hl_results: print(r)

def main():
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return

    broker_names = sorted(dev_dict.keys())
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:
        print("\n[STEP 2] Running Higher Highs & Higher Lows Analysis...")
        hh_hl_results = pool.map(higher_highs_higher_lows, broker_names)
        for r in hh_hl_results: print(r)


        print("\n[STEP 3] Running Lower Highs & Lower Lows Analysis...")
        lh_ll_results = pool.map(lower_highs_lower_lows, broker_names)
        for r in lh_ll_results: print(r)


        print("\n[STEP 4] Running Fair Value Gap Analysis...")
        fvg_results = pool.map(fair_value_gaps, broker_names)
        for r in fvg_results: print(r)
        
        hh_hl_results = pool.map(fvg_higherhighsandhigherlows, broker_names)
        for r in hh_hl_results: print(r)


        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(directional_bias, broker_names)
        for r in db_results: print(r)


        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(directional_bias_fvg, broker_names)
        for r in db_results: print(r)


        print("\n[STEP 6] Running Timeframes Communication Analysis...")
        tf_comm_results = pool.map(timeframes_communication, broker_names)
        for r in tf_comm_results: print(r)

        
        hh_hl_results = pool.map(enrich_receiver_comm_paths, broker_names)
        for r in hh_hl_results: print(r)

        hh_hl_results = pool.map(receiver_comm_higher_highs_higher_lows, broker_names)
        for r in hh_hl_results: print(r)

        hh_hl_results = pool.map(receiver_directional_bias, broker_names)
        for r in hh_hl_results: print(r)


        

    print("\n[SUCCESS] All tasks completed.")

if __name__ == "__main__":
    main()





                



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
from collections import defaultdict



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
    fvg_swing_type=None,                 
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
    end_x=None,
    box_w=None,          # External function provides this
    box_h=None,          # External function provides this
    box_alpha=0.3        # Transparency threshold
):
    color = object_color if object_color != (0, 255, 0) else arrow_color

    # Dimensions for markers
    shaft_length = 26
    head_size = 9
    thickness = 2
    wing_size = 7 if double_arrow else 6

    # 1. Determine Vertical Placement (Anchor Point)
    if label_position == "auto":
        place_at_high = not is_bullish_arrow  # HH/LH → top, ll/LL → bottom
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

        elif object_type == "lline":
            # Thin 1px horizontal line
            stop_x = end_x if end_x is not None else img.shape[1]
            cv2.line(img, (cx, tip_y), (int(stop_x), tip_y), color, 1)

        elif object_type == "box_transparent":
            if box_w is not None and box_h is not None:
                # Calculate coordinates based on passed width/height
                x1, y1 = cx - (box_w // 2), tip_y - (box_h // 2)
                x2, y2 = x1 + box_w, y1 + box_h
                
                # Overlay for transparency
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, box_alpha, img, 1 - box_alpha, 0, img)
                
                # 1px Border
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

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

    # 3. Text Placement Logic
    if not (custom_text or fvg_swing_type is not None):
        return

    if is_marked:
        is_vertical_obj = object_type in ["arrow", "reverse_arrow"]
        if is_vertical_obj:
            reach = (shaft_length + head_size + 4)
        elif object_type == "box_transparent" and box_h is not None:
            reach = (box_h // 2) + 4
        else:
            reach = 14
    else:
        reach = 4

    if place_at_high:
        base_text_y = tip_y - reach
    else:
        base_text_y = tip_y + reach + 10

    if custom_text:
        (tw, th), _ = cv2.getTextSize(custom_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.putText(img, custom_text, (cx - tw // 2, int(base_text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, arrow_color, text_thickness)
        fvg_swing_type_y = (base_text_y - 15) if place_at_high else (base_text_y + 15)
    else:
        fvg_swing_type_y = base_text_y

    if fvg_swing_type is not None:
        cv2.putText(img, str(fvg_swing_type), (cx - 8, int(fvg_swing_type_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(img, str(fvg_swing_type), (cx - 8, int(fvg_swing_type_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)                   

def lower_highs_higher_lows(broker_name):
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
    
    # --- CONFIG LOGIC UPDATE ---
    # We look for the Parent (HH/LL) config specifically to steal its BARS setting
    parent_keyword = "higherhighsandlowerlows"
    parent_cfg_list = [v for k, v in define_candles.items() if parent_keyword in k.lower()]
    parent_bars = parent_cfg_list[0].get("BARS", 101) if parent_cfg_list else 101
    
    keyword = "lowerhighsandhigherlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]

    if not matching_configs:
        return f"[{broker_name}] Error: No configuration found for '{keyword}'."

    log(f"--- STARTING IDENTIFICATION (CHILD): {broker_name} ---")
    log(f"Using Parent BARS: {parent_bars}")

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

    for config_key, llhl_cfg in matching_configs:
        log(f"Processing Config Key: [{config_key}]")
        # Overriding local BARS with parent_bars as requested
        bars = parent_bars 
        output_filename_base = llhl_cfg.get("filename", "lowers.json")
        direction = llhl_cfg.get("read_candles_from", "new_old")
        
        neighbor_left = llhl_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = llhl_cfg.get("NEIGHBOR_RIGHT", 5)

        label_cfg = llhl_cfg.get("label", {})
        lh_text = label_cfg.get("lowerhighs_text", "LH")
        ll_text = label_cfg.get("higherlows_text", "HL")
        cm_text = label_cfg.get("contourmaker_text", "m")

        label_at = label_cfg.get("label_at", {})
        lh_pos = label_at.get("lower_highs", "high").lower()
        ll_pos = label_at.get("higher_lows", "low").lower()

        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        lh_col = color_map.get(label_at.get("lower_highs_color", "red").lower(), (255, 0, 0))
        ll_col = color_map.get(label_at.get("higher_lows_color", "green").lower(), (0, 255, 0))

        lh_obj, lh_dbl = resolve_marker(label_at.get("lower_highs_marker", "arrow"))
        ll_obj, ll_dbl = resolve_marker(label_at.get("higher_lows_marker", "arrow"))
        lh_cm_obj, lh_cm_dbl = resolve_marker(label_at.get("lower_highs_contourmaker_marker", ""))
        ll_cm_obj, ll_cm_dbl = resolve_marker(label_at.get("higher_lows_contourmaker_marker", ""))

        symbols = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
        
        for sym in symbols:
            sym_p = os.path.join(base_folder, sym)
            timeframes = sorted(os.listdir(sym_p))

            for tf in timeframes:
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
                config_path = os.path.join(paths["output_dir"], "config.json")

                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                    continue

                try:
                    # 1. Load existing config to check for Parent (HH/LL) claims
                    parent_claimed_candles = set()
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            existing_cfg_data = json.load(f)
                            # Look for any key containing "higherhighsandlowerlows"
                            for k, v in existing_cfg_data.items():
                                if parent_keyword in k.lower() and isinstance(v, list):
                                    for candle in v:
                                        if candle.get("is_swing"):
                                            parent_claimed_candles.add(candle.get("candle_number"))

                    # 2. Load candle data
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

                    # Map coordinates
                    for idx, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        data[idx].update({
                            "candle_x": x + (w // 2), "candle_y": y,
                            "candle_width": w, "candle_height": h,
                            "candle_left": x, "candle_right": x + w,
                            "candle_top": y, "candle_bottom": y + h
                        })

                    n = len(data)
                    swing_count_in_chart = 0
                    for i in range(neighbor_left, n - neighbor_right):
                        fvg_swing_type = data[i].get('candle_number')
                        
                        # --- EXCLUSION CHECK ---
                        # If the Parent already claimed this candle, we skip it
                        if fvg_swing_type in parent_claimed_candles:
                            continue

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

                            label_objects_and_text(
                                img, data[i]["candle_x"], data[i]["candle_y"], data[i]["candle_height"], 
                                fvg_swing_type=fvg_swing_type,
                                custom_text=custom_text, object_type=obj_type,
                                is_bullish_arrow=is_bull, is_marked=True,
                                double_arrow=dbl_arrow, arrow_color=active_color,
                                label_position=position
                            )

                            # Handle Contour Maker
                            m_idx = i + neighbor_right
                            contour_maker_entry = None
                            if m_idx < n:
                                cm_obj = ll_cm_obj if is_bull else lh_cm_obj
                                cm_dbl = ll_cm_dbl if is_bull else lh_cm_dbl
                                
                                label_objects_and_text(
                                    img, data[m_idx]["candle_x"], data[m_idx]["candle_y"], data[m_idx]["candle_height"], 
                                    custom_text=cm_text, object_type=cm_obj,
                                    is_bullish_arrow=is_bull, is_marked=True,
                                    double_arrow=cm_dbl, arrow_color=active_color,
                                    label_position=position
                                )
                                contour_maker_entry = data[m_idx].copy()
                                contour_maker_entry.update({"is_contour_maker": True})

                            data[i].update({
                                "swing_type": "higher_low" if is_bull else "lower_high",
                                "is_swing": True, "active_color": active_color,
                                "draw_x": data[i]["candle_x"], "draw_y": data[i]["candle_y"],
                                "draw_w": data[i]["candle_width"], "draw_h": data[i]["candle_height"],
                                "contour_maker": contour_maker_entry,
                                "m_idx": m_idx if m_idx < n else None
                            })

                    # Save visual chart and Update config.json
                    os.makedirs(paths["output_dir"], exist_ok=True)
                    cv2.imwrite(paths["output_chart"], img)

                    config_content = {}
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            try: config_content = json.load(f)
                            except: config_content = {}

                    config_content[config_key] = data
                    config_content[f"{config_key}_candle_list"] = data

                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, indent=4)
                    
                    processed_charts_all += 1
                    total_marked_all += swing_count_in_chart

                except Exception as e:
                    log(f"Error in {sym}/{tf}: {e}", "ERROR")

    summary = f"COMPLETED. Broker: {broker_name} | Total Swings: {total_marked_all} | Total Charts: {processed_charts_all}"
    log(summary)
    return summary

def higher_highs_lower_lows(broker_name):

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
    keyword = "higherhighsandlowerlows"
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

    log(f"--- STARTING HH/ll ANALYSIS: {broker_name} ---")

    for config_key, hlll_cfg in matching_configs:
        bars = hlll_cfg.get("BARS", 101)
        output_filename_base = hlll_cfg.get("filename", "highers.json")
        direction = hlll_cfg.get("read_candles_from", "new_old")
        
        neighbor_left = hlll_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = hlll_cfg.get("NEIGHBOR_RIGHT", 5)
        label_cfg = hlll_cfg.get("label", {})
        hh_text = label_cfg.get("higherhighs_text", "HH")
        ll_text = label_cfg.get("lowerlows_text", "ll")
        cm_text = label_cfg.get("contourmaker_text", "m")
        label_at = label_cfg.get("label_at", {})
        hh_pos = label_at.get("higher_highs", "high").lower()
        ll_pos = label_at.get("lower_lows", "low").lower()
        
        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red").lower(), (255, 0, 0))
        ll_col = color_map.get(label_at.get("lower_lows_color", "green").lower(), (0, 255, 0))
        
        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        ll_obj, ll_dbl = resolve_marker(label_at.get("lower_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        ll_cm_obj, ll_cm_dbl = resolve_marker(label_at.get("lower_lows_contourmaker_marker", ""))

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            
            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
                config_path = os.path.join(paths["output_dir"], "config.json")
                
                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                    continue
                
                try:
                    # Logging specific pair and timeframe
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: 
                        log(f"   Skipping: Could not load image {paths['source_chart']}", "WARNING")
                        continue
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) == 0: 
                        log(f"   No contours found for {sym} {tf}")
                        continue

                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                    
                    if len(data) != len(contours):
                        min_len = min(len(data), len(contours))
                        data = data[:min_len]
                        contours = contours[:min_len]

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
                        is_ll = curr_l < min(l_l) and curr_l < min(r_l)
                        
                        if not (is_hh or is_ll):
                            continue
                        
                        swing_count_in_chart += 1
                        is_bull = is_ll
                        active_color = ll_col if is_bull else hh_col
                        custom_text = ll_text if is_bull else hh_text
                        obj_type = ll_obj if is_bull else hh_obj
                        dbl_arrow = ll_dbl if is_bull else hh_dbl
                        position = ll_pos if is_bull else hh_pos

                        label_objects_and_text(
                            img, data[i]["candle_x"], data[i]["candle_y"], data[i]["candle_height"],
                            fvg_swing_type=data[i]['candle_number'],
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
                            cm_obj = ll_cm_obj if is_bull else hh_cm_obj
                            cm_dbl = ll_cm_dbl if is_bull else hh_cm_dbl
                            
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

                            data[m_idx]["is_contour_maker"] = True
                            contour_maker_entry = data[m_idx].copy()
                            contour_maker_entry.update({
                                "draw_x": data[m_idx]["candle_x"], "draw_y": data[m_idx]["candle_y"],
                                "draw_w": data[m_idx]["candle_width"], "draw_h": data[m_idx]["candle_height"],
                                "draw_left": data[m_idx]["candle_left"], "draw_right": data[m_idx]["candle_right"],
                                "draw_top": data[m_idx]["candle_top"], "draw_bottom": data[m_idx]["candle_bottom"],
                                "is_contour_maker": True
                            })

                        data[i].update({
                            "swing_type": "lower_low" if is_bull else "higher_high",
                            "is_swing": True,
                            "active_color": active_color,
                            "draw_x": data[i]["candle_x"], "draw_y": data[i]["candle_y"],
                            "draw_w": data[i]["candle_width"], "draw_h": data[i]["candle_height"],
                            "draw_left": data[i]["candle_left"], "draw_right": data[i]["candle_right"],
                            "draw_top": data[i]["candle_top"], "draw_bottom": data[i]["candle_bottom"],
                            "contour_maker": contour_maker_entry,
                            "m_idx": m_idx if m_idx < n else None
                        })

                    # Finalize outputs for this specific TF
                    os.makedirs(paths["output_dir"], exist_ok=True)
                    cv2.imwrite(paths["output_chart"], img)

                    config_json = {}
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_json = json.load(f)
                        except:
                            config_json = {}
                    
                    config_json[config_key] = data
                    config_json[f"{config_key}_candle_list"] = data 

                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_json, f, indent=4)
                    
                    log(f"{sym} | {tf} | Key: {config_key} Swings found: {swing_count_in_chart}")
                    
                    processed_charts_all += 1
                    total_marked_all += swing_count_in_chart

                except Exception as e:
                    log(f"   [ERROR] Failed processing {sym}/{tf}: {e}", "ERROR")

    log(f"--- HH/ll COMPLETE --- Total Swings: {total_marked_all} | Total Charts: {processed_charts_all}")
    return f"Identify Done. Swings: {total_marked_all} | Charts: {processed_charts_all}"  

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

        # source_config_name is the key in config.json (e.g., "value")
        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        if not source_config: continue

        bars = source_config.get("BARS", 101)
        filename = source_config.get("filename", "output.json")
        
        is_hlll = "higherhighsandlowerlows" in source_config_name.lower()
        is_llhl = "lowerhighsandhigherlows"   in source_config_name.lower()

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in sorted(os.listdir(sym_p)):
                dev_output_dir = os.path.join(os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)), sym, tf)
                config_json_path = os.path.join(dev_output_dir, "config.json")
                
                # Check if the config file exists
                if not os.path.exists(config_json_path):
                    continue

                paths = get_analysis_paths(base_folder, broker_name, sym, tf, "new_old", bars, filename)
                
                # We still need the chart and source_json (candle prices) to process logic
                if not os.path.exists(paths.get("source_json")) or not os.path.exists(paths.get("source_chart")):
                    continue

                try:
                    # 1. Load Price Data (for calculations)
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        full_data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    # 2. Load the developer config.json (The actual Target)
                    with open(config_json_path, 'r', encoding='utf-8') as f:
                        local_config = json.load(f)

                    # Get the list data (e.g., local_config["value"])
                    input_structures = local_config.get(source_config_name, [])
                    if not input_structures:
                        continue

                    # 3. Setup CV2 Images
                    clean_img  = cv2.imread(paths["source_chart"])
                    marked_img = cv2.imread(paths["output_chart"]) if os.path.exists(paths["output_chart"]) else clean_img.copy()
                    if clean_img is None: continue

                    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(raw_contours, key=lambda c: cv2.boundingRect(c)[0])
                    n_candles = len(full_data)

                    final_flat_list = []
                    for structure in input_structures:
                        # Extract the base Marker
                        marker_candle = {k: v for k, v in structure.items() if k not in ["contour_maker", "directional_bias"]}
                        final_flat_list.append(marker_candle)

                        reference_idx = None
                        reference_high = None
                        reference_low = None
                        active_color = tuple(structure.get("active_color", [0,255,0]))

                        # Process Contour Maker
                        if (is_hlll or is_llhl) and target_type == "contourmaker":
                            cm_data = structure.get("contour_maker")
                            if cm_data:
                                cm_only = {k: v for k, v in cm_data.items() if k != "contour_maker_liquidity_candle"}
                                cm_only["is_contour_maker"] = True
                                final_flat_list.append(cm_only)
                                reference_idx = structure.get("m_idx")
                                reference_high, reference_low = cm_data["high"], cm_data["low"]

                        if reference_idx is None or reference_idx >= n_candles:
                            continue

                        # Process Level 1 Directional Bias
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
                            final_flat_list.append(first_db_info)

                            # Process Contour Maker Liquidity Sweep
                            for l_idx in range(db_idx + 1, n_candles):
                                l_candle = full_data[l_idx]
                                if (base_type == "support" and l_candle['low'] < reference_low) or \
                                   (base_type == "resistance" and l_candle['high'] > reference_high):
                                    liq_obj = {**l_candle, "idx": l_idx, "is_contour_maker_liquidity": True}
                                    final_flat_list.append(liq_obj)
                                    total_liq_marked += 1
                                    break

                            # Visual Marking
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

                            # Process Level 2 Bias (Self Apprehend)
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

                                    for dl_idx in range(s_idx + 1, n_candles):
                                        dl_candle = full_data[dl_idx]
                                        if (next_base_type == "support" and dl_candle['low'] < s_ref_l) or \
                                           (next_base_type == "resistance" and dl_candle['high'] > s_ref_h):
                                            final_flat_list.append({**dl_candle, "idx": dl_idx, "directional_bias_liquidity_candle": True})
                                            total_liq_marked += 1
                                            break

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

                    # Save visual chart
                    cv2.imwrite(paths["output_chart"], marked_img)
                    
                    # Update ONLY the specific key in local_config
                    local_config[source_config_name] = final_flat_list
                    # Optional: keep price data for other functions to use
                    local_config[f"{source_config_name}_candle_list"] = full_data
                    
                    with open(config_json_path, 'w', encoding='utf-8') as f:
                        json.dump(local_config, f, indent=4)
                    
                    log(f"Successfully finalized {sym} {tf}")

                except Exception as e:
                    log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Directional Bias Done. DB Markers: {total_db_marked}, Liq Sweeps: {total_liq_marked}"

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
                    # Storage for the FVG attributes to apply to candles 1, 2, and 3
                    potential_fvgs_map = {} 
                    c1_tags = {} # candle_num -> {fvg_c1: True, c1_for_fvg_number: X}
                    c3_tags = {} # candle_num -> {fvg_c3: True, c3_for_fvg_number: X}
                    
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
                                # Candle 2 (The FVG) Data
                                enriched_c2 = c2.copy()
                                enriched_c2.update({
                                    "fvg_type": fvg_type,
                                    "fvg_gap_size": gap_size,
                                    "fvg_gap_top": round(gap_top, 5),
                                    "fvg_gap_bottom": round(gap_bottom, 5),
                                    "c1_lookback_used": c1_lookback,
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
                                    "is_fvg": True,
                                    "_contour_idx": i,
                                    "_is_bull_c2": is_bullish_fvg_c
                                })
                                potential_fvgs_map[c2.get('candle_number')] = enriched_c2
                                
                                # Tag Candle 1 and Candle 3 for later list construction
                                c1_tags[c1.get('candle_number')] = {"fvg_c1": True, "c1_for_fvg_number": c2.get('candle_number')}
                                c3_tags[c3.get('candle_number')] = {"fvg_c3": True, "c3_for_fvg_number": c2.get('candle_number')}

                    # --- Build Final JSON (Flat List Structure) ---
                    fvg_results = []
                    max_gap_found = max([p["fvg_gap_size"] for p in potential_fvgs_map.values()], default=0)
                    
                    for idx, candle in enumerate(data):
                        fvg_swing_type = candle.get('candle_number')
                        
                        # Apply contour/coordinate data to every candle
                        if idx < len(contours):
                            x_c, y_c, w_c, h_c = cv2.boundingRect(contours[idx])
                            candle.update({
                                "candle_x": x_c + w_c // 2, "candle_y": y_c, "candle_width": w_c, "candle_height": h_c,
                                "candle_left": x_c, "candle_right": x_c + w_c, "candle_top": y_c, "candle_bottom": y_c + h_c,
                                "draw_x": x_c + w_c // 2, "draw_y": y_c, "draw_w": w_c, "draw_h": h_c,
                                "draw_left": x_c, "draw_right": x_c + w_c, "draw_top": y_c, "draw_bottom": y_c + h_c
                            })

                        # If this candle is C2 (The FVG)
                        if fvg_swing_type in potential_fvgs_map:
                            entry = potential_fvgs_map[fvg_swing_type]
                            # Sync coordinates from the base candle to the enriched one
                            coord_keys = ["candle_x", "candle_y", "candle_width", "candle_height", "candle_left", "candle_right", "candle_top", "candle_bottom", "draw_x", "draw_y", "draw_w", "draw_h", "draw_left", "draw_right", "draw_top", "draw_bottom"]
                            entry.update({k: candle[k] for k in coord_keys if k in candle})

                            is_tallest = (entry["fvg_gap_size"] == max_gap_found)
                            wick_compromised = (entry["c1_upper_and_lower_wick_higher"] or entry["c3_upper_and_lower_wick_higher"])
                            entry["constestant_fvg_chosed"] = is_tallest and (wick_compromised if apply_body_vs_wick_rule else True)
                            
                            body_condition_ok = ((entry["c1_body_higher"] and entry["c3_body_higher"]) or entry["constestant_fvg_chosed"]) if apply_body_vs_wick_rule else True
                            
                            if body_condition_ok:
                                c_idx = entry["_contour_idx"]
                                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[c_idx])
                                label_objects_and_text(
                                    img=img, cx=x_rect + w_rect // 2, y_rect=y_rect, h_rect=h_rect,
                                    fvg_swing_type=fvg_swing_type if (number_all or number_only_marked) else None,
                                    custom_text=bull_text if entry["fvg_type"] == "bullish" else bear_text,
                                    object_type=bull_obj if entry["fvg_type"] == "bullish" else bear_obj,
                                    is_bullish_arrow=entry["_is_bull_c2"], is_marked=True,
                                    double_arrow=bull_double if entry["fvg_type"] == "bullish" else bear_double,
                                    arrow_color=bullish_color if entry["_is_bull_c2"] else bearish_color,
                                    label_position="low" if entry["_is_bull_c2"] else "high"
                                )
                                final_entry = {k: v for k, v in entry.items() if not k.startswith("_") and k != "fvg_gap_size"}
                                fvg_results.append(final_entry)
                                marked_count += 1
                                continue 
                        
                        # If this candle is C1 or C3 for an FVG, apply the tags
                        if fvg_swing_type in c1_tags:
                            candle.update(c1_tags[fvg_swing_type])
                        if fvg_swing_type in c3_tags:
                            candle.update(c3_tags[fvg_swing_type])

                        fvg_results.append(candle)
                    
                    if marked_count > 0 or (number_all and len(data) > 0):
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        config_path = os.path.join(paths["output_dir"], "config.json")
                        try:
                            config_content = {}
                            if os.path.exists(config_path):
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    try:
                                        config_content = json.load(f)
                                        if not isinstance(config_content, dict): config_content = {}
                                    except: config_content = {}
                            config_content[config_key] = fvg_results
                            with open(config_path, 'w', encoding='utf-8') as f:
                                json.dump(config_content, f, indent=4)
                        except Exception as e:
                            log(f"Config sync failed for {sym}/{tf}: {e}", "WARN")

                        processed_charts += 1
                        total_marked += marked_count
                        
                except Exception as e:
                    log(f"Error processing {sym}/{tf} with config '{config_key}': {e}", "ERROR")

        log(f"Completed config '{config_key}': FVGs marked: {total_marked} | Charts: {processed_charts}")
        total_marked_all += total_marked
        processed_charts_all += processed_charts

    return f"Done (all FVG configs). Total FVGs: {total_marked_all} | Total Charts: {processed_charts_all}"

def fvg_higherhighsandlowerlows(broker_name):
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
        return f"[{broker_name}] Warning: No FVG config found → cannot determine parameters."

    config_key, fvg_cfg = fvg_configs[0]
    log(f"Using FVG config section: {config_key} for swing detection parameters")

    direction = fvg_cfg.get("read_candles_from", "new_old")
    bars = fvg_cfg.get("BARS", 301)
    output_filename_base = fvg_cfg.get("filename", "fvg.json")

    NEIGHBOR_LEFT = fvg_cfg.get("NEIGHBOR_LEFT", 5)
    NEIGHBOR_RIGHT = fvg_cfg.get("NEIGHBOR_RIGHT", 5)

    HH_TEXT = "fvg-HH"
    ll_TEXT = "fvg-ll"
    CM_TEXT = "m"  
    color_map = {"green": (0, 255, 0), "red": (255, 0, 0)}
    HH_COLOR = color_map["red"]
    ll_COLOR = color_map["green"]

    total_swings_added = 0
    processed_charts = 0

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p): continue

        for tf in sorted(os.listdir(sym_p)):
            tf_path = os.path.join(sym_p, tf)
            if not os.path.isdir(tf_path): continue

            paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
            config_path = os.path.join(paths["output_dir"], "config.json")
            chart_path = paths["output_chart"]

            if not os.path.exists(config_path) or not os.path.exists(chart_path):
                continue

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content = json.load(f)

                data = config_content.get(config_key)
                if not data or not isinstance(data, list):
                    continue

                data = sorted(data, key=lambda x: x.get('candle_number', 0))
                img = cv2.imread(chart_path)
                if img is None: continue

                n = len(data)
                min_req = NEIGHBOR_LEFT + NEIGHBOR_RIGHT + 1
                if n < min_req: continue

                swing_count = 0
                swings_found = []

                # --- STEP 1: DETECT SWINGS & MARK CONTOUR MAKERS ---
                for i in range(NEIGHBOR_LEFT, n - NEIGHBOR_RIGHT):
                    curr_h, curr_l = data[i]['high'], data[i]['low']
                    left_highs = [d['high'] for d in data[i - NEIGHBOR_LEFT:i]]
                    right_highs = [d['high'] for d in data[i + 1:i + 1 + NEIGHBOR_RIGHT]]
                    left_lows  = [d['low']  for d in data[i - NEIGHBOR_LEFT:i]]
                    right_lows  = [d['low']  for d in data[i + 1:i + 1 + NEIGHBOR_RIGHT]]

                    is_hh = curr_h > max(left_highs) and curr_h > max(right_highs)
                    is_ll = curr_l < min(left_lows)  and curr_l < min(right_lows)

                    if not (is_hh or is_ll): continue

                    swing_count += 1
                    is_bullish_swing = is_ll
                    swing_color = ll_COLOR if is_bullish_swing else HH_COLOR
                    
                    label_objects_and_text(
                        img=img, cx=data[i].get("candle_x"), y_rect=data[i].get("candle_y"), h_rect=data[i].get("candle_height"),
                        fvg_swing_type=data[i].get('candle_number'),
                        custom_text=ll_TEXT if is_bullish_swing else HH_TEXT, object_type="arrow",
                        is_bullish_arrow=is_bullish_swing, is_marked=True,
                        double_arrow=False, arrow_color=swing_color,
                        label_position="low" if is_bullish_swing else "high"
                    )

                    m_idx = i + NEIGHBOR_RIGHT
                    contour_maker_data = None
                    if m_idx < n:
                        label_objects_and_text(
                            img=img, cx=data[m_idx].get("candle_x"), y_rect=data[m_idx].get("candle_y"), h_rect=data[m_idx].get("candle_height"),
                            custom_text=CM_TEXT, object_type="dot",
                            is_bullish_arrow=is_bullish_swing, is_marked=True,
                            double_arrow=False, arrow_color=swing_color,
                            label_position="low" if is_bullish_swing else "high"
                        )
                        
                        data[m_idx].update({"is_contour_maker": True, "contour_maker_for_swing_candle": data[i]['candle_number']})
                        contour_maker_data = {
                            "candle_number": data[m_idx]['candle_number'],
                            "candle_x": data[m_idx].get("candle_x"),
                            "candle_y": data[m_idx].get("candle_y"),
                            "data_index": m_idx
                        }

                    swing_entry = {
                        "candle_number": data[i].get('candle_number'),
                        "swing_type": "higher_low" if is_bullish_swing else "higher_high",
                        "high": curr_h,
                        "low": curr_l
                    }
                    data[i].update({
                        "swing_type": swing_entry["swing_type"],
                        "is_swing": True,
                        "swing_color_bgr": [int(c) for c in swing_color],
                        "m_idx": m_idx if m_idx < n else None,
                        "contour_maker": contour_maker_data
                    })
                    swings_found.append(swing_entry)

                # --- STEP 2: ASSIGN BASE TYPE (SUPPORT/RESISTANCE) ---
                for candle in data:
                    ref_price = candle.get("low")
                    new_flag = None
                    for s in swings_found:
                        if s["swing_type"] == "higher_low" and s["low"] > ref_price:
                            new_flag = "support"
                            break
                        elif s["swing_type"] == "higher_high" and s["high"] < ref_price:
                            new_flag = "resistance"
                            break
                    
                    if new_flag:
                        if candle.get("fvg_c1") is True: candle["fvg_c1_base"] = new_flag
                        if candle.get("is_fvg") is True: candle["fvg_base"] = new_flag
                        if candle.get("fvg_c3") is True: candle["fvg_c3_base"] = new_flag

                # --- STEP 3: DEDUPLICATE ---
                unique_data_dict = {}
                for entry in data:
                    c_num = entry.get("candle_number")
                    if c_num not in unique_data_dict: unique_data_dict[c_num] = entry
                    else:
                        for key, val in entry.items():
                            if key not in unique_data_dict[c_num] or unique_data_dict[c_num][key] is None:
                                unique_data_dict[c_num][key] = val
                data = sorted(unique_data_dict.values(), key=lambda x: x.get('candle_number', 0))

                # --- STEP 4: ASSOCIATE SWINGS AND FIND LIQUIDITY ---
                for i in range(2, len(data)):
                    if data[i].get("fvg_c3") is True:
                        c1, c2, c3 = data[i-2], data[i-1], data[i]
                        family_str = f"{c1['candle_number']}, {c2['candle_number']}, {c3['candle_number']}"
                        
                        target_swing_idx = -1
                        cm_idx = -1
                        for j in range(i + 1, len(data)):
                            if data[j].get("is_swing"):
                                data[j]["swing_type_for_fvgf_number"] = family_str
                                target_swing_idx = j
                                cm_idx = data[j].get("m_idx", -1)
                                break
                        
                        if target_swing_idx != -1 and cm_idx != -1:
                            target_candles = [("c1", c1), ("c2", c2), ("c3", c3)]
                            
                            for label, triad_candle in target_candles:
                                triad_h = triad_candle['high']
                                triad_l = triad_candle['low']
                                found_h_liq = False
                                found_l_liq = False
                                
                                for k in range(cm_idx + 1, len(data)):
                                    liq_cand = data[k]
                                    l_open = liq_cand['open']
                                    l_high = liq_cand['high']
                                    l_low  = liq_cand['low']
                                    l_num  = liq_cand['candle_number']
                                    
                                    # --- High Liquidation Check ---
                                    if not found_h_liq:
                                        if l_open > triad_h:
                                            if l_low < triad_h: found_h_liq = True
                                        else:
                                            if l_high > triad_h: found_h_liq = True
                                        
                                        if found_h_liq:
                                            triad_candle[f"fvg_{label}_high_liquidated_by_candle_number"] = l_num
                                            liq_cand[f"liquidates_fvg_{label}_high"] = True

                                    # --- Low Liquidation Check ---
                                    if not found_l_liq:
                                        if l_open < triad_l:
                                            if l_high > triad_l: found_l_liq = True
                                        else:
                                            if l_low < triad_l: found_l_liq = True
                                        
                                        if found_l_liq:
                                            triad_candle[f"fvg_{label}_low_liquidated_by_candle_number"] = l_num
                                            liq_cand[f"liquidates_fvg_{label}_low"] = True

                                    if found_h_liq and found_l_liq: 
                                        break

                if swing_count > 0:
                    cv2.imwrite(chart_path, img)
                    config_content[config_key] = data
                    config_content[f"{config_key}_candle_list"] = data 
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_content, f, indent=4)
                    processed_charts += 1
                    total_swings_added += swing_count
                    log(f"Processed {sym}/{tf}: Swings, Base Types, & Liquidity mapped.")

            except Exception as e:
                log(f"Error in {sym}/{tf}: {e}", "ERROR")

    return f"Finished. Total Swings: {total_swings_added}, Charts: {processed_charts}"

def timeframes_communication(broker_name):
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
        rc={'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    )

    for apprehend_key, comm_cfg in tf_comm_section.items():
        if not isinstance(comm_cfg, dict) or not apprehend_key.startswith("apprehend_"):
            continue

        log(f"Processing communication strategy: '{apprehend_key}'")

        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        if not source_config:
            log(f"Source config '{source_config_name}' not found.", "ERROR")
            continue

        sender_raw = comm_cfg.get("timeframe_sender", "").strip()
        receiver_raw = comm_cfg.get("timeframe_receiver", "").strip()
        sender_tfs = [tf_normalize.get(s.strip().lower(), s.strip().lower()) for s in sender_raw.split(",") if s.strip()]
        receiver_tfs = [tf_normalize.get(r.strip().lower(), r.strip().lower()) for r in receiver_raw.split(",") if r.strip()]

        raw_targets = comm_cfg.get("target", "")
        targets = [t.strip().lower() for t in raw_targets.split(",") if t.strip()]
        
        source_filename = source_config.get("filename", "output.json")
        base_output_name = source_filename.replace(".json", "")
        bars = source_config.get("BARS", 101)

        for sym in sorted(os.listdir(base_folder)):
            sym_path = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_path): continue

            for sender_tf, receiver_tf in zip(sender_tfs, receiver_tfs):
                log(f"{sym}: Processing {sender_tf} → {receiver_tf}")

                sender_tf_path = os.path.join(sym_path, sender_tf)
                receiver_tf_path = os.path.join(sym_path, receiver_tf)

                if not os.path.isdir(sender_tf_path) or not os.path.isdir(receiver_tf_path):
                    continue

                dev_output_dir = os.path.join(
                    os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)),
                    sym, sender_tf
                )
                config_json_path = os.path.join(dev_output_dir, "config.json")

                if not os.path.exists(config_json_path): continue

                try:
                    # 1. LOAD CONFIG
                    with open(config_json_path, 'r', encoding='utf-8') as f:
                        local_config = json.load(f)

                    structures = local_config.get(source_config_name, [])
                    if not structures: continue

                    # 2. LOAD RECEIVER CANDLES
                    receiver_full_json = os.path.join(receiver_tf_path, "candlesdetails", "newest_oldest.json")
                    with open(receiver_full_json, 'r', encoding='utf-8') as f:
                        all_receiver_candles = json.load(f)

                    df_full = pd.DataFrame(all_receiver_candles)
                    df_full["time"] = pd.to_datetime(df_full["time"])
                    df_full = df_full.set_index("time").sort_index()
                    candle_index_map = {ts: idx for idx, ts in enumerate(df_full.index)}

                    config_updated = False

                    for target in targets:
                        matched_times = []
                        for candle_obj in structures:
                            is_match = False
                            if target == "contourmaker" and candle_obj.get("is_contour_maker") is True:
                                is_match = True
                            elif target == "directional_bias" and candle_obj.get("is_directional_bias") is True:
                                is_match = True
                            elif target == "next_bias" and candle_obj.get("is_next_bias_candle") is True:
                                is_match = True
                            elif target == candle_obj.get("target_label"):
                                is_match = True

                            if is_match and "time" in candle_obj:
                                ref_time_dt = pd.to_datetime(candle_obj["time"])
                                if ref_time_dt in df_full.index:
                                    matched_times.append(ref_time_dt)

                        matched_times_sorted = sorted(set(matched_times))
                        used_suffixes = {}

                        for signal_time in matched_times_sorted:
                            c_idx_start = candle_index_map[signal_time]
                            df_chart = df_full.loc[signal_time:].iloc[:bars]

                            if len(df_chart) < 5: continue

                            # 3. CONSTRUCT UNIQUE KEY NAME
                            base_name = f"{receiver_tf}_{base_output_name}_{target}_{sender_tf}_{c_idx_start}"
                            suffix = ""
                            if base_name in used_suffixes:
                                used_suffixes[base_name] += 1
                                suffix = f"_{chr(96 + used_suffixes[base_name])}"
                            else:
                                used_suffixes[base_name] = 0
                            
                            final_key_name = base_name + suffix

                            # 4. PLOT IMAGE (PNG)
                            scatter_data = pd.Series([float('nan')] * len(df_chart), index=df_chart.index)
                            scatter_data.iloc[0] = df_chart.iloc[0]["high"] * 1.001
                            addplots = [mpf.make_addplot(scatter_data, type='scatter', markersize=300, marker='o', color='yellow', alpha=0.9)]

                            fig, _ = mpf.plot(df_chart, type='candle', style=custom_style, addplot=addplots, 
                                              returnfig=True, figsize=(28, 10), tight_layout=False)
                            fig.suptitle(f"{sym} ({receiver_tf}) | Key: {final_key_name}", fontsize=16, fontweight='bold', y=0.95)
                            
                            os.makedirs(dev_output_dir, exist_ok=True)
                            fig.savefig(os.path.join(dev_output_dir, f"{final_key_name}.png"), bbox_inches="tight", dpi=120)
                            plt.close(fig)

                            # 5. BUILD THE LIST OF CANDLES (this will be the direct value)
                            forward_candles = []

                            for _, r in df_chart.iterrows():
                                current_idx = candle_index_map[r.name]
                                
                                candle_obj = {
                                    "time": r.name.strftime('%Y-%m-%d %H:%M:%S'), 
                                    "open": float(r["open"]), 
                                    "high": float(r["high"]), 
                                    "low": float(r["low"]), 
                                    "close": float(r["close"]), 
                                    "timeframe": receiver_tf,
                                    "candle_number": current_idx
                                }
                                forward_candles.append(candle_obj)

                            # 6. SAVE THE LIST DIRECTLY UNDER THE KEY
                            local_config[final_key_name] = forward_candles

                            config_updated = True
                            total_marked_all += 1
                            log(f"tf Communication {final_key_name} Processed")

                    # 7. SAVE UPDATED CONFIG.JSON (once per symbol/timeframe pair)
                    if config_updated:
                        with open(config_json_path, 'w', encoding='utf-8') as f:
                            json.dump(local_config, f, indent=4)
                        log(f"Successfully updated config.json for {sym}/{sender_tf}")

                except Exception as e:
                    log(f"FATAL ERROR {sym} ({sender_tf}→{receiver_tf}): {str(e)}", "ERROR")

    return f"Done. Total entries added to config files: {total_marked_all}"

def receiver_comm_higher_highs_lower_lows(broker_name):
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
    processed_keys_all = 0

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
        hlll_cfg = define_candles.get(source_config_name, {})
        if not hlll_cfg: 
            continue

        neighbor_left = hlll_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = hlll_cfg.get("NEIGHBOR_RIGHT", 5)
        source_filename = hlll_cfg.get("filename", "highers.json")
        bars = hlll_cfg.get("BARS", 101)
        direction = hlll_cfg.get("read_candles_from", "new_old")

        label_cfg = hlll_cfg.get("label", {})
        hh_text = label_cfg.get("higherhighs_text", "HH")
        ll_text = label_cfg.get("lowerlows_text", "ll")
        cm_text = label_cfg.get("contourmaker_text", "m")

        label_at = label_cfg.get("label_at", {})
        hh_pos = label_at.get("higher_highs", "high").lower()
        ll_pos = label_at.get("lower_lows", "low").lower()

        color_map = {"green": (0, 255, 0), "red": (255, 0, 0), "blue": (0, 0, 255)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red").lower(), (255, 0, 0))
        ll_col = color_map.get(label_at.get("lower_lows_color", "green").lower(), (0, 255, 0))

        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        ll_obj, ll_dbl = resolve_marker(label_at.get("lower_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        ll_cm_obj, ll_cm_dbl = resolve_marker(label_at.get("lower_lows_contourmaker_marker", ""))

        sender_tfs_raw = comm_cfg.get("timeframe_sender", "")
        receiver_tfs_raw = comm_cfg.get("timeframe_receiver", "")
        
        sender_tfs = [s.strip().lower() for s in sender_tfs_raw.split(",") if s.strip()]
        receiver_tfs = [r.strip().lower() for r in receiver_tfs_raw.split(",") if r.strip()]

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for s_tf, r_tf in zip(sender_tfs, receiver_tfs):
                # 1. FIND THE CONFIG.JSON
                dev_output_dir = os.path.join(
                    os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)),
                    sym, s_tf
                )
                config_path = os.path.join(dev_output_dir, "config.json")
                
                if not os.path.exists(config_path):
                    continue

                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    config_updated = False
                    # Pattern example: 5m_output_contourmaker_15m_45
                    search_pattern = f"{r_tf}_{source_filename.replace('.json','')}"

                    for key in list(config_data.keys()):
                        if not key.startswith(search_pattern):
                            continue
                        
                        # Now the value is directly a list of candles
                        raw_candles = config_data.get(key)
                        if not isinstance(raw_candles, list):
                            continue

                        # Sort ascending by candle_number (oldest → newest)
                        data = sorted(raw_candles, key=lambda x: x.get('candle_number', 0))
                        
                        png_path = os.path.join(dev_output_dir, f"{key}.png")
                        if not os.path.exists(png_path):
                            continue

                        img = cv2.imread(png_path)
                        if img is None: 
                            continue

                        # 2. IMAGE PROCESSING (Contour detection)
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | \
                               cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours: 
                            continue
                        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

                        min_len = min(len(data), len(contours))
                        data = data[:min_len]
                        contours = contours[:min_len]

                        # 3. RECORD COORDINATES
                        for i in range(min_len):
                            x, y, w, h = cv2.boundingRect(contours[i])
                            data[i].update({
                                "candle_x": int(x + w // 2),
                                "candle_y": int(y),
                                "candle_width": int(w),
                                "candle_height": int(h),
                                "candle_left": int(x),
                                "candle_right": int(x + w),
                                "candle_top": int(y),
                                "candle_bottom": int(y + h)
                            })

                        # 4. SWING DETECTION
                        modified = False
                        n = len(data)
                        for i in range(neighbor_left, n - neighbor_right):
                            curr_h, curr_l = data[i]['high'], data[i]['low']
                            
                            l_h = [d['high'] for d in data[i-neighbor_left : i]     if 'high' in d]
                            r_h = [d['high'] for d in data[i+1 : i+1+neighbor_right] if 'high' in d]
                            l_l = [d['low']  for d in data[i-neighbor_left : i]     if 'low'  in d]
                            r_l = [d['low']  for d in data[i+1 : i+1+neighbor_right] if 'low'  in d]

                            is_hh = curr_h > max(l_h) and curr_h > max(r_h) if l_h and r_h else False
                            is_ll = curr_l < min(l_l) and curr_l < min(r_l) if l_l and r_l else False

                            if not (is_hh or is_ll): 
                                continue

                            is_bull = is_ll
                            active_color = ll_col if is_bull else hh_col
                            label_text = ll_text if is_bull else hh_text
                            obj_type = ll_obj if is_bull else hh_obj
                            dbl_arrow = ll_dbl if is_bull else hh_dbl
                            pos = ll_pos if is_bull else hh_pos

                            # Draw on image
                            label_objects_and_text(
                                img, 
                                data[i]["candle_x"], 
                                data[i]["candle_y"], 
                                data[i]["candle_height"],
                                fvg_swing_type=data[i]['candle_number'],
                                custom_text=label_text, 
                                object_type=obj_type,
                                is_bullish_arrow=is_bull, 
                                is_marked=True,
                                double_arrow=dbl_arrow, 
                                arrow_color=active_color,
                                label_position=pos
                            )

                            data[i].update({
                                "swing_type": "lower_low" if is_bull else "higher_high",
                                "active_color": [int(c) for c in active_color],  # make serializable
                                "is_swing": True
                            })
                            
                            # Contour Maker logic
                            m_idx = i + neighbor_right
                            if m_idx < n:
                                data[m_idx]["is_contour_maker"] = True
                                label_objects_and_text(
                                    img, 
                                    data[m_idx]["candle_x"], 
                                    data[m_idx]["candle_y"], 
                                    data[m_idx]["candle_height"],
                                    custom_text=cm_text, 
                                    object_type=(ll_cm_obj if is_bull else hh_cm_obj),
                                    is_bullish_arrow=is_bull, 
                                    is_marked=True,
                                    double_arrow=(ll_cm_dbl if is_bull else hh_cm_dbl),
                                    arrow_color=active_color, 
                                    label_position=pos
                                )

                            modified = True
                            total_marked_all += 1

                        if modified:
                            cv2.imwrite(png_path, img)
                            # Write the list directly back under the key
                            config_data[key] = data
                            config_updated = True
                            processed_keys_all += 1
                            log(f"receiver {key} swing processed")

                    # 5. SAVE UPDATED CONFIG.JSON
                    if config_updated:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=4)

                except Exception as e:
                    log(f"Error in {sym} / {s_tf}: {e}", "ERROR")

    return f"Done. Updated {processed_keys_all} keys in config files with {total_marked_all} swings."

def liquidity_candles(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    def resolve_marker(raw):
        if not raw:
            return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
        if raw in ["doublearrow", "doublearrows"]: return "arrow", True
        if raw in ["rightarrow", "right_arrow"]: return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]: return "leftarrow", False
        if "dot" in raw: return "dot", False
        return raw, False

    log(f"--- STARTING SPACE-BASED LIQUIDITY ANALYSIS: {broker_name} ---")

    dev_dict = load_developers_dictionary() 
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"Error: Broker {broker_name} not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data:
        return "Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    liq_root = define_candles.get("liquidity_candle", {})
    
    total_liq_found = 0

    for apprehend_key, liq_cfg in liq_root.items():
        if not apprehend_key.startswith("apprehend_"): 
            continue
        
        source_def_name = apprehend_key.replace("apprehend_", "")
        source_def = define_candles.get(source_def_name, {})
        if not source_def: 
            continue

        raw_filename = source_def.get("filename", "")
        target_file_filter = raw_filename.replace(".json", "").lower()
        primary_png_name = raw_filename.replace(".json", ".png")

        apprentice_section = liq_cfg.get("liquidity_apprentice_candle", {})
        apprentice_cfg = apprentice_section.get("swing_types", {})
        
        is_bullish = any("higher" in k for k in apprentice_cfg.keys())
        swing_prefix = "higher" if is_bullish else "lower"

        sweeper_section = liq_cfg.get("liquidity_sweeper_candle", {})
        liq_label_at = sweeper_section.get("label_at", {})

        markers = {
            "liq_hh": resolve_marker(liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_marker")),
            "liq_ll": resolve_marker(liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_marker")),
            "liq_hh_txt": liq_label_at.get(f"{swing_prefix}_high_liquidity_candle_text", ""),
            "liq_ll_txt": liq_label_at.get(f"{swing_prefix}_low_liquidity_candle_text", ""),
            "app_hh": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_high_marker")),
            "app_ll": resolve_marker(apprentice_cfg.get("label_at", {}).get(f"swing_type_{swing_prefix}_low_marker"))
        }

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): 
                continue

            for tf in os.listdir(sym_p):
                tf_p = os.path.join(sym_p, tf)
                if not os.path.isdir(tf_p): 
                    continue
                
                dev_output_dir = os.path.join(
                    os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)), 
                    sym, 
                    tf
                )
                config_path = os.path.join(dev_output_dir, "config.json")

                if not os.path.exists(config_path): 
                    continue

                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    config_modified = False
                    
                    for file_key in list(config_data.keys()):
                        if not (file_key.lower() == source_def_name.lower() or 
                                target_file_filter in file_key.lower()):
                            continue

                        entry = config_data[file_key]

                        # ────────────────────────────────────────────────
                        # NEW: Only accept direct list of candles
                        # ────────────────────────────────────────────────
                        if not isinstance(entry, list):
                            log(f"Skipping {file_key} — value is not a list (expected direct array of candles)", "WARN")
                            continue

                        candles = entry

                        if len(candles) < 2:
                            log(f"Skipping {file_key} — fewer than 2 candles", "WARN")
                            continue

                        # Optional: basic validation that items look like candles
                        if not all(isinstance(c, dict) for c in candles):
                            log(f"Skipping {file_key} — not all items are dictionaries", "WARN")
                            continue
                        # ────────────────────────────────────────────────

                        png_path = os.path.join(dev_output_dir, f"{file_key}.png")
                        if not os.path.exists(png_path):
                            png_path = os.path.join(dev_output_dir, primary_png_name)

                        img = cv2.imread(png_path)
                        key_modified = False

                        for i in range(len(candles) - 1):
                            base_c = candles[i]
                            next_c = candles[i + 1]
                            
                            b_h = base_c.get("high")
                            b_l = base_c.get("low")
                            n_h = next_c.get("high")
                            n_l = next_c.get("low")
                            
                            if None in [b_h, b_l, n_h, n_l]: 
                                continue

                            target_side = None
                            ref_price = None

                            if n_h >= b_h and n_l <= b_l:
                                continue 
                            
                            elif n_h > b_h and n_l > b_l:
                                target_side = "low"
                                ref_price = b_l
                                
                            elif n_l < b_l and n_h < b_h:
                                target_side = "high"
                                ref_price = b_h
                            
                            else:
                                continue

                            for j in range(i + 2, len(candles)):
                                sweeper_c = candles[j]
                                swept = False

                                if target_side == "low" and sweeper_c.get("low", 999999) <= ref_price:
                                    swept = True
                                    pos = "low"
                                    m_key = "liq_ll"
                                    a_key = "app_ll"
                                elif target_side == "high" and sweeper_c.get("high", 0) >= ref_price:
                                    swept = True
                                    pos = "high"
                                    m_key = "liq_hh"
                                    a_key = "app_hh"

                                if swept:
                                    log(f"[SPACE SWEEP] {sym} {tf}: {pos} of candle {base_c.get('candle_number')} swept by {sweeper_c.get('candle_number')}")
                                    
                                    obj, dbl = markers[m_key]
                                    app_obj, app_dbl = markers[a_key]
                                    txt = markers.get(f"{m_key}_txt", "")

                                    sweeper_c.update({
                                        "is_liquidity_sweep": True, 
                                        "liquidity_price": ref_price
                                    })
                                    base_c.update({
                                        "swept_by_liquidity": True, 
                                        "swept_by_candle_number": sweeper_c.get("candle_number")
                                    })

                                    if img is not None:
                                        label_objects_and_text(img, 
                                            int(sweeper_c.get("candle_x", 0)), 
                                            int(sweeper_c.get("candle_y", 0)), 
                                            int(sweeper_c.get("candle_height", 0)),
                                            custom_text=txt, 
                                            object_type=obj, 
                                            is_bullish_arrow=(target_side == "low"),
                                            is_marked=True, 
                                            double_arrow=dbl, 
                                            arrow_color=(0, 255, 255), 
                                            label_position=pos)
                                        
                                        label_objects_and_text(img, 
                                            int(base_c.get("candle_x", 0)), 
                                            int(base_c.get("candle_y", 0)), 
                                            int(base_c.get("candle_height", 0)),
                                            custom_text="", 
                                            object_type=app_obj, 
                                            is_bullish_arrow=(target_side == "low"),
                                            is_marked=True, 
                                            double_arrow=app_dbl, 
                                            arrow_color=(255, 165, 0), 
                                            label_position=pos)

                                    key_modified = True
                                    config_modified = True
                                    total_liq_found += 1
                                    break 

                        if key_modified and img is not None:
                            cv2.imwrite(png_path, img)

                    if config_modified:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=4)

                except Exception as e:
                    log(f"Error processing {sym} ({tf}): {e}", "ERROR")

    log(f"--- LIQUIDITY COMPLETE --- Total Space Sweeps: {total_liq_found}")
    return f"Completed: {total_liq_found} sweeps."

def entry_point_of_interest_condition(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ─── Helper functions ────────────────────────────────────────────────
    def normalize_swing(s):
        if not s: return ""
        return str(s).lower().replace("_", "").replace(" ", "")

    def normalize_fvg_type(s):
        if not s: return ""
        return str(s).lower().replace("_", "").replace(" ", "")
    
    def normalize_fvg_family(s):
        """Normalize any FVG family type"""
        if not s: return ""
        norm = str(s).lower().replace("_", "").replace(" ", "")
        if "fvgc1" in norm:
            return "fvg_c1"
        elif "fvgc3" in norm:
            return "fvg_c3"
        elif "fvg" in norm and "c1" not in norm and "c3" not in norm and "swing" not in norm:
            return "fvg"
        return norm

    def is_valid_swing_type(swing):
        if not swing: return False
        norm = normalize_swing(swing)
        return norm in ["higherhigh", "lowerhigh", "higherlow", "lowerlow"]

    def is_valid_fvg_c1(fvg_flag):
        if fvg_flag is None: return False
        return bool(fvg_flag) if isinstance(fvg_flag, bool) else str(fvg_flag).lower() in ["true", "1", "yes"]
    
    def is_valid_fvg_c2(fvg_flag):
        if fvg_flag is None: return False
        return bool(fvg_flag) if isinstance(fvg_flag, bool) else str(fvg_flag).lower() in ["true", "1", "yes"]
    
    def is_valid_fvg_c3(fvg_flag):
        if fvg_flag is None: return False
        return bool(fvg_flag) if isinstance(fvg_flag, bool) else str(fvg_flag).lower() in ["true", "1", "yes"]

    def get_fvg_base_type(candle, fvg_type):
        """Get FVG base type (support/resistance) for specific FVG type"""
        if fvg_type == "fvg_c1":
            base = candle.get("fvg_c1_base")
        elif fvg_type == "fvg_c3":
            base = candle.get("fvg_c3_base")
        else:  # fvg
            base = candle.get("fvg_base")
        
        if not base:
            return None
        return str(base).lower().strip()

    def get_fvg_number(candle, fvg_type):
        """Get the fvg_number for a specific FVG type"""
        if not candle: return None
        
        if fvg_type == "fvg_c1":
            return candle.get("c1_for_fvg_number")
        elif fvg_type == "fvg_c3":
            return candle.get("c3_for_fvg_number")
        elif fvg_type == "fvg":
            return candle.get("fvg_number")
        
        return None

    def find_fvg_family_by_number(candles, target_fvg_number, target_fvg_type):
        """Find all FVG family members with the same fvg_number"""
        if target_fvg_number is None: return []
        
        family_members = []
        
        for candle in candles:
            # Check fvg_c1
            if target_fvg_type != "fvg_c1":
                c1_number = candle.get("c1_for_fvg_number")
                if c1_number == target_fvg_number and is_valid_fvg_c1(candle.get("fvg_c1")):
                    family_members.append({
                        "candle": candle,
                        "type": "fvg_c1",
                        "number": c1_number
                    })
            
            # Check fvg_c3
            if target_fvg_type != "fvg_c3":
                c3_number = candle.get("c3_for_fvg_number")
                if c3_number == target_fvg_number and is_valid_fvg_c3(candle.get("fvg_c3")):
                    family_members.append({
                        "candle": candle,
                        "type": "fvg_c3",
                        "number": c3_number
                    })
            
            # Check regular fvg
            if target_fvg_type != "fvg":
                fvg_number = candle.get("fvg_number")
                if fvg_number == target_fvg_number and is_valid_fvg_c2(candle.get("is_fvg")):
                    family_members.append({
                        "candle": candle,
                        "type": "fvg",
                        "number": fvg_number
                    })
        
        return family_members

    def get_opposite_swing_type(swing_type):
        if not swing_type: return None
        norm = normalize_swing(swing_type)
        if "high" in norm: return "lower_low"
        if "low" in norm: return "higher_high"
        return None
    
    def get_opposite_fvg_type(fvg_type):
        """Get opposite FVG type (support <-> resistance)"""
        if not fvg_type: return None
        return None

    def is_swing_match(candle, target_swing_name):
        """Check if candle matches target swing type"""
        if not candle.get("is_swing"): return False
        candle_swing = candle.get("swing_type")
        if not target_swing_name or not candle_swing: return False
        
        # Handle fvg_swing as a special case - it matches any swing_type
        if normalize_swing(target_swing_name) == "fvgswing":
            return True
        
        return target_swing_name.replace("_", "") in normalize_swing(candle_swing)
    
    def is_fvg_match(candle, target_fvg_name, reference_candle=None):
        """Check if candle matches target FVG type, with family linking support"""
        if not target_fvg_name: 
            return False
        
        norm_target = normalize_fvg_family(target_fvg_name)
        
        # Special handling for fvg_swing - it's NOT a FVG type
        if norm_target == "fvgswing":
            return False
        
        # Direct match check
        if norm_target == "fvg_c1":
            if is_valid_fvg_c1(candle.get("fvg_c1")):
                return True
        elif norm_target == "fvg_c3":
            if is_valid_fvg_c3(candle.get("fvg_c3")):
                return True
        elif norm_target == "fvg":
            if is_valid_fvg_c2(candle.get("is_fvg")):
                return True
        
        # Family linking check - if we have a reference candle with fvg_number
        if reference_candle:
            # Get fvg_number from reference candle based on its type
            ref_fvg_type = None
            if is_valid_fvg_c1(reference_candle.get("fvg_c1")):
                ref_fvg_type = "fvg_c1"
            elif is_valid_fvg_c3(reference_candle.get("fvg_c3")):
                ref_fvg_type = "fvg_c3"
            elif is_valid_fvg_c2(reference_candle.get("is_fvg")):
                ref_fvg_type = "fvg"
            
            if ref_fvg_type:
                ref_fvg_number = get_fvg_number(reference_candle, ref_fvg_type)
                if ref_fvg_number is not None:
                    # Check if current candle has matching fvg_number for target type
                    target_fvg_number = get_fvg_number(candle, norm_target)
                    if target_fvg_number == ref_fvg_number:
                        return True
        
        return False

    def is_fvg_family_candle(candle, target_name):
        """Check if candle is any FVG family type (fvg, fvg_c1, fvg_c3)"""
        if not target_name: return False
        
        norm_target = normalize_fvg_family(target_name)
        
        # fvg_swing is NOT a FVG family type
        if norm_target == "fvgswing":
            return False
        
        if is_fvg_match(candle, norm_target):
            return True
        
        if norm_target == "fvg":
            return (is_valid_fvg_c2(candle.get("is_fvg")) or 
                    is_valid_fvg_c1(candle.get("fvg_c1")) or 
                    is_valid_fvg_c3(candle.get("fvg_c3")))
        
        return False

    def is_fvg_swing_candle(candle):
        """Check if candle is a fvg_swing (regular swing candle)"""
        return is_valid_swing_type(candle.get("swing_type"))

    def get_allowed_acting_sweeper_types(swept_swing):
        if not swept_swing: return []
        norm = normalize_swing(swept_swing)
        if "higherhigh" in norm: return ["higher_high"]
        if "lowerhigh" in norm: return ["higher_high", "lower_high"]
        if "lowerlow" in norm: return ["lower_low"]
        if "higherlow" in norm: return ["higher_low", "lower_low"]
        return []
    
    def get_allowed_acting_sweeper_fvg_types(swept_fvg_type):
        """Get allowed acting sweeper types for FVG"""
        if not swept_fvg_type: return []
        norm = normalize_fvg_family(swept_fvg_type)
        return []

    def check_beyond_condition(candle, reference, new_to_find_type):
        norm = normalize_swing(new_to_find_type)
        if "high" in norm:
            ref_val = reference.get("high") if "high" in normalize_swing(reference.get("swing_type")) else reference.get("low")
            return candle.get("high", 0) > ref_val
        if "low" in norm:
            ref_val = reference.get("low") if "low" in normalize_swing(reference.get("swing_type")) else reference.get("high")
            return candle.get("low", 999999) < ref_val
        return False
    
    def check_beyond_condition_fvg(candle, reference, new_to_find_type, fvg_base):
        """Check beyond condition for FVG family"""
        if not fvg_base: return False
        
        norm_base = str(fvg_base).lower()
        if "resistance" in norm_base:
            ref_val = reference.get("high", 0)
            return candle.get("high", 0) > ref_val
        elif "support" in norm_base:
            ref_val = reference.get("low", 999999)
            return candle.get("low", 999999) < ref_val
        return False
    
    def check_beyond_fvg_condition(candle, reference, fvg_type):
        """Check beyond condition for specific FVG type"""
        # Get reference FVG base type
        ref_fvg_base = None
        if is_fvg_match(reference, "fvg"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg")
        elif is_fvg_match(reference, "fvg_c1"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c1")
        elif is_fvg_match(reference, "fvg_c3"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c3")
        
        if not ref_fvg_base: return False
        
        # Handle fvg_swing case
        if fvg_type == "fvg_swing":
            # For fvg_swing, check based on swing_type
            candle_swing_type = normalize_swing(candle.get("swing_type", ""))
            if not candle_swing_type: return False
            
            if "resistance" in ref_fvg_base:
                if "high" in candle_swing_type:
                    return candle.get("high", 0) > reference.get("high", 0)
            elif "support" in ref_fvg_base:
                if "low" in candle_swing_type:
                    return candle.get("low", 999999) < reference.get("low", 999999)
            return False
        
        # Handle regular FVG types
        candle_fvg_base = None
        if is_fvg_match(candle, "fvg"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg")
        elif is_fvg_match(candle, "fvg_c1"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg_c1")
        elif is_fvg_match(candle, "fvg_c3"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg_c3")
        
        if not candle_fvg_base: return False
        
        if "resistance" in ref_fvg_base and "resistance" in candle_fvg_base:
            return candle.get("high", 0) > reference.get("high", 0)
        elif "support" in ref_fvg_base and "support" in candle_fvg_base:
            return candle.get("low", 999999) < reference.get("low", 999999)
        elif "resistance" in ref_fvg_base and "support" in candle_fvg_base:
            return candle.get("low", 999999) < reference.get("high", 999999)
        elif "support" in ref_fvg_base and "resistance" in candle_fvg_base:
            return candle.get("high", 0) > reference.get("low", 999999)
        
        return False

    def check_behind_condition(candle, reference, new_to_find_type):
        norm = normalize_swing(new_to_find_type)
        ref_type = normalize_swing(reference.get("swing_type", ""))
        if "high" in norm:
            if "low" in ref_type:
                return candle.get("high", 0) > reference.get("low", 0)
            return candle.get("high", 0) < reference.get("high", 0)
        if "low" in norm:
            if "high" in ref_type:
                return candle.get("low", 999999) < reference.get("high", 999999)
            return candle.get("low", 999999) > reference.get("low", 999999)
        return False
    
    def check_behind_condition_fvg(candle, reference, new_to_find_type, fvg_base):
        """Check behind condition for FVG family"""
        if not fvg_base: return False
        
        norm_base = str(fvg_base).lower()
        ref_fvg_base = None
        
        if is_fvg_match(reference, "fvg"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg")
        elif is_fvg_match(reference, "fvg_c1"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c1")
        elif is_fvg_match(reference, "fvg_c3"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c3")
        
        if "resistance" in norm_base:
            if ref_fvg_base and "support" in ref_fvg_base:
                return candle.get("high", 0) > reference.get("low", 0)
            return candle.get("high", 0) < reference.get("high", 0)
        elif "support" in norm_base:
            if ref_fvg_base and "resistance" in ref_fvg_base:
                return candle.get("low", 999999) < reference.get("high", 999999)
            return candle.get("low", 999999) > reference.get("low", 999999)
        return False
    
    def check_behind_fvg_condition(candle, reference, fvg_type):
        """Check behind condition for specific FVG type"""
        # Get reference FVG base type
        ref_fvg_base = None
        if is_fvg_match(reference, "fvg"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg")
        elif is_fvg_match(reference, "fvg_c1"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c1")
        elif is_fvg_match(reference, "fvg_c3"):
            ref_fvg_base = get_fvg_base_type(reference, "fvg_c3")
        
        if not ref_fvg_base: return False
        
        # Handle fvg_swing case
        if fvg_type == "fvg_swing":
            # For fvg_swing, check based on swing_type
            candle_swing_type = normalize_swing(candle.get("swing_type", ""))
            if not candle_swing_type: return False
            
            if "resistance" in ref_fvg_base:
                if "high" in candle_swing_type:
                    return candle.get("high", 0) < reference.get("high", 0)
            elif "support" in ref_fvg_base:
                if "low" in candle_swing_type:
                    return candle.get("low", 999999) > reference.get("low", 999999)
            return False
        
        # Handle regular FVG types
        candle_fvg_base = None
        if is_fvg_match(candle, "fvg"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg")
        elif is_fvg_match(candle, "fvg_c1"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg_c1")
        elif is_fvg_match(candle, "fvg_c3"):
            candle_fvg_base = get_fvg_base_type(candle, "fvg_c3")
        
        if not candle_fvg_base: return False
        
        if "resistance" in ref_fvg_base and "resistance" in candle_fvg_base:
            return candle.get("high", 0) < reference.get("high", 0)
        elif "support" in ref_fvg_base and "support" in candle_fvg_base:
            return candle.get("low", 999999) > reference.get("low", 999999)
        elif "resistance" in ref_fvg_base and "support" in candle_fvg_base:
            return candle.get("low", 999999) > reference.get("high", 999999)
        elif "support" in ref_fvg_base and "resistance" in candle_fvg_base:
            return candle.get("high", 0) < reference.get("low", 999999)
        
        return False

    def find_liquidation_for_candle(target_candle, candle_list):
        """Find the liquidation candle for a target candle"""
        target_num = target_candle.get("candle_number")
        if target_num is None: return None
        swing_t = normalize_swing(target_candle.get("swing_type", ""))
        for search_c in candle_list:
            search_num = search_c.get("candle_number")
            if search_num is None or search_num <= target_num: continue
            if "high" in swing_t:
                if search_c.get("high", 0) >= target_candle.get("high", 0): return search_c
            elif "low" in swing_t:
                if search_c.get("low", 999999) <= target_candle.get("low", 999999): return search_c
        return None
    
    def find_liquidation_for_fvg(target_candle, candle_list):
        """Find liquidation for FVG candle using FVG-specific liquidation fields"""
        target_num = target_candle.get("candle_number")
        if target_num is None: return None
        
        # Determine FVG type
        fvg_type = None
        if is_fvg_match(target_candle, "fvg_c1"):
            fvg_type = "fvg_c1"
        elif is_fvg_match(target_candle, "fvg_c3"):
            fvg_type = "fvg_c3"
        elif is_fvg_match(target_candle, "fvg"):
            fvg_type = "fvg_c2"  # Using fvg_c2 for regular fvg
        
        if not fvg_type: return None
        
        # Get FVG base type
        fvg_base = get_fvg_base_type(target_candle, fvg_type.replace("_c2", ""))
        if not fvg_base: return None
        
        # Find liquidation based on FVG type and base
        for search_c in candle_list:
            search_num = search_c.get("candle_number")
            if search_num is None or search_num <= target_num: continue
            
            if "resistance" in fvg_base:
                # Check for high liquidation
                liq_field = f"{fvg_type}_high_liquidated_by_candle_number"
                liq_num = target_candle.get(liq_field)
                if liq_num and search_num == liq_num:
                    return search_c
            elif "support" in fvg_base:
                # Check for low liquidation
                liq_field = f"{fvg_type}_low_liquidated_by_candle_number"
                liq_num = target_candle.get(liq_field)
                if liq_num and search_num == liq_num:
                    return search_c
        
        return None

    def get_price_from_cfg(candle, cfg_block):
        if not candle: return None
        
        # Check if it's a fvg_swing (regular swing candle)
        if is_fvg_swing_candle(candle):
            stype = normalize_swing(candle.get("swing_type", ""))
            if "high" in stype:
                field = cfg_block.get("swing_higher_high_or_lower_high", "high_price")
                return candle.get("high") if field == "high_price" else candle.get("low")
            elif "low" in stype:
                field = cfg_block.get("swing_lower_low_or_higher_low", "low_price")
                return candle.get("high") if field == "high_price" else candle.get("low")
        
        # Check if it's a FVG family candle
        fvg_type = None
        if is_fvg_match(candle, "fvg_c1"):
            fvg_type = "fvg_c1"
        elif is_fvg_match(candle, "fvg_c3"):
            fvg_type = "fvg_c3"
        elif is_fvg_match(candle, "fvg"):
            fvg_type = "fvg"
        
        if fvg_type:
            fvg_base = get_fvg_base_type(candle, fvg_type)
            if fvg_base:
                if "resistance" in fvg_base:
                    field = cfg_block.get("resistance_fvg", "high_price")
                    return candle.get("high") if field == "high_price" else candle.get("low")
                elif "support" in fvg_base:
                    field = cfg_block.get("support_fvg", "low_price")
                    return candle.get("high") if field == "high_price" else candle.get("low")
        
        return None

    def draw_line_logic(img, x_start, x_end, y, tool_type, color=(0, 0, 0), thickness=2):
        """Helper to handle solid and dashed lines"""
        if tool_type == "horizontal_line":
            cv2.line(img, (int(x_start), int(y)), (int(x_end), int(y)), color, thickness)
        elif tool_type == "dashed_horizontal_line":
            dist = int(x_end - x_start)
            dash_len = 10
            for j in range(0, dist, dash_len * 2):
                end_segment = min(int(x_start + j + dash_len), int(x_end))
                cv2.line(img, (int(x_start + j), int(y)), (end_segment, int(y)), color, thickness)
    
    def filter_data_selection(items, start_search_with, data_selection_value, source_candles):
        """Filter items based on data_selection value (extreme, non_extreme, or all)"""
        if not items or len(items) <= 1:
            return items
            
        if data_selection_value == "all":
            return items
        
        swing_type_groups = {}
        
        for item in items:
            swept_candle_num = item.get("swept_candle_number")
            if swept_candle_num is None:
                continue
                
            swept_candle = next((c for c in source_candles if c.get("candle_number") == swept_candle_num), None)
            if not swept_candle:
                continue
                
            swing_type = swept_candle.get("swing_type")
            if not swing_type:
                continue
                
            normalized_swing = normalize_swing(swing_type)
            if normalized_swing not in swing_type_groups:
                swing_type_groups[normalized_swing] = []
            swing_type_groups[normalized_swing].append(item)
        
        filtered_items = []
        
        for swing_type, group_items in swing_type_groups.items():
            if len(group_items) <= 1:
                filtered_items.extend(group_items)
                continue
            
            items_with_prices = []
            for item in group_items:
                swept_candle_num = item.get("swept_candle_number")
                swept_candle = next((c for c in source_candles if c.get("candle_number") == swept_candle_num), None)
                if swept_candle:
                    items_with_prices.append({
                        "item": item,
                        "high": swept_candle.get("high", 0),
                        "low": swept_candle.get("low", 999999),
                        "swing_type": swept_candle.get("swing_type", "")
                    })
            
            if not items_with_prices:
                filtered_items.extend(group_items)
                continue
            
            if data_selection_value == "extreme":
                if "higherhigh" in swing_type or "lowerhigh" in swing_type:
                    best_item = max(items_with_prices, key=lambda x: x["high"])
                    filtered_items.append(best_item["item"])
                elif "lowerlow" in swing_type or "higherlow" in swing_type:
                    best_item = min(items_with_prices, key=lambda x: x["low"])
                    filtered_items.append(best_item["item"])
                else:
                    filtered_items.extend(group_items)
                    
            elif data_selection_value == "non_extreme":
                if "lowerlow" in swing_type or "higherlow" in swing_type:
                    best_item = max(items_with_prices, key=lambda x: x["low"])
                    filtered_items.append(best_item["item"])
                elif "higherhigh" in swing_type or "lowerhigh" in swing_type:
                    best_item = min(items_with_prices, key=lambda x: x["high"])
                    filtered_items.append(best_item["item"])
                else:
                    filtered_items.extend(group_items)
            else:
                filtered_items.extend(group_items)
        
        return filtered_items

    def extract_timeframe_from_key(key_id):
        """Extract timeframe from key ID (first timeframe in the key)"""
        import re
        timeframe_patterns = [
            r'\b1m\b', r'\b5m\b', r'\b15m\b', r'\b30m\b', 
            r'\b1h\b', r'\b2h\b', r'\b4h\b', r'\b1d\b',
            r'\b1w\b', r'\b1M\b'
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, key_id.lower())
            if match:
                return match.group(0)
        
        return None

    def check_intruder_violation(option_candle, last_swing_candle, subject_poi_candle, candles, if_intruder_value):
        """Check for intruder candles between option and last swing that violate subject synchronization"""
        if if_intruder_value != "terminate":
            return False
        
        if not option_candle or not last_swing_candle or not subject_poi_candle:
            return False
        
        option_num = option_candle.get("candle_number")
        last_swing_num = last_swing_candle.get("candle_number")
        
        if option_num is None or last_swing_num is None:
            return False
        
        start_idx = min(option_num, last_swing_num)
        end_idx = max(option_num, last_swing_num)
        
        # Check subject type (could be fvg_swing or FVG)
        is_subject_fvg_swing = is_fvg_swing_candle(subject_poi_candle)
        is_subject_fvg = is_fvg_family_candle(subject_poi_candle, "fvg")
        
        for candle in candles:
            c_num = candle.get("candle_number")
            if c_num is None or c_num <= start_idx or c_num >= end_idx:
                continue
            
            if is_subject_fvg_swing:
                # Check for matching swing_type
                if not candle.get("is_swing"):
                    continue
                
                candle_swing_type = normalize_swing(candle.get("swing_type", ""))
                subject_swing_type = normalize_swing(subject_poi_candle.get("swing_type", ""))
                
                if candle_swing_type != subject_swing_type:
                    continue
                
                if "high" in subject_swing_type:
                    if candle.get("high", 0) >= subject_poi_candle.get("high", 0):
                        return True
                elif "low" in subject_swing_type:
                    if candle.get("low", 999999) <= subject_poi_candle.get("low", 999999):
                        return True
            
            elif is_subject_fvg:
                # Check for matching FVG type or family member
                subject_fvg_type = None
                subject_fvg_number = None
                
                if is_fvg_match(subject_poi_candle, "fvg_c1"):
                    subject_fvg_type = "fvg_c1"
                    subject_fvg_number = get_fvg_number(subject_poi_candle, "fvg_c1")
                elif is_fvg_match(subject_poi_candle, "fvg_c3"):
                    subject_fvg_type = "fvg_c3"
                    subject_fvg_number = get_fvg_number(subject_poi_candle, "fvg_c3")
                elif is_fvg_match(subject_poi_candle, "fvg"):
                    subject_fvg_type = "fvg"
                    subject_fvg_number = get_fvg_number(subject_poi_candle, "fvg")
                
                if not subject_fvg_type:
                    continue
                
                # Check if candle matches the same FVG type OR is a family member
                candle_matches = False
                
                # Direct type match
                if is_fvg_match(candle, subject_fvg_type):
                    candle_matches = True
                # Family member match
                elif subject_fvg_number is not None:
                    # Check all possible FVG types for this candle
                    for check_type in ["fvg_c1", "fvg_c3", "fvg"]:
                        candle_fvg_number = get_fvg_number(candle, check_type)
                        if candle_fvg_number == subject_fvg_number and (
                            (check_type == "fvg_c1" and is_valid_fvg_c1(candle.get("fvg_c1"))) or
                            (check_type == "fvg_c3" and is_valid_fvg_c3(candle.get("fvg_c3"))) or
                            (check_type == "fvg" and is_valid_fvg_c2(candle.get("is_fvg")))
                        ):
                            candle_matches = True
                            break
                
                if not candle_matches:
                    continue
                
                # Check price violation
                subject_fvg_base = get_fvg_base_type(subject_poi_candle, subject_fvg_type)
                if not subject_fvg_base:
                    continue
                
                if "resistance" in subject_fvg_base:
                    if candle.get("high", 0) >= subject_poi_candle.get("high", 0):
                        return True
                elif "support" in subject_fvg_base:
                    if candle.get("low", 999999) <= subject_poi_candle.get("low", 999999):
                        return True
        
        return False

    def check_intruder_violation_between_candles(prev_candle, current_candle, condition_type, candles, if_intruder_value):
        """Check for intruder candles between two candles based on condition type"""
        if if_intruder_value != "terminate":
            return False
        
        if not prev_candle or not current_candle:
            return False
        
        prev_num = prev_candle.get("candle_number")
        current_num = current_candle.get("candle_number")
        
        if prev_num is None or current_num is None:
            return False
        
        # Determine the condition type
        is_behind_condition = "behind" in condition_type.lower()
        is_beyond_condition = "beyond" in condition_type.lower()
        
        if not is_behind_condition:
            # Only check intruder for "behind" conditions
            return False
        
        # Determine swing types
        prev_is_swing = is_valid_swing_type(prev_candle.get("swing_type"))
        current_is_swing = is_valid_swing_type(current_candle.get("swing_type"))
        
        # Determine what to check based on swing types
        check_high = False
        check_low = False
        
        if prev_is_swing:
            prev_swing_type = normalize_swing(prev_candle.get("swing_type", ""))
            check_high = "high" in prev_swing_type
            check_low = "low" in prev_swing_type
        else:
            # If not a swing, try to determine from other attributes
            # Check if it's a FVG candle
            if is_fvg_match(prev_candle, "fvg"):
                fvg_base = get_fvg_base_type(prev_candle, "fvg")
                if fvg_base and "resistance" in fvg_base:
                    check_high = True
                elif fvg_base and "support" in fvg_base:
                    check_low = True
            elif is_fvg_match(prev_candle, "fvg_c1"):
                fvg_base = get_fvg_base_type(prev_candle, "fvg_c1")
                if fvg_base and "resistance" in fvg_base:
                    check_high = True
                elif fvg_base and "support" in fvg_base:
                    check_low = True
            elif is_fvg_match(prev_candle, "fvg_c3"):
                fvg_base = get_fvg_base_type(prev_candle, "fvg_c3")
                if fvg_base and "resistance" in fvg_base:
                    check_high = True
                elif fvg_base and "support" in fvg_base:
                    check_low = True
        
        if not check_high and not check_low:
            return False
        
        # Check all candles between prev_candle and current_candle
        for candle in candles:
            c_num = candle.get("candle_number")
            if c_num is None or c_num <= prev_num or c_num >= current_num:
                continue
            
            # Check if this candle violates the "behind" condition
            if check_high:
                # For high swings/FVGs, all intervening highs must be lower
                if candle.get("high", 0) >= prev_candle.get("high", 0):
                    return True
            elif check_low:
                # For low swings/FVGs, all intervening lows must be higher
                if candle.get("low", 999999) <= prev_candle.get("low", 999999):
                    return True
        
        return False

    def get_fvg_liquidation_candle(fvg_candle, candles):
        """Get liquidation candle for FVG using FVG-specific liquidation fields"""
        if not fvg_candle: return None
        
        # Determine FVG type
        fvg_type = None
        if is_fvg_match(fvg_candle, "fvg_c1"):
            fvg_type = "fvg_c1"
        elif is_fvg_match(fvg_candle, "fvg_c3"):
            fvg_type = "fvg_c3"
        elif is_fvg_match(fvg_candle, "fvg"):
            fvg_type = "fvg_c2"
        
        if not fvg_type: return None
        
        # Get FVG base type
        fvg_base = get_fvg_base_type(fvg_candle, fvg_type.replace("_c2", ""))
        if not fvg_base: return None
        
        # Find liquidation based on FVG type and base
        if "resistance" in fvg_base:
            liq_field = f"{fvg_type}_high_liquidated_by_candle_number"
            liq_num = fvg_candle.get(liq_field)
        elif "support" in fvg_base:
            liq_field = f"{fvg_type}_low_liquidated_by_candle_number"
            liq_num = fvg_candle.get(liq_field)
        else:
            return None
        
        if not liq_num: return None
        
        # Find the liquidation candle
        for candle in candles:
            if candle.get("candle_number") == liq_num:
                return candle
        
        return None

    def check_fvg_mitigation(fvg_candle, candles):
        """Check if FVG candle is mitigated (liquidated)"""
        liq_candle = get_fvg_liquidation_candle(fvg_candle, candles)
        return liq_candle is not None

    def check_fvg_pending_entry(fvg_candle):
        """Check if FVG candle has pending entry level (not liquidated)"""
        # Determine FVG type
        fvg_type = None
        if is_fvg_match(fvg_candle, "fvg_c1"):
            fvg_type = "fvg_c1"
        elif is_fvg_match(fvg_candle, "fvg_c3"):
            fvg_type = "fvg_c3"
        elif is_fvg_match(fvg_candle, "fvg"):
            fvg_type = "fvg_c2"
        
        if not fvg_type: return False
        
        # Get FVG base type
        fvg_base = get_fvg_base_type(fvg_candle, fvg_type.replace("_c2", ""))
        if not fvg_base: return False
        
        # Check if liquidation fields exist
        if "resistance" in fvg_base:
            liq_field = f"{fvg_type}_high_liquidated_by_candle_number"
            liq_num = fvg_candle.get(liq_field)
            return liq_num is None or liq_num == ""
        elif "support" in fvg_base:
            liq_field = f"{fvg_type}_low_liquidated_by_candle_number"
            liq_num = fvg_candle.get(liq_field)
            return liq_num is None or liq_num == ""
        
        return False

    def find_fvg_family_member(candles, reference_candle, target_fvg_type):
        """Find a FVG family member based on reference candle's fvg_number"""
        if not reference_candle: return None
        
        # Determine reference FVG type and number
        ref_fvg_type = None
        ref_fvg_number = None
        
        if is_valid_fvg_c1(reference_candle.get("fvg_c1")):
            ref_fvg_type = "fvg_c1"
            ref_fvg_number = get_fvg_number(reference_candle, "fvg_c1")
        elif is_valid_fvg_c3(reference_candle.get("fvg_c3")):
            ref_fvg_type = "fvg_c3"
            ref_fvg_number = get_fvg_number(reference_candle, "fvg_c3")
        elif is_valid_fvg_c2(reference_candle.get("is_fvg")):
            ref_fvg_type = "fvg"
            ref_fvg_number = get_fvg_number(reference_candle, "fvg")
        
        if not ref_fvg_number: return None
        
        # If reference is already the target type, return it
        if ref_fvg_type == target_fvg_type:
            return reference_candle
        
        # Search for family member with same fvg_number
        for candle in candles:
            if candle.get("candle_number") == reference_candle.get("candle_number"):
                continue
            
            # Check if this candle has the target FVG type with matching fvg_number
            if target_fvg_type == "fvg_c1":
                if is_valid_fvg_c1(candle.get("fvg_c1")):
                    candle_fvg_number = get_fvg_number(candle, "fvg_c1")
                    if candle_fvg_number == ref_fvg_number:
                        return candle
            elif target_fvg_type == "fvg_c3":
                if is_valid_fvg_c3(candle.get("fvg_c3")):
                    candle_fvg_number = get_fvg_number(candle, "fvg_c3")
                    if candle_fvg_number == ref_fvg_number:
                        return candle
            elif target_fvg_type == "fvg":
                if is_valid_fvg_c2(candle.get("is_fvg")):
                    candle_fvg_number = get_fvg_number(candle, "fvg")
                    if candle_fvg_number == ref_fvg_number:
                        return candle
        
        return None

    def process_second_source_independently(entry_spec, config_data, entry_key, sym, tf, dev_output_dir, source_def_name):
        
        second_source_candidates = []
        for key_id, candle_data in config_data.items():
            if not isinstance(key_id, str) or key_id.endswith("_candle_list"):
                continue
            
            if source_def_name.lower() in key_id.lower():
                continue
            
            if isinstance(candle_data, list) and len(candle_data) >= 5:
                has_swings = any(c.get("is_swing") for c in candle_data[:10] if isinstance(c, dict))
                has_swept = any(c.get("swept_by_liquidity") for c in candle_data[:10] if isinstance(c, dict))
                has_fvg = any(is_valid_fvg_c2(c.get("is_fvg")) or is_valid_fvg_c1(c.get("fvg_c1")) or is_valid_fvg_c3(c.get("fvg_c3")) 
                            for c in candle_data[:10] if isinstance(c, dict))
                
                if has_swings or has_swept or has_fvg:
                    second_source_candidates.append({
                        "key_id": key_id,
                        "candle_data": candle_data,
                        "has_swings": has_swings,
                        "has_swept": has_swept,
                        "has_fvg": has_fvg
                    })
        
        if not second_source_candidates:
            return None
        
        all_second_source_results = []
        
        for candidate in second_source_candidates:
            try:
                raw_second_candles = candidate["candle_data"]
                merged_second_candles_map = {}
                
                for c in raw_second_candles:
                    c_num = c.get("candle_number")
                    if c_num is None: continue
                    if c_num not in merged_second_candles_map:
                        merged_second_candles_map[c_num] = c.copy()
                    else:
                        for key, value in c.items():
                            if key not in merged_second_candles_map[c_num] or merged_second_candles_map[c_num][key] in [None, ""]:
                                merged_second_candles_map[c_num][key] = value
                            if key == "swing_type" and value:
                                merged_second_candles_map[c_num][key] = value
                                merged_second_candles_map[c_num]["is_swing"] = True
                
                second_candles = [merged_second_candles_map[num] for num in sorted(merged_second_candles_map.keys())]
                second_new_candles = [candle.copy() for candle in second_candles]
                
                start_search_with = entry_spec.get("start_search_with", "swept_candle").lower()
                start_search_with_tool = entry_spec.get("start_search_with_tool", {})
                before_entry = entry_spec.get("before_entry", {})
                data_selection_value = before_entry.get("data_selection", "all")
                comm_swings_field = before_entry.get("communicate_only2_swings", "")
                comm_mode = before_entry.get("swings_communication_condition", "follow_rules")
                comm_pair = []
                if comm_swings_field:
                    parts = comm_swings_field.split("_and_")
                    if len(parts) == 2: comm_pair = [parts[0].strip(), parts[1].strip()]

                subject_cfg = entry_spec.get("subject", {})
                if_intruder_value = subject_cfg.get("if_intruder", "skip")
                record_prices_cfg = entry_spec.get("record_prices", {})
                poi_target_role = subject_cfg.get("poi", "swept_candle")
                find_after_key = entry_spec.get("find_entry_after", "").lower()
                
                option_raw = entry_spec.get("option", "")
                option_condition = entry_spec.get("option_condition", "default").lower()
                
                valid_option_conditions = ["default", "after_start"]
                if option_condition not in valid_option_conditions:
                    option_condition = "default"
                
                second_source_items = []
                second_pending_orders = []
                second_executed_orders = []
                
                for i, candle in enumerate(second_new_candles):
                    is_swept = candle.get("swept_by_liquidity")
                    is_swing = is_valid_swing_type(candle.get("swing_type"))
                    is_fvg_c1 = is_valid_fvg_c1(candle.get("fvg_c1"))
                    is_fvg_c2 = is_valid_fvg_c2(candle.get("is_fvg"))
                    is_fvg_c3 = is_valid_fvg_c3(candle.get("fvg_c3"))
                    
                    # Check start_search_with criteria - handle fvg_swing
                    if "fvg_swing" in start_search_with:
                        if not is_swing: 
                            continue
                    elif "swing" in start_search_with and "fvg_swing" not in start_search_with:
                        if not is_swing: 
                            continue
                    elif "fvg_c1" in start_search_with:
                        if not is_fvg_c1: 
                            continue
                    elif "fvg" in start_search_with and "fvg_swing" not in start_search_with:
                        if not is_fvg_c2: 
                            continue
                    elif "fvg_c3" in start_search_with:
                        if not is_fvg_c3: 
                            continue
                    else:
                        if not is_swept: 
                            continue

                    c_num = candle.get("candle_number")
                    ######log(f"Processing candle {c_num}: {start_search_with.upper()} found")
                    
                    anchor_swing_type = candle.get("swing_type")
                    
                    orig_sweeper_num = candle.get("swept_by_candle_number")
                    orig_sweeper_candle = next((c for c in second_new_candles if c.get("candle_number") == orig_sweeper_num), None)
                    if not orig_sweeper_candle:
                        orig_sweeper_candle, orig_sweeper_num = candle, c_num
                    orig_sweeper_idx = second_new_candles.index(orig_sweeper_candle)
                    
                    acting_sweeper_candle, acting_sweeper_idx, has_acting = None, None, False
                    if orig_sweeper_candle.get("is_swing"):
                        acting_sweeper_candle, acting_sweeper_idx, has_acting = orig_sweeper_candle, orig_sweeper_idx, True
                    else:
                        allowed = get_allowed_acting_sweeper_types(anchor_swing_type)
                        for k in range(orig_sweeper_idx + 1, len(second_new_candles)):
                            cand = second_new_candles[k]
                            if cand.get("is_swing") and any(normalize_swing(cand.get("swing_type")) == normalize_swing(t) for t in allowed):
                                acting_sweeper_candle, acting_sweeper_idx, has_acting = cand, k, True
                                break

                    required_opt = None
                    if "fvg_c1" in option_raw.lower() or "fvg" in option_raw.lower() or "fvg_c3" in option_raw.lower() or "fvg_swing" in option_raw.lower():
                        required_opt = option_raw
                    else:
                        required_opt = get_opposite_swing_type(anchor_swing_type) if "opposite" in option_raw.lower() else anchor_swing_type
                    
                    option_candle, option_idx = None, None
                    
                    #log(f"Processing candle {c_num}: Looking for option ({required_opt})...")
                    
                    if option_condition == "default":
                        bound_idx = acting_sweeper_idx if has_acting else orig_sweeper_idx
                        s_min, s_max = min(i, bound_idx), max(i, bound_idx)
                        for k in range(s_min + 1, s_max):
                            if "fvg" in option_raw.lower():
                                # Check direct match or family member
                                if is_fvg_match(second_new_candles[k], required_opt, option_candle):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break
                                elif "fvg_swing" in option_raw.lower() and is_fvg_swing_candle(second_new_candles[k]):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break
                            else:
                                if is_swing_match(second_new_candles[k], required_opt):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break

                    elif option_condition == "after_start":
                        start_idx = i + 1
                        
                        for k in range(start_idx, len(second_new_candles)):
                            if "fvg" in option_raw.lower():
                                # Check direct match or family member
                                if is_fvg_match(second_new_candles[k], required_opt, option_candle):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break
                                elif "fvg_swing" in option_raw.lower() and is_fvg_swing_candle(second_new_candles[k]):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break
                            else:
                                if is_swing_match(second_new_candles[k], required_opt):
                                    option_candle, option_idx = second_new_candles[k], k
                                    break

                    if not option_candle: 
                        #log(f"Processing candle {c_num}: ✗ No option found ({required_opt})")
                        continue
                    
                    #log(f"Processing candle {c_num}: ✓ Found option at candle {option_candle.get('candle_number')}")

                    search_start = None
                    if "original_sweeper" in find_after_key:
                        search_start = orig_sweeper_idx + 1
                    elif "acting_sweeper" in find_after_key:
                        if has_acting: search_start = acting_sweeper_idx + 1
                    elif any(x in find_after_key for x in ["swept_candle", "swing_candle"]):
                        search_start = i + 1
                    elif "option" in find_after_key:
                        search_start = option_idx + 1
                    elif "fvg_c1" in find_after_key:
                        if option_candle and is_fvg_match(option_candle, "fvg_c1"):
                            search_start = option_idx + 1
                    elif "fvg_c3" in find_after_key:
                        if option_candle and is_fvg_match(option_candle, "fvg_c3"):
                            search_start = option_idx + 1
                    elif "fvg" in find_after_key:
                        if option_candle and is_fvg_match(option_candle, "fvg"):
                            search_start = option_idx + 1
                    elif "fvg_swing" in find_after_key:
                        if option_candle and is_fvg_swing_candle(option_candle):
                            search_start = option_idx + 1

                    if search_start is None: 
                        #log(f"Processing candle {c_num}: ✗ Invalid find_entry_after: {find_after_key}")
                        continue
                    
                    #log(f"Processing candle {c_num}: find_entry_after = '{find_after_key}'")

                    all_met = True
                    ref_sweeper = acting_sweeper_candle if has_acting else orig_sweeper_candle
                    condition_history = {
                        "option": option_candle, 
                        "sweeper": ref_sweeper, 
                        "swept": candle, 
                        "swing": candle,
                        "swing_candle": candle,
                        "subject": None
                    }
                    
                    sorted_cond_keys = [k for k in sorted(before_entry.keys()) if k.startswith("swing_")]
                    temp_search_start, temp_last_idx = search_start, search_start - 1
                    last_swing_candle = None
                    prev_swing_candle = None
                    
                    def validate_single(cfg, start_from, history, outlaw_behavior="check_further", prev_swing=None):
                        t_str = cfg.get("swing", "").lower()
                        c_str = cfg.get("condition", "").lower()

                        base_obj = None
                        if "sweeper" in t_str:
                            base_obj = history["sweeper"]
                        elif "option" in t_str:
                            base_obj = history["option"]
                        elif any(x in t_str for x in ["swing_candle", "swept_candle"]):
                            base_obj = history["swing"]
                        elif "swing_" in t_str:
                            bk = t_str.replace("_opposite", "").replace("_identical", "")
                            base_obj = history.get(bk, {})
                        else:
                            base_obj = history.get("swept", {})
                        
                        is_fvg_base = is_fvg_family_candle(base_obj, "fvg")
                        is_fvg_swing_base = is_fvg_swing_candle(base_obj)
                        
                        target_type = None
                        if is_fvg_base:
                            if "opposite" in t_str:
                                # Get opposite FVG type
                                fvg_type = None
                                if is_fvg_match(base_obj, "fvg_c1"):
                                    fvg_type = "fvg_c1"
                                elif is_fvg_match(base_obj, "fvg_c3"):
                                    fvg_type = "fvg_c3"
                                elif is_fvg_match(base_obj, "fvg"):
                                    fvg_type = "fvg"
                                
                                if fvg_type:
                                    fvg_base_val = get_fvg_base_type(base_obj, fvg_type)
                                    if fvg_base_val and "resistance" in fvg_base_val:
                                        target_type = f"{fvg_type}_support" if fvg_type != "fvg" else "fvg_support"
                                    elif fvg_base_val and "support" in fvg_base_val:
                                        target_type = f"{fvg_type}_resistance" if fvg_type != "fvg" else "fvg_resistance"
                                    else:
                                        target_type = t_str
                                else:
                                    target_type = t_str
                            else:
                                target_type = t_str
                        elif is_fvg_swing_base:
                            b_type = base_obj.get("swing_type")
                            target_type = get_opposite_swing_type(b_type) if "opposite" in t_str else b_type
                        else:
                            b_type = base_obj.get("swing_type")
                            target_type = get_opposite_swing_type(b_type) if "opposite" in t_str else b_type

                        parts = c_str.split('_')
                        constraints = []
                        
                        idx = 0
                        while idx < len(parts):
                            mode = parts[idx]
                            if mode not in ["beyond", "behind", "beyond_fvg", "behind_fvg", 
                                         "beyond_fvg_c1", "behind_fvg_c1", "beyond_fvg_c3", "behind_fvg_c3"]:
                                idx += 1
                                continue
                                
                            ref_obj = None
                            if idx + 1 < len(parts):
                                next_part = parts[idx+1]
                                if next_part == "option":
                                    ref_obj = history.get("option")
                                    idx += 2
                                elif next_part == "sweeper":
                                    ref_obj = history.get("sweeper")
                                    idx += 2
                                elif next_part == "swing" or next_part == "swept":
                                    if idx + 2 < len(parts) and parts[idx+2] == "candle":
                                        ref_obj = history.get("swing")
                                        idx += 3
                                    elif idx + 2 < len(parts):
                                        target_key = f"swing_{parts[idx+2]}"
                                        ref_obj = history.get(target_key)
                                        idx += 3
                                    else:
                                        idx += 1
                            
                            if ref_obj:
                                constraints.append((mode, ref_obj))
                            else:
                                idx += 1

                        for k in range(start_from, len(second_new_candles)):
                            kc = second_new_candles[k]
                            
                            # Check for FVG family member if target is FVG type
                            is_match = False
                            if "fvg_swing" in target_type.lower():
                                is_match = is_fvg_swing_candle(kc)
                            elif is_fvg_base or "fvg" in target_type.lower():
                                # Check for specific FVG type or family member
                                if "fvg_c1" in target_type.lower():
                                    # Check direct match or family member
                                    is_match = is_fvg_match(kc, "fvg_c1", base_obj)
                                elif "fvg_c3" in target_type.lower():
                                    is_match = is_fvg_match(kc, "fvg_c3", base_obj)
                                elif "fvg" in target_type.lower():
                                    is_match = is_fvg_match(kc, "fvg", base_obj)
                            else:
                                is_match = is_swing_match(kc, target_type)
                            
                            if not is_match:
                                continue
                            
                            match_all_conditions = True
                            for mode, ref in constraints:
                                if mode in ["beyond", "behind"]:
                                    if "fvg_swing" in target_type.lower():
                                        # Handle fvg_swing with regular swing conditions
                                        if mode == "beyond":
                                            if not check_beyond_condition(kc, ref, target_type):
                                                match_all_conditions = False
                                                break
                                        elif mode == "behind":
                                            if not check_behind_condition(kc, ref, target_type):
                                                match_all_conditions = False
                                                break
                                    elif is_fvg_base or "fvg" in target_type.lower():
                                        # Get FVG base type for the current candle
                                        fvg_type = None
                                        if is_fvg_match(kc, "fvg_c1"):
                                            fvg_type = "fvg_c1"
                                        elif is_fvg_match(kc, "fvg_c3"):
                                            fvg_type = "fvg_c3"
                                        elif is_fvg_match(kc, "fvg"):
                                            fvg_type = "fvg"
                                        
                                        fvg_base = get_fvg_base_type(kc, fvg_type) if fvg_type else None
                                        
                                        if not fvg_base:
                                            match_all_conditions = False
                                            break
                                            
                                        if mode == "beyond":
                                            if not check_beyond_condition_fvg(kc, ref, target_type, fvg_base):
                                                match_all_conditions = False
                                                break
                                        elif mode == "behind":
                                            if not check_behind_condition_fvg(kc, ref, target_type, fvg_base):
                                                match_all_conditions = False
                                                break
                                    else:
                                        if mode == "beyond":
                                            if not check_beyond_condition(kc, ref, target_type):
                                                match_all_conditions = False
                                                break
                                        elif mode == "behind":
                                            if not check_behind_condition(kc, ref, target_type):
                                                match_all_conditions = False
                                                break
                                elif mode in ["beyond_fvg", "behind_fvg", "beyond_fvg_c1", "behind_fvg_c1", 
                                           "beyond_fvg_c3", "behind_fvg_c3"]:
                                    fvg_type = "fvg"
                                    if "c1" in mode:
                                        fvg_type = "fvg_c1"
                                    elif "c3" in mode:
                                        fvg_type = "fvg_c3"
                                    
                                    if "beyond" in mode:
                                        if not check_beyond_fvg_condition(kc, ref, fvg_type):
                                            match_all_conditions = False
                                            break
                                    elif "behind" in mode:
                                        if not check_behind_fvg_condition(kc, ref, fvg_type):
                                            match_all_conditions = False
                                            break
                            
                            if match_all_conditions:
                                # Check intruder violation for "behind" conditions
                                # Determine which reference candle to check against based on constraints
                                for constraint_mode, constraint_ref in constraints:
                                    if "behind" in constraint_mode:
                                        # Check intruder violation between constraint_ref and found candle
                                        intruder_violation = check_intruder_violation_between_candles(
                                            constraint_ref, kc, constraint_mode, second_new_candles, outlaw_behavior
                                        )
                                        if intruder_violation:
                                            return None, None
                                
                                return kc, k
                            elif outlaw_behavior == "terminate":
                                return None, None
                                
                        return None, None

                    for s_key in sorted_cond_keys:
                        s_cfg = before_entry[s_key]
                        if_outlaw_value = s_cfg.get("if_outlaw", "check_further")
                        if_intruder_value = s_cfg.get("if_intruder", "skip")
                        
                        #log(f"Processing candle {c_num}: Searching {s_key} ({s_cfg.get('condition')}) from candle {temp_search_start}...")
                        
                        if s_key in comm_pair and comm_mode == "rules_or_opposite":
                            other_key = comm_pair[1] if s_key == comm_pair[0] else comm_pair[0]
                            other_cfg = before_entry.get(other_key)
                            other_outlaw = other_cfg.get("if_outlaw", "check_further") if other_cfg else "check_further"
                            other_intruder = other_cfg.get("if_intruder", "skip") if other_cfg else "skip"
                            
                            found_c, found_idx = validate_single(s_cfg, temp_search_start, condition_history, if_outlaw_value, prev_swing_candle)
                            if not found_c and other_cfg:
                                found_c, found_idx = validate_single(other_cfg, temp_search_start, condition_history, other_outlaw, prev_swing_candle)
                            
                            if found_c:
                                # Check intruder violation for "behind" conditions
                                if if_intruder_value == "terminate" or other_intruder == "terminate":
                                    # Parse condition to find which reference is being used
                                    condition_str = s_cfg.get("condition", "").lower()
                                    if "behind" in condition_str:
                                        # Find which reference is being checked against
                                        parts = condition_str.split('_')
                                        for part_idx, part in enumerate(parts):
                                            if part == "behind":
                                                if part_idx + 1 < len(parts):
                                                    ref_name = parts[part_idx + 1]
                                                    ref_candle = None
                                                    
                                                    if ref_name == "option":
                                                        ref_candle = condition_history.get("option")
                                                    elif ref_name == "sweeper":
                                                        ref_candle = condition_history.get("sweeper")
                                                    elif ref_name == "swing" or ref_name == "swept":
                                                        if part_idx + 2 < len(parts) and parts[part_idx + 2] == "candle":
                                                            ref_candle = condition_history.get("swing")
                                                    elif ref_name.startswith("swing_"):
                                                        ref_candle = condition_history.get(ref_name)
                                                    
                                                    if ref_candle:
                                                        intruder_violation = check_intruder_violation_between_candles(
                                                            ref_candle, found_c, condition_str, second_new_candles, "terminate"
                                                        )
                                                        if intruder_violation:
                                                            all_met = False
                                                            break
                                
                                if all_met:
                                    condition_history[s_key], temp_search_start, temp_last_idx = found_c, found_idx + 1, found_idx
                                    prev_swing_candle = found_c
                                    last_swing_candle = found_c
                                    #log(f"Processing candle {c_num}: ✓ Found {s_key} at candle {found_c.get('candle_number')}")
                                else:
                                    #log(f"Processing candle {c_num}: ✗ Intruder violation for {s_key}")
                                    all_met = False
                                    break
                            else:
                                #log(f"Processing candle {c_num}: ✗ No {s_key} found ({s_cfg.get('condition')})")
                                all_met = False
                                break
                        else:
                            found_c, found_idx = validate_single(s_cfg, temp_search_start, condition_history, if_outlaw_value, prev_swing_candle)
                            
                            if found_c:
                                # Check intruder violation for "behind" conditions
                                if if_intruder_value == "terminate":
                                    # Parse condition to find which reference is being used
                                    condition_str = s_cfg.get("condition", "").lower()
                                    if "behind" in condition_str:
                                        # Find which reference is being checked against
                                        parts = condition_str.split('_')
                                        for part_idx, part in enumerate(parts):
                                            if part == "behind":
                                                if part_idx + 1 < len(parts):
                                                    ref_name = parts[part_idx + 1]
                                                    ref_candle = None
                                                    
                                                    if ref_name == "option":
                                                        ref_candle = condition_history.get("option")
                                                    elif ref_name == "sweeper":
                                                        ref_candle = condition_history.get("sweeper")
                                                    elif ref_name == "swing" or ref_name == "swept":
                                                        if part_idx + 2 < len(parts) and parts[part_idx + 2] == "candle":
                                                            ref_candle = condition_history.get("swing")
                                                    elif ref_name.startswith("swing_"):
                                                        ref_candle = condition_history.get(ref_name)
                                                    
                                                    if ref_candle:
                                                        intruder_violation = check_intruder_violation_between_candles(
                                                            ref_candle, found_c, condition_str, second_new_candles, "terminate"
                                                        )
                                                        if intruder_violation:
                                                            all_met = False
                                                            break
                                
                                if all_met:
                                    condition_history[s_key], temp_search_start, temp_last_idx = found_c, found_idx + 1, found_idx
                                    prev_swing_candle = found_c
                                    last_swing_candle = found_c
                                    #log(f"Processing candle {c_num}: ✓ Found {s_key} at candle {found_c.get('candle_number')}")
                                else:
                                    #log(f"Processing candle {c_num}: ✗ Intruder violation for {s_key}")
                                    all_met = False
                                    break
                            else:
                                #log(f"Processing candle {c_num}: ✗ No {s_key} found ({s_cfg.get('condition')})")
                                all_met = False
                                break

                    if not all_met:
                        #log(f"Processing candle {c_num}: ✗ FAILED at swing_conditions")
                        continue

                    # Handle subject POI with family linking
                    if "swing_candle" in poi_target_role or "swept_candle" in poi_target_role:
                        poi_candle = candle
                    elif "option" in poi_target_role:
                        poi_candle = option_candle
                    elif "sweeper" in poi_target_role:
                        poi_candle = ref_sweeper
                    elif "swing_" in poi_target_role:
                        lookup_key = poi_target_role.replace("_candle", "")
                        poi_candle = condition_history.get(lookup_key)
                    elif "fvg_c1" in poi_target_role or "fvg_c3" in poi_target_role or "fvg" in poi_target_role:
                        # This is a FVG family POI - we need to find it using family linking
                        target_fvg_type = normalize_fvg_family(poi_target_role)
                        
                        # Try to find based on option or other references
                        reference_for_family = None
                        if option_candle and is_fvg_family_candle(option_candle, "fvg"):
                            reference_for_family = option_candle
                        elif any(k.startswith("swing_") for k in condition_history):
                            # Look for any FVG in condition history
                            for key, cand in condition_history.items():
                                if key.startswith("swing_") and is_fvg_family_candle(cand, "fvg"):
                                    reference_for_family = cand
                                    break
                        
                        if reference_for_family:
                            # Try to find family member
                            poi_candle = find_fvg_family_member(second_new_candles, reference_for_family, target_fvg_type)
                            if not poi_candle:
                                #log(f"Processing candle {c_num}: ✗ Could not find FVG family member {target_fvg_type}")
                                continue
                        else:
                            # Fallback to regular search
                            poi_candle = None
                            for k in range(temp_last_idx + 1, len(second_new_candles)):
                                if is_fvg_match(second_new_candles[k], target_fvg_type):
                                    poi_candle = second_new_candles[k]
                                    break
                            
                            if not poi_candle:
                                #log(f"Processing candle {c_num}: ✗ Could not find {poi_target_role}")
                                continue
                    else:
                        poi_candle = candle

                    if if_intruder_value == "terminate" and poi_candle and last_swing_candle and option_candle:
                        has_intruder = check_intruder_violation(
                            option_candle, 
                            last_swing_candle, 
                            poi_candle, 
                            second_new_candles, 
                            if_intruder_value
                        )
                        if has_intruder:
                            #log(f"Processing candle {c_num}: ✗ Intruder violation found")
                            continue

                    mitigation_found, desc, final_mit_rec = False, "pending entry level", None
                    if poi_candle:
                        is_fvg_poi = is_fvg_family_candle(poi_candle, "fvg")
                        is_fvg_swing_poi = is_fvg_swing_candle(poi_candle)
                        
                        if is_fvg_poi:
                            # Check FVG mitigation using FVG-specific liquidation fields
                            mitigation_found = check_fvg_mitigation(poi_candle, second_new_candles)
                            if mitigation_found:
                                liq_candle = get_fvg_liquidation_candle(poi_candle, second_new_candles)
                                if liq_candle:
                                    liq_candle["mitigation_candle"] = True
                                    poi_candle["mitigated"] = True
                                    poi_candle["mitigated_by_candle_number"] = liq_candle.get("candle_number")
                                    final_mit_rec = liq_candle.copy()
                                    final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                    desc = "mitigated entry"
                            else:
                                # Check for pending entry level
                                if check_fvg_pending_entry(poi_candle):
                                    poi_candle["pending_entry_level"] = True
                                    desc = "pending entry level"
                        elif is_fvg_swing_poi:
                            p_type = normalize_swing(poi_candle.get("swing_type", ""))
                            for m_idx in range(temp_last_idx + 1, len(second_new_candles)):
                                mc, is_mit = second_new_candles[m_idx], False
                                if "high" in p_type:
                                    if mc.get("high", 0) >= poi_candle.get("high", 0): is_mit = True
                                elif "low" in p_type:
                                    if mc.get("low", 999999) <= poi_candle.get("low", 999999): is_mit = True
                                
                                if is_mit:
                                    mc["mitigation_candle"] = True
                                    poi_candle["mitigated"] = True
                                    poi_candle["mitigated_by_candle_number"] = mc.get("candle_number")
                                    final_mit_rec = mc.copy()
                                    final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                    mitigation_found, desc = True, "mitigated entry"
                                    break
                        else:
                            p_type = normalize_swing(poi_candle.get("swing_type", ""))
                            for m_idx in range(temp_last_idx + 1, len(second_new_candles)):
                                mc, is_mit = second_new_candles[m_idx], False
                                if "high" in p_type:
                                    if mc.get("high", 0) >= poi_candle.get("high", 0): is_mit = True
                                elif "low" in p_type:
                                    if mc.get("low", 999999) <= poi_candle.get("low", 999999): is_mit = True
                                
                                if is_mit:
                                    mc["mitigation_candle"] = True
                                    poi_candle["mitigated"] = True
                                    poi_candle["mitigated_by_candle_number"] = mc.get("candle_number")
                                    final_mit_rec = mc.copy()
                                    final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                    mitigation_found, desc = True, "mitigated entry"
                                    break
                        
                        if not mitigation_found and not poi_candle.get("pending_entry_level"): 
                            poi_candle["pending_entry_level"] = True
                            desc = "pending entry level"
                            #log(f"Processing candle {c_num}: ✓ Pattern found - PENDING ENTRY")
                        else:
                            #log(f"Processing candle {c_num}: ✓ Pattern found - {desc.upper()}")
                            log

                        source_timeframe = extract_timeframe_from_key(candidate['key_id'])
                        found_in_timeframe = tf
                        
                        trade_info = {
                            "symbol": sym,
                            "description": desc,
                            "pattern": entry_key,
                            "signal_from": f"{entry_key}_SECOND_SOURCE_{candidate['key_id']}",
                            "second_source_key": candidate['key_id'],
                            "issource": True,
                            "source_definition": "second_source_source",
                            "found_in_timeframe": found_in_timeframe,
                            "timeframe": source_timeframe or tf
                        }
                        
                        for role in ["entry", "exit", "target"]:
                            r_cfg = record_prices_cfg.get(role, {})
                            source_role_raw = r_cfg.get("record", "").replace("_candle", "")
                            if source_role_raw in ["swing", "swing_candle"]: source_role_raw = "swept"
                            price = get_price_from_cfg(condition_history.get(source_role_raw), r_cfg)
                            trade_info[role] = price
                        
                        ot_cfg = record_prices_cfg.get("order_type_if_entry_swing_is", {})
                        
                        # Determine order type based on POI candle type
                        if is_fvg_poi:
                            # FVG family candle
                            fvg_type = None
                            if is_fvg_match(poi_candle, "fvg_c1"):
                                fvg_type = "fvg_c1"
                            elif is_fvg_match(poi_candle, "fvg_c3"):
                                fvg_type = "fvg_c3"
                            elif is_fvg_match(poi_candle, "fvg"):
                                fvg_type = "fvg"
                            
                            if fvg_type:
                                fvg_base = get_fvg_base_type(poi_candle, fvg_type)
                                if fvg_base:
                                    if "resistance" in fvg_base:
                                        trade_info["order_type"] = ot_cfg.get("resistance_fvg")
                                    elif "support" in fvg_base:
                                        trade_info["order_type"] = ot_cfg.get("support_fvg")
                        elif is_fvg_swing_poi:
                            # FVG swing candle (regular swing)
                            if "high" in normalize_swing(poi_candle.get("swing_type", "")):
                                trade_info["order_type"] = ot_cfg.get("swing_higher_high_or_lower_high")
                            else:
                                trade_info["order_type"] = ot_cfg.get("swing_lower_low_or_higher_low")
                        
                        if mitigation_found:
                            second_executed_orders.append(trade_info)
                        else:
                            second_pending_orders.append(trade_info)

                        pattern_item = {
                            "pattern_type": f"{entry_key}_SECOND_SOURCE",
                            "swept_candle_number": candle.get("candle_number"),
                            "sweeper_candle_number": ref_sweeper.get("candle_number"),
                            "poi_target_role": poi_target_role,
                            "start_search_with_tool": start_search_with_tool,
                            "subject_drawing": subject_cfg.get("drawing", {}),
                            "option_drawing": entry_spec.get("option_tool", {}),
                            "swings_drawing": {k: v.get("drawing", {}) for k, v in before_entry.items() if k.startswith("swing_")},
                            "second_source_key": candidate['key_id'],
                            "option_condition": option_condition,
                            "family_linked": True if "fvg" in poi_target_role and reference_for_family else False
                        }

                        p_keys = []
                        roles_to_check = [("swept", candle), ("sweeper", ref_sweeper), ("option", option_candle)]
                        for skey in sorted(condition_history.keys()):
                            if skey not in ["swept", "sweeper", "option", "subject", "swing", "swing_candle"]: 
                                roles_to_check.append((skey, condition_history[skey]))

                        for role_name, rc in roles_to_check:
                            if not rc: continue
                            c_copy = rc.copy()
                            c_copy[role_name] = True
                            if role_name == "option" or role_name.startswith("swing_"):
                                is_fvg_candle = is_fvg_family_candle(rc, "fvg")
                                is_fvg_swing_candle_check = is_fvg_swing_candle(rc)
                                if is_fvg_candle:
                                    liq_sw = find_liquidation_for_fvg(rc, second_new_candles)
                                elif is_fvg_swing_candle_check:
                                    liq_sw = find_liquidation_for_candle(rc, second_new_candles)
                                else:
                                    liq_sw = find_liquidation_for_candle(rc, second_new_candles)
                                
                                if liq_sw:
                                    c_copy["is_liquidated"] = True
                                    c_copy["liquidated_by_candle_number"] = liq_sw.get("candle_number")
                                    sw_rec = liq_sw.copy()
                                    sw_rec["is_liquidity"] = True
                                    sw_rec["liquidated_candle_number"] = rc.get("candle_number")
                                    sw_rec[f"{role_name}_sweeper"] = True
                                    p_keys.extend([c_copy, sw_rec]); continue
                            p_keys.append(c_copy)

                        if final_mit_rec: p_keys.append(final_mit_rec)
                        pattern_item["pattern_keys"] = p_keys
                        second_source_items.append(pattern_item)
                
                if second_source_items:
                    filtered_items = filter_data_selection(
                        second_source_items, 
                        start_search_with, 
                        data_selection_value, 
                        second_candles
                    )
                    
                    result = {
                        "key_id": candidate['key_id'],
                        "candle_count": len(second_candles),
                        "patterns_found": len(second_source_items),
                        "filtered_patterns": len(filtered_items),
                        "items": filtered_items,
                        "pending_orders": second_pending_orders,
                        "executed_orders": second_executed_orders,
                        "candles": second_candles,
                        "chart_files": [f for f in os.listdir(dev_output_dir) 
                                      if f.lower().endswith('.png') and candidate['key_id'].lower() in f.lower()]
                    }
                    
                    all_second_source_results.append(result)
            
            except Exception as e:
                log
        
        return all_second_source_results if all_second_source_results else None

    # ─── Main logic ───────────────────────────────────────────────────────
    log(f"--- STARTING ENTRY POI CONDITION ANALYSIS: {broker_name} ---")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: return f"Error: Broker {broker_name} not in dictionary."

    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    if not am_data: return "Error: accountmanagement.json missing."

    define_candles = am_data.get("chart", {}).get("define_candles", {})
    entry_poi_root = define_candles.get("entries_poi_condition", {})

    total_entries_found = 0
    total_second_source_entries = 0

    for apprehend_key, entry_cfg in entry_poi_root.items():
        if not apprehend_key.startswith("apprehend_"): continue
        source_def_name = apprehend_key.replace("apprehend_", "")
        source_def = define_candles.get(source_def_name, {})
        source_png_filename = source_def.get("filename", "").replace(".json", ".png")

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in os.listdir(sym_p):
                tf_p = os.path.join(sym_p, tf)
                if not os.path.isdir(tf_p): continue
                
                dev_output_dir = os.path.join(os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name)), sym, tf)
                config_path = os.path.join(dev_output_dir, "config.json")
                if not os.path.exists(config_path): continue

                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    source_key = next((k for k in config_data if source_def_name.lower() in k.lower()), None)
                    if not source_key: continue

                    raw_source_candles = config_data.get(source_key)
                    if not isinstance(raw_source_candles, list): continue

                    merged_candles_map = {}
                    for c in raw_source_candles:
                        c_num = c.get("candle_number")
                        if c_num is None: continue
                        if c_num not in merged_candles_map:
                            merged_candles_map[c_num] = c.copy()
                        else:
                            for key, value in c.items():
                                if key not in merged_candles_map[c_num] or merged_candles_map[c_num][key] in [None, ""]:
                                    merged_candles_map[c_num][key] = value
                                if key == "swing_type" and value:
                                    merged_candles_map[c_num][key] = value
                                    merged_candles_map[c_num]["is_swing"] = True

                    source_candles = [merged_candles_map[num] for num in sorted(merged_candles_map.keys())]
                    new_candles = [candle.copy() for candle in source_candles]
                    
                    # Process each entry independently
                    for entry_key, entry_spec in entry_cfg.items():
                        if not entry_key.startswith("entry_"): continue
                        new_filename = entry_spec.get("new_filename", "").strip()
                        if not new_filename: continue
                        
                        # ─── CHECK SECOND SOURCE PROCESSING FLAG ─────────────────────
                        process_second_source = entry_spec.get("process_second_source_files", "").strip().lower()
                        should_process_second_source = (process_second_source == "yes")
                        
                        log(f"\n[PROCESSING] Entry: {entry_key} on {sym}/{tf}")
                        log(f"  Config summary:")
                        log(f"    start_search_with: '{entry_spec.get('start_search_with', 'swept_candle')}'")
                        log(f"    option: '{entry_spec.get('option', '')}'")
                        log(f"    option_condition: '{entry_spec.get('option_condition', 'default')}'")
                        log(f"    find_entry_after: '{entry_spec.get('find_entry_after', '')}'")
                        log(f"    second_source_processing: {'ENABLED' if should_process_second_source else 'DISABLED'}")
                        
                        # Get save configuration
                        save_cfg = entry_spec.get("save_to", {}).get("new_filename_folder", {})
                        chart_cfg = save_cfg.get("save_charts", {})
                        data_cfg = save_cfg.get("save_data", {})
                        
                        chart_folder = chart_cfg.get("folder", "chart")
                        pending_folder = data_cfg.get("pending_folder", "limit_orders")
                        mitigated_folder = data_cfg.get("mitigated_folder", "executed_orders")
                        
                        # ─── PROCESS main SOURCE (ALWAYS) ─────────────────────
                        main_items = []
                        main_pending_orders = []
                        main_executed_orders = []
                        
                        start_search_with = entry_spec.get("start_search_with", "swept_candle").lower()
                        start_search_with_tool = entry_spec.get("start_search_with_tool", {})
                        before_entry = entry_spec.get("before_entry", {})
                        data_selection_value = before_entry.get("data_selection", "all")
                        comm_swings_field = before_entry.get("communicate_only2_swings", "")
                        comm_mode = before_entry.get("swings_communication_condition", "follow_rules")
                        comm_pair = []
                        if comm_swings_field:
                            parts = comm_swings_field.split("_and_")
                            if len(parts) == 2: comm_pair = [parts[0].strip(), parts[1].strip()]

                        subject_cfg = entry_spec.get("subject", {})
                        if_intruder_value = subject_cfg.get("if_intruder", "skip")
                        record_prices_cfg = entry_spec.get("record_prices", {})
                        poi_target_role = subject_cfg.get("poi", "swept_candle")
                        find_after_key = entry_spec.get("find_entry_after", "").lower()
                        
                        option_raw = entry_spec.get("option", "")
                        option_condition = entry_spec.get("option_condition", "default").lower()
                        
                        valid_option_conditions = ["default", "after_start"]
                        if option_condition not in valid_option_conditions:
                            option_condition = "default"
                        
                        stage_failures = {
                            "start_search_with": 0,
                            "option": 0,
                            "find_entry_after": 0,
                            "swing_conditions": 0,
                            "intruder": 0
                        }
                        total_candles_checked = 0
                        
                        for i, candle in enumerate(new_candles):
                            total_candles_checked += 1
                            c_num = candle.get("candle_number")
                            
                            is_swept = candle.get("swept_by_liquidity")
                            is_swing = is_valid_swing_type(candle.get("swing_type"))
                            is_fvg_c1 = is_valid_fvg_c1(candle.get("fvg_c1"))
                            is_fvg_c2 = is_valid_fvg_c2(candle.get("is_fvg"))
                            is_fvg_c3 = is_valid_fvg_c3(candle.get("fvg_c3"))
                            
                            # Check start_search_with criteria - handle fvg_swing
                            if "fvg_swing" in start_search_with:
                                if not is_swing: 
                                    stage_failures["start_search_with"] += 1
                                    continue
                            elif "swing" in start_search_with and "fvg_swing" not in start_search_with:
                                if not is_swing: 
                                    stage_failures["start_search_with"] += 1
                                    continue
                            elif "fvg_c1" in start_search_with:
                                if not is_fvg_c1: 
                                    stage_failures["start_search_with"] += 1
                                    continue
                            elif "fvg" in start_search_with and "fvg_swing" not in start_search_with:
                                if not is_fvg_c2: 
                                    stage_failures["start_search_with"] += 1
                                    continue
                            elif "fvg_c3" in start_search_with:
                                if not is_fvg_c3: 
                                    stage_failures["start_search_with"] += 1
                                    continue
                            else:
                                if not is_swept: 
                                    stage_failures["start_search_with"] += 1
                                    continue

                            ######log(f"Processing candle {c_num}: {start_search_with.upper()} found")
                            
                            anchor_swing_type = candle.get("swing_type")
                            
                            orig_sweeper_num = candle.get("swept_by_candle_number")
                            orig_sweeper_candle = next((c for c in new_candles if c.get("candle_number") == orig_sweeper_num), None)
                            if not orig_sweeper_candle:
                                orig_sweeper_candle, orig_sweeper_num = candle, c_num
                            orig_sweeper_idx = new_candles.index(orig_sweeper_candle)
                            
                            acting_sweeper_candle, acting_sweeper_idx, has_acting = None, None, False
                            if orig_sweeper_candle.get("is_swing"):
                                acting_sweeper_candle, acting_sweeper_idx, has_acting = orig_sweeper_candle, orig_sweeper_idx, True
                            else:
                                allowed = get_allowed_acting_sweeper_types(anchor_swing_type)
                                for k in range(orig_sweeper_idx + 1, len(new_candles)):
                                    cand = new_candles[k]
                                    if cand.get("is_swing") and any(normalize_swing(cand.get("swing_type")) == normalize_swing(t) for t in allowed):
                                        acting_sweeper_candle, acting_sweeper_idx, has_acting = cand, k, True
                                        break

                            # Determine required option type - handle fvg_swing
                            required_opt = None
                            if "fvg_c1" in option_raw.lower() or "fvg" in option_raw.lower() or "fvg_c3" in option_raw.lower() or "fvg_swing" in option_raw.lower():
                                required_opt = option_raw
                            else:
                                required_opt = get_opposite_swing_type(anchor_swing_type) if "opposite" in option_raw.lower() else anchor_swing_type
                            
                            option_candle, option_idx = None, None
                            
                            #log(f"Processing candle {c_num}: Looking for option ({required_opt})...")
                            
                            if option_condition == "default":
                                bound_idx = acting_sweeper_idx if has_acting else orig_sweeper_idx
                                s_min, s_max = min(i, bound_idx), max(i, bound_idx)
                                for k in range(s_min + 1, s_max):
                                    if "fvg" in option_raw.lower():
                                        # Check direct match or family member
                                        if is_fvg_match(new_candles[k], required_opt, option_candle):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                                        elif "fvg_swing" in option_raw.lower() and is_fvg_swing_candle(new_candles[k]):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                                    else:
                                        if is_swing_match(new_candles[k], required_opt):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                            
                            elif option_condition == "after_start":
                                start_idx = i + 1
                                
                                for k in range(start_idx, len(new_candles)):
                                    if "fvg" in option_raw.lower():
                                        # Check direct match or family member
                                        if is_fvg_match(new_candles[k], required_opt, option_candle):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                                        elif "fvg_swing" in option_raw.lower() and is_fvg_swing_candle(new_candles[k]):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                                    else:
                                        if is_swing_match(new_candles[k], required_opt):
                                            option_candle, option_idx = new_candles[k], k
                                            break
                            
                            if not option_candle: 
                                #log(f"Processing candle {c_num}: ✗ No option found ({required_opt})")
                                stage_failures["option"] += 1
                                continue
                            
                            #log(f"Processing candle {c_num}: ✓ Found option at candle {option_candle.get('candle_number')}")

                            search_start = None
                            if "original_sweeper" in find_after_key:
                                search_start = orig_sweeper_idx + 1
                            elif "acting_sweeper" in find_after_key:
                                if has_acting: search_start = acting_sweeper_idx + 1
                            elif any(x in find_after_key for x in ["swept_candle", "swing_candle"]):
                                search_start = i + 1
                            elif "option" in find_after_key:
                                search_start = option_idx + 1
                            elif "fvg_c1" in find_after_key:
                                if option_candle and is_fvg_match(option_candle, "fvg_c1"):
                                    search_start = option_idx + 1
                            elif "fvg_c3" in find_after_key:
                                if option_candle and is_fvg_match(option_candle, "fvg_c3"):
                                    search_start = option_idx + 1
                            elif "fvg" in find_after_key:
                                if option_candle and is_fvg_match(option_candle, "fvg"):
                                    search_start = option_idx + 1
                            elif "fvg_swing" in find_after_key:
                                if option_candle and is_fvg_swing_candle(option_candle):
                                    search_start = option_idx + 1

                            if search_start is None: 
                                #log(f"Processing candle {c_num}: ✗ Invalid find_entry_after: {find_after_key}")
                                stage_failures["find_entry_after"] += 1
                                continue
                            
                            #log(f"Processing candle {c_num}: find_entry_after = '{find_after_key}'")

                            all_met = True
                            ref_sweeper = acting_sweeper_candle if has_acting else orig_sweeper_candle
                            condition_history = {
                                "option": option_candle, 
                                "sweeper": ref_sweeper, 
                                "swept": candle, 
                                "swing": candle,
                                "swing_candle": candle,
                                "subject": None
                            }
                            
                            sorted_cond_keys = [k for k in sorted(before_entry.keys()) if k.startswith("swing_")]
                            temp_search_start, temp_last_idx = search_start, search_start - 1
                            last_swing_candle = None
                            prev_swing_candle = None
                            
                            def validate_single(cfg, start_from, history, outlaw_behavior="check_further", prev_swing=None):
                                t_str = cfg.get("swing", "").lower()
                                c_str = cfg.get("condition", "").lower()

                                base_obj = None
                                if "sweeper" in t_str:
                                    base_obj = history["sweeper"]
                                elif "option" in t_str:
                                    base_obj = history["option"]
                                elif any(x in t_str for x in ["swing_candle", "swept_candle"]):
                                    base_obj = history["swing"]
                                elif "swing_" in t_str:
                                    bk = t_str.replace("_opposite", "").replace("_identical", "")
                                    base_obj = history.get(bk, {})
                                else:
                                    base_obj = history.get("swept", {})
                                
                                is_fvg_base = is_fvg_family_candle(base_obj, "fvg")
                                is_fvg_swing_base = is_fvg_swing_candle(base_obj)
                                
                                target_type = None
                                if is_fvg_base:
                                    if "opposite" in t_str:
                                        # Get opposite FVG type
                                        fvg_type = None
                                        if is_fvg_match(base_obj, "fvg_c1"):
                                            fvg_type = "fvg_c1"
                                        elif is_fvg_match(base_obj, "fvg_c3"):
                                            fvg_type = "fvg_c3"
                                        elif is_fvg_match(base_obj, "fvg"):
                                            fvg_type = "fvg"
                                        
                                        if fvg_type:
                                            fvg_base_val = get_fvg_base_type(base_obj, fvg_type)
                                            if fvg_base_val and "resistance" in fvg_base_val:
                                                target_type = f"{fvg_type}_support" if fvg_type != "fvg" else "fvg_support"
                                            elif fvg_base_val and "support" in fvg_base_val:
                                                target_type = f"{fvg_type}_resistance" if fvg_type != "fvg" else "fvg_resistance"
                                            else:
                                                target_type = t_str
                                        else:
                                            target_type = t_str
                                    else:
                                        target_type = t_str
                                elif is_fvg_swing_base:
                                    b_type = base_obj.get("swing_type")
                                    target_type = get_opposite_swing_type(b_type) if "opposite" in t_str else b_type
                                else:
                                    b_type = base_obj.get("swing_type")
                                    target_type = get_opposite_swing_type(b_type) if "opposite" in t_str else b_type

                                parts = c_str.split('_')
                                constraints = []
                                
                                idx = 0
                                while idx < len(parts):
                                    mode = parts[idx]
                                    if mode not in ["beyond", "behind", "beyond_fvg", "behind_fvg", 
                                                 "beyond_fvg_c1", "behind_fvg_c1", "beyond_fvg_c3", "behind_fvg_c3"]:
                                        idx += 1
                                        continue
                                        
                                    ref_obj = None
                                    if idx + 1 < len(parts):
                                        next_part = parts[idx+1]
                                        if next_part == "option":
                                            ref_obj = history.get("option")
                                            idx += 2
                                        elif next_part == "sweeper":
                                            ref_obj = history.get("sweeper")
                                            idx += 2
                                        elif next_part == "swing" or next_part == "swept":
                                            if idx + 2 < len(parts) and parts[idx+2] == "candle":
                                                ref_obj = history.get("swing")
                                                idx += 3
                                            elif idx + 2 < len(parts):
                                                target_key = f"swing_{parts[idx+2]}"
                                                ref_obj = history.get(target_key)
                                                idx += 3
                                            else:
                                                idx += 1
                                    
                                    if ref_obj:
                                        constraints.append((mode, ref_obj))
                                    else:
                                        idx += 1

                                for k in range(start_from, len(new_candles)):
                                    kc = new_candles[k]
                                    
                                    # Check for FVG family member if target is FVG type
                                    is_match = False
                                    if "fvg_swing" in target_type.lower():
                                        is_match = is_fvg_swing_candle(kc)
                                    elif is_fvg_base or "fvg" in target_type.lower():
                                        # Check for specific FVG type or family member
                                        if "fvg_c1" in target_type.lower():
                                            # Check direct match or family member
                                            is_match = is_fvg_match(kc, "fvg_c1", base_obj)
                                        elif "fvg_c3" in target_type.lower():
                                            is_match = is_fvg_match(kc, "fvg_c3", base_obj)
                                        elif "fvg" in target_type.lower():
                                            is_match = is_fvg_match(kc, "fvg", base_obj)
                                    else:
                                        is_match = is_swing_match(kc, target_type)
                                    
                                    if not is_match:
                                        continue
                                    
                                    match_all_conditions = True
                                    for mode, ref in constraints:
                                        if mode in ["beyond", "behind"]:
                                            if "fvg_swing" in target_type.lower():
                                                # Handle fvg_swing with regular swing conditions
                                                if mode == "beyond":
                                                    if not check_beyond_condition(kc, ref, target_type):
                                                        match_all_conditions = False
                                                        break
                                                elif mode == "behind":
                                                    if not check_behind_condition(kc, ref, target_type):
                                                        match_all_conditions = False
                                                        break
                                            elif is_fvg_base or "fvg" in target_type.lower():
                                                # Get FVG base type for the current candle
                                                fvg_type = None
                                                if is_fvg_match(kc, "fvg_c1"):
                                                    fvg_type = "fvg_c1"
                                                elif is_fvg_match(kc, "fvg_c3"):
                                                    fvg_type = "fvg_c3"
                                                elif is_fvg_match(kc, "fvg"):
                                                    fvg_type = "fvg"
                                                
                                                fvg_base = get_fvg_base_type(kc, fvg_type) if fvg_type else None
                                                
                                                if not fvg_base:
                                                    match_all_conditions = False
                                                    break
                                                    
                                                if mode == "beyond":
                                                    if not check_beyond_condition_fvg(kc, ref, target_type, fvg_base):
                                                        match_all_conditions = False
                                                        break
                                                elif mode == "behind":
                                                    if not check_behind_condition_fvg(kc, ref, target_type, fvg_base):
                                                        match_all_conditions = False
                                                        break
                                            else:
                                                if mode == "beyond":
                                                    if not check_beyond_condition(kc, ref, target_type):
                                                        match_all_conditions = False
                                                        break
                                                elif mode == "behind":
                                                    if not check_behind_condition(kc, ref, target_type):
                                                        match_all_conditions = False
                                                        break
                                        elif mode in ["beyond_fvg", "behind_fvg", "beyond_fvg_c1", "behind_fvg_c1", 
                                                   "beyond_fvg_c3", "behind_fvg_c3"]:
                                            fvg_type = "fvg"
                                            if "c1" in mode:
                                                fvg_type = "fvg_c1"
                                            elif "c3" in mode:
                                                fvg_type = "fvg_c3"
                                            
                                            if "beyond" in mode:
                                                if not check_beyond_fvg_condition(kc, ref, fvg_type):
                                                    match_all_conditions = False
                                                    break
                                            elif "behind" in mode:
                                                if not check_behind_fvg_condition(kc, ref, fvg_type):
                                                    match_all_conditions = False
                                                    break
                                    
                                    if match_all_conditions:
                                        # Check intruder violation for "behind" conditions
                                        for constraint_mode, constraint_ref in constraints:
                                            if "behind" in constraint_mode:
                                                # Check intruder violation between constraint_ref and found candle
                                                intruder_violation = check_intruder_violation_between_candles(
                                                    constraint_ref, kc, constraint_mode, new_candles, outlaw_behavior
                                                )
                                                if intruder_violation:
                                                    return None, None
                                        
                                        return kc, k
                                    elif outlaw_behavior == "terminate":
                                        return None, None
                                        
                                return None, None

                            for s_key in sorted_cond_keys:
                                s_cfg = before_entry[s_key]
                                if_outlaw_value = s_cfg.get("if_outlaw", "check_further")
                                if_intruder_value_swing = s_cfg.get("if_intruder", "skip")
                                
                                #log(f"Processing candle {c_num}: Searching {s_key} ({s_cfg.get('condition')}) from candle {temp_search_start}...")
                                
                                if s_key in comm_pair and comm_mode == "rules_or_opposite":
                                    other_key = comm_pair[1] if s_key == comm_pair[0] else comm_pair[0]
                                    other_cfg = before_entry.get(other_key)
                                    other_outlaw = other_cfg.get("if_outlaw", "check_further") if other_cfg else "check_further"
                                    other_intruder = other_cfg.get("if_intruder", "skip") if other_cfg else "skip"
                                    
                                    found_c, found_idx = validate_single(s_cfg, temp_search_start, condition_history, if_outlaw_value, prev_swing_candle)
                                    if not found_c and other_cfg:
                                        found_c, found_idx = validate_single(other_cfg, temp_search_start, condition_history, other_outlaw, prev_swing_candle)
                                    
                                    if found_c:
                                        # Check intruder violation for "behind" conditions
                                        if if_intruder_value_swing == "terminate" or other_intruder == "terminate":
                                            # Parse condition to find which reference is being used
                                            condition_str = s_cfg.get("condition", "").lower()
                                            if "behind" in condition_str:
                                                # Find which reference is being checked against
                                                parts = condition_str.split('_')
                                                for part_idx, part in enumerate(parts):
                                                    if part == "behind":
                                                        if part_idx + 1 < len(parts):
                                                            ref_name = parts[part_idx + 1]
                                                            ref_candle = None
                                                            
                                                            if ref_name == "option":
                                                                ref_candle = condition_history.get("option")
                                                            elif ref_name == "sweeper":
                                                                ref_candle = condition_history.get("sweeper")
                                                            elif ref_name == "swing" or ref_name == "swept":
                                                                if part_idx + 2 < len(parts) and parts[part_idx + 2] == "candle":
                                                                    ref_candle = condition_history.get("swing")
                                                            elif ref_name.startswith("swing_"):
                                                                ref_candle = condition_history.get(ref_name)
                                                            
                                                            if ref_candle:
                                                                intruder_violation = check_intruder_violation_between_candles(
                                                                    ref_candle, found_c, condition_str, new_candles, "terminate"
                                                                )
                                                                if intruder_violation:
                                                                    all_met = False
                                                                    break
                                        
                                        if all_met:
                                            condition_history[s_key], temp_search_start, temp_last_idx = found_c, found_idx + 1, found_idx
                                            prev_swing_candle = found_c
                                            last_swing_candle = found_c
                                            #log(f"Processing candle {c_num}: ✓ Found {s_key} at candle {found_c.get('candle_number')}")
                                        else:
                                            #log(f"Processing candle {c_num}: ✗ Intruder violation for {s_key}")
                                            stage_failures["intruder"] += 1
                                            all_met = False
                                            break
                                    else:
                                        #log(f"Processing candle {c_num}: ✗ No {s_key} found ({s_cfg.get('condition')})")
                                        stage_failures["swing_conditions"] += 1
                                        all_met = False
                                        break
                                else:
                                    found_c, found_idx = validate_single(s_cfg, temp_search_start, condition_history, if_outlaw_value, prev_swing_candle)
                                    
                                    if found_c:
                                        # Check intruder violation for "behind" conditions
                                        if if_intruder_value_swing == "terminate":
                                            # Parse condition to find which reference is being used
                                            condition_str = s_cfg.get("condition", "").lower()
                                            if "behind" in condition_str:
                                                # Find which reference is being checked against
                                                parts = condition_str.split('_')
                                                for part_idx, part in enumerate(parts):
                                                    if part == "behind":
                                                        if part_idx + 1 < len(parts):
                                                            ref_name = parts[part_idx + 1]
                                                            ref_candle = None
                                                            
                                                            if ref_name == "option":
                                                                ref_candle = condition_history.get("option")
                                                            elif ref_name == "sweeper":
                                                                ref_candle = condition_history.get("sweeper")
                                                            elif ref_name == "swing" or ref_name == "swept":
                                                                if part_idx + 2 < len(parts) and parts[part_idx + 2] == "candle":
                                                                    ref_candle = condition_history.get("swing")
                                                            elif ref_name.startswith("swing_"):
                                                                ref_candle = condition_history.get(ref_name)
                                                            
                                                            if ref_candle:
                                                                intruder_violation = check_intruder_violation_between_candles(
                                                                    ref_candle, found_c, condition_str, new_candles, "terminate"
                                                                )
                                                                if intruder_violation:
                                                                    all_met = False
                                                                    break
                                        
                                        if all_met:
                                            condition_history[s_key], temp_search_start, temp_last_idx = found_c, found_idx + 1, found_idx
                                            prev_swing_candle = found_c
                                            last_swing_candle = found_c
                                            #log(f"Processing candle {c_num}: ✓ Found {s_key} at candle {found_c.get('candle_number')}")
                                        else:
                                            #log(f"Processing candle {c_num}: ✗ Intruder violation for {s_key}")
                                            stage_failures["intruder"] += 1
                                            all_met = False
                                            break
                                    else:
                                        #log(f"Processing candle {c_num}: ✗ No {s_key} found ({s_cfg.get('condition')})")
                                        stage_failures["swing_conditions"] += 1
                                        all_met = False
                                        break

                            if not all_met:
                                #log(f"Processing candle {c_num}: ✗ FAILED at swing_conditions")
                                continue

                            # Handle subject POI with family linking
                            if "swing_candle" in poi_target_role or "swept_candle" in poi_target_role:
                                poi_candle = candle
                            elif "option" in poi_target_role:
                                poi_candle = option_candle
                            elif "sweeper" in poi_target_role:
                                poi_candle = ref_sweeper
                            elif "swing_" in poi_target_role:
                                lookup_key = poi_target_role.replace("_candle", "")
                                poi_candle = condition_history.get(lookup_key)
                            elif "fvg_c1" in poi_target_role or "fvg_c3" in poi_target_role or "fvg" in poi_target_role:
                                # This is a FVG family POI - we need to find it using family linking
                                target_fvg_type = normalize_fvg_family(poi_target_role)
                                
                                # Try to find based on option or other references
                                reference_for_family = None
                                if option_candle and is_fvg_family_candle(option_candle, "fvg"):
                                    reference_for_family = option_candle
                                elif any(k.startswith("swing_") for k in condition_history):
                                    # Look for any FVG in condition history
                                    for key, cand in condition_history.items():
                                        if key.startswith("swing_") and is_fvg_family_candle(cand, "fvg"):
                                            reference_for_family = cand
                                            break
                                
                                if reference_for_family:
                                    # Try to find family member
                                    poi_candle = find_fvg_family_member(new_candles, reference_for_family, target_fvg_type)
                                    if not poi_candle:
                                        #log(f"Processing candle {c_num}: ✗ Could not find FVG family member {target_fvg_type}")
                                        stage_failures["swing_conditions"] += 1
                                        continue
                                else:
                                    # Fallback to regular search
                                    poi_candle = None
                                    for k in range(temp_last_idx + 1, len(new_candles)):
                                        if is_fvg_match(new_candles[k], target_fvg_type):
                                            poi_candle = new_candles[k]
                                            break
                                    
                                    if not poi_candle:
                                        #log(f"Processing candle {c_num}: ✗ Could not find {poi_target_role}")
                                        stage_failures["swing_conditions"] += 1
                                        continue
                            else:
                                poi_candle = candle

                            if if_intruder_value == "terminate" and poi_candle and last_swing_candle and option_candle:
                                has_intruder = check_intruder_violation(
                                    option_candle, 
                                    last_swing_candle, 
                                    poi_candle, 
                                    new_candles, 
                                    if_intruder_value
                                )
                                if has_intruder:
                                    #log(f"Processing candle {c_num}: ✗ Intruder violation found")
                                    stage_failures["intruder"] += 1
                                    continue

                            mitigation_found, desc, final_mit_rec = False, "pending entry level", None
                            if poi_candle:
                                is_fvg_poi = is_fvg_family_candle(poi_candle, "fvg")
                                is_fvg_swing_poi = is_fvg_swing_candle(poi_candle)
                                
                                if is_fvg_poi:
                                    # Check FVG mitigation using FVG-specific liquidation fields
                                    mitigation_found = check_fvg_mitigation(poi_candle, new_candles)
                                    if mitigation_found:
                                        liq_candle = get_fvg_liquidation_candle(poi_candle, new_candles)
                                        if liq_candle:
                                            liq_candle["mitigation_candle"] = True
                                            poi_candle["mitigated"] = True
                                            poi_candle["mitigated_by_candle_number"] = liq_candle.get("candle_number")
                                            final_mit_rec = liq_candle.copy()
                                            final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                            desc = "mitigated entry"
                                    else:
                                        # Check for pending entry level
                                        if check_fvg_pending_entry(poi_candle):
                                            poi_candle["pending_entry_level"] = True
                                            desc = "pending entry level"
                                elif is_fvg_swing_poi:
                                    p_type = normalize_swing(poi_candle.get("swing_type", ""))
                                    for m_idx in range(temp_last_idx + 1, len(new_candles)):
                                        mc, is_mit = new_candles[m_idx], False
                                        if "high" in p_type:
                                            if mc.get("high", 0) >= poi_candle.get("high", 0): is_mit = True
                                        elif "low" in p_type:
                                            if mc.get("low", 999999) <= poi_candle.get("low", 999999): is_mit = True
                                        
                                        if is_mit:
                                            mc["mitigation_candle"] = True
                                            poi_candle["mitigated"] = True
                                            poi_candle["mitigated_by_candle_number"] = mc.get("candle_number")
                                            final_mit_rec = mc.copy()
                                            final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                            mitigation_found, desc = True, "mitigated entry"
                                            break
                                else:
                                    p_type = normalize_swing(poi_candle.get("swing_type", ""))
                                    for m_idx in range(temp_last_idx + 1, len(new_candles)):
                                        mc, is_mit = new_candles[m_idx], False
                                        if "high" in p_type:
                                            if mc.get("high", 0) >= poi_candle.get("high", 0): is_mit = True
                                        elif "low" in p_type:
                                            if mc.get("low", 999999) <= poi_candle.get("low", 999999): is_mit = True
                                        
                                        if is_mit:
                                            mc["mitigation_candle"] = True
                                            poi_candle["mitigated"] = True
                                            poi_candle["mitigated_by_candle_number"] = mc.get("candle_number")
                                            final_mit_rec = mc.copy()
                                            final_mit_rec["mitigated_candle_number"] = poi_candle.get("candle_number")
                                            mitigation_found, desc = True, "mitigated entry"
                                            break
                                
                                if not mitigation_found and not poi_candle.get("pending_entry_level"): 
                                    poi_candle["pending_entry_level"] = True
                                    desc = "pending entry level"
                                    #log(f"Processing candle {c_num}: ✓ Pattern found - PENDING ENTRY")
                                else:
                                    #log(f"Processing candle {c_num}: ✓ Pattern found - {desc.upper()}")
                                    log

                            trade_info = {
                                "symbol": sym,
                                "timeframe": tf,
                                "description": desc,
                                "pattern": entry_key,
                                "signal_from": new_filename,
                                "is_main": True,
                                "source_definition": "main_source",
                                "entry": None,
                                "exit": None,
                                "target": None,
                                "order_type": None,
                                "family_linked": True if "fvg" in poi_target_role and "reference_for_family" in locals() and reference_for_family else False
                            }
                            
                            for role in ["entry", "exit", "target"]:
                                r_cfg = record_prices_cfg.get(role, {})
                                source_role_raw = r_cfg.get("record", "").replace("_candle", "")
                                if source_role_raw in ["swing", "swing_candle"]: source_role_raw = "swept"
                                price = get_price_from_cfg(condition_history.get(source_role_raw), r_cfg)
                                trade_info[role] = price
                            
                            ot_cfg = record_prices_cfg.get("order_type_if_entry_swing_is", {})
                            
                            # Determine order type based on POI candle type
                            if is_fvg_poi:
                                # FVG family candle
                                fvg_type = None
                                if is_fvg_match(poi_candle, "fvg_c1"):
                                    fvg_type = "fvg_c1"
                                elif is_fvg_match(poi_candle, "fvg_c3"):
                                    fvg_type = "fvg_c3"
                                elif is_fvg_match(poi_candle, "fvg"):
                                    fvg_type = "fvg"
                                
                                if fvg_type:
                                    fvg_base = get_fvg_base_type(poi_candle, fvg_type)
                                    if fvg_base:
                                        if "resistance" in fvg_base:
                                            trade_info["order_type"] = ot_cfg.get("resistance_fvg")
                                        elif "support" in fvg_base:
                                            trade_info["order_type"] = ot_cfg.get("support_fvg")
                            elif is_fvg_swing_poi:
                                # FVG swing candle (regular swing)
                                if "high" in normalize_swing(poi_candle.get("swing_type", "")):
                                    trade_info["order_type"] = ot_cfg.get("swing_higher_high_or_lower_high")
                                else:
                                    trade_info["order_type"] = ot_cfg.get("swing_lower_low_or_higher_low")
                            
                            if mitigation_found:
                                main_executed_orders.append(trade_info)
                            else:
                                main_pending_orders.append(trade_info)

                            total_entries_found += 1
                            pattern_item = {
                                "pattern_type": entry_key,
                                "swept_candle_number": candle.get("candle_number"),
                                "sweeper_candle_number": ref_sweeper.get("candle_number"),
                                "poi_target_role": poi_target_role,
                                "start_search_with_tool": start_search_with_tool,
                                "subject_drawing": subject_cfg.get("drawing", {}),
                                "option_drawing": entry_spec.get("option_tool", {}),
                                "swings_drawing": {k: v.get("drawing", {}) for k, v in before_entry.items() if k.startswith("swing_")},
                                "is_main": True,
                                "option_condition": option_condition,
                                "family_linked": True if "fvg" in poi_target_role and "reference_for_family" in locals() and reference_for_family else False
                            }

                            p_keys = []
                            roles_to_check = [("swept", candle), ("sweeper", ref_sweeper), ("option", option_candle)]
                            for skey in sorted(condition_history.keys()):
                                if skey not in ["swept", "sweeper", "option", "subject", "swing", "swing_candle"]: 
                                    roles_to_check.append((skey, condition_history[skey]))

                            for role_name, rc in roles_to_check:
                                if not rc: continue
                                c_copy = rc.copy()
                                c_copy[role_name] = True
                                if role_name == "option" or role_name.startswith("swing_"):
                                    is_fvg_candle = is_fvg_family_candle(rc, "fvg")
                                    is_fvg_swing_candle_check = is_fvg_swing_candle(rc)
                                    if is_fvg_candle:
                                        liq_sw = find_liquidation_for_fvg(rc, new_candles)
                                    elif is_fvg_swing_candle_check:
                                        liq_sw = find_liquidation_for_candle(rc, new_candles)
                                    else:
                                        liq_sw = find_liquidation_for_candle(rc, new_candles)
                                    
                                    if liq_sw:
                                        c_copy["is_liquidated"] = True
                                        c_copy["liquidated_by_candle_number"] = liq_sw.get("candle_number")
                                        sw_rec = liq_sw.copy()
                                        sw_rec["is_liquidity"] = True
                                        sw_rec["liquidated_candle_number"] = rc.get("candle_number")
                                        sw_rec[f"{role_name}_sweeper"] = True
                                        p_keys.extend([c_copy, sw_rec]); continue
                                p_keys.append(c_copy)

                            if final_mit_rec: p_keys.append(final_mit_rec)
                            pattern_item["pattern_keys"] = p_keys
                            main_items.append(pattern_item)
                        
                        # Apply data selection filtering for main
                        filtered_main_items = filter_data_selection(
                            main_items, start_search_with, data_selection_value, source_candles
                        )
                        
                        log(f"  Stage analysis for {sym}/{tf}:")
                        log(f"    Total candles checked: {total_candles_checked}")
                        log(f"    Stage failures breakdown:")
                        log(f"      start_search_with: {stage_failures['start_search_with']} candles failed")
                        log(f"      option: {stage_failures['option']} candles failed")
                        log(f"      find_entry_after: {stage_failures['find_entry_after']} candles failed")
                        log(f"      swing_conditions: {stage_failures['swing_conditions']} candles failed")
                        log(f"      intruder: {stage_failures['intruder']} candles failed")
                        log(f"    Patterns found: {len(main_items)} (before data selection)")
                        log(f"    Patterns after data_selection='{data_selection_value}': {len(filtered_main_items)}")
                        
                        if len(filtered_main_items) > 0:
                            log(f"    ✓ Found {len(filtered_main_items)} valid patterns")
                        else:
                            log(f"    ✗ No patterns found after all filters")
                        
                        # ─── PROCESS SECOND SOURCE (CONDITIONALLY) ─────────────────────
                        second_source_results = None
                        if should_process_second_source:
                            second_source_results = process_second_source_independently(
                                entry_spec, config_data, entry_key, sym, tf, dev_output_dir, source_def_name
                            )
                        
                        # ─── SAVE ALL RESULTS TO CONFIG.JSON ─────────────────────
                        entry_output_dir = os.path.join(os.path.dirname(base_folder), "developers", broker_name, new_filename)
                        os.makedirs(entry_output_dir, exist_ok=True)
                        
                        config_structure = {
                            "metadata": {
                                "entry_name": entry_key,
                                "broker": broker_name,
                                "symbol": sym,
                                "base_timeframe": tf,
                                "source_definition": source_def_name,
                                "processed_at": datetime.now(lagos_tz).isoformat(),
                                "save_config": {
                                    "chart_folder": chart_folder,
                                    "pending_folder": pending_folder,
                                    "mitigated_folder": mitigated_folder
                                },
                                "second_source_processing": "enabled" if should_process_second_source else "disabled",
                                "process_second_source_files": process_second_source,
                                "option_condition": option_condition,
                                "stage_analysis": {
                                    "total_candles_checked": total_candles_checked,
                                    "stage_failures": stage_failures,
                                    "patterns_before_selection": len(main_items),
                                    "patterns_after_selection": len(filtered_main_items)
                                }
                            },
                            "main_source": {},
                            "second_source_sources": {},
                            "source_candles": [c.copy() for c in source_candles]
                        }
                        
                        if filtered_main_items:
                            config_structure["main_source"] = {
                                "patterns": filtered_main_items,
                                "pending_orders": main_pending_orders,
                                "executed_orders": main_executed_orders,
                                "patterns_count": len(filtered_main_items),
                                "pending_count": len(main_pending_orders),
                                "executed_count": len(main_executed_orders)
                            }
                        
                        if second_source_results:
                            for ss_result in second_source_results:
                                if ss_result["filtered_patterns"] > 0:
                                    total_second_source_entries += ss_result["filtered_patterns"]
                                    
                                    second_source_key = ss_result["key_id"]
                                    safe_key = second_source_key.replace("/", "_").replace("\\", "_")
                                    
                                    config_structure["second_source_sources"][safe_key] = {
                                        "original_key": second_source_key,
                                        "patterns": ss_result["items"],
                                        "pending_orders": ss_result["pending_orders"],
                                        "executed_orders": ss_result["executed_orders"],
                                        "patterns_count": ss_result["filtered_patterns"],
                                        "pending_count": len(ss_result["pending_orders"]),
                                        "executed_count": len(ss_result["executed_orders"]),
                                        "candle_count": ss_result["candle_count"],
                                        "chart_files": ss_result["chart_files"]
                                    }
                        
                        config_path = os.path.join(entry_output_dir, "config.json")
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_structure, f, indent=4, default=str)
                        
                        log(f"  ✓ Saved all data to config.json")
                        
                        # ─── SAVE COLLECTIVE PRICE RECORDS ─────────────────────
                        pending_dir = os.path.join(entry_output_dir, pending_folder)
                        mitigated_dir = os.path.join(entry_output_dir, mitigated_folder)
                        os.makedirs(pending_dir, exist_ok=True)
                        os.makedirs(mitigated_dir, exist_ok=True)
                        
                        if main_pending_orders:
                            main_pending_file = os.path.join(pending_dir, f"main_{pending_folder}.json")
                            with open(main_pending_file, 'w', encoding='utf-8') as f:
                                json.dump(main_pending_orders, f, indent=4, default=str)
                        
                        if main_executed_orders:
                            main_executed_file = os.path.join(mitigated_dir, f"main_{mitigated_folder}.json")
                            with open(main_executed_file, 'w', encoding='utf-8') as f:
                                json.dump(main_executed_orders, f, indent=4, default=str)
                        
                        if second_source_results:
                            second_source_pending_orders_all = []
                            second_source_executed_orders_all = []
                            
                            for ss_result in second_source_results:
                                if ss_result["pending_orders"]:
                                    second_source_pending_orders_all.extend(ss_result["pending_orders"])
                                if ss_result["executed_orders"]:
                                    second_source_executed_orders_all.extend(ss_result["executed_orders"])
                            
                            if second_source_pending_orders_all:
                                second_source_pending_file = os.path.join(pending_dir, f"second_source_{pending_folder}.json")
                                with open(second_source_pending_file, 'w', encoding='utf-8') as f:
                                    json.dump(second_source_pending_orders_all, f, indent=4, default=str)
                            
                            if second_source_executed_orders_all:
                                second_source_executed_file = os.path.join(mitigated_dir, f"second_source_{mitigated_folder}.json")
                                with open(second_source_executed_file, 'w', encoding='utf-8') as f:
                                    json.dump(second_source_executed_orders_all, f, indent=4, default=str)
                        
                        # ─── SAVE CHARTS ──────
                        chart_dir = os.path.join(entry_output_dir, chart_folder, sym)
                        os.makedirs(chart_dir, exist_ok=True)
                        
                        if filtered_main_items:
                            src_png_path = os.path.join(dev_output_dir, source_png_filename)
                            if not os.path.exists(src_png_path): 
                                src_png_path = os.path.join(dev_output_dir, f"{source_key}.png")
                            
                            if os.path.exists(src_png_path):
                                img = cv2.imread(src_png_path)
                                if img is not None:
                                    overlay, h, w = img.copy(), img.shape[0], img.shape[1]
                                    for item in filtered_main_items:
                                        pk_list = item.get("pattern_keys", [])
                                        mit_c = next((pk for pk in pk_list if pk.get("mitigation_candle")), None)
                                        
                                        start_search_tool_cfg = item.get("start_search_with_tool", {})
                                        if start_search_tool_cfg:
                                            swept_candle_num = item.get("swept_candle_number")
                                            swept_candle = next((pk for pk in pk_list if pk.get("candle_number") == swept_candle_num and pk.get("swept")), None)
                                            if swept_candle:
                                                tool = start_search_tool_cfg.get("tool")
                                                ct = swept_candle.get("candle_top")
                                                cb = swept_candle.get("candle_bottom")
                                                cl = swept_candle.get("candle_left")
                                                if all(v is not None for v in [ct, cb, cl]):
                                                    is_fvg_swept = is_fvg_family_candle(swept_candle, "fvg")
                                                    is_fvg_swing_swept = is_fvg_swing_candle(swept_candle)
                                                    if is_fvg_swept:
                                                        liq_sw = find_liquidation_for_fvg(swept_candle, new_candles)
                                                    elif is_fvg_swing_swept:
                                                        liq_sw = find_liquidation_for_candle(swept_candle, new_candles)
                                                    else:
                                                        liq_sw = find_liquidation_for_candle(swept_candle, new_candles)
                                                    
                                                    rb = int(w)
                                                    
                                                    if start_search_tool_cfg.get("stop_at_its") == "liquidator" and liq_sw:
                                                        liq_candle_left = liq_sw.get("candle_left")
                                                        if liq_candle_left is not None:
                                                            rb = int(liq_candle_left)
                                                    
                                                    # Determine which price to use based on candle type
                                                    st = normalize_swing(swept_candle.get("swing_type", ""))
                                                    is_fvg_candle = is_fvg_family_candle(swept_candle, "fvg")
                                                    
                                                    if is_fvg_candle:
                                                        # FVG family candle - use resistance_fvg or support_fvg
                                                        fvg_type = None
                                                        if is_fvg_match(swept_candle, "fvg_c1"):
                                                            fvg_type = "fvg_c1"
                                                        elif is_fvg_match(swept_candle, "fvg_c3"):
                                                            fvg_type = "fvg_c3"
                                                        elif is_fvg_match(swept_candle, "fvg"):
                                                            fvg_type = "fvg"
                                                        
                                                        fvg_base = get_fvg_base_type(swept_candle, fvg_type) if fvg_type else None
                                                        
                                                        if fvg_base:
                                                            if "resistance" in fvg_base:
                                                                field = start_search_tool_cfg.get("resistance_fvg", "high_price")
                                                                ly = cb if field == "low_price" else ct
                                                            elif "support" in fvg_base:
                                                                field = start_search_tool_cfg.get("support_fvg", "low_price")
                                                                ly = ct if field == "high_price" else cb
                                                    elif is_fvg_swing_swept:
                                                        # FVG swing candle (regular swing)
                                                        if "high" in st:
                                                            field = start_search_tool_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                            ly = cb if field == "low_price" else ct
                                                        elif "low" in st:
                                                            field = start_search_tool_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                            ly = ct if field == "high_price" else cb
                                                    else:
                                                        # Regular swing candle
                                                        if "high" in st:
                                                            field = start_search_tool_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                            ly = cb if field == "low_price" else ct
                                                        elif "low" in st:
                                                            field = start_search_tool_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                            ly = ct if field == "high_price" else cb
                                                    
                                                    if tool == "box": 
                                                        cv2.rectangle(overlay, (int(cl), int(ct)), (rb, int(cb)), (255, 0, 0), -1)
                                                    elif tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                        if ly is not None: 
                                                            draw_line_logic(overlay, cl, rb, ly, tool, color=(255, 0, 0))
                                        
                                        for role_prefix in ["option", "swing_"]:
                                            role_keys = [pk for pk in pk_list if any(k.startswith(role_prefix) and pk.get(k) is True for k in pk.keys())]
                                            for r_candle in role_keys:
                                                act_r = next((k for k in r_candle.keys() if (k.startswith(role_prefix) and r_candle[k] is True)), None)
                                                d_cfg = item.get("option_drawing", {}) if act_r == "option" else item.get("swings_drawing", {}).get(act_r, {})
                                                tool, ct, cb, cl = d_cfg.get("tool"), r_candle.get("candle_top"), r_candle.get("candle_bottom"), r_candle.get("candle_left")
                                                if any(v is None for v in [ct, cb, cl]): continue
                                                rb = int(w)
                                                
                                                # Check for FVG liquidation for drawing stop
                                                is_fvg_candle_role = is_fvg_family_candle(r_candle, "fvg")
                                                if d_cfg.get("stop_at_its") == "liquidator":
                                                    if is_fvg_candle_role:
                                                        liq_sw = find_liquidation_for_fvg(r_candle, new_candles)
                                                    else:
                                                        liq_sw = find_liquidation_for_candle(r_candle, new_candles)
                                                    
                                                    if liq_sw:
                                                        liq_candle_left = liq_sw.get("candle_left")
                                                        if liq_candle_left is not None:
                                                            rb = int(liq_candle_left)
                                                
                                                # Determine which price to use based on candle type
                                                st = normalize_swing(r_candle.get("swing_type", ""))
                                                is_fvg_candle_role = is_fvg_family_candle(r_candle, "fvg")
                                                is_fvg_swing_candle_role = is_fvg_swing_candle(r_candle)
                                                
                                                if is_fvg_candle_role:
                                                    # FVG family candle - use resistance_fvg or support_fvg
                                                    fvg_type = None
                                                    if is_fvg_match(r_candle, "fvg_c1"):
                                                        fvg_type = "fvg_c1"
                                                    elif is_fvg_match(r_candle, "fvg_c3"):
                                                        fvg_type = "fvg_c3"
                                                    elif is_fvg_match(r_candle, "fvg"):
                                                        fvg_type = "fvg"
                                                    
                                                    fvg_base = get_fvg_base_type(r_candle, fvg_type) if fvg_type else None
                                                    
                                                    if fvg_base:
                                                        if "resistance" in fvg_base:
                                                            field = d_cfg.get("resistance_fvg", "high_price")
                                                            ly = cb if field == "low_price" else ct
                                                        elif "support" in fvg_base:
                                                            field = d_cfg.get("support_fvg", "low_price")
                                                            ly = ct if field == "high_price" else cb
                                                elif is_fvg_swing_candle_role:
                                                    # FVG swing candle (regular swing)
                                                    if "high" in st:
                                                        field = d_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                        ly = cb if field == "low_price" else ct
                                                    elif "low" in st:
                                                        field = d_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                        ly = ct if field == "high_price" else cb
                                                else:
                                                    # Regular swing candle
                                                    if "high" in st:
                                                        field = d_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                        ly = cb if field == "low_price" else ct
                                                    elif "low" in st:
                                                        field = d_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                        ly = ct if field == "high_price" else cb
                                                
                                                if tool == "box": 
                                                    cv2.rectangle(overlay, (int(cl), int(ct)), (rb, int(cb)), (0, 0, 0), -1)
                                                elif tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                    if ly is not None: 
                                                        draw_line_logic(overlay, cl, rb, ly, tool)

                                        poi_target_role_clean = item.get("poi_target_role", "swept")
                                        if any(x in poi_target_role_clean for x in ["swing_candle", "swept_candle"]): search_flag = "swept"
                                        elif "swing_" in poi_target_role_clean: search_flag = poi_target_role_clean.replace("_candle", "")
                                        else: search_flag = poi_target_role_clean
                                        
                                        poi_draw = next((pk for pk in pk_list if pk.get(search_flag)), None)
                                        if poi_draw:
                                            sd_cfg = item.get("subject_drawing", {})
                                            s_tool, sct, scb, scl = sd_cfg.get("tool", "box"), poi_draw.get("candle_top"), poi_draw.get("candle_bottom"), poi_draw.get("candle_left")
                                            s_right = int(mit_c.get("candle_left")) if mit_c and mit_c.get("candle_left") else int(w)
                                            if all(v is not None for v in [sct, scb, scl]):
                                                s_st = normalize_swing(poi_draw.get("swing_type", ""))
                                                is_fvg_poi_draw = is_fvg_family_candle(poi_draw, "fvg")
                                                is_fvg_swing_poi_draw = is_fvg_swing_candle(poi_draw)
                                                
                                                if is_fvg_poi_draw:
                                                    # FVG family candle - use resistance_fvg or support_fvg
                                                    fvg_type = None
                                                    if is_fvg_match(poi_draw, "fvg_c1"):
                                                        fvg_type = "fvg_c1"
                                                    elif is_fvg_match(poi_draw, "fvg_c3"):
                                                        fvg_type = "fvg_c3"
                                                    elif is_fvg_match(poi_draw, "fvg"):
                                                        fvg_type = "fvg"
                                                    
                                                    fvg_base = get_fvg_base_type(poi_draw, fvg_type) if fvg_type else None
                                                    
                                                    if fvg_base:
                                                        if "resistance" in fvg_base:
                                                            field = sd_cfg.get("resistance_fvg", "high_price")
                                                            s_ly = scb if field == "low_price" else sct
                                                        elif "support" in fvg_base:
                                                            field = sd_cfg.get("support_fvg", "low_price")
                                                            s_ly = sct if field == "high_price" else scb
                                                elif is_fvg_swing_poi_draw:
                                                    # FVG swing candle (regular swing)
                                                    if "high" in s_st:
                                                        field = sd_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                        s_ly = scb if field == "low_price" else sct
                                                    elif "low" in s_st:
                                                        field = sd_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                        s_ly = sct if field == "high_price" else scb
                                                else:
                                                    # Regular swing candle
                                                    if "high" in s_st:
                                                        field = sd_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                        s_ly = scb if field == "low_price" else sct
                                                    elif "low" in s_st:
                                                        field = sd_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                        s_ly = sct if field == "high_price" else scb
                                                
                                                if s_tool == "box":
                                                    cv2.rectangle(overlay, (int(scl), int(sct)), (s_right, int(scb)), (0, 0, 0), -1)
                                                elif s_tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                    if s_ly is not None: 
                                                        draw_line_logic(overlay, scl, s_right, s_ly, s_tool)

                                    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                                    chart_path = os.path.join(chart_dir, f"{tf}__main.png")
                                    cv2.imwrite(chart_path, img)
                                    log(f"  ✓ Saved main chart to {chart_path}")
                        
                        if second_source_results:
                            for ss_result in second_source_results:
                                if ss_result["filtered_patterns"] > 0 and ss_result["chart_files"]:
                                    chart_file = ss_result["chart_files"][0]
                                    chart_path = os.path.join(dev_output_dir, chart_file)
                                    
                                    if os.path.exists(chart_path):
                                        second_source_key = ss_result["key_id"]
                                        safe_key = second_source_key.replace("/", "_").replace("\\", "_")
                                        
                                        new_chart_name = f"{tf}_{safe_key}.png"
                                        new_chart_path = os.path.join(chart_dir, new_chart_name)
                                        shutil.copy2(chart_path, new_chart_path)
                                        
                                        img = cv2.imread(chart_path)
                                        if img is not None:
                                            overlay, h, w = img.copy(), img.shape[0], img.shape[1]
                                            for item in ss_result["items"]:
                                                pk_list = item.get("pattern_keys", [])
                                                mit_c = next((pk for pk in pk_list if pk.get("mitigation_candle")), None)
                                                
                                                start_search_tool_cfg = item.get("start_search_with_tool", {})
                                                if start_search_tool_cfg:
                                                    swept_candle_num = item.get("swept_candle_number")
                                                    swept_candle = next((pk for pk in pk_list if pk.get("candle_number") == swept_candle_num and pk.get("swept")), None)
                                                    if swept_candle:
                                                        tool = start_search_tool_cfg.get("tool")
                                                        ct = swept_candle.get("candle_top")
                                                        cb = swept_candle.get("candle_bottom")
                                                        cl = swept_candle.get("candle_left")
                                                        if all(v is not None for v in [ct, cb, cl]):
                                                            is_fvg_swept = is_fvg_family_candle(swept_candle, "fvg")
                                                            is_fvg_swing_swept = is_fvg_swing_candle(swept_candle)
                                                            if is_fvg_swept:
                                                                liq_sw = find_liquidation_for_fvg(swept_candle, ss_result["candles"])
                                                            elif is_fvg_swing_swept:
                                                                liq_sw = find_liquidation_for_candle(swept_candle, ss_result["candles"])
                                                            else:
                                                                liq_sw = find_liquidation_for_candle(swept_candle, ss_result["candles"])
                                                            
                                                            rb = int(w)
                                                            
                                                            if start_search_tool_cfg.get("stop_at_its") == "liquidator" and liq_sw:
                                                                liq_candle_left = liq_sw.get("candle_left")
                                                                if liq_candle_left is not None:
                                                                    rb = int(liq_candle_left)
                                                            
                                                            # Determine which price to use based on candle type
                                                            st = normalize_swing(swept_candle.get("swing_type", ""))
                                                            is_fvg_candle = is_fvg_family_candle(swept_candle, "fvg")
                                                            
                                                            if is_fvg_candle:
                                                                # FVG family candle - use resistance_fvg or support_fvg
                                                                fvg_type = None
                                                                if is_fvg_match(swept_candle, "fvg_c1"):
                                                                    fvg_type = "fvg_c1"
                                                                elif is_fvg_match(swept_candle, "fvg_c3"):
                                                                    fvg_type = "fvg_c3"
                                                                elif is_fvg_match(swept_candle, "fvg"):
                                                                    fvg_type = "fvg"
                                                                
                                                                fvg_base = get_fvg_base_type(swept_candle, fvg_type) if fvg_type else None
                                                                
                                                                if fvg_base:
                                                                    if "resistance" in fvg_base:
                                                                        field = start_search_tool_cfg.get("resistance_fvg", "high_price")
                                                                        ly = cb if field == "low_price" else ct
                                                                    elif "support" in fvg_base:
                                                                        field = start_search_tool_cfg.get("support_fvg", "low_price")
                                                                        ly = ct if field == "high_price" else cb
                                                            elif is_fvg_swing_swept:
                                                                # FVG swing candle (regular swing)
                                                                if "high" in st:
                                                                    field = start_search_tool_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                    ly = cb if field == "low_price" else ct
                                                                elif "low" in st:
                                                                    field = start_search_tool_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                    ly = ct if field == "high_price" else cb
                                                            else:
                                                                # Regular swing candle
                                                                if "high" in st:
                                                                    field = start_search_tool_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                    ly = cb if field == "low_price" else ct
                                                                elif "low" in st:
                                                                    field = start_search_tool_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                    ly = ct if field == "high_price" else cb
                                                            
                                                            if tool == "box": 
                                                                cv2.rectangle(overlay, (int(cl), int(ct)), (rb, int(cb)), (255, 0, 0), -1)
                                                            elif tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                                if ly is not None: 
                                                                    draw_line_logic(overlay, cl, rb, ly, tool, color=(255, 0, 0))
                                                
                                                for role_prefix in ["option", "swing_"]:
                                                    role_keys = [pk for pk in pk_list if any(k.startswith(role_prefix) and pk.get(k) is True for k in pk.keys())]
                                                    for r_candle in role_keys:
                                                        act_r = next((k for k in r_candle.keys() if (k.startswith(role_prefix) and r_candle[k] is True)), None)
                                                        d_cfg = item.get("option_drawing", {}) if act_r == "option" else item.get("swings_drawing", {}).get(act_r, {})
                                                        tool, ct, cb, cl = d_cfg.get("tool"), r_candle.get("candle_top"), r_candle.get("candle_bottom"), r_candle.get("candle_left")
                                                        if any(v is not None for v in [ct, cb, cl]): continue
                                                        rb = int(w)
                                                        
                                                        # Check for FVG liquidation for drawing stop
                                                        is_fvg_candle_role = is_fvg_family_candle(r_candle, "fvg")
                                                        if d_cfg.get("stop_at_its") == "liquidator":
                                                            if is_fvg_candle_role:
                                                                liq_sw = find_liquidation_for_fvg(r_candle, ss_result["candles"])
                                                            else:
                                                                liq_sw = find_liquidation_for_candle(r_candle, ss_result["candles"])
                                                            
                                                            if liq_sw:
                                                                liq_candle_left = liq_sw.get("candle_left")
                                                                if liq_candle_left is not None:
                                                                    rb = int(liq_candle_left)
                                                        
                                                        # Determine which price to use based on candle type
                                                        st = normalize_swing(r_candle.get("swing_type", ""))
                                                        is_fvg_candle_role = is_fvg_family_candle(r_candle, "fvg")
                                                        is_fvg_swing_candle_role = is_fvg_swing_candle(r_candle)
                                                        
                                                        if is_fvg_candle_role:
                                                            # FVG family candle - use resistance_fvg or support_fvg
                                                            fvg_type = None
                                                            if is_fvg_match(r_candle, "fvg_c1"):
                                                                fvg_type = "fvg_c1"
                                                            elif is_fvg_match(r_candle, "fvg_c3"):
                                                                fvg_type = "fvg_c3"
                                                            elif is_fvg_match(r_candle, "fvg"):
                                                                fvg_type = "fvg"
                                                            
                                                            fvg_base = get_fvg_base_type(r_candle, fvg_type) if fvg_type else None
                                                            
                                                            if fvg_base:
                                                                if "resistance" in fvg_base:
                                                                    field = d_cfg.get("resistance_fvg", "high_price")
                                                                    ly = cb if field == "low_price" else ct
                                                                elif "support" in fvg_base:
                                                                    field = d_cfg.get("support_fvg", "low_price")
                                                                    ly = ct if field == "high_price" else cb
                                                        elif is_fvg_swing_candle_role:
                                                            # FVG swing candle (regular swing)
                                                            if "high" in st:
                                                                field = d_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                ly = cb if field == "low_price" else ct
                                                            elif "low" in st:
                                                                field = d_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                ly = ct if field == "high_price" else cb
                                                        else:
                                                            # Regular swing candle
                                                            if "high" in st:
                                                                field = d_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                ly = cb if field == "low_price" else ct
                                                            elif "low" in st:
                                                                field = d_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                ly = ct if field == "high_price" else cb
                                                        
                                                        if tool == "box": 
                                                            cv2.rectangle(overlay, (int(cl), int(ct)), (rb, int(cb)), (0, 0, 255), -1)
                                                        elif tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                            if ly is not None: 
                                                                draw_line_logic(overlay, cl, rb, ly, tool, color=(0, 0, 255))

                                                poi_target_role_clean = item.get("poi_target_role", "swept")
                                                if any(x in poi_target_role_clean for x in ["swing_candle", "swept_candle"]): search_flag = "swept"
                                                elif "swing_" in poi_target_role_clean: search_flag = poi_target_role_clean.replace("_candle", "")
                                                else: search_flag = poi_target_role_clean
                                                
                                                poi_draw = next((pk for pk in pk_list if pk.get(search_flag)), None)
                                                if poi_draw:
                                                    sd_cfg = item.get("subject_drawing", {})
                                                    s_tool, sct, scb, scl = sd_cfg.get("tool", "box"), poi_draw.get("candle_top"), poi_draw.get("candle_bottom"), poi_draw.get("candle_left")
                                                    s_right = int(mit_c.get("candle_left")) if mit_c and mit_c.get("candle_left") else int(w)
                                                    if all(v is not None for v in [sct, scb, scl]):
                                                        s_st = normalize_swing(poi_draw.get("swing_type", ""))
                                                        is_fvg_poi_draw = is_fvg_family_candle(poi_draw, "fvg")
                                                        is_fvg_swing_poi_draw = is_fvg_swing_candle(poi_draw)
                                                        
                                                        if is_fvg_poi_draw:
                                                            # FVG family candle - use resistance_fvg or support_fvg
                                                            fvg_type = None
                                                            if is_fvg_match(poi_draw, "fvg_c1"):
                                                                fvg_type = "fvg_c1"
                                                            elif is_fvg_match(poi_draw, "fvg_c3"):
                                                                fvg_type = "fvg_c3"
                                                            elif is_fvg_match(poi_draw, "fvg"):
                                                                fvg_type = "fvg"
                                                            
                                                            fvg_base = get_fvg_base_type(poi_draw, fvg_type) if fvg_type else None
                                                            
                                                            if fvg_base:
                                                                if "resistance" in fvg_base:
                                                                    field = sd_cfg.get("resistance_fvg", "high_price")
                                                                    s_ly = scb if field == "low_price" else sct
                                                                elif "support" in fvg_base:
                                                                    field = sd_cfg.get("support_fvg", "low_price")
                                                                    s_ly = sct if field == "high_price" else scb
                                                        elif is_fvg_swing_poi_draw:
                                                            # FVG swing candle (regular swing)
                                                            if "high" in s_st:
                                                                field = sd_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                s_ly = scb if field == "low_price" else sct
                                                            elif "low" in s_st:
                                                                field = sd_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                s_ly = sct if field == "high_price" else scb
                                                        else:
                                                            # Regular swing candle
                                                            if "high" in s_st:
                                                                field = sd_cfg.get("swing_higher_high_or_lower_high", "high_price")
                                                                s_ly = scb if field == "low_price" else sct
                                                            elif "low" in s_st:
                                                                field = sd_cfg.get("swing_lower_low_or_higher_low", "low_price")
                                                                s_ly = sct if field == "high_price" else scb
                                                        
                                                        if s_tool == "box":
                                                            cv2.rectangle(overlay, (int(scl), int(sct)), (s_right, int(scb)), (0, 255, 0), -1)
                                                        elif s_tool in ["horizontal_line", "dashed_horizontal_line"]:
                                                            if s_ly is not None: 
                                                                draw_line_logic(overlay, scl, s_right, s_ly, s_tool, color=(0, 255, 0))

                                            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                                            chart_path = os.path.join(chart_dir, f"{tf}_{safe_key}.png")
                                            cv2.imwrite(chart_path, img)
                                
                                elif ss_result["filtered_patterns"] == 0:
                                    log
                        
                        log(f"✓ Entry {entry_key} completed: {len(filtered_main_items)} main patterns")
                        if should_process_second_source and second_source_results:
                            second_source_count = len(second_source_results)
                            log(f"  second_source sources: {second_source_count} sources processed")
                        elif should_process_second_source:
                            log(f"  second_source sources: 0 sources processed (no patterns found)")
                        
                except Exception as e:
                    log(f"[{sym}|{tf}|{entry_key}] ERROR: {str(e)}", "ERROR")

    log(f"--- FINISHED ENTRY POI ANALYSIS ---")
    log(f"Total main patterns: {total_entries_found}")
    log(f"Total second_source source patterns: {total_second_source_entries}")
    
    return f"Success: Found {total_entries_found} main patterns and {total_second_source_entries} second_source patterns with family linking FVG support and swing-level intruder checking." 

def clear_symbol_folders(broker_name):
    """
    Identifies valid symbols from the source directory and deletes only those 
    matching folders in the developer output directory.
    """
    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    
    if not cfg:
        print(f"[{broker_name}] Error: Broker not in dictionary.")
        return False

    base_folder = cfg.get("BASE_FOLDER")
    # This is where your outputs live
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))

    if not os.path.exists(base_folder):
        print(f"[{broker_name}] Source folder missing: {base_folder}")
        return False

    # 1. Identify valid symbols by looking at the source directory
    # (Matches the logic: for sym in sorted(os.listdir(base_folder)))
    valid_symbols = [
        sym for sym in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, sym))
    ]

    deleted_count = 0
    
    # 2. Only delete folders in 'developers' that are actually symbols
    for sym in valid_symbols:
        target_path = os.path.join(dev_output_base, sym)
        
        if os.path.exists(target_path) and os.path.isdir(target_path):
            try:
                shutil.rmtree(target_path)
                deleted_count += 1
            except Exception as e:
                print("")
    return True

def single():  
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return


    broker_names = sorted(dev_dict.keys()) 
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:

        # STEP 2: Higher Highs & lower lows
        hh_ll_results = pool.map(entry_point_of_interest_condition, broker_names)
        for r in hh_ll_results: print(r)

def main():
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return

    broker_names = sorted(dev_dict.keys())
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:
        print("\n[STEP 2] Running Higher Highs & lower lows Analysis...")
        hh_ll_results = pool.map(higher_highs_lower_lows, broker_names)
        for r in hh_ll_results: print(r)


        print("\n[STEP 3] Running Lower Highs & Lower Lows Analysis...")
        lh_ll_results = pool.map(lower_highs_higher_lows, broker_names)
        for r in lh_ll_results: print(r)


        print("\n[STEP 4] Running Fair Value Gap Analysis...")
        fvg_results = pool.map(fair_value_gaps, broker_names)
        for r in fvg_results: print(r)
        print("\n[STEP 4] Running Fair Value Gap Analysis...")
        fvg_results = pool.map(fvg_higherhighsandlowerlows, broker_names)
        for r in fvg_results: print(r)
        


        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(directional_bias, broker_names)
        for r in db_results: print(r)


        print("\n[STEP 6] Running Timeframes Communication Analysis...")
        tf_comm_results = pool.map(timeframes_communication, broker_names)
        for r in tf_comm_results: print(r)

        hh_ll_results = pool.map(receiver_comm_higher_highs_lower_lows, broker_names)
        for r in hh_ll_results: print(r)

        hh_ll_results = pool.map(liquidity_candles, broker_names)
        for r in hh_ll_results: print(r)

        hh_ll_results = pool.map(entry_point_of_interest_condition, broker_names)
        for r in hh_ll_results: print(r)


        

    print("\n[SUCCESS] All tasks completed.")

if __name__ == "__main__":
    single()




                



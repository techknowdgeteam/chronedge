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

def get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base, 
                       receiver_tf=None, target=None):
    # Define the developer output root
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
    
    # Source: Where raw data currently lives
    source_json = os.path.join(base_folder, sym, tf, "candlesdetails", f"{direction}_{bars}.json")
    source_chart = os.path.join(base_folder, sym, tf, f"chart_{bars}.png")
    
    # New: Full bars source
    full_bars_source = os.path.join(base_folder, sym, tf, "candlesdetails", "newest_oldest.json")
    
    # Output: Where the results will be saved
    output_dir = os.path.join(dev_output_base, sym, tf)
    output_json = os.path.join(output_dir, output_filename_base)
    output_chart = os.path.join(output_dir, output_filename_base.replace(".json", ".png"))

    # Communication Paths (Constructor Logic)
    comm_paths = {}
    if receiver_tf and target:
        base_name = output_filename_base.replace(".json", "")
        # Format: {receiver_tf}_{base_output_name}_{target}_{sender_tf}
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
        "comm_paths": comm_paths  # New constructed paths
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
    label_position="auto"       
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

    # 2. Draw Objects (Arrows/Shapes) ONLY if is_marked is True
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

    # If marked, text goes beyond the arrow/shape.
    # If NOT marked (just numbering), we use a tiny 4-pixel gap.
    if is_marked:
        is_vertical_obj = object_type in ["arrow", "reverse_arrow"]
        reach = (shaft_length + head_size + 4) if is_vertical_obj else 14
    else:
        reach = 4

    if place_at_high:
        # Number sits directly above the top wick
        base_text_y = tip_y - reach
    else:
        # Number sits directly below the bottom wick (+10 accounts for text height)
        base_text_y = tip_y + reach + 10

    # Draw the Higher/Lower text (HH, LL, etc.)
    if custom_text:
        (tw, th), _ = cv2.getTextSize(custom_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.putText(img, custom_text, (cx - tw // 2, int(base_text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, arrow_color, text_thickness)
        # Shift the number slightly so it doesn't overlap the HH/LL text
        c_num_y = (base_text_y - 15) if place_at_high else (base_text_y + 15)
    else:
        c_num_y = base_text_y

    # Draw the candle number (c_num)
    if c_num is not None:
        cv2.putText(img, str(c_num), (cx - 8, int(c_num_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2) # Shadow for visibility
        cv2.putText(img, str(c_num), (cx - 8, int(c_num_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1) # White text


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
        if not raw:
            return None, False
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]:
            return "arrow", False
        if raw in ["doublearrow", "doublearrows"]:
            return "arrow", True
        if raw in ["reverse_arrow", "reversearrow"]:
            return "reverse_arrow", False
        if raw in ["reverse_doublearrow", "reverse_doublearrows"]:
            return "reverse_arrow", True
        if raw in ["rightarrow", "right_arrow"]:
            return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]:
            return "leftarrow", False
        if "dot" in raw:
            return "dot", False
        return raw, False

    for config_key, lhll_cfg in matching_configs:
        bars = lhll_cfg.get("BARS", 101)
        output_filename_base = lhll_cfg.get("filename", "lowers.json")
        
        # Set default to 'new_old' to ensure left-to-right processing
        direction = lhll_cfg.get("read_candles_from", "new_old")
        
        neighbor_left = lhll_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = lhll_cfg.get("NEIGHBOR_RIGHT", 5)

        label_cfg = lhll_cfg.get("label", {})
        # Note: These use the 'lower' variables now
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
                    
                    # FORCED Left-to-Right Sorting:
                    # Removed 'reverse' parameter to ensure index 0 is always the leftmost (oldest) candle.
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

                    swing_results = []
                    n = len(data)
                    for i in range(neighbor_left, n - neighbor_right):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        l_h = [d['high'] for d in data[i-neighbor_left:i]]
                        l_l = [d['low'] for d in data[i-neighbor_left:i]]
                        r_h = [d['high'] for d in data[i+1:i+neighbor_right+1]]
                        r_l = [d['low'] for d in data[i+1:i+neighbor_right+1]]

                        # Lower High (LH): Current high is lower than previous highs
                        # Lower Low (LL): Current low is lower than previous lows
                        is_lh = len(l_h) > 0 and len(r_h) > 0 and curr_h < max(l_h) and curr_h > max(r_h) # Logic varies based on your specific strategy definition
                        # Typical LH definition: curr_h is a peak (higher than neighbors) but lower than the PREVIOUS peak. 
                        # Using your specific neighbor logic:
                        is_peak = curr_h > max(l_h) and curr_h > max(r_h)
                        is_valley = curr_l < min(l_l) and curr_l < min(r_l)

                        if is_peak or is_valley:
                            m_idx = i + neighbor_right
                            if m_idx >= len(contours): continue

                            # Assigning based on LH/LL context
                            is_bull = is_valley # LL is bullish potential/reversal context
                            active_color = ll_col if is_bull else lh_col
                            custom_text = ll_text if is_bull else lh_text
                            obj_type = ll_obj if is_bull else lh_obj
                            dbl_arrow = ll_dbl if is_bull else lh_dbl
                            position = ll_pos if is_bull else lh_pos

                            # Draw main LH/LL label
                            x, y, w, h = cv2.boundingRect(contours[i])
                            label_objects_and_text(img, x+w//2, y, h, c_num=data[i]['candle_number'],
                                                 custom_text=custom_text,
                                                 object_type=obj_type,
                                                 is_bullish_arrow=is_bull,
                                                 is_marked=True,
                                                 double_arrow=dbl_arrow,
                                                 arrow_color=active_color,
                                                 label_position=position)

                            # Draw contour maker 'm'
                            mx, my, mw, mh = cv2.boundingRect(contours[m_idx])
                            cm_obj = ll_cm_obj if is_bull else lh_cm_obj
                            cm_dbl = ll_cm_dbl if is_bull else lh_cm_dbl
                            label_objects_and_text(img, mx+mw//2, my, mh, custom_text=cm_text,
                                                 object_type=cm_obj,
                                                 is_bullish_arrow=is_bull,
                                                 is_marked=True,
                                                 double_arrow=cm_dbl,
                                                 arrow_color=active_color,
                                                 label_position=position)

                            enriched = data[i].copy()
                            enriched.update({
                                "type": "lower_low" if is_bull else "lower_high",
                                "contour_maker": data[m_idx] if m_idx < n else None,
                                "m_idx": m_idx,
                                "active_color": active_color
                            })
                            swing_results.append(enriched)

                    if swing_results:
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w', encoding='utf-8') as f:
                            json.dump(swing_results, f, indent=4)
                        processed_charts_all += 1
                        total_marked_all += len(swing_results)

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
                    
                    if len(contours) == 0:
                        log(f"No candle contours detected for {sym}/{tf}", "WARN")
                        continue

                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

                    if len(data) != len(contours):
                        min_len = min(len(data), len(contours))
                        data = data[:min_len]
                        contours = contours[:min_len]

                    swing_results = []
                    n = len(data)

                    # NEW STRICT LOOP: Only process candles that have full neighbors on both sides
                    start_idx = neighbor_left
                    end_idx = n - neighbor_right  # Exclusive

                    for i in range(start_idx, end_idx):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        
                        # Left neighbors (full count required)
                        l_h = [d['high'] for d in data[i - neighbor_left:i]]
                        l_l = [d['low'] for d in data[i - neighbor_left:i]]
                        
                        # Right neighbors (full count required)
                        r_h = [d['high'] for d in data[i + 1:i + 1 + neighbor_right]]
                        r_l = [d['low'] for d in data[i + 1:i + 1 + neighbor_right]]

                        # Higher High: current high > all left highs AND > all right highs
                        is_hh = curr_h > max(l_h) and curr_h > max(r_h)
                        
                        # Higher Low: current low < all left lows AND < all right lows
                        is_hl = curr_l < min(l_l) and curr_l < min(r_l)

                        if not (is_hh or is_hl):
                            continue

                        is_bull = is_hl
                        active_color = hl_col if is_bull else hh_col
                        custom_text = hl_text if is_bull else hh_text
                        obj_type = hl_obj if is_bull else hh_obj
                        dbl_arrow = hl_dbl if is_bull else hh_dbl
                        position = hl_pos if is_bull else hh_pos

                        x, y, w, h = cv2.boundingRect(contours[i])
                        label_objects_and_text(
                            img, x + w // 2, y, h,
                            c_num=data[i]['candle_number'],
                            custom_text=custom_text,
                            object_type=obj_type,
                            is_bullish_arrow=is_bull,
                            is_marked=True,
                            double_arrow=dbl_arrow,
                            arrow_color=active_color,
                            label_position=position
                        )

                        # Contour maker logic (marks the candle neighbor_right steps to the right)
                        m_idx = i + neighbor_right
                        contour_maker_data = None
                        if m_idx < n:
                            mx, my, mw, mh = cv2.boundingRect(contours[m_idx])
                            cm_obj = hl_cm_obj if is_bull else hh_cm_obj
                            cm_dbl = hl_cm_dbl if is_bull else hh_cm_dbl
                            label_objects_and_text(
                                img, mx + mw // 2, my, mh,
                                custom_text=cm_text,
                                object_type=cm_obj,
                                is_bullish_arrow=is_bull,
                                is_marked=True,
                                double_arrow=cm_dbl,
                                arrow_color=active_color,
                                label_position=position
                            )
                            contour_maker_data = data[m_idx]

                        enriched = data[i].copy()
                        enriched.update({
                            "type": "higher_low" if is_bull else "higher_high",
                            "contour_maker": contour_maker_data,
                            "m_idx": m_idx if m_idx < n else None,
                            "active_color": active_color
                        })
                        swing_results.append(enriched)

                    if swing_results:
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w', encoding='utf-8') as f:
                            json.dump(swing_results, f, indent=4)
                        processed_charts_all += 1
                        total_marked_all += len(swing_results)

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
                    
                    # Sort data by candle number to ensure sequence matches Left-to-Right chart order
                    data = sorted(data, key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # FIXED: Always sort Left-to-Right (ascending x-coordinate)
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=False)
                    
                    fvg_results = []
                    marked_count = 0
                    potential_fvgs = []
                    
                    for i in range(1 + c1_lookback, len(data) - 1):
                        if i >= len(contours): break
                        
                        c_idx_1 = i - 1 - c1_lookback
                        c1, c2, c3 = data[c_idx_1], data[i], data[i+1]
                        
                        # --- Wick & Body Calculations ---
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
                                fvg_durability = (height1 * 2) <= height2
                            elif fvg_body_size_cfg == "multiply_c3_height_by_2":
                                fvg_durability = (height3 * 2) <= height2
                            
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
                                potential_fvgs.append(enriched)
                    
                    if potential_fvgs:
                        max_gap_found = max(p["fvg_gap_size"] for p in potential_fvgs)
                        
                        for entry in potential_fvgs:
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
                            
                            if not body_condition_ok:
                                continue
                            
                            idx = entry["_contour_idx"]
                            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[idx])
                            cx = x_rect + w_rect // 2
                            should_num = number_all or number_only_marked
                            
                            label_objects_and_text(
                                img=img, cx=cx, y_rect=y_rect, h_rect=h_rect,
                                c_num=entry.get('candle_number') if should_num else None,
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
                    
                    if marked_count > 0 or (number_all and len(data) > 0):
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w', encoding='utf-8') as f:
                            json.dump(fvg_results, f, indent=4)
                        processed_charts += 1
                        total_marked += marked_count
                        
                except Exception as e:
                    log(f"Error processing {sym}/{tf} with config '{config_key}': {e}", "ERROR")

        log(f"Completed config '{config_key}': FVGs: {total_marked} | Charts: {processed_charts}")
        total_marked_all += total_marked
        processed_charts_all += processed_charts

    return f"Done (all FVG configs). Total FVGs: {total_marked_all} | Total Charts: {processed_charts_all}"
    
def directional_bias(broker_name):
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
    
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    db_section = define_candles.get("directional_bias_candles", {})
    
    if not db_section:
        return f"[{broker_name}] Error: 'directional_bias_candles' section missing."

    total_db_marked = 0

    # ─── Self-apprehend (second level DB) config ───────────────────────────────
    self_apprehend_cfg = db_section.get("apprehend_directional_bias_candles", {})
    self_label_cfg = self_apprehend_cfg.get("label", {}) if self_apprehend_cfg else {}
    self_db_text = self_label_cfg.get("directional_bias_candles_text", "DB2")

    self_label_at = self_label_cfg.get("label_at", {})
    self_up_pos   = self_label_at.get("upward_directional_bias",   "high").lower()
    self_dn_pos   = self_label_at.get("downward_directional_bias", "high").lower()

    def resolve_marker(raw):
        raw = str(raw or "").lower().strip()
        if not raw: return None, False
        if "double" in raw: return "arrow", True
        if "arrow"  in raw: return "arrow", False
        if "dot" in raw or "circle" in raw: return "dot", False
        if "pentagon" in raw: return "pentagon", False
        return raw, False

    self_up_obj,  self_up_dbl  = resolve_marker(self_label_at.get("upward_directional_bias_marker"))
    self_dn_obj,  self_dn_dbl  = resolve_marker(self_label_at.get("downward_directional_bias_marker"))

    has_self_apprehend = bool(self_apprehend_cfg)

    # ─── Process every apprehend_* section ─────────────────────────────────────
    for apprehend_key, apprehend_cfg in db_section.items():
        if not isinstance(apprehend_cfg, dict):
            continue
        if apprehend_key == "apprehend_directional_bias_candles":
            continue 

        log(f"Processing directional bias apprehend: '{apprehend_key}'")

        target_type = apprehend_cfg.get("target", "").lower()
        if not target_type:
            log(f"Skipping {apprehend_key}: no 'target' specified.", "WARN")
            continue

        label_cfg = apprehend_cfg.get("label", {})
        db_text   = label_cfg.get("directional_bias_candles_text", "DB")
        label_at  = label_cfg.get("label_at", {})
        up_pos    = label_at.get("upward_directional_bias",   "high").lower()
        dn_pos    = label_at.get("downward_directional_bias", "high").lower()
        up_obj, up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker"))
        dn_obj, dn_dbl = resolve_marker(label_at.get("downward_directional_bias_marker"))

        source_config_name = apprehend_key.replace("apprehend_", "")
        source_config = define_candles.get(source_config_name)
        if not source_config:
            log(f"No source config '{source_config_name}' found for '{apprehend_key}'", "ERROR")
            continue

        bars      = source_config.get("BARS", 101)
        filename  = source_config.get("filename", "output.json")
        
        # FORCED TO new_old (Left to Right)
        direction = "new_old" 

        is_hhhl = "higherhighsandhigherlows" in source_config_name.lower()
        is_lhll = "lowerhighsandlowerlows"   in source_config_name.lower()
        is_fvg  = "fvg" in source_config_name.lower()

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue

            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, filename)
                
                required = ["source_json", "source_chart", "output_json", "output_chart"]
                if not all(os.path.exists(paths.get(p)) for p in required):
                    continue

                try:
                    # 1. Load and Sort JSON data Left-to-Right (Ascending candle number)
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        full_data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))

                    with open(paths["output_json"], 'r', encoding='utf-8') as f:
                        structures = json.load(f)

                    clean_img  = cv2.imread(paths["source_chart"])
                    marked_img = cv2.imread(paths["output_chart"])
                    if clean_img is None or marked_img is None:
                        continue

                    # 2. Extract and Sort Contours Left-to-Right (Ascending X coordinate)
                    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Sorting based on X coordinate (left to right)
                    contours = sorted(raw_contours, key=lambda c: cv2.boundingRect(c)[0])

                    n_candles = len(full_data)
                    if n_candles != len(contours):
                        log(f"Contour/data mismatch {sym}/{tf}. Data:{n_candles} Contours:{len(contours)}", "WARN")
                        continue

                    updated_structures = []

                    for structure in structures:
                        reference_idx   = None
                        reference_high  = None
                        reference_low   = None
                        active_color    = (0, 255, 0)
                        structure_type  = None
                        fvg_type        = None

                        # Determine reference candle index based on structure type
                        if (is_hhhl or is_lhll) and target_type == "contourmaker":
                            cm_data = structure.get("contour_maker")
                            if not cm_data:
                                updated_structures.append(structure)
                                continue
                            reference_idx = structure.get("m_idx")
                            reference_high = cm_data["high"]
                            reference_low  = cm_data["low"]
                            active_color   = tuple(structure.get("active_color", [0,255,0] if is_hhhl else [255,0,0]))
                            structure_type = "hhhl" if is_hhhl else "lhll"

                        elif is_fvg:
                            c1, c2, c3 = structure.get("c1_data"), structure.get("c2_data", structure), structure.get("c3_data")
                            ref_candle = None
                            if target_type == "fvg_c1": ref_candle = c1
                            elif target_type == "fvg_c2": ref_candle = c2
                            elif target_type == "fvg_c3": ref_candle = c3

                            if not ref_candle:
                                updated_structures.append(structure)
                                continue

                            ref_key = ref_candle.get("time") or ref_candle.get("candle_number")
                            reference_idx = next((i for i, d in enumerate(full_data) if d.get("time") == ref_key or d.get("candle_number") == ref_key), None)
                            
                            if reference_idx is not None:
                                reference_high = ref_candle["high"]
                                reference_low  = ref_candle["low"]
                                fvg_type       = structure.get("fvg_type", "").lower()
                                active_color   = (0, 255, 0) if "bullish" in fvg_type else (255, 0, 0)
                                structure_type = "fvg"

                        if reference_idx is None or reference_idx >= n_candles:
                            updated_structures.append(structure)
                            continue

                        # ─── First-level directional bias ─────────────────────────────
                        first_db_info = None
                        for k in range(reference_idx + 1, n_candles):
                            candle = full_data[k]
                            # Logic for identifying the break (DB)
                            if structure_type == "hhhl":
                                if candle['high'] < reference_low:
                                    first_db_info = {**candle, "idx": k, "type": "downward", "level": 1}
                                    break
                                if candle['low'] > reference_high:
                                    first_db_info = {**candle, "idx": k, "type": "upward", "level": 1}
                                    break
                            elif structure_type == "lhll":
                                if candle['low'] > reference_high:
                                    first_db_info = {**candle, "idx": k, "type": "upward", "level": 1}
                                    break
                                if candle['high'] < reference_low:
                                    first_db_info = {**candle, "idx": k, "type": "downward", "level": 1}
                                    break
                            elif structure_type == "fvg":
                                is_bull_fvg = "bullish" in fvg_type
                                if (is_bull_fvg and candle['low'] > reference_high) or (not is_bull_fvg and candle['low'] > reference_high):
                                    first_db_info = {**candle, "idx": k, "type": "upward", "level": 1}
                                    break
                                if (is_bull_fvg and candle['high'] < reference_low) or (not is_bull_fvg and candle['high'] < reference_low):
                                    first_db_info = {**candle, "idx": k, "type": "downward", "level": 1}
                                    break

                        if first_db_info:
                            db_idx = first_db_info["idx"]
                            x, y, w, h = cv2.boundingRect(contours[db_idx])
                            cx = x + w // 2
                            is_up = first_db_info["type"] == "upward"

                            label_objects_and_text(
                                img=marked_img, cx=cx, y_rect=y, h_rect=h,
                                custom_text=db_text, object_type=up_obj if is_up else dn_obj,
                                is_bullish_arrow=is_up, is_marked=True,
                                double_arrow=up_dbl if is_up else dn_dbl,
                                arrow_color=active_color, label_position=up_pos if is_up else dn_pos
                            )
                            structure["directional_bias"] = first_db_info
                            total_db_marked += 1

                            # ─── Second level (self-apprehend) ────────────────────────
                            if has_self_apprehend and db_idx + 1 < n_candles:
                                s_ref_h, s_ref_l = first_db_info["high"], first_db_info["low"]
                                second_db_info = None
                                for m in range(db_idx + 1, n_candles):
                                    c2 = full_data[m]
                                    if (structure_type == "hhhl" or structure_type == "lhll" or structure_type == "fvg"):
                                        if c2['high'] < s_ref_l:
                                            second_db_info = {**c2, "idx": m, "type": "downward", "level": 2}
                                            break
                                        if c2['low'] > s_ref_h:
                                            second_db_info = {**c2, "idx": m, "type": "upward", "level": 2}
                                            break
                                
                                if second_db_info:
                                    s_idx = second_db_info["idx"]
                                    sx, sy, sw, sh = cv2.boundingRect(contours[s_idx])
                                    s_is_up = second_db_info["type"] == "upward"

                                    label_objects_and_text(
                                        img=marked_img, cx=sx + sw // 2, y_rect=sy, h_rect=sh,
                                        custom_text=self_db_text, object_type=self_up_obj if s_is_up else self_dn_obj,
                                        is_bullish_arrow=s_is_up, is_marked=True,
                                        double_arrow=self_up_dbl if s_is_up else self_dn_dbl,
                                        arrow_color=active_color, label_position=self_up_pos if s_is_up else self_dn_pos
                                    )
                                    structure["directional_bias"]["next_bias"] = second_db_info
                                    total_db_marked += 1

                        updated_structures.append(structure)

                    # Save results
                    cv2.imwrite(paths["output_chart"], marked_img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(updated_structures, f, indent=4)

                except Exception as e:
                    log(f"Error processing {sym}/{tf} ({apprehend_key}): {e}", "ERROR")

    return f"Directional Bias Done. Total DB Markers: {total_db_marked}"  

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

        if not sender_list:
            log(f"No valid timeframes in {apprehend_key}", "WARN")
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
        bars = source_config.get("BARS", 101)  # Use same BARS as source analysis
        direction = source_config.get("read_candles_from", "new_old")
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

                sender_paths = get_analysis_paths(base_folder, broker_name, sym, sender_tf, direction, bars,
                                                  source_config.get("filename", "output.json"))

                if not os.path.exists(sender_paths["output_json"]):
                    log(f"{sym}/{sender_tf}: Source JSON missing", "WARN")
                    continue

                receiver_full_json = os.path.join(receiver_tf_path, "candlesdetails", "newest_oldest.json")
                if not os.path.exists(receiver_full_json):
                    log(f"{sym}/{receiver_tf}: newest_oldest.json NOT FOUND", "ERROR")
                    continue

                try:
                    with open(sender_paths["output_json"], 'r', encoding='utf-8') as f:
                        structures = json.load(f)
                    log(f"{sym}/{sender_tf}: Loaded {len(structures)} structures")

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
                                ref_time_dt = pd.to_datetime(ref_time)
                            else:
                                ref_time_dt = ref_time

                            if not isinstance(ref_time_dt, pd.Timestamp):
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

                        matched_times_sorted = sorted(matched_times)

                        used_suffixes = {}

                        for signal_time in matched_times_sorted:
                            if signal_time not in df_full.index:
                                continue

                            candle_idx = candle_index_map[signal_time]

                            # Start from signal candle and take up to 'bars' candles forward
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

                            # Yellow marker on the starting signal candle
                            scatter_data = pd.Series([float('nan')] * len(df_chart), index=df_chart.index)
                            high_val = df_chart.iloc[0]["high"]  # First candle = signal
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
                                fontsize=16, fontweight='bold', y=0.95, x=0.5, horizontalalignment='center'
                            )

                            output_dir = sender_paths["output_dir"]
                            os.makedirs(output_dir, exist_ok=True)

                            output_json_path = os.path.join(output_dir, f"{final_output_base}.json")
                            output_png_path = os.path.join(output_dir, f"{final_output_base}.png")

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
                                    "is_signal_candle": idx == 0  # First candle is the signal
                                }
                                for idx, (_, row) in enumerate(df_chart.iterrows())
                            ]

                            with open(output_json_path, 'w', encoding='utf-8') as f:
                                json.dump(forward_candles, f, indent=4)

                            processed_charts_all += 1
                            total_marked_all += 1
                            log(f"SAVED: {final_output_base}.{{json,png}} → {len(df_chart)} candles (max {bars}) from signal #{candle_idx}{suffix}")

                except Exception as e:
                    log(f"ERROR {sym} ({sender_tf}→{receiver_tf}): {e}", "ERROR")

    return f"Timeframes Communication Done. Total Individual Signals Charted: {total_marked_all} | Charts Generated: {processed_charts_all}"

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
    log(f"Errors encountered: {len(errors)}", "WARNING" if errors else "INFO")
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

                        log(f"Processing {os.path.basename(json_path)} | aligned {min_len} candles "
                            f"(json:{json_candles_count}, contours:{contour_count})", "DEBUG")

                        # ── 1. Record coordinates for EVERY candle ───────────────────────────────
                        for i in range(min_len):
                            x, y, w, h = cv2.boundingRect(contours[i])
                            
                            candle_info = {
                                "candle_x": int(x + w // 2),           # center x
                                "candle_y": int(y),                    # top y
                                "candle_width": int(w),
                                "candle_height": int(h),
                                "candle_left": int(x),
                                "candle_right": int(x + w),
                                "candle_top": int(y),
                                "candle_bottom": int(y + h)
                            }
                            
                            # Merge into existing candle dict (without overwriting other keys)
                            data[i].update(candle_info)

                        # Summary at index 0 (optional - you can remove if not needed)
                        highs_summary = {
                            "highs_summary": {
                                "json_candles_count": json_candles_count,
                                "contour_candles_count": contour_count,
                                "processed_candles_with_coords": min_len
                            }
                        }
                        data.insert(0, highs_summary)

                        # ── 2. Swing detection & drawing (same as before) ────────────────────────
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

                            # ── Swing candle drawing info ───────────────────────────────
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

                            # ── CONTOUR MAKER ───────────────────────────────────────────
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

                            # ── Update swing record ─────────────────────────────────────
                            if "type" not in swing_dict:
                                swing_dict["type"] = "higher_low" if is_bull else "higher_high"
                            if "active_color" not in swing_dict:
                                swing_dict["active_color"] = active_color

                            swing_dict["contour_maker"] = contour_maker_entry

                            modified = True
                            swing_results.append(swing_dict)

                        if modified:
                            cv2.imwrite(png_path, img)
                            
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=4)
                            
                            processed_charts_all += 1
                            total_marked_all += len(swing_results)
                            
                            log(f"Processed {os.path.basename(json_path)} → "
                                f"{len(swing_results)} swings | all {min_len} candles coords added", "INFO")

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
        """Checks if the item is a candle record and not a summary block."""
        return isinstance(item, dict) and item.get("candle_number") is not None

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
    total_files_processed = 0

    # ─── Config Extraction ────────────────────────────────────────────────────
    self_apprehend_cfg = db_section.get("apprehend_directional_bias_candles", {})
    self_label_cfg = self_apprehend_cfg.get("label", {}) if self_apprehend_cfg else {}
    self_db_text = self_label_cfg.get("directional_bias_candles_text", "DB2")
    self_label_at = self_label_cfg.get("label_at", {})
    self_up_pos   = self_label_at.get("upward_directional_bias",   "high").lower()
    self_dn_pos   = self_label_at.get("downward_directional_bias", "high").lower()
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
        db_text   = label_cfg.get("directional_bias_candles_text", "DB")
        label_at  = label_cfg.get("label_at", {})
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
                pattern = os.path.join(paths.get("output_dir"), f"{r_tf}_{source_filename.replace('.json', '')}_*_{s_tf}_*.json")
                
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

                        # Detect all contours (includes candles, arrows, dots, text)
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # ─── PRE-PROCESS CONTOURS ─────────────────────────────────
                        contour_list = []
                        for cnt in cnts:
                            x, y, w, h = cv2.boundingRect(cnt)
                            contour_list.append({
                                "cnt": cnt,
                                "cx": x + w/2,
                                "cy": y + h/2,
                                "bbox": (x, y, w, h)
                            })

                        # ─── SPATIAL MAPPING (FUZZY COORDINATE MATCH) ─────────────
                        # Map JSON candles to the closest detected contour center
                        candle_to_contour_map = {}
                        used_contour_indices = set()

                        for i, candle in enumerate(actual_candles):
                            # Target center from JSON
                            t_cx = candle.get("candle_left", 0) + (candle.get("candle_width", 0) / 2)
                            t_cy = candle.get("candle_top", 0) + (candle.get("candle_height", 0) / 2)
                            
                            best_dist = 15 # Max distance in pixels to consider a match
                            best_idx = None

                            for idx, c_info in enumerate(contour_list):
                                if idx in used_contour_indices: continue
                                dist = ((t_cx - c_info["cx"])**2 + (t_cy - c_info["cy"])**2)**0.5
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = idx
                            
                            if best_idx is not None:
                                candle_to_contour_map[i] = contour_list[best_idx]["cnt"]
                                used_contour_indices.add(best_idx)

                        # Align working data to matched visual candles
                        matched_indices = sorted(candle_to_contour_map.keys())
                        working_data = [actual_candles[i] for i in matched_indices]
                        ordered_contours = [candle_to_contour_map[i] for i in matched_indices]
                        
                        json_candles_count = len(actual_candles)
                        matched_count = len(working_data)

                        # Update Summary info
                        bias_summary_found = False
                        for summary in existing_summaries:
                            if "bias_summary" in summary:
                                summary["bias_summary"].update({
                                    "json_candles_count": json_candles_count,
                                    "contour_found_total": len(cnts),
                                    "matched_candles_count": matched_count,
                                    "total_iscandle_found": json_candles_count
                                })
                                bias_summary_found = True
                        
                        if not bias_summary_found:
                            existing_summaries.insert(0, {"bias_summary": {
                                "json_candles_count": json_candles_count,
                                "matched_candles_count": matched_count
                            }})

                        modified = False
                        marked_indices = set()
                        num_working = len(working_data)

                        # ─── Processing Loop ──────────────────────────────────────
                        for i, candle in enumerate(working_data):
                            candle["is_candle"] = True
                            
                            # Reference Candle Check
                            cm = candle.get("contour_maker")
                            if not isinstance(cm, dict): continue
                            
                            ref_num = cm.get("candle_number")
                            ref_idx = next((j for j, c in enumerate(working_data) if c.get("candle_number") == ref_num), None)
                            if ref_idx is None: continue

                            ref_h, ref_l = cm.get("high"), cm.get("low")
                            active_color = tuple(candle.get("active_color", [0, 255, 0]))
                            
                            # Finding Bias Candle
                            first_db_info = None
                            first_db_idx = None

                            for k in range(ref_idx + 1, num_working):
                                check_c = working_data[k]
                                if check_c['low'] > ref_h:
                                    first_db_info = {**check_c, "type": "upward", "level": 1}; first_db_idx = k; break
                                if check_c['high'] < ref_l:
                                    first_db_info = {**check_c, "type": "downward", "level": 1}; first_db_idx = k; break

                            if first_db_info and first_db_idx not in marked_indices:
                                # Get actual coordinates from the matched contour
                                x, y, w, h = cv2.boundingRect(ordered_contours[first_db_idx])
                                cx, is_up = x + w // 2, first_db_info["type"] == "upward"

                                label_objects_and_text(
                                    img=img, cx=cx, y_rect=y, h_rect=h, custom_text=db_text,
                                    object_type=up_obj if is_up else dn_obj, is_bullish_arrow=is_up,
                                    is_marked=True, double_arrow=up_dbl if is_up else dn_dbl,
                                    arrow_color=active_color, label_position=up_pos if is_up else dn_pos
                                )

                                first_db_info.update({
                                    "draw_x": int(cx), "draw_y": int(y), "draw_w": int(w), "draw_h": int(h),
                                    "draw_left": int(x), "draw_right": int(x + w),
                                    "draw_top": int(y), "draw_bottom": int(y + h)
                                })

                                candle["directional_bias"] = first_db_info
                                marked_indices.add(first_db_idx)
                                modified = True
                                total_db_marked += 1

                                # Level 2 Bias
                                if has_self_apprehend:
                                    s_ref_h, s_ref_l = first_db_info["high"], first_db_info["low"]
                                    for m in range(first_db_idx + 1, num_working):
                                        c2 = working_data[m]
                                        if c2['high'] < s_ref_l or c2['low'] > s_ref_h:
                                            if m not in marked_indices:
                                                s_up = c2['low'] > s_ref_h
                                                sx, sy, sw, sh = cv2.boundingRect(ordered_contours[m])
                                                label_objects_and_text(
                                                    img=img, cx=sx+sw//2, y_rect=sy, h_rect=sh, custom_text=self_db_text,
                                                    object_type=self_up_obj if s_up else self_dn_obj, is_bullish_arrow=s_up,
                                                    is_marked=True, double_arrow=self_up_dbl if s_up else self_dn_dbl,
                                                    arrow_color=active_color, label_position=self_up_pos if s_up else self_dn_pos
                                                )
                                                s_info = {**c2, "type": "upward" if s_up else "downward", "level": 2,
                                                          "draw_x": int(sx+sw//2), "draw_y": int(sy), "draw_w": int(sw), "draw_h": int(sh)}
                                                candle["directional_bias"]["next_bias"] = s_info
                                                marked_indices.add(m)
                                                total_db_marked += 1
                                            break

                        if modified:
                            final_output = existing_summaries + working_data
                            cv2.imwrite(img_path, img)
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(final_output, f, indent=4)
                            total_files_processed += 1
                            log(f"✓ Processed {os.path.basename(json_path)}", "INFO")

                    except Exception as e:
                        log(f"Error: {str(e)}", "ERROR")

    return f"Done. Processed {total_files_processed} files with {total_db_marked} markers."   

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
        hh_hl_results = pool.map(receiver_directional_bias, broker_names)
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


        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(directional_bias, broker_names)
        for r in db_results: print(r)


        print("\n[STEP 6] Running Timeframes Communication Analysis...")
        tf_comm_results = pool.map(timeframes_communication, broker_names)
        for r in tf_comm_results: print(r)

        
        hh_hl_results = pool.map(enrich_receiver_comm_paths, broker_names)
        for r in hh_hl_results: print(r)

        hh_hl_results = pool.map(receiver_comm_higher_highs_higher_lows, broker_names)
        for r in hh_hl_results: print(r)
        

    print("\n[SUCCESS] All tasks completed.")

if __name__ == "__main__":
   main()
   single()
    






                



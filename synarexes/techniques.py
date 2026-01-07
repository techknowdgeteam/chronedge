import os
import json
import re
import cv2
import numpy as np
import pytz
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta


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

def get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base):
    # Define the developer output root
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
    
    # Source: Where raw data currently lives (direction + bars specific)
    source_json = os.path.join(base_folder, sym, tf, "candlesdetails", f"{direction}_{bars}.json")
    source_chart = os.path.join(base_folder, sym, tf, f"chart_{bars}.png")
    
    # New: Full bars source - the complete newest_oldest.json (no direction/bars filtering)
    full_bars_source = os.path.join(base_folder, sym, tf, "candlesdetails", "newest_oldest.json")
    
    # Output: Where the results will be saved
    output_dir = os.path.join(dev_output_base, sym, tf)
    output_json = os.path.join(output_dir, output_filename_base)
    output_chart = os.path.join(output_dir, output_filename_base.replace(".json", ".png"))
    
    return {
        "dev_output_base": dev_output_base,
        "source_json": source_json,
        "source_chart": source_chart,
        "full_bars_source": full_bars_source,    # ← New path added here
        "output_dir": output_dir,
        "output_json": output_json,
        "output_chart": output_chart
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


def identify_higher_highs_higher_lows(broker_name):
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

                    # START CHANGE: Loop until the very last index (n)
                    for i in range(neighbor_left, n):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        
                        # Left side remains strict
                        l_h = [d['high'] for d in data[i-neighbor_left:i]]
                        l_l = [d['low'] for d in data[i-neighbor_left:i]]
                        
                        # Right side becomes dynamic (takes what is available)
                        r_h = [d['high'] for d in data[i+1 : i+1+neighbor_right]]
                        r_l = [d['low'] for d in data[i+1 : i+1+neighbor_right]]

                        # We only process if there is at least ONE candle to the right to compare
                        # If you want to mark the absolute last candle, remove the 'len(r_h) > 0' check
                        is_hh = len(l_h) > 0 and curr_h > max(l_h) and (not r_h or curr_h > max(r_h))
                        is_hl = len(l_l) > 0 and curr_l < min(l_l) and (not r_l or curr_l < min(r_l))

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

                        # Contour maker logic
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

def identify_lower_highs_lower_lows(broker_name):
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
  
def identify_fair_value_gaps(broker_name):
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
    
def identify_directional_bias(broker_name):
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

def mark_related_candles(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: return f"Error: {broker_name} not found."
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    define_candles = am_data.get("chart", {}).get("define_candles", {})
    comm_cfg = define_candles.get("timeframes_communication", {})
    
    if not comm_cfg: return "No timeframes_communication config found."

    total_related_marked = 0

    for app_key, app_val in comm_cfg.items():
        if not app_key.startswith("apprehend_"): continue
        
        comm_pair = app_val.get("timeframes_communication", "")
        import re
        tfs = re.findall(r'\d+[h|m]', comm_pair)
        if len(tfs) < 2: continue
        sender_tf, receiver_tf = tfs[0], tfs[1]
        
        target_type = app_val.get("target", "contourmaker")
        # Identify the source filename from the define_candles dictionary
        source_config_key = app_key.replace("apprehend_", "")
        source_filename = define_candles.get(source_config_key, {}).get("filename", "highers.json")

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            
            # Use analysis paths to get consistent folder locations
            s_paths = get_analysis_paths(base_folder, broker_name, sym, sender_tf, "new_old", 101, source_filename)
            r_paths = get_analysis_paths(base_folder, broker_name, sym, receiver_tf, "new_old", 101, comm_cfg.get("filename"))

            if not os.path.exists(s_paths["output_json"]) or not os.path.exists(r_paths["full_bars_source"]):
                continue

            try:
                with open(s_paths["output_json"], 'r') as f:
                    sender_structures = json.load(f)
                with open(r_paths["full_bars_source"], 'r') as f:
                    receiver_data = sorted(json.load(f), key=lambda x: x.get('time', ''))
                
                img = cv2.imread(r_paths["source_chart"])
                if img is None: continue

                # Get receiver contours
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) | cv2.inRange(hsv, (0,50,50),(10,255,255))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

                relation_results = []
                
                # Extract all relevant timestamps from the structure
                for struct in sender_structures:
                    times_to_mark = []
                    
                    # 1. Main Structure / ContourMaker
                    ref = struct.get("contour_maker") if target_type == "contourmaker" else struct
                    if ref and "time" in ref: times_to_mark.append(ref["time"])
                    
                    # 2. Apprehend Directional Bias
                    db = struct.get("directional_bias")
                    if db and "time" in db: times_to_mark.append(db["time"])
                    
                    # 3. Apprehend Next Bias (Level 2)
                    nb = db.get("next_bias") if db else None
                    if nb and "time" in nb: times_to_mark.append(nb["time"])

                    for target_time in times_to_mark:
                        # Find matching candles in receiver (15m) that belong to this sender (4h) candle
                        start_dt = datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')
                        # Duration window (e.g. 4 hours)
                        duration = 240 if "4h" in sender_tf else 60 if "1h" in sender_tf else 15
                        end_dt = start_dt + timedelta(minutes=duration)

                        for idx, r_can in enumerate(receiver_data):
                            r_dt = datetime.strptime(r_can["time"], '%Y-%m-%d %H:%M:%S')
                            
                            if start_dt <= r_dt < end_dt:
                                if idx < len(cnts):
                                    x, y, w, h = cv2.boundingRect(cnts[idx])
                                    
                                    # Marker Configuration
                                    label_at = app_val.get("label", {}).get("label_at", {})
                                    pos = label_at.get("upward_directional_bias", "high")
                                    marker = label_at.get("upward_directional_bias_marker", "dot")
                                    
                                    label_objects_and_text(
                                        img, x + w // 2, y, h,
                                        custom_text=app_val.get("label", {}).get("directional_bias_candles_text", ""),
                                        object_type=marker,
                                        is_bullish_arrow=True,
                                        is_marked=True,
                                        arrow_color=(0, 255, 0), # Default green for relation
                                        label_position=pos
                                    )
                                    
                                    r_can["related_to_sender_time"] = target_time
                                    relation_results.append(r_can)
                                    total_related_marked += 1

                if relation_results:
                    os.makedirs(r_paths["output_dir"], exist_ok=True)
                    cv2.imwrite(r_paths["output_chart"], img)
                    with open(r_paths["output_json"], 'w') as f:
                        json.dump(relation_results, f, indent=4)

            except Exception as e:
                log(f"Error {sym}: {e}", "ERROR")

    return f"Mark Related Done. Total Related Marked: {total_related_marked}"
   
def main():
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return

    broker_names = sorted(dev_dict.keys())
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:

        # STEP 2: Higher Highs & Higher Lows
        print("\n[STEP 2] Running Higher Highs & Higher Lows Analysis...")
        hh_hl_results = pool.map(identify_higher_highs_higher_lows, broker_names)
        for r in hh_hl_results: print(r)

        # STEP 3: Lower Highs & Lower Lows
        print("\n[STEP 3] Running Lower Highs & Lower Lows Analysis...")
        lh_ll_results = pool.map(identify_lower_highs_lower_lows, broker_names)
        for r in lh_ll_results: print(r)

        # STEP 4: Fair Value Gaps (FVG)
        print("\n[STEP 4] Running Fair Value Gap Analysis...")
        fvg_results = pool.map(identify_fair_value_gaps, broker_names)
        for r in fvg_results: print(r)

        # STEP 5: Directional Bias (DB)
        # Identifies breaks based on steps 2, 3, and 4
        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(identify_directional_bias, broker_names)
        for r in db_results: print(r)

        # ─── NEW STEP ──────────────────────────────────────────────────────────
        # STEP 6: Timeframe Communication (Multi-Timeframe Sync)
        # Synchronizes HTF markers with LTF full bar sources
        print("\n[STEP 6] Running Timeframe Communication Analysis...")
        sync_results = pool.map(mark_related_candles, broker_names)
        for r in sync_results: print(r)
        # ───────────────────────────────────────────────────────────────────────

    print("\n[SUCCESS] All tasks completed.")
    
if __name__ == "__main__":
    main()
    






                



import os
import json
import re
import cv2
import numpy as np
import pytz
from multiprocessing import Pool, cpu_count
from datetime import datetime

# === CONFIGURATION LOADERS ===
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
    """
    Centralized path factory. 
    Handles both the developer output root and specific file paths.
    """
    # Define the developer output root
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
    
    # Source: Where raw data currently lives
    source_json = os.path.join(base_folder, sym, tf, "candlesdetails", f"{direction}_{bars}.json")
    source_chart = os.path.join(base_folder, sym, tf, f"chart_{bars}.png")
    
    # Output: Where the results will be saved
    output_dir = os.path.join(dev_output_base, sym, tf)
    output_json = os.path.join(output_dir, output_filename_base)
    output_chart = os.path.join(output_dir, output_filename_base.replace(".json", ".png"))
    
    return {
        "dev_output_base": dev_output_base,
        "source_json": source_json,
        "source_chart": source_chart,
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


def identify_higher_highs_higher_lowss(broker_name):
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
    if not define_candles:
        return f"[{broker_name}] Error: 'define_candles' section missing."

    keyword = "higherhighsandhigherlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]

    if not matching_configs:
        return f"[{broker_name}] Error: No configuration found for '{keyword}'."

    total_marked_all = 0
    processed_charts_all = 0

    for config_key, hhhl_cfg in matching_configs:
        log(f"Processing configuration: '{config_key}'")

        bars = hhhl_cfg.get("BARS", 101)
        output_filename_base = hhhl_cfg.get("filename", "highers.json")
        direction = hhhl_cfg.get("read_candles_from", "new_old")
        neighbor_left = hhhl_cfg.get("NEIGHBOR_LEFT", 5)
        neighbor_right = hhhl_cfg.get("NEIGHBOR_RIGHT", 5)

        # Label Settings
        label_cfg = hhhl_cfg.get("label", {})
        hh_text = label_cfg.get("higherhighs_text", "HH")
        hl_text = label_cfg.get("higherlows_text", "HL")
        cm_text = label_cfg.get("contourmaker_text", "m")
        db_text = label_cfg.get("directional_bias_text", "db")
        
        label_at = label_cfg.get("label_at", {})
        hh_pos = label_at.get("higher_highs", "high").lower()
        hl_pos = label_at.get("higher_lows", "low").lower()
        
        def resolve_marker(raw):
            raw = str(raw).lower().strip()
            if not raw: return None, False
            if "double" in raw: return "arrow", True
            if "arrow" in raw: return "arrow", False
            if "dot" in raw or "circle" in raw: return "dot", False
            return raw, False

        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        hl_obj, hl_dbl = resolve_marker(label_at.get("higher_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        hl_cm_obj, hl_cm_dbl = resolve_marker(label_at.get("higher_lows_contourmaker_marker", ""))
        db_up_obj, db_up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker", ""))
        db_down_obj, db_down_dbl = resolve_marker(label_at.get("downward_directional_bias_marker", ""))

        color_map = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red"), (0, 0, 255))
        hl_col = color_map.get(label_at.get("higher_lows_color", "green"), (0, 255, 0))

        processed_charts, total_marked = 0, 0

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
                        data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue

                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=(direction == "new_old"))

                    swing_points = []
                    n = len(data)

                    for i in range(neighbor_left, n - neighbor_right):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        l_h = [d['high'] for d in data[i-neighbor_left:i]]
                        l_l = [d['low'] for d in data[i-neighbor_left:i]]
                        r_h = [d['high'] for d in data[i+1:i+neighbor_right+1]]
                        r_l = [d['low'] for d in data[i+1:i+neighbor_right+1]]

                        is_hh = curr_h > max(l_h) and curr_h > max(r_h)
                        is_hl = curr_l < min(l_l) and curr_l < min(r_l)

                        if is_hh or is_hl:
                            maker_idx = i + neighbor_right
                            if maker_idx >= n: continue
                            
                            price_used = data[maker_idx]['open']
                            s_type = "higher_high" if is_hh and price_used < curr_h else None
                            if not s_type:
                                s_type = "higher_low" if is_hl and price_used > curr_l else None

                            if s_type:
                                db_data = None
                                cm_high = data[maker_idx]['high']
                                cm_low = data[maker_idx]['low']
                                
                                for k in range(maker_idx + 1, n):
                                    if data[k]['high'] < cm_low:
                                        db_data = data[k].copy()
                                        db_data.update({"idx": k, "type": "downward"})
                                        break
                                    if data[k]['low'] > cm_high:
                                        db_data = data[k].copy()
                                        db_data.update({"idx": k, "type": "upward"})
                                        break
                                
                                swing_points.append({
                                    "swing_idx": i, 
                                    "type": s_type, 
                                    "price": curr_h if is_hh else curr_l,
                                    "maker_idx": maker_idx,
                                    "db": db_data
                                })

                    swing_results = []
                    for sp in swing_points:
                        i, m_idx = sp["swing_idx"], sp["maker_idx"]
                        is_bull = (sp["type"] == "higher_low")
                        # The color used for the swing/contour maker
                        active_color = hl_col if is_bull else hh_col
                        
                        # A. Draw Swing (HH/HL)
                        x, y, w, h = cv2.boundingRect(contours[i])
                        label_objects_and_text(img, x+w//2, y, h, c_num=data[i]['candle_number'],
                                              custom_text=hl_text if is_bull else hh_text,
                                              object_type=hl_obj if is_bull else hh_obj,
                                              is_bullish_arrow=is_bull, is_marked=True,
                                              double_arrow=hl_dbl if is_bull else hh_dbl,
                                              arrow_color=active_color,
                                              label_position=hl_pos if is_bull else hh_pos)

                        # B. Draw Contour Maker (m)
                        mx, my, mw, mh = cv2.boundingRect(contours[m_idx])
                        label_objects_and_text(img, mx+mw//2, my, mh, custom_text=cm_text,
                                              object_type=hl_cm_obj if is_bull else hh_cm_obj,
                                              is_bullish_arrow=is_bull, is_marked=True,
                                              double_arrow=hl_cm_dbl if is_bull else hh_cm_dbl,
                                              arrow_color=active_color,
                                              label_position=hl_pos if is_bull else hh_pos)

                        # C. Draw Directional Bias (db)
                        db = sp["db"]
                        if db and db["idx"] < len(contours):
                            dx, dy, dw, dh = cv2.boundingRect(contours[db["idx"]])
                            is_up_bias = (db["type"] == "upward")
                            
                            # Marker at Low for downward per request, and color matched to active swing color
                            db_label_pos = "low" if (is_up_bias or db["type"] == "downward") else "high"
                            
                            label_objects_and_text(img, dx+dw//2, dy, dh, custom_text=db_text,
                                                  object_type=db_up_obj if is_up_bias else db_down_obj,
                                                  is_bullish_arrow=is_up_bias, is_marked=True,
                                                  double_arrow=db_up_dbl if is_up_bias else db_down_dbl,
                                                  arrow_color=active_color, # MATCHED COLOR
                                                  label_position=db_label_pos)

                        enriched = data[i].copy()
                        enriched.update({
                            "type": sp["type"],
                            "contour_maker": data[m_idx],
                            "directional_bias": db
                        })
                        swing_results.append(enriched)

                    if swing_results:
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w') as f:
                            json.dump(swing_results, f, indent=4)
                        processed_charts += 1
                        total_marked += len(swing_results)

                except Exception as e:
                    log(f"Error: {e}", "ERROR")

        total_marked_all += total_marked
        processed_charts_all += processed_charts

    return f"Done. Swings: {total_marked_all} | Charts: {processed_charts_all}"   


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
    
    lhll_cfg = am_data.get("chart", {}).get("define_candles", {}).get("lowerhighsandlowerlows", {})
    if not lhll_cfg: return f"[{broker_name}] Error: Config section missing."

    bars = lhll_cfg.get("BARS", 101)
    output_filename_base = lhll_cfg.get("filename", "lowers.json")
    neighbor_left = lhll_cfg.get("NEIGHBOR_LEFT", 5)
    neighbor_right = lhll_cfg.get("NEIGHBOR_RIGHT", 5)
    direction = lhll_cfg.get("read_candles_from", "new_old")
    number_mode = lhll_cfg.get("number_candles", "define_candles").lower()

    label_cfg = lhll_cfg.get("label", {})
    lh_text, ll_text = label_cfg.get("lowerhighs_text", "LH"), label_cfg.get("lowerlows_text", "LL")
    label_at = label_cfg.get("label_at", {})
    lh_pos, ll_pos = label_at.get("lower_highs", "high").lower(), label_at.get("lower_lows", "low").lower()

    color_map = {"green": (0, 255, 0), "red": (0, 0, 255)}
    lh_col = color_map.get(label_at.get("lower_highs_color", "red").lower(), (0, 0, 255))
    ll_col = color_map.get(label_at.get("lower_lows_color", "green").lower(), (0, 255, 0))

    def resolve_marker(raw):
        raw = str(raw).lower().strip()
        if raw in ["arrow", "arrows", "singlearrow"]: return "arrow", False
        if raw in ["doublearrow", "doublearrows"]: return "arrow", True
        if raw in ["reverse_arrow", "reversearrow"]: return "reverse_arrow", False
        if raw in ["reverse_doublearrow", "reverse_doublearrows"]: return "reverse_arrow", True
        if raw in ["rightarrow", "right_arrow"]: return "rightarrow", False
        if raw in ["leftarrow", "left_arrow"]: return "leftarrow", False
        return raw, False
        
    lh_obj, lh_dbl = resolve_marker(label_at.get("lower_highs_marker", "arrow"))
    ll_obj, ll_dbl = resolve_marker(label_at.get("lower_lows_marker", "arrow"))

    number_all = (number_mode == "all")
    number_only_marked = number_mode in ["define_candles", "define_candle", "definecandle"]
    total_marked, processed_charts = 0, 0

    log(f"Starting Lower Analysis | BARS: {bars}")

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p): continue
        
        for tf in sorted(os.listdir(sym_p)):
            if not os.path.isdir(os.path.join(sym_p, tf)): continue

            # --- PATHS GENERATED VIA HELPER ---
            paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)

            if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]):
                continue

            try:
                with open(paths["source_json"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not data: continue

                data = sorted(data, key=lambda x: x.get('candle_number', 0))
                img = cv2.imread(paths["source_chart"])
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=(direction == "new_old"))

                swing_points = []
                n = len(data)
                for i in range(n):
                    if i + neighbor_right >= n: continue
                    curr_h, curr_l = data[i]['high'], data[i]['low']
                    l_h = [data[j]['high'] for j in range(max(0, i - neighbor_left), i)]
                    r_h = [data[j]['high'] for j in range(i + 1, min(i + neighbor_right + 1, n))]
                    l_l = [data[j]['low'] for j in range(max(0, i - neighbor_left), i)]
                    r_l = [data[j]['low'] for j in range(i + 1, min(i + neighbor_right + 1, n))]

                    if curr_h > max(r_h) and (not l_h or curr_h > max(l_h)):
                        swing_points.append((i, "lower_high", curr_h))
                    if curr_l < min(r_l) and (not l_l or curr_l < min(l_l)):
                        swing_points.append((i, "lower_low", curr_l))

                swing_candles, marked_count = [], 0
                for i, candle in enumerate(data):
                    if i >= len(contours): break
                    c_num = candle.get('candle_number', i)
                    x_r, y_r, w_r, h_r = cv2.boundingRect(contours[i])
                    cx = x_r + w_r // 2

                    matching = [sp for sp in swing_points if sp[0] == i]
                    is_swing = len(matching) > 0
                    should_num = number_all or (number_only_marked and is_swing)

                    if is_swing:
                        for _, s_type, price in matching:
                            is_bull = (s_type == "lower_low")
                            label_objects_and_text(
                                img=img, cx=cx, y_rect=y_r, h_rect=h_r,
                                c_num=c_num if should_num else None,
                                custom_text=ll_text if is_bull else lh_text,
                                object_type=ll_obj if is_bull else lh_obj,
                                is_bullish_arrow=is_bull, is_marked=True,
                                double_arrow=ll_dbl if is_bull else lh_dbl,
                                arrow_color=ll_col if is_bull else lh_col,
                                label_position=ll_pos if is_bull else lh_pos
                            )
                            enriched = candle.copy()
                            enriched[f"is_{s_type.replace('_','')}"] = True
                            enriched["swing_price"] = round(price, 5)
                            swing_candles.append(enriched)
                            marked_count += 1
                    elif should_num:
                        label_objects_and_text(img=img, cx=cx, y_rect=y_r, h_rect=h_r, c_num=c_num, is_marked=False, label_position="high")

                # --- OUTPUT SAVED VIA HELPER PATHS ---
                if marked_count > 0 or number_all:
                    os.makedirs(paths["output_dir"], exist_ok=True)
                    cv2.imwrite(paths["output_chart"], img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(swing_candles, f, indent=4)
                    processed_charts += 1
                    total_marked += marked_count

            except Exception as e:
                log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    return f"Done. Swings: {total_marked}"

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
        direction = fvg_cfg.get("read_candles_from", "old_new")
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

        log(f"Starting FVG Analysis with config '{config_key}' | Validate: {validate_filter} | Lookback: {c1_lookback} | Body>Wick: {apply_body_vs_wick_rule} | Durability: {fvg_body_size_cfg or 'none'}")

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
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=(direction == "new_old"))
                    
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
                            
                            # Core condition checks
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
                            
                            # Enhanced FVG Durability Check
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
                    
                    # Contest logic + body/wick filter
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
                            
                            # Draw & save
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

def identify_higher_highs_higher_lows(broker_name):
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
    keyword = "higherhighsandhigherlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]

    if not matching_configs:
        return f"[{broker_name}] Error: No configuration found for '{keyword}'."

    total_marked_all, processed_charts_all = 0, 0

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
        hh_pos, hl_pos = label_at.get("higher_highs", "high").lower(), label_at.get("higher_lows", "low").lower()
        
        def resolve_marker(raw):
            raw = str(raw).lower().strip()
            if not raw: return None, False
            return ("arrow", "double" in raw) if "arrow" in raw else (("dot", False) if "dot" in raw else (raw, False))

        hh_obj, hh_dbl = resolve_marker(label_at.get("higher_highs_marker", "arrow"))
        hl_obj, hl_dbl = resolve_marker(label_at.get("higher_lows_marker", "arrow"))
        hh_cm_obj, hh_cm_dbl = resolve_marker(label_at.get("higher_highs_contourmaker_marker", ""))
        hl_cm_obj, hl_cm_dbl = resolve_marker(label_at.get("higher_lows_contourmaker_marker", ""))

        color_map = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}
        hh_col = color_map.get(label_at.get("higher_highs_color", "red"), (0, 0, 255))
        hl_col = color_map.get(label_at.get("higher_lows_color", "green"), (0, 255, 0))

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, output_filename_base)
                if not os.path.exists(paths["source_json"]) or not os.path.exists(paths["source_chart"]): continue

                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    img = cv2.imread(paths["source_chart"])
                    if img is None: continue

                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=(direction == "new_old"))

                    swing_results = []
                    n = len(data)
                    for i in range(neighbor_left, n - neighbor_right):
                        curr_h, curr_l = data[i]['high'], data[i]['low']
                        l_h, l_l = [d['high'] for d in data[i-neighbor_left:i]], [d['low'] for d in data[i-neighbor_left:i]]
                        r_h, r_l = [d['high'] for d in data[i+1:i+neighbor_right+1]], [d['low'] for d in data[i+1:i+neighbor_right+1]]
                        
                        is_hh, is_hl = curr_h > max(l_h) and curr_h > max(r_h), curr_l < min(l_l) and curr_l < min(r_l)
                        if is_hh or is_hl:
                            m_idx = i + neighbor_right
                            if m_idx >= n: continue
                            
                            is_bull = is_hl
                            active_color = hl_col if is_bull else hh_col
                            
                            # Draw HH/HL
                            x, y, w, h = cv2.boundingRect(contours[i])
                            label_objects_and_text(img, x+w//2, y, h, c_num=data[i]['candle_number'],
                                                 custom_text=hl_text if is_bull else hh_text,
                                                 object_type=hl_obj if is_bull else hh_obj,
                                                 is_bullish_arrow=is_bull, is_marked=True,
                                                 double_arrow=hl_dbl if is_bull else hh_dbl,
                                                 arrow_color=active_color, label_position=hl_pos if is_bull else hh_pos)

                            # Draw Contour Maker (m)
                            mx, my, mw, mh = cv2.boundingRect(contours[m_idx])
                            label_objects_and_text(img, mx+mw//2, my, mh, custom_text=cm_text,
                                                 object_type=hl_cm_obj if is_bull else hh_cm_obj,
                                                 is_bullish_arrow=is_bull, is_marked=True,
                                                 double_arrow=hl_cm_dbl if is_bull else hh_cm_dbl,
                                                 arrow_color=active_color, label_position=hl_pos if is_bull else hh_pos)

                            enriched = data[i].copy()
                            enriched.update({"type": "higher_low" if is_bull else "higher_high", 
                                           "contour_maker": data[m_idx], "m_idx": m_idx, "active_color": active_color})
                            swing_results.append(enriched)

                    if swing_results:
                        os.makedirs(paths["output_dir"], exist_ok=True)
                        cv2.imwrite(paths["output_chart"], img)
                        with open(paths["output_json"], 'w') as f: json.dump(swing_results, f, indent=4)
                        processed_charts_all += 1
                        total_marked_all += len(swing_results)
                except Exception as e: log(f"Error: {e}", "ERROR")

    return f"Identify Done. Swings: {total_marked_all} | Charts: {processed_charts_all}"

def identify_directional_bias(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg: return f"[{broker_name}] Error"
    
    base_folder = cfg.get("BASE_FOLDER")
    am_data = get_account_management(broker_name)
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    
    keyword = "higherhighsandhigherlows"
    matching_configs = [(k, v) for k, v in define_candles.items() if keyword in k.lower()]

    total_db_marked = 0
    for config_key, hhhl_cfg in matching_configs:
        bars = hhhl_cfg.get("BARS", 101)
        filename = hhhl_cfg.get("filename", "highers.json")
        direction = hhhl_cfg.get("read_candles_from", "new_old")
        
        label_cfg = hhhl_cfg.get("label", {})
        db_text = label_cfg.get("directional_bias_text", "db")
        label_at = label_cfg.get("label_at", {})

        def resolve_marker(raw):
            raw = str(raw).lower().strip()
            if not raw: return None, False
            if "double" in raw: return "arrow", True
            if "arrow" in raw: return "arrow", False
            if "dot" in raw or "circle" in raw: return "dot", False
            return raw, False

        up_obj, up_dbl = resolve_marker(label_at.get("upward_directional_bias_marker", "dot"))
        dn_obj, dn_dbl = resolve_marker(label_at.get("downward_directional_bias_marker", "dot"))

        for sym in sorted(os.listdir(base_folder)):
            sym_p = os.path.join(base_folder, sym)
            if not os.path.isdir(sym_p): continue
            for tf in sorted(os.listdir(sym_p)):
                paths = get_analysis_paths(base_folder, broker_name, sym, tf, direction, bars, filename)
                # Ensure we have the source JSON, source Chart (clean), and output JSON
                if not os.path.exists(paths["source_json"]) or \
                   not os.path.exists(paths["source_chart"]) or \
                   not os.path.exists(paths["output_json"]): continue

                try:
                    with open(paths["source_json"], 'r', encoding='utf-8') as f:
                        full_data = sorted(json.load(f), key=lambda x: x.get('candle_number', 0))
                    with open(paths["output_json"], 'r', encoding='utf-8') as f:
                        swings = json.load(f)
                    
                    # LOAD BOTH IMAGES
                    clean_img = cv2.imread(paths["source_chart"]) # For detection
                    marked_img = cv2.imread(paths["output_chart"]) # For drawing
                    if clean_img is None or marked_img is None: continue

                    # 1. Detect contours on the CLEAN chart for accurate indexing
                    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Use same sorting as all-in-one
                    contours = sorted(raw_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=(direction == "new_old"))

                    updated_swings = []
                    for s in swings:
                        m_idx = s.get("m_idx") or s.get("maker_idx")
                        if m_idx is None or "contour_maker" not in s:
                            updated_swings.append(s)
                            continue

                        # Extract previous context
                        active_color = tuple(s.get("active_color", [0, 255, 0]))
                        cm_high = s["contour_maker"]["high"]
                        cm_low = s["contour_maker"]["low"]
                        
                        db_candle_info = None

                        # 2. EXACT Price Logic Loop from All-in-One
                        n = len(full_data)
                        for k in range(m_idx + 1, n):
                            # Downward check
                            if full_data[k]['high'] < cm_low:
                                db_candle_info = full_data[k].copy()
                                db_candle_info.update({"idx": k, "type": "downward"})
                                break
                            # Upward check
                            if full_data[k]['low'] > cm_high:
                                db_candle_info = full_data[k].copy()
                                db_candle_info.update({"idx": k, "type": "upward"})
                                break
                        
                        if db_candle_info:
                            f_idx = db_candle_info["idx"]
                            if f_idx < len(contours):
                                # Use coordinates from CLEAN chart, but draw on MARKED chart
                                dx, dy, dw, dh = cv2.boundingRect(contours[f_idx])
                                is_up = (db_candle_info["type"] == "upward")
                                
                                # Positioning: All-in-one always results in "low" for valid DB
                                label_pos = "low"

                                label_objects_and_text(
                                    marked_img, dx+dw//2, dy, dh, 
                                    custom_text=db_text,
                                    object_type=up_obj if is_up else dn_obj,
                                    is_bullish_arrow=is_up, 
                                    is_marked=True,
                                    double_arrow=up_dbl if is_up else dn_dbl,
                                    arrow_color=active_color, 
                                    label_position=label_pos
                                )
                                s["directional_bias"] = db_candle_info
                                total_db_marked += 1
                        
                        updated_swings.append(s)

                    # Save updated chart and JSON
                    cv2.imwrite(paths["output_chart"], marked_img)
                    with open(paths["output_json"], 'w', encoding='utf-8') as f:
                        json.dump(updated_swings, f, indent=4)

                except Exception as e:
                    log(f"DB Error {sym} {tf}: {e}", "ERROR")

    return f"DB Done. Total Markers: {total_db_marked}"   
    
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
        # Now strictly identifies swings and contour makers (m)
        print("\n[STEP 2] Running Higher Highs & Higher Lows Analysis...")
        hh_hl_results = pool.map(identify_higher_highs_higher_lows, broker_names)
        for r in hh_hl_results: print(r)

        # STEP 3: Lower Highs & Lower Lows
        # (Assuming you have a similar structure for LH/LL)
        print("\n[STEP 3] Running Lower Highs & Lower Lows Analysis...")
        lh_ll_results = pool.map(identify_lower_highs_lower_lows, broker_names)
        for r in lh_ll_results: print(r)

        # STEP 4: Fair Value Gaps (FVG)
        print("\n[STEP 4] Running Fair Value Gap Analysis...")
        fvg_results = pool.map(identify_fair_value_gaps, broker_names)
        for r in fvg_results:
            print(r)

        # STEP 5: Directional Bias (DB)
        # New independent function that apprehends the output from Step 2 & 3
        print("\n[STEP 5] Running Directional Bias Analysis...")
        db_results = pool.map(identify_directional_bias, broker_names)
        for r in db_results:
            print(r)

    print("\n[SUCCESS] All tasks completed.")

if __name__ == "__main__":
    main()
    






                



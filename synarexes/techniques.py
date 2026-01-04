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


def label_objects_and_text(
    img,
    cx,
    y_rect,
    h_rect,
    c_num=None,                  # Optional candle number
    custom_text=None,            # e.g., "LB", "HH", "HL"
    is_bullish_arrow=True,       # True → upward arrow (label below), False → downward (label above)
    is_marked=False,
    double_arrow=False,
    arrow_color=(0, 255, 0),
    font_scale=0.65,
    text_thickness=2
):
    """
    Draws arrow(s) and text labels on candle chart.
    - Upward arrow (is_bullish_arrow=True)  → marks lows  → text below
    - Downward arrow (False)               → marks highs → text above
    """

    shaft_length = 26
    head_size = 9
    thickness = 2
    wing_size = 7 if double_arrow else 6

    if is_marked:
        def draw_single_arrow(center_x: int):
            if is_bullish_arrow:
                # Upward arrow from bottom
                cy_tip = y_rect + h_rect
                shaft_start_y = cy_tip + head_size
                cv2.line(img, (center_x, shaft_start_y), (center_x, shaft_start_y + shaft_length),
                         arrow_color, thickness)
                pts = np.array([
                    [center_x, shaft_start_y],
                    [center_x - wing_size, shaft_start_y + head_size],
                    [center_x + wing_size, shaft_start_y + head_size]
                ], np.int32)
            else:
                # Downward arrow from top
                cy_tip = y_rect
                shaft_start_y = cy_tip - head_size
                cv2.line(img, (center_x, shaft_start_y), (center_x, shaft_start_y - shaft_length),
                         arrow_color, thickness)
                pts = np.array([
                    [center_x, shaft_start_y],
                    [center_x - wing_size, shaft_start_y - head_size],
                    [center_x + wing_size, shaft_start_y - head_size]
                ], np.int32)
            cv2.fillPoly(img, [pts], arrow_color)

        if double_arrow:
            offset = 5
            draw_single_arrow(cx - offset)
            draw_single_arrow(cx + offset)
        else:
            draw_single_arrow(cx)

    # === DRAW TEXT (custom + optional number) ===
    if not (custom_text or c_num is not None):
        return

    bottom_y = y_rect + h_rect
    top_y = y_rect

    if is_bullish_arrow:
        # Text BELOW the upward arrow
        base_text_y = bottom_y + shaft_length + head_size + 22
    else:
        # Text ABOVE the downward arrow
        base_text_y = top_y - shaft_length - head_size - 12

    # === CUSTOM TEXT (centered, no black border) ===
    if custom_text:
        # Get text size to center it properly
        (text_width, text_height), baseline = cv2.getTextSize(
            custom_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        text_x = cx - text_width // 2  # Perfect horizontal centering
        text_y = base_text_y

        # Only draw the colored text (no black outline/border)
        cv2.putText(img, custom_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, arrow_color, text_thickness)

    # === CANDLE NUMBER (unchanged as requested) ===
    if c_num is not None:
        num_y = base_text_y + 28 if custom_text else base_text_y
        cv2.putText(img, str(c_num), (cx - 10, num_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(img, str(c_num), (cx - 10, num_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

def analyze_tallest_body(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
    
    base_folder = cfg.get("BASE_FOLDER")
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
    
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."

    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    large_body_cfg = define_candles.get("large_body", {})
    
    if not large_body_cfg:
        return f"[{broker_name}] Error: define_candles.large_body section missing."

    bars = large_body_cfg.get("BARS", 101)
    output_filename_base = large_body_cfg.get("filename", "largebody.json")
    direction = large_body_cfg.get("read_candles_from", "new_old")
    number_mode = large_body_cfg.get("number_candles", "define_candles").lower()

    label_cfg = large_body_cfg.get("label", {})
    marker_type = label_cfg.get("marker", "arrow").lower()
    custom_text = label_cfg.get("text", "LB")

    output_chart_name = output_filename_base.replace(".json", ".png")
    output_json_name = output_filename_base

    source_json = f"new_old_{bars}.json" if direction == "new_old" else f"old_new_{bars}.json"
    source_chart = f"chart_{bars}.png"

    color_map = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}
    bullish_color = color_map.get(label_cfg.get("label_at", {}).get("bullish_color", "green").lower(), (0, 255, 0))
    bearish_color = color_map.get(label_cfg.get("label_at", {}).get("bearish_color", "red").lower(), (0, 0, 255))

    number_all = number_mode == "all"
    number_only_marked = number_mode in ["define_candles", "define_candle", "definecandle", "define_candle"]

    total_marked = 0
    processed_charts = 0

    log(f"Starting Large Body Analysis | BARS: {bars} | Direction: {direction} | Number: {number_mode}")

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p): continue

        for tf in sorted(os.listdir(sym_p)):
            tf_path = os.path.join(sym_p, tf)
            if not os.path.isdir(tf_path): continue

            source_json_path = os.path.join(tf_path, "candlesdetails", source_json)
            source_chart_path = os.path.join(tf_path, source_chart)

            if not os.path.exists(source_json_path) or not os.path.exists(source_chart_path):
                continue

            try:
                with open(source_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not data: continue

                data = sorted(data, key=lambda x: x.get('candle_number', 0))

                img = cv2.imread(source_chart_path)
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                should_reverse = (direction == "new_old")
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=should_reverse)

                marked_count = 0
                large_body_candles = []

                for i, candle in enumerate(data):
                    if i >= len(contours): break
                    
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[i])
                    cx = x_rect + w_rect // 2
                    c_num = candle.get('candle_number', i)
                    
                    o = candle['open']
                    c = candle['close']
                    body_size = abs(c - o)
                    if body_size == 0: continue
                    is_bullish = c > o
                    
                    upper_wick = (candle['high'] - max(o, c))
                    lower_wick = (min(o, c) - candle['low'])
                    is_tall_body = (body_size >= 2 * upper_wick and body_size >= 2 * lower_wick)

                    should_number = number_all or (number_only_marked and is_tall_body)

                    if is_tall_body or should_number:
                        label_objects_and_text(
                            img=img,
                            cx=cx,
                            y_rect=y_rect,
                            h_rect=h_rect,
                            c_num=c_num if should_number else None,
                            custom_text=custom_text if is_tall_body else None,
                            is_bullish_arrow=is_bullish,
                            is_marked=is_tall_body,
                            double_arrow=(marker_type == "doublearrows"),
                            arrow_color=bullish_color if is_bullish else bearish_color
                        )

                    if is_tall_body:
                        enriched = candle.copy()
                        enriched.update({
                            "is_large_body": True,
                            "body_size": round(body_size, 5),
                            "upper_wick": round(upper_wick, 5),
                            "lower_wick": round(lower_wick, 5),
                            "direction": "bullish" if is_bullish else "bearish"
                        })
                        large_body_candles.append(enriched)
                        marked_count += 1

                if marked_count > 0 or number_all:
                    output_dir = os.path.join(dev_output_base, sym, tf)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, output_chart_name), img)
                    with open(os.path.join(output_dir, output_json_name), 'w', encoding='utf-8') as f:
                        json.dump(large_body_candles, f, indent=4)
                    total_marked += marked_count
                    processed_charts += 1

            except Exception as e:
                log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    log(f"LARGE BODY ANALYSIS COMPLETE | {processed_charts} charts | {total_marked} marked")
    return f"=== LARGE BODY ANALYSIS COMPLETED ===\nBroker: {broker_name}\nBars: {bars} | Numbering: {number_mode}\nTotal: {total_marked} across {processed_charts} charts."


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
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
   
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    hhhl_cfg = define_candles.get("higherhighsandhigherlows", {})
    
    if not hhhl_cfg:
        return f"[{broker_name}] Error: define_candles.higherhighsandhigherlows section missing."

    bars = hhhl_cfg.get("BARS", 101)
    output_filename_base = hhhl_cfg.get("filename", "highers.json")
    neighbor_left = hhhl_cfg.get("NEIGHBOR_LEFT", 5)
    neighbor_right = hhhl_cfg.get("NEIGHBOR_RIGHT", 5)
    direction = hhhl_cfg.get("read_candles_from", "new_old")
    number_mode = hhhl_cfg.get("number_candles", "define_candles").lower()

    label_cfg = hhhl_cfg.get("label", {})
    marker_type = label_cfg.get("marker", "arrow").lower()
    hh_text = label_cfg.get("higherhighs_text", "HH")
    hl_text = label_cfg.get("higherlows_text", "HL")

    output_chart_name = output_filename_base.replace(".json", ".png")
    output_json_name = output_filename_base

    source_json = f"new_old_{bars}.json" if direction == "new_old" else f"old_new_{bars}.json"
    source_chart = f"chart_{bars}.png"

    color_map = {"green": (0, 255, 0), "red": (0, 0, 255)}
    label_at = label_cfg.get("label_at", {})
    hh_color = color_map.get(label_at.get("higher_highs_color", "red").lower(), (0, 0, 255))
    hl_color = color_map.get(label_at.get("higher_lows_color", "green").lower(), (0, 255, 0))

    number_all = number_mode == "all"
    number_only_marked = number_mode in ["define_candles", "define_candle", "definecandle", "define_candle"]

    total_marked = 0
    processed_charts = 0

    log(f"Starting Swing Points Analysis | BARS: {bars} | Number mode: {number_mode}")

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p): continue
        
        for tf in sorted(os.listdir(sym_p)):
            tf_path = os.path.join(sym_p, tf)
            if not os.path.isdir(tf_path): continue

            source_json_path = os.path.join(tf_path, "candlesdetails", source_json)
            source_chart_path = os.path.join(tf_path, source_chart)

            if not os.path.exists(source_json_path) or not os.path.exists(source_chart_path):
                continue

            try:
                with open(source_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not data: continue

                data = sorted(data, key=lambda x: x.get('candle_number', 0))

                img = cv2.imread(source_chart_path)
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                should_reverse = (direction == "new_old")
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=should_reverse)

                swing_points = []
                n = len(data)
                for i in range(n):
                    if i + neighbor_right >= n: 
                        continue  # Not enough candles on the right

                    current_high = data[i]['high']
                    current_low = data[i]['low']

                    left_highs = [data[j]['high'] for j in range(max(0, i - neighbor_left), i)]
                    left_lows  = [data[j]['low']  for j in range(max(0, i - neighbor_left), i)]
                    right_highs = [data[j]['high'] for j in range(i + 1, min(i + neighbor_right + 1, n))]
                    right_lows  = [data[j]['low']  for j in range(i + 1, min(i + neighbor_right + 1, n))]

                    # Higher High: current high > all right highs, and > all left highs if any exist
                    is_higher_high = current_high > max(right_highs) if right_highs else False
                    if left_highs:
                        is_higher_high = is_higher_high and current_high > max(left_highs)

                    # Higher Low: current low < all right lows, and < all left lows if any exist
                    is_higher_low = current_low < min(right_lows) if right_lows else False
                    if left_lows:
                        is_higher_low = is_higher_low and current_low < min(left_lows)

                    if is_higher_high:
                        swing_points.append((i, "higher_high", current_high))
                    if is_higher_low:
                        swing_points.append((i, "higher_low", current_low))

                swing_candles = []
                marked_count = 0

                for i, candle in enumerate(data):
                    if i >= len(contours): 
                        break

                    c_num = candle.get('candle_number', i)
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[i])
                    cx = x_rect + w_rect // 2

                    matching_swings = [sp for sp in swing_points if sp[0] == i]
                    is_swing = len(matching_swings) > 0
                    should_number = number_all or (number_only_marked and is_swing)

                    if is_swing or should_number:
                        for _, swing_type, price in matching_swings:
                            custom_text = hl_text if swing_type == "higher_low" else hh_text
                            arrow_up = (swing_type == "higher_low")  # Up arrow for HL, down for HH
                            color = hl_color if arrow_up else hh_color

                            label_objects_and_text(
                                img=img,
                                cx=cx,
                                y_rect=y_rect,
                                h_rect=h_rect,
                                c_num=c_num if should_number else None,
                                custom_text=custom_text if is_swing else None,
                                is_bullish_arrow=arrow_up,
                                is_marked=is_swing,
                                double_arrow=(marker_type == "doublearrows"),
                                arrow_color=color
                            )

                            # Enrich JSON
                            enriched = candle.copy()
                            if swing_type == "higher_high":
                                enriched["is_higherhigh"] = True
                            else:
                                enriched["is_higherlow"] = True
                            enriched["swing_price"] = round(price, 5)
                            enriched["direction"] = "bullish" if candle['close'] > candle['open'] else "bearish"
                            swing_candles.append(enriched)
                            marked_count += 1

                if marked_count > 0 or number_all:
                    output_dir = os.path.join(dev_output_base, sym, tf)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, output_chart_name), img)
                    with open(os.path.join(output_dir, output_json_name), 'w', encoding='utf-8') as f:
                        json.dump(swing_candles, f, indent=4)
                    total_marked += marked_count
                    processed_charts += 1

            except Exception as e:
                log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    log(f"SWING ANALYSIS COMPLETE | {processed_charts} charts | {total_marked} swings")
    return f"=== HIGHER HIGHS & HIGHER LOWS ANALYSIS COMPLETED ===\nBroker: {broker_name}\nBars: {bars} | Numbering: {number_mode}\nTotal swings: {total_marked} across {processed_charts} charts."

def identify_lower_highs_lower_lows(broker_name):
    lagos_tz = pytz.timezone('Africa/Lagos')
   
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    dev_dict = load_developers_dictionary()
    cfg = dev_dict.get(broker_name)
    if not cfg:
        return f"[{broker_name}] Error: Broker not in dictionary."
   
    base_folder = cfg.get("BASE_FOLDER")
    dev_output_base = os.path.abspath(os.path.join(base_folder, "..", "developers", broker_name))
   
    am_data = get_account_management(broker_name)
    if not am_data:
        return f"[{broker_name}] Error: accountmanagement.json missing."
    
    chart_cfg = am_data.get("chart", {})
    define_candles = chart_cfg.get("define_candles", {})
    lhll_cfg = define_candles.get("lowerhighsandlowerlows", {})
    
    if not lhll_cfg:
        return f"[{broker_name}] Error: define_candles.lowerhighsandlowerlows section missing."

    bars = lhll_cfg.get("BARS", 101)
    output_filename_base = lhll_cfg.get("filename", "lowers.json")
    neighbor_left = lhll_cfg.get("NEIGHBOR_LEFT", 5)
    neighbor_right = lhll_cfg.get("NEIGHBOR_RIGHT", 5)
    direction = lhll_cfg.get("read_candles_from", "new_old")
    number_mode = lhll_cfg.get("number_candles", "define_candles").lower()

    label_cfg = lhll_cfg.get("label", {})
    marker_type = label_cfg.get("marker", "arrow").lower()
    lh_text = label_cfg.get("lowerhighs_text", "lh")
    ll_text = label_cfg.get("lowerlows_text", "ll")

    output_chart_name = output_filename_base.replace(".json", ".png")
    output_json_name = output_filename_base

    source_json = f"new_old_{bars}.json" if direction == "new_old" else f"old_new_{bars}.json"
    source_chart = f"chart_{bars}.png"

    color_map = {"green": (0, 255, 0), "red": (0, 0, 255)}
    label_at = label_cfg.get("label_at", {})
    lh_color = color_map.get(label_at.get("lower_highs_color", "red").lower(), (0, 0, 255))
    ll_color = color_map.get(label_at.get("lower_lows_color", "green").lower(), (0, 255, 0))

    number_all = number_mode == "all"
    number_only_marked = number_mode in ["define_candles", "define_candle", "definecandle", "define_candle"]

    total_marked = 0
    processed_charts = 0

    log(f"Starting Swing Points Analysis | BARS: {bars} | Number mode: {number_mode}")

    for sym in sorted(os.listdir(base_folder)):
        sym_p = os.path.join(base_folder, sym)
        if not os.path.isdir(sym_p): continue
        
        for tf in sorted(os.listdir(sym_p)):
            tf_path = os.path.join(sym_p, tf)
            if not os.path.isdir(tf_path): continue

            source_json_path = os.path.join(tf_path, "candlesdetails", source_json)
            source_chart_path = os.path.join(tf_path, source_chart)

            if not os.path.exists(source_json_path) or not os.path.exists(source_chart_path):
                continue

            try:
                with open(source_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not data: continue

                data = sorted(data, key=lambda x: x.get('candle_number', 0))

                img = cv2.imread(source_chart_path)
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255)) | cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                should_reverse = (direction == "new_old")
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=should_reverse)

                swing_points = []
                n = len(data)
                for i in range(n):
                    if i + neighbor_right >= n: 
                        continue  # Not enough candles on the right

                    current_high = data[i]['high']
                    current_low = data[i]['low']

                    left_highs = [data[j]['high'] for j in range(max(0, i - neighbor_left), i)]
                    left_lows  = [data[j]['low']  for j in range(max(0, i - neighbor_left), i)]
                    right_highs = [data[j]['high'] for j in range(i + 1, min(i + neighbor_right + 1, n))]
                    right_lows  = [data[j]['low']  for j in range(i + 1, min(i + neighbor_right + 1, n))]

                    # lower High: current high > all right highs, and > all left highs if any exist
                    is_lower_high = current_high > max(right_highs) if right_highs else False
                    if left_highs:
                        is_lower_high = is_lower_high and current_high > max(left_highs)

                    # lower Low: current low < all right lows, and < all left lows if any exist
                    is_lower_low = current_low < min(right_lows) if right_lows else False
                    if left_lows:
                        is_lower_low = is_lower_low and current_low < min(left_lows)

                    if is_lower_high:
                        swing_points.append((i, "lower_high", current_high))
                    if is_lower_low:
                        swing_points.append((i, "lower_low", current_low))

                swing_candles = []
                marked_count = 0

                for i, candle in enumerate(data):
                    if i >= len(contours): 
                        break

                    c_num = candle.get('candle_number', i)
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contours[i])
                    cx = x_rect + w_rect // 2

                    matching_swings = [sp for sp in swing_points if sp[0] == i]
                    is_swing = len(matching_swings) > 0
                    should_number = number_all or (number_only_marked and is_swing)

                    if is_swing or should_number:
                        for _, swing_type, price in matching_swings:
                            custom_text = ll_text if swing_type == "lower_low" else lh_text
                            arrow_up = (swing_type == "lower_low")  # Up arrow for ll, down for lh
                            color = ll_color if arrow_up else lh_color

                            label_objects_and_text(
                                img=img,
                                cx=cx,
                                y_rect=y_rect,
                                h_rect=h_rect,
                                c_num=c_num if should_number else None,
                                custom_text=custom_text if is_swing else None,
                                is_bullish_arrow=arrow_up,
                                is_marked=is_swing,
                                double_arrow=(marker_type == "doublearrows"),
                                arrow_color=color
                            )

                            # Enrich JSON
                            enriched = candle.copy()
                            if swing_type == "lower_high":
                                enriched["is_lowerhigh"] = True
                            else:
                                enriched["is_lowerlow"] = True
                            enriched["swing_price"] = round(price, 5)
                            enriched["direction"] = "bullish" if candle['close'] > candle['open'] else "bearish"
                            swing_candles.append(enriched)
                            marked_count += 1

                if marked_count > 0 or number_all:
                    output_dir = os.path.join(dev_output_base, sym, tf)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, output_chart_name), img)
                    with open(os.path.join(output_dir, output_json_name), 'w', encoding='utf-8') as f:
                        json.dump(swing_candles, f, indent=4)
                    total_marked += marked_count
                    processed_charts += 1

            except Exception as e:
                log(f"Error processing {sym}/{tf}: {e}", "ERROR")

    log(f"SWING ANALYSIS COMPLETE | {processed_charts} charts | {total_marked} swings")
    return f"=== lower HIGHS & lower LOWS ANALYSIS COMPLETED ===\nBroker: {broker_name}\nBars: {bars} | Numbering: {number_mode}\nTotal swings: {total_marked} across {processed_charts} charts."
   
def main():
    dev_dict = load_developers_dictionary()
    if not dev_dict:
        print("No developers to process.")
        return

    broker_names = sorted(dev_dict.keys())
    cores = cpu_count()
    print(f"--- STARTING MULTIPROCESSING (Cores: {cores}) ---")

    with Pool(processes=cores) as pool:
        # STEP 1: Large Body Analysis
        print("\n[STEP 1] Running Large Body Candle Analysis...")
        large_results = pool.map(analyze_tallest_body, broker_names)
        for r in large_results:
            print(r)

        # STEP 2: Higher Highs & Higher Lows
        print("\n[STEP 2] Running Higher Highs & Higher Lows Analysis...")
        hh_hl_results = pool.map(identify_higher_highs_higher_lows, broker_names)
        for r in hh_hl_results:
            print(r)

        # STEP 3: Lower Highs & Lower Lows (appends to same files)
        print("\n[STEP 3] Running Lower Highs & Lower Lows Analysis...")
        lh_ll_results = pool.map(identify_lower_highs_lower_lows, broker_names)
        for r in lh_ll_results:
            print(r)

    print("\n[SUCCESS] All tasks completed.")
    
if __name__ == "__main__":
    main()
    






                



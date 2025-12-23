import os
import json
import cv2
import numpy as np
import os
import json
from datetime import datetime
import pytz
import re
import shutil

def login_to_my_account():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    base_dev_path = r"C:\xampp\htdocs\chronedge\synarex\chart\developers"

    # Load brokersdictionary.json
    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND!", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            all_brokers_data = json.load(f)
    except Exception as e:
        print(f"Failed to read brokersdictionary.json: {e}", "CRITICAL")
        return {}

    # Find session.json in any developer folder
    session_data = None
    matched_folder_name = None  # The actual folder name on disk (e.g., "Deriv 6")
    broker_key_in_json = None   # The key in brokersdictionary.json (e.g., "deriv6")

    if not os.path.exists(base_dev_path):
        print(f"Developers base path not found: {base_dev_path}", "CRITICAL")
        return {}

    for folder_name in os.listdir(base_dev_path):
        folder_path = os.path.join(base_dev_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        session_path = os.path.join(folder_path, "session.json")
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                matched_folder_name = folder_name
                break
            except Exception as e:
                print(f"Failed to read session.json in '{folder_name}': {e}", "WARNING")
                continue

    if session_data is None:
        print("No session.json found in any developer broker folder.", "WARNING")
        return {}

    email = session_data.get("email", "").strip()
    password = session_data.get("password", "").strip()

    if not email or not password:
        print("session.json is missing email or password.", "WARNING")
        return {}

    # Match credentials against any broker in the dictionary
    matched_cfg = None
    for broker_key, cfg in all_brokers_data.items():
        if (cfg.get("ACCOUNT_EMAIL") == email and
            cfg.get("ACCOUNT_PASSWORD") == password):
            matched_cfg = cfg
            broker_key_in_json = broker_key
            break

    if not matched_cfg:
        print("Credentials in session.json do not match any broker in brokersdictionary.json", "WARNING")
        return {}

    # Success! Use the broker key from JSON (e.g., "deriv6")
    print(f"Login successful! Broker '{broker_key_in_json}' authenticated via session.json "
          f"(folder: '{matched_folder_name}')", "SUCCESS")

    # Optional: Check allowed symbols file — only warn, do NOT fail login
    if matched_folder_name:
        allowed_file_path = os.path.join(base_dev_path, matched_folder_name, "allowedsymbolsandvolumes.json")
        if not os.path.exists(allowed_file_path):
            print(f"Warning: allowedsymbolsandvolumes.json missing in folder '{matched_folder_name}'", "WARNING")
        else:
            print(f"allowedsymbolsandvolumes.json found.", "INFO")

    # Return exactly what custom_breakout() expects: { "deriv6": { ...config... } }
    return {broker_key_in_json: matched_cfg}
# === USAGE ===
session = login_to_my_account()




def custom_horizontal_line():
    import cv2
    import numpy as np
    import os
    import json
    from datetime import datetime
    import pytz

    lagos_tz = pytz.timezone('Africa/Lagos')

    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ------------------------------------------------------------------
    # PROFESSIONAL COLORS (same as trendline & markers)
    # ------------------------------------------------------------------
    COLOR_MAP = {
        "ph": (255, 100, 0),       # Orange
        "pl": (200, 0, 200),       # Magenta
        "ch": (255, 200, 0),       # Cyan
        "cl": (0, 140, 255),       # Warm Orange
        "fvg_middle_(bullish)": (0, 255, 0),
        "fvg_middle_(bearish)": (60, 20, 220),
    }

    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))

    def is_level_match(candle, key):
        if key == "ph": return candle.get("is_ph")
        if key == "pl": return candle.get("is_pl")
        if key == "ch": return candle.get("is_ch")
        if key == "cl": return candle.get("is_cl")
        if key == "fvg_middle_(bullish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bullish"
        if key == "fvg_middle_(bearish)": return candle.get("is_fvg_middle") and candle.get("fvg_direction", "").lower() == "bearish"
        return False

    def get_y_position(positions, candle_num, key):
        pos = positions[candle_num]
        if key in ["ph", "ch", "fvg_middle_(bullish)"]:
            return pos["high_y"]
        else:
            return pos["low_y"]

    # ------------------------------------------------------------------
    # Get candle positions (right = newest = candle_number 0)
    # ------------------------------------------------------------------
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None:
            return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        raw_positions = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            raw_positions[idx] = {"x": x + w // 2, "high_y": y, "low_y": y + h}
        return img.copy(), raw_positions

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    developer_brokers = {
        k: v for k, v in session.items()
        if v.get("POSITION", "").lower() == "developer"
    }

    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "breakout.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "breakout.json")
        if not os.path.exists(technique_path):
            log(f"breakout.json missing → {broker_raw_name}", "WARNING")
            continue

        try:
            with open(technique_path, 'r', encoding='utf-8') as f:
                tech = json.load(f)
        except Exception as e:
            log(f"Failed loading breakout.json: {e}", "ERROR")
            continue

        if str(tech.get("drawings_switch", {}).get("horizontal_line", "no")).strip().lower() != "yes":
            log(f"Horizontal lines disabled for {broker_raw_name}", "INFO")
            continue

        horiz_configs = tech.get("horizontal_line", {})
        horiz_list = []
        for key in sorted([k for k in horiz_configs.keys() if str(k).isdigit()]):
            conf = horiz_configs[key]
            if not isinstance(conf, dict): continue
            fr = conf.get("FROM", "").strip().lower().replace(" ", "_")
            to = conf.get("TO", "").strip().lower().replace(" ", "_")
            if fr and to:
                horiz_list.append({"id": key, "FROM": fr, "TO": to})

        if not horiz_list:
            log(f"No horizontal lines defined → {broker_raw_name}", "INFO")
            continue

        log(f"Processing {broker_raw_name} → {len(horiz_list)} horizontal lines")

        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue

            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue

                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_oldest_newest_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")

                if not os.path.exists(chart_path) or not os.path.exists(json_path):
                    continue

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        candles = json.load(f)  # index 0 = newest → last = oldest
                except:
                    continue

                img, raw_positions = get_candle_positions(chart_path)
                if img is None:
                    continue

                # Build correct positions: candle_number → coordinates
                positions = {}
                for i, candle in enumerate(candles):
                    if i >= len(raw_positions):
                        break
                    cnum = candle["candle_number"]
                    pos = raw_positions[i]
                    positions[cnum] = {
                        "x": pos["x"],
                        "high_y": pos["high_y"],
                        "low_y": pos["low_y"]
                    }

                anything_drawn = False

                # DRAW MARKERS (same as before)
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy = positions[cnum]["high_y"]
                    ly = positions[cnum]["low_y"]

                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                        anything_drawn = True
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                        anything_drawn = True
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                        anything_drawn = True
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                        anything_drawn = True
                    if candle.get("is_fvg_middle"):
                        dir_key = "fvg_middle_(bullish)" if candle.get("fvg_direction", "").lower() == "bullish" else "fvg_middle_(bearish)"
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP.get(dir_key, (180,180,180)), -1)
                        anything_drawn = True

                # DRAW HORIZONTAL LINES — stops exactly at TO candle
                for conf in horiz_list:
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    line_id = f"H{conf['id']}"
                    color = get_color(from_key)

                    from_candle = None
                    to_candle = None

                    # Scan from oldest to newest
                    for candle in reversed(candles):
                        if is_level_match(candle, from_key):
                            from_candle = candle
                            break

                    if not from_candle or from_candle["candle_number"] not in positions:
                        continue

                    fx = positions[from_candle["candle_number"]]["x"]
                    fy = get_y_position(positions, from_candle["candle_number"], from_key)

                    # Find first TO candle AFTER the FROM candle
                    found_from = False
                    for candle in reversed(candles):
                        if candle["candle_number"] == from_candle["candle_number"]:
                            found_from = True
                            continue
                        if found_from and is_level_match(candle, to_key):
                            to_candle = candle
                            break

                    if not to_candle or to_candle["candle_number"] not in positions:
                        continue  # No valid TO found

                    tx = positions[to_candle["candle_number"]]["x"]
                    ty = get_y_position(positions, to_candle["candle_number"], to_key)

                    # Draw perfectly horizontal line from FROM.x to TO.x at FROM.y
                    cv2.line(img, (fx, fy), (tx, fy), color, 2)

                    # Label on the right side
                    label_x = tx + 10
                    label_y = fy
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    anything_drawn = True

                if anything_drawn:
                    cv2.imwrite(output_path, img)
                    log(f"HORIZONTAL LINES + MARKERS → {symbol_folder}/{tf_folder}", "SUCCESS")
                else:
                    log(f"Nothing drawn → {symbol_folder}/{tf_folder}", "INFO")

    log("=== CUSTOM HORIZONTAL LINE (STOPS AT TO) COMPLETED ===", "SUCCESS")


def custom_breakout():

    # --- INITIAL SETUP ---
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ==================================================================
    # LEVEL FAMILY CLASSIFICATION (UNCHANGED CORE LOGIC)
    # ==================================================================
    BEARISH_FAMILY = {"ph", "ch"}
    BULLISH_FAMILY = {"pl", "cl"}
    
    # NEW: Dynamic family mapping based on input key
    def get_level_family(key):
        if key in BEARISH_FAMILY: return "bearish"
        if key in BULLISH_FAMILY: return "bullish"
        return None

    # This map is used to find opposite levels (e.g., ph -> {pl, cl})
    OPPOSITE_MAP = {
        "ph": {"pl", "cl"}, "ch": {"pl", "cl"},
        "pl": {"ph", "ch"}, "cl": {"ph", "ch"}
    }
    
    def is_bearish_level(key): return key in BEARISH_FAMILY
    def is_bullish_level(key): return key in BULLISH_FAMILY
    def is_level_match(candle, key): return candle.get(f"is_{key}", False)
    
    PARENT_LEVELS = {"ph", "pl"}
    CHILD_LEVELS = {"ch", "cl"}
    def is_parent_level(key): return key in PARENT_LEVELS
    def is_child_level(key): return key in CHILD_LEVELS
    
    COLOR_MAP = {
        "ph": (255, 100, 0), # Orange
        "pl": (200, 0, 200), # Purple
        "ch": (255, 200, 0), # Yellow
        "cl": (0, 140, 255), # Blue
        "fvg_middle": (0, 255, 0),
    }
    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))
        
    def get_y_position(positions, candle_num, key):
        pos = positions[candle_num]
        return pos["high_y"] if key in ["ph", "ch", "fvg_middle"] else pos["low_y"]

    # ------------------------------------------------------------------
    # MARKERS (unchanged)
    # ------------------------------------------------------------------
    def mark_breakout_extreme_interceptor(img, x, body_y, color):
        cv2.circle(img, (x, body_y), 18, color, 4)
        cv2.circle(img, (x, body_y), 14, (0, 255, 255), 2)
        arrow_start_x = x - 70
        arrow_end_x = x - 20
        cv2.arrowedLine(img, (arrow_start_x, body_y), (arrow_end_x, body_y), color, thickness=4, tipLength=0.3)

    def mark_breakout_candle(img, x, body_y, color):
        label_x = x - 38
        label_y = body_y + 6
        cv2.putText(img, "B", (label_x + 1, label_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, "B", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "B", (label_x - 1, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def mark_continuation_extreme_interceptor(img, x, body_y, color):
        cv2.circle(img, (x, body_y), 18, color, 4)
        cv2.circle(img, (x, body_y), 14, (0, 255, 255), 2)
        arrow_start_x = x - 70
        arrow_end_x = x - 20
        cv2.arrowedLine(img, (arrow_start_x, body_y), (arrow_end_x, body_y), color, thickness=4, tipLength=0.3)

    def mark_continuation_candle(img, x, body_y, color):
        label_x = x - 38
        label_y = body_y + 6
        cv2.putText(img, "B", (label_x + 1, label_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, "B", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "B", (label_x - 1, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def draw_opposition_arrow(img, x, y, color, direction_up=True):
        size = 20
        thickness = 3
        shaft_length = 40
        if direction_up:
            cv2.line(img, (x, y + size), (x, y + size + shaft_length), color, thickness)
            pts = np.array([[x, y + size], [x - 12, y + size + 12], [x + 12, y + size + 12]], np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.line(img, (x, y - size), (x, y - size - shaft_length), color, thickness)
            pts = np.array([[x, y - size], [x - 12, y - size - 12], [x + 12, y - size - 12]], np.int32)
            cv2.fillPoly(img, [pts], color)

    def draw_double_retest_arrow(img, x, y_price, color, direction_up=True):
        offset = 12
        draw_opposition_arrow(img, x - offset, y_price, color, direction_up=direction_up)
        draw_opposition_arrow(img, x + offset, y_price, color, direction_up=direction_up)

    def draw_target_zone_marker(img, x, y_price, color, size=10):
        cv2.rectangle(img, (x - size, y_price - size), (x + size, y_price + size), color, -1)

    # ------------------------------------------------------------------
    # Candle positions from chart.png (unchanged)
    # ------------------------------------------------------------------
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None: return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        bounds = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            bounds[idx] = {"x": center_x, "high_y": y, "low_y": y + h, "width": w}
        return img.copy(), bounds

    # ------------------------------------------------------------------
    # Line–rectangle intersection (unchanged)
    # ------------------------------------------------------------------
    def line_intersects_rect(x1, y1, x2, y2, rect_left, rect_top, rect_right, rect_bottom):
        expand = 6
        rect_left -= expand; rect_right += expand; rect_top -= expand; rect_bottom += expand
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10: return 0
            return 1 if val > 0 else 2
        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2); o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1); o4 = orientation(p2, q2, q1)
            if o1 != o2 and o3 != o4: return True
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True
            return False
        edges = [((rect_left, rect_top), (rect_right, rect_top)),
                 ((rect_right, rect_top), (rect_right, rect_bottom)),
                 ((rect_right, rect_bottom), (rect_left, rect_bottom)),
                 ((rect_left, rect_bottom), (rect_left, rect_top))]
        for p2, q2 in edges:
            if do_intersect((x1, y1), (x2, y2), p2, q2):
                return True
        return False
        
    # ------------------------------------------------------------------
    # MAIN LOOP (with dynamic family and key loading)
    # ------------------------------------------------------------------
    developer_brokers = {k: v for k, v in globals().get("session", {}).items() if v.get("POSITION", "").lower() == "developer"}
    
    global_tech_data = {}
    
    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path = os.path.join(base_folder, "..", "developers", broker_raw_name, "breakout.json")
        if not os.path.exists(technique_path):
            technique_path = os.path.join(base_folder, "breakout.json")
        if not os.path.exists(technique_path):
            log(f"breakout.json missing → {broker_raw_name}", "WARNING")
            continue
        
        with open(technique_path, 'r', encoding='utf-8') as f:
            tech = json.load(f)
            global_tech_data = tech # Store for later access
            
        if str(tech.get("drawings_switch", {}).get("trendline", "no")).strip().lower() != "yes":
            continue
        
        trend_configs = tech.get("trendline", {})
        trend_list = []
        for key in sorted([k for k in trend_configs.keys() if str(k).isdigit()]):
            conf = trend_configs[key]
            if not isinstance(conf, dict): continue
            
            # --- DYNAMIC CONFIGURATION LOADING ---
            fr = conf.get("FROM", "").strip().lower()
            direction = conf.get("DIRECTION", "").strip().lower()
            trend_family = conf.get("TREND", "").strip().lower() # NEW: Load the trend family
            
            rules = conf.get("rules", {})
            breakout_cond = rules.get("breakout_condition", "").strip().lower()
            
            # Load the dynamic point keys
            define_points = rules.get("define_trend_points", {})
            sender_key_name = define_points.get("sender_candle", "").strip().lower()
            receiver_key_name = define_points.get("receiver_candle", "").strip().lower()
            opposition_key_name = define_points.get("opposition_candle", "").strip().lower()
            retest_key_name = define_points.get("retest_candle", "").strip().lower()
            
            seq_count = 0
            if breakout_cond and "_sequence_candle" in breakout_cond:
                match = re.search(r"(\d+)_sequence_candle", breakout_cond)
                if match:
                    seq_count = int(match.group(1))

            if fr:
                trend_list.append({
                    "id": key,
                    "FROM": fr,
                    "TO": conf.get("TO", "ray").strip().lower(),
                    "rule": rules.get("extreme_intruder", "continue").strip().lower(),
                    "sender_condition": rules.get("sender_condition", "none").strip().lower(),
                    "interceptor_enabled": str(conf.get("INTERCEPTOR", "no")).strip().lower() == "yes",
                    "direction": direction,
                    "breakout_sequence_count": seq_count if direction == "breakout" and seq_count > 0 else 0,
                    "continuation_sequence_count": seq_count if direction == "continuation" and seq_count > 0 else 0,
                    # NEW DYNAMIC KEYS
                    "trend_family": trend_family, 
                    "point_keys": {
                        "sender": sender_key_name,
                        "receiver": receiver_key_name,
                        "opposition": opposition_key_name,
                        "retest": retest_key_name,
                    },
                    "horizontal_line_subject": rules.get("HORIZONTAL_LINE_SUBJECT", {}) 
                })

        if not trend_list:
            continue
            
        # Get Neighbor Right settings
        parent_neighbor_right = global_tech_data.get("parenthighsandlows", {}).get("NEIGHBOR_RIGHT", 15)
        child_neighbor_right = global_tech_data.get("childhighsandlows", {}).get("NEIGHBOR_RIGHT", 7)
        
        log(f"Processing {broker_raw_name} → {len(trend_list)} institutional trendlines (Parent NR={parent_neighbor_right}, Child NR={child_neighbor_right})")
        
        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue
            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue
                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_oldest_newest_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")
                report_path = os.path.join(tf_path, "custom_levels.json")
                if not all(os.path.exists(p) for p in [chart_path, json_path]):
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    candles = json.load(f)
                img, raw_positions = get_candle_positions(chart_path)
                if img is None: continue
                positions = {}
                for idx, data in sorted(raw_positions.items(), key=lambda x: x[1]["x"], reverse=True):
                    if idx < len(candles):
                        cnum = candles[idx]["candle_number"]
                        positions[cnum] = data
                
                # Draw level markers (unchanged)
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy, ly = positions[cnum]["high_y"], positions[cnum]["low_y"]
                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                    if candle.get("is_fvg_middle"):
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP["fvg_middle"], -1)
                
                final_teams = {}
                final_trendlines_for_redraw = []
                
                def draw_trendline(line_id, fx, fy, tx, ty, color, extreme_cnum=None, extreme_y=None):
                    cv2.line(img, (fx, fy), (tx, ty), color, 3)
                    label_x = tx + 15 if tx > fx else fx + 15
                    label_y = ty - 20 if fy < ty else ty + 25
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    if extreme_cnum and extreme_y is not None:
                        ex_x = positions[extreme_cnum]["x"]
                        pts = np.array([[ex_x, extreme_y - 15], [ex_x - 10, extreme_y], [ex_x + 10, extreme_y]], np.int32)
                        cv2.fillPoly(img, [pts], color)
                        
                def validate_sender_condition(sender_cnum, receiver_cnum, key, condition):
                    if condition == "none": return True
                    if sender_cnum not in positions or receiver_cnum not in positions: return False
                    sender_candle = next(c for c in candles if c["candle_number"] == sender_cnum)
                    receiver_candle = next(c for c in candles if c["candle_number"] == receiver_cnum)
                    if is_bearish_level(key):
                        return sender_candle["high"] >= receiver_candle["high"] if condition == "beyond" else sender_candle["high"] <= receiver_candle["high"]
                    elif is_bullish_level(key):
                        return sender_candle["low"] <= receiver_candle["low"] if condition == "beyond" else sender_candle["low"] >= receiver_candle["low"]
                    return True
                    
                def process_trendline(conf, depth=0, max_depth=50):
                    if depth > max_depth:
                        log(f"Max recursion depth for T{conf['id']}", "WARNING")
                        return False
                    
                    line_id = f"T{conf['id']}"
                    from_key = conf["FROM"]
                    to_key = conf["TO"]
                    rule = conf["rule"]
                    sender_condition = conf["sender_condition"]
                    interceptor_enabled = conf["interceptor_enabled"]
                    direction = conf["direction"]
                    breakout_seq_count = conf["breakout_sequence_count"]
                    color = get_color(from_key)
                    
                    # --- CORE: FINDING FROM/SENDER POINT ---
                    from_candle = next((c for c in reversed(candles) if is_level_match(c, from_key)), None)
                    if not from_candle or from_candle["candle_number"] not in positions:
                        return False
                        
                    from_cnum = from_candle["candle_number"]
                    fx = positions[from_cnum]["x"]
                    fy = get_y_position(positions, from_cnum, from_key)
                    
                    # --- CORE: FINDING TO/RECEIVER POINT ---
                    to_cnum = None
                    tx, ty = img.shape[1] - 30, fy # Default ray endpoint
                    
                    found_from = False
                    for c in reversed(candles):
                        if c["candle_number"] == from_cnum:
                            found_from = True
                            continue
                        if found_from and (to_key == "ray" or is_level_match(c, to_key)):
                            to_cnum = c["candle_number"]
                            if to_cnum in positions:
                                tx = positions[to_cnum]["x"]
                                ty = get_y_position(positions, to_cnum, to_key)
                            break
                            
                    is_ray = (to_cnum is None)

                    # --- INTERMEDIATE TOUCHES FOR EXTREME INTRUDER RULE ---
                    touches = []
                    min_c = min(from_cnum, to_cnum or from_cnum + 99999)
                    max_c = max(from_cnum, to_cnum or from_cnum + 99999)
                    
                    for c in candles:
                        cn = c["candle_number"]
                        if cn in [from_cnum, to_cnum] or cn not in positions: continue
                        if not (min_c <= cn <= max_c): continue
                        
                        pos = positions[cn]
                        if line_intersects_rect(fx, fy, tx, ty,
                                               pos["x"] - pos["width"]//2, pos["high_y"],
                                               pos["x"] + pos["width"]//2, pos["low_y"]):
                            touches.append(cn)
                            
                    extreme_cnum = extreme_y = None
                    if touches:
                        s_min, s_max = min(touches), max(touches)
                        if is_bearish_level(from_key):
                            best = max((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["high"], default=None)
                        else:
                            best = min((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["low"], default=None)
                            
                        if best:
                            extreme_cnum = best["candle_number"]
                            extreme_y = positions[extreme_cnum]["high_y"] if is_bearish_level(from_key) else positions[extreme_cnum]["low_y"]
                            
                    # --- APPLYING EXTREME INTRUDER RULE (new_from/new_to) ---
                    final_fx, final_fy = fx, fy
                    final_tx, final_ty = tx, ty
                    final_from_cnum = from_cnum
                    final_to_cnum = to_cnum
                    applied_rule = "continue"
                    
                    if rule == "new_from" and extreme_cnum:
                        final_fx = positions[extreme_cnum]["x"]
                        final_fy = extreme_y
                        final_from_cnum = extreme_cnum
                        applied_rule = "new_from"
                    elif rule == "new_to" and extreme_cnum:
                        final_tx = positions[extreme_cnum]["x"]
                        final_ty = extreme_y
                        final_to_cnum = extreme_cnum
                        applied_rule = "new_to"
                        
                    # --- VALIDATE SENDER CONDITION ---
                    sender_cnum = final_from_cnum
                    receiver_cnum = final_to_cnum if not is_ray else final_from_cnum # For condition, if ray, receiver is sender
                    if not validate_sender_condition(sender_cnum, receiver_cnum, from_key, sender_condition):
                        return False
                        
                    # --- DRAW INITIAL LINE AND STORE FOR FINAL PROCESSING ---
                    draw_trendline(line_id, int(final_fx), int(final_fy), int(final_tx), int(final_ty), color,
                                    extreme_cnum, extreme_y if rule in ["new_from", "new_to"] else None)
                                    
                    final_trendlines_for_redraw.append({
                        "line_id": line_id,
                        "from_final_valid_sender_x": int(final_fx),
                        "from_final_valid_sender_y": int(final_fy),
                        "from_final_valid_receiver_x": int(final_tx),
                        "from_final_valid_receiver_y": int(final_ty),
                        "receiver_cnum": final_to_cnum if final_to_cnum else final_from_cnum, # Use the actual 'TO' or 'FROM' if ray
                        "from_key": from_key,
                        "color": color,
                        "interceptor_enabled": interceptor_enabled,
                        "direction": direction,
                        "breakout_sequence_count": breakout_seq_count,
                        "trend_family": conf["trend_family"], # NEW: Store trend family
                        "point_keys": conf["point_keys"],     # NEW: Store dynamic point keys
                    })
                    
                    final_teams[line_id] = {"team": {
                        "trendline_info": {
                            "line_id": line_id,
                            "from_candle": final_from_cnum,
                            "to_candle": final_to_cnum,
                            "receiver_candle": final_to_cnum if final_to_cnum else final_from_cnum,
                            "is_ray": is_ray,
                            "intermediate_touches": len(touches),
                            "touched_candles": touches,
                            "extreme_intruder_candle": extreme_cnum,
                            "rule_applied": applied_rule,
                            "color": list(map(int, color)),
                            "interceptors": [],
                            "opposition_candle": None,
                            "breakout_extreme_interceptor_candle": None,
                            "breakout_sequence_candles": [],
                            "retest_candle": None,
                            "target_zone_candle": None 
                        }
                    }}
                    return True
                    
                for conf in trend_list:
                    process_trendline(conf)
                    
                # ==================================================================+
                # FINAL PROCESSING — INTERCEPTORS, OPPOSITION, RETEST, & TARGET ZONE|
                # ==================================================================+
                def CONTINUATION_EXTREME_INTERCEPTOR(final_valid_trends):
                    for trend in final_trendlines_for_redraw:
                        fx, fy = trend["from_final_valid_sender_x"], trend["from_final_valid_sender_y"]
                        tx, ty = trend["from_final_valid_receiver_x"], trend["from_final_valid_receiver_y"]
                        color = trend["color"]
                        line_id = trend["line_id"]
                        receiver_cnum = trend["receiver_cnum"]
                        sender_cnum = trend.get("sender_cnum") # Ensure you have sender_cnum in your trend dict
                        from_key = trend["from_key"]

                        if tx - fx == 0:
                            continue

                        # --- NEW: Identify Point Types (PH, CH, PL, CL) ---
                        # Get 'FROM' type from from_key (e.g., "bullish_ph" -> "ph")
                        from_point_type = from_key.split('_')[-1] if from_key else "unknown"
                        
                        # Get 'TO' type by checking the receiver candle's label in positions/candles
                        to_point_type = "unknown"
                        if receiver_cnum in positions:
                            # We look for the label assigned to this specific candle in our positions data
                            to_point_type = positions[receiver_cnum].get("label", "unknown").lower()

                        # 1. Redraw Ray
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))
                        cv2.line(img, (fx, fy), (pending_entry, extend_y), color, 3)
                        cv2.putText(img, line_id, (fx + 20, fy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                        # Mark Receiver
                        if receiver_cnum and receiver_cnum in positions:
                            rx = positions[receiver_cnum]["x"]
                            ry = get_y_position(positions, receiver_cnum, from_key)
                            cv2.circle(img, (rx, ry), 12, (0, 0, 0), -1)
                            cv2.putText(img, "R", (rx - 8, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # 2. Find continuation_interceptors + prepare enriched list
                        continuation_interceptors = []
                        touched_continuation_interceptors_enriched = []
                        receiver_x = positions.get(receiver_cnum, {}).get("x", -99999)

                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions:
                                continue
                            pos = positions[cn]
                            if pos["x"] <= receiver_x:
                                continue

                            intersects = line_intersects_rect(fx, fy, pending_entry, extend_y,
                                                            pos["x"] - pos["width"]//2, pos["high_y"],
                                                            pos["x"] + pos["width"]//2, pos["low_y"])

                            body_y = (pos["high_y"] + pos["low_y"]) // 2
                            candle_color = "green" if c["close"] > c["open"] else "red" if c["close"] < c["open"] else "doji"

                            touched_continuation_interceptors_enriched.append({
                                "candle_number": cn,
                                "color": candle_color,
                                "is_extreme": False,
                                "is_mutual": False
                            })

                            if intersects:
                                continuation_interceptors.append({
                                    "candle_number": cn,
                                    "x": pos["x"],
                                    "y": body_y,
                                    "high": c["high"],
                                    "low": c["low"],
                                    "color": candle_color
                                })

                        # 3. Find Extreme Interceptor (oldest opposing)
                        extreme_interceptor_cnum = None
                        extreme_interceptor_data = None
                        is_bullish_trend = is_bullish_level(from_key)

                        if continuation_interceptors:
                            continuation_interceptors.sort(key=lambda x: x["candle_number"])

                            if is_bullish_trend:
                                for intr in continuation_interceptors:
                                    if intr["color"] == "red":
                                        extreme_interceptor_cnum = intr["candle_number"]
                                        extreme_interceptor_data = intr
                            else:
                                for intr in continuation_interceptors:
                                    if intr["color"] == "green":
                                        extreme_interceptor_cnum = intr["candle_number"]
                                        extreme_interceptor_data = intr

                        # 4. Check DUAL RESPECT RULE
                        continuation_extreme_interceptor_mutual = None
                        if extreme_interceptor_data and extreme_interceptor_cnum:
                            next_cnum = extreme_interceptor_cnum - 1
                            if next_cnum > 0 and next_cnum in positions:
                                pos = positions[next_cnum]
                                candle = next((c for c in candles if c["candle_number"] == next_cnum), None)
                                if candle:
                                    body_y = (pos["high_y"] + pos["low_y"]) // 2
                                    ex_high = extreme_interceptor_data["high"]
                                    ex_low  = extreme_interceptor_data["low"]

                                    if candle["high"] <= ex_high and candle["low"] >= ex_low:
                                        continuation_extreme_interceptor_mutual = {
                                            "candle_number": next_cnum,
                                            "x": pos["x"],
                                            "y": body_y,
                                            "high": candle["high"],
                                            "low": candle["low"],
                                            "color": "green" if candle["close"] > candle["open"] else "red"
                                        }

                        # 5. Drawing
                        if extreme_interceptor_data:
                            ex = extreme_interceptor_data
                            mark_continuation_extreme_interceptor(img, ex["x"], ex["y"], color)
                            if continuation_extreme_interceptor_mutual:
                                mx, my = continuation_extreme_interceptor_mutual["x"], continuation_extreme_interceptor_mutual["y"]
                                cv2.circle(img, (mx, my), 7, color, 2)
                                cv2.circle(img, (mx, my), 6, (*color[:3], 120), -1)
                                cv2.putText(img, "M", (mx-6, my+6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                        # 5.5 Handle Limit (3rd position)
                        limit_interceptor_cnum = None
                        if extreme_interceptor_cnum:
                            limit_interceptor_cnum = extreme_interceptor_cnum - 2
                            if limit_interceptor_cnum > 0 and limit_interceptor_cnum in positions:
                                lpos = positions[limit_interceptor_cnum]
                                lx, ly = lpos["x"], (lpos["high_y"] + lpos["low_y"]) // 2
                                cv2.rectangle(img, (lx-8, ly-8), (lx+8, ly+8), color, 2)
                                cv2.putText(img, "L", (lx-5, ly+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # 6. Update JSON flags and Point Types
                        if extreme_interceptor_cnum:
                            for item in touched_continuation_interceptors_enriched:
                                if item["candle_number"] == extreme_interceptor_cnum:
                                    item["is_extreme"] = True
                                if continuation_extreme_interceptor_mutual and item["candle_number"] == continuation_extreme_interceptor_mutual["candle_number"]:
                                    item["is_mutual"] = True

                        # 7. Final Save to custom level JSON structure
                        if line_id in final_teams:
                            final_teams[line_id]["team"]["trendline_info"].update({
                                "FROM": from_point_type,  # ph, ch, pl, cl
                                "TO": to_point_type,      # ph, ch, pl, cl
                                "continuation_interceptors": continuation_interceptors,
                                "touched_continuation_interceptors": touched_continuation_interceptors_enriched,
                                "continuation_extreme_interceptor_candle": extreme_interceptor_cnum,
                                "Continuation_extreme_Interceptor_limit": limit_interceptor_cnum,
                                "continuation_extreme_interceptor_mutual": (
                                    {
                                        "candle_number": continuation_extreme_interceptor_mutual["candle_number"],
                                        "low": continuation_extreme_interceptor_mutual["low"],
                                        "high": continuation_extreme_interceptor_mutual["high"]
                                    } if continuation_extreme_interceptor_mutual else None
                                )
                            })

                            # Clean old keys
                            for key in ["opposition_candle", "retest_candle", "target_zone_candle",
                                        "continuation_sequence_candles", "breakout_extreme_interceptor_candle",
                                        "continuation_extreme_interceptor_mutuals"]:
                                final_teams[line_id]["team"]["trendline_info"][key] = None

                            final_valid_trends[line_id] = trend.copy()
                            final_valid_trends[line_id]['valid'] = True

                    log(f"→ {symbol_folder}/{tf_folder} | {len(final_valid_trends)} Trends Validated (Points: {from_point_type}->{to_point_type})", "SUCCESS")
                    return final_valid_trends                                     

                def OPPOSITION_BREAKOUT_INTERCEPTORS_RETEST_TARGET_ZONE(final_valid_trends):

                    def draw_final_trendline(trend, fx, fy, tx, ty, color):
                        if tx - fx == 0:
                            return
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))
                        cv2.line(img, (fx, fy), (pending_entry, extend_y), color, 3)
                        cv2.putText(img, trend["line_id"], (fx + 20, fy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                        # Mark receiver
                        receiver_cnum = trend["receiver_cnum"]
                        if receiver_cnum in positions:
                            rx = positions[receiver_cnum]["x"]
                            ry = get_y_position(positions, receiver_cnum, trend["from_key"])
                            cv2.circle(img, (rx, ry), 12, (0, 0, 0), -1)
                            cv2.putText(img, "R", (rx - 8, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    def get_candle_level(cnum):
                        """Helper to return the level string (ph/ch/pl/cl) for a given candle number, or None."""
                        if cnum not in [c["candle_number"] for c in candles]:
                            return None
                        candle = next(c for c in candles if c["candle_number"] == cnum)
                        for lvl in ["ph", "ch", "pl", "cl"]:
                            if candle.get(f"is_{lvl}"):
                                return lvl
                        return None

                    def process_single_trend(trend, depth=0, max_depth=5):
                        if depth > max_depth:
                            log(f"Max adaptation depth reached for {trend['line_id']} → DISCARDING", "WARNING")
                            return False, trend

                        line_id = trend["line_id"]
                        fx, fy = trend["from_final_valid_sender_x"], trend["from_final_valid_sender_y"]
                        tx, ty = trend["from_final_valid_receiver_x"], trend["from_final_valid_receiver_y"]
                        color = trend["color"]
                        from_key = trend["from_key"]
                        direction = trend["direction"]
                        seq_count = trend["breakout_sequence_count"]
                        point_keys = trend["point_keys"]
                        opposition_key_name = point_keys["opposition"]
                        retest_key_name = point_keys["retest"]
                        sender_condition = trend.get("sender_condition", "none")
                        extreme_rule = trend.get("extreme_rule", "continue")

                        # --- Initialize variables ---
                        target_zone_cnum = None
                        target_zone_candle = None
                        target_zone_candle_found = False
                        extreme_target_reached_candle = None
                        extreme_target_reached_candle_found = False
                        target_reached_limit_found = None  # New: the candle that gets the diamond
                        extreme_target_reached_mutual_candle = []
                        extreme_target_reached_mutual_candle_found = False
                        target_zone_mutuals_cnums = []
                        target_zone_mutuals_found = False
                        target_zone_mutual_limit_cnum = None
                        target_zone_mutual_limit_found = False
                        extreme_opposition_cnum = None
                        extreme_opposition_reason = None

                        # === Sender candle ===
                        if "sender_cnum" not in trend:
                            sender_candle = next((c for c in reversed(candles) if is_level_match(c, from_key)), None)
                            trend["sender_cnum"] = sender_candle["candle_number"] if sender_candle else None
                        sender_cnum = trend["sender_cnum"]
                        if not sender_cnum or sender_cnum not in positions:
                            return False, trend

                        receiver_cnum = trend["receiver_cnum"]
                        if tx - fx == 0:
                            return False, trend

                        # Pre-compute level types for final sender & receiver
                        final_sender_level = get_candle_level(sender_cnum)
                        final_receiver_level = get_candle_level(receiver_cnum)

                        # === Extend ray for detection ===
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))

                        # === 1. Find future interceptors ===
                        interceptors = []
                        receiver_x = positions.get(receiver_cnum, {}).get("x", -99999)
                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions or positions[cn]["x"] <= receiver_x:
                                continue
                            pos = positions[cn]
                            if line_intersects_rect(fx, fy, pending_entry, extend_y,
                                                    pos["x"] - pos["width"]//2, pos["high_y"],
                                                    pos["x"] + pos["width"]//2, pos["low_y"]):
                                body_y = (pos["high_y"] + pos["low_y"]) // 2
                                interceptors.append({
                                    "candle_number": cn, "x": pos["x"], "y": body_y,
                                    "high": c["high"], "low": c["low"], "close": c["close"], "open": c["open"],
                                    "candle": c
                                })

                        if not interceptors:
                            draw_final_trendline(trend, fx, fy, tx, ty, color)
                            final_teams[line_id] = {"team": {"trendline_info": {
                                "line_id": line_id,
                                "interceptors": [],
                                "touched_interceptors": [],
                                "opposition_candle": None,
                                "further_oppositions": [],
                                "extreme_opposition_candle": None,
                                "extreme_opposition_reason": "No interceptors found",
                                "valid": True,
                                "adapted": depth > 0,
                                "final_sender_cnum": trend["sender_cnum"],
                                "final_receiver_cnum": trend["receiver_cnum"],
                                "final_sender_level": final_sender_level,
                                "final_receiver_level": final_receiver_level,
                            }}}
                            log(f"{line_id}: No interceptors → Valid & drawn", "INFO")
                            return True, trend

                        # === 2. Find opposition candle(s) ===
                        opposition_cnum = None
                        further_oppositions = []
                        if receiver_cnum and opposition_key_name:
                            oldest_int_cnum = min(i["candle_number"] for i in interceptors)
                            receiver_candle = next((c for c in candles if c["candle_number"] == receiver_cnum), None)
                            if receiver_candle:
                                receiver_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                                if receiver_level:
                                    target_levels = BULLISH_FAMILY if opposition_key_name == "bullish" else BEARISH_FAMILY
                                    if is_parent_level(receiver_level):
                                        target_levels = {lvl for lvl in target_levels if is_parent_level(lvl)}
                                    for cn in range(receiver_cnum - 1, oldest_int_cnum - 1, -1):
                                        if cn <= 0: break
                                        candle = next((c for c in candles if c["candle_number"] == cn), None)
                                        if candle and any(candle.get(f"is_{lvl}") for lvl in target_levels):
                                            if opposition_cnum is None:
                                                opposition_cnum = cn
                                            else:
                                                further_oppositions.append(cn)

                        if opposition_cnum is None:
                            log(f"{line_id}: No opposition candle found → DISCARDING trendline", "INFO")
                            return False, trend

                        # === 3. Find extreme interceptor ===
                        extreme_interceptor_cnum = None
                        extreme_interceptor_data = None
                        pre_opp_ints = [i for i in interceptors if i["candle_number"] < opposition_cnum]
                        if pre_opp_ints:
                            pre_opp_ints.sort(key=lambda i: i["candle_number"])
                            opposing_red = is_bullish_level(from_key)
                            for intr in reversed(pre_opp_ints):
                                is_red = intr["close"] < intr["open"]
                                if (opposing_red and is_red) or (not opposing_red and not is_red):
                                    extreme_interceptor_cnum = intr["candle_number"]
                                    extreme_interceptor_data = intr
                                    break

                        if extreme_interceptor_cnum is None:
                            log(f"{line_id}: No extreme interceptor found → DISCARDING trendline", "INFO")
                            return False, trend

                        # === 4. Check breakout sequence ===
                        breakout_sequence_cnums = []
                        has_breakout = False
                        if direction == "breakout" and seq_count > 0:
                            ext_c = next(c for c in candles if c["candle_number"] == extreme_interceptor_cnum)
                            younger = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                            younger.sort(key=lambda x: x["candle_number"], reverse=True)
                            for i in range(len(younger) - seq_count + 1):
                                seq = younger[i:i + seq_count]
                                if is_bullish_level(from_key):
                                    if all(c["high"] < ext_c["high"] and c["low"] < ext_c["low"] for c in seq):
                                        breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                        break
                                else:
                                    if all(c["high"] > ext_c["high"] and c["low"] > ext_c["low"] for c in seq):
                                        breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                        break
                            has_breakout = bool(breakout_sequence_cnums)

                        # === BREAKOUT → Valid & draw current line ===
                        if has_breakout:
                            draw_final_trendline(trend, fx, fy, tx, ty, color)
                            if extreme_interceptor_data:
                                mark_breakout_extreme_interceptor(img, extreme_interceptor_data["x"], extreme_interceptor_data["y"], color)
                            for cnum in breakout_sequence_cnums:
                                if cnum in positions:
                                    body_y = (positions[cnum]["high_y"] + positions[cnum]["low_y"]) // 2
                                    mark_breakout_candle(img, positions[cnum]["x"], body_y, color)

                            # === Retest ===
                            retest_cnum = None
                            retest_candle = None
                            if retest_key_name and breakout_sequence_cnums:
                                younger = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                                younger.sort(key=lambda x: x["candle_number"], reverse=True)
                                allowed = BULLISH_FAMILY if retest_key_name == "bullish" else BEARISH_FAMILY
                                receiver_candle = next((c for c in candles if c["candle_number"] == receiver_cnum), None)
                                if receiver_candle:
                                    r_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                                    if r_level and is_parent_level(r_level):
                                        allowed = {lvl for lvl in allowed if is_parent_level(lvl)}
                                for c in younger:
                                    if c["candle_number"] in breakout_sequence_cnums: continue
                                    if any(c.get(f"is_{lvl}") for lvl in allowed):
                                        retest_cnum = c["candle_number"]
                                        retest_candle = c
                                        break

                            if retest_cnum and retest_cnum in positions and retest_candle:
                                level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                                is_bullish_retest = level in {"pl", "cl"}
                                y_price = positions[retest_cnum]["low_y"] if is_bullish_retest else positions[retest_cnum]["high_y"]
                                draw_double_retest_arrow(img, positions[retest_cnum]["x"], y_price, color, direction_up=is_bullish_retest)

                            # === Target Zone Marker ===
                            if retest_cnum and retest_candle:
                                level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                                nr = parent_neighbor_right if is_parent_level(level) else child_neighbor_right
                                if nr > 0:
                                    target_zone_cnum = retest_cnum - nr
                                    if target_zone_cnum > 0:
                                        target_zone_candle = next((c for c in candles if c["candle_number"] == target_zone_cnum), None)
                                        if target_zone_cnum in positions:
                                            target_zone_candle_found = True
                                            txz = positions[target_zone_cnum]["x"]
                                            tyz = positions[target_zone_cnum]["high_y"] - 20 if is_bullish_level(from_key) else positions[target_zone_cnum]["low_y"] + 20
                                            draw_target_zone_marker(img, txz, tyz, color)

                            # === Target Reached Diamond Marker ===
                            target_reached_cnums = []
                            if target_zone_cnum:
                                target_reach_candidates = [i for i in interceptors if i["candle_number"] < target_zone_cnum]
                                for c_data in target_reach_candidates:
                                    target_reached_cnums.append(c_data["candle_number"])
                                if target_reached_cnums:
                                    sorted_reached = sorted(target_reached_cnums, reverse=True)
                                    extreme_target_reached_candle = sorted_reached[0] if sorted_reached else None
                                    extreme_target_reached_candle_found = extreme_target_reached_candle is not None
                                    mutual_candidates = sorted_reached[1:3]
                                    extreme_target_reached_mutual_candle = []
                                    if extreme_target_reached_candle is not None:
                                        extreme_candle_data = next((c for c in candles if c["candle_number"] == extreme_target_reached_candle), None)
                                        if extreme_candle_data:
                                            is_bullish = is_bullish_level(from_key)
                                            for cnum in mutual_candidates:
                                                candidate_candle_data = next((c for c in candles if c["candle_number"] == cnum), None)
                                                if candidate_candle_data:
                                                    if (is_bullish and candidate_candle_data["high"] > extreme_candle_data["high"]) or \
                                                    (not is_bullish and candidate_candle_data["low"] < extreme_candle_data["low"]):
                                                        extreme_target_reached_mutual_candle.append(cnum)
                                    extreme_target_reached_mutual_candle_found = bool(extreme_target_reached_mutual_candle)

                                    # === NEW: Determine which candle gets the diamond (2 candles before extreme) ===
                                    target_reached_limit_found = extreme_target_reached_candle
                                    if extreme_target_reached_candle is not None:
                                        mark_cnum = extreme_target_reached_candle - 2
                                        if mark_cnum > 0 and mark_cnum in positions:
                                            target_reached_limit_found = mark_cnum

                                    # Draw diamond on the selected candle
                                    if target_reached_limit_found in positions:
                                        pos = positions[target_reached_limit_found]
                                        body_y = (pos["high_y"] + pos["low_y"]) // 2
                                        cv2.drawMarker(img, (pos["x"], body_y), color,
                                                    markerType=cv2.MARKER_DIAMOND,
                                                    markerSize=12, thickness=3)

                            # === Target Zone Mutuals ===
                            if target_zone_cnum and target_zone_candle:
                                is_bullish = is_bullish_level(from_key)
                                target_zone_price = target_zone_candle["high"] if is_bullish else target_zone_candle["low"]
                                younger_c1_num = target_zone_cnum - 1
                                younger_c2_num = target_zone_cnum - 2
                                younger_c3_num = target_zone_cnum - 3
                                candidates = []
                                if younger_c1_num > 0:
                                    c1 = next((c for c in candles if c["candle_number"] == younger_c1_num), None)
                                    if c1: candidates.append((younger_c1_num, c1))
                                if younger_c2_num > 0:
                                    c2 = next((c for c in candles if c["candle_number"] == younger_c2_num), None)
                                    if c2: candidates.append((younger_c2_num, c2))
                                if younger_c3_num > 0:
                                    c3 = next((c for c in candles if c["candle_number"] == younger_c3_num), None)
                                    if c3: candidates.append((younger_c3_num, c3))
                                for cn, candle in candidates[:2]:
                                    meets_condition = False
                                    if is_bullish:
                                        if candle["high"] >= target_zone_price:
                                            meets_condition = True
                                    else:
                                        if candle["low"] <= target_zone_price:
                                            meets_condition = True
                                    if meets_condition:
                                        target_zone_mutuals_cnums.append(cn)
                                target_zone_mutuals_found = bool(target_zone_mutuals_cnums)
                                target_zone_mutual_limit_cnum = None
                                if len(candidates) >= 3:
                                    cn, candle = candidates[2]
                                    target_zone_mutual_limit_cnum = cn
                                    target_zone_mutual_limit_found = True

                                for cn in target_zone_mutuals_cnums:
                                    if cn in positions:
                                        pos = positions[cn]
                                        body_y = (pos["high_y"] + pos["low_y"]) // 2
                                        cv2.drawMarker(img, (pos["x"], body_y), color,
                                                    markerType=cv2.MARKER_CROSS,
                                                    markerSize=10, thickness=2)
                                if target_zone_mutual_limit_cnum and target_zone_mutual_limit_cnum in positions:
                                    pos = positions[target_zone_mutual_limit_cnum]
                                    body_y = (pos["high_y"] + pos["low_y"]) // 2
                                    cv2.circle(img, (pos["x"], body_y), 8, color, 2)

                            # === Extreme Opposition Logic ===
                            extreme_opposition_reason = "Not calculated"
                            if target_zone_cnum is not None:
                                opposition_group = [opposition_cnum] + further_oppositions
                                valid_oppositions = [cn for cn in opposition_group if target_zone_cnum <= cn <= opposition_cnum]

                                if valid_oppositions:
                                    opp_candles = []
                                    for cn in valid_oppositions:
                                        candle = next((c for c in candles if c["candle_number"] == cn), None)
                                        if candle:
                                            opp_candles.append((cn, candle))

                                    if opp_candles:
                                        is_bullish_trend = is_bullish_level(from_key)
                                        if is_bullish_trend:
                                            extreme_opposition_cnum = max(opp_candles, key=lambda x: x[1]["high"])[0]
                                        else:
                                            extreme_opposition_cnum = min(opp_candles, key=lambda x: x[1]["low"])[0]

                                        if extreme_opposition_cnum and extreme_opposition_cnum in positions:
                                            opp_candle = next(c for c in candles if c["candle_number"] == extreme_opposition_cnum)
                                            opp_level = next(k for k in ["ph","ch","pl","cl"] if opp_candle.get(f"is_{k}"))
                                            direction_up = opp_level in {"pl", "cl"}
                                            arrow_y = positions[extreme_opposition_cnum]["low_y"] if direction_up else positions[extreme_opposition_cnum]["high_y"]
                                            draw_opposition_arrow(img, positions[extreme_opposition_cnum]["x"], arrow_y, color, direction_up=direction_up)

                                        extreme_opposition_reason = f"Selected extreme opposition: {extreme_opposition_cnum}"
                                    else:
                                        extreme_opposition_reason = "No candle data found for valid opposition candles"
                                else:
                                    extreme_opposition_reason = f"No opposition candles in range [{target_zone_cnum} - {opposition_cnum}]"
                            else:
                                extreme_opposition_reason = "No target zone found → extreme opposition skipped"

                            # === Special "X" marker for invalid target zone ===
                            invalid_target_zone = False
                            if target_zone_cnum is not None and extreme_opposition_cnum is not None and target_zone_candle is not None:
                                opp_candle = next((c for c in candles if c["candle_number"] == extreme_opposition_cnum), None)
                                tz_candle = target_zone_candle

                                if opp_candle and tz_candle:
                                    tz_x = positions[target_zone_cnum]["x"]
                                    tz_high_y = positions[target_zone_cnum]["high_y"]
                                    tz_low_y = positions[target_zone_cnum]["low_y"]

                                    if not is_bullish_level(from_key):  # Bearish
                                        if tz_candle["low"] < opp_candle["high"]:
                                            invalid_target_zone = True
                                            cv2.putText(img, "X", (tz_x, tz_low_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                                            cv2.putText(img, "X", (tz_x, tz_low_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                                    else:  # Bullish
                                        if tz_candle["high"] > opp_candle["low"]:
                                            invalid_target_zone = True
                                            cv2.putText(img, "X", (tz_x, tz_high_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                                            cv2.putText(img, "X", (tz_x, tz_high_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                            # === Final dictionary ===
                            final_info = {
                                "line_id": line_id,
                                "interceptors": [i for i in interceptors if "candle" not in i],
                                "touched_interceptors": [i["candle_number"] for i in interceptors],
                                "breakout_extreme_interceptor_candle": extreme_interceptor_cnum,
                                "breakout_sequence_candles": breakout_sequence_cnums,
                                "retest_candle": retest_cnum,
                                "target_zone_candle": target_zone_cnum,
                                "target_reached_candles": target_reached_cnums,
                                "extreme_target_reached_candle": extreme_target_reached_candle,
                                "target_reached_limit_found": target_reached_limit_found,  # ← NEW
                                "extreme_target_reached_mutual_candle": extreme_target_reached_mutual_candle,
                                "target_zone_mutuals": target_zone_mutuals_cnums,
                                "target_zone_mutual_limit": target_zone_mutual_limit_cnum,
                                "valid": True,
                                "adapted": depth > 0,
                                "target_zone_candle_found": target_zone_candle_found,
                                "extreme_target_reached_candle_found": extreme_target_reached_candle_found,
                                "extreme_target_reached_mutual_candle_found": extreme_target_reached_mutual_candle_found,
                                "target_zone_mutuals_found": target_zone_mutuals_found,
                                "target_zone_mutual_limit_found": target_zone_mutual_limit_found,
                                "final_sender_cnum": trend["sender_cnum"],
                                "final_receiver_cnum": trend["receiver_cnum"],
                                "final_sender_level": final_sender_level,
                                "final_receiver_level": final_receiver_level,
                            }

                            if invalid_target_zone:
                                final_info.update({
                                    "opposition_candle": None,
                                    "further_oppositions": [],
                                    "extreme_opposition_candle": None,
                                    "extreme_opposition_reason": "Removed – invalid target zone (X marker)",
                                    "breakout_extreme_interceptor_candle": None,
                                    "breakout_sequence_candles": [],
                                    "retest_candle": None,
                                    "target_zone_candle": None,
                                    "target_reached_candles": [],
                                    "extreme_target_reached_candle": None,
                                    "target_reached_limit_found": None,  # ← also cleared on invalid
                                    "extreme_target_reached_mutual_candle": [],
                                    "target_zone_mutuals": [],
                                    "target_zone_mutual_limit": None,
                                    "target_zone_candle_found": False,
                                    "target_zone_mutuals_found": False,
                                    "target_zone_mutual_limit_found": False,
                                    "extreme_target_reached_candle_found": False,
                                    "extreme_target_reached_mutual_candle_found": False,
                                    "valid": False,
                                    "final_sender_cnum": trend["sender_cnum"],
                                    "final_receiver_cnum": trend["receiver_cnum"],
                                    "final_sender_level": final_sender_level,
                                    "final_receiver_level": final_receiver_level,
                                })
                            else:
                                final_info.update({
                                    "opposition_candle": opposition_cnum,
                                    "further_oppositions": further_oppositions,
                                    "extreme_opposition_candle": extreme_opposition_cnum,
                                    "extreme_opposition_reason": extreme_opposition_reason,
                                })

                            final_teams[line_id] = {"team": {"trendline_info": final_info}}

                            log(f"{line_id}: BREAKOUT confirmed → Valid & drawn", "SUCCESS")
                            return True, trend

                        # === NO BREAKOUT → Try to adapt ===
                        if depth < max_depth:
                            touches = []
                            min_c = min(sender_cnum, receiver_cnum)
                            max_c = max(sender_cnum, receiver_cnum)
                            for c in candles:
                                cn = c["candle_number"]
                                if cn in [sender_cnum, receiver_cnum] or cn not in positions: continue
                                if not (min_c <= cn <= max_c): continue
                                pos = positions[cn]
                                if line_intersects_rect(fx, fy, tx, ty,
                                                        pos["x"] - pos["width"]//2, pos["high_y"],
                                                        pos["x"] + pos["width"]//2, pos["low_y"]):
                                    touches.append(cn)

                            extreme_cnum = extreme_y = None
                            if touches and extreme_rule != "continue":
                                s_min, s_max = min(touches), max(touches)
                                if is_bearish_level(from_key):
                                    best = max((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["high"], default=None)
                                else:
                                    best = min((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["low"], default=None)
                                if best:
                                    extreme_cnum = best["candle_number"]
                                    extreme_y = positions[extreme_cnum]["high_y"] if is_bearish_level(from_key) else positions[extreme_cnum]["low_y"]

                            new_fx, new_fy = fx, fy
                            new_tx, new_ty = tx, ty
                            new_receiver_cnum = receiver_cnum
                            if extreme_rule == "new_from" and extreme_cnum:
                                new_fx, new_fy = positions[extreme_cnum]["x"], extreme_y
                                trend["sender_cnum"] = extreme_cnum
                            elif extreme_rule == "new_to" and extreme_cnum:
                                new_tx, new_ty = positions[extreme_cnum]["x"], extreme_y
                                new_receiver_cnum = extreme_cnum

                            if (new_fx, new_fy, new_tx, new_ty) != (fx, fy, tx, ty):
                                if validate_sender_condition(sender_cnum, new_receiver_cnum, from_key, sender_condition):
                                    log(f"{line_id}: Adapting via extreme_intruder → {extreme_cnum}", "INFO")
                                    trend.update({
                                        "from_final_valid_sender_x": new_fx, "from_final_valid_sender_y": new_fy,
                                        "from_final_valid_receiver_x": new_tx, "from_final_valid_receiver_y": new_ty,
                                        "receiver_cnum": new_receiver_cnum,
                                        "line_id": f"{line_id}_adapted{depth+1}"
                                    })
                                    return process_single_trend(trend, depth + 1, max_depth)

                            # Fallback price extreme adaptation
                            if is_bullish_level(from_key):
                                candidate = min(interceptors, key=lambda i: i["low"])
                            else:
                                candidate = max(interceptors, key=lambda i: i["high"])

                            base_price = next(c["low"] if is_bullish_level(from_key) else c["high"] for c in candles if c["candle_number"] == receiver_cnum)
                            should_adapt = (is_bullish_level(from_key) and candidate["low"] < base_price) or \
                                        (not is_bullish_level(from_key) and candidate["high"] > base_price)

                            if should_adapt and validate_sender_condition(sender_cnum, candidate["candle_number"], from_key, sender_condition):
                                log(f"{line_id}: Adapting to price extreme → {candidate['candle_number']}", "INFO")
                                trend.update({
                                    "from_final_valid_receiver_x": candidate["x"],
                                    "from_final_valid_receiver_y": get_y_position(positions, candidate["candle_number"], from_key),
                                    "receiver_cnum": candidate["candle_number"],
                                    "line_id": f"{line_id}_adapted{depth+1}"
                                })
                                return process_single_trend(trend, depth + 1, max_depth)

                        # === DISCARD ===
                        log(f"{line_id}: No breakout + no valid adaptation → DISCARDING trendline", "INFO")
                        return False, trend

                    # --- MAIN EXECUTION ---
                    for trend in final_trendlines_for_redraw[:]:
                        orig_id = trend["line_id"].split("_")[0].lstrip("T")
                        orig_conf = next((c for c in trend_list if c["id"] == orig_id), None)
                        if orig_conf:
                            trend["sender_condition"] = orig_conf.get("sender_condition", "none")
                            trend["extreme_rule"] = orig_conf.get("rule", "continue")

                        was_valid, final_state = process_single_trend(trend, depth=0)

                        if was_valid:
                            base_id = final_state["line_id"].split("_")[0]
                            final_valid_trends[base_id] = final_state

                    log(f"→ {symbol_folder}/{tf_folder} | {len(final_valid_trends)} Trendlines validated for BREAKOUT (strict validation)", "SUCCESS")

                    return final_valid_trends

                def DRAW_ALL_POINTS_LEVELS(valid_trends):
                    if not valid_trends:
                        log("No valid trendlines → nothing to draw for points levels.", "INFO")
                        return

                    drawn_levels = 0
                    pending_entry = img.shape[1] - 10

                    for trend in valid_trends.values():
                        color = trend["color"]
                        base_id = trend["line_id"].split("_")[0].lstrip("T")
                        line_id = trend["line_id"].split("_")[0]
                        orig_conf = next((c for c in trend_list if c["id"] == base_id), None)
                        if not orig_conf:
                            continue

                        direction = orig_conf.get("direction", "continuation").lower()
                        horiz_cfg = orig_conf.get("horizontal_line_subject", {})
                        subject_wanted = horiz_cfg.get("subject", "").strip().lower()
                        entry_wanted = horiz_cfg.get("entry", "").strip().lower()

                        if not subject_wanted or entry_wanted not in {"high_price", "low_price"}:
                            continue

                        draw_high = entry_wanted == "high_price"
                        draw_low = entry_wanted == "low_price"

                        # === GET TARGET CANDLE ===
                        def get_target_cnum_and_label():
                            info = final_teams.get(line_id, {}).get("team", {}).get("trendline_info", {})
                            if subject_wanted == "sender":
                                x_check = trend["from_final_valid_sender_x"]
                                for cnum, pos in positions.items():
                                    if abs(pos["x"] - x_check) < 12:
                                        return cnum, "S"
                            elif subject_wanted == "receiver":
                                x_check = trend["from_final_valid_receiver_x"]
                                for cnum, pos in positions.items():
                                    if abs(pos["x"] - x_check) < 12:
                                        return cnum, "R"
                            elif subject_wanted in {"opposition", "opp"}:
                                cnum = info.get("extreme_opposition_candle") or info.get("opposition_candle")
                                if cnum and cnum in positions:
                                    return cnum, "EXT-OPP" if info.get("extreme_opposition_candle") else "OPP"
                            elif subject_wanted == "extreme":
                                cnum = info.get("breakout_extreme_interceptor_candle") or info.get("continuation_extreme_interceptor_candle")
                                if cnum and cnum in positions:
                                    return cnum, "EXT"
                            elif subject_wanted == "retest":
                                cnum = info.get("retest_candle")
                                if cnum and cnum in positions:
                                    return cnum, "RET"
                            elif subject_wanted in {"target_zone", "target", "tz"}:
                                cnum = info.get("target_zone_candle")
                                if cnum and cnum in positions:
                                    return cnum, "TZ"
                            return None, ""

                        target_cnum, label = get_target_cnum_and_label()
                        if not target_cnum or target_cnum not in positions:
                            continue

                        # For breakout: require target zone
                        if direction == "breakout":
                            tz_cnum = final_teams.get(line_id, {}).get("team", {}).get("trendline_info", {}).get("target_zone_candle")
                            if tz_cnum is None:
                                continue

                        # === FULL CANDLE DATA ===
                        source_candle = next(c for c in candles if c["candle_number"] == target_cnum)
                        high_price = source_candle["high"]
                        low_price = source_candle["low"]
                        pos = positions[target_cnum]
                        high_y = pos["high_y"]
                        low_y = pos["low_y"]

                        # === TOUCHES & BLOCKER ===
                        touched_candles = []
                        start_x = pos["x"]
                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions or positions[cn]["x"] <= start_x:
                                continue
                            p = positions[cn]
                            touched = False
                            if draw_high and p["high_y"] <= high_y <= p["low_y"]:
                                touched = True
                            elif draw_low and p["high_y"] <= low_y <= p["low_y"]:
                                touched = True
                            elif draw_high and abs(p["high_y"] - high_y) < 8:
                                touched = True
                            elif draw_low and abs(p["low_y"] - low_y) < 8:
                                touched = True
                            if touched:
                                body_y = (p["high_y"] + p["low_y"]) // 2
                                touched_candles.append({
                                    "candle_number": cn, "x": p["x"], "y": body_y,
                                    "high": c["high"], "low": c["low"], "close": c["close"], "open": c["open"]
                                })

                        extreme_touch_cnum = None
                        extreme_touch_x = pending_entry
                        target_zone_cnum = final_teams.get(line_id, {}).get("team", {}).get("trendline_info", {}).get("target_zone_candle")
                        if target_zone_cnum is not None and touched_candles:
                            candidates = [t for t in touched_candles if t["candle_number"] < target_zone_cnum]
                            if candidates:
                                extreme_touch = max(candidates, key=lambda x: x["candle_number"])
                                extreme_touch_cnum = extreme_touch["candle_number"]
                                extreme_touch_x = positions[extreme_touch_cnum]["x"]
                                cv2.putText(img, "blck", (extreme_touch["x"] + 20, extreme_touch["y"] - 10),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

                        end_x = extreme_touch_x

                        # === BREAKOUT VALIDATION RULE ===
                        if direction == "breakout" and target_zone_cnum is not None:
                            tz_candle = next((c for c in candles if c["candle_number"] == target_zone_cnum), None)
                            if tz_candle:
                                tz_high = tz_candle["high"]
                                tz_low = tz_candle["low"]

                                should_draw = True
                                if "bear" in direction or "down" in direction.lower():
                                    if draw_high and tz_high <= low_price:
                                        should_draw = False
                                    elif draw_low and tz_high <= high_price:
                                        should_draw = False
                                elif "bull" in direction or "up" in direction.lower():
                                    if draw_high and tz_low >= high_price:
                                        should_draw = False
                                    elif draw_low and tz_low >= high_price:
                                        should_draw = False

                                if not should_draw:
                                    continue

                        # === DRAW LINES ===
                        if draw_high:
                            cv2.line(img, (pos["x"], high_y), (end_x, high_y), color, 1, cv2.LINE_AA)
                            cv2.putText(img, f"{label}-H", (pos["x"] - 58, high_y + 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                        if draw_low:
                            cv2.line(img, (pos["x"], low_y), (end_x, low_y), color, 1, cv2.LINE_AA)
                            cv2.putText(img, f"{label}-L", (pos["x"] - 58, low_y + 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                        drawn_levels += 1

                        # === SAVE PERFECT DATA (with exit_price and ordered placement) ===
                        if line_id not in final_teams:
                            final_teams[line_id] = {"team": {"trendline_info": {}}}

                        info = final_teams[line_id]["team"]["trendline_info"]

                        subject_map = {
                            "sender": "Sender", "receiver": "Receiver", "opposition": "Opposition",
                            "extreme": "Extreme", "retest": "Retest", "target_zone": "Target Zone"
                        }
                        display_subject = subject_map.get(subject_wanted, subject_wanted.title())

                        entry_price = high_price if draw_high else low_price
                        exit_price = low_price if draw_high else high_price  # Opposite extreme

                        # Build enhanced pending_entry_point with exit_price
                        pending_entry_point = {
                            "pending_entry_high_price": high_price,
                            "pending_entry_low_price": low_price,
                            "candle_number": target_cnum,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "type": "high" if draw_high else "low",
                            "subject": display_subject
                        }

                        # Preserve all existing data except pending_entry_point (to avoid duplication)
                        preserved_data = {k: v for k, v in info.items() if k != "pending_entry_point"}

                        # Reconstruct info with desired order: line_id → pending_entry_point → everything else
                        new_info = {
                            "line_id": info.get("line_id", line_id),
                            "pending_entry_point": pending_entry_point,  # Now high in JSON output
                        }
                        new_info.update(preserved_data)  # Add back interceptors, etc.

                        # Add the rest of the pending entry fields
                        new_info.update({
                            "pending_entry_subject": display_subject,
                            "pending_entry_candle": target_cnum,
                            "pending_entry_high_price": high_price,
                            "pending_entry_low_price": low_price,
                            "pending_entry_is_full_extension": extreme_touch_cnum is None,
                            "horizontal_level_blockers": touched_candles,
                            "extreme_horizontal_level_blocker": extreme_touch_cnum,
                        })

                        # Apply the new structure
                        info.clear()
                        info.update(new_info)

                    log(f"→ {symbol_folder}/{tf_folder} | {drawn_levels} Horizontal levels drawn (with exit_price in pending_entry_point)", "SUCCESS")
 
                def enrich_candle_details(candles_list, final_teams, final_valid_trends, report_path, technique_path):
                    import os
                    import json
                    
                    candle_lookup = {c["candle_number"]: c for c in candles_list}
                    enriched_count = 0
                    ESSENTIAL_FIELDS = ["open", "high", "low", "close", "time", "timeframe", "symbol"]

                    # === 1. Load breakout.json to determine if it's breakout-only ===
                    is_breakout_strategy = False
                    try:
                        with open(technique_path, 'r', encoding='utf-8') as f:
                            technique_config = json.load(f)
                        
                        trendline_configs = technique_config.get("trendline", {})
                        if isinstance(trendline_configs, dict):
                            for config in trendline_configs.values():
                                if isinstance(config, dict):
                                    direction = config.get("DIRECTION", "").strip().lower()
                                    if direction == "breakout":
                                        is_breakout_strategy = True
                                        break  # One breakout config → treat whole technique as breakout
                    except Exception as e:
                        log(f"Failed to load technique JSON for direction check: {e}", "ERROR")
                        # Fallback: assume continuation if cannot determine
                        is_breakout_strategy = False

                    if not is_breakout_strategy:
                        log("Technique is continuation-only → processing as continuation", "INFO")
                    else:
                        log("Technique is breakout-configured → processing as breakout", "INFO")

                    # === Helpers ===
                    def enrich_if_needed(container, key):
                        nonlocal enriched_count
                        val = container.get(key)
                        if isinstance(val, int) and val in candle_lookup:
                            full = candle_lookup[val]
                            clean = {"candle_number": val}
                            for f in ESSENTIAL_FIELDS:
                                clean[f] = full[f]
                            container[key] = clean
                            enriched_count += 1

                    def enrich_list_if_needed(container, key):
                        nonlocal enriched_count
                        items = container.get(key, [])
                        if not isinstance(items, list):
                            return
                        new_list = []
                        for item in items:
                            if isinstance(item, int):
                                cn = item
                                if cn in candle_lookup:
                                    full = candle_lookup[cn]
                                    clean = {"candle_number": cn}
                                    for f in ESSENTIAL_FIELDS:
                                        clean[f] = full[f]
                                    new_list.append(clean)
                                    enriched_count += 1
                                else:
                                    new_list.append({"candle_number": cn})
                            elif isinstance(item, dict):
                                cn = item.get("candle_number")
                                if isinstance(cn, int):
                                    cleaned = {"candle_number": cn}
                                    for f in ESSENTIAL_FIELDS:
                                        cleaned[f] = item.get(f, candle_lookup.get(cn, {}).get(f))
                                    new_list.append(cleaned)
                                    enriched_count += 1
                                else:
                                    new_list.append(item)
                            else:
                                new_list.append(item)
                        if new_list:
                            container[key] = new_list

                    # === MAIN LOOP ===
                    for line_id, data in final_teams.items():
                        info = data["team"]["trendline_info"]

                        # === Direction is now globally determined from breakout.json ===
                        direction = "breakout" if is_breakout_strategy else "continuation"

                        # === Determine trendline_connection_type ===
                        trendline_connection_type = "unknown"
                        if direction == "continuation":
                            from_type = info.get("FROM")
                            if from_type in ["pl", "cl"]:
                                trendline_connection_type = "bullish"
                            elif from_type in ["ph", "ch"]:
                                trendline_connection_type = "bearish"
                        else:  # breakout
                            sender_level = info.get("final_sender_level")
                            if sender_level in ["pl", "cl"]:
                                trendline_connection_type = "bullish"
                            elif sender_level in ["ph", "ch"]:
                                trendline_connection_type = "bearish"
                            else:
                                connection_type = info.get("connection_type", "unknown")
                                if connection_type in ["ph to ph", "ch to ch"]:
                                    trendline_connection_type = "bearish"
                                elif connection_type in ["pl to pl", "cl to cl"]:
                                    trendline_connection_type = "bullish"

                        # === Order type based on connection type ===
                        order_type = None
                        if direction == "continuation":
                            if trendline_connection_type == "bullish":
                                order_type = "buy now"
                            elif trendline_connection_type == "bearish":
                                order_type = "sell now"
                        else:  # breakout
                            if trendline_connection_type == "bullish":
                                order_type = "sell now"
                            elif trendline_connection_type == "bearish":
                                order_type = "buy now"

                        # === Common enrichments ===
                        common_single = ["pending_entry_candle", "extreme_horizontal_level_blocker"]
                        for key in common_single:
                            enrich_if_needed(info, key)
                        common_lists = ["touched_interceptors", "horizontal_level_blockers", "interceptors"]
                        for key in common_lists:
                            enrich_list_if_needed(info, key)

                        entry = info.get("pending_entry_point")
                        if isinstance(entry, dict) and isinstance(entry.get("candle_number"), int):
                            cn = entry["candle_number"]
                            if cn in candle_lookup:
                                full = candle_lookup[cn]
                                entry["candle"] = {"candle_number": cn}
                                for f in ESSENTIAL_FIELDS:
                                    entry["candle"][f] = full[f]
                                enriched_count += 1

                        # === Pre-compute booleans ===
                        pre_bools = {
                            "valid": info.get("valid", False),
                            "adapted": info.get("adapted", False),
                            "retest_candle_found": bool(info.get("retest_candle")),
                            "target_zone_candle_found": bool(info.get("target_zone_candle")),
                            "target_zone_mutuals_found": bool(info.get("target_zone_mutuals")),
                            "target_zone_mutual_limit_found": bool(info.get("target_zone_mutual_limit")),
                            "extreme_target_reached_candle_found": bool(info.get("extreme_target_reached_candle")),
                            "extreme_target_reached_mutual_candle_found": bool(info.get("extreme_target_reached_mutual_candle")),
                            "extreme_target_reached_limit_found": bool(info.get("target_reached_limit_found")),
                            "continuation_extreme_interceptor_found": bool(info.get("continuation_extreme_interceptor_candle")),
                            "continuation_extreme_interceptor_limit_found": bool(info.get("Continuation_extreme_Interceptor_limit")),
                            "pending_entry_is_full_extension": info.get("pending_entry_is_full_extension", False),
                            "extreme_opposition_candle_found": bool(info.get("extreme_opposition_candle")),
                            "extreme_horizontal_level_blocker_found": bool(info.get("extreme_horizontal_level_blocker")),
                        }

                        # === Keys to remove based on direction ===
                        breakout_only_keys = [
                            "breakout_extreme_interceptor_candle", "breakout_sequence_candles", "retest_candle",
                            "target_zone_candle", "target_reached_candles", "extreme_target_reached_candle",
                            "target_reached_limit_found", "extreme_target_reached_mutual_candle",
                            "target_zone_mutuals", "target_zone_mutual_limit", "opposition_candle",
                            "further_oppositions", "extreme_opposition_candle", "extreme_opposition_reason",
                        ]
                        continuation_only_keys = [
                            "continuation_extreme_interceptor_candle",
                            "Continuation_extreme_Interceptor_limit",
                            "continuation_extreme_interceptor_mutual",
                            "touched_continuation_interceptors",
                            "FROM", "TO",
                        ]

                        # === Rebuild new_info ===
                        new_info = {
                            "line_id": info.get("line_id", line_id),
                            "final_sender_level": info.get("final_sender_level"),
                            "final_receiver_level": info.get("final_receiver_level"),
                        }
                        active_order = {"reason": "no active order"}

                        if direction == "continuation":
                            for key in breakout_only_keys:
                                info.pop(key, None)

                            enrich_if_needed(info, "continuation_extreme_interceptor_candle")
                            enrich_if_needed(info, "Continuation_extreme_Interceptor_limit")
                            if "continuation_extreme_interceptor_mutual" in info:
                                mutual_data = info["continuation_extreme_interceptor_mutual"]
                                if isinstance(mutual_data, dict) and isinstance(mutual_data.get("candle_number"), int):
                                    cn = mutual_data["candle_number"]
                                    if cn in candle_lookup:
                                        full = candle_lookup[cn]
                                        clean = {"candle_number": cn}
                                        for f in ESSENTIAL_FIELDS:
                                            clean[f] = full[f]
                                        info["continuation_extreme_interceptor_mutual"] = clean
                                        enriched_count += 1
                                else:
                                    info.pop("continuation_extreme_interceptor_mutual", None)
                            enrich_list_if_needed(info, "touched_continuation_interceptors")

                            # Continuation instant order logic
                            ext_cnum = info.get("continuation_extreme_interceptor_candle")
                            ext_candle = ext_cnum if isinstance(ext_cnum, dict) else candle_lookup.get(ext_cnum) if isinstance(ext_cnum, int) else None
                            ext_open = ext_candle.get("open") if ext_candle else None
                            mutual_candle = info.get("continuation_extreme_interceptor_mutual")
                            has_mutual = bool(mutual_candle)
                            mutual_open = mutual_candle.get("open") if isinstance(mutual_candle, dict) else None
                            limit_found = pre_bools["continuation_extreme_interceptor_limit_found"]
                            ext_found = pre_bools["continuation_extreme_interceptor_found"]

                            if ext_found and not has_mutual and not limit_found and ext_open is not None:
                                active_order = {
                                    "instant_entry": ext_open,
                                    "order_type": order_type,
                                    "fact": "continuation extreme interceptor is active",
                                    "order_from": "continuation_extreme_interceptor"
                                }
                            elif has_mutual and not limit_found and mutual_open is not None:
                                active_order = {
                                    "instant_entry": mutual_open,
                                    "order_type": order_type,
                                    "fact": "continuation extreme interceptor mutuals is now active",
                                    "order_from": "continuation_extreme_interceptor_mutuals"
                                }
                            elif limit_found:
                                active_order = {"reason": "no active order", "order_from": "none"}

                            new_info.update({
                                "valid": pre_bools["valid"],
                                "adapted": pre_bools["adapted"],
                                "continuation_extreme_interceptor_found": ext_found,
                                "continuation_extreme_interceptor_limit_found": limit_found,
                                "trendline_connection_type": trendline_connection_type,
                                "active_instant_order": active_order
                            })

                        else:  # breakout
                            for key in continuation_only_keys:
                                info.pop(key, None)

                            enrich_if_needed(info, "retest_candle")
                            enrich_if_needed(info, "target_zone_candle")
                            enrich_if_needed(info, "extreme_target_reached_candle")
                            enrich_if_needed(info, "target_reached_limit_found")
                            enrich_if_needed(info, "extreme_opposition_candle")
                            enrich_list_if_needed(info, "breakout_sequence_candles")
                            enrich_list_if_needed(info, "target_zone_mutuals")
                            enrich_list_if_needed(info, "extreme_target_reached_mutual_candle")

                            # Breakout instant order logic
                            if (pre_bools["valid"] or pre_bools["adapted"]) and order_type and pre_bools["retest_candle_found"] and pre_bools["target_zone_candle_found"]:
                                tz_candle = info.get("target_zone_candle")
                                tz_mutuals = info.get("target_zone_mutuals", [])
                                ext_candle = info.get("extreme_target_reached_candle")
                                ext_mutuals = info.get("extreme_target_reached_mutual_candle", [])
                                tz_open = tz_candle.get("open") if isinstance(tz_candle, dict) else None
                                latest_mutual_open = None
                                if tz_mutuals:
                                    sorted_mutuals = sorted(tz_mutuals, key=lambda x: x.get("candle_number", 0), reverse=True)
                                    latest_mutual = sorted_mutuals[0]
                                    latest_mutual_open = latest_mutual.get("open")
                                latest_ext_open = None
                                if ext_mutuals:
                                    sorted_ext = sorted(ext_mutuals, key=lambda x: x.get("candle_number", 0), reverse=True)
                                    latest_ext = sorted_ext[0]
                                    latest_ext_open = latest_ext.get("open")
                                ext_open = ext_candle.get("open") if isinstance(ext_candle, dict) else None

                                if (not pre_bools["target_zone_mutuals_found"] and
                                    not pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if tz_open is not None:
                                        active_order = {
                                            "instant_entry": tz_open,
                                            "order_type": order_type,
                                            "fact": "price retested and at target zone",
                                            "order_from": "target_zone"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    not pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if latest_mutual_open is not None:
                                        active_order = {
                                            "instant_entry": latest_mutual_open,
                                            "order_type": order_type,
                                            "fact": "price retested, target zone is no more active but using most target zone mutuals",
                                            "order_from": "target_zone_mutuals"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    active_order = {
                                        "reason": "target zone and its mutual limit has been reached no orders for now",
                                        "order_from": "none"
                                    }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if ext_open is not None:
                                        active_order = {
                                            "instant_entry": ext_open,
                                            "order_type": order_type,
                                            "fact": "target reached candle is active",
                                            "order_from": "target reached candle"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if latest_ext_open is not None:
                                        active_order = {
                                            "instant_entry": latest_ext_open,
                                            "order_type": order_type,
                                            "fact": "target reached candle mutual is active",
                                            "order_from": "target reached mutual candle"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    pre_bools["extreme_target_reached_limit_found"]):
                                    active_order = {
                                        "reason": "no active order, all limit is reached",
                                        "order_from": "instant orders limit reached"
                                    }

                            new_info.update({
                                "valid": pre_bools["valid"],
                                "adapted": pre_bools["adapted"],
                                "extreme_opposition_candle_found": pre_bools["extreme_opposition_candle_found"],
                                "retest_candle_found": pre_bools["retest_candle_found"],
                                "target_zone_candle_found": pre_bools["target_zone_candle_found"],
                                "target_zone_mutuals_found": pre_bools["target_zone_mutuals_found"],
                                "target_zone_mutual_limit_found": pre_bools["target_zone_mutual_limit_found"],
                                "extreme_target_reached_candle_found": pre_bools["extreme_target_reached_candle_found"],
                                "extreme_target_reached_mutual_candle_found": pre_bools["extreme_target_reached_mutual_candle_found"],
                                "extreme_target_reached_limit_found": pre_bools["extreme_target_reached_limit_found"],
                                "extreme_horizontal_level_blocker_found": pre_bools["extreme_horizontal_level_blocker_found"],
                                "pending_entry_is_full_extension": pre_bools["pending_entry_is_full_extension"],
                                "trendline_connection_type": trendline_connection_type,
                                "active_instant_order": active_order
                            })

                        # === Copy remaining fields (safely) ===
                        for k, v in info.items():
                            if k not in new_info and k not in {"final_sender_level", "final_receiver_level", "valid", "adapted"}:
                                if k == "continuation_extreme_interceptor_mutual" and direction == "continuation" and v is not None:
                                    new_info[k] = v
                                elif k != "continuation_extreme_interceptor_mutual":
                                    new_info[k] = v

                        data["team"]["trendline_info"] = new_info

                    # === Enrich final_valid_trends ===
                    for trend in final_valid_trends.values():
                        enrich_if_needed(trend, "sender_cnum")
                        enrich_if_needed(trend, "receiver_cnum")

                    log(f"FINAL ENRICHMENT COMPLETE → {enriched_count} fields enriched | Direction determined from breakout.json (breakout={is_breakout_strategy})", "SUCCESS")                               

                def categorize_developer_techniqueq(
                        broker_raw_name,
                        symbol_folder,
                        tf_folder,
                        chart_path,
                        output_path,  # This is the FINAL drawn chart (with all annotations)
                        technique_path,
                        report_path
                    ):

                    # === 1. Load configuration from continuation.json ===
                    strategy_name = None
                    pending_entry_value = None
                    target_zone_name = None
                    target_reached_name = None
                    continuation_name = None
                    continuation_extreme_name = None
                    categories = []
                    is_breakout_strategy = False
                    is_continuation_strategy = False

                    try:
                        with open(technique_path, 'r', encoding='utf-8') as f:
                            technique_conf_2ig = json.load(f)
                        
                        for key, value in technique_conf_2ig.items():
                            k = key.strip().upper()
                            if k == "STRATEGY_NAME" and isinstance(value, str):
                                strategy_name = value.strip()
                            elif k == "PENDING_ENTRY" and isinstance(value, str):
                                pending_entry_value = value.strip()
                            elif k == "TARGET_ZONE_NAME" and isinstance(value, str):
                                target_zone_name = value.strip()
                            elif k == "TARGET_REACHED_NAME" and isinstance(value, str):
                                target_reached_name = value.strip()
                            elif k == "CONTINUATION_NAME" and isinstance(value, str):
                                continuation_name = value.strip()
                            elif k == "CONTINUATION_EXTREME_NAME" and isinstance(value, str):
                                continuation_extreme_name = value.strip()
                            elif k == "CATEGORIES_SPLIT" and isinstance(value, str):
                                categories = [cat.strip().lower().replace(" ", "_").replace("-", "_") for cat in value.split(",") if cat.strip()]
                        
                        trendline_conf_2igs = technique_conf_2ig.get("trendline", {})
                        if isinstance(trendline_conf_2igs, dict):
                            for conf_2ig in trendline_conf_2igs.values():
                                if isinstance(conf_2ig, dict):
                                    direction = conf_2ig.get("DIRECTION", "").strip().lower()
                                    if direction == "breakout":
                                        is_breakout_strategy = True
                                    elif direction == "continuation":
                                        is_continuation_strategy = True
                        
                        if not strategy_name:
                            strategy_name = os.path.basename(os.path.dirname(technique_path))
                        if not continuation_name:
                            continuation_name = f"{strategy_name}_Continuation"  # fallback if not specified
                        if not pending_entry_value:
                            log("No PENDING_ENTRY found → skipping aggregation", "WARNING")
                    except Exception as e:
                        log(f"Failed to load technique JSON: {e}", "ERROR")
                        return

                    # === Early exit if no pending entry name ===
                    if not pending_entry_value:
                        log("No PENDING_ENTRY name → aggregation skipped but chart may still be saved", "WARNING")

                    # === 2. Determine which strategies to process ===
                    if not (is_breakout_strategy or is_continuation_strategy):
                        log(f"Technique is neither breakout nor continuation → skipping categorization for {symbol_folder}/{tf_folder}", "INFO")
                        return

                    base_dev_path = r"C:\xampp\htdocs\chronedge\synarex\chart\developers"
                    broker_folder = os.path.join(base_dev_path, broker_raw_name)

                    # Safe names
                    pending_safe = pending_entry_value.strip().replace(" ", "_") if pending_entry_value else None
                    target_zone_safe = (target_zone_name or "Instant_Orders").strip().replace(" ", "_")
                    target_reached_safe = (target_reached_name or "Trend_Instant_Orders").strip().replace(" ", "_")
                    continuation_extreme_safe = (continuation_extreme_name or "Instant_Continuation").strip().replace(" ", "_")

                    safe_symbol = symbol_folder.replace("/", "_").replace("\\", "_").replace(":", "_")
                    timeframe_filename = f"{tf_folder}.png"

                    expected_tfs = ["5m", "15m", "30m", "1h", "4h"]

                    # === 3. CHART SAVING: Save in respective main folders based on direction ===
                    def save_chart_to_strategy(main_strategy_name):
                        if not os.path.exists(output_path):
                            log(f"Chart not found, skipped copy → {output_path}", "WARNING")
                            return

                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        os.makedirs(strategy_folder, exist_ok=True)

                        chart_base_dir = os.path.join(strategy_folder, "chart")
                        os.makedirs(chart_base_dir, exist_ok=True)

                        market_chart_dir = os.path.join(chart_base_dir, safe_symbol)
                        os.makedirs(market_chart_dir, exist_ok=True)

                        destination_chart_path = os.path.join(market_chart_dir, timeframe_filename)

                        try:
                            shutil.copy2(output_path, destination_chart_path)
                            log(f"Chart saved → {destination_chart_path} ({main_strategy_name})", "INFO")
                        except Exception as e:
                            log(f"Failed to copy chart to {main_strategy_name}: {e}", "ERROR")

                    if is_breakout_strategy:
                        save_chart_to_strategy(strategy_name)
                    if is_continuation_strategy:
                        save_chart_to_strategy(continuation_name)

                    # If no pending → stop aggregation but chart already saved
                    if not pending_entry_value:
                        return

                    # === 4. Load report JSON ===
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            custom_data = json.load(f)
                    except Exception as e:
                        log(f"Failed to read custom JSON: {e}", "ERROR")
                        return

                    # === 5. Extract confirmed pending entries → new structure ===
                    base_confirmed_entries = []
                    for line_id, entry in custom_data.items():
                        info = entry.get("team", {}).get("trendline_info", {})
                        if not info.get("valid", False):
                            continue
                        full_extension = info.get("pending_entry_is_full_extension", False)
                        blocker = info.get("extreme_horizontal_level_blocker")
                        target_zone_found = info.get("target_zone_candle_found", False)
                        if full_extension and blocker is None and target_zone_found:
                            pep = info.get("pending_entry_point")
                            if not pep or not isinstance(pep, dict):
                                continue
                            entry_price = pep.get("entry_price")
                            exit_price = pep.get("exit_price")
                            etype = pep.get("type")
                            candle = pep.get("candle", {})
                            candle_number = candle.get("candle_number") if isinstance(candle, dict) else pep.get("candle_number")
                            time = candle.get("time") if isinstance(candle, dict) else None

                            if entry_price is None:
                                continue

                            order_type = "buy_limit" if etype == "high" else "sell_limit" if etype == "low" else None
                            if order_type is None:
                                continue

                            order_record = {
                                "market_name": symbol_folder,
                                "timeframe": tf_folder,
                                "entry_price": entry_price,
                                "order_type": order_type,
                                "candle_number": candle_number,
                                "candletimestamp": time
                            }
                            if exit_price is not None:
                                order_record["exit_price"] = exit_price

                            base_confirmed_entries.append(order_record)

                    has_pending_entries = len(base_confirmed_entries) > 0

                    # === 6. Extract instant orders → new structure ===
                    instant_target_zone_entries = []
                    instant_target_reached_entries = []
                    instant_continuation_extreme_entries = []
                    no_active_order_entries = []

                    for line_id, entry in custom_data.items():
                        info = entry.get("team", {}).get("trendline_info", {})
                        active_order = info.get("active_instant_order", {})

                        if not isinstance(active_order, dict):
                            continue

                        if active_order.get("reason") == "no active order":
                            no_active_order_entries.append({
                                "market_name": symbol_folder,
                                "timeframe": tf_folder,
                                "reason": "no active order",
                                "order_from": active_order.get("order_from", "none")
                            })
                            continue

                        instant_price = active_order.get("instant_entry")
                        order_type_str = active_order.get("order_type")
                        order_from = active_order.get("order_from")

                        if instant_price is None or order_type_str is None or order_from is None:
                            continue

                        order_type = "buy" if "buy" in order_type_str.lower() else "sell"

                        record = {
                            "market_name": symbol_folder,
                            "timeframe": tf_folder,
                            "entry_price": instant_price,
                            "order_type": order_type,
                            "candle_number": None,  # instant orders usually don't have candle number
                            "candletimestamp": None,
                            "fact": active_order.get("fact", "")
                        }

                        if order_from in {"target_zone", "target_zone_mutuals"}:
                            instant_target_zone_entries.append(record)
                        elif order_from in {"target reached candle", "target reached mutual candle"}:
                            instant_target_reached_entries.append(record)
                        elif order_from in {"extreme interceptor", "extreme_interceptor_mutual"}:
                            instant_continuation_extreme_entries.append(record)

                    has_tz_instant = len(instant_target_zone_entries) > 0
                    has_tr_instant = len(instant_target_reached_entries) > 0
                    has_cont_ext_instant = len(instant_continuation_extreme_entries) > 0
                    has_no_active_order = len(no_active_order_entries) > 0

                    # === 7. Helper: Save aggregation data (updated structure) ===
                    def save_aggregation(main_strategy_name, subfolder_name, entries_list, has_entries, no_active_list=None):
                        if not entries_list and not no_active_list:
                            return

                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        subfolder = os.path.join(strategy_folder, subfolder_name)
                        os.makedirs(subfolder, exist_ok=True)

                        for cat in categories:
                            cat_folder = os.path.join(subfolder, cat)
                            os.makedirs(cat_folder, exist_ok=True)

                            file_name = f"{subfolder_name}.json"
                            no_file_name = f"no_{subfolder_name}.json"

                            full_path = os.path.join(cat_folder, file_name)
                            no_full_path = os.path.join(cat_folder, no_file_name)

                            data_key = f"{subfolder_name}_orders"
                            markets_key = f"{subfolder_name}_markets"
                            total_key = f"total_{subfolder_name}"

                            # Load or init main data
                            aggregate_data = {markets_key: 0, total_key: 0, data_key: {}}
                            if os.path.exists(full_path):
                                try:
                                    with open(full_path, 'r', encoding='utf-8') as f:
                                        aggregate_data = json.load(f)
                                except:
                                    pass

                            aggregate_data.setdefault(data_key, {})
                            if symbol_folder not in aggregate_data[data_key]:
                                aggregate_data[data_key][symbol_folder] = {tf: [] for tf in expected_tfs}

                            if has_entries:
                                aggregate_data[data_key][symbol_folder][tf_folder] = entries_list
                                log(f"Saved {len(entries_list)} orders → {full_path} ({cat})", "SUCCESS")
                            else:
                                aggregate_data[data_key][symbol_folder][tf_folder] = []

                            # Clean up empty symbols
                            if symbol_folder in aggregate_data[data_key] and all(len(e) == 0 for e in aggregate_data[data_key][symbol_folder].values()):
                                del aggregate_data[data_key][symbol_folder]

                            # Update counts
                            total_markets = sum(1 for sym, tfs in aggregate_data[data_key].items() if any(len(e) > 0 for e in tfs.values()))
                            total_entries = sum(len(e) for sym in aggregate_data[data_key].values() for e in sym.values())
                            aggregate_data[markets_key] = total_markets
                            aggregate_data[total_key] = total_entries

                            with open(full_path, 'w', encoding='utf-8') as f:
                                json.dump(aggregate_data, f, indent=2, ensure_ascii=False)

                            # Handle no-active-order tracking
                            if no_active_list:
                                no_data = {"markets_with_no_active_order": [], "count": 0}
                                if os.path.exists(no_full_path):
                                    try:
                                        with open(no_full_path, 'r', encoding='utf-8') as f:
                                            no_data = json.load(f)
                                    except:
                                        pass
                                no_data.setdefault("markets_with_no_active_order", [])

                                symbol_has_orders = symbol_folder in aggregate_data[data_key] and any(len(e) > 0 for e in aggregate_data[data_key][symbol_folder].values())
                                if symbol_has_orders:
                                    if symbol_folder in no_data["markets_with_no_active_order"]:
                                        no_data["markets_with_no_active_order"].remove(symbol_folder)
                                else:
                                    if symbol_folder not in no_data["markets_with_no_active_order"]:
                                        no_data["markets_with_no_active_order"].append(symbol_folder)

                                no_data["count"] = len(no_data["markets_with_no_active_order"])
                                with open(no_full_path, 'w', encoding='utf-8') as f:
                                    json.dump(no_data, f, indent=2, ensure_ascii=False)

                    # === 8. Save Pending Entries ===
                    if pending_safe:
                        save_aggregation(strategy_name, pending_safe, base_confirmed_entries, has_pending_entries)

                    # === 9. Breakout: Target Zone & Target Reached Instant Orders ===
                    if is_breakout_strategy:
                        save_aggregation(strategy_name, target_zone_safe, instant_target_zone_entries, has_tz_instant, no_active_list=no_active_order_entries if has_no_active_order else None)
                        save_aggregation(strategy_name, target_reached_safe, instant_target_reached_entries, has_tr_instant, no_active_list=no_active_order_entries if has_no_active_order else None)

                    # === 10. Continuation: Extreme Interceptor Instant Orders ===
                    if is_continuation_strategy:
                        save_aggregation(continuation_name, continuation_extreme_safe, instant_continuation_extreme_entries, has_cont_ext_instant, no_active_list=no_active_order_entries if has_no_active_order else None)

                    log(f"Categorization complete for {symbol_folder}/{tf_folder} | "
                        f"Pending: {len(base_confirmed_entries)} | "
                        f"TZ: {len(instant_target_zone_entries)} | TR: {len(instant_target_reached_entries)} | "
                        f"Cont Ext: {len(instant_continuation_extreme_entries)} | No Active: {len(no_active_order_entries)}", "SUCCESS")

                def categorize_developer_technique(
                    broker_raw_name,
                    symbol_folder,
                    tf_folder,
                    chart_path,
                    output_path,
                    technique_path,
                    report_path
                ):
                    import os
                    import json
                    import shutil
                    # Assuming log is already defined elsewhere

                    # === 1. Load configuration ===
                    strategy_name = None
                    pending_entry_value = None
                    target_zone_name = None
                    target_reached_name = None
                    categories = []
                    is_breakout_strategy = False

                    try:
                        with open(technique_path, 'r', encoding='utf-8') as f:
                            technique_conf = json.load(f)
                        
                        for key, value in technique_conf.items():
                            k = key.strip().upper()
                            if k == "STRATEGY_NAME" and isinstance(value, str):
                                strategy_name = value.strip()
                            elif k == "PENDING_ENTRY" and isinstance(value, str):
                                pending_entry_value = value.strip()
                            elif k == "TARGET_ZONE_NAME" and isinstance(value, str):
                                target_zone_name = value.strip()
                            elif k == "TARGET_REACHED_NAME" and isinstance(value, str):
                                target_reached_name = value.strip()
                            elif k == "CATEGORIES_SPLIT" and isinstance(value, str):
                                categories = [cat.strip().lower().replace(" ", "_").replace("-", "_") 
                                            for cat in value.split(",") if cat.strip()]

                        # Check trendline directions
                        trendline_conf = technique_conf.get("trendline", {})
                        if isinstance(trendline_conf, dict):
                            for conf in trendline_conf.values():
                                if isinstance(conf, dict):
                                    direction = conf.get("DIRECTION", "").strip().lower()
                                    if direction == "breakout":
                                        is_breakout_strategy = True

                        # Fallback
                        if not strategy_name:
                            strategy_name = os.path.basename(os.path.dirname(technique_path))

                    except Exception as e:
                        log(f"Failed to load technique JSON: {e}", "ERROR")
                        return

                    # Skip if not breakout
                    if not is_breakout_strategy:
                        log(f"Technique is not a breakout strategy → skipping categorization", "INFO")
                        return

                    base_dev_path = r"C:\xampp\htdocs\chronedge\synarex\chart\developers"
                    broker_folder = os.path.join(base_dev_path, broker_raw_name)

                    # Safe names for folders
                    pending_safe = pending_entry_value.strip().replace(" ", "_") if pending_entry_value else None
                    target_zone_safe = (target_zone_name or "Instant_Orders").strip().replace(" ", "_")
                    target_reached_safe = (target_reached_name or "Trend_Instant_Orders").strip().replace(" ", "_")

                    safe_symbol = symbol_folder.replace("/", "_").replace("\\", "_").replace(":", "_")
                    timeframe_filename = f"{tf_folder}.png"

                    # === Chart saving ===
                    def save_chart_to_strategy(main_strategy_name):
                        if not os.path.exists(output_path):
                            log(f"Chart not found, skipped copy → {output_path}", "WARNING")
                            return
                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        os.makedirs(strategy_folder, exist_ok=True)
                        chart_base_dir = os.path.join(strategy_folder, "chart")
                        os.makedirs(chart_base_dir, exist_ok=True)
                        market_chart_dir = os.path.join(chart_base_dir, safe_symbol)
                        os.makedirs(market_chart_dir, exist_ok=True)
                        destination_chart_path = os.path.join(market_chart_dir, timeframe_filename)
                        try:
                            shutil.copy2(output_path, destination_chart_path)
                            log(f"Chart saved → {destination_chart_path} ({main_strategy_name})", "INFO")
                        except Exception as e:
                            log(f"Failed to copy chart to {main_strategy_name}: {e}", "ERROR")

                    save_chart_to_strategy(strategy_name)

                    # === Load report JSON ===
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            custom_data = json.load(f)
                    except Exception as e:
                        log(f"Failed to read custom JSON: {e}", "ERROR")
                        return

                    # === Extract real orders ===
                    base_confirmed_entries = []             # Pending (Limit Orders)
                    instant_target_zone_entries = []        # Instant from target zone
                    instant_target_reached_entries = []     # Instant from target reached

                    for line_id, entry in custom_data.items():
                        info = entry.get("team", {}).get("trendline_info", {})
                        if not info.get("valid", False):
                            continue

                        # Pending Logic
                        if (info.get("pending_entry_is_full_extension") and 
                            info.get("extreme_horizontal_level_blocker") is None and 
                            info.get("target_zone_candle_found")):
                            
                            pep = info.get("pending_entry_point")
                            if pep and isinstance(pep, dict):
                                base_confirmed_entries.append({
                                    "market_name": symbol_folder,
                                    "timeframe": tf_folder,
                                    "entry_price": pep.get("entry_price"),
                                    "exit_price": pep.get("exit_price"),
                                    "order_type": "buy_limit" if pep.get("type") == "high" else "sell_limit" if pep.get("type") == "low" else None,
                                    "candle_number": pep.get("candle", {}).get("candle_number") if isinstance(pep.get("candle"), dict) else pep.get("candle_number"),
                                    "candletimestamp": pep.get("candle", {}).get("time") if isinstance(pep.get("candle"), dict) else None
                                })

                        # Instant Logic
                        active_order = info.get("active_instant_order", {})
                        if isinstance(active_order, dict) and active_order.get("reason") != "no active order":
                            order_from = active_order.get("order_from")
                            record = {
                                "market_name": symbol_folder,
                                "timeframe": tf_folder,
                                "entry_price": active_order.get("instant_entry"),
                                "exit_price": active_order.get("exit_price"),
                                "order_type": "buy" if "buy" in str(active_order.get("order_type")).lower() else "sell",
                                "candle_number": None,
                                "candletimestamp": None,
                                "fact": active_order.get("fact", "")
                            }
                            if order_from in {"target_zone", "target_zone_mutuals"}:
                                instant_target_zone_entries.append(record)
                            elif order_from in {"target reached candle", "target reached mutual candle"}:
                                instant_target_reached_entries.append(record)

                    # === Save aggregation helper ===
                    def save_aggregation(main_strategy_name, subfolder_name, new_entries, allow_coords=True):
                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        subfolder = os.path.join(strategy_folder, subfolder_name)
                        os.makedirs(subfolder, exist_ok=True)

                        for cat in categories:
                            if cat == "co_ords" and not allow_coords:
                                continue

                            cat_folder = os.path.join(subfolder, cat)
                            os.makedirs(cat_folder, exist_ok=True)
                            full_path = os.path.join(cat_folder, f"{subfolder_name}.json")

                            try:
                                if os.path.exists(full_path):
                                    with open(full_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        orders = data.get("orders", [])
                                else:
                                    orders = []
                            except:
                                orders = []

                            orders = [o for o in orders if not (o.get("market_name") == symbol_folder and o.get("timeframe") == tf_folder)]
                            orders = [o for o in orders if o.get("market_name") != "none"]

                            processed_new_entries = []
                            for entry in new_entries:
                                entry_copy = entry.copy()
                                if cat != "co_ords":
                                    entry_copy.pop("exit_price", None)
                                processed_new_entries.append(entry_copy)

                            orders.extend(processed_new_entries)

                            real_orders = [o for o in orders if o.get("market_name") not in {"none", None}]
                            result = {
                                "total_orders": len(real_orders),
                                "total_markets": len({o["market_name"] for o in real_orders}),
                                "orders": orders
                            }

                            with open(full_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)

                    # === Execution (breakout only) ===
                    if pending_safe:
                        save_aggregation(strategy_name, pending_safe, base_confirmed_entries, allow_coords=True)
                    save_aggregation(strategy_name, target_zone_safe, instant_target_zone_entries, allow_coords=False)
                    save_aggregation(strategy_name, target_reached_safe, instant_target_reached_entries, allow_coords=False)

                    # === Final Placeholder Logic ===
                    total_real = len(base_confirmed_entries) + len(instant_target_zone_entries) + len(instant_target_reached_entries)

                    placeholder = [{
                        "market_name": "none", "timeframe": "none", "entry_price": None,
                        "order_type": "none", "candle_number": None, "candletimestamp": None, "exit_price": None
                    }]

                    if total_real == 0:
                        if pending_safe:
                            save_aggregation(strategy_name, pending_safe, placeholder, allow_coords=True)
                        save_aggregation(strategy_name, target_zone_safe, placeholder, allow_coords=False)
                        save_aggregation(strategy_name, target_reached_safe, placeholder, allow_coords=False)

                    log(f"Breakout categorization complete for {symbol_folder}/{tf_folder}", "SUCCESS")                          

                def trend_direction():
                    # 1. Initialize the central data store
                    final_valid_trends = {}
                    for trend in final_trendlines_for_redraw:
                        direction = trend["direction"]
                        if direction == "breakout":
                            final_valid_trends = OPPOSITION_BREAKOUT_INTERCEPTORS_RETEST_TARGET_ZONE(final_valid_trends)
                        elif direction == "continuation":
                            final_valid_trends = CONTINUATION_EXTREME_INTERCEPTOR(final_valid_trends)
                        else:
                            log(f"Unknown direction '{direction}' for trend {trend['line_id']} - skipping processing", "WARNING")

                    # 3. Draw horizontal levels (uses final_valid_trends)
                    DRAW_ALL_POINTS_LEVELS(final_valid_trends)

                    # ==== ENRICH CUSTOM JSON WITH FULL CANDLE DETAILS ====
                    enrich_candle_details(candles, final_teams, final_valid_trends, report_path, technique_path)

                    # Final save after processing all trends (local)
                    cv2.imwrite(output_path, img)
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(final_teams, f, indent=2, ensure_ascii=False, default=str)
                    log(f"{symbol_folder}/{tf_folder} | {len(final_teams)} Trendlines processed & enriched with full candle details", "SUCCESS")

                    # ===== NEW: SAVE TO CENTRAL DEVELOPER STRATEGY FOLDER =====
                    # We need broker_raw_name and the technique_path from outer scope
                    # These variables are available in the main loop context
                    categorize_developer_technique(
                        broker_raw_name=broker_raw_name,
                        symbol_folder=symbol_folder,
                        tf_folder=tf_folder,
                        chart_path=chart_path,
                        output_path=output_path,
                        technique_path=technique_path,
                        report_path=report_path
                    )

                trend_direction()
                

    log("=== INSTITUTIONAL TRENDLINE ENGINE v10.1 — DYNAMIC CONFIGURATION & TARGET ZONE ADDED ===", "SUCCESS")

def custom_continuation():
    # --- INITIAL SETUP ---
    lagos_tz = pytz.timezone('Africa/Lagos')
    def log(msg, level="INFO"):
        ts = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] [{level}] {msg}")

    # ==================================================================
    # LEVEL FAMILY CLASSIFICATION (UNCHANGED CORE LOGIC)
    # ==================================================================
    BEARISH_FAMILY = {"ph", "ch"}
    BULLISH_FAMILY = {"pl", "cl"}
    
    # NEW: Dynamic family mapping based on input key
    def get_level_family(key):
        if key in BEARISH_FAMILY: return "bearish"
        if key in BULLISH_FAMILY: return "bullish"
        return None

    # This map is used to find opposite levels (e.g., ph -> {pl, cl})
    OPPOSITE_MAP = {
        "ph": {"pl", "cl"}, "ch": {"pl", "cl"},
        "pl": {"ph", "ch"}, "cl": {"ph", "ch"}
    }
    
    def is_bearish_level(key): return key in BEARISH_FAMILY
    def is_bullish_level(key): return key in BULLISH_FAMILY
    def is_level_match(candle, key): return candle.get(f"is_{key}", False)
    
    PARENT_LEVELS = {"ph", "pl"}
    CHILD_LEVELS = {"ch", "cl"}
    def is_parent_level(key): return key in PARENT_LEVELS
    def is_child_level(key): return key in CHILD_LEVELS
    
    COLOR_MAP = {
        "ph": (255, 100, 0), # Orange
        "pl": (200, 0, 200), # Purple
        "ch": (255, 200, 0), # Yellow
        "cl": (0, 140, 255), # Blue
        "fvg_middle": (0, 255, 0),
    }
    def get_color(key):
        return COLOR_MAP.get(key, (180, 180, 180))
        
    def get_y_position(positions, candle_num, key):
        pos = positions[candle_num]
        return pos["high_y"] if key in ["ph", "ch", "fvg_middle"] else pos["low_y"]

    # ------------------------------------------------------------------
    # MARKERS (unchanged)
    # ------------------------------------------------------------------
    def mark_breakout_extreme_interceptor(img, x, body_y, color):
        cv2.circle(img, (x, body_y), 18, color, 4)
        cv2.circle(img, (x, body_y), 14, (0, 255, 255), 2)
        arrow_start_x = x - 70
        arrow_end_x = x - 20
        cv2.arrowedLine(img, (arrow_start_x, body_y), (arrow_end_x, body_y), color, thickness=4, tipLength=0.3)

    def mark_breakout_candle(img, x, body_y, color):
        label_x = x - 38
        label_y = body_y + 6
        cv2.putText(img, "B", (label_x + 1, label_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, "B", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "B", (label_x - 1, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def mark_continuation_extreme_interceptor(img, x, body_y, color):
        cv2.circle(img, (x, body_y), 18, color, 4)
        cv2.circle(img, (x, body_y), 14, (0, 255, 255), 2)
        arrow_start_x = x - 70
        arrow_end_x = x - 20
        cv2.arrowedLine(img, (arrow_start_x, body_y), (arrow_end_x, body_y), color, thickness=4, tipLength=0.3)

    def mark_continuation_candle(img, x, body_y, color):
        label_x = x - 38
        label_y = body_y + 6
        cv2.putText(img, "B", (label_x + 1, label_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, "B", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(img, "B", (label_x - 1, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def draw_opposition_arrow(img, x, y, color, direction_up=True):
        size = 20
        thickness = 3
        shaft_length = 40
        if direction_up:
            cv2.line(img, (x, y + size), (x, y + size + shaft_length), color, thickness)
            pts = np.array([[x, y + size], [x - 12, y + size + 12], [x + 12, y + size + 12]], np.int32)
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.line(img, (x, y - size), (x, y - size - shaft_length), color, thickness)
            pts = np.array([[x, y - size], [x - 12, y - size - 12], [x + 12, y - size - 12]], np.int32)
            cv2.fillPoly(img, [pts], color)

    def draw_double_retest_arrow(img, x, y_price, color, direction_up=True):
        offset = 12
        draw_opposition_arrow(img, x - offset, y_price, color, direction_up=direction_up)
        draw_opposition_arrow(img, x + offset, y_price, color, direction_up=direction_up)

    def draw_target_zone_marker(img, x, y_price, color, size=10):
        cv2.rectangle(img, (x - size, y_price - size), (x + size, y_price + size), color, -1)

    # ------------------------------------------------------------------
    # Candle positions from chart.png (unchanged)
    # ------------------------------------------------------------------
    def get_candle_positions(chart_path):
        img = cv2.imread(chart_path)
        if img is None: return None, {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        mask |= cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        bounds = {}
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            bounds[idx] = {"x": center_x, "high_y": y, "low_y": y + h, "width": w}
        return img.copy(), bounds

    # ------------------------------------------------------------------
    # Line–rectangle intersection (unchanged)
    # ------------------------------------------------------------------
    def line_intersects_rect(x1, y1, x2, y2, rect_left, rect_top, rect_right, rect_bottom):
        expand = 6
        rect_left -= expand; rect_right += expand; rect_top -= expand; rect_bottom += expand
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10: return 0
            return 1 if val > 0 else 2
        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2); o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1); o4 = orientation(p2, q2, q1)
            if o1 != o2 and o3 != o4: return True
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True
            return False
        edges = [((rect_left, rect_top), (rect_right, rect_top)),
                 ((rect_right, rect_top), (rect_right, rect_bottom)),
                 ((rect_right, rect_bottom), (rect_left, rect_bottom)),
                 ((rect_left, rect_bottom), (rect_left, rect_top))]
        for p2, q2 in edges:
            if do_intersect((x1, y1), (x2, y2), p2, q2):
                return True
        return False
        
    # ------------------------------------------------------------------
    # MAIN LOOP (with dynamic family and key loading)
    # ------------------------------------------------------------------
    developer_brokers = {k: v for k, v in globals().get("session", {}).items() if v.get("POSITION", "").lower() == "developer"}
    
    global_tech_data = {}
    
    for broker_raw_name, cfg in developer_brokers.items():
        base_folder = cfg["BASE_FOLDER"]
        technique_path_2 = os.path.join(base_folder, "..", "developers", broker_raw_name, "continuation.json")
        if not os.path.exists(technique_path_2):
            technique_path_2 = os.path.join(base_folder, "continuation.json")
        if not os.path.exists(technique_path_2):
            log(f"continuation.json missing → {broker_raw_name}", "WARNING")
            continue
        
        with open(technique_path_2, 'r', encoding='utf-8') as f:
            tech = json.load(f)
            global_tech_data = tech # Store for later access
            
        if str(tech.get("drawings_switch", {}).get("trendline", "no")).strip().lower() != "yes":
            continue
        
        trend_conf_2igs = tech.get("trendline", {})
        trend_list = []
        for key in sorted([k for k in trend_conf_2igs.keys() if str(k).isdigit()]):
            conf_2 = trend_conf_2igs[key]
            if not isinstance(conf_2, dict): continue
            
            # --- DYNAMIC conf_2IGURATION LOADING ---
            fr = conf_2.get("FROM", "").strip().lower()
            direction = conf_2.get("DIRECTION", "").strip().lower()
            trend_family = conf_2.get("TREND", "").strip().lower() # NEW: Load the trend family
            
            rules = conf_2.get("rules", {})
            breakout_cond = rules.get("breakout_condition", "").strip().lower()
            
            # Load the dynamic point keys
            define_points = rules.get("define_trend_points", {})
            sender_key_name = define_points.get("sender_candle", "").strip().lower()
            receiver_key_name = define_points.get("receiver_candle", "").strip().lower()
            opposition_key_name = define_points.get("opposition_candle", "").strip().lower()
            retest_key_name = define_points.get("retest_candle", "").strip().lower()
            
            seq_count = 0
            if breakout_cond and "_sequence_candle" in breakout_cond:
                match = re.search(r"(\d+)_sequence_candle", breakout_cond)
                if match:
                    seq_count = int(match.group(1))

            if fr:
                trend_list.append({
                    "id": key,
                    "FROM": fr,
                    "TO": conf_2.get("TO", "ray").strip().lower(),
                    "rule": rules.get("extreme_intruder", "continue").strip().lower(),
                    "sender_condition": rules.get("sender_condition", "none").strip().lower(),
                    "interceptor_enabled": str(conf_2.get("INTERCEPTOR", "no")).strip().lower() == "yes",
                    "direction": direction,
                    "breakout_sequence_count": seq_count if direction == "breakout" and seq_count > 0 else 0,
                    "continuation_sequence_count": seq_count if direction == "continuation" and seq_count > 0 else 0,
                    # NEW DYNAMIC KEYS
                    "trend_family": trend_family, 
                    "point_keys": {
                        "sender": sender_key_name,
                        "receiver": receiver_key_name,
                        "opposition": opposition_key_name,
                        "retest": retest_key_name,
                    },
                    "horizontal_line_subject": rules.get("HORIZONTAL_LINE_SUBJECT", {}) 
                })

        if not trend_list:
            continue
            
        # Get Neighbor Right settings
        parent_neighbor_right = global_tech_data.get("parenthighsandlows", {}).get("NEIGHBOR_RIGHT", 15)
        child_neighbor_right = global_tech_data.get("childhighsandlows", {}).get("NEIGHBOR_RIGHT", 7)
        
        log(f"Processing {broker_raw_name} → {len(trend_list)} institutional trendlines (Parent NR={parent_neighbor_right}, Child NR={child_neighbor_right})")
        
        for symbol_folder in os.listdir(base_folder):
            sym_path = os.path.join(base_folder, symbol_folder)
            if not os.path.isdir(sym_path): continue
            for tf_folder in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf_folder)
                if not os.path.isdir(tf_path): continue
                chart_path = os.path.join(tf_path, "chart.png")
                json_path = os.path.join(tf_path, "all_oldest_newest_candles.json")
                output_path = os.path.join(tf_path, "chart_custom.png")
                report_path = os.path.join(tf_path, "custom_levels.json")
                if not all(os.path.exists(p) for p in [chart_path, json_path]):
                    continue
                with open(json_path, 'r', encoding='utf-8') as f:
                    candles = json.load(f)
                img, raw_positions = get_candle_positions(chart_path)
                if img is None: continue
                positions = {}
                for idx, data in sorted(raw_positions.items(), key=lambda x: x[1]["x"], reverse=True):
                    if idx < len(candles):
                        cnum = candles[idx]["candle_number"]
                        positions[cnum] = data
                
                # Draw level markers (unchanged)
                for candle in reversed(candles):
                    cnum = candle["candle_number"]
                    if cnum not in positions: continue
                    x = positions[cnum]["x"]
                    hy, ly = positions[cnum]["high_y"], positions[cnum]["low_y"]
                    if candle.get("is_ph"):
                        pts = np.array([[x, hy-10], [x-10, hy+5], [x+10, hy+5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ph"])
                    if candle.get("is_pl"):
                        pts = np.array([[x, ly+10], [x-10, ly-5], [x+10, ly-5]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["pl"])
                    if candle.get("is_ch"):
                        pts = np.array([[x, hy-8], [x-7, hy+4], [x+7, hy+4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["ch"])
                    if candle.get("is_cl"):
                        pts = np.array([[x, ly+8], [x-7, ly-4], [x+7, ly-4]])
                        cv2.fillPoly(img, [pts], COLOR_MAP["cl"])
                    if candle.get("is_fvg_middle"):
                        cv2.circle(img, (x, (hy + ly) // 2), 6, COLOR_MAP["fvg_middle"], -1)
                
                final_teams_2 = {}
                final_trendlines_for_redraw = []
                
                def draw_trendline(line_id, fx, fy, tx, ty, color, extreme_cnum=None, extreme_y=None):
                    cv2.line(img, (fx, fy), (tx, ty), color, 3)
                    label_x = tx + 15 if tx > fx else fx + 15
                    label_y = ty - 20 if fy < ty else ty + 25
                    cv2.putText(img, line_id, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    if extreme_cnum and extreme_y is not None:
                        ex_x = positions[extreme_cnum]["x"]
                        pts = np.array([[ex_x, extreme_y - 15], [ex_x - 10, extreme_y], [ex_x + 10, extreme_y]], np.int32)
                        cv2.fillPoly(img, [pts], color)
                        
                def validate_sender_condition(sender_cnum, receiver_cnum, key, condition):
                    if condition == "none": return True
                    if sender_cnum not in positions or receiver_cnum not in positions: return False
                    sender_candle = next(c for c in candles if c["candle_number"] == sender_cnum)
                    receiver_candle = next(c for c in candles if c["candle_number"] == receiver_cnum)
                    if is_bearish_level(key):
                        return sender_candle["high"] >= receiver_candle["high"] if condition == "beyond" else sender_candle["high"] <= receiver_candle["high"]
                    elif is_bullish_level(key):
                        return sender_candle["low"] <= receiver_candle["low"] if condition == "beyond" else sender_candle["low"] >= receiver_candle["low"]
                    return True
                    
                def process_trendline(conf_2, depth_2=0, max_depth_2=50):
                    if depth_2 > max_depth_2:
                        log(f"Max recursion depth_2 for T{conf_2['id']}", "WARNING")
                        return False
                    
                    line_id = f"T{conf_2['id']}"
                    from_key = conf_2["FROM"]
                    to_key = conf_2["TO"]
                    rule = conf_2["rule"]
                    sender_condition = conf_2["sender_condition"]
                    interceptor_enabled = conf_2["interceptor_enabled"]
                    direction = conf_2["direction"]
                    breakout_seq_count = conf_2["breakout_sequence_count"]
                    color = get_color(from_key)
                    
                    # --- CORE: FINDING FROM/SENDER POINT ---
                    from_candle = next((c for c in reversed(candles) if is_level_match(c, from_key)), None)
                    if not from_candle or from_candle["candle_number"] not in positions:
                        return False
                        
                    from_cnum = from_candle["candle_number"]
                    fx = positions[from_cnum]["x"]
                    fy = get_y_position(positions, from_cnum, from_key)
                    
                    # --- CORE: FINDING TO/RECEIVER POINT ---
                    to_cnum = None
                    tx, ty = img.shape[1] - 30, fy # Default ray endpoint
                    
                    found_from = False
                    for c in reversed(candles):
                        if c["candle_number"] == from_cnum:
                            found_from = True
                            continue
                        if found_from and (to_key == "ray" or is_level_match(c, to_key)):
                            to_cnum = c["candle_number"]
                            if to_cnum in positions:
                                tx = positions[to_cnum]["x"]
                                ty = get_y_position(positions, to_cnum, to_key)
                            break
                            
                    is_ray = (to_cnum is None)

                    # --- INTERMEDIATE TOUCHES FOR EXTREME INTRUDER RULE ---
                    touches = []
                    min_c = min(from_cnum, to_cnum or from_cnum + 99999)
                    max_c = max(from_cnum, to_cnum or from_cnum + 99999)
                    
                    for c in candles:
                        cn = c["candle_number"]
                        if cn in [from_cnum, to_cnum] or cn not in positions: continue
                        if not (min_c <= cn <= max_c): continue
                        
                        pos = positions[cn]
                        if line_intersects_rect(fx, fy, tx, ty,
                                               pos["x"] - pos["width"]//2, pos["high_y"],
                                               pos["x"] + pos["width"]//2, pos["low_y"]):
                            touches.append(cn)
                            
                    extreme_cnum = extreme_y = None
                    if touches:
                        s_min, s_max = min(touches), max(touches)
                        if is_bearish_level(from_key):
                            best = max((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["high"], default=None)
                        else:
                            best = min((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["low"], default=None)
                            
                        if best:
                            extreme_cnum = best["candle_number"]
                            extreme_y = positions[extreme_cnum]["high_y"] if is_bearish_level(from_key) else positions[extreme_cnum]["low_y"]
                            
                    # --- APPLYING EXTREME INTRUDER RULE (new_from/new_to) ---
                    final_fx, final_fy = fx, fy
                    final_tx, final_ty = tx, ty
                    final_from_cnum = from_cnum
                    final_to_cnum = to_cnum
                    applied_rule = "continue"
                    
                    if rule == "new_from" and extreme_cnum:
                        final_fx = positions[extreme_cnum]["x"]
                        final_fy = extreme_y
                        final_from_cnum = extreme_cnum
                        applied_rule = "new_from"
                    elif rule == "new_to" and extreme_cnum:
                        final_tx = positions[extreme_cnum]["x"]
                        final_ty = extreme_y
                        final_to_cnum = extreme_cnum
                        applied_rule = "new_to"
                        
                    # --- VALIDATE SENDER CONDITION ---
                    sender_cnum = final_from_cnum
                    receiver_cnum = final_to_cnum if not is_ray else final_from_cnum # For condition, if ray, receiver is sender
                    if not validate_sender_condition(sender_cnum, receiver_cnum, from_key, sender_condition):
                        return False
                        
                    # --- DRAW INITIAL LINE AND STORE FOR FINAL PROCESSING ---
                    draw_trendline(line_id, int(final_fx), int(final_fy), int(final_tx), int(final_ty), color,
                                    extreme_cnum, extreme_y if rule in ["new_from", "new_to"] else None)
                                    
                    final_trendlines_for_redraw.append({
                        "line_id": line_id,
                        "from_final_valid_sender_x": int(final_fx),
                        "from_final_valid_sender_y": int(final_fy),
                        "from_final_valid_receiver_x": int(final_tx),
                        "from_final_valid_receiver_y": int(final_ty),
                        "receiver_cnum": final_to_cnum if final_to_cnum else final_from_cnum, # Use the actual 'TO' or 'FROM' if ray
                        "from_key": from_key,
                        "color": color,
                        "interceptor_enabled": interceptor_enabled,
                        "direction": direction,
                        "breakout_sequence_count": breakout_seq_count,
                        "trend_family": conf_2["trend_family"], # NEW: Store trend family
                        "point_keys": conf_2["point_keys"],     # NEW: Store dynamic point keys
                    })
                    
                    final_teams_2[line_id] = {"team": {
                        "trendline_info": {
                            "line_id": line_id,
                            "from_candle": final_from_cnum,
                            "to_candle": final_to_cnum,
                            "receiver_candle": final_to_cnum if final_to_cnum else final_from_cnum,
                            "is_ray": is_ray,
                            "intermediate_touches": len(touches),
                            "touched_candles": touches,
                            "extreme_intruder_candle": extreme_cnum,
                            "rule_applied": applied_rule,
                            "color": list(map(int, color)),
                            "interceptors": [],
                            "opposition_candle": None,
                            "breakout_extreme_interceptor_candle": None,
                            "breakout_sequence_candles": [],
                            "retest_candle": None,
                            "target_zone_candle": None 
                        }
                    }}
                    return True
                    
                for conf_2 in trend_list:
                    process_trendline(conf_2)
                    
                # ==================================================================+
                # FINAL PROCESSING — INTERCEPTORS, OPPOSITION, RETEST, & TARGET ZONE|
                # ==================================================================+
                def CONTINUATION_EXTREME_INTERCEPTOR(final_valid_trends_2):
                    for trend in final_trendlines_for_redraw:
                        fx, fy = trend["from_final_valid_sender_x"], trend["from_final_valid_sender_y"]
                        tx, ty = trend["from_final_valid_receiver_x"], trend["from_final_valid_receiver_y"]
                        color = trend["color"]
                        line_id = trend["line_id"]
                        receiver_cnum = trend["receiver_cnum"]
                        sender_cnum = trend.get("sender_cnum") # Ensure you have sender_cnum in your trend dict
                        from_key = trend["from_key"]

                        if tx - fx == 0:
                            continue

                        # --- NEW: Identify Point Types (PH, CH, PL, CL) ---
                        # Get 'FROM' type from from_key (e.g., "bullish_ph" -> "ph")
                        from_point_type = from_key.split('_')[-1] if from_key else "unknown"
                        
                        # Get 'TO' type by checking the receiver candle's label in positions/candles
                        to_point_type = "unknown"
                        if receiver_cnum in positions:
                            # We look for the label assigned to this specific candle in our positions data
                            to_point_type = positions[receiver_cnum].get("label", "unknown").lower()

                        # 1. Redraw Ray
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))
                        cv2.line(img, (fx, fy), (pending_entry, extend_y), color, 3)
                        cv2.putText(img, line_id, (fx + 20, fy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                        # Mark Receiver
                        if receiver_cnum and receiver_cnum in positions:
                            rx = positions[receiver_cnum]["x"]
                            ry = get_y_position(positions, receiver_cnum, from_key)
                            cv2.circle(img, (rx, ry), 12, (0, 0, 0), -1)
                            cv2.putText(img, "R", (rx - 8, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # 2. Find continuation_interceptors + prepare enriched list
                        continuation_interceptors = []
                        touched_continuation_interceptors_enriched = []
                        receiver_x = positions.get(receiver_cnum, {}).get("x", -99999)

                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions:
                                continue
                            pos = positions[cn]
                            if pos["x"] <= receiver_x:
                                continue

                            intersects = line_intersects_rect(fx, fy, pending_entry, extend_y,
                                                            pos["x"] - pos["width"]//2, pos["high_y"],
                                                            pos["x"] + pos["width"]//2, pos["low_y"])

                            body_y = (pos["high_y"] + pos["low_y"]) // 2
                            candle_color = "green" if c["close"] > c["open"] else "red" if c["close"] < c["open"] else "doji"

                            touched_continuation_interceptors_enriched.append({
                                "candle_number": cn,
                                "color": candle_color,
                                "is_extreme": False,
                                "is_mutual": False
                            })

                            if intersects:
                                continuation_interceptors.append({
                                    "candle_number": cn,
                                    "x": pos["x"],
                                    "y": body_y,
                                    "high": c["high"],
                                    "low": c["low"],
                                    "color": candle_color
                                })

                        # 3. Find Extreme Interceptor (oldest opposing)
                        extreme_interceptor_cnum = None
                        extreme_interceptor_data = None
                        is_bullish_trend = is_bullish_level(from_key)

                        if continuation_interceptors:
                            continuation_interceptors.sort(key=lambda x: x["candle_number"])

                            if is_bullish_trend:
                                for intr in continuation_interceptors:
                                    if intr["color"] == "red":
                                        extreme_interceptor_cnum = intr["candle_number"]
                                        extreme_interceptor_data = intr
                            else:
                                for intr in continuation_interceptors:
                                    if intr["color"] == "green":
                                        extreme_interceptor_cnum = intr["candle_number"]
                                        extreme_interceptor_data = intr

                        # 4. Check DUAL RESPECT RULE
                        continuation_extreme_interceptor_mutual = None
                        if extreme_interceptor_data and extreme_interceptor_cnum:
                            next_cnum = extreme_interceptor_cnum - 1
                            if next_cnum > 0 and next_cnum in positions:
                                pos = positions[next_cnum]
                                candle = next((c for c in candles if c["candle_number"] == next_cnum), None)
                                if candle:
                                    body_y = (pos["high_y"] + pos["low_y"]) // 2
                                    ex_high = extreme_interceptor_data["high"]
                                    ex_low  = extreme_interceptor_data["low"]

                                    if candle["high"] <= ex_high and candle["low"] >= ex_low:
                                        continuation_extreme_interceptor_mutual = {
                                            "candle_number": next_cnum,
                                            "x": pos["x"],
                                            "y": body_y,
                                            "high": candle["high"],
                                            "low": candle["low"],
                                            "color": "green" if candle["close"] > candle["open"] else "red"
                                        }

                        # 5. Drawing
                        if extreme_interceptor_data:
                            ex = extreme_interceptor_data
                            mark_continuation_extreme_interceptor(img, ex["x"], ex["y"], color)
                            if continuation_extreme_interceptor_mutual:
                                mx, my = continuation_extreme_interceptor_mutual["x"], continuation_extreme_interceptor_mutual["y"]
                                cv2.circle(img, (mx, my), 7, color, 2)
                                cv2.circle(img, (mx, my), 6, (*color[:3], 120), -1)
                                cv2.putText(img, "M", (mx-6, my+6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                        # 5.5 Handle Limit (3rd position)
                        limit_interceptor_cnum = None
                        if extreme_interceptor_cnum:
                            limit_interceptor_cnum = extreme_interceptor_cnum - 2
                            if limit_interceptor_cnum > 0 and limit_interceptor_cnum in positions:
                                lpos = positions[limit_interceptor_cnum]
                                lx, ly = lpos["x"], (lpos["high_y"] + lpos["low_y"]) // 2
                                cv2.rectangle(img, (lx-8, ly-8), (lx+8, ly+8), color, 2)
                                cv2.putText(img, "L", (lx-5, ly+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # 6. Update JSON flags and Point Types
                        if extreme_interceptor_cnum:
                            for item in touched_continuation_interceptors_enriched:
                                if item["candle_number"] == extreme_interceptor_cnum:
                                    item["is_extreme"] = True
                                if continuation_extreme_interceptor_mutual and item["candle_number"] == continuation_extreme_interceptor_mutual["candle_number"]:
                                    item["is_mutual"] = True

                        # 7. Final Save to custom level JSON structure
                        if line_id in final_teams_2:
                            final_teams_2[line_id]["team"]["trendline_info"].update({
                                "FROM": from_point_type,  # ph, ch, pl, cl
                                "TO": to_point_type,      # ph, ch, pl, cl
                                "continuation_interceptors": continuation_interceptors,
                                "touched_continuation_interceptors": touched_continuation_interceptors_enriched,
                                "continuation_extreme_interceptor_candle": extreme_interceptor_cnum,
                                "Continuation_extreme_Interceptor_limit": limit_interceptor_cnum,
                                "continuation_extreme_interceptor_mutual": (
                                    {
                                        "candle_number": continuation_extreme_interceptor_mutual["candle_number"],
                                        "low": continuation_extreme_interceptor_mutual["low"],
                                        "high": continuation_extreme_interceptor_mutual["high"]
                                    } if continuation_extreme_interceptor_mutual else None
                                )
                            })

                            # Clean old keys
                            for key in ["opposition_candle", "retest_candle", "target_zone_candle",
                                        "continuation_sequence_candles", "breakout_extreme_interceptor_candle",
                                        "continuation_extreme_interceptor_mutuals"]:
                                final_teams_2[line_id]["team"]["trendline_info"][key] = None

                            final_valid_trends_2[line_id] = trend.copy()
                            final_valid_trends_2[line_id]['valid'] = True

                    log(f"→ {symbol_folder}/{tf_folder} | {len(final_valid_trends_2)} Trends Validated (Points: {from_point_type}->{to_point_type})", "SUCCESS")
                    return final_valid_trends_2                                     

                def OPPOSITION_BREAKOUT_INTERCEPTORS_RETEST_TARGET_ZONE(final_valid_trends_2):

                    def draw_final_trendline(trend, fx, fy, tx, ty, color):
                        if tx - fx == 0:
                            return
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))
                        cv2.line(img, (fx, fy), (pending_entry, extend_y), color, 3)
                        cv2.putText(img, trend["line_id"], (fx + 20, fy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                        # Mark receiver
                        receiver_cnum = trend["receiver_cnum"]
                        if receiver_cnum in positions:
                            rx = positions[receiver_cnum]["x"]
                            ry = get_y_position(positions, receiver_cnum, trend["from_key"])
                            cv2.circle(img, (rx, ry), 12, (0, 0, 0), -1)
                            cv2.putText(img, "R", (rx - 8, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    def get_candle_level(cnum):
                        """Helper to return the level string (ph/ch/pl/cl) for a given candle number, or None."""
                        if cnum not in [c["candle_number"] for c in candles]:
                            return None
                        candle = next(c for c in candles if c["candle_number"] == cnum)
                        for lvl in ["ph", "ch", "pl", "cl"]:
                            if candle.get(f"is_{lvl}"):
                                return lvl
                        return None

                    def process_single_trend(trend, depth_2=0, max_depth_2=5):
                        if depth_2 > max_depth_2:
                            log(f"Max adaptation depth_2 reached for {trend['line_id']} → DISCARDING", "WARNING")
                            return False, trend

                        line_id = trend["line_id"]
                        fx, fy = trend["from_final_valid_sender_x"], trend["from_final_valid_sender_y"]
                        tx, ty = trend["from_final_valid_receiver_x"], trend["from_final_valid_receiver_y"]
                        color = trend["color"]
                        from_key = trend["from_key"]
                        direction = trend["direction"]
                        seq_count = trend["breakout_sequence_count"]
                        point_keys = trend["point_keys"]
                        opposition_key_name = point_keys["opposition"]
                        retest_key_name = point_keys["retest"]
                        sender_condition = trend.get("sender_condition", "none")
                        extreme_rule = trend.get("extreme_rule", "continue")

                        # --- Initialize variables ---
                        target_zone_cnum = None
                        target_zone_candle = None
                        target_zone_candle_found = False
                        extreme_target_reached_candle = None
                        extreme_target_reached_candle_found = False
                        target_reached_limit_found = None  # New: the candle that gets the diamond
                        extreme_target_reached_mutual_candle = []
                        extreme_target_reached_mutual_candle_found = False
                        target_zone_mutuals_cnums = []
                        target_zone_mutuals_found = False
                        target_zone_mutual_limit_cnum = None
                        target_zone_mutual_limit_found = False
                        extreme_opposition_cnum = None
                        extreme_opposition_reason = None

                        # === Sender candle ===
                        if "sender_cnum" not in trend:
                            sender_candle = next((c for c in reversed(candles) if is_level_match(c, from_key)), None)
                            trend["sender_cnum"] = sender_candle["candle_number"] if sender_candle else None
                        sender_cnum = trend["sender_cnum"]
                        if not sender_cnum or sender_cnum not in positions:
                            return False, trend

                        receiver_cnum = trend["receiver_cnum"]
                        if tx - fx == 0:
                            return False, trend

                        # Pre-compute level types for final sender & receiver
                        final_sender_level = get_candle_level(sender_cnum)
                        final_receiver_level = get_candle_level(receiver_cnum)

                        # === Extend ray for detection ===
                        slope = (ty - fy) / (tx - fx)
                        pending_entry = img.shape[1] - 10
                        extend_y = int(fy + slope * (pending_entry - fx))

                        # === 1. Find future interceptors ===
                        interceptors = []
                        receiver_x = positions.get(receiver_cnum, {}).get("x", -99999)
                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions or positions[cn]["x"] <= receiver_x:
                                continue
                            pos = positions[cn]
                            if line_intersects_rect(fx, fy, pending_entry, extend_y,
                                                    pos["x"] - pos["width"]//2, pos["high_y"],
                                                    pos["x"] + pos["width"]//2, pos["low_y"]):
                                body_y = (pos["high_y"] + pos["low_y"]) // 2
                                interceptors.append({
                                    "candle_number": cn, "x": pos["x"], "y": body_y,
                                    "high": c["high"], "low": c["low"], "close": c["close"], "open": c["open"],
                                    "candle": c
                                })

                        if not interceptors:
                            draw_final_trendline(trend, fx, fy, tx, ty, color)
                            final_teams_2[line_id] = {"team": {"trendline_info": {
                                "line_id": line_id,
                                "interceptors": [],
                                "touched_interceptors": [],
                                "opposition_candle": None,
                                "further_oppositions": [],
                                "extreme_opposition_candle": None,
                                "extreme_opposition_reason": "No interceptors found",
                                "valid": True,
                                "adapted": depth_2 > 0,
                                "final_sender_cnum": trend["sender_cnum"],
                                "final_receiver_cnum": trend["receiver_cnum"],
                                "final_sender_level": final_sender_level,
                                "final_receiver_level": final_receiver_level,
                            }}}
                            log(f"{line_id}: No interceptors → Valid & drawn", "INFO")
                            return True, trend

                        # === 2. Find opposition candle(s) ===
                        opposition_cnum = None
                        further_oppositions = []
                        if receiver_cnum and opposition_key_name:
                            oldest_int_cnum = min(i["candle_number"] for i in interceptors)
                            receiver_candle = next((c for c in candles if c["candle_number"] == receiver_cnum), None)
                            if receiver_candle:
                                receiver_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                                if receiver_level:
                                    target_levels = BULLISH_FAMILY if opposition_key_name == "bullish" else BEARISH_FAMILY
                                    if is_parent_level(receiver_level):
                                        target_levels = {lvl for lvl in target_levels if is_parent_level(lvl)}
                                    for cn in range(receiver_cnum - 1, oldest_int_cnum - 1, -1):
                                        if cn <= 0: break
                                        candle = next((c for c in candles if c["candle_number"] == cn), None)
                                        if candle and any(candle.get(f"is_{lvl}") for lvl in target_levels):
                                            if opposition_cnum is None:
                                                opposition_cnum = cn
                                            else:
                                                further_oppositions.append(cn)

                        if opposition_cnum is None:
                            log(f"{line_id}: No opposition candle found → DISCARDING trendline", "INFO")
                            return False, trend

                        # === 3. Find extreme interceptor ===
                        extreme_interceptor_cnum = None
                        extreme_interceptor_data = None
                        pre_opp_ints = [i for i in interceptors if i["candle_number"] < opposition_cnum]
                        if pre_opp_ints:
                            pre_opp_ints.sort(key=lambda i: i["candle_number"])
                            opposing_red = is_bullish_level(from_key)
                            for intr in reversed(pre_opp_ints):
                                is_red = intr["close"] < intr["open"]
                                if (opposing_red and is_red) or (not opposing_red and not is_red):
                                    extreme_interceptor_cnum = intr["candle_number"]
                                    extreme_interceptor_data = intr
                                    break

                        if extreme_interceptor_cnum is None:
                            log(f"{line_id}: No extreme interceptor found → DISCARDING trendline", "INFO")
                            return False, trend

                        # === 4. Check breakout sequence ===
                        breakout_sequence_cnums = []
                        has_breakout = False
                        if direction == "breakout" and seq_count > 0:
                            ext_c = next(c for c in candles if c["candle_number"] == extreme_interceptor_cnum)
                            younger = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                            younger.sort(key=lambda x: x["candle_number"], reverse=True)
                            for i in range(len(younger) - seq_count + 1):
                                seq = younger[i:i + seq_count]
                                if is_bullish_level(from_key):
                                    if all(c["high"] < ext_c["high"] and c["low"] < ext_c["low"] for c in seq):
                                        breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                        break
                                else:
                                    if all(c["high"] > ext_c["high"] and c["low"] > ext_c["low"] for c in seq):
                                        breakout_sequence_cnums = [c["candle_number"] for c in seq]
                                        break
                            has_breakout = bool(breakout_sequence_cnums)

                        # === BREAKOUT → Valid & draw current line ===
                        if has_breakout:
                            draw_final_trendline(trend, fx, fy, tx, ty, color)
                            if extreme_interceptor_data:
                                mark_breakout_extreme_interceptor(img, extreme_interceptor_data["x"], extreme_interceptor_data["y"], color)
                            for cnum in breakout_sequence_cnums:
                                if cnum in positions:
                                    body_y = (positions[cnum]["high_y"] + positions[cnum]["low_y"]) // 2
                                    mark_breakout_candle(img, positions[cnum]["x"], body_y, color)

                            # === Retest ===
                            retest_cnum = None
                            retest_candle = None
                            if retest_key_name and breakout_sequence_cnums:
                                younger = [c for c in candles if c["candle_number"] < extreme_interceptor_cnum]
                                younger.sort(key=lambda x: x["candle_number"], reverse=True)
                                allowed = BULLISH_FAMILY if retest_key_name == "bullish" else BEARISH_FAMILY
                                receiver_candle = next((c for c in candles if c["candle_number"] == receiver_cnum), None)
                                if receiver_candle:
                                    r_level = next((k for k in ["ph","ch","pl","cl"] if receiver_candle.get(f"is_{k}")), None)
                                    if r_level and is_parent_level(r_level):
                                        allowed = {lvl for lvl in allowed if is_parent_level(lvl)}
                                for c in younger:
                                    if c["candle_number"] in breakout_sequence_cnums: continue
                                    if any(c.get(f"is_{lvl}") for lvl in allowed):
                                        retest_cnum = c["candle_number"]
                                        retest_candle = c
                                        break

                            if retest_cnum and retest_cnum in positions and retest_candle:
                                level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                                is_bullish_retest = level in {"pl", "cl"}
                                y_price = positions[retest_cnum]["low_y"] if is_bullish_retest else positions[retest_cnum]["high_y"]
                                draw_double_retest_arrow(img, positions[retest_cnum]["x"], y_price, color, direction_up=is_bullish_retest)

                            # === Target Zone Marker ===
                            if retest_cnum and retest_candle:
                                level = next(k for k in ["ph","ch","pl","cl"] if retest_candle.get(f"is_{k}"))
                                nr = parent_neighbor_right if is_parent_level(level) else child_neighbor_right
                                if nr > 0:
                                    target_zone_cnum = retest_cnum - nr
                                    if target_zone_cnum > 0:
                                        target_zone_candle = next((c for c in candles if c["candle_number"] == target_zone_cnum), None)
                                        if target_zone_cnum in positions:
                                            target_zone_candle_found = True
                                            txz = positions[target_zone_cnum]["x"]
                                            tyz = positions[target_zone_cnum]["high_y"] - 20 if is_bullish_level(from_key) else positions[target_zone_cnum]["low_y"] + 20
                                            draw_target_zone_marker(img, txz, tyz, color)

                            # === Target Reached Diamond Marker ===
                            target_reached_cnums = []
                            if target_zone_cnum:
                                target_reach_candidates = [i for i in interceptors if i["candle_number"] < target_zone_cnum]
                                for c_data in target_reach_candidates:
                                    target_reached_cnums.append(c_data["candle_number"])
                                if target_reached_cnums:
                                    sorted_reached = sorted(target_reached_cnums, reverse=True)
                                    extreme_target_reached_candle = sorted_reached[0] if sorted_reached else None
                                    extreme_target_reached_candle_found = extreme_target_reached_candle is not None
                                    mutual_candidates = sorted_reached[1:3]
                                    extreme_target_reached_mutual_candle = []
                                    if extreme_target_reached_candle is not None:
                                        extreme_candle_data = next((c for c in candles if c["candle_number"] == extreme_target_reached_candle), None)
                                        if extreme_candle_data:
                                            is_bullish = is_bullish_level(from_key)
                                            for cnum in mutual_candidates:
                                                candidate_candle_data = next((c for c in candles if c["candle_number"] == cnum), None)
                                                if candidate_candle_data:
                                                    if (is_bullish and candidate_candle_data["high"] > extreme_candle_data["high"]) or \
                                                    (not is_bullish and candidate_candle_data["low"] < extreme_candle_data["low"]):
                                                        extreme_target_reached_mutual_candle.append(cnum)
                                    extreme_target_reached_mutual_candle_found = bool(extreme_target_reached_mutual_candle)

                                    # === NEW: Determine which candle gets the diamond (2 candles before extreme) ===
                                    target_reached_limit_found = extreme_target_reached_candle
                                    if extreme_target_reached_candle is not None:
                                        mark_cnum = extreme_target_reached_candle - 2
                                        if mark_cnum > 0 and mark_cnum in positions:
                                            target_reached_limit_found = mark_cnum

                                    # Draw diamond on the selected candle
                                    if target_reached_limit_found in positions:
                                        pos = positions[target_reached_limit_found]
                                        body_y = (pos["high_y"] + pos["low_y"]) // 2
                                        cv2.drawMarker(img, (pos["x"], body_y), color,
                                                    markerType=cv2.MARKER_DIAMOND,
                                                    markerSize=12, thickness=3)

                            # === Target Zone Mutuals ===
                            if target_zone_cnum and target_zone_candle:
                                is_bullish = is_bullish_level(from_key)
                                target_zone_price = target_zone_candle["high"] if is_bullish else target_zone_candle["low"]
                                younger_c1_num = target_zone_cnum - 1
                                younger_c2_num = target_zone_cnum - 2
                                younger_c3_num = target_zone_cnum - 3
                                candidates = []
                                if younger_c1_num > 0:
                                    c1 = next((c for c in candles if c["candle_number"] == younger_c1_num), None)
                                    if c1: candidates.append((younger_c1_num, c1))
                                if younger_c2_num > 0:
                                    c2 = next((c for c in candles if c["candle_number"] == younger_c2_num), None)
                                    if c2: candidates.append((younger_c2_num, c2))
                                if younger_c3_num > 0:
                                    c3 = next((c for c in candles if c["candle_number"] == younger_c3_num), None)
                                    if c3: candidates.append((younger_c3_num, c3))
                                for cn, candle in candidates[:2]:
                                    meets_condition = False
                                    if is_bullish:
                                        if candle["high"] >= target_zone_price:
                                            meets_condition = True
                                    else:
                                        if candle["low"] <= target_zone_price:
                                            meets_condition = True
                                    if meets_condition:
                                        target_zone_mutuals_cnums.append(cn)
                                target_zone_mutuals_found = bool(target_zone_mutuals_cnums)
                                target_zone_mutual_limit_cnum = None
                                if len(candidates) >= 3:
                                    cn, candle = candidates[2]
                                    target_zone_mutual_limit_cnum = cn
                                    target_zone_mutual_limit_found = True

                                for cn in target_zone_mutuals_cnums:
                                    if cn in positions:
                                        pos = positions[cn]
                                        body_y = (pos["high_y"] + pos["low_y"]) // 2
                                        cv2.drawMarker(img, (pos["x"], body_y), color,
                                                    markerType=cv2.MARKER_CROSS,
                                                    markerSize=10, thickness=2)
                                if target_zone_mutual_limit_cnum and target_zone_mutual_limit_cnum in positions:
                                    pos = positions[target_zone_mutual_limit_cnum]
                                    body_y = (pos["high_y"] + pos["low_y"]) // 2
                                    cv2.circle(img, (pos["x"], body_y), 8, color, 2)

                            # === Extreme Opposition Logic ===
                            extreme_opposition_reason = "Not calculated"
                            if target_zone_cnum is not None:
                                opposition_group = [opposition_cnum] + further_oppositions
                                valid_oppositions = [cn for cn in opposition_group if target_zone_cnum <= cn <= opposition_cnum]

                                if valid_oppositions:
                                    opp_candles = []
                                    for cn in valid_oppositions:
                                        candle = next((c for c in candles if c["candle_number"] == cn), None)
                                        if candle:
                                            opp_candles.append((cn, candle))

                                    if opp_candles:
                                        is_bullish_trend = is_bullish_level(from_key)
                                        if is_bullish_trend:
                                            extreme_opposition_cnum = max(opp_candles, key=lambda x: x[1]["high"])[0]
                                        else:
                                            extreme_opposition_cnum = min(opp_candles, key=lambda x: x[1]["low"])[0]

                                        if extreme_opposition_cnum and extreme_opposition_cnum in positions:
                                            opp_candle = next(c for c in candles if c["candle_number"] == extreme_opposition_cnum)
                                            opp_level = next(k for k in ["ph","ch","pl","cl"] if opp_candle.get(f"is_{k}"))
                                            direction_up = opp_level in {"pl", "cl"}
                                            arrow_y = positions[extreme_opposition_cnum]["low_y"] if direction_up else positions[extreme_opposition_cnum]["high_y"]
                                            draw_opposition_arrow(img, positions[extreme_opposition_cnum]["x"], arrow_y, color, direction_up=direction_up)

                                        extreme_opposition_reason = f"Selected extreme opposition: {extreme_opposition_cnum}"
                                    else:
                                        extreme_opposition_reason = "No candle data found for valid opposition candles"
                                else:
                                    extreme_opposition_reason = f"No opposition candles in range [{target_zone_cnum} - {opposition_cnum}]"
                            else:
                                extreme_opposition_reason = "No target zone found → extreme opposition skipped"

                            # === Special "X" marker for invalid target zone ===
                            invalid_target_zone = False
                            if target_zone_cnum is not None and extreme_opposition_cnum is not None and target_zone_candle is not None:
                                opp_candle = next((c for c in candles if c["candle_number"] == extreme_opposition_cnum), None)
                                tz_candle = target_zone_candle

                                if opp_candle and tz_candle:
                                    tz_x = positions[target_zone_cnum]["x"]
                                    tz_high_y = positions[target_zone_cnum]["high_y"]
                                    tz_low_y = positions[target_zone_cnum]["low_y"]

                                    if not is_bullish_level(from_key):  # Bearish
                                        if tz_candle["low"] < opp_candle["high"]:
                                            invalid_target_zone = True
                                            cv2.putText(img, "X", (tz_x, tz_low_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                                            cv2.putText(img, "X", (tz_x, tz_low_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                                    else:  # Bullish
                                        if tz_candle["high"] > opp_candle["low"]:
                                            invalid_target_zone = True
                                            cv2.putText(img, "X", (tz_x, tz_high_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                                            cv2.putText(img, "X", (tz_x, tz_high_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                            # === Final dictionary ===
                            final_info = {
                                "line_id": line_id,
                                "interceptors": [i for i in interceptors if "candle" not in i],
                                "touched_interceptors": [i["candle_number"] for i in interceptors],
                                "breakout_extreme_interceptor_candle": extreme_interceptor_cnum,
                                "breakout_sequence_candles": breakout_sequence_cnums,
                                "retest_candle": retest_cnum,
                                "target_zone_candle": target_zone_cnum,
                                "target_reached_candles": target_reached_cnums,
                                "extreme_target_reached_candle": extreme_target_reached_candle,
                                "target_reached_limit_found": target_reached_limit_found,  # ← NEW
                                "extreme_target_reached_mutual_candle": extreme_target_reached_mutual_candle,
                                "target_zone_mutuals": target_zone_mutuals_cnums,
                                "target_zone_mutual_limit": target_zone_mutual_limit_cnum,
                                "valid": True,
                                "adapted": depth_2 > 0,
                                "target_zone_candle_found": target_zone_candle_found,
                                "extreme_target_reached_candle_found": extreme_target_reached_candle_found,
                                "extreme_target_reached_mutual_candle_found": extreme_target_reached_mutual_candle_found,
                                "target_zone_mutuals_found": target_zone_mutuals_found,
                                "target_zone_mutual_limit_found": target_zone_mutual_limit_found,
                                "final_sender_cnum": trend["sender_cnum"],
                                "final_receiver_cnum": trend["receiver_cnum"],
                                "final_sender_level": final_sender_level,
                                "final_receiver_level": final_receiver_level,
                            }

                            if invalid_target_zone:
                                final_info.update({
                                    "opposition_candle": None,
                                    "further_oppositions": [],
                                    "extreme_opposition_candle": None,
                                    "extreme_opposition_reason": "Removed – invalid target zone (X marker)",
                                    "breakout_extreme_interceptor_candle": None,
                                    "breakout_sequence_candles": [],
                                    "retest_candle": None,
                                    "target_zone_candle": None,
                                    "target_reached_candles": [],
                                    "extreme_target_reached_candle": None,
                                    "target_reached_limit_found": None,  # ← also cleared on invalid
                                    "extreme_target_reached_mutual_candle": [],
                                    "target_zone_mutuals": [],
                                    "target_zone_mutual_limit": None,
                                    "target_zone_candle_found": False,
                                    "target_zone_mutuals_found": False,
                                    "target_zone_mutual_limit_found": False,
                                    "extreme_target_reached_candle_found": False,
                                    "extreme_target_reached_mutual_candle_found": False,
                                    "valid": False,
                                    "final_sender_cnum": trend["sender_cnum"],
                                    "final_receiver_cnum": trend["receiver_cnum"],
                                    "final_sender_level": final_sender_level,
                                    "final_receiver_level": final_receiver_level,
                                })
                            else:
                                final_info.update({
                                    "opposition_candle": opposition_cnum,
                                    "further_oppositions": further_oppositions,
                                    "extreme_opposition_candle": extreme_opposition_cnum,
                                    "extreme_opposition_reason": extreme_opposition_reason,
                                })

                            final_teams_2[line_id] = {"team": {"trendline_info": final_info}}

                            log(f"{line_id}: BREAKOUT conf_2irmed → Valid & drawn", "SUCCESS")
                            return True, trend

                        # === NO BREAKOUT → Try to adapt ===
                        if depth_2 < max_depth_2:
                            touches = []
                            min_c = min(sender_cnum, receiver_cnum)
                            max_c = max(sender_cnum, receiver_cnum)
                            for c in candles:
                                cn = c["candle_number"]
                                if cn in [sender_cnum, receiver_cnum] or cn not in positions: continue
                                if not (min_c <= cn <= max_c): continue
                                pos = positions[cn]
                                if line_intersects_rect(fx, fy, tx, ty,
                                                        pos["x"] - pos["width"]//2, pos["high_y"],
                                                        pos["x"] + pos["width"]//2, pos["low_y"]):
                                    touches.append(cn)

                            extreme_cnum = extreme_y = None
                            if touches and extreme_rule != "continue":
                                s_min, s_max = min(touches), max(touches)
                                if is_bearish_level(from_key):
                                    best = max((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["high"], default=None)
                                else:
                                    best = min((c for c in candles if s_min <= c["candle_number"] <= s_max), key=lambda c: c["low"], default=None)
                                if best:
                                    extreme_cnum = best["candle_number"]
                                    extreme_y = positions[extreme_cnum]["high_y"] if is_bearish_level(from_key) else positions[extreme_cnum]["low_y"]

                            new_fx, new_fy = fx, fy
                            new_tx, new_ty = tx, ty
                            new_receiver_cnum = receiver_cnum
                            if extreme_rule == "new_from" and extreme_cnum:
                                new_fx, new_fy = positions[extreme_cnum]["x"], extreme_y
                                trend["sender_cnum"] = extreme_cnum
                            elif extreme_rule == "new_to" and extreme_cnum:
                                new_tx, new_ty = positions[extreme_cnum]["x"], extreme_y
                                new_receiver_cnum = extreme_cnum

                            if (new_fx, new_fy, new_tx, new_ty) != (fx, fy, tx, ty):
                                if validate_sender_condition(sender_cnum, new_receiver_cnum, from_key, sender_condition):
                                    log(f"{line_id}: Adapting via extreme_intruder → {extreme_cnum}", "INFO")
                                    trend.update({
                                        "from_final_valid_sender_x": new_fx, "from_final_valid_sender_y": new_fy,
                                        "from_final_valid_receiver_x": new_tx, "from_final_valid_receiver_y": new_ty,
                                        "receiver_cnum": new_receiver_cnum,
                                        "line_id": f"{line_id}_adapted{depth_2+1}"
                                    })
                                    return process_single_trend(trend, depth_2 + 1, max_depth_2)

                            # Fallback price extreme adaptation
                            if is_bullish_level(from_key):
                                candidate = min(interceptors, key=lambda i: i["low"])
                            else:
                                candidate = max(interceptors, key=lambda i: i["high"])

                            base_price = next(c["low"] if is_bullish_level(from_key) else c["high"] for c in candles if c["candle_number"] == receiver_cnum)
                            should_adapt = (is_bullish_level(from_key) and candidate["low"] < base_price) or \
                                        (not is_bullish_level(from_key) and candidate["high"] > base_price)

                            if should_adapt and validate_sender_condition(sender_cnum, candidate["candle_number"], from_key, sender_condition):
                                log(f"{line_id}: Adapting to price extreme → {candidate['candle_number']}", "INFO")
                                trend.update({
                                    "from_final_valid_receiver_x": candidate["x"],
                                    "from_final_valid_receiver_y": get_y_position(positions, candidate["candle_number"], from_key),
                                    "receiver_cnum": candidate["candle_number"],
                                    "line_id": f"{line_id}_adapted{depth_2+1}"
                                })
                                return process_single_trend(trend, depth_2 + 1, max_depth_2)

                        # === DISCARD ===
                        log(f"{line_id}: No breakout + no valid adaptation → DISCARDING trendline", "INFO")
                        return False, trend

                    # --- MAIN EXECUTION ---
                    for trend in final_trendlines_for_redraw[:]:
                        orig_id = trend["line_id"].split("_")[0].lstrip("T")
                        orig_conf_2 = next((c for c in trend_list if c["id"] == orig_id), None)
                        if orig_conf_2:
                            trend["sender_condition"] = orig_conf_2.get("sender_condition", "none")
                            trend["extreme_rule"] = orig_conf_2.get("rule", "continue")

                        was_valid, final_state = process_single_trend(trend, depth_2=0)

                        if was_valid:
                            base_id = final_state["line_id"].split("_")[0]
                            final_valid_trends_2[base_id] = final_state

                    log(f"→ {symbol_folder}/{tf_folder} | {len(final_valid_trends_2)} Trendlines validated for BREAKOUT (strict validation)", "SUCCESS")

                    return final_valid_trends_2

                def DRAW_ALL_POINTS_LEVELS(valid_trends_2):
                    if not valid_trends_2:
                        log("No valid trendlines → nothing to draw for points levels.", "INFO")
                        return

                    drawn_levels = 0
                    pending_entry = img.shape[1] - 10

                    for trend in valid_trends_2.values():
                        color = trend["color"]
                        base_id = trend["line_id"].split("_")[0].lstrip("T")
                        line_id = trend["line_id"].split("_")[0]
                        orig_conf_2 = next((c for c in trend_list if c["id"] == base_id), None)
                        if not orig_conf_2:
                            continue

                        direction = orig_conf_2.get("direction", "continuation").lower()
                        horiz_cfg = orig_conf_2.get("horizontal_line_subject", {})
                        subject_wanted = horiz_cfg.get("subject", "").strip().lower()
                        entry_wanted = horiz_cfg.get("entry", "").strip().lower()

                        if not subject_wanted or entry_wanted not in {"high_price", "low_price"}:
                            continue

                        draw_high = entry_wanted == "high_price"
                        draw_low = entry_wanted == "low_price"

                        # === GET TARGET CANDLE ===
                        def get_target_cnum_and_label():
                            info = final_teams_2.get(line_id, {}).get("team", {}).get("trendline_info", {})
                            if subject_wanted == "sender":
                                x_check = trend["from_final_valid_sender_x"]
                                for cnum, pos in positions.items():
                                    if abs(pos["x"] - x_check) < 12:
                                        return cnum, "S"
                            elif subject_wanted == "receiver":
                                x_check = trend["from_final_valid_receiver_x"]
                                for cnum, pos in positions.items():
                                    if abs(pos["x"] - x_check) < 12:
                                        return cnum, "R"
                            elif subject_wanted in {"opposition", "opp"}:
                                cnum = info.get("extreme_opposition_candle") or info.get("opposition_candle")
                                if cnum and cnum in positions:
                                    return cnum, "EXT-OPP" if info.get("extreme_opposition_candle") else "OPP"
                            elif subject_wanted == "extreme":
                                cnum = info.get("breakout_extreme_interceptor_candle") or info.get("continuation_extreme_interceptor_candle")
                                if cnum and cnum in positions:
                                    return cnum, "EXT"
                            elif subject_wanted == "retest":
                                cnum = info.get("retest_candle")
                                if cnum and cnum in positions:
                                    return cnum, "RET"
                            elif subject_wanted in {"target_zone", "target", "tz"}:
                                cnum = info.get("target_zone_candle")
                                if cnum and cnum in positions:
                                    return cnum, "TZ"
                            return None, ""

                        target_cnum, label = get_target_cnum_and_label()
                        if not target_cnum or target_cnum not in positions:
                            continue

                        # For breakout: require target zone
                        if direction == "breakout":
                            tz_cnum = final_teams_2.get(line_id, {}).get("team", {}).get("trendline_info", {}).get("target_zone_candle")
                            if tz_cnum is None:
                                continue

                        # === FULL CANDLE DATA ===
                        source_candle = next(c for c in candles if c["candle_number"] == target_cnum)
                        high_price = source_candle["high"]
                        low_price = source_candle["low"]
                        pos = positions[target_cnum]
                        high_y = pos["high_y"]
                        low_y = pos["low_y"]

                        # === TOUCHES & BLOCKER ===
                        touched_candles = []
                        start_x = pos["x"]
                        for c in candles:
                            cn = c["candle_number"]
                            if cn not in positions or positions[cn]["x"] <= start_x:
                                continue
                            p = positions[cn]
                            touched = False
                            if draw_high and p["high_y"] <= high_y <= p["low_y"]:
                                touched = True
                            elif draw_low and p["high_y"] <= low_y <= p["low_y"]:
                                touched = True
                            elif draw_high and abs(p["high_y"] - high_y) < 8:
                                touched = True
                            elif draw_low and abs(p["low_y"] - low_y) < 8:
                                touched = True
                            if touched:
                                body_y = (p["high_y"] + p["low_y"]) // 2
                                touched_candles.append({
                                    "candle_number": cn, "x": p["x"], "y": body_y,
                                    "high": c["high"], "low": c["low"], "close": c["close"], "open": c["open"]
                                })

                        extreme_touch_cnum = None
                        extreme_touch_x = pending_entry
                        target_zone_cnum = final_teams_2.get(line_id, {}).get("team", {}).get("trendline_info", {}).get("target_zone_candle")
                        if target_zone_cnum is not None and touched_candles:
                            candidates = [t for t in touched_candles if t["candle_number"] < target_zone_cnum]
                            if candidates:
                                extreme_touch = max(candidates, key=lambda x: x["candle_number"])
                                extreme_touch_cnum = extreme_touch["candle_number"]
                                extreme_touch_x = positions[extreme_touch_cnum]["x"]
                                cv2.putText(img, "blck", (extreme_touch["x"] + 20, extreme_touch["y"] - 10),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

                        end_x = extreme_touch_x

                        # === BREAKOUT VALIDATION RULE ===
                        if direction == "breakout" and target_zone_cnum is not None:
                            tz_candle = next((c for c in candles if c["candle_number"] == target_zone_cnum), None)
                            if tz_candle:
                                tz_high = tz_candle["high"]
                                tz_low = tz_candle["low"]

                                should_draw = True
                                if "bear" in direction or "down" in direction.lower():
                                    if draw_high and tz_high <= low_price:
                                        should_draw = False
                                    elif draw_low and tz_high <= high_price:
                                        should_draw = False
                                elif "bull" in direction or "up" in direction.lower():
                                    if draw_high and tz_low >= high_price:
                                        should_draw = False
                                    elif draw_low and tz_low >= high_price:
                                        should_draw = False

                                if not should_draw:
                                    continue

                        # === DRAW LINES ===
                        if draw_high:
                            cv2.line(img, (pos["x"], high_y), (end_x, high_y), color, 1, cv2.LINE_AA)
                            cv2.putText(img, f"{label}-H", (pos["x"] - 58, high_y + 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                        if draw_low:
                            cv2.line(img, (pos["x"], low_y), (end_x, low_y), color, 1, cv2.LINE_AA)
                            cv2.putText(img, f"{label}-L", (pos["x"] - 58, low_y + 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                        drawn_levels += 1

                        # === SAVE PERFECT DATA (with exit_price and ordered placement) ===
                        if line_id not in final_teams_2:
                            final_teams_2[line_id] = {"team": {"trendline_info": {}}}

                        info = final_teams_2[line_id]["team"]["trendline_info"]

                        subject_map = {
                            "sender": "Sender", "receiver": "Receiver", "opposition": "Opposition",
                            "extreme": "Extreme", "retest": "Retest", "target_zone": "Target Zone"
                        }
                        display_subject = subject_map.get(subject_wanted, subject_wanted.title())

                        entry_price = high_price if draw_high else low_price
                        exit_price = low_price if draw_high else high_price  # Opposite extreme

                        # Build enhanced pending_entry_point with exit_price
                        pending_entry_point = {
                            "pending_entry_high_price": high_price,
                            "pending_entry_low_price": low_price,
                            "candle_number": target_cnum,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "type": "high" if draw_high else "low",
                            "subject": display_subject
                        }

                        # Preserve all existing data except pending_entry_point (to avoid duplication)
                        preserved_data = {k: v for k, v in info.items() if k != "pending_entry_point"}

                        # Reconstruct info with desired order: line_id → pending_entry_point → everything else
                        new_info = {
                            "line_id": info.get("line_id", line_id),
                            "pending_entry_point": pending_entry_point,  # Now high in JSON output
                        }
                        new_info.update(preserved_data)  # Add back interceptors, etc.

                        # Add the rest of the pending entry fields
                        new_info.update({
                            "pending_entry_subject": display_subject,
                            "pending_entry_candle": target_cnum,
                            "pending_entry_high_price": high_price,
                            "pending_entry_low_price": low_price,
                            "pending_entry_is_full_extension": extreme_touch_cnum is None,
                            "horizontal_level_blockers": touched_candles,
                            "extreme_horizontal_level_blocker": extreme_touch_cnum,
                        })

                        # Apply the new structure
                        info.clear()
                        info.update(new_info)

                    log(f"→ {symbol_folder}/{tf_folder} | {drawn_levels} Horizontal levels drawn (with exit_price in pending_entry_point)", "SUCCESS")
 
                def enrich_candle_details(candles_list_2, final_teams_2, final_valid_trends_2, report_path, technique_path_2):
                    import os
                    import json
                    
                    candle_lookup = {c["candle_number"]: c for c in candles_list_2}
                    enriched_count = 0
                    ESSENTIAL_FIELDS = ["open", "high", "low", "close", "time", "timeframe", "symbol"]

                    # === 1. Load continuation.json to determine if it's breakout-only ===
                    is_breakout_strategy = False
                    try:
                        with open(technique_path_2, 'r', encoding='utf-8') as f:
                            technique_conf_2ig = json.load(f)
                        
                        trendline_conf_2igs = technique_conf_2ig.get("trendline", {})
                        if isinstance(trendline_conf_2igs, dict):
                            for conf_2ig in trendline_conf_2igs.values():
                                if isinstance(conf_2ig, dict):
                                    direction = conf_2ig.get("DIRECTION", "").strip().lower()
                                    if direction == "breakout":
                                        is_breakout_strategy = True
                                        break  # One breakout conf_2ig → treat whole technique as breakout
                    except Exception as e:
                        log(f"Failed to load technique JSON for direction check: {e}", "ERROR")
                        # Fallback: assume continuation if cannot determine
                        is_breakout_strategy = False

                    if not is_breakout_strategy:
                        log("Technique is continuation-only → processing as continuation", "INFO")
                    else:
                        log("Technique is breakout-conf_2igured → processing as breakout", "INFO")

                    # === Helpers ===
                    def enrich_if_needed(container, key):
                        nonlocal enriched_count
                        val = container.get(key)
                        if isinstance(val, int) and val in candle_lookup:
                            full = candle_lookup[val]
                            clean = {"candle_number": val}
                            for f in ESSENTIAL_FIELDS:
                                clean[f] = full[f]
                            container[key] = clean
                            enriched_count += 1

                    def enrich_list_if_needed(container, key):
                        nonlocal enriched_count
                        items = container.get(key, [])
                        if not isinstance(items, list):
                            return
                        new_list = []
                        for item in items:
                            if isinstance(item, int):
                                cn = item
                                if cn in candle_lookup:
                                    full = candle_lookup[cn]
                                    clean = {"candle_number": cn}
                                    for f in ESSENTIAL_FIELDS:
                                        clean[f] = full[f]
                                    new_list.append(clean)
                                    enriched_count += 1
                                else:
                                    new_list.append({"candle_number": cn})
                            elif isinstance(item, dict):
                                cn = item.get("candle_number")
                                if isinstance(cn, int):
                                    cleaned = {"candle_number": cn}
                                    for f in ESSENTIAL_FIELDS:
                                        cleaned[f] = item.get(f, candle_lookup.get(cn, {}).get(f))
                                    new_list.append(cleaned)
                                    enriched_count += 1
                                else:
                                    new_list.append(item)
                            else:
                                new_list.append(item)
                        if new_list:
                            container[key] = new_list

                    # === MAIN LOOP ===
                    for line_id, data in final_teams_2.items():
                        info = data["team"]["trendline_info"]

                        # === Direction is now globally determined from continuation.json ===
                        direction = "breakout" if is_breakout_strategy else "continuation"

                        # === Determine trendline_connection_type ===
                        trendline_connection_type = "unknown"
                        if direction == "continuation":
                            from_type = info.get("FROM")
                            if from_type in ["pl", "cl"]:
                                trendline_connection_type = "bullish"
                            elif from_type in ["ph", "ch"]:
                                trendline_connection_type = "bearish"
                        else:  # breakout
                            sender_level = info.get("final_sender_level")
                            if sender_level in ["pl", "cl"]:
                                trendline_connection_type = "bullish"
                            elif sender_level in ["ph", "ch"]:
                                trendline_connection_type = "bearish"
                            else:
                                connection_type = info.get("connection_type", "unknown")
                                if connection_type in ["ph to ph", "ch to ch"]:
                                    trendline_connection_type = "bearish"
                                elif connection_type in ["pl to pl", "cl to cl"]:
                                    trendline_connection_type = "bullish"

                        # === Order type based on connection type ===
                        order_type = None
                        if direction == "continuation":
                            if trendline_connection_type == "bullish":
                                order_type = "buy now"
                            elif trendline_connection_type == "bearish":
                                order_type = "sell now"
                        else:  # breakout
                            if trendline_connection_type == "bullish":
                                order_type = "sell now"
                            elif trendline_connection_type == "bearish":
                                order_type = "buy now"

                        # === Common enrichments ===
                        common_single = ["pending_entry_candle", "extreme_horizontal_level_blocker"]
                        for key in common_single:
                            enrich_if_needed(info, key)
                        common_lists = ["touched_interceptors", "horizontal_level_blockers", "interceptors"]
                        for key in common_lists:
                            enrich_list_if_needed(info, key)

                        entry = info.get("pending_entry_point")
                        if isinstance(entry, dict) and isinstance(entry.get("candle_number"), int):
                            cn = entry["candle_number"]
                            if cn in candle_lookup:
                                full = candle_lookup[cn]
                                entry["candle"] = {"candle_number": cn}
                                for f in ESSENTIAL_FIELDS:
                                    entry["candle"][f] = full[f]
                                enriched_count += 1

                        # === Pre-compute booleans ===
                        pre_bools = {
                            "valid": info.get("valid", False),
                            "adapted": info.get("adapted", False),
                            "retest_candle_found": bool(info.get("retest_candle")),
                            "target_zone_candle_found": bool(info.get("target_zone_candle")),
                            "target_zone_mutuals_found": bool(info.get("target_zone_mutuals")),
                            "target_zone_mutual_limit_found": bool(info.get("target_zone_mutual_limit")),
                            "extreme_target_reached_candle_found": bool(info.get("extreme_target_reached_candle")),
                            "extreme_target_reached_mutual_candle_found": bool(info.get("extreme_target_reached_mutual_candle")),
                            "extreme_target_reached_limit_found": bool(info.get("target_reached_limit_found")),
                            "continuation_extreme_interceptor_found": bool(info.get("continuation_extreme_interceptor_candle")),
                            "continuation_extreme_interceptor_limit_found": bool(info.get("Continuation_extreme_Interceptor_limit")),
                            "pending_entry_is_full_extension": info.get("pending_entry_is_full_extension", False),
                            "extreme_opposition_candle_found": bool(info.get("extreme_opposition_candle")),
                            "extreme_horizontal_level_blocker_found": bool(info.get("extreme_horizontal_level_blocker")),
                        }

                        # === Keys to remove based on direction ===
                        breakout_only_keys = [
                            "breakout_extreme_interceptor_candle", "breakout_sequence_candles", "retest_candle",
                            "target_zone_candle", "target_reached_candles", "extreme_target_reached_candle",
                            "target_reached_limit_found", "extreme_target_reached_mutual_candle",
                            "target_zone_mutuals", "target_zone_mutual_limit", "opposition_candle",
                            "further_oppositions", "extreme_opposition_candle", "extreme_opposition_reason",
                        ]
                        continuation_only_keys = [
                            "continuation_extreme_interceptor_candle",
                            "Continuation_extreme_Interceptor_limit",
                            "continuation_extreme_interceptor_mutual",
                            "touched_continuation_interceptors",
                            "FROM", "TO",
                        ]

                        # === Rebuild new_info ===
                        new_info = {
                            "line_id": info.get("line_id", line_id),
                            "final_sender_level": info.get("final_sender_level"),
                            "final_receiver_level": info.get("final_receiver_level"),
                        }
                        active_order = {"reason": "no active order"}

                        if direction == "continuation":
                            for key in breakout_only_keys:
                                info.pop(key, None)

                            enrich_if_needed(info, "continuation_extreme_interceptor_candle")
                            enrich_if_needed(info, "Continuation_extreme_Interceptor_limit")
                            if "continuation_extreme_interceptor_mutual" in info:
                                mutual_data = info["continuation_extreme_interceptor_mutual"]
                                if isinstance(mutual_data, dict) and isinstance(mutual_data.get("candle_number"), int):
                                    cn = mutual_data["candle_number"]
                                    if cn in candle_lookup:
                                        full = candle_lookup[cn]
                                        clean = {"candle_number": cn}
                                        for f in ESSENTIAL_FIELDS:
                                            clean[f] = full[f]
                                        info["continuation_extreme_interceptor_mutual"] = clean
                                        enriched_count += 1
                                else:
                                    info.pop("continuation_extreme_interceptor_mutual", None)
                            enrich_list_if_needed(info, "touched_continuation_interceptors")

                            # Continuation instant order logic
                            ext_cnum = info.get("continuation_extreme_interceptor_candle")
                            ext_candle = ext_cnum if isinstance(ext_cnum, dict) else candle_lookup.get(ext_cnum) if isinstance(ext_cnum, int) else None
                            ext_open = ext_candle.get("open") if ext_candle else None
                            mutual_candle = info.get("continuation_extreme_interceptor_mutual")
                            has_mutual = bool(mutual_candle)
                            mutual_open = mutual_candle.get("open") if isinstance(mutual_candle, dict) else None
                            limit_found = pre_bools["continuation_extreme_interceptor_limit_found"]
                            ext_found = pre_bools["continuation_extreme_interceptor_found"]

                            if ext_found and not has_mutual and not limit_found and ext_open is not None:
                                active_order = {
                                    "instant_entry": ext_open,
                                    "order_type": order_type,
                                    "fact": "continuation extreme interceptor is active",
                                    "order_from": "continuation_extreme_interceptor"
                                }
                            elif has_mutual and not limit_found and mutual_open is not None:
                                active_order = {
                                    "instant_entry": mutual_open,
                                    "order_type": order_type,
                                    "fact": "continuation extreme interceptor mutuals is now active",
                                    "order_from": "continuation_extreme_interceptor_mutuals"
                                }
                            elif limit_found:
                                active_order = {"reason": "no active order", "order_from": "none"}

                            new_info.update({
                                "valid": pre_bools["valid"],
                                "adapted": pre_bools["adapted"],
                                "continuation_extreme_interceptor_found": ext_found,
                                "continuation_extreme_interceptor_limit_found": limit_found,
                                "trendline_connection_type": trendline_connection_type,
                                "active_instant_order": active_order
                            })

                        else:  # breakout
                            for key in continuation_only_keys:
                                info.pop(key, None)

                            enrich_if_needed(info, "retest_candle")
                            enrich_if_needed(info, "target_zone_candle")
                            enrich_if_needed(info, "extreme_target_reached_candle")
                            enrich_if_needed(info, "target_reached_limit_found")
                            enrich_if_needed(info, "extreme_opposition_candle")
                            enrich_list_if_needed(info, "breakout_sequence_candles")
                            enrich_list_if_needed(info, "target_zone_mutuals")
                            enrich_list_if_needed(info, "extreme_target_reached_mutual_candle")

                            # Breakout instant order logic
                            if (pre_bools["valid"] or pre_bools["adapted"]) and order_type and pre_bools["retest_candle_found"] and pre_bools["target_zone_candle_found"]:
                                tz_candle = info.get("target_zone_candle")
                                tz_mutuals = info.get("target_zone_mutuals", [])
                                ext_candle = info.get("extreme_target_reached_candle")
                                ext_mutuals = info.get("extreme_target_reached_mutual_candle", [])
                                tz_open = tz_candle.get("open") if isinstance(tz_candle, dict) else None
                                latest_mutual_open = None
                                if tz_mutuals:
                                    sorted_mutuals = sorted(tz_mutuals, key=lambda x: x.get("candle_number", 0), reverse=True)
                                    latest_mutual = sorted_mutuals[0]
                                    latest_mutual_open = latest_mutual.get("open")
                                latest_ext_open = None
                                if ext_mutuals:
                                    sorted_ext = sorted(ext_mutuals, key=lambda x: x.get("candle_number", 0), reverse=True)
                                    latest_ext = sorted_ext[0]
                                    latest_ext_open = latest_ext.get("open")
                                ext_open = ext_candle.get("open") if isinstance(ext_candle, dict) else None

                                if (not pre_bools["target_zone_mutuals_found"] and
                                    not pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if tz_open is not None:
                                        active_order = {
                                            "instant_entry": tz_open,
                                            "order_type": order_type,
                                            "fact": "price retested and at target zone",
                                            "order_from": "target_zone"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    not pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if latest_mutual_open is not None:
                                        active_order = {
                                            "instant_entry": latest_mutual_open,
                                            "order_type": order_type,
                                            "fact": "price retested, target zone is no more active but using most target zone mutuals",
                                            "order_from": "target_zone_mutuals"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    not pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    active_order = {
                                        "reason": "target zone and its mutual limit has been reached no orders for now",
                                        "order_from": "none"
                                    }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    not pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if ext_open is not None:
                                        active_order = {
                                            "instant_entry": ext_open,
                                            "order_type": order_type,
                                            "fact": "target reached candle is active",
                                            "order_from": "target reached candle"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    not pre_bools["extreme_target_reached_limit_found"]):
                                    if latest_ext_open is not None:
                                        active_order = {
                                            "instant_entry": latest_ext_open,
                                            "order_type": order_type,
                                            "fact": "target reached candle mutual is active",
                                            "order_from": "target reached mutual candle"
                                        }
                                elif (pre_bools["target_zone_mutuals_found"] and
                                    pre_bools["target_zone_mutual_limit_found"] and
                                    pre_bools["extreme_target_reached_candle_found"] and
                                    pre_bools["extreme_target_reached_mutual_candle_found"] and
                                    pre_bools["extreme_target_reached_limit_found"]):
                                    active_order = {
                                        "reason": "no active order, all limit is reached",
                                        "order_from": "instant orders limit reached"
                                    }

                            new_info.update({
                                "valid": pre_bools["valid"],
                                "adapted": pre_bools["adapted"],
                                "extreme_opposition_candle_found": pre_bools["extreme_opposition_candle_found"],
                                "retest_candle_found": pre_bools["retest_candle_found"],
                                "target_zone_candle_found": pre_bools["target_zone_candle_found"],
                                "target_zone_mutuals_found": pre_bools["target_zone_mutuals_found"],
                                "target_zone_mutual_limit_found": pre_bools["target_zone_mutual_limit_found"],
                                "extreme_target_reached_candle_found": pre_bools["extreme_target_reached_candle_found"],
                                "extreme_target_reached_mutual_candle_found": pre_bools["extreme_target_reached_mutual_candle_found"],
                                "extreme_target_reached_limit_found": pre_bools["extreme_target_reached_limit_found"],
                                "extreme_horizontal_level_blocker_found": pre_bools["extreme_horizontal_level_blocker_found"],
                                "pending_entry_is_full_extension": pre_bools["pending_entry_is_full_extension"],
                                "trendline_connection_type": trendline_connection_type,
                                "active_instant_order": active_order
                            })

                        # === Copy remaining fields (safely) ===
                        for k, v in info.items():
                            if k not in new_info and k not in {"final_sender_level", "final_receiver_level", "valid", "adapted"}:
                                if k == "continuation_extreme_interceptor_mutual" and direction == "continuation" and v is not None:
                                    new_info[k] = v
                                elif k != "continuation_extreme_interceptor_mutual":
                                    new_info[k] = v

                        data["team"]["trendline_info"] = new_info

                    # === Enrich final_valid_trends_2 ===
                    for trend in final_valid_trends_2.values():
                        enrich_if_needed(trend, "sender_cnum")
                        enrich_if_needed(trend, "receiver_cnum")

                    log(f"FINAL ENRICHMENT COMPLETE → {enriched_count} fields enriched | Direction determined from continuation.json (breakout={is_breakout_strategy})", "SUCCESS")                               

                def categorize_developer_technique(
                    broker_raw_name,
                    symbol_folder,
                    tf_folder,
                    chart_path,
                    output_path,
                    technique_path_2,
                    report_path
                ):
                    import os
                    import json
                    import shutil
                    # Assuming log is already defined elsewhere

                    # === 1. Load configuration ===
                    strategy_name = None
                    continuation_extreme_name = None
                    categories = []
                    is_continuation_strategy = False

                    try:
                        with open(technique_path_2, 'r', encoding='utf-8') as f:
                            technique_conf = json.load(f)
                        
                        for key, value in technique_conf.items():
                            k = key.strip().upper()
                            if k == "STRATEGY_NAME" and isinstance(value, str):
                                strategy_name = value.strip()
                            elif k == "CONTINUATION_EXTREME_NAME" and isinstance(value, str):
                                continuation_extreme_name = value.strip()
                            elif k == "CATEGORIES_SPLIT" and isinstance(value, str):
                                categories = [cat.strip().lower().replace(" ", "_").replace("-", "_") 
                                            for cat in value.split(",") if cat.strip()]

                        # Check trendline directions
                        trendline_conf = technique_conf.get("trendline", {})
                        if isinstance(trendline_conf, dict):
                            for conf in trendline_conf.values():
                                if isinstance(conf, dict):
                                    direction = conf.get("DIRECTION", "").strip().lower()
                                    if direction == "continuation":
                                        is_continuation_strategy = True

                        # Fallback
                        if not strategy_name:
                            strategy_name = os.path.basename(os.path.dirname(technique_path_2))

                    except Exception as e:
                        log(f"Failed to load technique JSON: {e}", "ERROR")
                        return

                    # Skip if not continuation
                    if not is_continuation_strategy:
                        log(f"Technique is not a continuation strategy → skipping categorization", "INFO")
                        return

                    base_dev_path = r"C:\xampp\htdocs\chronedge\synarex\chart\developers"
                    broker_folder = os.path.join(base_dev_path, broker_raw_name)

                    # Safe names for folders
                    continuation_extreme_safe = (continuation_extreme_name or "Instant_Continuation").strip().replace(" ", "_")

                    safe_symbol = symbol_folder.replace("/", "_").replace("\\", "_").replace(":", "_")
                    timeframe_filename = f"{tf_folder}.png"

                    # === Chart saving ===
                    def save_chart_to_strategy(main_strategy_name):
                        if not os.path.exists(output_path):
                            log(f"Chart not found, skipped copy → {output_path}", "WARNING")
                            return
                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        os.makedirs(strategy_folder, exist_ok=True)
                        chart_base_dir = os.path.join(strategy_folder, "chart")
                        os.makedirs(chart_base_dir, exist_ok=True)
                        market_chart_dir = os.path.join(chart_base_dir, safe_symbol)
                        os.makedirs(market_chart_dir, exist_ok=True)
                        destination_chart_path = os.path.join(market_chart_dir, timeframe_filename)
                        try:
                            shutil.copy2(output_path, destination_chart_path)
                            log(f"Chart saved → {destination_chart_path} ({main_strategy_name})", "INFO")
                        except Exception as e:
                            log(f"Failed to copy chart to {main_strategy_name}: {e}", "ERROR")

                    save_chart_to_strategy(strategy_name)

                    # === Load report JSON ===
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            custom_data = json.load(f)
                    except Exception as e:
                        log(f"Failed to read custom JSON: {e}", "ERROR")
                        return

                    # === Extract real orders ===
                    instant_continuation_extreme_entries = []  # Instant extreme (continuation)

                    for line_id, entry in custom_data.items():
                        info = entry.get("team", {}).get("trendline_info", {})
                        if not info.get("valid", False):
                            continue

                        # Instant Logic — only collect extreme for continuation
                        active_order = info.get("active_instant_order", {})
                        if isinstance(active_order, dict) and active_order.get("reason") != "no active order":
                            order_from = active_order.get("order_from")
                            record = {
                                "market_name": symbol_folder,
                                "timeframe": tf_folder,
                                "entry_price": active_order.get("instant_entry"),
                                "order_type": "buy" if "buy" in str(active_order.get("order_type")).lower() else "sell",
                                "candle_number": None,
                                "candletimestamp": None,
                                "fact": active_order.get("fact", ""),
                                "exit_price": active_order.get("exit_price")
                            }
                            if order_from in {"extreme interceptor", "extreme_interceptor_mutual"}:
                                instant_continuation_extreme_entries.append(record)

                    # === Save aggregation helper ===
                    def save_aggregation(main_strategy_name, subfolder_name, new_entries, allow_coords=True):
                        strategy_folder = os.path.join(broker_folder, main_strategy_name)
                        subfolder = os.path.join(strategy_folder, subfolder_name)
                        os.makedirs(subfolder, exist_ok=True)

                        for cat in categories:
                            if cat == "co_ords" and not allow_coords:
                                continue

                            cat_folder = os.path.join(subfolder, cat)
                            os.makedirs(cat_folder, exist_ok=True)
                            full_path = os.path.join(cat_folder, f"{subfolder_name}.json")

                            try:
                                if os.path.exists(full_path):
                                    with open(full_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        orders = data.get("orders", [])
                                else:
                                    orders = []
                            except:
                                orders = []

                            orders = [o for o in orders if not (o.get("market_name") == symbol_folder and o.get("timeframe") == tf_folder)]
                            orders = [o for o in orders if o.get("market_name") != "none"]

                            processed_new_entries = []
                            for entry in new_entries:
                                entry_copy = entry.copy()
                                if cat != "co_ords":
                                    entry_copy.pop("exit_price", None)
                                processed_new_entries.append(entry_copy)

                            orders.extend(processed_new_entries)

                            real_orders = [o for o in orders if o.get("market_name") not in {"none", None}]
                            result = {
                                "total_orders": len(real_orders),
                                "total_markets": len({o["market_name"] for o in real_orders}),
                                "orders": orders
                            }

                            with open(full_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)

                    # === Execution (continuation only) ===
                    save_aggregation(strategy_name, continuation_extreme_safe, instant_continuation_extreme_entries, allow_coords=False)

                    # === Final Placeholder Logic ===
                    total_real = len(instant_continuation_extreme_entries)

                    placeholder = [{
                        "market_name": "none", "timeframe": "none", "entry_price": None,
                        "order_type": "none", "candle_number": None, "candletimestamp": None, "exit_price": None
                    }]

                    if total_real == 0:
                        save_aggregation(strategy_name, continuation_extreme_safe, placeholder, allow_coords=False)

                    log(f"Continuation categorization complete for {symbol_folder}/{tf_folder}", "SUCCESS")
  

                def trend_direction():
                    # 1. Initialize the central data store
                    final_valid_trends_2 = {}
                    for trend in final_trendlines_for_redraw:
                        direction = trend["direction"]
                        if direction == "breakout":
                            final_valid_trends_2 = OPPOSITION_BREAKOUT_INTERCEPTORS_RETEST_TARGET_ZONE(final_valid_trends_2)
                        elif direction == "continuation":
                            final_valid_trends_2 = CONTINUATION_EXTREME_INTERCEPTOR(final_valid_trends_2)
                        else:
                            log(f"Unknown direction '{direction}' for trend {trend['line_id']} - skipping processing", "WARNING")

                    # 3. Draw horizontal levels (uses final_valid_trends_2)
                    DRAW_ALL_POINTS_LEVELS(final_valid_trends_2)

                    # ==== ENRICH CUSTOM JSON WITH FULL CANDLE DETAILS ====
                    enrich_candle_details(candles, final_teams_2, final_valid_trends_2, report_path, technique_path_2)

                    # Final save after processing all trends (local)
                    cv2.imwrite(output_path, img)
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(final_teams_2, f, indent=2, ensure_ascii=False, default=str)
                    log(f"{symbol_folder}/{tf_folder} | {len(final_teams_2)} Trendlines processed & enriched with full candle details", "SUCCESS")

                    # ===== NEW: SAVE TO CENTRAL DEVELOPER STRATEGY FOLDER =====
                    # We need broker_raw_name and the technique_path_2 from outer scope
                    # These variables are available in the main loop context
                    categorize_developer_technique(
                        broker_raw_name=broker_raw_name,
                        symbol_folder=symbol_folder,
                        tf_folder=tf_folder,
                        chart_path=chart_path,
                        output_path=output_path,
                        technique_path_2=technique_path_2,
                        report_path=report_path
                    )

                trend_direction()
                

    log("=== INSTITUTIONAL TRENDLINE ENGINE v10.1 — DYNAMIC conf_2IGURATION & TARGET ZONE ADDED ===", "SUCCESS")



if session:
    custom_breakout()
    custom_continuation()






import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ==============================
# 1. FOREX
# ==============================
def calculate_forex_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\forexvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"

    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd",
        1.0: "risk_1_usd",
        2.0: "risk_2_usd",
        3.0: "risk_3_usd",
        4.0: "risk_4_usd",
        8.0: "risk_8_usd",
        16.0: "risk_16_usd"
    }

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    for section in data.values():
        if not isinstance(section, list):
            continue
        for entry in section:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)

    print(f"Loaded forex orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 1e-5))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"Invalid limit type {limit_type}", "WARNING")
                continue

            digits = 5 if tick_size == 1e-5 else 3
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "forexcalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"Forex SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 2. SYNTHETICS
# ==============================
def calculate_synthetics_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\syntheticsvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Synthetics] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Synthetics] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Synthetics] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Synthetics] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Synthetics][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "syntheticscalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Synthetics][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Synthetics][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Synthetics] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 3. CRYPTO
# ==============================
def calculate_crypto_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\cryptovolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Crypto] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Crypto] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Crypto] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Crypto] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Crypto][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "cryptocalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Crypto][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Crypto][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Crypto] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 4. BASKET INDICES
# ==============================
def calculate_basketindices_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\basketindicesvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[BasketIndices] Unexpected JSON structure", "ERROR")
        return False

    print(f"[BasketIndices] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[BasketIndices] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[BasketIndices] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[BasketIndices][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "basketindicescalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[BasketIndices][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[BasketIndices][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[BasketIndices] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 5. INDICES
# ==============================
def calculate_indices_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\indicesvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Indices] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Indices] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Indices] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Indices] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Indices][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "indicescalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Indices][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Indices][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Indices] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 6. METALS
# ==============================
def calculate_metals_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\metalsvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Metals] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Metals] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Metals] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Metals] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Metals][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "metalscalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Metals][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Metals][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Metals] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 7. STOCKS
# ==============================
def calculate_stocks_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\stocksvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Stocks] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Stocks] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Stocks] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Stocks] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Stocks][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "stockscalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Stocks][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Stocks][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Stocks] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 8. ETFs
# ==============================
def calculate_etfs_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\etfsvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[ETFs] Unexpected JSON structure", "ERROR")
        return False

    print(f"[ETFs] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[ETFs] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[ETFs] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[ETFs][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "etfscalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[ETFs][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[ETFs][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[ETFs] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 9. EQUITIES
# ==============================
def calculate_equities_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\equitiesvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Equities] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Equities] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Equities] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Equities] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Equities][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "equitiescalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Equities][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Equities][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Equities] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 10. ENERGIES
# ==============================
def calculate_energies_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\energiesvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Energies] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Energies] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Energies] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Energies] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Energies][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "energiescalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Energies][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Energies][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Energies] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# 11. COMMODITIES
# ==============================
def calculate_commodities_sl_tp_markets():
    INPUT_JSON = r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\commoditiesvolumesandrisk.json"
    BASE_OUTPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}

    in_file = Path(INPUT_JSON)
    if not in_file.is_file():
        print(f"INPUT FILE NOT FOUND: {INPUT_JSON}", "ERROR")
        return False

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                for entry in section:
                    broker = entry.get("broker", "unknown")
                    risk_usd = float(entry.get("riskusd_amount", 0))
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            risk_usd = float(entry.get("riskusd_amount", 0))
            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)
    else:
        print("[Commodities] Unexpected JSON structure", "ERROR")
        return False

    print(f"[Commodities] Loaded orders by broker & risk", "INFO")

    saved = 0
    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}
        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                print(f"[Commodities] No orders for risk ${risk_usd} in {broker}", "WARNING")
                continue

            entry = risk_orders[0]
            market = entry.get("market")
            limit_type = entry["limit_order"]
            entry_price = float(entry["entry_price"])
            volume = float(entry["volume"])
            tick_value = float(entry["tick_value"])
            tick_size = float(entry.get("tick_size", 0.01))

            pip_size = 10 * tick_size
            pip_value_usd = tick_value * volume * (pip_size / tick_size)
            sl_pips = risk_usd / pip_value_usd
            tp_pips = sl_pips * 3

            if limit_type == "buy_limit":
                sl_price = entry_price - (sl_pips * pip_size)
                tp_price = entry_price + (tp_pips * pip_size)
            elif limit_type == "sell_limit":
                sl_price = entry_price + (sl_pips * pip_size)
                tp_price = entry_price - (tp_pips * pip_size)
            else:
                print(f"[Commodities] Invalid limit type {limit_type}", "WARNING")
                continue

            digits = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            digits = max(digits, 2)
            sl_price = round(sl_price, digits)
            tp_price = round(tp_price, digits)

            calc_entry = {
                "market": market,
                "limit_order": limit_type,
                "timeframe": entry.get("timeframe", ""),
                "entry_price": entry_price,
                "volume": volume,
                "riskusd_amount": risk_usd,
                "sl_price": sl_price,
                "sl_pips": round(sl_pips, 2),
                "tp_price": tp_price,
                "tp_pips": round(tp_pips, 2),
                "rr_ratio": 3.0,
                "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selection_criteria": "first_valid_order",
                "broker": broker
            }
            results_by_risk[risk_usd].append(calc_entry)

            print(
                f"[Commodities][{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                f"SL {sl_price} ({calc_entry['sl_pips']} pips) | TP {tp_price} ({calc_entry['tp_pips']} pips)",
                "INFO",
            )

        for risk_usd, calc_list in results_by_risk.items():
            if not calc_list:
                continue
            broker_dir = Path(BASE_OUTPUT_DIR) / broker / RISK_FOLDERS[risk_usd]
            broker_dir.mkdir(parents=True, exist_ok=True)
            out_file = broker_dir / "commoditiescalculatedprices.json"
            try:
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(calc_list, f, indent=2)
                saved += len(calc_list)
                print(f"[Commodities][{broker}] Saved {len(calc_list)} calc(s) for risk ${risk_usd}", "SUCCESS")
            except Exception as e:
                print(f"[Commodities][{broker}] Failed to save for risk ${risk_usd}: {e}", "ERROR")

    print(f"[Commodities] SL/TP calculations done – {saved} entries saved.", "SUCCESS")
    return True


# ==============================
# PROMOTION FUNCTION
# ==============================
def scale_lowerorders_proportionally():
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices"
    RISK_LEVELS = [0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0]
    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd",
        3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"
    }
    ASSET_CLASSES = {
        "forex": "forexcalculatedprices.json",
        "synthetics": "syntheticscalculatedprices.json",
        "crypto": "cryptocalculatedprices.json",
        "basketindices": "basketindicescalculatedprices.json",
        "indices": "indicescalculatedprices.json",
        "metals": "metalscalculatedprices.json",
        "stocks": "stockscalculatedprices.json",
        "etfs": "etfscalculatedprices.json",
        "equities": "equitiescalculatedprices.json",
        "energies": "energiescalculatedprices.json",
        "commodities": "commoditiescalculatedprices.json",
    }

    total_promoted = 0

    # Loop over each broker
    for broker_dir in Path(BASE_INPUT_DIR).iterdir():
        if not broker_dir.is_dir():
            continue
        broker = broker_dir.name
        print(f"\n[Promoter] Processing broker: {broker}", "INFO")

        # For this broker, collect lowest-risk entry per asset
        seed_entries = {}  # asset → (risk, entry)

        for asset, filename in ASSET_CLASSES.items():
            lowest_risk = None
            seed_entry = None

            for risk in RISK_LEVELS:
                file_path = broker_dir / RISK_FOLDERS[risk] / filename
                if not file_path.is_file():
                    continue
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data:
                            entry = data[0]  # Use first entry as representative
                            if lowest_risk is None or risk < lowest_risk:
                                lowest_risk = risk
                                seed_entry = entry
                except Exception as e:
                    print(f"[Promoter] Failed to read {file_path}: {e}", "ERROR")

            if seed_entry:
                seed_entries[asset] = (lowest_risk, seed_entry)
                print(f"[Promoter] {asset.upper()} seed: ${lowest_risk} ({seed_entry['market']})", "INFO")

        if not seed_entries:
            print(f"[Promoter] No seed data for {broker}", "WARNING")
            continue

        # Now promote from lowest to all higher risks (per asset)
        for asset, (base_risk, base_entry) in seed_entries.items():
            filename = ASSET_CLASSES[asset]

            for target_risk in RISK_LEVELS:
                if target_risk <= base_risk:
                    continue

                scale_factor = target_risk / base_risk
                new_volume = base_entry["volume"] * scale_factor

                promoted_entry = {
                    "market": base_entry["market"],
                    "limit_order": base_entry["limit_order"],
                    "timeframe": base_entry.get("timeframe", ""),
                    "entry_price": base_entry["entry_price"],
                    "volume": round(new_volume, 8),
                    "riskusd_amount": target_risk,
                    "sl_price": base_entry["sl_price"],
                    "sl_pips": base_entry["sl_pips"],
                    "tp_price": base_entry["tp_price"],
                    "tp_pips": base_entry["tp_pips"],
                    "rr_ratio": 3.0,
                    "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "selection_criteria": f"promoted_from_${base_risk}_scaled_x{scale_factor}",
                    "broker": broker
                }

                target_file = broker_dir / RISK_FOLDERS[target_risk] / filename
                existing_data = []
                if target_file.is_file():
                    try:
                        with target_file.open("r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = []

                # Avoid duplicates
                already_exists = any(
                    e.get("selection_criteria", "").startswith(f"promoted_from_${base_risk}")
                    and e["market"] == promoted_entry["market"]
                    for e in existing_data
                )
                if already_exists:
                    continue

                existing_data.append(promoted_entry)
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    with target_file.open("w", encoding="utf-8") as f:
                        json.dump(existing_data, f, indent=2)
                    print(f"[Promoter] {asset.upper()}: ${base_risk}→${target_risk} (×{scale_factor:.1f}) in {broker}", "SUCCESS")
                    total_promoted += 1
                except Exception as e:
                    print(f"[Promoter] Save failed {target_file}: {e}", "ERROR")

    print(f"\n[Promoter] Promotion complete – {total_promoted} entries promoted.", "SUCCESS")
    return True

    

# ==============================
# CATEGORISE STRATEGY
# ==============================
def categorise_strategy():
    BASE_DIR = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices")
    RISK_FOLDERS = {0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd", 3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"}
    CALC_FILES = {
        "forex": "forexcalculatedprices.json",
        "synthetics": "syntheticscalculatedprices.json",
        "crypto": "cryptocalculatedprices.json",
        "basketindices": "basketindicescalculatedprices.json",
        "indices": "indicescalculatedprices.json",
        "metals": "metalscalculatedprices.json",
        "stocks": "stockscalculatedprices.json",
        "etfs": "etfscalculatedprices.json",
        "equities": "equitiescalculatedprices.json",
        "energies": "energiescalculatedprices.json",
        "commodities": "commoditiescalculatedprices.json",
    }
    TIMEFRAME_ORDER = ["4h", "1h", "30m", "15m", "5m"]

    total_high = 0
    total_low  = 0

    for broker_dir in BASE_DIR.iterdir():
        if not broker_dir.is_dir():
            continue
        broker = broker_dir.name

        for risk_usd, folder_name in RISK_FOLDERS.items():
            folder_path = broker_dir / folder_name
            if not folder_path.is_dir():
                continue

            symbol_data = defaultdict(list)
            source_map  = defaultdict(list)

            for source, fname in CALC_FILES.items():
                fpath = folder_path / fname
                if not fpath.is_file():
                    continue
                try:
                    with fpath.open("r", encoding="utf-8") as f:
                        entries = json.load(f)
                    for e in entries:
                        market = e["market"]
                        symbol_data[market].append(e)
                        if fname not in source_map[market]:
                            source_map[market].append(fname)
                except Exception as exc:
                    print(f"[Categorise][{broker}] Error reading {fpath}: {exc}", "ERROR")

            if not symbol_data:
                continue

            def price_distance(e1, e2):
                if e1["limit_order"] == "buy_limit":
                    return abs(e1["sl_price"] - e2["entry_price"])
                else:
                    return abs(e2["sl_price"] - e1["entry_price"])

            # HIGH-TO-LOW
            hightolow_entries = []
            market_sources = defaultdict(set)
            for market, all_entries in symbol_data.items():
                tf_groups = defaultdict(list)
                for e in all_entries:
                    tf = e.get("timeframe", "").strip()
                    if tf not in TIMEFRAME_ORDER:
                        tf = "unknown"
                    tf_groups[tf].append(e)

                buy = sell = None
                for tf in TIMEFRAME_ORDER:
                    if tf not in tf_groups:
                        continue
                    candidates = tf_groups[tf]
                    buys  = [c for c in candidates if c["limit_order"] == "buy_limit"]
                    sells = [c for c in candidates if c["limit_order"] == "sell_limit"]

                    if buys:
                        best = min(buys, key=lambda x: x["entry_price"])
                        if buy is None or best["entry_price"] < buy["entry_price"]:
                            buy = best
                    if sells:
                        best = max(sells, key=lambda x: x["entry_price"])
                        if sell is None or best["entry_price"] > sell["entry_price"]:
                            sell = best

                    if buy and sell:
                        tick_sz = buy.get("tick_size", 1e-5)
                        pip_sz = 10 * tick_sz
                        required = 3 * min(buy["sl_pips"], sell["sl_pips"]) * pip_sz
                        dist = price_distance(buy, sell)
                        if dist >= required:
                            hightolow_entries.extend([buy, sell])
                            market_sources[market].update(source_map[market])
                            break
                        else:
                            if buy["calculated_at"] <= sell["calculated_at"]:
                                sell = None
                            else:
                                buy = None

                if buy and not sell:
                    hightolow_entries.append(buy)
                    market_sources[market].update(source_map[market])
                if sell and not buy:
                    hightolow_entries.append(sell)
                    market_sources[market].update(source_map[market])

            # LOW-TO-HIGH
            lowtohigh_entries = []
            market_sources_low = defaultdict(set)
            for market, all_entries in symbol_data.items():
                tf_groups = defaultdict(list)
                for e in all_entries:
                    tf = e.get("timeframe", "").strip()
                    if tf not in TIMEFRAME_ORDER:
                        tf = "unknown"
                    tf_groups[tf].append(e)

                buy = sell = None
                for tf in TIMEFRAME_ORDER:
                    if tf not in tf_groups:
                        continue
                    candidates = tf_groups[tf]
                    buys  = [c for c in candidates if c["limit_order"] == "buy_limit"]
                    sells = [c for c in candidates if c["limit_order"] == "sell_limit"]

                    if buys:
                        best = max(buys, key=lambda x: x["entry_price"])
                        if buy is None or best["entry_price"] > buy["entry_price"]:
                            buy = best
                    if sells:
                        best = min(sells, key=lambda x: x["entry_price"])
                        if sell is None or best["entry_price"] < sell["entry_price"]:
                            sell = best

                    if buy and sell:
                        tick_sz = buy.get("tick_size", 1e-5)
                        pip_sz = 10 * tick_sz
                        required = 3 * min(buy["sl_pips"], sell["sl_pips"]) * pip_sz
                        dist = price_distance(buy, sell)
                        if dist >= required:
                            lowtohigh_entries.extend([buy, sell])
                            market_sources_low[market].update(source_map[market])
                            break
                        else:
                            if buy["calculated_at"] <= sell["calculated_at"]:
                                sell = None
                            else:
                                buy = None

                if buy and not sell:
                    lowtohigh_entries.append(buy)
                    market_sources_low[market].update(source_map[market])
                if sell and not buy:
                    lowtohigh_entries.append(sell)
                    market_sources_low[market].update(source_map[market])

            # Summary
            def build_summary(market_sources_dict):
                counts = {"allmarketssymbols": len(market_sources_dict)}
                for src in CALC_FILES.keys():
                    key = f"{src}symbols"
                    counts[key] = sum(1 for srcs in market_sources_dict.values() if CALC_FILES[src] in srcs)
                return counts

            summary_high = build_summary(market_sources)
            summary_low  = build_summary(market_sources_low)

            # Write
            out_folder = folder_path
            out_folder.mkdir(parents=True, exist_ok=True)

            high_path = out_folder / "hightolow.json"
            try:
                with high_path.open("w", encoding="utf-8") as f:
                    json.dump({"summary": summary_high, "entries": hightolow_entries}, f, indent=2)
                print(f"[Categorise][{broker}] ${risk_usd} hightolow.json → {len(hightolow_entries)} entries", "SUCCESS")
                total_high += len(hightolow_entries)
            except Exception as e:
                print(f"[Categorise][{broker}] Failed hightolow.json ${risk_usd}: {e}", "ERROR")

            low_path = out_folder / "lowtohigh.json"
            try:
                with low_path.open("w", encoding="utf-8") as f:
                    json.dump({"summary": summary_low, "entries": lowtohigh_entries}, f, indent=2)
                print(f"[Categorise][{broker}] ${risk_usd} lowtohigh.json → {len(lowtohigh_entries)} entries", "SUCCESS")
                total_low += len(lowtohigh_entries)
            except Exception as e:
                print(f"[Categorise][{broker}] Failed lowtohigh.json ${risk_usd}: {e}", "ERROR")

    print(f"\n[Categorise] Strategy categorisation complete – "
          f"{total_high} HIGH→LOW entries | {total_low} LOW→HIGH entries across all brokers & risks.", "SUCCESS")
    return True


# ==============================
# MAIN EXECUTION BLOCK
# ==============================
def main():
    print("\n" + "="*60, "HEADER")
    print("SL/TP CALCULATOR + PROMOTER + STRATEGY CATEGORISER", "HEADER")
    print("Starting full pipeline...", "INFO")
    print("="*60 + "\n", "HEADER")

    # Phase 1: Calculate SL/TP for all asset classes
    print("PHASE 1: Calculating SL/TP per broker & risk level...", "PHASE")
    calculations = [
        calculate_forex_sl_tp_markets,
        calculate_synthetics_sl_tp_markets,
        calculate_crypto_sl_tp_markets,
        calculate_basketindices_sl_tp_markets,
        calculate_indices_sl_tp_markets,
        calculate_metals_sl_tp_markets,
        calculate_stocks_sl_tp_markets,
        calculate_etfs_sl_tp_markets,
        calculate_equities_sl_tp_markets,
        calculate_energies_sl_tp_markets,
        calculate_commodities_sl_tp_markets,
    ]

    calc_success = 0
    calc_failed = 0

    for calc_func in calculations:
        try:
            if calc_func():
                calc_success += 1
            else:
                calc_failed += 1
                asset = calc_func.__name__.replace("calculate_", "").replace("_sl_tp_markets", "").upper()
                print(f"[{asset}] Failed (check logs above)", "ERROR")
        except FileNotFoundError:
            asset = calc_func.__name__.replace("calculate_", "").replace("_sl_tp_markets", "").upper()
            print(f"[{asset}] Input file missing — SKIPPING", "SKIP")
            calc_failed += 1
        except Exception as e:
            asset = calc_func.__name__.replace("calculate_", "").replace("_sl_tp_markets", "").upper()
            print(f"[{asset}] Unexpected error: {e}", "ERROR")
            calc_failed += 1

    if calc_success == 0:
        print(f"No calculations succeeded. Aborting.", "FATAL")
        return False

    print(f"\n{calc_success} calculation(s) succeeded, {calc_failed} skipped/failed. Continuing...\n", "INFO")

    # Phase 2: Promote lower risk orders
    print("PHASE 2: Promoting lower-risk orders proportionally...", "PHASE")
    if not scale_lowerorders_proportionally():
        print("Promotion phase failed.", "ERROR")
        return False
    print("Promotion phase completed.\n", "SUCCESS")

    # Phase 3: Categorise strategies
    print("PHASE 3: Categorising strategies (HIGH→LOW / LOW→HIGH)...", "PHASE")
    if not categorise_strategy():
        print("Strategy categorisation failed.", "ERROR")
        return False
    print("Strategy categorisation completed.\n", "SUCCESS")

    print("="*60, "FOOTER")
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!", "SUCCESS")
    print("="*60 + "\n", "FOOTER")
    return True

# ==============================
# SCRIPT ENTRY POINT
# ==============================
if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("Pipeline failed at some stage.", "ERROR")
            exit(1)
    except KeyboardInterrupt:
        print("\nScript interrupted by user.", "INTERRUPT")
        exit(130)
    except Exception as e:
        print(f"Unexpected error in main: {e}", "CRITICAL")
        exit(1)
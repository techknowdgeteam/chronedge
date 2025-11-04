import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict



def symbolsorderfiltering():
    """
    Filters ALL order data based on allowedmarkets.json:
    - volumesandrisk.json (input)
    - calculatedprices.json (output)
    - hightolow.json / lowtohigh.json (categorized strategies)

    Rules:
      - limited: true + allowed list → keep only listed markets
      - limited: true + empty list → delete ALL for that market
      - limited: false → keep all

    Returns True on success.
    """
    from pathlib import Path
    import json

    # ------------------------------------------------------------------
    # PATHS & CONFIG
    # ------------------------------------------------------------------
    ALLOWED_MARKETS_PATH = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points\symbols\allowedmarkets.json")
    INPUT_ROOT = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points")
    OUTPUT_ROOT = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices")

    INPUT_FILES = [
        "forexvolumesandrisk.json", "syntheticsvolumesandrisk.json", "cryptovolumesandrisk.json",
        "basketindicesvolumesandrisk.json", "indicesvolumesandrisk.json", "metalsvolumesandrisk.json",
        "stocksvolumesandrisk.json", "etfsvolumesandrisk.json", "equitiesvolumesandrisk.json",
        "energiesvolumesandrisk.json", "commoditiesvolumesandrisk.json",
    ]

    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd",
        3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"
    }

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

    CATEGORY_FILES = ["hightolow.json", "lowtohigh.json"]
    FILENAME_TO_MARKET = {v: k for k, v in CALC_FILES.items()}

    total_removed = 0
    total_files   = 0

    # ------------------------------------------------------------------
    # 1. Load allowed markets
    # ------------------------------------------------------------------
    if not ALLOWED_MARKETS_PATH.is_file():
        return False
    try:
        with ALLOWED_MARKETS_PATH.open("r", encoding="utf-8") as f:
            allowed_config = json.load(f)
    except Exception:
        return False

    market_rules = {}
    for market_key, cfg in allowed_config.items():
        limited = cfg.get("limited", False)
        allowed = set(cfg.get("allowed", []))
        market_rules[market_key] = (limited, allowed)

    # ------------------------------------------------------------------
    # Helper: print header + rules
    # ------------------------------------------------------------------
    print("\n" + "="*90)
    print("SYMBOLS ORDER FILTERING (INPUT + OUTPUT + CATEGORIZED)".center(90))
    print("="*90 + "\n")

    for market_key, (limited, allowed) in market_rules.items():
        status = "DELETE ALL" if limited and not allowed else \
                 f"KEEP {len(allowed)}" if limited else "KEEP ALL"
        print(f"[RULE] {market_key.upper():12} → {status}")

    # ------------------------------------------------------------------
    # Helper: filter list/dict by market
    # ------------------------------------------------------------------
    def filter_market_data(data, market_key):
        nonlocal total_removed
        limited, allowed = market_rules.get(market_key, (False, set()))
        if not limited:
            return 0

        removed = 0
        if isinstance(data, dict):
            for key in list(data.keys()):
                if not isinstance(data[key], list):
                    continue
                orig = len(data[key])
                if not allowed:
                    data[key][:] = []
                else:
                    data[key][:] = [e for e in data[key] if e.get("market") in allowed]
                removed += orig - len(data[key])
        elif isinstance(data, list):
            orig = len(data)
            if not allowed:
                data[:] = []
            else:
                data[:] = [e for e in data if e.get("market") in allowed]
            removed = orig - len(data)
        total_removed += removed
        return removed

    # ------------------------------------------------------------------
    # Helper: rebuild summary for categorized files
    # ------------------------------------------------------------------
    def rebuild_summary(entries, source_map):
        markets = {e["market"] for e in entries}
        counts = {"allmarketssymbols": len(markets)}
        for src in CALC_FILES.keys():
            key = f"{src}symbols"
            counts[key] = sum(1 for m in markets if any(CALC_FILES[src] in s for s in source_map.get(m, [])))
        return counts

    # ------------------------------------------------------------------
    # 2. FILTER INPUT FILES (silent)
    # ------------------------------------------------------------------
    for fname in INPUT_FILES:
        market_key = fname.split("volumesandrisk")[0].rstrip("_")
        if market_key not in market_rules:
            continue
        fpath = INPUT_ROOT / fname
        if not fpath.is_file():
            continue
        try:
            with fpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        filter_market_data(data, market_key)
        total_files += 1
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # 3. FILTER OUTPUT & CATEGORIZED FILES (silent)
    # ------------------------------------------------------------------
    for broker_dir in OUTPUT_ROOT.iterdir():
        if not broker_dir.is_dir():
            continue
        for risk, folder in RISK_FOLDERS.items():
            risk_dir = broker_dir / folder
            if not risk_dir.is_dir():
                continue

            # ---- source_map -------------------------------------------------
            source_map = {}
            for calc_fname in CALC_FILES.values():
                p = risk_dir / calc_fname
                if not p.is_file():
                    continue
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    for e in (data if isinstance(data, list) else data.values()):
                        if isinstance(e, dict) and "market" in e:
                            m = e["market"]
                            source_map.setdefault(m, []).append(calc_fname)
                except Exception:
                    pass

            # ---- calculatedprices -------------------------------------------
            for market_key, calc_fname in CALC_FILES.items():
                p = risk_dir / calc_fname
                if not p.is_file():
                    continue
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                filter_market_data(data, market_key)
                total_files += 1
                with p.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            # ---- hightolow / lowtohigh --------------------------------------
            for cat_file in CATEGORY_FILES:
                p = risk_dir / cat_file
                if not p.is_file():
                    continue
                try:
                    with p.open("r", encoding="utf-8") as f:
                        content = json.load(f)
                    entries = content.get("entries", [])
                    if not isinstance(entries, list):
                        continue
                    old = len(entries)
                except Exception:
                    continue

                filtered = []
                for e in entries:
                    market = e.get("market")
                    if not market:
                        continue

                    # market category from source_map
                    market_key = None
                    for src in source_map.get(market, []):
                        market_key = FILENAME_TO_MARKET.get(src)
                        if market_key:
                            break
                    if not market_key:
                        # fallback – any calc file in the folder
                        for cf in CALC_FILES.values():
                            if (risk_dir / cf).is_file():
                                market_key = FILENAME_TO_MARKET.get(cf)
                                break
                    if not market_key:
                        filtered.append(e)
                        continue

                    limited, allowed = market_rules.get(market_key, (False, set()))
                    if not limited or (allowed and market in allowed):
                        filtered.append(e)

                # rebuild summary
                content["entries"] = filtered
                content["summary"] = rebuild_summary(filtered, source_map)

                with p.open("w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2)

                total_removed += old - len(filtered)
                total_files   += 1

    # ------------------------------------------------------------------
    # FINAL REPORT (only what you asked for)
    # ------------------------------------------------------------------
    print("\n" + "="*90)
    print(f"SYMBOLS FILTERING COMPLETE")
    print(f"   • {total_removed:,} entries removed")
    print(f"   • {total_files} files processed (input + output + categorized)")
    print(f"   • Rules: {ALLOWED_MARKETS_PATH.name}")
    print("="*90 + "\n")
    return True 


def clean_5m_timeframes():
    """
    1. Scans every *input* file (volumesandrisk.json) and deletes entries
       where timeframe == "5m".
    2. Scans every *output* file (calculatedprices.json) and deletes the
       same entries.
    Returns True on success, False if any step fails.
    """
    from pathlib import Path
    import json
    from datetime import datetime

    # ------------------------------------------------------------------
    # 1. INPUT FILES (volumesandrisk)
    # ------------------------------------------------------------------
    INPUT_ROOT = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_volumes_points")
    INPUT_FILES = [
        "forexvolumesandrisk.json",
        "syntheticsvolumesandrisk.json",
        "cryptovolumesandrisk.json",
        "basketindicesvolumesandrisk.json",
        "indicesvolumesandrisk.json",
        "metalsvolumesandrisk.json",
        "stocksvolumesandrisk.json",
        "etfsvolumesandrisk.json",
        "equitiesvolumesandrisk.json",
        "energiesvolumesandrisk.json",
        "commoditiesvolumesandrisk.json",
    ]

    # ------------------------------------------------------------------
    # 2. OUTPUT FILES (calculatedprices)
    # ------------------------------------------------------------------
    OUTPUT_ROOT = Path(r"C:\xampp\htdocs\chronedge\chart\symbols_calculated_prices")
    RISK_FOLDERS = {
        0.5: "risk_0_50cent_usd", 1.0: "risk_1_usd", 2.0: "risk_2_usd",
        3.0: "risk_3_usd", 4.0: "risk_4_usd", 8.0: "risk_8_usd", 16.0: "risk_16_usd"
    }
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

    total_removed = 0
    total_files = 0

    print("\n" + "="*70, "HEADER")
    print("CLEANING 5m TIMEFRAMES – INPUTS & OUTPUTS", "HEADER")
    print("="*70 + "\n", "HEADER")

    # ------------------------------------------------------------------
    # Helper: filter a list / dict-of-lists and return removed count
    # ------------------------------------------------------------------
    def filter_5m(data):
        removed = 0
        if isinstance(data, dict):
            for key in list(data.keys()):
                if not isinstance(data[key], list):
                    continue
                original_len = len(data[key])
                data[key] = [
                    e for e in data[key]
                    if str(e.get("timeframe", "")).strip().lower() != "5m"
                ]
                removed += original_len - len(data[key])
        elif isinstance(data, list):
            original_len = len(data)
            data[:] = [
                e for e in data
                if str(e.get("timeframe", "")).strip().lower() != "5m"
            ]
            removed = original_len - len(data)
        return removed

    # ------------------------------------------------------------------
    # 1. CLEAN INPUTS
    # ------------------------------------------------------------------
    print("PHASE 1 – Cleaning INPUT files …", "PHASE")
    for fname in INPUT_FILES:
        fpath = INPUT_ROOT / fname
        if not fpath.is_file():
            print(f"[INPUT] SKIP (missing): {fname}", "SKIP")
            continue

        try:
            with fpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[INPUT] READ ERROR {fname}: {e}", "ERROR")
            continue

        removed = filter_5m(data)
        total_removed += removed
        total_files += 1

        if removed:
            try:
                with fpath.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[INPUT] {fname} → {removed} '5m' removed", "SUCCESS")
            except Exception as e:
                print(f"[INPUT] WRITE ERROR {fname}: {e}", "ERROR")
        else:
            print(f"[INPUT] {fname} → no '5m' found", "INFO")

    # ------------------------------------------------------------------
    # 2. CLEAN OUTPUTS (after calculations have been written)
    # ------------------------------------------------------------------
    print("\nPHASE 2 – Cleaning OUTPUT files …", "PHASE")
    for broker_dir in OUTPUT_ROOT.iterdir():
        if not broker_dir.is_dir():
            continue
        broker = broker_dir.name

        for risk, folder_name in RISK_FOLDERS.items():
            risk_dir = broker_dir / folder_name
            if not risk_dir.is_dir():
                continue

            for asset, filename in CALC_FILES.items():
                fpath = risk_dir / filename
                if not fpath.is_file():
                    continue

                try:
                    with fpath.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[OUTPUT] READ ERROR {broker}/{folder_name}/{filename}: {e}", "ERROR")
                    continue

                removed = filter_5m(data)
                total_removed += removed
                total_files += 1

                if removed:
                    try:
                        with fpath.open("w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        print(f"[OUTPUT] {broker}/{folder_name}/{filename} → {removed} '5m' removed", "SUCCESS")
                    except Exception as e:
                        print(f"[OUTPUT] WRITE ERROR {broker}/{folder_name}/{filename}: {e}", "ERROR")
                else:
                    # No need to spam – just count the file
                    pass

    print(f"\n[CLEAN 5m] Finished – {total_removed} '5m' entries removed from {total_files} files.", "SUCCESS")
    print("="*70 + "\n", "FOOTER")
    return True


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

    try:
        with in_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}", "ERROR")
        return False

    # Group orders: broker → risk → list of entries
    orders_by_broker_risk = defaultdict(lambda: {risk: [] for risk in RISK_FOLDERS})
    total_input_entries = 0

    for section_key, section in data.items():
        if not isinstance(section, list):
            continue
        for entry in section:
            total_input_entries += 1
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue

            if risk_usd in RISK_FOLDERS:
                orders_by_broker_risk[broker][risk_usd].append(entry)

    print(f"Loaded {total_input_entries} total entries from JSON", "INFO")
    processed_count = 0
    saved = 0

    # Debug: Show how many per broker/risk
    for broker, risk_dict in orders_by_broker_risk.items():
        for risk_usd, orders in risk_dict.items():
            if orders:
                print(f"[{broker}] Risk ${risk_usd}: {len(orders)} order(s) grouped", "DEBUG")

    for broker, risk_dict in orders_by_broker_risk.items():
        results_by_risk = {risk: [] for risk in RISK_FOLDERS}

        for risk_usd in RISK_FOLDERS:
            risk_orders = risk_dict.get(risk_usd, [])
            if not risk_orders:
                continue

            for entry in risk_orders:  # CHANGED: Loop over ALL entries
                market = entry.get("market")
                if not market:
                    print(f"[{broker}] Skipped: missing 'market'", "WARNING")
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ["buy_limit", "sell_limit"]:
                    print(f"[{broker}] Skipped {market}: invalid limit_order '{limit_type}'", "WARNING")
                    continue

                # Required fields
                required = ["entry_price", "volume", "tick_value"]
                missing = [field for field in required if field not in entry]
                if missing:
                    print(f"[{broker}] Skipped {market}: missing fields {missing}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 1e-5))
                except (ValueError, TypeError) as e:
                    print(f"[{broker}] Skipped {market}: invalid numeric value - {e}", "WARNING")
                    continue

                # Pip logic
                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)

                if pip_value_usd <= 0:
                    print(f"[{broker}] Skipped {market}: pip_value_usd <= 0 ({pip_value_usd})", "WARNING")
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                # Price calculation
                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

                # Round to correct digits
                digits = 5 if tick_size <= 1e-5 else 3
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
                    "selection_criteria": "all_valid_orders",
                    "broker": broker
                }

                results_by_risk[risk_usd].append(calc_entry)
                processed_count += 1

                print(
                    f"[{broker}] Risk ${risk_usd}: {market} {limit_type} @ {entry_price} → "
                    f"SL {sl_price} ({round(sl_pips, 2)} pips) | TP {tp_price} ({round(tp_pips, 2)} pips)",
                    "INFO"
                )

        # Save per risk folder
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
                print(f"[{broker}] Saved {len(calc_list)} calculation(s) for risk ${risk_usd} → {out_file}", "SUCCESS")
            except Exception as e:
                print(f"[{broker}] Failed to save risk ${risk_usd}: {e}", "ERROR")

    print(f"Forex SL/TP calculations done – Processed: {processed_count}, Saved: {saved} entries.", "SUCCESS")
    return True


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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Synthetics] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Synthetics] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Crypto] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Crypto] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[BasketIndices] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[BasketIndices] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Indices] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Indices] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Metals] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Metals] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Stocks] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Stocks] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[ETFs] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[ETFs] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Equities] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Equities] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Energies] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Energies] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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
                    try:
                        risk_usd = float(entry.get("riskusd_amount", 0))
                    except (TypeError, ValueError):
                        continue
                    if risk_usd in RISK_FOLDERS:
                        orders_by_broker_risk[broker][risk_usd].append(entry)
    elif isinstance(data, list):
        for entry in data:
            broker = entry.get("broker", "unknown")
            try:
                risk_usd = float(entry.get("riskusd_amount", 0))
            except (TypeError, ValueError):
                continue
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

            # FIXED: loop over ALL entries
            for entry in risk_orders:
                market = entry.get("market")
                if not market:
                    continue

                limit_type = entry.get("limit_order")
                if limit_type not in ("buy_limit", "sell_limit"):
                    print(f"[Commodities] Invalid limit type {limit_type}", "WARNING")
                    continue

                try:
                    entry_price = float(entry["entry_price"])
                    volume = float(entry["volume"])
                    tick_value = float(entry["tick_value"])
                    tick_size = float(entry.get("tick_size", 0.01))
                except (KeyError, ValueError):
                    print(f"[Commodities] Missing/invalid field in {market}", "WARNING")
                    continue

                pip_size = 10 * tick_size
                pip_value_usd = tick_value * volume * (pip_size / tick_size)
                if pip_value_usd <= 0:
                    continue

                sl_pips = risk_usd / pip_value_usd
                tp_pips = sl_pips * 3

                if limit_type == "buy_limit":
                    sl_price = entry_price - (sl_pips * pip_size)
                    tp_price = entry_price + (tp_pips * pip_size)
                else:  # sell_limit
                    sl_price = entry_price + (sl_pips * pip_size)
                    tp_price = entry_price - (tp_pips * pip_size)

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
                    "selection_criteria": "all_valid_orders",
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

    for broker_dir in Path(BASE_INPUT_DIR).iterdir():
        if not broker_dir.is_dir():
            continue
        broker = broker_dir.name
        print(f"\n[Promoter] Processing broker: {broker}", "INFO")

        # Collect ALL valid source entries from ALL risk levels
        source_entries = {}  # (asset, market, direction) → (src_risk, entry)

        for asset, filename in ASSET_CLASSES.items():
            for risk in RISK_LEVELS:
                file_path = broker_dir / RISK_FOLDERS[risk] / filename
                if not file_path.is_file():
                    continue
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    for entry in data:
                        market = entry["market"]
                        direction = entry["limit_order"]  # buy_limit/sell_limit
                        key = (asset, market, direction)

                        # Only use native (non-promoted) entries as source
                        criteria = entry.get("selection_criteria", "")
                        if criteria.startswith("promoted_from_"):
                            continue

                        # Keep if not seen or from lower risk
                        if key not in source_entries or risk < source_entries[key][0]:
                            source_entries[key] = (risk, entry)
                except Exception as e:
                    print(f"[Promoter] Failed to read {file_path}: {e}", "ERROR")

        if not source_entries:
            print(f"[Promoter] No source data for {broker}", "WARNING")
            continue

        # Promote EVERY source entry to EVERY higher risk
        for (asset, market, direction), (base_risk, base_entry) in source_entries.items():
            filename = ASSET_CLASSES[asset]

            for target_risk in RISK_LEVELS:
                if target_risk <= base_risk:
                    continue

                scale_factor = target_risk / base_risk
                new_volume = round(base_entry["volume"] * scale_factor, 8)

                promoted_entry = {
                    "market": market,
                    "limit_order": direction,
                    "timeframe": base_entry.get("timeframe", ""),
                    "entry_price": base_entry["entry_price"],
                    "volume": new_volume,
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
                    and e["market"] == market
                    and e["limit_order"] == direction
                    for e in existing_data
                )
                if already_exists:
                    continue

                existing_data.append(promoted_entry)
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    with target_file.open("w", encoding="utf-8") as f:
                        json.dump(existing_data, f, indent=2)
                    print(f"[Promoter] {asset.upper()} | {market} {direction} | ${base_risk}→${target_risk} (×{scale_factor})", "SUCCESS")
                    total_promoted += 1
                except Exception as e:
                    print(f"[Promoter] Save failed {target_file}: {e}", "ERROR")

    print(f"\n[Promoter] Promotion complete – {total_promoted} entries promoted.", "SUCCESS")
    checkriskorders()  # Ensure integrity
    return True



def checkriskorders():
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

    total_removed_invalid_placement = 0
    total_removed_unscaled = 0
    brokers_needing_rescaling = set()

    # Build risk hierarchy: higher risk should NOT appear in lower risk folders
    RISK_HIERARCHY = {r: [x for x in RISK_LEVELS if x < r] for r in RISK_LEVELS}

    # Map each risk to its valid source (only equal or lower, but we care about promotion source)
    VALID_SOURCE_FOR = {}
    for target in RISK_LEVELS:
        VALID_SOURCE_FOR[target] = [src for src in RISK_LEVELS if src <= target]

    print(f"\n[Checker] Starting risk order integrity check...", "INFO")

    for broker_dir in Path(BASE_INPUT_DIR).iterdir():
        if not broker_dir.is_dir():
            continue
        broker = broker_dir.name
        print(f"\n[Checker] Checking broker: {broker}", "INFO")

        # Collect all base (lowest risk) entries per asset for volume reference
        base_volumes = {}  # (asset, market) -> (risk, volume)

        for asset, filename in ASSET_CLASSES.items():
            file_path = broker_dir / RISK_FOLDERS[0.5] / filename
            if file_path.is_file():
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            if entry.get("riskusd_amount") == 0.5:
                                base_volumes[(asset, entry["market"])] = (0.5, entry["volume"])
                except Exception as e:
                    print(f"[Checker] Failed to read base {file_path}: {e}", "ERROR")

            # Also check higher risks for potential base if 0.5 missing
            for risk in RISK_LEVELS:
                if risk == 0.5:
                    continue
                file_path = broker_dir / RISK_FOLDERS[risk] / filename
                if file_path.is_file():
                    try:
                        with file_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                            for entry in data:
                                key = (asset, entry["market"])
                                if entry.get("selection_criteria", "").startswith("promoted_from_"):
                                    continue  # skip promoted
                                if key not in base_volumes:
                                    base_volumes[key] = (risk, entry["volume"])
                    except Exception as e:
                        print(f"[Checker] Failed to read {file_path}: {e}", "ERROR")

        # Now scan all risk levels
        for risk in RISK_LEVELS:
            risk_folder = broker_dir / RISK_FOLDERS[risk]
            if not risk_folder.exists():
                continue

            for asset, filename in ASSET_CLASSES.items():
                file_path = risk_folder / filename
                if not file_path.is_file():
                    continue

                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[Checker] Failed to read {file_path}: {e}", "ERROR")
                    continue

                valid_entries = []
                modified = False

                for entry in data:
                    market = entry.get("market")
                    entry_risk = entry.get("riskusd_amount")
                    volume = entry.get("volume", 0)
                    criteria = entry.get("selection_criteria", "")
                    key = (asset, market)

                    remove = False
                    reason = ""

                    # RULE 1: Higher risk orders must not appear in lower risk folders
                    if entry_risk > risk:
                        remove = True
                        reason = f"risk ${entry_risk} found in ${risk} folder (invalid placement)"
                        total_removed_invalid_placement += 1
                    # RULE 2: Promoted entries must have correct scaling
                    elif criteria.startswith("promoted_from_"):
                        # Extract source risk
                        try:
                            src_str = criteria.split("promoted_from_$")[1].split("_")[0]
                            src_risk = float(src_str)
                        except:
                            src_risk = None

                        if src_risk is not None and src_risk < risk:
                            expected_scale = risk / src_risk
                            base_volume = None

                            # Find base volume
                            if key in base_volumes:
                                base_risk, base_vol = base_volumes[key]
                                if base_risk == src_risk:
                                    base_volume = base_vol

                            # If no base, check if volume matches expected
                            if base_volume is None:
                                # Fallback: assume volume should be scaled from source
                                remove = True
                                reason = f"promoted entry missing base volume reference"
                                total_removed_unscaled += 1
                            else:
                                expected_volume = round(base_volume * expected_scale, 8)
                                if abs(volume - expected_volume) > 1e-8:
                                    remove = True
                                    reason = f"volume {volume} ≠ expected {expected_volume} (scale ×{expected_scale})"
                                    total_removed_unscaled += 1

                    if remove:
                        print(f"[Checker] REMOVING {asset.upper()} | {market} | ${entry_risk} in ${risk} folder | {reason}", "WARNING")
                        modified = True
                        if criteria.startswith("promoted_from_") and not reason.startswith("risk $"):
                            brokers_needing_rescaling.add(broker)
                    else:
                        valid_entries.append(entry)

                # Write back cleaned data
                if modified and valid_entries:
                    try:
                        with file_path.open("w", encoding="utf-8") as f:
                            json.dump(valid_entries, f, indent=2)
                        print(f"[Checker] Cleaned {file_path.name} in {RISK_FOLDERS[risk]} ({len(data)} → {len(valid_entries)})", "SUCCESS")
                    except Exception as e:
                        print(f"[Checker] Failed to save {file_path}: {e}", "ERROR")

        # End of broker

    # Final summary
    print(f"\n[Checker] Integrity check complete.", "INFO")
    print(f"   • {total_removed_invalid_placement} entries removed due to invalid risk placement", "INFO")
    print(f"   • {total_removed_unscaled} promoted entries removed due to incorrect scaling", "INFO")

    # Trigger re-promotion if needed
    if brokers_needing_rescaling:
        print(f"[Checker] Re-running scale_lowerorders_proportionally() for {len(brokers_needing_rescaling)} broker(s) with scaling escapes...", "INFO")
        scale_lowerorders_proportionally()
    else:
        print(f"[Checker] No rescaling needed.", "INFO")

    return True

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

def symbolvolumeupdater():
    '''NUCLEAR COMMAND: ALL RECORDS OF A SYMBOL = YOUR RULE
    Now also triggers recalculation of SL/TP for affected asset classes.'''
    from pathlib import Path
    import json
    from datetime import datetime

    CONTROL = Path("C:/xampp/htdocs/chronedge/chart/symbols_volumes_points/allowedmarkets/allsymbolsvolumesandrisk.json")
    ROOT = Path("C:/xampp/htdocs/chronedge/chart/symbols_volumes_points")

    if not CONTROL.exists():
        print("[NUKE] CONTROL FILE MISSING", "ERROR")
        return False

    try:
        allowed = json.loads(CONTROL.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[NUKE] Failed to parse control file: {e}", "ERROR")
        return False

    changes = 0
    changed_symbols = defaultdict(set)  # asset_class → {symbols}

    # Build command dictionary: symbol → {risk, volume, asset_class}
    commands = {}
    for risk_key, groups in allowed.items():
        try:
            risk = float(risk_key.split(":")[1].strip())
        except:
            continue
        for asset, symbols in groups.items():
            for s in symbols:
                sym = s.get("symbol")
                vol = s.get("volume")
                if sym and vol is not None:
                    commands[sym] = {"risk": risk, "volume": vol, "asset": asset}

    if not commands:
        print("[NUKE] NO COMMANDS FOUND", "INFO")
        return True

    # Map filename → asset class
    FILENAME_TO_ASSET = {
        "forexvolumesandrisk.json": "forex",
        "syntheticsvolumesandrisk.json": "synthetics",
        "cryptovolumesandrisk.json": "crypto",
        "basketindicesvolumesandrisk.json": "basketindices",
        "indicesvolumesandrisk.json": "indices",
        "metalsvolumesandrisk.json": "metals",
        "stocksvolumesandrisk.json": "stocks",
        "etfsvolumesandrisk.json": "etfs",
        "equitiesvolumesandrisk.json": "equities",
        "energiesvolumesandrisk.json": "energies",
        "commoditiesvolumesandrisk.json": "commodities",
    }

    # Master files to update
    masters = list(FILENAME_TO_ASSET.keys())

    # === PHASE 1: Apply changes to input files ===
    for file in masters:
        path = ROOT / file
        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[NUKE] Failed to read {file}: {e}", "ERROR")
            continue

        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    entries.extend(v)

        asset_class = FILENAME_TO_ASSET[file]
        changed_in_file = False

        for entry in entries:
            market = entry.get("market")
            if not market or market not in commands:
                continue

            cmd = commands[market]
            if cmd["asset"] != asset_class:
                continue  # Wrong asset class

            old_risk = entry.get("riskusd_amount")
            old_vol = entry.get("volume")

            if old_risk != cmd["risk"] or str(old_vol) != str(cmd["volume"]):
                print(f"[NUKE] {market} | ${old_risk}→${cmd['risk']} | {old_vol}→{cmd['volume']} [{asset_class.upper()}]")
                entry["riskusd_amount"] = cmd["risk"]
                entry["volume"] = cmd["volume"]
                entry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changes += 1
                changed_in_file = True
                changed_symbols[asset_class].add(market)

        if changed_in_file:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[NUKE] SAVED {file}")

    if changes == 0:
        print("[NUKE] NO CHANGES DETECTED – SKIPPING RECALCULATION", "INFO")
        return True

    # === PHASE 2: Recalculate SL/TP for affected asset classes ===
    print(f"\n[NUKE] {changes} changes detected. Recalculating SL/TP for affected assets...", "INFO")

    # Map asset class → calculation function
    CALC_FUNCTIONS = {
        "forex": calculate_forex_sl_tp_markets,
        "synthetics": calculate_synthetics_sl_tp_markets,
        "crypto": calculate_crypto_sl_tp_markets,
        "basketindices": calculate_basketindices_sl_tp_markets,
        "indices": calculate_indices_sl_tp_markets,
        "metals": calculate_metals_sl_tp_markets,
        "stocks": calculate_stocks_sl_tp_markets,
        "etfs": calculate_etfs_sl_tp_markets,
        "equities": calculate_equities_sl_tp_markets,
        "energies": calculate_energies_sl_tp_markets,
        "commodities": calculate_commodities_sl_tp_markets,
    }

    recalc_success = 0
    recalc_failed = 0

    for asset_class, symbols in changed_symbols.items():
        calc_func = CALC_FUNCTIONS.get(asset_class)
        if not calc_func:
            print(f"[NUKE] No calculator for {asset_class}", "WARNING")
            continue

        print(f"[NUKE] Recalculating {asset_class.upper()} for {len(symbols)} symbol(s): {', '.join(list(symbols)[:5])}{'...' if len(symbols)>5 else ''}")
        try:
            if calc_func():
                recalc_success += 1
            else:
                recalc_failed += 1
        except Exception as e:
            print(f"[NUKE] {asset_class.upper()} recalc failed: {e}", "ERROR")
            recalc_failed += 1

    print(f"[NUKE] Recalculation complete: {recalc_success} succeeded, {recalc_failed} failed.", "INFO")

    # === PHASE 3: Re-promote & re-categorize ===
    print("[NUKE] Running scale_lowerorders_proportionally()...", "INFO")
    scale_lowerorders_proportionally()

    print("[NUKE] Running categorise_strategy()...", "INFO")
    categorise_strategy()

    print(f"\n[NUKE] NUCLEAR SYNC COMPLETE | {changes} records updated | {recalc_success} asset(s) recalculated.", "SUCCESS")
    return True  

def main():
    symbolsorderfiltering()
    clean_5m_timeframes()
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
    clean_5m_timeframes()
    print("="*60, "FOOTER")
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!", "SUCCESS")
    print("="*60 + "\n", "FOOTER")
    return True


if __name__ == "__main__":
    main()
   
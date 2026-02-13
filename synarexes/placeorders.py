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
import time
import threading
import traceback
from datetime import timedelta
import traceback
import shutil
from datetime import datetime
import math
from pathlib import Path

INVESTOR_USERS = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors\investors.json"
INV_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors"
NORMALIZE_SYMBOLS_PATH = r"C:\xampp\htdocs\chronedge\synarex\symbols_normalization.json"
DEFAULT_ACCOUNTMANAGEMENT = r"C:\xampp\htdocs\chronedge\synarex\default_accountmanagement.json"

def load_investors_dictionary():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors\investors.json"
    """Load brokers config from JSON file with error handling and fallback."""
    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Convert numeric strings back to int where needed
        for user_brokerid, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in investors.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load investors.json: {e}", "CRITICAL")
        return {}
usersdictionary = load_investors_dictionary()

def sort_orders():
    if not os.path.exists(INV_PATH):
        print(f"Error: Investor path {INV_PATH} not found.")
        return False

    # 1. Identify all investor directories
    investor_ids = [f for f in os.listdir(INV_PATH) if os.path.isdir(os.path.join(INV_PATH, f))]

    for inv_id in investor_ids:
        inv_root = os.path.join(INV_PATH, inv_id)
        acc_mgmt_path = os.path.join(inv_root, "accountmanagement.json")

        if not os.path.exists(acc_mgmt_path):
            continue

        # 2. Load and potentially update the config for THIS specific investor
        try:
            with open(acc_mgmt_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check if field exists; if not, add it with default [3]
                if "selected_risk_reward" not in data:
                    data["selected_risk_reward"] = [3]
                    # Move pointer to start and overwrite file with updated data
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()

                # Convert list to set of strings for fast lookup
                allowed_ratios = {str(r) for r in data.get("selected_risk_reward", [])}
                
        except Exception as e:
            print(f" ! Skip {inv_id}: Error processing config: {e}")
            continue

        # 3. Deep search using os.walk (Bottom-Up traversal)
        # topdown=False is preferred when deleting contents to avoid pathing errors
        for root, dirs, files in os.walk(inv_root, topdown=False):
            for dir_name in dirs:
                if dir_name.startswith("risk_reward_"):
                    # Extract the ratio suffix (the 'X' in 'risk_reward_X')
                    ratio_suffix = dir_name.replace("risk_reward_", "")

                    # 4. If the ratio found in the folder name is NOT allowed, delete it
                    if ratio_suffix not in allowed_ratios:
                        full_path = os.path.join(root, dir_name)
                        try:
                            shutil.rmtree(full_path)
                        except Exception as e:
                            print(f" ! Failed to delete {full_path}: {e}")

    print("--- Orders filtration completed ---")
    return True

def debug_print_all_broker_symbols():
    """
    Connects to the currently active MT5 terminal and prints 
    every available symbol name to the console.
    """
    # Ensure MT5 is initialized (if not already)
    if not mt5.initialize():
        print(f"FAILED to initialize MT5: {mt5.last_error()}")
        return

    # Get all symbols from the terminal
    symbols = mt5.symbols_get()
    
    if symbols is None:
        print("No symbols found. This might be a connection issue.")
    else:
        print(f"\n{'='*40}")
        print(f"BROKER: {mt5.account_info().server if mt5.account_info() else 'Unknown'}")
        print(f"TOTAL SYMBOLS FOUND: {len(symbols)}")
        print(f"{'='*40}")
        
        # Extract names and sort them alphabetically for easier reading
        all_names = sorted([s.name for s in symbols])
        
        for i, name in enumerate(all_names, 1):
            print(f"{i}. {name}")
            
        print(f"{'='*40}\nEND OF LIST\n{'='*40}")

def get_normalized_symbol(record_symbol, norm_map):
    """
    Finds the correct broker symbol, prioritizing those that allow full trading.
    Checks for suffixes like '+', '.', 'm', '..', etc.
    """
    if not record_symbol:
        return None

    search_term = record_symbol.replace(" ", "").replace("_", "").replace(".", "").upper()
    norm_data = norm_map.get("NORMALIZATION", {})
    target_synonyms = []

    # 1. Identify potential base names from the normalization map
    for standard_key, synonyms in norm_data.items():
        clean_key = standard_key.replace("_", "").upper()
        clean_syns = [s.replace(" ", "").replace("_", "").replace("/", "").upper() for s in synonyms]
        
        if search_term == clean_key or search_term in clean_syns:
            target_synonyms = list(synonyms)
            break

    if not target_synonyms:
        target_synonyms = [record_symbol, search_term]

    # 2. Get all symbols from broker
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return None
    
    available_names = [s.name for s in all_symbols]

    # 3. Helper to check if a symbol is actually tradeable
    def is_tradeable(sym_name):
        info = mt5.symbol_info(sym_name)
        if info is None:
            mt5.symbol_select(sym_name, True)
            info = mt5.symbol_info(sym_name)
        # Check if symbol exists AND trade_mode allows full access (2)
        return info is not None and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    # 4. Search Strategy:
    # First, try exact matches. If not tradeable, look for suffix variations.
    common_suffixes = ["+", "m", ".", "..", "#", "i", "z"]
    
    for option in target_synonyms:
        clean_opt = option.replace("/", "").upper()
        
        # Priority A: Check for exact match first
        if clean_opt in available_names and is_tradeable(clean_opt):
            return clean_opt
        
        # Priority B: Look for any broker symbol that starts with our option (e.g., AUDUSD -> AUDUSD+)
        for broker_name in available_names:
            if broker_name.upper().startswith(clean_opt):
                if is_tradeable(broker_name):
                    return broker_name

    # Priority C: Manual Suffix Append (Last Resort)
    for option in target_synonyms:
        clean_opt = option.replace("/", "").upper()
        for suffix in common_suffixes:
            test_name = f"{clean_opt}{suffix}"
            if test_name in available_names and is_tradeable(test_name):
                return test_name

    return None

def get_filling_mode(symbol):
    """Helper to detect the correct filling mode for the broker."""
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return mt5.ORDER_FILLING_IOC # Fallback
    
    # Corrected attribute names for bitwise checking
    filling_mode = symbol_info.filling_mode
    
    if filling_mode & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        # Most common for Deriv/Indices if FOK/IOC are restricted
        return mt5.ORDER_FILLING_RETURN
    
def deduplicate_orders():
    """
    Scans all risk bucket JSON files and removes duplicate orders based on:
    Symbol, Timeframe, Order Type, and Entry Price.
    """
    print(f"DEDUPLICATING ORDERS.") 
    total_files_cleaned = 0
    total_duplicates_removed = 0

    # Path to the base investor directory
    inv_base_path = Path(INV_PATH)

    # 1. Iterate through all investor folders
    for inv_folder in inv_base_path.iterdir():
        if not inv_folder.is_dir():
            continue

        # 2. Search for all risk bucket JSON files
        # Matches: .../risk_reward_3.0/2usd_risk/2usd_risk.json
        search_pattern = "**/risk_reward_*/*usd_risk/*.json"
        order_files = list(inv_folder.rglob(search_pattern))

        for file_path in order_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    orders = json.load(f)

                if not orders:
                    continue

                original_count = len(orders)
                seen_orders = set()
                unique_orders = []

                for order in orders:
                    # Create a unique key based on your requirements
                    # We use entry price as well to ensure different setups on 
                    # the same symbol/TF are preserved
                    unique_key = (
                        str(order.get("symbol")).strip(),
                        str(order.get("timeframe")).strip(),
                        str(order.get("order_type")).strip(),
                        float(order.get("entry", 0))
                    )

                    if unique_key not in seen_orders:
                        seen_orders.add(unique_key)
                        unique_orders.append(order)
                
                # 3. Only write back if duplicates were actually found
                if len(unique_orders) < original_count:
                    removed = original_count - len(unique_orders)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(unique_orders, f, indent=4)
                    
                    total_duplicates_removed += removed
                    total_files_cleaned += 1

            except Exception as e:
                print(f" [✗] Error processing {file_path.name}: {e}")
    print(f"DEDUPLICATION COMPLETED")
    return True

def check_limit_orders_risk():

    """
    Function 3: Validates live pending orders against the account's current risk bucket.
    Synchronized with the stable initialization logic of place_usd_orders.
    """
    print("\n" + "="*80)
    print("STARTING CHECK_LIMIT_ORDERS_RISK (SYNCHRONIZED INIT)")
    print("="*80)

    # --- DATA INITIALIZATION ---
    try:
        if not os.path.exists(NORMALIZE_SYMBOLS_PATH):
            print("CRITICAL ERROR: Normalization map path does not exist.")
            return False
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        print(f"\n{'-'*80}\nAUDITING RISK LIMITS FOR: {user_brokerid}\n{'-'*80}")
        inv_root = Path(INV_PATH) / user_brokerid
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists():
            print(f"  [SKIP] accountmanagement.json not found for {user_brokerid}")
            continue

        # --- LOAD RISK CONFIG ---
        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            risk_map = config.get("account_balance_default_risk_management", {})
        except Exception as e:
            print(f"  [ERROR] Failed to read config: {e}")
            continue

        # --- START STABLE INIT LOGIC ---
        mt5.shutdown() 
        login_id = int(broker_cfg['LOGIN_ID'])
        mt5_path = broker_cfg["TERMINAL_PATH"]
        
        print(f"  Initializing terminal at: {mt5_path}")
        if not mt5.initialize(path=mt5_path, timeout=180000):
            print(f"  [ERROR] initialize() failed: {mt5.last_error()}")
            continue

        # Login check
        acc_info = mt5.account_info()
        if acc_info is None or acc_info.login != login_id:
            if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                print(f"  [ERROR] Login failed: {mt5.last_error()}")
                continue
            acc_info = mt5.account_info() # Refresh after login
            print(f"  [OK] Logged into {login_id}")
        else:
            print(f"  [OK] Already logged into {login_id}")
        # --- END STABLE INIT LOGIC ---

        balance = acc_info.balance

        # Determine Primary Risk Value (Current Bucket)
        primary_risk = None
        for range_str, r_val in risk_map.items():
            try:
                raw_range = range_str.split("_")[0]
                low, high = map(float, raw_range.split("-"))
                if low <= balance <= high:
                    primary_risk = int(r_val)
                    break
            except: continue

        if primary_risk is None:
            print(f"  [WARN] No risk mapping found for balance {balance}")
            mt5.shutdown()
            continue

        print(f"  [INFO] Balance: {balance} | Target Risk: {primary_risk} USD")

        # Check Live Pending Orders
        pending_orders = mt5.orders_get()
        if pending_orders:
            for order in pending_orders:
                # Process only Limit Orders
                if order.type not in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
                    continue

                calc_type = mt5.ORDER_TYPE_BUY if order.type == mt5.ORDER_TYPE_BUY_LIMIT else mt5.ORDER_TYPE_SELL
                
                # Calculate live risk (Entry to SL)
                sl_profit = mt5.order_calc_profit(calc_type, order.symbol, order.volume_initial, order.price_open, order.sl)
                
                if sl_profit is not None:
                    order_risk_usd = round(abs(sl_profit), 2)
                    
                    # CANCEL if order risk differs significantly from the primary risk bucket
                    # We allow a 1.0 USD tolerance for spread/fee variations
                    if abs(order_risk_usd - primary_risk) > 1.0: 
                        print(f"  [!] RISK MISMATCH: Order {order.ticket} ({order.symbol}) = {order_risk_usd} USD")
                        print(f"      Required: {primary_risk} USD. Removing order...")
                        
                        cancel_request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket
                        }
                        result = mt5.order_send(cancel_request)
                        
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"      [X] Cancel failed: {result.comment}")
                        else:
                            print(f"      [V] Order {order.ticket} successfully removed.")
                else:
                    print(f"  [WARN] Could not calculate risk for order {order.ticket}")

        mt5.shutdown()
        print(f"<<< [FINISHED: {user_brokerid}] Audit complete.")

    print("\n" + "="*80)
    print("FINSHED AND REMOVED ORDERS IN OVER RISK.") 
    print("="*80)
    return True

def place_usd_orders():
    # --- SUB-FUNCTION 1: DATA INITIALIZATION ---
    def load_normalization_map():
        try:
            if not os.path.exists(NORMALIZE_SYMBOLS_PATH):
                return {}
            with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Critical Error: Could not load normalization map: {e}")
            return None

    # --- SUB-FUNCTION 2: RISK & FILE AGGREGATION ---
    def collect_and_deduplicate_entries(inv_root, risk_map, balance, pull_lower, selected_rr, norm_map):
        primary_risk = None
        print(f"  [DEBUG] Determining primary risk for balance: {balance}")
        for range_str, r_val in risk_map.items():
            try:
                raw_range = range_str.split("_")[0]
                low, high = map(float, raw_range.split("-"))
                if low <= balance <= high:
                    primary_risk = int(r_val)
                    print(f"    [OK] Balance {balance} in range {low}-{high} → Risk Level: {primary_risk}")
                    break
            except Exception as e:
                print(f"    [WARN] Error parsing risk range '{range_str}': {e}")
                continue

        if primary_risk is None:
            print(f"  [ERROR] No matching risk range found for balance: {balance}")
            return None, []

        risk_levels = [primary_risk]
        if pull_lower:
            start_lookback = max(1, primary_risk - 9)
            risk_levels = list(range(start_lookback, primary_risk + 1))
            print(f"  [INFO] Pull lower enabled, scanning risk levels: {risk_levels}")
        else:
            print(f"  [INFO] Scanning only primary risk level: {risk_levels}")

        unique_entries_dict = {}
        target_rr_folder = f"risk_reward_{selected_rr}"
        
        for r_val in reversed(risk_levels):
            risk_folder_name = f"{r_val}usd_risk"
            risk_filename = f"{r_val}usd_risk.json"
            search_pattern = f"**/{target_rr_folder}/{risk_folder_name}/{risk_filename}"
            
            found_files = False
            for path in inv_root.rglob(search_pattern):
                found_files = True
                if path.is_file():
                    try:
                        print(f"      [FILE] Reading: {path}")
                        with open(path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for entry in data:
                                    symbol = get_normalized_symbol(entry["symbol"], norm_map)
                                    key = f"{entry.get('timeframe','NA')}|{symbol}|{entry.get('order_type','NA')}|{round(float(entry['entry']), 5)}"
                                    if key not in unique_entries_dict:
                                        unique_entries_dict[key] = entry
                                        print(f"          [ADD] Added entry: {symbol} @ {entry['entry']}")
                    except Exception as e:
                        print(f"      [ERROR] Failed to process {path}: {e}")
                break # Matched this risk level
        
        print(f"  [RESULT] Total unique entries collected: {len(unique_entries_dict)}")
        return risk_levels, list(unique_entries_dict.values())

    # --- SUB-FUNCTION 3: BROKER CLEANUP ---
    def cleanup_unauthorized_orders(all_entries, norm_map):
        print("  [CLEANUP] Checking for unauthorized orders...")
        try:
            current_orders = mt5.orders_get()
            if not current_orders:
                print("  [CLEANUP] No pending orders found")
                return
            
            deleted_count = 0
            for order in current_orders:
                is_authorized = False
                for entry in all_entries:
                    vol_key = next((k for k in entry.keys() if k.endswith("_volume")), None)
                    if not vol_key: continue
                    
                    e_vol = round(float(entry[vol_key]), 2)
                    e_price = round(float(entry["entry"]), 5)
                    e_symbol = get_normalized_symbol(entry["symbol"], norm_map)

                    if (order.symbol == e_symbol and 
                        round(order.price_open, 5) == e_price and 
                        round(order.volume_initial, 2) == e_vol):
                        is_authorized = True
                        break
                
                if not is_authorized:
                    print(f"  [DELETE] Unauthorized order - Ticket: {order.ticket}")
                    res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        deleted_count += 1
            print(f"  [CLEANUP] Deleted {deleted_count} unauthorized orders")
        except Exception as e:
            print(f"  [ERROR] Cleanup failed: {e}")

    # --- SUB-FUNCTION 4: ORDER EXECUTION ---
    def execute_missing_orders(all_entries, norm_map, default_magic, selected_rr, trade_allowed):
        placed = failed = skipped = 0
        print(f"  [EXECUTION] Processing {len(all_entries)} entries...")
        
        for idx, entry in enumerate(all_entries):
            try:
                # The normalization function now ensures we get a TRADEABLE symbol
                symbol = get_normalized_symbol(entry["symbol"], norm_map)
                
                if not symbol:
                    print(f"      [SKIP] {entry['symbol']} - No tradeable symbol found on broker.")
                    failed += 1
                    continue

                # Ensure symbol is visible in Market Watch
                if not mt5.symbol_select(symbol, True):
                    print(f"      [FAIL] {symbol} - Could not select symbol.")
                    failed += 1
                    continue

                symbol_info = mt5.symbol_info(symbol)
                vol_key = next((k for k in entry.keys() if k.endswith("_volume")), None)
                
                # Check for existing positions or orders
                existing_orders = mt5.orders_get(symbol=symbol) or []
                existing_pos = mt5.positions_get(symbol=symbol) or []
                
                entry_price = round(float(entry["entry"]), symbol_info.digits)
                
                if existing_pos or any(round(o.price_open, symbol_info.digits) == entry_price for o in existing_orders):
                    skipped += 1
                    continue

                volume = float(entry[vol_key])
                if symbol_info.volume_step > 0:
                    volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
                
                # Clamp volume to broker limits
                volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))

                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": symbol,
                    "volume": round(volume, 2),
                    "type": mt5.ORDER_TYPE_BUY_LIMIT if entry["order_type"] == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT,
                    "price": entry_price,
                    "sl": round(float(entry["exit"]), symbol_info.digits),
                    "tp": round(float(entry["target"]), symbol_info.digits),
                    "magic": int(entry.get("magic", default_magic)),
                    "comment": f"Risk_Agg_RR{selected_rr}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                res = mt5.order_send(request)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"      [SUCCESS] Order placed: {symbol} Ticket: {res.order}")
                    placed += 1
                else:
                    ret_msg = res.comment if res else "No response"
                    print(f"      [FAIL] {symbol} @ {entry_price} - Error: {ret_msg} (Code: {res.retcode if res else 'N/A'})")
                    failed += 1
            except Exception as e:
                print(f"      [ERROR] Fatal error placing {entry.get('symbol')}: {e}")
                failed += 1
                
        return placed, failed, skipped

    # --- MAIN EXECUTION FLOW ---
    print("\n" + "="*80)
    print("STARTING PLACE_USD_ORDERS (VERIFIED STABLE)")
    print("="*80)
    
    norm_map = load_normalization_map()
    if norm_map is None: return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        print(f"\n{'-'*80}\nPROCESSING INVESTOR: {user_brokerid}\n{'-'*80}")
        inv_root = Path(INV_PATH) / user_brokerid
        acc_mgmt_path = inv_root / "accountmanagement.json"
        if not acc_mgmt_path.exists(): continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # --- START STABLE INIT LOGIC ---
            mt5.shutdown() 
            login_id = int(broker_cfg['LOGIN_ID'])
            mt5_path = broker_cfg["TERMINAL_PATH"]
            
            print(f"  Initializing terminal at: {mt5_path}")
            if not mt5.initialize(path=mt5_path, timeout=180000):
                print(f"  [ERROR] initialize() failed: {mt5.last_error()}")
                continue

            # Login check
            acc = mt5.account_info()
            if acc is None or acc.login != login_id:
                if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                    print(f"  [ERROR] Login failed: {mt5.last_error()}")
                    continue
                print(f"  [OK] Logged into {login_id}")
            else:
                print(f"  [OK] Already logged into {login_id}")
            # --- END STABLE INIT LOGIC ---

            # Settings Extraction
            settings = config.get("settings", {})
            pull_lower = settings.get("pull_orders_from_lower", False)
            selected_rr = config.get("selected_risk_reward", [None])[0]
            risk_map = config.get("account_balance_default_risk_management", {})
            default_magic = config.get("magic_number", 123456)
            
            acc_info = mt5.account_info()
            term_info = mt5.terminal_info()
            
            # AutoTrading Check
            print(f"  [INFO] Terminal AutoTrading Allowed: {term_info.trade_allowed}")

            print(f"\n  [STAGE 1] Risk determination and file loading")
            risk_lvls, all_entries = collect_and_deduplicate_entries(inv_root, risk_map, acc_info.balance, pull_lower, selected_rr, norm_map)
            
            if all_entries:
                print(f"\n  [STAGE 2] Cleaning up unauthorized orders")
                cleanup_unauthorized_orders(all_entries, norm_map)
                
                print(f"\n  [STAGE 3] Executing missing orders")
                p, f, s = execute_missing_orders(all_entries, norm_map, default_magic, selected_rr, term_info.trade_allowed)
                print(f"\n  [SUMMARY] {user_brokerid}: Placed:{p}, Failed:{f}, Skipped:{s}")
            else:
                print(f"  [INFO] No entries to process for {user_brokerid}")

        except Exception as e:
            print(f"  [ERROR] System Error for {user_brokerid}: {e}")
        
    mt5.shutdown()
    print("\n" + "="*80 + "\nCOMPLETED\n" + "="*80)
    return True  

def default_price_repair():
    """
    Synchronizes 'exit' and 'target' prices from limit_orders_backup.json 
    to all active risk bucket files ONLY if 'default_price' is set to true 
    in accountmanagement.json.
    """
    print("--- STARTING DEFAULT-PRICE MODIFICATION---")
    
    if not os.path.exists(INV_PATH):
        print(f"Error: Investor path {INV_PATH} not found.")
        return False

    investor_ids = [f for f in os.listdir(INV_PATH) if os.path.isdir(os.path.join(INV_PATH, f))]

    for inv_id in investor_ids:
        inv_root = Path(INV_PATH) / inv_id
        acc_mgmt_path = inv_root / "accountmanagement.json"
        
        # --- 1. PERMISSION CHECK ---
        if not acc_mgmt_path.exists():
            print(f" > Skipping {inv_id}: accountmanagement.json missing.")
            continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check the nested setting: settings -> default_price
            settings = config.get("settings", {})
            is_allowed = settings.get("default_price", False)
            
            if not is_allowed:
                continue
        except Exception as e:
            print(f" ! Error reading config for {inv_id}: {e}")
            continue

        # --- 2. BACKUP LOCATION ---
        backup_files = list(inv_root.rglob("limit_orders_backup.json"))
        if not backup_files:
            print(f" > Skipping {inv_id}: No limit_orders_backup.json found.")
            continue
            
        backup_path = backup_files[0]
        print(f"{inv_id} [AUTHORIZED] default price")

        # --- 3. LOAD MASTER DATA ---
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_entries = json.load(f)
            
            master_map = {}
            for b_entry in backup_entries:
                key = (b_entry.get("symbol"), b_entry.get("entry"), b_entry.get("order_type"))
                master_map[key] = b_entry
                
        except Exception as e:
            print(f" ! Error loading backup data for {inv_id}: {e}")
            continue

        # --- 4. APPLY UPDATES TO RISK BUCKETS ---
        risk_files = list(inv_root.rglob("*usd_risk.json"))
        investor_updates = 0
        total_orders_patched = 0

        for target_file in risk_files:
            # Avoid self-referencing if backup uses the same naming convention
            if target_file.name == "limit_orders_backup.json":
                continue

            file_changed = False
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    active_entries = json.load(f)

                if not isinstance(active_entries, list):
                    continue

                for active_entry in active_entries:
                    key = (active_entry.get("symbol"), active_entry.get("entry"), active_entry.get("order_type"))
                    
                    if key in master_map:
                        backup_ref = master_map[key]
                        
                        # Apply Exit repair (if not 0)
                        b_exit = backup_ref.get("exit", 0)
                        if b_exit != 0 and active_entry.get("exit") != b_exit:
                            active_entry["exit"] = b_exit
                            file_changed = True
                            total_orders_patched += 1
                        
                        # Apply Target repair (if not 0)
                        b_target = backup_ref.get("target", 0)
                        if b_target != 0 and active_entry.get("target") != b_target:
                            active_entry["target"] = b_target
                            file_changed = True
                            total_orders_patched += 1

                if file_changed:
                    with open(target_file, 'w', encoding='utf-8') as f:
                        json.dump(active_entries, f, indent=4)
                    investor_updates += 1

            except Exception as e:
                print(f" ! Error processing {target_file.name}: {e}")

        print(f" [✓] {inv_id}: Successfully repaired {total_orders_patched} prices.")

    print("--- Default Price Repair Completed ---")
    return True

def place_orders():
    sort_orders()
    deduplicate_orders()
    place_usd_orders()
    check_limit_orders_risk()
    default_price_repair()


if __name__ == "__main__":
   place_orders()


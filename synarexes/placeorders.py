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
    1. Takes "US Oil" from record.
    2. Cleans it to "USOIL".
    3. Finds the list in Normalization JSON.
    4. Checks broker for any name in that list.
    """
    if not record_symbol:
        return None

    # Step 1: Clean the record name for searching (remove spaces, underscores, dots)
    # "US Oil" -> "USOIL" | "US_OIL" -> "USOIL"
    search_term = record_symbol.replace(" ", "").replace("_", "").replace(".", "").upper()
    
    norm_data = norm_map.get("NORMALIZATION", {})
    target_synonyms = []

    # Step 2: Go to Normalization straight
    for standard_key, synonyms in norm_data.items():
        # Clean the standard key and all synonyms for a fair comparison
        clean_key = standard_key.replace("_", "").upper()
        clean_syns = [s.replace(" ", "").replace("_", "").replace("/", "").upper() for s in synonyms]
        
        if search_term == clean_key or search_term in clean_syns:
            target_synonyms = synonyms
            break

    # If the record symbol isn't in our map, at least try the cleaned version
    if not target_synonyms:
        target_synonyms = [record_symbol, search_term]

    # Step 3: Check Broker for any of these possibilities
    # Fetch all available names once to save time
    available_symbols = [s.name for s in mt5.symbols_get()]
    
    for option in target_synonyms:
        # Check for Exact Match (e.g., "USOUSD")
        if option in available_symbols:
            return option
            
        # Check for Suffix Match (e.g., "USOUSD+")
        # We check if any broker symbol starts with our synonym
        clean_opt = option.replace("/", "").upper()
        for broker_name in available_symbols:
            if broker_name.upper().startswith(clean_opt):
                return broker_name

    print(f"[!] No broker match found for {record_symbol} even after normalization check.")
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

def check_orders_live_volume():
    """
    Function 2: Validates and modifies volumes to match broker constraints.
    - Ensures volume is not below 'volume_min'.
    - Rounds volume to the nearest 'volume_step' (fix decimal issues).
    - Updates the specific volume key in the JSON file.
    """
    total_files_updated = 0

    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        
        if not mt5.initialize(
            path=broker_cfg["TERMINAL_PATH"], 
            login=int(broker_cfg["LOGIN_ID"]), 
            password=broker_cfg["PASSWORD"], 
            server=broker_cfg["SERVER"]
        ):
            print(f"    [!] CONNECTION ERROR for {inv_id}")
            continue

        search_pattern = "**/risk_reward_*/*usd_risk/*usd_risk.json"
        found_files = list(inv_root.rglob(search_pattern))

        for target_file_path in found_files:
            try:
                with open(target_file_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)

                file_changed = False
                for entry in entries:
                    raw_symbol = entry.get("symbol")
                    symbol = get_normalized_symbol(raw_symbol, norm_map)
                    
                    # Get broker specific symbol info
                    info = mt5.symbol_info(symbol)
                    if info is None:
                        continue

                    # Identify the volume key (e.g., "1h_volume" or "volume")
                    vol_key = next((k for k in entry.keys() if k.endswith("_volume")), "volume")
                    current_vol = float(entry.get(vol_key, 0.0))

                    # 1. Minimum Volume Check
                    # If it's too low, we set it to the minimum the broker allows
                    new_vol = max(current_vol, info.volume_min)

                    # 2. Volume Step / Decimal Check
                    # Deriv often uses steps of 0.1 or 0.01. This math ensures alignment.
                    step = info.volume_step
                    if step > 0:
                        # math.floor is safer to avoid accidentally increasing risk
                        new_vol = round(math.floor(new_vol / step) * step, 2)
                    
                    # Final check: Ensure we didn't drop below min after rounding
                    if new_vol < info.volume_min:
                        new_vol = info.volume_min

                    # Update only if different
                    if new_vol != current_vol:
                        entry[vol_key] = new_vol
                        file_changed = True
                        #print(f"    [FIX] {symbol}: Adjusted {current_vol} -> {new_vol} ({broker_cfg['SERVER']})")

                if file_changed:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        json.dump(entries, f, indent=4)
                    total_files_updated += 1

            except Exception as e:
                print(f"    [!] Error in {target_file_path}: {e}")

        mt5.shutdown()
    
    return True

def check_orders_live_risk():
    """
    Function 1: Only calculates and adds live risk/reward fields to orders.
    Does NOT move files between buckets - only adds broker-specific risk fields.
    """
    total_files_processed = 0
    total_errors_encountered = 0

    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        broker_name = broker_cfg.get("SERVER", "UnknownBroker")
        
        print(f"\n>>> [INVESTOR: {inv_id}] Calculating live risk on {broker_name}...")
        investor_files_updated = 0

        if not mt5.initialize(
            path=broker_cfg["TERMINAL_PATH"], 
            login=int(broker_cfg["LOGIN_ID"]), 
            password=broker_cfg["PASSWORD"], 
            server=broker_cfg["SERVER"]
        ):
            print(f"    [!] CONNECTION ERROR for {inv_id}: {mt5.last_error()}")
            total_errors_encountered += 1
            continue

        try:
            search_pattern = "**/risk_reward_*/*usd_risk/*usd_risk.json"
            found_files = list(inv_root.rglob(search_pattern))
            
            for target_file_path in found_files:
                try:
                    with open(target_file_path, 'r', encoding='utf-8') as f:
                        entries = json.load(f)

                    file_changed = False
                    updated_entries = []

                    for entry in entries:
                        # Remove old broker-specific risk fields before adding fresh ones
                        for key in list(entry.keys()):
                            if key.endswith("_sl_risk_amount") or key.endswith("_tp_reward_amount"):
                                entry.pop(key, None)

                        raw_symbol = entry.get("symbol")
                        symbol = get_normalized_symbol(raw_symbol, norm_map)
                        
                        if not symbol or mt5.symbol_info(symbol) is None:
                            total_errors_encountered += 1
                            updated_entries.append(entry)
                            continue

                        info = mt5.symbol_info(symbol)
                        if not info.visible: 
                            mt5.symbol_select(symbol, True)

                        # Extract values
                        entry_p = float(entry.get("entry", 0))
                        exit_p = float(entry.get("exit", 0))
                        target_p = float(entry.get("target", 0))
                        vol_key = next((k for k in entry.keys() if k.endswith("_volume")), "volume")
                        current_vol = float(entry.get(vol_key, 0.0))
                        
                        order_type_str = entry.get("order_type", "").lower()
                        calc_type = mt5.ORDER_TYPE_BUY if "buy" in order_type_str else mt5.ORDER_TYPE_SELL

                        # Calculate live risk and reward
                        sl_risk = mt5.order_calc_profit(calc_type, symbol, current_vol, entry_p, exit_p)
                        
                        if sl_risk is not None:
                            risk_amt = round(abs(sl_risk), 2)
                            reward_amt = round(abs(mt5.order_calc_profit(calc_type, symbol, current_vol, entry_p, target_p)), 2)
                            
                            # Add broker-specific live risk fields
                            entry[f"{broker_name}_sl_risk_amount"] = risk_amt
                            entry[f"{broker_name}_tp_reward_amount"] = reward_amt
                            file_changed = True
                        
                        updated_entries.append(entry)

                    # Save updated entries with new risk fields
                    if file_changed:
                        with open(target_file_path, 'w', encoding='utf-8') as f:
                            json.dump(updated_entries, f, indent=4)
                        investor_files_updated += 1

                except Exception as e:
                    print(f"    [!] Error processing file {target_file_path}: {e}")
                    total_errors_encountered += 1
                    continue

        except Exception as e:
            print(f"    [!] CRASH for {inv_id}: {e}")
            total_errors_encountered += 1
        
        mt5.shutdown()
        print(f"<<< [FINISHED: {inv_id}] Updated {investor_files_updated} files with live risk fields.")
        total_files_processed += investor_files_updated
        
    return True

def repair_order_buckets():
    """
    Function 2: Identifies and fixes bucket violations for both overflow and underflow.
    Moves orders to the correct bucket based on: floor(live_risk_amt).
    """
    total_files_processed = 0
    total_errors_encountered = 0
    total_moved_orders = 0

    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        broker_name = broker_cfg.get("SERVER", "UnknownBroker")
        
        print(f"\n>>> [INVESTOR: {inv_id}] Repairing bucket violations on {broker_name}...")
        investor_files_updated = 0
        investor_orders_moved = 0

        if not mt5.initialize(
            path=broker_cfg["TERMINAL_PATH"], 
            login=int(broker_cfg["LOGIN_ID"]), 
            password=broker_cfg["PASSWORD"], 
            server=broker_cfg["SERVER"]
        ):
            print(f"    [!] CONNECTION ERROR for {inv_id}: {mt5.last_error()}")
            total_errors_encountered += 1
            continue

        try:
            search_pattern = "**/risk_reward_*/*usd_risk/*usd_risk.json"
            found_files = list(inv_root.rglob(search_pattern))
            
            for target_file_path in found_files:
                try:
                    # current_bucket_limit is the integer name (e.g., 2 for '2usd_risk')
                    current_bucket_limit = int(target_file_path.stem.replace('usd_risk', ''))
                except:
                    current_bucket_limit = 0
                    print(f"    [!] Could not parse bucket limit from {target_file_path.name}")

                try:
                    with open(target_file_path, 'r', encoding='utf-8') as f:
                        entries = json.load(f)

                    staying_entries = []
                    file_changed = False

                    for entry in entries:
                        risk_field = f"{broker_name}_sl_risk_amount"
                        live_risk_amt = entry.get(risk_field)
                        
                        if live_risk_amt is None:
                            staying_entries.append(entry)
                            continue

                        # --- NEW LOGIC: VALIDATE BUCKET BOUNDARIES ---
                        # Correct bucket for $1.50 is bucket 1. Correct for $2.10 is bucket 2.
                        correct_bucket_val = int(math.floor(live_risk_amt))
                        
                        # Check if order is in the wrong place (too high OR too low)
                        is_violation = correct_bucket_val != current_bucket_limit

                        if not is_violation:
                            # Order is exactly where it belongs
                            staying_entries.append(entry)
                        else:
                            # --- MOVE LOGIC ---
                            raw_symbol = entry.get("symbol")
                            symbol = get_normalized_symbol(raw_symbol, norm_map)
                            
                            # Log the reason
                            direction = "OVERFLOW" if live_risk_amt >= (current_bucket_limit + 1) else "UNDERFLOW"
                            new_bucket_name = f"{correct_bucket_val}usd_risk"
                            

                            # File Migration Logic
                            parent_rr_folder = target_file_path.parent.parent
                            new_dir = parent_rr_folder / new_bucket_name
                            new_dir.mkdir(parents=True, exist_ok=True)
                            new_file_path = new_dir / f"{new_bucket_name}.json"

                            # Load or create destination data
                            dest_data = []
                            if new_file_path.exists():
                                try:
                                    with open(new_file_path, 'r', encoding='utf-8') as nf:
                                        dest_data = json.load(nf)
                                except: pass
                            
                            dest_data.append(entry)
                            with open(new_file_path, 'w', encoding='utf-8') as nf:
                                json.dump(dest_data, nf, indent=4)
                            
                            file_changed = True
                            investor_orders_moved += 1
                            total_moved_orders += 1

                    # Save updated source file (removing moved orders)
                    if file_changed:
                        with open(target_file_path, 'w', encoding='utf-8') as f:
                            json.dump(staying_entries, f, indent=4)
                        investor_files_updated += 1

                except Exception as e:
                    print(f"    [!] Error processing file {target_file_path}: {e}")
                    total_errors_encountered += 1

        except Exception as e:
            print(f"    [!] CRASH for {inv_id}: {e}")
            total_errors_encountered += 1
        
        mt5.shutdown()
        print(f"<<< [FINISHED: {inv_id}] Updated {investor_files_updated} files, moved {investor_orders_moved} orders.")
        total_files_processed += investor_files_updated
        
    print(f"\n=== BUCKET REPAIR COMPLETE: Processed {total_files_processed} files, moved {total_moved_orders} orders, {total_errors_encountered} errors ===")
    return True

def enrich_orphanage_buckets():
    """
    Function 3: Creates volume-varied copies of orders across all possible risk buckets.
    Handles decimal buckets (e.g., 0.5) and integer buckets (e.g., 10).
    """
    total_files_processed = 0
    total_errors_encountered = 0
    start_time = time.time()

    print("ENRICHING & CLEANING ORPHANAGE BUCKETS - GLOBAL START")
    
    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
        print(f"[✓] Loaded normalization map with {len(norm_map)} symbols")
    except Exception as e:
        print(f"[✗] CRITICAL ERROR: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        broker_name = broker_cfg.get("SERVER", "UnknownBroker")
        
        inv_files_handled = 0
        inv_misfits_removed = 0
        inv_copies_generated = 0

        print(f"\n>>> STARTING INVESTOR: {inv_id} ({broker_name})")
        
        account_mgmt_path = inv_root / "accountmanagement.json"
        available_risks = []
        
        try:
            with open(account_mgmt_path, 'r') as f:
                account_data = json.load(f)
                available_risks = sorted([float(r) for r in account_data.get("RISKS", [])])
        except Exception as e:
            print(f"    [✗] Could not load accountmanagement.json: {e}")
            total_errors_encountered += 1
            continue

        if not available_risks:
            print(f"    [!] No risk buckets defined. Skipping.")
            continue

        # Dynamic range calculation: Step is the distance to the next bucket, default to 1.0
        max_allowed_risk = max(available_risks)

        if not mt5.initialize(
            path=broker_cfg["TERMINAL_PATH"], 
            login=int(broker_cfg["LOGIN_ID"]), 
            password=broker_cfg["PASSWORD"], 
            server=broker_cfg["SERVER"]
        ):
            print(f"    [✗] CONNECTION FAILED for terminal.")
            total_errors_encountered += 1
            continue

        try:
            search_pattern = "**/risk_reward_*/*usd_risk/*usd_risk.json"
            found_files = list(inv_root.rglob(search_pattern))
            
            for source_file_path in found_files:
                try:
                    # FIX: Use float() instead of int() to capture 0.5
                    try:
                        current_bucket_base = float(source_file_path.stem.replace('usd_risk', ''))
                        
                        # Find the next bucket up to define the upper limit
                        next_buckets = [r for r in available_risks if r > current_bucket_base]
                        bucket_max_limit = next_buckets[0] - 0.01 if next_buckets else current_bucket_base + 0.99
                    except:
                        current_bucket_base = 0.0
                        bucket_max_limit = 0.0

                    with open(source_file_path, 'r', encoding='utf-8') as f:
                        entries = json.load(f)

                    original_count = len(entries)
                    valid_entries_for_this_file = []
                    
                    for entry in entries:
                        # 1. VALIDATION
                        symbol = get_normalized_symbol(entry.get("symbol"), norm_map)
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info is None: continue 

                        entry_p = float(entry.get("entry", 0))
                        exit_p = float(entry.get("exit", 0))
                        target_p = float(entry.get("target", 0))
                        vol_key = next((k for k in entry.keys() if k.endswith("_volume")), "volume")
                        current_vol = float(entry.get(vol_key, 0.0))
                        order_type_str = entry.get("order_type", "").lower()
                        calc_type = mt5.ORDER_TYPE_BUY if "buy" in order_type_str else mt5.ORDER_TYPE_SELL

                        live_risk_val = mt5.order_calc_profit(calc_type, symbol, current_vol, entry_p, exit_p)
                        if live_risk_val is None: continue
                        actual_risk_amt = round(abs(live_risk_val), 2)
                        
                        # --- DYNAMIC CLEANUP LOGIC ---
                        if not (current_bucket_base <= actual_risk_amt <= bucket_max_limit):
                            inv_misfits_removed += 1
                            continue
                        
                        valid_entries_for_this_file.append(entry)

                        # 2. ENRICHMENT
                        if entry.get("is_volume_copy", False): continue

                        volume_risk_map = {}
                        vol = symbol_info.volume_min
                        iteration = 0
                        
                        while vol <= symbol_info.volume_max and iteration < 1000:
                            risk_val = mt5.order_calc_profit(calc_type, symbol, vol, entry_p, exit_p)
                            if risk_val is not None:
                                risk_amt = round(abs(risk_val), 2)
                                
                                # Find which bucket this volume belongs to
                                target_bucket = None
                                for i, r in enumerate(available_risks):
                                    upper = available_risks[i+1] if (i+1) < len(available_risks) else r + 1.0
                                    if r <= risk_amt < upper:
                                        target_bucket = r
                                        break
                                
                                if target_bucket is not None:
                                    if target_bucket not in volume_risk_map or vol > volume_risk_map[target_bucket]['volume']:
                                        volume_risk_map[target_bucket] = {
                                            'volume': vol,
                                            'risk': risk_amt,
                                            'reward': round(abs(mt5.order_calc_profit(calc_type, symbol, vol, entry_p, target_p)), 2)
                                        }
                            
                            vol = round(vol + symbol_info.volume_step, 8)
                            iteration += 1

                        # Save Copies
                        parent_rr_folder = source_file_path.parent.parent
                        for target_bucket, vol_data in volume_risk_map.items():
                            # Format bucket name to match file system (e.g., 0.5 -> "0.5", 10.0 -> "10")
                            b_str = str(int(target_bucket)) if target_bucket.is_integer() else str(target_bucket)
                            if target_bucket == current_bucket_base: continue
                            
                            order_copy = entry.copy()
                            order_copy.update({
                                vol_key: vol_data['volume'],
                                f"{broker_name}_sl_risk_amount": vol_data['risk'],
                                f"{broker_name}_tp_reward_amount": vol_data['reward'],
                                "is_volume_copy": True,
                                "parent_bucket": b_str,
                                "enriched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                            target_bucket_name = f"{b_str}usd_risk"
                            target_file_path = parent_rr_folder / target_bucket_name / f"{target_bucket_name}.json"
                            target_file_path.parent.mkdir(parents=True, exist_ok=True)

                            dest_data = []
                            if target_file_path.exists():
                                with open(target_file_path, 'r', encoding='utf-8') as nf:
                                    dest_data = json.load(nf)
                            
                            if not any(ex.get("symbol") == order_copy["symbol"] and 
                                       ex.get("entry") == order_copy["entry"] and 
                                       abs(ex.get(vol_key, 0) - vol_data['volume']) < 0.0001 
                                       for ex in dest_data):
                                dest_data.append(order_copy)
                                with open(target_file_path, 'w', encoding='utf-8') as nf:
                                    json.dump(dest_data, nf, indent=4)
                                inv_copies_generated += 1

                    if len(valid_entries_for_this_file) != original_count:
                        with open(source_file_path, 'w', encoding='utf-8') as f:
                            json.dump(valid_entries_for_this_file, f, indent=4)
                    
                    inv_files_handled += 1

                except Exception as e:
                    print(f"    [✗] Error processing file {source_file_path.name}: {e}")
                    total_errors_encountered += 1

        except Exception as e:
            print(f"    [✗] CRITICAL INVESTOR CRASH for {inv_id}: {e}")
            total_errors_encountered += 1
        
        mt5.shutdown()
    return True

def deduplicate_orders():
    """
    Scans all risk bucket JSON files and removes duplicate orders based on:
    Symbol, Timeframe, Order Type, and Entry Price.
    """

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
    print(f"Deduplication completed")
    return True

def place_usd_orders():
    # --- SUB-FUNCTION 1: DATA INITIALIZATION ---
    def load_normalization_map():
        try:
            with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Critical Error: Could not load normalization map: {e}")
            return None

    # --- SUB-FUNCTION 2: RISK & FILE AGGREGATION ---
    def collect_and_deduplicate_entries(inv_root, risk_map, balance, pull_lower, selected_rr, norm_map):
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
            return None, []

        # Determine which risk levels to scan
        risk_levels = [primary_risk]
        if pull_lower:
            start_lookback = max(1, primary_risk - 9)
            risk_levels = list(range(start_lookback, primary_risk + 1))

        unique_entries_dict = {}
        target_rr_folder = f"risk_reward_{selected_rr}"

        # Process higher risk levels first (precedence logic)
        for r_val in reversed(risk_levels):
            risk_folder_name = f"{r_val}usd_risk"
            risk_filename = f"{r_val}usd_risk.json"
            search_pattern = f"**/{target_rr_folder}/{risk_folder_name}/{risk_filename}"
            
            for path in inv_root.rglob(search_pattern):
                if path.is_file():
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for entry in data:
                                    # Create unique key: timeframe|symbol|type|price
                                    symbol = get_normalized_symbol(entry["symbol"], norm_map)
                                    key = f"{entry.get('timeframe','NA')}|{symbol}|{entry.get('order_type','NA')}|{round(float(entry['entry']), 5)}"
                                    if key not in unique_entries_dict:
                                        unique_entries_dict[key] = entry
                    except Exception: continue
                    break 
        
        return risk_levels, list(unique_entries_dict.values())

    # --- SUB-FUNCTION 3: BROKER CLEANUP (Unauthorized Orders) ---
    def cleanup_unauthorized_orders(all_entries, norm_map):
        current_orders = mt5.orders_get()
        if not current_orders: return

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
                print(f" [-] Deleting unauthorized order: {order.symbol} @ {order.price_open}")
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})

    # --- SUB-FUNCTION 4: ORDER EXECUTION ---
    def execute_missing_orders(all_entries, norm_map, default_magic, selected_rr):
        placed = failed = skipped = 0
        for entry in all_entries:
            symbol = get_normalized_symbol(entry["symbol"], norm_map)
            vol_key = next((k for k in entry.keys() if k.endswith("_volume")), None)
            if not vol_key or mt5.symbol_info(symbol) is None:
                failed += 1
                continue

            # Skip if position exists or limit already exists
            if any(p.symbol == symbol for p in (mt5.positions_get(symbol=symbol) or [])):
                skipped += 1
                continue
            
            entry_p = round(float(entry["entry"]), 5)
            existing_orders = mt5.orders_get(symbol=symbol)
            if existing_orders and any(round(o.price_open, 5) == entry_p for o in existing_orders):
                skipped += 1
                continue

            # Place Order
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": float(entry[vol_key]),
                "type": mt5.ORDER_TYPE_BUY_LIMIT if entry["order_type"] == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": entry_p,
                "sl": float(entry["exit"]),
                "tp": float(entry["target"]),
                "magic": int(entry.get("magic", default_magic)),
                "comment": f"Risk_Agg_RR{selected_rr}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE: placed += 1
            else: failed += 1
        return placed, failed, skipped

    # --- MAIN EXECUTION FLOW ---
    norm_map = load_normalization_map()
    if not norm_map: return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists(): continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract config
            settings = config.get("settings", {})
            pull_lower = settings.get("pull_orders_from_lower", False)
            selected_rr = config.get("selected_risk_reward", [None])[0]
            risk_map = config.get("account_balance_default_risk_management", {})
            default_magic = config.get("magic_number", 123456)

            if not mt5.initialize(path=broker_cfg["TERMINAL_PATH"], login=int(broker_cfg["LOGIN_ID"]), 
                                   password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                continue

            balance = mt5.account_info().balance

            # STAGE 1: Risk determination and file loading
            risk_lvls, all_entries = collect_and_deduplicate_entries(inv_root, risk_map, balance, pull_lower, selected_rr, norm_map)
            
            if not all_entries:
                mt5.shutdown()
                continue

            # STAGE 2: Cleanup unauthorized orders
            cleanup_unauthorized_orders(all_entries, norm_map)

            # STAGE 3: Execute placement
            p, f, s = execute_missing_orders(all_entries, norm_map, default_magic, selected_rr)
            
            print(f"Investor {inv_id} Summary: Placed: {p}, Failed: {f}, Skipped: {s} (Checked {len(risk_lvls)} risk levels)")
            mt5.shutdown()

        except Exception as e:
            print(f"Error processing {inv_id}: {e}")
            mt5.shutdown()

    return True

def check_limit_orders_risk():
    """
    Function to verify that all live limit orders on the broker match the 
    investor's current risk bucket based on their account balance.
    If an order's risk does not align with the current balance bucket, it is canceled.
    """
    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f"Critical Error: Could not load normalization map: {e}")
        return False

    for user_brokerid, broker_cfg in usersdictionary.items():
        inv_id = user_brokerid
        inv_root = Path(INV_PATH) / inv_id
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists():
            print(f"Skipping {inv_id}: accountmanagement.json not found.")
            continue

        # 1. Load Risk Configuration
        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            risk_map = config.get("account_balance_default_risk_management", {})
        except Exception as e:
            print(f"Error reading config for {inv_id}: {e}")
            continue

        # 2. MT5 Init & Login
        if not mt5.initialize(path=broker_cfg["TERMINAL_PATH"], login=int(broker_cfg["LOGIN_ID"]), 
                               password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
            print(f"MT5 Init failed for {inv_id}: {mt5.last_error()}")
            continue

        acc_info = mt5.account_info()
        if not acc_info:
            print(f"Failed to get account info for {inv_id}")
            mt5.shutdown()
            continue

        balance = acc_info.balance

        # 3. Determine Primary Risk Value (Current Bucket)
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
            print(f"No risk mapping found for balance {balance} on {inv_id}")
            mt5.shutdown()
            continue

        print(f">>> [INVESTOR: {inv_id}] Balance: {balance} | Current Risk Bucket: {primary_risk} USD")

        # 4. Check Live Pending Orders
        pending_orders = mt5.orders_get()
        if pending_orders:
            for order in pending_orders:
                # We only care about Limit Orders
                if order.type not in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
                    continue

                # Calculate live risk for this order
                # order_calc_profit requires (action, symbol, volume, open_price, close_price)
                calc_type = mt5.ORDER_TYPE_BUY if order.type == mt5.ORDER_TYPE_BUY_LIMIT else mt5.ORDER_TYPE_SELL
                
                # Risk is the difference between Entry and SL
                sl_profit = mt5.order_calc_profit(calc_type, order.symbol, order.volume_initial, order.price_open, order.sl)
                
                if sl_profit is not None:
                    order_risk_usd = round(abs(sl_profit), 2)
                    
                    # 5. Cancel if order risk does not match the primary risk bucket
                    # Note: You can add a small tolerance here (e.g., +/- 1.0) if broker fees/spreads vary
                    if abs(order_risk_usd - primary_risk) > 1.0: 
                        print(f" [!] RISK MISMATCH: Order {order.ticket} ({order.symbol}) has {order_risk_usd} USD risk.")
                        print(f"     Expected: {primary_risk} USD. Canceling order...")
                        
                        cancel_request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket
                        }
                        result = mt5.order_send(cancel_request)
                        
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"     [X] Failed to cancel order {order.ticket}: {result.comment}")
                        else:
                            print(f"     [V] Order {order.ticket} canceled successfully.")
                else:
                    print(f" [!] Could not calculate risk for order {order.ticket} ({order.symbol})")

        mt5.shutdown()
        print(f"<<< [FINISHED: {inv_id}] Risk check complete.")

    return True

def main():
    sort_orders()
    check_orders_live_volume()
    check_orders_live_risk()
    repair_order_buckets()
    enrich_orphanage_buckets()
    deduplicate_orders()
    place_usd_orders()
    check_limit_orders_risk()


if __name__ == "__main__":
   place_usd_orders()
   check_limit_orders_risk()


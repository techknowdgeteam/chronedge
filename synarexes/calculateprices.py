import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import glob
import MetaTrader5 as mt5
import copy
import math
import shutil

# --- GLOBALS ---
BROKER_DICT_PATH = r"C:\xampp\htdocs\chronedge\synarex\ohlc.json"
USERS_PATH = r"C:\xampp\htdocs\chronedge\synarex\users.json"
SYMBOL_CATEGORY_PATH = r"C:\xampp\htdocs\chronedge\synarex\symbolscategory.json"
DEV_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\developers"


def clean_risk_folders():
    """
    Scans the DEV_PATH and permanently deletes all folders ending in 'usd_risk'.
    This clears out all calculated risk buckets and their contained JSON files.
    """
    print("STARTING RISK FOLDER CLEANUP...")
    
    if not os.path.exists(DEV_PATH):
        print(f"Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    # Find all directories that end with 'usd_risk'
    # We use recursive glob to find them at any depth within user folders
    risk_folders = glob.glob(os.path.join(DEV_PATH, "**", "*usd_risk"), recursive=True)

    if not risk_folders:
        print("No risk folders found to delete.")
        return True

    deleted_count = 0
    for folder_path in risk_folders:
        # Double check it is actually a directory before deleting
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                deleted_count += 1
            except Exception as e:
                print(f" ! Error deleting {folder_path}: {e}")

    print(f"FINISHED CLEANUP: {deleted_count} folders removed.")
    return True

def purge_unauthorized_symbols():
    """
    Iterates through all users, identifies allowed symbols across all categories,
    and removes any orders from limit_orders.json that use unauthorized symbols.
    """
    try:
        # 1. Load User IDs
        if not os.path.exists(USERS_PATH):
            print(f"Users file not found: {USERS_PATH}")
            return False

        with open(USERS_PATH, 'r') as f:
            users_data = json.load(f)

        for user_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, user_broker_id)
            volumes_path = os.path.join(user_folder, "allowedsymbolsandvolumes.json")
            
            # Skip user if they don't have a config file
            if not os.path.exists(volumes_path):
                print(f"Config missing for user {user_broker_id}, skipping purge.")
                continue

            # 2. Extract ALL allowed symbols from all categories (forex, crypto, indices, etc.)
            allowed_symbols = set()
            with open(volumes_path, 'r') as f:
                v_data = json.load(f)
                for category in v_data:
                    for item in v_data[category]:
                        if 'symbol' in item:
                            # We use uppercase for a case-insensitive comparison safety check
                            allowed_symbols.add(item['symbol'].upper())

            # 3. Locate all limit_orders.json files for this user
            limit_order_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)

            for limit_path in limit_order_files:
                with open(limit_path, 'r') as f:
                    orders = json.load(f)

                # 4. Filter the orders: Keep only if the symbol is in the allowed set
                original_count = len(orders)
                purged_orders = [
                    order for order in orders 
                    if order.get('symbol', '').upper() in allowed_symbols
                ]

                # 5. Save back if any orders were removed
                if len(purged_orders) < original_count:
                    with open(limit_path, 'w') as f:
                        json.dump(purged_orders, f, indent=4)
                    
        print(f"Purged unauthorized orders")
        return True

    except Exception as e:
        print(f"Error during symbol purge: {e}")
        return False

def backup_limit_orders():
    """
    Traverses all user folders and creates a copy of every 'limit_orders.json' 
    file named 'limit_orders_backup.json' in the same directory.
    """
    try:
        # 1. Load User IDs to know which folders to scan
        if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
            print(f"Users file not found or empty: {USERS_PATH}")
            return False
            
        with open(USERS_PATH, 'r') as f:
            users_data = json.load(f)
        
        backup_count = 0
        
        for user_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, user_broker_id)
            
            # 2. Find all limit_orders.json files recursively
            # We search for the exact filename to avoid backing up backups
            limit_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)
            
            for limit_path in limit_files:
                # Skip secondary files that aren't primary limit order records
                if "risk_reward_" in limit_path or "limit_orders_backup.json" in limit_path:
                    continue
                
                if not os.path.exists(limit_path) or os.path.getsize(limit_path) == 0:
                    continue
                
                # 3. Define the destination path in the same folder
                directory = os.path.dirname(limit_path)
                backup_path = os.path.join(directory, "limit_orders_backup.json")
                
                try:
                    # Use copy2 to preserve metadata (timestamps, etc.)
                    shutil.copy2(limit_path, backup_path)
                    backup_count += 1
                except Exception as e:
                    print(f"Failed to backup {limit_path}: {e}")

        return True

    except Exception as e:
        print(f"Critical Error during backup: {e}")
        return False
    
def enforce_risk():
    def enforce_risk_v2():
        try:
            if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
                return False
                
            with open(USERS_PATH, 'r') as f:
                users_data = json.load(f)
            
            for user_broker_id in users_data.keys():
                user_folder = os.path.join(DEV_PATH, user_broker_id)
                
                # 1. Identify all config files
                acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
                secondary_configs = []
                if os.path.exists(acc_mgmt_path):
                    with open(acc_mgmt_path, 'r') as f:
                        acc_data = json.load(f)
                    poi_conditions = acc_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
                    for app_val in poi_conditions.values():
                        if isinstance(app_val, dict):
                            for ent_val in app_val.values():
                                if isinstance(ent_val, dict) and ent_val.get("new_filename"):
                                    secondary_configs.append(os.path.join(user_folder, ent_val["new_filename"], "allowedsymbolsandvolumes.json"))

                all_configs = [os.path.join(user_folder, "allowedsymbolsandvolumes.json")] + secondary_configs
                
                # 2. Build the Master Lookup (Symbol -> Timeframe -> Risk)
                risk_master_data = {}
                for config_path in all_configs:
                    if not os.path.exists(config_path): continue
                    with open(config_path, 'r') as f:
                        c_data = json.load(f)
                    
                    for category_list in c_data.values():
                        if not isinstance(category_list, list): continue
                        for item in category_list:
                            sym = item.get("symbol")
                            if not sym: continue
                            
                            if sym not in risk_master_data: 
                                risk_master_data[sym] = {}
                            
                            # Extract specs (e.g., "5m_specs" -> "5m")
                            for key, val in item.items():
                                if "_specs" in key and isinstance(val, dict):
                                    tf_clean = key.replace("_specs", "")
                                    risk_val = val.get("usd_risk", 0)
                                    # Only store if it's a valid non-zero risk
                                    if risk_val > 0:
                                        risk_master_data[sym][tf_clean] = risk_val

                # 3. Apply Enforcement based on Record Integrity
                limit_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)
                for limit_path in limit_files:
                    if "risk_reward_" in limit_path or not os.path.exists(limit_path): continue
                    
                    with open(limit_path, 'r') as f:
                        orders = json.load(f)
                    
                    modified = False
                    for order in orders:
                        sym, tf = order.get('symbol'), order.get('timeframe')
                        
                        # TRIGGER: Both exit and target are essentially empty/zero
                        is_missing_targets = order.get('exit') in [0, "0", None] and \
                                            order.get('target') in [0, "0", None]
                        
                        if is_missing_targets:
                            # Attempt to grab risk from master table
                            found_risk = risk_master_data.get(sym, {}).get(tf)
                            
                            if found_risk:
                                order['exit'] = 0
                                order['target'] = 0
                                order['usd_risk'] = found_risk
                                order['usd_based_risk_only'] = True
                                modified = True
                                
                            else:
                                print(f"❌ Failed: {sym} ({tf}) has no exit/target but no usd_risk found in configs.")

                    if modified:
                        with open(limit_path, 'w') as f:
                            json.dump(orders, f, indent=4)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False     
    enforce_risk_v2()
    try:
        # 1. Load User IDs
        if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
            print(f"Users file not found or empty: {USERS_PATH}")
            return False
            
        with open(USERS_PATH, 'r') as f:
            users_data = json.load(f)
        
        for user_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, user_broker_id)
            
            # --- Get Config Paths (Primary + Secondary from Account Management) ---
            acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
            secondary_configs = []
            if os.path.exists(acc_mgmt_path):
                with open(acc_mgmt_path, 'r') as f:
                    acc_data = json.load(f)
                poi_conditions = acc_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
                for app_val in poi_conditions.values():
                    if isinstance(app_val, dict):
                        for ent_val in app_val.values():
                            if isinstance(ent_val, dict) and ent_val.get("new_filename"):
                                secondary_configs.append(os.path.join(user_folder, ent_val["new_filename"], "allowedsymbolsandvolumes.json"))

            all_config_files = [os.path.join(user_folder, "allowedsymbolsandvolumes.json")] + secondary_configs
            
            for config_path in all_config_files:
                if not os.path.exists(config_path): continue
                
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # --- Build Universal Lookup Table ---
                risk_lookup = {}
                
                # Iterate through all categories (forex, crypto, etc.)
                for category in config_data.values():
                    if not isinstance(category, list): continue
                    
                    for item in category:
                        symbol = item.get("symbol")
                        if not symbol: continue
                        
                        if symbol not in risk_lookup: risk_lookup[symbol] = {}
                        
                        for key, value in item.items():
                            if key.endswith("_specs") and isinstance(value, dict):
                                tf = key.replace("_specs", "")
                                is_enforced = str(value.get("enforce_usd_risk", "no")).lower() == "yes"
                                risk_val = value.get("usd_risk", 0)
                                
                                risk_lookup[symbol][tf] = {
                                    "enforce": is_enforced,
                                    "usd_risk": risk_val
                                }

                # --- Apply to Limit Orders ---
                config_dir = os.path.dirname(config_path)
                limit_files = glob.glob(os.path.join(config_dir, "**", "limit_orders.json"), recursive=True)
                
                for limit_path in limit_files:
                    if "risk_reward_" in limit_path: continue
                    if not os.path.exists(limit_path) or os.path.getsize(limit_path) == 0: continue
                    
                    with open(limit_path, 'r') as f:
                        orders = json.load(f)
                    
                    modified = False
                    for order in orders:
                        sym, tf = order.get('symbol'), order.get('timeframe')
                        
                        if sym in risk_lookup and tf in risk_lookup[sym]:
                            rule = risk_lookup[sym][tf]
                            
                            if rule["enforce"]:
                                # 1. Set exit/target to zero
                                order['exit'] = 0
                                order['target'] = 0
                                
                                # 2. Extract and assign the risk value
                                order['usd_risk'] = rule["usd_risk"]
                                
                                # 3. Tag the record for system awareness
                                order['usd_based_risk_only'] = True
                                
                                modified = True
                    
                    if modified:
                        with open(limit_path, 'w') as f:
                            json.dump(orders, f, indent=4)
        print(f"Usd risk Based completed")
        return True
    except Exception as e:
        print(f"Critical Error: {e}")
        return False

def calculate_symbols_orderss():
    try:
        # 1. Load User IDs
        if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
            print(f"Users file not found: {USERS_PATH}")
            return False
            
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
            
        # 2. Load Global Symbols (Flatten all categories into one set)
        if not os.path.exists(SYMBOL_CATEGORY_PATH) or os.path.getsize(SYMBOL_CATEGORY_PATH) == 0:
            print(f"Symbol category file not found: {SYMBOL_CATEGORY_PATH}")
            return False

        all_valid_symbols = set()
        with open(SYMBOL_CATEGORY_PATH, 'r', encoding='utf-8') as f:
            categories = json.load(f)
            for category_list in categories.values():
                for sym in category_list:
                    all_valid_symbols.add(sym.upper())

        # 3. Iterate through each User
        for user_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, user_broker_id)
            acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
            primary_volumes_path = os.path.join(user_folder, "allowedsymbolsandvolumes.json")

            if not os.path.exists(acc_mgmt_path):
                continue

            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                acc_mgmt_data = json.load(f)
            
            rr_ratios = acc_mgmt_data.get("risk_reward_ratios", [1.0])
            poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
            
            # --- Handle Secondary Config Directories (Automation) ---
            secondary_paths = []
            for apprehend_val in poi_conditions.values():
                if isinstance(apprehend_val, dict):
                    for entry_val in apprehend_val.values():
                        if isinstance(entry_val, dict) and entry_val.get("new_filename"):
                            target_dir = os.path.join(user_folder, entry_val["new_filename"])
                            secondary_file = os.path.join(target_dir, "allowedsymbolsandvolumes.json")
                            if not os.path.exists(secondary_file) and os.path.exists(primary_volumes_path):
                                os.makedirs(target_dir, exist_ok=True)
                                shutil.copy2(primary_volumes_path, secondary_file)
                            secondary_paths.append(secondary_file)

            all_config_files = list(set([primary_volumes_path] + secondary_paths))

            for volumes_path in all_config_files:
                if not os.path.exists(volumes_path): continue

                # Dynamically load all symbols from allowedsymbolsandvolumes.json regardless of key
                user_config = {}
                with open(volumes_path, 'r', encoding='utf-8') as f:
                    v_data = json.load(f)
                    for category_key in v_data: 
                        if isinstance(v_data[category_key], list):
                            for item in v_data[category_key]:
                                user_config[item['symbol'].upper()] = item

                config_folder = os.path.dirname(volumes_path)
                limit_order_files = glob.glob(os.path.join(config_folder, "**", "limit_orders.json"), recursive=True)

                for limit_path in limit_order_files:
                    if "risk_reward_" in limit_path: continue
                    with open(limit_path, 'r', encoding='utf-8') as f:
                        original_orders = json.load(f)

                    base_dir = os.path.dirname(limit_path)

                    for current_rr in rr_ratios:
                        orders_copy = copy.deepcopy(original_orders)
                        updated = False

                        for order in orders_copy:
                            symbol = order.get('symbol', '').upper()
                            if symbol not in all_valid_symbols: continue

                            try:
                                # Data Extraction
                                entry = float(order.get('entry', 0))
                                rr_ratio = float(current_rr)
                                order_type = order.get('order_type', '').upper()
                                tick_size = float(order.get('tick_size', 0.00001))
                                tick_value = float(order.get('tick_value', 0))
                                tf = order.get('timeframe', '1h')
                                
                                # --- DYNAMIC PRECISION LOGIC ---
                                # Converts tick_size 0.00001 to 5, or 0.1 to 1, or 1.0 to 0
                                if tick_size < 1:
                                    digits = len(str(tick_size).split('.')[-1])
                                else:
                                    digits = 0
                                
                                tf_specs = user_config.get(symbol, {}).get(f"{tf}_specs", {})
                                volume = float(tf_specs.get('volume', 0.01))
                                
                                # --- LOGIC BRANCH A: USD RISK BASED ---
                                if order.get("usd_based_risk_only") is True:
                                    risk_val = float(order.get("usd_risk", tf_specs.get("usd_risk", 0)))
                                    
                                    if risk_val > 0 and tick_value > 0:
                                        # Pip size is usually 10 ticks in Forex, but 1 tick in many indices.
                                        # For a truly universal approach, we use the Tick Value directly.
                                        # Risk = (Distance / TickSize) * TickValue * Volume
                                        # Distance = (Risk * TickSize) / (TickValue * Volume)
                                        
                                        sl_dist = (risk_val * tick_size) / (tick_value * volume)
                                        tp_dist = sl_dist * rr_ratio

                                        if "BUY" in order_type:
                                            order["exit"] = round(entry - sl_dist, digits)
                                            order["target"] = round(entry + tp_dist, digits)
                                        else:
                                            order["exit"] = round(entry + sl_dist, digits)
                                            order["target"] = round(entry - tp_dist, digits)

                                # --- LOGIC BRANCH B: DISTANCE BASED ---
                                else:
                                    sl_price = float(order.get('exit', 0))
                                    tp_price = float(order.get('target', 0))

                                    if sl_price == 0 and tp_price > 0:
                                        tp_dist = abs(tp_price - entry)
                                        risk_dist = tp_dist / rr_ratio
                                        order['exit'] = round(entry - risk_dist if "BUY" in order_type else entry + risk_dist, digits)
                                    
                                    elif sl_price > 0:
                                        risk_dist = abs(entry - sl_price)
                                        order['target'] = round(entry + (risk_dist * rr_ratio) if "BUY" in order_type else entry - (risk_dist * rr_ratio), digits)

                                # Finalize
                                order.pop("usd_risk", None) # Remove if exists
                                order[f"{tf}_volume"] = volume
                                order['risk_reward'] = rr_ratio
                                order['status'] = "Calculated"
                                order['calculated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                updated = True

                            except Exception as e:
                                print(f"Error calculating {symbol}: {e}")
                                continue

                        if updated:
                            target_out_dir = os.path.join(base_dir, f"risk_reward_{current_rr}")
                            os.makedirs(target_out_dir, exist_ok=True)
                            with open(os.path.join(target_out_dir, "limit_orders.json"), 'w', encoding='utf-8') as f:
                                json.dump(orders_copy, f, indent=4)
            
            print(f"Processed: {user_broker_id}")

        return True
    except Exception as e:
        print(f"Critical Error: {e}")
        return False

def calculate_symbols_orders():
    try:
        # 1. Load User IDs
        if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
            print(f"Users file not found: {USERS_PATH}")
            return False
            
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
            
        # 2. Load Global Symbols (Flatten every category into one master set)
        all_valid_symbols = set()
        if os.path.exists(SYMBOL_CATEGORY_PATH) and os.path.getsize(SYMBOL_CATEGORY_PATH) > 0:
            with open(SYMBOL_CATEGORY_PATH, 'r', encoding='utf-8') as f:
                categories = json.load(f)
                for category_content in categories.values():
                    if isinstance(category_content, list):
                        for sym in category_content:
                            # Normalize: "us oil" -> "US OIL"
                            all_valid_symbols.add(str(sym).strip().upper())

        # 3. Iterate through each User
        for user_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, user_broker_id)
            acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
            primary_volumes_path = os.path.join(user_folder, "allowedsymbolsandvolumes.json")

            if not os.path.exists(acc_mgmt_path):
                continue

            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                acc_mgmt_data = json.load(f)
            
            rr_ratios = acc_mgmt_data.get("risk_reward_ratios", [1.0])
            poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
            
            # --- Auto-generate secondary folders if they don't exist ---
            secondary_paths = []
            for apprehend_val in poi_conditions.values():
                if isinstance(apprehend_val, dict):
                    for entry_val in apprehend_val.values():
                        if isinstance(entry_val, dict) and entry_val.get("new_filename"):
                            target_dir = os.path.join(user_folder, entry_val["new_filename"])
                            secondary_file = os.path.join(target_dir, "allowedsymbolsandvolumes.json")
                            if not os.path.exists(secondary_file) and os.path.exists(primary_volumes_path):
                                os.makedirs(target_dir, exist_ok=True)
                                shutil.copy2(primary_volumes_path, secondary_file)
                            secondary_paths.append(secondary_file)

            all_config_files = list(set([primary_volumes_path] + secondary_paths))

            for volumes_path in all_config_files:
                if not os.path.exists(volumes_path): continue

                # --- UNIVERSAL CONFIG LOAD: Bypass Category Names ---
                user_config = {}
                with open(volumes_path, 'r', encoding='utf-8') as f:
                    v_data = json.load(f)
                    # Loop through EVERY key (forex, metals, energies, etc.)
                    for category_key in v_data:
                        symbol_list = v_data[category_key]
                        if isinstance(symbol_list, list):
                            for item in symbol_list:
                                sym_name = str(item.get('symbol', '')).strip().upper()
                                user_config[sym_name] = item

                config_folder = os.path.dirname(volumes_path)
                limit_order_files = glob.glob(os.path.join(config_folder, "**", "limit_orders.json"), recursive=True)

                for limit_path in limit_order_files:
                    if "risk_reward_" in limit_path: continue
                    with open(limit_path, 'r', encoding='utf-8') as f:
                        original_orders = json.load(f)

                    base_dir = os.path.dirname(limit_path)

                    for current_rr in rr_ratios:
                        orders_copy = copy.deepcopy(original_orders)
                        updated_any_order = False

                        for order in orders_copy:
                            symbol = str(order.get('symbol', '')).strip().upper()
                            
                            # Ensure symbol exists in our master flattened list
                            if symbol not in user_config:
                                continue

                            try:
                                # Data Extraction
                                entry = float(order.get('entry', 0))
                                rr_ratio = float(current_rr)
                                order_type = order.get('order_type', '').upper()
                                tick_size = float(order.get('tick_size', 0.00001))
                                tick_value = float(order.get('tick_value', 0))
                                tf = order.get('timeframe', '1h')
                                
                                # --- DYNAMIC PRECISION LOGIC ---
                                # Determines decimals based on tick_size (e.g., 0.00001 -> 5, 1.0 -> 0)
                                if tick_size < 1:
                                    # Count decimals by splitting at dot
                                    digits = len(str(tick_size).split('.')[-1])
                                else:
                                    digits = 0
                                
                                # Fetch specs from our flattened config
                                tf_specs = user_config.get(symbol, {}).get(f"{tf}_specs", {})
                                volume = float(tf_specs.get('volume', 0.01))
                                
                                # --- LOGIC BRANCH A: USD RISK BASED ---
                                if order.get("usd_based_risk_only") is True:
                                    risk_val = float(order.get("usd_risk", tf_specs.get("usd_risk", 0)))
                                    
                                    if risk_val > 0 and tick_value > 0:
                                        # Universal Formula: Distance = (Risk * TickSize) / (TickValue * Volume)
                                        sl_dist = (risk_val * tick_size) / (tick_value * volume)
                                        tp_dist = sl_dist * rr_ratio

                                        if "BUY" in order_type:
                                            order["exit"] = round(entry - sl_dist, digits)
                                            order["target"] = round(entry + tp_dist, digits)
                                        else:
                                            order["exit"] = round(entry + sl_dist, digits)
                                            order["target"] = round(entry - tp_dist, digits)

                                # --- LOGIC BRANCH B: DISTANCE BASED ---
                                else:
                                    sl_price = float(order.get('exit', 0))
                                    tp_price = float(order.get('target', 0))

                                    if sl_price == 0 and tp_price > 0:
                                        tp_dist = abs(tp_price - entry)
                                        risk_dist = tp_dist / rr_ratio
                                        order['exit'] = round(entry - risk_dist if "BUY" in order_type else entry + risk_dist, digits)
                                    
                                    elif sl_price > 0:
                                        risk_dist = abs(entry - sl_price)
                                        order['target'] = round(entry + (risk_dist * rr_ratio) if "BUY" in order_type else entry - (risk_dist * rr_ratio), digits)

                                # Meta-data and Cleanup
                                order.pop("usd_risk", None) 
                                order[f"{tf}_volume"] = volume
                                order['risk_reward'] = rr_ratio
                                order['status'] = "Calculated"
                                order['calculated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                updated_any_order = True

                            except Exception as e:
                                print(f"Error calculating {symbol}: {e}")
                                continue

                        if updated_any_order:
                            target_out_dir = os.path.join(base_dir, f"risk_reward_{current_rr}")
                            os.makedirs(target_out_dir, exist_ok=True)
                            with open(os.path.join(target_out_dir, "limit_orders.json"), 'w', encoding='utf-8') as f:
                                json.dump(orders_copy, f, indent=4)
            
            print(f"Processed: {user_broker_id}")

        return True
    except Exception as e:
        print(f"Critical Error: {e}")
        return False

def live_risk_reward_amounts_and_scale():
    if not os.path.exists(BROKER_DICT_PATH):
        print(f"Error: Broker dictionary not found.")
        return False

    with open(BROKER_DICT_PATH, 'r') as f:
        try:
            broker_configs = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {BROKER_DICT_PATH} is empty or invalid JSON.")
            return False

    print("STARTING LIVE RISK CALCULATION...")

    for user_broker_id, config in broker_configs.items():
        user_folder = os.path.join(DEV_PATH, user_broker_id)
        
        # 1. Load Account Management
        acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
        if not os.path.exists(acc_mgmt_path):
            print(f" ! Skipping {user_broker_id}: accountmanagement.json not found.")
            continue

        with open(acc_mgmt_path, 'r') as f:
            try:
                acc_mgmt_data = json.load(f)
            except json.JSONDecodeError:
                print(f" ! Error: {acc_mgmt_path} is invalid.")
                continue

        allowed_risks = acc_mgmt_data.get("RISKS", [])
        max_allowed_risk = max(allowed_risks) if allowed_risks else 50.00

        # 2. Identify all Secondary "new_filename" paths
        poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
        
        target_subfolders = [user_folder]  # Start with Primary folder
        for apprehend_key, apprehend_val in poi_conditions.items():
            if apprehend_key.startswith("apprehend_") and isinstance(apprehend_val, dict):
                for entry_key, entry_val in apprehend_val.items():
                    if entry_key.startswith("entry_") and isinstance(entry_val, dict):
                        new_filename = entry_val.get("new_filename")
                        if new_filename:
                            subfolder_path = os.path.join(user_folder, new_filename)
                            if os.path.exists(subfolder_path):
                                target_subfolders.append(subfolder_path)

        # 3. Initialize MT5 Connection
        TERMINAL_PATH = config.get("TERMINAL_PATH", "")
        LOGIN_ID = config.get("LOGIN_ID")
        PASSWORD = config.get("PASSWORD")
        SERVER = config.get("SERVER")

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            print(f" ! MT5 Init failed for {user_broker_id}")
            continue

        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            continue
        
        acc_currency = account_info.currency
        print(f" > Processing {user_broker_id.upper()} (Max Risk: {max_allowed_risk})...")

        # 4. Iterate through Primary and all Secondary subfolders
        for current_search_path in target_subfolders:
            limit_order_files = glob.glob(os.path.join(current_search_path, "**", "limit_orders.json"), recursive=True)

            for limit_path in limit_order_files:
                # Skip folders created by this script or the logic folder
                if "risk_reward_" not in limit_path or "_risk" in limit_path:
                    continue

                if not os.path.exists(limit_path) or os.path.getsize(limit_path) == 0:
                    continue

                with open(limit_path, 'r') as f:
                    try:
                        orders = json.load(f)
                    except json.JSONDecodeError:
                        continue

                risk_buckets = {}

                for order in orders:
                    if order.get('status') != "Calculated":
                        continue

                    symbol = order.get('symbol')
                    entry = float(order.get('entry', 0))
                    exit_price = float(order.get('exit', 0))
                    target_price = float(order.get('target', 0))
                    tf = order.get('timeframe', '1h')
                    
                    info = mt5.symbol_info(symbol)
                    if info is None: continue
                    
                    filled_buckets_for_this_order = set()
                    base_volume = float(order.get(f"{tf}_volume", 0.01))
                    current_volume = base_volume
                    max_volume_iterations = 5000
                    iteration_count = 0
                    
                    while iteration_count < max_volume_iterations:
                        action = mt5.ORDER_TYPE_BUY if "BUY" in order['order_type'].upper() else mt5.ORDER_TYPE_SELL
                        sl_risk = mt5.order_calc_profit(action, symbol, current_volume, entry, exit_price)
                        tp_reward = mt5.order_calc_profit(action, symbol, current_volume, entry, target_price)

                        # Fallback calculation if MT5 returns None
                        if sl_risk is None:
                            sl_risk = -(abs(entry - exit_price) / info.point * (info.trade_contract_size * info.point) * current_volume)
                        if tp_reward is None:
                            tp_reward = abs(target_price - entry) / info.point * (info.trade_contract_size * info.point) * current_volume

                        abs_risk = round(abs(sl_risk), 2)
                        
                        if abs_risk > max_allowed_risk: 
                            break
                        
                        assigned_risk_bucket = None
                        if abs_risk < 1.00:
                            if 0.5 in allowed_risks: assigned_risk_bucket = 0.5
                        else:
                            floored_risk = int(math.floor(abs_risk))
                            if floored_risk in allowed_risks: assigned_risk_bucket = floored_risk
                        
                        if assigned_risk_bucket is not None and assigned_risk_bucket not in filled_buckets_for_this_order:
                            if assigned_risk_bucket not in risk_buckets:
                                risk_buckets[assigned_risk_bucket] = []
                            
                            order_copy = order.copy()
                            order_copy[f"{tf}_volume"] = round(current_volume, 2)
                            order_copy['live_sl_risk_amount'] = abs_risk
                            order_copy['live_tp_reward_amount'] = round(abs(tp_reward), 2)
                            order_copy['account_currency'] = acc_currency
                            
                            risk_buckets[assigned_risk_bucket].append(order_copy)
                            filled_buckets_for_this_order.add(assigned_risk_bucket)
                        
                        current_volume += 0.01
                        iteration_count += 1
                
                # 5. Save the scaled orders into risk-specific buckets
                base_rr_folder = os.path.dirname(limit_path)
                for risk_val, grouped_orders in risk_buckets.items():
                    target_dir = os.path.join(base_rr_folder, f"{risk_val}usd_risk")
                    os.makedirs(target_dir, exist_ok=True)
                    output_path = os.path.join(target_dir, f"{risk_val}usd_risk.json")
                    
                    existing_orders = []
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        try:
                            with open(output_path, 'r') as f:
                                existing_orders = json.load(f)
                        except json.JSONDecodeError:
                            existing_orders = []
                    
                    with open(output_path, 'w') as f:
                        json.dump(existing_orders + grouped_orders, f, indent=4)
        
        mt5.shutdown()

    print("FINISHED: ALL DIRECTORIES MAPPED.")
    return True

def usd_based_risk_scaling():
    print("PROCESSING PRICE RE-ADJUSTMENT PROMOTION...")
    
    if not os.path.exists(DEV_PATH):
        print(f"Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    # Get all primary user directories
    user_folders = [f.path for f in os.scandir(DEV_PATH) if f.is_dir()]

    for user_folder in user_folders:
        acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
        
        # 1. Load Account Management to find strategy subfolders
        if not os.path.exists(acc_mgmt_path) or os.path.getsize(acc_mgmt_path) == 0:
            continue

        with open(acc_mgmt_path, 'r') as f:
            try:
                acc_mgmt_data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        allowed_risks = acc_mgmt_data.get("RISKS", [])
        if not allowed_risks:
            continue

        # 2. Identify all strategy-specific paths (new_filename)
        poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
        
        target_search_paths = [user_folder]  # Always include the primary folder
        for app_key, app_val in poi_conditions.items():
            if app_key.startswith("apprehend_") and isinstance(app_val, dict):
                for ent_key, ent_val in app_val.items():
                    if ent_key.startswith("entry_") and isinstance(ent_val, dict):
                        new_filename = ent_val.get("new_filename")
                        if new_filename:
                            strat_path = os.path.join(user_folder, new_filename)
                            if os.path.exists(strat_path):
                                target_search_paths.append(strat_path)

        # 3. Process each folder (Primary + All new_filenames)
        for search_root in target_search_paths:
            # Locate all bucketed risk files within this specific scope
            risk_json_files = glob.glob(os.path.join(search_root, "**", "*usd_risk", "*usd_risk.json"), recursive=True)

            for file_path in risk_json_files:
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    continue

                with open(file_path, 'r') as f:
                    try:
                        orders = json.load(f)
                    except json.JSONDecodeError:
                        continue

                if not isinstance(orders, list):
                    continue

                for order in orders:
                    try:
                        sl_risk = order.get('live_sl_risk_amount', 0)
                        if sl_risk == 0: continue
                        
                        fractional_part = sl_risk - int(sl_risk)

                        # Promotion Logic (e.g., 1.55 becomes 2.0)
                        if fractional_part >= 0.51:
                            target_risk = float(math.ceil(sl_risk))
                            
                            if target_risk not in allowed_risks:
                                continue 

                            # Extract data for re-calculation
                            entry = float(order['entry'])
                            rr_ratio = float(order['risk_reward'])
                            tick_size = float(order['tick_size'])
                            tick_value = float(order['tick_value'])
                            tf = order['timeframe']
                            volume = float(order[f"{tf}_volume"])
                            order_type = order['order_type'].upper()

                            if tick_value == 0 or volume == 0: continue

                            # Determine decimal precision based on tick_size
                            tick_str = format(tick_size, 'f').rstrip('0').rstrip('.')
                            precision = len(tick_str.split('.')[1]) if '.' in tick_str else 0

                            # Core Calculation: Re-space SL/TP to match higher risk amount
                            risk_dist = (target_risk / (tick_value * volume)) * tick_size
                            new_order = copy.deepcopy(order)
                            
                            if "BUY" in order_type:
                                new_order['exit'] = round(entry - risk_dist, precision)
                                new_order['target'] = round(entry + (risk_dist * rr_ratio), precision)
                            else:
                                new_order['exit'] = round(entry + risk_dist, precision)
                                new_order['target'] = round(entry - (risk_dist * rr_ratio), precision)

                            # Update Metadata
                            new_order['live_sl_risk_amount'] = target_risk
                            new_order['live_tp_reward_amount'] = round(target_risk * rr_ratio, 2)
                            new_order['status'] = "Adjusted_To_Next_Bucket"
                            new_order['adjusted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Determine destination (relative to the strategy folder being processed)
                            parent_rr_dir = os.path.dirname(os.path.dirname(file_path))
                            new_bucket_folder = os.path.join(parent_rr_dir, f"{int(target_risk)}usd_risk")
                            os.makedirs(new_bucket_folder, exist_ok=True)
                            
                            target_json_path = os.path.join(new_bucket_folder, f"{int(target_risk)}usd_risk.json")

                            # Merge into the higher bucket
                            existing_data = []
                            if os.path.exists(target_json_path) and os.path.getsize(target_json_path) > 0:
                                with open(target_json_path, 'r') as tf_file:
                                    try:
                                        existing_data = json.load(tf_file)
                                    except json.JSONDecodeError:
                                        existing_data = []
                            
                            existing_data.append(new_order)
                            
                            with open(target_json_path, 'w') as tf_file:
                                json.dump(existing_data, tf_file, indent=4)

                    except (KeyError, ValueError, ZeroDivisionError, TypeError):
                        continue

    print("FINISHED RE-ADJUSTMENT PROMOTION.")
    return True

def deduplicate_risk_bucket_orders():
    print("STARTING DEDUPLICATION OF RISK BUCKETS...")
    
    if not os.path.exists(DEV_PATH):
        print(f"Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    risk_json_files = glob.glob(os.path.join(DEV_PATH, "**", "*usd_risk", "*usd_risk.json"), recursive=True)

    for file_path in risk_json_files:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            continue

        with open(file_path, 'r') as f:
            try:
                orders = json.load(f)
            except json.JSONDecodeError:
                continue

        if not isinstance(orders, list) or not orders:
            continue

        initial_count = len(orders)
        
        # Dictionary to store the 'best' order for each unique category
        # Key: (symbol, timeframe, order_type)
        best_orders = {}

        for order in orders:
            symbol = order.get('symbol')
            tf = order.get('timeframe', '1h')
            # Normalize order type to catch "buy_limit" vs "BUY_LIMIT"
            direction = order.get('order_type', '').upper()
            risk_amt = float(order.get('live_sl_risk_amount', 0))

            # Define the unique group: Symbol + Timeframe + Direction
            group_key = (symbol, tf, direction)

            if group_key not in best_orders:
                # First time seeing this combination
                best_orders[group_key] = order
            else:
                # If we already have one, keep the one with the LOWER risk amount
                existing_risk = float(best_orders[group_key].get('live_sl_risk_amount', 0))
                if risk_amt < existing_risk:
                    best_orders[group_key] = order

        # Convert dictionary back to list
        unique_orders = list(best_orders.values())

        # Only write back if duplicates/higher risks were removed
        if len(unique_orders) < initial_count:
            try:
                with open(file_path, 'w') as f:
                    json.dump(unique_orders, f, indent=4)
            except Exception as e:
                print(f" ! Failed to save cleaned file {file_path}: {e}")

    print("FINISHED DEDUPLICATION.")
    return True
    
def calculate_orders():
    purge_unauthorized_symbols()
    clean_risk_folders()
    backup_limit_orders()
    enforce_risk()
    calculate_symbols_orders()
    live_risk_reward_amounts_and_scale()
    usd_based_risk_scaling()
    deduplicate_risk_bucket_orders()
    print(f"✅ Symbols orders price levels calculation completed.")

if __name__ == "__main__":
    calculate_symbols_orders()
     
    
   
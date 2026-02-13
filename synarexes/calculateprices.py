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
import re

# --- GLOBALS ---
BROKER_DICT_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokers.json"
DEVELOPER_USERS = r"C:\xampp\htdocs\chronedge\synarex\usersdata\developers\developers.json"
INVESTOR_USERS = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors\investors.json"
SYMBOL_CATEGORY_PATH = r"C:\xampp\htdocs\chronedge\synarex\symbolscategory.json"
DEV_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\developers"
INV_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors"
DEFAULT_ACCOUNTMANAGEMENT = r"C:\xampp\htdocs\chronedge\synarex\default_accountmanagement.json"
NORMALIZE_SYMBOLS_PATH = r"C:\xampp\htdocs\chronedge\synarex\symbols_normalization.json"



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

def clean_risk_folders():
    """
    Scans the DEV_PATH and permanently deletes all folders ending in 'usd_risk'.
    This clears out all calculated risk buckets and their contained JSON files.
    """
    print(f"\n{'='*10} RISK FOLDER CLEANUP {'='*10}")
    
    if not os.path.exists(DEV_PATH):
        print(f" [!] Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    # Find all directories that end with 'usd_risk'
    # Recursive search ensures we hit sub-strategy folders
    risk_folders = glob.glob(os.path.join(DEV_PATH, "**", "*usd_risk"), recursive=True)

    if not risk_folders:
        print(f" 🛡️  System clean: No risk buckets found.")
        print(f"{'='*10} CLEANUP COMPLETE {'='*10}\n")
        return True

    deleted_count = 0
    print(f" 🧹 Starting deep purge of risk directories...")

    for folder_path in risk_folders:
        # Check if it exists and is a directory (shutil.rmtree requires a dir)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                deleted_count += 1
            except Exception as e:
                print(f"  └─ ❌ Error purging {os.path.basename(folder_path)}: {e}")

    if deleted_count > 0:
        print(f"  └─ ✅ Successfully purged {deleted_count} risk bucket folders")
    else:
        print(f"  └─ 🔘 No folders required deletion")

    print(f"{'='*10} CLEANUP COMPLETE {'='*10}\n")
    return True

def purge_unauthorized_symbols():
    """
    Iterates through all users, identifies allowed symbols across all categories,
    and removes any orders from limit_orders.json that use unauthorized symbols.
    """
    print(f"\n{'='*10} PURGING UNAUTHORIZED SYMBOLS {'='*10}")
    
    try:
        # 1. Load User IDs
        if not os.path.exists(DEVELOPER_USERS):
            print(f" [!] Error: Users file not found: {DEVELOPER_USERS}")
            return False

        with open(DEVELOPER_USERS, 'r') as f:
            users_data = json.load(f)

        total_purged_overall = 0

        for dev_broker_id in users_data.keys():
            print(f" [{dev_broker_id}] 🔍 Auditing symbol permissions...")
            
            user_folder = os.path.join(DEV_PATH, dev_broker_id)
            volumes_path = os.path.join(user_folder, "allowedsymbolsandvolumes.json")
            
            # Skip user if they don't have a config file
            if not os.path.exists(volumes_path):
                print(f"  └─ ⚠️  Config missing: Skipping audit")
                continue

            # 2. Extract ALL allowed symbols
            allowed_symbols = set()
            try:
                with open(volumes_path, 'r') as f:
                    v_data = json.load(f)
                    for category in v_data.values():
                        if isinstance(category, list):
                            for item in category:
                                if 'symbol' in item:
                                    allowed_symbols.add(item['symbol'].upper())
            except Exception as e:
                print(f"  └─ ❌ Error reading config: {e}")
                continue

            # 3. Locate and Filter limit_orders.json files
            limit_order_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)
            user_purged_count = 0

            for limit_path in limit_order_files:
                # We skip the calculated risk_reward folders to avoid redundant processing
                if "risk_reward_" in limit_path: continue
                
                try:
                    with open(limit_path, 'r') as f:
                        orders = json.load(f)
                except: continue

                original_count = len(orders)
                # Keep only if the symbol is in the allowed set
                purged_orders = [
                    order for order in orders 
                    if order.get('symbol', '').upper() in allowed_symbols
                ]

                # 4. Save back if any orders were removed
                diff = original_count - len(purged_orders)
                if diff > 0:
                    try:
                        with open(limit_path, 'w') as f:
                            json.dump(purged_orders, f, indent=4)
                        user_purged_count += diff
                    except Exception as e:
                        print(f"  └─ ❌ Failed to save {limit_path}: {e}")

            if user_purged_count > 0:
                print(f"  └─ ✅ Purged {user_purged_count} unauthorized orders")
                total_purged_overall += user_purged_count
            else:
                print(f"  └─ 🔘 All active symbols authorized")

        print(f"{'='*10} PURGE COMPLETE: {total_purged_overall} REMOVED {'='*10}\n")
        return True

    except Exception as e:
        print(f" [!] Critical Error during symbol purge: {e}")
        return False

def backup_limit_orders():
    """
    Traverses all user folders and creates a copy of every 'limit_orders.json' 
    file named 'limit_orders_backup.json' in the same directory.
    """
    print(f"\n{'='*10} BACKING UP LIMIT ORDERS {'='*10}")
    
    try:
        # 1. Load User IDs
        if not os.path.exists(DEVELOPER_USERS) or os.path.getsize(DEVELOPER_USERS) == 0:
            print(f" [!] Error: Users file missing or empty: {DEVELOPER_USERS}")
            return False
            
        with open(DEVELOPER_USERS, 'r') as f:
            users_data = json.load(f)
        
        total_backups = 0
        
        for dev_broker_id in users_data.keys():
            print(f" [{dev_broker_id}] 💾 Creating safety snapshots...")
            
            user_folder = os.path.join(DEV_PATH, dev_broker_id)
            user_backup_count = 0
            
            # 2. Find all limit_orders.json files recursively
            limit_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)
            
            for limit_path in limit_files:
                # Filter logic: Skip secondary files and existing backups
                if "risk_reward_" in limit_path or "limit_orders_backup.json" in limit_path:
                    continue
                
                if not os.path.exists(limit_path) or os.path.getsize(limit_path) == 0:
                    continue
                
                # 3. Define the destination path
                directory = os.path.dirname(limit_path)
                backup_path = os.path.join(directory, "limit_orders_backup.json")
                
                try:
                    # Preserve metadata with copy2
                    shutil.copy2(limit_path, backup_path)
                    user_backup_count += 1
                except Exception as e:
                    print(f"  └─ ❌ Failed to backup {os.path.basename(directory)}: {e}")

            if user_backup_count > 0:
                print(f"  └─ ✅ Secured {user_backup_count} backup files")
                total_backups += user_backup_count
            else:
                print(f"  └─ 🔘 No active orders found to back up")

        print(f"{'='*10} BACKUP COMPLETE: {total_backups} FILES {'='*10}\n")
        return True

    except Exception as e:
        print(f" [!] Critical Error during backup: {e}")
        return False

def enforce_risk():
    """
    Synchronizes USD risk rules from configuration files to limit orders.
    Uses symbol normalization and cleaning to ensure accurate config lookups.
    """
    try:
        # 1. Basic Path Validation
        if not os.path.exists(DEVELOPER_USERS) or os.path.getsize(DEVELOPER_USERS) == 0:
            print(f" [!] Error: Users file not found at {DEVELOPER_USERS}")
            return False

        # 2. Load Normalization Map (to match live_risk_reward logic)
        norm_map = {}
        if os.path.exists(NORMALIZE_SYMBOLS_PATH):
            try:
                with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
                    norm_map = json.load(f)
            except Exception as e:
                print(f" [!] Warning: Could not load normalization map: {e}")

        with open(DEVELOPER_USERS, 'r') as f:
            users_data = json.load(f)

        print(f"\n{'='*10} ENFORCING RISK RULES {'='*10}")

        for dev_broker_id in users_data.keys():
            user_folder = os.path.join(DEV_PATH, dev_broker_id)
            print(f" [{dev_broker_id}] 🛡️  Syncing risk enforcement...")

            # 3. Identify all config files (Primary + Secondary)
            acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
            secondary_configs = []
            if os.path.exists(acc_mgmt_path):
                try:
                    with open(acc_mgmt_path, 'r') as f:
                        acc_data = json.load(f)
                    poi_conditions = acc_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
                    for app_val in poi_conditions.values():
                        if isinstance(app_val, dict):
                            for ent_val in app_val.values():
                                if isinstance(ent_val, dict) and ent_val.get("new_filename"):
                                    secondary_configs.append(os.path.join(user_folder, ent_val["new_filename"], "allowedsymbolsandvolumes.json"))
                except: pass

            all_config_files = [os.path.join(user_folder, "allowedsymbolsandvolumes.json")] + secondary_configs
            total_enforced = 0

            # 4. Process each config and its related limit orders
            for config_path in all_config_files:
                if not os.path.exists(config_path): continue
                
                # Build Master Lookup (Normalized)
                risk_lookup = {}
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                for category in config_data.values():
                    if not isinstance(category, list): continue
                    for item in category:
                        symbol = item.get("symbol")
                        if not symbol: continue
                        
                        # Clean symbol: Remove special characters and set to UPPER
                        clean_sym = re.sub(r'[^a-zA-Z0-9]', '', symbol).upper()
                        
                        specs = {}
                        for key, value in item.items():
                            if key.endswith("_specs") and isinstance(value, dict):
                                tf = key.replace("_specs", "").upper()
                                specs[tf] = {
                                    "enforce": str(value.get("enforce_usd_risk", "no")).lower() == "yes",
                                    "usd_risk": value.get("usd_risk", 0)
                                }
                        risk_lookup[clean_sym] = specs

                # 5. Apply to Limit Orders
                config_dir = os.path.dirname(config_path)
                limit_files = glob.glob(os.path.join(config_dir, "**", "limit_orders.json"), recursive=True)
                
                for limit_path in limit_files:
                    if "risk_reward_" in limit_path: continue
                    try:
                        with open(limit_path, 'r') as f: orders = json.load(f)
                    except: continue

                    modified = False
                    for order in orders:
                        raw_sym = order.get('symbol', '')
                        raw_tf = str(order.get('timeframe', '')).upper()

                        # --- Normalization Logic ---
                        # Step A: Use the norm_map if it exists (e.g., "XAUUSD+" -> "Gold")
                        target_sym = norm_map.get(raw_sym, raw_sym)
                        # Step B: Filter out special characters (e.g., "DOW.N" -> "DOWN")
                        lookup_key = re.sub(r'[^a-zA-Z0-9]', '', target_sym).upper()
                        
                        rule = risk_lookup.get(lookup_key, {}).get(raw_tf)

                        # Determine Enforcement
                        is_missing_targets = order.get('exit') in [0, "0", None] and \
                                             order.get('target') in [0, "0", None]

                        should_enforce = False
                        risk_to_apply = 0

                        if rule and rule["enforce"]:
                            should_enforce = True
                            risk_to_apply = rule["usd_risk"]
                        elif is_missing_targets and rule:
                            should_enforce = True
                            risk_to_apply = rule["usd_risk"]
                        elif is_missing_targets and not rule:
                            # Log detailed error for the user to troubleshoot
                            print(f"  └─ ❌ Error: {raw_sym} ({raw_tf}) [Key: {lookup_key}] missing config risk.")

                        if should_enforce:
                            order['exit'] = 0
                            order['target'] = 0
                            order['usd_risk'] = risk_to_apply
                            order['usd_based_risk_only'] = True
                            modified = True
                            total_enforced += 1
                    
                    if modified:
                        with open(limit_path, 'w') as f:
                            json.dump(orders, f, indent=4)

            # Summary per broker
            if total_enforced > 0:
                print(f"  └─ ✅ Enforced USD risk rules on {total_enforced} orders")
            else:
                print(f"  └─ 🔘 Risk profiles already synchronized")

        print(f"{'='*10} RISK ENFORCEMENT COMPLETE {'='*10}\n")
        return True

    except Exception as e:
        print(f" [!] Critical Error in enforce_risk: {e}")
        return False

def preprocess_limit_orders_with_broker_data():
    """
    Simplified Pre-processor:
    Focuses on Broker-level logging and hides long file paths.
    """
    if not os.path.exists(BROKER_DICT_PATH):
        print(f" [!] Error: Broker config missing at {BROKER_DICT_PATH}")
        return False

    # Load Normalization Map
    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f" [!] Critical: Normalization map error: {e}")
        return False

    with open(BROKER_DICT_PATH, 'r') as f:
        broker_configs = json.load(f)

    print(f"\n{'='*10} STARTING BROKER PRE-PROCESSOR {'='*10}")

    for dev_broker_id, config in broker_configs.items():
        # --- MT5 CONNECTION ---
        if not mt5.initialize(
            path=config.get("TERMINAL_PATH", ""), 
            login=int(config.get("LOGIN_ID")), 
            password=config.get("PASSWORD"), 
            server=config.get("SERVER")
        ):
            print(f" [{dev_broker_id}] ❌ Connection Failed: {mt5.last_error()}")
            continue

        print(f" [{dev_broker_id}] 🟢 Connected to {config.get('SERVER')}")

        user_folder = os.path.join(DEV_PATH, dev_broker_id)
        limit_files = glob.glob(os.path.join(user_folder, "**", "limit_orders.json"), recursive=True)

        for file_path in limit_files:
            if "risk_reward_" in file_path:
                continue

            # Extract just the parent folder name (e.g., 'EURUSD_POI') instead of the full path
            folder_context = os.path.basename(os.path.dirname(file_path))

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    orders = json.load(f)
            except: continue

            file_changed = False
            for order in orders:
                raw_symbol = order.get('symbol')
                normalized_symbol = get_normalized_symbol(raw_symbol, norm_map)

                if not normalized_symbol:
                    print(f"  └─ [{folder_context}] ⚠️  Unknown symbol: {raw_symbol}")
                    continue

                mt5.symbol_select(normalized_symbol, True)
                info = mt5.symbol_info(normalized_symbol)

                if info is None:
                    print(f"  └─ [{folder_context}] ❓ {normalized_symbol} not on server")
                    continue

                # Update Logic
                if order['symbol'] != normalized_symbol or order.get('tick_size') != info.trade_tick_size:
                    order['symbol'] = normalized_symbol
                    order['tick_size'] = info.trade_tick_size
                    order['tick_value'] = info.trade_tick_value
                    file_changed = True

            if file_changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(orders, f, indent=4)
                print(f"  └─ [{folder_context}] ✅ Specs Updated")

        mt5.shutdown()

    print(f"{'='*10} PRE-PROCESSOR COMPLETE {'='*10}\n")
    return True

def validate_orders_with_live_volume():
    """
    Developer Version: Validates and modifies volumes in the INPUT configuration files.
    Targets 'allowedsymbolsandvolumes.json' so that subsequent calculations use valid data.
    """
    if not os.path.exists(BROKER_DICT_PATH):
        print(f" [!] Error: Broker dictionary not found.")
        return False

    # Load Normalization Map
    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f" [!] Critical: Normalization map error: {e}")
        return False

    try:
        with open(BROKER_DICT_PATH, 'r') as f:
            broker_configs = json.load(f)
    except:
        return False

    print(f"\n{'='*10} PRE-CALCULATION VOLUME VALIDATION {'='*10}")
    overall_files_updated = 0

    for dev_broker_id, config in broker_configs.items():
        user_folder = os.path.join(DEV_PATH, dev_broker_id)
        broker_files_updated = 0
        symbols_fixed = []
        
        # 1. MT5 Initialization
        if not mt5.initialize(
            path=config.get("TERMINAL_PATH", ""), 
            login=int(config.get("LOGIN_ID")), 
            password=config.get("PASSWORD"), 
            server=config.get("SERVER")
        ):
            print(f" [{dev_broker_id}] ❌ MT5 Init Failed")
            continue

        account_info = mt5.account_info()
        broker_server_name = account_info.server if account_info else config.get("SERVER")
        print(f" [{dev_broker_id}] 🔍 Checking Inputs: {broker_server_name}")

        # 2. TARGET THE INPUT CONFIGURATION FILES
        # We look for the allowedsymbolsandvolumes.json files
        search_pattern = os.path.join(user_folder, "**", "allowedsymbolsandvolumes.json")
        found_files = glob.glob(search_pattern, recursive=True)

        for target_file_path in found_files:
            try:
                file_changed = False
                with open(target_file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # allowedsymbolsandvolumes.json is a dict of lists (Forex, Crypto, etc.)
                for category, items in config_data.items():
                    if not isinstance(items, list): continue
                    
                    for item in items:
                        raw_symbol = item.get("symbol")
                        symbol = get_normalized_symbol(raw_symbol, norm_map)
                        
                        mt5.symbol_select(symbol, True)
                        info = mt5.symbol_info(symbol)
                        if info is None: continue

                        # In these files, volumes are usually inside tf_specs (e.g., "1h_specs": {"volume": 0.1})
                        # We iterate through all keys to find specs dictionaries
                        for key, value in item.items():
                            if "_specs" in key and isinstance(value, dict):
                                current_vol = float(value.get("volume", 0.0))
                                
                                # --- MT5 CONSTRAINT LOGIC ---
                                new_vol = max(current_vol, info.volume_min)
                                step = info.volume_step
                                if step > 0:
                                    # Floor to nearest step
                                    new_vol = round(math.floor(new_vol / step + 1e-9) * step, 2)
                                
                                if new_vol > info.volume_max:
                                    new_vol = info.volume_max

                                # Check for change
                                if abs(new_vol - current_vol) > 1e-7:
                                    value["volume"] = new_vol
                                    value["vol_validated"] = datetime.now().strftime("%H:%M")
                                    file_changed = True
                                    symbols_fixed.append(f"{symbol}({current_vol}->{new_vol})")

                if file_changed:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=4)
                    broker_files_updated += 1
                    overall_files_updated += 1

            except Exception as e:
                continue

        # --- Cool Simplified Print ---
        if broker_files_updated > 0:
            fix_summary = ", ".join(symbols_fixed[:3])
            more = f" (+{len(symbols_fixed)-3} more)" if len(symbols_fixed) > 3 else ""
            print(f"  └─ ✅ Cleaned {broker_files_updated} config files. Fixes: {fix_summary}{more}")
        else:
            print(f"  └─ 🔘 All input volumes are valid")

        mt5.shutdown()

    print(f"\n{'='*10} PRE-CALCULATION CHECK COMPLETE {'='*10}\n")
    return True 
    
def calculate_symbols_orders():
    """
    Calculates Exit/Target prices based on Risk:Reward ratios.
    Processes primary and secondary strategy folders with a clean visual summary.
    """
    try:
        # 1. Load Data
        if not os.path.exists(DEVELOPER_USERS) or os.path.getsize(DEVELOPER_USERS) == 0:
            print(f" [!] Error: Users file not found: {DEVELOPER_USERS}")
            return False

        with open(DEVELOPER_USERS, 'r', encoding='utf-8') as f:
            users_data = json.load(f)

        print(f"\n{'='*10} CALCULATING SYMBOL ORDERS {'='*10}")

        # 2. Iterate through each User
        for dev_broker_id in users_data.keys():
            print(f" [{dev_broker_id}] 🧮 Processing Risk:Reward scaling...")
            
            user_folder = os.path.join(DEV_PATH, dev_broker_id)
            acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
            primary_volumes_path = os.path.join(user_folder, "allowedsymbolsandvolumes.json")

            if not os.path.exists(acc_mgmt_path):
                print(f"  └─ ⚠️  Account management file missing")
                continue

            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                acc_mgmt_data = json.load(f)
            
            rr_ratios = acc_mgmt_data.get("risk_reward_ratios", [1.0])
            poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
            
            # --- Auto-generate secondary folders & paths ---
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
            total_files_updated = 0

            for volumes_path in all_config_files:
                if not os.path.exists(volumes_path): continue

                # Load Config for this specific subfolder
                user_config = {}
                with open(volumes_path, 'r', encoding='utf-8') as f:
                    v_data = json.load(f)
                    for category in v_data.values():
                        if isinstance(category, list):
                            for item in category:
                                sym_name = str(item.get('symbol', '')).strip().upper()
                                user_config[sym_name] = {k.lower(): v for k, v in item.items()}

                config_folder = os.path.dirname(volumes_path)
                limit_order_files = glob.glob(os.path.join(config_folder, "**", "limit_orders.json"), recursive=True)

                for limit_path in limit_order_files:
                    if "risk_reward_" in limit_path: continue
                    
                    try:
                        with open(limit_path, 'r', encoding='utf-8') as f:
                            original_orders = json.load(f)
                    except: continue

                    base_dir = os.path.dirname(limit_path)

                    for current_rr in rr_ratios:
                        orders_copy = copy.deepcopy(original_orders)
                        updated_any_order = False

                        for order in orders_copy:
                            symbol = str(order.get('symbol', '')).strip().upper()
                            if symbol not in user_config: continue

                            try:
                                # Data Extraction
                                entry = float(order['entry'])
                                rr_ratio = float(current_rr)
                                order_type = str(order.get('order_type', '')).upper()
                                tick_size = float(order['tick_size'])
                                tick_value = float(order['tick_value'])
                                tf = str(order.get('timeframe', '1h')).lower()
                                
                                digits = len(str(tick_size).split('.')[-1]) if tick_size < 1 else 0
                                tf_specs = user_config.get(symbol, {}).get(f"{tf}_specs")
                                
                                if not tf_specs or tf_specs.get('volume') is None: continue
                                volume = float(tf_specs['volume'])

                                # Calculation Logic
                                if order.get("usd_based_risk_only") is True:
                                    risk_val = order.get("usd_risk") if order.get("usd_risk") is not None else tf_specs.get("usd_risk")
                                    if risk_val is None: continue
                                    
                                    sl_dist = (float(risk_val) * tick_size) / (tick_value * volume)
                                    tp_dist = sl_dist * rr_ratio

                                    if "BUY" in order_type:
                                        order["exit"] = round(entry - sl_dist, digits)
                                        order["target"] = round(entry + tp_dist, digits)
                                    else:
                                        order["exit"] = round(entry + sl_dist, digits)
                                        order["target"] = round(entry - tp_dist, digits)
                                else:
                                    sl_price = float(order.get('exit', 0))
                                    tp_price = float(order.get('target', 0))

                                    if sl_price == 0 and tp_price > 0:
                                        risk_dist = abs(tp_price - entry) / rr_ratio
                                        order['exit'] = round(entry - risk_dist if "BUY" in order_type else entry + risk_dist, digits)
                                    elif sl_price > 0:
                                        risk_dist = abs(entry - sl_price)
                                        order['target'] = round(entry + (risk_dist * rr_ratio) if "BUY" in order_type else entry - (risk_dist * rr_ratio), digits)

                                # Final Metadata
                                order.pop("usd_risk", None) 
                                order[f"{tf}_volume"] = volume
                                order['risk_reward'] = rr_ratio
                                order['status'] = "Calculated"
                                order['calculated_at'] = datetime.now().strftime("%H:%M:%S")
                                updated_any_order = True

                            except: continue

                        if updated_any_order:
                            target_out_dir = os.path.join(base_dir, f"risk_reward_{current_rr}")
                            os.makedirs(target_out_dir, exist_ok=True)
                            with open(os.path.join(target_out_dir, "limit_orders.json"), 'w', encoding='utf-8') as f:
                                json.dump(orders_copy, f, indent=4)
                            total_files_updated += 1
            
            if total_files_updated > 0:
                print(f"  └─ ✅ Generated {total_files_updated} R:R specific order files")
            else:
                print(f"  └─ 🔘 No new orders required calculation")

        print(f"{'='*10} CALCULATION COMPLETE {'='*10}\n")
        return True

    except Exception as e:
        print(f" [!] Critical Error: {e}")
        return False

def live_risk_reward_amounts_and_volume_scale():
    """
    Scans for limit orders and calculates volume based on USD risk targets.
    Logs each broker process clearly with a status summary.
    """
    if not os.path.exists(BROKER_DICT_PATH):
        print(f" [!] Error: Broker dictionary not found.")
        return False

    # Load Normalization Map
    try:
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f" [!] Critical: Normalization map error: {e}")
        return False

    with open(BROKER_DICT_PATH, 'r') as f:
        try:
            broker_configs = json.load(f)
        except:
            return False

    print(f"\n{'='*10} LIVE RISKS BUCKETING & VOLUME SCALING {'='*10}")

    for dev_broker_id, config in broker_configs.items():
        user_folder = os.path.join(DEV_PATH, dev_broker_id)
        
        # 1. MT5 Initialization
        if not mt5.initialize(
            path=config.get("TERMINAL_PATH", ""), 
            login=int(config.get("LOGIN_ID")), 
            password=config.get("PASSWORD"), 
            server=config.get("SERVER")
        ):
            print(f" [{dev_broker_id}] ❌ MT5 Init Failed")
            continue

        account_info = mt5.account_info()
        if account_info is None:
            print(f" [{dev_broker_id}] ❌ Connection Error")
            mt5.shutdown()
            continue
        
        broker_server_name = account_info.server
        acc_currency = account_info.currency
        print(f" [{dev_broker_id}] 🟢 {broker_server_name} ({acc_currency})")

        # 2. Load Risks
        acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
        try:
            with open(acc_mgmt_path, 'r') as f:
                acc_mgmt_data = json.load(f)
            allowed_risks = acc_mgmt_data.get("RISKS", [])
            max_allowed_risk = max(allowed_risks) if allowed_risks else 50.0
        except:
            print(f"  └─ ⚠️  Account Config Missing")
            mt5.shutdown()
            continue

        # 3. Identify subfolders
        poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
        target_subfolders = [user_folder]
        for app_val in poi_conditions.values():
            if isinstance(app_val, dict):
                for ent_val in app_val.values():
                    if isinstance(ent_val, dict) and ent_val.get("new_filename"):
                        sub_path = os.path.join(user_folder, ent_val.get("new_filename"))
                        if os.path.exists(sub_path):
                            target_subfolders.append(sub_path)

        total_orders_calculated = 0
        buckets_found = set()

        for current_search_path in target_subfolders:
            limit_order_files = glob.glob(os.path.join(current_search_path, "**", "limit_orders.json"), recursive=True)
            
            for limit_path in limit_order_files:
                if "risk_reward_" not in limit_path or "_risk" in limit_path:
                    continue

                try:
                    with open(limit_path, 'r') as f: orders = json.load(f)
                except: continue

                risk_buckets = {}
                for order in orders:
                    if order.get('status') != "Calculated": continue

                    symbol = get_normalized_symbol(order.get('symbol'), norm_map)
                    if not symbol or not mt5.symbol_select(symbol, True): continue
                    
                    info = mt5.symbol_info(symbol)
                    if info is None: continue

                    entry, exit_p, target_p = float(order['entry']), float(order['exit']), float(order['target'])
                    current_volume = info.volume_min
                    filled_buckets = set()

                    # Volume loop
                    for _ in range(5000):
                        action = mt5.ORDER_TYPE_BUY if "BUY" in order['order_type'].upper() else mt5.ORDER_TYPE_SELL
                        sl_risk = mt5.order_calc_profit(action, symbol, current_volume, entry, exit_p)
                        
                        if sl_risk is None: # Fallback
                            sl_risk = -(abs(entry - exit_p) / info.trade_tick_size * info.trade_tick_value * current_volume)
                        
                        abs_risk = round(abs(sl_risk), 2)
                        if abs_risk > max_allowed_risk: break
                        
                        assigned_risk = None
                        if abs_risk < 1.00 and 0.5 in allowed_risks: assigned_risk = 0.5
                        elif int(math.floor(abs_risk)) in allowed_risks: assigned_risk = int(math.floor(abs_risk))
                        
                        if assigned_risk and assigned_risk not in filled_buckets:
                            risk_buckets.setdefault(assigned_risk, []).append({
                                **order,
                                'symbol': symbol,
                                f"{broker_server_name}_tick_size": info.trade_tick_size,
                                f"{broker_server_name}_tick_value": info.trade_tick_value,
                                'live_sl_risk_amount': abs_risk,
                                'calculated_at': datetime.now().strftime("%H:%M:%S")
                            })
                            filled_buckets.add(assigned_risk)
                            buckets_found.add(assigned_risk)
                            total_orders_calculated += 1
                        
                        current_volume += info.volume_step

                # Save Results
                if risk_buckets:
                    base_dir = os.path.dirname(limit_path)
                    for r_val, grouped in risk_buckets.items():
                        out_dir = os.path.join(base_dir, f"{r_val}usd_risk")
                        os.makedirs(out_dir, exist_ok=True)
                        out_file = os.path.join(out_dir, f"{r_val}usd_risk.json")
                        
                        existing = []
                        if os.path.exists(out_file):
                            try:
                                with open(out_file, 'r') as f: existing = json.load(f)
                            except: pass
                            
                        with open(out_file, 'w') as f:
                            json.dump(existing + grouped, f, indent=4)

        # Broker-level summary
        if total_orders_calculated > 0:
            bucket_str = ", ".join([f"${b}" for b in sorted(list(buckets_found))])
            print(f"  └─ ✅ Scaled {total_orders_calculated} orders into: {bucket_str}")
        else:
            print(f"  └─ 🔘 No pending orders found for scaling")

        mt5.shutdown()

    print(f"{'='*10} RISK SCALING COMPLETE {'='*10}\n")
    return True

def ajdust_order_price_closer_in_99cent_to_next_bucket():
    """
    Promotes fractional risk orders (e.g., $1.55) to the next whole bucket (e.g., $2.0).
    Processes folders silently per broker with a clean visual summary.
    """
    if not os.path.exists(DEV_PATH):
        print(f" [!] Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    print(f"\n{'='*10} PRICE RE-ADJUSTMENT PROMOTION {'='*10}")

    # Get all primary user directories
    user_folders = [f.path for f in os.scandir(DEV_PATH) if f.is_dir()]

    for user_folder in user_folders:
        dev_broker_id = os.path.basename(user_folder)
        acc_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
        
        # 1. Validation & Initialization
        if not os.path.exists(acc_mgmt_path) or os.path.getsize(acc_mgmt_path) == 0:
            continue

        try:
            with open(acc_mgmt_path, 'r') as f:
                acc_mgmt_data = json.load(f)
        except:
            continue
        
        allowed_risks = acc_mgmt_data.get("RISKS", [])
        if not allowed_risks:
            continue

        # Indicator that we are starting this broker
        print(f" [{dev_broker_id}] ⚖️  Scaling fractional risks...")

        # 2. Identify strategy paths
        poi_conditions = acc_mgmt_data.get("chart", {}).get("define_candles", {}).get("entries_poi_condition", {})
        target_search_paths = [user_folder]
        for app_val in poi_conditions.values():
            if isinstance(app_val, dict):
                for ent_val in app_val.values():
                    if isinstance(ent_val, dict) and ent_val.get("new_filename"):
                        strat_path = os.path.join(user_folder, ent_val.get("new_filename"))
                        if os.path.exists(strat_path):
                            target_search_paths.append(strat_path)

        promotion_count = 0

        # 3. Process Folders
        for search_root in target_search_paths:
            risk_json_files = glob.glob(os.path.join(search_root, "**", "*usd_risk", "*usd_risk.json"), recursive=True)

            for file_path in risk_json_files:
                try:
                    with open(file_path, 'r') as f:
                        orders = json.load(f)
                except: continue

                if not isinstance(orders, list): continue

                for order in orders:
                    try:
                        sl_risk = order.get('live_sl_risk_amount', 0)
                        fractional_part = sl_risk - int(sl_risk)

                        # Promotion Logic (e.g., 1.51 becomes 2.0)
                        if fractional_part >= 0.99:
                            target_risk = float(math.ceil(sl_risk))
                            if target_risk not in allowed_risks: continue 

                            # Re-calculation Data
                            entry = float(order['entry'])
                            rr_ratio = float(order['risk_reward'])
                            tick_size = float(order['tick_size'])
                            tick_value = float(order['tick_value'])
                            tf = order['timeframe']
                            volume = float(order[f"{tf}_volume"])
                            
                            if tick_value == 0 or volume == 0: continue

                            # Precision & New Levels
                            tick_str = format(tick_size, 'f').rstrip('0').rstrip('.')
                            precision = len(tick_str.split('.')[1]) if '.' in tick_str else 0
                            risk_dist = (target_risk / (tick_value * volume)) * tick_size
                            
                            new_order = copy.deepcopy(order)
                            if "BUY" in order['order_type'].upper():
                                new_order['exit'] = round(entry - risk_dist, precision)
                                new_order['target'] = round(entry + (risk_dist * rr_ratio), precision)
                            else:
                                new_order['exit'] = round(entry + risk_dist, precision)
                                new_order['target'] = round(entry - (risk_dist * rr_ratio), precision)

                            new_order.update({
                                'live_sl_risk_amount': target_risk,
                                'live_tp_reward_amount': round(target_risk * rr_ratio, 2),
                                'status': "Adjusted_To_Next_Bucket",
                                'adjusted_at': datetime.now().strftime("%H:%M:%S")
                            })

                            # Save to Destination
                            parent_rr_dir = os.path.dirname(os.path.dirname(file_path))
                            new_bucket_folder = os.path.join(parent_rr_dir, f"{int(target_risk)}usd_risk")
                            os.makedirs(new_bucket_folder, exist_ok=True)
                            
                            target_json_path = os.path.join(new_bucket_folder, f"{int(target_risk)}usd_risk.json")
                            existing_data = []
                            if os.path.exists(target_json_path):
                                with open(target_json_path, 'r') as tf_file:
                                    try: existing_data = json.load(tf_file)
                                    except: pass
                            
                            existing_data.append(new_order)
                            with open(target_json_path, 'w') as tf_file:
                                json.dump(existing_data, tf_file, indent=4)
                            
                            promotion_count += 1
                    except: continue

        if promotion_count > 0:
            print(f"  └─ ✅ Promoted {promotion_count} orders to higher buckets")
        else:
            print(f"  └─ 🔘 No fractional risks required promotion")

    print(f"{'='*10} PROMOTION COMPLETE {'='*10}\n")
    return True

def fix_risk_buckets_according_to_orders_risk():
    """
    Developer Version: Identifies and fixes bucket violations in the DEV_PATH locally.
    Logic: Risk < $1.00 -> '0.5usd_risk', Risk >= $1.00 -> floor(risk) bucket.
    """
    print(f"\n{'='*15} 🛠️  BUCKET INTEGRITY REPAIR (DEV MODE) {'='*15}")
    print(f" >>> Scanning Developer Path: {DEV_PATH}")
    
    overall_moved = 0
    overall_files_fixed = 0
    broker_stats = []

    try:
        with open(BROKER_DICT_PATH, 'r') as f:
            broker_configs = json.load(f)
    except Exception as e:
        print(f" [!] Critical: Could not load broker configs: {e}")
        return False

    # Risk field key already exists in your JSON data
    risk_field = "live_sl_risk_amount"

    for dev_broker_id in broker_configs.keys():
        print(f" ⚙️  Processing Developer Broker: {dev_broker_id}...")
        
        user_folder = os.path.join(DEV_PATH, dev_broker_id)
        investor_moved = 0
        
        # Search for: **/risk_reward_*/*usd_risk/*.json
        search_pattern = os.path.join(user_folder, "**", "*usd_risk", "*.json")
        found_files = glob.glob(search_pattern, recursive=True)

        for target_file_path in found_files:
            try:
                filename = os.path.basename(target_file_path)
                try:
                    # Extracts '5.0' from '5usd_risk.json'
                    current_bucket_val = float(filename.replace('usd_risk.json', ''))
                except: continue

                with open(target_file_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)

                if not isinstance(entries, list): continue

                staying_entries = []
                file_changed = False

                for entry in entries:
                    live_risk_amt = entry.get(risk_field)
                    
                    if live_risk_amt is None:
                        staying_entries.append(entry)
                        continue

                    # --- HYBRID BUCKET LOGIC ---
                    if live_risk_amt < 1.0:
                        correct_bucket_val = 0.5
                    else:
                        correct_bucket_val = float(math.floor(live_risk_amt))

                    # Check for violation
                    if math.isclose(correct_bucket_val, current_bucket_val):
                        staying_entries.append(entry)
                    else:
                        # MIGRATION LOGIC
                        new_bucket_name = "0.5usd_risk" if correct_bucket_val == 0.5 else f"{int(correct_bucket_val)}usd_risk"
                        
                        # Navigate up from .../5usd_risk/5usd_risk.json to the RR folder
                        parent_rr_dir = os.path.dirname(os.path.dirname(target_file_path))
                        new_dir = os.path.join(parent_rr_dir, new_bucket_name)
                        os.makedirs(new_dir, exist_ok=True)
                        
                        new_file_path = os.path.join(new_dir, f"{new_bucket_name}.json")

                        # Append to target bucket
                        dest_data = []
                        if os.path.exists(new_file_path):
                            try:
                                with open(new_file_path, 'r', encoding='utf-8') as nf:
                                    dest_data = json.load(nf)
                            except: pass
                        
                        dest_data.append(entry)
                        with open(new_file_path, 'w', encoding='utf-8') as nf:
                            json.dump(dest_data, nf, indent=4)
                        
                        file_changed = True
                        investor_moved += 1
                        overall_moved += 1

                # Clean up the source file if items were moved out
                if file_changed:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        json.dump(staying_entries, f, indent=4)
                    overall_files_fixed += 1

            except Exception:
                continue

        # Log completion for this broker
        status_icon = "🛠️" if investor_moved > 0 else "✨"
        print(f"  └─ {status_icon} Finished {dev_broker_id}: {investor_moved} shifts made.")
        broker_stats.append({"id": dev_broker_id, "moved": investor_moved})
    return True

def enrich_orphanage_buckets():
    """
    Developer Version: Enrichment & Scaling.
    Takes the orders repaired in Function 1 and creates copies for all other risk buckets.
    Announces each broker in real-time and skips the final summary table.
    """
    print(f"\n{'='*15} 🚀 BUCKET ENRICHMENT (DEV MODE) {'='*15}")
    print(f" >>> Scaling Orders across Developer Path: {DEV_PATH}")

    overall_generated = 0

    try:
        with open(BROKER_DICT_PATH, 'r') as f:
            broker_configs = json.load(f)
    except Exception as e:
        print(f" [!] Critical: Could not load broker configs: {e}")
        return False

    for dev_broker_id in broker_configs.keys():
        # Immediate notification of current work
        print(f" ⚙️  Enriching Developer Broker: {dev_broker_id}...")
        
        user_folder = os.path.join(DEV_PATH, dev_broker_id)
        
        # Load available risks for this specific developer setup
        account_mgmt_path = os.path.join(user_folder, "accountmanagement.json")
        try:
            with open(account_mgmt_path, 'r') as f:
                account_data = json.load(f)
                available_risks = sorted([float(r) for r in account_data.get("RISKS", [])])
        except:
            print(f"    [!] Skipping {dev_broker_id}: No accountmanagement.json found.")
            continue

        broker_copies_count = 0
        search_pattern = os.path.join(user_folder, "**", "*usd_risk", "*.json")
        found_files = glob.glob(search_pattern, recursive=True)

        for source_file_path in found_files:
            try:
                filename = os.path.basename(source_file_path)
                current_bucket_base = float(filename.replace('usd_risk.json', ''))

                with open(source_file_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)

                for entry in entries:
                    if entry.get("is_volume_copy"): continue

                    current_risk = float(entry.get("live_sl_risk_amount", 0))
                    current_vol = float(entry.get("volume", 0))
                    
                    if current_risk <= 0 or current_vol <= 0: continue

                    # Scaling Ratio (Linear)
                    vol_per_dollar = current_vol / current_risk
                    parent_rr_dir = os.path.dirname(os.path.dirname(source_file_path))
                    
                    for target_risk in available_risks:
                        if math.isclose(target_risk, current_bucket_base): continue

                        new_vol = round(vol_per_dollar * target_risk, 2)
                        new_bucket_name = "0.5usd_risk" if target_risk == 0.5 else f"{int(target_risk)}usd_risk"
                        
                        order_copy = entry.copy()
                        order_copy.update({
                            "volume": new_vol,
                            "live_sl_risk_amount": target_risk,
                            "is_volume_copy": True,
                            "parent_source_bucket": f"{current_bucket_base}usd",
                            "enriched_at": datetime.now().strftime("%H:%M:%S")
                        })

                        target_dir = os.path.join(parent_rr_dir, new_bucket_name)
                        os.makedirs(target_dir, exist_ok=True)
                        target_file = os.path.join(target_dir, f"{new_bucket_name}.json")

                        dest_data = []
                        if os.path.exists(target_file):
                            with open(target_file, 'r', encoding='utf-8') as nf:
                                try: dest_data = json.load(nf)
                                except: dest_data = []

                        if not any(d.get('symbol') == order_copy['symbol'] and 
                                   d.get('volume') == order_copy['volume'] for d in dest_data):
                            dest_data.append(order_copy)
                            with open(target_file, 'w', encoding='utf-8') as nf:
                                json.dump(dest_data, nf, indent=4)
                            
                            broker_copies_count += 1
                            overall_generated += 1

            except Exception:
                continue

        # Status update for the broker just finished
        status_icon = "🧬" if broker_copies_count > 0 else "✨"
        print(f"  └─ {status_icon} Finished {dev_broker_id}: {broker_copies_count} variations generated.")

    print(f"\n ✅ DEV ENRICHMENT COMPLETE: {overall_generated} New Orders Created")
    print(f"{'='*45}\n")

    return True

def deduplicate_risk_bucket_orders():
    """
    Cleans up risk buckets by keeping only the most efficient order 
    (lowest risk) for each Symbol/Timeframe/Direction pair.
    """
    if not os.path.exists(DEV_PATH):
        print(f" [!] Error: DEV_PATH {DEV_PATH} does not exist.")
        return False

    print(f"\n{'='*10} RISK BUCKET DEDUPLICATION {'='*10}")

    # We first find all broker folders to group the output
    user_folders = [f.path for f in os.scandir(DEV_PATH) if f.is_dir()]

    for user_folder in user_folders:
        dev_broker_id = os.path.basename(user_folder)
        risk_json_files = glob.glob(os.path.join(user_folder, "**", "*usd_risk", "*usd_risk.json"), recursive=True)
        
        if not risk_json_files:
            continue

        print(f" [{dev_broker_id}] 🧹 Cleaning redundant orders...")
        total_removed = 0

        for file_path in risk_json_files:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                continue

            try:
                with open(file_path, 'r') as f:
                    orders = json.load(f)
            except:
                continue

            if not isinstance(orders, list) or not orders:
                continue

            initial_count = len(orders)
            # Key: (symbol, timeframe, order_type)
            best_orders = {}

            for order in orders:
                symbol = order.get('symbol')
                tf = order.get('timeframe', '1h')
                direction = order.get('order_type', '').upper()
                risk_amt = float(order.get('live_sl_risk_amount', 0))

                group_key = (symbol, tf, direction)

                if group_key not in best_orders:
                    best_orders[group_key] = order
                else:
                    # Keep the one with the LOWER risk amount (most conservative)
                    existing_risk = float(best_orders[group_key].get('live_sl_risk_amount', 0))
                    if risk_amt < existing_risk:
                        best_orders[group_key] = order

            unique_orders = list(best_orders.values())
            removed_in_file = initial_count - len(unique_orders)

            if removed_in_file > 0:
                try:
                    with open(file_path, 'w') as f:
                        json.dump(unique_orders, f, indent=4)
                    total_removed += removed_in_file
                except:
                    continue

        if total_removed > 0:
            print(f"  └─ ✅ Pruned {total_removed} redundant entries")
        else:
            print(f"  └─ 🔘 Risk buckets already optimized")

    print(f"{'='*10} DEDUPLICATION COMPLETE {'='*10}\n")
    return True

def sync_dev_investors():
    """
    Synchronizes investor accounts with developer strategy data.
    Logs each investor process clearly with a status summary.
    """
    def compact_json_format(data):
        """Custom formatter to keep lists on one line while indenting dictionaries."""
        res = json.dumps(data, indent=4)
        res = re.sub(r'\[\s+([^\[\]]+?)\s+\]', 
                    lambda m: "[" + ", ".join([line.strip() for line in m.group(1).splitlines()]).replace('"', '"') + "]", 
                    res)
        res = res.replace(",,", ",")
        return res

    try:
        # 1. Load Data
        if not all(os.path.exists(f) for f in [INVESTOR_USERS, DEVELOPER_USERS, DEFAULT_ACCOUNTMANAGEMENT]):
            print(" [!] Error: Configuration files missing.")
            return False

        with open(DEFAULT_ACCOUNTMANAGEMENT, 'r', encoding='utf-8') as f:
            default_acc_data = json.load(f)
            default_risk_mgmt = default_acc_data.get("account_balance_default_risk_management", {})

        with open(INVESTOR_USERS, 'r', encoding='utf-8') as f:
            investors_data = json.load(f)
        
        with open(DEVELOPER_USERS, 'r', encoding='utf-8') as f:
            developers_data = json.load(f)

        print(f"\n{'='*10} SYNCING INVESTOR ACCOUNTS {'='*10}")

        # 2. Iterate through Investors
        for inv_broker_id, inv_info in investors_data.items():
            print(f" [{inv_broker_id}] 🔄 Processing Sync...")

            invested_string = inv_info.get("INVESTED_WITH", "")
            inv_server = inv_info.get("SERVER", "")
            
            if "_" not in invested_string:
                print(f"  └─ ⚠️  Invalid 'INVESTED_WITH' format: {invested_string}")
                continue
            
            parts = invested_string.split("_", 1)
            dev_broker_id, target_strat_name = parts[0], parts[1]

            if dev_broker_id not in developers_data:
                print(f"  └─ ❌ Linked Dev {dev_broker_id} not found in database")
                continue

            # Broker Matching Logic
            dev_broker_name = developers_data[dev_broker_id].get("BROKER", "").lower()
            if dev_broker_name not in inv_server.lower():
                print(f"  └─ ❌ Broker Mismatch: Dev requires {dev_broker_name.upper()}")
                continue

            dev_user_folder = os.path.join(DEV_PATH, dev_broker_id)
            inv_user_folder = os.path.join(INV_PATH, inv_broker_id)
            dev_acc_path = os.path.join(dev_user_folder, "accountmanagement.json")
            inv_acc_path = os.path.join(inv_user_folder, "accountmanagement.json")

            # 3. Sync Account Management
            if os.path.exists(dev_acc_path):
                with open(dev_acc_path, 'r', encoding='utf-8') as f:
                    dev_acc_data = json.load(f)
                
                os.makedirs(inv_user_folder, exist_ok=True)
                inv_acc_data = {}
                if os.path.exists(inv_acc_path):
                    try:
                        with open(inv_acc_path, 'r', encoding='utf-8') as f:
                            inv_acc_data = json.load(f)
                    except: pass

                is_reset = inv_acc_data.get("reset_all", False)
                if is_reset: inv_acc_data = {"reset_all": False}
                
                needs_save = is_reset 
                keys_to_sync = ["RISKS", "risk_reward_ratios", "symbols_priority", "settings"]
                
                for key in keys_to_sync:
                    if key not in inv_acc_data or not inv_acc_data[key]:
                        inv_acc_data[key] = dev_acc_data.get(key, []) if key != "settings" else dev_acc_data.get(key, {})
                        needs_save = True

                if "account_balance_default_risk_management" not in inv_acc_data:
                    inv_acc_data["account_balance_default_risk_management"] = default_risk_mgmt
                    needs_save = True
                
                if needs_save:
                    with open(inv_acc_path, 'w', encoding='utf-8') as f:
                        f.write(compact_json_format(inv_acc_data))
                    print(f"  └─ ✅ accountmanagement.json synced")
            else:
                print(f"  └─ ⚠️  Dev accountmanagement.json missing")

            # 4. Clone Strategy Folder
            dev_strat_path = os.path.join(dev_user_folder, target_strat_name)
            inv_strat_path = os.path.join(inv_user_folder, target_strat_name)

            if os.path.exists(dev_strat_path):
                try:
                    if os.path.exists(inv_strat_path):
                        shutil.rmtree(inv_strat_path)
                    
                    # Selective copy/clean logic
                    shutil.copytree(dev_strat_path, inv_strat_path, dirs_exist_ok=True)
                    for item in os.listdir(inv_strat_path):
                        item_path = os.path.join(inv_strat_path, item)
                        if item == "pending_orders": continue
                        if os.path.isdir(item_path): shutil.rmtree(item_path)
                        else: os.remove(item_path)
                    
                    print(f"  └─ 📁 Strategy Cloned: {target_strat_name}")
                except Exception as e:
                    print(f"  └─ ❌ Folder Sync Error: {e}")
            else:
                print(f"  └─ ⚠️  Dev Strategy folder '{target_strat_name}' missing")

        print(f"{'='*10} INVESTOR SYNC COMPLETE {'='*10}\n")
        return True

    except Exception as e:
        print(f"\n [!] Enrichment Error: {e}")
        return False

def calculate_orders():
    purge_unauthorized_symbols()
    clean_risk_folders()
    backup_limit_orders()
    enforce_risk()
    preprocess_limit_orders_with_broker_data()
    validate_orders_with_live_volume()
    calculate_symbols_orders()
    live_risk_reward_amounts_and_volume_scale()
    fix_risk_buckets_according_to_orders_risk()
    enrich_orphanage_buckets()
    deduplicate_risk_bucket_orders()
    sync_dev_investors()
    print(f"✅ Symbols order price levels calculation completed.")


if __name__ == "__main__":
    sync_dev_investors()
    
   
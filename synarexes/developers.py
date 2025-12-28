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
import re
import placeorders
import insiders_server
import timeorders
from collections import defaultdict
from typing import List, Dict, Any
import multiprocessing
from typing import List
import json
import os
import sys
import os
import json
import importlib.util
import multiprocessing
from typing import List, Dict, Any

def get_developers_dictionary_path():
    PATH_CONFIG_FILE = r"C:\xampp\htdocs\chronedge\synarex\developersdictionarypath.json"

    if not os.path.exists(PATH_CONFIG_FILE):
        raise FileNotFoundError(f"Path config file not found: {PATH_CONFIG_FILE}")

    try:
        with open(PATH_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        brokers_path = config.get("developers_dictionary_path")
        if not brokers_path:
            raise ValueError("Key 'developers_dictionary_path' missing in config file")
        
        # Optional: Normalize path (handles both / and \)
        brokers_path = os.path.normpath(brokers_path)
        
        if not os.path.exists(brokers_path):
            raise FileNotFoundError(f"Brokers file not found at: {brokers_path}")
        
        return brokers_path

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {PATH_CONFIG_FILE}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load path config: {e}")


def load_developers_dictionary(account_email_or_login="", account_password=""):
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json"

    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Normalize some fields (still useful even if we don't match on them)
        for user_brokerid, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        # If credentials are provided → find matching broker using ONLY ACCOUNT_EMAIL and ACCOUNT_PASSWORD
        if account_email_or_login and account_password:
            for user_brokerid, cfg in data.items():
                # Check ONLY ACCOUNT_EMAIL and ACCOUNT_PASSWORD
                email_match = (
                    "ACCOUNT_EMAIL" in cfg and 
                    cfg["ACCOUNT_EMAIL"] == account_email_or_login
                )
                password_match = (
                    "ACCOUNT_PASSWORD" in cfg and 
                    cfg["ACCOUNT_PASSWORD"] == account_password
                )
                
                if email_match and password_match:
                    print(f"Successfully matched and loaded broker '{user_brokerid}' using ACCOUNT_EMAIL and ACCOUNT_PASSWORD.", "SUCCESS")
                    return {user_brokerid: cfg}
            
            # No match found
            print("No broker matched the provided ACCOUNT_EMAIL and ACCOUNT_PASSWORD.", "WARNING")
            return {}
        
        # No credentials provided → load all
        print(f"Brokers config loaded successfully → {len(data)} broker(s)", "SUCCESS")
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in developersdictionary.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load developersdictionary.json: {e}", "CRITICAL")
        return {}
developersdictionary = load_developers_dictionary()


def append_myfunctions_to_developers_functions():
    import os
    import json

    # Paths
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json"
    brokers_dir = os.path.dirname(BROKERS_JSON_PATH)
    developers_functions_path = os.path.join(brokers_dir, "developers_functions.json")

    # Load developer brokers from the global dictionary
    developer_brokers = {
        name: cfg for name, cfg in globals().get("developersdictionary", {}).items()
        if cfg.get("POSITION", "").lower() == "developer"
    }

    if not developer_brokers:
        print("[ERROR] No developer brokers found in developersdictionary")
        return

    print(f"[INFO] Found {len(developer_brokers)} developer broker(s): {list(developer_brokers.keys())}")

    # Load existing developers_functions.json to maintain other (non-developer) brokers
    if os.path.exists(developers_functions_path):
        try:
            with open(developers_functions_path, 'r', encoding='utf-8') as f:
                users_funcs_dict = json.load(f)
            if not isinstance(users_funcs_dict, dict):
                users_funcs_dict = {}
        except Exception as e:
            print(f"[WARNING] Could not read developers_functions.json ({e}) → starting fresh")
            users_funcs_dict = {}
    else:
        users_funcs_dict = {}

    changes_made = False
    updated_brokers = []

    for user_brokerid, cfg in developer_brokers.items():
        print(f"\n[PROCESSING] Broker: {user_brokerid}")

        base_folder = cfg.get("BASE_FOLDER", "")
        if not base_folder:
            continue

        # Locate continuation.json to find the developer's source directory
        possible_continuation_paths = [
            os.path.join(base_folder, "..", "developers", user_brokerid, "continuation.json"),
            os.path.join(base_folder, "continuation.json")
        ]

        myfunctions_dir = None
        for cont_path in possible_continuation_paths:
            if os.path.exists(cont_path):
                myfunctions_dir = os.path.dirname(cont_path)
                break

        if myfunctions_dir is None:
            print(f"[SKIP] Could not locate continuation.json for broker '{user_brokerid}'")
            continue

        myfunctions_path = os.path.join(myfunctions_dir, "myfunctions.json")

        if not os.path.exists(myfunctions_path):
            print(f"[SKIP] myfunctions.json NOT FOUND at: {myfunctions_path}")
            continue

        # Load fresh items from developer's myfunctions.json
        try:
            with open(myfunctions_path, 'r', encoding='utf-8') as f:
                my_items = json.load(f)
            if not isinstance(my_items, list):
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load myfunctions.json: {e}")
            continue

        # --- BUILD FRESH LIST FOR THIS BROKER (OVERWRITE LOGIC) ---
        new_broker_list = []

        # 1. Developer Tag
        new_broker_list.append(f"developer: {user_brokerid}")

        # 2. Extract specific types
        filename_entries = [str(i).strip() for i in my_items if str(i).strip().startswith("filename:") and "nofilename.py" not in str(i)]
        orders_entries = [str(i).strip() for i in my_items if str(i).strip().startswith("orders_jsonfile:")]
        module_func_entries = [str(i).strip() for i in my_items if str(i).strip().startswith("module_function:")]

        # 3. Apply Filename Logic
        if filename_entries:
            new_broker_list.extend(filename_entries)
        else:
            new_broker_list.append("filename: nofilename.py")

        # 4. Apply Orders and Module Functions
        new_broker_list.extend(orders_entries)
        new_broker_list.extend(module_func_entries)

        # 5. Apply everything else (actual function names)
        prefixes = ("filename:", "orders_jsonfile:", "developer:", "module_function:")
        for item in my_items:
            item_str = str(item).strip()
            if not any(item_str.startswith(p) for p in prefixes) and item_str:
                if item_str not in new_broker_list:
                    new_broker_list.append(item_str)

        # Check if the new list is different from what we currently have
        old_list = users_funcs_dict.get(user_brokerid, [])
        if old_list != new_broker_list:
            users_funcs_dict[user_brokerid] = new_broker_list
            changes_made = True
            updated_brokers.append(user_brokerid)
            print(f"[SYNC] Overwrote '{user_brokerid}' with {len(new_broker_list)} current items.")
        else:
            print(f"[INFO] '{user_brokerid}' is already perfectly in sync.")

    # === SAVE CHANGES ===
    if changes_made:
        try:
            with open(developers_functions_path, 'w', encoding='utf-8') as f:
                json.dump(users_funcs_dict, f, indent=2, ensure_ascii=False)
            print(f"\n[SUCCESS] developers_functions.json UPDATED. Brokers synced: {updated_brokers}")
        except Exception as e:
            print(f"[ERROR] Failed to save developers_functions.json: {e}")
    else:
        print("\n[INFO] All brokers were already in sync. No file write needed.")


def load_users_functions_with_modules() -> List[Dict[str, Any]]:
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\developersdictionary.json"
    brokers_dir = os.path.dirname(BROKERS_JSON_PATH)
    developers_functions_path = os.path.join(brokers_dir, "developers_functions.json")

    if not os.path.exists(developers_functions_path):
        print(f"[WARNING] developers_functions.json not found at {developers_functions_path}")
        return []

    try:
        with open(developers_functions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        developersdictionary = globals().get("developersdictionary", {})

        for user_brokerid, items in data.items():
            if not isinstance(items, list):
                continue

            functions = []
            filename = None
            for item in items:
                item_str = str(item).strip()
                if item_str.startswith("filename:"):
                    filename = item_str[len("filename:"):].strip()
                elif not item_str.startswith("developer:"):
                    functions.append(item_str)

            if not filename or not filename.endswith(".py"):
                continue
            if user_brokerid not in developersdictionary:
                continue

            cfg = developersdictionary[user_brokerid]
            base_folder = cfg["BASE_FOLDER"]
            
            # Locate the developer folder
            developer_folder = None
            possible_paths = [
                os.path.join(base_folder, "..", "developers", user_brokerid, "continuation.json"),
                os.path.join(base_folder, "continuation.json")
            ]
            for cp in possible_paths:
                if os.path.exists(cp):
                    developer_folder = os.path.dirname(cp)
                    break

            if not developer_folder:
                continue
            
            full_file_path = os.path.normpath(os.path.join(developer_folder, filename))
            if not os.path.isfile(full_file_path):
                continue

            # === NO MORE mybroker SOURCE INSPECTION ===
            # We now rely solely on session.json + credential validation in developers_functions()

            results.append({
                "user_brokerid": user_brokerid,
                "filename": filename,
                "module_name": os.path.splitext(filename)[0],
                "functions": functions,
                "file_path": full_file_path,
                # Removed: "mybroker_found"
            })

        return results

    except Exception as e:
        print(f"[ERROR] Critical failure in load_users_functions_with_modules: {e}")
        return []

def validate_developer_credentials(user_brokerid: str, base_folder: str) -> bool:
    """
    Validate that the broker has a valid session.json with credentials
    that EXACTLY match ACCOUNT_EMAIL and ACCOUNT_PASSWORD in developersdictionary.json
    
    Returns True only if fully valid.
    """
    global developersdictionary
    
    if user_brokerid not in developersdictionary:
        print(f"[REJECTED] Broker '{user_brokerid}' not found in developersdictionary.json")
        return False
    
    cfg = developersdictionary[user_brokerid]
    expected_email = cfg.get("ACCOUNT_EMAIL")
    expected_password = cfg.get("ACCOUNT_PASSWORD")
    
    if not expected_email or not expected_password:
        print(f"[REJECTED] Broker '{user_brokerid}' missing ACCOUNT_EMAIL or ACCOUNT_PASSWORD in developersdictionary.json")
        return False
    
    # Construct expected developer folder path
    developers_path = os.path.join(base_folder, "..", "developers", user_brokerid)
    if not os.path.exists(developers_path):
        # Fallback: some setups have flat structure
        developers_path = base_folder
    
    session_path = os.path.join(developers_path, "session.json")
    if not os.path.exists(session_path):
        print(f"[REJECTED] No session.json found for '{user_brokerid}' at {session_path}")
        return False
    
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        session_email = session_data.get("email", "").strip()
        session_password = session_data.get("password", "").strip()
        
        if session_email == expected_email and session_password == expected_password:
            print(f"[VALID] Credentials match for broker '{user_brokerid}'")
            return True
        else:
            print(f"[REJECTED] Credentials mismatch for '{user_brokerid}' "
                  f"(email: {'match' if session_email == expected_email else 'no match'}, "
                  f"password: {'match' if session_password == expected_password else 'no match'})")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to read session.json for '{user_brokerid}': {e}")
        return False

def locate_developer_order_jsons():
    import os
    import shutil
    import json
    from collections import defaultdict

    # Path to the global "dictator" symbol categories file
    DICTATOR_PATH = r"C:\xampp\htdocs\chronedge\synarex\chart\symbolscategory\symbolscategory.json"
    
    # Path to the combined symbolstick.json (tick info)
    SYMBOLSTICK_PATH = r"C:\xampp\htdocs\chronedge\synarex\chart\symbolstick\symbolstick.json"

    # Helper to normalize symbols (spaces/underscores, case-insensitive)
    def normalize_symbol(s):
        if not s:
            return ""
        return " ".join(s.replace("_", " ").split()).upper()

    # Helper to strip numbers from broker name (e.g., "deriv6" -> "deriv")
    def clean_broker(broker_name):
        if not broker_name:
            return ""
        return ''.join([char for char in broker_name if not char.isdigit()]).lower()

    # Load dictator symbol categories
    dictator = {}
    if os.path.exists(DICTATOR_PATH):
        try:
            with open(DICTATOR_PATH, 'r', encoding='utf-8') as f:
                dictator = json.load(f)
            print("[INFO] Dictator symbol categories loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load dictator file: {e}. All symbols will be removed.")
    else:
        print("[WARNING] Dictator file not found. All symbols will be removed.")

    # Precompute normalized symbols from dictator
    all_dictator_symbols = set()
    symbol_to_category = {}
    for cat, sym_list in dictator.items():
        for s in sym_list:
            norm_s = normalize_symbol(s)
            all_dictator_symbols.add(norm_s)
            if norm_s not in symbol_to_category:
                symbol_to_category[norm_s] = cat
            else:
                print(f"[WARNING] Normalized symbol '{norm_s}' appears in multiple categories! Using first one.")

    # Load symbolstick.json (tick info) once
    symbolstick_data = {}
    if os.path.exists(SYMBOLSTICK_PATH):
        try:
            with open(SYMBOLSTICK_PATH, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # raw_data is dict with keys like "AUD_BASKET", "EURGBP", etc.
                for key, info in raw_data.items():
                    norm_key = normalize_symbol(key)
                    cleaned_broker = clean_broker(info.get("broker", ""))
                    symbolstick_data[norm_key] = {
                        "broker": cleaned_broker,
                        "tick_size": info.get("tick_size"),
                        "tick_value": info.get("tick_value"),
                        "original_market": info.get("market")  # preserve original display name
                    }
            print(f"[INFO] Loaded {len(symbolstick_data)} symbols from symbolstick.json with tick info.")
        except Exception as e:
            print(f"[WARNING] Failed to load symbolstick.json: {e}. Tick info will be unavailable.")
    else:
        print("[WARNING] symbolstick.json not found. No tick info will be added.")

    # 1. Sync data first
    append_myfunctions_to_developers_functions()  # Assuming this exists

    # 2. Load module metadata
    broker_data = load_users_functions_with_modules()  # Assuming this exists
    if not broker_data:
        print("[INFO] No developer modules loaded — nothing to scan.")
        return {}

    results = {}
    global_stats = {
        "found": 0,
        "copied": 0,
        "restructured": 0,
        "configs_copied": 0,
        "filtered": 0,
        "orphan_folders_found": 0,
        "orphan_folders_deleted": 0
    }

    for entry in broker_data:
        user_brokerid = entry["user_brokerid"]
        filename = entry["filename"]
        file_path = entry["file_path"]
        functions = entry["functions"]

        cleaned_user_broker = clean_broker(user_brokerid)  # e.g., "deriv6" -> "deriv"

        developer_folder = os.path.dirname(os.path.abspath(file_path))

        # === STEP 1: CREDENTIAL VALIDATION ===
        if not validate_developer_credentials(user_brokerid, developer_folder):
            print(f"[SKIPPED] Broker '{user_brokerid}' failed credential validation. Skipping module '{filename}'.")
            continue

        # === STEP 2: CHECK REQUIRED CONFIGURATION FILES ===
        allowed_file_path = os.path.join(developer_folder, "allowedsymbolsandvolumes.json")
        tradeslimit_file_path = os.path.join(developer_folder, "disabledorders.json")
        accountmanagement_file_path = os.path.join(developer_folder, "accountmanagement.json")

        required_files = [
            (allowed_file_path, "allowedsymbolsandvolumes.json"),
            (tradeslimit_file_path, "disabledorders.json"),
            (accountmanagement_file_path, "accountmanagement.json")
        ]

        missing = [name for path, name in required_files if not os.path.exists(path)]
        if missing:
            print(f"[REJECTED] Missing required files for broker '{user_brokerid}': {missing}. Skipping.")
            continue

        print(f"[VALID] All required config files found for broker '{user_brokerid}' ({filename})")

        # === Collect order JSON file names (case-insensitive) ===
        order_entries = [
            item[len("orders_jsonfile:"):].strip().lower()
            for item in functions
            if isinstance(item, str) and item.lower().startswith("orders_jsonfile:")
        ]

        if not order_entries:
            print(f"[INFO] No order JSON files declared for broker '{user_brokerid}'")
            results[user_brokerid] = {
                "order_files": [],
                "temp_files": [],
                "found": 0,
                "copied": 0,
                "restructured": 0,
                "configs_copied": 0,
                "filtered": 0,
                "orphan_market_folders_found": [],
                "orphan_market_folders_deleted": [],
                "base_path": developer_folder
            }
            continue

        found_count = 0
        copied_count = 0
        restructured_count = 0
        configs_copied_count = 0
        filtered_count = 0
        orphan_folders_found = set()
        orphan_folders_deleted = set()

        # Collect removed symbols for folder cleanup
        removed_symbols_normalized = set()

        for root, _, files in os.walk(developer_folder):
            for f in files:
                if not f.lower().endswith('.json'):
                    continue

                full_path = os.path.join(root, f)
                base_name_no_ext_lower = os.path.splitext(f)[0].lower()

                if base_name_no_ext_lower.endswith('_temp'):
                    continue

                if base_name_no_ext_lower in order_entries:
                    found_count += 1
                    global_stats["found"] += 1

                    original_base_name = os.path.splitext(f)[0]
                    prefix = original_base_name + "_"

                    temp_filename = original_base_name + "_temp.json"
                    temp_path = os.path.join(root, temp_filename)

                    try:
                        # Read original JSON
                        with open(full_path, 'r', encoding='utf-8') as f_in:
                            original_data = json.load(f_in)

                        # === Extract all orders ===
                        orders_list = []
                        if isinstance(original_data, list):
                            orders_list = original_data
                        elif isinstance(original_data, dict):
                            if "orders" in original_data and isinstance(original_data["orders"], list):
                                orders_list = original_data["orders"]
                            elif "pending_orders" in original_data:
                                for sym_data in original_data["pending_orders"].values():
                                    for tf_list in sym_data.values():
                                        orders_list.extend(tf_list)
                            else:
                                orders_list = list(original_data.values())
                        orders_list = [o for o in orders_list if isinstance(o, dict)]

                        # === Load prefixed config ===
                        config_path = os.path.join(root, prefix + "allowedsymbolsandvolumes.json")
                        if not os.path.exists(config_path):
                            print(f"[WARNING] Prefixed config not found for {original_base_name}. All symbols filtered.")
                            config = {}
                        else:
                            with open(config_path, 'r', encoding='utf-8') as f_config:
                                config = json.load(f_config)

                        # === Filter and validate orders ===
                        valid_orders = []
                        removed_in_this_file = 0

                        for order in orders_list:
                            symbol = order.get("market_name") or order.get("symbol")
                            if not symbol:
                                removed_in_this_file += 1
                                continue

                            norm_symbol = normalize_symbol(symbol)

                            if norm_symbol not in all_dictator_symbols:
                                removed_symbols_normalized.add(norm_symbol)
                                removed_in_this_file += 1
                                continue

                            category = symbol_to_category.get(norm_symbol)
                            if not category:
                                removed_symbols_normalized.add(norm_symbol)
                                removed_in_this_file += 1
                                continue

                            cat_data = config.get(category)
                            if cat_data is None:
                                removed_symbols_normalized.add(norm_symbol)
                                removed_in_this_file += 1
                                continue

                            is_limited = cat_data.get("limited", False)
                            if is_limited:
                                allowed_symbols = {
                                    normalize_symbol(item.get("symbol", "")) for item in cat_data.get("allowed", [])
                                }
                                if norm_symbol not in allowed_symbols:
                                    removed_symbols_normalized.add(norm_symbol)
                                    removed_in_this_file += 1
                                    continue

                            valid_orders.append(order)

                        filtered_count += removed_in_this_file
                        global_stats["filtered"] += removed_in_this_file

                        # === Group orders by symbol and timeframe ===
                        grouped = defaultdict(lambda: defaultdict(list))
                        market_set = set()

                        for order in valid_orders:
                            symbol = order.get("market_name") or order.get("symbol")
                            timeframe = order.get("timeframe")
                            if symbol and timeframe:
                                grouped[symbol][timeframe].append(order)
                                market_set.add(symbol)

                        # === Build new restructured data with tick info at market level ===
                        new_orders_structure = {
                            "orders": {
                                "total_orders": len(valid_orders),
                                "total_markets": len(market_set)
                            }
                        }

                        orders_dict = new_orders_structure["orders"]

                        for market_name, tf_dict in grouped.items():
                            norm_market = normalize_symbol(market_name)

                            # Look up tick info (broker name already cleaned in symbolstick_data)
                            tick_info = symbolstick_data.get(norm_market, {})

                            # Only include if broker matches (cleaned comparison)
                            if tick_info and tick_info.get("broker") == cleaned_user_broker:
                                market_entry = {
                                    "broker": cleaned_user_broker,  # e.g., "deriv"
                                    "tick_size": tick_info.get("tick_size"),
                                    "tick_value": tick_info.get("tick_value"),
                                    **tf_dict  # Add all timeframe lists
                                }
                            else:
                                # Fallback: no tick info or broker mismatch
                                market_entry = dict(tf_dict)

                            orders_dict[market_name] = market_entry  # Use original market_name as key (preserves casing/spacing)

                        # === Save restructured _temp file ===
                        with open(temp_path, 'w', encoding='utf-8') as f_out:
                            json.dump(new_orders_structure, f_out, indent=2, ensure_ascii=False)

                        copied_count += 1
                        restructured_count += 1
                        global_stats["copied"] += 1
                        global_stats["restructured"] += 1

                        # === Copy prefixed config files ===
                        for src_path, original_name in required_files:
                            new_filename = prefix + original_name
                            dest_path = os.path.join(root, new_filename)
                            shutil.copy2(src_path, dest_path)
                            configs_copied_count += 1
                            global_stats["configs_copied"] += 1

                    except Exception as e:
                        print(f"[ERROR] Failed to process {full_path}: {e}")
                        continue

        # === Clean up orphan folders based on removed symbols ===
        if removed_symbols_normalized:
            print(f"[CLEANUP] Removing {len(removed_symbols_normalized)} disallowed/orphan market folders...")
            for root, dirs, _ in os.walk(developer_folder, topdown=True):
                for d in list(dirs):
                    norm_dir = normalize_symbol(d)
                    if norm_dir in removed_symbols_normalized:
                        full_dir_path = os.path.join(root, d)
                        try:
                            shutil.rmtree(full_dir_path)
                            print(f"[DELETED] Orphan folder: {full_dir_path}")
                            orphan_folders_found.add(full_dir_path)
                            orphan_folders_deleted.add(full_dir_path)
                            global_stats["orphan_folders_found"] += 1
                            global_stats["orphan_folders_deleted"] += 1
                            dirs.remove(d)
                        except Exception as del_e:
                            print(f"[ERROR] Could not delete {full_dir_path}: {del_e}")
                            orphan_folders_found.add(full_dir_path)
                            global_stats["orphan_folders_found"] += 1

        # === Per-broker summary ===
        print(f"\n{user_brokerid} ({cleaned_user_broker}): "
              f"Found {found_count}, Created {copied_count} _temp, "
              f"Restructured {restructured_count}, Copied {configs_copied_count} configs, "
              f"Filtered {filtered_count} orders, Deleted {len(orphan_folders_deleted)} orphan folders")

        results[user_brokerid] = {
            "order_files": order_entries,
            "temp_files": [f"{name}_temp.json" for name in order_entries],
            "found": found_count,
            "copied": copied_count,
            "restructured": restructured_count,
            "configs_copied": configs_copied_count,
            "filtered": filtered_count,
            "orphan_market_folders_found": sorted(list(orphan_folders_found)),
            "orphan_market_folders_deleted": sorted(list(orphan_folders_deleted)),
            "base_path": developer_folder
        }

    # === Global summary ===
    print(f"\n[GLOBAL SUMMARY]")
    print(f"Verified order files found: {global_stats['found']}")
    print(f"_temp files created: {global_stats['copied']}")
    print(f"Restructured with tick info: {global_stats['restructured']}")
    print(f"Config files copied: {global_stats['configs_copied']}")
    print(f"Invalid orders filtered: {global_stats['filtered']}")
    print(f"Orphan folders detected: {global_stats['orphan_folders_found']}")
    print(f"Orphan folders deleted: {global_stats['orphan_folders_deleted']}")

    return results   
 
def module_function_worker(user_brokerid: str, module_path: str, func_name: str):
    """
    Executed in each child process.
    Reloads the developer's module and runs the target function.
    All output (prints, errors, tracebacks) is completely silenced and safe from encoding issues.
    """
    import importlib.util
    import os
    import sys

    # Create a devnull stream that accepts any Unicode and ignores encoding errors
    devnull_stream = open(os.devnull, 'w', encoding='utf-8', errors='replace')

    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Redirect everything to devnull (safe from cp1252 Unicode errors)
    sys.stdout = devnull_stream
    sys.stderr = devnull_stream

    try:
        # Add developer folder to path
        developer_folder = os.path.dirname(module_path)
        if developer_folder not in sys.path:
            sys.path.insert(0, developer_folder)

        # Load the module dynamically
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Execute the target function if it exists and is callable
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if callable(func):
                func()  # All output inside the developer's function is silenced

    except Exception:
        # Completely silent – no traceback, no error printing
        # This prevents UnicodeEncodeError when tracebacks contain → or other chars
        pass
    finally:
        # Restore original streams (best effort)
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass
        try:
            devnull_stream.close()
        except:
            pass

def developers_functions():
    # 1. Sync data
    append_myfunctions_to_developers_functions()

    # 2. Get module metadata
    broker_data = load_users_functions_with_modules()

    import multiprocessing
    from collections import deque
    import os
    import shutil
    import json
    from collections import defaultdict

    prefix = "module_function:"
    FUNCTIONS_PER_BATCH = 10
    BROKERS_PER_ROUND = 20  # 20 brokers * 10 funcs = 200 concurrent max

    # === Paths used in the per-broker JSON processing ===
    DICTATOR_PATH = r"C:\xampp\htdocs\chronedge\synarex\chart\symbolscategory\symbolscategory.json"
    SYMBOLSTICK_PATH = r"C:\xampp\htdocs\chronedge\synarex\chart\symbolstick\symbolstick.json"

    # === Pre-load global data once (shared across all brokers) ===
    def load_global_data():
        def normalize_symbol(s):
            if not s:
                return ""
            return " ".join(s.replace("_", " ").split()).upper()

        def clean_broker(broker_name):
            if not broker_name:
                return ""
            return ''.join([char for char in broker_name if not char.isdigit()]).lower()

        # Load dictator
        dictator = {}
        if os.path.exists(DICTATOR_PATH):
            try:
                with open(DICTATOR_PATH, 'r', encoding='utf-8') as f:
                    dictator = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load dictator file: {e}")
        else:
            print("[WARNING] Dictator file not found.")

        all_dictator_symbols = set()
        symbol_to_category = {}
        for cat, sym_list in dictator.items():
            for s in sym_list:
                norm_s = normalize_symbol(s)
                all_dictator_symbols.add(norm_s)
                if norm_s not in symbol_to_category:
                    symbol_to_category[norm_s] = cat

        # Load symbolstick
        symbolstick_data = {}
        if os.path.exists(SYMBOLSTICK_PATH):
            try:
                with open(SYMBOLSTICK_PATH, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                for key, info in raw_data.items():
                    norm_key = normalize_symbol(key)
                    cleaned_broker = clean_broker(info.get("broker", ""))
                    symbolstick_data[norm_key] = {
                        "broker": cleaned_broker,
                        "tick_size": info.get("tick_size"),
                        "tick_value": info.get("tick_value"),
                        "original_market": info.get("market")
                    }
            except Exception as e:
                print(f"[WARNING] Failed to load symbolstick.json: {e}")
        else:
            print("[WARNING] symbolstick.json not found.")

        return all_dictator_symbols, symbol_to_category, symbolstick_data

    all_dictator_symbols, symbol_to_category, symbolstick_data = load_global_data()

    # === Per-broker JSON order processing helper (exact logic from original locate_developer_order_jsons) ===
    def process_broker_orders(user_brokerid, developer_folder, order_entries):
        def normalize_symbol(s):
            if not s:
                return ""
            return " ".join(s.replace("_", " ").split()).upper()

        def clean_broker(broker_name):
            if not broker_name:
                return ""
            return ''.join([char for char in broker_name if not char.isdigit()]).lower()

        cleaned_user_broker = clean_broker(user_brokerid)

        # Required config paths (root level)
        allowed_file_path = os.path.join(developer_folder, "allowedsymbolsandvolumes.json")
        tradeslimit_file_path = os.path.join(developer_folder, "disabledorders.json")
        accountmanagement_file_path = os.path.join(developer_folder, "accountmanagement.json")

        required_files = [
            (allowed_file_path, "allowedsymbolsandvolumes.json"),
            (tradeslimit_file_path, "disabledorders.json"),
            (accountmanagement_file_path, "accountmanagement.json")
        ]

        missing = [name for path, name in required_files if not os.path.exists(path)]
        if missing:
            print(f"[REJECTED] Missing required root files for broker '{user_brokerid}': {missing}. Skipping order processing.")
            return

        found_count = copied_count = restructured_count = configs_copied_count = filtered_count = 0
        orphan_folders_found = set()
        orphan_folders_deleted = set()
        removed_symbols_normalized = set()

        for root, _, files in os.walk(developer_folder):
            for f in files:
                if not f.lower().endswith('.json'):
                    continue
                full_path = os.path.join(root, f)
                base_name_no_ext_lower = os.path.splitext(f)[0].lower()

                if base_name_no_ext_lower.endswith('_temp'):
                    continue

                if base_name_no_ext_lower not in order_entries:
                    continue

                found_count += 1
                original_base_name = os.path.splitext(f)[0]
                prefix_name = original_base_name + "_"
                temp_path = os.path.join(root, original_base_name + "_temp.json")

                try:
                    with open(full_path, 'r', encoding='utf-8') as f_in:
                        original_data = json.load(f_in)

                    # Extract orders
                    orders_list = []
                    if isinstance(original_data, list):
                        orders_list = original_data
                    elif isinstance(original_data, dict):
                        if "orders" in original_data and isinstance(original_data["orders"], list):
                            orders_list = original_data["orders"]
                        elif "pending_orders" in original_data:
                            for sym_data in original_data["pending_orders"].values():
                                for tf_list in sym_data.values():
                                    orders_list.extend(tf_list)
                        else:
                            orders_list = list(original_data.values())
                    orders_list = [o for o in orders_list if isinstance(o, dict)]

                    # Load prefixed config
                    config_path = os.path.join(root, prefix_name + "allowedsymbolsandvolumes.json")
                    if not os.path.exists(config_path):
                        print(f"[WARNING] Prefixed config not found for {original_base_name}. All symbols filtered.")
                        config = {}
                    else:
                        with open(config_path, 'r', encoding='utf-8') as f_config:
                            config = json.load(f_config)

                    # Filter orders
                    valid_orders = []
                    removed_in_this_file = 0
                    for order in orders_list:
                        symbol = order.get("market_name") or order.get("symbol")
                        if not symbol:
                            removed_in_this_file += 1
                            continue
                        norm_symbol = normalize_symbol(symbol)
                        if norm_symbol not in all_dictator_symbols:
                            removed_symbols_normalized.add(norm_symbol)
                            removed_in_this_file += 1
                            continue
                        category = symbol_to_category.get(norm_symbol)
                        if not category:
                            removed_symbols_normalized.add(norm_symbol)
                            removed_in_this_file += 1
                            continue
                        cat_data = config.get(category)
                        if cat_data is None:
                            removed_symbols_normalized.add(norm_symbol)
                            removed_in_this_file += 1
                            continue
                        is_limited = cat_data.get("limited", False)
                        if is_limited:
                            allowed_symbols = {normalize_symbol(item.get("symbol", "")) for item in cat_data.get("allowed", [])}
                            if norm_symbol not in allowed_symbols:
                                removed_symbols_normalized.add(norm_symbol)
                                removed_in_this_file += 1
                                continue
                        valid_orders.append(order)

                    filtered_count += removed_in_this_file

                    # Group by symbol & timeframe
                    grouped = defaultdict(lambda: defaultdict(list))
                    market_set = set()
                    for order in valid_orders:
                        symbol = order.get("market_name") or order.get("symbol")
                        timeframe = order.get("timeframe")
                        if symbol and timeframe:
                            grouped[symbol][timeframe].append(order)
                            market_set.add(symbol)

                    # Build new structure with tick info AND category
                    new_orders_structure = {
                        "orders": {
                            "total_orders": len(valid_orders),
                            "total_markets": len(market_set)
                        }
                    }
                    orders_dict = new_orders_structure["orders"]
                    for market_name, tf_dict in grouped.items():
                        norm_market = normalize_symbol(market_name)
                        category = symbol_to_category.get(norm_market, "unknown")
                        tick_info = symbolstick_data.get(norm_market, {})
                        if tick_info and tick_info.get("broker") == cleaned_user_broker:
                            market_entry = {
                                "broker": cleaned_user_broker,
                                "category": category,
                                "tick_size": tick_info.get("tick_size"),
                                "tick_value": tick_info.get("tick_value"),
                                **tf_dict
                            }
                        else:
                            market_entry = {
                                "category": category,
                                **tf_dict
                            }
                        orders_dict[market_name] = market_entry

                    # Write _temp file
                    with open(temp_path, 'w', encoding='utf-8') as f_out:
                        json.dump(new_orders_structure, f_out, indent=2, ensure_ascii=False)

                    copied_count += 1
                    restructured_count += 1

                    # Copy prefixed configs
                    for src_path, original_name in required_files:
                        dest_path = os.path.join(root, prefix_name + original_name)
                        shutil.copy2(src_path, dest_path)
                        configs_copied_count += 1

                except Exception as e:
                    print(f"[ERROR] Failed to process {full_path}: {e}")
                    continue

        # Orphan folder cleanup
        if removed_symbols_normalized:
            print(f"[CLEANUP] Removing {len(removed_symbols_normalized)} orphan market folders for {user_brokerid}...")
            for root, dirs, _ in os.walk(developer_folder, topdown=True):
                for d in list(dirs):
                    norm_dir = normalize_symbol(d)
                    if norm_dir in removed_symbols_normalized:
                        full_dir_path = os.path.join(root, d)
                        try:
                            shutil.rmtree(full_dir_path)
                            print(f"[DELETED] Orphan folder: {full_dir_path}")
                            orphan_folders_deleted.add(full_dir_path)
                        except Exception as del_e:
                            print(f"[ERROR] Could not delete {full_dir_path}: {del_e}")
                        orphan_folders_found.add(full_dir_path)
                        dirs.remove(d)  # prevent os.walk from entering it

        # Per-broker summary
        print(f"\n{user_brokerid} ORDER PROCESSING COMPLETE: "
              f"Found {found_count}, Created {copied_count} _temp, "
              f"Restructured {restructured_count}, Copied {configs_copied_count} configs, "
              f"Filtered {filtered_count} orders, Deleted {len(orphan_folders_deleted)} orphan folders")

    # === Main processing starts here ===
    print("[INFO] Collecting and validating developer functions...")

    broker_functions = {}  # broker_id -> deque of (func_name, file_path, developer_folder, order_entries)
    broker_order_entries = {}  # broker_id -> list of lowercase order filenames

    for entry in broker_data:
        user_brokerid = entry["user_brokerid"]
        filename = entry["filename"]
        developer_folder = os.path.dirname(entry["file_path"])

        if not validate_developer_credentials(user_brokerid, developer_folder):
            print(f"[SKIPPED] Broker '{user_brokerid}' failed credential validation. Skipping module '{filename}'.")
            continue

        required_files = [
            "allowedsymbolsandvolumes.json",
            "disabledorders.json",
            "accountmanagement.json"
        ]
        missing = [f for f in required_files if not os.path.exists(os.path.join(developer_folder, f))]
        if missing:
            print(f"[REJECTED] Missing file(s): {missing} for broker '{user_brokerid}'. Skipping.")
            continue

        print(f"[VALID] All required config files found for '{user_brokerid}'.")

        try:
            spec = importlib.util.spec_from_file_location(entry["module_name"], entry["file_path"])
            if spec is None or spec.loader is None:
                print(f"[ERROR] Could not create module spec for {filename}")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[entry["module_name"]] = module
            spec.loader.exec_module(module)

            print(f"[APPROVED] Module loaded: {user_brokerid} → {filename}")

            valid_funcs = []
            for item in entry["functions"]:
                item_str = item.strip()
                if item_str.startswith(prefix):
                    func_name = item_str[len(prefix):].strip()
                    if hasattr(module, func_name) and callable(getattr(module, func_name)):
                        valid_funcs.append((func_name, entry["file_path"]))

            # Also collect order_jsonfile entries for later processing
            order_entries_lower = [
                item[len("orders_jsonfile:"):].strip().lower()
                for item in entry["functions"]
                if isinstance(item, str) and item.lower().startswith("orders_jsonfile:")
            ]
            broker_order_entries[user_brokerid] = order_entries_lower

            if valid_funcs:
                broker_functions[user_brokerid] = deque([
                    (func_name, file_path, developer_folder, order_entries_lower)
                    for func_name, file_path in valid_funcs
                ])
                print(f"[QUEUED] {len(valid_funcs)} function(s) queued for broker '{user_brokerid}'.")

        except Exception as e:
            print(f"[ERROR] Failed to load module {filename}: {e}")

    if not broker_functions:
        print("[INFO] No developer functions to run.")
        return

    broker_list = list(broker_functions.keys())
    total_rounds = 0

    print(f"[START] Beginning round-robin processing: {FUNCTIONS_PER_BATCH} functions per broker, "
          f"{BROKERS_PER_ROUND} brokers per round (max 200 concurrent).")

    while any(broker_functions.values()):
        total_rounds += 1
        print(f"\n=== ROUND {total_rounds} STARTED ===")

        active_processes = []
        launched_this_round = 0

        for broker in broker_list:
            if not broker_functions.get(broker):
                continue

            funcs_to_launch = min(FUNCTIONS_PER_BATCH, len(broker_functions[broker]))

            for _ in range(funcs_to_launch):
                if launched_this_round >= BROKERS_PER_ROUND * FUNCTIONS_PER_BATCH:
                    break

                func_name, file_path, developer_folder, order_entries_lower = broker_functions[broker].popleft()

                print(f"[START] Launching {broker} → {func_name}() (output silenced)")

                p = multiprocessing.Process(
                    target=module_function_worker,
                    args=(broker, file_path, func_name)
                )
                p.start()

                active_processes.append((broker, func_name, p, developer_folder, order_entries_lower))
                launched_this_round += 1

            if launched_this_round >= BROKERS_PER_ROUND * FUNCTIONS_PER_BATCH:
                break

        if not active_processes:
            print("[INFO] No functions launched this round — all done.")
            break

        print(f"[WAITING] Waiting for {len(active_processes)} functions in Round {total_rounds} to complete...")

        # Wait for all processes in this round
        for broker, func_name, proc, developer_folder, order_entries_lower in active_processes:
            proc.join()
            status = "successfully" if proc.exitcode == 0 else f"with error (code {proc.exitcode})"
            print(f"[DONE] {broker} → {func_name}() completed {status}")

            # === AFTER function completes: if this was the LAST function for this broker, run order processing ===
            remaining = len(broker_functions.get(broker, []))
            if remaining == 0 and broker in broker_order_entries and broker_order_entries[broker]:
                print(f"[ORDER PROCESSING] Starting JSON order file processing for broker '{broker}'...")
                process_broker_orders(broker, developer_folder, broker_order_entries[broker])

        print(f"=== ROUND {total_rounds} COMPLETED ===\n")

    # Final cleanup: any brokers that finished in the last round but weren't caught above
    for broker in broker_list:
        if broker in broker_order_entries and broker_order_entries[broker]:
            # Check if any functions were ever queued — if no queue existed, still process orders if declared
            if broker not in broker_functions or not broker_functions[broker]:
                print(f"[ORDER PROCESSING] Final pass for broker '{broker}' (no functions or already processed)...")
                # Find developer_folder again if needed (fallback from broker_data)
                for entry in broker_data:
                    if entry["user_brokerid"] == broker:
                        dev_folder = os.path.dirname(entry["file_path"])
                        process_broker_orders(broker, dev_folder, broker_order_entries[broker])
                        break

    print("[FINISHED] All developer functions and per-broker JSON order processing completed.")




if __name__ == "__main__":
   developers_functions() 

    
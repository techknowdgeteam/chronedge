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

def get_brokers_dictionary_path():
    PATH_CONFIG_FILE = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionarypath.json"

    if not os.path.exists(PATH_CONFIG_FILE):
        raise FileNotFoundError(f"Path config file not found: {PATH_CONFIG_FILE}")

    try:
        with open(PATH_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        brokers_path = config.get("brokers_dictionary_path")
        if not brokers_path:
            raise ValueError("Key 'brokers_dictionary_path' missing in config file")
        
        # Optional: Normalize path (handles both / and \)
        brokers_path = os.path.normpath(brokers_path)
        
        if not os.path.exists(brokers_path):
            raise FileNotFoundError(f"Brokers file not found at: {brokers_path}")
        
        return brokers_path

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {PATH_CONFIG_FILE}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load path config: {e}")


def load_brokers_dictionary(account_email_or_login="", account_password=""):
    """
    Load brokers config from JSON file with error handling and fallback.
    
    If account_email_or_login and account_password are provided (non-empty),
    ONLY return the matching broker's config if BOTH:
      - ACCOUNT_EMAIL matches account_email_or_login
      - ACCOUNT_PASSWORD matches account_password
    Otherwise return all brokers.
    
    Matching is done STRICTLY using only ACCOUNT_EMAIL and ACCOUNT_PASSWORD fields.
    """
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"

    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Normalize some fields (still useful even if we don't match on them)
        for broker_name, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        # If credentials are provided → find matching broker using ONLY ACCOUNT_EMAIL and ACCOUNT_PASSWORD
        if account_email_or_login and account_password:
            for broker_name, cfg in data.items():
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
                    print(f"Successfully matched and loaded broker '{broker_name}' using ACCOUNT_EMAIL and ACCOUNT_PASSWORD.", "SUCCESS")
                    return {broker_name: cfg}
            
            # No match found
            print("No broker matched the provided ACCOUNT_EMAIL and ACCOUNT_PASSWORD.", "WARNING")
            return {}
        
        # No credentials provided → load all
        print(f"Brokers config loaded successfully → {len(data)} broker(s)", "SUCCESS")
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in brokersdictionary.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load brokersdictionary.json: {e}", "CRITICAL")
        return {}
brokersdictionary = load_brokers_dictionary()


def append_myfunctions_to_usersfunctions():
    import os
    import json

    # Paths
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    brokers_dir = os.path.dirname(BROKERS_JSON_PATH)
    usersfunctions_path = os.path.join(brokers_dir, "usersfunctions.json")

    # Load developer brokers from the global dictionary
    developer_brokers = {
        name: cfg for name, cfg in globals().get("brokersdictionary", {}).items()
        if cfg.get("POSITION", "").lower() == "developer"
    }

    if not developer_brokers:
        print("[ERROR] No developer brokers found in brokersdictionary")
        return

    print(f"[INFO] Found {len(developer_brokers)} developer broker(s): {list(developer_brokers.keys())}")

    # Load existing usersfunctions.json to maintain other (non-developer) brokers
    if os.path.exists(usersfunctions_path):
        try:
            with open(usersfunctions_path, 'r', encoding='utf-8') as f:
                users_funcs_dict = json.load(f)
            if not isinstance(users_funcs_dict, dict):
                users_funcs_dict = {}
        except Exception as e:
            print(f"[WARNING] Could not read usersfunctions.json ({e}) → starting fresh")
            users_funcs_dict = {}
    else:
        users_funcs_dict = {}

    changes_made = False
    updated_brokers = []

    for broker_name, cfg in developer_brokers.items():
        print(f"\n[PROCESSING] Broker: {broker_name}")

        base_folder = cfg.get("BASE_FOLDER", "")
        if not base_folder:
            continue

        # Locate continuation.json to find the developer's source directory
        possible_continuation_paths = [
            os.path.join(base_folder, "..", "developers", broker_name, "continuation.json"),
            os.path.join(base_folder, "continuation.json")
        ]

        myfunctions_dir = None
        for cont_path in possible_continuation_paths:
            if os.path.exists(cont_path):
                myfunctions_dir = os.path.dirname(cont_path)
                break

        if myfunctions_dir is None:
            print(f"[SKIP] Could not locate continuation.json for broker '{broker_name}'")
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
        new_broker_list.append(f"developer: {broker_name}")

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
        old_list = users_funcs_dict.get(broker_name, [])
        if old_list != new_broker_list:
            users_funcs_dict[broker_name] = new_broker_list
            changes_made = True
            updated_brokers.append(broker_name)
            print(f"[SYNC] Overwrote '{broker_name}' with {len(new_broker_list)} current items.")
        else:
            print(f"[INFO] '{broker_name}' is already perfectly in sync.")

    # === SAVE CHANGES ===
    if changes_made:
        try:
            with open(usersfunctions_path, 'w', encoding='utf-8') as f:
                json.dump(users_funcs_dict, f, indent=2, ensure_ascii=False)
            print(f"\n[SUCCESS] usersfunctions.json UPDATED. Brokers synced: {updated_brokers}")
        except Exception as e:
            print(f"[ERROR] Failed to save usersfunctions.json: {e}")
    else:
        print("\n[INFO] All brokers were already in sync. No file write needed.")




def load_users_functions_with_modules() -> List[Dict[str, Any]]:
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    brokers_dir = os.path.dirname(BROKERS_JSON_PATH)
    usersfunctions_path = os.path.join(brokers_dir, "usersfunctions.json")

    if not os.path.exists(usersfunctions_path):
        print(f"[WARNING] usersfunctions.json not found at {usersfunctions_path}")
        return []

    try:
        with open(usersfunctions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        brokersdictionary = globals().get("brokersdictionary", {})

        for broker_name, items in data.items():
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
            if broker_name not in brokersdictionary:
                continue

            cfg = brokersdictionary[broker_name]
            base_folder = cfg["BASE_FOLDER"]
            
            # Locate the developer folder
            developer_folder = None
            possible_paths = [
                os.path.join(base_folder, "..", "developers", broker_name, "continuation.json"),
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
            # We now rely solely on session.json + credential validation in usersfunctions()

            results.append({
                "broker_name": broker_name,
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

def validate_broker_credentials(broker_name: str, base_folder: str) -> bool:
    """
    Validate that the broker has a valid session.json with credentials
    that EXACTLY match ACCOUNT_EMAIL and ACCOUNT_PASSWORD in brokersdictionary.json
    
    Returns True only if fully valid.
    """
    global brokersdictionary
    
    if broker_name not in brokersdictionary:
        print(f"[REJECTED] Broker '{broker_name}' not found in brokersdictionary.json")
        return False
    
    cfg = brokersdictionary[broker_name]
    expected_email = cfg.get("ACCOUNT_EMAIL")
    expected_password = cfg.get("ACCOUNT_PASSWORD")
    
    if not expected_email or not expected_password:
        print(f"[REJECTED] Broker '{broker_name}' missing ACCOUNT_EMAIL or ACCOUNT_PASSWORD in brokersdictionary.json")
        return False
    
    # Construct expected developer folder path
    developers_path = os.path.join(base_folder, "..", "developers", broker_name)
    if not os.path.exists(developers_path):
        # Fallback: some setups have flat structure
        developers_path = base_folder
    
    session_path = os.path.join(developers_path, "session.json")
    if not os.path.exists(session_path):
        print(f"[REJECTED] No session.json found for '{broker_name}' at {session_path}")
        return False
    
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        session_email = session_data.get("email", "").strip()
        session_password = session_data.get("password", "").strip()
        
        if session_email == expected_email and session_password == expected_password:
            print(f"[VALID] Credentials match for broker '{broker_name}'")
            return True
        else:
            print(f"[REJECTED] Credentials mismatch for '{broker_name}' "
                  f"(email: {'match' if session_email == expected_email else 'no match'}, "
                  f"password: {'match' if session_password == expected_password else 'no match'})")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to read session.json for '{broker_name}': {e}")
        return False

def usersfunctions():
    # 1. Sync data
    append_myfunctions_to_usersfunctions()  

    # 2. Get module metadata
    broker_data = load_users_functions_with_modules()

    processes = []
    prefix = "module_function:"

    for entry in broker_data:
        broker_name = entry["broker_name"]
        filename = entry["filename"]
        developer_folder = os.path.dirname(entry["file_path"])

        # === STEP 1: CREDENTIAL VALIDATION ===
        if not validate_broker_credentials(broker_name, developer_folder):
            print(f"[SKIPPED] Broker '{broker_name}' failed credential validation. Skipping module '{filename}'.")
            continue

        # === STEP 2: CHECK REQUIRED CONFIGURATION FILES ===
        allowed_file_path = os.path.join(developer_folder, "allowedsymbolsandvolumes.json")
        tradeslimit_file_path = os.path.join(developer_folder, "tradeslimit.json")
        accountmanagement_file_path = os.path.join(developer_folder, "accountmanagement.json")

        # Check allowedsymbolsandvolumes.json
        if not os.path.exists(allowed_file_path):
            print(f"[REJECTED] Missing required file: allowedsymbolsandvolumes.json "
                  f"for broker '{broker_name}' in {developer_folder}. Skipping module '{filename}'.")
            continue

        # Check tradeslimit.json
        if not os.path.exists(tradeslimit_file_path):
            print(f"[REJECTED] Missing required file: tradeslimit.json "
                  f"for broker '{broker_name}' in {developer_folder}. Skipping module '{filename}'.")
            continue

        # Check accountmanagement.json
        if not os.path.exists(accountmanagement_file_path):
            print(f"[REJECTED] Missing required file: accountmanagement.json "
                  f"for broker '{broker_name}' in {developer_folder}. Skipping module '{filename}'.")
            continue

        print(f"[VALID] Both allowedsymbolsandvolumes.json, tradeslimit.json and account balance rules found for '{broker_name}'.")

        # === ALL CHECKS PASSED → PROCEED TO LOAD AND EXECUTE MODULE ===
        try:
            # Load and execute the module
            spec = importlib.util.spec_from_file_location(entry["module_name"], entry["file_path"])
            if spec is None or spec.loader is None:
                print(f"[ERROR] Could not create module spec for {filename}")
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[entry["module_name"]] = module
            spec.loader.exec_module(module)

            print(f"[APPROVED] Loading fully authenticated & configured module: {broker_name} -> {filename}")

            # Launch declared functions
            for item in entry["functions"]:
                item_str = item.strip()
                if item_str.startswith(prefix):
                    func_name = item_str.replace(prefix, "").strip()

                    if hasattr(module, func_name):
                        func = getattr(module, func_name)
                        if callable(func):
                            print(f"[START] Launching {broker_name} -> {func_name}()")
                            p = multiprocessing.Process(target=func)
                            p.start()
                            processes.append((broker_name, func_name, p))
                        else:
                            print(f"[WARNING] '{func_name}' exists but is not callable in {filename}")
                    else:
                        print(f"[ERROR] Function '{func_name}' not found in module {filename}")

        except Exception as e:
            print(f"[ERROR] Failed to load or execute module {filename}: {e}")

    # Wait for all valid processes to complete
    if processes:
        print(f"[WAITING] Waiting for {len(processes)} function(s) to complete...")
        for b_name, f_name, p in processes:
            p.join()
            print(f"[DONE] {b_name} -> {f_name}() completed")
    else:
        print("[INFO] No fully authenticated and configured brokers with functions to run.")       

def locate_broker_order_jsonfiles():
    """
    Scans all loaded broker modules (from usersfunctions.json) and locates
    any JSON order files referenced via 'orders_jsonfile:' entries.
    
    Validation steps (same as usersfunctions()):
    - Validates broker credentials
    - Checks existence of allowedsymbolsandvolumes.json, tradeslimit.json, and accountmanagement.json
    
    For each matching, validated JSON file:
    - Creates/overwrites a backup copy with '_temp.json' appended
    - Restructures the content into a clean, standardized format
    - Copies the three configuration files (allowedsymbols, tradeslimit, accountmanagement)
      into the same directory as the _temp.json file
    
    Output is limited to summary only.
    """
    import os
    import shutil
    import json
    from collections import defaultdict

    # 1. Sync data first
    append_myfunctions_to_usersfunctions()  # Assuming this function exists

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
        "configs_copied": 0
    }

    for entry in broker_data:
        broker_name = entry["broker_name"]
        filename = entry["filename"]
        file_path = entry["file_path"]
        functions = entry["functions"]

        developer_folder = os.path.dirname(os.path.abspath(file_path))

        # === STEP 1: CREDENTIAL VALIDATION ===
        if not validate_broker_credentials(broker_name, developer_folder):
            print(f"[SKIPPED] Broker '{broker_name}' failed credential validation. Skipping module '{filename}'.")
            continue

        # === STEP 2: CHECK REQUIRED CONFIGURATION FILES ===
        allowed_file_path = os.path.join(developer_folder, "allowedsymbolsandvolumes.json")
        tradeslimit_file_path = os.path.join(developer_folder, "tradeslimit.json")
        accountmanagement_file_path = os.path.join(developer_folder, "accountmanagement.json")

        if not os.path.exists(allowed_file_path):
            print(f"[REJECTED] Missing allowedsymbolsandvolumes.json for broker '{broker_name}' in {developer_folder}. Skipping.")
            continue

        if not os.path.exists(tradeslimit_file_path):
            print(f"[REJECTED] Missing tradeslimit.json for broker '{broker_name}' in {developer_folder}. Skipping.")
            continue

        if not os.path.exists(accountmanagement_file_path):
            print(f"[REJECTED] Missing accountmanagement.json for broker '{broker_name}' in {developer_folder}. Skipping.")
            continue

        print(f"[VALID] All required config files found for broker '{broker_name}' ({filename})")

        # === Collect order JSON file names ===
        order_entries = [
            item[len("orders_jsonfile:"):].strip().lower()
            for item in functions
            if isinstance(item, str) and item.startswith("orders_jsonfile:")
        ]

        if not order_entries:
            print(f"[INFO] No order JSON files declared for broker '{broker_name}'")
            results[broker_name] = {
                "order_files": [],
                "temp_files": [],
                "found": 0,
                "copied": 0,
                "restructured": 0,
                "configs_copied": 0,
                "base_path": developer_folder
            }
            continue

        found_count = 0
        copied_count = 0
        restructured_count = 0
        configs_copied_count = 0

        for root, _, files in os.walk(developer_folder):
            for f in files:
                if not f.lower().endswith('.json'):
                    continue

                full_path = os.path.join(root, f)
                base_name_no_ext = os.path.splitext(f)[0].lower()

                if base_name_no_ext.endswith('_temp'):
                    continue

                if base_name_no_ext in order_entries:
                    found_count += 1
                    global_stats["found"] += 1

                    temp_filename = os.path.splitext(f)[0] + "_temp.json"
                    temp_path = os.path.join(root, temp_filename)

                    try:
                        # Read original JSON
                        with open(full_path, 'r', encoding='utf-8') as f_in:
                            original_data = json.load(f_in)

                        # === Collect all orders ===
                        orders_list = []
                        if isinstance(original_data, list):
                            orders_list = original_data
                        elif isinstance(original_data, dict):
                            if "orders" in original_data and isinstance(original_data["orders"], list):
                                orders_list = original_data["orders"]
                            elif isinstance(original_data.get("pending_orders"), dict):
                                for sym_data in original_data.get("pending_orders", {}).values():
                                    for tf_list in sym_data.values():
                                        orders_list.extend(tf_list)
                            else:
                                orders_list = list(original_data.values())

                        orders_list = [o for o in orders_list if isinstance(o, dict)]

                        # === Group orders ===
                        grouped = defaultdict(lambda: defaultdict(list))
                        market_set = set()

                        for order in orders_list:
                            symbol = order.get("market_name") or order.get("symbol") or None
                            timeframe = order.get("timeframe") or None
                            if symbol and timeframe:
                                grouped[symbol][timeframe].append(order)
                                market_set.add(symbol)

                        # === Calculate fresh summary ===
                        total_orders = len(orders_list)
                        total_markets = len(market_set)

                        # === Build new structure ===
                        restructured = {
                            "orders": {
                                "total_orders": total_orders,
                                "total_markets": total_markets,
                                **dict(grouped)
                            }
                        }

                        # === Save to _temp ===
                        with open(temp_path, 'w', encoding='utf-8') as f_out:
                            json.dump(restructured, f_out, indent=2, ensure_ascii=False)

                        copied_count += 1
                        restructured_count += 1
                        global_stats["copied"] += 1
                        global_stats["restructured"] += 1

                        # === Copy the three configuration files to the same directory ===
                        for config_path in [allowed_file_path, tradeslimit_file_path, accountmanagement_file_path]:
                            config_filename = os.path.basename(config_path)
                            dest_config_path = os.path.join(root, config_filename)
                            shutil.copy2(config_path, dest_config_path)
                            configs_copied_count += 1
                            global_stats["configs_copied"] += 1

                    except Exception as e:
                        print(f"[WARNING] Failed to process {full_path} for broker '{broker_name}': {e}")
                        continue

        # === Per-broker summary ===
        print(f"{broker_name}: Found {found_count} verified order JSON file(s), "
              f"created/overwritten {copied_count} _temp backup(s), "
              f"restructured {restructured_count} file(s), "
              f"copied {configs_copied_count} config file(s) to order directories")

        results[broker_name] = {
            "order_files": order_entries,
            "temp_files": [f"{name}_temp.json" for name in order_entries],
            "found": found_count,
            "copied": copied_count,
            "restructured": restructured_count,
            "configs_copied": configs_copied_count,
            "base_path": developer_folder
        }

    # === Global summary ===
    print(f"\n[GLOBAL] Total verified original order JSON files found: {global_stats['found']}")
    print(f"[GLOBAL] Total _temp backups created/overwritten: {global_stats['copied']}")
    print(f"[GLOBAL] Total _temp files restructured: {global_stats['restructured']}")
    print(f"[GLOBAL] Total config files copied to order destinations: {global_stats['configs_copied']}")

    return results

if __name__ == "__main__":
    locate_broker_order_jsonfiles()
    
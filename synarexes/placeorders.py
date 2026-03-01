import os
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import traceback
import shutil
from datetime import datetime
import re
from pathlib import Path
import math
import multiprocessing as mp
from pathlib import Path
import time
import random

INVESTOR_USERS = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors\investors.json"
INV_PATH = r"C:\xampp\htdocs\chronedge\synarex\usersdata\investors"
NORMALIZE_SYMBOLS_PATH = r"C:\xampp\htdocs\chronedge\synarex\symbols_normalization.json"
DEFAULT_ACCOUNTMANAGEMENT = r"C:\xampp\htdocs\chronedge\synarex\default_accountmanagement.json"
DEFAULT_PATH = r"C:\xampp\htdocs\chronedge\synarex"
NORM_FILE_PATH = Path(DEFAULT_PATH) / "symbols_normalization.json"

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

def accountmanagement_manager(inv_id):
    """
    Updates accountmanagement.json field-by-field.
    Maintains 'Flex' look: lists on one line, dictionaries vertical.
    Handles both Default and Maximum risk management tables.
    """
    print(f"\n{'='*10} ⚙️ MANAGING ACCOUNT MANAGEMENT: {inv_id} {'='*10}")
    
    # 1. Setup Paths
    inv_folder = Path(INV_PATH) / inv_id
    inv_acc_mgmt_path = inv_folder / "accountmanagement.json"
    
    # 2. Identify Broker Template
    broker_cfg = usersdictionary.get(inv_id)
    if not broker_cfg:
        print(f" [!] Error: No broker config for {inv_id}")
        return False
    
    server = broker_cfg.get('SERVER', '')
    inv_broker_name = server.split('-')[0].split('.')[0].lower() if server else 'broker'
    template_filename = f"default{inv_broker_name}_accountmanagement.json"
    template_path = Path(DEFAULT_PATH) / template_filename

    if not template_path.exists():
        template_path = Path(DEFAULT_PATH) / "default_accountmanagement.json"
        if not template_path.exists(): return False

    # 3. Load Data
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        if inv_acc_mgmt_path.exists():
            with open(inv_acc_mgmt_path, 'r', encoding='utf-8') as f:
                inv_data = json.load(f)
        else:
            inv_data = {}
    except Exception as e:
        print(f" [!] Load Error: {e}"); return False

    modified = False

    # --- FIELD-BY-FIELD FLEXIBLE LOGIC ---

    # selected_risk_reward
    curr_rr = inv_data.get("selected_risk_reward")
    if "selected_risk_reward" not in inv_data or curr_rr in [None, 0, [], [0], ""]:
        inv_data["selected_risk_reward"] = template_data.get("selected_risk_reward", [3])
        modified = True

    # symbols_dictionary
    if "symbols_dictionary" not in inv_data or not inv_data.get("symbols_dictionary"):
        inv_data["symbols_dictionary"] = template_data.get("symbols_dictionary", {})
        modified = True

    # settings (Sub-field check)
    template_settings = template_data.get("settings", {})
    if "settings" not in inv_data:
        inv_data["settings"] = template_settings
        modified = True
    else:
        for key, value in template_settings.items():
            if key not in inv_data["settings"]:
                inv_data["settings"][key] = value
                modified = True

    # account_balance_default_risk_management
    def_risk_key = "account_balance_default_risk_management"
    if def_risk_key not in inv_data or not inv_data.get(def_risk_key):
        inv_data[def_risk_key] = template_data.get(def_risk_key, {})
        modified = True

    # account_balance_maximum_risk_management (NEW)
    max_risk_key = "account_balance_maximum_risk_management"
    if max_risk_key not in inv_data or not inv_data.get(max_risk_key):
        inv_data[max_risk_key] = template_data.get(max_risk_key, {})
        modified = True
        print(f" └─ ✅ Added/Updated Maximum Risk Management table")

    # 4. Save with "Flex" Formatting
    if modified:
        try:
            # Generate standard JSON
            json_string = json.dumps(inv_data, indent=4)
            
            # Logic to flatten ONLY lists [ ... ] to keep them 'flex'
            # This regex captures lists but leaves large dictionaries vertical
            flex_format = re.sub(
                r'\[\s+([^\]\{\}]+?)\s+\]', 
                lambda m: "[ " + re.sub(r'\s+', ' ', m.group(1)).strip() + " ]", 
                json_string
            )

            with open(inv_acc_mgmt_path, 'w', encoding='utf-8') as f:
                f.write(flex_format)
                
            print(f" └─ 💾 {inv_id} accountmanagement.json synced successfully.")
            return True
        except Exception as e:
            print(f" └─ ❌ Save Error: {e}"); return False
    
    print(f" └─ 🔘 {inv_id} already contains all required fields.")
    return True

def get_normalized_symbol(record_symbol, risk_keys=None):
    """
    Standardizes symbols with a 'Broker-First' priority.
    If 'US OIL' is passed, it finds the USOIL family, then checks if the broker
    uses USOUSD, USOIL, or WTI.
    """
    if not record_symbol: return None

    NORM_PATH = Path(r"C:\xampp\htdocs\chronedge\synarex\symbols_normalization.json")
    
    def clean(s): 
        return str(s).replace(" ", "").replace("_", "").replace("/", "").replace(".", "").upper()

    search_term = clean(record_symbol)
    
    # 1. Load Normalization Map
    norm_data = {}
    if NORM_PATH.exists():
        try:
            with open(NORM_PATH, 'r', encoding='utf-8') as f:
                norm_data = json.load(f).get("NORMALIZATION", {})
        except: pass

    # 2. Find the "Family"
    target_family_key = None
    all_family_variants = []
    
    for std_key, synonyms in norm_data.items():
        family_variants = [clean(std_key)] + [clean(s) for s in synonyms]
        if any(search_term == v or search_term.startswith(v) or v.startswith(search_term) for v in family_variants):
            target_family_key = std_key
            all_family_variants = family_variants
            break

    # 3. IF RISK_KEYS ARE PROVIDED (For Risk Enforcement)
    if risk_keys:
        clean_risk_map = {clean(k): k for k in risk_keys}
        if target_family_key and clean(target_family_key) in clean_risk_map:
            return clean_risk_map[clean(target_family_key)]
        for v in all_family_variants:
            if v in clean_risk_map: return clean_risk_map[v]

    # 4. IF NO RISK_KEYS (For Populating Order Fields / MT5 Specs)
    # Check what the broker actually has in MarketWatch
    all_symbols = mt5.symbols_get()
    if all_symbols:
        broker_symbols = {clean(s.name): s.name for s in all_symbols}
        
        # Try to find which variant the broker uses
        for v in all_family_variants:
            if v in broker_symbols:
                return broker_symbols[v]
            # Handle suffixes (e.g., USOIL.m)
            for b_clean, b_raw in broker_symbols.items():
                if b_clean.startswith(v):
                    return b_raw

    # Fallback
    return target_family_key if target_family_key else record_symbol.upper()

def deduplicate_orders(inv_id=None):
    """
    Scans all pending_orders/limit_orders.json, pending_orders/limit_orders_backup.json, 
    and pending_orders/signals.json files and removes duplicate orders based on: 
    Symbol, Timeframe, Order Type, and Entry Price.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        This function does NOT require MT5.
    
    Returns:
        bool: True if any duplicates were removed, False otherwise
    """
    print(f"\n{'='*10} 🧹 DEDUPLICATING ORDERS {'='*10}")
    
    total_files_cleaned = 0
    total_duplicates_removed = 0
    total_limit_files_cleaned = 0
    total_signal_files_cleaned = 0
    total_limit_backup_files_cleaned = 0
    total_limit_duplicates = 0
    total_signal_duplicates = 0
    total_limit_backup_duplicates = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    # Determine which investors to process
    if inv_id:
        inv_folder = inv_base_path / inv_id
        investor_folders = [inv_folder] if inv_folder.exists() else []
        if not investor_folders:
            print(f" [!] Error: Investor folder {inv_id} does not exist.")
            return False
    else:
        investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found for deduplication.")
        return False

    any_duplicates_removed = False

    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f"\n [{current_inv_id}] 🔍 Checking for duplicate entries...")

        # 2. Search for pending_orders folders
        pending_orders_folders = list(inv_folder.rglob("*/pending_orders/"))
        
        investor_limit_duplicates = 0
        investor_signal_duplicates = 0
        investor_limit_backup_duplicates = 0
        investor_limit_files_cleaned = 0
        investor_signal_files_cleaned = 0
        investor_limit_backup_files_cleaned = 0

        for pending_folder in pending_orders_folders:
            # Process limit_orders.json
            limit_file = pending_folder / "limit_orders.json"
            if limit_file.exists():
                try:
                    with open(limit_file, 'r', encoding='utf-8') as f:
                        orders = json.load(f)

                    if orders:
                        original_count = len(orders)
                        seen_orders = set()
                        unique_orders = []

                        for order in orders:
                            # Create a unique key based on Symbol, Timeframe, Order Type, and Entry
                            unique_key = (
                                str(order.get("symbol", "")).strip(),
                                str(order.get("timeframe", "")).strip(),
                                str(order.get("order_type", "")).strip(),
                                float(order.get("entry", 0))
                            )

                            if unique_key not in seen_orders:
                                seen_orders.add(unique_key)
                                unique_orders.append(order)
                        
                        # Only write back if duplicates were actually found
                        if len(unique_orders) < original_count:
                            removed = original_count - len(unique_orders)
                            with open(limit_file, 'w', encoding='utf-8') as f:
                                json.dump(unique_orders, f, indent=4)
                            
                            investor_limit_duplicates += removed
                            investor_limit_files_cleaned += 1
                            total_limit_duplicates += removed
                            total_limit_files_cleaned += 1
                            any_duplicates_removed = True
                            
                            folder_name = pending_folder.parent.name
                            print(f"  └─ 📄 {folder_name}/limit_orders.json - Removed {removed} duplicates")

                except Exception as e:
                    print(f"  └─ ❌ Error processing {limit_file}: {e}")

            # Process limit_orders_backup.json
            limit_backup_file = pending_folder / "limit_orders_backup.json"
            if limit_backup_file.exists():
                try:
                    with open(limit_backup_file, 'r', encoding='utf-8') as f:
                        backup_orders = json.load(f)

                    if backup_orders:
                        original_count = len(backup_orders)
                        seen_orders = set()
                        unique_backup_orders = []

                        for order in backup_orders:
                            # Create a unique key based on Symbol, Timeframe, Order Type, and Entry
                            unique_key = (
                                str(order.get("symbol", "")).strip(),
                                str(order.get("timeframe", "")).strip(),
                                str(order.get("order_type", "")).strip(),
                                float(order.get("entry", 0))
                            )

                            if unique_key not in seen_orders:
                                seen_orders.add(unique_key)
                                unique_backup_orders.append(order)
                        
                        # Only write back if duplicates were actually found
                        if len(unique_backup_orders) < original_count:
                            removed = original_count - len(unique_backup_orders)
                            with open(limit_backup_file, 'w', encoding='utf-8') as f:
                                json.dump(unique_backup_orders, f, indent=4)
                            
                            investor_limit_backup_duplicates += removed
                            investor_limit_backup_files_cleaned += 1
                            total_limit_backup_duplicates += removed
                            total_limit_backup_files_cleaned += 1
                            any_duplicates_removed = True
                            
                            folder_name = pending_folder.parent.name
                            print(f"  └─ 📄 {folder_name}/limit_orders_backup.json - Removed {removed} duplicates")

                except Exception as e:
                    print(f"  └─ ❌ Error processing {limit_backup_file}: {e}")

            # Process signals.json
            signals_file = pending_folder / "signals.json"
            if signals_file.exists():
                try:
                    with open(signals_file, 'r', encoding='utf-8') as f:
                        signals = json.load(f)

                    if signals:
                        original_count = len(signals)
                        seen_orders = set()
                        unique_signals = []

                        for signal in signals:
                            # Create a unique key based on Symbol, Timeframe, Order Type, and Entry
                            unique_key = (
                                str(signal.get("symbol", "")).strip(),
                                str(signal.get("timeframe", "")).strip(),
                                str(signal.get("order_type", "")).strip(),
                                float(signal.get("entry", 0))
                            )

                            if unique_key not in seen_orders:
                                seen_orders.add(unique_key)
                                unique_signals.append(signal)
                        
                        # Only write back if duplicates were actually found
                        if len(unique_signals) < original_count:
                            removed = original_count - len(unique_signals)
                            with open(signals_file, 'w', encoding='utf-8') as f:
                                json.dump(unique_signals, f, indent=4)
                            
                            investor_signal_duplicates += removed
                            investor_signal_files_cleaned += 1
                            total_signal_duplicates += removed
                            total_signal_files_cleaned += 1
                            any_duplicates_removed = True
                            
                            folder_name = pending_folder.parent.name
                            print(f"  └─ 📄 {folder_name}/signals.json - Removed {removed} duplicates")

                except Exception as e:
                    print(f"  └─ ❌ Error processing {signals_file}: {e}")

        # Summary for the current investor
        if investor_limit_duplicates > 0 or investor_signal_duplicates > 0 or investor_limit_backup_duplicates > 0:
            print(f"\n  └─ ✨ Investor {current_inv_id} Cleanup Summary:")
            if investor_limit_duplicates > 0:
                print(f"      • limit_orders.json: Cleaned {investor_limit_files_cleaned} files | Removed {investor_limit_duplicates} duplicates")
            if investor_limit_backup_duplicates > 0:
                print(f"      • limit_orders_backup.json: Cleaned {investor_limit_backup_files_cleaned} files | Removed {investor_limit_backup_duplicates} duplicates")
            if investor_signal_duplicates > 0:
                print(f"      • signals.json: Cleaned {investor_signal_files_cleaned} files | Removed {investor_signal_duplicates} duplicates")
        else:
            print(f"  └─ ✅ No duplicates found in any order files")

    # Final Global Summary
    print(f"\n{'='*10} DEDUPLICATION COMPLETE {'='*10}")
    
    total_files_cleaned = total_limit_files_cleaned + total_signal_files_cleaned + total_limit_backup_files_cleaned
    total_duplicates_removed = total_limit_duplicates + total_signal_duplicates + total_limit_backup_duplicates
    
    if total_duplicates_removed > 0:
        print(f" Total Duplicates Purged: {total_duplicates_removed}")
        print(f" Total Files Modified:    {total_files_cleaned}")
        print(f"\n Breakdown by file type:")
        print(f"   • limit_orders.json:        {total_limit_files_cleaned} files | {total_limit_duplicates} duplicates")
        print(f"   • limit_orders_backup.json: {total_limit_backup_files_cleaned} files | {total_limit_backup_duplicates} duplicates")
        print(f"   • signals.json:             {total_signal_files_cleaned} files | {total_signal_duplicates} duplicates")
    else:
        print(" ✅ Everything was already clean - no duplicates found!")
    print(f"{'='*33}\n")
    
    return any_duplicates_removed

def filter_unauthorized_symbols(inv_id=None):
    """
    Verifies and filters pending order files based on allowed symbols defined in accountmanagement.json.
    Now filters both limit_orders.json and signals.json files, removing any entries with unauthorized symbols.
    Matches sanitized versions of symbols to handle broker suffixes (e.g., EURUSDm vs EURUSD).
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        This function does NOT require MT5.
    
    Returns:
        bool: True if any unauthorized symbols were removed, False otherwise
    """
    print(f"\n{'='*10} 🛡️  SYMBOL AUTHORIZATION FILTER {'='*10}")

    def sanitize(sym):
        if not sym: return ""
        # Remove non-alphanumeric, uppercase, and strip trailing M/PRO suffixes
        clean = re.sub(r'[^a-zA-Z0-9]', '', str(sym)).upper()
        return re.sub(r'(PRO|M)$', '', clean)

    if not os.path.exists(INV_PATH):
        print(f" [!] Error: Investor path {INV_PATH} not found.")
        return False

    # Determine which investors to process
    if inv_id:
        investor_ids = [inv_id]
    else:
        investor_ids = [f for f in os.listdir(INV_PATH) if os.path.isdir(os.path.join(INV_PATH, f))]
    
    if not investor_ids:
        print(" └─ 🔘 No investor directories found for filtering.")
        return False

    total_files_cleaned = 0
    total_entries_removed = 0
    total_limit_files_cleaned = 0
    total_signal_files_cleaned = 0
    total_limit_removed = 0
    total_signal_removed = 0
    any_symbols_removed = False

    for current_inv_id in investor_ids:
        print(f"\n [{current_inv_id}] 🔍 Verifying symbol permissions...")
        inv_folder = Path(INV_PATH) / current_inv_id
        acc_mgmt_path = inv_folder / "accountmanagement.json"
        
        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  Account config missing. Skipping.")
            continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract and sanitize the list of allowed symbols
            sym_dict = config.get("symbols_dictionary", {})
            allowed_sanitized = {sanitize(s) for sublist in sym_dict.values() for s in sublist}
            
            if not allowed_sanitized:
                print(f"  └─ 🔘 No symbols defined in dictionary. Skipping filter.")
                continue

            print(f"  └─ ✅ Found {len(allowed_sanitized)} authorized symbols")

            # Search for pending_orders folders
            pending_orders_folders = list(inv_folder.rglob("*/pending_orders/"))
            
            investor_limit_removed = 0
            investor_signal_removed = 0
            investor_limit_files_cleaned = 0
            investor_signal_files_cleaned = 0

            for pending_folder in pending_orders_folders:
                # Process limit_orders.json
                limit_file = pending_folder / "limit_orders.json"
                if limit_file.exists():
                    try:
                        with open(limit_file, 'r', encoding='utf-8') as f:
                            orders = json.load(f)

                        if orders and isinstance(orders, list):
                            original_count = len(orders)
                            
                            # Filter: Keep only if the sanitized symbol exists in our allowed set
                            filtered_orders = [
                                order for order in orders 
                                if sanitize(order.get("symbol", "")) in allowed_sanitized
                            ]
                            
                            # Only write back if entries were actually removed
                            if len(filtered_orders) < original_count:
                                removed = original_count - len(filtered_orders)
                                with open(limit_file, 'w', encoding='utf-8') as f:
                                    json.dump(filtered_orders, f, indent=4)
                                
                                investor_limit_removed += removed
                                investor_limit_files_cleaned += 1
                                total_limit_removed += removed
                                total_limit_files_cleaned += 1
                                any_symbols_removed = True
                                
                                folder_name = pending_folder.parent.name
                                print(f"    └─ 📄 {folder_name}/limit_orders.json - Removed {removed} unauthorized entries")
                            elif original_count > 0:
                                folder_name = pending_folder.parent.name
                                print(f"    └─ ✅ {folder_name}/limit_orders.json - All symbols authorized ({original_count} entries)")

                    except Exception as e:
                        print(f"    └─ ❌ Error processing {limit_file}: {e}")

                # Process signals.json
                signals_file = pending_folder / "signals.json"
                if signals_file.exists():
                    try:
                        with open(signals_file, 'r', encoding='utf-8') as f:
                            signals = json.load(f)

                        if signals and isinstance(signals, list):
                            original_count = len(signals)
                            
                            # Filter: Keep only if the sanitized symbol exists in our allowed set
                            filtered_signals = [
                                signal for signal in signals 
                                if sanitize(signal.get("symbol", "")) in allowed_sanitized
                            ]
                            
                            # Only write back if entries were actually removed
                            if len(filtered_signals) < original_count:
                                removed = original_count - len(filtered_signals)
                                with open(signals_file, 'w', encoding='utf-8') as f:
                                    json.dump(filtered_signals, f, indent=4)
                                
                                investor_signal_removed += removed
                                investor_signal_files_cleaned += 1
                                total_signal_removed += removed
                                total_signal_files_cleaned += 1
                                any_symbols_removed = True
                                
                                folder_name = pending_folder.parent.name
                                print(f"    └─ 📄 {folder_name}/signals.json - Removed {removed} unauthorized entries")
                            elif original_count > 0:
                                folder_name = pending_folder.parent.name
                                print(f"    └─ ✅ {folder_name}/signals.json - All symbols authorized ({original_count} entries)")

                    except Exception as e:
                        print(f"    └─ ❌ Error processing {signals_file}: {e}")

            # Summary for the current investor
            if investor_limit_removed > 0 or investor_signal_removed > 0:
                print(f"\n  └─ ✨ Investor {current_inv_id} Filter Summary:")
                if investor_limit_removed > 0:
                    print(f"      • limit_orders.json: Cleaned {investor_limit_files_cleaned} files | Removed {investor_limit_removed} unauthorized entries")
                if investor_signal_removed > 0:
                    print(f"      • signals.json: Cleaned {investor_signal_files_cleaned} files | Removed {investor_signal_removed} unauthorized entries")
            else:
                # Check if any files were found at all
                if pending_orders_folders:
                    print(f"  └─ ✅ All symbols in order files are authorized")
                else:
                    print(f"  └─ 🔘 No pending_orders folders found")

        except Exception as e:
            print(f"  └─ ❌ Error processing {current_inv_id}: {e}")

    # Final Global Summary
    print(f"\n{'='*10} SYMBOL FILTERING COMPLETE {'='*10}")
    
    total_files_cleaned = total_limit_files_cleaned + total_signal_files_cleaned
    total_entries_removed = total_limit_removed + total_signal_removed
    
    if total_entries_removed > 0:
        print(f" Total Unauthorized Entries Removed: {total_entries_removed}")
        print(f" Total Files Modified:               {total_files_cleaned}")
        print(f"\n Breakdown by file type:")
        print(f"   • limit_orders.json:   {total_limit_files_cleaned} files | {total_limit_removed} entries removed")
        print(f"   • signals.json:        {total_signal_files_cleaned} files | {total_signal_removed} entries removed")
    else:
        if total_files_cleaned == 0:
            print(" ✅ No files needed filtering - all symbols were already authorized!")
        else:
            print(" ✅ All files checked and verified - no unauthorized symbols found!")
    print(f"{'='*39}\n")
    
    return any_symbols_removed

def filter_unauthorized_timeframes(inv_id=None):
    """
    Verifies and filters pending order files based on restricted timeframes defined in accountmanagement.json.
    Now filters both limit_orders.json and signals.json files, removing any entries with restricted timeframes.
    Matches the 'timeframe' key in order files against the 'restrict_order_from_timeframe' setting.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        This function does NOT require MT5.
    
    Returns:
        bool: True if any restricted timeframes were removed, False otherwise
    """
    print(f"\n{'='*10} 🛡️  TIMEFRAME AUTHORIZATION FILTER {'='*10}")

    def sanitize_tf(tf):
        if not tf: return ""
        # Ensure uniform comparison (lowercase, stripped)
        return str(tf).strip().lower()

    if not os.path.exists(INV_PATH):
        print(f" [!] Error: Investor path {INV_PATH} not found.")
        return False

    # Determine which investors to process
    if inv_id:
        investor_ids = [inv_id]
    else:
        investor_ids = [f for f in os.listdir(INV_PATH) if os.path.isdir(os.path.join(INV_PATH, f))]
    
    if not investor_ids:
        print(" └─ 🔘 No investor directories found for filtering.")
        return False

    total_files_cleaned = 0
    total_entries_removed = 0
    total_limit_files_cleaned = 0
    total_signal_files_cleaned = 0
    total_limit_removed = 0
    total_signal_removed = 0
    any_timeframes_removed = False

    for current_inv_id in investor_ids:
        print(f"\n [{current_inv_id}] 🔍 Checking timeframe restrictions...")
        inv_folder = Path(INV_PATH) / current_inv_id
        acc_mgmt_path = inv_folder / "accountmanagement.json"
        
        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  Account config missing. Skipping.")
            continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract restriction setting
            # Supports: "5m" OR ["1m", "5m"]
            raw_restrictions = config.get("settings", {}).get("restrict_order_from_timeframe", [])
            
            if isinstance(raw_restrictions, str):
                # Handle comma separated strings or single strings
                restricted_list = [s.strip() for s in raw_restrictions.split(',')]
            elif isinstance(raw_restrictions, list):
                restricted_list = raw_restrictions
            else:
                restricted_list = []

            restricted_set = {sanitize_tf(t) for t in restricted_list if t}

            if not restricted_set:
                print(f"  └─ ✅ No timeframe restrictions active.")
                continue

            print(f"  └─ 🚫 Restricted timeframes: {', '.join(restricted_set)}")

            # Search for pending_orders folders
            pending_orders_folders = list(inv_folder.rglob("*/pending_orders/"))
            
            investor_limit_removed = 0
            investor_signal_removed = 0
            investor_limit_files_cleaned = 0
            investor_signal_files_cleaned = 0

            for pending_folder in pending_orders_folders:
                # Process limit_orders.json
                limit_file = pending_folder / "limit_orders.json"
                if limit_file.exists():
                    try:
                        with open(limit_file, 'r', encoding='utf-8') as f:
                            orders = json.load(f)

                        if orders and isinstance(orders, list):
                            original_count = len(orders)
                            
                            # Filter: Keep only if the entry's timeframe is NOT in the restricted set
                            filtered_orders = [
                                order for order in orders 
                                if sanitize_tf(order.get("timeframe")) not in restricted_set
                            ]
                            
                            # Only write back if entries were actually removed
                            if len(filtered_orders) < original_count:
                                removed = original_count - len(filtered_orders)
                                with open(limit_file, 'w', encoding='utf-8') as f:
                                    json.dump(filtered_orders, f, indent=4)
                                
                                investor_limit_removed += removed
                                investor_limit_files_cleaned += 1
                                total_limit_removed += removed
                                total_limit_files_cleaned += 1
                                any_timeframes_removed = True
                                
                                folder_name = pending_folder.parent.name
                                print(f"    └─ 📄 {folder_name}/limit_orders.json - Removed {removed} restricted timeframe entries")
                            elif original_count > 0:
                                folder_name = pending_folder.parent.name
                                print(f"    └─ ✅ {folder_name}/limit_orders.json - All timeframes authorized ({original_count} entries)")

                    except Exception as e:
                        print(f"    └─ ❌ Error processing {limit_file}: {e}")

                # Process signals.json
                signals_file = pending_folder / "signals.json"
                if signals_file.exists():
                    try:
                        with open(signals_file, 'r', encoding='utf-8') as f:
                            signals = json.load(f)

                        if signals and isinstance(signals, list):
                            original_count = len(signals)
                            
                            # Filter: Keep only if the entry's timeframe is NOT in the restricted set
                            filtered_signals = [
                                signal for signal in signals 
                                if sanitize_tf(signal.get("timeframe")) not in restricted_set
                            ]
                            
                            # Only write back if entries were actually removed
                            if len(filtered_signals) < original_count:
                                removed = original_count - len(filtered_signals)
                                with open(signals_file, 'w', encoding='utf-8') as f:
                                    json.dump(filtered_signals, f, indent=4)
                                
                                investor_signal_removed += removed
                                investor_signal_files_cleaned += 1
                                total_signal_removed += removed
                                total_signal_files_cleaned += 1
                                any_timeframes_removed = True
                                
                                folder_name = pending_folder.parent.name
                                print(f"    └─ 📄 {folder_name}/signals.json - Removed {removed} restricted timeframe entries")
                            elif original_count > 0:
                                folder_name = pending_folder.parent.name
                                print(f"    └─ ✅ {folder_name}/signals.json - All timeframes authorized ({original_count} entries)")

                    except Exception as e:
                        print(f"    └─ ❌ Error processing {signals_file}: {e}")

            # Summary for the current investor
            if investor_limit_removed > 0 or investor_signal_removed > 0:
                print(f"\n  └─ ✨ Investor {current_inv_id} Filter Summary:")
                if investor_limit_removed > 0:
                    print(f"      • limit_orders.json: Cleaned {investor_limit_files_cleaned} files | Removed {investor_limit_removed} restricted entries")
                if investor_signal_removed > 0:
                    print(f"      • signals.json: Cleaned {investor_signal_files_cleaned} files | Removed {investor_signal_removed} restricted entries")
                print(f"     (Blocked timeframes: {', '.join(restricted_set)})")
            else:
                # Check if any files were found at all
                if pending_orders_folders:
                    print(f"  └─ ✅ All timeframes in order files are authorized")
                else:
                    print(f"  └─ 🔘 No pending_orders folders found")

        except Exception as e:
            print(f"  └─ ❌ Error processing {current_inv_id}: {e}")

    # Final Global Summary
    print(f"\n{'='*10} TIMEFRAME FILTERING COMPLETE {'='*10}")
    
    total_files_cleaned = total_limit_files_cleaned + total_signal_files_cleaned
    total_entries_removed = total_limit_removed + total_signal_removed
    
    if total_entries_removed > 0:
        print(f" Total Restricted Entries Removed: {total_entries_removed}")
        print(f" Total Files Modified:              {total_files_cleaned}")
        print(f"\n Breakdown by file type:")
        print(f"   • limit_orders.json:   {total_limit_files_cleaned} files | {total_limit_removed} entries removed")
        print(f"   • signals.json:        {total_signal_files_cleaned} files | {total_signal_removed} entries removed")
    else:
        if total_files_cleaned == 0:
            print(" ✅ No files needed filtering - no restricted timeframes found!")
        else:
            print(" ✅ All files checked and verified - no restricted timeframes found!")
    print(f"{'='*41}\n")
    
    return any_timeframes_removed

def backup_limit_orders(inv_id=None):
    """
    Finds all limit_orders.json files and creates a copy named 
    limit_orders_backup.json in the same directory.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. 
                               If None, processes all investors.
    """
    print(f"\n{'='*10} 📂 CREATING LIMIT ORDERS BACKUP {'='*10}")
    
    inv_base_path = Path(INV_PATH)
    total_backups_created = 0

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    # 1. Determine which investors to process
    if inv_id:
        investor_folders = [inv_base_path / inv_id]
    else:
        investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]

    # 2. Loop through each investor folder
    for inv_folder in investor_folders:
        if not inv_folder.exists():
            continue
            
        print(f" [{inv_folder.name}] Scanning for limit_orders.json...")

        # 3. Find all limit_orders.json files (using rglob for subfolders)
        # Specifically targeting the 'pending_orders' subfolder pattern
        target_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))

        for source_path in target_files:
            # Define the backup path in the same directory
            backup_path = source_path.parent / "limit_orders_backup.json"
            
            try:
                # 4. Create the copy (overwrites existing backup)
                shutil.copy2(source_path, backup_path)
                
                print(f"  └─ ✅ Backed up: {source_path.parent.parent.name} -> limit_orders_backup.json")
                total_backups_created += 1
                
            except Exception as e:
                print(f"  └─ ❌ Error backing up {source_path}: {e}")

    print(f"\n{'='*10} BACKUP PROCESS COMPLETE {'='*10}")
    print(f" Total backups created: {total_backups_created}")
    return total_backups_created > 0

def populate_orders_missing_fields(inv_id=None, callback_function=None):
    print(f"\n{'='*10} 📊 POPULATING ORDER FIELDS {'='*10}")
    
    total_files_updated = 0
    total_orders_updated = 0
    total_symbols_normalized = 0
    inv_base_path = Path(INV_PATH)

    investor_folders = [inv_base_path / inv_id] if inv_id else [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f" [{current_inv_id}] 🔍 Processing orders...")

        # Local Cache for this investor to prevent redundant lookups
        # Format: { "raw_symbol": {"broker_sym": "normalized", "info": mt5_obj} }
        resolution_cache = {}

        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        if not order_files: continue
            
        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg: continue
            
        server = broker_cfg.get('SERVER', '')
        broker_prefix = server.split('-')[0].split('.')[0].lower() if server else 'broker'
        v_field, ts_field, tv_field = f"{broker_prefix}_volume", f"{broker_prefix}_tick_size", f"{broker_prefix}_tick_value"

        for file_path in order_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    orders = json.load(f)
                if not orders: continue
                
                modified = False
                for order in orders:
                    raw_symbol = order.get("symbol")
                    if not raw_symbol: continue

                    # Check Cache First
                    if raw_symbol in resolution_cache:
                        res = resolution_cache[raw_symbol]
                        broker_symbol = res['broker_sym']
                        symbol_info = res['info']
                    else:
                        # Perform mapping only once
                        broker_symbol = get_normalized_symbol(raw_symbol)
                        symbol_info = mt5.symbol_info(broker_symbol)
                        
                        resolution_cache[raw_symbol] = {'broker_sym': broker_symbol, 'info': symbol_info}
                        
                        # Detailed Log only on first discovery
                        if symbol_info:
                            if broker_symbol != raw_symbol:
                                print(f"    └─ ✅ {raw_symbol} -> {broker_symbol} (Mapped & Cached)")
                                total_symbols_normalized += 1
                        else:
                            print(f"    └─ ❌ MT5: '{broker_symbol}' (from '{raw_symbol}') not found in MarketWatch")

                    if symbol_info:
                        order['symbol'] = broker_symbol
                        
                        # Cleanup and Update
                        for key in list(order.keys()):
                            if any(x in key.lower() for x in ['volume', 'tick_size', 'tick_value']) and key not in [v_field, ts_field, tv_field]:
                                del order[key]

                        order[v_field] = symbol_info.volume_min
                        order[ts_field] = symbol_info.trade_tick_size
                        order[tv_field] = symbol_info.trade_tick_value
                        total_orders_updated += 1
                        modified = True

                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(orders, f, indent=4)
                    total_files_updated += 1

            except Exception as e:
                print(f"  └─ ❌ Error: {e}")

    print(f"\n{'='*10} POPULATION COMPLETE {'='*10}")
    print(f" Total Orders Updated:      {total_orders_updated}")
    print(f" Total Symbols Normalized:  {total_symbols_normalized}")
    return True

def activate_usd_based_risk_on_empty_pricelevels(inv_id=None):
    print(f"\n{'='*10} 📊 INVESTOR EMPTY TARGET CHECK - USD RISK ENFORCEMENT {'='*10}")
    
    total_orders_processed = 0
    total_orders_enforced = 0
    total_files_updated = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    investor_folders = [inv_base_path / inv_id] if inv_id else [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f"\n [{current_inv_id}] 🔍 Processing empty target check...")

        # Cache for risk mappings to avoid re-calculating family logic 1000s of times
        risk_map_cache = {}

        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found")
            continue
        
        broker_name = broker_cfg.get('BROKER_NAME', '').lower() or \
                      broker_cfg.get('SERVER', 'default').split('-')[0].split('.')[0].lower()

        default_config_path = Path(DEFAULT_PATH) / f"{broker_name}_default_allowedsymbolsandvolumes.json"
        
        risk_lookup = {}
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    default_config = json.load(f)
                    for category, items in default_config.items():
                        if not isinstance(items, list): continue
                        for item in items:
                            sym = str(item.get("symbol", "")).upper()
                            if sym:
                                risk_lookup[sym] = {
                                    k.replace("_specs", "").upper(): v.get("usd_risk", 0)
                                    for k, v in item.items() if k.endswith("_specs")
                                }
                print(f"  └─ ✅ Loaded risk config for {len(risk_lookup)} symbols")
            except Exception as e:
                print(f"  └─ ❌ Risk config error: {e}")
                continue

        known_risk_symbols = list(risk_lookup.keys())
        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        signals_files = list(inv_folder.rglob("*/signals/signals.json"))
        
        for file_list, label in [(order_files, "LIMITS"), (signals_files, "SIGNALS")]:
            for file_path in file_list:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not data: continue
                    modified = False
                    
                    for item in data:
                        if item.get('exit') in [0, "0", None, 0.0] and \
                           item.get('target') in [0, "0", None, 0.0]:
                            
                            total_orders_processed += 1
                            raw_sym = str(item.get('symbol', '')).upper()
                            raw_tf = str(item.get('timeframe', '')).upper()
                            
                            # Cache Logic for Risk Mapping
                            if raw_sym not in risk_map_cache:
                                matched_sym = get_normalized_symbol(raw_sym, risk_keys=known_risk_symbols)
                                risk_map_cache[raw_sym] = matched_sym
                                
                                # Print mapping only once per symbol type
                                if matched_sym not in risk_lookup:
                                    print(f"      ❌ [{label}] {raw_sym}: Not in risk config (Mapped as: {matched_sym})")
                            else:
                                matched_sym = risk_map_cache[raw_sym]
                            
                            if matched_sym in risk_lookup:
                                tf_risks = risk_lookup[matched_sym]
                                risk_value = tf_risks.get(raw_tf, 0)
                                
                                if risk_value > 0:
                                    item.update({
                                        'exit': 0, 'target': 0, 'usd_risk': risk_value,
                                        'usd_based_risk_only': True, 'symbol': matched_sym
                                    })
                                    modified = True
                                    total_orders_enforced += 1
                                    
                                    # Logic to avoid spamming the same enforcement 1000 times in logs
                                    # Only log the first time we enforce this symbol/tf pair for this file
                                    if f"{raw_sym}_{raw_tf}" not in risk_map_cache:
                                        print(f"      ✅ [{label}] {matched_sym} ({raw_sym}) {raw_tf}: Enforced ${risk_value} risk")
                                        risk_map_cache[f"{raw_sym}_{raw_tf}"] = True
                                
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4)
                        total_files_updated += 1

                except Exception as e:
                    print(f"    └─ ❌ Error processing {file_path.name}: {e}")

    print(f"\n{'='*10} ENFORCEMENT COMPLETE {'='*10}")
    print(f" Total Targetless Found:  {total_orders_processed}")
    print(f" Total Risk Enforced:    {total_orders_enforced}")
    print(f" Files Updated:          {total_files_updated}")
    
    return total_orders_enforced > 0

def enforce_investors_risk(inv_id=None):
    """
    Enforces risk rules for investors based on accountmanagement.json settings.
    Enhanced with Smart Normalization Caching and optimized lookup logic.
    """
    print(f"\n{'='*10} 📊 SMART INVESTOR RISK ENFORCEMENT {'='*10}")
    
    total_orders_processed = 0
    total_orders_enforced = 0
    total_files_updated = 0
    total_symbols_normalized = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    investor_folders = [inv_base_path / inv_id] if inv_id else [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    any_orders_enforced = False
    
    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        
        # --- INVESTOR LOCAL CACHE ---
        # Stores: { "RAW_SYM": {"matched": "NORM_SYM", "is_norm": True/False, "risk": {TF_DATA}} }
        resolution_cache = {}
        
        print(f"\n [{current_inv_id}] 🔍 Initializing smart enforcement...")

        # 1. Load accountmanagement.json
        acc_mgmt_path = inv_folder / "accountmanagement.json"
        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  accountmanagement.json not found, skipping")
            continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                acc_mgmt_data = json.load(f)
            
            enforce_default = acc_mgmt_data.get("settings", {}).get("enforce_default_usd_risk", False)
            print(f"  └─ 🎯 Master Switch: {enforce_default}")
            
            if not enforce_default:
                print(f"  └─ ⏭️  Master switch is OFF - skipping")
                continue
        except Exception as e:
            print(f"  └─ ❌ Failed to load accountmanagement.json: {e}")
            continue

        # 2. Get Broker and Config Path
        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg:
            continue
        
        broker_name = broker_cfg.get('BROKER_NAME', '').lower() or \
                      broker_cfg.get('SERVER', 'default').split('-')[0].split('.')[0].lower()

        default_config_path = Path(DEFAULT_PATH) / f"{broker_name}_default_allowedsymbolsandvolumes.json"
        if not default_config_path.exists():
            print(f"  └─ ❌ Default config not found: {default_config_path.name}")
            continue
        
        # 3. Build Risk Lookup Table
        risk_lookup = {}
        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                default_config = json.load(f)
                for category, items in default_config.items():
                    if not isinstance(items, list): continue
                    for item in items:
                        sym = str(item.get("symbol", "")).upper()
                        if sym:
                            risk_lookup[sym] = {
                                k.replace("_specs", "").upper(): {
                                    "volume": v.get("volume", 0.01),
                                    "usd_risk": v.get("usd_risk", 0)
                                } for k, v in item.items() if k.endswith("_specs")
                            }
            known_risk_symbols = list(risk_lookup.keys())
            print(f"  └─ ✅ Loaded risk config for {len(risk_lookup)} symbols")
        except Exception as e:
            print(f"  └─ ❌ Failed to parse default config: {e}")
            continue

        # 4. Gather Files
        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        signals_files = list(inv_folder.rglob("*/signals/signals.json"))
        
        investor_orders_enforced = 0
        investor_files_updated = 0
        
        # 5. Process Unified Pipeline
        for file_list, label in [(order_files, "LIMITS"), (signals_files, "SIGNALS")]:
            for file_path in file_list:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if not data: continue
                    
                    modified = False
                    for item in data:
                        total_orders_processed += 1
                        raw_sym = str(item.get('symbol', '')).upper()
                        raw_tf = str(item.get('timeframe', '')).upper()
                        
                        # --- SMART RESOLUTION LOGIC ---
                        if raw_sym not in resolution_cache:
                            # Helper does the heavy lifting: USOUSD -> USOIL
                            matched_sym = get_normalized_symbol(raw_sym, risk_keys=known_risk_symbols)
                            was_normalized = (matched_sym != raw_sym)
                            
                            # Cache the result
                            resolution_cache[raw_sym] = {
                                "matched": matched_sym,
                                "is_norm": was_normalized,
                                "risk_data": risk_lookup.get(matched_sym, {})
                            }
                            
                            # Log first-time discovery
                            if was_normalized and matched_sym in risk_lookup:
                                print(f"    └─ ✅ Normalized: {raw_sym} -> {matched_sym}")
                                total_symbols_normalized += 1
                        
                        res = resolution_cache[raw_sym]
                        matched_sym = res["matched"]
                        tf_data = res["risk_data"].get(raw_tf)

                        if tf_data and tf_data["usd_risk"] > 0:
                            # Apply Enforcement
                            item.update({
                                'exit': 0,
                                'target': 0,
                                'usd_risk': tf_data["usd_risk"],
                                'usd_based_risk_only': True,
                                'symbol': matched_sym
                            })
                            
                            # Update volume if specified
                            if tf_data["volume"] > 0:
                                for key in list(item.keys()):
                                    if 'volume' in key.lower():
                                        item[key] = tf_data["volume"]
                                        break
                                        
                            modified = True
                            investor_orders_enforced += 1
                            total_orders_enforced += 1
                        
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4)
                        investor_files_updated += 1
                        total_files_updated += 1
                        
                except Exception as e:
                    print(f"    └─ ❌ Error in {file_path.name}: {e}")

        # Summary for this investor
        if investor_orders_enforced > 0:
            any_orders_enforced = True
            print(f"  └─ 📊 {current_inv_id} Complete: Enforced {investor_orders_enforced} orders across {investor_files_updated} files.")

    # Final Global Summary
    print(f"\n{'='*10} RISK ENFORCEMENT COMPLETE {'='*10}")
    print(f" Total Files Updated:   {total_files_updated}")
    print(f" Total Enforced:        {total_orders_enforced} / {total_orders_processed}")
    print(f" Symbols Normalized:    {total_symbols_normalized}")
    print(f"{'='*50}\n")
    
    return any_orders_enforced
    
def calculate_investor_symbols_orders(inv_id=None, callback_function=None):
    """
    Calculates Exit/Target prices for ALL orders in limit_orders.json files for investors.
    Uses the selected_risk_reward value from accountmanagement.json for each investor.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        callback_function (callable, optional): A function to call with the opened file data.
            The callback will receive (inv_id, file_path, orders_list) parameters.
    
    Returns:
        bool: True if any orders were calculated, False otherwise
    """
    print(f"\n{'='*10} 📊 CALCULATING INVESTOR ORDER PRICES {'='*10}")
    
    total_files_updated = 0
    total_orders_processed = 0
    total_orders_calculated = 0
    total_orders_skipped = 0
    total_symbols_normalized = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    # Determine which investors to process
    if inv_id:
        inv_folder = inv_base_path / inv_id
        investor_folders = [inv_folder] if inv_folder.exists() else []
        if not investor_folders:
            print(f" [!] Error: Investor folder {inv_id} does not exist.")
            return False
    else:
        investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    any_orders_calculated = False

    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f" [{current_inv_id}] 🔍 Processing orders...")

        # --- INVESTOR LOCAL CACHE for symbol normalization ---
        resolution_cache = {}

        # 1. Load accountmanagement.json to get selected_risk_reward
        acc_mgmt_path = inv_folder / "accountmanagement.json"
        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  accountmanagement.json not found for {current_inv_id}, skipping")
            continue

        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                acc_mgmt_data = json.load(f)
            
            # Get selected_risk_reward value (default to 1.0 if not found)
            selected_rr = acc_mgmt_data.get("selected_risk_reward", [1.0])
            if isinstance(selected_rr, list) and len(selected_rr) > 0:
                rr_ratio = float(selected_rr[0])
            else:
                rr_ratio = float(selected_rr) if selected_rr else 1.0
            
            print(f"  └─ 📊 Using selected R:R ratio: {rr_ratio}")
            
        except Exception as e:
            print(f"  └─ ❌ Failed to load accountmanagement.json: {e}")
            continue

        # 2. Get broker config for potential symbol mapping context
        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg:
            print(f"  └─ ⚠️  No broker config found for {current_inv_id}")
            # Continue anyway as normalization might still work

        # 3. Find all limit_orders.json files
        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        
        if not order_files:
            print(f"  └─ 🔘 No limit order files found")
            continue
            
        investor_files_updated = 0
        investor_orders_processed = 0
        investor_orders_calculated = 0
        investor_orders_skipped = 0
        
        # Process each file individually
        for file_path in order_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    orders = json.load(f)
                
                if not orders:
                    continue
                
                # Call callback function if provided with the original data
                if callback_function:
                    try:
                        callback_function(current_inv_id, file_path, orders)
                    except Exception as e:
                        print(f"    └─ ⚠️  Callback error for {file_path.name}: {e}")
                
                # Track original orders for this file
                original_count = len(orders)
                investor_orders_processed += original_count
                
                # Process each order in this file
                orders_updated = False
                file_orders_calculated = 0
                file_orders_skipped = 0
                
                for order in orders:
                    try:
                        # --- SYMBOL NORMALIZATION with Caching ---
                        raw_symbol = order.get("symbol", "")
                        if not raw_symbol:
                            file_orders_skipped += 1
                            continue
                        
                        # Check Cache First
                        if raw_symbol in resolution_cache:
                            normalized_symbol = resolution_cache[raw_symbol]
                        else:
                            # Perform mapping only once
                            normalized_symbol = get_normalized_symbol(raw_symbol)
                            resolution_cache[raw_symbol] = normalized_symbol
                            
                            # Log normalization on first discovery
                            if normalized_symbol != raw_symbol:
                                print(f"    └─ ✅ {raw_symbol} -> {normalized_symbol} (Mapped & Cached)")
                                total_symbols_normalized += 1
                        
                        # Update the symbol in the order
                        if normalized_symbol:
                            order['symbol'] = normalized_symbol
                        
                        # --- CHECK FOR USD-BASED RISK FIRST (doesn't require volume) ---
                        if order.get("usd_based_risk_only") is True:
                            risk_val = float(order.get("usd_risk", 0))
                            
                            if risk_val > 0:
                                # For USD-based, we need volume but it might be named differently
                                # Try to find volume field
                                volume_value = None
                                for key, value in order.items():
                                    if 'volume' in key.lower() and isinstance(value, (int, float)):
                                        volume_value = float(value)
                                        break
                                
                                if volume_value is None or volume_value <= 0:
                                    print(f"      ⚠️  USD-based order missing volume for {order.get('symbol', 'Unknown')}, skipping")
                                    file_orders_skipped += 1
                                    continue
                                
                                # Find tick_size field
                                tick_size_value = None
                                for key, value in order.items():
                                    if 'tick_size' in key.lower() and isinstance(value, (int, float)):
                                        tick_size_value = float(value)
                                        break
                                
                                if tick_size_value is None or tick_size_value <= 0:
                                    tick_size_value = 0.00001
                                    print(f"      ⚠️  No tick_size found for {order.get('symbol', 'Unknown')}, using default")
                                
                                # Find tick_value field
                                tick_value_value = None
                                for key, value in order.items():
                                    if 'tick_value' in key.lower() and isinstance(value, (int, float)):
                                        tick_value_value = float(value)
                                        break
                                
                                if tick_value_value is None or tick_value_value <= 0:
                                    tick_value_value = 1.0
                                    print(f"      ⚠️  No tick_value found for {order.get('symbol', 'Unknown')}, using default")
                                
                                # Extract required order data
                                entry = float(order.get('entry', 0))
                                if entry == 0:
                                    file_orders_skipped += 1
                                    continue
                                    
                                order_type = str(order.get('order_type', '')).upper()
                                
                                # Calculate digits for rounding based on tick_size
                                if tick_size_value < 1:
                                    digits = len(str(tick_size_value).split('.')[-1])
                                else:
                                    digits = 0
                                
                                # Calculate using USD risk
                                sl_dist = (risk_val * tick_size_value) / (tick_value_value * volume_value)
                                tp_dist = sl_dist * rr_ratio
                                
                                if "BUY" in order_type:
                                    order["exit"] = round(entry - sl_dist, digits)
                                    order["target"] = round(entry + tp_dist, digits)
                                elif "SELL" in order_type:
                                    order["exit"] = round(entry + sl_dist, digits)
                                    order["target"] = round(entry - tp_dist, digits)
                                else:
                                    file_orders_skipped += 1
                                    continue
                                
                                file_orders_calculated += 1
                                any_orders_calculated = True
                                
                                # Update metadata
                                order['risk_reward'] = rr_ratio
                                order['status'] = "Calculated"
                                order['calculated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                orders_updated = True
                                continue  # Skip the rest of the processing for this order
                            else:
                                file_orders_skipped += 1
                                continue
                        
                        # --- NON-USD BASED ORDERS (require volume) ---
                        # Check for required volume field
                        volume_field = None
                        volume_value = None
                        
                        for key, value in order.items():
                            if 'volume' in key.lower() and isinstance(value, (int, float)):
                                volume_field = key
                                volume_value = float(value)
                                break
                        
                        if volume_value is None or volume_value <= 0:
                            file_orders_skipped += 1
                            continue
                        
                        # Find tick_size field
                        tick_size_field = None
                        tick_size_value = None
                        
                        for key, value in order.items():
                            if 'tick_size' in key.lower() and isinstance(value, (int, float)):
                                tick_size_field = key
                                tick_size_value = float(value)
                                break
                        
                        if tick_size_value is None or tick_size_value <= 0:
                            tick_size_value = 0.00001
                            print(f"      ⚠️  No tick_size found for {order.get('symbol', 'Unknown')}, using default")
                        
                        # Find tick_value field
                        tick_value_field = None
                        tick_value_value = None
                        
                        for key, value in order.items():
                            if 'tick_value' in key.lower() and isinstance(value, (int, float)):
                                tick_value_field = key
                                tick_value_value = float(value)
                                break
                        
                        if tick_value_value is None or tick_value_value <= 0:
                            tick_value_value = 1.0
                            print(f"      ⚠️  No tick_value found for {order.get('symbol', 'Unknown')}, using default")
                        
                        # Extract required order data
                        entry = float(order.get('entry', 0))
                        if entry == 0:
                            file_orders_skipped += 1
                            continue
                            
                        order_type = str(order.get('order_type', '')).upper()
                        
                        # Calculate digits for rounding based on tick_size
                        if tick_size_value < 1:
                            digits = len(str(tick_size_value).split('.')[-1])
                        else:
                            digits = 0
                        
                        # Standard calculation based on exit or target
                        sl_price = float(order.get('exit', 0))
                        tp_price = float(order.get('target', 0))
                        
                        # Case 1: Target provided, need to calculate exit
                        if sl_price == 0 and tp_price > 0:
                            risk_dist = abs(tp_price - entry) / rr_ratio
                            if "BUY" in order_type:
                                order['exit'] = round(entry - risk_dist, digits)
                            elif "SELL" in order_type:
                                order['exit'] = round(entry + risk_dist, digits)
                            else:
                                file_orders_skipped += 1
                                continue
                            
                            file_orders_calculated += 1
                            any_orders_calculated = True
                        
                        # Case 2: Exit provided, need to calculate target
                        elif sl_price > 0:
                            risk_dist = abs(entry - sl_price)
                            if "BUY" in order_type:
                                order['target'] = round(entry + (risk_dist * rr_ratio), digits)
                            elif "SELL" in order_type:
                                order['target'] = round(entry - (risk_dist * rr_ratio), digits)
                            else:
                                file_orders_skipped += 1
                                continue
                            
                            file_orders_calculated += 1
                            any_orders_calculated = True
                            print(f"      ✅ Exit-based: {order.get('symbol')} - Target calculated: {order['target']}")
                        
                        # Case 3: Neither exit nor target provided, skip
                        else:
                            file_orders_skipped += 1
                            continue
                        
                        # --- METADATA UPDATES ---
                        order['risk_reward'] = rr_ratio
                        order['status'] = "Calculated"
                        order['calculated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        orders_updated = True
                        
                    except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                        file_orders_skipped += 1
                        print(f"      ⚠️  Error processing order {order.get('symbol', 'Unknown')}: {e}")
                        continue
                
                # Save the updated orders back to the same file
                if orders_updated:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(orders, f, indent=4)
                        
                        investor_files_updated += 1
                        total_files_updated += 1
                        
                        # Update counters
                        investor_orders_calculated += file_orders_calculated
                        investor_orders_skipped += file_orders_skipped
                        
                        print(f"    └─ 📁 {file_path.parent.name}/limit_orders.json: "
                              f"Processed: {original_count}, Calculated: {file_orders_calculated}, "
                              f"Skipped: {file_orders_skipped}")
                        
                    except Exception as e:
                        print(f"    └─ ❌ Failed to save {file_path}: {e}")
                
            except Exception as e:
                print(f"    └─ ❌ Error reading {file_path}: {e}")
                continue
        
        # Summary for current investor
        if investor_orders_processed > 0:
            total_orders_processed += investor_orders_processed
            total_orders_calculated += investor_orders_calculated
            total_orders_skipped += investor_orders_skipped
            
            print(f"  └─ ✨ Investor {current_inv_id} Summary:")
            print(f"      Files updated: {investor_files_updated}")
            print(f"      Orders processed: {investor_orders_processed}")
            print(f"      Orders calculated: {investor_orders_calculated}")
            print(f"      Orders skipped: {investor_orders_skipped}")
            
            if investor_orders_processed > 0:
                calc_rate = (investor_orders_calculated / investor_orders_processed) * 100
                print(f"      Calculation rate: {calc_rate:.1f}%")
        else:
            print(f"  └─ ⚠️  No orders processed for {current_inv_id}")

    # Final Global Summary
    print(f"\n{'='*10} INVESTOR CALCULATION COMPLETE {'='*10}")
    if total_orders_processed > 0:
        print(f" Total Files Modified:    {total_files_updated}")
        print(f" Total Orders Processed:  {total_orders_processed}")
        print(f" Total Orders Calculated: {total_orders_calculated}")
        print(f" Total Orders Skipped:    {total_orders_skipped}")
        print(f" Symbols Normalized:      {total_symbols_normalized}")
        
        if total_orders_processed > 0:
            overall_rate = (total_orders_calculated / total_orders_processed) * 100
            print(f" Overall Calculation Rate: {overall_rate:.1f}%")
    else:
        print(" No orders were processed.")
    
    return any_orders_calculated

def padding_tight_usd_risk(inv_id=None):
    """
    Ranks orders, adjusts 'too tight' risk to 50% of the next order in line,
    and saves results back to the original limit_orders.json.
    """
    print(f"\n{'='*10} ⚖️ DYNAMIC RISK RANKING & SPACING {'='*10}")
    
    inv_base_path = Path(INV_PATH) 
    if not inv_base_path.exists(): 
        return False

    investor_folders = [inv_base_path / inv_id] if inv_id and (inv_base_path / inv_id).exists() else [f for f in inv_base_path.iterdir() if f.is_dir()]

    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        # Target the standard limit_orders.json files
        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        
        for file_path in order_files:
            try:
                # 1. Load the original file
                with open(file_path, 'r', encoding='utf-8') as f:
                    orders = json.load(f)
                if not orders: 
                    continue

                # 2. Group by timeframe and extract agnostic broker data
                timeframe_groups = {}
                for order in orders:
                    tf = order.get("timeframe", "Unknown")
                    if tf not in timeframe_groups: 
                        timeframe_groups[tf] = []
                    
                    # Agnostic extraction (suffix-based)
                    vol = next((v for k, v in order.items() if k.endswith("_volume")), 0)
                    t_size = next((v for k, v in order.items() if k.endswith("_tick_size")), 0)
                    t_val = next((v for k, v in order.items() if k.endswith("_tick_value")), 0)
                    entry = order.get("entry", 0)
                    exit_p = order.get("exit", 0)

                    # Initial risk calc
                    if vol > 0 and t_size > 0 and entry > 0 and exit_p > 0:
                        ticks = abs(entry - exit_p) / t_size
                        order["live_risk_usd"] = round(vol * ticks * t_val, 2)
                    else:
                        order["live_risk_usd"] = 0.0
                    
                    timeframe_groups[tf].append(order)

                rank_words = ["first", "second", "third", "fourth", "fifth"]
                updated_orders_list = []

                # 3. Process each group for the 50% spacing rule
                for tf, group in timeframe_groups.items():
                    # Sort by risk lowest -> highest
                    group.sort(key=lambda x: x.get("live_risk_usd", 0))
                    
                    for i in range(len(group)):
                        current = group[i]
                        
                        # Apply spacing if there's a higher risk order next in line
                        if i + 1 < len(group):
                            next_order = group[i+1]
                            curr_risk = current.get("live_risk_usd", 0)
                            next_risk = next_order.get("live_risk_usd", 0)
                            
                            threshold = round(next_risk / 2, 2)
                            
                            if curr_risk < threshold and threshold > 0:
                                vol = next((v for k, v in current.items() if k.endswith("_volume")), 0)
                                t_size = next((v for k, v in current.items() if k.endswith("_tick_size")), 0)
                                t_val = next((v for k, v in current.items() if k.endswith("_tick_value")), 0)
                                entry = current.get("entry")
                                
                                if vol > 0 and t_val > 0:
                                    new_dist = (threshold * t_size) / (vol * t_val)
                                    if "buy" in current.get("order_type", "").lower():
                                        current["exit"] = round(entry - new_dist, 5)
                                    else:
                                        current["exit"] = round(entry + new_dist, 5)
                                    
                                    current["live_risk_usd"] = threshold
                                    current["adjustment_note"] = f"Spaced to 50% of rank {i+2} (${threshold})"

                        # Remove old rank flags
                        for k in list(current.keys()):
                            if "_usd_risk" in k and any(w in k for w in rank_words):
                                del current[k]

                        # Apply current ranking flag (only for True values as requested)
                        if i < len(rank_words):
                            flag_name = f"{rank_words[i]}_lowest_{tf}_usd_risk"
                            current[flag_name] = True
                        
                        current["risk_calculated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        updated_orders_list.append(current)

                # 4. Save back to original limit_orders.json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_orders_list, f, indent=4)
                
                # 5. Cleanup: Remove the order_risks.json file if it exists
                legacy_risks_file = file_path.parent / "order_risks.json"
                if legacy_risks_file.exists():
                    os.remove(legacy_risks_file)
                
                print(f"  └─ [{current_inv_id}] Updated {file_path.name} and cleaned legacy files.")

            except Exception as e:
                print(f"  └─ ❌ Error processing {file_path}: {e}")

    return True

def live_usd_risk_and_scaling(inv_id=None, callback_function=None):
    """
    Calculates and populates the live USD risk for all orders in pending_orders/limit_orders.json files.
    Deduplicates orders first, scales volume, and SPLITS orders if they exceed max volume.
    """
    print(f"\n{'='*10} 💰 CALCULATING LIVE USD RISK WITH DEDUPLICATION & SPLITTING {'='*10}")
    
    total_files_updated = 0
    total_orders_updated = 0
    total_risk_usd = 0.0
    total_signals_created = 0
    total_symbols_normalized = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    if inv_id:
        inv_folder = inv_base_path / inv_id
        investor_folders = [inv_folder] if inv_folder.exists() else []
        if not investor_folders:
            print(f" [!] Error: Investor folder {inv_id} does not exist.")
            return False
    else:
        investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    any_orders_processed = False

    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f"\n [{current_inv_id}] 🔍 Initializing pre-process cleanup and risk scaling...")

        resolution_cache = {}

        # 1. Load account management data
        account_mgmt_path = inv_folder / "accountmanagement.json"
        risk_ranges = {}
        account_balance = None
        
        if account_mgmt_path.exists():
            try:
                with open(account_mgmt_path, 'r', encoding='utf-8') as f:
                    account_data = json.load(f)
                risk_ranges = account_data.get('account_balance_default_risk_management', {})
                print(f"  └─ 📊 Loaded risk management ranges: {len(risk_ranges)} ranges")
            except Exception as e:
                print(f"  └─ ⚠️  Could not load accountmanagement.json: {e}")
        else:
            print(f"  └─ ⚠️  No accountmanagement.json found, skipping risk-based scaling")
            continue

        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found for {current_inv_id}")
            continue
        
        account_info = mt5.account_info()
        if account_info:
            account_balance = account_info.balance
            print(f"  └─ 💵 Live account balance: ${account_balance:,.2f}")
        else:
            print(f"  └─ ⚠️  Could not fetch account balance from broker")
            continue
        
        required_risk = 0
        tolerance_min = 0
        tolerance_max = 0
        
        for range_str, risk_value in risk_ranges.items():
            try:
                if '_risk' in range_str:
                    range_part = range_str.replace('_risk', '')
                    if '-' in range_part:
                        min_val, max_val = map(float, range_part.split('-'))
                        if min_val <= account_balance <= max_val:
                            required_risk = float(risk_value)
                            tolerance_min = required_risk
                            tolerance_max = required_risk + 0.99
                            print(f"  └─ 🎯 Balance ${account_balance:,.2f} falls in range {range_part}")
                            print(f"  └─ 🎯 Required risk: ${required_risk:.2f} (tolerance: ${tolerance_min:.2f} - ${tolerance_max:.2f})")
                            break
            except Exception as e:
                continue
        
        if required_risk == 0:
            print(f"  └─ ⚠️  No matching risk range found for balance ${account_balance:,.2f}")
            continue
        
        order_files = list(inv_folder.rglob("*/pending_orders/limit_orders.json"))
        
        if not order_files:
            print(f"  └─ 🔘 No limit order files found")
            continue
            
        investor_files_updated = 0
        investor_orders_updated = 0
        investor_risk_usd = 0.0
        investor_signals_count = 0
        
        broker_prefix = broker_cfg.get('BROKER_NAME', '').lower()
        if not broker_prefix:
            server = broker_cfg.get('SERVER', '')
            broker_prefix = server.split('-')[0].split('.')[0].lower() if server else 'broker'
        
        print(f"  └─ 🏷️  Using broker prefix: '{broker_prefix}' for field names")
        
        for file_path in order_files:
            try:
                # --- PRE-PROCESS DEDUPLICATION ---
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_orders = json.load(f)
                
                if not raw_orders:
                    continue

                # Deduplicate based on symbol, entry, and exit
                unique_orders = []
                seen_keys = set()
                for o in raw_orders:
                    key = (o.get('symbol'), o.get('entry'), o.get('exit'))
                    if key not in seen_keys:
                        unique_orders.append(o)
                        seen_keys.add(key)
                
                if len(unique_orders) < len(raw_orders):
                    print(f"    └─ 🧹 Cleaned {len(raw_orders) - len(unique_orders)} duplicate orders from {file_path.name}")
                
                orders = unique_orders

                # Clear signals.json for this specific folder to prevent stale data mixing with splits
                signals_path = file_path.parent / "signals.json"
                if signals_path.exists():
                    with open(signals_path, 'w', encoding='utf-8') as f:
                        json.dump([], f)
                    print(f"    └─ 🚿 Cleared existing signals.json for fresh split generation")

                # --- START PROCESSING ---
                if callback_function:
                    try:
                        callback_function(current_inv_id, file_path, orders)
                    except Exception as e:
                        print(f"    └─ ⚠️  Callback error: {e}")
                
                orders_modified = False
                file_risk_total = 0.0
                file_signals = []
                
                for order in orders:
                    raw_symbol = order.get("symbol", "")
                    if not raw_symbol:
                        continue
                    
                    if raw_symbol in resolution_cache:
                        normalized_symbol = resolution_cache[raw_symbol]
                    else:
                        normalized_symbol = get_normalized_symbol(raw_symbol)
                        resolution_cache[raw_symbol] = normalized_symbol
                        if normalized_symbol != raw_symbol:
                            print(f"    └─ ✅ {raw_symbol} -> {normalized_symbol} (Mapped & Cached)")
                            total_symbols_normalized += 1
                    
                    symbol = normalized_symbol if normalized_symbol else raw_symbol
                    order['symbol'] = symbol
                    
                    volume_field = f"{broker_prefix}_volume"
                    tick_size_field = f"{broker_prefix}_tick_size"
                    tick_value_field = f"{broker_prefix}_tick_value"
                    
                    current_volume = order.get(volume_field)
                    tick_size = order.get(tick_size_field)
                    tick_value = order.get(tick_value_field)
                    
                    if None in (current_volume, tick_size, tick_value):
                        print(f"    └─ ⚠️  Missing broker fields for {symbol}, skipping")
                        continue
                    
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info:
                        print(f"    └─ ⚠️  Could not fetch current price for {symbol}, skipping")
                        continue
                    
                    entry_price = order.get("entry")
                    exit_price = order.get("exit")
                    
                    if not entry_price or not exit_price:
                        continue
                    
                    stop_distance_pips = abs(entry_price - exit_price)
                    ticks_in_stop = stop_distance_pips / tick_size if tick_size > 0 else 0
                    
                    volume_step = symbol_info.volume_step
                    min_volume = symbol_info.volume_min
                    max_volume = symbol_info.volume_max
                    
                    best_volume = current_volume
                    best_risk = 0
                    test_volume = current_volume if current_volume >= min_volume else min_volume
                    test_risk = test_volume * ticks_in_stop * tick_value
                    
                    print(f"    └─ 📈 {symbol}: Starting volume {test_volume} -> risk ${test_risk:.2f}")
                    
                    if test_risk > tolerance_max:
                        print(f"      └─ ⬇️  Risk too high (${test_risk:.2f} > ${tolerance_max:.2f}), scaling down...")
                        while test_risk > tolerance_max and test_volume > min_volume:
                            test_volume = max(min_volume, test_volume - volume_step)
                            test_risk = test_volume * ticks_in_stop * tick_value
                        best_volume = test_volume
                        best_risk = test_risk
                    
                    elif test_risk < tolerance_min:
                        print(f"      └─ ⬆️  Risk too low (${test_risk:.2f} < ${tolerance_min:.2f}), scaling up...")
                        while test_risk < tolerance_min:
                            previous_volume = test_volume
                            previous_risk = test_risk
                            test_volume = test_volume + volume_step
                            test_risk = test_volume * ticks_in_stop * tick_value
                            
                            if test_risk > tolerance_max:
                                best_volume = previous_volume
                                best_risk = previous_risk
                                print(f"      └─ ✅ Using volume: {best_volume:.3f} (risk ${best_risk:.2f}) to avoid overshoot")
                                break
                        else:
                            best_volume = test_volume
                            best_risk = test_risk
                    else:
                        best_volume = test_volume
                        best_risk = test_risk
                        print(f"      └─ ✅ Already within tolerance: volume {best_volume:.3f} (risk ${best_risk:.2f})")

                    order[volume_field] = round(best_volume, 2)
                    order["risk_in_usd"] = round(best_risk, 2)
                    order["risk_calculated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    order["required_risk_target"] = required_risk
                    order["risk_tolerance_min"] = round(tolerance_min, 2)
                    order["risk_tolerance_max"] = round(tolerance_max, 2)
                    order["account_balance_at_calc"] = round(account_balance, 2)
                    order["current_bid"] = round(symbol_info.bid, 6) if hasattr(symbol_info, 'bid') else None
                    order["current_ask"] = round(symbol_info.ask, 6) if hasattr(symbol_info, 'ask') else None
                    
                    orders_modified = True
                    investor_orders_updated += 1
                    total_orders_updated += 1
                    file_risk_total += best_risk
                    
                    # --- SPLITTING LOGIC ---
                    if best_risk >= tolerance_min * 0.5 and best_risk > 0:
                        remaining_volume = best_volume
                        is_split = best_volume > max_volume
                        
                        while remaining_volume > 0.0001:
                            chunk_volume = min(remaining_volume, max_volume)
                            if chunk_volume < min_volume and remaining_volume != best_volume:
                                break
                                
                            signal_order = order.copy()
                            signal_order[volume_field] = round(chunk_volume, 2)
                            signal_order["split_order"] = is_split
                            signal_order["moved_to_signals_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            file_signals.append(signal_order)
                            investor_signals_count += 1
                            total_signals_created += 1
                            remaining_volume -= chunk_volume

                        if is_split:
                            print(f"      └─ 🟢 Qualified & SPLIT (Total Vol: {best_volume:.2f}, Max Vol: {max_volume})")
                        else:
                            print(f"      └─ 🟢 Qualified for signals.json (risk ${best_risk:.2f})")
                
                # Save cleaned and updated limit orders
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(orders, f, indent=4)
                
                investor_files_updated += 1
                total_files_updated += 1
                investor_risk_usd += file_risk_total
                total_risk_usd += file_risk_total
                any_orders_processed = True
                
                if file_signals:
                    try:
                        # We already cleared signals.json at the start, so we just write the new ones
                        with open(signals_path, 'w', encoding='utf-8') as f:
                            json.dump(file_signals, f, indent=4)
                        
                        print(f"  └─ 📊 signals.json: Created {len(file_signals)} clean signals (splits included)")
                    except Exception as e:
                        print(f"  └─ ❌ Error writing signals.json: {e}")
                
            except Exception as e:
                print(f"  └─ ❌ Error processing {file_path}: {e}")
                continue
        
        # Summary for current investor
        if investor_orders_updated > 0:
            print(f"\n  └─ {'='*40}")
            print(f"  └─ ✨ Investor {current_inv_id} Summary:")
            print(f"  └─     Files Processed:    {investor_files_updated}")
            print(f"  └─     Orders Risk-Scaled: {investor_orders_updated}")
            print(f"  └─     Total Risk:         ${investor_risk_usd:,.2f}")
            print(f"  └─     Signals Generated:  {investor_signals_count}")
            print(f"  └─ {'='*40}")

    print(f"\n{'='*10} USD RISK CALCULATION COMPLETE {'='*10}")
    return any_orders_processed

def apply_default_prices(inv_id=None, callback_function=None):
    """
    Applies default prices from limit_orders_backup.json to signals.json when default_price is true.
    Copies exit/target prices from backup to matching orders in signals.json, handling symbol normalization.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        callback_function (callable, optional): A function to call with the opened file data.
            The callback will receive (inv_id, backup_file_path, signals_file_path, modifications) parameters.
    
    Returns:
        bool: True if any orders were modified, False otherwise
    """
    print(f"\n{'='*10} 💰 APPLYING DEFAULT PRICES FROM BACKUP {'='*10}")
    
    total_orders_modified = 0
    total_files_updated = 0
    total_symbols_normalized = 0
    inv_base_path = Path(INV_PATH)

    if not inv_base_path.exists():
        print(f" [!] Error: Investor path {INV_PATH} does not exist.")
        return False

    # Determine which investors to process
    if inv_id:
        inv_folder = inv_base_path / inv_id
        investor_folders = [inv_folder] if inv_folder.exists() else []
        if not investor_folders:
            print(f" [!] Error: Investor folder {inv_id} does not exist.")
            return False
    else:
        investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    any_orders_modified = False

    for inv_folder in investor_folders:
        current_inv_id = inv_folder.name
        print(f"\n [{current_inv_id}] 🔍 Checking default price setting...")

        # --- INVESTOR LOCAL CACHE for symbol normalization ---
        resolution_cache = {}

        # 1. Load accountmanagement.json to check default_price setting
        account_mgmt_path = inv_folder / "accountmanagement.json"
        if not account_mgmt_path.exists():
            print(f"  └─ ⚠️  accountmanagement.json not found, skipping")
            continue

        try:
            with open(account_mgmt_path, 'r', encoding='utf-8') as f:
                account_data = json.load(f)
            
            settings = account_data.get('settings', {})
            default_price_enabled = settings.get('default_price', False)
            
            if not default_price_enabled:
                print(f"  └─ ⏭️  default_price is FALSE - skipping investor (set to true to apply default prices)")
                continue
                
            print(f"  └─ ✅ default_price is TRUE - will apply prices from backup")
            
        except Exception as e:
            print(f"  └─ ❌ Error reading accountmanagement.json: {e}")
            continue

        # 2. Load broker config for symbol handling
        broker_cfg = usersdictionary.get(current_inv_id)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found for {current_inv_id}")
            continue

        # 3. Find all limit_orders_backup.json files
        backup_files = list(inv_folder.rglob("*/pending_orders/limit_orders_backup.json"))
        
        if not backup_files:
            print(f"  └─ 🔘 No limit_orders_backup.json files found")
            continue
        
        print(f"  └─ 📁 Found {len(backup_files)} backup files to process")

        investor_orders_modified = 0
        investor_files_updated = 0
        investor_symbols_normalized = 0

        # 4. Process each backup file
        for backup_path in backup_files:
            folder_path = backup_path.parent.parent  # Gets the strategy folder (e.g., double-levels)
            signals_path = backup_path.parent / "signals.json"  # Same directory as backup
            
            # Check if signals.json exists
            if not signals_path.exists():
                print(f"  └─ ⚠️  No signals.json found in {backup_path.parent} (same folder as backup), skipping")
                continue
            
            print(f"\n  └─ 📂 Processing folder: {folder_path.name}")
            print(f"      ├─ Backup: {backup_path.name}")
            print(f"      └─ Signals: {signals_path.name}")
            
            try:
                # Load backup orders
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_orders = json.load(f)
                
                # Load signals
                with open(signals_path, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
                
                if not backup_orders:
                    print(f"    └─ ⚠️  Empty backup file")
                    continue
                    
                if not signals:
                    print(f"    └─ ⚠️  Empty signals file")
                    continue
                
                # Create lookup dictionaries for backup orders with multiple matching strategies
                backup_lookup = {}  # (symbol, timeframe, order_type) -> order
                
                print(f"    └─ 📊 Processing {len(backup_orders)} backup orders and {len(signals)} signals")
                
                # First, let's analyze what's in the backup for AUDCAD 15m sell_limit
                audcad_backups = []
                for order in backup_orders:
                    # --- SYMBOL NORMALIZATION for backup symbols ---
                    raw_symbol = str(order.get('symbol', '')).upper()
                    if not raw_symbol:
                        continue
                    
                    # Check Cache First for backup symbol
                    if raw_symbol in resolution_cache:
                        normalized_symbol = resolution_cache[raw_symbol]
                    else:
                        # Perform mapping only once
                        normalized_symbol = get_normalized_symbol(raw_symbol)
                        resolution_cache[raw_symbol] = normalized_symbol
                        
                        # Log normalization on first discovery
                        if normalized_symbol != raw_symbol:
                            print(f"      └─ ✅ Backup: {raw_symbol} -> {normalized_symbol} (Mapped & Cached)")
                            investor_symbols_normalized += 1
                            total_symbols_normalized += 1
                    
                    # Use normalized symbol for lookup
                    symbol = normalized_symbol if normalized_symbol else raw_symbol
                    timeframe = str(order.get('timeframe', '')).upper()
                    order_type = str(order.get('order_type', '')).lower()
                    
                    if symbol == 'AUDCAD' and timeframe == '15M' and order_type == 'sell_limit':
                        audcad_backups.append(order)
                        print(f"      └─ 📌 Found AUDCAD 15M sell_limit in backup with exit: {order.get('exit')}")
                    
                    if symbol and timeframe and order_type:
                        # Store with normalized symbol
                        key = (symbol, timeframe, order_type)
                        backup_lookup[key] = order
                
                print(f"    └─ 📊 Created lookup for {len(backup_lookup)} backup orders")
                
                # Process signals and apply default prices
                modified = False
                signals_modified_count = 0
                modifications_log = []
                
                for signal in signals:
                    # --- SYMBOL NORMALIZATION for signal symbols ---
                    raw_signal_symbol = str(signal.get('symbol', '')).upper()
                    if not raw_signal_symbol:
                        continue
                    
                    # Check Cache First for signal symbol
                    if raw_signal_symbol in resolution_cache:
                        signal_symbol = resolution_cache[raw_signal_symbol]
                    else:
                        # Perform mapping only once
                        signal_symbol = get_normalized_symbol(raw_signal_symbol)
                        resolution_cache[raw_signal_symbol] = signal_symbol
                        
                        # Log normalization on first discovery
                        if signal_symbol != raw_signal_symbol:
                            print(f"      └─ ✅ Signal: {raw_signal_symbol} -> {signal_symbol} (Mapped & Cached)")
                            investor_symbols_normalized += 1
                            total_symbols_normalized += 1
                    
                    signal_timeframe = str(signal.get('timeframe', '')).upper()
                    signal_type = str(signal.get('order_type', '')).lower()
                    
                    if not all([signal_symbol, signal_timeframe, signal_type]):
                        print(f"      └─ ⚠️  Signal missing required fields: {signal}")
                        continue
                    
                    # Special debug for AUDCAD+ 15M (now normalized to AUDCAD)
                    if raw_signal_symbol == 'AUDCAD+' and signal_timeframe == '15M' and signal_type == 'sell_limit':
                        print(f"      └─ 🔍 DEBUG: Processing AUDCAD+ 15M sell_limit signal (normalized to {signal_symbol})")
                        print(f"          Current exit: {signal.get('exit')}, target: {signal.get('target')}")
                    
                    # Try to find matching backup order
                    matched_backup = None
                    match_method = None
                    
                    # Method 1: Direct symbol match with normalized symbols
                    backup_key = (signal_symbol, signal_timeframe, signal_type)
                    if backup_key in backup_lookup:
                        matched_backup = backup_lookup[backup_key]
                        match_method = "direct match"
                        if raw_signal_symbol == 'AUDCAD+':
                            print(f"      └─ ✓ Found direct match for normalized AUDCAD")
                    
                    if matched_backup:
                        # Check if we need to update any prices
                        updates_made = False
                        
                        # Get backup values
                        backup_exit = matched_backup.get('exit', 0)
                        backup_target = matched_backup.get('target', 0)
                        
                        # Current signal values
                        current_exit = signal.get('exit', 0)
                        current_target = signal.get('target', 0)
                        
                        update_details = []
                        
                        # Apply backup exit if not zero and different from current
                        if backup_exit != 0 and backup_exit != current_exit:
                            signal['exit'] = backup_exit
                            updates_made = True
                            update_details.append(f"exit: {current_exit} -> {backup_exit}")
                        
                        # Apply backup target if not zero and different from current
                        if backup_target != 0 and backup_target != current_target:
                            signal['target'] = backup_target
                            updates_made = True
                            update_details.append(f"target: {current_target} -> {backup_target}")
                        
                        if updates_made:
                            # Add metadata about the update
                            signal['price_updated_from_backup'] = True
                            signal['price_updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            signal['backup_match_method'] = match_method
                            signal['original_symbol'] = raw_signal_symbol
                            
                            signals_modified_count += 1
                            investor_orders_modified += 1
                            total_orders_modified += 1
                            any_orders_modified = True
                            modified = True
                            
                            modifications_log.append({
                                'symbol': raw_signal_symbol,
                                'normalized_symbol': signal_symbol,
                                'timeframe': signal_timeframe,
                                'type': signal_type,
                                'updates': {
                                    'exit': backup_exit if backup_exit != 0 else None,
                                    'target': backup_target if backup_target != 0 else None
                                },
                                'match_method': match_method,
                                'update_details': ', '.join(update_details)
                            })
                            
                            print(f"      └─ 🔄 [{raw_signal_symbol} -> {signal_symbol}] {', '.join(update_details)} [{match_method}]")
                        else:
                            if raw_signal_symbol == 'AUDCAD+':
                                print(f"      └─ ✓ AUDCAD+ already has correct prices (exit={current_exit}, target={current_target})")
                    else:
                        # Debug: Show unmatched signals with more detail
                        if raw_signal_symbol == 'AUDCAD+':
                            print(f"      └─ ❌ FAILED to find match for AUDCAD+ 15M sell_limit (normalized to {signal_symbol})")
                            print(f"          Looking for backup_key: ({signal_symbol}, {signal_timeframe}, {signal_type})")
                            
                            # Show all available backup keys
                            print(f"          Available backup keys:")
                            for (bsym, btf, btype) in list(backup_lookup.keys())[:10]:
                                if btf == signal_timeframe and btype == signal_type:
                                    print(f"            • ({bsym}, {btf}, {btype})")
                        else:
                            print(f"      └─ ⚠️  No backup match for: {raw_signal_symbol} -> {signal_symbol} ({signal_timeframe}, {signal_type})")
                
                # Save modified signals file
                if modified:
                    try:
                        with open(signals_path, 'w', encoding='utf-8') as f:
                            json.dump(signals, f, indent=4)
                        
                        investor_files_updated += 1
                        total_files_updated += 1
                        
                        print(f"    └─ 📝 Updated {signals_modified_count} orders in signals.json")
                        
                        # Call callback if provided
                        if callback_function:
                            try:
                                callback_function(current_inv_id, backup_path, signals_path, modifications_log)
                            except Exception as e:
                                print(f"    └─ ⚠️  Callback error: {e}")
                        
                        # Show summary of modifications
                        if modifications_log:
                            print(f"    └─ 📋 Modification Summary:")
                            for mod in modifications_log[:5]:  # Show first 5
                                norm_info = f" -> {mod['normalized_symbol']}" if mod['symbol'] != mod['normalized_symbol'] else ""
                                print(f"      • {mod['symbol']}{norm_info} ({mod['timeframe']}): {mod['update_details']} [{mod['match_method']}]")
                            if len(modifications_log) > 5:
                                print(f"      • ... and {len(modifications_log) - 5} more")
                    
                    except Exception as e:
                        print(f"    └─ ❌ Error saving signals.json: {e}")
                else:
                    print(f"    └─ ✓ No price updates needed for signals in {folder_path.name}")
                
            except Exception as e:
                print(f"  └─ ❌ Error processing {backup_path}: {e}")
                continue

        # Investor summary
        if investor_orders_modified > 0:
            print(f"\n  └─ {'='*40}")
            print(f"  └─ ✨ Investor {current_inv_id} Summary:")
            print(f"  └─    Folders Processed:   {len(backup_files)}")
            print(f"  └─    Signals Files Updated: {investor_files_updated}")
            print(f"  └─    Orders Modified:     {investor_orders_modified}")
            if investor_symbols_normalized > 0:
                print(f"  └─    Symbols Normalized:   {investor_symbols_normalized}")
            print(f"  └─ {'='*40}")
        else:
            print(f"\n  └─ ⚠️  No modifications made for {current_inv_id}")

    # Final Global Summary
    print(f"\n{'='*10} DEFAULT PRICE APPLICATION COMPLETE {'='*10}")
    if total_orders_modified > 0:
        print(f" Total Files Updated:       {total_files_updated}")
        print(f" Total Orders Modified:     {total_orders_modified}")
        if total_symbols_normalized > 0:
            print(f" Total Symbols Normalized:  {total_symbols_normalized}")
        print(f"\n ✓ Default prices successfully applied from backup files")
    else:
        print(" No orders were modified.")
        print(" └─ Possible reasons:")
        print("    • default_price is false in accountmanagement.json")
        print("    • No matching orders found between backup and signals")
        print("    • All exit/target prices already match backup values")
        print("    • No limit_orders_backup.json files found")
    
    return any_orders_modified

def place_usd_orders(inv_id=None):
    """
    Places pending orders from signals.json files for investors.
    Performs a global existence check at the start to filter out duplicates.
    """
    
    # --- SUB-FUNCTION 2: COLLECT ORDERS FROM SIGNALS.JSON ---
    def collect_orders_from_signals(inv_root, resolution_cache):
        print(f"  📁 Scanning for signals.json files...")
        signals_files = list(inv_root.rglob("*/pending_orders/signals.json"))
        
        if not signals_files:
            print(f"  📁 No signals.json files found")
            return []
        
        entries_with_paths = [] 
        
        for signals_path in signals_files:
            if not signals_path.is_file(): continue
            try:
                with open(signals_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list): continue
                
                for entry in data:
                    raw_symbol = entry.get("symbol", "")
                    if not raw_symbol: continue
                    
                    # Symbol normalization
                    if raw_symbol in resolution_cache:
                        normalized_symbol = resolution_cache[raw_symbol]
                    else:
                        # Assuming get_normalized_symbol is defined globally
                        normalized_symbol = get_normalized_symbol(raw_symbol)
                        resolution_cache[raw_symbol] = normalized_symbol
                    
                    entry['symbol'] = normalized_symbol
                    entries_with_paths.append({'data': entry, 'path': signals_path})
                    
            except Exception as e:
                print(f"    ❌ Error reading {signals_path.name}: {e}")
        
        return entries_with_paths

    # --- SUB-FUNCTION 4: ORDER EXECUTION (MODIFIED) ---
    def execute_missing_orders(valid_entries, default_magic, trade_allowed):
        if not trade_allowed:
            print("  ⚠️  AutoTrading is DISABLED in Terminal")
            return 0, 0, 0
            
        placed = failed = skipped = 0
        
        for idx, entry_wrapper in enumerate(valid_entries, 1):
            entry = entry_wrapper['data']
            symbol = entry["symbol"]
            
            try:
                if not mt5.symbol_select(symbol, True):
                    print(f"      ❌ SYMBOL ERROR: {symbol} not found/selected")
                    failed += 1; continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    failed += 1; continue

                # VOLUME EXTRACTION & NORMALIZATION
                vol_key = next((k for k in entry.keys() if k.endswith("volume")), "volume")
                raw_vol = float(entry.get(vol_key, 0))
                
                volume = raw_vol
                if symbol_info.volume_step > 0:
                    volume = round(raw_vol / symbol_info.volume_step) * symbol_info.volume_step
                
                # Boundary Check
                volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))

                entry_price = round(float(entry["entry"]), symbol_info.digits)
                sl_price = round(float(entry["exit"]), symbol_info.digits)
                tp_price = round(float(entry["target"]), symbol_info.digits)

                ot_str = entry.get("order_type", "").lower()
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT if "buy" in ot_str else mt5.ORDER_TYPE_SELL_LIMIT

                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": symbol,
                    "volume": round(volume, 2),
                    "type": mt5_order_type,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "magic": int(entry.get("magic", default_magic)),
                    "comment": f"RR{entry.get('risk_reward', '?')}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                res = mt5.order_send(request)
                
                if res is None:
                    print(f"      ❌ CRITICAL: No response from MT5 for {symbol}")
                    failed += 1
                    continue

                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"      ✅ SUCCESS: {symbol} @ {entry_price}")
                    placed += 1
                else:
                    # SMART ERROR INTERPRETATION
                    error_msg = res.comment
                    if res.retcode == mt5.TRADE_RETCODE_LIMIT_ORDERS:
                        error_msg = "BROKER LIMIT: Maximum number of pending orders reached."
                    elif res.retcode == mt5.TRADE_RETCODE_LIMIT_VOLUME:
                        error_msg = "LIQUIDITY LIMIT: Maximum aggregate volume for this symbol reached."
                    elif res.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
                        error_msg = f"INVALID VOL: {volume} is outside broker steps/limits."
                    elif res.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                        error_msg = "INSUFFICIENT MARGIN: Cannot afford order."

                    print(f"      ⚠️  REJECTED: {symbol} -> {error_msg} (Code: {res.retcode})")
                    failed += 1

            except Exception as e:
                print(f"      💥 ERROR: {symbol} - {e}")
                failed += 1
                
        return placed, failed, skipped

    # --- MAIN EXECUTION FLOW ---
    print("\n" + "="*80)
    print("🚀 STARTING USD ORDER PLACEMENT ENGINE (GLOBAL CHECK)")
    print("="*80)
    
    investor_ids = [inv_id] if inv_id else list(usersdictionary.keys()) # Assumes usersdictionary exists
    any_orders_placed = False

    for user_brokerid in investor_ids:
        print(f"\n📋 INVESTOR: {user_brokerid}")
        resolution_cache = {}
        # Assumes INV_PATH is defined globally
        inv_root = Path(INV_PATH) / user_brokerid 
        
        if not inv_root.exists():
            print(f"  ❌ Path not found: {inv_root}")
            continue

        # 1. Collect all signals
        entries_with_paths = collect_orders_from_signals(inv_root, resolution_cache)
        if not entries_with_paths: continue

        # 2. GLOBAL CHECK
        print(f"  🔍 STAGE 1.5: Performing Global Existence Check...")
        
        active_positions = mt5.positions_get() or []
        pending_orders = mt5.orders_get() or []
        
        existing_lookup = set()
        for p in active_positions:
            existing_lookup.add((p.symbol, round(p.price_open, 5), round(p.volume, 2)))
        for o in pending_orders:
            existing_lookup.add((o.symbol, round(o.price_open, 5), round(o.volume_initial, 2)))

        to_place = []
        to_move = [] 

        for item in entries_with_paths:
            data = item['data']
            vol_key = next((k for k in data.keys() if k.endswith("volume")), "volume")
            sig_vol = round(float(data.get(vol_key, 0)), 2)
            sig_price = round(float(data['entry']), 5)
            
            sig_key = (data['symbol'], sig_price, sig_vol)

            if sig_key in existing_lookup:
                to_move.append(item)
            else:
                to_place.append(item)

        # 3. Move Existing to placed_orders.json
        if to_move:
            print(f"  ⏭️  Moving {len(to_move)} existing records to history...")
            for item in to_move:
                hist_path = item['path'].parent / "placed_orders.json"
                
                current_hist = []
                if hist_path.exists():
                    try:
                        with open(hist_path, 'r') as hf: current_hist = json.load(hf)
                    except: pass
                current_hist.append(item['data'])
                with open(hist_path, 'w') as hf: json.dump(current_hist, hf, indent=4)

                try:
                    with open(item['path'], 'r') as sf: current_sigs = json.load(sf)
                    new_sigs = [s for s in current_sigs if not (s['symbol'] == item['data']['symbol'] and float(s['entry']) == float(item['data']['entry']))]
                    with open(item['path'], 'w') as sf: json.dump(new_sigs, sf, indent=4)
                except Exception as e: print(f" Error cleaning file: {e}")

        # 4. Final Placement
        if to_place:
            try:
                acc_mgmt_path = inv_root / "accountmanagement.json"
                if acc_mgmt_path.exists():
                    with open(acc_mgmt_path, 'r') as f: config = json.load(f)
                    p, f, s = execute_missing_orders(to_place, config.get("magic_number", 123456), mt5.terminal_info().trade_allowed)
                    if p > 0: any_orders_placed = True
                else:
                    print(f"  ⚠️  Missing accountmanagement.json for {user_brokerid}")
            except Exception as e:
                print(f"  💥 Execution Error: {e}")
        else:
            print("  ℹ️  No new unique orders to place.")

    print("\n✅ PROCESS COMPLETE")
    return any_orders_placed

def check_pending_orders_risk(inv_id=None):
    """
    Function 3: Validates live pending orders against the account's current risk bucket.
     VERSION: Uses the EXACT account initialization logic from place_usd_orders_for_accounts()
    Only removes orders with risk HIGHER than allowed (lower risk orders are kept).
    
    NOW CHECKS: ALL pending orders (LIMIT, STOP, STOP-LIMIT)
    
    RISK CONFIGURATION LOGIC:
    - If enable_maximum_account_balance_management = true -> use account_balance_maximum_risk_management
    - Else if enable_default_account_balance_management = true -> use account_balance_default_risk_management
    - Else (both false) -> default to account_balance_default_risk_management
    
    Args:
        inv_id: Optional specific investor ID to process. If None, processes all investors.
        
    Returns:
        dict: Statistics about the processing
    """
    print(f"\n{'='*10} 🛡️ LIVE RISK AUDIT: ALL PENDING ORDERS (LIMIT + STOP)  {'='*10}")
    if inv_id:
        print(f" Processing single investor: {inv_id}")

    # --- DATA INITIALIZATION ---
    stats = {
        "investor_id": inv_id if inv_id else "all",
        "orders_checked": 0,
        "orders_removed": 0,
        "orders_kept_lower": 0,
        "orders_kept_in_range": 0,
        "risk_config_used": None,
        "processing_success": False
    }
    
    try:
        if not os.path.exists(NORMALIZE_SYMBOLS_PATH):
            print(" [!] CRITICAL ERROR: Normalization map path missing.")
            return stats
        with open(NORMALIZE_SYMBOLS_PATH, 'r') as f:
            norm_map = json.load(f)
    except Exception as e:
        print(f" [!] CRITICAL ERROR: Normalization map load failed: {e}")
        return stats

    # Define MT5 order types for better readability
    ORDER_TYPES = {
        mt5.ORDER_TYPE_BUY_LIMIT: "BUY LIMIT",
        mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
        mt5.ORDER_TYPE_BUY_STOP: "BUY STOP",
        mt5.ORDER_TYPE_SELL_STOP: "SELL STOP",
        mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY STOP-LIMIT",
        mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL STOP-LIMIT"
    }

    # Determine which investors to process
    investors_to_process = [inv_id] if inv_id else usersdictionary.keys()
    total_investors = len(investors_to_process) if not inv_id else 1
    processed = 0

    for user_brokerid in investors_to_process:
        processed += 1
        print(f"\n[{processed}/{total_investors}] {user_brokerid} 🔍 Auditing live risk limits...")
        
        # Get broker config
        broker_cfg = usersdictionary.get(user_brokerid)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found")
            continue
        
        inv_root = Path(INV_PATH) / user_brokerid
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  Account config missing. Skipping.")
            continue

        # --- LOAD CONFIG AND DETERMINE RISK CONFIGURATION TO USE ---
        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Get settings flags
            settings = config.get("settings", {})
            enable_default = settings.get("enable_default_account_balance_management", False)
            enable_maximum = settings.get("enable_maximum_account_balance_management", False)
            
            print(f"  └─ ⚙️  Risk Configuration Settings:")
            print(f"      • enable_default_account_balance_management: {enable_default}")
            print(f"      • enable_maximum_account_balance_management: {enable_maximum}")
            
            # Determine which risk config to use
            risk_map = None
            risk_config_used = None
            
            # LOGIC: If maximum is enabled, use maximum (even if default is also enabled)
            if enable_maximum:
                risk_map = config.get("account_balance_maximum_risk_management", {})
                risk_config_used = "maximum"
                print(f"      📋 USING: account_balance_maximum_risk_management (maximum enabled)")
            
            # Else if default is enabled (and maximum is false), use default
            elif enable_default:
                risk_map = config.get("account_balance_default_risk_management", {})
                risk_config_used = "default"
                print(f"      📋 USING: account_balance_default_risk_management (default enabled)")
            
            # Else (both false), default to default risk management
            else:
                risk_map = config.get("account_balance_default_risk_management", {})
                risk_config_used = "default (fallback)"
                print(f"      📋 USING: account_balance_default_risk_management (fallback - both flags false)")
            
            if not risk_map:
                print(f"  └─ ⚠️  Selected risk configuration is empty or missing")
                continue
                
        except Exception as e:
            print(f"  └─ ❌ Failed to read config: {e}")
            continue

        # --- ACCOUNT CONNECTION CHECK (NO INIT/SHUTDOWN) ---
        print(f"  └─ 🔌 Checking account connection...")
        
        login_id = int(broker_cfg['LOGIN_ID'])
        mt5_path = broker_cfg["TERMINAL_PATH"]
        
        print(f"      • Terminal Path: {mt5_path}")
        print(f"      • Login ID: {login_id}")

        # Check if already logged into correct account
        acc = mt5.account_info()
        if acc is None or acc.login != login_id:
            print(f"  └─ ❌ Not logged into the correct account. Expected: {login_id}, Found: {acc.login if acc else 'None'}")
            continue
        else:
            print(f"      ✅ Connected to account: {acc.login}")

        acc_info = mt5.account_info()
        if not acc_info:
            print(f"  └─ ❌ Failed to get account info")
            continue
            
        balance = acc_info.balance

        # Get terminal info for additional details
        term_info = mt5.terminal_info()
        
        print(f"\n  └─ 📊 Account Details:")
        print(f"      • Balance: ${acc_info.balance:,.2f}")
        print(f"      • Equity: ${acc_info.equity:,.2f}")
        print(f"      • Free Margin: ${acc_info.margin_free:,.2f}")
        print(f"      • Margin Level: {acc_info.margin_level:.2f}%" if acc_info.margin_level else "      • Margin Level: N/A")
        print(f"      • AutoTrading: {'✅ ENABLED' if term_info.trade_allowed else '❌ DISABLED'}")

        # Determine Primary Risk Value based on selected risk map
        primary_risk = None
        for range_str, r_val in risk_map.items():
            try:
                raw_range = range_str.split("_")[0]
                low, high = map(float, raw_range.split("-"))
                if low <= balance <= high:
                    primary_risk = float(r_val)
                    break
            except Exception as e:
                print(f"  └─ ⚠️  Error parsing range '{range_str}': {e}")
                continue

        if primary_risk is None:
            print(f"  └─ ⚠️  No risk mapping for balance ${balance:,.2f} in selected config")
            continue

        print(f"\n  └─ 💰 Target Risk (from {risk_config_used} config): ${primary_risk:.2f}")
        
        # Store which config was used in stats
        stats["risk_config_used"] = risk_config_used

        # Check ALL Live Pending Orders (LIMIT, STOP, STOP-LIMIT)
        pending_orders = mt5.orders_get()
        investor_orders_checked = 0
        investor_orders_removed = 0
        investor_orders_kept_lower = 0
        investor_orders_kept_in_range = 0

        if pending_orders:
            print(f"  └─ 🔍 Scanning {len(pending_orders)} pending orders (ALL types)...")
            
            for order in pending_orders:
                # Skip if not a pending order type
                if order.type not in ORDER_TYPES.keys():
                    continue

                investor_orders_checked += 1
                stats["orders_checked"] += 1
                
                order_type_name = ORDER_TYPES.get(order.type, f"Unknown Type {order.type}")
                
                # Determine order direction for calculations
                is_buy = order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT]
                calc_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                
                # Calculate risk (stop loss distance in money)
                if order.sl == 0:
                    print(f"    └─ ⚠️  Order #{order.ticket} | {order_type_name} | {order.symbol} - No SL set, skipping risk check")
                    continue
                
                sl_profit = mt5.order_calc_profit(calc_type, order.symbol, order.volume_initial, 
                                                  order.price_open, order.sl)
                
                if sl_profit is not None:
                    order_risk_usd = round(abs(sl_profit), 2)
                    
                    # Use a percentage-based threshold instead of absolute dollar difference
                    # For small balances, absolute differences can be misleading
                    risk_difference = order_risk_usd - primary_risk
                    
                    # For very small balances (like $2), a difference of $0.50 is significant
                    # Use a relative threshold: 20% of primary risk or $0.50, whichever is smaller
                    relative_threshold = max(0.50, primary_risk * 0.2)
                    
                    print(f"    └─ 📋 Order #{order.ticket} | {order_type_name} | {order.symbol}")
                    print(f"       Risk: ${order_risk_usd:.2f} | Target Risk: ${primary_risk:.2f}")
                    
                    # Only remove if risk is significantly higher than allowed
                    if risk_difference > relative_threshold: 
                        print(f"       🗑️ PURGING: Risk too high")
                        print(f"       Risk: ${order_risk_usd:.2f} > Allowed: ${primary_risk:.2f} (Δ: ${risk_difference:.2f})")
                        
                        cancel_request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket
                        }
                        result = mt5.order_send(cancel_request)
                        
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            investor_orders_removed += 1
                            stats["orders_removed"] += 1
                            print(f"       ✅ Order removed successfully")
                        else:
                            error_msg = result.comment if result else "No response"
                            print(f"       ❌ Cancel failed: {error_msg}")
                    
                    elif order_risk_usd < primary_risk - relative_threshold:
                        # Lower risk - keep it (good for the account)
                        investor_orders_kept_lower += 1
                        stats["orders_kept_lower"] += 1
                        print(f"       ✅ KEEPING: Lower risk than allowed")
                        print(f"       Risk: ${order_risk_usd:.2f} < Allowed: ${primary_risk:.2f} (Δ: ${primary_risk - order_risk_usd:.2f})")
                    
                    else:
                        # Within tolerance - keep it
                        investor_orders_kept_in_range += 1
                        stats["orders_kept_in_range"] += 1
                        print(f"       ✅ KEEPING: Risk within tolerance")
                        print(f"       Risk: ${order_risk_usd:.2f} vs Allowed: ${primary_risk:.2f} (Δ: ${abs(risk_difference):.2f})")
                else:
                    print(f"    └─ ⚠️  Order #{order.ticket} - Could not calculate risk")

        # Investor final summary
        if investor_orders_checked > 0:
            print(f"\n  └─ 📊 Audit Results for {user_brokerid}:")
            print(f"       • Risk config used: {risk_config_used}")
            print(f"       • Orders checked: {investor_orders_checked}")
            if investor_orders_kept_lower > 0:
                print(f"       • Kept (lower risk): {investor_orders_kept_lower}")
            if investor_orders_kept_in_range > 0:
                print(f"       • Kept (in tolerance): {investor_orders_kept_in_range}")
            if investor_orders_removed > 0:
                print(f"       • Removed (too high): {investor_orders_removed}")
            else:
                print(f"       ✅ No orders needed removal")
            stats["processing_success"] = True
        else:
            print(f"  └─ 🔘 No pending orders found.")

    # --- FINAL SUMMARY ---
    print(f"\n{'='*10} 📊 RISK AUDIT SUMMARY {'='*10}")
    print(f"   Investor ID: {stats['investor_id']}")
    print(f"   Risk config used: {stats['risk_config_used']}")
    print(f"   Orders checked: {stats['orders_checked']}")
    print(f"   Orders removed: {stats['orders_removed']}")
    print(f"   Orders kept (lower risk): {stats['orders_kept_lower']}")
    print(f"   Orders kept (in tolerance): {stats['orders_kept_in_range']}")
    
    if stats['orders_checked'] > 0:
        removal_rate = (stats['orders_removed'] / stats['orders_checked']) * 100
        print(f"   Removal rate: {removal_rate:.1f}%")
    
    print(f"\n{'='*10} 🏁 RISK AUDIT COMPLETE {'='*10}\n")
    return stats

def orders_risk_correction(inv_id=None):
    """
    Function: Checks both live pending orders AND open positions (LIMIT, STOP, and MARKET)
    and adjusts their take profit levels based on the selected risk-reward ratio from
    accountmanagement.json. Only executes if risk_reward_correction setting is True.
    
     VERSION: Uses the EXACT account initialization logic from place_usd_orders_for_accounts()
    
    Args:
        inv_id: Optional specific investor ID to process. If None, processes all investors.
        
    Returns:
        dict: Statistics about the processing
    """
    print(f"\n{'='*10} 📐 RISK-REWARD CORRECTION: ALL POSITIONS & PENDING ORDERS (MARKET + LIMIT + STOP)  {'='*10}")
    if inv_id:
        print(f" Processing single investor: {inv_id}")

    # Track statistics
    stats = {
        "investor_id": inv_id if inv_id else "all",
        "orders_checked": 0,
        "orders_adjusted": 0,
        "orders_skipped": 0,
        "orders_error": 0,
        "positions_checked": 0,
        "positions_adjusted": 0,
        "processing_success": False
    }

    # Determine which investors to process
    investors_to_process = [inv_id] if inv_id else usersdictionary.keys()
    total_investors = len(investors_to_process) if not inv_id else 1
    processed = 0

    for user_brokerid in investors_to_process:
        processed += 1
        print(f"\n[{processed}/{total_investors}] {user_brokerid} 🔍 Checking risk-reward configurations...")
        
        # Get broker config
        broker_cfg = usersdictionary.get(user_brokerid)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found")
            continue
        
        inv_root = Path(INV_PATH) / user_brokerid
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  Account config missing. Skipping.")
            continue

        # --- LOAD CONFIG AND CHECK SETTINGS ---
        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check if risk_reward_correction is enabled
            settings = config.get("settings", {})
            if not settings.get("risk_reward_correction", False):
                print(f"  └─ ⏭️  Risk-reward correction disabled in settings. Skipping.")
                continue
            
            # Get selected risk-reward ratios
            selected_rr = config.get("selected_risk_reward", [2])
            if not selected_rr:
                print(f"  └─ ⚠️  No risk-reward ratios selected. Using default [2]")
                selected_rr = [2]
            
            # Use the first ratio in the list (typically the preferred one)
            target_rr_ratio = float(selected_rr[0])
            print(f"  └─ ✅ Target R:R Ratio: 1:{target_rr_ratio}")
            
            # Get risk management mapping for balance-based risk
            risk_map = config.get("account_balance_default_risk_management", {})
            
        except Exception as e:
            print(f"  └─ ❌ Failed to read config: {e}")
            stats["orders_error"] += 1
            continue

        # --- ACCOUNT INITIALIZATION (EXACT COPY FROM place_usd_orders_for_accounts) ---
        print(f"  └─ 🔌 Initializing account connection...")
        
        login_id = int(broker_cfg['LOGIN_ID'])
        mt5_path = broker_cfg["TERMINAL_PATH"]
        
        print(f"      • Terminal Path: {mt5_path}")
        print(f"      • Login ID: {login_id}")

        # Check login status
        acc = mt5.account_info()
        if acc is None or acc.login != login_id:
            print(f"      🔑 Logging into account...")
            if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                error = mt5.last_error()
                print(f"  └─ ❌  login failed: {error}")
                stats["orders_error"] += 1
                continue
            print(f"      ✅ Successfully logged into account")
        else:
            print(f"      ✅ Already logged into account")
        # --- END EXACT INITIALIZATION COPY ---

        acc_info = mt5.account_info()
        if not acc_info:
            print(f"  └─ ❌ Failed to get account info")
            stats["orders_error"] += 1
            continue
            
        balance = acc_info.balance

        # Get terminal info for additional details
        term_info = mt5.terminal_info()
        
        print(f"\n  └─ 📊 Account Details:")
        print(f"      • Balance: ${acc_info.balance:,.2f}")
        print(f"      • Equity: ${acc_info.equity:,.2f}")
        print(f"      • Free Margin: ${acc_info.margin_free:,.2f}")
        print(f"      • Margin Level: {acc_info.margin_level:.2f}%" if acc_info.margin_level else "      • Margin Level: N/A")
        print(f"      • AutoTrading: {'✅ ENABLED' if term_info.trade_allowed else '❌ DISABLED'}")

        # --- DETERMINE PRIMARY RISK VALUE BASED ON BALANCE ---
        primary_risk = None
        for range_str, r_val in risk_map.items():
            try:
                raw_range = range_str.split("_")[0]
                low, high = map(float, raw_range.split("-"))
                if low <= balance <= high:
                    primary_risk = float(r_val)
                    break
            except Exception as e:
                print(f"  └─ ⚠️  Error parsing range '{range_str}': {e}")
                continue

        if primary_risk is None:
            print(f"  └─ ⚠️  No risk mapping for balance ${balance:,.2f}")
            stats["orders_skipped"] += 1
            continue

        print(f"\n  └─ 💰 Balance: ${balance:,.2f} | Base Risk: ${primary_risk:.2f} | Target R:R: 1:{target_rr_ratio}")

        # --- CHECK AND ADJUST ALL POSITIONS (OPEN MARKET ORDERS) ---
        positions = mt5.positions_get()
        investor_positions_checked = 0
        investor_positions_adjusted = 0
        investor_positions_skipped = 0
        investor_positions_error = 0

        # Define MT5 position/order types for better readability
        POSITION_TYPES = {
            mt5.POSITION_TYPE_BUY: "BUY (MARKET)",
            mt5.POSITION_TYPE_SELL: "SELL (MARKET)"
        }
        
        ORDER_TYPES = {
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY STOP-LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL STOP-LIMIT"
        }

        # Process OPEN POSITIONS first
        if positions:
            print(f"\n  └─ 🔍 Scanning {len(positions)} open positions (MARKET)...")
            
            for position in positions:
                investor_positions_checked += 1
                stats["positions_checked"] += 1
                
                position_type_name = POSITION_TYPES.get(position.type, f"Unknown Type {position.type}")
                
                # Get symbol info
                symbol_info = mt5.symbol_info(position.symbol)
                if not symbol_info:
                    print(f"    └─ ⚠️  Cannot get symbol info for {position.symbol}")
                    investor_positions_skipped += 1
                    stats["orders_skipped"] += 1
                    continue

                # Determine position direction
                is_buy = position.type == mt5.POSITION_TYPE_BUY
                
                print(f"\n    └─ 📋 Position #{position.ticket} | {position_type_name} | {position.symbol}")
                
                # Calculate current risk (stop loss distance in money)
                if position.sl == 0:
                    print(f"       ⚠️  No SL set - cannot calculate risk. Skipping TP adjustment.")
                    investor_positions_skipped += 1
                    stats["orders_skipped"] += 1
                    continue
                
                # For positions, risk is from current price to SL (or entry to SL if price moved favorably)
                # We use the more conservative approach: risk based on original entry to SL
                # This ensures we don't reduce TP if price has moved in our favor
                if is_buy:
                    # For BUY: entry price is position.price_open
                    risk_price = position.price_open - position.sl
                else:
                    # For SELL: entry price is position.price_open
                    risk_price = position.sl - position.price_open
                
                # Calculate risk in money
                risk_points = abs(risk_price) / symbol_info.point
                point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size * symbol_info.point
                current_risk_usd = round(risk_points * point_value * position.volume, 2)
                
                # Alternative: calculate using MT5 profit calculator for accuracy
                calc_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                sl_profit = mt5.order_calc_profit(calc_type, position.symbol, position.volume, 
                                                  position.price_open, position.sl)
                
                if sl_profit is not None:
                    current_risk_usd = round(abs(sl_profit), 2)
                
                # Calculate required take profit based on risk and target R:R ratio
                target_profit_usd = current_risk_usd * target_rr_ratio
                
                print(f"       Risk (from entry): ${current_risk_usd:.2f} | Target Profit: ${target_profit_usd:.2f}")
                
                # Calculate the take profit price that would achieve this profit
                tick_value = symbol_info.trade_tick_value
                tick_size = symbol_info.trade_tick_size
                
                if tick_value > 0 and tick_size > 0:
                    # Calculate how many ticks we need to move to achieve target profit
                    ticks_needed = target_profit_usd / (position.volume * tick_value)
                    
                    # Convert ticks to price movement
                    price_move_needed = ticks_needed * tick_size
                    
                    # Round to symbol digits
                    digits = symbol_info.digits
                    price_move_needed = round(price_move_needed, digits)
                    
                    # Calculate new take profit price based on position type (from entry price, not current price)
                    if is_buy:
                        # For BUY positions: TP above entry price
                        new_tp = round(position.price_open + price_move_needed, digits)
                    else:
                        # For SELL positions: TP below entry price
                        new_tp = round(position.price_open - price_move_needed, digits)
                    
                    # Check if current TP is significantly different from calculated TP
                    if position.tp == 0:
                        target_move = abs(new_tp - position.price_open)
                        print(f"       📝 No TP currently set")
                        print(f"       Target TP: {new_tp:.{digits}f} (Move from entry: {target_move:.{digits}f})")
                        should_adjust = True
                    else:
                        current_move = abs(position.tp - position.price_open)
                        target_move = abs(new_tp - position.price_open)
                        
                        # Calculate threshold (10% of target move or 2 pips, whichever is larger)
                        pip_threshold = max(target_move * 0.1, symbol_info.point * 20)
                        
                        if abs(current_move - target_move) > pip_threshold:
                            print(f"       📐 TP needs adjustment")
                            print(f"       Current TP: {position.tp:.{digits}f} (Move from entry: {current_move:.{digits}f})")
                            print(f"       Target TP:  {new_tp:.{digits}f} (Move from entry: {target_move:.{digits}f})")
                            should_adjust = True
                        else:
                            print(f"       ✅ TP already correct")
                            print(f"       TP: {position.tp:.{digits}f} | Risk: ${current_risk_usd:.2f}")
                            investor_positions_skipped += 1
                            stats["orders_skipped"] += 1
                            continue
                    
                    if should_adjust:
                        # Prepare modification request for position
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "sl": position.sl,
                            "tp": new_tp,
                        }
                        
                        # Send modification
                        result = mt5.order_send(modify_request)
                        
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            investor_positions_adjusted += 1
                            stats["positions_adjusted"] += 1
                            stats["orders_adjusted"] += 1
                            print(f"       ✅ TP adjusted successfully to {new_tp:.{digits}f}")
                        else:
                            investor_positions_error += 1
                            stats["orders_error"] += 1
                            error_msg = result.comment if result else f"Error code: {result.retcode if result else 'Unknown'}"
                            print(f"       ❌ Modification failed: {error_msg}")
                else:
                    print(f"       ⚠️  Invalid tick values - using fallback calculation")
                    # Fallback method using profit calculation for a small price movement
                    try:
                        test_move = symbol_info.point * 10
                        if is_buy:
                            test_price = position.price_open + test_move
                        else:
                            test_price = position.price_open - test_move
                            
                        test_profit = mt5.order_calc_profit(calc_type, position.symbol, position.volume, 
                                                            position.price_open, test_price)
                        
                        if test_profit and test_profit != 0:
                            point_value = abs(test_profit) / 10
                            price_move_needed = target_profit_usd / point_value * symbol_info.point
                            
                            digits = symbol_info.digits
                            price_move_needed = round(price_move_needed, digits)
                            
                            if is_buy:
                                new_tp = round(position.price_open + price_move_needed, digits)
                            else:
                                new_tp = round(position.price_open - price_move_needed, digits)
                            
                            print(f"       Using fallback calculation")
                            
                            if position.tp == 0 or abs(position.tp - new_tp) > symbol_info.point * 20:
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": position.ticket,
                                    "sl": position.sl,
                                    "tp": new_tp,
                                }
                                
                                result = mt5.order_send(modify_request)
                                
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    investor_positions_adjusted += 1
                                    stats["positions_adjusted"] += 1
                                    stats["orders_adjusted"] += 1
                                    print(f"       ✅ TP adjusted using fallback method to {new_tp:.{digits}f}")
                                else:
                                    investor_positions_error += 1
                                    stats["orders_error"] += 1
                                    print(f"       ❌ Fallback modification failed")
                            else:
                                investor_positions_skipped += 1
                                stats["orders_skipped"] += 1
                                print(f"       ✅ TP already correct (fallback check)")
                        else:
                            investor_positions_skipped += 1
                            stats["orders_skipped"] += 1
                            print(f"       ⚠️  Cannot calculate using fallback method")
                    except Exception as e:
                        investor_positions_error += 1
                        stats["orders_error"] += 1
                        print(f"       ❌ Fallback calculation error: {e}")
        
        # --- CHECK AND ADJUST ALL PENDING ORDERS (LIMIT AND STOP) ---
        pending_orders = mt5.orders_get()
        investor_orders_checked = 0
        investor_orders_adjusted = 0
        investor_orders_skipped = 0
        investor_orders_error = 0

        if pending_orders:
            print(f"\n  └─ 🔍 Scanning {len(pending_orders)} pending orders (LIMIT & STOP)...")
            
            for order in pending_orders:
                # Skip if not a pending order (only process pending order types)
                if order.type not in ORDER_TYPES.keys():
                    continue

                investor_orders_checked += 1
                stats["orders_checked"] += 1
                
                order_type_name = ORDER_TYPES.get(order.type, f"Unknown Type {order.type}")
                
                # Get symbol info
                symbol_info = mt5.symbol_info(order.symbol)
                if not symbol_info:
                    print(f"    └─ ⚠️  Cannot get symbol info for {order.symbol}")
                    investor_orders_skipped += 1
                    stats["orders_skipped"] += 1
                    continue

                # Determine order direction for calculations
                is_buy = order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT]
                calc_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                
                print(f"\n    └─ 📋 Order #{order.ticket} | {order_type_name} | {order.symbol}")
                
                # Calculate current risk (stop loss distance in money)
                if order.sl == 0:
                    print(f"       ⚠️  No SL set - cannot calculate risk. Skipping TP adjustment.")
                    investor_orders_skipped += 1
                    stats["orders_skipped"] += 1
                    continue
                    
                # For pending orders, risk is from entry to SL
                sl_profit = mt5.order_calc_profit(calc_type, order.symbol, order.volume_initial, 
                                                  order.price_open, order.sl)
                
                if sl_profit is None:
                    print(f"       ⚠️  Cannot calculate risk. Skipping.")
                    investor_orders_skipped += 1
                    stats["orders_skipped"] += 1
                    continue

                current_risk_usd = round(abs(sl_profit), 2)
                
                # Calculate required take profit based on risk and target R:R ratio
                target_profit_usd = current_risk_usd * target_rr_ratio
                
                print(f"       Risk: ${current_risk_usd:.2f} | Target Profit: ${target_profit_usd:.2f}")
                
                # Calculate the take profit price that would achieve this profit
                tick_value = symbol_info.trade_tick_value
                tick_size = symbol_info.trade_tick_size
                
                if tick_value > 0 and tick_size > 0:
                    # Calculate how many ticks we need to move to achieve target profit
                    ticks_needed = target_profit_usd / (order.volume_initial * tick_value)
                    
                    # Convert ticks to price movement
                    price_move_needed = ticks_needed * tick_size
                    
                    # Round to symbol digits
                    digits = symbol_info.digits
                    price_move_needed = round(price_move_needed, digits)
                    
                    # Calculate new take profit price based on order type
                    if is_buy:
                        # For BUY orders: TP above entry
                        new_tp = round(order.price_open + price_move_needed, digits)
                    else:
                        # For SELL orders: TP below entry
                        new_tp = round(order.price_open - price_move_needed, digits)
                    
                    # Check if current TP is significantly different from calculated TP
                    current_move = abs(order.tp - order.price_open) if order.tp != 0 else 0
                    target_move = abs(new_tp - order.price_open)
                    
                    # Calculate threshold (10% of target move or 2 pips, whichever is larger)
                    pip_threshold = max(target_move * 0.1, symbol_info.point * 20)
                    
                    should_adjust = False
                    
                    if order.tp == 0:
                        print(f"       📝 No TP currently set")
                        print(f"       Target TP: {new_tp:.{digits}f} (Move: {target_move:.{digits}f})")
                        should_adjust = True
                    elif abs(current_move - target_move) > pip_threshold:
                        print(f"       📐 TP needs adjustment")
                        print(f"       Current TP: {order.tp:.{digits}f} (Move: {current_move:.{digits}f})")
                        print(f"       Target TP:  {new_tp:.{digits}f} (Move: {target_move:.{digits}f})")
                        should_adjust = True
                    else:
                        print(f"       ✅ TP already correct")
                        print(f"       TP: {order.tp:.{digits}f} | Risk: ${current_risk_usd:.2f}")
                        investor_orders_skipped += 1
                        stats["orders_skipped"] += 1
                        continue
                    
                    if should_adjust:
                        # Prepare modification request
                        modify_request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": order.ticket,
                            "price": order.price_open,
                            "sl": order.sl,
                            "tp": new_tp,
                        }
                        
                        # Send modification
                        result = mt5.order_send(modify_request)
                        
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            investor_orders_adjusted += 1
                            stats["orders_adjusted"] += 1
                            print(f"       ✅ TP adjusted successfully to {new_tp:.{digits}f}")
                        else:
                            investor_orders_error += 1
                            stats["orders_error"] += 1
                            error_msg = result.comment if result else f"Error code: {result.retcode if result else 'Unknown'}"
                            print(f"       ❌ Modification failed: {error_msg}")
                else:
                    print(f"       ⚠️  Invalid tick values - using fallback calculation")
                    # Fallback method using profit calculation for a small price movement
                    try:
                        test_move = symbol_info.point * 10
                        if is_buy:
                            test_price = order.price_open + test_move
                        else:
                            test_price = order.price_open - test_move
                            
                        test_profit = mt5.order_calc_profit(calc_type, order.symbol, order.volume_initial, 
                                                            order.price_open, test_price)
                        
                        if test_profit and test_profit != 0:
                            point_value = abs(test_profit) / 10
                            price_move_needed = target_profit_usd / point_value * symbol_info.point
                            
                            digits = symbol_info.digits
                            price_move_needed = round(price_move_needed, digits)
                            
                            if is_buy:
                                new_tp = round(order.price_open + price_move_needed, digits)
                            else:
                                new_tp = round(order.price_open - price_move_needed, digits)
                            
                            print(f"       Using fallback calculation")
                            
                            if order.tp == 0 or abs(order.tp - new_tp) > symbol_info.point * 20:
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_MODIFY,
                                    "order": order.ticket,
                                    "price": order.price_open,
                                    "sl": order.sl,
                                    "tp": new_tp,
                                }
                                
                                result = mt5.order_send(modify_request)
                                
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    investor_orders_adjusted += 1
                                    stats["orders_adjusted"] += 1
                                    print(f"       ✅ TP adjusted using fallback method to {new_tp:.{digits}f}")
                                else:
                                    investor_orders_error += 1
                                    stats["orders_error"] += 1
                                    print(f"       ❌ Fallback modification failed")
                            else:
                                investor_orders_skipped += 1
                                stats["orders_skipped"] += 1
                                print(f"       ✅ TP already correct (fallback check)")
                        else:
                            investor_orders_skipped += 1
                            stats["orders_skipped"] += 1
                            print(f"       ⚠️  Cannot calculate using fallback method")
                    except Exception as e:
                        investor_orders_error += 1
                        stats["orders_error"] += 1
                        print(f"       ❌ Fallback calculation error: {e}")

        # --- INVESTOR SUMMARY ---
        total_checked = investor_positions_checked + investor_orders_checked
        total_adjusted = investor_positions_adjusted + investor_orders_adjusted
        
        if total_checked > 0:
            print(f"\n  └─ 📊 Risk-Reward Correction Results for {user_brokerid}:")
            if investor_positions_checked > 0:
                print(f"       • Positions checked: {investor_positions_checked}")
                print(f"       • Positions adjusted: {investor_positions_adjusted}")
                print(f"       • Positions skipped: {investor_positions_skipped}")
            if investor_orders_checked > 0:
                print(f"       • Pending orders checked: {investor_orders_checked}")
                print(f"       • Pending orders adjusted: {investor_orders_adjusted}")
                print(f"       • Pending orders skipped: {investor_orders_skipped}")
            if investor_positions_error + investor_orders_error > 0:
                print(f"       • Errors: {investor_positions_error + investor_orders_error}")
            else:
                print(f"       ✅ All adjustments completed successfully")
            stats["processing_success"] = True
        else:
            print(f"  └─ 🔘 No positions or pending orders found.")

    # --- FINAL SUMMARY ---
    print(f"\n{'='*10} 📊 RISK-REWARD CORRECTION SUMMARY {'='*10}")
    print(f"   Investor ID: {stats['investor_id']}")
    print(f"   Positions checked: {stats['positions_checked']}")
    print(f"   Positions adjusted: {stats['positions_adjusted']}")
    print(f"   Pending orders checked: {stats['orders_checked']}")
    print(f"   Pending orders adjusted: {stats['orders_adjusted']}")
    print(f"   Total checked: {stats['positions_checked'] + stats['orders_checked']}")
    print(f"   Total adjusted: {stats['positions_adjusted'] + stats['orders_adjusted']}")
    print(f"   Orders skipped: {stats['orders_skipped']}")
    print(f"   Errors: {stats['orders_error']}")
    
    total_checked = stats['positions_checked'] + stats['orders_checked']
    total_adjusted = stats['positions_adjusted'] + stats['orders_adjusted']
    if total_checked > 0:
        success_rate = (total_adjusted / total_checked) * 100
        print(f"   Adjustment success rate: {success_rate:.1f}%")
    
    print(f"\n{'='*10} 🏁 POSITIONS & PENDING ORDERS RISK-REWARD CORRECTION COMPLETE {'='*10}\n")
    return stats

def history_closed_orders_removal_in_pendingorders(inv_id=None):
    """
    Scans history for the last 48 hours. If a position was closed, 
    any pending limit orders with the same first 4 digits in the price 
    are cancelled to prevent re-entry.
    
    Args:
        inv_id (str, optional): Specific investor ID to process. If None, processes all investors.
        MT5 should already be initialized and logged in for this investor.
    
    Returns:
        bool: True if any orders were removed, False otherwise
    """
    from datetime import datetime, timedelta
    print(f"\n{'='*10} 📜 HISTORY AUDIT: PREVENTING RE-ENTRY {'='*10}")

    # Determine which investors to process
    if inv_id:
        investor_ids = [inv_id]
    else:
        investor_ids = list(usersdictionary.keys())
    
    if not investor_ids:
        print(" └─ 🔘 No investors found.")
        return False

    any_orders_removed = False

    for user_brokerid in investor_ids:
        print(f" [{user_brokerid}] 🔍 Checking 48h history for duplicates...")
        
        broker_cfg = usersdictionary.get(user_brokerid)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found for {user_brokerid}")
            continue

        # 1. Define the 48-hour window
        from_date = datetime.now() - timedelta(hours=48)
        to_date = datetime.now()

        # 2. Get Closed Positions (Deals)
        history_deals = mt5.history_deals_get(from_date, to_date)
        if history_deals is None:
            print(f"  └─ ⚠️ Could not access history for {user_brokerid}")
            continue

        # 3. Create a set of "Used Price Prefixes"
        # We store: (symbol, price_prefix)
        used_entries = set()
        for deal in history_deals:
            # Only look at actual trades (buy/sell) that were closed
            if deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                # Extract first 3 significant digits of the price
                # We remove the decimal to handle 0.856 and 1901 uniformly
                clean_price = str(deal.price).replace('.', '')[:4]
                used_entries.add((deal.symbol, clean_price))

        if not used_entries:
            print(f"  └─ ✅ No closed orders found in last 48h.")
            continue

        # 4. Check Current Pending Orders
        pending_orders = mt5.orders_get()
        removed_count = 0
        orders_checked = 0

        if pending_orders:
            for order in pending_orders:
                # Only target limit orders
                if order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
                    orders_checked += 1
                    order_price_prefix = str(order.price_open).replace('.', '')[:4]
                    
                    # If this symbol + price prefix exists in history, kill the order
                    if (order.symbol, order_price_prefix) in used_entries:
                        print(f"  └─ 🚫 DUPLICATE FOUND: {order.symbol} at {order.price_open}")
                        print(f"     Match found in history (Prefix: {order_price_prefix}). Cancelling...")
                        
                        cancel_request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket
                        }
                        res = mt5.order_send(cancel_request)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            removed_count += 1
                            any_orders_removed = True
                            print(f"     ✅ Order #{order.ticket} cancelled successfully")
                        else:
                            error_msg = res.comment if res else f"Error code: {res.retcode if res else 'Unknown'}"
                            print(f"     ❌ Failed to cancel #{order.ticket}: {error_msg}")

        print(f"  └─ 📊 Cleanup Result: {removed_count} duplicate limit orders removed out of {orders_checked} checked.")

    print(f"\n{'='*10} 🏁 HISTORY AUDIT COMPLETE {'='*10}\n")
    return any_orders_removed

def apply_dynamic_breakeven(inv_id=None):
    """
    Function: Dynamically moves stop loss to breakeven or partial profit levels based on
    running profit reward multiples. Uses breakeven_dictionary from accountmanagement.json
    to determine at which reward levels to adjust SL.
    
    Args:
        inv_id: Optional specific investor ID to process. If None, processes all investors.
        
    Returns:
        dict: Statistics about the processing
    """
    print(f"\n{'='*10} 🎯 DYNAMIC BREAKEVEN: MONITORING RUNNING PROFIT REWARDS {'='*10}")
    if inv_id:
        print(f" Processing single investor: {inv_id}")

    # Track statistics
    stats = {
        "investor_id": inv_id if inv_id else "all",
        "positions_checked": 0,
        "positions_adjusted": 0,
        "positions_skipped": 0,
        "positions_error": 0,
        "breakeven_events": 0,
        "processing_success": False
    }

    # Determine which investors to process
    investors_to_process = [inv_id] if inv_id else usersdictionary.keys()
    total_investors = len(investors_to_process) if not inv_id else 1
    processed = 0

    for user_brokerid in investors_to_process:
        processed += 1
        print(f"\n[{processed}/{total_investors}] {user_brokerid} 🔍 Checking breakeven configurations...")
        
        # Get broker config
        broker_cfg = usersdictionary.get(user_brokerid)
        if not broker_cfg:
            print(f"  └─ ❌ No broker config found")
            continue
        
        inv_root = Path(INV_PATH) / user_brokerid
        acc_mgmt_path = inv_root / "accountmanagement.json"

        if not acc_mgmt_path.exists():
            print(f"  └─ ⚠️  Account config missing. Skipping.")
            continue

        # --- LOAD CONFIG AND CHECK SETTINGS ---
        try:
            with open(acc_mgmt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check if breakeven is enabled
            settings = config.get("settings", {})
            if not settings.get("enable_breakeven", False):
                print(f"  └─ ⏭️  Breakeven disabled in settings. Skipping.")
                continue
            
            # Get breakeven dictionary
            breakeven_config = settings.get("breakeven_dictionary", [])
            if not breakeven_config:
                print(f"  └─ ⚠️  No breakeven configuration found. Using default.")
                # Default configuration if none provided
                breakeven_config = [
                    {"reward": 1, "breakeven_at_reward": 0.5},
                    {"reward": 2, "breakeven_at_reward": 1},
                    {"reward": 3, "breakeven_at_reward": 1.5}
                ]
            
            # Sort by reward level (ascending) to process in order
            breakeven_config.sort(key=lambda x: x["reward"])
            
            print(f"  └─ ✅ Breakeven enabled with {len(breakeven_config)} reward levels:")
            for level in breakeven_config:
                print(f"       • At {level['reward']}R profit → Move SL to {level['breakeven_at_reward']}R")
            
        except Exception as e:
            print(f"  └─ ❌ Failed to read config: {e}")
            stats["positions_error"] += 1
            continue

        # --- ACCOUNT INITIALIZATION ---
        print(f"  └─ 🔌 Initializing account connection...")
        
        login_id = int(broker_cfg['LOGIN_ID'])
        mt5_path = broker_cfg["TERMINAL_PATH"]
        
        print(f"      • Terminal Path: {mt5_path}")
        print(f"      • Login ID: {login_id}")

        # Check login status
        acc = mt5.account_info()
        if acc is None or acc.login != login_id:
            print(f"      🔑 Logging into account...")
            if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                error = mt5.last_error()
                print(f"  └─ ❌ login failed: {error}")
                stats["positions_error"] += 1
                continue
            print(f"      ✅ Successfully logged into account")
        else:
            print(f"      ✅ Already logged into account")

        acc_info = mt5.account_info()
        if not acc_info:
            print(f"  └─ ❌ Failed to get account info")
            stats["positions_error"] += 1
            continue
            
        balance = acc_info.balance
        print(f"\n  └─ 📊 Account Balance: ${balance:,.2f}")

        # --- CHECK ALL OPEN POSITIONS ---
        positions = mt5.positions_get()
        investor_positions_checked = 0
        investor_positions_adjusted = 0
        investor_positions_skipped = 0
        investor_positions_error = 0
        investor_breakeven_events = 0

        # Define position types for better readability
        POSITION_TYPES = {
            mt5.POSITION_TYPE_BUY: "BUY",
            mt5.POSITION_TYPE_SELL: "SELL"
        }

        if positions:
            print(f"\n  └─ 🔍 Scanning {len(positions)} open positions for breakeven opportunities...")
            
            for position in positions:
                investor_positions_checked += 1
                stats["positions_checked"] += 1
                
                position_type_name = POSITION_TYPES.get(position.type, f"Unknown Type {position.type}")
                
                # Skip positions without SL
                if position.sl == 0:
                    print(f"\n    └─ 📋 Position #{position.ticket} | {position_type_name} | {position.symbol}")
                    print(f"       ⚠️  No SL set - cannot manage breakeven. Skipping.")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1
                    continue

                # Get symbol info
                symbol_info = mt5.symbol_info(position.symbol)
                if not symbol_info:
                    print(f"\n    └─ 📋 Position #{position.ticket} | {position.symbol}")
                    print(f"       ⚠️  Cannot get symbol info. Skipping.")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1
                    continue

                # Determine position direction
                is_buy = position.type == mt5.POSITION_TYPE_BUY
                
                print(f"\n    └─ 📋 Position #{position.ticket} | {position_type_name} | {position.symbol}")
                
                # Calculate current risk (from entry to original SL)
                if is_buy:
                    risk_distance = position.price_open - position.sl
                else:
                    risk_distance = position.sl - position.price_open
                
                risk_points = abs(risk_distance) / symbol_info.point
                
                # Calculate point value
                tick_value = symbol_info.trade_tick_value
                tick_size = symbol_info.trade_tick_size
                
                if tick_value > 0 and tick_size > 0:
                    point_value = tick_value / tick_size * symbol_info.point
                    risk_usd = round(risk_points * point_value * position.volume, 2)
                else:
                    # Fallback: calculate risk using profit calculator
                    calc_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                    sl_profit = mt5.order_calc_profit(calc_type, position.symbol, position.volume, 
                                                      position.price_open, position.sl)
                    if sl_profit is not None:
                        risk_usd = round(abs(sl_profit), 2)
                    else:
                        print(f"       ⚠️  Cannot calculate risk. Skipping.")
                        investor_positions_skipped += 1
                        stats["positions_skipped"] += 1
                        continue

                # Calculate current profit in R multiples
                current_profit_usd = position.profit
                
                if risk_usd > 0:
                    current_r_multiple = current_profit_usd / risk_usd
                else:
                    print(f"       ⚠️  Invalid risk value. Skipping.")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1
                    continue

                print(f"       • Risk: ${risk_usd:.2f} | Current P/L: ${current_profit_usd:.2f} ({current_r_multiple:.2f}R)")

                # Skip if position is not in profit
                if current_profit_usd <= 0:
                    print(f"       ⏭️  Position not in profit. Skipping.")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1
                    continue

                # Find applicable breakeven rules
                applicable_rules = []
                for rule in breakeven_config:
                    reward_threshold = rule["reward"]
                    if current_r_multiple >= reward_threshold:
                        applicable_rules.append(rule)
                
                if not applicable_rules:
                    print(f"       ⏭️  No breakeven threshold reached (current: {current_r_multiple:.2f}R)")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1
                    continue

                # Get the highest applicable rule (last in sorted list)
                highest_rule = applicable_rules[-1]
                target_reward = highest_rule["breakeven_at_reward"]
                
                print(f"       🎯 Reached {highest_rule['reward']}R threshold")
                print(f"       Target SL position: {target_reward}R")

                # Calculate target SL price based on target reward
                if target_reward >= 0:
                    # For positive target reward, we want SL at entry + (risk_distance * target_reward)
                    # But direction matters
                    if is_buy:
                        # For BUY: entry + (risk * target_reward)
                        target_sl_price = position.price_open + (risk_distance * target_reward)
                    else:
                        # For SELL: entry - (risk * target_reward)
                        target_sl_price = position.price_open - (risk_distance * target_reward)
                    
                    # Round to symbol digits
                    digits = symbol_info.digits
                    target_sl_price = round(target_sl_price, digits)
                    
                    print(f"       Current SL: {position.sl:.{digits}f}")
                    print(f"       Target SL:  {target_sl_price:.{digits}f} ({target_reward}R)")
                    
                    # Check if SL needs adjustment
                    current_sl_distance = abs(position.sl - position.price_open) if position.sl != 0 else 0
                    target_sl_distance = abs(target_sl_price - position.price_open)
                    
                    # Calculate threshold (10% of target distance or 2 pips)
                    pip_threshold = max(target_sl_distance * 0.1, symbol_info.point * 20)
                    
                    should_adjust = False
                    
                    if position.sl == 0:
                        print(f"       📝 No SL currently set")
                        should_adjust = True
                    elif abs(current_sl_distance - target_sl_distance) > pip_threshold:
                        print(f"       📐 SL needs adjustment")
                        should_adjust = True
                    else:
                        # Check if we're moving in the right direction (should only move SL towards profit)
                        if is_buy and target_sl_price > position.sl:
                            print(f"       ✅ SL already at or beyond target")
                            investor_positions_skipped += 1
                            stats["positions_skipped"] += 1
                            continue
                        elif not is_buy and target_sl_price < position.sl:
                            print(f"       ✅ SL already at or beyond target")
                            investor_positions_skipped += 1
                            stats["positions_skipped"] += 1
                            continue
                        else:
                            should_adjust = True
                    
                    if should_adjust:
                        # Ensure we're only moving SL in the profit direction
                        if is_buy and target_sl_price <= position.sl:
                            print(f"       ⚠️  Target SL would not improve position. Skipping.")
                            investor_positions_skipped += 1
                            stats["positions_skipped"] += 1
                            continue
                        elif not is_buy and target_sl_price >= position.sl:
                            print(f"       ⚠️  Target SL would not improve position. Skipping.")
                            investor_positions_skipped += 1
                            stats["positions_skipped"] += 1
                            continue
                        
                        # Prepare modification request
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "sl": target_sl_price,
                            "tp": position.tp,  # Keep existing TP
                        }
                        
                        # Send modification
                        result = mt5.order_send(modify_request)
                        
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            investor_positions_adjusted += 1
                            investor_breakeven_events += 1
                            stats["positions_adjusted"] += 1
                            stats["breakeven_events"] += 1
                            print(f"       ✅ SL adjusted successfully to {target_sl_price:.{digits}f} ({target_reward}R)")
                        else:
                            investor_positions_error += 1
                            stats["positions_error"] += 1
                            error_msg = result.comment if result else f"Error code: {result.retcode if result else 'Unknown'}"
                            print(f"       ❌ Modification failed: {error_msg}")
                else:
                    print(f"       ⚠️  Invalid target reward: {target_reward}")
                    investor_positions_skipped += 1
                    stats["positions_skipped"] += 1

        # --- INVESTOR SUMMARY ---
        if investor_positions_checked > 0:
            print(f"\n  └─ 📊 Breakeven Results for {user_brokerid}:")
            print(f"       • Positions checked: {investor_positions_checked}")
            print(f"       • Positions adjusted: {investor_positions_adjusted}")
            print(f"       • Breakeven events: {investor_breakeven_events}")
            print(f"       • Positions skipped: {investor_positions_skipped}")
            if investor_positions_error > 0:
                print(f"       • Errors: {investor_positions_error}")
            else:
                print(f"       ✅ All breakeven checks completed successfully")
            stats["processing_success"] = True
        else:
            print(f"  └─ 🔘 No open positions found.")

    # --- FINAL SUMMARY ---
    print(f"\n{'='*10} 📊 DYNAMIC BREAKEVEN SUMMARY {'='*10}")
    print(f"   Investor ID: {stats['investor_id']}")
    print(f"   Positions checked: {stats['positions_checked']}")
    print(f"   Positions adjusted: {stats['positions_adjusted']}")
    print(f"   Breakeven events: {stats['breakeven_events']}")
    print(f"   Positions skipped: {stats['positions_skipped']}")
    print(f"   Errors: {stats['positions_error']}")
    
    if stats['positions_checked'] > 0:
        adjustment_rate = (stats['positions_adjusted'] / stats['positions_checked']) * 100
        print(f"   Adjustment rate: {adjustment_rate:.1f}%")
    
    print(f"\n{'='*10} 🏁 DYNAMIC BREAKEVEN MONITORING COMPLETE {'='*10}\n")
    return stats

# real accounts 
def process_single_investor(inv_folder):
    """
    WORKER FUNCTION: Handles the entire pipeline for ONE investor.
    Each process calls this independently.
    """
    inv_id = inv_folder.name
    # Results dictionary to pass back to the main process for statistics
    account_stats = {"inv_id": inv_id, "success": False, "details": {}}
    
    # 1. Get broker config
    broker_cfg = usersdictionary.get(inv_id)
    if not broker_cfg:
        print(f" [{inv_id}] ❌ No broker config found")
        return account_stats

    # --- ISOLATION START ---
    # Give a small random offset to avoid exact simultaneous initialization hits on the OS
    time.sleep(random.uniform(0.1, 2.0)) 
    
    login_id = int(broker_cfg['LOGIN_ID'])
    mt5_path = broker_cfg["TERMINAL_PATH"]

    try:
        # Initialize and Login (Local to this process)
        if not mt5.initialize(path=mt5_path, timeout=180000):
            print(f" [{inv_id}] ❌ MT5 Init failed at {mt5_path}")
            return account_stats

        if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
            print(f" [{inv_id}] ❌ Login failed")
            mt5.shutdown()
            return account_stats

        # --- RUN ALL SEQUENTIAL STEPS FOR THIS BROKER ---
        # Note: All your functions (deduplicate_orders, etc.) must accept inv_id
        accountmanagement_manager(inv_id=inv_id)
        deduplicate_orders(inv_id=inv_id)
        filter_unauthorized_symbols(inv_id=inv_id)
        filter_unauthorized_timeframes(inv_id=inv_id)
        backup_limit_orders(inv_id=inv_id)
        populate_orders_missing_fields(inv_id=inv_id)
        activate_usd_based_risk_on_empty_pricelevels(inv_id=inv_id)
        enforce_investors_risk(inv_id=inv_id)
        calculate_investor_symbols_orders(inv_id=inv_id)
        padding_tight_usd_risk(inv_id=inv_id)
        live_usd_risk_and_scaling(inv_id=inv_id)
        apply_default_prices(inv_id=inv_id)
        place_usd_orders(inv_id=inv_id)
        orders_risk_correction(inv_id=inv_id)
        check_pending_orders_risk(inv_id=inv_id)
        history_closed_orders_removal_in_pendingorders(inv_id=inv_id)
        apply_dynamic_breakeven(inv_id=inv_id)

        mt5.shutdown()
        account_stats["success"] = True
        print(f" [{inv_id}] ✅ Processing complete")
        
    except Exception as e:
        print(f" [{inv_id}] ❌ Critical Error: {e}")
        mt5.shutdown()
    
    return account_stats

def place_orders_parallel():
    """
    ORCHESTRATOR: Spawns multiple processes to handle investors in parallel.
    """
    print(f"\n{'='*10} 🚀 STARTING MULTIPROCESSING ENGINE {'='*10}")
    
    inv_base_path = Path(INV_PATH)
    investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    # Create a pool based on the number of accounts (or CPU cores)
    # This will run 'process_single_investor' for all folders at the same time
    with mp.Pool(processes=len(investor_folders)) as pool:
        results = pool.map(process_single_investor, investor_folders)

    # Summary logic
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*10} PARALLEL PROCESSING COMPLETE {'='*10}")
    print(f" Total: {len(results)} | Successful: {successful} | Failed: {len(results)-successful}")
    return successful > 0
#--


# demo
def process_demo_single_investor(inv_folder):
    """
    WORKER FUNCTION: Handles the entire pipeline for ONE investor (DEMO VERSION).
    Uses demo-style initialization with account info check before login.
    """
    inv_id = inv_folder.name
    
    account_stats = {
        "inv_id": inv_id, 
        "success": False, 
        "price_collection_stats": {},
        "candle_fetch_stats": {},
        "crosser_analysis_stats": {},
        "trapped_analysis_stats": {},
        "liquidator_analysis_stats": {},
        "ranging_analysis_stats": {},
        "order_placement_stats": {},
        "risk_correction_stats": {},
        "risk_audit_stats": {},
        "symbols_filtered": 0,
        "orders_filtered": 0,
        "symbols_processed": 0,
        "symbols_successful": 0,
        "orders_placed": 0,
        "counter_orders_placed": 0,
        "total_active_orders": 0,
        "orders_adjusted": 0,
        "orders_removed": 0,
        "current_candle_forming": False,
        "bid_wins": 0,
        "ask_wins": 0,
        "trapped_candles_found": 0,
        "symbols_with_trapped": 0,
        "symbols_with_liquidator": 0,
        "liquidator_candles_found": 0,
        "bullish_liquidators": 0,
        "bearish_liquidators": 0,
        "symbols_ranging": 0,
        "avg_ranging_cycles": 0
    }
    
    # 1. Get broker config
    broker_cfg = usersdictionary.get(inv_id)
    if not broker_cfg:
        print(f" [{inv_id}] ❌ No broker config found")
        return account_stats

    # --- ISOLATION START ---
    # Give a small random offset to avoid exact simultaneous initialization hits on the OS
    import random
    import time
    time.sleep(random.uniform(0.1, 2.0)) 
    
    login_id = int(broker_cfg['LOGIN_ID'])
    mt5_path = broker_cfg["TERMINAL_PATH"]

    try:
        # DEMO STYLE INITIALIZATION: Initialize first, then check account info
        if not mt5.initialize(path=mt5_path, timeout=180000):
            print(f" [{inv_id}] ❌ MT5 Init failed at {mt5_path}")
            return account_stats

        # DEMO STYLE LOGIN: Check if already logged in correctly
        acc = mt5.account_info()
        if acc is None or acc.login != login_id:
            if not mt5.login(login_id, password=broker_cfg["PASSWORD"], server=broker_cfg["SERVER"]):
                print(f" [{inv_id}] ❌ Login failed")
                mt5.shutdown()
                return account_stats

        # --- RUN ALL SEQUENTIAL STEPS FOR THIS BROKER (SAME AS REAL VERSION) ---
        accountmanagement_manager(inv_id=inv_id)
        deduplicate_orders(inv_id=inv_id)
        filter_unauthorized_symbols(inv_id=inv_id)
        filter_unauthorized_timeframes(inv_id=inv_id)
        backup_limit_orders(inv_id=inv_id)
        populate_orders_missing_fields(inv_id=inv_id)
        activate_usd_based_risk_on_empty_pricelevels(inv_id=inv_id)
        enforce_investors_risk(inv_id=inv_id)
        calculate_investor_symbols_orders(inv_id=inv_id)
        padding_tight_usd_risk(inv_id=inv_id)
        live_usd_risk_and_scaling(inv_id=inv_id)
        apply_default_prices(inv_id=inv_id)
        place_usd_orders(inv_id=inv_id)
        orders_risk_correction(inv_id=inv_id)
        check_pending_orders_risk(inv_id=inv_id)
        history_closed_orders_removal_in_pendingorders(inv_id=inv_id)
        apply_dynamic_breakeven(inv_id=inv_id)
        

        # Update demo-specific stats if needed
        # (You might want to collect stats from these functions)
        
        mt5.shutdown()
        account_stats["success"] = True
        print(f" [{inv_id}] ✅ Demo processing complete")
        
    except Exception as e:
        print(f" [{inv_id}] ❌ Critical Error: {e}")
        try:
            mt5.shutdown()
        except:
            pass
    
    return account_stats

def place_demo_orders_parallel():
    """
    ORCHESTRATOR: Spawns multiple processes to handle investors in parallel.
    """
    print(f"\n{'='*10} 🚀 STARTING MULTIPROCESSING ENGINE {'='*10}")
    
    inv_base_path = Path(INV_PATH)
    investor_folders = [f for f in inv_base_path.iterdir() if f.is_dir()]
    
    if not investor_folders:
        print(" └─ 🔘 No investor directories found.")
        return False

    # Create a pool based on the number of accounts (or CPU cores)
    # This will run 'process_single_investor' for all folders at the same time
    with mp.Pool(processes=len(investor_folders)) as pool:
        results = pool.map(process_demo_single_investor, investor_folders)

    # Summary logic
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*10} PARALLEL PROCESSING COMPLETE {'='*10}")
    print(f" Total: {len(results)} | Successful: {successful} | Failed: {len(results)-successful}")
    return successful > 0
#---

if __name__ == "__main__":
    place_demo_orders_parallel()



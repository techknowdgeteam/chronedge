import connectwithinfinitydb as db
import json
import os
import shutil
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for cross-platform terminal colors
init()

OUTPUT_FILE_PATH = r"C:\xampp\htdocs\chronedge\usersdictionary.json"
MT5_TEMPLATE_SOURCE_DIR = r"C:\xampp\htdocs\chronedge\mt5\MetaTrader 5"
BROKERS_OUTPUT_FILE_PATH = r"C:\xampp\htdocs\chronedge\brokersdictionary.json"

# --- HELPER FUNCTIONS ---

def log_and_print(message, level="INFO"):
    """Helper function to print formatted messages with color coding and spacing."""
    indent = "    "
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "INFO": Fore.CYAN,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED,
        "TITLE": Fore.MAGENTA,
        "WARNING": Fore.CYAN,
    }
    color = level_colors.get(level, Fore.WHITE)
    formatted_message = f"[ {timestamp} ] │ {level:7} │ {indent}{message}"
    print(f"{color}{formatted_message}{Style.RESET_ALL}")

def safe_float(value):
    """
    Safely converts a value (which might be None or the string 'None') to a float, 
    defaulting to 0.00.
    """
    if value is None:
        return 0.00
    
    value_str = str(value).strip().lower()
    
    if value_str in ('none', ''):
        return 0.00
    
    try:
        return float(value)
    except ValueError:
        log_and_print(f"WARNING: Non-numeric value encountered for float conversion: '{value}'. Defaulting to 0.00.", "WARNING")
        return 0.00

def update_history_string(current_history, new_value):
    """Appends a new value (which can be a raw string like 'None' or a number) to a comma-separated history string."""
    new_value_str = str(new_value).strip()
    
    # Check if the new value is insignificant or already captured
    if not new_value_str or new_value_str.lower() in ('null', ''):
        return str(current_history).strip()

    # Treat 'None' (from DB) as an empty string for concatenation purposes if it's the only entry
    current_history_str = str(current_history).strip()
    if current_history_str.lower() in ('none', 'null', ''):
        current_history_str = ""
    
    if not current_history_str:
        return new_value_str
    
    # Check if the new value is already the last entry
    last_entry = current_history_str.split(',')[-1].strip()
    
    # Use string comparison for raw values (like 'None')
    if last_entry == new_value_str:
        return current_history_str
    
    # Try numeric comparison if both values are convertible (for 0.0 vs 0.00 consistency)
    try:
        if safe_float(last_entry) == safe_float(new_value_str):
             return current_history_str
    except ValueError:
        # Ignore if conversion fails (e.g., comparing 'None' to '1.0')
        pass
    
    return f"{current_history_str},{new_value_str}"


def fetch_insiders_server_rows():
    """
    Connects to the database, fetches all rows, filters out entries with 'None' broker,
    and writes the processed data to the specified JSON file based on verification status.
    
    FIX: Ensures the in-memory data updated during the DB RESET in Pass 1 is correctly 
    transferred to the final JSON output, overriding the potentially stale values 
    from the Pass 2 database fetch.
    
    MODIFICATION: Sets BASE_FOLDER to the parent directory (e.g., ...\chart\deriv 3) 
    instead of the subfolder (...deriv3symbols), and ensures this applies to all users.
    """
    # Configuration Section
    INSIDERS_TABLE = "insiders_server"
    
    # --- Default Configuration to Append ---
    DEFAULT_CONFIG = {
        "ACCOUNT": "real",
        "STRATEGY": "hightolow",
        "SCALE": "consistency",
        "RISKREWARD": 3,
        "SYMBOLS": "all",
        "MARTINGALE_MARKETS": "neth25, usdjpy",
        "ACCOUNT_VERIFICATION": "waiting",
        # NEW FIELDS for database interaction and JSON structure
        "BROKER_BALANCE": 0.00,
        "PROFITANDLOSS": 0.00,
        "RESET_EXECUTION_DATE_AND_BROKER_BALANCE": "none", # JSON COMMAND FIELD
        # New history fields (initialized as empty strings for consistency)
        "BROKER_BALANCE_HISTORY": "",
        "EXECUTION_DATES_HISTORY": "",
        "PROFITANDLOSS_HISTORY": "",
        "TRADES": "",
        "LOYALTIES": "low", # <--- Default value for loyalties
    }
    
    # The dictionary that will hold the state updated during Pass 1 (JSON-to-DB)
    # and then used as the base for the final Pass 2 (DB-to-JSON).
    users_dictionary = {}
    existing_users_dictionary = {}
    
    skipped_count = 0
    updated_db_declined_count = 0
    updated_db_approved_count = 0
    skipped_verified_count = 0
    updated_db_pending_count = 0
    updated_db_reset_count = 0
    updated_db_pnl_count = 0
    updated_db_balance_count = 0

    try:
        print("\n")
        log_and_print(f"===== Fetching ALL ROWS for Table: {INSIDERS_TABLE} =====", "TITLE")
        
        # --- LOGIC: Load existing JSON data for status check ---
        if os.path.exists(OUTPUT_FILE_PATH):
            try:
                with open(OUTPUT_FILE_PATH, 'r') as f:
                    existing_users_dictionary = json.load(f)
                
                # Pre-populate users_dictionary with existing JSON data 
                # This is the in-memory state we will modify.
                for key, data in existing_users_dictionary.items():
                    users_dictionary[key] = data.copy()
            except (IOError, json.JSONDecodeError):
                log_and_print("WARNING: Existing JSON file found but could not be loaded/parsed. Starting fresh dictionary.", "WARNING")
                pass
        
        # 1. Define the SQL query to fetch all data, INCLUDING the new history and loyalties columns.
        select_all_query = (
            f"SELECT id, broker, login, password, server, execution_start_date, application_status, broker_balance, profitandloss, "
            f"broker_balance_history, execution_dates_history, profitandlosshistory, trades, loyalties " # loyalties ADDED HERE
            f"FROM {INSIDERS_TABLE}"
        )
        log_and_print(f"Sending query: {select_all_query}", "INFO")
        
        # 2. Execute the query (Initial DB state fetch for Pass 1)
        result = db.execute_query(select_all_query)
        
        print("\n")
        log_and_print("--- Database Query Results ---", "TITLE")

        # Convert initial DB rows to map for easy access during Pass 1 updates
        db_rows_map_initial = {}
        if result.get('status') == 'success' and isinstance(result.get('results'), list):
            rows_initial = result['results']
            db_rows_map_initial = {str(row['id']): row for row in rows_initial}
        
        
        if db_rows_map_initial:
            log_and_print(f"Successfully fetched {len(rows_initial)} rows from '{INSIDERS_TABLE}'.", "SUCCESS")
            
            # --- JSON-to-DB UPDATE LOGIC (Pass 1) ---
            log_and_print("--- Executing JSON-to-DB Update Passes ---", "TITLE")
            
            for json_key, json_data in existing_users_dictionary.items():
                
                # Reverse lookup ID from JSON key (broker+ID)
                id_number = next((row['id'] for row in rows_initial if f"{str(row['broker']).lower().replace(' ', '')}{row['id']}" == json_key), None)
                if not id_number:
                    continue
                    
                id_number_str = str(id_number)
                db_row = db_rows_map_initial.get(id_number_str)
                
                if not db_row:
                    continue
                    
                current_verification_status = str(json_data.get("ACCOUNT_VERIFICATION", DEFAULT_CONFIG["ACCOUNT_VERIFICATION"])).lower().strip()
                reset_flag = str(json_data.get("RESET_EXECUTION_DATE_AND_BROKER_BALANCE", DEFAULT_CONFIG["RESET_EXECUTION_DATE_AND_BROKER_BALANCE"])).lower().strip()
                
                json_broker_balance_float = safe_float(json_data.get("BROKER_BALANCE", DEFAULT_CONFIG["BROKER_BALANCE"]))
                json_profitandloss_raw = json_data.get("PROFITANDLOSS")
                if json_profitandloss_raw is None:
                    json_profitandloss_raw = json_data.get("PROFIT_AND_LOSS", DEFAULT_CONFIG["PROFITANDLOSS"])
                json_profitandloss_float = safe_float(json_profitandloss_raw)
                
                # Get current DB values for comparison/history
                db_balance_float = safe_float(db_row.get('broker_balance'))
                db_pnl_float = safe_float(db_row.get('profitandloss'))
                db_status = str(db_row.get('application_status', '')).lower().strip()

                # --- 1. RESET LOGIC: 'reset' flag ---
                if reset_flag == "reset":
                    db_balance_history = str(db_row.get('broker_balance_history', ''))
                    db_dates_history = str(db_row.get('execution_dates_history', ''))
                    db_pnl_history = str(db_row.get('profitandlosshistory', ''))
                    
                    raw_db_balance = str(db_row.get('broker_balance', '0.00'))
                    raw_db_execution_date = str(db_row.get('execution_start_date', 'None'))
                    raw_db_pnl = str(db_row.get('profitandloss', '0.00'))
                    
                    new_balance_history = update_history_string(db_balance_history, raw_db_balance)
                    new_dates_history = update_history_string(db_dates_history, raw_db_execution_date)
                    new_pnl_history = update_history_string(db_pnl_history, raw_db_pnl)

                    update_query = (
                        f"UPDATE {INSIDERS_TABLE} SET "
                        f"execution_start_date = NULL, "
                        f"broker_balance = {json_broker_balance_float}, "
                        f"broker_balance_history = '{new_balance_history}', "
                        f"execution_dates_history = '{new_dates_history}', "
                        f"profitandlosshistory = '{new_pnl_history}' "
                        f"WHERE id = {id_number_str}"
                    )
                    update_result = db.execute_query(update_query)

                    if update_result.get('status') == 'success':
                        log_and_print(f"User {json_key}: DB RESET performed. Balance: {db_balance_float} -> {json_broker_balance_float}. History updated.", "WARNING")
                        updated_db_reset_count += 1
                        
                        # --- CRITICAL FIX: Update IN-MEMORY data with new values ---
                        # This ensures the new history and null/cleared values are present 
                        # in the `users_dictionary` for the final JSON write.
                        if json_key in users_dictionary:
                            users_dictionary[json_key]["BROKER_BALANCE_HISTORY"] = new_balance_history
                            users_dictionary[json_key]["EXECUTION_DATES_HISTORY"] = new_dates_history
                            users_dictionary[json_key]["PROFITANDLOSS_HISTORY"] = new_pnl_history
                            users_dictionary[json_key]["EXECUTION_START_DATE"] = None # Set to None (null in DB)
                            users_dictionary[json_key]["BROKER_BALANCE"] = json_broker_balance_float # Set new balance
                            
                    else:
                        log_and_print(f"ERROR: Failed DB RESET for ID {id_number_str}. Msg: {update_result.get('message')}", "ERROR")

                # --- 2. ALWAYS UPDATE PROFITANDLOSS LOGIC (JSON to DB) ---
                if abs(json_profitandloss_float - db_pnl_float) > 0.0001:
                    pnl_update_query = (
                        f"UPDATE {INSIDERS_TABLE} SET "
                        f"profitandloss = {json_profitandloss_float} "
                        f"WHERE id = {id_number_str}"
                    )
                    pnl_update_result = db.execute_query(pnl_update_query)

                    if pnl_update_result.get('status') == 'success':
                        updated_db_pnl_count += 1
                        # Update in-memory data for consistency
                        if json_key in users_dictionary:
                            users_dictionary[json_key]["PROFITANDLOSS"] = json_profitandloss_float
                    else:
                        log_and_print(f"ERROR: Failed to update profitandloss for ID {id_number_str}. Msg: {pnl_update_result.get('message')}", "ERROR")

                # --- 3. 'none' flag BALANCE UPDATE LOGIC (JSON to DB) ---
                if reset_flag == "none":
                    if abs(json_broker_balance_float - db_balance_float) > 0.0001:
                        balance_update_query = (
                            f"UPDATE {INSIDERS_TABLE} SET "
                            f"broker_balance = {json_broker_balance_float} "
                            f"WHERE id = {id_number_str}"
                        )
                        balance_update_result = db.execute_query(balance_update_query)

                        if balance_update_result.get('status') == 'success':
                            log_and_print(f"User {json_key}: 'none' flag update. DB Balance updated to {json_broker_balance_float}.", "INFO")
                            updated_db_balance_count += 1
                            # Update in-memory data for consistency
                            if json_key in users_dictionary:
                                users_dictionary[json_key]["BROKER_BALANCE"] = json_broker_balance_float
                        else:
                            log_and_print(f"ERROR: Failed to update broker_balance for ID {id_number_str} ('none' flag). Msg: {balance_update_result.get('message')}", "ERROR")
                
                # --- 4. ACCOUNT_VERIFICATION Status Update to DB ---
                if current_verification_status == "verified" and db_status != 'approved':
                    update_query = f"UPDATE {INSIDERS_TABLE} SET application_status = 'approved' WHERE id = {id_number_str}"
                    update_result = db.execute_query(update_query)
                    if update_result.get('status') == 'success':
                        updated_db_approved_count += 1
                        # Update in-memory data
                        if json_key in users_dictionary:
                            users_dictionary[json_key]["DB_APPLICATION_STATUS"] = 'approved'
                    else:
                        log_and_print(f"ERROR: Failed to update status to 'approved' for ID {id_number_str}. Msg: {update_result.get('message')}", "ERROR")
                        
                elif current_verification_status == "invalidcredentials" and db_status != 'declined':
                    update_query = f"UPDATE {INSIDERS_TABLE} SET application_status = 'declined' WHERE id = {id_number_str}"
                    update_result = db.execute_query(update_query)
                    if update_result.get('status') == 'success':
                        updated_db_declined_count += 1
                        # Update in-memory data
                        if json_key in users_dictionary:
                            users_dictionary[json_key]["DB_APPLICATION_STATUS"] = 'declined'
                    else:
                        log_and_print(f"ERROR: Failed to update status to 'declined' for ID {id_number_str}. Msg: {update_result.get('message')}", "ERROR")
                        
            log_and_print("--- Finished JSON-to-DB Update Passes ---", "TITLE")
            
            # --- DB-to-JSON Processing and File/Folder Creation (Pass 2) ---
            log_and_print("--- Executing DB-to-JSON and File/Folder Creation ---", "TITLE")

            # Re-run query after updates to get the most current DB state 
            result = db.execute_query(select_all_query)
            if result.get('status') == 'success' and isinstance(result.get('results'), list):
                rows_current = result['results']
            else:
                log_and_print("WARNING: Second DB fetch failed. Using original data for JSON processing.", "WARNING")
                rows_current = rows_initial # Fallback to initial rows
                
            
            new_users_dictionary = {} # Use a new dictionary for the output
            
            for i, row in enumerate(rows_current, 1):
                broker_value = row.get('broker')
                
                # 🔔 FILTERING LOGIC: Skip rows if 'broker' is None, or the string 'None'
                if broker_value is None or str(broker_value).strip().lower() == 'none':
                    skipped_count += 1
                    continue
                    
                # Get necessary values (as strings initially)
                broker = str(broker_value).lower().replace(" ", "")
                login = str(row.get('login', 'None'))
                password = str(row.get('password', 'None'))
                server = str(row.get('server', 'None'))
                id_number = str(row.get('id', '0'))
                db_execution_start_date = str(row.get('execution_start_date', 'None'))
                db_broker_balance = str(row.get('broker_balance', '0.00'))
                db_profitandloss = str(row.get('profitandloss', '0.00'))
                db_broker_balance_history = str(row.get('broker_balance_history', ''))
                db_execution_dates_history = str(row.get('execution_dates_history', ''))
                db_pnl_history = str(row.get('profitandlosshistory', ''))
                db_trades = str(row.get('trades', ''))
                db_application_status = str(row.get('application_status', 'None')).lower().strip()
                db_loyalties = str(row.get('loyalties', DEFAULT_CONFIG["LOYALTIES"]))
                
                json_key = f"{broker}{id_number}"
                in_memory_user_data = users_dictionary.get(json_key) # Get the updated data from Pass 1
                
                # *** MODIFIED PATH LOGIC START ***
                mt5_terminal_dir = rf"C:\xampp\htdocs\chronedge\mt5\MetaTrader 5 {broker} {id_number}"
                terminal_path = os.path.join(mt5_terminal_dir, "terminal64.exe")
                
                # Set BASE_FOLDER to the PARENT folder directory path (e.g., C:\...\chart\deriv 3)
                base_folder_dir = rf"C:\xampp\htdocs\chronedge\chart\{broker} {id_number}" 
                base_folder_path = base_folder_dir
                # *** MODIFIED PATH LOGIC END ***

                # --- NEW LOGIC: Check Execution Dates History for "justjoined" Loyalty ---
                if not db_execution_dates_history or db_execution_dates_history.lower() in ('none', 'null'):
                    db_loyalties = "justjoined"
                    log_and_print(f"User {json_key}: Loyalty set to 'justjoined' because execution_dates_history is empty or 'None'.", "INFO")


                # --- JSON OVERWRITE PREVENTION FOR VERIFIED USERS ---
                if in_memory_user_data and in_memory_user_data.get("ACCOUNT_VERIFICATION", "").lower().strip() == "verified":
                    
                    # Start with the in-memory data (which holds the current config and any Pass 1 reset updates)
                    user_data = in_memory_user_data.copy()
                    
                    # Update DB-sourced fields with the latest DB values (from the second fetch)
                    # ONLY if the 'reset' flag was not active for that field in Pass 1.
                    if user_data.get("RESET_EXECUTION_DATE_AND_BROKER_BALANCE", "none") == "none":
                        user_data["BROKER_BALANCE"] = safe_float(db_broker_balance)
                        user_data["EXECUTION_START_DATE"] = db_execution_start_date if db_execution_start_date != 'None' else None
                        
                        # Use the DB history, but if the history was updated in Pass 1, in_memory already has the correct value.
                        if not user_data["BROKER_BALANCE_HISTORY"]:
                            user_data["BROKER_BALANCE_HISTORY"] = db_broker_balance_history
                        if not user_data["EXECUTION_DATES_HISTORY"]:
                            user_data["EXECUTION_DATES_HISTORY"] = db_execution_dates_history
                        if not user_data["PROFITANDLOSS_HISTORY"]:
                            user_data["PROFITANDLOSS_HISTORY"] = db_pnl_history
                        
                    # PNL and status are always updated to the latest DB value (which was updated in Pass 1)
                    user_data["PROFITANDLOSS"] = safe_float(db_profitandloss)
                    user_data["TRADES"] = db_trades
                    user_data["DB_APPLICATION_STATUS"] = db_application_status
                    user_data["LOYALTIES"] = db_loyalties # Use the potentially updated 'justjoined' value

                    # *** CRITICAL FIX: Ensure BASE_FOLDER is updated for existing/verified users ***
                    user_data["BASE_FOLDER"] = base_folder_path
                    # ******************************************************************************
                    
                    new_users_dictionary[json_key] = user_data
                    skipped_verified_count += 1
                    continue
                    
                # --- DB APPLICATION STATUS Normalization (For new/non-verified/non-declined users) ---
                if db_application_status not in ["pending", "approved", "declined"]:
                    db_application_status = 'pending'

                # Console print of the row (only first 5 valid rows printed)
                row_str = ", ".join([f"{k}: {repr(v)}" for k, v in row.items()])
                
                valid_index = i - skipped_count - skipped_verified_count
                if valid_index <= 5 and valid_index >= 1:
                    print(f"{Fore.GREEN}Row {i}: DB App Status '{db_application_status}': {row_str}{Style.RESET_ALL}")
                    
                # --- Check and Copy MT5 Template (unchanged) ---
                if not os.path.isdir(mt5_terminal_dir):
                    log_and_print(f"MT5 folder for user {id_number} does not exist. Copying template...", "WARNING")
                    try:
                        shutil.copytree(MT5_TEMPLATE_SOURCE_DIR, mt5_terminal_dir)
                        log_and_print(f"Successfully created MT5 folder: {mt5_terminal_dir}", "SUCCESS")
                    except FileExistsError:
                        pass
                    except Exception as copy_error:
                        log_and_print(f"ERROR: Failed to copy MT5 template for user {id_number}: {copy_error}", "ERROR")
                else:
                    pass
                    
                # --- Ensure Chart Base Folder Exists (unchanged, but uses the modified base_folder_dir) ---
                try:
                    os.makedirs(base_folder_dir, exist_ok=True)
                except Exception:
                    pass


                # Construct the value dictionary from the DB row (Second fetch - DB is the source of truth for current state)
                user_data = {
                    "TERMINAL_PATH": terminal_path,
                    "LOGIN_ID": login,
                    "PASSWORD": password,
                    "SERVER": server,
                    "BASE_FOLDER": base_folder_path, # Now set to the parent directory, correctly
                    "EXECUTION_START_DATE": db_execution_start_date if db_execution_start_date != 'None' else None,
                    "BROKER_BALANCE": safe_float(db_broker_balance),
                    "PROFITANDLOSS": safe_float(db_profitandloss),
                    "DB_APPLICATION_STATUS": db_application_status,
                    "BROKER_BALANCE_HISTORY": db_broker_balance_history,
                    "EXECUTION_DATES_HISTORY": db_execution_dates_history,
                    "PROFITANDLOSS_HISTORY": db_pnl_history,
                    "TRADES": db_trades,
                    "LOYALTIES": db_loyalties # Use the potentially updated 'justjoined' value
                }
                
                user_config = DEFAULT_CONFIG.copy()
                
                # --- CRITICAL FIX: Merge with in_memory_user_data for all users ---
                if in_memory_user_data:
                    # 1. Merge configuration keys (strategy, riskreward, etc.)
                    for key in DEFAULT_CONFIG.keys():
                        if key not in user_data: # Don't overwrite the core DB fields
                            user_config[key] = in_memory_user_data.get(key, DEFAULT_CONFIG[key])
                    
                    # 2. Overwrite DB data with the UPDATED in-memory history/balance if a reset occurred.
                    if in_memory_user_data.get("RESET_EXECUTION_DATE_AND_BROKER_BALANCE", "none") == "reset":
                        user_data["BROKER_BALANCE_HISTORY"] = in_memory_user_data["BROKER_BALANCE_HISTORY"]
                        user_data["EXECUTION_DATES_HISTORY"] = in_memory_user_data["EXECUTION_DATES_HISTORY"]
                        user_data["PROFITANDLOSS_HISTORY"] = in_memory_user_data["PROFITANDLOSS_HISTORY"]
                        user_data["EXECUTION_START_DATE"] = in_memory_user_data["EXECUTION_START_DATE"]
                        user_data["BROKER_BALANCE"] = in_memory_user_data["BROKER_BALANCE"]
                        
                    # 3. Always ensure the JSON flag is preserved from in-memory data
                    user_config["RESET_EXECUTION_DATE_AND_BROKER_BALANCE"] = in_memory_user_data.get("RESET_EXECUTION_DATE_AND_BROKER_BALANCE", "none")

                # Append the configuration 
                user_data.update(user_config)
                
                # Add to the main dictionary 
                new_users_dictionary[json_key] = user_data
            
            # Update the main dictionary reference for writing
            users_dictionary = new_users_dictionary
            
            # --- Reporting and Finalizing ---
            log_and_print("--- Final Processing Summary ---", "TITLE")
            if skipped_count > 0:
                log_and_print(f"Skipped {skipped_count} row(s) due to missing 'broker' value.", "INFO")
            if skipped_verified_count > 0:
                log_and_print(f"Preserved {skipped_verified_count} user(s) from existing JSON (status 'verified').", "INFO")
            if updated_db_approved_count > 0:
                log_and_print(f"Approved {updated_db_approved_count} user(s) in DB for 'verified' JSON status.", "SUCCESS")
            if updated_db_declined_count > 0:
                log_and_print(f"Declined {updated_db_declined_count} user(s) in DB for 'invalidcredentials' JSON status.", "WARNING")
            if updated_db_pending_count > 0:
                log_and_print(f"Set {updated_db_pending_count} user(s) to 'pending' in DB for normalization.", "INFO")
            if updated_db_reset_count > 0:
                log_and_print(f"Executed {updated_db_reset_count} DB RESET(s) with history update (Balance, Dates, PNL). JSON flags not reset.", "WARNING")
            if updated_db_balance_count > 0:
                log_and_print(f"Updated {updated_db_balance_count} DB broker_balance(s) from JSON ('none' flag).", "INFO")
            if updated_db_pnl_count > 0:
                log_and_print(f"Updated {updated_db_pnl_count} user(s) profitandloss in DB.", "INFO")

            valid_processed_count = len(users_dictionary)
            if len(rows_current) > 5 and valid_processed_count > 5:
                log_and_print(f"... and {valid_processed_count - 5} more valid row(s) processed (not all displayed).", "INFO")
            
            print("---")
            log_and_print("Finished processing all fetched rows.", "SUCCESS")
            
            # 4. Write the final dictionary to the JSON file
            log_and_print(f"Attempting to write data to file: {OUTPUT_FILE_PATH}", "INFO")
            try:
                os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
                
                if users_dictionary:
                    with open(OUTPUT_FILE_PATH, 'w') as f:
                        json.dump(users_dictionary, f, indent=4)
                    log_and_print(f"Successfully wrote {len(users_dictionary)} user configurations to JSON.", "SUCCESS")
                else:
                    log_and_print("No valid user data to write to JSON after filtering and skipping.", "WARNING")
                    
            except IOError as file_error:
                log_and_print(f"Failed to write to JSON file. Check path/permissions: {file_error}", "ERROR")

        else:
            log_and_print(f"The query was successful but returned 0 rows for '{INSIDERS_TABLE}'. The table is empty. No JSON file created/updated.", "WARNING")
    
    except Exception as e:
        log_and_print(f"An unexpected error occurred during database operations or processing: {str(e)}", "ERROR")
        
    finally:
        log_and_print("--- Cleanup Operations ---", "TITLE")
        db.shutdown()
        log_and_print("Database connection closed.", "SUCCESS")
        print("\n")
        log_and_print("===== Row Fetch and JSON Export Completed =====", "TITLE")
        
def copy_verified_users_to_brokers_dictionary():
    """
    Reads the main user dictionary, filters users where:
    1. ACCOUNT_VERIFICATION is 'verified'
    2. LOYALTIES is 'justjoined' OR 'elligible'
    
    Copies these records to a NEW, empty brokers dictionary in memory, and then 
    overwrites the BROKERS_OUTPUT_FILE_PATH completely.
    """
    log_and_print("--- Starting Copy Verified Users to Brokers Dictionary (OVERWRITE MODE) ---", "TITLE")
    
    # --- 1. Load existing data (Source only) ---
    users_dictionary = {}
    brokers_dictionary = {} # **Starts as an empty dictionary to ensure a complete overwrite**
    
    try:
        # Load the main users dictionary (source)
        if os.path.exists(OUTPUT_FILE_PATH):
            with open(OUTPUT_FILE_PATH, 'r') as f:
                users_dictionary = json.load(f)
        else:
            log_and_print("Main users dictionary not found. Nothing to copy.", "WARNING")
            return
            
    except Exception as e:
        log_and_print(f"ERROR: Failed to load users JSON file: {str(e)}", "ERROR")
        return

    # --- 2. Filter and copy users into the empty brokers dictionary ---
    copied_count = 0
    
    for key, user_data in users_dictionary.items():
        account_verified = str(user_data.get("ACCOUNT_VERIFICATION", "")).lower().strip() == "verified"
        loyalty_status = str(user_data.get("LOYALTIES", "")).lower().strip()
        loyalty_check = loyalty_status in ("justjoined", "elligible")
        
        if account_verified and loyalty_check:
            # 🔔 Action: Copy user data to the brokers dictionary
            brokers_dictionary[key] = user_data.copy()
            copied_count += 1
            log_and_print(f"Copying user {key} (Loyalty: {loyalty_status}) to brokers dictionary.", "INFO")
            
    # --- 3. Write output file (Completely overwrites the file with only the new data) ---
    
    if copied_count > 0 or os.path.exists(BROKERS_OUTPUT_FILE_PATH):
        try:
            os.makedirs(os.path.dirname(BROKERS_OUTPUT_FILE_PATH), exist_ok=True)
            with open(BROKERS_OUTPUT_FILE_PATH, 'w') as f:
                json.dump(brokers_dictionary, f, indent=4) 
            log_and_print(f"Successfully **copied** {copied_count} user(s) and **completely OVERWROTE** the brokers JSON file. Total brokers: {len(brokers_dictionary)}.", "SUCCESS")
        except IOError as file_error:
            log_and_print(f"ERROR: Failed to write brokers JSON file: {file_error}", "ERROR")
    else:
        log_and_print("No qualified users found to copy. Brokers JSON file remains unchanged.", "INFO")

    log_and_print("--- Finished Copy Verified Users to Brokers Dictionary ---", "TITLE")
    
if __name__ == "__main__":
    fetch_insiders_server_rows()
    fetch_insiders_server_rows()
    copy_verified_users_to_brokers_dictionary()
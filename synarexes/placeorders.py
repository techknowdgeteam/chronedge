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



def load_brokers_dictionary():
    BROKERS_JSON_PATH = r"C:\xampp\htdocs\chronedge\synarex\brokersdictionary.json"
    """Load brokers config from JSON file with error handling and fallback."""
    if not os.path.exists(BROKERS_JSON_PATH):
        print(f"CRITICAL: {BROKERS_JSON_PATH} NOT FOUND! Using empty config.", "CRITICAL")
        return {}

    try:
        with open(BROKERS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Optional: Convert numeric strings back to int where needed
        for broker_name, cfg in data.items():
            if "LOGIN_ID" in cfg and isinstance(cfg["LOGIN_ID"], str):
                cfg["LOGIN_ID"] = cfg["LOGIN_ID"].strip()
            if "RISKREWARD" in cfg and isinstance(cfg["RISKREWARD"], (str, float)):
                cfg["RISKREWARD"] = int(cfg["RISKREWARD"])
        
        print(f"Brokers config loaded successfully → {len(data)} broker(s)", "SUCCESS")
        return data

    except json.JSONDecodeError as e:
        print(f"Invalid JSON in brokersdictionary.json: {e}", "CRITICAL")
        return {}
    except Exception as e:
        print(f"Failed to load brokersdictionary.json: {e}", "CRITICAL")
        return {}
brokersdictionary = load_brokers_dictionary()


def log_and_print(message, level="INFO"):
    """Log and print messages in a structured format."""
    timestamp = datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level:8} | {message}")


def _0_50_4_orders():
    def _0_50cent_usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $0.60
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 0.60 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (0.50 <= balance < 3.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_0_50cent_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 0.60 ---
                    if risk_usd <= 0.60:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $0.60 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_0.60": kept,
                "removed_risk_>_0.60": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 0.60",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $0.60) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_0_50cent_usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_0_50cent_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (0.50 <= balance < 3.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue


            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 0.50 and balance >= 0.50:
                log_and_print(f"Equity ${equity:.2f} < $0.50 while Balance ${balance:.2f} ≥ $0.50 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 0.50 and balance < 0.50:
                log_and_print(f"Equity ${equity:.2f} > $0.50 while Balance ${balance:.2f} < $0.50 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (0.50 <= balance < 3.99):
                log_and_print(f"Balance ${balance:.2f} not in $0.50–$3.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")



            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 4.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True

    def _0_50cent_usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $0.60  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 0.60
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 0.50 and balance >= 0.50:
                log_and_print(f"Equity ${equity:.2f} < $0.50 while Balance ${balance:.2f} ≥ $0.50 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 0.50 and balance < 0.50:
                log_and_print(f"Equity ${equity:.2f} > $0.50 while Balance ${balance:.2f} < $0.50 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (0.50 <= balance < 3.99):
                log_and_print(f"Balance ${balance:.2f} not in $0.50–$3.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_0_50cent_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 4. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_0_50cent_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _0_50cent_usd_ratio_levels():
        """
        0_50cent_usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 4... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("0_50cent_usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (0.50 <= balance < 3.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _0_50cent_usd_live_sl_tp_amounts()
    place_0_50cent_usd_orders()
    _0_50cent_usd_history_and_deduplication()
    _0_50cent_usd_ratio_levels()

def _4_8_orders():
    def _1usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $1.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 1.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (4.0 <= balance < 7.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_1_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 1.10 ---
                    if risk_usd <= 1.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $1.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_1.10": kept,
                "removed_risk_>_1.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 1.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $1.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_1usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_1_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue


            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 4.0 and balance >= 4.0:
                log_and_print(f"Equity ${equity:.2f} < $4.0 while Balance ${balance:.2f} ≥ $4.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 4.0 and balance < 4.0:
                log_and_print(f"Equity ${equity:.2f} > $4.0 while Balance ${balance:.2f} < $4.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (4.0 <= balance < 7.99):
                log_and_print(f"Balance ${balance:.2f} not in $4–$7.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            

            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 4.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True

    def _1usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $1.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 1.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 4.0 and balance >= 4.0:
                log_and_print(f"Equity ${equity:.2f} < $4.0 while Balance ${balance:.2f} ≥ $4.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 4.0 and balance < 4.0:
                log_and_print(f"Equity ${equity:.2f} > $4.0 while Balance ${balance:.2f} < $4.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (4.0 <= balance < 7.99):
                log_and_print(f"Balance ${balance:.2f} not in $4–$7.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_1_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 4. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_1_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _1usd_ratio_levels():
        """
        1usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 4... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("1usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (4.0 <= balance < 7.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _1usd_live_sl_tp_amounts()
    place_1usd_orders()
    _1usd_history_and_deduplication()
    _1usd_ratio_levels()

def _8_12_orders():
    def _2usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $2.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 2.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (8.0 <= balance < 11.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_2_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 2.10 ---
                    if risk_usd <= 2.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $2.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_2.10": kept,
                "removed_risk_>_2.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 2.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $2.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_2usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_2_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue


            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 8.0 and balance >= 8.0:
                log_and_print(f"Equity ${equity:.2f} < $8.0 while Balance ${balance:.2f} ≥ $8.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 8.0 and balance < 8.0:
                log_and_print(f"Equity ${equity:.2f} > $8.0 while Balance ${balance:.2f} < $8.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (8.0 <= balance < 11.99):
                log_and_print(f"Balance ${balance:.2f} not in $8–$11.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")


            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 4.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True

    def _2usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $2.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 2.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 8.0 and balance >= 8.0:
                log_and_print(f"Equity ${equity:.2f} < $8.0 while Balance ${balance:.2f} ≥ $8.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 8.0 and balance < 8.0:
                log_and_print(f"Equity ${equity:.2f} > $8.0 while Balance ${balance:.2f} < $8.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (8.0 <= balance < 11.99):
                log_and_print(f"Balance ${balance:.2f} not in $8–$11.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_2_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 4. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_2_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _2usd_ratio_levels():
        """
        2usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 4... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("2usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (8.0 <= balance < 11.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _2usd_live_sl_tp_amounts()
    place_2usd_orders()
    _2usd_history_and_deduplication()
    _2usd_ratio_levels()

def _12_20_orders():
    def _3usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $3.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 3.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (12.0 <= balance < 19.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_3_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 3.10 ---
                    if risk_usd <= 3.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $3.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_3.10": kept,
                "removed_risk_>_3.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 3.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $3.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_3usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_3_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue



            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 12.0 and balance >= 12.0:
                log_and_print(f"Equity ${equity:.2f} < $12.0 while Balance ${balance:.2f} ≥ $12.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 12.0 and balance < 12.0:
                log_and_print(f"Equity ${equity:.2f} > $12.0 while Balance ${balance:.2f} < $12.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (12.0 <= balance < 19.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$19.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")



            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 4.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True

    def _3usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $3.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 3.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 12.0 and balance >= 12.0:
                log_and_print(f"Equity ${equity:.2f} < $12.0 while Balance ${balance:.2f} ≥ $12.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 12.0 and balance < 12.0:
                log_and_print(f"Equity ${equity:.2f} > $12.0 while Balance ${balance:.2f} < $12.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (12.0 <= balance < 19.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$19.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_3_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 4. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_3_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _3usd_ratio_levels():
        """
        3USD RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 4... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("3USD RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (12.0 <= balance < 19.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _3usd_live_sl_tp_amounts()
    place_3usd_orders()
    _3usd_history_and_deduplication()
    _3usd_ratio_levels()

def _20_80_orders():
    def _4usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $4.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 4.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (20.0 <= balance < 79.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_4_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 4.10 ---
                    if risk_usd <= 4.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $4.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_4.10": kept,
                "removed_risk_>_4.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 4.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $4.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_4usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_4_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 20.0 and balance >= 20.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 20.0 and balance < 20.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (20.0 <= balance < 79.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")



            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 4.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True
  
    def _4usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $4.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 4.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 20.0 and balance >= 20.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 20.0 and balance < 20.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (20.0 <= balance < 79.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_4_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 4. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_4_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _4usd_ratio_levels():
        """
        4usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 4... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("4usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (20.0 <= balance < 79.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _4usd_live_sl_tp_amounts()
    place_4usd_orders()
    _4usd_history_and_deduplication()
    _4usd_ratio_levels()

def _80_160_orders():
    def _8usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $8.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 8.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (80.0 <= balance < 159.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_8_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 8.10 ---
                    if risk_usd <= 8.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $8.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_8.10": kept,
                "removed_risk_>_8.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 8.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $8.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_8usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_8_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 80.0 and balance >= 80.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 80.0 and balance < 80.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (80.0 <= balance < 159.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")



            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123856,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 8.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True
  
    def _8usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $8.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 8.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 80.0 and balance >= 80.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 80.0 and balance < 80.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (80.0 <= balance < 159.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_8_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 8. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_8_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _8usd_ratio_levels():
        """
        8usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 8... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("8usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (80.0 <= balance < 159.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _8usd_live_sl_tp_amounts()
    place_8usd_orders()
    _8usd_history_and_deduplication()
    _8usd_ratio_levels()

def _160_320_orders():
    def _16usd_live_sl_tp_amounts():
        
        """
        READS: hightolow.json
        CALCULATES: Live $3 risk & profit
        PRINTS: 3-line block for every market
        SAVES:
            - live_risk_profit_all.json → only valid ≤ $16.10
            - OVERWRITES hightolow.json → REMOVES bad orders PERMANENTLY
        FILTER: Delete any order with live_risk_usd > 16.10 from BOTH files
        """

        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        INPUT_FILE = "hightolow.json"
        OUTPUT_FILE = "live_risk_profit_all.json"

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID = cfg["LOGIN_ID"]
            PASSWORD = cfg["PASSWORD"]
            SERVER = cfg["SERVER"]

            log_and_print(f"\n{'='*60}", "INFO")
            log_and_print(f"PROCESSING BROKER: {broker_name.upper()}", "INFO")
            log_and_print(f"{'='*60}", "INFO")

            # ------------------- CONNECT TO MT5 -------------------
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=60000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            if not (160.0 <= balance < 319.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Connected → Balance: ${balance:.2f} {currency}", "INFO")

            # ------------------- LOAD JSON -------------------
            json_path = Path(BASE_DIR) / broker_name / "risk_16_usd" / INPUT_FILE
            if not json_path.exists():
                log_and_print(f"JSON not found: {json_path}", "ERROR")
                mt5.shutdown()
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    original_data = json.load(f)
                entries = original_data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read JSON: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in JSON.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Loaded {len(entries)} entries → Calculating LIVE risk...", "INFO")

            # ------------------- PROCESS & FILTER -------------------
            valid_entries = []        # For overwriting hightolow.json
            results = []              # For live_risk_profit_all.json
            total = len(entries)
            kept = 0
            removed = 0

            for i, entry in enumerate(entries, 1):
                market = entry["market"]
                try:
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type = entry["limit_order"]
                    sl_pips = float(entry.get("sl_pips", 0))
                    tp_pips = float(entry.get("tp_pips", 0))

                    # --- LIVE DATA ---
                    info = mt5.symbol_info(market)
                    tick = mt5.symbol_info_tick(market)

                    if not info or not tick:
                        log_and_print(f"NO LIVE DATA for {market} → Using fallback", "WARNING")
                        pip_value = 0.1
                        risk_usd = volume * sl_pips * pip_value
                        profit_usd = volume * tp_pips * pip_value
                    else:
                        point = info.point
                        contract = info.trade_contract_size

                        risk_points = abs(price - sl) / point
                        profit_points = abs(tp - price) / point

                        point_val = contract * point
                        if "JPY" in market and currency == "USD":
                            point_val /= 100

                        risk_ac = risk_points * point_val * volume
                        profit_ac = profit_points * point_val * volume

                        risk_usd = risk_ac
                        profit_usd = profit_ac

                        if currency != "USD":
                            conv = f"USD{currency}"
                            rate_tick = mt5.symbol_info_tick(conv)
                            rate = rate_tick.bid if rate_tick else 1.0
                            risk_usd /= rate
                            profit_usd /= rate

                    risk_usd = round(risk_usd, 2)
                    profit_usd = round(profit_usd, 2)

                    # --- PRINT ALL ---
                    print(f"market: {market}")
                    print(f"risk: {risk_usd} USD")
                    print(f"profit: {profit_usd} USD")
                    print("---")

                    # --- FILTER: KEEP ONLY <= 16.10 ---
                    if risk_usd <= 16.10:
                        # Keep in BOTH files
                        valid_entries.append(entry)  # Original format
                        results.append({
                            "market": market,
                            "order_type": order_type,
                            "entry_price": round(price, 6),
                            "sl": round(sl, 6),
                            "tp": round(tp, 6),
                            "volume": round(volume, 5),
                            "live_risk_usd": risk_usd,
                            "live_profit_usd": profit_usd,
                            "sl_pips": round(sl_pips, 2),
                            "tp_pips": round(tp_pips, 2),
                            "has_live_tick": bool(info and tick),
                            "current_bid": round(tick.bid, 6) if tick else None,
                            "current_ask": round(tick.ask, 6) if tick else None,
                        })
                        kept += 1
                    else:
                        removed += 1
                        log_and_print(f"REMOVED {market}: live risk ${risk_usd} > $16.10 → DELETED FROM BOTH JSON FILES", "WARNING")

                except Exception as e:
                    log_and_print(f"ERROR on {market}: {e}", "ERROR")
                    removed += 1

                if i % 5 == 0 or i == total:
                    log_and_print(f"Processed {i}/{total} | Kept: {kept} | Removed: {removed}", "INFO")

            # ------------------- SAVE OUTPUT: live_risk_profit_all.json -------------------
            out_path = json_path.parent / OUTPUT_FILE
            report = {
                "broker": broker_name,
                "account_currency": currency,
                "generated_at": datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z"),
                "source_file": str(json_path),
                "total_entries": total,
                "kept_risk_<=_16.10": kept,
                "removed_risk_>_16.10": removed,
                "filter_applied": "Delete from both input & output if live_risk_usd > 16.10",
                "orders": results
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"SAVED → {out_path} | Kept: {kept} | Removed: {removed}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save failed: {e}", "ERROR")

            # ------------------- OVERWRITE INPUT: hightolow.json -------------------
            cleaned_input = original_data.copy()
            cleaned_input["entries"] = valid_entries  # Only good ones

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(cleaned_input, f, indent=2)
                log_and_print(f"OVERWRITTEN → {json_path} | Now has {len(valid_entries)} entries (removed {removed})", "SUCCESS")
            except Exception as e:
                log_and_print(f"Failed to overwrite input JSON: {e}", "ERROR")

            mt5.shutdown()
            log_and_print(f"FINISHED {broker_name} → {kept}/{total} valid orders in BOTH files", "SUCCESS")

        log_and_print("\nALL DONE – BAD ORDERS (> $16.10) DELETED FROM INPUT & OUTPUT!", "SUCCESS")
        return True
    
    def place_16usd_orders():
        

        BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        RISK_FOLDER = "risk_16_usd"
        STRATEGY_FILE = "hightolow.json"
        REPORT_SUFFIX = "forex_order_report.json"
        ISSUES_FILE = "ordersissues.json"

        for broker_name, broker_cfg in brokersdictionary.items():
            TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
            LOGIN_ID = broker_cfg["LOGIN_ID"]
            PASSWORD = broker_cfg["PASSWORD"]
            SERVER = broker_cfg["SERVER"]

            log_and_print(f"Processing broker: {broker_name} (Balance $12–$20 mode)", "INFO")

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue
            balance = account_info.balance
            equity = account_info.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 160.0 and balance >= 160.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 160.0 and balance < 160.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (160.0 <= balance < 319.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue
            # === Only reaches here if: equity >= 8 AND balance in [8, 11.99) ===
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")



            log_and_print(f"Balance: ${balance:.2f} → Using {RISK_FOLDER} + {STRATEGY_FILE}", "INFO")

            # === Load hightolow.json ===
            file_path = Path(BASE_INPUT_DIR) / broker_name / RISK_FOLDER / STRATEGY_FILE
            if not file_path.exists():
                log_and_print(f"File not found: {file_path}", "WARNING")
                mt5.shutdown()
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
            except Exception as e:
                log_and_print(f"Failed to read {file_path}: {e}", "ERROR")
                mt5.shutdown()
                continue

            if not entries:
                log_and_print("No entries in hightolow.json", "INFO")
                mt5.shutdown()
                continue

            # === Load existing orders & positions ===
            existing_pending = {}  # (symbol, type) → ticket
            running_positions = set()  # symbols with open position

            for order in (mt5.orders_get() or []):
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    existing_pending[(order.symbol, order.type)] = order.ticket

            for pos in (mt5.positions_get() or []):
                running_positions.add(pos.symbol)

            # === Reporting ===
            report_file = file_path.parent / REPORT_SUFFIX
            existing_reports = json.load(report_file.open("r", encoding="utf-8")) if report_file.exists() else []
            issues_list = []
            now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f+01:00")
            placed = failed = skipped = 0

            for entry in entries:
                try:
                    symbol = entry["market"]
                    price = float(entry["entry_price"])
                    sl = float(entry["sl_price"])
                    tp = float(entry["tp_price"])
                    volume = float(entry["volume"])
                    order_type_str = entry["limit_order"]
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type_str == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT

                    # === SKIP: Already running or pending ===
                    if symbol in running_positions:
                        skipped += 1
                        log_and_print(f"{symbol} has running position → SKIPPED", "INFO")
                        continue

                    key = (symbol, order_type)
                    if key in existing_pending:
                        skipped += 1
                        log_and_print(f"{symbol} {order_type_str} already pending → SKIPPED", "INFO")
                        continue

                    # === Symbol check ===
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        issues_list.append({"symbol": symbol, "reason": "Symbol not available"})
                        failed += 1
                        continue

                    # === Volume fix ===
                    vol_step = symbol_info.volume_step
                    volume = max(symbol_info.volume_min,
                                round(volume / vol_step) * vol_step)
                    volume = min(volume, symbol_info.volume_max)

                    # === Price distance check ===
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        issues_list.append({"symbol": symbol, "reason": "No tick data"})
                        failed += 1
                        continue

                    point = symbol_info.point
                    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                        if price >= tick.ask or (tick.ask - price) < 10 * point:
                            skipped += 1
                            continue
                    else:
                        if price <= tick.bid or (price - tick.bid) < 10 * point:
                            skipped += 1
                            continue

                    # === Build & send order ===
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123856,
                        "comment": "Risk3_Auto",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result is None:
                        result = type('obj', (), {'retcode': 10000, 'comment': 'order_send returned None'})()

                    success = result.retcode == mt5.TRADE_RETCODE_DONE
                    if success:
                        existing_pending[key] = result.order
                        placed += 1
                        log_and_print(f"{symbol} {order_type_str} @ {price} → PLACED (ticket {result.order})", "SUCCESS")
                    else:
                        failed += 1
                        issues_list.append({"symbol": symbol, "reason": result.comment})

                    # === Report ===
                    if "cent" in RISK_FOLDER:
                        risk_usd = 0.5
                    else:
                        risk_usd = float(RISK_FOLDER.split("_")[1].replace("usd", ""))

                    # === Report ===
                    report_entry = {
                        "symbol": symbol,
                        "order_type": order_type_str,
                        "price": price,
                        "volume": volume,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": risk_usd,   # ← Now correct: 0.5, 1.0, 2.0, 3.0, 8.0
                        "ticket": result.order if success else None,
                        "success": success,
                        "error_code": result.retcode if not success else None,
                        "error_msg": result.comment if not success else None,
                        "timestamp": now_str
                    }
                    existing_reports.append(report_entry)
                    try:
                        with report_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_reports, f, indent=2)
                    except:
                        pass

                except Exception as e:
                    failed += 1
                    issues_list.append({"symbol": symbol, "reason": f"Exception: {e}"})
                    log_and_print(f"Error processing {symbol}: {e}", "ERROR")

            # === Save issues ===
            issues_path = file_path.parent / ISSUES_FILE
            try:
                existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
                with issues_path.open("w", encoding="utf-8") as f:
                    json.dump(existing_issues + issues_list, f, indent=2)
            except:
                pass

            mt5.shutdown()
            log_and_print(
                f"{broker_name} DONE → Placed: {placed}, Failed: {failed}, Skipped: {skipped}",
                "SUCCESS"
            )

        log_and_print("All $12–$20 accounts processed.", "SUCCESS")
        return True
  
    def _16usd_history_and_deduplication():
        """
        HISTORY + PENDING + POSITION DUPLICATE DETECTOR + RISK SNIPER
        - Cancels risk > $16.10  (even if TP=0)
        - Cancels HISTORY DUPLICATES
        - Cancels PENDING LIMIT DUPLICATES
        - Cancels PENDING if POSITION already exists
        - Shows duplicate market name on its own line
        ONLY PROCESSES ACCOUNTS WITH BALANCE $12.00 – $19.99
        """
        BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
        REPORT_NAME = "pending_risk_profit_per_order.json"
        MAX_RISK_USD = 16.10
        LOOKBACK_DAYS = 5
        PRICE_PRECISION = 5
        TZ = pytz.timezone("Africa/Lagos")

        five_days_ago = datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS)

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg["TERMINAL_PATH"]
            LOGIN_ID     = cfg["LOGIN_ID"]
            PASSWORD     = cfg["PASSWORD"]
            SERVER       = cfg["SERVER"]

            log_and_print(f"\n{'='*80}", "INFO")
            log_and_print(f"BROKER: {broker_name.upper()} | FULL DUPLICATE + RISK GUARD", "INFO")
            log_and_print(f"{'='*80}", "INFO")

            # ---------- MT5 Init ----------
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue
            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue
            if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account = mt5.account_info()
            if not account:
                log_and_print("No account info.", "ERROR")
                mt5.shutdown()
                continue

            balance = account.balance
            equity = account.equity
            log_and_print(f"Balance: ${balance:.2f}, Equity: ${equity:.2f}", "INFO")
            if equity < 160.0 and balance >= 160.0:
                log_and_print(f"Equity ${equity:.2f} < $20.0 while Balance ${balance:.2f} ≥ $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if equity >= 160.0 and balance < 160.0:
                log_and_print(f"Equity ${equity:.2f} > $20.0 while Balance ${balance:.2f} < $20.0 → IN DRAWDOWN → SKIPPED", "WARNING")
                mt5.shutdown()
                continue
            if not (160.0 <= balance < 319.99):
                log_and_print(f"Balance ${balance:.2f} not in $20–$99.99 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            currency = account.currency
            log_and_print(f"Account: {account.login} | Balance: ${balance:.2f} {currency} → Proceeding with risk_16_usd checks", "INFO")

            # ---------- Get Data ----------
            pending_orders = [o for o in (mt5.orders_get() or [])
                            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
            positions = mt5.positions_get()
            history_deals = mt5.history_deals_get(int(five_days_ago.timestamp()), int(datetime.now(TZ).timestamp()))

            if not pending_orders:
                log_and_print("No pending orders.", "INFO")
                mt5.shutdown()
                continue

            # ---------- BUILD DATABASES ----------
            log_and_print(f"Building duplicate databases...", "INFO")

            # 1. Historical Setups
            historical_keys = {}  # (symbol, entry, sl) → details
            if history_deals:
                for deal in history_deals:
                    if deal.entry != mt5.DEAL_ENTRY_IN: continue
                    if deal.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL): continue

                    order = mt5.history_orders_get(ticket=deal.order)
                    if not order: continue
                    order = order[0]
                    if order.sl == 0: continue

                    symbol = deal.symbol
                    entry = round(deal.price, PRICE_PRECISION)
                    sl = round(order.sl, PRICE_PRECISION)

                    key = (symbol, entry, sl)
                    if key not in historical_keys:
                        profit = sum(d.profit for d in history_deals if d.order == deal.order and d.entry == mt5.DEAL_ENTRY_OUT)
                        historical_keys[key] = {
                            "time": datetime.fromtimestamp(deal.time, TZ).strftime("%Y-%m-%d %H:%M"),
                            "profit": round(profit, 2),
                            "symbol": symbol
                        }

            # 2. Open Positions (by symbol)
            open_symbols = {pos.symbol for pos in positions} if positions else set()

            # 3. Pending Orders Key Map
            pending_keys = {}  # (symbol, entry, sl) → [order_tickets]
            for order in pending_orders:
                key = (order.symbol, round(order.price_open, PRICE_PRECISION), round(order.sl, PRICE_PRECISION))
                pending_keys.setdefault(key, []).append(order.ticket)

            log_and_print(f"Loaded: {len(historical_keys)} history | {len(open_symbols)} open | {len(pending_keys)} unique pending setups", "INFO")

            # ---------- Process & Cancel ----------
            per_order_data = []
            kept = cancelled_risk = cancelled_hist = cancelled_pend_dup = cancelled_pos_dup = skipped = 0

            for order in pending_orders:
                symbol = order.symbol
                ticket = order.ticket
                volume = order.volume_current
                entry = round(order.price_open, PRICE_PRECISION)
                sl = round(order.sl, PRICE_PRECISION)
                tp = order.tp                     # may be 0

                # ---- NEW: ONLY REQUIRE SL, TP CAN BE 0 ----
                if sl == 0:
                    log_and_print(f"SKIP {ticket} | {symbol} | No SL", "WARNING")
                    skipped += 1
                    continue

                info = mt5.symbol_info(symbol)
                if not info or not mt5.symbol_info_tick(symbol):
                    log_and_print(f"SKIP {ticket} | {symbol} | No symbol data", "WARNING")
                    skipped += 1
                    continue

                point = info.point
                contract = info.trade_contract_size
                point_val = contract * point
                if "JPY" in symbol and currency == "USD":
                    point_val /= 100

                # ---- RISK CALCULATION (always possible with SL) ----
                risk_points = abs(entry - sl) / point
                risk_usd = risk_points * point_val * volume
                if currency != "USD":
                    rate = mt5.symbol_info_tick(f"USD{currency}")
                    if not rate:
                        log_and_print(f"SKIP {ticket} | No USD{currency} rate", "WARNING")
                        skipped += 1
                        continue
                    risk_usd /= rate.bid

                # ---- PROFIT CALCULATION (only if TP exists) ----
                profit_usd = None
                if tp != 0:
                    profit_usd = abs(tp - entry) / point * point_val * volume
                    if currency != "USD":
                        profit_usd /= rate.bid

                # ---- DUPLICATE KEYS ----
                key = (symbol, entry, sl)
                dup_hist = historical_keys.get(key)
                is_position_open = symbol in open_symbols
                is_pending_duplicate = len(pending_keys.get(key, [])) > 1

                print(f"\nmarket: {symbol}")
                print(f"risk: {risk_usd:.2f} USD | profit: {profit_usd if profit_usd is not None else 'N/A'} USD")

                cancel_reason = None
                cancel_type = None

                # === 1. RISK CANCEL (works even if TP=0) ===
                if risk_usd > MAX_RISK_USD:
                    cancel_reason = f"RISK > ${MAX_RISK_USD}"
                    cancel_type = "RISK"
                    print(f"{cancel_reason} → CANCELLED")

                # === 2. HISTORY DUPLICATE ===
                elif dup_hist:
                    cancel_reason = "HISTORY DUPLICATE"
                    cancel_type = "HIST_DUP"
                    print("HISTORY DUPLICATE ORDER FOUND!")
                    print(dup_hist["symbol"])
                    print(f"entry: {entry} | sl: {sl}")
                    print(f"used: {dup_hist['time']} | P/L: {dup_hist['profit']:+.2f} {currency}")
                    print("→ HISTORY DUPLICATE CANCELLED")
                    print("!" * 60)

                # === 3. PENDING DUPLICATE ===
                elif is_pending_duplicate:
                    cancel_reason = "PENDING DUPLICATE"
                    cancel_type = "PEND_DUP"
                    print("PENDING LIMIT DUPLICATE FOUND!")
                    print(symbol)
                    print(f"→ DUPLICATE PENDING ORDER CANCELLED")
                    print("-" * 60)

                # === 8. POSITION EXISTS (Cancel Pending) ===
                elif is_position_open:
                    cancel_reason = "POSITION ALREADY OPEN"
                    cancel_type = "POS_DUP"
                    print("POSITION ALREADY RUNNING!")
                    print(symbol)
                    print(f"→ PENDING ORDER CANCELLED (POSITION ACTIVE)")
                    print("^" * 60)

                # === NO ISSUE → KEEP ===
                else:
                    print("No duplicate. Order kept.")
                    kept += 1
                    per_order_data.append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk_usd": round(risk_usd, 2),
                        "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                        "status": "KEPT"
                    })
                    continue  # Skip cancel

                # === CANCEL ORDER ===
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    log_and_print(f"{cancel_type} CANCELLED {ticket} | {symbol} | {cancel_reason}", "WARNING")
                    if cancel_type == "RISK": cancelled_risk += 1
                    elif cancel_type == "HIST_DUP": cancelled_hist += 1
                    elif cancel_type == "PEND_DUP": cancelled_pend_dup += 1
                    elif cancel_type == "POS_DUP": cancelled_pos_dup += 1
                else:
                    log_and_print(f"CANCEL FAILED {ticket} | {res.comment}", "ERROR")

                per_order_data.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "risk_usd": round(risk_usd, 2),
                    "profit_usd": round(profit_usd, 2) if profit_usd is not None else None,
                    "status": "CANCELLED",
                    "reason": cancel_reason,
                    "duplicate_time": dup_hist["time"] if dup_hist else None,
                    "duplicate_pl": dup_hist["profit"] if dup_hist else None
                })

            # === SUMMARY ===
            log_and_print(f"\nSUMMARY:", "SUCCESS")
            log_and_print(f"KEPT: {kept}", "INFO")
            log_and_print(f"CANCELLED → RISK: {cancelled_risk} | HIST DUP: {cancelled_hist} | "
                        f"PEND DUP: {cancelled_pend_dup} | POS DUP: {cancelled_pos_dup} | SKIPPED: {skipped}", "WARNING")

            # === SAVE REPORT ===
            out_dir = Path(BASE_DIR) / broker_name / "risk_16_usd"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / REPORT_NAME

            report = {
                "broker": broker_name,
                "checked_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "max_risk_usd": MAX_RISK_USD,
                "lookback_days": LOOKBACK_DAYS,
                "summary": {
                    "kept": kept,
                    "cancelled_risk": cancelled_risk,
                    "cancelled_history_duplicate": cancelled_hist,
                    "cancelled_pending_duplicate": cancelled_pend_dup,
                    "cancelled_position_duplicate": cancelled_pos_dup,
                    "skipped": skipped
                },
                "orders": per_order_data
            }

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                log_and_print(f"Report saved: {out_path}", "SUCCESS")
            except Exception as e:
                log_and_print(f"Save error: {e}", "ERROR")

            mt5.shutdown()

        log_and_print("\nALL $12–$20 ACCOUNTS: DUPLICATE SCAN + RISK GUARD = DONE", "SUCCESS")
        return True

    def _16usd_ratio_levels():
        """
        16usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING POSITIONS) – BROKER-SAFE
        - Balance $12–$19.99 only
        - Auto-supports riskreward: 1, 2, 3, 8... (any integer)
        - Case-insensitive config
        - consistency → Dynamic TP = RISKREWARD × Risk
        - martingale → TP = 1R (always), ignores RISKREWARD
        - Smart ratio ladder (shows 1R, 2R, 3R only when needed)
        """
        TZ = pytz.timezone("Africa/Lagos")

        log_and_print(f"\n{'='*80}", "INFO")
        log_and_print("16usd RATIO LEVELS + TP UPDATE (PENDING + RUNNING) – CONSISTENCY: N×R | MARTINGALE: 1R", "INFO")
        log_and_print(f"{'='*80}", "INFO")

        for broker_name, cfg in brokersdictionary.items():
            TERMINAL_PATH = cfg.get("TERMINAL_PATH") or cfg.get("terminal_path")
            LOGIN_ID      = cfg.get("LOGIN_ID")      or cfg.get("login_id")
            PASSWORD      = cfg.get("PASSWORD")      or cfg.get("password")
            SERVER        = cfg.get("SERVER")        or cfg.get("server")
            SCALE         = (cfg.get("SCALE")        or cfg.get("scale")        or "").strip().lower()
            STRATEGY      = (cfg.get("STRATEGY")    or cfg.get("strategy")    or "").strip().lower()

            # === Case-insensitive riskreward lookup ===
            riskreward_raw = None
            for key in cfg:
                if key.lower() == "riskreward":
                    riskreward_raw = cfg[key]
                    break

            if riskreward_raw is None:
                riskreward_raw = 2
                log_and_print(f"{broker_name}: 'riskreward' not found → using default 2R", "WARNING")

            log_and_print(
                f"\nProcessing broker: {broker_name} | Scale: {SCALE.upper()} | "
                f"Strategy: {STRATEGY.upper()} | riskreward: {riskreward_raw}R", "INFO"
            )

            # === Validate required fields ===
            missing = []
            for f in ("TERMINAL_PATH", "LOGIN_ID", "PASSWORD", "SERVER", "SCALE"):
                if not locals()[f]: missing.append(f)
            if missing:
                log_and_print(f"Missing config: {', '.join(missing)} → SKIPPED", "ERROR")
                continue

            # === MT5 Init ===
            if not os.path.exists(TERMINAL_PATH):
                log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
                continue

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                                server=SERVER, timeout=30000):
                log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
                continue

            if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"MT5 login failed: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            account_info = mt5.account_info()
            if not account_info:
                log_and_print(f"Failed to get account info: {mt5.last_error()}", "ERROR")
                mt5.shutdown()
                continue

            balance = account_info.balance
            if not (160.0 <= balance < 319.99):
                log_and_print(f"Balance ${balance:.2f} not in $12–$20 range → SKIPPED", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Balance: ${balance:.2f} → Scanning positions & pending orders...", "INFO")

            # === Determine effective RR ===
            try:
                config_rr = int(float(riskreward_raw))
                if config_rr < 1: config_rr = 1
            except (ValueError, TypeError):
                config_rr = 2
                log_and_print(f"Invalid riskreward '{riskreward_raw}' → using 2R", "WARNING")

            effective_rr = 1 if SCALE == "martingale" else config_rr
            rr_source = "MARTINGALE (forced 1R)" if SCALE == "martingale" else f"CONFIG ({effective_rr}R)"
            log_and_print(f"Effective TP: {effective_rr}R [{rr_source}]", "INFO")

            # ------------------------------------------------------------------ #
            # 1. PENDING LIMIT ORDERS
            # ------------------------------------------------------------------ #
            pending_orders = [
                o for o in (mt5.orders_get() or [])
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
                and getattr(o, 'sl', 0) != 0 and getattr(o, 'tp', 0) != 0
            ]

            # ------------------------------------------------------------------ #
            # 2. RUNNING POSITIONS
            # ------------------------------------------------------------------ #
            running_positions = [
                p for p in (mt5.positions_get() or [])
                if p.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
                and p.sl != 0 and p.tp != 0
            ]

            # Merge into a single iterable with a flag
            items_to_process = []
            for o in pending_orders:
                items_to_process.append(('PENDING', o))
            for p in running_positions:
                items_to_process.append(('RUNNING', p))

            if not items_to_process:
                log_and_print("No valid pending orders or running positions found.", "INFO")
                mt5.shutdown()
                continue

            log_and_print(f"Found {len(pending_orders)} pending + {len(running_positions)} running → total {len(items_to_process)}", "INFO")

            processed_symbols = set()
            updated_count = 0

            for kind, obj in items_to_process:
                symbol   = obj.symbol
                ticket   = getattr(obj, 'ticket', None) or getattr(obj, 'order', None)
                entry_price = getattr(obj, 'price_open', None) or getattr(obj, 'price_current', None)
                sl_price = obj.sl
                current_tp = obj.tp
                is_buy   = obj.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY)

                if symbol in processed_symbols:
                    continue

                risk_distance = abs(entry_price - sl_price)
                if risk_distance <= 0:
                    log_and_print(f"Zero risk distance on {symbol} ({kind}) → skipped", "WARNING")
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    log_and_print(f"Symbol info missing: {symbol}", "WARNING")
                    continue

                digits = symbol_info.digits
                def r(p): return round(p, digits)

                entry_price = r(entry_price)
                sl_price    = r(sl_price)
                current_tp  = r(current_tp)
                direction   = 1 if is_buy else -1
                target_tp   = r(entry_price + direction * effective_rr * risk_distance)

                # ----- Ratio ladder (display only) -----
                ratio1 = r(entry_price + direction * 1 * risk_distance)
                ratio2 = r(entry_price + direction * 2 * risk_distance)
                ratio3 = r(entry_price + direction * 3 * risk_distance) if effective_rr >= 3 else None

                print(f"\n{symbol} | {kind} | Target: {effective_rr}R ({SCALE.upper()})")
                print(f"  Entry : {entry_price}")
                print(f"  1R    : {ratio1}")
                print(f"  2R    : {ratio2}")
                if ratio3:
                    print(f"  3R    : {ratio3}")
                print(f"  TP    : {current_tp} → ", end="")

                # ----- Modify TP -----
                tolerance = 10 ** -digits
                if abs(current_tp - target_tp) > tolerance:
                    if kind == "PENDING":
                        # modify pending order
                        request = {
                            "action": mt5.TRADE_ACTION_MODIFY,
                            "order": ticket,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": target_tp,
                            "type": obj.type,
                            "type_time": obj.type_time,
                            "type_filling": obj.type_filling,
                            "magic": getattr(obj, 'magic', 0),
                            "comment": getattr(obj, 'comment', "")
                        }
                        if hasattr(obj, 'expiration') and obj.expiration:
                            request["expiration"] = obj.expiration
                    else:  # RUNNING
                        # modify open position (SL/TP only)
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "sl": sl_price,
                            "tp": target_tp,
                            "symbol": symbol
                        }

                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{target_tp} [UPDATED]")
                        log_and_print(
                            f"TP → {effective_rr}R | {symbol} | {kind} | {current_tp} → {target_tp} [{SCALE.upper()}]",
                            "SUCCESS"
                        )
                        updated_count += 1
                    else:
                        err = result.comment if result else "Unknown"
                        print(f"{current_tp} [FAILED: {err}]")
                        log_and_print(f"TP UPDATE FAILED | {symbol} | {kind} | {err}", "ERROR")
                else:
                    print(f"{current_tp} [OK]")

                print(f"  SL    : {sl_price}")
                processed_symbols.add(symbol)

            mt5.shutdown()
            log_and_print(
                f"{broker_name} → {len(processed_symbols)} symbol(s) | "
                f"{updated_count} TP(s) set to {effective_rr}R [{SCALE.upper()}]",
                "SUCCESS"
            )

        log_and_print(
            "\nALL $12–$20 ACCOUNTS: R:R UPDATE (PENDING + RUNNING) – "
            "consistency=N×R, martingale=1R = DONE",
            "SUCCESS"
        )
        return True
    _16usd_live_sl_tp_amounts()
    place_16usd_orders()
    _16usd_history_and_deduplication()
    _16usd_ratio_levels()



def collect_all_brokers_limit_orders():
    BASE_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets"
    REPORT_NAME = "brokerslimitorders.json"
    TZ = pytz.timezone("Africa/Lagos")
    OUTPUT_PATH = Path(BASE_DIR) / REPORT_NAME

    MAX_AGE_SECONDS = 2 * 24 * 60 * 60  # 2 days (for stale pending)
    MAX_HISTORY_SECONDS = 5 * 60 * 60   # 5 hours (for recent filled/canceled)

    all_pending_orders = []
    all_open_positions = []
    all_history_orders = []  # ← NEW
    total_pending = 0
    total_positions = 0
    total_history = 0
    failed_brokers = []
    deleted_count = 0

    # Helper: Convert seconds to human-readable age string
    def format_age(seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        minutes = minutes % 60
        if hours < 24:
            return f"{hours}h {minutes}m" if minutes else f"{hours}h"
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h" if hours else f"{days}d"

    log_and_print(f"\n{'='*100}", "INFO")
    log_and_print(f"COLLECTING PENDING LIMITS + OPEN POSITIONS + RECENT HISTORY (<5h)", "INFO")
    log_and_print(f"{'='*100}", "INFO")

    # Per-broker tracking
    broker_symbol_data = {}

    for broker_name, cfg in brokersdictionary.items():
        TERMINAL_PATH = cfg["TERMINAL_PATH"]
        LOGIN_ID     = cfg["LOGIN_ID"]
        PASSWORD     = cfg["PASSWORD"]
        SERVER       = cfg["SERVER"]

        log_and_print(f"\n→ Broker: {broker_name.upper()}", "INFO")

        # ---------- MT5 Init ----------
        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"Terminal not found: {TERMINAL_PATH}", "ERROR")
            failed_brokers.append(broker_name)
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
            log_and_print(f"MT5 init failed: {mt5.last_error()}", "ERROR")
            failed_brokers.append(broker_name)
            continue

        if not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
            log_and_print(f"Login failed: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            failed_brokers.append(broker_name)
            continue

        account = mt5.account_info()
        if not account:
            log_and_print("No account info.", "ERROR")
            mt5.shutdown()
            failed_brokers.append(broker_name)
            continue

        balance = account.balance
        currency = account.currency
        log_and_print(f"Connected: Account {account.login} | Balance: ${balance:.2f} {currency}", "INFO")

        # Initialize broker data
        broker_symbol_data[broker_name] = {}
        current_time = datetime.now(TZ)

        # ---------- 1. PENDING LIMIT ORDERS ----------
        pending_orders_raw = mt5.orders_get() or []
        pending_orders = [
            o for o in pending_orders_raw
            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
        ]

        pending_count = len(pending_orders)
        total_pending += pending_count

        to_delete = []  # Per-broker stale pending

        if pending_count:
            log_and_print(f"Found {pending_count} pending limit order(s).", "INFO")

            for order in pending_orders:
                symbol = order.symbol

                if symbol not in broker_symbol_data[broker_name]:
                    broker_symbol_data[broker_name][symbol] = {
                        "has_open": False,
                        "pending": {"BUY": None, "SELL": None},
                        "account_login": account.login,
                        "account_currency": currency
                    }

                order_type_str = "BUY LIMIT" if order.type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL LIMIT"
                order_time = datetime.fromtimestamp(order.time_setup, TZ)
                age_seconds = (current_time - order_time).total_seconds()

                side_key = "BUY" if order.type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL"
                broker_symbol_data[broker_name][symbol]["pending"][side_key] = {
                    "ticket": order.ticket,
                    "volume": order.volume_current,
                    "entry_price": round(order.price_open, 6),
                    "sl": round(order.sl, 6) if order.sl != 0 else None,
                    "tp": round(order.tp, 6) if order.tp != 0 else None,
                    "setup_time": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "comment": order.comment.strip() if order.comment else None,
                    "magic": order.magic,
                    "age_seconds": age_seconds
                }

                if age_seconds > MAX_AGE_SECONDS:
                    to_delete.append((order.ticket, symbol, order_type_str, format_age(age_seconds)))

            if to_delete:
                log_and_print(f"Found {len(to_delete)} order(s) older than 2 days. Will delete after open position check.", "WARNING")
        else:
            log_and_print("No pending limit orders.", "INFO")

        # ---------- 2. OPEN POSITIONS ----------
        positions = mt5.positions_get()
        position_count = len(positions) if positions else 0
        total_positions += position_count

        if position_count:
            log_and_print(f"Found {position_count} open position(s).", "INFO")
            for pos in positions:
                symbol = pos.symbol
                pos_type_str = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                open_time = datetime.fromtimestamp(pos.time, TZ).strftime("%Y-%m-%d %H:%M:%S")

                if symbol not in broker_symbol_data[broker_name]:
                    broker_symbol_data[broker_name][symbol] = {
                        "has_open": True,
                        "pending": {"BUY": None, "SELL": None},
                        "account_login": account.login,
                        "account_currency": currency
                    }
                else:
                    broker_symbol_data[broker_name][symbol]["has_open"] = True

                all_open_positions.append({
                    "broker": broker_name,
                    "account_login": account.login,
                    "account_currency": currency,
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos_type_str,
                    "status": "OPEN",
                    "volume": pos.volume,
                    "entry_price": round(pos.price_open, 6),
                    "current_price": round(pos.price_current, 6),
                    "sl": round(pos.sl, 6) if pos.sl != 0 else None,
                    "tp": round(pos.tp, 6) if pos.tp != 0 else None,
                    "open_time": open_time,
                    "profit": round(pos.profit, 2),
                    "swap": round(pos.swap, 2),
                    "comment": pos.comment.strip() if pos.comment else None,
                    "magic": pos.magic
                })
        else:
            log_and_print("No open positions.", "INFO")

        # ---------- 3. HISTORY: RECENT FILLED/CANCELED LIMIT ORDERS (<5h) ----------
        from_datetime = datetime.now(TZ) - timedelta(seconds=MAX_HISTORY_SECONDS)
        to_datetime = datetime.now(TZ)

        # Convert to UTC timestamps
        from_ts = int(from_datetime.timestamp())
        to_ts = int(to_datetime.timestamp())

        history = mt5.history_orders_get(from_ts, to_ts) or []
        recent_limit_history = [
            h for h in history
            if h.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
        ]

        history_count = len(recent_limit_history)
        total_history += history_count

        if history_count:
            log_and_print(f"Found {history_count} recent limit order(s) in history (<5h).", "INFO")
            for h in recent_limit_history:
                symbol = h.symbol
                order_type_str = "BUY LIMIT" if h.type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL LIMIT"
                fill_time = datetime.fromtimestamp(h.time_done, TZ).strftime("%Y-%m-%d %H:%M:%S")
                age_seconds = (current_time - datetime.fromtimestamp(h.time_done, TZ)).total_seconds()
                age_str = format_age(age_seconds)

                status = "FILLED" if h.state == mt5.ORDER_STATE_FILLED else "CANCELED"

                entry = {
                    "broker": broker_name,
                    "account_login": account.login,
                    "account_currency": currency,
                    "ticket": h.ticket,
                    "symbol": symbol,
                    "type": order_type_str,
                    "status": status,
                    "volume": h.volume_current,
                    "entry_price": round(h.price_open, 6),
                    "fill_price": round(h.price_current, 6) if h.price_current != 0 else None,
                    "fill_time": fill_time,
                    "setup_time": datetime.fromtimestamp(h.time_setup, TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "comment": h.comment.strip() if h.comment else None,
                    "magic": h.magic,
                    "profit": round(h.profit, 2) if hasattr(h, 'profit') else None,
                    "age": age_str
                }
                all_history_orders.append(entry)
        else:
            log_and_print("No recent limit orders in history (<5h).", "INFO")

        mt5.shutdown()

        # ---------- 4. DELETE STALE PENDING ORDERS (>2 days) ----------
        if to_delete:
            log_and_print(f"Attempting to delete {len(to_delete)} stale limit order(s) on {broker_name.upper()}...", "INFO")

            if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD, server=SERVER, timeout=30000):
                log_and_print(f"Re-init failed: {mt5.last_error()}", "ERROR")
            elif not mt5.login(int(LOGIN_ID), password=PASSWORD, server=SERVER):
                log_and_print(f"Re-login failed: {mt5.last_error()}", "ERROR")
            else:
                for ticket, symbol, order_type, age_str in to_delete:
                    sym_data = broker_symbol_data[broker_name].get(symbol, {})
                    if sym_data.get("has_open", False):
                        log_and_print(f"SKIPPED: {symbol} [{order_type}] has open position", "INFO")
                        continue

                    current_orders = mt5.orders_get(ticket=ticket)
                    if not current_orders:
                        log_and_print(f"SKIP: Order {ticket} no longer exists", "INFO")
                        side = "BUY" if "BUY" in order_type else "SELL"
                        if broker_symbol_data[broker_name][symbol]["pending"][side]:
                            broker_symbol_data[broker_name][symbol]["pending"][side] = None
                        continue

                    request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                    result = mt5.order_send(request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        log_and_print(f"DELETED: {symbol} [{order_type}] | Ticket: {ticket} | Age: {age_str}", "SUCCESS")
                        deleted_count += 1
                        side = "BUY" if "BUY" in order_type else "SELL"
                        broker_symbol_data[broker_name][symbol]["pending"][side] = None
                    else:
                        log_and_print(f"FAILED: {symbol} [{order_type}] | Ticket: {ticket} | Error: {result.comment}", "ERROR")

                mt5.shutdown()

    # ========== POST-PROCESS: Build final pending list ==========
    log_and_print(f"\nProcessing age for remaining pending-only symbols...", "INFO")

    for broker_name, symbols_data in broker_symbol_data.items():
        for symbol, data in symbols_data.items():
            has_open = data["has_open"]
            pending = data["pending"]
            buy_data = pending["BUY"]
            sell_data = pending["SELL"]

            if not buy_data and not sell_data:
                continue

            if has_open:
                for side, order in [("BUY", buy_data), ("SELL", sell_data)]:
                    if order:
                        all_pending_orders.append({
                            "broker": broker_name,
                            "account_login": data["account_login"],
                            "account_currency": data["account_currency"],
                            "ticket": order["ticket"],
                            "symbol": symbol,
                            "type": f"{side} LIMIT",
                            "status": "PENDING",
                            "volume": order["volume"],
                            "entry_price": order["entry_price"],
                            "sl": order["sl"],
                            "tp": order["tp"],
                            "setup_time": order["setup_time"],
                            "comment": order["comment"],
                            "magic": order["magic"]
                        })
            else:
                for side, order in [("BUY", buy_data), ("SELL", sell_data)]:
                    if order:
                        age_str = format_age(order["age_seconds"])
                        all_pending_orders.append({
                            "broker": broker_name,
                            "account_login": data["account_login"],
                            "account_currency": data["account_currency"],
                            "ticket": order["ticket"],
                            "symbol": symbol,
                            "type": f"{side} LIMIT",
                            "status": "PENDING",
                            "volume": order["volume"],
                            "entry_price": order["entry_price"],
                            "sl": order["sl"],
                            "tp": order["tp"],
                            "setup_time": order["setup_time"],
                            "comment": order["comment"],
                            "magic": order["magic"],
                            "age": age_str
                        })

    # ========== FINAL SUMMARY ==========
    log_and_print(f"\n{'='*100}", "SUCCESS")
    log_and_print(f"COLLECTION COMPLETE", "SUCCESS")
    log_and_print(f"Total Brokers: {len(brokersdictionary)} | Failed: {len(failed_brokers)}", "INFO")
    if failed_brokers:
        log_and_print(f"Failed Brokers: {', '.join(failed_brokers)}", "WARNING")
    log_and_print(f"Pending Limit Orders (after cleanup): {len(all_pending_orders)}", "INFO")
    log_and_print(f"Open Positions: {total_positions}", "INFO")
    log_and_print(f"Recent History Orders (<5h): {total_history}", "INFO")
    log_and_print(f"Stale Orders Deleted (>2 days): {deleted_count}", "WARNING" if deleted_count else "INFO")
    log_and_print(f"Total Entries: {len(all_pending_orders) + total_positions + total_history}", "SUCCESS")

    # ========== SAVE TO JSON ==========
    report = {
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "total_brokers": len(brokersdictionary),
        "failed_brokers": failed_brokers,
        "cleanup": {
            "stale_orders_deleted": deleted_count,
            "max_age_allowed_seconds": MAX_AGE_SECONDS
        },
        "history_window_seconds": MAX_HISTORY_SECONDS,
        "summary": {
            "pending_orders": len(all_pending_orders),
            "open_positions": total_positions,
            "history_orders": total_history,
            "total": len(all_pending_orders) + total_positions + total_history
        },
        "pending_orders": all_pending_orders,
        "open_positions": all_open_positions,
        "history_orders": all_history_orders  # ← NEW
    }

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log_and_print(f"REPORT SAVED: {OUTPUT_PATH}", "SUCCESS")
    except Exception as e:
        log_and_print(f"FAILED TO SAVE REPORT: {e}", "ERROR")

    log_and_print(f"{'='*100}", "INFO")
    return True

def deduplicate_pending_orders():
    r"""
    Deduplicate pending BUY_LIMIT / SELL_LIMIT orders.
    Rules:
      1. Only ONE pending BUY_LIMIT per symbol
      2. Only ONE pending SELL_LIMIT per symbol
      3. If a BUY position is open → delete ALL pending BUY_LIMIT on that symbol
      4. If a SELL position is open → delete ALL pending SELL_LIMIT on that symbol
      5. When multiple pendings exist → use STRATEGY (lowtohigh/hightolow) to keep best price
         or keep oldest (lowest ticket) if no strategy.
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
    DEDUP_REPORT = "dedup_report.json"
    ISSUES_FILE = "ordersissues.json"

    # ------------------------------------------------------------------ #
    def _order_type_str(mt5_type):
        return "BUY_LIMIT" if mt5_type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL_LIMIT"

    def _decide_winner(existing, candidate, order_type, strategy):
        """Return (keep_existing, reason)"""
        is_buy = order_type == mt5.ORDER_TYPE_BUY_LIMIT

        if strategy == "lowtohigh":
            if is_buy:
                better = candidate["price"] > existing["price"]
                reason = f"lowtohigh → new {candidate['price']} > old {existing['price']}"
            else:
                better = candidate["price"] < existing["price"]
                reason = f"lowtohigh → new {candidate['price']} < old {existing['price']}"
        elif strategy == "hightolow":
            if is_buy:
                better = candidate["price"] < existing["price"]
                reason = f"hightolow → new {candidate['price']} < old {existing['price']}"
            else:
                better = candidate["price"] > existing["price"]
                reason = f"hightolow → new {candidate['price']} > old {existing['price']}"
        else:
            better = candidate["ticket"] < existing["ticket"]
            reason = f"no strategy → keep oldest ticket {candidate['ticket']} < {existing['ticket']}"

        return (not better, reason)  # True → keep existing

    # ------------------------------------------------------------------ #
    for broker_name, broker_cfg in brokersdictionary.items():
        account_type = broker_cfg.get("ACCOUNT", "").lower()
        if account_type not in ("demo", "real"):
            log_and_print(f"Skipping {broker_name} (account type: {account_type})", "INFO")
            continue

        strategy_key = broker_cfg.get("STRATEGY", "").lower()
        if strategy_key and strategy_key not in ("lowtohigh", "hightolow"):
            log_and_print(f"{broker_name}: Unknown STRATEGY '{strategy_key}' – using oldest ticket", "WARNING")
            strategy_key = ""

        TERMINAL_PATH = broker_cfg["TERMINAL_PATH"]
        LOGIN_ID      = broker_cfg["LOGIN_ID"]
        PASSWORD      = broker_cfg["PASSWORD"]
        SERVER        = broker_cfg["SERVER"]

        log_and_print(f"Deduplicating pending orders for {broker_name} ({account_type})", "INFO")

        # ------------------- MT5 connection -------------------
        if not os.path.exists(TERMINAL_PATH):
            log_and_print(f"{broker_name}: Terminal path missing", "ERROR")
            continue

        if not mt5.initialize(path=TERMINAL_PATH, login=int(LOGIN_ID), password=PASSWORD,
                              server=SERVER, timeout=30000):
            log_and_print(f"{broker_name}: MT5 init failed: {mt5.last_error()}", "ERROR")
            continue

        if not mt5.login(login=int(LOGIN_ID), password=PASSWORD, server=SERVER):
            log_and_print(f"{broker_name}: MT5 login failed: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            continue

        # ------------------- Get running positions -------------------
        running_positions = {}  # symbol → direction: 1=buy, -1=sell
        positions = mt5.positions_get()
        for pos in (positions or []):
            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            running_positions[pos.symbol] = direction

        # ------------------- Get pending orders -------------------
        pending = mt5.orders_get()
        pending_by_key = {}  # (symbol, type) → list of {'ticket':, 'price':}
        for order in (pending or []):
            if order.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue
            key = (order.symbol, order.type)
            pending_by_key.setdefault(key, []).append({
                "ticket": order.ticket,
                "price":  order.price_open
            })

        # ------------------- Deduplication -------------------
        total_deleted = total_kept = 0
        dedup_report = []
        issues_list   = []
        now_str = datetime.now(pytz.timezone("Africa/Lagos")).strftime(
            "%Y-%m-%d %H:%M:%S.%f+01:00")

        for (symbol, otype), orders in pending_by_key.items():
            new_dir = 1 if otype == mt5.ORDER_TYPE_BUY_LIMIT else -1
            type_str = _order_type_str(otype)

            # === RULE: If same-direction position is running → delete ALL pending of this type ===
            if symbol in running_positions and running_positions[symbol] == new_dir:
                for order in orders:
                    del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order["ticket"]}
                    del_res = mt5.order_send(del_req)

                    status = "DELETED"
                    err_msg = None
                    if del_res is None:
                        status = "DELETE FAILED (None)"
                        err_msg = "order_send returned None"
                    elif del_res.retcode != mt5.TRADE_RETCODE_DONE:
                        status = f"DELETE FAILED ({del_res.retcode})"
                        err_msg = del_res.comment

                    log_and_print(
                        f"{broker_name} | {symbol} {type_str} "
                        f"ticket {order['ticket']} @ {order['price']} → {status} "
                        f"(running { 'BUY' if new_dir==1 else 'SELL' } position)",
                        "INFO" if status == "DELETED" else "WARNING"
                    )

                    dedup_report.append({
                        "symbol": symbol,
                        "order_type": type_str,
                        "ticket": order["ticket"],
                        "price": order["price"],
                        "action": status.split()[0],
                        "reason": "Deleted: same-direction position already running",
                        "error_msg": err_msg,
                        "timestamp": now_str
                    })

                    if status == "DELETED":
                        total_deleted += 1
                    else:
                        issues_list.append({"symbol": symbol, "diagnosed_reason": f"Delete failed: {err_msg}"})
                continue  # skip to next symbol

            # === RULE: Only one pending per type → deduplicate if >1 ===
            if len(orders) <= 1:
                total_kept += 1
                continue

            # Sort by ticket (oldest first) for fallback
            orders.sort(key=lambda x: x["ticket"])

            keep = orders[0]
            for cand in orders[1:]:
                keep_it, reason = _decide_winner(keep, cand, otype, strategy_key)
                to_delete = cand if keep_it else keep

                del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": to_delete["ticket"]}
                del_res = mt5.order_send(del_req)

                status = "DELETED"
                err_msg = None
                if del_res is None:
                    status = "DELETE FAILED (None)"
                    err_msg = "order_send returned None"
                elif del_res.retcode != mt5.TRADE_RETCODE_DONE:
                    status = f"DELETE FAILED ({del_res.retcode})"
                    err_msg = del_res.comment

                log_and_print(
                    f"{broker_name} | {symbol} {type_str} "
                    f"ticket {to_delete['ticket']} @ {to_delete['price']} → {status} | {reason}",
                    "INFO" if status == "DELETED" else "WARNING"
                )

                dedup_report.append({
                    "symbol": symbol,
                    "order_type": type_str,
                    "ticket": to_delete["ticket"],
                    "price": to_delete["price"],
                    "action": status.split()[0],
                    "reason": reason,
                    "error_msg": err_msg,
                    "timestamp": now_str
                })

                if status == "DELETED":
                    total_deleted += 1
                    if not keep_it:
                        keep = cand  # promote winner
                else:
                    issues_list.append({"symbol": symbol, "diagnosed_reason": f"Delete failed: {err_msg}"})

            total_kept += 1  # one survivor

        # ------------------- Save reports -------------------
        broker_dir = Path(BASE_INPUT_DIR) / broker_name
        dedup_file = broker_dir / DEDUP_REPORT
        try:
            existing = json.load(dedup_file.open("r", encoding="utf-8")) if dedup_file.exists() else []
        except:
            existing = []
        all_report = existing + dedup_report
        try:
            with dedup_file.open("w", encoding="utf-8") as f:
                json.dump(all_report, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to write {DEDUP_REPORT}: {e}", "ERROR")

        issues_path = broker_dir / ISSUES_FILE
        try:
            existing_issues = json.load(issues_path.open("r", encoding="utf-8")) if issues_path.exists() else []
            with issues_path.open("w", encoding="utf-8") as f:
                json.dump(existing_issues + issues_list, f, indent=2)
        except Exception as e:
            log_and_print(f"{broker_name}: Failed to update {ISSUES_FILE}: {e}", "ERROR")

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Deduplication complete – Kept: {total_kept}, Deleted: {total_deleted}",
            "SUCCESS"
        )

    log_and_print("All brokers deduplicated successfully.", "SUCCESS")

def BreakevenRunningPositions():
    r"""
    Staged Breakeven:
      • Ratio 1 → SL to 0.25 (actual price shown)
      • Ratio 2 → SL to 0.50 (actual price shown)
    Clean logs, full precision, MT5-safe.
    """
    BASE_INPUT_DIR = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices"
    BREAKEVEN_REPORT = "breakeven_report.json"
    ISSUES_FILE = "ordersissues.json"

    # === BREAKEVEN STAGES ===
    BE_STAGE_1 = 0.25   # SL moves here at ratio 1
    BE_STAGE_2 = 0.50   # SL moves here at ratio 2
    RATIO_1 = 1.0
    RATIO_2 = 2.0

    # === Helper: Round to symbol digits ===
    def _round_price(price, symbol):
        digits = mt5.symbol_info(symbol).digits
        return round(price, digits)

    # === Helper: Price at ratio ===
    def _ratio_price(entry, sl, tp, ratio, is_buy):
        risk = abs(entry - sl) or 1e-9
        return entry + risk * ratio * (1 if is_buy else -1)

    # === Helper: Modify SL ===
    def _modify_sl(pos, new_sl_raw):
        new_sl = _round_price(new_sl_raw, pos.symbol)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp,
            "magic": pos.magic,
            "comment": pos.comment
        }
        return mt5.order_send(req)

    # === Helper: Print block ===
    def _log_block(lines):
        log_and_print("\n".join(lines), "INFO")

    # === Helper: Safe JSON read (handles corrupted/multi-object files) ===
    def _safe_read_json(path):
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                # Handle multiple JSON objects by parsing line-by-line
                objs = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            objs.extend(obj)
                        elif isinstance(obj, dict):
                            objs.append(obj)
                    except json.JSONDecodeError:
                        continue
                return objs
        except Exception as e:
            log_and_print(f"Failed to read {path.name}: {e}. Starting fresh.", "WARNING")
            return []

    # === Helper: Safe JSON write ===
    def _safe_write_json(path, data):
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Ensure file ends cleanly
            return True
        except Exception as e:
            log_and_print(f"Failed to write {path.name}: {e}", "ERROR")
            return False

    # ------------------------------------------------------------------ #
    for broker_name, cfg in brokersdictionary.items():
        # ---- MT5 Connection ------------------------------------------------
        if not mt5.initialize(path=cfg["TERMINAL_PATH"], login=int(cfg["LOGIN_ID"]),
                              password=cfg["PASSWORD"], server=cfg["SERVER"], timeout=30000):
            log_and_print(f"{broker_name}: MT5 init failed", "ERROR")
            continue
        if not mt5.login(int(cfg["LOGIN_ID"]), cfg["PASSWORD"], cfg["SERVER"]):
            log_and_print(f"{broker_name}: MT5 login failed", "ERROR")
            mt5.shutdown()
            continue

        broker_dir = Path(BASE_INPUT_DIR) / broker_name
        report_path = broker_dir / BREAKEVEN_REPORT
        issues_path = broker_dir / ISSUES_FILE

        # Load existing report (unchanged)
        existing_report = []
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    existing_report = json.load(f)
            except Exception as e:
                log_and_print(f"{broker_name}: Failed to load breakeven_report.json – {e}", "WARNING")

        issues = []
        now = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S.%f%z")
        now = f"{now[:-2]}:{now[-2:]}"  # Format +01:00 properly
        updated = pending_info = 0

        positions = mt5.positions_get() or []
        pending   = mt5.orders_get()   or []

        # ---- Group pending orders by symbol ----
        pending_by_sym = {}
        for o in pending:
            if o.type not in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                continue
            pending_by_sym.setdefault(o.symbol, {})[o.type] = {
                "price": o.price_open, "sl": o.sl, "tp": o.tp
            }

        # ==================================================================
        # === PROCESS RUNNING POSITIONS ===
        # ==================================================================
        for pos in positions:
            if pos.sl == 0 or pos.tp == 0:
                continue

            sym = pos.symbol
            tick = mt5.symbol_info_tick(sym)
            info = mt5.symbol_info(sym)
            if not tick or not info:
                continue

            cur_price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            typ = "BUY" if is_buy else "SELL"

            # Key levels
            r1_price = _ratio_price(pos.price_open, pos.sl, pos.tp, RATIO_1, is_buy)
            r2_price = _ratio_price(pos.price_open, pos.sl, pos.tp, RATIO_2, is_buy)
            be_025   = _ratio_price(pos.price_open, pos.sl, pos.tp, BE_STAGE_1, is_buy)
            be_050   = _ratio_price(pos.price_open, pos.sl, pos.tp, BE_STAGE_2, is_buy)

            stage1 = (cur_price >= r1_price) if is_buy else (cur_price <= r1_price)
            stage2 = (cur_price >= r2_price) if is_buy else (cur_price <= r2_price)

            # Base block
            block = [
                f"┌─ {broker_name} ─ {sym} ─ {typ} (ticket {pos.ticket})",
                f"│ Entry : {pos.price_open:.{info.digits}f}   SL : {pos.sl:.{info.digits}f}   TP : {pos.tp:.{info.digits}f}",
                f"│ Now   : {cur_price:.{info.digits}f}"
            ]

            # === STAGE 2: SL to 0.50 ===
            if stage2 and abs(pos.sl - be_050) > info.point:
                res = _modify_sl(pos, be_050)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    block += [
                        f"│ BE @ 0.25 → {be_025:.{info.digits}f}",
                        f"│ BE @ 0.50 → {be_050:.{info.digits}f}  ← SL MOVED",
                        f"└─ All left to market"
                    ]
                    updated += 1
                else:
                    issues.append({"symbol": sym, "diagnosed_reason": "SL modify failed (stage 2)"})
                    block.append(f"└─ SL move FAILED")
                _log_block(block)
                continue

            # === STAGE 1: SL to 0.25 ===
            if stage1 and abs(pos.sl - be_025) > info.point:
                res = _modify_sl(pos, be_025)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    block += [
                        f"│ BE @ 0.25 → {be_025:.{info.digits}f}  ← SL MOVED",
                        f"│ Waiting ratio 2 @ {r2_price:.{info.digits}f} → BE @ 0.50 → {be_050:.{info.digits}f}"
                    ]
                    updated += 1
                else:
                    issues.append({"symbol": sym, "diagnosed_reason": "SL modify failed (stage 1)"})
                    block.append(f"└─ SL move FAILED")
                _log_block(block)
                continue

            # === STAGE 1 REACHED, WAITING STAGE 2 ===
            if stage1:
                block += [
                    f"│ BE @ 0.25 → {be_025:.{info.digits}f}",
                    f"│ Waiting ratio 2 @ {r2_price:.{info.digits}f} → BE @ 0.50 → {be_050:.{info.digits}f}"
                ]
            # === WAITING STAGE 1 ===
            else:
                block += [
                    f"│ Waiting ratio 1 @ {r1_price:.{info.digits}f} → BE @ 0.25 → {be_025:.{info.digits}f}"
                ]

            block.append("")
            _log_block(block)

        # ==================================================================
        # === PROCESS PENDING ORDERS (INFO ONLY) ===
        # ==================================================================
        for sym, orders in pending_by_sym.items():
            for otype, o in orders.items():
                if o["sl"] == 0 or o["tp"] == 0:
                    continue
                info = mt5.symbol_info(sym)
                if not info:
                    continue
                is_buy = otype == mt5.ORDER_TYPE_BUY_LIMIT
                typ = "BUY_LIMIT" if is_buy else "SELL_LIMIT"

                r1_price = _ratio_price(o["price"], o["sl"], o["tp"], RATIO_1, is_buy)
                r2_price = _ratio_price(o["price"], o["sl"], o["tp"], RATIO_2, is_buy)
                be_025   = _ratio_price(o["price"], o["sl"], o["tp"], BE_STAGE_1, is_buy)
                be_050   = _ratio_price(o["price"], o["sl"], o["tp"], BE_STAGE_2, is_buy)

                block = [
                    f"┌─ {broker_name} ─ {sym} ─ PENDING {typ}",
                    f"│ Entry : {o['price']:.{info.digits}f}   SL : {o['sl']:.{info.digits}f}   TP : {o['tp']:.{info.digits}f}",
                    f"│ Target 1 → {r1_price:.{info.digits}f}  |  BE @ 0.25 → {be_025:.{info.digits}f}",
                    f"│ Target 2 → {r2_price:.{info.digits}f}  |  BE @ 0.50 → {be_050:.{info.digits}f}",
                    f"└─ Order not running – waiting…"
                ]
                _log_block(block)
                pending_info += 1

        # === SAVE BREAKEVEN REPORT (unchanged) ===
        _safe_write_json(report_path, existing_report)

        # === SAVE ISSUES – ROBUST MERGE ===
        current_issues = _safe_read_json(issues_path)
        all_issues = current_issues + issues
        _safe_write_json(issues_path, all_issues)

        mt5.shutdown()
        log_and_print(
            f"{broker_name}: Breakeven done – SL Updated: {updated} | Pending Info: {pending_info}",
            "SUCCESS"
        )

    log_and_print("All brokers breakeven processed.", "SUCCESS")

def martingale_enforcement():
    """
    MARTINGALE ENFORCER v5.2 – SMART KILL + REAL HISTORY SCALING
    ------------------------------------------------------------
    • Kills unwanted pending orders
    • Uses mt5.history_deals_get() with smart filtering
    • Checks last 2 closed trades per symbol
    • Scales pending limit order volume ×2 for each losing symbol
    • Delete + recreate if volume change needed
    • Works on Bybit MT5 (tested with real history)
    """
    import time
    from collections import defaultdict, deque
    from datetime import datetime, timedelta

    log_and_print(f"\n{'='*100}", "INFO")
    log_and_print("MARTINGALE ENFORCER v5.2 – SMART KILL + HISTORY SCALING", "INFO")
    log_and_print(f"{'='*100}", "INFO")

    for broker_name, cfg in brokersdictionary.items():
        SCALE = (cfg.get("SCALE") or cfg.get("scale") or "").lower()
        if SCALE != "martingale":
            continue

        TERMINAL_PATH = cfg["TERMINAL_PATH"]
        LOGIN_ID      = int(cfg["LOGIN_ID"])
        PASSWORD      = cfg["PASSWORD"]
        SERVER        = cfg["SERVER"]
        raw           = cfg.get("MARTINGALE_MARKETS", "")
        allowed       = {s.strip().lower() for s in raw.replace(",", " ").split() if s.strip()}

        if not allowed:
            continue

        log_and_print(f"\n{broker_name.upper()} → LOCKING TO: {', '.join(sorted(allowed)).upper()}", "INFO")

        # ------------------------------------------------------------------ #
        # 1. CONNECT / RECONNECT
        # ------------------------------------------------------------------ #
        def connect():
            mt5.shutdown()
            time.sleep(0.3)
            if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN_ID,
                                  password=PASSWORD, server=SERVER, timeout=60000):
                return False
            if not mt5.login(LOGIN_ID, password=PASSWORD, server=SERVER):
                return False
            time.sleep(0.7)
            return True

        if not connect():
            log_and_print("INITIAL CONNECTION FAILED", "ERROR")
            continue

        # ------------------------------------------------------------------ #
        # 2. KILL UNWANTED PENDING ORDERS
        # ------------------------------------------------------------------ #
        def get_orders():
            return mt5.orders_get() or []

        orders = get_orders()
        unwanted = [
            o for o in orders
            if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)
            and o.symbol.lower() not in allowed
        ]

        killed = skipped = failed = 0
        for order in unwanted:
            symbol = order.symbol
            ticket = order.ticket
            log_and_print(f"{symbol} PENDING → Attempting removal...", "WARNING")

            if not connect():
                log_and_print(f"{symbol} → Reconnect failed", "ERROR")
                failed += 1
                continue

            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
            res = mt5.order_send(req)

            if not res:
                log_and_print(f"{symbol} → No response", "ERROR")
                failed += 1
                continue

            if res.retcode == mt5.TRADE_RETCODE_DONE:
                log_and_print(f"{symbol} → REMOVED", "SUCCESS")
                killed += 1
            elif "market closed" in res.comment.lower():
                log_and_print(f"{symbol} → Market closed → SKIPPED (safe)", "INFO")
                skipped += 1
            elif res.retcode in (mt5.TRADE_RETCODE_TRADE_DISABLED, mt5.TRADE_RETCODE_NO_CONNECTION):
                log_and_print(f"{symbol} → {res.comment} → SKIPPED", "INFO")
                skipped += 1
            else:
                log_and_print(f"{symbol} → FAILED: {res.comment}", "ERROR")
                failed += 1
            time.sleep(0.4)

        # ------------------------------------------------------------------ #
        # 3. GET CLOSED HISTORY (LAST 2 TRADES PER SYMBOL)
        # ------------------------------------------------------------------ #
        if not connect():
            mt5.shutdown()
            continue

        # Pull recent deals (last 24h should be enough)
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=24)
        all_deals = mt5.history_deals_get(from_date, to_date) or []

        # Filter: only closed positions (DEAL_ENTRY_OUT) and our symbols
        closed_deals = [
            d for d in all_deals
            if d.entry == mt5.DEAL_ENTRY_OUT
            and d.symbol.lower() in allowed
            and d.profit is not None
        ]

        # Sort newest first
        closed_deals.sort(key=lambda x: x.time, reverse=True)

        log_and_print(f"Found {len(closed_deals)} closed deal(s) in last 24h for Martingale markets", "INFO")

        # Build: symbol → list of (deal, volume, profit) — newest first
        history_per_symbol = defaultdict(list)
        for deal in closed_deals:
            sym = deal.symbol.lower()
            history_per_symbol[sym].append({
                'deal': deal,
                'volume': deal.volume,
                'profit': deal.profit,
                'time': deal.time
            })

        # ------------------------------------------------------------------ #
        # 4. DETERMINE WHICH SYMBOLS TO SCALE
        # ------------------------------------------------------------------ #
        symbols_to_scale = {}  # sym → (original_volume, price, order_type)

        # We look at **last 2 closed trades globally**, but per symbol
        recent_losses = []
        for deal in closed_deals[:10]:  # safety cap
            if deal.profit < 0:
                recent_losses.append({
                    'symbol': deal.symbol.lower(),
                    'volume': deal.volume,
                    'profit': deal.profit,
                    'time': deal.time
                })
            if len(recent_losses) >= 2:
                break

        log_and_print(f"Last {len(recent_losses)} losing trade(s): {[d['symbol'].upper() for d in recent_losses]}", "INFO")

        # Rule: If last 2 are losses → scale both (if different), or only last (if same)
        if len(recent_losses) >= 1:
            last = recent_losses[0]
            sym1 = last['symbol']
            vol1 = last['volume']

            # Find pending order
            pending = [o for o in get_orders() if o.symbol.lower() == sym1
                       and o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]

            if pending:
                order = pending[0]
                if order.volume_current < vol1 * 2:
                    symbols_to_scale[sym1] = (vol1, order.price_open, order.type)
                    log_and_print(f"{sym1.upper()} → Last loss {vol1} → will scale pending to {vol1*2}", "INFO")
                else:
                    log_and_print(f"{sym1.upper()} → Already scaled (current {order.volume_current} ≥ {vol1*2})", "INFO")
            else:
                log_and_print(f"{sym1.upper()} → No pending order → cannot scale", "INFO")

            # If 2nd loss exists and is DIFFERENT symbol
            if len(recent_losses) >= 2:
                second = recent_losses[1]
                sym2 = second['symbol']
                vol2 = second['volume']

                if sym2 != sym1:
                    pending2 = [o for o in get_orders() if o.symbol.lower() == sym2
                                and o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]
                    if pending2:
                        order2 = pending2[0]
                        if order2.volume_current < vol2 * 2:
                            symbols_to_scale[sym2] = (vol2, order2.price_open, order2.type)
                            log_and_print(f"{sym2.upper()} → 2nd loss {vol2} → will scale pending to {vol2*2}", "INFO")
                        else:
                            log_and_print(f"{sym2.upper()} → Already scaled", "INFO")
                    else:
                        log_and_print(f"{sym2.upper()} → No pending order → cannot scale", "INFO")

        # ------------------------------------------------------------------ #
        # 5. APPLY SCALING: DELETE + RECREATE
        # ------------------------------------------------------------------ #
        scaled = not_scaled = 0
        for sym, (orig_vol, price, order_type) in symbols_to_scale.items():
            if not connect():
                log_and_print(f"{sym.upper()} → Reconnect failed before scaling", "ERROR")
                continue

            # Re-get orders
            current_orders = get_orders()
            pending = [o for o in current_orders if o.symbol.lower() == sym
                       and o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT)]

            if not pending:
                log_and_print(f"{sym.upper()} → Pending order vanished → SKIPPED", "WARNING")
                not_scaled += 1
                continue

            order = pending[0]
            new_vol = orig_vol * 2

            if order.volume_current >= new_vol:
                log_and_print(f"{sym.upper()} → Already at {order.volume_current} → SKIPPED", "INFO")
                not_scaled += 1
                continue

            # DELETE
            del_req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            del_res = mt5.order_send(del_req)
            if del_res.retcode != mt5.TRADE_RETCODE_DONE:
                log_and_print(f"{sym.upper()} → DELETE FAILED: {del_res.comment}", "ERROR")
                continue

            time.sleep(0.3)

            # RECREATE
            new_req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": sym.upper(),
                "volume": new_vol,
                "type": order_type,
                "price": price,
                "sl": order.sl,
                "tp": order.tp,
                "deviation": 20,
                "magic": order.magic,
                "comment": f"MartingaleScaled_{new_vol}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            new_res = mt5.order_send(new_req)
            if new_res and new_res.retcode == mt5.TRADE_RETCODE_DONE:
                log_and_print(f"{sym.upper()} → SCALED {order.volume_current} → {new_vol} @ {price}", "SUCCESS")
                scaled += 1
            else:
                comment = new_res.comment if new_res else "None"
                log_and_print(f"{sym.upper()} → PLACE FAILED: {comment}", "ERROR")

            time.sleep(0.5)

        # ------------------------------------------------------------------ #
        # 6. 1R ENFORCEMENT (placeholder)
        # ------------------------------------------------------------------ #
        if connect():
            for pos in mt5.positions_get() or []:
                if pos.symbol.lower() in allowed:
                    pass  # ← your 1R logic

        mt5.shutdown()

        # ------------------------------------------------------------------ #
        # 7. FINAL REPORT
        # ------------------------------------------------------------------ #
        log_and_print(f"\n{broker_name.upper()} → ENFORCEMENT COMPLETE", "SUCCESS")
        log_and_print(f"   REMOVED     : {killed}", "SUCCESS")
        log_and_print(f"   SKIPPED     : {skipped} (market closed / safe)", "INFO")
        log_and_print(f"   Failed      : {failed}", "WARNING")
        log_and_print(f"   SCALED      : {scaled}", "SUCCESS")
        log_and_print(f"   NOT SCALED  : {not_scaled}", "INFO")

    log_and_print("\nMARTINGALE v5.2 → HISTORY CHECKED. SCALED. DONE.", "SUCCESS")
    return True


def main():
    _12_20_orders()
    _0_50_4_orders()
    _4_8_orders()
    _8_12_orders()
    _20_80_orders()
    _80_160_orders()
    _160_320_orders()
    deduplicate_pending_orders()
    collect_all_brokers_limit_orders()
    martingale_enforcement()

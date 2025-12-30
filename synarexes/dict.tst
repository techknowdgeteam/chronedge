def fetch_ohlcv_data(symbol, mt5_timeframe, bars):
    """Fetch OHLCV data for a given symbol and timeframe."""
    error_log = []
    if not mt5.symbol_select(symbol, True):
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to select symbol {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to select symbol {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve rates for {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to retrieve rates for {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "tick_volume": float, "spread": int, "real_volume": float
    })
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    log_and_print(f"OHLCV data fetched for {symbol}", "INFO")
    return df, error_log
def generate_and_save_oldest_newest_chart(df, symbol, timeframe_str, timeframe_folder, neighborcandles_left, neighborcandles_right):
    """Generate and save a basic candlestick chart as chart.png, then identify PH/PL and save as chartanalysed.png with markers."""
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "oldest_newest.png")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    trendline_log = []

    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green",
                down="red",
                edge="inherit",
                wick={"up": "green", "down": "red"},
                volume="gray"
            )
        )

        # Step 1: Save basic candlestick chart as chart.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str})",
            returnfig=True,
            warn_too_much_data=5000  # Add this line
        )

        # Adjust wick thickness for basic chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        log_and_print(f"Basic chart saved for {symbol} ({timeframe_str}) as {chart_path}", "SUCCESS")

        # Step 2: Identify PH/PL
        ph_labels, pl_labels, phpl_errors = identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right)
        error_log.extend(phpl_errors)

        # Step 3: Prepare annotations for analyzed chart with PH/PL markers
        apds = []
        if ph_labels:
            ph_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ph_labels:
                ph_series.loc[t] = price
            apds.append(mpf.make_addplot(
                ph_series,
                type='scatter',
                markersize=100,
                marker='^',
                color='blue'
            ))
        if pl_labels:
            pl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in pl_labels:
                pl_series.loc[t] = price
            apds.append(mpf.make_addplot(
                pl_series,
                type='scatter',
                markersize=100,
                marker='v',
                color='purple'
            ))

        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "team_type": "initial",
            "status": "info",
            "reason": f"Found {len(ph_labels)} PH points and {len(pl_labels)} PL points",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })

        # Save Trendline Log (only PH/PL info, no trendlines)
        try:
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(f"Trendline log saved for {symbol} ({timeframe_str})", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        # Step 4: Save analyzed chart with PH/PL markers as chartanalysed.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str}) - Analysed",
            addplot=apds if apds else None,
            returnfig=True
        )

        # Adjust wick thickness for analyzed chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(True, linestyle='--')
        fig.savefig(chart_analysed_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        log_and_print(f"Analysed chart saved for {symbol} ({timeframe_str}) as {chart_analysed_path}", "SUCCESS")

        return chart_path, error_log, ph_labels, pl_labels
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "status": "failed",
            "reason": f"Chart generation failed: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        with open(trendline_log_json_path, 'w') as f:
            json.dump(trendline_log, f, indent=4)
        save_errors(error_log)
        log_and_print(f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        return chart_path if os.path.exists(chart_path) else None, error_log, [], []
def fetch_charts_all_brokers(
    bars,
    neighborcandles_left,
    neighborcandles_right
):
    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    backup_developers_dictionary()
    delete_all_category_jsons()
    delete_all_calculated_risk_jsons()
    delete_issue_jsons()
    clear_unknown_broker()
    required_allowed_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    fallback_allowed_path = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allowedmarkets.json"
    allsymbols_path       = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\allsymbolsvolumesandrisk.json"
    match_path            = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\symbolsmatch.json"
    brokers_report_path   = r"C:\xampp\htdocs\chronedge\synarex\chart\symbols_volumes_points\allowedmarkets\brokerslimitorders.json"

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    IMPORTANT_TFS = {"15m", "30m", "1h", "4h"}

    def normalize_broker_key(name: str) -> str:
        """Normalize broker name: remove digits, spaces, case-insensitive"""
        return re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', '', name.strip())).lower()

    def clean_folder_name(name: str) -> str:
        """Convert 'Deriv 2', 'deriv6', 'Bybit 10' → 'Deriv', 'Bybit' (Title case)"""
        cleaned = re.sub(r'\d+', '', re.sub(r'[\/\s\-_]+', ' ', name.strip()))
        return cleaned.strip().title()

    def normalize_symbol(s: str) -> str:
        return re.sub(r'[\/\s\-_]+', '', s.strip()).upper() if s else ""

    def symbol_needs_processing(symbol: str, base_folder: str) -> bool:
        log_and_print(f"QUEUED {symbol} → will be processed", "INFO")
        return True

    def delete_symbol_folder(symbol: str, base_folder: str, reason: str = ""):
        sym_folder = os.path.join(base_folder, symbol.replace(" ", "_"))
        if os.path.exists(sym_folder):
            try:
                shutil.rmtree(sym_folder)
                log_and_print(f"DELETED {sym_folder} {reason}", "INFO")
            except Exception as e:
                log_and_print(f"FAILED to delete {sym_folder}: {e}", "ERROR")
        os.makedirs(base_folder, exist_ok=True)

    def delete_all_non_blocked_symbol_folders(broker_cfg: dict, blocked_symbols: set):
        base_folder = broker_cfg["BASE_FOLDER"]
        if not os.path.exists(base_folder):
            return
        deleted = 0
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if not os.path.isdir(item_path):
                continue
            symbol = item.replace("_", " ")
            if symbol in blocked_symbols:
                log_and_print(f"KEEPING folder {item} → {symbol} is BLOCKED", "INFO")
                continue
            try:
                shutil.rmtree(item_path)
                deleted += 1
            except Exception as e:
                log_and_print(f"FAILED to delete {item_path}: {e}", "ERROR")
        log_and_print(f"CLEANED {deleted} non-blocked symbol folders in {base_folder}", "SUCCESS")

    def mark_chosen_broker(original_broker_key: str, user_brokerid: str, balance: float):
        """Create chosenbroker.json in symbols_calculated_prices\<original_key>\chosenbroker.json"""
        target_dir = fr"C:\xampp\htdocs\chronedge\synarex\chart\symbols_calculated_prices\{original_broker_key}"
        os.makedirs(target_dir, exist_ok=True)
        chosen_path = os.path.join(target_dir, "chosenbroker.json")
        
        chosen_data = {
            "chosen": True,
            "broker_display_name": user_brokerid,
            "original_key": original_broker_key,
            "balance": round(balance, 2),
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "Highest balance among same broker type"
        }
        
        try:
            with open(chosen_path, "w", encoding="utf-8") as f:
                json.dump(chosen_data, f, indent=4)
            log_and_print(f"MARKED AS CHOSEN → {chosen_path} (Balance: {balance})", "SUCCESS")
        except Exception as e:
            log_and_print(f"FAILED to write chosenbroker.json for {original_broker_key}: {e}", "ERROR")

    def breakeven_worker():
        while True:
            try:
                BreakevenRunningPositions()
            except Exception as e:
                log_and_print(f"BREAKEVEN ERROR: {e}", "CRITICAL")
            time.sleep(10)

    threading.Thread(target=breakeven_worker, daemon=True).start()
    log_and_print("Breakeven thread started", "SUCCESS")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    while True:
        error_log = []
        log_and_print("\n=== NEW FULL CYCLE STARTED ===", "INFO")

        try:
            # ------------------------------------------------------------------
            # 0. LOAD AND AGGREGATE BLOCKED SYMBOLS BY NORMALIZED BROKER KEY
            # ------------------------------------------------------------------
            normalized_blocked_symbols = {}  # normalized_broker -> set of blocked symbols

            if os.path.exists(brokers_report_path):
                try:
                    with open(brokers_report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)

                    for section in ["pending_orders", "open_positions", "history_orders"]:
                        items = report.get(section, [])
                        for item in items:
                            broker_raw = item.get("broker", "")
                            symbol = item.get("symbol", "")
                            if not broker_raw or not symbol:
                                continue

                            norm_broker = normalize_broker_key(broker_raw)
                            normalized_blocked_symbols.setdefault(norm_broker, set())

                            if section == "history_orders":
                                age_str = item.get("age", "")
                                if "d" in age_str:
                                    days = int(age_str.split("d")[0])
                                    if days >= 5:
                                        continue
                                elif not any(t in age_str for t in ["h", "m", "s"]):
                                    continue

                            normalized_blocked_symbols[norm_broker].add(symbol)

                    log_and_print(f"Loaded & merged blocked symbols from {len(normalized_blocked_symbols)} unique brokers", "INFO")
                except Exception as e:
                    log_and_print(f"FAILED to load brokerslimitorders.json: {e}", "ERROR")

            # ------------------------------------------------------------------
            # 1. SELECT ONLY ONE BROKER PER UNIQUE TYPE (HIGHEST BALANCE) + MARK CHOSEN
            # ------------------------------------------------------------------
            selected_brokers = {}  # normalized_key -> (original_name, config, balance, original_dict_key)

            for original_key, cfg in developersdictionary.items():  # original_key = "deriv2", "bybit10", etc.
                user_brokerid = cfg.get("original_name", original_key)  # fallback if not set
                norm_key = normalize_broker_key(user_brokerid)

                balance = 0.0
                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if ok:
                    try:
                        account_info = mt5.account_info()
                        if account_info:
                            balance = account_info.balance
                    except:
                        pass
                    mt5.shutdown()

                current = selected_brokers.get(norm_key)
                if current is None or balance > current[2]:
                    cfg_copy = cfg.copy()
                    cfg_copy["balance"] = balance
                    cfg_copy["original_name"] = user_brokerid
                    selected_brokers[norm_key] = (user_brokerid, cfg_copy, balance, original_key)

            # Now mark all selected brokers as "chosen" with their original dictionary key
            unique_brokers = {}
            for norm_key, (user_brokerid, cfg, balance, original_key) in selected_brokers.items():
                unique_brokers[user_brokerid] = cfg
                mark_chosen_broker(original_key, user_brokerid, balance)  # <-- THIS IS THE NEW FEATURE

            log_and_print(f"Selected & MARKED {len(unique_brokers)} unique brokers (highest balance): {list(unique_brokers.keys())}", "SUCCESS")

            # ------------------------------------------------------------------
            # 0.5 DELETE NON-BLOCKED FOLDERS FOR SELECTED BROKERS
            # ------------------------------------------------------------------
            for bn, cfg in unique_brokers.items():
                norm_key = normalize_broker_key(bn)
                blocked = normalized_blocked_symbols.get(norm_key, set())
                delete_all_non_blocked_symbol_folders(cfg, blocked)

            # ------------------------------------------------------------------
            # 1. Load allowed markets
            # ------------------------------------------------------------------
            if not os.path.exists(required_allowed_path):
                if os.path.exists(fallback_allowed_path):
                    os.makedirs(os.path.dirname(required_allowed_path), exist_ok=True)
                    shutil.copy2(fallback_allowed_path, required_allowed_path)
                    log_and_print("AUTO-COPIED allowedmarkets.json", "INFO")
                else:
                    log_and_print("CRITICAL: allowedmarkets.json missing!", "CRITICAL")
                    time.sleep(600); continue

            with open(required_allowed_path, "r", encoding="utf-8") as f:
                allowed_config = json.load(f)

            normalized_allowed = {
                cat: {normalize_symbol(s) for s in cfg.get("allowed", [])}
                for cat, cfg in allowed_config.items()
            }

            # ------------------------------------------------------------------
            # 2. Symbol → category map
            # ------------------------------------------------------------------
            if not os.path.exists(allsymbols_path):
                log_and_print(f"Missing {allsymbols_path}", "CRITICAL")
                time.sleep(600); continue

            symbol_to_category = {}
            with open(allsymbols_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for markets in data.values():
                for cat in markets:
                    for item in markets.get(cat, []):
                        if sym := item.get("symbol"):
                            norm = normalize_symbol(sym)
                            symbol_to_category[norm] = cat
                            symbol_to_category[sym] = cat

            # ------------------------------------------------------------------
            # 3. Load symbolsmatch
            # ------------------------------------------------------------------
            if not os.path.exists(match_path):
                log_and_print(f"Missing {match_path}", "CRITICAL")
                time.sleep(600); continue
            with open(match_path, "r", encoding="utf-8") as f:
                symbolsmatch_data = json.load(f)

            # ------------------------------------------------------------------
            # 5. Build candidate list — ONLY UNIQUE BROKERS
            # ------------------------------------------------------------------
            all_cats = ["stocks","forex","crypto","synthetics","indices","commodities","equities","energies","etfs","basket_indices","metals"]
            candidates = {}
            total_to_do = 0

            for user_brokerid, cfg in unique_brokers.items():
                norm_key = normalize_broker_key(user_brokerid)
                blocked = normalized_blocked_symbols.get(norm_key, set())
                candidates[user_brokerid] = {c: [] for c in all_cats}

                broker_symbols_raw = cfg.get("SYMBOLS", "").strip()
                broker_allowed_symbols = None
                if broker_symbols_raw and broker_symbols_raw.lower() != "all":
                    broker_allowed_symbols = {normalize_symbol(s) for s in broker_symbols_raw.split(",") if s.strip()}

                ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                error_log.extend(errs)
                if not ok:
                    mt5.shutdown(); continue
                avail, _ = get_symbols()
                mt5.shutdown()

                for entry in symbolsmatch_data.get("main_symbols", []):
                    canonical = entry.get("symbol")
                    if not canonical:
                        continue
                    norm_canonical = normalize_symbol(canonical)

                    found = False
                    broker_symbols_list = []
                    for possible_key in [norm_key, norm_key.title(), norm_key.upper()]:
                        if possible_key in entry:
                            broker_symbols_list = entry.get(possible_key, [])
                            found = True
                            break
                    if not found:
                        continue

                    for sym_mt5 in broker_symbols_list:
                        if sym_mt5 not in avail or sym_mt5 in blocked:
                            continue

                        cat = symbol_to_category.get(norm_canonical) or symbol_to_category.get(canonical)
                        if not cat or cat not in all_cats:
                            continue

                        if allowed_config.get(cat, {}).get("limited", False):
                            if norm_canonical not in normalized_allowed.get(cat, set()):
                                continue

                        if broker_allowed_symbols is not None and norm_canonical not in broker_allowed_symbols:
                            continue

                        if symbol_needs_processing(sym_mt5, cfg["BASE_FOLDER"]):
                            delete_symbol_folder(sym_mt5, cfg["BASE_FOLDER"], "(pre-process cleanup)")
                            candidates[user_brokerid][cat].append(sym_mt5)

                for cat in all_cats:
                    cnt = len(candidates[user_brokerid][cat])
                    if cnt:
                        log_and_print(f"{user_brokerid.upper()} → {cat.upper():10} : {cnt:3} queued", "INFO")
                        total_to_do += cnt

            if total_to_do == 0:
                log_and_print("No symbols to process – sleeping 30 min", "WARNING")
                time.sleep(1800)
                continue

            log_and_print(f"TOTAL TO PROCESS: {total_to_do}", "SUCCESS")

            # ------------------------------------------------------------------
            # 6. ROUND-ROBIN PROCESSING ACROSS UNIQUE BROKERS ONLY
            # ------------------------------------------------------------------
            remaining = {b: {c: candidates[b][c][:] for c in all_cats} for b in unique_brokers}
            indices   = {b: {c: 0 for c in all_cats} for b in unique_brokers}

            round_no = 1
            while any(any(remaining[b][c]) for b in unique_brokers for c in all_cats):
                log_and_print(f"\n--- ROUND {round_no} ---", "INFO")

                for cat in all_cats:
                    for bn, cfg in unique_brokers.items():
                        if not remaining[bn][cat]:
                            continue

                        idx = indices[bn][cat]
                        if idx >= len(remaining[bn][cat]):
                            remaining[bn][cat] = []
                            continue

                        symbol = remaining[bn][cat][idx]
                        norm_key = normalize_broker_key(bn)
                        if symbol in normalized_blocked_symbols.get(norm_key, set()):
                            indices[bn][cat] += 1
                            continue

                        for run in (1, 2):
                            ok, errs = initialize_mt5(cfg["TERMINAL_PATH"], cfg["LOGIN_ID"], cfg["PASSWORD"], cfg["SERVER"])
                            error_log.extend(errs)
                            if not ok:
                                log_and_print(f"MT5 INIT FAILED → {bn}/{symbol} (run {run})", "ERROR")
                                mt5.shutdown()
                                continue

                            log_and_print(f"RUN {run} – PROCESSING {symbol} ({cat}) on {bn.upper()}", "INFO")

                            sym_folder = os.path.join(cfg["BASE_FOLDER"], symbol.replace(" ", "_"))
                            os.makedirs(sym_folder, exist_ok=True)

                            def roundgoblin():
                                for tf_str, mt5_tf in TIMEFRAME_MAP.items():
                                    tf_folder = os.path.join(sym_folder, tf_str)
                                    os.makedirs(tf_folder, exist_ok=True)

                                    df, errs = fetch_ohlcv_data(symbol, mt5_tf, bars)
                                    error_log.extend(errs)
                                    if df is None:
                                        log_and_print(f"NO DATA for {symbol} {tf_str}", "WARNING")
                                        continue

                                    df["symbol"] = symbol
                                    chart_path, ch_errs, ph, pl = generate_and_save_oldest_newest_chart(
                                        df, symbol, tf_str, tf_folder,
                                        neighborcandles_left, neighborcandles_right
                                    )
                                    error_log.extend(ch_errs)

                                    generate_and_save_newest_oldest_chart(
                                        df, symbol, tf_str, tf_folder,
                                        neighborcandles_left, neighborcandles_right
                                    )
                                    error_log.extend(ch_errs)

                                    save_oldest_newest(df, symbol, tf_str, tf_folder, ph, pl)
                                    next_errs = save_next_oldest_newest_candles(df, symbol, tf_str, tf_folder, ph, pl)
                                    error_log.extend(next_errs)

                                    save_newest_oldest(df, symbol, tf_str, tf_folder, ph, pl)
                                    next_errs = save_next_newest_oldest_candles(df, symbol, tf_str, tf_folder, ph, pl)
                                    error_log.extend(next_errs)

                                    if chart_path:
                                        crop_chart(chart_path, symbol, tf_str, tf_folder)

                                mt5.shutdown()

                            roundgoblin()
                            ticks_value(symbol, sym_folder, bn, cfg["BASE_FOLDER"], candidates[bn][cat])
                            calc_and_placeorders()

                        indices[bn][cat] += 1

                round_no += 1

            save_errors(error_log)
            calc_and_placeorders()
            log_and_print("CYCLE 100% COMPLETED (UNIQUE BROKERS ONLY)", "SUCCESS")

            # ------------------------------------------------------------------
            # FINAL STEP: RENAME BASE_FOLDERS TO REMOVE NUMBERS (AFTER CYCLE)
            # ------------------------------------------------------------------
            log_and_print("Starting post-cycle BASE_FOLDER renaming (removing numbers)...", "INFO")
            renamed = 0
            for original_name, cfg in unique_brokers.items():
                old_path = cfg["BASE_FOLDER"]
                if not os.path.exists(old_path):
                    continue

                parent_dir = os.path.dirname(old_path)
                new_name = clean_folder_name(original_name)
                new_path = os.path.join(parent_dir, new_name)

                if old_path == new_path:
                    continue

                if os.path.exists(new_path):
                    log_and_print(f"Target already exists: {new_path} — skipping rename from {old_path}", "WARNING")
                    continue

                try:
                    os.rename(old_path, new_path)
                    cfg["BASE_FOLDER"] = new_path
                    log_and_print(f"RENAMED FOLDER: {old_path} → {new_path}", "SUCCESS")
                    renamed += 1
                except Exception as e:
                    log_and_print(f"FAILED RENAME {old_path} → {new_path}: {e}", "ERROR")

            log_and_print(f"Folder renaming complete. {renamed} folder(s) cleaned.", "SUCCESS" if renamed > 0 else "INFO")

            log_and_print("Sleeping 30 minutes before next cycle...", "INFO")
            time.sleep(1800)

        except Exception as e:
            log_and_print(f"MAIN LOOP CRASH: {e}\n{traceback.format_exc()}", "CRITICAL")
            time.sleep(600)
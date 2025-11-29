import ccxt
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import os

def fetch_ohlcv_ccxt(symbol, timeframe='5m', limit=500, exchange_id='mexc'):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}  # important for MEXC perps
    })
    
    # Auto-fix symbol format for MEXC (e.g., BTC/USDT:USDT → BTC_USDT)
    if exchange_id in ['mexc', 'mexc3']:
        symbol = symbol.replace('/', '_').replace(':USDT', '')

    print(f"Fetching {limit} candles of {symbol} ({timeframe}) from {exchange_id}...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    if not ohlcv:
        print("Empty response – check symbol or market type")
        return pd.DataFrame()
        
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    
    print(f"Data fetched successfully! Last candle: {df.index[-1]}")
    return df

# Fetch the data
df = fetch_ohlcv_ccxt('BTC/USDT:USDT', '5m', 500, 'mexc')

if not df.empty:
    # Create a nice candlestick chart
    save_path = f"BTCUSDT_PERP_5m_MEXC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    # Optional: Add some style and indicators
    apds = [
        mpf.make_addplot(df['volume'], panel=1, type='bar', ylabel='Volume', secondary_y=False),
    ]
    
    mpf.plot(
        df,
        type='candle',
        style='charles',           # clean style (you can try 'yahoo', 'binance', 'blueskies', etc.)
        title=f'BTC/USDT:USDT Perpetual - MEXC - 5m Chart\nLast update: {df.index[-1]}',
        ylabel='Price (USDT)',
        volume=True,
        addplot=apds,
        savefig=dict(fname=save_path, dpi=300, bbox_inches='tight'),
        figsize=(16, 10),
        panel_ratios=(3, 1),
        tight_layout=True
    )
    
    print(f"Chart saved as: {os.path.abspath(save_path)}")
else:
    print("No data to plot.")
from pykrx import stock
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

tickers = stock.get_market_ticker_list(market="KOSPI")
for symbol in tickers:
    name = stock.get_market_ticker_name(symbol)
    print(f"{symbol}.KS|{name}")

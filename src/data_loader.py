import yfinance as yf

def get_data(stock,start,end):
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        return None
    return data

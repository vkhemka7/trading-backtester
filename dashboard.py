import streamlit as st # type: ignore
import sys
sys.path.append("src")
from compare_strategies import compare_strategies
from data_loader import get_data
import pandas as pd


st.title("QuantBacktester")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start = st.sidebar.text_input("Start Date", value="2020-01-01")
end = st.sidebar.text_input("End Date", value="2024-01-01")

st.write(f"Running strategy for {ticker} from {start} to {end}")


sys.path.append("src")

if st.sidebar.button("Run"):
    with st.spinner("Running strategies..."):
        result, equity_curves, price = compare_strategies(ticker, start, end)
        st.dataframe(result)
        st.line_chart(equity_curves)

        st.subheader("Stock Price")
        st.line_chart(price)
        


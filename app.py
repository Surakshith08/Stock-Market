import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Global Market Influence on Nifty 50")

st.write("This app analyzes global indices and predicts Nifty open direction.")

# Download latest data
nse = yf.download("^NSEI", period="1y")
dji = yf.download("^DJI", period="1y")

# returns
nse["Return"] = nse["Close"].pct_change()
dji["Return"] = dji["Close"].pct_change()

df = pd.DataFrame({
    "NSE_Return": nse["Return"],
    "DJI_Return": dji["Return"]
}).dropna()

# target
df["Target"] = (df["NSE_Return"] > 0).astype(int)

X = df[["DJI_Return"]]
y = df["Target"]

model = LogisticRegression()
model.fit(X, y)

latest = [[df["DJI_Return"].iloc[-1]]]

prediction = model.predict(latest)

st.subheader("Prediction")

if prediction[0] == 1:
    st.success("Nifty likely to open UP")
else:
    st.error("Nifty likely to open DOWN")

st.line_chart(nse["Close"])

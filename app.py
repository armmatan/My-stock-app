import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Alpha Expert AI", layout="wide")

# פונקציה לחישוב ציון מומחה
def calculate_expert_score(rsi, peg, sma50, current_price):
    score = 50 # נקודת פתיחה
    reasons = []
    
    # טכני: RSI (מכירת יתר זה טוב לקנייה)
    if rsi < 35: 
        score += 20
        reasons.append("Oversold (RSI)")
    elif rsi > 70: 
        score -= 20
        reasons.append("Overbought (Caution)")
        
    # פנדמנטלי: PEG (מתחת ל-1 זה זול וצומח)
    if peg < 1:
        score += 25
        reasons.append("High Growth/Low Price (PEG)")
    elif peg > 2.5:
        score -= 15
        reasons.append("Expensive Growth")

    # מגמה: מעל ממוצע 50
    if current_price > sma50:
        score += 10
        reasons.append("Positive Momentum")
        
    return score, ", ".join(reasons)

st.title("🚀 Alpha Expert Trader")
st.sidebar.header("Control Panel")
tickers_input = st.sidebar.text_input("Enter Tickers (CSV)", "NVDA,AAPL,TSLA,AMD,PLTR")
run_analysis = st.sidebar.button("Analyze as Expert")

if run_analysis:
    results = []
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    with st.spinner("Calculating Expert Scores..."):
        for t in tickers:
            try:
                stock = yf.Ticker(t)
                hist = stock.history(period="1y")
                info = stock.info
                
                # חישוב נתונים
                price = hist['Close'].iloc[-1]
                sma50 = hist['Close'].rolling(50).mean().iloc[-1]
                
                # RSI חישוב ידני מהיר
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]
                
                peg = info.get('pegRatio', 2.0)
                
                score, reason = calculate_expert_score(rsi, peg, sma50, price)
                
                # קביעת המלצה
                if score >= 70: rec = "Strong Buy"
                elif score >= 50: rec = "Hold/Buy"
                else: rec = "Avoid/Sell"
                
                results.append({"Ticker": t, "Score": score, "Recommendation": rec, "Reason": reason, "Price": round(price, 2)})
            except:
                st.warning(f"Could not analyze {t}")

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df.sort_values(by="Score", ascending=False), use_container_width=True)
        
        # תצוגת כרטיסיות לנייד
        for res in results:
            if res['Score'] >= 70:
                st.success(f"💎 **{res['Ticker']}**: {res['Recommendation']} (Score: {res['Score']}) - {res['Reason']}")

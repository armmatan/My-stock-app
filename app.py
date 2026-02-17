import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import time
from datetime import datetime

# --- הגדרת דף (חייבת להיות ראשונה) ---
st.set_page_config(page_title="Pro Trader AI", layout="wide", page_icon="📈")

# --- פונקציות עזר וחישובים ---

@st.cache_data(ttl=300) # שומר נתונים ל-5 דקות
def fetch_stock_data(tickers):
    """מושך נתונים טכניים ופנדמנטליים לכל הטיקרים"""
    data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # 1. שליפת היסטוריה (טכני)
            hist = stock.history(period="6mo")
            if hist.empty:
                continue
                
            # חישוב אינדיקטורים
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # חישוב RSI ידני
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            current_price = hist['Close'].iloc[-1]
            current_rsi = hist['RSI'].iloc[-1]
            sma_50 = hist['SMA_50'].iloc[-1]
            sma_200 = hist['SMA_200'].iloc[-1]
            
            # 2. שליפת פנדמנטלס (שווי חברה)
            info = stock.info
            pe_ratio = info.get('trailingPE', None)
            peg_ratio = info.get('pegRatio', None) # יחס צמיחה למחיר
            target_price = info.get('targetMeanPrice', None)
            
            # 3. מנוע הדירוג (The Expert Brain)
            score = 0
            reasons = []
            
            # ניקוד טכני (מקסימום 60 נקודות)
            if current_rsi < 30:
                score += 25
                reasons.append("Oversold (RSI)")
            elif current_rsi > 70:
                score -= 20
                reasons.append("Overbought (RSI)")
            elif 40 < current_rsi < 60:
                score += 10 # מומנטום יציב
                
            if current_price > sma_50:
                score += 15
                reasons.append("Above SMA50 (Uptrend)")
            else:
                score -= 10
                
            if current_price > sma_200:
                score += 20
                reasons.append("Long Term Bullish")
            
            # ניקוד פנדמנטלי (מקסימום 40 נקודות)
            if peg_ratio and peg_ratio < 1:
                score += 20
                reasons.append("Undervalued Growth (PEG<1)")
            elif peg_ratio and peg_ratio > 2:
                score -= 10
                
            if target_price and current_price < target_price:
                upside = ((target_price - current_price) / current_price) * 100
                if upside > 15:
                    score += 20
                    reasons.append(f"Analyst Upside ({upside:.1f}%)")

            # קביעת המלצה סופית
            signal = "HOLD"
            color = "gray"
            if score >= 65:
                signal = "STRONG BUY"
                color = "green"
            elif score >= 40:
                signal = "BUY"
                color = "lightgreen"
            elif score <= 10:
                signal = "STRONG SELL"
                color = "red"
            elif score <= 25:
                signal = "SELL"
                color = "orange"

            data.append({
                "Ticker": ticker,
                "Price": current_price,
                "Signal": signal,
                "Score": score,
                "RSI": round(current_rsi, 2),
                "P/E": round(pe_ratio, 2) if pe_ratio else "N/A",
                "Reasons": ", ".join(reasons),
                "Color": color,
                "History": hist
            })
            
        except Exception as e:
            continue
            
    return data

# --- ממשק משתמש (UI) ---

st.title("🚀 Pro Trader AI Dashboard")
st.markdown("Automatic expert analysis based on Technicals & Fundamentals")

# Sidebar
st.sidebar.header("Settings")
default_tickers = "NVDA,AAPL,TSLA,AMD,MSFT,GOOGL,AMZN,PLTR"
tickers_input = st.sidebar.text_area("Watchlist (comma separated)", default_tickers)
tickers_list = [x.strip().upper() for x in tickers_input.split(',')]

if st.sidebar.button("Run AI Analysis"):
    with st.spinner('Analyzing Market Data...'):
        market_data = fetch_stock_data(tickers_list)
        
        if not market_data:
            st.error("No data found. Check tickers.")
        else:
            # המרת הנתונים ל-DataFrame לתצוגה
            df_display = pd.DataFrame(market_data)
            
            # תצוגת מדדים עיקריים (Top Picks)
            st.subheader("🏆 Top Opportunities")
            best_stocks = [s for s in market_data if "BUY" in s['Signal']]
            
            if best_stocks:
                cols = st.columns(len(best_stocks))
                for idx, stock in enumerate(best_stocks):
                    with cols[idx]:
                        st.metric(
                            label=stock['Ticker'], 
                            value=f"${stock['Price']:.2f}",
                            delta=f"Score: {stock['Score']}"
                        )
                        st.markdown(f":green[**{stock['Signal']}**]")
            else:
                st.info("No strong buying opportunities found right now.")

            # טבלה מפורטת
            st.divider()
            st.subheader("📊 Detailed Analysis")
            
            # עיצוב הטבלה
            def highlight_signal(val):
                color = 'red' if 'SELL' in val else 'green' if 'BUY' in val else 'gray'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                df_display[["Ticker", "Price", "Signal", "Score", "RSI", "P/E", "Reasons"]].style.applymap(
                    highlight_signal, subset=['Signal']
                ),
                use_container_width=True
            )

            # אזור גרפים אינטראקטיבי
            st.divider()
            st.subheader("📈 Technical Chart")
            selected_ticker = st.selectbox("Select stock to view chart", [d['Ticker'] for d in market_data])
            
            selected_data = next(item for item in market_data if item["Ticker"] == selected_ticker)
            hist_df = selected_data['History']
            
            fig = go.Figure()
            # נרות יפניים
            fig.add_trace(go.Candlestick(x=hist_df.index,
                            open=hist_df['Open'], high=hist_df['High'],
                            low=hist_df['Low'], close=hist_df['Close'], name='Price'))
            
            # ממוצעים נעים
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
            
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Run AI Analysis' in the sidebar to start.")

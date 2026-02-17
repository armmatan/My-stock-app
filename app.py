"""
Stock Trading Dashboard - Streamlit Application
================================================
A comprehensive stock screening and analysis dashboard with real-time data,
technical indicators, and historical tracking.

Author: Claude AI Assistant
Version: 2.0 - Enhanced Mobile & Watchlist Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
from typing import List, Dict, Optional
import warnings
import time
import re

warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_ticker_list(ticker_input: str) -> List[str]:
    """
    Clean and parse ticker list from user input
    
    Args:
        ticker_input: Comma-separated ticker string
    
    Returns:
        List of cleaned ticker symbols
    """
    if not ticker_input:
        return []
    
    # Split by comma and clean each ticker
    tickers = [
        re.sub(r'[^A-Z0-9\-.]', '', ticker.strip().upper())
        for ticker in ticker_input.split(',')
    ]
    
    # Remove empty strings and duplicates while preserving order
    seen = set()
    cleaned_tickers = []
    for ticker in tickers:
        if ticker and ticker not in seen:
            seen.add(ticker)
            cleaned_tickers.append(ticker)
    
    return cleaned_tickers

# ============================================================================
# DATABASE LAYER
# ============================================================================

class DatabaseManager:
    """Manages SQLite database operations for scan history"""
    
    def __init__(self, db_path: str = "scan_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with scans table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                close REAL,
                score INTEGER,
                sentiment REAL,
                signal TEXT,
                rsi REAL,
                volume REAL,
                sma_20 REAL,
                sma_50 REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_scan_results(self, results: List[Dict]):
        """Save scan results to database"""
        if not results:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in results:
            cursor.execute("""
                INSERT INTO scans (scan_time, ticker, close, score, sentiment, 
                                  signal, rsi, volume, sma_20, sma_50)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_time,
                result.get('ticker', ''),
                result.get('close', 0),
                result.get('score', 0),
                result.get('sentiment', 0),
                result.get('signal', 'HOLD'),
                result.get('rsi', 50),
                result.get('volume', 0),
                result.get('sma_20', 0),
                result.get('sma_50', 0)
            ))
        
        conn.commit()
        conn.close()
    
    def load_latest_scan(self) -> pd.DataFrame:
        """Load most recent scan results"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT ticker, close, score, sentiment, signal, rsi, volume, 
                   sma_20, sma_50, scan_time
            FROM scans
            WHERE scan_time = (SELECT MAX(scan_time) FROM scans)
            ORDER BY score DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def load_history(self, ticker: str) -> pd.DataFrame:
        """Load historical scans for a specific ticker"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT scan_time, close, score, sentiment, signal, rsi, volume
            FROM scans
            WHERE ticker = ?
            ORDER BY scan_time DESC
            LIMIT 50
        """
        
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        
        return df

# ============================================================================
# MARKET DATA SERVICE
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    Fetch stock price data using yfinance
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    
    Returns:
        DataFrame with OHLCV data or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_stock_info(ticker: str) -> Dict:
    """Fetch stock information and metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return {}

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def detect_volume_spike(volume: pd.Series, threshold: float = 1.5) -> bool:
    """Detect if current volume is significantly higher than average"""
    if len(volume) < 20:
        return False
    
    avg_volume = volume[-20:].mean()
    current_volume = volume.iloc[-1]
    
    return current_volume > (avg_volume * threshold)

def detect_breakout(close: pd.Series, high: pd.Series) -> bool:
    """Detect if stock is breaking out of recent range"""
    if len(close) < 20:
        return False
    
    recent_high = high[-20:-1].max()
    current_close = close.iloc[-1]
    
    return current_close > recent_high

# ============================================================================
# STOCK ANALYSIS ENGINE
# ============================================================================

def analyze_stock(ticker: str, rsi_low: int = 30, rsi_high: int = 70) -> Optional[Dict]:
    """
    Analyze a stock and generate buy/sell signals
    
    Args:
        ticker: Stock ticker symbol
        rsi_low: RSI oversold threshold
        rsi_high: RSI overbought threshold
    
    Returns:
        Dictionary with analysis results
    """
    df = fetch_stock_data(ticker, period="3mo")
    
    if df is None or df.empty:
        return None
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    # Get latest values
    latest = df.iloc[-1]
    rsi = latest['RSI']
    close = latest['Close']
    volume = latest['Volume']
    sma_20 = latest['SMA_20']
    sma_50 = latest['SMA_50']
    
    # Volume and breakout detection
    volume_spike = detect_volume_spike(df['Volume'])
    breakout = detect_breakout(df['Close'], df['High'])
    
    # Calculate score (0-100)
    score = 50  # Base score
    
    # RSI component
    if rsi < rsi_low:
        score += 20  # Oversold - bullish
    elif rsi > rsi_high:
        score -= 20  # Overbought - bearish
    
    # Moving average component
    if close > sma_20 > sma_50:
        score += 15  # Bullish trend
    elif close < sma_20 < sma_50:
        score -= 15  # Bearish trend
    
    # Volume spike
    if volume_spike:
        score += 10
    
    # Breakout
    if breakout:
        score += 10
    
    # Ensure score is in range
    score = max(0, min(100, score))
    
    # Generate signal
    if score >= 70:
        signal = "BUY"
    elif score <= 30:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # Mock sentiment (in real app, would fetch from news API)
    sentiment = np.random.uniform(-1, 1)
    
    return {
        'ticker': ticker,
        'close': round(close, 2),
        'score': int(score),
        'sentiment': round(sentiment, 2),
        'signal': signal,
        'rsi': round(rsi, 2),
        'volume': int(volume),
        'sma_20': round(sma_20, 2),
        'sma_50': round(sma_50, 2),
        'volume_spike': volume_spike,
        'breakout': breakout
    }

def run_screener(tickers: List[str], rsi_low: int, rsi_high: int) -> List[Dict]:
    """Run stock screener on multiple tickers"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Analyzing {ticker}... ({idx + 1}/{len(tickers)})")
        result = analyze_stock(ticker, rsi_low, rsi_high)
        
        if result:
            results.append(result)
        
        # Sleep to prevent API rate limiting
        if idx < len(tickers) - 1:  # Don't sleep on last iteration
            time.sleep(1.2)
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    return results

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def render_sentiment_gauge(avg_sentiment: float):
    """
    Render sentiment gauge chart
    
    Args:
        avg_sentiment: Average sentiment value between -1 and 1
    """
    # Convert sentiment to 0-100 scale
    gauge_value = (avg_sentiment + 1) * 50  # -1 to 1 becomes 0 to 100
    
    # Determine color based on sentiment
    if gauge_value >= 60:
        color = "green"
        sentiment_label = "Positive"
    elif gauge_value >= 40:
        color = "orange"
        sentiment_label = "Neutral"
    else:
        color = "red"
        sentiment_label = "Negative"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overall Sentiment<br><span style='font-size:0.8em'>{sentiment_label}</span>"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "rgba(255, 0, 0, 0.2)"},
                {'range': [33, 66], 'color': "rgba(255, 165, 0, 0.2)"},
                {'range': [66, 100], 'color': "rgba(0, 255, 0, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "gray", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_candlestick_chart(ticker: str, period: str = "3mo"):
    """Render interactive candlestick chart with indicators"""
    df = fetch_stock_data(ticker, period)
    
    if df is None or df.empty:
        st.warning(f"No data available for {ticker}")
        return
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Chart', 'RSI')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )
    
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Update layout - MOBILE OPTIMIZED HEIGHT
    fig.update_layout(
        height=300,  # Reduced height for mobile
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_score_history_chart(df_history: pd.DataFrame):
    """Render score trend over time"""
    if df_history.empty:
        st.info("No historical data available")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_history['scan_time'],
        y=df_history['score'],
        mode='lines+markers',
        name='Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Score History',
        xaxis_title='Date',
        yaxis_title='Score',
        height=300,  # Mobile optimized
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state"""
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
    
    if 'scan_time' not in st.session_state:
        st.session_state.scan_time = None
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def render_sidebar_controls() -> Dict:
    """Render sidebar controls and return user selections"""
    st.sidebar.header("⚙️ Controls")
    
    # API Key input (optional, for future Alpha Vantage integration)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Configuration")
    api_key = st.sidebar.text_input(
        "Alpha Vantage API Key (Optional)",
        type="password",
        help="Enter your Alpha Vantage API key for additional features"
    )
    
    if api_key:
        st.sidebar.success("✓ API Key configured")
    else:
        st.sidebar.info("💡 Using free data source (yfinance)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Custom Watchlist")
    
    # Custom watchlist input
    watchlist_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value="",
        height=80,
        placeholder="NVDA, AAPL, TSLA, GOOGL",
        help="Enter stock tickers separated by commas. Spaces will be automatically cleaned."
    )
    
    # Parse watchlist
    custom_tickers = clean_ticker_list(watchlist_input)
    
    if custom_tickers:
        st.sidebar.success(f"✓ {len(custom_tickers)} tickers loaded")
        with st.sidebar.expander("View Watchlist"):
            st.write(", ".join(custom_tickers))
    
    st.sidebar.markdown("---")
    
    # Default tickers (only shown if no custom watchlist)
    default_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'BRK-B', 'V', 'JPM',
        'JNJ', 'WMT', 'PG', 'MA', 'HD',
        'DIS', 'NFLX', 'PYPL', 'INTC', 'CSCO'
    ]
    
    # Ticker selection (use custom or default)
    if custom_tickers:
        selected_tickers = custom_tickers
        st.sidebar.info("Using custom watchlist")
    else:
        selected_tickers = st.sidebar.multiselect(
            "📊 Or Select from Defaults",
            options=default_tickers,
            default=default_tickers[:5],
            help="Choose stocks to analyze"
        )
    
    # RSI thresholds
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 RSI Settings")
    rsi_low = st.sidebar.slider(
        "Oversold Threshold",
        min_value=10,
        max_value=40,
        value=30,
        help="Below this value = oversold (buy signal)"
    )
    
    rsi_high = st.sidebar.slider(
        "Overbought Threshold",
        min_value=60,
        max_value=90,
        value=70,
        help="Above this value = overbought (sell signal)"
    )
    
    st.sidebar.markdown("---")
    
    # Run scan button
    run_scan = st.sidebar.button(
        "🚀 Run Scan",
        type="primary",
        use_container_width=True
    )
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 Quick Guide
    1. **Custom Watchlist**: Enter your tickers
    2. **Or**: Select from defaults
    3. **Adjust**: RSI thresholds
    4. **Click**: 'Run Scan'
    5. **Select**: Any row for details
    
    💡 *Scans run with 1.2s delay between stocks for stability*
    """)
    
    return {
        'selected_tickers': selected_tickers,
        'rsi_low': rsi_low,
        'rsi_high': rsi_high,
        'run_scan_clicked': run_scan,
        'api_key': api_key
    }

def render_metrics(stock_data: Dict):
    """Render KPI metrics for selected stock with card-style design"""
    
    # Custom CSS for metric cards
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create card-style metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = stock_data.get('close', 0)
        st.metric(
            label="💰 Current Price",
            value=f"${price:.2f}",
            help="Latest closing price"
        )
    
    with col2:
        score = stock_data.get('score', 0)
        
        # Color indicator
        if score >= 70:
            score_emoji = "🟢"
        elif score <= 30:
            score_emoji = "🔴"
        else:
            score_emoji = "🟡"
        
        st.metric(
            label="📊 Analysis Score",
            value=f"{score_emoji} {score}/100",
            help="Composite technical score"
        )
    
    with col3:
        rsi = stock_data.get('rsi', 50)
        
        if rsi < 30:
            rsi_status = "Oversold"
            rsi_delta = "📉"
        elif rsi > 70:
            rsi_status = "Overbought"
            rsi_delta = "📈"
        else:
            rsi_status = "Neutral"
            rsi_delta = "➡️"
        
        st.metric(
            label=f"{rsi_delta} RSI",
            value=f"{rsi:.1f}",
            delta=rsi_status,
            help="Relative Strength Index"
        )
    
    with col4:
        signal = stock_data.get('signal', 'HOLD')
        
        if signal == 'BUY':
            signal_emoji = "🟢"
            signal_color = "#28a745"
        elif signal == 'SELL':
            signal_emoji = "🔴"
            signal_color = "#dc3545"
        else:
            signal_emoji = "🟡"
            signal_color = "#ffc107"
        
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 10px;
            background-color: {signal_color}22;
            border-radius: 8px;
            border: 2px solid {signal_color};
        ">
            <div style="font-size: 12px; color: #555; margin-bottom: 4px;">
                Trading Signal
            </div>
            <div style="font-size: 28px; font-weight: bold; color: {signal_color};">
                {signal_emoji} {signal}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_stock_details(stock_data: Dict):
    """Render detailed stock information panel"""
    st.subheader(f"📋 Details: {stock_data['ticker']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Technical Indicators:**")
        st.write(f"- RSI: {stock_data['rsi']:.2f}")
        st.write(f"- SMA 20: ${stock_data['sma_20']:.2f}")
        st.write(f"- SMA 50: ${stock_data['sma_50']:.2f}")
    
    with col2:
        st.markdown("**Market Conditions:**")
        st.write(f"- Volume: {stock_data['volume']:,}")
        st.write(f"- Volume Spike: {'🔥 Yes' if stock_data.get('volume_spike') else '❌ No'}")
        st.write(f"- Breakout: {'🚀 Yes' if stock_data.get('breakout') else '❌ No'}")

def render_results_table(df_results: pd.DataFrame):
    """Render interactive results table with CSS styling"""
    if df_results.empty:
        st.info("No results to display. Run a scan to get started!")
        return
    
    # Add custom CSS for BUY signal highlighting
    st.markdown("""
    <style>
    /* Style for BUY signal rows */
    [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        font-family: 'Arial', sans-serif;
    }
    
    /* Green background for positive signals */
    .buy-signal {
        background-color: #d4edda !important;
        color: #155724 !important;
        font-weight: bold;
    }
    
    /* Red background for negative signals */
    .sell-signal {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    
    /* Yellow background for neutral signals */
    .hold-signal {
        background-color: #fff3cd !important;
        color: #856404 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Scan Results")
    
    # Prepare display dataframe
    display_df = df_results.copy()
    
    # Format columns with clear English headers
    if 'close' in display_df.columns:
        display_df['Price'] = display_df['close'].apply(lambda x: f"${x:.2f}")
    
    if 'score' in display_df.columns:
        display_df['Score'] = display_df['score'].astype(int)
    
    if 'signal' in display_df.columns:
        display_df['Signal'] = display_df['signal']
    
    if 'rsi' in display_df.columns:
        display_df['RSI'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
    
    if 'sentiment' in display_df.columns:
        display_df['Sentiment'] = display_df['sentiment'].apply(lambda x: f"{x:.2f}")
    
    if 'volume' in display_df.columns:
        display_df['Volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}")
    
    # Select and rename columns for display
    display_columns = {
        'ticker': 'Ticker',
        'Price': 'Price',
        'Score': 'Score',
        'Signal': 'Signal',
        'RSI': 'RSI',
        'Sentiment': 'Sentiment',
        'Volume': 'Volume'
    }
    
    # Create final display dataframe
    final_df = pd.DataFrame()
    for old_col, new_col in display_columns.items():
        if old_col in display_df.columns:
            final_df[new_col] = display_df[old_col]
        elif new_col in display_df.columns:
            final_df[new_col] = display_df[new_col]
    
    # Apply styling function for signal colors
    def style_signal_row(row):
        if 'Signal' in row.index:
            if row['Signal'] == 'BUY':
                return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
            elif row['Signal'] == 'SELL':
                return ['background-color: #f8d7da; color: #721c24'] * len(row)
            else:
                return ['background-color: #fff3cd; color: #856404'] * len(row)
        return [''] * len(row)
    
    # Display summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_count = len(df_results[df_results['signal'] == 'BUY'])
        st.metric("🟢 BUY Signals", buy_count)
    
    with col2:
        sell_count = len(df_results[df_results['signal'] == 'SELL'])
        st.metric("🔴 SELL Signals", sell_count)
    
    with col3:
        hold_count = len(df_results[df_results['signal'] == 'HOLD'])
        st.metric("🟡 HOLD Signals", hold_count)
    
    with col4:
        avg_score = df_results['score'].mean()
        st.metric("📊 Avg Score", f"{avg_score:.1f}")
    
    st.markdown("---")
    
    # Display with selection
    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="results_table",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Score": st.column_config.NumberColumn("Score", width="small"),
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "RSI": st.column_config.TextColumn("RSI", width="small"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Volume": st.column_config.TextColumn("Volume", width="medium")
        }
    )
    
    # Handle selection
    if st.session_state.get('results_table'):
        selection = st.session_state.results_table.get('selection')
        if selection and selection.get('rows'):
            selected_idx = selection['rows'][0]
            selected_ticker = df_results.iloc[selected_idx]['ticker']
            st.session_state.selected_ticker = selected_ticker

def render_history_view():
    """Render historical scan viewer"""
    st.markdown("---")
    st.subheader("📜 Historical Scan Data")
    
    db_manager = st.session_state.db_manager
    
    # Check if we have any data
    latest_scan = db_manager.load_latest_scan()
    
    if latest_scan.empty:
        st.info("No historical data yet. Run a scan to start tracking!")
        return
    
    # Get unique tickers from latest scan
    tickers = sorted(latest_scan['ticker'].unique().tolist())
    
    selected_history_ticker = st.selectbox(
        "Select Ticker for History",
        options=tickers,
        key="history_ticker"
    )
    
    if selected_history_ticker:
        df_history = db_manager.load_history(selected_history_ticker)
        
        if not df_history.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Recent Scan Results:**")
                
                # Format the history dataframe with clear English headers
                display_history = df_history.copy()
                display_history['Scan Time'] = display_history['scan_time']
                display_history['Price'] = display_history['close'].apply(lambda x: f"${x:.2f}")
                display_history['Score'] = display_history['score'].astype(int)
                display_history['Signal'] = display_history['signal']
                display_history['RSI'] = display_history['rsi'].apply(lambda x: f"{x:.1f}")
                
                # Select columns for display
                final_display = display_history[['Scan Time', 'Price', 'Score', 'Signal', 'RSI']].head(10)
                
                st.dataframe(
                    final_display,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Scan Time": st.column_config.TextColumn("Scan Time", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Score": st.column_config.NumberColumn("Score", width="small"),
                        "Signal": st.column_config.TextColumn("Signal", width="small"),
                        "RSI": st.column_config.TextColumn("RSI", width="small")
                    }
                )
            
            with col2:
                render_score_history_chart(df_history)
        else:
            st.info(f"No history available for {selected_history_ticker}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Stock Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for mobile optimization
    st.markdown("""
    <style>
    /* Mobile-friendly padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card-style containers */
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* BUY signal highlighting */
    .buy-highlight {
        background-color: #d4edda !important;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    /* Responsive font sizes */
    @media (max-width: 768px) {
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Header with card design
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    ">
        <h1 style="color: white; margin: 0; text-align: center;">
            📈 Stock Trading Dashboard
        </h1>
        <p style="color: #e0e0e0; text-align: center; margin: 10px 0 0 0;">
            Real-time screening & technical analysis powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar and get controls
    controls = render_sidebar_controls()
    
    # Main content area
    db_manager = st.session_state.db_manager
    
    # Handle scan execution
    if controls['run_scan_clicked']:
        if not controls['selected_tickers']:
            st.warning("⚠️ Please select at least one ticker to scan")
        else:
            with st.spinner("🔍 Running stock screener..."):
                results = run_screener(
                    controls['selected_tickers'],
                    controls['rsi_low'],
                    controls['rsi_high']
                )
                
                if results:
                    # Save to database
                    db_manager.save_scan_results(results)
                    
                    # Update session state
                    st.session_state.scan_results = pd.DataFrame(results)
                    st.session_state.scan_time = datetime.now()
                    
                    st.success(f"✅ Scan completed! Analyzed {len(results)} stocks")
                else:
                    st.error("❌ Scan failed. Please check your ticker symbols")
    
    # Load results (either from scan or from database)
    if st.session_state.scan_results is not None:
        df_results = st.session_state.scan_results
    else:
        df_results = db_manager.load_latest_scan()
        if not df_results.empty:
            st.session_state.scan_results = df_results
    
    # Display results table
    if not df_results.empty:
        # TOP CARD SECTION - Market Overview
        st.markdown("### 📊 Market Overview")
        
        overview_col1, overview_col2, overview_col3 = st.columns([2, 2, 3])
        
        with overview_col1:
            # Summary stats card
            total_stocks = len(df_results)
            buy_signals = len(df_results[df_results['signal'] == 'BUY'])
            sell_signals = len(df_results[df_results['signal'] == 'SELL'])
            
            st.markdown(f"""
            <div class="card">
                <h4 style="margin-top: 0;">📈 Summary</h4>
                <p><strong>Total Stocks:</strong> {total_stocks}</p>
                <p><strong>🟢 Buy Signals:</strong> {buy_signals}</p>
                <p><strong>🔴 Sell Signals:</strong> {sell_signals}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with overview_col2:
            # Top performers card
            top_stocks = df_results.nlargest(3, 'score')
            
            st.markdown("""
            <div class="card">
                <h4 style="margin-top: 0;">⭐ Top Performers</h4>
            """, unsafe_allow_html=True)
            
            for _, stock in top_stocks.iterrows():
                st.markdown(f"**{stock['ticker']}**: {stock['score']}/100")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with overview_col3:
            # Sentiment gauge
            avg_sentiment = df_results['sentiment'].mean()
            render_sentiment_gauge(avg_sentiment)
        
        st.markdown("---")
        
        # Show scan time
        if st.session_state.scan_time:
            st.caption(f"🕒 Last scan: {st.session_state.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        elif 'scan_time' in df_results.columns:
            st.caption(f"🕒 Last scan: {df_results['scan_time'].iloc[0]}")
        
        # Render table
        render_results_table(df_results)
        
        # If a stock is selected, show details
        if st.session_state.selected_ticker:
            selected_data = df_results[df_results['ticker'] == st.session_state.selected_ticker].iloc[0].to_dict()
            
            st.markdown("---")
            st.markdown(f"## 🎯 Selected Stock: **{st.session_state.selected_ticker}**")
            
            # Metrics row with card design
            render_metrics(selected_data)
            
            st.markdown("---")
            
            # Details and chart
            col1, col2 = st.columns([1, 2])
            
            with col1:
                render_stock_details(selected_data)
            
            with col2:
                # Show chart for ANY selected stock (not just BUY signals)
                st.subheader(f"📊 Price Chart")
                
                period = st.selectbox(
                    "Time Period",
                    options=['1mo', '3mo', '6mo', '1y'],
                    index=1,
                    key="chart_period"
                )
                
                render_candlestick_chart(st.session_state.selected_ticker, period)
    
    else:
        # Empty state with helpful message
        st.markdown("""
        <div style="
            text-align: center;
            padding: 60px 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin: 40px 0;
        ">
            <h2>👋 Welcome to Your Trading Dashboard!</h2>
            <p style="font-size: 18px; color: #666; margin: 20px 0;">
                Get started by selecting stocks and running your first scan
            </p>
            <p style="color: #888;">
                📌 Use the sidebar to configure your watchlist and settings
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Historical viewer
    render_history_view()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>📊 Stock Trading Dashboard v2.0</strong></p>
        <p style='font-size: 12px;'>
            Built with Streamlit • Data: Yahoo Finance • Mobile Optimized
        </p>
        <p style='font-size: 11px; color: #999;'>
            ⚠️ Not financial advice • For educational purposes only
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()

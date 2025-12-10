import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import os

# --- Configuration ---
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# ==============================================================================
# PART 1: SHARED HELPERS
# ==============================================================================

def infer_asset_class(ticker):
    """Guess asset class based on ticker string."""
    ticker = str(ticker).upper().strip()
    if ticker.endswith('3') or ticker.endswith('4'):
        return 'Ações'
    elif ticker.endswith('11'):
        return 'FIIs / ETFs'
    elif ticker.endswith('34'):
        return 'BDRs'
    else:
        return 'Outros'

def format_ticker_for_yfinance(ticker):
    """
    Applies logic to determine if suffix .SA is needed.
    Logic: Ends in number -> .SA (B3), otherwise assume US/Global.
    Exceptions: IVVB11, HASH11, etc are handled generally by 'Ends in number'.
    """
    t = str(ticker).upper().strip()
    # Check if it already has a suffix or is a known crypto pair
    if "." in t or "-" in t: 
        return t
    
    # Heuristic: Brazilian tickers usually end in digits (PETR4, BOVA11)
    # US tickers usually don't (AAPL, TSLA)
    if any(char.isdigit() for char in t):
        return f"{t}.SA"
    
    return t

@st.cache_data
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Helper: Clean currency columns
        def clean_currency(x):
            if isinstance(x, str):
                clean_str = x.replace('R$', '').replace('.', '').replace(',', '.').strip()
                try: return float(clean_str)
                except ValueError: return 0.0
            return x

        currency_cols = ['Preço', 'Custo total com taxas', 'Preço médio com taxas', 'Taxas totais']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_currency)

        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
            
        return df
    except FileNotFoundError:
        return None

# ==============================================================================
# PART 2: PORTFOLIO DASHBOARD LOGIC (Your existing code)
# ==============================================================================

def process_ticker_transactions(group):
    group = group.sort_values('Data')
    current_qty = 0
    current_cost = 0
    
    for _, row in group.iterrows():
        qty = row['Quantidade']
        cost = row['Custo total com taxas']
        op = str(row['Operação']).strip().capitalize()
        
        if op == 'Compra':
            current_qty += qty
            current_cost += cost
        elif op == 'Venda':
            if current_qty > 0:
                avg_price = current_cost / current_qty
                current_qty -= qty
                current_cost = current_qty * avg_price
                
    return pd.Series({'Quantidade': current_qty, 'Custo total com taxas': current_cost})

def calculate_portfolio(df):
    if 'Classe' not in df.columns:
        df['Classe'] = df['Ticker'].apply(infer_asset_class)

    portfolio = df.groupby(['Ticker', 'Classe']).apply(process_ticker_transactions).reset_index()
    portfolio = portfolio[portfolio['Quantidade'] > 0.0001]
    portfolio['Average Price'] = portfolio['Custo total com taxas'] / portfolio['Quantidade']
    return portfolio

@st.cache_data(ttl=300)
def get_live_prices_bulk(tickers):
    formatted = [format_ticker_for_yfinance(t) for t in tickers]
    if not formatted: return {}
    try:
        data = yf.download(formatted, period="1d", progress=False)['Close']
        if isinstance(data, pd.Series):
            if len(data) > 0:
                 return {formatted[0].replace('.SA', ''): data.iloc[-1]}
            else: return {}
        if data.empty: return {}
        # Map back to original ticker name (remove .SA for display)
        return {col.replace('.SA', ''): data[col].iloc[-1] for col in data.columns}
    except:
        return {}

def render_portfolio_tab(df_transactions):
    st.markdown("### 📈 Portfolio Performance")
    
    portfolio_df = calculate_portfolio(df_transactions)
    
    if portfolio_df.empty:
        st.info("No active assets found (all sold or empty file).")
        return

    # Fetch Prices
    with st.spinner('Updating portfolio prices...'):
        unique_tickers = portfolio_df['Ticker'].unique().tolist()
        live_prices = get_live_prices_bulk(unique_tickers)
    
    portfolio_df['Current Price'] = portfolio_df['Ticker'].map(live_prices)
    portfolio_df['Current Price'] = portfolio_df['Current Price'].fillna(portfolio_df['Average Price'])

    # KPIs
    portfolio_df['Current Value'] = portfolio_df['Quantidade'] * portfolio_df['Current Price']
    portfolio_df['Gain/Loss Value'] = portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']
    portfolio_df['Gain/Loss %'] = ((portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']) / portfolio_df['Custo total com taxas']) * 100
    portfolio_df = portfolio_df.sort_values(by='Gain/Loss %', ascending=False)

    # Metrics
    total_invested = portfolio_df['Custo total com taxas'].sum()
    curr_val = portfolio_df['Current Value'].sum()
    diff = curr_val - total_invested
    pct = (diff / total_invested) * 100 if total_invested > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Invested", f"R$ {total_invested:,.2f}")
    c2.metric("Current Value", f"R$ {curr_val:,.2f}", delta=f"{diff:,.2f} ({pct:.2f}%)")
    best = portfolio_df.iloc[0]
    c3.metric("Best Asset", best['Ticker'], delta=f"{best['Gain/Loss %']:.2f}%")
    
    st.divider()

    # Sunburst
    st.subheader("Allocation")
    fig_sun = px.sunburst(
        portfolio_df, path=['Classe', 'Ticker'], values='Current Value',
        color='Classe', color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_sun.update_traces(textinfo="label+percent parent", hovertemplate="<b>%{label}</b><br>R$ %{value:,.2f}<br>%{percentRoot:.1%}")
    fig_sun.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)

    # Bar Charts
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Returns (%)")
        fig_p = px.bar(portfolio_df, x='Gain/Loss %', y='Ticker', orientation='h', text='Gain/Loss %',
                       color='Gain/Loss %', color_continuous_scale=['red', 'yellow', 'green'], range_color=[-20, 20])
        fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_p, use_container_width=True)
    with c_right:
        st.subheader("Price: Avg vs Current")
        comp_df = portfolio_df[['Ticker', 'Average Price', 'Current Price']].melt(id_vars='Ticker', var_name='Type', value_name='Price')
        fig_c = px.bar(comp_df, x='Ticker', y='Price', color='Type', barmode='group',
                       color_discrete_map={'Average Price': 'gray', 'Current Price': '#00CC96'})
        st.plotly_chart(fig_c, use_container_width=True)

# ==============================================================================
# PART 3: TECH SCANNER LOGIC (Golden Cross)
# ==============================================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900) # Cache for 15 mins
def run_scanner_analysis(tickers):
    results = []
    
    # Clean tickers
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = list(set(tickers)) # Remove duplicates
    
    if not tickers:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        yf_ticker = format_ticker_for_yfinance(ticker)
        
        try:
            # Download 1 year of data to ensure enough for SMA200
            df = yf.download(yf_ticker, period="2y", progress=False)
            
            if df.empty or len(df) < 200:
                continue
                
            # Handle MultiIndex columns if present (common in new yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(yf_ticker, level=1, axis=1)
                except:
                    # Fallback if structure is different, try taking 'Close'
                    if 'Close' in df.columns:
                        pass # standard structure
            
            # Indicators
            close = df['Close']
            df['SMA50'] = close.rolling(window=50).mean()
            df['SMA200'] = close.rolling(window=200).mean()
            df['RSI'] = calculate_rsi(close)
            
            # Logic
            curr_sma50 = df['SMA50'].iloc[-1]
            curr_sma200 = df['SMA200'].iloc[-1]
            prev_sma50 = df['SMA50'].iloc[-2]
            prev_sma200 = df['SMA200'].iloc[-2]
            current_price = close.iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            status = "Neutral"
            signal_strength = "Hold"
            
            # Golden Cross: 50 crosses ABOVE 200
            if prev_sma50 < prev_sma200 and curr_sma50 > curr_sma200:
                status = "🌟 GOLDEN CROSS"
                signal_strength = "Strong Buy"
            # Death Cross: 50 crosses BELOW 200
            elif prev_sma50 > prev_sma200 and curr_sma50 < curr_sma200:
                status = "💀 DEATH CROSS"
                signal_strength = "Strong Sell"
            # Watchlist (Close together)
            elif abs(curr_sma50 - curr_sma200) / curr_sma200 < 0.02: # 2% diff
                if curr_sma50 < curr_sma200:
                    status = "👀 Watchlist (Nearing Golden)"
                else:
                    status = "⚠️ Watchlist (Nearing Death)"
            elif curr_sma50 > curr_sma200:
                status = "✅ Bullish Trend"
            elif curr_sma50 < curr_sma200:
                status = "🔻 Bearish Trend"
                
            results.append({
                'Ticker': ticker,
                'Price': current_price,
                'Status': status,
                'RSI': rsi,
                'SMA50': curr_sma50,
                'SMA200': curr_sma200,
                'Signal': signal_strength,
                'History': df # Store dataframe for plotting later
            })
            
        except Exception as e:
            # print(f"Error {ticker}: {e}")
            pass
            
        progress_bar.progress((i + 1) / len(tickers))
        
    progress_bar.empty()
    return pd.DataFrame(results)

def render_scanner_tab(default_portfolio_df):
    st.markdown("### 🔍 Technical Scanner (Golden Cross)")
    
    # Check session state for existing results
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = None

    # --- Input Section (Wrapped in Form for Enter Key support) ---
    with st.form(key='scanner_form'):
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.markdown("**1. Select Source**")
            use_portfolio = st.checkbox("Scan Current Portfolio", value=True)
            
        with c2:
            st.markdown("**2. Repository CSVs**")
            # Find CSVs in current directory
            repo_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != 'requirements.txt']
            repo_files.insert(0, "None")
            selected_csv = st.selectbox("Add tickers from file:", repo_files)
        
        with c3:
            st.markdown("**3. Manual Add**")
            manual_tickers = st.text_input("Tickers (comma separated):", placeholder="AAPL, TSLA34, BTC-USD")

        # Submit Button triggers on Enter key in text_input
        submit_button = st.form_submit_button("🚀 Run Scanner", type="primary")

    # --- Processing Logic (Triggered by Form Submit) ---
    if submit_button:
        # Build Ticker List
        scan_list = []
        
        if use_portfolio and not default_portfolio_df.empty:
            scan_list.extend(default_portfolio_df['Ticker'].unique().tolist())
            
        if selected_csv != "None":
            try:
                csv_df = pd.read_csv(selected_csv)
                possible_cols = [c for c in csv_df.columns if 'Ticker' in c or 'Ativo' in c or 'Symbol' in c]
                if possible_cols:
                    scan_list.extend(csv_df[possible_cols[0]].astype(str).tolist())
                else:
                    st.error(f"Could not find a 'Ticker' column in {selected_csv}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                
        if manual_tickers:
            scan_list.extend([t.strip() for t in manual_tickers.split(',')])

        if not scan_list:
            st.warning("No tickers selected.")
        else:
            with st.spinner("Analyzing market data..."):
                results_df = run_scanner_analysis(scan_list)
                # Store results in Session State to persist across re-runs (like changing chart tabs)
                st.session_state.scanner_results = results_df
    
    # --- Output Section (Rendered from Session State) ---
    if st.session_state.scanner_results is not None and not st.session_state.scanner_results.empty:
        results_df = st.session_state.scanner_results
        
        # 1. Summary Metrics
        n_golden = len(results_df[results_df['Status'].str.contains("GOLDEN")])
        n_watch = len(results_df[results_df['Status'].str.contains("Watchlist")])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Assets Scanned", len(results_df))
        m2.metric("Golden Crosses", n_golden)
        m3.metric("On Watchlist", n_watch)
        
        # 2. Results Table
        st.subheader("Analysis Results")
        
        def highlight_status(val):
            if 'GOLDEN' in val: color = '#90EE90'
            elif 'DEATH' in val: color = '#FFcccb'
            elif 'Watchlist' in val: color = '#FFFACD'
            elif 'Bullish' in val: color = '#E0FFF0'
            else: color = ''
            return f'background-color: {color}; color: black'

        display_df = results_df[['Ticker', 'Status', 'Price', 'RSI', 'Signal']].copy()
        st.dataframe(
            display_df.style.applymap(highlight_status, subset=['Status'])
            .format({'Price': '{:.2f}', 'RSI': '{:.1f}'}),
            use_container_width=True
        )
        
        # 3. Drill Down Chart
        st.divider()
        st.subheader("📉 Technical Chart")
        
        # The selectbox is OUTSIDE the form, so it triggers a re-run.
        # But because we read results from st.session_state, the data persists.
        selected_ticker = st.selectbox("Select asset to view chart:", results_df['Ticker'].unique())
        
        if selected_ticker:
            row = results_df[results_df['Ticker'] == selected_ticker].iloc[0]
            hist_data = row['History']
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'], high=hist_data['High'],
                low=hist_data['Low'], close=hist_data['Close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA50'], line=dict(color='orange', width=2), name='SMA 50'))
            fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA200'], line=dict(color='blue', width=2), name='SMA 200'))
            
            fig.update_layout(
                title=f"{selected_ticker} - SMA 50/200 Analysis",
                xaxis_title="Date", yaxis_title="Price", height=600, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif submit_button: # Only if button pressed but no results (e.g. empty)
        st.warning("No data found for selected tickers.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

st.title("📊 Investment Command Center")

# Load Default Portfolio
df_main = load_csv_data("data.csv")
if df_main is None:
    st.error("Primary file 'data.csv' not found. Please upload it.")
    df_main = pd.DataFrame() # Empty to prevent crashes

# Create Tabs
tab1, tab2 = st.tabs(["💼 Portfolio Performance", "🔬 Technical Scanner"])

with tab1:
    render_portfolio_tab(df_main)

with tab2:
    # Need basic portfolio df for ticker list
    if not df_main.empty:
        # Calculate it briefly just to get the list of held assets
        pf_calc = calculate_portfolio(df_main)
        render_scanner_tab(pf_calc)
    else:
        render_scanner_tab(pd.DataFrame())
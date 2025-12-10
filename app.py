import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# --- Configuration ---
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# --- 1. Data Loading & Cleaning ---
@st.cache_data
def load_data():
    file_path = "data.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # Helper: Convert "R$ 1.000,00" -> 1000.00
        def clean_currency(x):
            if isinstance(x, str):
                clean_str = x.replace('R$', '').replace('.', '').replace(',', '.').strip()
                try:
                    return float(clean_str)
                except ValueError:
                    return 0.0
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

# --- Helper: Auto-categorize Assets ---
def infer_asset_class(ticker):
    """
    Simple logic to guess asset class based on B3 ticker patterns.
    User can override this by adding a 'Classe' column to their CSV.
    """
    ticker = str(ticker).upper().strip()
    if ticker.endswith('3') or ticker.endswith('4'):
        return 'Ações'
    elif ticker.endswith('11'):
        return 'FIIs / ETFs'
    elif ticker.endswith('34'):
        return 'BDRs'
    else:
        return 'Outros'

# --- 2. Live Price Fetching ---
@st.cache_data(ttl=300)
def get_live_prices(tickers):
    # Prepare tickers for Yahoo Finance (add .SA for Brazil if not present)
    formatted_tickers = [f"{t}.SA" if not t.endswith('.SA') and not t.upper() in ['IVVB11', 'HASH11'] else f"{t}.SA" for t in tickers]
    
    if not formatted_tickers:
        return {}

    try:
        data = yf.download(formatted_tickers, period="1d", progress=False)['Close']
        
        if isinstance(data, pd.Series):
            if len(data) > 0:
                 current_prices = {formatted_tickers[0].replace('.SA', ''): data.iloc[-1]}
            else:
                 current_prices = {}
        else:
            # Check if dataframe is empty or columns missing
            if data.empty:
                return {}
            current_prices = {col.replace('.SA', ''): data[col].iloc[-1] for col in data.columns}
            
        return current_prices
    except Exception as e:
        return {}

# --- 3. Portfolio Calculation (With Sales Logic) ---
def process_ticker_transactions(group):
    """
    Iterates through transactions to calculate current position and cost basis.
    Logic:
    - Buy: Increase Qty, Increase Cost
    - Sell: Decrease Qty, Decrease Cost PROPORTIONALLY (maintaining Avg Price)
    """
    # Sort by date is crucial for FIFO/Avg Price logic
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
                # Calculate avg price before sale
                avg_price = current_cost / current_qty
                
                # Reduce quantity
                current_qty -= qty
                
                # Reduce cost proportionally (Cost Basis Adjustment)
                # New Cost = Remaining Qty * Avg Price
                current_cost = current_qty * avg_price
            else:
                # Error state: Selling what you don't have
                pass
                
    return pd.Series({'Quantidade': current_qty, 'Custo total com taxas': current_cost})

def calculate_portfolio(df):
    # Determine Asset Class if missing
    if 'Classe' not in df.columns:
        df['Classe'] = df['Ticker'].apply(infer_asset_class)

    # Apply the transaction processor to each Ticker
    portfolio = df.groupby(['Ticker', 'Classe']).apply(process_ticker_transactions).reset_index()

    # Filter out closed positions (Qty <= 0) and very small residuals
    portfolio = portfolio[portfolio['Quantidade'] > 0.0001]

    # Calculate Average Price based on the final positions
    portfolio['Average Price'] = portfolio['Custo total com taxas'] / portfolio['Quantidade']
    
    return portfolio

# --- Main App Logic ---
df = load_data()

if df is not None:
    st.title("📊 Market Performance Dashboard")

    # 1. Prepare Portfolio Data (Now handling Sales)
    portfolio_df = calculate_portfolio(df)
    
    if portfolio_df.empty:
        st.warning("No active assets found in portfolio (everything sold or empty file).")
    else:
        # 2. Get Live Prices
        with st.spinner('Fetching latest market data...'):
            unique_tickers = portfolio_df['Ticker'].unique().tolist()
            live_prices = get_live_prices(unique_tickers)
        
        # 3. Merge Live Data
        portfolio_df['Current Price'] = portfolio_df['Ticker'].map(live_prices)
        portfolio_df['Current Price'] = portfolio_df['Current Price'].fillna(portfolio_df['Average Price'])

        # 4. Calculate KPIs
        portfolio_df['Current Value'] = portfolio_df['Quantidade'] * portfolio_df['Current Price']
        portfolio_df['Gain/Loss Value'] = portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']
        portfolio_df['Gain/Loss %'] = ((portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']) / portfolio_df['Custo total com taxas']) * 100

        # Sort for charts
        portfolio_df = portfolio_df.sort_values(by='Gain/Loss %', ascending=False)

        # --- Section: Top Level Metrics ---
        total_invested = portfolio_df['Custo total com taxas'].sum()
        current_portfolio_value = portfolio_df['Current Value'].sum()
        total_gain_loss = current_portfolio_value - total_invested
        total_gain_loss_pct = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"R$ {total_invested:,.2f}")
        col2.metric("📈 Current Value", f"R$ {current_portfolio_value:,.2f}", delta=f"{total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%)")
        
        if not portfolio_df.empty:
            best_stock = portfolio_df.iloc[0]
            col3.metric("🏆 Best Performer", f"{best_stock['Ticker']}", delta=f"{best_stock['Gain/Loss %']:.2f}%")

        st.divider()

        # --- Section: Portfolio Composition (Sunburst) ---
        st.subheader("🎨 Portfolio Composition")
        st.caption("Click on a sector (e.g., 'Ações') to drill down and see specific assets.")
        
        fig_sun = px.sunburst(
            portfolio_df,
            path=['Classe', 'Ticker'],
            values='Current Value',
            color='Classe',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            custom_data=['Gain/Loss %', 'Current Value']
        )
        fig_sun.update_traces(
            textinfo="label+percent parent",
            hovertemplate="<b>%{label}</b><br>Value: R$ %{value:,.2f}<br>Share: %{percentRoot:.1%}<extra></extra>"
        )
        fig_sun.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)
        
        st.divider()

        # --- Section: Visualizations ---
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("🚀 Performance by Asset (%)")
            fig_perf = px.bar(
                portfolio_df, 
                x='Gain/Loss %', 
                y='Ticker', 
                orientation='h',
                text='Gain/Loss %',
                title="Return (%) per Asset",
                color='Gain/Loss %',
                color_continuous_scale=['red', 'yellow', 'green'],
                range_color=[-20, 20]
            )
            fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_perf.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_chart2:
            st.subheader("⚖️ Average Cost vs. Current Price")
            comparison_df = portfolio_df[['Ticker', 'Average Price', 'Current Price']].melt(
                id_vars='Ticker', var_name='Price Type', value_name='Price'
            )
            fig_comp = px.bar(
                comparison_df, 
                x='Ticker', y='Price', 
                color='Price Type', 
                barmode='group',
                title="Price Comparison: Paid vs. Market",
                color_discrete_map={'Average Price': 'gray', 'Current Price': '#00CC96'}
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # --- Section: Winners & Losers ---
        c1, c2 = st.columns(2)
        with c1:
            st.caption("📉 Worst Performing Assets")
            st.dataframe(portfolio_df[['Ticker', 'Classe', 'Average Price', 'Current Price', 'Gain/Loss %']].sort_values('Gain/Loss %').head(3).style.format({'Average Price': 'R$ {:.2f}', 'Current Price': 'R$ {:.2f}', 'Gain/Loss %': '{:.2f}%'}))
        
        with c2:
            st.caption("📈 Best Performing Assets")
            st.dataframe(portfolio_df[['Ticker', 'Classe', 'Average Price', 'Current Price', 'Gain/Loss %']].sort_values('Gain/Loss %', ascending=False).head(3).style.format({'Average Price': 'R$ {:.2f}', 'Current Price': 'R$ {:.2f}', 'Gain/Loss %': '{:.2f}%'}))

        # --- Data Table ---
        with st.expander("View Consolidated Portfolio Data"):
            st.dataframe(portfolio_df.style.format({
                'Quantidade': '{:.0f}',
                'Custo total com taxas': 'R$ {:,.2f}',
                'Average Price': 'R$ {:,.2f}',
                'Current Price': 'R$ {:,.2f}',
                'Current Value': 'R$ {:,.2f}',
                'Gain/Loss Value': 'R$ {:,.2f}',
                'Gain/Loss %': '{:,.2f}%'
            }))
        
else:
    st.error("Could not find 'data.csv'. Please ensure it is in the directory.")
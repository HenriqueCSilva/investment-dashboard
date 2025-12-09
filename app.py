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

# --- 2. Live Price Fetching ---
@st.cache_data(ttl=300)  # Cache data for 5 minutes to avoid spamming Yahoo Finance
def get_live_prices(tickers):
    # Prepare tickers for Yahoo Finance (add .SA for Brazil if not present)
    # Note: If you have international stocks, this logic might need adjustment
    formatted_tickers = [f"{t}.SA" if not t.endswith('.SA') else t for t in tickers]
    
    if not formatted_tickers:
        return {}

    try:
        # Bulk download is faster
        data = yf.download(formatted_tickers, period="1d", progress=False)['Close']
        
        # If we only have one ticker, yf returns a Series, not a DataFrame. We fix that.
        if isinstance(data, pd.Series):
            current_prices = {formatted_tickers[0].replace('.SA', ''): data.iloc[-1]}
        else:
            # Get the most recent closing price for each ticker
            current_prices = {col.replace('.SA', ''): data[col].iloc[-1] for col in data.columns}
            
        return current_prices
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return {}

# --- 3. Portfolio Calculation ---
def calculate_portfolio(df):
    # Group by Ticker to get Total Quantity and Total Invested
    portfolio = df.groupby('Ticker').agg({
        'Quantidade': 'sum',
        'Custo total com taxas': 'sum'
    }).reset_index()

    # Calculate Average Price
    portfolio['Average Price'] = portfolio['Custo total com taxas'] / portfolio['Quantidade']
    
    return portfolio

# --- Main App Logic ---
df = load_data()

if df is not None:
    st.title("📊 Market Performance Dashboard")
    st.markdown("### Real-time vs. Acquisition Costs")

    # 1. Prepare Portfolio Data
    portfolio_df = calculate_portfolio(df)
    
    # 2. Get Live Prices
    with st.spinner('Fetching latest market data...'):
        unique_tickers = portfolio_df['Ticker'].unique().tolist()
        live_prices = get_live_prices(unique_tickers)
    
    # 3. Merge Live Data
    # Map the live prices to our portfolio dataframe
    portfolio_df['Current Price'] = portfolio_df['Ticker'].map(live_prices)
    
    # Handle missing prices (e.g., if a ticker changed or wasn't found)
    # We fill NaNs with the Average Price to assume 0% gain/loss for missing data
    portfolio_df['Current Price'] = portfolio_df['Current Price'].fillna(portfolio_df['Average Price'])

    # 4. Calculate KPIs
    portfolio_df['Current Value'] = portfolio_df['Quantidade'] * portfolio_df['Current Price']
    portfolio_df['Gain/Loss Value'] = portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']
    portfolio_df['Gain/Loss %'] = ((portfolio_df['Current Value'] - portfolio_df['Custo total com taxas']) / portfolio_df['Custo total com taxas']) * 100

    # Sort by performance for charts
    portfolio_df = portfolio_df.sort_values(by='Gain/Loss %', ascending=False)

    # --- Section: Top Level Metrics ---
    total_invested = portfolio_df['Custo total com taxas'].sum()
    current_portfolio_value = portfolio_df['Current Value'].sum()
    total_gain_loss = current_portfolio_value - total_invested
    total_gain_loss_pct = (total_gain_loss / total_invested) * 100

    col1, col2, col3 = st.columns(3)
    
    col1.metric("💰 Total Invested", f"R$ {total_invested:,.2f}")
    
    col2.metric(
        "📈 Current Portfolio Value", 
        f"R$ {current_portfolio_value:,.2f}",
        delta=f"{total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%)"
    )

    best_stock = portfolio_df.iloc[0]
    col3.metric(
        "🏆 Best Performer",
        f"{best_stock['Ticker']}",
        delta=f"{best_stock['Gain/Loss %']:.2f}%"
    )

    st.divider()

    # --- Section: Visualizations ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("🚀 Performance by Asset (%)")
        
        # Color logic: Green for positive, Red for negative
        fig_perf = px.bar(
            portfolio_df, 
            x='Gain/Loss %', 
            y='Ticker', 
            orientation='h',
            text='Gain/Loss %',
            title="Return (%) per Asset",
            color='Gain/Loss %',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[-20, 20] # Adjusts the color intensity range
        )
        fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_perf.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_chart2:
        st.subheader("⚖️ Average Cost vs. Current Price")
        
        # Comparison Chart
        # We reshape the data to "Long Format" for Plotly Grouped Bar Chart
        comparison_df = portfolio_df[['Ticker', 'Average Price', 'Current Price']].melt(
            id_vars='Ticker', 
            var_name='Price Type', 
            value_name='Price'
        )
        
        fig_comp = px.bar(
            comparison_df, 
            x='Ticker', 
            y='Price', 
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
        st.dataframe(
            portfolio_df[['Ticker', 'Average Price', 'Current Price', 'Gain/Loss %']]
            .sort_values('Gain/Loss %', ascending=True)
            .head(3)
            .style.format({
                'Average Price': 'R$ {:.2f}', 
                'Current Price': 'R$ {:.2f}', 
                'Gain/Loss %': '{:.2f}%'
            })
        )
    
    with c2:
        st.caption("📈 Best Performing Assets")
        st.dataframe(
            portfolio_df[['Ticker', 'Average Price', 'Current Price', 'Gain/Loss %']]
            .sort_values('Gain/Loss %', ascending=False)
            .head(3)
            .style.format({
                'Average Price': 'R$ {:.2f}', 
                'Current Price': 'R$ {:.2f}', 
                'Gain/Loss %': '{:.2f}%'
            })
        )

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
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Investment Dashboard", layout="wide")

@st.cache_data
def load_data():
    # If you didn't rename the file, change 'data.csv' to your exact filename
    file_path = "data.csv"
    
    try:
        # Load csv
        df = pd.read_csv(file_path)
        
        # --- Cleaning Data ---
        # Helper function to convert "R$ 1.000,00" -> 1000.00
        def clean_currency(x):
            if isinstance(x, str):
                clean_str = x.replace('R$', '').replace('.', '').replace(',', '.').strip()
                try:
                    return float(clean_str)
                except ValueError:
                    return 0.0
            return x

        # Clean specific columns
        currency_cols = ['Preço', 'Custo total com taxas', 'Preço médio com taxas', 'Taxas totais']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_currency)

        # Fix Dates
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
            
        return df
        
    except FileNotFoundError:
        return None

# --- App Logic ---
df = load_data()

if df is not None:
    st.title("📊 My Investment Dashboard")

    # --- Metrics Section ---
    total_invested = df['Custo total com taxas'].sum()
    total_assets = df['Quantidade'].sum()
    
    col1, col2 = st.columns(2)
    col1.metric("💰 Total Invested", f"R$ {total_invested:,.2f}")
    col2.metric("📦 Total Assets", int(total_assets))
    
    st.divider()

    # --- Charts Section ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Allocation by Asset")
        # Sum costs by Ticker
        df_ticker = df.groupby('Ticker')['Custo total com taxas'].sum().reset_index()
        fig_pie = px.pie(df_ticker, values='Custo total com taxas', names='Ticker', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Investment Evolution")
        # Sort by date to show timeline
        df_time = df.sort_values('Data')
        fig_line = px.bar(df_time, x='Data', y='Custo total com taxas', title="Investments over Time")
        st.plotly_chart(fig_line, use_container_width=True)

    # --- Data Table ---
    with st.expander("View Raw Data"):
        st.dataframe(df)
        
else:
    st.error("Could not find the file. Please make sure 'data.csv' is in the same folder as this script.")
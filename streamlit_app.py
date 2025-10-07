import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# Company name to ticker mapping (add more as needed)
COMPANY_TICKER_MAP = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'meta': 'META',
    'facebook': 'META',
    'nvidia': 'NVDA',
    'netflix': 'NFLX',
    'amd': 'AMD',
    'intel': 'INTC',
    'coca cola': 'KO',
    'pepsi': 'PEP',
    'disney': 'DIS',
    'walmart': 'WMT',
    'jpmorgan': 'JPM',
    'visa': 'V',
    'mastercard': 'MA',
    'boeing': 'BA',
    'nike': 'NKE',
    'starbucks': 'SBUX',
    'mcdonalds': 'MCD',
}

def get_ticker(query):
    """Convert company name or ticker to valid ticker symbol"""
    query_lower = query.lower().strip()
    
    if query_lower in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[query_lower]
    
    try:
        test_ticker = yf.Ticker(query.upper())
        info = test_ticker.info
        
        if info and len(info) > 5:
            return query.upper()
    except:
        pass
    
    return query.upper()

def get_stock_data(ticker):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        return stock, info, hist
    except Exception as e:
        return None, None, None

def calculate_pe_history(hist, info):
    """Calculate P/E ratio history based on price and EPS"""
    eps = info.get('trailingEps')
    if eps and eps > 0 and not hist.empty:
        pe_history = hist['Close'] / eps
        return pe_history
    return None

def export_to_csv(data_dict, ticker):
    """Export data to CSV"""
    df = pd.DataFrame(list(data_dict.items()), columns=['Metric', 'Value'])
    csv = df.to_csv(index=False)
    return csv

def display_analyst_recommendations(stock, info):
    """Display analyst recommendations"""
    st.subheader("Analyst Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_high = info.get('targetHighPrice', 'N/A')
        st.metric("Target High", f"${target_high:.2f}" if target_high != 'N/A' else 'N/A')
    
    with col2:
        target_mean = info.get('targetMeanPrice', 'N/A')
        st.metric("Target Mean", f"${target_mean:.2f}" if target_mean != 'N/A' else 'N/A')
    
    with col3:
        target_low = info.get('targetLowPrice', 'N/A')
        st.metric("Target Low", f"${target_low:.2f}" if target_low != 'N/A' else 'N/A')
    
    # Recommendation details
    recommendation = info.get('recommendationKey', 'N/A')
    num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
    
    st.write(f"**Recommendation:** {recommendation.upper() if recommendation != 'N/A' else 'N/A'}")
    st.write(f"**Number of Analysts:** {num_analysts}")
    
    # Try to get recommendation trend
    try:
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            st.write("**Recent Recommendations:**")
            st.dataframe(recommendations.tail(10), use_container_width=True)
    except:
        pass

def display_financial_statements(stock):
    """Display financial statements"""
    st.subheader("Financial Statements")
    
    # Toggle between Annual and Quarterly
    period_type = st.radio("Select Period:", ["Annual", "Quarterly"], horizontal=True)
    
    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
    with tab1:
        try:
            if period_type == "Annual":
                income_stmt = stock.financials
            else:
                income_stmt = stock.quarterly_financials
            
            if income_stmt is not None and not income_stmt.empty:
                # Format numbers for better readability
                income_stmt_formatted = income_stmt.copy()
                for col in income_stmt_formatted.columns:
                    income_stmt_formatted[col] = income_stmt_formatted[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                    )
                st.dataframe(income_stmt_formatted, use_container_width=True)
            else:
                st.write(f"No {period_type.lower()} income statement data available")
        except:
            st.write("Unable to fetch income statement")
    
    with tab2:
        try:
            if period_type == "Annual":
                balance_sheet = stock.balance_sheet
            else:
                balance_sheet = stock.quarterly_balance_sheet
            
            if balance_sheet is not None and not balance_sheet.empty:
                # Format numbers for better readability
                balance_sheet_formatted = balance_sheet.copy()
                for col in balance_sheet_formatted.columns:
                    balance_sheet_formatted[col] = balance_sheet_formatted[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                    )
                
                # Organize balance sheet into sections
                st.write("**ASSETS**")
                st.write("---")
                
                # Filter for asset-related rows
                asset_keywords = ['Asset', 'Cash', 'Receivable', 'Inventory', 'Investment', 
                                 'Property', 'Equipment', 'Goodwill', 'Intangible']
                asset_rows = balance_sheet_formatted.index[
                    balance_sheet_formatted.index.str.contains('|'.join(asset_keywords), case=False, na=False)
                ]
                
                if len(asset_rows) > 0:
                    st.dataframe(balance_sheet_formatted.loc[asset_rows], use_container_width=True)
                
                st.write("")
                st.write("**LIABILITIES**")
                st.write("---")
                
                # Filter for liability-related rows
                liability_keywords = ['Liabilit', 'Debt', 'Payable', 'Deferred', 'Accrued']
                liability_rows = balance_sheet_formatted.index[
                    balance_sheet_formatted.index.str.contains('|'.join(liability_keywords), case=False, na=False)
                ]
                
                if len(liability_rows) > 0:
                    st.dataframe(balance_sheet_formatted.loc[liability_rows], use_container_width=True)
                
                st.write("")
                st.write("**EQUITY**")
                st.write("---")
                
                # Filter for equity-related rows
                equity_keywords = ['Equity', 'Stock', 'Retained Earnings', 'Capital']
                equity_rows = balance_sheet_formatted.index[
                    balance_sheet_formatted.index.str.contains('|'.join(equity_keywords), case=False, na=False)
                ]
                
                if len(equity_rows) > 0:
                    st.dataframe(balance_sheet_formatted.loc[equity_rows], use_container_width=True)
                
                # Show any remaining rows that didn't match
                all_categorized = list(asset_rows) + list(liability_rows) + list(equity_rows)
                remaining_rows = balance_sheet_formatted.index.difference(all_categorized)
                
                if len(remaining_rows) > 0:
                    st.write("")
                    st.write("**OTHER**")
                    st.write("---")
                    st.dataframe(balance_sheet_formatted.loc[remaining_rows], use_container_width=True)
                
            else:
                st.write(f"No {period_type.lower()} balance sheet data available")
        except Exception as e:
            st.write("Unable to fetch balance sheet")
    
    with tab3:
        try:
            if period_type == "Annual":
                cash_flow = stock.cashflow
            else:
                cash_flow = stock.quarterly_cashflow
            
            if cash_flow is not None and not cash_flow.empty:
                # Format numbers for better readability
                cash_flow_formatted = cash_flow.copy()
                for col in cash_flow_formatted.columns:
                    cash_flow_formatted[col] = cash_flow_formatted[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                    )
                st.dataframe(cash_flow_formatted, use_container_width=True)
            else:
                st.write(f"No {period_type.lower()} cash flow data available")
        except:
            st.write("Unable to fetch cash flow statement")

def display_dividend_info(stock, info, hist):
    """Display dividend information"""
    st.subheader("Dividend Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        div_yield = info.get('dividendYield', 'N/A')
        if div_yield != 'N/A' and div_yield:
            st.metric("Dividend Yield", f"{div_yield * 100:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
    
    with col2:
        div_rate = info.get('dividendRate', 'N/A')
        st.metric("Annual Dividend", f"${div_rate:.2f}" if div_rate != 'N/A' else 'N/A')
    
    with col3:
        payout_ratio = info.get('payoutRatio', 'N/A')
        if payout_ratio != 'N/A' and payout_ratio:
            st.metric("Payout Ratio", f"{payout_ratio * 100:.2f}%")
        else:
            st.metric("Payout Ratio", "N/A")
    
    with col4:
        ex_div_date = info.get('exDividendDate', 'N/A')
        if ex_div_date != 'N/A':
            ex_div_date_str = datetime.fromtimestamp(ex_div_date).strftime('%Y-%m-%d')
            st.metric("Ex-Dividend Date", ex_div_date_str)
        else:
            st.metric("Ex-Dividend Date", "N/A")
    
    # Dividend history
    try:
        dividends = stock.dividends
        if dividends is not None and not dividends.empty and len(dividends) > 0:
            st.write("**Dividend History:**")
            
            # Create dividend chart
            fig_div = go.Figure()
            fig_div.add_trace(go.Bar(
                x=dividends.index,
                y=dividends.values,
                name='Dividends',
                marker_color='green'
            ))
            fig_div.update_layout(
                title='Dividend History',
                xaxis_title='Date',
                yaxis_title='Dividend ($)',
                template='plotly_white'
            )
            st.plotly_chart(fig_div, use_container_width=True)
    except:
        st.write("No dividend history available")

def display_earnings_calendar(stock, info):
    """Display earnings information"""
    st.subheader("Earnings Calendar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        earnings_date = info.get('earningsTimestamp', 'N/A')
        if earnings_date != 'N/A':
            earnings_date_str = datetime.fromtimestamp(earnings_date).strftime('%Y-%m-%d %H:%M')
            st.write(f"**Next Earnings Date:** {earnings_date_str}")
        else:
            st.write("**Next Earnings Date:** N/A")
    
    with col2:
        earnings_high = info.get('earningsQuarterlyGrowth', 'N/A')
        if earnings_high != 'N/A' and earnings_high:
            st.write(f"**Earnings Growth (QoQ):** {earnings_high * 100:.2f}%")
        else:
            st.write("**Earnings Growth (QoQ):** N/A")
    
    # Earnings history
    try:
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            st.write("**Earnings History:**")
            st.dataframe(earnings.head(10), use_container_width=True)
    except:
        pass
    
    # Quarterly earnings
    try:
        quarterly_earnings = stock.quarterly_earnings
        if quarterly_earnings is not None and not quarterly_earnings.empty:
            st.write("**Quarterly Earnings:**")
            st.dataframe(quarterly_earnings, use_container_width=True)
    except:
        pass

def display_institutional_holdings(stock):
    """Display institutional holdings"""
    st.subheader("Institutional Holdings")
    
    try:
        institutional = stock.institutional_holders
        if institutional is not None and not institutional.empty:
            st.dataframe(institutional, use_container_width=True)
        else:
            st.write("No institutional holdings data available")
    except:
        st.write("Unable to fetch institutional holdings")
    
    try:
        major_holders = stock.major_holders
        if major_holders is not None and not major_holders.empty:
            st.write("**Major Holders Summary:**")
            st.dataframe(major_holders, use_container_width=True)
    except:
        pass

def display_stock_comparison(tickers_list):
    """Compare multiple stocks"""
    st.subheader("Stock Comparison")
    
    comparison_data = []
    
    for ticker_input in tickers_list:
        ticker = get_ticker(ticker_input.strip())
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and len(info) > 1:
            comparison_data.append({
                'Ticker': ticker,
                'Name': info.get('longName', ticker),
                'Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52W Low': info.get('fiftyTwoWeekLow', 'N/A')
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Format the dataframe
        if 'Market Cap' in df_comparison.columns:
            df_comparison['Market Cap'] = df_comparison['Market Cap'].apply(
                lambda x: f"${x/1e9:.2f}B" if x != 'N/A' else 'N/A'
            )
        
        if 'Dividend Yield' in df_comparison.columns:
            df_comparison['Dividend Yield'] = df_comparison['Dividend Yield'].apply(
                lambda x: f"{x*100:.2f}%" if x != 'N/A' else 'N/A'
            )
        
        st.dataframe(df_comparison, use_container_width=True)
        
        # Price comparison chart
        fig_comparison = go.Figure()
        
        for ticker_input in tickers_list:
            ticker = get_ticker(ticker_input.strip())
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if not hist.empty:
                # Normalize to percentage change
                normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                fig_comparison.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=ticker
                ))
        
        fig_comparison.update_layout(
            title='Price Comparison (% Change from 1 Year Ago)',
            xaxis_title='Date',
            yaxis_title='% Change',
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.write("Unable to fetch comparison data")

def display_stock_info(ticker, info, hist, stock):
    """Display stock information in a structured way"""
    
    st.header(f"{info.get('longName', ticker)} ({ticker})")
    
    # Export button
    export_data = {
        'Ticker': ticker,
        'Name': info.get('longName', ticker),
        'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
        'Market Cap': info.get('marketCap', 'N/A'),
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'Beta': info.get('beta', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A')
    }
    
    csv_data = export_to_csv(export_data, ticker)
    st.download_button(
        label="Download Key Metrics (CSV)",
        data=csv_data,
        file_name=f"{ticker}_metrics.csv",
        mime="text/csv"
    )
    
    # Time period selector for charts
    st.subheader("Chart Time Period")
    time_period = st.selectbox(
        "Select time period for charts:",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max (All Available Data)"],
        index=3  # Default to 1 Year
    )
    
    # Map selection to yfinance period parameter
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max (All Available Data)": "max"
    }
    
    selected_period = period_map[time_period]
    
    # Fetch historical data for selected period
    with st.spinner(f"Loading {time_period} data..."):
        hist_selected = stock.history(period=selected_period)
    
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        st.metric("Current Price", f"${current_price:.2f}" if current_price != 'N/A' else 'N/A')
    
    with col2:
        prev_close = info.get('previousClose', 'N/A')
        change = current_price - prev_close if current_price != 'N/A' and prev_close != 'N/A' else 'N/A'
        change_pct = (change / prev_close * 100) if change != 'N/A' else 'N/A'
        st.metric("Change", 
                  f"${change:.2f} ({change_pct:.2f}%)" if change != 'N/A' else 'N/A',
                  delta=f"{change_pct:.2f}%" if change_pct != 'N/A' else None)
    
    with col3:
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap_formatted = f"${market_cap / 1e9:.2f}B"
        else:
            market_cap_formatted = 'N/A'
        st.metric("Market Cap", market_cap_formatted)
    
    with col4:
        volume = info.get('volume', 'N/A')
        if volume != 'N/A':
            volume_formatted = f"{volume / 1e6:.2f}M"
        else:
            volume_formatted = 'N/A'
        st.metric("Volume", volume_formatted)
    
    # Financial Ratios Section
    st.subheader("Financial Ratios & Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Valuation Ratios**")
        pe_ratio = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        peg_ratio = info.get('pegRatio', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        ps_ratio = info.get('priceToSalesTrailing12Months', 'N/A')
        
        st.write(f"P/E Ratio (TTM): **{pe_ratio if pe_ratio != 'N/A' else 'N/A'}**")
        st.write(f"Forward P/E: **{forward_pe if forward_pe != 'N/A' else 'N/A'}**")
        st.write(f"PEG Ratio: **{peg_ratio if peg_ratio != 'N/A' else 'N/A'}**")
        st.write(f"Price/Book: **{pb_ratio if pb_ratio != 'N/A' else 'N/A'}**")
        st.write(f"Price/Sales: **{ps_ratio if ps_ratio != 'N/A' else 'N/A'}**")
    
    with col2:
        st.write("**Profitability Ratios**")
        profit_margin = info.get('profitMargins', 'N/A')
        operating_margin = info.get('operatingMargins', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        roa = info.get('returnOnAssets', 'N/A')
        
        if profit_margin != 'N/A':
            st.write(f"Profit Margin: **{profit_margin * 100:.2f}%**")
        else:
            st.write(f"Profit Margin: **N/A**")
        
        if operating_margin != 'N/A':
            st.write(f"Operating Margin: **{operating_margin * 100:.2f}%**")
        else:
            st.write(f"Operating Margin: **N/A**")
        
        if roe != 'N/A':
            st.write(f"Return on Equity: **{roe * 100:.2f}%**")
        else:
            st.write(f"Return on Equity: **N/A**")
        
        if roa != 'N/A':
            st.write(f"Return on Assets: **{roa * 100:.2f}%**")
        else:
            st.write(f"Return on Assets: **N/A**")
    
    with col3:
        st.write("**Risk Metrics**")
        beta = info.get('beta', 'N/A')
        debt_to_equity = info.get('debtToEquity', 'N/A')
        current_ratio = info.get('currentRatio', 'N/A')
        quick_ratio = info.get('quickRatio', 'N/A')
        
        st.write(f"Beta: **{beta if beta != 'N/A' else 'N/A'}**")
        st.write(f"Debt/Equity: **{debt_to_equity if debt_to_equity != 'N/A' else 'N/A'}**")
        st.write(f"Current Ratio: **{current_ratio if current_ratio != 'N/A' else 'N/A'}**")
        st.write(f"Quick Ratio: **{quick_ratio if quick_ratio != 'N/A' else 'N/A'}**")
        
        # 52 Week Range
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        st.write(f"52W Range: **${week_52_low} - ${week_52_high}**")
    
    # Company Information
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
        st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
    
    with col2:
        st.write(f"**EPS (TTM):** {info.get('trailingEps', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
    
    # Business Summary
    if 'longBusinessSummary' in info:
        st.subheader("Business Summary")
        st.write(info['longBusinessSummary'])
    
    # Analyst Recommendations
    display_analyst_recommendations(stock, info)
    
    # Dividend Information
    display_dividend_info(stock, info, hist)
    
    # Earnings Calendar
    display_earnings_calendar(stock, info)
    
    # Charts Section
    st.subheader("Interactive Charts")
    
    # Price Chart
    if not hist_selected.empty:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist_selected.index,
            y=hist_selected['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_price.update_layout(
            title=f'Stock Price History ({time_period})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume Chart
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=hist_selected.index,
            y=hist_selected['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        fig_volume.update_layout(
            title=f'Trading Volume ({time_period})',
            xaxis_title='Date',
            yaxis_title='Volume',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # P/E Ratio History Chart
        pe_history = calculate_pe_history(hist_selected, info)
        if pe_history is not None:
            fig_pe = go.Figure()
            fig_pe.add_trace(go.Scatter(
                x=pe_history.index,
                y=pe_history,
                mode='lines',
                name='P/E Ratio',
                line=dict(color='green', width=2)
            ))
            fig_pe.update_layout(
                title=f'P/E Ratio History ({time_period})',
                xaxis_title='Date',
                yaxis_title='P/E Ratio',
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_pe, use_container_width=True)
    else:
        st.write("No historical data available for selected period")
    
    # Financial Statements
    display_financial_statements(stock)
    
    # Institutional Holdings
    display_institutional_holdings(stock)
    
    # Historical Data Table
    st.subheader("Recent Trading Data")
    if not hist_selected.empty:
        display_hist = hist_selected[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
        display_hist = display_hist.sort_index(ascending=False)
        st.dataframe(display_hist, use_container_width=True)
        
        # Export historical data
        csv_hist = hist_selected.to_csv()
        st.download_button(
            label=f"Download Historical Data ({time_period}) - CSV",
            data=csv_hist,
            file_name=f"{ticker}_historical_data_{selected_period}.csv",
            mime="text/csv"
        )

# Streamlit App
st.set_page_config(page_title="Stock Information Scraper", page_icon="📈", layout="wide")

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None

st.title("Stock Information Scraper")
st.write("Enter a stock ticker to get detailed information or compare multiple stocks")

# Tabs for different features
tab1, tab2 = st.tabs(["Single Stock Analysis", "Stock Comparison"])

with tab1:
    # Input
    query = st.text_input("Enter ticker symbol:", placeholder="e.g., AAPL, TSLA, NVDA, GME, 0700.HK")

    if st.button("Get Stock Info"):
        if query:
            with st.spinner("Fetching stock data..."):
                ticker = get_ticker(query)
                stock, info, hist = get_stock_data(ticker)
                
                if info and len(info) > 1:
                    st.session_state.stock_data = {
                        'ticker': ticker,
                        'stock': stock,
                        'info': info,
                        'hist': hist
                    }
                else:
                    st.error(f"Could not find stock data for '{query}'. Please check the ticker symbol.")
                    st.info("Tip: Use the exact ticker symbol (e.g., AAPL, TSLA, MSFT). For international stocks, include the exchange suffix (e.g., 0700.HK for Tencent, SAP.DE for SAP)")
                    st.session_state.stock_data = None
        else:
            st.warning("Please enter a ticker symbol")
    
    # Display stock info if data exists in session state
    if st.session_state.stock_data is not None:
        data = st.session_state.stock_data
        display_stock_info(data['ticker'], data['info'], data['hist'], data['stock'])

with tab2:
    st.write("Compare up to 5 stocks side by side")
    
    comparison_input = st.text_input(
        "Enter ticker symbols separated by commas:",
        placeholder="e.g., AAPL, MSFT, GOOGL"
    )
    
    if st.button("Compare Stocks"):
        if comparison_input:
            tickers_list = [t.strip() for t in comparison_input.split(',')]
            
            if len(tickers_list) > 5:
                st.warning("Please compare a maximum of 5 stocks at a time")
            elif len(tickers_list) < 2:
                st.warning("Please enter at least 2 stocks to compare")
            else:
                with st.spinner("Fetching comparison data..."):
                    st.session_state.comparison_data = tickers_list
        else:
            st.warning("Please enter ticker symbols to compare")
    
    # Display comparison if data exists in session state
    if st.session_state.comparison_data is not None:
        display_stock_comparison(st.session_state.comparison_data)

# Sidebar with app info
st.sidebar.header("About This App")
st.sidebar.write("""
This Stock Information Scraper provides comprehensive financial data and analysis for any publicly traded stock worldwide.

**Features:**
- Real-time stock prices and key metrics
- Financial ratios and valuation metrics
- Analyst recommendations and target prices
- Complete financial statements
- Dividend history and analysis
- Earnings calendar and history
- Institutional holdings information
- Stock comparison tool
- Export data to CSV
- Interactive charts and visualizations

**How to Use:**
1. Enter any stock ticker symbol
2. Click "Get Stock Info" for detailed analysis
3. Use the comparison tab to compare multiple stocks
4. Download data using export buttons

**Data Source:**
Powered by Yahoo Finance (yfinance)
""")

st.sidebar.divider()
st.sidebar.caption("Works with any stock available on Yahoo Finance")
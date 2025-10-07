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
            st.metric("Dividend Yield", f"{div_yield:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
    
    with col2:
        div_rate = info.get('dividendRate', 'N/A')
        st.metric("Annual Dividend", f"${div_rate:.2f}" if div_rate != 'N/A' else 'N/A')
    
    with col3:
        payout_ratio = info.get('payoutRatio', 'N/A')
        if payout_ratio != 'N/A' and payout_ratio:
            st.metric("Payout Ratio", f"{payout_ratio:.2f}%")
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

def display_insider_trading(stock):
    """Display insider trading information"""
    st.subheader("Insider Trading Activity")
    
    try:
        insider_trades = stock.insider_transactions
        if insider_trades is not None and not insider_trades.empty:
            # Format the dataframe for better readability
            insider_trades_display = insider_trades.copy()
            
            # Sort by date (most recent first)
            if 'Start Date' in insider_trades_display.columns:
                insider_trades_display = insider_trades_display.sort_values('Start Date', ascending=False)
            
            st.dataframe(insider_trades_display.head(20), use_container_width=True)
            
            # Summary statistics
            st.write("**Recent Insider Activity Summary:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Transaction' in insider_trades.columns:
                    buys = len(insider_trades[insider_trades['Transaction'].str.contains('Buy|Purchase', case=False, na=False)])
                    st.metric("Recent Buys", buys)
            
            with col2:
                if 'Transaction' in insider_trades.columns:
                    sells = len(insider_trades[insider_trades['Transaction'].str.contains('Sale|Sell', case=False, na=False)])
                    st.metric("Recent Sells", sells)
        else:
            st.write("No insider trading data available")
    except Exception as e:
        st.write("Unable to fetch insider trading data")

def display_options_data(stock, ticker):
    """Display options data if available"""
    st.subheader("Options Data")
    
    try:
        # Get available expiration dates
        expirations = stock.options
        
        if expirations and len(expirations) > 0:
            # Let user select expiration date
            selected_expiration = st.selectbox(
                "Select Options Expiration Date:",
                expirations[:10],  # Show first 10 expiration dates
                key="options_expiration"
            )
            
            # Get options chain for selected date
            opt_chain = stock.option_chain(selected_expiration)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Call Options**")
                if not opt_chain.calls.empty:
                    calls_display = opt_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head(10)
                    st.dataframe(calls_display, use_container_width=True)
                else:
                    st.write("No call options data")
            
            with col2:
                st.write("**Put Options**")
                if not opt_chain.puts.empty:
                    puts_display = opt_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head(10)
                    st.dataframe(puts_display, use_container_width=True)
                else:
                    st.write("No put options data")
            
            # Options metrics
            st.write("**Options Metrics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_call_volume = opt_chain.calls['volume'].sum() if not opt_chain.calls.empty else 0
                st.metric("Total Call Volume", f"{total_call_volume:,.0f}")
            
            with col2:
                total_put_volume = opt_chain.puts['volume'].sum() if not opt_chain.puts.empty else 0
                st.metric("Total Put Volume", f"{total_put_volume:,.0f}")
            
            with col3:
                if total_call_volume > 0 and total_put_volume > 0:
                    put_call_ratio = total_put_volume / total_call_volume
                    st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
                else:
                    st.metric("Put/Call Ratio", "N/A")
        else:
            st.write("No options data available for this stock")
    except Exception as e:
        st.write("Unable to fetch options data")




def generate_ai_insights(info, stock):
    """Generate AI-powered insights based on financial metrics"""
    st.subheader("AI-Powered Insights")
    
    insights = []
    warnings = []
    positives = []
    
    # Valuation insights
    pe_ratio = info.get('trailingPE', None)
    if pe_ratio and pe_ratio != 'N/A':
        if pe_ratio < 15:
            positives.append(f"✓ Low P/E ratio ({pe_ratio:.1f}) suggests the stock may be undervalued")
        elif pe_ratio > 30:
            warnings.append(f"⚠ High P/E ratio ({pe_ratio:.1f}) indicates premium valuation")
    
    # Profitability insights
    profit_margin = info.get('profitMargins', None)
    if profit_margin and profit_margin != 'N/A':
        if profit_margin > 0.20:
            positives.append(f"✓ Strong profit margin ({profit_margin*100:.1f}%) indicates efficient operations")
        elif profit_margin < 0.05:
            warnings.append(f"⚠ Low profit margin ({profit_margin*100:.1f}%) may indicate competitive pressure")
    
    # Debt insights
    debt_to_equity = info.get('debtToEquity', None)
    if debt_to_equity and debt_to_equity != 'N/A':
        if debt_to_equity > 100:
            warnings.append(f"⚠ High debt-to-equity ratio ({debt_to_equity:.1f}) indicates financial risk")
        elif debt_to_equity < 50:
            positives.append(f"✓ Low debt-to-equity ratio ({debt_to_equity:.1f}) suggests strong balance sheet")
    
    # ROE insights
    roe = info.get('returnOnEquity', None)
    if roe and roe != 'N/A':
        if roe > 0.15:
            positives.append(f"✓ Strong ROE ({roe*100:.1f}%) shows effective use of equity")
        elif roe < 0.05:
            warnings.append(f"⚠ Low ROE ({roe*100:.1f}%) may indicate poor capital efficiency")
    
    # Beta insights
    beta = info.get('beta', None)
    if beta and beta != 'N/A':
        if beta > 1.5:
            warnings.append(f"⚠ High beta ({beta:.2f}) indicates high volatility relative to market")
        elif beta < 0.8:
            positives.append(f"✓ Low beta ({beta:.2f}) suggests lower volatility than market")
    
    # Dividend insights
    div_yield = info.get('dividendYield', None)
    payout_ratio = info.get('payoutRatio', None)
    if div_yield and div_yield != 'N/A' and div_yield > 0:
        if div_yield > 0.03:
            positives.append(f"✓ Attractive dividend yield ({div_yield*100:.1f}%)")
        if payout_ratio and payout_ratio != 'N/A':
            if payout_ratio > 0.8:
                warnings.append(f"⚠ High payout ratio ({payout_ratio*100:.1f}%) may not be sustainable")
    
    # Growth insights
    revenue_growth = info.get('revenueGrowth', None)
    earnings_growth = info.get('earningsGrowth', None)
    if revenue_growth and revenue_growth != 'N/A':
        if revenue_growth > 0.15:
            positives.append(f"✓ Strong revenue growth ({revenue_growth*100:.1f}%)")
        elif revenue_growth < 0:
            warnings.append(f"⚠ Declining revenue ({revenue_growth*100:.1f}%)")
    
    # Current ratio insights
    current_ratio = info.get('currentRatio', None)
    if current_ratio and current_ratio != 'N/A':
        if current_ratio < 1:
            warnings.append(f"⚠ Current ratio below 1 ({current_ratio:.2f}) may indicate liquidity issues")
        elif current_ratio > 2:
            positives.append(f"✓ Strong current ratio ({current_ratio:.2f}) suggests good liquidity")
    
    # Display insights
    if positives:
        st.success("**Positive Signals:**")
        for insight in positives:
            st.write(insight)
    
    if warnings:
        st.warning("**Areas of Concern:**")
        for warning in warnings:
            st.write(warning)
    
    if not positives and not warnings:
        st.info("Insufficient data to generate insights")
    
    # Overall sentiment
    st.write("---")
    sentiment_score = len(positives) - len(warnings)
    
    if sentiment_score > 2:
        st.success("**Overall Sentiment: BULLISH** 📈")
    elif sentiment_score < -2:
        st.error("**Overall Sentiment: BEARISH** 📉")
    else:
        st.info("**Overall Sentiment: NEUTRAL** ➡️")
    
    st.caption("Note: These insights are algorithmic assessments based on financial metrics and should not be considered as investment advice.")


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
                'Forward P/E': info.get('forwardPE', 'N/A'),
                'PEG Ratio': info.get('pegRatio', 'N/A'),
                'Price/Book': info.get('priceToBook', 'N/A'),
                'Price/Sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'Profit Margin': info.get('profitMargins', 'N/A'),
                'Operating Margin': info.get('operatingMargins', 'N/A'),
                'ROE': info.get('returnOnEquity', 'N/A'),
                'ROA': info.get('returnOnAssets', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                'Debt/Equity': info.get('debtToEquity', 'N/A'),
                'Current Ratio': info.get('currentRatio', 'N/A'),
                '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
                'Revenue Growth': info.get('revenueGrowth', 'N/A'),
                'Earnings Growth': info.get('earningsGrowth', 'N/A')
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Overview Section
        st.write("### Overview")
        overview_cols = ['Ticker', 'Name', 'Price', 'Market Cap', '52W High', '52W Low']
        df_overview = df_comparison[overview_cols].copy()
        
        # Format overview
        if 'Market Cap' in df_overview.columns:
            df_overview['Market Cap'] = df_overview['Market Cap'].apply(
                lambda x: f"${x/1e9:.2f}B" if x != 'N/A' else 'N/A'
            )
        if 'Price' in df_overview.columns:
            df_overview['Price'] = df_overview['Price'].apply(
                lambda x: f"${x:.2f}" if x != 'N/A' else 'N/A'
            )
        
        st.dataframe(df_overview, use_container_width=True)
        
        # Valuation Metrics Section
        st.write("### Valuation Metrics")
        valuation_cols = ['Ticker', 'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book', 'Price/Sales']
        df_valuation = df_comparison[valuation_cols].copy()
        
        # Format valuation metrics
        for col in ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book', 'Price/Sales']:
            if col in df_valuation.columns:
                df_valuation[col] = df_valuation[col].apply(
                    lambda x: f"{x:.2f}" if x != 'N/A' and x is not None else 'N/A'
                )
        
        st.dataframe(df_valuation, use_container_width=True)
        
        # Profitability Metrics Section
        st.write("### Profitability Metrics")
        profitability_cols = ['Ticker', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']
        df_profitability = df_comparison[profitability_cols].copy()
        
        # Format profitability metrics
        for col in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
            if col in df_profitability.columns:
                df_profitability[col] = df_profitability[col].apply(
                    lambda x: f"{x*100:.2f}%" if x != 'N/A' and x is not None else 'N/A'
                )
        
        st.dataframe(df_profitability, use_container_width=True)
        
        # Growth Metrics Section
        st.write("### Growth Metrics")
        growth_cols = ['Ticker', 'Revenue Growth', 'Earnings Growth']
        df_growth = df_comparison[growth_cols].copy()
        
        # Format growth metrics
        for col in ['Revenue Growth', 'Earnings Growth']:
            if col in df_growth.columns:
                df_growth[col] = df_growth[col].apply(
                    lambda x: f"{x*100:.2f}%" if x != 'N/A' and x is not None else 'N/A'
                )
        
        st.dataframe(df_growth, use_container_width=True)
        
        # Risk Metrics Section
        st.write("### Risk Metrics")
        risk_cols = ['Ticker', 'Beta', 'Debt/Equity', 'Current Ratio', 'Dividend Yield']
        df_risk = df_comparison[risk_cols].copy()
        
        # Format risk metrics
        if 'Dividend Yield' in df_risk.columns:
            df_risk['Dividend Yield'] = df_risk['Dividend Yield'].apply(
                lambda x: f"{x*100:.2f}%" if x != 'N/A' and x is not None else 'N/A'
            )
        for col in ['Beta', 'Debt/Equity', 'Current Ratio']:
            if col in df_risk.columns:
                df_risk[col] = df_risk[col].apply(
                    lambda x: f"{x:.2f}" if x != 'N/A' and x is not None else 'N/A'
                )
        
        st.dataframe(df_risk, use_container_width=True)
        
        # Visualization Section
        st.write("### Visual Comparisons")
        
        # Create tabs for different comparison charts
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "Price Performance", 
            "Valuation Comparison", 
            "Profitability Comparison",
            "Risk Comparison"
        ])
        
        with chart_tab1:
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
        
        with chart_tab2:
            # Valuation metrics bar chart
            valuation_metrics = ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book']
            
            fig_valuation = go.Figure()
            
            for metric in valuation_metrics:
                values = []
                tickers = []
                for item in comparison_data:
                    if item[metric] != 'N/A' and item[metric] is not None:
                        values.append(item[metric])
                        tickers.append(item['Ticker'])
                
                if values:
                    fig_valuation.add_trace(go.Bar(
                        name=metric,
                        x=tickers,
                        y=values
                    ))
            
            fig_valuation.update_layout(
                title='Valuation Metrics Comparison',
                xaxis_title='Stock',
                yaxis_title='Ratio',
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig_valuation, use_container_width=True)
        
        with chart_tab3:
            # Profitability metrics bar chart
            profitability_metrics = ['Profit Margin', 'Operating Margin', 'ROE', 'ROA']
            
            fig_profitability = go.Figure()
            
            for metric in profitability_metrics:
                values = []
                tickers = []
                for item in comparison_data:
                    if item[metric] != 'N/A' and item[metric] is not None:
                        values.append(item[metric] * 100)
                        tickers.append(item['Ticker'])
                
                if values:
                    fig_profitability.add_trace(go.Bar(
                        name=metric,
                        x=tickers,
                        y=values
                    ))
            
            fig_profitability.update_layout(
                title='Profitability Metrics Comparison',
                xaxis_title='Stock',
                yaxis_title='Percentage (%)',
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig_profitability, use_container_width=True)
        
        with chart_tab4:
            # Risk metrics comparison
            fig_risk = go.Figure()
            
            # Beta comparison
            betas = []
            tickers = []
            for item in comparison_data:
                if item['Beta'] != 'N/A' and item['Beta'] is not None:
                    betas.append(item['Beta'])
                    tickers.append(item['Ticker'])
            
            if betas:
                fig_risk.add_trace(go.Bar(
                    name='Beta',
                    x=tickers,
                    y=betas,
                    marker_color='lightcoral'
                ))
            
            fig_risk.update_layout(
                title='Beta Comparison (Market Risk)',
                xaxis_title='Stock',
                yaxis_title='Beta',
                template='plotly_white'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.info("Beta = 1: Stock moves with the market | Beta > 1: More volatile | Beta < 1: Less volatile")
        
        # Export comparison data
        csv_comparison = df_comparison.to_csv(index=False)
        st.download_button(
            label="Download Comparison Data (CSV)",
            data=csv_comparison,
            file_name="stock_comparison.csv",
            mime="text/csv"
        )
        
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
    
    # Stock Price Chart at the top for quick overview
    st.subheader("Stock Price Overview")
    
    # Time period selector for charts
    time_period = st.selectbox(
        "Select time period:",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max (All Available Data)"],
        index=3,
        key="time_period_selector"
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
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.write("No historical data available for selected period")
    
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
    display_dividend_info(stock, info, hist_selected)
    
    # Earnings Calendar
    display_earnings_calendar(stock, info)
    
    # Additional Charts Section
    st.subheader("Additional Charts")
    
    # Volume Chart
    if not hist_selected.empty:
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
    
    # Financial Statements
    display_financial_statements(stock)
    
    # Insider Trading
    display_insider_trading(stock)
    
    # Institutional Holdings
    display_institutional_holdings(stock)
    
    # Options Data
    display_options_data(stock, ticker)
   
    # AI-Powered Insights
    generate_ai_insights(info, stock)
    
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
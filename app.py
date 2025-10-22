import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re
import requests

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False

st.set_page_config(page_title="Planning Intelligence", page_icon="🏗️", layout="wide")

# CSS
st.markdown("""
    <style>
    .main {background-color: #FFFFFF; padding: 2rem;}
    h1 {font-family: -apple-system, sans-serif; font-weight: 700; color: #1d1d1f; font-size: 3.5rem;}
    h2 {font-family: -apple-system, sans-serif; font-weight: 600; color: #1d1d1f; font-size: 2.5rem; margin-top: 3rem;}
    .hero-gradient {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 3rem; border-radius: 28px; color: white; box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);}
    .insight-card {background: #f5f5f7; padding: 2rem; border-radius: 24px; margin: 2rem 0; border-left: 6px solid #007AFF;}
    .success-card {background: linear-gradient(135deg, #f0fff4 0%, #e0f7e9 100%); border-left: 6px solid #34C759;}
    .urgent-card {background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%); border-left: 6px solid #FF3B30;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_boe_rate():
    """Fetch BoE base rate from official Bank of England website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = 'https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp'
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Parse HTML table with pandas
            tables = pd.read_html(response.text)
            if len(tables) > 0:
                df = tables[0]
                # The first row has the most recent rate
                if len(df) > 0 and 'Rate' in df.columns:
                    latest_rate = float(df.iloc[0]['Rate'])
                    latest_date = str(df.iloc[0]['Date Changed'])
                    
                    return {
                        'rate': latest_rate,
                        'date': latest_date,
                        'source': 'Bank of England (Live)'
                    }
    except Exception as e:
        st.sidebar.caption(f"BoE API note: Using fallback ({str(e)[:30]}...)")
    
    # Fallback to current known rate
    return {
        'rate': 4.00,
        'date': '07 Aug 2025',
        'source': 'Bank of England (Current)'
    }

@st.cache_data(ttl=600)
def load_from_google_sheets():
    if not GOOGLE_SHEETS_AVAILABLE:
        return None
    
    try:
        SPREADSHEET_ID = "1rWZAD_Oy_tuypTYgTFaHClC9keg5cX0VJhUjQtk-LOw"
        
        if "gcp_service_account" not in st.secrets:
            return None
        
        credentials = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        
        service = build('sheets', 'v4', credentials=credentials)
        spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets = [s['properties']['title'] for s in spreadsheet.get('sheets', [])]
        
        for sheet_name in sheets:
            try:
                result = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID, range=sheet_name
                ).execute()
                values = result.get('values', [])
                if values and len(values) > 1:
                    st.sidebar.success(f"✅ Loaded {len(values)-1:,} rows from '{sheet_name}'")
                    return pd.DataFrame(values[1:], columns=values[0])
            except:
                continue
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data
def analyze_data(df):
    """Analyze data with column name flexibility"""
    
    # Show available columns
    st.sidebar.caption("Columns found:")
    for col in list(df.columns)[:5]:
        st.sidebar.caption(f"  • {col}")
    
    # Ensure required columns exist (with defaults if missing)
    required_cols = {
        'LPA': 'Unknown',
        'Status': 'Unknown', 
        'Proposal': '',
        'App_Ref': '',
        'Use_Class_Hint': ''
    }
    
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
    
    # Date columns
    date_cols = ['Valid_Date', 'Decision_Date', 'Run_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = pd.NaT
    
    # Processing days
    df['Processing_Days'] = (df['Decision_Date'] - df['Valid_Date']).dt.days
    
    # Categorize
    df['Status'] = df['Status'].fillna('Unknown')
    df['Status_Category'] = df['Status'].apply(lambda s: 
        'Approved' if any(w in str(s).lower() for w in ['approve', 'grant', 'permit']) else
        'Rejected' if any(w in str(s).lower() for w in ['refuse', 'reject']) else
        'Withdrawn' if 'withdraw' in str(s).lower() else
        'Pending' if 'pending' in str(s).lower() else 'Other')
    
    df['Proposal_Type'] = df['Proposal'].apply(lambda p:
        'Redevelopment' if 'demolition' in str(p).lower() and 'erection' in str(p).lower() else
        'Change of Use' if 'change of use' in str(p).lower() else
        'New Build' if 'erection' in str(p).lower() else 'Other')
    
    df['Is_Residential'] = df['Proposal'].apply(lambda x: 
        any(w in str(x).lower() for w in ['dwelling', 'residential', 'house', 'flat']))
    df['Is_Commercial'] = df['Proposal'].apply(lambda x:
        any(w in str(x).lower() for w in ['commercial', 'retail', 'office']))
    df['Is_Mixed_Use'] = df['Is_Residential'] & df['Is_Commercial']
    
    df['Unit_Count'] = df['Proposal'].apply(lambda p: 
        int(re.search(r'(\d+)\s*(?:dwelling|unit|flat)', str(p).lower()).group(1))
        if re.search(r'(\d+)\s*(?:dwelling|unit|flat)', str(p).lower()) else 0)
    
    # Value score
    df['Value_Score'] = df.apply(lambda r:
        min(100, (40 if r['Status_Category']=='Approved' else 20 if r['Status_Category']=='Pending' else 0) +
        (20 if r['Is_Residential'] else 0) + (10 if r['Is_Mixed_Use'] else 0) +
        (20 if r['Unit_Count']>=10 else 15 if r['Unit_Count']>=5 else 10 if r['Unit_Count']>=1 else 0) +
        (10 if r['Proposal_Type'] in ['New Build', 'Redevelopment'] else 0)), axis=1)
    
    # Time fields
    df['Application_Month'] = df['Valid_Date'].dt.to_period('M')
    df['Month_Name'] = df['Valid_Date'].dt.month_name()
    
    return df

def show_hero(df, base_rate):
    st.markdown("# Planning Intelligence")
    st.markdown("### Real estate investment decision engine")
    st.markdown("")
    
    st.markdown(f"""
    <div class='hero-gradient'>
        <h3 style='margin:0; color:white; font-size:1.3rem;'>🏦 Bank of England Base Rate</h3>
        <h1 style='margin:15px 0; color:white; font-size:5rem; font-weight:800;'>{base_rate['rate']:.2f}%</h1>
        <p style='margin:0; font-size:1.1rem; opacity:0.95;'>
            Development Finance: ~{base_rate['rate']+2.5:.2f}% | Monthly per £100k: £{((base_rate['rate']+2.5)/100*100000/12):,.0f}
        </p>
        <p style='margin-top:10px; font-size:0.95rem; opacity:0.8;'>
            Source: {base_rate['source']} | Updated: {base_rate['date']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(df)
    approved = (df['Status_Category']=='Approved').sum()
    
    with col1:
        st.metric("Applications", f"{total:,}")
    with col2:
        st.metric("Approval Rate", f"{(approved/total*100):.0f}%" if total>0 else "N/A")
    with col3:
        st.metric("High-Value", f"{len(df[df['Value_Score']>=70]):,}")
    with col4:
        median = df['Processing_Days'].median()
        st.metric("Median Days", f"{median:.0f}" if pd.notna(median) else "N/A")
    with col5:
        st.metric("Residential", f"{(df['Is_Residential'].sum()/total*100):.0f}%")

def show_opportunities(df):
    st.markdown("## 🎯 High-Value Opportunities")
    
    high_value = df[(df['Value_Score']>=70) & (df['Status_Category']=='Approved')].sort_values('Value_Score', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759;'>✓ {len(high_value)} High-Value Approved Applications</h3>
        <p><strong>Action:</strong> Contact authorities immediately. Time-sensitive opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Unit_Count', 'Decision_Date', 'Value_Score'] if c in high_value.columns]
        st.dataframe(high_value[display_cols].head(20), use_container_width=True, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prop_counts = df['Proposal_Type'].value_counts().head(6)
        fig = px.bar(x=prop_counts.values, y=prop_counts.index, orientation='h')
        fig.update_traces(marker_color='#007AFF')
        fig.update_layout(title='Proposal Types', height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cat_data = pd.DataFrame({
            'Category': ['Residential', 'Commercial', 'Mixed Use'],
            'Count': [df['Is_Residential'].sum(), df['Is_Commercial'].sum(), df['Is_Mixed_Use'].sum()]
        })
        fig = px.pie(cat_data, values='Count', names='Category', hole=0.6,
                     color_discrete_sequence=['#34C759', '#007AFF', '#FF9500'])
        fig.update_layout(title='Development Categories', height=350)
        st.plotly_chart(fig, use_container_width=True)

def show_market_intelligence(df):
    st.markdown("## 🏛️ Market Intelligence")
    
    lpa_stats = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Approved').sum()
    }).reset_index()
    lpa_stats.columns = ['LPA', 'Total', 'Approved']
    lpa_stats['Approval_Rate'] = (lpa_stats['Approved']/lpa_stats['Total']*100).round(1)
    
    friendly = lpa_stats[(lpa_stats['Total']>=20) & (lpa_stats['Approval_Rate']>=70)].sort_values('Approval_Rate', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759;'>🎯 {len(friendly)} Development-Friendly Authorities</h3>
        <p><strong>Top LPAs:</strong> {', '.join(friendly['LPA'].head(5).tolist()) if len(friendly)>0 else 'None found'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top = lpa_stats.nlargest(10, 'Total')
        fig = px.bar(top, x='Total', y='LPA', orientation='h', color='Approval_Rate',
                     color_continuous_scale='RdYlGn', range_color=[0,100])
        fig.update_layout(title='Most Active LPAs', height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        processing = df[(df['Processing_Days']>0) & (df['Processing_Days']<500)]
        if len(processing) > 0:
            fig = px.histogram(processing, x='Processing_Days', nbins=50)
            fig.add_vline(x=processing['Processing_Days'].median(), line_dash="dash")
            fig.update_traces(marker_color='#007AFF')
            fig.update_layout(title='Processing Time Distribution', height=450)
            st.plotly_chart(fig, use_container_width=True)

def show_executive_summary(df, base_rate):
    st.markdown("## 📊 Executive Summary")
    
    show_hero(df, base_rate)
    
    st.markdown("---")
    
    # Simple analysis without complex groupby
    total = len(df)
    approved = (df['Status_Category']=='Approved').sum()
    high_value = df[(df['Status_Category']=='Approved') & (df['Value_Score']>=70)]
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759;'>🎯 Priority 1: Focus Here</h3>
        <p><strong>{len(high_value)} high-value approved applications</strong> ready for immediate acquisition.</p>
        <p><strong>Overall approval rate:</strong> {(approved/total*100):.1f}%</p>
        <p><strong>Action:</strong> Review detailed opportunities in the Opportunities tab.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Decision_Date', 'Value_Score'] if c in high_value.columns]
        st.dataframe(high_value[display_cols].head(15), use_container_width=True)

def main():
    with st.spinner("Loading planning data..."):
        df_raw = load_from_google_sheets()
    
    if df_raw is None:
        st.error("Unable to load data. Check Google Sheets connection.")
        return
    
    df = analyze_data(df_raw)
    base_rate = get_boe_rate()
    
    if 'Run_Date' in df.columns:
        latest = df['Run_Date'].max()
        if pd.notna(latest):
            st.sidebar.success(f"✅ Updated: {latest.strftime('%Y-%m-%d')}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🎯 Opportunities", "🏛️ Market Intelligence"])
    
    with tab1:
        show_executive_summary(df, base_rate)
    with tab2:
        show_opportunities(df)
    with tab3:
        show_market_intelligence(df)
    
    st.markdown("---")
    st.caption("Planning Intelligence | Real-time data from Google Sheets")

if __name__ == "__main__":
    main()
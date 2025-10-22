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

st.set_page_config(page_title="Planning Intelligence", page_icon="🏗️", layout="wide", initial_sidebar_state="collapsed")

# Steve Jobs aesthetic
st.markdown("""
    <style>
    .main {background-color: #FFFFFF; padding: 2rem 3rem;}
    h1 {font-family: -apple-system, sans-serif; font-weight: 700; color: #1d1d1f; font-size: 3.5rem; letter-spacing: -0.03em;}
    h2 {font-family: -apple-system, sans-serif; font-weight: 600; color: #1d1d1f; font-size: 2.5rem; margin-top: 3rem;}
    h3 {font-family: -apple-system, sans-serif; font-weight: 500; color: #6e6e73; font-size: 1.6rem;}
    .stMetric {background: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%); 
               padding: 2rem; border-radius: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);}
    .hero-gradient {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 3rem; border-radius: 28px; color: white; box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);}
    .insight-card {background: #f5f5f7; padding: 2.5rem; border-radius: 24px; margin: 2rem 0; 
                   border-left: 6px solid #007AFF; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    .success-card {background: linear-gradient(135deg, #f0fff4 0%, #e0f7e9 100%); border-left: 6px solid #34C759;}
    .urgent-card {background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%); border-left: 6px solid #FF3B30;}
    .warning-card {background: linear-gradient(135deg, #fff9f0 0%, #ffedd5 100%); border-left: 6px solid #FF9500;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_boe_rate():
    """Fetch real BoE rate from official website"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = 'https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp'
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            if len(tables) > 0 and len(tables[0]) > 0:
                df = tables[0]
                latest_rate = float(df.iloc[0]['Rate'])
                latest_date = str(df.iloc[0]['Date Changed'])
                return {'rate': latest_rate, 'date': latest_date, 'source': 'Bank of England (Live)'}
    except:
        pass
    return {'rate': 4.00, 'date': '07 Aug 2025', 'source': 'Bank of England'}

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
                    st.sidebar.success(f"✅ Loaded {len(values)-1:,} applications")
                    return pd.DataFrame(values[1:], columns=values[0])
            except:
                continue
        return None
    except Exception as e:
        st.error(f"Google Sheets error: {str(e)}")
        return None

@st.cache_data
def analyze_data(df):
    """Comprehensive data analysis"""
    required_cols = {'LPA': 'Unknown', 'Status': 'Unknown', 'Proposal': '', 
                    'App_Ref': '', 'Use_Class_Hint': ''}
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
    
    # Dates
    for col in ['Valid_Date', 'Decision_Date', 'Run_Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = pd.NaT
    
    df['Processing_Days'] = (df['Decision_Date'] - df['Valid_Date']).dt.days
    df['Status'] = df['Status'].fillna('Unknown')
    df['Status_Category'] = df['Status'].apply(lambda s: 
        'Approved' if any(w in str(s).lower() for w in ['approve', 'grant', 'permit']) else
        'Rejected' if any(w in str(s).lower() for w in ['refuse', 'reject']) else
        'Withdrawn' if 'withdraw' in str(s).lower() else
        'Pending' if 'pending' in str(s).lower() else 'Other')
    
    df['Proposal_Type'] = df['Proposal'].apply(lambda p:
        'Redevelopment' if 'demolition' in str(p).lower() and 'erection' in str(p).lower() else
        'Change of Use' if 'change of use' in str(p).lower() or 'conversion' in str(p).lower() else
        'Extension/Alteration' if 'extension' in str(p).lower() or 'alteration' in str(p).lower() else
        'New Build' if 'erection' in str(p).lower() or 'construction' in str(p).lower() else
        'Subdivision' if 'subdivision' in str(p).lower() else 'Other')
    
    df['Is_Residential'] = df['Proposal'].apply(lambda x: 
        any(w in str(x).lower() for w in ['dwelling', 'residential', 'house', 'flat', 'apartment', 'c3']))
    df['Is_Commercial'] = df['Proposal'].apply(lambda x:
        any(w in str(x).lower() for w in ['commercial', 'retail', 'office', 'shop', 'warehouse', 'industrial']))
    df['Is_Mixed_Use'] = df['Is_Residential'] & df['Is_Commercial']
    
    df['Unit_Count'] = df['Proposal'].apply(lambda p: 
        int(re.search(r'(\d+)\s*(?:dwelling|unit|flat|apartment)', str(p).lower()).group(1))
        if re.search(r'(\d+)\s*(?:dwelling|unit|flat|apartment)', str(p).lower()) else 0)
    
    df['Value_Score'] = df.apply(lambda r:
        min(100, (40 if r['Status_Category']=='Approved' else 20 if r['Status_Category']=='Pending' else 0) +
        (20 if r['Is_Residential'] else 0) + (10 if r['Is_Mixed_Use'] else 0) +
        (20 if r['Unit_Count']>=10 else 15 if r['Unit_Count']>=5 else 10 if r['Unit_Count']>=1 else 0) +
        (10 if r['Proposal_Type'] in ['New Build', 'Redevelopment'] else 0)), axis=1)
    
    df['Application_Month'] = df['Valid_Date'].dt.to_period('M')
    df['Application_Quarter'] = df['Valid_Date'].dt.to_period('Q')
    df['Month_Name'] = df['Valid_Date'].dt.month_name()
    
    return df

def show_hero(df, base_rate):
    st.markdown("# Planning Intelligence")
    st.markdown("### Real estate investment decision engine")
    st.markdown("")
    
    st.markdown(f"""
    <div class='hero-gradient'>
        <h3 style='margin:0; color:white; font-size:1.3rem; font-weight:500;'>🏦 Bank of England Base Rate</h3>
        <h1 style='margin:15px 0; color:white; font-size:5rem; font-weight:800;'>{base_rate['rate']:.2f}%</h1>
        <p style='margin:0; font-size:1.1rem; opacity:0.95;'>
            Development Finance: ~{base_rate['rate']+2.5:.2f}% | Monthly per £100k: £{((base_rate['rate']+2.5)/100*100000/12):,.0f}
        </p>
        <p style='margin-top:10px; font-size:0.95rem; opacity:0.8;'>
            Updated: {base_rate['date']} | Source: {base_rate['source']}
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
        st.metric("Approval Rate", f"{(approved/total*100):.0f}%" if total>0 else "0%")
    with col3:
        st.metric("High-Value", f"{len(df[df['Value_Score']>=70]):,}")
    with col4:
        median = df['Processing_Days'].median()
        st.metric("Median Days", f"{median:.0f}" if pd.notna(median) else "N/A")
    with col5:
        st.metric("Residential", f"{(df['Is_Residential'].sum()/total*100):.0f}%")

def show_opportunity_identification(df):
    """Objective 1: Opportunity Identification"""
    st.markdown("## 🎯 Opportunity Identification")
    st.markdown("### Which applications represent the highest-value development opportunities?")
    
    high_value = df[(df['Value_Score']>=70) & (df['Status_Category']=='Approved')].sort_values('Value_Score', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759; margin-top:0;'>✓ {len(high_value)} High-Value Approved Applications</h3>
        <p style='font-size:1.1rem;'><strong>Immediate Action:</strong> These scored 70+ and are approved. 
        Contact planning authorities and landowners today.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Unit_Count', 'Decision_Date', 'Value_Score'] if c in high_value.columns]
        st.dataframe(high_value[display_cols].head(25), use_container_width=True, height=500)
    
    st.markdown("### What types of proposals are most prevalent?")
    
    col1, col2 = st.columns(2)
    with col1:
        prop_counts = df['Proposal_Type'].value_counts().head(6)
        fig = px.bar(x=prop_counts.values, y=prop_counts.index, orientation='h',
                     labels={'x': 'Count', 'y': 'Type'}, title='Most Prevalent Proposal Types')
        fig.update_traces(marker_color='#007AFF')
        fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cat_data = pd.DataFrame({
            'Category': ['Residential', 'Commercial', 'Mixed Use'],
            'Count': [df['Is_Residential'].sum(), df['Is_Commercial'].sum(), df['Is_Mixed_Use'].sum()]
        })
        fig = px.pie(cat_data, values='Count', names='Category', hole=0.6,
                     title='Development Categories',
                     color_discrete_sequence=['#34C759', '#007AFF', '#FF9500'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### High-Density Opportunities (10+ Units)")
    high_density = df[df['Unit_Count'] >= 10]
    
    if len(high_density) > 0:
        density_stats = high_density.groupby('Status_Category').agg({
            'Unit_Count': ['count', 'sum', 'mean']
        }).round(0)
        density_stats.columns = ['Projects', 'Total Units', 'Avg Units']
        st.dataframe(density_stats, use_container_width=True)
        
        top_approved = high_density[high_density['Status_Category']=='Approved'].nlargest(15, 'Unit_Count')
        if len(top_approved) > 0:
            st.markdown(f"""
            <div class='insight-card success-card'>
                <h3 style='color:#34C759;'>🏢 {len(top_approved)} High-Density Approved</h3>
                <p>Avg {top_approved['Unit_Count'].mean():.0f} units | Total: {top_approved['Unit_Count'].sum():,.0f} units</p>
            </div>
            """, unsafe_allow_html=True)
            display_cols = [c for c in ['LPA', 'Proposal', 'Unit_Count', 'Decision_Date'] if c in top_approved.columns]
            st.dataframe(top_approved[display_cols], use_container_width=True)

def show_market_intelligence(df):
    """Objective 2: Market Intelligence"""
    st.markdown("## 🏛️ Market Intelligence")
    
    lpa_stats = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Approved').sum(),
        'Processing_Days': 'median'
    }).reset_index()
    lpa_stats.columns = ['LPA', 'Total', 'Approved', 'Median_Days']
    lpa_stats['Approval_Rate'] = (lpa_stats['Approved']/lpa_stats['Total']*100).round(1)
    
    friendly = lpa_stats[(lpa_stats['Total']>=20) & (lpa_stats['Approval_Rate']>=70)].sort_values('Approval_Rate', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759; margin-top:0;'>🎯 {len(friendly)} Development-Friendly Authorities</h3>
        <p style='font-size:1.1rem;'><strong>Top LPAs:</strong> {', '.join(friendly['LPA'].head(5).tolist()) if len(friendly)>0 else 'None with 20+ apps at 70%+'}</p>
        <p><strong>Action:</strong> Focus acquisition in these areas. Set up weekly monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Active Planning Authorities")
        top = lpa_stats.nlargest(12, 'Total')
        fig = px.bar(top, x='Total', y='LPA', orientation='h', color='Approval_Rate',
                     color_continuous_scale='RdYlGn', range_color=[0,100])
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Fastest Processing Times")
        fastest = lpa_stats[lpa_stats['Total']>=10].nsmallest(12, 'Median_Days')
        fig = px.bar(fastest, x='Median_Days', y='LPA', orientation='h')
        fig.update_traces(marker_color='#34C759')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Approval Rates by Proposal Type")
    type_perf = df.groupby('Proposal_Type').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Approved').sum()
    })
    type_perf['Approval_Rate'] = (type_perf['Status_Category']/type_perf['App_Ref']*100).round(1)
    type_perf = type_perf[type_perf['App_Ref']>=10].sort_values('Approval_Rate', ascending=False)
    
    fig = px.bar(type_perf, x=type_perf.index, y='Approval_Rate',
                 labels={'Approval_Rate': 'Approval Rate (%)', 'index': 'Type'})
    fig.update_traces(marker_color=['#34C759' if x>70 else '#FF9500' if x>50 else '#FF3B30' 
                                    for x in type_perf['Approval_Rate']])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Processing Time Analysis")
    processing = df[(df['Processing_Days']>0) & (df['Processing_Days']<500)]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if len(processing) > 0:
            fig = px.histogram(processing, x='Processing_Days', nbins=50)
            fig.add_vline(x=processing['Processing_Days'].median(), 
                         line_dash="dash", line_color="red",
                         annotation_text=f"Median: {processing['Processing_Days'].median():.0f}d")
            fig.update_traces(marker_color='#007AFF')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(processing) > 0:
            st.metric("Mean", f"{processing['Processing_Days'].mean():.0f} days")
            st.metric("Median", f"{processing['Processing_Days'].median():.0f} days")
            st.metric("90th %ile", f"{processing['Processing_Days'].quantile(0.9):.0f} days")
    
    st.markdown("### Seasonal Patterns")
    if df['Month_Name'].notna().any():
        monthly = df['Month_Name'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).fillna(0)
        
        fig = px.bar(x=monthly.index, y=monthly.values,
                     labels={'x': 'Month', 'y': 'Applications'})
        fig.update_traces(marker_color='#FF9500')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        best = monthly.nlargest(3)
        st.markdown(f"""
        <div class='insight-card'>
            <strong>📅 Best Submission Months:</strong> {', '.join(best.index.tolist())}<br>
            Historical data shows higher volumes in these months.
        </div>
        """, unsafe_allow_html=True)

def show_pipeline_management(df):
    """Objective 3: Pipeline Management"""
    st.markdown("## 📋 Deal Pipeline Management")
    
    status_counts = df['Status_Category'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Approved", f"{status_counts.get('Approved', 0):,}", delta="Ready")
    with col2:
        st.metric("⏳ Pending", f"{status_counts.get('Pending', 0):,}", delta="Monitor")
    with col3:
        st.metric("❌ Rejected", f"{status_counts.get('Rejected', 0):,}")
    with col4:
        st.metric("🚫 Withdrawn", f"{status_counts.get('Withdrawn', 0):,}")
    
    recent_approved = df[
        (df['Status_Category']=='Approved') &
        (df['Decision_Date'] > datetime.now() - timedelta(days=180))
    ].sort_values('Decision_Date', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card urgent-card'>
        <h3 style='color:#FF3B30; margin-top:0;'>⚡ {len(recent_approved)} Recently Approved (Last 6 Months)</h3>
        <p style='font-size:1.1rem;'><strong>Urgent:</strong> These were approved in the last 180 days. 
        Contact authorities NOW before these opportunities disappear.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(recent_approved) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Decision_Date', 'Value_Score'] if c in recent_approved.columns]
        st.dataframe(recent_approved[display_cols].head(30), use_container_width=True, height=500)
    
    pending_monitor = df[
        (df['Status_Category']=='Pending') &
        (df['Value_Score']>=60)
    ].sort_values('Value_Score', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card warning-card'>
        <h3 style='margin-top:0;'>👀 {len(pending_monitor)} High-Value Pending</h3>
        <p><strong>Strategy:</strong> Set up decision alerts. Pre-negotiate with landowners. Begin due diligence now.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(pending_monitor) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Valid_Date', 'Value_Score'] if c in pending_monitor.columns]
        st.dataframe(pending_monitor[display_cols].head(30), use_container_width=True, height=500)
    
    st.markdown("### Application Pipeline Funnel")
    funnel = df['Status_Category'].value_counts().reset_index()
    funnel.columns = ['Stage', 'Count']
    fig = px.funnel(funnel, x='Count', y='Stage')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment(df):
    """Objective 4: Risk Assessment"""
    st.markdown("## ⚠️ Risk Assessment")
    
    total = len(df)
    rejected = (df['Status_Category']=='Rejected').sum()
    rejection_rate = (rejected/total*100) if total>0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
    with col2:
        withdrawn = (df['Status_Category']=='Withdrawn').sum()
        st.metric("Withdrawal Rate", f"{(withdrawn/total*100):.1f}%")
    with col3:
        avg_reject = df[df['Status_Category']=='Rejected']['Processing_Days'].median()
        st.metric("Median to Rejection", f"{avg_reject:.0f}d" if pd.notna(avg_reject) else "N/A")
    
    lpa_risk = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Rejected').sum()
    })
    lpa_risk['Rejection_Rate'] = (lpa_risk['Status_Category']/lpa_risk['App_Ref']*100).round(1)
    high_risk = lpa_risk[(lpa_risk['App_Ref']>=10) & (lpa_risk['Rejection_Rate']>30)].sort_values('Rejection_Rate', ascending=False)
    
    if len(high_risk) > 0:
        st.markdown(f"""
        <div class='insight-card urgent-card'>
            <h3 style='color:#FF3B30; margin-top:0;'>🚨 {len(high_risk)} High-Risk Authorities</h3>
            <p style='font-size:1.1rem;'><strong>Avoid:</strong> {', '.join(high_risk.head(5).index.tolist())}</p>
            <p><strong>Action:</strong> If pursuing here, budget 50%+ extra for planning. Expect delays.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.bar(high_risk.head(15), x='Rejection_Rate', y=high_risk.head(15).index,
                     orientation='h', color='Rejection_Rate', color_continuous_scale='Reds')
        fig.update_layout(height=550, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Rejection by Use Class")
    use_risk = df.groupby('Use_Class_Hint').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Rejected').sum()
    })
    use_risk['Rejection_Rate'] = (use_risk['Status_Category']/use_risk['App_Ref']*100).round(1)
    risky = use_risk[(use_risk['App_Ref']>=5) & (use_risk['Rejection_Rate']>30)].sort_values('Rejection_Rate', ascending=False)
    
    if len(risky) > 0:
        fig = px.bar(risky.head(10), x='Rejection_Rate', y=risky.head(10).index,
                     orientation='h', color='Rejection_Rate', color_continuous_scale='Oranges')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_strategic_insights(df):
    """Objective 5: Strategic Insights"""
    st.markdown("## 🎯 Strategic Insights")
    
    recent = df[df['Valid_Date'] > datetime.now() - timedelta(days=365)]
    hotspots = recent.groupby('LPA').agg({
        'App_Ref': 'count',
        'Unit_Count': 'sum',
        'Value_Score': 'mean'
    }).sort_values('App_Ref', ascending=False).head(15)
    hotspots.columns = ['Applications', 'Total_Units', 'Avg_Value']
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759; margin-top:0;'>🔥 Top 5 Emerging Hotspots (Last 12mo)</h3>
        <p style='font-size:1.1rem;'><strong>{', '.join(hotspots.head(5).index.tolist())}</strong></p>
        <p>High volumes = strong momentum. Focus land acquisition here.</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(hotspots.reset_index(), x='Applications', y='Total_Units', 
                     size='Avg_Value', hover_data=['LPA'],
                     labels={'Applications': 'Apps (12mo)', 'Total_Units': 'Units Proposed'})
    fig.update_traces(marker=dict(color='#007AFF', line=dict(width=2, color='white')))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Trending Property Types")
    if df['Application_Quarter'].notna().any():
        quarterly = df.groupby(['Application_Quarter', 'Proposal_Type']).size().reset_index(name='Count')
        quarterly['Quarter'] = quarterly['Application_Quarter'].astype(str)
        top_types = df['Proposal_Type'].value_counts().head(5).index
        quarterly_filtered = quarterly[quarterly['Proposal_Type'].isin(top_types)]
        
        fig = px.line(quarterly_filtered, x='Quarter', y='Count', 
                     color='Proposal_Type', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Underserved Markets (Low Competition)")
    use_competition = df.groupby(['LPA', 'Use_Class_Hint']).size().reset_index(name='Count')
    underserved = use_competition[use_competition['Count']<=3].sort_values('Count')
    
    if len(underserved) > 0:
        st.markdown(f"""
        <div class='insight-card warning-card'>
            <h3>💡 {len(underserved)} Underserved Opportunities</h3>
            <p>Low competition in specific LPA + Use Class combinations. First-mover advantage.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(underserved.head(30), use_container_width=True, height=400)
    
    if 'Matched_Parcel_ID' in df.columns and df['Matched_Parcel_ID'].notna().any():
        parcel_counts = df['Matched_Parcel_ID'].value_counts()
        complex_parcels = parcel_counts[parcel_counts > 1]
        
        if len(complex_parcels) > 0:
            st.markdown(f"""
            <div class='insight-card'>
                <h3>🏗️ {len(complex_parcels)} Complex Parcels</h3>
                <p>Multiple applications indicate high-value/complex sites or challenging history.</p>
            </div>
            """, unsafe_allow_html=True)
            
            complex_data = df[df['Matched_Parcel_ID'].isin(complex_parcels.index)].groupby('Matched_Parcel_ID').agg({
                'App_Ref': 'count',
                'LPA': 'first',
                'Status_Category': lambda x: ', '.join(x.unique()),
                'Proposal': 'first'
            }).reset_index()
            complex_data.columns = ['Parcel_ID', 'App_Count', 'LPA', 'Statuses', 'Sample_Proposal']
            st.dataframe(complex_data.head(20), use_container_width=True)

def show_executive_summary(df, base_rate):
    """Executive Summary with Actionable Recommendations"""
    st.markdown("## 📊 Executive Summary")
    st.markdown("### Actionable Investment Recommendations")
    
    show_hero(df, base_rate)
    
    st.markdown("---")
    
    # Analysis for recommendations
    lpa_stats = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Approved').sum(),
        'Processing_Days': 'median'
    })
    lpa_stats['Approval_Rate'] = (lpa_stats['Status_Category']/lpa_stats['App_Ref']*100)
    
    best_lpas = lpa_stats[(lpa_stats['App_Ref']>=20) & (lpa_stats['Approval_Rate']>=70)].sort_values('Approval_Rate', ascending=False).head(5)
    high_risk_lpas = lpa_stats[(lpa_stats['App_Ref']>=10) & (lpa_stats['Approval_Rate']<50)].sort_values('Approval_Rate').head(5)
    
    recent_approved = df[(df['Status_Category']=='Approved') & (df['Decision_Date'] > datetime.now() - timedelta(days=90))]
    high_value = df[(df['Status_Category']=='Approved') & (df['Value_Score']>=70)]
    
    # Recommendation 1: Where to focus
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color:#34C759;'>🎯 Priority 1: Focus Acquisition Here</h3>
        <p style='font-size:1.2rem;'><strong>{', '.join(best_lpas.head(3).index.tolist()) if len(best_lpas)>0 else 'Analyze more data needed'}</strong></p>
        <p><strong>Why:</strong> ≥70% approval rates, ≥20 applications, fast processing.</p>
        <p><strong>Action:</strong> Weekly automated searches. Contact local agents proactively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation 2: Immediate opportunities
    st.markdown(f"""
    <div class='insight-card urgent-card'>
        <h3 style='color:#FF3B30;'>⚡ Priority 2: {len(recent_approved)} Time-Sensitive Opportunities</h3>
        <p><strong>Recently Approved (90d):</strong> {len(recent_approved):,} | <strong>High-Value (70+):</strong> {len(high_value):,}</p>
        <p><strong>Action:</strong> Contact planning authorities tomorrow. Move fast.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value) > 0:
        display_cols = [c for c in ['App_Ref', 'LPA', 'Proposal', 'Unit_Count', 'Decision_Date', 'Value_Score'] if c in high_value.columns]
        st.dataframe(high_value[display_cols].head(20), use_container_width=True, height=400)
    
    # Recommendation 3: Best types
    type_perf = df.groupby('Proposal_Type').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x=='Approved').sum()
    })
    type_perf['Approval_Rate'] = (type_perf['Status_Category']/type_perf['App_Ref']*100)
    best_types = type_perf[type_perf['App_Ref']>=10].sort_values('Approval_Rate', ascending=False).head(3)
    
    if len(best_types) > 0:
        st.markdown(f"""
        <div class='insight-card'>
            <h3>🏗️ Priority 3: Best Development Types</h3>
            <p><strong>Highest Success:</strong></p>
            <p>1. {best_types.index[0]}: {best_types['Approval_Rate'].values[0]:.0f}% approval</p>
            <p>2. {best_types.index[1]}: {best_types['Approval_Rate'].values[1]:.0f}% approval</p>
            <p>3. {best_types.index[2]}: {best_types['Approval_Rate'].values[2]:.0f}% approval</p>
            <p><strong>Action:</strong> Prioritize {best_types.index[0].lower()} for best returns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation 4: Areas to avoid
    if len(high_risk_lpas) > 0:
        st.markdown(f"""
        <div class='insight-card urgent-card'>
            <h3 style='color:#FF3B30;'>⚠️ Priority 4: Avoid These Areas</h3>
            <p><strong>High-Risk LPAs:</strong> {', '.join(high_risk_lpas.index.tolist())}</p>
            <p><strong>Risk:</strong> <50% approval. Extended timelines. Unpredictable.</p>
            <p><strong>Action:</strong> Avoid unless exceptional. Budget 2x planning costs if pursuing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market context
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Financing Context")
        st.markdown(f"""
        <div class='hero-gradient' style='padding: 2rem;'>
            <h3 style='color: white; margin: 0;'>BoE Base: {base_rate['rate']:.2f}%</h3>
            <p style='color: white; margin-top: 10px; opacity: 0.9;'>
            <strong>Dev Finance: ~{base_rate['rate']+2.5:.2f}%</strong><br>
            Monthly per £1M: £{((base_rate['rate']+2.5)/100*1000000/12):,.0f}
            </p>
            <p style='color: white; margin-top: 10px; opacity: 0.9;'>
            <strong>Strategy:</strong> Focus on 12-18mo projects to minimize carry costs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Timeline Planning")
        median = df['Processing_Days'].median()
        p90 = df['Processing_Days'].quantile(0.9)
        
        st.markdown(f"""
        <div class='insight-card'>
            <h3>⏱️ Planning Benchmarks</h3>
            <p><strong>Median:</strong> {median:.0f} days</p>
            <p><strong>90th %ile:</strong> {p90:.0f} days</p>
            <p><strong>Pro Forma:</strong> Use {p90:.0f}d for conservative planning</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application"""
    with st.spinner("Loading planning data..."):
        df_raw = load_from_google_sheets()
    
    if df_raw is None:
        st.error("Unable to load data from Google Sheets.")
        st.info("Ensure spreadsheet is shared with: planning@realestate-466709.iam.gserviceaccount.com")
        return
    
    df = analyze_data(df_raw)
    base_rate = get_boe_rate()
    
    if 'Run_Date' in df.columns:
        latest = df['Run_Date'].max()
        if pd.notna(latest):
            st.sidebar.success(f"✅ Updated: {latest.strftime('%Y-%m-%d')}")
    
    st.sidebar.markdown(f"### 🏦 BoE Rate: {base_rate['rate']:.2f}%")
    st.sidebar.caption(f"Updated: {base_rate['date']}")
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary",
        "🎯 Opportunities", 
        "🏛️ Market Intelligence",
        "📋 Pipeline",
        "⚠️ Risk Assessment",
        "🎯 Strategic Insights"
    ])
    
    with tab1:
        show_executive_summary(df, base_rate)
    
    with tab2:
        show_opportunity_identification(df)
    
    with tab3:
        show_market_intelligence(df)
    
    with tab4:
        show_pipeline_management(df)
    
    with tab5:
        show_risk_assessment(df)
    
    with tab6:
        show_strategic_insights(df)
    
    st.markdown("---")
    st.caption(f"Planning Intelligence | {len(df):,} applications analyzed | Powered by real-time data")

if __name__ == "__main__":
    main()
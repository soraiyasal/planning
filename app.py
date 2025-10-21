import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Planning Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Steve Jobs aesthetic - clean, bold, focused
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');
    
    .main {background-color: #FFFFFF; padding: 2rem 3rem;}
    h1 {font-family: 'SF Pro Display', -apple-system, sans-serif; 
        font-weight: 700; color: #1d1d1f; font-size: 4rem; 
        letter-spacing: -0.03em; margin-bottom: 0.5rem;}
    h2 {font-family: 'SF Pro Display', -apple-system, sans-serif; 
        font-weight: 600; color: #1d1d1f; font-size: 2.8rem; margin-top: 4rem;}
    h3 {font-family: 'SF Pro Display', -apple-system, sans-serif; 
        font-weight: 500; color: #6e6e73; font-size: 1.6rem; margin-top: 2rem;}
    .stMetric {background: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%); 
               padding: 2.5rem; border-radius: 24px; 
               box-shadow: 0 4px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0;}
    .metric-label {font-size: 0.9rem; color: #6e6e73; font-weight: 500;}
    .metric-value {font-size: 2.5rem; font-weight: 700; color: #1d1d1f;}
    
    .insight-card {
        background: #f5f5f7; padding: 2.5rem; border-radius: 24px; 
        margin: 2rem 0; border-left: 6px solid #007AFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .urgent-card {background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%); 
                  border-left: 6px solid #FF3B30;}
    .success-card {background: linear-gradient(135deg, #f0fff4 0%, #e0f7e9 100%); 
                   border-left: 6px solid #34C759;}
    .warning-card {background: linear-gradient(135deg, #fff9f0 0%, #ffedd5 100%); 
                   border-left: 6px solid #FF9500;}
    
    .hero-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem; border-radius: 28px; color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    div[data-testid="stMetricValue"] {font-size: 2rem !important;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_from_google_sheets():
    """Auto-load from Google Sheets using secrets"""
    if not GOOGLE_SHEETS_AVAILABLE:
        st.error("Google Sheets unavailable - install google-api-python-client")
        return None
    
    try:
        # HARDCODED VALUES - easier than fighting TOML
        SPREADSHEET_ID = "1rWZAD_Oy_tuypTYgTFaHClC9keg5cX0VJhUjQtk-LOw"
        SHEET_NAME = "Planning_Applications"
        
        # Check for service account
        if "gcp_service_account" not in st.secrets:
            st.error("Missing 'gcp_service_account' in secrets")
            return None
        
        st.sidebar.info(f"📊 Loading from Google Sheets...")
        st.sidebar.caption(f"Sheet: {SHEET_NAME}")
        
        credentials_dict = dict(st.secrets["gcp_service_account"])
        
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        
        service = build('sheets', 'v4', credentials=credentials)
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, 
            range=SHEET_NAME
        ).execute()
        
        values = result.get('values', [])
        if not values:
            st.error(f"No data found in sheet '{SHEET_NAME}'")
            return None
        
        st.sidebar.success(f"✅ Loaded {len(values)-1:,} rows")
        return pd.DataFrame(values[1:], columns=values[0])
        
    except Exception as e:
        st.error(f"Error loading Google Sheets: {str(e)}")
        st.exception(e)
        return None

@st.cache_data
def analyze_data(df):
    """Comprehensive data analysis"""
    df['Valid_Date'] = pd.to_datetime(df['Valid_Date'], errors='coerce')
    df['Decision_Date'] = pd.to_datetime(df['Decision_Date'], errors='coerce')
    df['Run_Date'] = pd.to_datetime(df['Run_Date'], errors='coerce')
    df['Processing_Days'] = (df['Decision_Date'] - df['Valid_Date']).dt.days
    
    df['Status'] = df['Status'].fillna('Unknown')
    df['Status_Category'] = df['Status'].apply(categorize_status)
    df['Proposal_Type'] = df['Proposal'].apply(extract_proposal_type)
    df['Is_Residential'] = df['Proposal'].apply(lambda x: 'dwelling' in str(x).lower() or 'residential' in str(x).lower() or 'house' in str(x).lower() or 'flat' in str(x).lower())
    df['Is_Commercial'] = df['Proposal'].apply(lambda x: 'commercial' in str(x).lower() or 'retail' in str(x).lower() or 'office' in str(x).lower())
    df['Is_Mixed_Use'] = df['Is_Residential'] & df['Is_Commercial']
    df['Unit_Count'] = df['Proposal'].apply(extract_units)
    df['Value_Score'] = df.apply(calc_value_score, axis=1)
    
    df['Application_Month'] = df['Valid_Date'].dt.to_period('M')
    df['Application_Quarter'] = df['Valid_Date'].dt.to_period('Q')
    df['Month_Name'] = df['Valid_Date'].dt.month_name()
    
    return df

def categorize_status(s):
    if pd.isna(s): return 'Unknown'
    s = str(s).lower()
    if any(w in s for w in ['approve', 'grant', 'permit']): return 'Approved'
    elif any(w in s for w in ['refuse', 'reject', 'dismiss']): return 'Rejected'
    elif any(w in s for w in ['withdraw', 'withdrawn']): return 'Withdrawn'
    elif any(w in s for w in ['pending', 'submitted', 'valid']): return 'Pending'
    return 'Other'

def extract_proposal_type(p):
    if pd.isna(p): return 'Unknown'
    p = str(p).lower()
    if 'demolition' in p and 'erection' in p: return 'Redevelopment'
    elif 'change of use' in p or 'conversion' in p: return 'Change of Use'
    elif 'extension' in p or 'alteration' in p: return 'Extension/Alteration'
    elif 'erection' in p or 'construction' in p: return 'New Build'
    return 'Other'

def extract_units(p):
    if pd.isna(p): return 0
    for pattern in [r'(\d+)\s*(?:dwelling|unit|flat|apartment)', r'(\d+)\s*bed']:
        match = re.search(pattern, str(p).lower())
        if match: return int(match.group(1))
    return 0

def calc_value_score(row):
    score = 0
    if row['Status_Category'] == 'Approved': score += 40
    elif row['Status_Category'] == 'Pending': score += 20
    if row['Is_Residential']: score += 20
    if row['Is_Mixed_Use']: score += 10
    units = row['Unit_Count']
    if units >= 10: score += 20
    elif units >= 5: score += 15
    elif units >= 1: score += 10
    if row['Proposal_Type'] in ['New Build', 'Redevelopment']: score += 10
    return min(score, 100)

def show_hero(df):
    """Hero section"""
    st.markdown("# Planning Intelligence")
    st.markdown("### Real estate investment decision engine")
    st.markdown("")
    
    # BoE Rate - corrected to 4%
    st.markdown(f"""
    <div class='hero-gradient'>
        <h3 style='margin: 0; color: white; font-size: 1.3rem; font-weight: 500;'>🏦 Bank of England Base Rate</h3>
        <h1 style='margin: 15px 0; color: white; font-size: 5rem; font-weight: 800;'>4.00%</h1>
        <p style='margin: 0; font-size: 1.1rem; opacity: 0.95;'>
            Development Finance: ~6.5% | Monthly cost per £100k: £542
        </p>
        <p style='margin-top: 10px; font-size: 0.95rem; opacity: 0.8;'>
            Higher rates favor quick-turnaround projects. Focus on approved applications with 12-18 month completion timelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(df)
    approved = (df['Status_Category'] == 'Approved').sum()
    approval_rate = (approved / total * 100) if total > 0 else 0
    
    with col1:
        st.metric("Applications", f"{total:,}")
    with col2:
        st.metric("Approval Rate", f"{approval_rate:.0f}%")
    with col3:
        st.metric("High-Value Opps", f"{len(df[df['Value_Score'] >= 70]):,}")
    with col4:
        median = df['Processing_Days'].median()
        st.metric("Median Processing", f"{median:.0f}d" if pd.notna(median) else "N/A")
    with col5:
        st.metric("Residential", f"{(df['Is_Residential'].sum()/total*100):.0f}%")

def show_opportunity_identification(df):
    """Objective 1: Opportunity Identification"""
    st.markdown("## 🎯 Opportunity Identification")
    st.markdown("### Which applications represent the highest-value opportunities?")
    
    high_value = df[(df['Value_Score'] >= 70) & (df['Status_Category'] == 'Approved')].sort_values('Value_Score', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='margin-top: 0; color: #34C759; font-size: 1.8rem;'>✓ {len(high_value)} High-Value Approved Applications Ready Now</h3>
        <p style='font-size: 1.1rem;'><strong>Immediate Action:</strong> These applications scored 70+ and are approved. 
        Contact planning authorities and landowners immediately. Time-sensitive opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value) > 0:
        st.dataframe(high_value[['App_Ref', 'LPA', 'Proposal', 'Unit_Count', 'Value_Score', 'Decision_Date']].head(20), 
                     use_container_width=True, height=400)
    
    # Proposal type prevalence
    st.markdown("### What types of proposals are most prevalent?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prop_counts = df['Proposal_Type'].value_counts()
        fig = px.bar(x=prop_counts.values, y=prop_counts.index, orientation='h',
                     labels={'x': 'Count', 'y': 'Type'}, title='Proposal Types')
        fig.update_traces(marker_color='#007AFF')
        fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class='insight-card'>
            <strong>Top 3 Types:</strong><br>
            1. {prop_counts.index[0]}: {prop_counts.values[0]:,} ({prop_counts.values[0]/prop_counts.sum()*100:.0f}%)<br>
            2. {prop_counts.index[1]}: {prop_counts.values[1]:,} ({prop_counts.values[1]/prop_counts.sum()*100:.0f}%)<br>
            3. {prop_counts.index[2]}: {prop_counts.values[2]:,} ({prop_counts.values[2]/prop_counts.sum()*100:.0f}%)
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # High-density opportunities
    st.markdown("### High-Density Opportunities Worth Pursuing (10+ Units)")
    
    high_density = df[df['Unit_Count'] >= 10]
    
    if len(high_density) > 0:
        density_stats = high_density.groupby('Status_Category').agg({
            'Unit_Count': ['count', 'sum', 'mean']
        }).round(0)
        density_stats.columns = ['Projects', 'Total Units', 'Avg Units/Project']
        st.dataframe(density_stats, use_container_width=True)
        
        top_density_approved = high_density[high_density['Status_Category'] == 'Approved'].nlargest(10, 'Unit_Count')
        
        if len(top_density_approved) > 0:
            st.markdown(f"""
            <div class='insight-card success-card'>
                <h3 style='color: #34C759;'>🏢 {len(top_density_approved)} High-Density Approved Projects</h3>
                <p>Average {top_density_approved['Unit_Count'].mean():.0f} units per project. 
                Total: {top_density_approved['Unit_Count'].sum():,.0f} units ready for development.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(top_density_approved[['LPA', 'Proposal', 'Unit_Count', 'Decision_Date', 'Value_Score']], 
                        use_container_width=True)

def show_market_intelligence(df):
    """Objective 2: Market Intelligence"""
    st.markdown("## 🏛️ Market Intelligence")
    
    # LPA Performance
    lpa_stats = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Approved').sum(),
        'Processing_Days': 'median'
    }).reset_index()
    lpa_stats.columns = ['LPA', 'Total', 'Approved', 'Median_Days']
    lpa_stats['Approval_Rate'] = (lpa_stats['Approved'] / lpa_stats['Total'] * 100).round(1)
    
    # Development-friendly LPAs
    friendly = lpa_stats[(lpa_stats['Total'] >= 20) & (lpa_stats['Approval_Rate'] >= 70)].sort_values('Approval_Rate', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='margin-top: 0; color: #34C759;'>🎯 {len(friendly)} Development-Friendly Authorities Identified</h3>
        <p style='font-size: 1.1rem;'><strong>Top 5 LPAs:</strong> {', '.join(friendly['LPA'].head(5).tolist())}</p>
        <p><strong>Recommendation:</strong> These authorities have ≥70% approval rates with 20+ applications. 
        Focus acquisition searches in these areas. Set up weekly monitoring of their planning portals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Active Planning Authorities")
        top_active = lpa_stats.nlargest(10, 'Total')
        fig = px.bar(top_active, x='Total', y='LPA', orientation='h',
                     color='Approval_Rate', color_continuous_scale='RdYlGn',
                     range_color=[0, 100])
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Fastest Processing Times")
        fastest = lpa_stats[lpa_stats['Total'] >= 10].nsmallest(10, 'Median_Days')
        fig = px.bar(fastest, x='Median_Days', y='LPA', orientation='h')
        fig.update_traces(marker_color='#34C759')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Approval rates by proposal type
    st.markdown("### Approval Rates by Proposal Type")
    
    type_perf = df.groupby('Proposal_Type').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Approved').sum()
    })
    type_perf['Approval_Rate'] = (type_perf['Status_Category'] / type_perf['App_Ref'] * 100).round(1)
    type_perf = type_perf[type_perf['App_Ref'] >= 10].sort_values('Approval_Rate', ascending=False)
    
    fig = px.bar(type_perf, x=type_perf.index, y='Approval_Rate',
                 labels={'Approval_Rate': 'Approval Rate (%)', 'index': 'Proposal Type'})
    fig.update_traces(marker_color=['#34C759' if x > 70 else '#FF9500' if x > 50 else '#FF3B30' 
                                    for x in type_perf['Approval_Rate']])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Processing time
    st.markdown("### Processing Time Analysis")
    
    processing = df[(df['Processing_Days'] > 0) & (df['Processing_Days'] < 500)]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(processing, x='Processing_Days', nbins=50)
        fig.add_vline(x=processing['Processing_Days'].median(), 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Median: {processing['Processing_Days'].median():.0f} days")
        fig.update_traces(marker_color='#007AFF')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Mean", f"{processing['Processing_Days'].mean():.0f} days")
        st.metric("Median", f"{processing['Processing_Days'].median():.0f} days")
        st.metric("90th %ile", f"{processing['Processing_Days'].quantile(0.9):.0f} days")
        
        st.markdown(f"""
        <div class='insight-card warning-card' style='margin-top: 1rem;'>
            <strong>Planning:</strong> Budget {processing['Processing_Days'].quantile(0.9):.0f} days 
            for planning approval to cover 90% of cases.
        </div>
        """, unsafe_allow_html=True)
    
    # Seasonal patterns
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
        
        best_months = monthly.nlargest(3)
        st.markdown(f"""
        <div class='insight-card'>
            <strong>📅 Optimal Submission Months:</strong> {', '.join(best_months.index.tolist())}<br>
            Historical data shows highest application volumes in these months. 
            Consider timing major applications accordingly.
        </div>
        """, unsafe_allow_html=True)

def show_pipeline_management(df):
    """Objective 3: Deal Pipeline Management"""
    st.markdown("## 📋 Deal Pipeline Management")
    
    # Stage distribution
    status_counts = df['Status_Category'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Approved", f"{status_counts.get('Approved', 0):,}", 
                 delta="Ready for action")
    with col2:
        st.metric("⏳ Pending", f"{status_counts.get('Pending', 0):,}",
                 delta="Monitor closely")
    with col3:
        st.metric("❌ Rejected", f"{status_counts.get('Rejected', 0):,}")
    with col4:
        st.metric("🚫 Withdrawn", f"{status_counts.get('Withdrawn', 0):,}")
    
    # Recently approved
    recent_approved = df[
        (df['Status_Category'] == 'Approved') &
        (df['Decision_Date'] > datetime.now() - timedelta(days=180))
    ].sort_values('Decision_Date', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card urgent-card'>
        <h3 style='margin-top: 0; color: #FF3B30;'>⚡ {len(recent_approved)} Recently Approved (Last 6 Months)</h3>
        <p style='font-size: 1.1rem;'><strong>Urgent Action Required:</strong> These applications were approved in the last 180 days. 
        Contact authorities and landowners immediately before these opportunities are gone.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(recent_approved) > 0:
        st.dataframe(
            recent_approved[['App_Ref', 'LPA', 'Proposal', 'Decision_Date', 'Value_Score']].head(25),
            use_container_width=True, height=500
        )
    
    # High-value pending
    pending_monitor = df[
        (df['Status_Category'] == 'Pending') &
        (df['Value_Score'] >= 60)
    ].sort_values('Value_Score', ascending=False)
    
    st.markdown(f"""
    <div class='insight-card warning-card'>
        <h3 style='margin-top: 0;'>👀 {len(pending_monitor)} High-Value Pending Applications</h3>
        <p><strong>Monitor Strategy:</strong> Set up decision notification alerts. Pre-negotiate with landowners. 
        Begin due diligence now to move quickly upon approval.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(pending_monitor) > 0:
        st.dataframe(
            pending_monitor[['App_Ref', 'LPA', 'Proposal', 'Valid_Date', 'Value_Score']].head(25),
            use_container_width=True, height=500
        )
    
    # Funnel
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
    rejected = (df['Status_Category'] == 'Rejected').sum()
    rejection_rate = (rejected / total * 100) if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
    with col2:
        withdrawn = (df['Status_Category'] == 'Withdrawn').sum()
        st.metric("Withdrawal Rate", f"{(withdrawn/total*100):.1f}%")
    with col3:
        avg_reject = df[df['Status_Category'] == 'Rejected']['Processing_Days'].median()
        st.metric("Median to Rejection", f"{avg_reject:.0f} days" if pd.notna(avg_reject) else "N/A")
    
    # High-risk LPAs
    lpa_risk = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Rejected').sum()
    })
    lpa_risk['Rejection_Rate'] = (lpa_risk['Status_Category'] / lpa_risk['App_Ref'] * 100).round(1)
    high_risk = lpa_risk[(lpa_risk['App_Ref'] >= 10) & (lpa_risk['Rejection_Rate'] > 30)].sort_values('Rejection_Rate', ascending=False)
    
    if len(high_risk) > 0:
        st.markdown(f"""
        <div class='insight-card urgent-card'>
            <h3 style='margin-top: 0; color: #FF3B30;'>🚨 {len(high_risk)} High-Risk Planning Authorities</h3>
            <p style='font-size: 1.1rem;'><strong>Avoid These Areas:</strong> {', '.join(high_risk.head(5).index.tolist())}</p>
            <p><strong>Action:</strong> If pursuing opportunities here, budget 50%+ extra for planning contingencies. 
            Consider pre-application consultations mandatory. Expect extended timelines.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.bar(high_risk.head(15), x='Rejection_Rate', y=high_risk.head(15).index,
                     orientation='h', color='Rejection_Rate', color_continuous_scale='Reds')
        fig.update_layout(height=550, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Use class risk
    use_risk = df.groupby('Use_Class_Hint').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Rejected').sum()
    })
    use_risk['Rejection_Rate'] = (use_risk['Status_Category'] / use_risk['App_Ref'] * 100).round(1)
    risky_uses = use_risk[(use_risk['App_Ref'] >= 5) & (use_risk['Rejection_Rate'] > 30)].sort_values('Rejection_Rate', ascending=False)
    
    if len(risky_uses) > 0:
        st.markdown("### High-Risk Use Classes")
        st.markdown(f"""
        <div class='insight-card warning-card'>
            <strong>Risky Use Classes:</strong> {', '.join(risky_uses.head(5).index.tolist())}<br>
            These use classes show >30% rejection rates. Proceed with caution.
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.bar(risky_uses.head(10), x='Rejection_Rate', y=risky_uses.head(10).index,
                     orientation='h', color='Rejection_Rate', color_continuous_scale='Oranges')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_strategic_insights(df):
    """Objective 5: Strategic Insights"""
    st.markdown("## 🎯 Strategic Insights")
    
    # Emerging hotspots
    recent = df[df['Valid_Date'] > datetime.now() - timedelta(days=365)]
    hotspots = recent.groupby('LPA').agg({
        'App_Ref': 'count',
        'Unit_Count': 'sum',
        'Value_Score': 'mean'
    }).sort_values('App_Ref', ascending=False).head(15)
    hotspots.columns = ['Applications', 'Total_Units', 'Avg_Value']
    
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='margin-top: 0; color: #34C759;'>🔥 Top 5 Emerging Hotspots (Last 12 Months)</h3>
        <p style='font-size: 1.1rem;'><strong>{', '.join(hotspots.head(5).index.tolist())}</strong></p>
        <p>High application volumes = strong development momentum. Focus land acquisition in these areas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(hotspots.reset_index(), x='Applications', y='Total_Units', 
                     size='Avg_Value', hover_data=['LPA'],
                     labels={'Applications': 'Applications (Last 12mo)', 
                            'Total_Units': 'Total Units Proposed'})
    fig.update_traces(marker=dict(color='#007AFF', line=dict(width=2, color='white')))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trending property types
    if df['Application_Quarter'].notna().any():
        st.markdown("### Property Types Trending Upward")
        
        quarterly = df.groupby(['Application_Quarter', 'Proposal_Type']).size().reset_index(name='Count')
        quarterly['Quarter'] = quarterly['Application_Quarter'].astype(str)
        
        top_types = df['Proposal_Type'].value_counts().head(5).index
        quarterly_filtered = quarterly[quarterly['Proposal_Type'].isin(top_types)]
        
        fig = px.line(quarterly_filtered, x='Quarter', y='Count', 
                     color='Proposal_Type', markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate growth
        latest_q = quarterly['Quarter'].max()
        prev_q = quarterly['Quarter'].unique()[-2] if len(quarterly['Quarter'].unique()) > 1 else latest_q
        
        latest_data = quarterly[quarterly['Quarter'] == latest_q].set_index('Proposal_Type')['Count']
        prev_data = quarterly[quarterly['Quarter'] == prev_q].set_index('Proposal_Type')['Count']
        
        growth = ((latest_data - prev_data) / prev_data * 100).dropna().sort_values(ascending=False)
        
        if len(growth) > 0:
            st.markdown(f"""
            <div class='insight-card success-card'>
                <strong>📈 Fastest Growing:</strong> {growth.index[0]} (+{growth.values[0]:.0f}% QoQ)<br>
                <strong>Action:</strong> Increase focus on {growth.index[0].lower()} opportunities.
            </div>
            """, unsafe_allow_html=True)
    
    # Underserved markets
    st.markdown("### Underserved Market Opportunities")
    
    use_competition = df.groupby(['LPA', 'Use_Class_Hint']).size().reset_index(name='Count')
    underserved = use_competition[use_competition['Count'] <= 3].sort_values('Count')
    
    if len(underserved) > 0:
        st.markdown(f"""
        <div class='insight-card warning-card'>
            <h3>💡 {len(underserved)} Underserved LPA + Use Class Combinations</h3>
            <p>Low competition = first-mover advantage. Consider feasibility studies for these niches.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(underserved.head(30), use_container_width=True, height=400)
    
    # Complex parcels
    if 'Matched_Parcel_ID' in df.columns and df['Matched_Parcel_ID'].notna().any():
        parcel_counts = df['Matched_Parcel_ID'].value_counts()
        complex_parcels = parcel_counts[parcel_counts > 1]
        
        if len(complex_parcels) > 0:
            st.markdown(f"""
            <div class='insight-card'>
                <h3>🏗️ {len(complex_parcels)} Parcels with Multiple Applications</h3>
                <p>Multiple applications indicate either: (1) High-value complex sites, or (2) Challenging planning history. 
                Requires detailed investigation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            complex_data = df[df['Matched_Parcel_ID'].isin(complex_parcels.index)].groupby('Matched_Parcel_ID').agg({
                'App_Ref': 'count',
                'LPA': 'first',
                'Status_Category': lambda x: ', '.join(x.unique()),
                'Proposal': 'first'
            }).reset_index()
            complex_data.columns = ['Parcel_ID', 'App_Count', 'LPA', 'Statuses', 'Sample_Proposal']
            
            st.dataframe(complex_data.head(20), use_container_width=True, height=400)

def show_executive_summary(df):
    """Executive dashboard with actionable insights"""
    st.markdown("## 📊 Executive Summary")
    st.markdown("### Actionable Investment Recommendations")
    
    # Calculate key insights
    lpa_stats = df.groupby('LPA').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Approved').sum(),
        'Processing_Days': 'median'
    })
    lpa_stats['Approval_Rate'] = (lpa_stats['Status_Category'] / lpa_stats['App_Ref'] * 100)
    
    best_lpas = lpa_stats[(lpa_stats['App_Ref'] >= 20) & (lpa_stats['Approval_Rate'] >= 70)].sort_values('Approval_Rate', ascending=False).head(5)
    high_risk_lpas = lpa_stats[(lpa_stats['App_Ref'] >= 10) & (lpa_stats['Approval_Rate'] < 50)].sort_values('Approval_Rate').head(5)
    
    recent_approved = df[(df['Status_Category'] == 'Approved') & (df['Decision_Date'] > datetime.now() - timedelta(days=90))]
    high_value_approved = df[(df['Status_Category'] == 'Approved') & (df['Value_Score'] >= 70)]
    
    # Recommendation 1: Where to focus
    st.markdown(f"""
    <div class='insight-card success-card'>
        <h3 style='color: #34C759;'>🎯 Priority 1: Focus Acquisition Efforts Here</h3>
        <p style='font-size: 1.2rem;'><strong>{', '.join(best_lpas.head(3).index.tolist())}</strong></p>
        <p><strong>Why:</strong> These LPAs have ≥70% approval rates, ≥20 applications, and fast processing times.</p>
        <p><strong>Action:</strong> Set up weekly automated searches for new listings in these areas. 
        Contact local agents and landowners proactively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation 2: Immediate opportunities
    st.markdown(f"""
    <div class='insight-card urgent-card'>
        <h3 style='color: #FF3B30;'>⚡ Priority 2: {len(recent_approved)} Time-Sensitive Opportunities</h3>
        <p><strong>Recently Approved (Last 90 Days):</strong> {len(recent_approved):,} applications</p>
        <p><strong>High-Value (Score 70+):</strong> {len(high_value_approved):,} applications</p>
        <p><strong>Action:</strong> Review detailed list below. Contact planning authorities tomorrow for landowner details. 
        Move fast - these won't last.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_value_approved) > 0:
        st.dataframe(
            high_value_approved[['App_Ref', 'LPA', 'Proposal', 'Unit_Count', 'Decision_Date', 'Value_Score']].head(20),
            use_container_width=True, height=400
        )
    
    # Recommendation 3: What types to pursue
    type_perf = df.groupby('Proposal_Type').agg({
        'App_Ref': 'count',
        'Status_Category': lambda x: (x == 'Approved').sum()
    })
    type_perf['Approval_Rate'] = (type_perf['Status_Category'] / type_perf['App_Ref'] * 100)
    best_types = type_perf[type_perf['App_Ref'] >= 10].sort_values('Approval_Rate', ascending=False).head(3)
    
    st.markdown(f"""
    <div class='insight-card'>
        <h3>🏗️ Priority 3: Best Development Types</h3>
        <p><strong>Highest Success Rates:</strong></p>
        <p>1. {best_types.index[0]}: {best_types['Approval_Rate'].values[0]:.0f}% approval rate</p>
        <p>2. {best_types.index[1]}: {best_types['Approval_Rate'].values[1]:.0f}% approval rate</p>
        <p>3. {best_types.index[2]}: {best_types['Approval_Rate'].values[2]:.0f}% approval rate</p>
        <p><strong>Action:</strong> Prioritize {best_types.index[0].lower()} opportunities for best risk-adjusted returns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation 4: Areas to avoid
    if len(high_risk_lpas) > 0:
        st.markdown(f"""
        <div class='insight-card urgent-card'>
            <h3 style='color: #FF3B30;'>⚠️ Priority 4: High-Risk Areas to Avoid</h3>
            <p><strong>Challenging LPAs:</strong> {', '.join(high_risk_lpas.index.tolist())}</p>
            <p><strong>Risk:</strong> <50% approval rates. Extended timelines. Unpredictable outcomes.</p>
            <p><strong>Action:</strong> Avoid unless exceptional opportunity. If pursuing, budget 2x planning costs and 50% longer timeline.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market context
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Market Context: Interest Rates")
        st.markdown("""
        <div class='hero-gradient' style='padding: 2rem;'>
            <h3 style='color: white; margin: 0;'>BoE Base Rate: 4.00%</h3>
            <p style='color: white; margin-top: 10px; opacity: 0.9;'>
            <strong>Development Finance: ~6.5%</strong><br>
            Monthly interest per £1M loan: £5,417
            </p>
            <p style='color: white; margin-top: 10px; opacity: 0.9;'>
            <strong>Strategy:</strong> Higher rates favor quick-turnaround projects. 
            Focus on approved applications with 12-18 month completion timelines to minimize carry costs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Timeline Planning")
        median_process = df['Processing_Days'].median()
        p90_process = df['Processing_Days'].quantile(0.9)
        
        st.markdown(f"""
        <div class='insight-card'>
            <h3>⏱️ Planning Timeline Benchmarks</h3>
            <p><strong>Median Decision Time:</strong> {median_process:.0f} days</p>
            <p><strong>90th Percentile:</strong> {p90_process:.0f} days</p>
            <p><strong>Pro Forma Assumption:</strong> Use {p90_process:.0f} days for conservative planning</p>
            <p><strong>Best Case LPAs:</strong> {df.groupby('LPA')['Processing_Days'].median().nsmallest(3).index[0]} 
            ({df.groupby('LPA')['Processing_Days'].median().nsmallest(3).values[0]:.0f}d median)</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application - seamless Google Sheets integration"""
    
    # Auto-load data from Google Sheets
    with st.spinner("Loading planning data..."):
        df_raw = load_from_google_sheets()
    
    if df_raw is None:
        st.error("Unable to load data from Google Sheets. Check your secrets configuration.")
        st.info("Ensure spreadsheet_id and gcp_service_account are set in Streamlit secrets.")
        return
    
    # Analyze data
    df = analyze_data(df_raw)
    
    # Show last update
    if 'Run_Date' in df.columns and df['Run_Date'].notna().any():
        latest = df['Run_Date'].max()
        st.sidebar.success(f"✅ Updated: {latest.strftime('%Y-%m-%d')}")
    
    # Hero section
    show_hero(df)
    
    st.markdown("---")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary",
        "🎯 Opportunities", 
        "🏛️ Market Intelligence",
        "📋 Pipeline",
        "⚠️ Risk Assessment",
        "🎯 Strategic Insights"
    ])
    
    with tab1:
        show_executive_summary(df)
    
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
    
    # Footer
    st.markdown("---")
    st.caption("Planning Intelligence | Powered by real-time data from Google Sheets")

if __name__ == "__main__":
    main()
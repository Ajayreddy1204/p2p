import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import json
import math
from datetime import date, timedelta
from typing import Dict, Any
import boto3
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

# ------------------------------
# 1. AWS Athena Connection
# ------------------------------
ATHENA_S3_STAGING_DIR = st.secrets.get("ATHENA_S3_STAGING_DIR", "s3://yignite-procurespendiq-miniature-landing/procure2pay_dev/athena-results/")#s3://yignite-procurespendiq-miniature-landing/procure2pay_dev/athena-results/
ATHENA_REGION = st.secrets.get("ATHENA_REGION", "us-east-1")
ATHENA_DATABASE = st.secrets.get("ATHENA_DATABASE", "procure2pay")
ATHENA_CATALOG = "AwsDataCatalog"

@st.cache_resource
def get_athena_connection():
    return connect(
        s3_staging_dir=ATHENA_S3_STAGING_DIR,
        region_name=ATHENA_REGION,
        catalog_name=ATHENA_CATALOG,
        schema_name=ATHENA_DATABASE,
        cursor_class=PandasCursor,
    )

def run_df(sql: str) -> pd.DataFrame:
    try:
        conn = get_athena_connection()
        return conn.cursor().execute(sql).as_pandas()
    except Exception as e:
        st.warning(f"Query failed: {e}\nSQL: {sql}")
        return pd.DataFrame()

# ------------------------------
# 2. AWS Bedrock (Nova) Helper
# ------------------------------
BEDROCK_MODEL_ID = "amazon.nova-pro-v1:0"
bedrock_client = None

def get_bedrock_client():
    global bedrock_client
    if bedrock_client is None:
        try:
            bedrock_client = boto3.client("bedrock-runtime", region_name=ATHENA_REGION)
        except Exception:
            bedrock_client = None
    return bedrock_client

def call_bedrock(prompt: str) -> str:
    client = get_bedrock_client()
    if not client:
        return ""
    try:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0.3}
        })
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        result = json.loads(response["body"].read().decode("utf-8"))
        return result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
    except Exception as e:
        st.warning(f"Bedrock call failed: {e}")
        return ""

# ------------------------------
# 3. Utility Functions
# ------------------------------
def sql_date(d: date) -> str:
    return f"DATE '{d.strftime('%Y-%m-%d')}'"

def safe_number(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def safe_int(val, default=0):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return int(float(val))
    except Exception:
        return default

def abbr_currency(v: float, currency_symbol: str = "$") -> str:
    n = abs(v)
    sign = "-" if v < 0 else ""
    if n >= 1_000_000_000: return f"{sign}{currency_symbol}{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:     return f"{sign}{currency_symbol}{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{sign}{currency_symbol}{n/1_000:.1f}K"
    return f"{sign}{currency_symbol}{n:.0f}"

def compute_range_preset(preset: str):
    today = date.today()
    if preset == "Last 30 Days":
        return today - timedelta(days=30), today
    if preset == "QTD":
        start = date(today.year, ((today.month - 1)//3)*3 + 1, 1)
        return start, today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    return today.replace(day=1), today

def prior_window(start: date, end: date):
    from calendar import monthrange
    # If full calendar month, compare to same month previous year
    if start.day == 1 and end.day == monthrange(end.year, end.month)[1]:
        prev_start = date(start.year - 1, start.month, 1)
        prev_end = date(end.year - 1, end.month, monthrange(end.year - 1, end.month)[1])
        return prev_start, prev_end
    days = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return prev_start, prev_end

def pct_delta(cur: float, prev: float):
    if prev == 0:
        return None, True, False
    change = (cur - prev) / prev * 100.0
    if abs(change) < 0.05:
        return "0%", True, True
    sign = "+" if change >= 0 else "−"
    return f"{sign}{abs(change):.1f}%", change >= 0, False

def build_vendor_where(selected_vendor: str) -> str:
    if selected_vendor == "All Vendors":
        return ""
    safe_vendor = selected_vendor.replace("'", "''")
    return f" AND UPPER(v.vendor_name) = UPPER('{safe_vendor}') "

def _pick_chart_columns(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return (None, None)
    cols = list(df.columns)
    cat_prefer = ("opportunity_area", "aging_bucket", "invoice_status", "po_purpose", "vendor_name", "driver_value", "driver", "status", "month")
    num_prefer = ("amount", "total_amount", "spend_change", "invoice_count", "total_spend", "spend", "cnt")
    upper = {str(c).upper(): c for c in cols}
    x = None
    for name in cat_prefer:
        if name.upper() in upper:
            x = upper[name.upper()]
            break
    if not x:
        for c in cols:
            if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
                x = c
                break
    if not x:
        x = cols[0]
    y = None
    for name in num_prefer:
        if name.upper() in upper and upper[name.upper()] != x:
            y = upper[name.upper()]
            break
    if not y:
        for c in cols:
            if c != x and pd.api.types.is_numeric_dtype(df[c]):
                y = c
                break
    if not y and len(cols) > 1:
        y = cols[1]
    return (x, y)

# ------------------------------
# 4. Altair Charts
# ------------------------------
def alt_bar(df, x, y, title=None, horizontal=False, color="#1459d2", height=320):
    if df.empty:
        st.info("No data for chart")
        return
    if horizontal:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X(y, type='quantitative', axis=alt.Axis(grid=False, title=None, format="~s")),
            y=alt.Y(x, type='nominal', sort='-x', axis=alt.Axis(grid=False, title=None)),
            tooltip=[x, alt.Tooltip(y, title="Value", format=",.0f")]
        )
    else:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X(x, type='nominal', axis=alt.Axis(grid=False, title=None, labelAngle=-45 if len(df[x].unique()) > 5 else 0)),
            y=alt.Y(y, type='quantitative', axis=alt.Axis(grid=False, title=None, format="~s")),
            tooltip=[x, alt.Tooltip(y, title="Value", format=",.0f")]
        )
    if title:
        chart = chart.properties(title=title).configure_title(color="#0f172a")
    chart = chart.properties(height=height).configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True)

def alt_line_monthly(df, month_col='month', value_col='value', height=140, title=None):
    if df.empty:
        st.info("No data for trend")
        return
    df = df.copy()
    try:
        df['_dt'] = pd.to_datetime(df[month_col].astype(str) + '-01')
        df = df.sort_values('_dt')
        df['month_label'] = df['_dt'].dt.strftime('%b %Y')
    except:
        df['month_label'] = df[month_col].astype(str)
    chart = alt.Chart(df).mark_line(point=True, color='#1e88e5').encode(
        x=alt.X('month_label:N', sort=None, axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y(value_col, type='quantitative', axis=alt.Axis(grid=False, title=None, format="~s")),
        tooltip=[alt.Tooltip('month_label:N', title='Month'), alt.Tooltip(value_col, format=",.0f")]
    ).properties(height=height).configure_view(stroke=None)
    if title:
        chart = chart.properties(title=title).configure_title(color='#0f172a')
    st.altair_chart(chart, use_container_width=True)

def alt_donut_status(df, label_col, value_col, title=None, height=340):
    if df.empty or df[value_col].sum() == 0:
        st.info("No data for status distribution")
        return
    total = df[value_col].sum()
    df['pct'] = df[value_col] / total
    order = ["Paid", "Pending", "Disputed", "Other"]
    palette = {"Paid": "#22C55E", "Pending": "#FBBF24", "Disputed": "#EF4444", "Other": "#1E88E5"}
    base = alt.Chart(df).encode(
        theta=alt.Theta(value_col, type='quantitative', stack=True),
        color=alt.Color(label_col, type='nominal', scale=alt.Scale(domain=order, range=[palette.get(k, "#1E88E5") for k in order]), legend=alt.Legend(title=None, orient='right')),
        tooltip=[label_col, alt.Tooltip(value_col, format=",.0f"), alt.Tooltip('pct', format=".1%")]
    )
    arc = base.mark_arc(innerRadius=40, outerRadius=100)
    text = base.transform_filter(alt.datum.pct >= 0.01).mark_text(radius=115, fontSize=12, fontWeight='bold').encode(text=alt.Text('pct', format='.1%'))
    center = alt.Chart(pd.DataFrame({'total': [total]})).mark_text(fontSize=24, fontWeight='bold', color='#0f172a').encode(text='total:Q')
    sub = alt.Chart(pd.DataFrame({'lbl': ['TOTAL']})).mark_text(dy=18, fontSize=11, color='#64748b').encode(text='lbl:N')
    chart = (arc + text + center + sub).properties(height=height).configure_view(stroke=None)
    if title:
        chart = chart.properties(title=title).configure_title(color='#0f172a')
    st.altair_chart(chart, use_container_width=True)

# ------------------------------
# 5. UI Components
# ------------------------------
def branding_bar():
    cur_page = st.session_state.get('page', 'dashboard')
    st.markdown("""
    <style>
    .brandbar {
        position: sticky; top: 0; z-index: 100;
        background: white; border-bottom: 1px solid #e5e7eb;
        padding: 8px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .brandrow {
        max-width: 1180px; margin: 0 auto;
        display: flex; justify-content: space-between; align-items: center;
    }
    .brand-left { display: flex; align-items: baseline; gap: 20px; }
    .brand-title { font-size: 24px; font-weight: 900; color: #0f172a; }
    .brand-sub { font-size: 12px; color: #64748b; }
    .topnav { display: flex; gap: 8px; }
    .nav-item {
        padding: 8px 16px; border-radius: 999px; font-weight: 600;
        color: #475569; background: transparent; cursor: pointer;
        text-decoration: none;
    }
    .nav-item.active {
        background: #2563eb; color: white;
    }
    .brand-right img { height: 40px; }
    </style>
    <div class="brandbar">
        <div class="brandrow">
            <div class="brand-left">
                <div>
                    <div class="brand-title">ProcureIQ</div>
                    <div class="brand-sub">P2P Analytics</div>
                </div>
            </div>
            <div class="topnav">
    """, unsafe_allow_html=True)
    pages = [("dashboard", "Dashboard"), ("genie", "Genie"), ("cash_flow", "Forecast"), ("invoice", "Invoices")]
    for key, label in pages:
        active = "active" if cur_page == key else ""
        st.markdown(f'<a href="#" class="nav-item {active}" onclick="window.parent.dispatchEvent(new CustomEvent(\'set-page\', {{detail: \'{key}\'}}))">{label}</a>', unsafe_allow_html=True)
    st.markdown("""
            </div>
            <div class="brand-right">
                <img src="https://upload.wikimedia.org/wikipedia/commons/2/2e/Yash_Technologies_logo.png" />
            </div>
        </div>
    </div>
    <hr style="margin:0 0 16px 0;">
    """, unsafe_allow_html=True)
    # Handle custom event via query params
    if st.query_params.get('page'):
        st.session_state.page = st.query_params.get('page')
        st.rerun()

def kpi_tile(title: str, value: str, delta_text: str = None, is_up_change: bool = True):
    arrow_up_svg = '<svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path d="M10 3l6 6H4l6-6zm0 14V6h-2v11h2z"/></svg>'
    arrow_down_svg = '<svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path d="M10 17l-6-6h12l-6 6zm0-14v11h2V3h-2z"/></svg>'
    if delta_text and delta_text != '—':
        delta_class = "up" if is_up_change else "down"
        arrow = arrow_up_svg if is_up_change else arrow_down_svg
        delta_html = f'<div class="delta {delta_class}" style="margin-top:4px; font-size:14px;">{arrow} {delta_text}</div>'
    else:
        delta_html = ""
    st.markdown(f"""
    <div class="kpi" style="background:#fff; border:1px solid #e6e8ee; border-radius:12px; padding:12px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">
        <div class="title" style="font-size:12px; color:#64748b; font-weight:800;">{title}</div>
        <div class="value" style="font-size:28px; font-weight:900; margin-top:4px;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# 6. Page Renderers
# ------------------------------
def render_dashboard():
    # Date and vendor filters
    if "preset" not in st.session_state:
        st.session_state.preset = "Last 30 Days"
    if "date_range" not in st.session_state:
        st.session_state.date_range = compute_range_preset(st.session_state.preset)
    if "vendor" not in st.session_state:
        st.session_state.vendor = "All Vendors"

    col1, col2, col3 = st.columns([1, 1, 1.8])
    with col1:
        rng_start, rng_end = st.date_input("Date Range", value=st.session_state.date_range, format="YYYY-MM-DD", label_visibility="collapsed")
        if isinstance(rng_start, (list, tuple)):
            rng_start, rng_end = rng_start[0], rng_start[1]
        st.session_state.date_range = (rng_start, rng_end)
        if st.session_state.preset != "Custom" and (rng_start, rng_end) != compute_range_preset(st.session_state.preset):
            st.session_state.preset = "Custom"
    with col2:
        vendors_df = run_df("SELECT DISTINCT vendor_name FROM dim_vendor_vw ORDER BY vendor_name")
        vendor_list = ["All Vendors"] + vendors_df['vendor_name'].dropna().tolist() if not vendors_df.empty else ["All Vendors"]
        vendor = st.selectbox("Vendor", vendor_list, index=vendor_list.index(st.session_state.vendor) if st.session_state.vendor in vendor_list else 0, label_visibility="collapsed")
        st.session_state.vendor = vendor
    with col3:
        presets = ["Last 30 Days", "QTD", "YTD", "Custom"]
        pcols = st.columns(4)
        for i, p in enumerate(presets):
            with pcols[i]:
                if st.button(p, key=f"preset_{p}", use_container_width=True, type="primary" if p == st.session_state.preset else "secondary"):
                    st.session_state.preset = p
                    if p != "Custom":
                        st.session_state.date_range = compute_range_preset(p)
                    st.rerun()

    start_lit, end_lit = sql_date(rng_start), sql_date(rng_end)
    p_start, p_end = prior_window(rng_start, rng_end)
    p_start_lit, p_end_lit = sql_date(p_start), sql_date(p_end)
    vendor_where = build_vendor_where(vendor)

    # KPI queries (current vs prior)
    kpi_sql = f"""
    WITH base AS (
        SELECT f.*, v.vendor_name
        FROM fact_all_sources_vw f
        LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
    )
    SELECT
        COUNT(DISTINCT CASE WHEN upper(invoice_status) = 'OPEN' THEN purchase_order_reference END) AS active_pos,
        COUNT(DISTINCT purchase_order_reference) AS total_pos,
        SUM(CASE WHEN upper(invoice_status) NOT IN ('CANCELLED','REJECTED') THEN coalesce(invoice_amount_local,0) ELSE 0 END) AS total_spend,
        COUNT(DISTINCT vendor_name) AS active_vendors,
        COUNT(DISTINCT CASE WHEN upper(invoice_status) = 'OPEN' THEN invoice_number END) AS pending_inv
    FROM base
    """
    kpi_cur = run_df(kpi_sql)
    kpi_prev_sql = kpi_sql.replace(f"BETWEEN {start_lit} AND {end_lit}", f"BETWEEN {p_start_lit} AND {p_end_lit}")
    kpi_prev = run_df(kpi_prev_sql)

    cur_spend = safe_number(kpi_cur.at[0,'total_spend'] if not kpi_cur.empty else 0)
    cur_pos = safe_int(kpi_cur.at[0,'active_pos'] if not kpi_cur.empty else 0)
    cur_total_pos = safe_int(kpi_cur.at[0,'total_pos'] if not kpi_cur.empty else 0)
    cur_vend = safe_int(kpi_cur.at[0,'active_vendors'] if not kpi_cur.empty else 0)
    cur_pend = safe_int(kpi_cur.at[0,'pending_inv'] if not kpi_cur.empty else 0)

    prev_spend = safe_number(kpi_prev.at[0,'total_spend'] if not kpi_prev.empty else 0)
    prev_pos = safe_int(kpi_prev.at[0,'active_pos'] if not kpi_prev.empty else 0)
    prev_total_pos = safe_int(kpi_prev.at[0,'total_pos'] if not kpi_prev.empty else 0)
    prev_vend = safe_int(kpi_prev.at[0,'active_vendors'] if not kpi_prev.empty else 0)
    prev_pend = safe_int(kpi_prev.at[0,'pending_inv'] if not kpi_prev.empty else 0)

    d_spend, up_spend, _ = pct_delta(cur_spend, prev_spend)
    d_pos, up_pos, _ = pct_delta(cur_pos, prev_pos)
    d_total_pos, up_total_pos, _ = pct_delta(cur_total_pos, prev_total_pos)
    d_vend, up_vend, _ = pct_delta(cur_vend, prev_vend)
    d_pend, up_pend, _ = pct_delta(cur_pend, prev_pend)

    kpis = [
        ("TOTAL SPEND", abbr_currency(cur_spend), d_spend, up_spend),
        ("ACTIVE PO'S", f"{cur_pos:,}", d_pos, up_pos),
        ("TOTAL PO'S", f"{cur_total_pos:,}", d_total_pos, up_total_pos),
        ("ACTIVE VENDORS", f"{cur_vend:,}", d_vend, up_vend),
        ("PENDING INVOICES", f"{cur_pend:,}", d_pend, up_pend)
    ]
    cols = st.columns(5)
    for i, (label, val, delta, up) in enumerate(kpis):
        with cols[i]:
            kpi_tile(label, val, delta, up)

    st.markdown("---")

    # Needs Attention Tabs
    st.subheader("Needs Attention")
    na_counts = run_df(f"""
        SELECT
            SUM(CASE WHEN due_date < CURRENT_DATE AND upper(invoice_status) = 'OVERDUE' THEN 1 ELSE 0 END) AS overdue,
            SUM(CASE WHEN upper(invoice_status) = 'DISPUTED' THEN 1 ELSE 0 END) AS disputed,
            SUM(CASE WHEN due_date >= CURRENT_DATE AND upper(invoice_status) = 'OPEN' THEN 1 ELSE 0 END) AS due
        FROM fact_all_sources_vw
        WHERE posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
    """)
    overdue_cnt = safe_int(na_counts.at[0,'overdue'] if not na_counts.empty else 0)
    disputed_cnt = safe_int(na_counts.at[0,'disputed'] if not na_counts.empty else 0)
    due_cnt = safe_int(na_counts.at[0,'due'] if not na_counts.empty else 0)

    na_tab = st.radio("", ["Overdue", "Disputed", "Due"], horizontal=True, label_visibility="collapsed")
    if na_tab == "Overdue":
        na_sql = f"""
        SELECT f.invoice_number, f.invoice_amount_local, f.due_date, upper(f.invoice_status) as status, v.vendor_name, f.aging_days
        FROM fact_all_sources_vw f
        LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
          AND f.due_date < CURRENT_DATE AND upper(f.invoice_status) = 'OVERDUE'
        ORDER BY f.due_date
        LIMIT 8
        """
    elif na_tab == "Disputed":
        na_sql = f"""
        SELECT f.invoice_number, f.invoice_amount_local, f.due_date, upper(f.invoice_status) as status, v.vendor_name, f.aging_days
        FROM fact_all_sources_vw f
        LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
          AND upper(f.invoice_status) = 'DISPUTED'
        ORDER BY f.due_date
        LIMIT 8
        """
    else:
        na_sql = f"""
        SELECT f.invoice_number, f.invoice_amount_local, f.due_date, upper(f.invoice_status) as status, v.vendor_name, f.aging_days
        FROM fact_all_sources_vw f
        LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
          AND f.due_date >= CURRENT_DATE AND upper(f.invoice_status) = 'OPEN'
        ORDER BY f.due_date
        LIMIT 8
        """
    na_df = run_df(na_sql)
    if na_df.empty:
        st.info(f"No {na_tab.lower()} invoices in this period.")
    else:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(na_df.iterrows()):
            with cols[idx % 4]:
                ref = row['invoice_number']
                amt = abbr_currency(row['invoice_amount_local'])
                due = row['due_date'].strftime('%Y-%m-%d') if pd.notna(row['due_date']) else '—'
                vendor = row['vendor_name']
                status = row['status'].lower()
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:8px; margin-bottom:8px;">
                    <div><strong>{ref}</strong></div>
                    <div style="font-size:12px; color:#64748b;">{vendor}</div>
                    <div>{amt}</div>
                    <div style="font-size:11px;">Due: {due}</div>
                    <span class="tag {status}">{status}</span>
                </div>
                """, unsafe_allow_html=True)
        if len(na_df) > 4:
            st.caption(f"Showing first 8 of {len(na_df)} {na_tab.lower()} invoices.")

    st.markdown("---")
    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        # Invoice status donut
        status_df = run_df(f"""
            SELECT CASE
                WHEN upper(invoice_status) IN ('PAID','CLEARED') THEN 'Paid'
                WHEN upper(invoice_status) IN ('OPEN','PENDING') THEN 'Pending'
                WHEN upper(invoice_status) = 'DISPUTED' THEN 'Disputed'
                ELSE 'Other'
            END AS status, COUNT(*) AS cnt
            FROM fact_all_sources_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
            GROUP BY 1
        """)
        alt_donut_status(status_df, 'status', 'cnt', title="Invoice Status", height=300)
    with col_b:
        # Top 10 vendors
        vendors_df = run_df(f"""
            SELECT v.vendor_name, SUM(f.invoice_amount_local) AS spend
            FROM fact_all_sources_vw f
            LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
            GROUP BY 1
            ORDER BY spend DESC
            LIMIT 10
        """)
        alt_bar(vendors_df, x='vendor_name', y='spend', title="Top 10 Vendors by Spend", horizontal=True, color="#22C55E", height=300)

def render_genie():
    st.header("ProcureIQ Genie")
    st.markdown("Ask natural language questions about your procurement data.")

    # Quick analysis buttons
    quick_questions = {
        "Spending Overview": "Show me total spend YTD, monthly trends, and top 5 vendors",
        "Vendor Analysis": "Analyze vendor concentration and dependency",
        "Payment Performance": "Show payment delays and cycle time issues",
        "Invoice Aging": "Show overdue invoices by aging buckets",
        "Cost Reduction": "Suggest ways to reduce procurement costs based on our spend data",
        "First Pass PO's": "Show me first pass PO's - purchase orders where all invoices were paid without disputes or overdue"
    }
    cols = st.columns(3)
    for i, (label, question) in enumerate(quick_questions.items()):
        with cols[i % 3]:
            if st.button(label, use_container_width=True):
                with st.spinner("Analyzing..."):
                    response = process_genie_query(question)
                    st.session_state.analyst_response = response
                    st.session_state.show_analysis = True
                    st.rerun()

    # Custom query input
    with st.form("genie_form"):
        user_q = st.text_input("Ask anything:", placeholder="e.g., What is our total spend this month?")
        submitted = st.form_submit_button("Ask Genie")
        if submitted and user_q:
            with st.spinner("Analyzing..."):
                response = process_genie_query(user_q)
                st.session_state.analyst_response = response
                st.session_state.show_analysis = True
                st.rerun()

    if st.session_state.get("show_analysis") and st.session_state.get("analyst_response"):
        response = st.session_state.analyst_response
        if "error" in response:
            st.error(response["error"])
        elif "layout" in response and response["layout"] == "quick":
            # Quick analysis result from pre‑defined SQL
            m = response.get("metrics", {})
            if m:
                col1, col2, col3 = st.columns(3)
                if "total_ytd" in m:
                    col1.metric("Total Spend YTD", abbr_currency(m["total_ytd"]))
                if "mom_pct" in m:
                    col2.metric("MoM Change", f"{m['mom_pct']:.1f}%")
                if "top5_pct" in m:
                    col3.metric("Top 5 Vendors Share", f"{m['top5_pct']:.0f}%")
            if response.get("monthly_df") is not None:
                alt_line_monthly(response["monthly_df"], month_col='month', value_col='value', title="Monthly Spend Trend")
            if response.get("vendors_df") is not None:
                alt_bar(response["vendors_df"], x='vendor_name', y='spend', title="Top Vendors", horizontal=True)
            # Prescriptive from Bedrock
            pres = call_bedrock(f"Based on this spend data: {response.get('metrics')} and top vendors, give 3 specific actions to reduce costs or improve payment terms.")
            if pres:
                st.markdown("### Prescriptive Recommendations")
                st.info(pres)
        else:
            # General SQL response
            for block in response.get("content", []):
                if block.get("type") == "text":
                    st.markdown(block.get("text"))
                elif block.get("type") == "sql":
                    with st.expander("View data"):
                        df = run_df(block.get("statement"))
                        if not df.empty:
                            x, y = _pick_chart_columns(df)
                            if x and y:
                                alt_bar(df, x, y, height=300)
                            st.dataframe(df)
        if st.button("New Analysis"):
            st.session_state.show_analysis = False
            st.session_state.analyst_response = None
            st.rerun()

def process_genie_query(query: str) -> Dict[str, Any]:
    """Map natural language to verified queries or run generic SQL."""
    q_lower = query.lower()
    # First pass PO's
    if "first pass po" in q_lower:
        sql = """
        WITH inv_flags AS (
            SELECT purchase_order_reference AS po_number,
                   MAX(CASE WHEN upper(status) IN ('DISPUTED','OVERDUE') THEN 1 ELSE 0 END) AS has_issue,
                   MAX(CASE WHEN upper(status) IN ('PAID','CLEARED') THEN 1 ELSE 0 END) AS is_paid
            FROM invoice_status_history_vw
            WHERE purchase_order_reference IS NOT NULL
            GROUP BY purchase_order_reference, invoice_number
        ), po_agg AS (
            SELECT po_number,
                   COUNT(*) AS invoice_count,
                   SUM(is_paid) AS paid_count,
                   SUM(has_issue) AS issue_count
            FROM inv_flags
            GROUP BY po_number
        )
        SELECT po_number, invoice_count, paid_count, issue_count
        FROM po_agg
        WHERE issue_count = 0 AND paid_count = invoice_count AND invoice_count > 0
        LIMIT 20
        """
        df = run_df(sql)
        return {"layout": "quick", "vendors_df": df, "content": []}
    # Spending overview
    elif "spending overview" in q_lower or "total spend" in q_lower:
        total = run_df("SELECT SUM(invoice_amount_local) AS total FROM fact_all_sources_vw WHERE upper(invoice_status) NOT IN ('CANCELLED','REJECTED')")
        monthly = run_df("SELECT date_format(posting_date, '%Y-%m') AS month, SUM(invoice_amount_local) AS value FROM fact_all_sources_vw WHERE upper(invoice_status) NOT IN ('CANCELLED','REJECTED') GROUP BY 1 ORDER BY 1")
        top5 = run_df("SELECT v.vendor_name, SUM(f.invoice_amount_local) AS spend FROM fact_all_sources_vw f LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id GROUP BY 1 ORDER BY spend DESC LIMIT 5")
        # Calculate MoM change
        mom = 0
        if len(monthly) >= 2:
            cur = monthly.iloc[-1]['value']
            prev = monthly.iloc[-2]['value']
            mom = ((cur - prev) / prev * 100) if prev != 0 else 0
        top5_pct = (top5['spend'].sum() / total.at[0,'total'] * 100) if not total.empty and total.at[0,'total'] != 0 else 0
        return {"layout": "quick", "metrics": {"total_ytd": total.at[0,'total'] if not total.empty else 0, "mom_pct": mom, "top5_pct": top5_pct}, "monthly_df": monthly, "vendors_df": top5}
    # Vendor analysis
    elif "vendor analysis" in q_lower or "vendor concentration" in q_lower:
        vendors = run_df("SELECT v.vendor_name, SUM(f.invoice_amount_local) AS spend, COUNT(*) AS cnt FROM fact_all_sources_vw f LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id GROUP BY 1 ORDER BY spend DESC")
        return {"layout": "quick", "vendors_df": vendors}
    # Payment performance
    elif "payment performance" in q_lower or "payment delays" in q_lower:
        payment = run_df("SELECT date_format(posting_date, '%Y-%m') AS month, AVG(datediff('day', posting_date, payment_date)) AS avg_days FROM fact_all_sources_vw WHERE payment_date IS NOT NULL GROUP BY 1 ORDER BY 1")
        return {"layout": "quick", "monthly_df": payment}
    # Invoice aging
    elif "invoice aging" in q_lower or "overdue invoices" in q_lower:
        aging = run_df("""
            SELECT CASE WHEN aging_days <= 30 THEN '0-30 days' WHEN aging_days <= 60 THEN '31-60 days' WHEN aging_days <= 90 THEN '61-90 days' ELSE '90+ days' END AS bucket,
                   COUNT(*) AS cnt, SUM(invoice_amount_local) AS amount
            FROM fact_all_sources_vw
            WHERE upper(invoice_status) = 'OVERDUE' AND aging_days > 0
            GROUP BY 1
        """)
        return {"layout": "quick", "vendors_df": aging}
    # Cost reduction
    elif "cost reduction" in q_lower or "reduce costs" in q_lower:
        df = run_df("SELECT po_purpose, SUM(invoice_amount_local) AS spend FROM fact_all_sources_vw GROUP BY po_purpose ORDER BY spend DESC")
        pres = call_bedrock(f"Given spend by category: {df.to_dict()}, suggest 3 cost reduction actions.")
        return {"layout": "quick", "vendors_df": df, "content": [{"type": "text", "text": pres}] if pres else []}
    else:
        # Generic: run a simple SQL that tries to answer
        sql = f"""
        SELECT 'Total Spend' AS metric, SUM(invoice_amount_local) AS value
        FROM fact_all_sources_vw
        WHERE upper(invoice_status) NOT IN ('CANCELLED','REJECTED')
        """
        df = run_df(sql)
        return {"layout": "quick", "metrics": {"total": df.at[0,'value'] if not df.empty else 0}, "content": []}

def render_forecast():
    st.header("Cash Flow & GR/IR Forecast")
    tab1, tab2 = st.tabs(["Cash Flow Forecast", "GR/IR Reconciliation"])
    with tab1:
        st.subheader("Unpaid Obligations by Due Date")
        cf_sql = """
        SELECT
            CASE WHEN days_until_due < 0 THEN 'Overdue'
                 WHEN days_until_due <= 7 THEN 'Due 0-7 days'
                 WHEN days_until_due <= 30 THEN 'Due 8-30 days'
                 WHEN days_until_due <= 60 THEN 'Due 31-60 days'
                 WHEN days_until_due <= 90 THEN 'Due 61-90 days'
                 ELSE 'Beyond 90 days'
            END AS bucket,
            COUNT(*) AS invoice_count,
            SUM(invoice_amount_local) AS total_amount,
            MIN(due_date) AS earliest_due,
            MAX(due_date) AS latest_due
        FROM cash_flow_unpaid_obligations_vw
        GROUP BY 1
        ORDER BY MIN(due_date)
        """
        cf = run_df(cf_sql)
        if not cf.empty:
            st.dataframe(cf, use_container_width=True)
            total = cf['total_amount'].sum()
            st.metric("Total Unpaid", abbr_currency(total))
            alt_bar(cf, x='bucket', y='total_amount', title="Cash Outflow by Bucket", color="#1e88e5")
        else:
            st.info("No unpaid obligations found.")

        st.markdown("### Payment Strategy")
        if st.button("Ask Genie for optimal payment timing"):
            st.session_state.page = "genie"
            st.session_state.genie_prefill = "Which invoices should we pay early to capture discounts?"
            st.rerun()

    with tab2:
        st.subheader("GR/IR Outstanding Balance")
        grir = run_df("""
            SELECT year, month, invoice_count, total_grir_blnc
            FROM gr_ir_outstanding_balance_vw
            ORDER BY year DESC, month DESC
            LIMIT 12
        """)
        if not grir.empty:
            st.dataframe(grir, use_container_width=True)
            alt_bar(grir, x='month', y='total_grir_blnc', title="GR/IR Balance Trend", color="#ef4444")
        else:
            st.info("No GR/IR data.")

def render_invoices():
    st.header("Invoice Management")
    search_term = st.text_input("Search by Invoice or PO Number", key="inv_search")
    if search_term:
        df = run_df(f"""
            SELECT f.invoice_number, f.purchase_order_reference, f.invoice_amount_local, f.posting_date, f.due_date, upper(f.invoice_status) as status, v.vendor_name
            FROM fact_all_sources_vw f
            LEFT JOIN dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.invoice_number LIKE '%{search_term}%' OR f.purchase_order_reference LIKE '%{search_term}%'
            ORDER BY f.posting_date DESC
            LIMIT 100
        """)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            # AI suggestion for the first invoice
            inv_num = df.iloc[0]['invoice_number']
            prompt = f"Analyze invoice {inv_num}: status {df.iloc[0]['status']}, amount {df.iloc[0]['invoice_amount_local']}, due {df.iloc[0]['due_date']}. Suggest action."
            suggestion = call_bedrock(prompt)
            if suggestion:
                st.info(f"Genie Suggestion: {suggestion}")
        else:
            st.info("No invoices found.")
    else:
        st.info("Enter an invoice or PO number to search.")

# ------------------------------
# 7. Main App
# ------------------------------
st.set_page_config(page_title="ProcureIQ", layout="wide", page_icon=None)
# Load minimal CSS (inline to avoid external files)
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.kpi { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; }
.tag.overdue { background: #fee2e2; color: #b91c1c; }
.tag.disputed { background: #fff4e5; color: #b54708; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "analyst_response" not in st.session_state:
    st.session_state.analyst_response = None

branding_bar()

# Page routing
if st.session_state.page == "dashboard":
    render_dashboard()
elif st.session_state.page == "genie":
    render_genie()
elif st.session_state.page == "cash_flow":
    render_forecast()
elif st.session_state.page == "invoice":
    render_invoices()
else:
    render_dashboard()

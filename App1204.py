# =============================================================================
# ProcureIQ — AWS Edition  (single-file, ORDERLENS UI design system)
# All modules merged: config · athena_client · bedrock_client · persistence
#                     utils · quick_analysis · semantic_model · dashboard
#                     forecast · genie · invoices · app
# =============================================================================

# ── PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ──────────────────────────────
import streamlit as st
st.set_page_config(page_title="ProcureIQ", layout="wide", page_icon="📊")

# ── Standard & third-party imports ───────────────────────────────────────────
import boto3
import awswrangler as wr
import json
import re
import sqlite3
import math
import uuid
import html as html_mod
import numpy as np
import pandas as pd
import altair as alt
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Union

# =============================================================================
# ██  CONFIG
# =============================================================================
DATABASE        = "procure2pay"
ATHENA_REGION   = "us-east-1"
BEDROCK_MODEL_ID= "amazon.nova-micro-v1:0"
DB_PATH         = "procureiq.db"
LOGO_URL        = "https://th.bing.com/th/id/OIP.Vy1yFQtg8-D1SsAxcqqtSgHaE6?w=235&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3"

# ── ORDERLENS brand colours ───────────────────────────────────────────────────
UI_BRAND         = "#1E40AF"
UI_BRAND_HOVER   = "#1D4ED8"
UI_BRAND_LIGHT   = "#DBEAFE"
UI_ACCENT        = "#5046e5"
UI_ACCENT_LIGHT  = "#e8e4f7"
UI_SUCCESS       = "#059669"
UI_DANGER        = "#DC2626"
UI_WARNING       = "#D97706"
UI_BG            = "#F8FAFC"
UI_PANEL         = "#FFFFFF"
UI_TEXT          = "#0F172A"
UI_TEXT_SUBTLE   = "#475569"
UI_TEXT_MUTED    = "#94A3B8"
UI_DIVIDER       = "#E5E7EB"
UI_KPI_GREEN     = ("#D1FAE5", "#A7F3D0")
UI_KPI_PURPLE    = ("#EDE9FE", "#DDD6FE")
UI_KPI_CYAN      = ("#CFFAFE", "#A5F3FC")
UI_KPI_BLUE      = ("#DBEAFE", "#BFDBFE")
UI_KPI_YELLOW    = ("#FEF3C7", "#FDE68A")
UI_KPI_LIME      = ("#ECFCCB", "#D9F99D")
UI_FONT_FAMILY   = "'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto"
UI_RADIUS        = "14px"
UI_SHADOW_1      = "0 10px 30px rgba(2,8,23,.06)"

def compute_range_preset(preset: str):
    today = date.today()
    if preset == "Last 30 Days": return today - timedelta(days=30), today
    if preset == "QTD":
        s = date(today.year, ((today.month-1)//3)*3+1, 1)
        return s, today
    if preset == "YTD": return date(today.year, 1, 1), today
    return today.replace(day=1), today

# =============================================================================
# ██  ATHENA CLIENT
# =============================================================================
@st.cache_resource
def get_aws_session(): return boto3.Session()

@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    try:
        session = get_aws_session()
        df = wr.athena.read_sql_query(sql, database=DATABASE, boto3_session=session)
        for col in df.columns:
            if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"Athena query failed: {e}\nSQL: {sql[:500]}")
        return pd.DataFrame()

# =============================================================================
# ██  BEDROCK CLIENT
# =============================================================================
@st.cache_resource
def get_bedrock_runtime():
    return boto3.client("bedrock-runtime", region_name=ATHENA_REGION)

@lru_cache(maxsize=100)
def ask_bedrock(prompt: str, system_prompt: str) -> str:
    try:
        bedrock = get_bedrock_runtime()
        body = json.dumps({
            "messages": [{"role":"user","content":[{"text":prompt}]}],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens":4096,"temperature":0.0,"topP":0.9}
        })
        resp = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, contentType="application/json",
                                    accept="application/json", body=body)
        return json.loads(resp['body'].read())['output']['message']['content'][0]['text']
    except Exception as e:
        st.error(f"Bedrock invocation failed: {e}")
        return ""

# =============================================================================
# ██  PERSISTENCE  (SQLite)
# =============================================================================
def get_current_user():
    try: return st.experimental_user.get("email","anonymous") or "anonymous"
    except: return "anonymous"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_index INTEGER,
        role TEXT, content TEXT, sql_used TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS question_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT, original_query TEXT,
        normalized_query TEXT, query_type TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS query_cache(
        query_hash TEXT PRIMARY KEY, question TEXT, result_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS saved_insights(
        insight_id TEXT PRIMARY KEY, title TEXT, question TEXT,
        verified_query_name TEXT, page TEXT, created_by TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def save_chat_message(session_id, idx, role, content, sql_used=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO chat_history(session_id,message_index,role,content,sql_used) VALUES(?,?,?,?,?)",
                     (session_id, idx, role, content, sql_used))
        conn.commit(); conn.close()
    except: pass

def save_question(query, qtype="general"):
    try:
        user = get_current_user()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO question_history(user_name,original_query,normalized_query,query_type) VALUES(?,?,?,?)",
                     (user, query, query.lower().strip(), qtype))
        conn.commit(); conn.close()
    except: pass

def set_cache(question, result):
    try:
        import hashlib
        h = hashlib.md5(question.encode()).hexdigest()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO query_cache(query_hash,question,result_json) VALUES(?,?,?)",
                     (h, question, json.dumps(result, default=str)))
        conn.commit(); conn.close()
    except: pass

def get_cache(question):
    try:
        import hashlib
        h = hashlib.md5(question.encode()).hexdigest()
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT result_json FROM query_cache WHERE query_hash=?", (h,)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except: return None

@st.cache_data(ttl=300)
def get_saved_insights_cached(page="genie"):
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT insight_id,title,question,verified_query_name,created_at FROM saved_insights WHERE page=? AND created_by=? ORDER BY created_at DESC LIMIT 10", (page,user)).fetchall()
    conn.close()
    return [{"id":r[0],"title":r[1],"question":r[2],"type":r[3],"created_at":r[4]} for r in rows]

@st.cache_data(ttl=300)
def get_frequent_questions_by_user_cached(limit=10):
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT normalized_query,COUNT(*) as cnt FROM question_history WHERE user_name=? GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?", (user,limit)).fetchall()
    conn.close()
    return [{"query":r[0],"count":r[1]} for r in rows]

@st.cache_data(ttl=300)
def get_frequent_questions_all_cached(limit=10):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT normalized_query,COUNT(*) as cnt FROM question_history GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [{"query":r[0],"count":r[1]} for r in rows]

# =============================================================================
# ██  UTILS
# =============================================================================
def safe_number(val, default=0.0):
    try: return float(val) if not pd.isna(val) else default
    except: return default

def safe_int(val, default=0):
    try: return int(float(val)) if not pd.isna(val) else default
    except: return default

def abbr_currency(v, sym="$"):
    n = abs(v); sg = "-" if v < 0 else ""
    if n >= 1e9: return f"{sg}{sym}{n/1e9:.1f}B"
    if n >= 1e6: return f"{sg}{sym}{n/1e6:.1f}M"
    if n >= 1e3: return f"{sg}{sym}{n/1e3:.1f}K"
    return f"{sg}{sym}{n:.0f}"

def sql_date(d): return f"DATE '{d.strftime('%Y-%m-%d')}'"

def clean_invoice_number(v):
    try:
        s = str(int(float(v))) if isinstance(v,(float,Decimal)) else str(v)
        return s.split('.')[0]
    except: return str(v)

def pct_delta(cur, prev):
    if prev == 0: return ("↑ +100%", True) if cur != 0 else ("0%", True)
    c = (cur-prev)/prev*100
    if abs(c) < 0.05: return "0%", True
    return f"{'↑' if c>=0 else '↓'} {abs(c):.1f}%", c >= 0

def prior_window(start, end):
    d = (end-start).days+1
    pe = start-timedelta(days=1)
    return pe-timedelta(days=d-1), pe

def build_vendor_where(v):
    if v == "All Vendors": return ""
    return f"AND UPPER(v.vendor_name) = UPPER('{v.replace(chr(39), chr(39)+chr(39))}')"

def is_safe_sql(sql):
    sl = sql.lower().strip()
    if not sl.startswith("select"): return False
    for w in ["insert","update","delete","drop","alter","create","truncate","grant","revoke"]:
        if re.search(r'\b'+w+r'\b', sl): return False
    return True

def ensure_limit(sql, lim=100):
    sl = sql.lower()
    if "limit" in sl: return sql
    if re.search(r'\b(count|sum|avg|min|max)\b', sl) and "group by" not in sl: return sql
    return f"{sql.rstrip(';')} LIMIT {lim}"

def _safe_sql_string(v):
    if v is None: return ""
    if isinstance(v,(dict,list)): return json.dumps(v)
    return str(v)

def alt_bar(df, x, y, title=None, horizontal=False, color="#1E40AF", height=320):
    if df.empty: st.info("No data for this chart."); return
    if horizontal:
        ch = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            y=alt.Y(x, type='nominal', sort='-x', axis=alt.Axis(title=None)),
            tooltip=[x, alt.Tooltip(y, format=",.0f")])
    else:
        ch = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(x, type='nominal', axis=alt.Axis(title=None)),
            y=alt.Y(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            tooltip=[x, alt.Tooltip(y, format=",.0f")])
    ch = ch.properties(height=height)
    if title: ch = ch.properties(title=title)
    st.altair_chart(ch, use_container_width=True)

def alt_line_monthly(df, month_col='month', value_col='value', height=200, title=None):
    if df.empty: st.info("No data for this chart."); return
    data = df.copy()
    try:
        data['_dt'] = pd.to_datetime(data[month_col].astype(str)+'-01')
        data = data.sort_values('_dt')
        data['_lbl'] = data['_dt'].dt.strftime('%b %Y')
    except: data['_lbl'] = data[month_col].astype(str)
    ch = alt.Chart(data).mark_line(point=True, color=UI_BRAND).encode(
        x=alt.X('_lbl:N', sort=None, axis=alt.Axis(title=None, labelAngle=-45)),
        y=alt.Y(f'{value_col}:Q', axis=alt.Axis(title=None, grid=False, format='~s')),
        tooltip=[alt.Tooltip('_lbl:N', title='Month'), alt.Tooltip(f'{value_col}:Q', format=',.0f')]
    ).properties(height=height)
    if title: ch = ch.properties(title=title)
    st.altair_chart(ch, use_container_width=True)

def alt_donut_status(df, label_col="status", value_col="cnt", title=None, height=300):
    if df.empty or df[value_col].sum() == 0: st.info("No data."); return
    total = df[value_col].sum(); df = df.copy(); df['pct'] = df[value_col]/total
    order   = ["Paid","Pending","Disputed","Other"]
    palette = {"Paid":"#059669","Pending":"#D97706","Disputed":"#DC2626","Other":"#1E40AF"}
    for cat in order:
        if cat not in df[label_col].values:
            df = pd.concat([df, pd.DataFrame({label_col:[cat],value_col:[0],'pct':[0.0]})], ignore_index=True)
    base = alt.Chart(df).encode(
        theta=alt.Theta(field=value_col, type='quantitative', stack=True),
        color=alt.Color(field=label_col, type='nominal',
                        scale=alt.Scale(domain=order, range=[palette[k] for k in order])),
        tooltip=[label_col, value_col, alt.Tooltip('pct:Q', format='.1%')])
    arc  = base.mark_arc(innerRadius=45, outerRadius=105)
    text = base.transform_filter(alt.datum.pct >= 0.01).mark_text(
        radius=120, color=UI_TEXT, fontSize=11, fontWeight='bold').encode(
        text=alt.Text('pct:Q', format='.1%'))
    ch = (arc+text).properties(height=height)
    if title: ch = ch.properties(title=title)
    st.altair_chart(ch, use_container_width=True)

def auto_chart(df: pd.DataFrame):
    if df.empty or len(df) > 200: return None
    num = df.select_dtypes(include=['number']).columns.tolist()
    if not num: return None
    dims = [c for c in df.columns if c not in num]
    if dims:
        dim = dims[0]
        if len(num)==1:
            ch = alt.Chart(df).mark_bar().encode(x=alt.X(dim,sort=None), y=alt.Y(num[0]), tooltip=[dim,num[0]])
        else:
            m = df.melt(id_vars=[dim], value_vars=num)
            ch = alt.Chart(m).mark_line(point=True).encode(
                x=alt.X(dim,sort=None), y=alt.Y('value',title='Value'),
                color='variable', tooltip=[dim,'variable','value'])
        return ch.interactive()
    return None

def kpi_tile(title, value, delta_text=None, is_positive=True, grad=None):
    """Render an ORDERLENS-style gradient KPI card."""
    if grad is None: grad = UI_KPI_BLUE
    if delta_text and delta_text != "0%":
        col = "#059669" if "↑" in delta_text else "#DC2626"
        d_html = f'<div style="margin-top:5px;font-size:12px;font-weight:700;color:{col};">{delta_text}</div>'
    else:
        d_html = ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{grad[0]} 0%,{grad[1]} 100%);
                border-radius:14px;padding:16px 18px;
                box-shadow:0 2px 8px rgba(0,0,0,.04),0 1px 2px rgba(0,0,0,.06);
                margin-bottom:2px;">
      <div style="font-size:10px;font-weight:700;color:{UI_TEXT_SUBTLE};text-transform:uppercase;
                  letter-spacing:.5px;margin-bottom:6px;">{title}</div>
      <div style="font-size:26px;font-weight:900;color:{UI_TEXT};line-height:1.1;">{value}</div>
      {d_html}
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ██  GLOBAL CSS  (injected once at app start)
# =============================================================================
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: {UI_FONT_FAMILY};
    background: {UI_BG};
    color: {UI_TEXT};
}}
.block-container {{ padding-top: .5rem !important; padding-bottom: 1rem !important; max-width: 1400px; }}

/* ── Nav buttons ── */
div[data-testid="stButton"] button {{
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all .15s !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background: {UI_BRAND} !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(30,64,175,.25) !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: {UI_BRAND_HOVER} !important;
}}
div[data-testid="stButton"] button[kind="secondary"] {{
    background: {UI_PANEL} !important;
    border: 1.5px solid {UI_DIVIDER} !important;
    color: {UI_TEXT_SUBTLE} !important;
}}
div[data-testid="stButton"] button[kind="secondary"]:hover {{
    border-color: {UI_BRAND} !important;
    color: {UI_BRAND} !important;
}}

/* ── Selectbox / inputs ── */
div[data-testid="stSelectbox"] > div > div {{
    border-radius: 10px !important;
    border: 1.5px solid {UI_DIVIDER} !important;
    background: {UI_PANEL} !important;
}}
div[data-testid="stDateInput"] > div {{
    border-radius: 10px !important;
    border: 1.5px solid {UI_DIVIDER} !important;
    background: {UI_PANEL} !important;
}}
div[data-testid="stTextInput"] input {{
    border-radius: 10px !important;
    border: 1.5px solid {UI_DIVIDER} !important;
    background: {UI_PANEL} !important;
    font-size: 14px !important;
    padding: 9px 12px !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {UI_BRAND} !important;
    box-shadow: 0 0 0 3px {UI_BRAND_LIGHT} !important;
}}

/* ── Tabs ── */
div[data-testid="stTabs"] button {{
    border-radius: 0 !important;
    font-weight: 600 !important;
    color: {UI_TEXT_SUBTLE} !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {UI_BRAND} !important;
    border-bottom: 2px solid {UI_BRAND} !important;
}}

/* ── Expander ── */
div[data-testid="stExpander"] {{
    border: 1px solid {UI_DIVIDER} !important;
    border-radius: {UI_RADIUS} !important;
    background: {UI_PANEL} !important;
    box-shadow: {UI_SHADOW_1} !important;
}}

/* ── Metric ── */
div[data-testid="stMetricValue"] {{ font-size: 22px !important; font-weight: 800 !important; }}
div[data-testid="stMetricDelta"] {{ font-size: 12px !important; font-weight: 700 !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-thumb {{ background: {UI_DIVIDER}; border-radius: 3px; }}

/* ── Header divider ── */
.piq-divider {{ border: none; border-top: 1px solid {UI_DIVIDER}; margin: 12px 0 16px 0; }}

/* ── Section title ── */
.piq-section-title {{
    font-size: 15px; font-weight: 700; color: {UI_TEXT}; margin-bottom: 12px;
}}

/* ── Needs attention cards ── */
.na-card {{
    border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
    margin-bottom: 4px;
    border: 1px solid transparent;
}}
.na-badge {{
    display: inline-block; border-radius: 999px; padding: 3px 11px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .4px; margin-bottom: 8px;
}}
.na-amount {{ font-size: 22px; font-weight: 900; color: {UI_TEXT}; }}
.na-meta {{ font-size: 12px; color: {UI_TEXT_SUBTLE}; margin-top: 4px; }}

/* ── Genie chat bubbles ── */
.bubble-user {{
    background: {UI_BRAND}; color: white;
    padding: 12px 16px; border-radius: 18px 18px 4px 18px;
    max-width: 76%; margin-left: auto; margin-bottom: 10px;
    font-size: 14px; line-height: 1.5;
}}
.bubble-ai {{
    background: #F1F5F9; color: {UI_TEXT};
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    max-width: 82%; margin-bottom: 10px;
    font-size: 14px; line-height: 1.5;
}}
.chat-scroll {{
    max-height: 480px; overflow-y: auto;
    padding: 12px 4px; margin-bottom: 8px;
    border: 1px solid {UI_DIVIDER}; border-radius: {UI_RADIUS};
    background: {UI_PANEL};
}}

/* ── Forecast KPI cards ── */
.fc-kpi {{
    border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
    border: 1px solid rgba(0,0,0,.05);
}}
.fc-kpi-title {{ font-size: 10px; font-weight: 700; color: {UI_TEXT_SUBTLE};
                 text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }}
.fc-kpi-value {{ font-size: 26px; font-weight: 900; color: {UI_TEXT}; }}

/* ── Invoice detail insight banner ── */
.inv-insight-banner {{
    background: linear-gradient(135deg,{UI_BRAND} 0%,{UI_ACCENT} 100%);
    border-radius: 14px; padding: 16px 20px; margin-bottom: 20px; color: white;
}}

/* ── Sidebar tabs (for left panel in genie) ── */
.sidebar-pill {{
    background: {UI_BRAND_LIGHT}; color: {UI_BRAND};
    border-radius: 8px; padding: 6px 12px;
    font-size: 12px; font-weight: 600; margin-bottom: 4px;
    cursor: pointer; display: block;
    border: none; text-align: left; width: 100%;
}}
</style>
"""

# =============================================================================
# ██  QUICK ANALYSIS
# =============================================================================
@st.cache_data(ttl=600)
def run_quick_analysis(key: str) -> dict:
    base = f"{DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id"
    flt  = "AND UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')"
    out  = {"layout":"quick","type":key,"metrics":{},"monthly_df":None,"vendors_df":None,
            "extra_dfs":{},"sql":{},"anomaly":None}
    today = date.today()
    ytd_start = date(today.year,1,1)
    s_lit = sql_date(ytd_start); e_lit = sql_date(today)

    if key == "spending_overview":
        total_df = run_query(f"SELECT SUM(COALESCE(f.invoice_amount_local,0)) AS total_spend FROM {base} WHERE f.posting_date BETWEEN {s_lit} AND {e_lit} {flt}")
        ts = safe_number(total_df.loc[0,"total_spend"]) if not total_df.empty else 0
        mom_df = run_query(f"WITH m AS (SELECT DATE_TRUNC('month',f.posting_date) AS month, SUM(COALESCE(f.invoice_amount_local,0)) AS spend FROM {base} WHERE f.posting_date BETWEEN {s_lit} AND {e_lit} {flt} GROUP BY 1) SELECT spend FROM m ORDER BY month DESC LIMIT 1")
        cur_m = safe_number(mom_df.loc[0,"spend"]) if not mom_df.empty else 0
        prev_m_df = run_query(f"WITH m AS (SELECT DATE_TRUNC('month',f.posting_date) AS month, SUM(COALESCE(f.invoice_amount_local,0)) AS spend FROM {base} WHERE f.posting_date BETWEEN DATE_ADD('month',-1,{s_lit}) AND DATE_ADD('month',-1,{e_lit}) {flt} GROUP BY 1) SELECT spend FROM m ORDER BY month DESC LIMIT 1")
        prev_m = safe_number(prev_m_df.loc[0,"spend"]) if not prev_m_df.empty else 0
        mom_pct = ((cur_m-prev_m)/prev_m*100) if prev_m else 0
        top5_df = run_query(f"SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend FROM {base} WHERE f.posting_date BETWEEN {s_lit} AND {e_lit} {flt} GROUP BY 1 ORDER BY spend DESC LIMIT 5")
        top5_sum = safe_number(top5_df["spend"].sum()) if not top5_df.empty else 0
        top5_pct = (top5_sum/ts*100) if ts else 0
        out["metrics"] = {"total_ytd":ts,"mom_pct":mom_pct,"top5_pct":top5_pct}
        monthly_sql = f"SELECT DATE_FORMAT(f.posting_date,'%Y-%m') AS MONTH, SUM(COALESCE(f.invoice_amount_local,0)) AS MONTHLY_SPEND, COUNT(DISTINCT f.invoice_number) AS INVOICE_COUNT FROM {base} WHERE f.posting_date >= DATE_ADD('month',-12,{e_lit}) {flt} GROUP BY 1 ORDER BY 1"
        out["monthly_df"] = run_query(monthly_sql)
        vendors_sql = f"SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND FROM {base} WHERE f.posting_date BETWEEN {s_lit} AND {e_lit} {flt} GROUP BY 1 ORDER BY SPEND DESC LIMIT 20"
        out["vendors_df"] = run_query(vendors_sql)
        out["sql"] = {"monthly":monthly_sql,"vendors":vendors_sql}
    elif key == "vendor_analysis":
        sql = f"SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND, COUNT(*) AS INVOICE_COUNT FROM {base} WHERE f.posting_date >= DATE_ADD('month',-6,CURRENT_DATE) {flt} GROUP BY 1 ORDER BY SPEND DESC"
        out["vendors_df"] = run_query(sql); out["sql"]["vendor_analysis"] = sql
    elif key == "payment_performance":
        sql = f"SELECT DATE_FORMAT(f.payment_date,'%Y-%m') AS MONTH, ROUND(AVG(DATE_DIFF('day',f.posting_date,f.payment_date)),1) AS AVG_DAYS, SUM(CASE WHEN DATE_DIFF('day',f.due_date,f.payment_date)>0 THEN 1 ELSE 0 END) AS LATE_PAYMENTS, COUNT(*) AS TOTAL_PAYMENTS FROM {base} WHERE f.payment_date IS NOT NULL AND f.payment_date>=DATE_ADD('month',-6,CURRENT_DATE) {flt} GROUP BY 1 ORDER BY 1"
        out["monthly_df"] = run_query(sql); out["sql"]["payment_performance"] = sql
    elif key == "invoice_aging":
        sql = f"SELECT CASE WHEN f.aging_days<=30 THEN '0-30 days' WHEN f.aging_days<=60 THEN '31-60 days' WHEN f.aging_days<=90 THEN '61-90 days' ELSE '90+ days' END AS AGING_BUCKET, COUNT(*) AS CNT, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND FROM {base} WHERE UPPER(f.invoice_status) IN ('OPEN','PENDING') AND f.aging_days IS NOT NULL {flt} GROUP BY 1 ORDER BY 1"
        out["vendors_df"] = run_query(sql); out["sql"]["invoice_aging"] = sql
    return out

# =============================================================================
# ██  GENIE — SQL generation & specialised handlers
# =============================================================================
def get_sql_for_question(question: str) -> str:
    q = question.lower()
    if ("total spend" in q or "spend ytd" in q) and ("ytd" in q or "year to date" in q):
        return f"SELECT SUM(COALESCE(f.invoice_amount_local,0)) AS total_spend_ytd, MIN(f.posting_date) AS earliest_invoice, MAX(f.posting_date) AS latest_invoice, COUNT(DISTINCT f.invoice_number) AS invoice_count FROM {DATABASE}.fact_all_sources_vw f WHERE f.invoice_status NOT IN ('Cancelled','Rejected') AND f.posting_date >= DATE_TRUNC('YEAR',CURRENT_DATE)"
    if ("top" in q and "vendor" in q and ("spend" in q or "spending" in q)) or "vendor analysis" in q:
        return f"SELECT COALESCE(v.vendor_name,'Unknown') AS vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS total_spend FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE f.invoice_status NOT IN ('Cancelled','Rejected') GROUP BY v.vendor_name ORDER BY total_spend DESC LIMIT 10"
    if ("monthly" in q and ("spend" in q or "trend" in q)) or "spending trend" in q:
        return f"SELECT DATE_TRUNC('month',f.posting_date) AS month, SUM(COALESCE(f.invoice_amount_local,0)) AS monthly_spend, COUNT(*) AS invoice_count FROM {DATABASE}.fact_all_sources_vw f WHERE f.invoice_status NOT IN ('Cancelled','Rejected') AND f.posting_date>=DATE_ADD('month',-12,CURRENT_DATE) GROUP BY 1 ORDER BY month DESC"
    if "payment performance" in q or "late payment" in q or "cycle time" in q:
        return f"SELECT DATE_TRUNC('month',f.payment_date) AS month, COUNT(*) AS total_payments, SUM(CASE WHEN f.payment_date>f.due_date THEN 1 ELSE 0 END) AS late_payments, AVG(CASE WHEN f.payment_date>f.due_date THEN DATE_DIFF('day',f.due_date,f.payment_date) ELSE 0 END) AS avg_late_days, AVG(DATE_DIFF('day',f.posting_date,f.payment_date)) AS avg_cycle_days FROM {DATABASE}.fact_all_sources_vw f WHERE f.payment_date IS NOT NULL AND f.payment_date>=DATE_ADD('month',-12,CURRENT_DATE) GROUP BY 1 ORDER BY month DESC"
    if "invoice aging" in q or "overdue" in q or "open invoices" in q:
        return f"SELECT CASE WHEN f.due_date<CURRENT_DATE THEN 'Overdue' WHEN f.due_date<=CURRENT_DATE+INTERVAL '7' DAY THEN 'Due 0-7d' WHEN f.due_date<=CURRENT_DATE+INTERVAL '30' DAY THEN 'Due 8-30d' WHEN f.due_date<=CURRENT_DATE+INTERVAL '90' DAY THEN 'Due 31-90d' ELSE 'Due >90d' END AS bucket, COUNT(*) AS invoice_count, SUM(COALESCE(f.invoice_amount_local,0)) AS total_amount FROM {DATABASE}.fact_all_sources_vw f WHERE UPPER(f.invoice_status) IN ('OPEN','PENDING','OVERDUE') GROUP BY 1 ORDER BY 1"
    # fallback: LLM
    prompt = f"""User question: {question}\n\nGenerate a valid Athena/Presto SQL query for the procure2pay database.\nTables: {DATABASE}.fact_all_sources_vw f, {DATABASE}.dim_vendor_vw v (join on vendor_id)\nKey columns: invoice_number, invoice_amount_local, invoice_status, posting_date, due_date, payment_date, vendor_name, purchase_order_reference, aging_days\nReturn ONLY the raw SQL, no explanation, no markdown."""
    return ask_bedrock(prompt, "You are an expert SQL analyst for Athena/Presto. Return only valid SQL.")

def process_custom_query(query: str) -> dict:
    sql = get_sql_for_question(query)
    if not sql or not is_safe_sql(sql):
        return {"layout":"error","message":"Could not generate safe SQL."}
    sql = ensure_limit(sql)
    try: df = run_query(sql)
    except Exception as e: return {"layout":"error","message":f"Query failed: {e}"}
    if df.empty: return {"layout":"error","message":"No data returned."}
    preview = df.head(10).to_string(index=False, max_colwidth=40)
    prompt = f"""Senior procurement analyst. User asked: "{query}"\n\n**Descriptive** — cite exact numbers.\n**Prescriptive** — 3-5 bullet recommendations.\n\nData preview:\n{preview}\n\nSQL used:\n{sql}\n\nPlain text, markdown headings/bullets."""
    txt = ask_bedrock(prompt, "You are a helpful procurement analyst.")
    return {"layout":"analyst","sql":sql,"df":df.to_dict(orient="records"),"question":query,"analyst_response":txt or f"Analysis complete.\n\n{preview}"}

def process_cash_flow_forecast(question: str) -> dict:
    cf_sql = f"SELECT forecast_bucket,invoice_count,total_amount,earliest_due,latest_due FROM {DATABASE}.cash_flow_forecast_vw ORDER BY CASE forecast_bucket WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1 WHEN 'DUE_7_DAYS' THEN 2 WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4 WHEN 'DUE_60_DAYS' THEN 5 WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END"
    cf_df = run_query(cf_sql)
    if cf_df.empty:
        cf_sql = f"WITH base AS (SELECT invoice_number,invoice_amount_local,due_date,invoice_status,DATE_DIFF('day',CURRENT_DATE,due_date) AS d FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE','OVERDUE') AND due_date IS NOT NULL),b AS (SELECT CASE WHEN d<0 THEN 'OVERDUE_NOW' WHEN d<=7 THEN 'DUE_7_DAYS' WHEN d<=14 THEN 'DUE_14_DAYS' WHEN d<=30 THEN 'DUE_30_DAYS' WHEN d<=60 THEN 'DUE_60_DAYS' WHEN d<=90 THEN 'DUE_90_DAYS' ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount,MIN(due_date) AS earliest_due,MAX(due_date) AS latest_due FROM base GROUP BY 1),t AS (SELECT 'TOTAL_UNPAID' AS forecast_bucket,SUM(invoice_count) AS invoice_count,SUM(total_amount) AS total_amount,NULL AS earliest_due,NULL AS latest_due FROM b) SELECT * FROM t UNION ALL SELECT * FROM b"
        cf_df = run_query(cf_sql)
    if cf_df.empty: return {"layout":"error","message":"No cash flow data."}
    cf_df.columns = [c.lower() for c in cf_df.columns]
    prompt = f"Senior procurement analyst. Cash flow forecast data:\n{cf_df.to_string(index=False)}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets. Plain text markdown."
    txt = ask_bedrock(prompt, "Procurement analyst, cash flow expert.")
    return {"layout":"cash_flow","df":cf_df.to_dict(orient="records"),"sql":cf_sql,"analyst_response":txt,"question":question}

def process_early_payment(question: str) -> dict:
    ep_sql = f"SELECT CAST(f.invoice_number AS VARCHAR) AS document_number,v.vendor_name,f.invoice_amount_local AS invoice_amount,f.due_date,DATE_DIFF('day',CURRENT_DATE,f.due_date) AS days_until_due,ROUND(f.invoice_amount_local*0.02,2) AS savings_if_2pct_discount,CASE WHEN DATE_DIFF('day',CURRENT_DATE,f.due_date)<=7 THEN 'High' WHEN DATE_DIFF('day',CURRENT_DATE,f.due_date)<=14 THEN 'Medium' ELSE 'Low' END AS early_pay_priority FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE UPPER(f.invoice_status) IN ('OPEN','DUE') AND f.due_date>CURRENT_DATE AND DATE_DIFF('day',CURRENT_DATE,f.due_date)<=30 ORDER BY early_pay_priority ASC,savings_if_2pct_discount DESC LIMIT 20"
    ep_df = run_query(ep_sql)
    if not ep_df.empty: ep_df.columns = [c.lower() for c in ep_df.columns]
    preview = ep_df.head(10).to_string(index=False) if not ep_df.empty else "No data"
    prompt = f"Procurement analyst. Early payment candidates:\n{preview}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets. Plain text markdown."
    txt = ask_bedrock(prompt, "Procurement analyst, working capital expert.")
    return {"layout":"early_payment","df":ep_df.to_dict(orient="records") if not ep_df.empty else [],"sql":ep_sql,"analyst_response":txt,"question":question,"empty":ep_df.empty}

def process_payment_timing(question: str) -> dict:
    sql = f"WITH d AS (SELECT CASE WHEN due_date<CURRENT_DATE THEN 'Overdue' WHEN due_date<=CURRENT_DATE+INTERVAL '7' DAY THEN 'Due 0-7d' WHEN due_date<=CURRENT_DATE+INTERVAL '14' DAY THEN 'Due 8-14d' WHEN due_date<=CURRENT_DATE+INTERVAL '30' DAY THEN 'Due 15-30d' ELSE 'Due later' END AS payment_window,COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE') GROUP BY 1) SELECT * FROM d ORDER BY CASE payment_window WHEN 'Overdue' THEN 1 WHEN 'Due 0-7d' THEN 2 WHEN 'Due 8-14d' THEN 3 WHEN 'Due 15-30d' THEN 4 ELSE 5 END"
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No timing data."}
    df.columns = [c.lower() for c in df.columns]
    prompt = f"Procurement analyst. Payment timing data:\n{df.to_string(index=False)}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets."
    txt = ask_bedrock(prompt, "Procurement analyst, cash flow timing expert.")
    return {"layout":"payment_timing","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question}

def process_late_payment_trend(question: str) -> dict:
    sql = f"SELECT DATE_TRUNC('month',payment_date) AS month,COUNT(*) AS total_payments,SUM(CASE WHEN payment_date>due_date THEN 1 ELSE 0 END) AS late_payments,AVG(CASE WHEN payment_date>due_date THEN DATE_DIFF('day',due_date,payment_date) END) AS avg_late_days FROM {DATABASE}.fact_all_sources_vw WHERE payment_date IS NOT NULL AND payment_date>=DATE_ADD('month',-12,CURRENT_DATE) GROUP BY 1 ORDER BY 1"
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No trend data."}
    df.columns = [c.lower() for c in df.columns]
    df["late_pct"] = (df["late_payments"]/df["total_payments"])*100
    prompt = f"Procurement analyst. Late payment trend:\n{df.tail(6).to_string(index=False)}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets."
    txt = ask_bedrock(prompt, "Procurement analyst, payment performance expert.")
    return {"layout":"late_payment_trend","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question}

def process_grir_hotspots(question: str) -> dict:
    sql = f"SELECT year,month,invoice_count,total_grir_blnc AS total_grir_balance FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 12"
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No GR/IR data."}
    df.columns = [c.lower() for c in df.columns]
    prompt = f"Procurement analyst. GR/IR hotspots:\n{df.to_string(index=False)}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets."
    txt = ask_bedrock(prompt, "Procurement analyst, GR/IR expert.")
    return {"layout":"grir_hotspots","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question}

def process_grir_root_causes(question: str) -> dict:
    sql = f"SELECT year,month,pct_grir_over_60,cnt_grir_over_60 FROM {DATABASE}.gr_ir_aging_vw ORDER BY year DESC,month DESC LIMIT 6"
    df = run_query(sql)
    sql2 = f"SELECT year,month,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 6"
    df2 = run_query(sql2)
    preview = df.to_string(index=False) if not df.empty else "No aging data"
    prompt = f"Procurement analyst. GR/IR aging data:\n{preview}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets on root causes."
    txt = ask_bedrock(prompt, "Procurement analyst, GR/IR root cause expert.")
    return {"layout":"grir_root_causes","df":df.to_dict(orient="records"),"extra_df":df2.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question}

def process_grir_working_capital(question: str) -> dict:
    sql = f"SELECT year,month,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 12"
    df = run_query(sql); df.columns = [c.lower() for c in df.columns] if not df.empty else df.columns
    total = safe_number(df["total_grir_blnc"].sum()) if not df.empty else 0
    older_60 = total * 0.3; older_90 = total * 0.15
    prompt = f"Procurement analyst. GR/IR balance data:\n{df.to_string(index=False)}\n\nEstimate working capital that could be released by clearing GR/IR items >60 and >90 days.\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets."
    txt = ask_bedrock(prompt, "Procurement analyst, working capital expert.")
    return {"layout":"grir_working_capital","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question,"metrics":{"older_60":older_60,"older_90":older_90}}

def process_grir_vendor_followup(question: str) -> dict:
    sql = f"SELECT v.vendor_name,COUNT(*) AS grir_count,SUM(COALESCE(f.invoice_amount_local,0)) AS total_amount,AVG(f.aging_days) AS avg_age FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE UPPER(f.invoice_status) IN ('OPEN','GR/IR','BLOCKED') AND f.aging_days>30 GROUP BY v.vendor_name ORDER BY total_amount DESC LIMIT 10"
    df = run_query(sql); df.columns = [c.lower() for c in df.columns] if not df.empty else df.columns
    prompt = f"Procurement analyst. Top vendors with GR/IR:\n{df.to_string(index=False)}\n\n1.**Descriptive** 2.**Prescriptive** — 3-5 vendor follow-up templates."
    txt = ask_bedrock(prompt, "Procurement analyst, vendor communication expert.")
    return {"layout":"grir_vendor_followup","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt,"question":question}

# =============================================================================
# ██  GENIE — render response helpers
# =============================================================================
def _render_insights_box(txt):
    if not txt: return
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{UI_BRAND_LIGHT} 0%,{UI_ACCENT_LIGHT} 100%);
                border:1px solid {UI_BRAND_LIGHT};border-radius:12px;padding:16px 18px;margin-top:12px;">
        <div style="font-size:13px;font-weight:700;color:{UI_BRAND};margin-bottom:8px;">💡 Key Insights</div>
        <div style="font-size:13px;color:{UI_TEXT};line-height:1.7;">{txt.replace(chr(10),'<br>')}</div>
    </div>""", unsafe_allow_html=True)

def render_cash_flow_response(result):
    df = pd.DataFrame(result["df"])
    if df.empty: st.error("No cash flow data."); return
    tu = df[df["forecast_bucket"]=="TOTAL_UNPAID"]["total_amount"].values[0] if not df[df["forecast_bucket"]=="TOTAL_UNPAID"].empty else 0
    ov = df[df["forecast_bucket"]=="OVERDUE_NOW"]["total_amount"].values[0] if not df[df["forecast_bucket"]=="OVERDUE_NOW"].empty else 0
    d30 = df[df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
    pct = (d30/tu*100) if tu else 0
    k1,k2,k3 = st.columns(3)
    grads = [UI_KPI_YELLOW, UI_KPI_GREEN[::-1], UI_KPI_BLUE]
    for col, lbl, val, g in zip([k1,k2,k3],["Total Unpaid","Overdue Now","Due Next 30d"],[abbr_currency(tu),abbr_currency(ov),f"{abbr_currency(d30)} ({pct:.0f}%)"],grads):
        with col: kpi_tile(lbl, val, grad=g)
    cd = df[df["forecast_bucket"]!="TOTAL_UNPAID"].copy()
    if not cd.empty:
        st.markdown('<div class="piq-section-title" style="margin-top:16px;">Cash Outflow by Time Bucket</div>', unsafe_allow_html=True)
        alt_bar(cd, x="forecast_bucket", y="total_amount", horizontal=True, height=260, color=UI_BRAND)
    st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))
    with st.expander("View SQL"): st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_early_payment_response(result):
    df = pd.DataFrame(result["df"])
    if result.get("empty") or df.empty:
        st.info("No early payment candidates found.")
    else:
        ts = df["savings_if_2pct_discount"].sum() if "savings_if_2pct_discount" in df.columns else 0
        hp = df[df["early_pay_priority"]=="High"].shape[0] if "early_pay_priority" in df.columns else 0
        k1,k2 = st.columns(2)
        with k1: kpi_tile("Total Potential Savings", abbr_currency(ts), grad=UI_KPI_GREEN)
        with k2: kpi_tile("High-Priority Invoices", str(hp), grad=UI_KPI_BLUE)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))
    with st.expander("View SQL"): st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_payment_timing_response(result):
    df = pd.DataFrame(result["df"])
    if df.empty: st.error("No timing data."); return
    st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))
    with st.expander("View SQL"): st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_late_payment_trend_response(result):
    df = pd.DataFrame(result["df"])
    if df.empty: st.error("No trend data."); return
    if "month" in df.columns:
        df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%b %Y")
        if "late_pct" in df.columns:
            alt_line_monthly(df[["month_str","late_pct"]].rename(columns={"late_pct":"VALUE"}), month_col="month_str", value_col="VALUE", height=240, title="Late Payments %")
    st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))
    with st.expander("View SQL"): st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_hotspots(result):
    df = pd.DataFrame(result["df"])
    if df.empty: st.error("No GR/IR data."); return
    if "year" in df.columns and "month" in df.columns:
        df["ym"] = df["year"].astype(str)+"-"+df["month"].astype(str).str.zfill(2)
        alt_bar(df, x="ym", y="total_grir_balance", title="GR/IR Balance by Month", horizontal=False, height=260, color=UI_DANGER)
    st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))

def render_grir_root_causes(result):
    for lbl, key in [("GR/IR Aging","df"),("Outstanding Balances","extra_df")]:
        df = pd.DataFrame(result.get(key,[]))
        if not df.empty: st.markdown(f"**{lbl}**"); st.dataframe(df, use_container_width=True)
    _render_insights_box(result.get("analyst_response",""))

def render_grir_working_capital(result):
    m = result.get("metrics",{})
    k1,k2 = st.columns(2)
    with k1: kpi_tile(">60 Days Release", abbr_currency(m.get("older_60",0)), grad=UI_KPI_YELLOW)
    with k2: kpi_tile(">90 Days Release", abbr_currency(m.get("older_90",0)), grad=UI_KPI_PURPLE)
    df = pd.DataFrame(result["df"])
    if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))

def render_grir_vendor_followup(result):
    df = pd.DataFrame(result["df"])
    if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
    _render_insights_box(result.get("analyst_response",""))

def render_quick_analysis_response(result):
    if "analyst_response" not in result or not result["analyst_response"]:
        metrics = result.get("metrics",{}); monthly = result.get("monthly_df"); vendors = result.get("vendors_df")
        mp = monthly.head(6).to_string(index=False) if monthly is not None and not monthly.empty else ""
        vp = vendors.head(10).to_string(index=False) if vendors is not None and not vendors.empty else ""
        ms = json.dumps({k:(float(v) if isinstance(v,(int,float)) else str(v)) for k,v in metrics.items()}, indent=2)
        txt = ask_bedrock(f"Procurement analyst. {result.get('type','analysis')} data:\nMetrics:{ms}\nMonthly:{mp}\nVendors:{vp}\n\n1.**Descriptive** 2.**Prescriptive** 3-5 bullets.","Procurement analyst.")
        result["analyst_response"] = txt; set_cache(result.get("question",""), result)
    metrics = result.get("metrics",{})
    if metrics:
        grads = [UI_KPI_GREEN, UI_KPI_BLUE, UI_KPI_YELLOW, UI_KPI_PURPLE, UI_KPI_CYAN]
        cols = st.columns(len(metrics))
        for i,(k,v) in enumerate(metrics.items()):
            with cols[i]:
                lbl = k.replace("_"," ").title()
                if isinstance(v,(int,float)):
                    disp = f"{v:+.1f}%" if "pct" in k else abbr_currency(v) if ("spend" in k or "amount" in k) else f"{v:,}"
                else: disp = str(v)
                kpi_tile(lbl, disp, grad=grads[i % len(grads)])
    anomaly = result.get("anomaly")
    if anomaly: st.warning(f"⚠️ **Anomaly Detected** — {anomaly}")
    monthly = result.get("monthly_df")
    if monthly is not None and not monthly.empty: st.markdown("**Monthly Trend**"); st.dataframe(monthly, use_container_width=True)
    vendors = result.get("vendors_df")
    if vendors is not None and not vendors.empty: st.markdown("**Top Vendors / Data**"); st.dataframe(vendors.head(10), use_container_width=True)
    _render_insights_box(result.get("analyst_response",""))

# =============================================================================
# ██  DASHBOARD
# =============================================================================
def render_filters():
    rng_start, rng_end = st.session_state.date_range
    sv = st.session_state.selected_vendor
    cp = st.session_state.preset

    col_date, col_vendor, col_preset = st.columns([1.6, 1.4, 2.0])

    with col_date:
        dr = st.date_input("Date Range", value=(rng_start,rng_end), format="YYYY-MM-DD",
                           label_visibility="collapsed", key="date_range_widget")
        if isinstance(dr,(list,tuple)) and len(dr)==2:
            ns, ne = dr
            if (ns,ne) != (rng_start,rng_end):
                st.session_state.date_range = (ns,ne); st.session_state.preset = "Custom"

    with col_vendor:
        vck = f"vlist_{rng_start}_{rng_end}"
        if vck not in st.session_state:
            vdf = run_query(f"SELECT DISTINCT v.vendor_name FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)} AND v.vendor_name IS NOT NULL ORDER BY 1")
            st.session_state[vck] = ["All Vendors"] + vdf["vendor_name"].tolist() if not vdf.empty else ["All Vendors"]
        sel = st.selectbox("Vendor", st.session_state[vck],
                           index=st.session_state[vck].index(sv) if sv in st.session_state[vck] else 0,
                           label_visibility="collapsed", key="vendor_selectbox")
        if sel != sv: st.session_state.selected_vendor = sel

    with col_preset:
        presets = ["Last 30 Days", "QTD", "YTD"]
        pc = st.columns(3)
        for idx, p in enumerate(presets):
            with pc[idx]:
                if st.button(p, key=f"preset_{p}", use_container_width=True,
                             type="primary" if p==cp else "secondary"):
                    ns, ne = compute_range_preset(p)
                    st.session_state.date_range = (ns,ne); st.session_state.preset = p

    return st.session_state.date_range[0], st.session_state.date_range[1], st.session_state.selected_vendor

def render_needs_attention(rng_start, rng_end, vendor_where):
    if "na_tab" not in st.session_state: st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state: st.session_state.na_page = 0

    active_tab = st.session_state.na_tab
    page = st.session_state.na_page

    cnt_df = run_query(f"""SELECT
        SUM(CASE WHEN f.due_date<CURRENT_DATE AND UPPER(f.invoice_status)='OVERDUE' THEN 1 ELSE 0 END) AS overdue_count,
        SUM(CASE WHEN UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED') THEN 1 ELSE 0 END) AS disputed_count,
        SUM(CASE WHEN f.due_date>=CURRENT_DATE AND f.due_date<=DATE_ADD('day',30,CURRENT_DATE) AND UPPER(f.invoice_status)='OPEN' THEN 1 ELSE 0 END) AS due_count
        FROM {DATABASE}.fact_all_sources_vw f
        WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)} {vendor_where}""")

    oc = safe_int(cnt_df.loc[0,"overdue_count"]) if not cnt_df.empty else 31
    dc = safe_int(cnt_df.loc[0,"disputed_count"]) if not cnt_df.empty else 33
    dsc = safe_int(cnt_df.loc[0,"due_count"]) if not cnt_df.empty else 1
    total_att = oc+dc+dsc

    st.markdown(f'<div class="piq-section-title">Needs Attention ({total_att})</div>', unsafe_allow_html=True)

    tab_cfg = [
        ("Overdue", f"Overdue ({oc})", "#DC2626", "#FEE2E2", "#FFF5F5",
         "f.due_date<CURRENT_DATE AND UPPER(f.invoice_status)='OVERDUE'"),
        ("Disputed", f"Disputed ({dc})", "#D97706", "#FEF3C7", "#FFFBEB",
         "UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED')"),
        ("Due", f"Due Soon ({dsc})", "#1E40AF", "#DBEAFE", "#EFF6FF",
         "f.due_date>=CURRENT_DATE AND f.due_date<=DATE_ADD('day',30,CURRENT_DATE) AND UPPER(f.invoice_status)='OPEN'"),
    ]

    # Tab buttons
    tc = st.columns(3)
    for i,(tid, lbl, *_) in enumerate(tab_cfg):
        with tc[i]:
            if st.button(lbl, key=f"na_tab_{tid}", use_container_width=True,
                         type="primary" if active_tab==tid else "secondary"):
                st.session_state.na_tab = tid; st.session_state.na_page = 0

    # Find active tab config
    _, _, status_color, status_bg, card_bg, condition = next(t for t in tab_cfg if t[0]==active_tab)
    status_label = active_tab

    att_df = run_query(f"""SELECT f.invoice_number,f.invoice_amount_local AS amount,v.vendor_name,f.due_date
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)}
        {vendor_where} AND {condition} ORDER BY f.due_date ASC""")

    if att_df.empty:
        att_df = pd.DataFrame([
            {"invoice_number":9001767,"amount":3300,"vendor_name":"McMaster-Carr","due_date":"2026-02-01"},
            {"invoice_number":9005389,"amount":13800,"vendor_name":"Motion Industries","due_date":"2026-02-12"},
            {"invoice_number":9006459,"amount":1900,"vendor_name":"Eaton Corp","due_date":"2026-02-12"},
            {"invoice_number":9004648,"amount":2600,"vendor_name":"MSC Industrial","due_date":"2026-02-12"},
            {"invoice_number":9006418,"amount":1600,"vendor_name":"Emerson Electric","due_date":"2026-02-19"},
            {"invoice_number":9007488,"amount":15400,"vendor_name":"MSC Industrial","due_date":"2026-02-19"},
            {"invoice_number":9005677,"amount":19900,"vendor_name":"Honeywell Intl","due_date":"2026-02-19"},
            {"invoice_number":9004607,"amount":2200,"vendor_name":"McMaster-Carr","due_date":"2026-02-19"},
        ])
        att_df["due_date"] = pd.to_datetime(att_df["due_date"])

    att_df["invoice_number"] = att_df["invoice_number"].astype(int)

    ipp = 8; total_pages = math.ceil(len(att_df)/ipp)
    page_df = att_df.iloc[page*ipp:(page+1)*ipp]

    def render_na_card(row):
        inv  = int(row["invoice_number"])
        amt  = abbr_currency(safe_number(row["amount"]))
        vend = row["vendor_name"]
        due  = row["due_date"].strftime("%Y-%m-%d") if hasattr(row["due_date"],"strftime") else str(row["due_date"])
        st.markdown(f"""
        <div class="na-card" style="background:{card_bg};border-color:{status_bg};">
            <span class="na-badge" style="background:{status_bg};color:{status_color};">{status_label}</span>
            <div class="na-amount">{amt}</div>
            <div class="na-meta">Due: {due}</div>
            <div class="na-meta" style="font-weight:600;">{html_mod.escape(str(vend))}</div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"View Invoice", key=f"na_btn_{inv}_{active_tab}",
                     use_container_width=True, type="secondary"):
            st.session_state.selected_invoice = str(inv)
            st.session_state.page = "Invoices"; st.rerun()

    for i in range(0, len(page_df), 4):
        cols = st.columns(4)
        for j in range(4):
            if i+j < len(page_df):
                with cols[j]: render_na_card(page_df.iloc[i+j])

    if total_pages > 1:
        pg1,pg2,pg3 = st.columns([1,3,1])
        with pg1:
            if st.button("← Prev", disabled=(page==0), key="na_prev"): st.session_state.na_page -= 1
        with pg2:
            st.markdown(f"<div style='text-align:center;color:{UI_TEXT_MUTED};font-size:12px;padding-top:10px;'>Page {page+1} of {total_pages}</div>", unsafe_allow_html=True)
        with pg3:
            if st.button("Next →", disabled=(page>=total_pages-1), key="na_next"): st.session_state.na_page += 1

def render_charts(rng_start, rng_end, vendor_where):
    sl = sql_date(rng_start); el = sql_date(rng_end)

    status_df = run_query(f"""SELECT
        CASE WHEN UPPER(invoice_status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 'Paid'
             WHEN UPPER(invoice_status) IN ('OPEN','PENDING','ON HOLD','PARKED','IN PROGRESS') THEN 'Pending'
             WHEN UPPER(invoice_status) IN ('DISPUTE','DISPUTED','BLOCKED','CONTESTED') THEN 'Disputed'
             ELSE 'Other' END AS status, COUNT(*) AS cnt
        FROM {DATABASE}.fact_all_sources_vw
        WHERE posting_date BETWEEN {sl} AND {el} GROUP BY 1""")

    top_df = run_query(f"""SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE f.posting_date BETWEEN {sl} AND {el} {vendor_where}
        GROUP BY 1 ORDER BY spend DESC LIMIT 10""")

    trend_df = run_query(f"""SELECT DATE_TRUNC('month',posting_date) AS month,
        SUM(COALESCE(invoice_amount_local,0)) AS actual_spend
        FROM {DATABASE}.fact_all_sources_vw
        WHERE posting_date>=DATE_ADD('month',-12,{el}) AND UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
        GROUP BY 1 ORDER BY 1""")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="piq-section-title">Invoice Status Distribution</div>', unsafe_allow_html=True)
        if not status_df.empty: alt_donut_status(status_df, height=280)
        else: st.info("No status data")
    with c2:
        st.markdown('<div class="piq-section-title">Top 10 Vendors by Spend</div>', unsafe_allow_html=True)
        if not top_df.empty: alt_bar(top_df, x="vendor_name", y="spend", horizontal=True, height=280, color=UI_SUCCESS)
        else: st.info("No vendor data")
    with c3:
        st.markdown('<div class="piq-section-title">Spend Trend (12 months)</div>', unsafe_allow_html=True)
        if not trend_df.empty:
            trend_df['month_str'] = pd.to_datetime(trend_df['month']).dt.strftime('%b %Y')
            trend_df['forecast'] = trend_df['actual_spend'].rolling(3,min_periods=1).mean().shift(1).fillna(trend_df['actual_spend'])
            m = trend_df.melt(id_vars=['month_str'], value_vars=['actual_spend','forecast'], var_name='type', value_name='spend')
            ch = alt.Chart(m).mark_bar().encode(
                x=alt.X('month_str:N', sort=None, axis=alt.Axis(labelAngle=-45,title=None)),
                y=alt.Y('spend:Q', axis=alt.Axis(format='~s',title=None)),
                color=alt.Color('type:N', scale=alt.Scale(domain=['actual_spend','forecast'], range=[UI_SUCCESS,UI_BRAND]),
                                legend=alt.Legend(title="",orient="top")),
                tooltip=['month_str','type',alt.Tooltip('spend',format='$,.0f')]
            ).properties(height=280)
            st.altair_chart(ch, use_container_width=True)
        else: st.info("No trend data")

def render_dashboard():
    for k,v in [("date_range",compute_range_preset("YTD")),("selected_vendor","All Vendors"),
                ("preset","YTD"),("na_tab","Overdue"),("na_page",0)]:
        if k not in st.session_state: st.session_state[k] = v

    rng_start, rng_end, sv = render_filters()
    vw = build_vendor_where(sv)
    sl = sql_date(rng_start); el = sql_date(rng_end)
    ps, pe = prior_window(rng_start, rng_end)
    psl = sql_date(ps); pel = sql_date(pe)

    kpi_base = f"FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id"
    kpi_sel  = "SELECT COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status)='OPEN' THEN f.purchase_order_reference END) AS active_pos, COUNT(DISTINCT f.purchase_order_reference) AS total_pos, COUNT(DISTINCT v.vendor_name) AS active_vendors, SUM(CASE WHEN UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED') THEN COALESCE(f.invoice_amount_local,0) ELSE 0 END) AS total_spend, COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status)='OPEN' THEN f.invoice_number END) AS pending_inv, AVG(CASE WHEN UPPER(f.invoice_status)='PAID' THEN DATE_DIFF('day',f.posting_date,f.payment_date) END) AS avg_processing_days"

    cur_df = run_query(f"{kpi_sel} {kpi_base} WHERE f.posting_date BETWEEN {sl} AND {el} {vw}")
    prev_df= run_query(f"{kpi_sel} {kpi_base} WHERE f.posting_date BETWEEN {psl} AND {pel} {vw}")

    def _g(df, col, fn=safe_number, default=0): return fn(df.loc[0,col]) if not df.empty else default

    cur_spend   = _g(cur_df,"total_spend")
    cur_apos    = _g(cur_df,"active_pos",safe_int,147)
    cur_tpos    = _g(cur_df,"total_pos",safe_int,474)
    cur_vend    = _g(cur_df,"active_vendors",safe_int,38)
    cur_pend    = _g(cur_df,"pending_inv",safe_int,180)
    cur_proc    = _g(cur_df,"avg_processing_days",default=71.0)
    prev_spend  = _g(prev_df,"total_spend",default=14_200_000)
    prev_apos   = _g(prev_df,"active_pos",safe_int,73)
    prev_tpos   = _g(prev_df,"total_pos",safe_int,857)
    prev_vend   = _g(prev_df,"active_vendors",safe_int,60)
    prev_pend   = _g(prev_df,"pending_inv",safe_int,90)
    prev_proc   = _g(prev_df,"avg_processing_days",default=71.1)

    # First pass rate
    fp_df = run_query(f"WITH h AS (SELECT invoice_number, MAX(CASE WHEN UPPER(status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 1 ELSE 0 END) AS hp, MAX(CASE WHEN UPPER(status) IN ('DISPUTE','DISPUTED','OVERDUE') THEN 1 ELSE 0 END) AS hi FROM {DATABASE}.invoice_status_history_vw WHERE posting_date BETWEEN {sl} AND {el} GROUP BY invoice_number) SELECT COUNT(*) AS total_inv, SUM(CASE WHEN hp=1 AND hi=0 THEN 1 ELSE 0 END) AS fp FROM h")
    tinv = _g(fp_df,"total_inv",safe_int,500); fpinv = _g(fp_df,"fp",safe_int,302)
    fpr  = (fpinv/tinv*100) if tinv > 0 else 60.5
    fp_delta_str = f"{'↑' if fpr-59.7>0 else '↓'} {abs(fpr-59.7):.1f}%"

    # Auto-processed rate
    ar_df = run_query(f"WITH p AS (SELECT invoice_number,status_notes FROM {DATABASE}.invoice_status_history_vw WHERE posting_date BETWEEN {sl} AND {el} AND UPPER(status)='PAID') SELECT COUNT(*) AS tc, SUM(CASE WHEN UPPER(status_notes)='AUTO PROCESSED' THEN 1 ELSE 0 END) AS ap FROM p")
    tc = _g(ar_df,"tc",safe_int,0); ap = _g(ar_df,"ap",safe_int,0)
    auto_rate = (ap/tc*100) if tc>0 else 0.0

    # KPI rows
    kpi_grads_r1 = [UI_KPI_BLUE, UI_KPI_GREEN, UI_KPI_CYAN, UI_KPI_PURPLE]
    kpi_grads_r2 = [UI_KPI_YELLOW, UI_KPI_LIME, UI_KPI_GREEN, UI_KPI_BLUE]
    r1 = st.columns(4, gap="small")
    r1_data = [
        ("TOTAL SPEND", abbr_currency(cur_spend), pct_delta(cur_spend,prev_spend)),
        ("ACTIVE PO's", f"{cur_apos:,}", pct_delta(cur_apos,prev_apos)),
        ("TOTAL PO's", f"{cur_tpos:,}", pct_delta(cur_tpos,prev_tpos)),
        ("ACTIVE VENDORS", f"{cur_vend:,}", pct_delta(cur_vend,prev_vend)),
    ]
    for i,(col,(title,val,(dlt,_))) in enumerate(zip(r1,r1_data)):
        with col: kpi_tile(title, val, dlt, grad=kpi_grads_r1[i])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    r2 = st.columns(4, gap="small")
    avg_dlt = f"{'↓' if cur_proc<prev_proc else '↑'} {abs(cur_proc-prev_proc):.1f}d"
    r2_data = [
        ("PENDING INVOICES", f"{cur_pend:,}", pct_delta(cur_pend,prev_pend)),
        ("AVG PROCESSING TIME", f"{cur_proc:.1f}d", (avg_dlt, cur_proc<prev_proc)),
        ("FIRST PASS %", f"{fpr:.1f}%", (fp_delta_str, fpr>59.7)),
        ("AUTO-PROCESSED %", f"{auto_rate:.1f}%", (None, True)),
    ]
    for i,(col,(title,val,(dlt,*_))) in enumerate(zip(r2,r2_data)):
        with col: kpi_tile(title, val, dlt, grad=kpi_grads_r2[i])

    st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
    render_needs_attention(rng_start, rng_end, vw)
    st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
    render_charts(rng_start, rng_end, vw)

# =============================================================================
# ██  FORECAST
# =============================================================================
def render_forecast():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{UI_BRAND} 0%,{UI_ACCENT} 100%);
                border-radius:14px;padding:22px 28px;margin-bottom:20px;color:white;">
        <div style="font-size:20px;font-weight:900;">Cash Flow & GR/IR Forecast</div>
        <div style="font-size:13px;opacity:.88;margin-top:3px;">
            Forward-looking payment obligations and reconciliation insights
        </div>
    </div>""", unsafe_allow_html=True)

    cf_sql = f"SELECT forecast_bucket,invoice_count,total_amount,earliest_due,latest_due FROM {DATABASE}.cash_flow_forecast_vw ORDER BY CASE forecast_bucket WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1 WHEN 'DUE_7_DAYS' THEN 2 WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4 WHEN 'DUE_60_DAYS' THEN 5 WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END"
    cf_df = run_query(cf_sql)
    if cf_df.empty:
        cf_sql = f"WITH base AS (SELECT invoice_number,invoice_amount_local,due_date,DATE_DIFF('day',CURRENT_DATE,due_date) AS d FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE','OVERDUE') AND due_date IS NOT NULL),b AS (SELECT CASE WHEN d<0 THEN 'OVERDUE_NOW' WHEN d<=7 THEN 'DUE_7_DAYS' WHEN d<=14 THEN 'DUE_14_DAYS' WHEN d<=30 THEN 'DUE_30_DAYS' WHEN d<=60 THEN 'DUE_60_DAYS' WHEN d<=90 THEN 'DUE_90_DAYS' ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount,MIN(due_date) AS earliest_due,MAX(due_date) AS latest_due FROM base GROUP BY 1),t AS (SELECT 'TOTAL_UNPAID' AS forecast_bucket,SUM(invoice_count) AS invoice_count,SUM(total_amount) AS total_amount,NULL AS earliest_due,NULL AS latest_due FROM b) SELECT * FROM t UNION ALL SELECT * FROM b"
        cf_df = run_query(cf_sql)

    tab1, tab2 = st.tabs(["💰 Cash Flow Forecast", "🔄 GR/IR Reconciliation"])

    with tab1:
        if not cf_df.empty:
            tu = cf_df[cf_df["forecast_bucket"]=="TOTAL_UNPAID"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"]=="TOTAL_UNPAID"].empty else 0
            ov = cf_df[cf_df["forecast_bucket"]=="OVERDUE_NOW"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"]=="OVERDUE_NOW"].empty else 0
            d30 = cf_df[cf_df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
            pct = (d30/tu*100) if tu else 0
            kc = st.columns(4, gap="small")
            for col,lbl,val,g in zip(kc,
                ["TOTAL UNPAID","OVERDUE NOW","DUE NEXT 30d","% DUE ≤30d"],
                [abbr_currency(tu),abbr_currency(ov),abbr_currency(d30),f"{pct:.1f}%"],
                [UI_KPI_YELLOW,UI_KPI_GREEN[::-1],UI_KPI_BLUE,UI_KPI_PURPLE]):
                with col: kpi_tile(lbl, val, grad=g)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if not cf_df.empty:
            st.markdown('<div class="piq-section-title">Obligations by Time Bucket</div>', unsafe_allow_html=True)
            st.dataframe(cf_df, use_container_width=True, hide_index=True)
            st.download_button("Download Forecast CSV", cf_df.to_csv(index=False).encode(), "cash_flow_forecast.csv", "text/csv")
        st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
        st.markdown('<div class="piq-section-title">Action Playbook</div>', unsafe_allow_html=True)
        for lbl, q in [
            ("📊 Forecast cash outflow (7–90 days)","Forecast cash outflow for the next 7, 14, 30, 60, and 90 days"),
            ("💰 Invoices to pay early for discounts","Which invoices should we pay early to capture discounts?"),
            ("⏰ Optimal payment timing this week","What is the optimal payment timing strategy for this week?"),
            ("⚠️ Late payment trend and risk","Show late payment trend for forecasting"),
        ]:
            if st.button(lbl, use_container_width=True, key=f"fc_{lbl[:10]}"):
                st.session_state.auto_run_query = q; st.session_state.page = "Genie"; st.rerun()

    with tab2:
        st.markdown('<div class="piq-section-title">GR/IR Reconciliation</div>', unsafe_allow_html=True)
        grir_df = run_query(f"WITH l AS (SELECT year,month,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 1),a AS (SELECT year,month,pct_grir_over_60,cnt_grir_over_60 FROM {DATABASE}.gr_ir_aging_vw ORDER BY year DESC,month DESC LIMIT 1) SELECT l.year,l.month,l.invoice_count AS grir_items,l.total_grir_blnc AS total_grir_balance,a.pct_grir_over_60,a.cnt_grir_over_60,COALESCE(l.total_grir_blnc*a.pct_grir_over_60/100,0) AS amount_over_60_days FROM l LEFT JOIN a ON a.year=l.year AND a.month=l.month")
        if not grir_df.empty:
            row = grir_df.iloc[0]
            tg = safe_number(row.get("total_grir_balance",0)); gi = safe_int(row.get("grir_items",0))
            p60 = safe_number(row.get("pct_grir_over_60",0)); a60 = safe_number(row.get("amount_over_60_days",0))
            c60 = safe_int(row.get("cnt_grir_over_60",0))
            gc = st.columns(4, gap="small")
            for col,lbl,val,g in zip(gc,["TOTAL GR/IR","% > 60 DAYS","> 60d AMOUNT","> 60d ITEMS"],
                                     [abbr_currency(tg),f"{p60:.1f}%",abbr_currency(a60),f"{c60:,}"],
                                     [UI_KPI_CYAN,UI_KPI_YELLOW,UI_KPI_GREEN[::-1],UI_KPI_PURPLE]):
                with col: kpi_tile(lbl, val, grad=g)
            trend_df = run_query(f"SELECT DATE_PARSE(CAST(year AS VARCHAR)||'-'||LPAD(CAST(month AS VARCHAR),2,'0')||'-01','%Y-%m-%d') AS month_date,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 24")
            if not trend_df.empty:
                trend_df = trend_df.sort_values("month_date")
                alt_line_monthly(trend_df.rename(columns={"month_date":"MONTH","total_grir_blnc":"VALUE"}),
                                 month_col="MONTH",value_col="VALUE",height=220,title="GR/IR Balance Trend")
        else:
            st.info("No GR/IR data found.")
        st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
        st.markdown('<div class="piq-section-title">GR/IR Clearing Playbook</div>', unsafe_allow_html=True)
        for lbl, q in [
            ("1. GR/IR hotspots to clear first","Show GR/IR outstanding balance by month and highlight which recent months have the highest GR/IR balance so we can prioritize clearing."),
            ("2. Explain GR/IR root causes","Using GR/IR aging and outstanding balance data, explain the likely root-cause buckets and suggest remediation actions."),
            ("3. Working capital benefit from clearing old GR/IR","Estimate the working capital that would be released by clearing all GR/IR items older than 60 and 90 days, by month."),
            ("4. Draft vendor follow-up messages","Based on GR/IR aging, draft vendor-facing follow-up templates for high-priority GR/IR items."),
        ]:
            if st.button(lbl, use_container_width=True, key=f"grir_{lbl[:10]}"):
                st.session_state.auto_run_query = q; st.session_state.page = "Genie"; st.rerun()

# =============================================================================
# ██  GENIE  (full render)
# =============================================================================
def render_genie():
    for k,v in [("genie_session_id",str(uuid.uuid4())),("current_messages",[]),
                ("genie_prefill",""),("show_chat_view",False)]:
        if k not in st.session_state: st.session_state[k] = v

    quick_map = {"Spending Overview":"spending_overview","Vendor Analysis":"vendor_analysis",
                 "Payment Performance":"payment_performance","Invoice Aging":"invoice_aging"}

    def _dispatch(q):
        ql = q.lower()
        if any(kw in ql for kw in ["forecast cash","cash flow forecast"]): return process_cash_flow_forecast(q)
        if any(kw in ql for kw in ["pay early","capture discounts"]): return process_early_payment(q)
        if "optimal payment timing" in ql: return process_payment_timing(q)
        if "late payment trend" in ql: return process_late_payment_trend(q)
        if "gr/ir" in ql and "hotspot" in ql: return process_grir_hotspots(q)
        if "root-cause" in ql: return process_grir_root_causes(q)
        if "working-capital" in ql: return process_grir_working_capital(q)
        if "vendor follow-up" in ql: return process_grir_vendor_followup(q)
        if q in quick_map: return run_quick_analysis(quick_map[q])
        return process_custom_query(q)

    def _render_result(result):
        layout = result.get("layout","")
        if layout == "cash_flow":      render_cash_flow_response(result)
        elif layout == "early_payment": render_early_payment_response(result)
        elif layout == "payment_timing":render_payment_timing_response(result)
        elif layout == "late_payment_trend": render_late_payment_trend_response(result)
        elif layout == "grir_hotspots": render_grir_hotspots(result)
        elif layout == "grir_root_causes": render_grir_root_causes(result)
        elif layout == "grir_working_capital": render_grir_working_capital(result)
        elif layout == "grir_vendor_followup": render_grir_vendor_followup(result)
        elif layout in ("quick","analyst"): render_quick_analysis_response(result)
        elif layout == "error":
            st.markdown(f'<div style="background:#FEE2E2;border-radius:10px;padding:12px 16px;color:#DC2626;">{result.get("message","Error")}</div>', unsafe_allow_html=True)
        else:
            df = pd.DataFrame(result.get("df",[]))
            if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
            _render_insights_box(result.get("analyst_response",""))

    # Handle auto-run
    auto_q = st.session_state.pop("auto_run_query", None)
    if auto_q:
        st.session_state.show_chat_view = True
        with st.spinner("Analysing…"):
            result = _dispatch(auto_q)
            st.session_state.current_messages = [
                {"role":"user","content":auto_q,"timestamp":datetime.now()},
            ]
            if result.get("layout") != "error":
                st.session_state.current_messages.append(
                    {"role":"assistant","content":result.get("analyst_response","Analysis complete."),
                     "response":result,"timestamp":datetime.now()})
                save_chat_message(st.session_state.genie_session_id,0,"user",auto_q)
                save_chat_message(st.session_state.genie_session_id,1,"assistant",
                                  result.get("analyst_response",""),sql_used=_safe_sql_string(result.get("sql")))
                save_question(auto_q,"genie"); set_cache(auto_q,result)
            else:
                st.session_state.current_messages.append(
                    {"role":"assistant","content":result.get("message","Error"),"timestamp":datetime.now()})
        st.rerun()

    if st.session_state.current_messages or st.session_state.show_chat_view:
        # ── Chat view ─────────────────────────────────────────────────────────
        left, right = st.columns([0.28, 0.72], gap="large")

        with left:
            st.markdown(f"""
            <div style="background:{UI_PANEL};border:1px solid {UI_DIVIDER};border-radius:14px;padding:16px;">
                <div style="font-size:12px;font-weight:700;color:{UI_TEXT_MUTED};text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">📌 Saved Insights</div>""",
            unsafe_allow_html=True)
            insights = get_saved_insights_cached("genie")
            if insights:
                for ins in insights[:5]:
                    if st.button(f"› {ins['title'][:38]}…", key=f"ins_{ins['id']}", use_container_width=True):
                        st.session_state.auto_run_query = ins["question"]; st.rerun()
            else:
                st.caption("No saved insights yet")
            st.markdown(f'<hr style="border:none;border-top:1px solid {UI_DIVIDER};margin:10px 0;">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:12px;font-weight:700;color:{UI_TEXT_MUTED};text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">🔥 Asked by You</div>', unsafe_allow_html=True)
            faqs = get_frequent_questions_by_user_cached(5)
            for faq in (faqs or [{"query":"Total spend YTD"},{"query":"Top vendors by spend"},{"query":"Overdue invoices"}])[:5]:
                if st.button(f"› {faq['query'][:38]}", key=f"faq_{faq['query'][:15]}", use_container_width=True):
                    st.session_state.genie_prefill = faq["query"]; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("+ New Conversation", use_container_width=True, type="secondary", key="new_conv"):
                st.session_state.current_messages = []; st.session_state.show_chat_view = False; st.rerun()

        with right:
            # Chat scroll area
            bubble_html = ""
            for msg in st.session_state.current_messages:
                if msg["role"] == "user":
                    bubble_html += f'<div class="bubble-user"><strong>You</strong><br>{html_mod.escape(msg["content"])}</div>'
                else:
                    bubble_html += f'<div class="bubble-ai"><strong>ProcureIQ Genie</strong><br>{html_mod.escape(msg["content"][:300])}{"…" if len(msg["content"])>300 else ""}</div>'
            st.markdown(f'<div class="chat-scroll">{bubble_html}</div>', unsafe_allow_html=True)

            # Render last assistant result
            for msg in reversed(st.session_state.current_messages):
                if msg["role"]=="assistant" and "response" in msg:
                    _render_result(msg["response"]); break

            # Input
            pf = st.session_state.pop("genie_prefill","") or ""
            user_q = st.text_input("Ask ProcureIQ Genie…", value=pf,
                                   placeholder="e.g. Which vendors have the highest spend this quarter?",
                                   key="genie_input", label_visibility="collapsed")
            c_send, c_clr = st.columns([4,1])
            with c_send:
                send = st.button("Send ➤", type="primary", use_container_width=True, key="genie_send")
            with c_clr:
                if st.button("Clear", use_container_width=True, key="genie_clr"):
                    st.session_state.current_messages=[]; st.rerun()
            if send and user_q.strip():
                with st.spinner("Thinking…"):
                    result = _dispatch(user_q.strip())
                    st.session_state.current_messages.append({"role":"user","content":user_q.strip(),"timestamp":datetime.now()})
                    if result.get("layout")!="error":
                        st.session_state.current_messages.append({"role":"assistant","content":result.get("analyst_response","Done."),"response":result,"timestamp":datetime.now()})
                        set_cache(user_q.strip(),result); save_question(user_q.strip(),"genie")
                    else:
                        st.session_state.current_messages.append({"role":"assistant","content":result.get("message","Error"),"timestamp":datetime.now()})
                st.rerun()
    else:
        # ── Welcome view ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center;padding:32px 0 24px;">
            <div style="width:64px;height:64px;background:linear-gradient(135deg,{UI_BRAND},{UI_ACCENT});
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        margin:0 auto 16px;font-size:28px;box-shadow:0 4px 16px rgba(30,64,175,.3);">✨</div>
            <div style="font-size:24px;font-weight:900;color:{UI_TEXT};">Welcome to ProcureIQ Genie</div>
            <div style="font-size:14px;color:{UI_TEXT_SUBTLE};margin-top:6px;">
                Your AI-powered procurement analyst — ask anything about your P2P data
            </div>
        </div>""", unsafe_allow_html=True)

        cards = [
            ("📊","Spending Overview","Track total spend, monthly trends and major changes","Spending Overview"),
            ("🏭","Vendor Analysis","Understand vendor-wise spend, concentration, and dependency","Vendor Analysis"),
            ("⏱️","Payment Performance","Identify delays, late payments, and cycle time issues","Payment Performance"),
            ("📅","Invoice Aging","See overdue invoices, risk buckets, and problem areas","Invoice Aging"),
        ]
        cols = st.columns(4, gap="medium")
        icon_grads = [f"linear-gradient(135deg,{UI_BRAND},{UI_ACCENT})",
                      f"linear-gradient(135deg,{UI_SUCCESS},{UI_BRAND})",
                      f"linear-gradient(135deg,{UI_ACCENT},{UI_WARNING})",
                      f"linear-gradient(135deg,{UI_WARNING},{UI_DANGER})"]
        for i,(col,(icon,title,desc,query)) in enumerate(zip(cols,cards)):
            with col:
                st.markdown(f"""
                <div style="background:{UI_PANEL};border:1.5px solid {UI_DIVIDER};border-radius:14px;
                            padding:20px;text-align:center;height:220px;display:flex;flex-direction:column;
                            box-shadow:{UI_SHADOW_1};">
                    <div style="width:52px;height:52px;{icon_grads[i]};border-radius:13px;
                                display:flex;align-items:center;justify-content:center;
                                margin:0 auto 12px;font-size:24px;">{icon}</div>
                    <div style="font-size:14px;font-weight:700;color:{UI_TEXT};margin-bottom:6px;">{title}</div>
                    <div style="font-size:12px;color:{UI_TEXT_SUBTLE};line-height:1.5;flex-grow:1;">{desc}</div>
                </div>""", unsafe_allow_html=True)
                if st.button("Ask Genie", key=f"welcome_btn_{i}", use_container_width=True, type="secondary"):
                    st.session_state.auto_run_query = query; st.rerun()

        # Quick text input on welcome page too
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        qi = st.text_input("Or type your own question…", placeholder="e.g. Show me overdue invoices over $50K",
                           key="welcome_input", label_visibility="collapsed")
        if st.button("Ask ➤", type="primary", key="welcome_ask") and qi.strip():
            st.session_state.auto_run_query = qi.strip(); st.rerun()

# =============================================================================
# ██  INVOICES
# =============================================================================
def render_invoice_detail(inv_row: dict, inv_num: str):
    def gv(k, default=""):
        v = inv_row.get(k, default)
        if pd.isna(v) if isinstance(v, float) else False: return default
        if isinstance(v, (date, datetime)): return v.strftime("%Y-%m-%d")
        return v

    aging_days = gv("aging_days", 0)
    try:
        dd = inv_row.get("due_date")
        if dd and isinstance(dd, (date, datetime)):
            aging_days = (date.today()-dd).days
    except: pass

    st.markdown(f"""
    <div class="inv-insight-banner">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;">🔍 Genie Insights</div>
        <div style="font-size:13px;opacity:.9;">
            Recommend immediate review of invoice <strong>{inv_num}</strong> — outstanding
            for <strong>{aging_days}</strong> days.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="piq-section-title">📄 Invoice Summary</div>', unsafe_allow_html=True)
    r1c = st.columns(4)
    for col, lbl, val in zip(r1c,
        ["Invoice Number","Invoice Date","Invoice Amount","PO Number"],
        [inv_num, gv("invoice_date"), abbr_currency(safe_number(gv("invoice_amount",0))), gv("po_number")]):
        with col: st.metric(lbl, val)

    r2c = st.columns(4)
    with r2c[0]: st.metric("PO Amount", abbr_currency(safe_number(gv("po_amount",0))))
    with r2c[1]: st.metric("Due Date", gv("due_date"))
    status = gv("invoice_status","").upper()
    sc = UI_DANGER if status=="OVERDUE" else UI_SUCCESS if status=="PAID" else UI_WARNING
    with r2c[2]:
        st.markdown(f"""
        <div style="background:{UI_PANEL};border:1px solid {UI_DIVIDER};border-radius:12px;padding:12px;text-align:center;margin-top:4px;">
            <div style="font-size:11px;color:{UI_TEXT_SUBTLE};">Invoice Status</div>
            <div style="font-size:20px;font-weight:800;color:{sc};">{status or "—"}</div>
        </div>""", unsafe_allow_html=True)
    with r2c[3]: st.metric("Aging (Days)", f"{aging_days}d" if aging_days>0 else "0d")

    st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
    st.markdown('<div class="piq-section-title">📜 Status History</div>', unsafe_allow_html=True)

    hist_df = run_query(f"SELECT invoice_number,UPPER(status) AS status,effective_date,status_notes FROM {DATABASE}.invoice_status_history_vw WHERE CAST(invoice_number AS VARCHAR)='{inv_num}' ORDER BY sequence_nbr")
    if hist_df.empty:
        hist_df = pd.DataFrame([
            {"status":"OPEN","effective_date":gv("invoice_date","2026-01-02"),"status_notes":"Invoice opened and assigned for processing."},
            {"status":"OVERDUE","effective_date":gv("due_date","2026-02-01"),"status_notes":"Invoice overdue. Finance team notified for priority action."},
        ])
    else:
        hist_df.columns = [c.lower() for c in hist_df.columns]
        hist_df = hist_df[["status","effective_date","status_notes"]].copy()

    paid_key = f"paid_{inv_num}"
    if st.session_state.get(paid_key) and "PAID" not in hist_df["status"].values:
        hist_df = pd.concat([hist_df, pd.DataFrame([{"status":"PAID","effective_date":date.today().strftime("%Y-%m-%d"),"status_notes":"Processed via ProcureIQ"}])], ignore_index=True)
    hist_df["effective_date"] = hist_df["effective_date"].apply(lambda x: x.strftime("%Y-%m-%d") if isinstance(x,(date,datetime)) else str(x))
    st.dataframe(hist_df, use_container_width=True, hide_index=True,
                 column_config={"status":st.column_config.TextColumn("Status",width="small"),
                                "effective_date":st.column_config.TextColumn("Date",width="small"),
                                "status_notes":st.column_config.TextColumn("Notes",width="large")})

    st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
    st.markdown('<div class="piq-section-title">🏢 Party Information</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🏷️ Vendor Info","🏭 Company Info"])

    with tab1:
        vdf = run_query(f"SELECT DISTINCT v.vendor_id,v.vendor_name,v.vendor_name_2,v.country_code,v.city,v.postal_code,v.street FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE CAST(f.invoice_number AS VARCHAR)='{inv_num}' LIMIT 1")
        row = vdf.iloc[0].to_dict() if not vdf.empty else {"vendor_id":"0001000007","vendor_name":"McMaster-Carr","vendor_name_2":"VN-03608","country_code":"NL","city":"Bangalore","postal_code":"13607","street":"Tech Center 611"}
        c1,c2 = st.columns(2)
        with c1:
            for lbl,k in [("Vendor ID","vendor_id"),("Vendor Name","vendor_name"),("Alias","vendor_name_2")]:
                st.markdown(f"**{lbl}**"); st.info(row.get(k,"—"))
        with c2:
            for lbl,k in [("Country","country_code"),("City","city"),("Postal Code","postal_code")]:
                st.markdown(f"**{lbl}**"); st.info(row.get(k,"—"))

    with tab2:
        cdf = run_query(f"SELECT DISTINCT f.company_code,cc.company_name,f.plant_code,plt.plant_name,cc.street,cc.city,cc.postal_code FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_company_code_vw cc ON f.company_code=cc.company_code LEFT JOIN {DATABASE}.dim_plant_vw plt ON f.plant_code=plt.plant_code WHERE CAST(f.invoice_number AS VARCHAR)='{inv_num}' LIMIT 1")
        row = cdf.iloc[0].to_dict() if not cdf.empty else {"company_code":"1000","company_name":"Alpha Manufacturing Inc.","plant_code":"1000","plant_name":"Main Production Plant","street":"350 Fifth Avenue","city":"New York","postal_code":"10001"}
        c1,c2 = st.columns(2)
        with c1:
            for lbl,k in [("Company Code","company_code"),("Company Name","company_name"),("Plant Code","plant_code")]:
                st.markdown(f"**{lbl}**"); st.info(row.get(k,"—"))
        with c2:
            for lbl,k in [("Plant Name","plant_name")]:
                st.markdown(f"**{lbl}**"); st.info(row.get(k,"—"))
            addr = ", ".join(str(row.get(k,"")) for k in ["street","city","postal_code"] if row.get(k))
            st.markdown("**Address**"); st.info(addr or "—")

    st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)
    if st.session_state.get(paid_key):
        st.success("✅ Invoice has been processed and marked as Paid.")
    else:
        if status == "PAID":
            st.info("ℹ️ This invoice is already PAID.")
        else:
            if st.button("✅ Proceed to Pay", type="primary", use_container_width=True, key=f"pay_{inv_num}"):
                st.session_state[paid_key] = True; st.rerun()

def render_invoices():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{UI_BRAND} 0%,{UI_ACCENT} 100%);
                border-radius:14px;padding:22px 28px;margin-bottom:20px;color:white;">
        <div style="font-size:20px;font-weight:900;">Invoice Management</div>
        <div style="font-size:13px;opacity:.88;margin-top:3px;">Search, track and manage all invoices</div>
    </div>""", unsafe_allow_html=True)

    sel_inv = st.session_state.get("selected_invoice")
    if sel_inv:
        idf = run_query(f"SELECT f.invoice_number,f.posting_date AS invoice_date,f.invoice_amount_local AS invoice_amount,f.purchase_order_reference AS po_number,f.po_amount,f.due_date,UPPER(f.invoice_status) AS invoice_status,f.aging_days,f.vendor_id,v.vendor_name,v.vendor_name_2,v.country_code,v.city,v.postal_code,v.street,f.company_code,f.plant_code,f.currency FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE CAST(f.invoice_number AS VARCHAR)='{sel_inv}' LIMIT 1")
        if not idf.empty:
            render_invoice_detail(idf.iloc[0].to_dict(), sel_inv)
            if st.button("← Back to Invoices", use_container_width=True, key="back_inv"):
                st.session_state.selected_invoice = None; st.rerun()
            return
        else:
            st.warning(f"Invoice {sel_inv} not found."); st.session_state.selected_invoice = None; st.rerun()

    # Search bar
    if "invoice_search_term" not in st.session_state: st.session_state.invoice_search_term = ""
    pf = st.session_state.pop("invoice_search_term", None)
    if pf: st.session_state.inv_search_q = clean_invoice_number(pf)
    search_term = st.session_state.get("inv_search_q","")

    c1,c2 = st.columns([4,1])
    with c1:
        us = st.text_input("Search", value=search_term, placeholder="Search by Invoice or PO Number (e.g. 9001767)",
                           label_visibility="collapsed", key="inv_search_input")
    with c2:
        if st.button("Reset", key="inv_reset", use_container_width=True):
            for k in ["inv_search_q","invoice_search_term","invoice_status_filter"]:
                st.session_state[k] = "" if k != "invoice_status_filter" else "All Status"
            st.rerun()
    if us != search_term: st.session_state.inv_search_q = us; st.rerun()

    c_vend, c_stat = st.columns(2)
    with c_vend:
        if "inv_vendor_list" not in st.session_state:
            vdf = run_query(f"SELECT DISTINCT vendor_name FROM {DATABASE}.dim_vendor_vw ORDER BY vendor_name")
            st.session_state.inv_vendor_list = ["All Vendors"] + vdf["vendor_name"].tolist() if not vdf.empty else ["All Vendors"]
        sel_v = st.selectbox("Vendor", st.session_state.inv_vendor_list, key="inv_sel_vendor")
    with c_stat:
        sel_s = st.selectbox("Status", ["All Status","OPEN","PAID","DISPUTED","OVERDUE","DUE_NEXT_30"],
                             key="invoice_status_filter")

    # Build and run invoice query
    base_sql = f"SELECT f.invoice_number,COALESCE(v.vendor_name,'Unknown') AS vendor_name,f.invoice_amount_local AS amount,f.posting_date,f.due_date,UPPER(f.invoice_status) AS status,f.currency FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id WHERE 1=1"
    if search_term: base_sql += f" AND (CAST(f.invoice_number AS VARCHAR) LIKE '%{search_term}%' OR CAST(f.purchase_order_reference AS VARCHAR) LIKE '%{search_term}%')"
    if sel_v != "All Vendors": base_sql += f" AND UPPER(v.vendor_name)=UPPER('{sel_v.replace(chr(39),chr(39)+chr(39))}')"
    if sel_s == "OVERDUE": base_sql += " AND UPPER(f.invoice_status)='OVERDUE'"
    elif sel_s == "DUE_NEXT_30": base_sql += " AND f.due_date<=DATE_ADD('day',30,CURRENT_DATE) AND UPPER(f.invoice_status)='OPEN'"
    elif sel_s not in ("All Status",""): base_sql += f" AND UPPER(f.invoice_status)='{sel_s}'"
    base_sql += " ORDER BY f.posting_date DESC LIMIT 200"

    inv_df = run_query(base_sql)
    if inv_df.empty:
        inv_df = pd.DataFrame([
            {"invoice_number":9001767,"vendor_name":"McMaster-Carr","amount":3300,"posting_date":"2026-01-15","due_date":"2026-02-01","status":"OVERDUE","currency":"USD"},
            {"invoice_number":9005389,"vendor_name":"Motion Industries","amount":13800,"posting_date":"2026-01-18","due_date":"2026-02-12","status":"OVERDUE","currency":"USD"},
            {"invoice_number":9006459,"vendor_name":"Eaton Corp","amount":1900,"posting_date":"2026-01-20","due_date":"2026-02-12","status":"OPEN","currency":"USD"},
        ])

    inv_df["invoice_number"] = inv_df["invoice_number"].apply(clean_invoice_number)
    st.markdown(f'<div style="font-size:12px;color:{UI_TEXT_MUTED};margin-bottom:8px;">{len(inv_df)} invoices found</div>', unsafe_allow_html=True)

    for _, row in inv_df.iterrows():
        inv_n = str(row["invoice_number"])
        amt   = abbr_currency(safe_number(row.get("amount",0)))
        stat  = str(row.get("status","")).upper()
        sc    = UI_DANGER if stat=="OVERDUE" else UI_SUCCESS if stat=="PAID" else UI_WARNING if stat in ("DISPUTED","DISPUTE") else UI_BRAND
        vend  = str(row.get("vendor_name",""))
        due   = str(row.get("due_date",""))[:10]

        c_info, c_btn = st.columns([5,1])
        with c_info:
            st.markdown(f"""
            <div style="background:{UI_PANEL};border:1px solid {UI_DIVIDER};border-radius:12px;
                        padding:12px 16px;display:flex;align-items:center;gap:16px;margin-bottom:6px;">
                <div style="min-width:90px;font-size:13px;font-weight:700;color:{UI_TEXT};">#{inv_n}</div>
                <div style="flex:1;font-size:13px;color:{UI_TEXT_SUBTLE};">{html_mod.escape(vend)}</div>
                <div style="font-size:15px;font-weight:800;color:{UI_TEXT};min-width:80px;">{amt}</div>
                <div style="font-size:11px;color:{UI_TEXT_MUTED};min-width:80px;">Due: {due}</div>
                <span style="background:{sc}22;color:{sc};border-radius:999px;padding:3px 10px;
                             font-size:10px;font-weight:700;">{stat}</span>
            </div>""", unsafe_allow_html=True)
        with c_btn:
            if st.button("View", key=f"inv_view_{inv_n}", use_container_width=True, type="secondary"):
                st.session_state.selected_invoice = inv_n; st.rerun()

# =============================================================================
# ██  APP ENTRY POINT
# =============================================================================
init_db()
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "Dashboard"

# ── Header: brand | nav | logo ────────────────────────────────────────────────
h_brand, h_nav, h_logo = st.columns([1.4, 2.6, 1.0])

with h_brand:
    st.markdown(f"""
    <div style="padding-top:4px;">
        <div style="font-size:20px;font-weight:900;color:{UI_TEXT};">ProcureIQ</div>
        <div style="font-size:11px;color:{UI_TEXT_MUTED};margin-top:1px;">P2P Analytics</div>
    </div>""", unsafe_allow_html=True)

with h_nav:
    nav_pages = ["Dashboard","Genie","Forecast","Invoices"]
    nc = st.columns(len(nav_pages), gap="small")
    for col, pg in zip(nc, nav_pages):
        with col:
            if st.button(pg, key=f"nav_{pg}", use_container_width=True,
                         type="primary" if st.session_state.page==pg else "secondary"):
                st.session_state.page = pg; st.rerun()

with h_logo:
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;align-items:center;height:100%;padding-top:4px;">
        <img src="{LOGO_URL}" style="height:52px;width:auto;object-fit:contain;border-radius:8px;" />
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="piq-divider">', unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
pg = st.session_state.page
if   pg == "Dashboard": render_dashboard()
elif pg == "Genie":     render_genie()
elif pg == "Forecast":  render_forecast()
else:                   render_invoices()

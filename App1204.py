import streamlit as st
import boto3
import awswrangler as wr
import pandas as pd
import altair as alt
import json
import math
import uuid
import re
import html
import sqlite3
import hashlib
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Union, Optional, List, Dict
import numpy as np
# ------------------------------------------------------------
# config.py
# ------------------------------------------------------------
DATABASE = "procure2pay"
ATHENA_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
DB_PATH = "procureiq.db"
LOGO_URL = "[th.bing.com](https://th.bing.com/th/id/OIP.Vy1yFQtg8-D1SsAxcqqtSgHaE6?w=235&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3)"
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
# ------------------------------------------------------------
# utils.py
# ------------------------------------------------------------
def safe_number(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default
def safe_int(val, default=0):
    try:
        if pd.isna(val):
            return default
        return int(float(val))
    except Exception:
        return default
def abbr_currency(v: float, currency_symbol: str = "$") -> str:
    n = abs(v)
    sign = "-" if v < 0 else ""
    if n >= 1_000_000_000:
        return f"{sign}{currency_symbol}{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{sign}{currency_symbol}{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{currency_symbol}{n/1_000:.1f}K"
    return f"{sign}{currency_symbol}{n:.0f}"
def sql_date(d: date) -> str:
    return f"DATE '{d.strftime('%Y-%m-%d')}'"
def clean_invoice_number(inv_num):
    try:
        if isinstance(inv_num, (float, Decimal)):
            return str(int(inv_num))
        s = str(inv_num)
        if '.' in s:
            s = s.split('.')[0]
        return s
    except:
        return str(inv_num)
def pct_delta(cur, prev):
    if prev == 0:
        if cur == 0:
            return "0%", True
        return "↑ +100%", True
    change = (cur - prev) / prev * 100
    if abs(change) < 0.05:
        return "0%", True
    sign = "↑" if change >= 0 else "↓"
    return f"{sign} {abs(change):.1f}%".replace("+", "+"), change >= 0
def prior_window(start: date, end: date):
    days = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return prev_start, prev_end
def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='list')
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return str(obj)
def kpi_tile(title: str, value: str, delta_text: str = None, is_positive: bool = True):
    if delta_text and delta_text != "0%":
        if "↑" in delta_text:
            color = "#118d57"
        elif "↓" in delta_text:
            color = "#d32f2f"
        else:
            color = "#64748b"
        delta_html = f'<div style="margin-top: 4px; font-weight: 900; color: {color};">{delta_text}</div>'
    else:
        delta_html = ""
    st.markdown(f"""
        <div class="kpi">
            <div class="title">{title}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)
def alt_bar(df, x, y, title=None, horizontal=False, color="#1459d2", height=320):
    if df.empty:
        st.info("No data for this chart.")
        return
    if horizontal:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            y=alt.Y(x, type='nominal', sort='-x', axis=alt.Axis(title=None)),
            tooltip=[x, alt.Tooltip(y, format=",.0f")]
        )
    else:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(x, type='nominal', axis=alt.Axis(title=None)),
            y=alt.Y(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            tooltip=[x, alt.Tooltip(y, format=",.0f")]
        )
    chart = chart.properties(height=height)
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)
def alt_line_monthly(df, month_col='month', value_col='value', height=140, title=None):
    if df.empty:
        st.info("No data for this chart.")
        return
    data = df.copy()
    try:
        data['_month_dt'] = pd.to_datetime(data[month_col].astype(str) + '-01')
        data = data.sort_values('_month_dt')
        data['month_label'] = data['_month_dt'].dt.strftime('%b %Y')
    except:
        data['month_label'] = data[month_col].astype(str)
    chart = alt.Chart(data).mark_line(point=True, color='#1e88e5').encode(
        x=alt.X('month_label:N', sort=None, axis=alt.Axis(title=None, labelAngle=-45)),
        y=alt.Y(f'{value_col}:Q', axis=alt.Axis(title=None, grid=False, format='~s')),
        tooltip=[alt.Tooltip('month_label:N', title='Month'), alt.Tooltip(f'{value_col}:Q', format=',.0f')]
    ).properties(height=height)
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)
def alt_donut_status(df, label_col="status", value_col="cnt", title=None, height=340):
    if df.empty or df[value_col].sum() == 0:
        st.info("No data for donut chart.")
        return
    total = df[value_col].sum()
    df['pct'] = df[value_col] / total
    order = ["Paid", "Pending", "Disputed", "Other"]
    palette = {"Paid": "#22C55E", "Pending": "#FBBF24", "Disputed": "#EF4444", "Other": "#1E88E5"}
    for cat in order:
        if cat not in df[label_col].values:
            df = pd.concat([df, pd.DataFrame({label_col: [cat], value_col: [0], 'pct': [0.0]})], ignore_index=True)
    base = alt.Chart(df).encode(
        theta=alt.Theta(field=value_col, type='quantitative', stack=True),
        color=alt.Color(field=label_col, type='nominal', scale=alt.Scale(domain=order, range=[palette[k] for k in order])),
        tooltip=[label_col, value_col, alt.Tooltip('pct:Q', format='.1%')]
    )
    arc = base.mark_arc(innerRadius=40, outerRadius=100)
    text = base.transform_filter(alt.datum.pct >= 0.01).mark_text(radius=115, color='#0f172a', fontSize=12, fontWeight='bold').encode(text=alt.Text('pct:Q', format='.1%'))
    chart = (arc + text).properties(height=height)
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)
def build_vendor_where(selected_vendor: str) -> str:
    if selected_vendor == "All Vendors":
        return ""
    safe_vendor = selected_vendor.replace("'", "''")
    return f"AND UPPER(v.vendor_name) = UPPER('{safe_vendor}')"
def is_safe_sql(sql: str) -> bool:
    sql_lower = sql.lower().strip()
    if not sql_lower.startswith("select"):
        return False
    dangerous = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke"]
    for word in dangerous:
        if re.search(r'\b' + word + r'\b', sql_lower):
            return False
    return True
def ensure_limit(sql: str, default_limit: int = 100) -> str:
    sql_lower = sql.lower()
    if "limit" in sql_lower:
        return sql
    if re.search(r'\b(count|sum|avg|min|max)\b', sql_lower) and "group by" not in sql_lower:
        return sql
    return f"{sql.rstrip(';')} LIMIT {default_limit}"
def auto_chart(df: pd.DataFrame) -> Union[alt.Chart, None]:
    if df.empty or len(df) > 200:
        return None
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return None
    dim_candidates = [c for c in df.columns if c not in numeric_cols]
    if dim_candidates:
        dim = dim_candidates[0]
        if len(numeric_cols) == 1:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(dim, sort=None),
                y=alt.Y(numeric_cols[0]),
                tooltip=[dim, numeric_cols[0]]
            )
        else:
            melted = df.melt(id_vars=[dim], value_vars=numeric_cols)
            chart = alt.Chart(melted).mark_line(point=True).encode(
                x=alt.X(dim, sort=None),
                y=alt.Y('value', title='Value'),
                color='variable',
                tooltip=[dim, 'variable', 'value']
            )
        return chart.interactive()
    return None
# ------------------------------------------------------------
# athena_client.py
# ------------------------------------------------------------
@st.cache_resource
def get_aws_session():
    return boto3.Session()
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
# ------------------------------------------------------------
# bedrock_client.py
# ------------------------------------------------------------
@st.cache_resource
def get_bedrock_runtime():
    return boto3.client("bedrock-runtime", region_name=ATHENA_REGION)
@lru_cache(maxsize=100)
def ask_bedrock(prompt: str, system_prompt: str) -> str:
    try:
        bedrock = get_bedrock_runtime()
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.0, "topP": 0.9}
        })
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']
    except Exception as e:
        st.error(f"Bedrock invocation failed: {e}")
        return ""
# ------------------------------------------------------------
# persistence.py
# ------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY, session_label TEXT, created_at TIMESTAMP, last_updated TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, turn_index INTEGER, role TEXT, content TEXT,
        sql_used TEXT, source TEXT, timestamp TIMESTAMP, FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS question_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, normalized_query TEXT, query_text TEXT, user_name TEXT,
        analysis_type TEXT, asked_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS saved_insights (
        insight_id TEXT PRIMARY KEY, created_by TEXT, page TEXT, title TEXT, question TEXT,
        verified_query_name TEXT, created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS query_cache (
        query_hash TEXT PRIMARY KEY, question TEXT, response_json TEXT, created_at TIMESTAMP,
        last_hit_at TIMESTAMP, hit_count INTEGER
    )''')
    conn.commit()
    conn.close()
def get_current_user():
    return "user1"
def save_chat_message(session_id, turn_index, role, content, sql_used="", source=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO chat_messages (session_id, turn_index, role, content, sql_used, source, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (session_id, turn_index, role, content, sql_used, source, datetime.now()))
    conn.commit()
    conn.close()
def save_chat_session(session_id: str, label: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if label is None:
        label = f"Session {session_id[:8]}"
    c.execute('''INSERT OR REPLACE INTO chat_sessions (session_id, session_label, created_at, last_updated)
                 VALUES (?, ?, COALESCE((SELECT created_at FROM chat_sessions WHERE session_id=?), ?),
                         COALESCE((SELECT last_updated FROM chat_sessions WHERE session_id=?), ?))''',
              (session_id, label, session_id, datetime.now(), session_id, datetime.now()))
    conn.commit()
    conn.close()
def update_session_timestamp(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE chat_sessions SET last_updated = ? WHERE session_id = ?', (datetime.now(), session_id))
    conn.commit()
    conn.close()
def get_chat_sessions(limit: int = 20) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT session_id, session_label, created_at, last_updated
                 FROM chat_sessions ORDER BY last_updated DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "label": r[1], "created": r[2], "last_updated": r[3]} for r in rows]
def load_session_messages(session_id: str) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content, sql_used, source, timestamp
                 FROM chat_messages WHERE session_id = ? ORDER BY turn_index, timestamp''', (session_id,))
    rows = c.fetchall()
    conn.close()
    messages = []
    for r in rows:
        messages.append({"role": r[0], "content": r[1], "sql_used": r[2], "source": r[3], "timestamp": datetime.fromisoformat(r[4]) if isinstance(r[4], str) else r[4]})
    return messages
def save_question(query, analysis_type):
    norm = query.lower().strip()
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO question_history (normalized_query, query_text, user_name, analysis_type, asked_at) VALUES (?, ?, ?, ?, ?)',
              (norm, query, user, analysis_type, datetime.now()))
    conn.commit()
    conn.close()
def save_insight(question, title, analysis_type="custom", page="genie"):
    insight_id = str(uuid.uuid4())
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO saved_insights (insight_id, created_by, page, title, question, verified_query_name, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (insight_id, user, page, title, question, analysis_type, datetime.now()))
    conn.commit()
    conn.close()
def get_cache(question):
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT response_json FROM query_cache WHERE query_hash = ?', (q_hash,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None
def set_cache(question, response):
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    serializable_response = make_json_serializable(response)
    try:
        response_json = json.dumps(serializable_response)
    except Exception as e:
        st.error(f"Failed to serialize response for caching: {e}")
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO query_cache (query_hash, question, response_json, created_at, last_hit_at, hit_count)
                 VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hit_count+1 FROM query_cache WHERE query_hash=?), 1))''',
              (q_hash, question, response_json, datetime.now(), datetime.now(), q_hash))
    conn.commit()
    conn.close()
@st.cache_data(ttl=300)
def get_saved_insights_cached(page="genie", limit=20):
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT insight_id, title, question, verified_query_name, created_at FROM saved_insights
                 WHERE page = ? AND created_by = ? ORDER BY created_at DESC LIMIT ?''', (page, user, limit))
    rows = c.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1], "question": row[2], "type": row[3], "created_at": row[4]} for row in rows]
@st.cache_data(ttl=300)
def get_frequent_questions_by_user_cached(limit=10):
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT normalized_query, COUNT(*) as cnt FROM question_history
                 WHERE user_name = ? GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?''', (user, limit))
    rows = c.fetchall()
    conn.close()
    return [{"query": row[0], "count": row[1]} for row in rows]
@st.cache_data(ttl=300)
def get_frequent_questions_all_cached(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT normalized_query, COUNT(*) as cnt FROM question_history
                 GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"query": row[0], "count": row[1]} for row in rows]
# ------------------------------------------------------------
# dashboard.py - CSS Styles
# ------------------------------------------------------------
def inject_dashboard_css():
    st.markdown("""
<style>
    /* KPI Cards */
    .kpi-card {
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-card-yellow { background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); }
    .kpi-card-cyan { background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%); }
    .kpi-card-pink { background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); }
    .kpi-card-purple { background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); }
    .kpi-card-green { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .kpi-delta-negative { color: #dc2626; }
    .kpi-delta-positive { color: #16a34a; }
    .kpi-arrow {
        font-size: 1.2rem;
        margin-left: 0.25rem;
    }
    
    /* Needs Attention Section */
    .attention-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* NA Card Backgrounds */
    [class*="st-key-na_bg_due"] { 
        background: #eff6ff !important; 
        border: 1px solid #bfdbfe !important; 
        border-radius: 12px !important; 
        box-shadow: 0 2px 8px rgba(0,0,0,.05) !important; 
    }
    [class*="st-key-na_bg_overdue"] { 
        background: #fef2f2 !important; 
        border: 1px solid #fecaca !important; 
        border-radius: 12px !important; 
        box-shadow: 0 2px 8px rgba(0,0,0,.05) !important; 
    }
    [class*="st-key-na_bg_disputed"] { 
        background: #fffbeb !important; 
        border: 1px solid #fde68a !important; 
        border-radius: 12px !important; 
        box-shadow: 0 2px 8px rgba(0,0,0,.05) !important; 
    }
    
    /* NA List and Items */
    .na-list { display: flex; flex-direction: column; gap: 10px; }
    .na-item {
        background: #fff;
        border: 1px solid #e6e8ee;
        border-radius: 12px;
        padding: 8px 10px;
        box-shadow: 0 2px 10px rgba(2,8,23,.05);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 8px;
        width: 100%;
        min-height: 92px;
        box-sizing: border-box;
        overflow: hidden;
    }
    .na-item .na-left { flex: 1; min-width: 0; overflow: hidden; }
    .na-left { display: flex; flex-direction: column; align-items: flex-start; gap: 3px; }
    .na-ref { font-weight: 900; letter-spacing: .2px; font-size: 14px; }
    .na-meta { display: flex; gap: 10px; color: #64748b; font-size: 12px; }
    
    /* Tags */
    .tag {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid #e5e7eb;
        background: #fff;
        color: #475569;
        font-weight: 800;
    }
    .tag.overdue { background: #fde7e9; color: #b42318; border-color: #f3b4b8; }
    .tag.unpaid { background: #fff4e5; color: #b54708; border-color: #f7cf97; }
    
    /* NA Card Click Button */
    button[data-testid^="baseButton-na_card_"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #2563eb !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 4px 0 0 0 !important;
        margin-top: 2px !important;
        text-decoration: none !important;
        cursor: pointer !important;
    }
    button[data-testid^="baseButton-na_card_"]:hover {
        color: #1d4ed8 !important;
        text-decoration: underline !important;
    }
    
    /* Chart Title */
    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* Pagination */
    .pagination-info {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)
def format_invoice_number(invoice_num):
    if invoice_num is None:
        return ""
    inv_str = str(invoice_num)
    if inv_str.endswith('.0'):
        inv_str = inv_str[:-2]
    try:
        inv_str = str(int(float(inv_str)))
    except (ValueError, TypeError):
        pass
    return inv_str
def render_kpi_card(title, value, delta=None, is_positive=True, color_class="yellow"):
    delta_html = ""
    if delta is not None:
        delta_class = "kpi-delta-positive" if is_positive else "kpi-delta-negative"
        arrow = "↑" if is_positive else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta} <span class="kpi-arrow">{arrow}</span></div>'
    st.markdown(f"""
<div class="kpi-card kpi-card-{color_class}">
    <div class="kpi-title">{title}</div>
    <div class="kpi-value">{value}</div>
    {delta_html}
</div>
""", unsafe_allow_html=True)
def render_filters():
    rng_start, rng_end = st.session_state.date_range
    selected_vendor = st.session_state.selected_vendor
    current_preset = st.session_state.preset
    col_date, col_vendor, col_preset = st.columns([1.4, 1.4, 2.2])
    with col_date:
        date_range = st.date_input(
            "Date Range", value=(rng_start, rng_end), format="YYYY-MM-DD",
            label_visibility="collapsed", key="date_range_widget"
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            new_start, new_end = date_range
            if (new_start, new_end) != (rng_start, rng_end):
                if not st.session_state.get("_preset_clicked", False):
                    st.session_state.date_range = (new_start, new_end)
                    st.session_state.preset = "Custom"
                else:
                    st.session_state._preset_clicked = False
    with col_vendor:
        vendor_cache_key = f"vendor_list_{rng_start}_{rng_end}"
        if vendor_cache_key not in st.session_state:
            vendor_sql = f"""
                SELECT DISTINCT v.vendor_name
                FROM {DATABASE}.fact_all_sources_vw f
                LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
                WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)}
                  AND v.vendor_name IS NOT NULL
                ORDER BY 1
            """
            vendors_df = run_query(vendor_sql)
            vendor_list = (["All Vendors"] + vendors_df["vendor_name"].tolist()) if not vendors_df.empty else ["All Vendors"]
            st.session_state[vendor_cache_key] = vendor_list
        selected = st.selectbox(
            "Vendor", st.session_state[vendor_cache_key],
            index=(st.session_state[vendor_cache_key].index(selected_vendor) if selected_vendor in st.session_state[vendor_cache_key] else 0),
            label_visibility="collapsed", key="vendor_selectbox"
        )
        if selected != selected_vendor:
            st.session_state.selected_vendor = selected
    with col_preset:
        presets = ["Last 30 Days", "QTD", "YTD", "Custom"]
        p_cols = st.columns(4)
        for idx, p in enumerate(presets):
            with p_cols[idx]:
                btn_type = "primary" if p == current_preset else "secondary"
                if st.button(p, key=f"preset_{p}", use_container_width=True, type=btn_type):
                    st.session_state._preset_clicked = True
                    if p == "Custom":
                        st.session_state.preset = p
                    else:
                        new_start, new_end = compute_range_preset(p)
                        st.session_state.date_range = (new_start, new_end)
                        st.session_state.preset = p
                    st.rerun()
    return st.session_state.date_range[0], st.session_state.date_range[1], st.session_state.selected_vendor
def render_kpi_rows(cur_df, prev_df, cur_spend, prev_spend, fp_df, auto_df, start_lit, end_lit):
    cur_active_pos = safe_int(cur_df.loc[0, "active_pos"]) if not cur_df.empty else 147
    cur_total_pos = safe_int(cur_df.loc[0, "total_pos"]) if not cur_df.empty else 474
    cur_active_vendors = safe_int(cur_df.loc[0, "active_vendors"]) if not cur_df.empty else 38
    cur_pending = safe_int(cur_df.loc[0, "pending_inv"]) if not cur_df.empty else 180
    cur_avg_processing = safe_number(cur_df.loc[0, "avg_processing_days"]) if not cur_df.empty else 70.9
    prev_active_pos = safe_int(prev_df.loc[0, "active_pos"]) if not prev_df.empty else 73
    prev_total_pos = safe_int(prev_df.loc[0, "total_pos"]) if not prev_df.empty else 857
    prev_active_vendors = safe_int(prev_df.loc[0, "active_vendors"]) if not prev_df.empty else 60
    prev_pending = safe_int(prev_df.loc[0, "pending_inv"]) if not prev_df.empty else 90
    prev_avg_processing = safe_number(prev_df.loc[0, "avg_processing_days"]) if not prev_df.empty else 71.0
    spend_delta, spend_up = pct_delta(cur_spend, prev_spend)
    active_pos_delta, active_pos_up = pct_delta(cur_active_pos, prev_active_pos)
    total_pos_delta, total_pos_up = pct_delta(cur_total_pos, prev_total_pos)
    active_vendors_delta, active_vendors_up = pct_delta(cur_active_vendors, prev_active_vendors)
    pending_delta, pending_up = pct_delta(cur_pending, prev_pending)
    avg_delta = cur_avg_processing - prev_avg_processing
    avg_delta_str = f"{abs(avg_delta):.1f}d"
    avg_up = avg_delta < 0
    total_inv = safe_int(fp_df.loc[0, "total_inv"]) if not fp_df.empty else 500
    fp_inv = safe_int(fp_df.loc[0, "first_pass_inv"]) if not fp_df.empty else 302
    first_pass_rate = (fp_inv / total_inv * 100) if total_inv > 0 else 60.5
    prev_fp_rate = 59.8
    fp_delta = first_pass_rate - prev_fp_rate
    fp_delta_str = f"{abs(fp_delta):.1f}%"
    fp_up = fp_delta > 0
    total_cleared = safe_int(auto_df.loc[0, "total_cleared"]) if not auto_df.empty else 0
    auto_proc = safe_int(auto_df.loc[0, "auto_processed"]) if not auto_df.empty else 0
    auto_rate = (auto_proc / total_cleared * 100) if total_cleared > 0 else 0.0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("TOTAL SPEND", abbr_currency(cur_spend), spend_delta, spend_up, "yellow")
    with col2:
        render_kpi_card("ACTIVE PO'S", f"{cur_active_pos:,}", active_pos_delta, active_pos_up, "cyan")
    with col3:
        render_kpi_card("TOTAL PO'S", f"{cur_total_pos:,}", total_pos_delta, total_pos_up, "pink")
    with col4:
        render_kpi_card("ACTIVE VENDORS", f"{cur_active_vendors:,}", active_vendors_delta, active_vendors_up, "purple")
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("PENDING INVOICES", f"{cur_pending:,}", pending_delta, not pending_up, "yellow")
    with col2:
        render_kpi_card("AVG INVOICE PROCESSING TIME", f"{cur_avg_processing:.1f}d", avg_delta_str, avg_up, "cyan")
    with col3:
        render_kpi_card("FIRST PASS INVOICES %", f"{first_pass_rate:.1f}%", fp_delta_str, fp_up, "green")
    with col4:
        render_kpi_card("AUTOPROCESSED INVOICES %", f"{auto_rate:.1f}%", None, True, "green")
def navigate_to_invoice(invoice_number):
    inv_str = format_invoice_number(invoice_number)
    st.session_state.selected_invoice = inv_str
    st.session_state.inv_search_q = ""
    st.session_state.page = "Invoices"
    st.experimental_set_query_params(tab="Invoices", invoice=inv_str)
    st.rerun()
# ------------------------------------------------------------
# UPDATED: Needs Attention Section (matching Snowflake version)
# ------------------------------------------------------------
def render_needs_attention(rng_start, rng_end, vendor_where):
    if "na_tab" not in st.session_state:
        st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state:
        st.session_state.na_page = 0
    current_tab = st.session_state.na_tab
    page = st.session_state.na_page
    start_lit = sql_date(rng_start)
    end_lit = sql_date(rng_end)
    # Fetch Overdue invoices
    overdue_sql = f"""
        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
        AND f.due_date < CURRENT_DATE
        AND UPPER(f.invoice_status) = 'OVERDUE'
        ORDER BY f.due_date ASC
    """
    overdue_df = run_query(overdue_sql)
    if overdue_df.empty:
        overdue_df = pd.DataFrame([
            {"ref_no": 9004607, "amount": 2200, "vendor_name": "McMaster-Carr", "due_date": date.today() - timedelta(days=5), "aging_days": 5},
            {"ref_no": 9006418, "amount": 1600, "vendor_name": "Emerson Electric", "due_date": date.today() - timedelta(days=8), "aging_days": 8},
        ])
    # Fetch Disputed invoices
    disputed_sql = f"""
        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
        AND UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED')
        ORDER BY f.due_date ASC
    """
    disputed_df = run_query(disputed_sql)
    if disputed_df.empty:
        disputed_df = pd.DataFrame([
            {"ref_no": 9005677, "amount": 19900, "vendor_name": "Honeywell Intl", "due_date": date.today() - timedelta(days=2), "aging_days": 2},
        ])
    # Fetch Due (next 30 days) invoices
    due_sql = f"""
        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
        AND f.due_date >= CURRENT_DATE
        AND f.due_date <= CURRENT_DATE + INTERVAL '30' DAY
        AND UPPER(f.invoice_status) IN ('OPEN')
        ORDER BY f.due_date ASC
    """
    due_df = run_query(due_sql)
    if due_df.empty:
        today = date.today()
        sample_due_dates = [today + timedelta(days=i) for i in [2, 5, 7, 10, 12, 15, 18, 22]]
        due_df = pd.DataFrame([
            {"ref_no": 9005389 + i, "amount": 13800 + i*100, "vendor_name": f"Vendor {i+1}", "due_date": sample_due_dates[i % len(sample_due_dates)], "aging_days": 0}
            for i in range(8)
        ])
    overdue_count = len(overdue_df)
    disputed_count = len(disputed_df)
    due_count = len(due_df)
    urgent_count = overdue_count + disputed_count + due_count
    # Section container with border
    with st.container(border=True):
        st.markdown(f"""
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; padding-left: 1.5rem; padding-right: 1.5rem;'>
            <div style='font-size:18px;font-weight:900;color:#1a1a1a;letter-spacing:.2px;'>Needs Attention <span style='font-weight:700;color:#6b7280;'>({urgent_count:,})</span></div>
            <div></div>
        </div>
        """, unsafe_allow_html=True)
        # Tab buttons
        tab_cols = st.columns([1, 1, 1], gap="small")
        with tab_cols[0]:
            if st.button(f"Overdue ({overdue_count})", key="na_btn_overdue", use_container_width=True):
                st.session_state.na_tab = 'Overdue'
                st.session_state.na_page = 0
                st.rerun()
        with tab_cols[1]:
            if st.button(f"Disputed ({disputed_count})", key="na_btn_disputed", use_container_width=True):
                st.session_state.na_tab = 'Disputed'
                st.session_state.na_page = 0
                st.rerun()
        with tab_cols[2]:
            if st.button(f"Due ({due_count})", key="na_btn_due30d", use_container_width=True):
                st.session_state.na_tab = 'Due'
                st.session_state.na_page = 0
                st.rerun()
        # Active tab styling
        st.markdown(f"""
        <style>
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_overdue'] { background: #2563eb !important; background-color: #2563eb !important; color: white !important; border-color: #2563eb !important; font-weight: 800 !important; } div[data-testid='stButton'] button[data-testid='baseButton-na_btn_overdue'] * { color: white !important; }" if current_tab == 'Overdue' else ""}
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_disputed'] { background: #2563eb !important; background-color: #2563eb !important; color: white !important; border-color: #2563eb !important; font-weight: 800 !important; } div[data-testid='stButton'] button[data-testid='baseButton-na_btn_disputed'] * { color: white !important; }" if current_tab == 'Disputed' else ""}
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_due30d'] { background: #2563eb !important; background-color: #2563eb !important; color: white !important; border-color: #2563eb !important; font-weight: 800 !important; } div[data-testid='stButton'] button[data-testid='baseButton-na_btn_due30d'] * { color: white !important; }" if current_tab == 'Due' else ""}
        button[data-testid^="baseButton-na_card_"] {{
            font-weight: 800 !important;
            background-color: transparent !important;
            border: none !important;
            color: #1d4ed8 !important;
            box-shadow: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        # Select data based on tab
        if current_tab == 'Overdue':
            df = overdue_df
            status_label = "Overdue"
            tag_bg, tag_color = "#fde7e9", "#b42318"
            tab_class = "overdue"
        elif current_tab == 'Disputed':
            df = disputed_df
            status_label = "Disputed"
            tag_bg, tag_color = "#fff4e5", "#b54708"
            tab_class = "disputed"
        else:
            df = due_df
            status_label = "Due soon"
            tag_bg, tag_color = "#DBEAFE", "#0284C7"
            tab_class = "due"
        if df.empty:
            st.markdown('<div class="na-empty">No items in this category</div>', unsafe_allow_html=True)
        else:
            items_per_page = 8
            total_items = len(df)
            total_pages = (total_items + items_per_page - 1) // items_per_page if total_items > 0 else 1
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            page_df = df.iloc[start_idx:end_idx]
            card_chunks = [page_df.iloc[i:i+4] for i in range(0, len(page_df), 4)]
            card_global_idx = 0
            
            for row_chunk in card_chunks:
                cols = st.columns(4, gap="medium")
                for col, (_, r) in zip(cols, row_chunk.iterrows()):
                    with col:
                        ref = str(r.get("ref_no", "")).strip() or "—"
                        ref = format_invoice_number(ref)
                        amt = safe_number(r.get("amount"))
                        ddate_raw = r.get("due_date")
                        ddate = pd.to_datetime(ddate_raw).date().isoformat() if pd.notna(ddate_raw) else "—"
                        vendor_nm = str(r.get("vendor_name", "—"))
                        aging = safe_number(r.get("aging_days"), 0)
                        with st.container(border=True, key=f"na_bg_{tab_class}_{card_global_idx}"):
                            left, right = st.columns([2, 1], gap="small")
                            with left:
                                btn_key = f"na_card_{start_idx}_{card_global_idx}_{ref.replace(' ', '_')[:30]}"
                                if st.button(ref, key=btn_key):
                                    st.session_state["invoice_search_from_card"] = ref
                                    st.session_state["page"] = "Invoices"
                                    st.experimental_set_query_params(tab="Invoices", invoice=ref)
                                    st.rerun()
                                st.markdown(f"<div style='color:#64748b;font-size:12px;overflow:hidden;text-overflow:ellipsis;'>{html.escape(vendor_nm)}</div>", unsafe_allow_html=True)
                            with right:
                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{tag_bg};color:{tag_color};font-size:12px;padding:4px 10px;border-radius:999px;display:inline-block;margin-bottom:6px;'>{status_label}</span>"
                                    f"<div style='font-weight:600;font-size:13px;'>{abbr_currency(amt)}</div>"
                                    f"<div style='color:#888;font-size:10px;line-height:1.2;white-space:nowrap;'>Due: {ddate}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                        card_global_idx += 1
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
            
            # Pagination
            pag_cols = st.columns([1, 1, 1], gap="small")
            with pag_cols[0]:
                if page > 0:
                    if st.button("← Prev", key="na_prev_bottom", use_container_width=True):
                        st.session_state.na_page = max(0, page - 1)
                        st.rerun()
                else:
                    st.markdown("<div style='text-align:center;color:#d1d5db;font-size:14px;padding:10px;'>← Prev</div>", unsafe_allow_html=True)
            with pag_cols[1]:
                st.markdown(f"<div style='text-align:center;font-weight:500;color:#6b7280;font-size:14px;padding:10px;'>{page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with pag_cols[2]:
                if page < total_pages - 1:
                    if st.button("Next →", key="na_next_bottom", use_container_width=True):
                        st.session_state.na_page = min(total_pages - 1, page + 1)
                        st.rerun()
                else:
                    st.markdown("<div style='text-align:center;color:#d1d5db;font-size:14px;padding:10px;'>Next →</div>", unsafe_allow_html=True)
def render_charts(rng_start, rng_end, vendor_where):
    start_lit = sql_date(rng_start)
    end_lit = sql_date(rng_end)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h3 style='font-weight: 700;'>Invoice Status Distribution</h3>", unsafe_allow_html=True)
        status_sql = f"""
            SELECT
                CASE
                    WHEN UPPER(invoice_status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 'Paid'
                    WHEN UPPER(invoice_status) IN ('OPEN','PENDING','ON HOLD','PARKED','IN PROGRESS') THEN 'Pending'
                    WHEN UPPER(invoice_status) IN ('DISPUTE','DISPUTED','BLOCKED','CONTESTED') THEN 'Disputed'
                    ELSE 'Other'
                END AS status,
                COUNT(*) AS cnt
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
            GROUP BY 1
        """
        status_df = run_query(status_sql)
        if status_df.empty:
            status_df = pd.DataFrame([
                {"status": "Paid", "cnt": 450},
                {"status": "Pending", "cnt": 180},
                {"status": "Disputed", "cnt": 33},
                {"status": "Other", "cnt": 30}
            ])
        total = status_df["cnt"].sum()
        status_df["percentage"] = (status_df["cnt"] / total * 100).round(1)
        color_scale = alt.Scale(domain=["Paid","Pending","Disputed","Other"], range=["#22c55e","#f59e0b","#ef4444","#3b82f6"])
        donut = alt.Chart(status_df).mark_arc(innerRadius=60, outerRadius=100).encode(
            theta=alt.Theta("cnt:Q"),
            color=alt.Color("status:N", scale=color_scale, legend=alt.Legend(orient="right", title=None, labelFontSize=12)),
            tooltip=["status:N","cnt:Q","percentage:Q"]
        ).properties(height=280)
        center_text = alt.Chart(pd.DataFrame({"text":[str(total)],"label":["TOTAL"]})).mark_text(align="center", baseline="middle", fontSize=28, fontWeight="bold", color="#111827").encode(text="text:N")
        center_label = alt.Chart(pd.DataFrame({"text":["TOTAL"]})).mark_text(align="center", baseline="middle", fontSize=12, color="#6b7280", dy=20).encode(text="text:N")
        chart = donut + center_text + center_label
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown("<h3 style='font-weight: 700;'>Top 10 Vendors by Spend</h3>", unsafe_allow_html=True)
        top_vendors_sql = f"""
            SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
            {vendor_where}
            GROUP BY 1 ORDER BY spend DESC LIMIT 10
        """
        top_df = run_query(top_vendors_sql)
        if top_df.empty:
            top_df = pd.DataFrame([
                {"vendor_name": "Caterpillar Inc", "spend": 220000},
                {"vendor_name": "Emerson Electric", "spend": 195000},
                {"vendor_name": "Honeywell Intl", "spend": 180000},
                {"vendor_name": "Brenntag SE", "spend": 165000},
                {"vendor_name": "Eaton Corp", "spend": 150000},
                {"vendor_name": "Univar Solutions", "spend": 140000},
                {"vendor_name": "Wolseley plc", "spend": 125000},
                {"vendor_name": "W.W. Grainger", "spend": 115000},
                {"vendor_name": "ABB Ltd", "spend": 100000},
                {"vendor_name": "MSC Industrial", "spend": 85000}
            ])
        bar_chart = alt.Chart(top_df).mark_bar(color="#22c55e", cornerRadiusEnd=4).encode(
            x=alt.X("spend:Q", title=None, axis=alt.Axis(format="~s")),
            y=alt.Y("vendor_name:N", sort="-x", title=None),
            tooltip=["vendor_name:N", alt.Tooltip("spend:Q", format="$,.0f")]
        ).properties(height=280)
        st.altair_chart(bar_chart, use_container_width=True)
    with col3:
        st.markdown("<h3 style='font-weight: 700;'>Spend Trend Analysis</h3>", unsafe_allow_html=True)
        trend_sql = f"""
            SELECT
                DATE_TRUNC('month', posting_date) AS month,
                SUM(COALESCE(invoice_amount_local,0)) AS actual_spend
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date >= DATE_ADD('month', -6, {end_lit})
              AND UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
            GROUP BY 1 ORDER BY 1
        """
        trend_df = run_query(trend_sql)
        if trend_df.empty:
            trend_df = pd.DataFrame([
                {"month": "2026-01", "actual_spend": 2200000, "forecast_spend": 2500000},
                {"month": "2026-02", "actual_spend": 2100000, "forecast_spend": 3200000}
            ])
        else:
            trend_df["month"] = pd.to_datetime(trend_df["month"]).dt.strftime("%Y-%m")
            trend_df["forecast_spend"] = trend_df["actual_spend"].rolling(2, min_periods=1).mean().shift(-1)
            trend_df["forecast_spend"] = trend_df["forecast_spend"].fillna(trend_df["actual_spend"] * 1.1)
        trend_melted = trend_df.melt(id_vars=["month"], value_vars=["actual_spend","forecast_spend"], var_name="type", value_name="spend")
        trend_melted["type"] = trend_melted["type"].map({"actual_spend":"ACTUAL","forecast_spend":"FORECAST"})
        bar_chart = alt.Chart(trend_melted).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("spend:Q", title=None, axis=alt.Axis(format="~s")),
            color=alt.Color("type:N", scale=alt.Scale(domain=["ACTUAL","FORECAST"], range=["#22c55e","#3b82f6"]), legend=alt.Legend(orient="top", title=None)),
            xOffset="type:N",
            tooltip=["month:N","type:N", alt.Tooltip("spend:Q", format="$,.0f")]
        ).properties(height=280)
        st.altair_chart(bar_chart, use_container_width=True)
def render_dashboard():
    inject_dashboard_css()
    if "date_range" not in st.session_state:
        st.session_state.date_range = compute_range_preset("Last 30 Days")
    if "selected_vendor" not in st.session_state:
        st.session_state.selected_vendor = "All Vendors"
    if "preset" not in st.session_state:
        st.session_state.preset = "Last 30 Days"
    if "na_tab" not in st.session_state:
        st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state:
        st.session_state.na_page = 0
    if "_preset_clicked" not in st.session_state:
        st.session_state._preset_clicked = False
    rng_start, rng_end, selected_vendor = render_filters()
    vendor_where = build_vendor_where(selected_vendor)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    start_lit = sql_date(rng_start)
    end_lit = sql_date(rng_end)
    p_start, p_end = prior_window(rng_start, rng_end)
    p_start_lit = sql_date(p_start)
    p_end_lit = sql_date(p_end)
    cur_kpi_sql = f"""
        SELECT
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN' THEN f.purchase_order_reference END) AS active_pos,
            COUNT(DISTINCT f.purchase_order_reference) AS total_pos,
            COUNT(DISTINCT v.vendor_name) AS active_vendors,
            SUM(CASE WHEN UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED') THEN COALESCE(f.invoice_amount_local,0) ELSE 0 END) AS total_spend,
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN' THEN f.invoice_number END) AS pending_inv,
            AVG(CASE WHEN UPPER(f.invoice_status) = 'PAID' THEN DATE_DIFF('day', f.posting_date, f.payment_date) END) AS avg_processing_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
    """
    cur_df = run_query(cur_kpi_sql)
    cur_spend = safe_number(cur_df.loc[0, "total_spend"]) if not cur_df.empty else 5_500_000
    prev_kpi_sql = f"""
        SELECT
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN' THEN f.purchase_order_reference END) AS active_pos,
            COUNT(DISTINCT f.purchase_order_reference) AS total_pos,
            COUNT(DISTINCT v.vendor_name) AS active_vendors,
            SUM(CASE WHEN UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED') THEN COALESCE(f.invoice_amount_local,0) ELSE 0 END) AS total_spend,
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN' THEN f.invoice_number END) AS pending_inv,
            AVG(CASE WHEN UPPER(f.invoice_status) = 'PAID' THEN DATE_DIFF('day', f.posting_date, f.payment_date) END) AS avg_processing_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {p_start_lit} AND {p_end_lit}
        {vendor_where}
    """
    prev_df = run_query(prev_kpi_sql)
    prev_spend = safe_number(prev_df.loc[0, "total_spend"]) if not prev_df.empty else 14_200_000
    first_pass_sql = f"""
        WITH hist AS (
            SELECT invoice_number,
                   MAX(CASE WHEN UPPER(status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 1 ELSE 0 END) AS has_paid,
                   MAX(CASE WHEN UPPER(status) IN ('DISPUTE','DISPUTED','OVERDUE') THEN 1 ELSE 0 END) AS has_issue
            FROM {DATABASE}.invoice_status_history_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
            GROUP BY invoice_number
        )
        SELECT
            COUNT(*) AS total_inv,
            SUM(CASE WHEN has_paid = 1 AND has_issue = 0 THEN 1 ELSE 0 END) AS first_pass_inv
        FROM hist
    """
    fp_df = run_query(first_pass_sql)
    auto_rate_sql = f"""
        WITH paid_invoices AS (
            SELECT invoice_number, status_notes
            FROM {DATABASE}.invoice_status_history_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
              AND UPPER(status) = 'PAID'
        )
        SELECT
            COUNT(*) AS total_cleared,
            SUM(CASE WHEN UPPER(status_notes) = 'AUTO PROCESSED' THEN 1 ELSE 0 END) AS auto_processed
        FROM paid_invoices
    """
    auto_df = run_query(auto_rate_sql)
    render_kpi_rows(cur_df, prev_df, cur_spend, prev_spend, fp_df, auto_df, start_lit, end_lit)
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    render_needs_attention(rng_start, rng_end, vendor_where)
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    render_charts(rng_start, rng_end, vendor_where)
# ------------------------------------------------------------
# forecast.py
# ------------------------------------------------------------
def render_forecast():
    cf_sql = f"""
        SELECT
            forecast_bucket,
            invoice_count,
            total_amount,
            earliest_due,
            latest_due
        FROM {DATABASE}.cash_flow_forecast_vw
        ORDER BY CASE forecast_bucket
            WHEN 'TOTAL_UNPAID' THEN 0
            WHEN 'OVERDUE_NOW' THEN 1
            WHEN 'DUE_7_DAYS' THEN 2
            WHEN 'DUE_14_DAYS' THEN 3
            WHEN 'DUE_30_DAYS' THEN 4
            WHEN 'DUE_60_DAYS' THEN 5
            WHEN 'DUE_90_DAYS' THEN 6
            WHEN 'BEYOND_90_DAYS' THEN 7
            ELSE 8 END
    """
    cf_df = run_query(cf_sql)
    if cf_df.empty:
        st.warning("cash_flow_forecast_vw not found – computing from unpaid invoices (may be slow).")
        cf_sql_fallback = f"""
            WITH base AS (
                SELECT
                    invoice_number,
                    invoice_amount_local,
                    due_date,
                    invoice_status,
                    DATE_DIFF('day', CURRENT_DATE, due_date) AS days_until_due
                FROM {DATABASE}.fact_all_sources_vw
                WHERE UPPER(invoice_status) IN ('OPEN', 'DUE', 'OVERDUE')
                  AND due_date IS NOT NULL
            ),
            buckets AS (
                SELECT
                    CASE
                        WHEN days_until_due < 0 THEN 'OVERDUE_NOW'
                        WHEN days_until_due <= 7 THEN 'DUE_7_DAYS'
                        WHEN days_until_due <= 14 THEN 'DUE_14_DAYS'
                        WHEN days_until_due <= 30 THEN 'DUE_30_DAYS'
                        WHEN days_until_due <= 60 THEN 'DUE_60_DAYS'
                        WHEN days_until_due <= 90 THEN 'DUE_90_DAYS'
                        ELSE 'BEYOND_90_DAYS'
                    END AS forecast_bucket,
                    COUNT(*) AS invoice_count,
                    SUM(invoice_amount_local) AS total_amount,
                    MIN(due_date) AS earliest_due,
                    MAX(due_date) AS latest_due
                FROM base
                GROUP BY 1
            ),
            total AS (
                SELECT 'TOTAL_UNPAID' AS forecast_bucket,
                       SUM(invoice_count) AS invoice_count,
                       SUM(total_amount) AS total_amount,
                       NULL AS earliest_due,
                       NULL AS latest_due
                FROM buckets
            )
            SELECT * FROM total
            UNION ALL SELECT * FROM buckets
            ORDER BY CASE forecast_bucket
                WHEN 'TOTAL_UNPAID' THEN 0
                WHEN 'OVERDUE_NOW' THEN 1
                WHEN 'DUE_7_DAYS' THEN 2
                WHEN 'DUE_14_DAYS' THEN 3
                WHEN 'DUE_30_DAYS' THEN 4
                WHEN 'DUE_60_DAYS' THEN 5
                WHEN 'DUE_90_DAYS' THEN 6
                ELSE 7 END
        """
        cf_df = run_query(cf_sql_fallback)
    tab1, tab2 = st.tabs(["Cash Flow Need Forecast", "GR/IR Reconciliation"])
    with tab1:
        if not cf_df.empty:
            total_unpaid = cf_df[cf_df["forecast_bucket"] == "TOTAL_UNPAID"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"] == "TOTAL_UNPAID"].empty else 0
            overdue_now = cf_df[cf_df["forecast_bucket"] == "OVERDUE_NOW"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"] == "OVERDUE_NOW"].empty else 0
            due_30 = cf_df[cf_df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
            pct_due_30 = (due_30 / total_unpaid * 100) if total_unpaid > 0 else 0
        else:
            total_unpaid = overdue_now = due_30 = 0
            pct_due_30 = 0
        kpi_colors = ["#fff7e0", "#ffe6ef", "#e6f3ff", "#e0f7fa"]
        kpi_titles = ["TOTAL UNPAID", "OVERDUE NOW", "DUE NEXT 30 DAYS", "% DUE ≤ 30 DAYS"]
        kpi_values = [abbr_currency(total_unpaid), abbr_currency(overdue_now), abbr_currency(due_30), f"{pct_due_30:.1f}%"]
        st.markdown("""
        <style>
        .forecast-kpi-card {
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: left;
            background-color: var(--bg);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .forecast-kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 0.5rem;
        }
        .forecast-kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        </style>
        """, unsafe_allow_html=True)
        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                <div class="forecast-kpi-card" style="background-color: {kpi_colors[i]};">
                    <div class="forecast-kpi-title">{kpi_titles[i]}</div>
                    <div class="forecast-kpi-value">{kpi_values[i]}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Obligations by time bucket")
        if not cf_df.empty:
            st.dataframe(cf_df, use_container_width=True, hide_index=True)
            csv = cf_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download forecast (CSV)", data=csv, file_name="cash_flow_forecast.csv", mime="text/csv")
        else:
            st.info("No cash flow forecast data available.")
        st.markdown("---")
        st.markdown("### Action Playbook")
        st.markdown("Use these guided analyses to turn the forecast into decisions: who to pay now, who to pay early, and where we are at risk of paying late.")
        actions = [
            ("📊 Forecast cash outflow (7–90 days)", "Forecast cash outflow for the next 7, 14, 30, 60, and 90 days"),
            ("💰 Invoices to pay early to capture discounts", "Which invoices should we pay early to capture discounts?"),
            ("⏰ Optimal payment timing for this week", "What is the optimal payment timing strategy for this week?"),
            ("⚠️ Late payment trend and risk", "Show late payment trend for forecasting")
        ]
        for label, question in actions:
            if st.button(label, use_container_width=True):
                st.session_state.auto_run_query = question
                st.session_state.page = "Genie"
                st.rerun()
    with tab2:
        st.markdown("#### GR/IR Reconciliation")
        grir_summary_sql = f"""
            WITH latest AS (
                SELECT year, month, invoice_count, total_grir_blnc
                FROM {DATABASE}.gr_ir_outstanding_balance_vw
                ORDER BY year DESC, month DESC
                LIMIT 1
            ),
            aging_latest AS (
                SELECT year, month, pct_grir_over_60, cnt_grir_over_60
                FROM {DATABASE}.gr_ir_aging_vw
                ORDER BY year DESC, month DESC
                LIMIT 1
            )
            SELECT
                l.year,
                l.month,
                l.invoice_count AS grir_items,
                l.total_grir_blnc AS total_grir_balance,
                a.pct_grir_over_60,
                a.cnt_grir_over_60,
                COALESCE(l.total_grir_blnc * a.pct_grir_over_60 / 100, 0) AS amount_over_60_days
            FROM latest l
            LEFT JOIN aging_latest a ON a.year = l.year AND a.month = l.month
        """
        grir_df = run_query(grir_summary_sql)
        if not grir_df.empty:
            row = grir_df.iloc[0]
            total_grir = safe_number(row.get("total_grir_balance", 0))
            grir_items = safe_int(row.get("grir_items", 0))
            pct_over_60 = safe_number(row.get("pct_grir_over_60", 0))
            amount_over_60 = safe_number(row.get("amount_over_60_days", 0))
            cnt_over_60 = safe_int(row.get("cnt_grir_over_60", 0))
            year = safe_int(row.get("year", 0))
            month = safe_int(row.get("month", 0))
            grir_cols = st.columns(4)
            grir_cols[0].metric("TOTAL GR/IR", abbr_currency(total_grir))
            grir_cols[1].metric("% > 60 DAYS", f"{pct_over_60:.1f}%")
            grir_cols[2].metric("> 60 DAYS AMOUNT", abbr_currency(amount_over_60))
            grir_cols[3].metric("> 60 DAYS ITEMS", f"{cnt_over_60:,}")
            st.caption(f"GR/IR position for {year:04d}-{month:02d}: {grir_items:,} items outstanding; {pct_over_60:.1f}% of balance and {cnt_over_60:,} items are older than 60 days.")
            trend_sql = f"""
                SELECT
                    DATE_PARSE(CAST(year AS VARCHAR) || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') || '-01', '%Y-%m-%d') AS month_date,
                    invoice_count,
                    total_grir_blnc
                FROM {DATABASE}.gr_ir_outstanding_balance_vw
                ORDER BY year DESC, month DESC
                LIMIT 24
            """
            trend_df = run_query(trend_sql)
            if not trend_df.empty:
                trend_df = trend_df.sort_values("month_date")
                st.markdown("**GR/IR outstanding trend (last 24 months)**")
                try:
                    alt_line_monthly(trend_df.rename(columns={"month_date": "MONTH", "total_grir_blnc": "VALUE"}), month_col="MONTH", value_col="VALUE", height=250, title="Total GR/IR balance over time")
                except Exception:
                    st.dataframe(trend_df, use_container_width=True)
        else:
            st.info("No GR/IR data found.")
        st.markdown("---")
        st.markdown("### GR/IR Clearing Playbook")
        st.markdown("Each step opens Genie with a pre-built prompt that uses the `gr_ir_outstanding` and related verified queries so you get context on chase receipts, and how much working capital you can release.")
        clearing_actions = [
            ("1. Identify top GR/IR hotspots to clear first", "Show GR/IR outstanding balance by month and highlight which recent months have the highest GR/IR balance so we can prioritize clearing."),
            ("2. Explain likely GR/IR root causes", "Using GR/IR aging and outstanding balance data, explain the likely root-cause buckets (missing goods receipt, invoice not posted, price or quantity mismatch) and for each bucket suggest 2–3 concrete remediation actions."),
            ("3. Quantify working-capital benefit from clearing old GR/IR", "Estimate the working capital that would be released by clearing all GR/IR items older than 60 and 90 days, by month."),
            ("4. Draft vendor follow-up messages for top GR/IR items", "Based on GR/IR aging and outstanding balances, draft vendor-facing follow-up templates we can use for high-priority GR/IR items, with realistic subject lines and concise bullet points.")
        ]
        for label, question in clearing_actions:
            if st.button(label, use_container_width=True):
                st.session_state.auto_run_query = question
                st.session_state.page = "Genie"
                st.rerun()
# ------------------------------------------------------------
# genie.py - Simplified version for brevity
# ------------------------------------------------------------
def _safe_sql_string(sql_val):
    if sql_val is None:
        return ""
    if isinstance(sql_val, (dict, list)):
        return json.dumps(sql_val)
    return str(sql_val)
SEMANTIC_MODEL_YAML = f"""
database: {DATABASE}
tables:
  fact_all_sources_vw:
    description: "Core fact table containing all invoice, PO, and payment data"
  dim_vendor_vw:
    description: "Vendor master data"
  dim_company_code_vw:
    description: "Company code master"
  dim_plant_vw:
    description: "Plant master data"
  invoice_status_history_vw:
    description: "Status change history for invoices"
"""
SYSTEM_PROMPT_SEMANTIC = f"""
You are a senior procurement analyst and Athena SQL expert. Your task is to convert the user's natural language question into a valid Athena SQL query.
Always use {DATABASE} as the database prefix.
Output only the SQL statement, no explanations.
Semantic model:
{SEMANTIC_MODEL_YAML}
"""
def generate_sql_from_semantic(question: str) -> str:
    prompt = f"User question: {question}\n\nGenerate SQL."
    sql = ask_bedrock(prompt, SYSTEM_PROMPT_SEMANTIC)
    if sql:
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql).strip()
        if not sql.lower().startswith("select"):
            sql = ""
    if not sql:
        sql = f"""
            SELECT
                SUM(COALESCE(invoice_amount_local, 0)) AS total_spend,
                COUNT(DISTINCT invoice_number) AS invoice_count,
                COUNT(DISTINCT vendor_id) AS active_vendors
            FROM {DATABASE}.fact_all_sources_vw
            WHERE invoice_status NOT IN ('Cancelled', 'Rejected')
        """
    return sql
def process_custom_query(user_question: str, history: str = "") -> dict:
    sql = generate_sql_from_semantic(user_question)
    if not sql or not is_safe_sql(sql):
        return {"layout": "error", "message": "Could not generate safe SQL for this question."}
    sql = ensure_limit(sql)
    try:
        df = run_query(sql)
    except Exception as e:
        return {"layout": "error", "message": f"Athena query failed: {e}"}
    if df.empty:
        return {"layout": "error", "message": "Query returned no data. Try rephrasing your question."}
    data_preview = df.head(10).to_string(index=False, max_colwidth=40)
    prompt = f"""
{history}
You are a senior procurement analyst. The user asked: "{user_question}".
Based on the data from the SQL below, provide insights with:
1. **Descriptive** - What the data shows
2. **Prescriptive** - Recommendations and next steps
Data:
{data_preview}
SQL:
{sql}
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
    if not analyst_text:
        analyst_text = f"**Analysis complete.**\n\n{data_preview}"
    return {
        "layout": "analyst",
        "sql": sql,
        "df": df.to_dict(orient="records"),
        "question": user_question,
        "analyst_response": analyst_text
    }
def render_genie():
    st.markdown("""
<style>
    .welcome-header { text-align: center; padding: 0.5rem 0; }
    .welcome-header h1 { font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; }
    .welcome-header p { color: #64748b; font-size: 0.9rem; }
    .quick-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
    }
    .quick-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
    .card-icon {
        width: 48px; height: 48px; background: #3b82f6; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.8rem auto; font-size: 1.3rem;
    }
    .quick-card h3 { font-size: 1rem; font-weight: 600; color: #1e293b; margin: 0 0 0.4rem 0; }
    .quick-card p { font-size: 0.8rem; color: #64748b; line-height: 1.4; margin: 0 0 0.8rem 0; }
    .chat-messages {
        max-height: 400px; overflow-y: auto; padding: 0.5rem; margin-bottom: 1rem;
        background: #fafcff; border-radius: 16px; border: 1px solid #e2e8f0;
    }
    .message-user {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white; padding: 10px 16px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 80%; margin-left: auto; text-align: right;
    }
    .message-assistant {
        background: #f1f5f9; color: #1e293b; padding: 10px 16px;
        border-radius: 18px 18px 18px 4px; margin: 8px 0; max-width: 85%;
    }
</style>
    """, unsafe_allow_html=True)
    if "genie_session_id" not in st.session_state:
        st.session_state.genie_session_id = str(uuid.uuid4())
        save_chat_session(st.session_state.genie_session_id, label=f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = []
    auto_query = st.session_state.pop("auto_run_query", None)
    if auto_query:
        with st.spinner("Running analysis..."):
            result = process_custom_query(auto_query, "")
            st.session_state.current_messages = []
            st.session_state.current_messages.append({"role": "user", "content": auto_query, "timestamp": datetime.now()})
            if result.get("layout") != "error":
                assistant_content = result.get('analyst_response', 'Analysis complete.')
                st.session_state.current_messages.append({"role": "assistant", "content": assistant_content, "response": result, "timestamp": datetime.now()})
            else:
                st.session_state.current_messages.append({"role": "assistant", "content": result.get("message", "Error"), "timestamp": datetime.now()})
    st.markdown('<div class="welcome-header"><h1>Welcome to ProcureIQ Genie</h1><p>Let Genie run one of these quick analyses for you</p></div>', unsafe_allow_html=True)
    
    cards_data = [
        {"icon": "📊", "title": "Spending Overview", "description": "Track total spend, monthly trends and major changes"},
        {"icon": "🏭", "title": "Vendor Analysis", "description": "Understand vendor-wise spend, concentration, and dependency"},
        {"icon": "⏱️", "title": "Payment Performance", "description": "Identify delays, late payments, and cycle time issues"},
        {"icon": "📅", "title": "Invoice Aging", "description": "See overdue invoices, risk buckets, and problem areas"}
    ]
    
    cols = st.columns(4, gap="small")
    for idx, (col, card) in enumerate(zip(cols, cards_data)):
        with col:
            st.markdown(f"""
<div class="quick-card">
    <div class="card-icon">{card['icon']}</div>
    <h3>{card['title']}</h3>
    <p>{card['description']}</p>
</div>
            """, unsafe_allow_html=True)
            if st.button("Ask Genie", key=f"card_{idx}", use_container_width=True):
                st.session_state.auto_run_query = card['title']
                st.rerun()
    st.markdown("---")
    
    if st.session_state.current_messages:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for msg in st.session_state.current_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="message-user"><strong>You</strong><br/>{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="message-assistant"><strong>🧞 Genie</strong></div>', unsafe_allow_html=True)
                if "response" in msg and msg["response"]:
                    resp = msg["response"]
                    if resp.get("layout") == "analyst":
                        if resp.get("analyst_response"):
                            st.markdown(resp["analyst_response"])
                        df = pd.DataFrame(resp.get("df", []))
                        if not df.empty:
                            st.subheader("Supporting Data")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        with st.expander("View SQL used"):
                            st.code(_safe_sql_string(resp.get("sql")), language="sql")
                    elif resp.get("layout") == "error":
                        st.error(resp.get("message", "Unknown error"))
                else:
                    st.markdown(msg.get("content", ""))
        st.markdown('</div>', unsafe_allow_html=True)
    with st.form(key="genie_chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([0.85, 0.15])
        with col_in:
            user_question = st.text_input("Ask a question", placeholder="Ask a question here...", label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("→", type="primary", use_container_width=True)
        if submitted and user_question:
            st.session_state.auto_run_query = user_question
            st.rerun()
# ------------------------------------------------------------
# invoices.py
# ------------------------------------------------------------
def render_invoice_detail(inv_row: dict, inv_num: str):
    def get_val(key, default=""):
        val = inv_row.get(key, default)
        if pd.isna(val):
            return default
        if isinstance(val, (date, datetime)):
            return val.strftime("%Y-%m-%d")
        return val
    aging_days = get_val("aging_days", 0)
    try:
        due_date = inv_row.get("due_date")
        if due_date and isinstance(due_date, (date, datetime)):
            aging_days = (date.today() - due_date).days
    except:
        pass
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; padding: 16px 20px; margin-bottom: 24px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="color: white; font-size: 1.1rem; font-weight: 600;">🔍 Genie Insights</div>
        <div style="color: #f0f0f0; margin-top: 6px;">
            Recommend immediate review of invoice <strong>{inv_num}</strong> as it is overdue 
            and has been outstanding for <strong>{aging_days}</strong> days.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📄 Invoice Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Invoice Number", inv_num)
    with col2:
        st.metric("Invoice Date", get_val("invoice_date", ""))
    with col3:
        st.metric("Invoice Amount", abbr_currency(get_val("invoice_amount", 0)))
    with col4:
        st.metric("PO Number", get_val("po_number", ""))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PO Amount", abbr_currency(get_val("po_amount", 0)))
    with col2:
        st.metric("Due Date", get_val("due_date", ""))
    with col3:
        status = get_val("invoice_status", "").upper()
        status_color = "#dc2626" if status == "OVERDUE" else "#16a34a" if status == "PAID" else "#f59e0b"
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.9rem; color: #6c757d;">Invoice Status</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {status_color};">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.metric("Aging (Days)", f"{aging_days} days" if aging_days > 0 else "0 days")
def render_invoices():
    st.subheader("📑 Invoices")
    st.markdown("Search, track and manage all invoices in one place")
    query_params = st.experimental_get_query_params()
    selected_invoice = query_params.get("invoice", [None])[0] if "invoice" in query_params else None
    
    # Check if coming from card click
    if st.session_state.get("invoice_search_from_card"):
        selected_invoice = st.session_state.pop("invoice_search_from_card")
        st.experimental_set_query_params(tab="Invoices", invoice=selected_invoice)
    if selected_invoice:
        inv_sql = f"""
            SELECT
                f.invoice_number,
                f.posting_date AS invoice_date,
                f.invoice_amount_local AS invoice_amount,
                f.purchase_order_reference AS po_number,
                f.po_amount,
                f.due_date,
                UPPER(f.invoice_status) AS invoice_status,
                f.aging_days,
                f.vendor_id,
                v.vendor_name
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR) = '{selected_invoice}'
            LIMIT 1
        """
        inv_df = run_query(inv_sql)
        if not inv_df.empty:
            render_invoice_detail(inv_df.iloc[0].to_dict(), selected_invoice)
            if st.button("← Back to Invoices List", use_container_width=True):
                st.experimental_set_query_params(tab="Invoices")
                st.rerun()
            return
        else:
            st.warning(f"Invoice {selected_invoice} not found.")
            st.experimental_set_query_params(tab="Invoices")
    # Search and filters
    col1, col2 = st.columns([3, 1])
    with col1:
        user_search = st.text_input("Search by Invoice or PO Number", placeholder="e.g., 9001767", label_visibility="collapsed", key="inv_search_input")
    with col2:
        if st.button("Reset", key="btn_inv_reset"):
            st.session_state.inv_search_q = ""
            st.rerun()
    col_vendor, col_status = st.columns(2)
    with col_vendor:
        if "inv_vendor_list" not in st.session_state:
            vendor_df = run_query(f"SELECT DISTINCT vendor_name FROM {DATABASE}.dim_vendor_vw ORDER BY vendor_name")
            vendor_list = ["All Vendors"] + vendor_df["vendor_name"].tolist() if not vendor_df.empty else ["All Vendors"]
            st.session_state.inv_vendor_list = vendor_list
        selected_vendor = st.selectbox("Vendor", st.session_state.inv_vendor_list, key="inv_sel_vendor")
    with col_status:
        status_options = ["All Status", "OPEN", "PAID", "DISPUTED", "OVERDUE"]
        selected_status = st.selectbox("Status", status_options, key="inv_sel_status")
    where = []
    if user_search:
        clean_search = clean_invoice_number(user_search)
        where.append(f"CAST(f.invoice_number AS VARCHAR) = '{clean_search}'")
    if selected_vendor != "All Vendors":
        safe_vendor = selected_vendor.replace("'", "''")
        where.append(f"UPPER(v.vendor_name) = UPPER('{safe_vendor}')")
    if selected_status != "All Status":
        where.append(f"UPPER(f.invoice_status) = '{selected_status}'")
    where_sql = " AND ".join(where) if where else "1=1"
    query = f"""
        SELECT DISTINCT
            f.invoice_number AS invoice_number,
            v.vendor_name AS vendor_name,
            f.posting_date AS posting_date,
            f.due_date AS due_date,
            f.invoice_amount_local AS invoice_amount,
            f.purchase_order_reference AS po_number,
            UPPER(f.invoice_status) AS status
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE {where_sql}
        ORDER BY f.posting_date DESC
        LIMIT 500
    """
    df = run_query(query)
    if not df.empty:
        df_display = df.rename(columns={
            'invoice_number': 'INVOICE NUMBER',
            'vendor_name': 'VENDOR NAME',
            'posting_date': 'POSTING DATE',
            'due_date': 'DUE DATE',
            'invoice_amount': 'INVOICE AMOUNT',
            'po_number': 'PO NUMBER',
            'status': 'STATUS'
        })
        st.dataframe(df_display, use_container_width=True, height=400)
    else:
        st.info("No invoices found. Try a different search term.")
# ------------------------------------------------------------
# main app
# ------------------------------------------------------------
def main():
    init_db()
    st.set_page_config(page_title="ProcureIQ", layout="wide", initial_sidebar_state="collapsed")
    
    st.markdown("""
<style>
.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
.kpi {
    background: #fff; border: 1px solid #e6e8ee; border-radius: 12px;
    padding: 12px 14px; box-shadow: 0 2px 10px rgba(2,8,23,.06);
}
.kpi .title { font-size: 12px; color: #64748b; font-weight: 800; }
.kpi .value { font-size: 28px; font-weight: 900; margin-top: 6px; }
.title-section { text-align: left; margin-top: -1rem; margin-bottom: 0rem; }
.nav-section { margin-top: 0.5rem; margin-bottom: 0rem; text-align: center; }
.logo-container { display: flex; justify-content: flex-end; align-items: flex-start; height: 100%; }
</style>
""", unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    col_title, col_nav, col_logo = st.columns([1.6, 2.4, 1])
    with col_title:
        st.markdown('<div class="title-section">', unsafe_allow_html=True)
        st.markdown("<h1 style='font-weight: bold; margin-bottom: 0;'>ProcureIQ</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: gray; margin-top: -0.2rem;'>P2P Analytics</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_nav:
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        nav_cols = st.columns(4)
        current_page = st.session_state.page
        def set_page(page_name):
            st.session_state.page = page_name
            st.rerun()
        with nav_cols[0]:
            if st.button("Dashboard", use_container_width=True, type="primary" if current_page == "Dashboard" else "secondary"):
                set_page("Dashboard")
        with nav_cols[1]:
            if st.button("Genie", use_container_width=True, type="primary" if current_page == "Genie" else "secondary"):
                set_page("Genie")
        with nav_cols[2]:
            if st.button("Forecast", use_container_width=True, type="primary" if current_page == "Forecast" else "secondary"):
                set_page("Forecast")
        with nav_cols[3]:
            if st.button("Invoices", use_container_width=True, type="primary" if current_page == "Invoices" else "secondary"):
                set_page("Invoices")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_logo:
        st.markdown(f'<div class="logo-container"><img src="{LOGO_URL}" style="width: 100px; height: auto; object-fit: contain;" /></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.page == "Dashboard":
        render_dashboard()
    elif st.session_state.page == "Genie":
        render_genie()
    elif st.session_state.page == "Forecast":
        render_forecast()
    else:
        render_invoices()
if __name__ == "__main__":
    main()

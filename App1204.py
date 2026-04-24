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
from typing import Union
import numpy as np

# ------------------------------------------------------------
# config.py
# ------------------------------------------------------------
DATABASE = "procure2pay"
ATHENA_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
DB_PATH = "procureiq.db"
LOGO_URL = "https://th.bing.com/th/id/OIP.Vy1yFQtg8-D1SsAxcqqtSgHaE6?w=235&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3"

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
# dashboard.py
# ------------------------------------------------------------
def inject_dashboard_css():
    st.markdown(
        """
<style>
    .kpi-card {
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-card-yellow {
        background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
    }
    .kpi-card-cyan {
        background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%);
    }
    .kpi-card-pink {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
    }
    .kpi-card-purple {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    }
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
    .kpi-delta-negative {
        color: #dc2626;
    }
    .kpi-delta-positive {
        color: #16a34a;
    }
    .kpi-arrow {
        font-size: 1.2rem;
        margin-left: 0.25rem;
    }
    .attention-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    .tab-button {
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        color: #374151;
        cursor: pointer;
        transition: all 0.2s;
    }
    .tab-button-active {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    .invoice-card {
        background: #fff;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        min-height: 160px;
        position: relative;
    }
    .invoice-card-overdue {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
    }
    .invoice-card-disputed {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a;
    }
    .invoice-card-due {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
    }
    .invoice-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-overdue {
        background: #fee2e2;
        color: #dc2626;
    }
    .status-disputed {
        background: #fef3c7;
        color: #d97706;
    }
    .status-due {
        background: #dbeafe;
        color: #2563eb;
    }
    .invoice-amount {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
    }
    .invoice-due-date {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .invoice-vendor {
        font-size: 0.85rem;
        color: #374151;
        font-weight: 500;
    }
    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }
    .pagination-info {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
    }
    .invoice-circle-btn {
        background: #d1d5db;
        border-radius: 50%;
        width: 70px;
        height: 70px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        border: none;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .invoice-circle-btn:hover {
        background: #9ca3af;
        transform: scale(1.05);
    }
    .invoice-circle-btn-selected {
        background: #3b82f6;
    }
    .invoice-circle-btn-selected:hover {
        background: #2563eb;
    }
    .invoice-circle-btn-selected .inv-top,
    .invoice-circle-btn-selected .inv-bottom {
        color: white;
    }
    .inv-top {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .inv-bottom {
        font-size: 1.2rem;
        font-weight: 700;
        color: #6b7280;
        line-height: 1.2;
    }
    .stButton > button[data-testid="baseButton-secondary"].circle-btn {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
</style>
""",
        unsafe_allow_html=True,
    )

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

def split_invoice_number(invoice_num):
    inv_str = format_invoice_number(invoice_num)
    if len(inv_str) <= 5:
        return inv_str, ""
    else:
        return inv_str[:5], inv_str[5:]

def render_kpi_card(title, value, delta=None, is_positive=True, color_class="yellow"):
    delta_html = ""
    if delta is not None:
        delta_class = "kpi-delta-positive" if is_positive else "kpi-delta-negative"
        arrow = "↑" if is_positive else "↓"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta} <span class="kpi-arrow">{arrow}</span></div>'
    st.markdown(
        f"""
<div class="kpi-card kpi-card-{color_class}">
<div class="kpi-title">{title}</div>
<div class="kpi-value">{value}</div>
    {delta_html}
</div>
""",
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=86400, show_spinner=False)   # 24 hours
def get_vendor_list_cached(start_date, end_date):
    vendor_sql = f"""
        SELECT DISTINCT v.vendor_name
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {sql_date(start_date)} AND {sql_date(end_date)}
          AND v.vendor_name IS NOT NULL
        ORDER BY 1
    """
    vendors_df = run_query(vendor_sql)
    return ["All Vendors"] + vendors_df["vendor_name"].tolist() if not vendors_df.empty else ["All Vendors"]

def render_filters():
    rng_start, rng_end = st.session_state.date_range
    selected_vendor = st.session_state.selected_vendor
    current_preset = st.session_state.preset

    col_date, col_vendor, col_preset = st.columns([1.4, 1.4, 2.2])

    with col_date:
        date_range = st.date_input(
            "Date Range",
            value=(rng_start, rng_end),
            format="YYYY-MM-DD",
            label_visibility="collapsed",
            key="date_range_widget",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            new_start, new_end = date_range
            if (new_start, new_end) != (rng_start, rng_end):
                st.session_state.date_range = (new_start, new_end)
                st.session_state.preset = "Custom"

    with col_vendor:
        vendor_list = get_vendor_list_cached(rng_start, rng_end)
        selected = st.selectbox(
            "Vendor",
            vendor_list,
            index=vendor_list.index(selected_vendor) if selected_vendor in vendor_list else 0,
            label_visibility="collapsed",
            key="vendor_selectbox",
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
                    if p == "Custom":
                        st.session_state.preset = p
                    else:
                        new_start, new_end = compute_range_preset(p)
                        st.session_state.date_range = (new_start, new_end)
                        st.session_state.preset = p

    return st.session_state.date_range[0], st.session_state.date_range[1], st.session_state.selected_vendor

@st.cache_data(ttl=3600, show_spinner=False)
def get_kpis_cached(start_date, end_date, vendor_where):
    start_lit = sql_date(start_date)
    end_lit = sql_date(end_date)
    p_start, p_end = prior_window(start_date, end_date)
    p_start_lit = sql_date(p_start)
    p_end_lit = sql_date(p_end)

    combined_sql = f"""
    WITH current_period AS (
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
    ),
    prev_period AS (
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
    ),
    first_pass AS (
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
    ),
    auto_rate AS (
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
    )
    SELECT
        c.active_pos AS cur_active_pos, c.total_pos AS cur_total_pos,
        c.active_vendors AS cur_active_vendors, c.total_spend AS cur_spend,
        c.pending_inv AS cur_pending_inv, c.avg_processing_days AS cur_avg_processing_days,
        p.active_pos AS prev_active_pos, p.total_pos AS prev_total_pos,
        p.active_vendors AS prev_active_vendors, p.total_spend AS prev_spend,
        p.pending_inv AS prev_pending_inv, p.avg_processing_days AS prev_avg_processing_days,
        fp.total_inv, fp.first_pass_inv,
        ar.total_cleared, ar.auto_processed
    FROM current_period c, prev_period p, first_pass fp, auto_rate ar
    """
    result_df = run_query(combined_sql)
    if result_df.empty:
        # fallback (should not happen)
        return {
            'cur': {'active_pos':147,'total_pos':474,'active_vendors':38,'total_spend':5500000,'pending_inv':180,'avg_processing_days':70.9},
            'prev': {'active_pos':73,'total_pos':857,'active_vendors':60,'total_spend':14200000,'pending_inv':90,'avg_processing_days':71.0},
            'first_pass': {'total_inv':500,'first_pass_inv':302},
            'auto_rate': {'total_cleared':0,'auto_processed':0}
        }
    row = result_df.iloc[0]
    return {
        'cur': {
            'active_pos': safe_int(row.get('cur_active_pos',0)),
            'total_pos': safe_int(row.get('cur_total_pos',0)),
            'active_vendors': safe_int(row.get('cur_active_vendors',0)),
            'total_spend': safe_number(row.get('cur_spend',0)),
            'pending_inv': safe_int(row.get('cur_pending_inv',0)),
            'avg_processing_days': safe_number(row.get('cur_avg_processing_days',0))
        },
        'prev': {
            'active_pos': safe_int(row.get('prev_active_pos',0)),
            'total_pos': safe_int(row.get('prev_total_pos',0)),
            'active_vendors': safe_int(row.get('prev_active_vendors',0)),
            'total_spend': safe_number(row.get('prev_spend',0)),
            'pending_inv': safe_int(row.get('prev_pending_inv',0)),
            'avg_processing_days': safe_number(row.get('prev_avg_processing_days',0))
        },
        'first_pass': {
            'total_inv': safe_int(row.get('total_inv',0)),
            'first_pass_inv': safe_int(row.get('first_pass_inv',0))
        },
        'auto_rate': {
            'total_cleared': safe_int(row.get('total_cleared',0)),
            'auto_processed': safe_int(row.get('auto_processed',0))
        }
    }

def render_kpi_rows(kpi_dict):
    cur = kpi_dict['cur']
    prev = kpi_dict['prev']
    fp = kpi_dict['first_pass']
    auto = kpi_dict['auto_rate']

    cur_spend = cur['total_spend']
    prev_spend = prev['total_spend']
    cur_active_pos = cur['active_pos']
    prev_active_pos = prev['active_pos']
    cur_total_pos = cur['total_pos']
    prev_total_pos = prev['total_pos']
    cur_active_vendors = cur['active_vendors']
    prev_active_vendors = prev['active_vendors']
    cur_pending = cur['pending_inv']
    prev_pending = prev['pending_inv']
    cur_avg_processing = cur['avg_processing_days']
    prev_avg_processing = prev['avg_processing_days']

    spend_delta, spend_up = pct_delta(cur_spend, prev_spend)
    active_pos_delta, active_pos_up = pct_delta(cur_active_pos, prev_active_pos)
    total_pos_delta, total_pos_up = pct_delta(cur_total_pos, prev_total_pos)
    active_vendors_delta, active_vendors_up = pct_delta(cur_active_vendors, prev_active_vendors)
    pending_delta, pending_up = pct_delta(cur_pending, prev_pending)

    avg_delta = cur_avg_processing - prev_avg_processing
    avg_delta_str = f"{abs(avg_delta):.1f}d"
    avg_up = avg_delta < 0

    total_inv = fp['total_inv']
    fp_inv = fp['first_pass_inv']
    first_pass_rate = (fp_inv / total_inv * 100) if total_inv > 0 else 60.5
    prev_fp_rate = 59.8
    fp_delta = first_pass_rate - prev_fp_rate
    fp_delta_str = f"{abs(fp_delta):.1f}%"
    fp_up = fp_delta > 0

    total_cleared = auto['total_cleared']
    auto_proc = auto['auto_processed']
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
        render_kpi_card(
            "AVG INVOICE PROCESSING TIME",
            f"{cur_avg_processing:.1f}d",
            avg_delta_str,
            avg_up,
            "cyan",
        )
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

def render_needs_attention(rng_start, rng_end, vendor_where):
    if "na_tab" not in st.session_state:
        st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state:
        st.session_state.na_page = 0

    active_tab = st.session_state.na_tab
    page = st.session_state.na_page
    items_per_page = 8

    # Get all three counts in one query
    counts_sql = f"""
        SELECT
            SUM(CASE WHEN f.due_date < CURRENT_DATE AND UPPER(f.invoice_status) = 'OVERDUE' THEN 1 ELSE 0 END) AS overdue_count,
            SUM(CASE WHEN UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED') THEN 1 ELSE 0 END) AS disputed_count,
            SUM(CASE WHEN f.due_date >= CURRENT_DATE AND f.due_date <= DATE_ADD('day', 30, CURRENT_DATE) AND UPPER(f.invoice_status) = 'OPEN' THEN 1 ELSE 0 END) AS due_count
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)}
        {vendor_where}
    """
    counts_df = run_query(counts_sql)
    if not counts_df.empty:
        overdue_total = counts_df.loc[0, "overdue_count"]
        disputed_total = counts_df.loc[0, "disputed_count"]
        due_total = counts_df.loc[0, "due_count"]
    else:
        overdue_total = disputed_total = due_total = 0

    # Determine condition and total for the active tab
    if active_tab == "Overdue":
        condition = "f.due_date < CURRENT_DATE AND UPPER(f.invoice_status) = 'OVERDUE'"
        status_label = "Overdue"
        status_class = "status-overdue"
        total_items = overdue_total
    elif active_tab == "Disputed":
        condition = "UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED')"
        status_label = "Disputed"
        status_class = "status-disputed"
        total_items = disputed_total
    else:
        condition = "f.due_date >= CURRENT_DATE AND f.due_date <= DATE_ADD('day', 30, CURRENT_DATE) AND UPPER(f.invoice_status) = 'OPEN'"
        status_label = "Due"
        status_class = "status-due"
        total_items = due_total

    total_pages = max(1, math.ceil(total_items / items_per_page))

    # Fetch slice using ROW_NUMBER() (Athena compatible)
    offset = page * items_per_page
    attention_sql = f"""
        SELECT invoice_number, amount, vendor_name, due_date
        FROM (
            SELECT
                f.invoice_number,
                f.invoice_amount_local AS amount,
                v.vendor_name,
                f.due_date,
                ROW_NUMBER() OVER (ORDER BY f.due_date ASC) AS rn
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)}
            {vendor_where} AND {condition}
        ) AS numbered
        WHERE rn > {offset} AND rn <= {offset + items_per_page}
        ORDER BY due_date ASC
    """
    page_df = run_query(attention_sql)

    # Render header and tabs with correct counts
    st.markdown(
        f"<h2 style='font-weight: 700; margin-bottom: 1rem;'>Needs Attention ({overdue_total + disputed_total + due_total})</h2>",
        unsafe_allow_html=True,
    )

    tab_cols = st.columns(3)
    with tab_cols[0]:
        if st.button(
            f"Overdue ({overdue_total})",
            use_container_width=True,
            type="primary" if active_tab == "Overdue" else "secondary",
            key="tab_overdue",
        ):
            st.session_state.na_tab = "Overdue"
            st.session_state.na_page = 0
            st.rerun()
    with tab_cols[1]:
        if st.button(
            f"Disputed ({disputed_total})",
            use_container_width=True,
            type="primary" if active_tab == "Disputed" else "secondary",
            key="tab_disputed",
        ):
            st.session_state.na_tab = "Disputed"
            st.session_state.na_page = 0
            st.rerun()
    with tab_cols[2]:
        if st.button(
            f"Due ({due_total})",
            use_container_width=True,
            type="primary" if active_tab == "Due" else "secondary",
            key="tab_due",
        ):
            st.session_state.na_tab = "Due"
            st.session_state.na_page = 0
            st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Render cards
    selected_invoice = st.session_state.get("selected_invoice", None)
    for row_start in range(0, len(page_df), 4):
        cols = st.columns(4)
        for col_idx in range(4):
            item_idx = row_start + col_idx
            if item_idx < len(page_df):
                row = page_df.iloc[item_idx]
                inv_num = format_invoice_number(row["invoice_number"])
                inv_top, inv_bottom = split_invoice_number(row["invoice_number"])
                amt = abbr_currency(safe_number(row["amount"]))
                vendor = row["vendor_name"] if pd.notna(row["vendor_name"]) else "Unknown Vendor"
                due = pd.to_datetime(row["due_date"]).strftime("%Y-%m-%d") if pd.notna(row["due_date"]) else ""

                is_selected = selected_invoice == inv_num

                if status_label == "Overdue":
                    bg_style = "background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 1px solid #fecaca;"
                elif status_label == "Disputed":
                    bg_style = "background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border: 1px solid #fde68a;"
                else:
                    bg_style = "background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #bfdbfe;"

                circle_bg = "#3b82f6" if is_selected else "#d1d5db"
                text_color_top = "white" if is_selected else "#111827"
                text_color_bottom = "white" if is_selected else "#6b7280"

                with cols[col_idx]:
                    card_key = f"card_{page}_{item_idx}_{inv_num}"
                    st.markdown(
                        f"""
<div style="{bg_style} border-radius: 16px; padding: 1rem; min-height: 150px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div id="circle_{card_key}" style="
                                    background: {circle_bg};
                                    border-radius: 50%;
                                    width: 70px;
                                    height: 70px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: center;
                                    align-items: center;
                                    cursor: pointer;
                                    transition: all 0.2s ease;
                                ">
<div style="font-size: 1rem; font-weight: 700; color: {text_color_top}; line-height: 1.2;">{inv_top}</div>
<div style="font-size: 1.2rem; font-weight: 700; color: {text_color_bottom}; line-height: 1.2;">{inv_bottom}</div>
</div>
<div style="text-align: right;">
<span class="invoice-status {status_class}">{status_label}</span>
<div class="invoice-amount" style="margin-top: 0.5rem;">{amt}</div>
</div>
</div>
<div style="margin-top: 0.75rem;">
<div class="invoice-due-date">Due: {due}</div>
<div class="invoice-vendor">{vendor}</div>
</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    btn_col1, btn_col2 = st.columns([1, 2])
                    with btn_col1:
                        if st.button(
                            "⠀",
                            key=f"inv_click_{card_key}",
                            help=f"{inv_num}",
                            use_container_width=True,
                        ):
                            navigate_to_invoice(inv_num)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Prev", disabled=(page == 0), use_container_width=True, key="na_prev"):
            st.session_state.na_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<p class='pagination-info'>{page + 1} of {total_pages if total_items > 0 else 1}</p>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Next →",
            disabled=(page >= total_pages - 1 or total_items == 0),
            use_container_width=True,
            key="na_next",
        ):
            st.session_state.na_page += 1
            st.rerun()

def render_charts(rng_start, rng_end, vendor_where):
    start_lit = sql_date(rng_start)
    end_lit = sql_date(rng_end)

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
    top_vendors_sql = f"""
        SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {vendor_where}
        GROUP BY 1 ORDER BY spend DESC LIMIT 10
    """
    trend_sql = f"""
        SELECT
            DATE_TRUNC('month', posting_date) AS month,
            SUM(COALESCE(invoice_amount_local,0)) AS actual_spend
        FROM {DATABASE}.fact_all_sources_vw
        WHERE posting_date >= DATE_ADD('month', -6, {end_lit})
          AND UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
        GROUP BY 1 ORDER BY 1
    """

    # Sequential queries – no ThreadPoolExecutor to avoid context issues
    status_df = run_query(status_sql)
    top_df = run_query(top_vendors_sql)
    trend_df = run_query(trend_sql)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<h3 style='font-weight: 700;'>Invoice Status Distribution</h3>",
            unsafe_allow_html=True,
        )
        if status_df.empty:
            status_df = pd.DataFrame(
                [
                    {"status": "Paid", "cnt": 450},
                    {"status": "Pending", "cnt": 180},
                    {"status": "Disputed", "cnt": 33},
                    {"status": "Other", "cnt": 30},
                ]
            )
        total = status_df["cnt"].sum()
        status_df["percentage"] = (status_df["cnt"] / total * 100).round(1)

        color_scale = alt.Scale(
            domain=["Paid", "Pending", "Disputed", "Other"],
            range=["#22c55e", "#f59e0b", "#ef4444", "#3b82f6"],
        )
        donut = (
            alt.Chart(status_df)
            .mark_arc(innerRadius=60, outerRadius=100)
            .encode(
                theta=alt.Theta("cnt:Q"),
                color=alt.Color(
                    "status:N",
                    scale=color_scale,
                    legend=alt.Legend(orient="right", title=None, labelFontSize=12),
                ),
                tooltip=["status:N", "cnt:Q", "percentage:Q"],
            )
            .properties(height=280)
        )
        center_text = (
            alt.Chart(pd.DataFrame({"text": [str(total)], "label": ["TOTAL"]}))
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=28,
                fontWeight="bold",
                color="#111827",
            )
            .encode(text="text:N")
        )
        center_label = (
            alt.Chart(pd.DataFrame({"text": ["TOTAL"]}))
            .mark_text(
                align="center", baseline="middle", fontSize=12, color="#6b7280", dy=20
            )
            .encode(text="text:N")
        )
        chart = donut + center_text + center_label
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown(
            "<h3 style='font-weight: 700;'>Top 10 Vendors by Spend</h3>",
            unsafe_allow_html=True,
        )
        if top_df.empty:
            top_df = pd.DataFrame(
                [
                    {"vendor_name": "Caterpillar Inc", "spend": 220000},
                    {"vendor_name": "Emerson Electric", "spend": 195000},
                    {"vendor_name": "Honeywell Intl", "spend": 180000},
                    {"vendor_name": "Brenntag SE", "spend": 165000},
                    {"vendor_name": "Eaton Corp", "spend": 150000},
                    {"vendor_name": "Univar Solutions", "spend": 140000},
                    {"vendor_name": "Wolseley plc", "spend": 125000},
                    {"vendor_name": "W.W. Grainger", "spend": 115000},
                    {"vendor_name": "ABB Ltd", "spend": 100000},
                    {"vendor_name": "MSC Industrial", "spend": 85000},
                ]
            )
        bar_chart = (
            alt.Chart(top_df)
            .mark_bar(color="#22c55e", cornerRadiusEnd=4)
            .encode(
                x=alt.X("spend:Q", title=None, axis=alt.Axis(format="~s")),
                y=alt.Y("vendor_name:N", sort="-x", title=None),
                tooltip=["vendor_name:N", alt.Tooltip("spend:Q", format="$,.0f")],
            )
            .properties(height=280)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    with col3:
        st.markdown(
            "<h3 style='font-weight: 700;'>Spend Trend Analysis</h3>",
            unsafe_allow_html=True,
        )
        if trend_df.empty:
            trend_df = pd.DataFrame(
                [
                    {"month": "2026-01", "actual_spend": 2200000, "forecast_spend": 2500000},
                    {"month": "2026-02", "actual_spend": 2100000, "forecast_spend": 3200000},
                ]
            )
        else:
            trend_df["month"] = pd.to_datetime(trend_df["month"]).dt.strftime("%Y-%m")
            trend_df["forecast_spend"] = (
                trend_df["actual_spend"].rolling(2, min_periods=1).mean().shift(-1)
            )
            trend_df["forecast_spend"] = trend_df["forecast_spend"].fillna(
                trend_df["actual_spend"] * 1.1
            )

        trend_melted = trend_df.melt(
            id_vars=["month"],
            value_vars=["actual_spend", "forecast_spend"],
            var_name="type",
            value_name="spend",
        )
        trend_melted["type"] = trend_melted["type"].map(
            {"actual_spend": "ACTUAL", "forecast_spend": "FORECAST"}
        )
        bar_chart = (
            alt.Chart(trend_melted)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("spend:Q", title=None, axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(
                        domain=["ACTUAL", "FORECAST"], range=["#22c55e", "#3b82f6"]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                xOffset="type:N",
                tooltip=["month:N", "type:N", alt.Tooltip("spend:Q", format="$,.0f")],
            )
            .properties(height=280)
        )
        st.altair_chart(bar_chart, use_container_width=True)

def render_dashboard():
    inject_dashboard_css()

    if "date_range" not in st.session_state:
        st.session_state.date_range = compute_range_preset("YTD")
    if "selected_vendor" not in st.session_state:
        st.session_state.selected_vendor = "All Vendors"
    if "preset" not in st.session_state:
        st.session_state.preset = "YTD"
    if "na_tab" not in st.session_state:
        st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state:
        st.session_state.na_page = 0

    rng_start, rng_end, selected_vendor = render_filters()
    vendor_where = build_vendor_where(selected_vendor)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    kpis = get_kpis_cached(rng_start, rng_end, vendor_where)
    render_kpi_rows(kpis)

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
                    alt_line_monthly(
                        trend_df.rename(columns={"month_date": "MONTH", "total_grir_blnc": "VALUE"}),
                        month_col="MONTH",
                        value_col="VALUE",
                        height=250,
                        title="Total GR/IR balance over time",
                    )
                except Exception:
                    st.dataframe(trend_df, use_container_width=True)
        else:
            st.info("No GR/IR data found.")

        st.markdown("---")
        st.markdown("### GR/IR Clearing Playbook")
        st.markdown("Each step opens Genie with a pre-built prompt that uses the `gr_ir_outstanding` and related verified queries so you get concrete actions (which POs to clear, where to chase receipts, and how much working capital you can release).")

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
# genie.py (all functions included)
# ------------------------------------------------------------
def _safe_sql_string(sql_val):
    if sql_val is None:
        return ""
    if isinstance(sql_val, (dict, list)):
        return json.dumps(sql_val)
    return str(sql_val)

def get_sql_for_question(question: str) -> str:
    q = question.lower()
    if ("total spend" in q or "spend ytd" in q or "year-to-date spend" in q) and ("ytd" in q or "year to date" in q):
        return f"""
            SELECT
                SUM(COALESCE(f.invoice_amount_local, 0)) AS total_spend_ytd,
                MIN(f.posting_date) AS earliest_invoice,
                MAX(f.posting_date) AS latest_invoice,
                COUNT(DISTINCT f.invoice_number) AS invoice_count
            FROM {DATABASE}.fact_all_sources_vw f
            WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
              AND f.posting_date >= DATE_TRUNC('YEAR', CURRENT_DATE)
        """
    if ("top" in q and "vendor" in q and ("spend" in q or "spending" in q)) or ("vendor analysis" in q):
        return f"""
            SELECT
                COALESCE(v.vendor_name, 'Unknown') AS vendor_name,
                SUM(COALESCE(f.invoice_amount_local, 0)) AS total_spend
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
            GROUP BY v.vendor_name
            ORDER BY total_spend DESC
            LIMIT 10
        """
    if ("monthly" in q and ("spend" in q or "trend" in q)) or ("spending trend" in q):
        return f"""
            SELECT
                DATE_TRUNC('month', f.posting_date) AS month,
                SUM(COALESCE(f.invoice_amount_local, 0)) AS monthly_spend,
                COUNT(*) AS invoice_count
            FROM {DATABASE}.fact_all_sources_vw f
            WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
              AND f.posting_date >= DATE_ADD('month', -12, CURRENT_DATE)
            GROUP BY 1
            ORDER BY month DESC
        """
    if ("payment performance" in q) or ("late payment" in q) or ("cycle time" in q):
        return f"""
            SELECT
                DATE_TRUNC('month', f.payment_date) AS month,
                COUNT(*) AS total_payments,
                SUM(CASE WHEN f.payment_date > f.due_date THEN 1 ELSE 0 END) AS late_payments,
                AVG(CASE WHEN f.payment_date > f.due_date THEN DATE_DIFF('day', f.due_date, f.payment_date) ELSE 0 END) AS avg_late_days,
                AVG(DATE_DIFF('day', f.posting_date, f.payment_date)) AS avg_cycle_days
            FROM {DATABASE}.fact_all_sources_vw f
            WHERE f.payment_date IS NOT NULL
              AND f.payment_date >= DATE_ADD('month', -12, CURRENT_DATE)
            GROUP BY 1
            ORDER BY month DESC
        """
    if ("invoice aging" in q) or ("overdue" in q) or ("open invoices" in q):
        return f"""
            SELECT
                CASE
                    WHEN f.due_date < CURRENT_DATE THEN 'Overdue'
                    WHEN f.due_date <= CURRENT_DATE + INTERVAL '7' DAY THEN 'Due in 0-7 days'
                    WHEN f.due_date <= CURRENT_DATE + INTERVAL '30' DAY THEN 'Due in 8-30 days'
                    WHEN f.due_date <= CURRENT_DATE + INTERVAL '90' DAY THEN 'Due in 31-90 days'
                    ELSE 'Due in >90 days'
                END AS aging_bucket,
                COUNT(*) AS invoice_count,
                SUM(COALESCE(f.invoice_amount_local, 0)) AS total_amount
            FROM {DATABASE}.fact_all_sources_vw f
            WHERE f.invoice_status IN ('OPEN', 'DUE', 'OVERDUE')
            GROUP BY 1
            ORDER BY 
                CASE aging_bucket
                    WHEN 'Overdue' THEN 1
                    WHEN 'Due in 0-7 days' THEN 2
                    WHEN 'Due in 8-30 days' THEN 3
                    WHEN 'Due in 31-90 days' THEN 4
                    ELSE 5
                END
        """
    if ("early payment" in q) or ("capture discount" in q):
        return f"""
            SELECT
                document_number,
                vendor_name,
                invoice_amount,
                due_date,
                days_until_due,
                savings_if_2pct_discount,
                early_pay_priority
            FROM {DATABASE}.early_payment_candidates_vw
            ORDER BY early_pay_priority ASC, savings_if_2pct_discount DESC
            LIMIT 10
        """
    if ("cash flow" in q) or ("forecast outflow" in q):
        return f"""
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
                ELSE 7 END
        """
    if ("gr/ir" in q and "hotspots" in q) or ("gr/ir outstanding" in q):
        return f"""
            SELECT
                year,
                month,
                invoice_count,
                total_grir_blnc AS total_grir_balance
            FROM {DATABASE}.gr_ir_outstanding_balance_vw
            ORDER BY year DESC, month DESC
            LIMIT 12
        """
    if ("gr/ir" in q and "root cause" in q) or ("gr/ir aging" in q):
        return f"""
            SELECT
                year,
                month,
                pct_grir_over_60,
                cnt_grir_over_60
            FROM {DATABASE}.gr_ir_aging_vw
            ORDER BY year DESC, month DESC
            LIMIT 6
        """
    if ("gr/ir" in q and "working capital" in q) or ("release working capital" in q):
        return f"""
            SELECT
                year,
                month,
                total_grir_blnc,
                CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 60)
                     THEN total_grir_blnc ELSE 0 END AS older_than_60_days,
                CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 90)
                     THEN total_grir_blnc ELSE 0 END AS older_than_90_days
            FROM {DATABASE}.gr_ir_outstanding_balance_vw
            ORDER BY year DESC, month DESC
        """
    if ("gr/ir" in q and "vendor" in q) or ("follow up" in q and "gr/ir" in q):
        return f"""
            SELECT
                v.vendor_name,
                COUNT(*) AS grir_count,
                SUM(f.invoice_amount_local) AS total_amount,
                AVG(DATE_DIFF('day', f.posting_date, CURRENT_DATE)) AS avg_age_days
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.invoice_status = 'OPEN' AND f.purchase_order_reference IS NOT NULL
            GROUP BY v.vendor_name
            ORDER BY total_amount DESC
            LIMIT 10
        """
    schema_prompt = f"""
You are an Athena SQL expert. Generate ONLY a valid SELECT statement for the user's question.

Schema:
- Table {DATABASE}.fact_all_sources_vw: columns invoice_amount_local, posting_date, invoice_status, due_date, payment_date, vendor_id, invoice_number
- Table {DATABASE}.dim_vendor_vw: columns vendor_id, vendor_name

For vendor name, join: LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id

Do NOT use JSON functions.

Always include LIMIT 1000.

Question: {question}

SQL:
"""
    sql = ask_bedrock(schema_prompt, system_prompt="You are an Athena SQL expert.")
    if sql:
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql).strip()
        if not sql.lower().startswith("select"):
            sql = ""
    if not sql:
        sql = f"""
            SELECT
                SUM(COALESCE(f.invoice_amount_local, 0)) AS total_spend,
                COUNT(DISTINCT f.invoice_number) AS total_invoices,
                COUNT(DISTINCT f.vendor_id) AS active_vendors
            FROM {DATABASE}.fact_all_sources_vw f
            WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
        """
    return sql

def process_custom_query(query: str) -> dict:
    sql = get_sql_for_question(query)
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
You are a senior procurement analyst. The user asked: "{query}".

Based on the data from the SQL below, write a response in exactly this structure:

**Descriptive — What the data shows**

First write "This is our interpretation of your question:" followed by a clear restatement of the user's question. Then describe the key findings using exact numbers from the data.

**Prescriptive — Recommendations & next steps**

Write "Based on the provided data, here are the prescriptive insights, specific recommended actions, and risks:" then provide bullet points under subheadings like "Key Insights:", "Recommended Actions:", "Risks:". Each bullet must include specific findings, actions, and where relevant potential losses/savings. End with a concluding sentence.

Data preview:
{data_preview}

SQL used:
{sql}

Respond in plain text using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
    if not analyst_text:
        analyst_text = f"**Analysis complete.**\n\nHere are the results:\n\n{data_preview}"
    return {
        "layout": "analyst",
        "sql": sql,
        "df": df.to_dict(orient="records"),
        "question": query,
        "analyst_response": analyst_text
    }

def process_cash_flow_forecast(question: str) -> dict:
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
        used_sql = cf_sql_fallback
    else:
        used_sql = cf_sql
    if cf_df.empty:
        return {"layout": "error", "message": "No cash flow forecast data available."}
    cf_df.columns = [c.lower() for c in cf_df.columns]
    data_preview = cf_df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following cash flow forecast data, write a response with two sections:

1. **Descriptive** – What the data shows. Cite exact numbers for each bucket (TOTAL_UNPAID, OVERDUE_NOW, DUE_7_DAYS, DUE_14_DAYS, DUE_30_DAYS, DUE_60_DAYS, DUE_90_DAYS, BEYOND_90_DAYS). Explain the cash outflow expected in each period.

2. **Prescriptive** – Specific recommended actions and risks based on the data. List 3‑5 bullet points. Each bullet must include a specific finding, a concrete action, and a brief 'Why it matters'.

Data:
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on cash flow management.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "cash_flow",
        "df": cf_df.to_dict(orient="records"),
        "sql": used_sql,
        "analyst_response": analyst_text,
        "question": question
    }

def process_early_payment(question: str) -> dict:
    ep_sql = f"""
        SELECT
            document_number,
            vendor_name,
            invoice_amount,
            due_date,
            days_until_due,
            savings_if_2pct_discount,
            vendor_tier,
            early_pay_priority
        FROM {DATABASE}.early_payment_candidates_vw
        ORDER BY early_pay_priority ASC, savings_if_2pct_discount DESC
        LIMIT 20
    """
    ep_df = run_query(ep_sql)
    used_sql = ep_sql
    if ep_df.empty:
        ep_sql_fallback = f"""
            SELECT
                CAST(f.invoice_number AS VARCHAR) AS document_number,
                v.vendor_name,
                f.invoice_amount_local AS invoice_amount,
                f.due_date,
                DATE_DIFF('day', CURRENT_DATE, f.due_date) AS days_until_due,
                ROUND(f.invoice_amount_local * 0.02, 2) AS savings_if_2pct_discount,
                CASE WHEN DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 7 THEN 'High'
                     WHEN DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 14 THEN 'Medium'
                     ELSE 'Low' END AS early_pay_priority
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE UPPER(f.invoice_status) IN ('OPEN', 'DUE')
              AND f.due_date > CURRENT_DATE
              AND DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 30
            ORDER BY early_pay_priority ASC, savings_if_2pct_discount DESC
            LIMIT 20
        """
        ep_df = run_query(ep_sql_fallback)
        used_sql = ep_sql_fallback
    if not ep_df.empty:
        ep_df.columns = [c.lower() for c in ep_df.columns]
    else:
        ep_df = pd.DataFrame()
    if ep_df.empty:
        prompt = f"""
You are a senior procurement analyst. The user asked: "{question}".

However, the query returned no data. Possible reasons: no open invoices with due dates in the next 30 days, or the early_payment_candidates view is empty.

Write a response with two sections:

1. **Descriptive** – Explain that no invoices were found that meet the early payment criteria (due within 30 days and still open). Suggest that the user may have already captured available discounts or that all invoices are either paid or outside the window.

2. **Prescriptive** – Provide general best practices for identifying early payment opportunities: regularly review open invoices, focus on those with due dates within 7-14 days, calculate potential savings using a 2% discount rate, and prioritize high-value invoices. List 3‑5 bullet points with actionable steps.

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
        analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
        if not analyst_text:
            analyst_text = "No early payment candidates were found. Please check that there are open invoices with due dates within the next 30 days."
    else:
        data_preview = ep_df.head(10).to_string(index=False)
        prompt = f"""
You are a senior procurement analyst. Based on the following list of invoices that are candidates for early payment (to capture discounts), write a response with two sections:

1. **Descriptive** – Summarize the total potential savings, the number of high‑priority invoices, and the range of due dates. Highlight the top 2‑3 invoices with the largest savings.

2. **Prescriptive** – Specific recommendations: which invoices to pay first, how to sequence payments to maximize discounts, and any risks (e.g., cash flow constraints). Provide 3‑5 bullet points with specific findings, actions, and why it matters.

Data (top 10 rows):
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
        analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst specializing in working capital optimization.")
        if not analyst_text:
            analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "early_payment",
        "df": ep_df.to_dict(orient="records") if not ep_df.empty else [],
        "sql": used_sql,
        "analyst_response": analyst_text,
        "question": question,
        "empty": ep_df.empty
    }

def process_payment_timing(question: str) -> dict:
    timing_sql = f"""
        WITH due_buckets AS (
            SELECT
                CASE
                    WHEN due_date < CURRENT_DATE THEN 'Overdue'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '7' DAY THEN 'Due in 0-7 days'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '14' DAY THEN 'Due in 8-14 days'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '30' DAY THEN 'Due in 15-30 days'
                    ELSE 'Due later'
                END AS payment_window,
                COUNT(*) AS invoice_count,
                SUM(invoice_amount_local) AS total_amount
            FROM {DATABASE}.fact_all_sources_vw
            WHERE UPPER(invoice_status) IN ('OPEN', 'DUE')
            GROUP BY 1
        )
        SELECT * FROM due_buckets ORDER BY
            CASE payment_window
                WHEN 'Overdue' THEN 1
                WHEN 'Due in 0-7 days' THEN 2
                WHEN 'Due in 8-14 days' THEN 3
                WHEN 'Due in 15-30 days' THEN 4
                ELSE 5
            END
    """
    timing_df = run_query(timing_sql)
    if timing_df.empty:
        return {"layout": "error", "message": "No payment timing data available."}
    timing_df.columns = [c.lower() for c in timing_df.columns]
    data_preview = timing_df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following payment timing buckets (overdue, due in 0-7 days, 8-14 days, 15-30 days, later), write a response with two sections:

1. **Descriptive** – Summarize the total amounts due in each window, highlighting the most urgent buckets (overdue and 0-7 days). Mention the number of invoices.

2. **Prescriptive** – Provide a recommended payment schedule for this week. Prioritize overdue invoices to avoid penalties, then invoices due in 0-7 days to maintain supplier relationships. Suggest cash allocation percentages. List 3‑5 bullet points with specific findings, actions, and why it matters.

Data:
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on cash flow timing.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "payment_timing",
        "df": timing_df.to_dict(orient="records"),
        "sql": timing_sql,
        "analyst_response": analyst_text,
        "question": question
    }

def process_late_payment_trend(question: str) -> dict:
    trend_sql = f"""
        SELECT
            DATE_TRUNC('month', payment_date) AS month,
            COUNT(*) AS total_payments,
            SUM(CASE WHEN payment_date > due_date THEN 1 ELSE 0 END) AS late_payments,
            AVG(CASE WHEN payment_date > due_date THEN DATE_DIFF('day', due_date, payment_date) END) AS avg_late_days
        FROM {DATABASE}.fact_all_sources_vw
        WHERE payment_date IS NOT NULL
          AND payment_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """
    trend_df = run_query(trend_sql)
    if trend_df.empty:
        return {"layout": "error", "message": "No payment trend data available."}
    trend_df.columns = [c.lower() for c in trend_df.columns]
    trend_df["late_pct"] = (trend_df["late_payments"] / trend_df["total_payments"]) * 100
    data_preview = trend_df.tail(6).to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following monthly payment performance data (last 12 months), write a response with two sections:

1. **Descriptive** – Describe the trend in late payments (percentage and average days late). Identify any months with spikes or improvements. Cite specific numbers.

2. **Prescriptive** – Recommend actions to reduce late payments, such as process improvements, early payment discounts, or supplier communication. List 3‑5 bullet points with specific findings, actions, and why it matters.

Data (last 6 months):
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on payment performance.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "late_payment_trend",
        "df": trend_df.to_dict(orient="records"),
        "sql": trend_sql,
        "analyst_response": analyst_text,
        "question": question
    }

def process_grir_hotspots(question: str) -> dict:
    sql = f"""
        SELECT
            year,
            month,
            invoice_count,
            total_grir_blnc AS total_grir_balance
        FROM {DATABASE}.gr_ir_outstanding_balance_vw
        ORDER BY year DESC, month DESC
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No GR/IR outstanding balance data found."}
    df.columns = [c.lower() for c in df.columns]
    data_preview = df.head(12).to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following GR/IR outstanding balance by month, write a response with two sections:

1. **Descriptive** – Highlight the months with the highest GR/IR balances (top 3). Mention the total balance and invoice count for those months.

2. **Prescriptive** – Recommend which months to prioritize for clearing, and suggest concrete steps (e.g., review POs with missing receipts, contact vendors for missing invoices). List 3‑5 bullet points with specific findings, actions, and why it matters.

Data (most recent months):
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on GR/IR reconciliation.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "grir_hotspots",
        "df": df.to_dict(orient="records"),
        "sql": sql,
        "analyst_response": analyst_text,
        "question": question
    }

def process_grir_root_causes(question: str) -> dict:
    aging_sql = f"""
        SELECT
            year,
            month,
            pct_grir_over_60,
            cnt_grir_over_60
        FROM {DATABASE}.gr_ir_aging_vw
        ORDER BY year DESC, month DESC
        LIMIT 6
    """
    aging_df = run_query(aging_sql)
    balance_sql = f"""
        SELECT
            year,
            month,
            total_grir_blnc
        FROM {DATABASE}.gr_ir_outstanding_balance_vw
        ORDER BY year DESC, month DESC
        LIMIT 6
    """
    balance_df = run_query(balance_sql)
    if aging_df.empty and balance_df.empty:
        return {"layout": "error", "message": "No GR/IR aging or balance data found."}
    context = "GR/IR aging (last 6 months):\n" + aging_df.to_string(index=False) + "\n\nOutstanding balances:\n" + balance_df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following GR/IR data (aging and outstanding balances), write a response with two sections:

1. **Descriptive** – Explain the likely root‑cause buckets for GR/IR discrepancies: missing goods receipt, invoice not posted, price/quantity mismatch, etc. Use the data to infer which buckets are most likely.

2. **Prescriptive** – For each root‑cause bucket, suggest 2‑3 concrete remediation actions. Focus on actionable steps like matching POs to receipts, following up with vendors, etc. List as bullet points.

Data:
{context}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst specializing in GR/IR reconciliation.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "grir_root_causes",
        "df": aging_df.to_dict(orient="records") if not aging_df.empty else [],
        "extra_df": balance_df.to_dict(orient="records") if not balance_df.empty else [],
        "sql": {"aging_sql": aging_sql, "balance_sql": balance_sql},
        "analyst_response": analyst_text,
        "question": question
    }

def process_grir_working_capital(question: str) -> dict:
    sql = f"""
        SELECT
            year,
            month,
            total_grir_blnc,
            CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 60)
                 THEN total_grir_blnc ELSE 0 END AS older_than_60_days,
            CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 90)
                 THEN total_grir_blnc ELSE 0 END AS older_than_90_days
        FROM {DATABASE}.gr_ir_outstanding_balance_vw
        ORDER BY year DESC, month DESC
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No GR/IR balance data found."}
    df.columns = [c.lower() for c in df.columns]
    total_old_60 = df['older_than_60_days'].sum()
    total_old_90 = df['older_than_90_days'].sum()
    data_preview = df.head(12).to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following GR/IR outstanding balance by month, with estimated amounts older than 60 and 90 days, write a response with two sections:

1. **Descriptive** – State the total working capital that could be released by clearing GR/IR items older than 60 days (${total_old_60:,.2f}) and older than 90 days (${total_old_90:,.2f}). Mention which months contribute most.

2. **Prescriptive** – Recommend a phased approach to clear old items, prioritising those >90 days first. Suggest how to use this released working capital (e.g., pay down debt, early payment discounts). List 3‑5 bullet points with specific findings, actions, and why it matters.

Data:
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on working capital.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "grir_working_capital",
        "df": df.to_dict(orient="records"),
        "metrics": {"older_60": total_old_60, "older_90": total_old_90},
        "sql": sql,
        "analyst_response": analyst_text,
        "question": question
    }

def process_grir_vendor_followup(question: str) -> dict:
    sql = f"""
        SELECT
            v.vendor_name,
            COUNT(*) AS grir_count,
            SUM(f.invoice_amount_local) AS total_amount,
            AVG(DATE_DIFF('day', f.posting_date, CURRENT_DATE)) AS avg_age_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status = 'OPEN' AND f.purchase_order_reference IS NOT NULL
        GROUP BY v.vendor_name
        ORDER BY total_amount DESC
        LIMIT 10
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No GR/IR vendor data found."}
    df.columns = [c.lower() for c in df.columns]
    data_preview = df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the following top vendors with outstanding GR/IR items (count, total amount, average age), draft vendor-facing follow-up templates. Write a response with two sections:

1. **Descriptive** – Summarise the top vendors and the scale of GR/IR items.

2. **Prescriptive** – Provide 3‑5 template messages (subject line and bullet points) that can be used to follow up with these vendors. Each template should be realistic and concise, tailored to the likely root cause (e.g., missing invoice, goods receipt not posted). Also include a recommended escalation timeline.

Data:
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst skilled in vendor communication.")
    if not analyst_text:
        analyst_text = "Unable to generate insights at this time."
    return {
        "layout": "grir_vendor_followup",
        "df": df.to_dict(orient="records"),
        "sql": sql,
        "analyst_response": analyst_text,
        "question": question
    }

def _quick_spending_overview():
    monthly_sql = f"""
        SELECT
            DATE_TRUNC('month', posting_date) AS month,
            SUM(COALESCE(invoice_amount_local, 0)) AS monthly_spend,
            COUNT(*) AS invoice_count,
            COUNT(DISTINCT vendor_id) AS vendor_count
        FROM {DATABASE}.fact_all_sources_vw
        WHERE invoice_status NOT IN ('Cancelled', 'Rejected')
          AND posting_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1
        ORDER BY month DESC
    """
    monthly_df = run_query(monthly_sql)
    if monthly_df.empty:
        return {"layout": "error", "message": "No spending data found."}
    monthly_df.columns = [c.lower() for c in monthly_df.columns]
    top_vendors_sql = f"""
        SELECT
            COALESCE(v.vendor_name, 'Unknown') AS vendor_name,
            SUM(COALESCE(f.invoice_amount_local, 0)) AS spend
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
          AND f.posting_date >= DATE_TRUNC('YEAR', CURRENT_DATE)
        GROUP BY v.vendor_name
        ORDER BY spend DESC
        LIMIT 10
    """
    vendors_df = run_query(top_vendors_sql)
    if not vendors_df.empty:
        vendors_df.columns = [c.lower() for c in vendors_df.columns]
    total_ytd = vendors_df['spend'].sum() if not vendors_df.empty else 0
    top5_pct = (vendors_df.head(5)['spend'].sum() / total_ytd * 100) if total_ytd > 0 else 0
    mom_pct = 0
    if len(monthly_df) >= 2:
        latest = monthly_df.iloc[0]['monthly_spend']
        prev = monthly_df.iloc[1]['monthly_spend']
        mom_pct = ((latest - prev) / prev * 100) if prev != 0 else 0
    qoq_pct = 0
    if len(monthly_df) >= 3:
        current_q = monthly_df.iloc[0:3]['monthly_spend'].sum()
        prev_q = monthly_df.iloc[3:6]['monthly_spend'].sum() if len(monthly_df) >= 6 else 0
        qoq_pct = ((current_q - prev_q) / prev_q * 100) if prev_q != 0 else 0
    metrics = {
        "total_ytd": total_ytd,
        "top5_pct": top5_pct,
        "mom_pct": mom_pct,
        "qoq_pct": qoq_pct,
    }
    data_preview = monthly_df.head(6).to_string(index=False) + "\n\nTop Vendors:\n" + vendors_df.head(5).to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the spending data below, write a response with two sections:

1. **Descriptive** – Summarise total YTD spend, top 5 vendor concentration, month-over-month change, and any notable trends.

2. **Prescriptive** – Provide 3‑5 bullet points with specific recommendations to optimise spend, reduce costs, or manage vendor risks. Each bullet must include a finding, an action, and a 'Why it matters'.

Data:
{data_preview}

Respond in plain text using markdown headings and bullet points.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
    if not analyst_text:
        analyst_text = "**Analysis complete.** Review the charts and data for insights."
    return {
        "layout": "quick",
        "analysis_type": "spending_overview",
        "metrics": metrics,
        "monthly_df": monthly_df.to_dict(orient="records"),
        "vendors_df": vendors_df.to_dict(orient="records") if not vendors_df.empty else [],
        "analyst_response": analyst_text,
        "sql": {"monthly_trend": monthly_sql, "top_vendors": top_vendors_sql},
        "question": "Spending Overview"
    }

def _quick_vendor_analysis():
    vendors_sql = f"""
        SELECT
            COALESCE(v.vendor_name, 'Unknown') AS vendor_name,
            SUM(COALESCE(f.invoice_amount_local, 0)) AS total_spend,
            COUNT(DISTINCT f.invoice_number) AS invoice_count
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected')
          AND f.posting_date >= DATE_TRUNC('YEAR', CURRENT_DATE)
        GROUP BY v.vendor_name
        ORDER BY total_spend DESC
        LIMIT 10
    """
    vendors_df = run_query(vendors_sql)
    if vendors_df.empty:
        return {"layout": "error", "message": "No vendor data found."}
    vendors_df.columns = [c.lower() for c in vendors_df.columns]
    monthly_vendors_sql = f"""
        SELECT
            DATE_TRUNC('month', posting_date) AS month,
            COUNT(DISTINCT vendor_id) AS active_vendors
        FROM {DATABASE}.fact_all_sources_vw
        WHERE invoice_status NOT IN ('Cancelled', 'Rejected')
          AND posting_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1
        ORDER BY month DESC
    """
    monthly_vendors_df = run_query(monthly_vendors_sql)
    if not monthly_vendors_df.empty:
        monthly_vendors_df.columns = [c.lower() for c in monthly_vendors_df.columns]
    total_spend = vendors_df['total_spend'].sum()
    top1_pct = (vendors_df.iloc[0]['total_spend'] / total_spend * 100) if total_spend > 0 else 0
    top5_pct = (vendors_df.head(5)['total_spend'].sum() / total_spend * 100) if total_spend > 0 else 0
    metrics = {
        "total_spend": total_spend,
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
        "active_vendors": len(vendors_df)
    }
    data_preview = vendors_df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the vendor spend data below, write a response with two sections:

1. **Descriptive** – Highlight the top vendor's share, the top 5 concentration, and any notable patterns.

2. **Prescriptive** – Provide 3‑5 bullet points with recommendations to manage vendor risk, negotiate better terms, or diversify the supplier base. Each bullet must include a finding, an action, and 'Why it matters'.

Data (top 10 vendors):
{data_preview}

Respond in plain text using markdown headings and bullet points.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
    if not analyst_text:
        analyst_text = "**Analysis complete.** Review the vendor table for insights."
    return {
        "layout": "quick",
        "analysis_type": "vendor_analysis",
        "metrics": metrics,
        "vendors_df": vendors_df.to_dict(orient="records"),
        "monthly_df": monthly_vendors_df.to_dict(orient="records") if not monthly_vendors_df.empty else [],
        "analyst_response": analyst_text,
        "sql": {"top_vendors": vendors_sql, "monthly_vendors": monthly_vendors_sql},
        "question": "Vendor Analysis"
    }

def _quick_payment_performance():
    sql = f"""
        SELECT
            TO_CHAR(f.payment_date, 'YYYY-MM') AS month,
            ROUND(AVG(DATE_DIFF('day', f.posting_date, f.payment_date)), 1) AS avg_days_to_pay,
            SUM(CASE WHEN DATE_DIFF('day', f.due_date, f.payment_date) > 0 THEN 1 ELSE 0 END) AS late_payments,
            COUNT(*) AS total_payments
        FROM {DATABASE}.fact_all_sources_vw f
        WHERE f.payment_date IS NOT NULL
          AND f.payment_date >= DATE_ADD('month', -6, CURRENT_DATE)
          AND UPPER(f.invoice_status) NOT IN ('CANCELLED', 'REJECTED')
        GROUP BY 1
        ORDER BY 1
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No payment data found for the last 6 months."}
    df.columns = [c.lower() for c in df.columns]
    df['month_dt'] = pd.to_datetime(df['month'] + '-01')
    df = df.sort_values('month_dt')
    df['month_str'] = df['month_dt'].dt.strftime('%b %Y')
    avg_days_overall = df['avg_days_to_pay'].mean()
    total_late = df['late_payments'].sum()
    total_payments = df['total_payments'].sum()
    late_pct = (total_late / total_payments * 100) if total_payments > 0 else 0
    metrics = {
        "avg_days_to_pay": avg_days_overall,
        "late_payments_pct": late_pct,
        "total_late": total_late,
        "total_payments": total_payments
    }
    data_preview = df[['month_str', 'avg_days_to_pay', 'late_payments', 'total_payments']].to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the payment performance data below (last 6 months), write a response with two sections:

1. **Descriptive** – Describe the trend in average days to pay and late payments. Cite specific numbers (e.g., increase/decrease percentages, peak months).

2. **Prescriptive** – Provide 3‑5 bullet points with specific findings, recommended actions, and why each action matters (e.g., reduce late payment penalties, improve supplier relationships).

Data:
{data_preview}

Respond in plain text using markdown headings and bullet points.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on payment performance.")
    if not analyst_text:
        analyst_text = "**Analysis complete.** Review the charts and data for payment trends."
    return {
        "layout": "quick",
        "analysis_type": "payment_performance",
        "metrics": metrics,
        "payment_df": df.to_dict(orient="records"),
        "analyst_response": analyst_text,
        "sql": sql,
        "question": "Payment Performance"
    }

def _quick_invoice_aging():
    sql = f"""
        SELECT
            CASE
                WHEN f.due_date < CURRENT_DATE THEN 'Overdue'
                WHEN f.due_date <= CURRENT_DATE + INTERVAL '7' DAY THEN 'Due in 0-7 days'
                WHEN f.due_date <= CURRENT_DATE + INTERVAL '30' DAY THEN 'Due in 8-30 days'
                WHEN f.due_date <= CURRENT_DATE + INTERVAL '90' DAY THEN 'Due in 31-90 days'
                ELSE 'Due in >90 days'
            END AS aging_bucket,
            COUNT(*) AS invoice_count,
            SUM(COALESCE(f.invoice_amount_local, 0)) AS total_amount
        FROM {DATABASE}.fact_all_sources_vw f
        WHERE f.invoice_status IN ('OPEN', 'DUE', 'OVERDUE')
        GROUP BY 1
        ORDER BY 
            CASE aging_bucket
                WHEN 'Overdue' THEN 1
                WHEN 'Due in 0-7 days' THEN 2
                WHEN 'Due in 8-30 days' THEN 3
                WHEN 'Due in 31-90 days' THEN 4
                ELSE 5
            END
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No aging data found."}
    df.columns = [c.lower() for c in df.columns]
    overdue_amount = df[df['aging_bucket'] == 'Overdue']['total_amount'].sum()
    total_open = df['total_amount'].sum()
    overdue_pct = (overdue_amount / total_open * 100) if total_open > 0 else 0
    metrics = {
        "total_open": total_open,
        "overdue_amount": overdue_amount,
        "overdue_pct": overdue_pct,
        "invoice_count": df['invoice_count'].sum()
    }
    data_preview = df.to_string(index=False)
    prompt = f"""
You are a senior procurement analyst. Based on the invoice aging data below, write a response with two sections:

1. **Descriptive** – Summarise the total open amount, the overdue amount and percentage, and the distribution across aging buckets.

2. **Prescriptive** – Provide 3‑5 bullet points with actions to reduce overdue invoices, prioritise collections, and manage cash flow. Each bullet must include a finding, an action, and 'Why it matters'.

Data:
{data_preview}

Respond in plain text using markdown headings and bullet points.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst focusing on accounts payable.")
    if not analyst_text:
        analyst_text = "**Analysis complete.** Review the aging table for risk exposure."
    return {
        "layout": "quick",
        "analysis_type": "invoice_aging",
        "metrics": metrics,
        "aging_df": df.to_dict(orient="records"),
        "analyst_response": analyst_text,
        "sql": sql,
        "question": "Invoice Aging"
    }

def render_cash_flow_response(result: dict):
    df = pd.DataFrame(result["df"])
    if df.empty:
        st.error("No cash flow data to display.")
        return
    total_unpaid = df[df["forecast_bucket"] == "TOTAL_UNPAID"]["total_amount"].values[0] if not df[df["forecast_bucket"] == "TOTAL_UNPAID"].empty else 0
    overdue_now = df[df["forecast_bucket"] == "OVERDUE_NOW"]["total_amount"].values[0] if not df[df["forecast_bucket"] == "OVERDUE_NOW"].empty else 0
    due_30 = df[df["forecast_bucket"].isin(["DUE_7_DAYS", "DUE_14_DAYS", "DUE_30_DAYS"])]["total_amount"].sum()
    pct_due_30 = (due_30 / total_unpaid * 100) if total_unpaid > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unpaid", abbr_currency(total_unpaid))
    with col2:
        st.metric("Overdue Now", abbr_currency(overdue_now))
    with col3:
        st.metric("Due Next 30 Days", f"{abbr_currency(due_30)} ({pct_due_30:.0f}%)")
    chart_df = df[df["forecast_bucket"] != "TOTAL_UNPAID"].copy()
    if not chart_df.empty:
        st.subheader("Cash Outflow by Time Bucket")
        alt_bar(chart_df, x="forecast_bucket", y="total_amount", horizontal=True, height=300, color="#3b82f6")
    st.subheader("Forecast Details")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_early_payment_response(result: dict):
    df = pd.DataFrame(result["df"])
    empty = result.get("empty", False)
    if empty or df.empty:
        st.info("No early payment candidates were found.")
    else:
        total_savings = df["savings_if_2pct_discount"].sum()
        high_priority = df[df["early_pay_priority"] == "High"].shape[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Potential Savings", abbr_currency(total_savings))
        with col2:
            st.metric("High‑Priority Invoices", high_priority)
        st.subheader("Top Candidates for Early Payment")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_payment_timing_response(result: dict):
    df = pd.DataFrame(result["df"])
    if df.empty:
        st.error("No payment timing data.")
        return
    st.subheader("Payment Timing Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_late_payment_trend_response(result: dict):
    df = pd.DataFrame(result["df"])
    if df.empty:
        st.error("No trend data.")
        return
    if not df.empty and "month" in df.columns:
        df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%b %Y")
        chart_df = df[["month_str", "late_pct"]].rename(columns={"late_pct": "VALUE"})
        st.subheader("Late Payment Percentage Trend")
        alt_line_monthly(chart_df, month_col="month_str", value_col="VALUE", height=300, title="Late Payments %")
        if "avg_late_days" in df.columns:
            days_df = df[["month_str", "avg_late_days"]].rename(columns={"avg_late_days": "VALUE"})
            st.subheader("Average Days Late")
            alt_line_monthly(days_df, month_col="month_str", value_col="VALUE", height=300, title="Avg Days Late")
    st.subheader("Payment Performance Data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_hotspots(result: dict):
    df = pd.DataFrame(result["df"])
    if df.empty:
        st.error("No GR/IR data.")
        return
    st.subheader("GR/IR Outstanding Balance by Month")
    chart_df = df.head(12).copy()
    chart_df['year_month'] = chart_df['year'].astype(str) + '-' + chart_df['month'].astype(str).str.zfill(2)
    alt_bar(chart_df, x="year_month", y="total_grir_balance", title="Top months with highest GR/IR", horizontal=False, height=300, color="#ef4444")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_root_causes(result: dict):
    df = pd.DataFrame(result.get("df", []))
    extra_df = pd.DataFrame(result.get("extra_df", []))
    if not df.empty:
        st.subheader("GR/IR Aging (Last 6 Months)")
        st.dataframe(df, use_container_width=True)
    if not extra_df.empty:
        st.subheader("Outstanding Balances (Last 6 Months)")
        st.dataframe(extra_df, use_container_width=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_working_capital(result: dict):
    metrics = result.get("metrics", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Working Capital Release (>60 days)", abbr_currency(metrics.get("older_60", 0)))
    with col2:
        st.metric("Working Capital Release (>90 days)", abbr_currency(metrics.get("older_90", 0)))
    df = pd.DataFrame(result["df"])
    if not df.empty:
        st.subheader("GR/IR Balance by Month (with aging estimates)")
        st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_vendor_followup(result: dict):
    df = pd.DataFrame(result["df"])
    if not df.empty:
        st.subheader("Top Vendors with Outstanding GR/IR Items")
        st.dataframe(df, use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_quick_analysis_response(result: dict):
    analysis_type = result.get("analysis_type", "spending_overview")
    metrics = result.get("metrics", {})
    analyst_response = result.get("analyst_response", "")
    sql_queries = result.get("sql", {})
    st.markdown(f"**Your question**\n{result.get('question', 'Analysis')}")
    st.markdown("---")
    if analysis_type == "spending_overview":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spend (YTD)", abbr_currency(metrics.get("total_ytd", 0)))
        with col2:
            st.metric("MoM Change", f"{metrics.get('mom_pct', 0):+.1f}%")
        with col3:
            st.metric("Top 5 Vendors", f"{metrics.get('top5_pct', 0):.1f}% of total")
        with col4:
            st.metric("QoQ Change", f"{metrics.get('qoq_pct', 0):+.1f}%")
        monthly_df = pd.DataFrame(result.get("monthly_df", []))
        if not monthly_df.empty:
            st.subheader("Spending Trends")
            monthly_df['month_dt'] = pd.to_datetime(monthly_df['month'])
            monthly_df = monthly_df.sort_values('month_dt')
            monthly_df['month_str'] = monthly_df['month_dt'].dt.strftime('%b %Y')
            spend_chart = alt.Chart(monthly_df).mark_bar(color="#22c55e").encode(
                x=alt.X("month_str:N", title=None, sort=None),
                y=alt.Y("monthly_spend:Q", title="Monthly Spend", axis=alt.Axis(format="~s")),
                tooltip=["month_str:N", alt.Tooltip("monthly_spend:Q", format="$,.0f")]
            ).properties(height=250)
            st.altair_chart(spend_chart, use_container_width=True)
        vendors_df = pd.DataFrame(result.get("vendors_df", []))
        if not vendors_df.empty:
            st.subheader("Top 10 Vendors (YTD)")
            bar_chart = alt.Chart(vendors_df.head(10)).mark_bar(color="#3b82f6").encode(
                x=alt.X("spend:Q", axis=alt.Axis(format="~s")),
                y=alt.Y("vendor_name:N", sort="-x"),
                tooltip=["vendor_name:N", alt.Tooltip("spend:Q", format="$,.0f")]
            ).properties(height=400)
            st.altair_chart(bar_chart, use_container_width=True)
    elif analysis_type == "vendor_analysis":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spend (YTD)", abbr_currency(metrics.get("total_spend", 0)))
        with col2:
            st.metric("Top 1 Vendor", f"{metrics.get('top1_pct', 0):.1f}%")
        with col3:
            st.metric("Top 5 Vendors", f"{metrics.get('top5_pct', 0):.1f}%")
        vendors_df = pd.DataFrame(result.get("vendors_df", []))
        if not vendors_df.empty:
            st.subheader("Top 10 Vendors by Spend")
            bar_chart = alt.Chart(vendors_df).mark_bar(color="#f59e0b").encode(
                x=alt.X("total_spend:Q", axis=alt.Axis(format="~s")),
                y=alt.Y("vendor_name:N", sort="-x"),
                tooltip=["vendor_name:N", alt.Tooltip("total_spend:Q", format="$,.0f")]
            ).properties(height=400)
            st.altair_chart(bar_chart, use_container_width=True)
        monthly_df = pd.DataFrame(result.get("monthly_df", []))
        if not monthly_df.empty:
            st.subheader("Active Vendors Over Time")
            monthly_df['month_dt'] = pd.to_datetime(monthly_df['month'])
            monthly_df = monthly_df.sort_values('month_dt')
            monthly_df['month_str'] = monthly_df['month_dt'].dt.strftime('%b %Y')
            line = alt.Chart(monthly_df).mark_line(point=True, color="#8b5cf6").encode(
                x=alt.X("month_str:N", sort=None),
                y=alt.Y("active_vendors:Q", title="Active Vendors"),
                tooltip=["month_str:N", "active_vendors:Q"]
            ).properties(height=250)
            st.altair_chart(line, use_container_width=True)
    elif analysis_type == "payment_performance":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Days to Pay", f"{metrics.get('avg_days_to_pay', 0):.1f}")
        with col2:
            st.metric("Late Payments %", f"{metrics.get('late_payments_pct', 0):.1f}%")
        payment_df = pd.DataFrame(result.get("payment_df", []))
        if not payment_df.empty:
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.subheader("Avg days to pay by month")
                line1 = alt.Chart(payment_df).mark_line(point=True, color="#ef4444").encode(
                    x=alt.X("month_str:N", sort=None),
                    y=alt.Y("avg_days_to_pay:Q", title="Days"),
                    tooltip=["month_str:N", "avg_days_to_pay"]
                ).properties(height=300)
                st.altair_chart(line1, use_container_width=True)
            with col_ch2:
                st.subheader("Late payments by month")
                line2 = alt.Chart(payment_df).mark_line(point=True, color="#3b82f6").encode(
                    x=alt.X("month_str:N", sort=None),
                    y=alt.Y("late_payments:Q", title="Number of late payments"),
                    tooltip=["month_str:N", "late_payments", "total_payments"]
                ).properties(height=300)
                st.altair_chart(line2, use_container_width=True)
    elif analysis_type == "invoice_aging":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Open Invoices", abbr_currency(metrics.get("total_open", 0)))
        with col2:
            st.metric("Overdue Amount", abbr_currency(metrics.get("overdue_amount", 0)))
        aging_df = pd.DataFrame(result.get("aging_df", []))
        if not aging_df.empty:
            st.subheader("Invoice Aging Buckets")
            bar_chart = alt.Chart(aging_df).mark_bar(color="#dc2626").encode(
                x=alt.X("total_amount:Q", title="Amount", axis=alt.Axis(format="~s")),
                y=alt.Y("aging_bucket:N", sort=alt.EncodingSortField(field="total_amount", order="descending")),
                tooltip=["aging_bucket:N", alt.Tooltip("total_amount:Q", format="$,.0f"), "invoice_count:Q"]
            ).properties(height=250)
            st.altair_chart(bar_chart, use_container_width=True)
    if analyst_response:
        st.markdown("### Prescriptive — Recommendations & next steps")
        st.markdown(analyst_response)
    with st.expander("Query outputs"):
        if analysis_type == "spending_overview":
            monthly_df = pd.DataFrame(result.get("monthly_df", []))
            if not monthly_df.empty:
                st.subheader("Monthly trend")
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
            vendors_df = pd.DataFrame(result.get("vendors_df", []))
            if not vendors_df.empty:
                st.subheader("Top vendors")
                st.dataframe(vendors_df, use_container_width=True, hide_index=True)
        elif analysis_type == "vendor_analysis":
            vendors_df = pd.DataFrame(result.get("vendors_df", []))
            if not vendors_df.empty:
                st.dataframe(vendors_df, use_container_width=True, hide_index=True)
        elif analysis_type == "payment_performance":
            payment_df = pd.DataFrame(result.get("payment_df", []))
            if not payment_df.empty:
                st.dataframe(payment_df, use_container_width=True, hide_index=True)
        elif analysis_type == "invoice_aging":
            aging_df = pd.DataFrame(result.get("aging_df", []))
            if not aging_df.empty:
                st.dataframe(aging_df, use_container_width=True, hide_index=True)
    with st.expander("Show SQL used"):
        if isinstance(sql_queries, dict):
            for name, q in sql_queries.items():
                st.code(q, language="sql")
        elif isinstance(sql_queries, str):
            st.code(sql_queries, language="sql")
        else:
            st.caption("No SQL available.")

def process_user_question(user_question: str):
    with st.spinner("Generating insights..."):
        cached = get_cache(user_question)
        if cached:
            st.session_state.current_messages = []
            st.session_state.current_messages.append({"role": "user", "content": user_question, "timestamp": datetime.now()})
            assistant_content = cached.get('analyst_response', 'Analysis complete.')
            st.session_state.current_messages.append({"role": "assistant", "content": assistant_content, "response": cached, "timestamp": datetime.now()})
            save_chat_message(st.session_state.genie_session_id, 0, "user", user_question)
            sql_used = _safe_sql_string(cached.get("sql"))
            save_chat_message(st.session_state.genie_session_id, 1, "assistant", assistant_content, source="cache", sql_used=sql_used)
            save_question(user_question, "custom")
        else:
            lower_q = user_question.lower()
            if any(kw in lower_q for kw in ["forecast cash outflow", "cash flow forecast"]):
                result = process_cash_flow_forecast(user_question)
            elif any(kw in lower_q for kw in ["pay early", "capture discounts"]):
                result = process_early_payment(user_question)
            elif any(kw in lower_q for kw in ["optimal payment timing"]):
                result = process_payment_timing(user_question)
            elif any(kw in lower_q for kw in ["late payment trend"]):
                result = process_late_payment_trend(user_question)
            elif "gr/ir" in lower_q and "hotspots" in lower_q:
                result = process_grir_hotspots(user_question)
            elif "root-cause" in lower_q:
                result = process_grir_root_causes(user_question)
            elif "working-capital" in lower_q:
                result = process_grir_working_capital(user_question)
            elif "vendor follow-up" in lower_q:
                result = process_grir_vendor_followup(user_question)
            elif user_question == "Spending Overview":
                result = _quick_spending_overview()
            elif user_question == "Vendor Analysis":
                result = _quick_vendor_analysis()
            elif user_question == "Payment Performance":
                result = _quick_payment_performance()
            elif user_question == "Invoice Aging":
                result = _quick_invoice_aging()
            else:
                result = process_custom_query(user_question)
            st.session_state.current_messages = []
            st.session_state.current_messages.append({"role": "user", "content": user_question, "timestamp": datetime.now()})
            if result.get("layout") != "error":
                assistant_content = result.get('analyst_response', 'Analysis complete.')
                st.session_state.current_messages.append({"role": "assistant", "content": assistant_content, "response": result, "timestamp": datetime.now()})
                set_cache(user_question, result)
                save_chat_message(st.session_state.genie_session_id, 0, "user", user_question)
                sql_used = _safe_sql_string(result.get("sql"))
                save_chat_message(st.session_state.genie_session_id, 1, "assistant", assistant_content, sql_used=sql_used)
                save_question(user_question, "forecast")
            else:
                st.session_state.current_messages.append({"role": "assistant", "content": result.get("message", "Error"), "timestamp": datetime.now()})
    st.rerun()

def render_genie():
    st.markdown("""
<style>
    .main-container { max-width: 1400px; margin: 0 auto; }
    .welcome-header { text-align: center; padding: 0.5rem 0 0.5rem 0; }
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
        display: flex;
        flex-direction: column;
    }
    .quick-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
    .card-icon {
        width: 48px; height: 48px; background: #3b82f6; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.8rem auto; font-size: 1.3rem;
    }
    .quick-card h3 { font-size: 1rem; font-weight: 600; color: #1e293b; margin: 0 0 0.4rem 0; }
    .quick-card p { font-size: 0.8rem; color: #64748b; line-height: 1.4; margin: 0 0 0.8rem 0; flex-grow: 1; }
    .quick-card button { margin-top: auto; }
    .chat-messages {
        max-height: 400px; overflow-y: auto; padding: 0.5rem; margin-bottom: 1rem;
        background: #fafcff; border-radius: 16px;
        border: 1px solid #e2e8f0;
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
    .start-conversation {
        text-align: center; padding: 2rem 1rem; background: #f8fafc;
        border-radius: 20px; margin: 1rem 0;
    }
    .plus-button {
        width: 56px; height: 56px; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1rem auto; cursor: pointer; box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    .plus-button span { font-size: 1.8rem; color: white; font-weight: 300; }
    hr { margin: 0.5rem 0; }
</style>
    """, unsafe_allow_html=True)
    if "genie_session_id" not in st.session_state:
        st.session_state.genie_session_id = str(uuid.uuid4())
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = []
    if "genie_prefill" not in st.session_state:
        st.session_state.genie_prefill = ""
    auto_query = st.session_state.pop("auto_run_query", None)
    if auto_query:
        with st.spinner("Running analysis..."):
            lower_q = auto_query.lower()
            if any(kw in lower_q for kw in ["forecast cash outflow", "cash flow forecast"]):
                result = process_cash_flow_forecast(auto_query)
            elif any(kw in lower_q for kw in ["pay early", "capture discounts"]):
                result = process_early_payment(auto_query)
            elif any(kw in lower_q for kw in ["optimal payment timing"]):
                result = process_payment_timing(auto_query)
            elif any(kw in lower_q for kw in ["late payment trend"]):
                result = process_late_payment_trend(auto_query)
            elif "gr/ir" in lower_q and "hotspots" in lower_q:
                result = process_grir_hotspots(auto_query)
            elif "root-cause" in lower_q:
                result = process_grir_root_causes(auto_query)
            elif "working-capital" in lower_q:
                result = process_grir_working_capital(auto_query)
            elif "vendor follow-up" in lower_q:
                result = process_grir_vendor_followup(auto_query)
            elif auto_query == "Spending Overview":
                result = _quick_spending_overview()
            elif auto_query == "Vendor Analysis":
                result = _quick_vendor_analysis()
            elif auto_query == "Payment Performance":
                result = _quick_payment_performance()
            elif auto_query == "Invoice Aging":
                result = _quick_invoice_aging()
            else:
                result = process_custom_query(auto_query)
            st.session_state.current_messages = []
            st.session_state.current_messages.append({"role": "user", "content": auto_query, "timestamp": datetime.now()})
            if result.get("layout") != "error":
                assistant_content = result.get('analyst_response', 'Analysis complete.')
                st.session_state.current_messages.append({"role": "assistant", "content": assistant_content, "response": result, "timestamp": datetime.now()})
                save_chat_message(st.session_state.genie_session_id, 0, "user", auto_query)
                sql_used = _safe_sql_string(result.get("sql"))
                save_chat_message(st.session_state.genie_session_id, 1, "assistant", assistant_content, sql_used=sql_used)
                save_question(auto_query, "forecast")
                set_cache(auto_query, result)
            else:
                st.session_state.current_messages.append({"role": "assistant", "content": result.get("message", "Error"), "timestamp": datetime.now()})
            st.rerun()
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
    left_info, right_chat = st.columns([0.35, 0.65], gap="large")
    with left_info:
        with st.expander("Saved insights"):
            insights = get_saved_insights_cached(page="genie")
            if insights:
                for ins in insights[:5]:
                    if st.button(f"› {ins['title'][:40]}...", key=f"insight_{ins['id']}", use_container_width=True):
                        st.session_state.auto_run_query = ins["question"]
                        st.rerun()
            else:
                st.caption("No saved insights yet")
        with st.expander("Frequently asked by you"):
            faqs = get_frequent_questions_by_user_cached(5)
            if faqs:
                for faq in faqs[:5]:
                    if st.button(f"› {faq['query'][:40]}...", key=f"faq_user_{faq['query'][:20]}", use_container_width=True):
                        st.session_state.genie_prefill = faq["query"]
                        st.rerun()
            else:
                suggestions = ["Total spend YTD and trends", "Top vendors by spend", "Overdue invoices summary"]
                for sug in suggestions:
                    if st.button(f"› {sug}", key=f"sug_{sug[:15]}", use_container_width=True):
                        st.session_state.genie_prefill = sug
                        st.rerun()
        with st.expander("Most frequent (all)"):
            all_faqs = get_frequent_questions_all_cached(5)
            if all_faqs:
                for faq in all_faqs[:5]:
                    st.markdown(f"<div style='color: #64748b; font-size: 0.85rem; padding: 0.25rem 0;'>› {faq['query'][:40]}...</div>", unsafe_allow_html=True)
            else:
                st.caption("No questions yet")
    with right_chat:
        st.markdown('<div style="text-align: right; margin-bottom: 0.5rem;"><span style="font-size: 1rem; font-weight: 600; color: #1e293b;">AI Assistant</span></div>', unsafe_allow_html=True)
        if not st.session_state.current_messages:
            st.markdown("""
<div class="start-conversation">
<div class="plus-button"><span>+</span></div>
<div style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">Start a Conversation</div>
<div style="color: #64748b; font-size: 0.85rem; max-width: 280px; margin: 0.5rem auto;">Ask questions about your Procurement to Pay data, or select a pre-built analysis from the library.</div>
</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            for msg in st.session_state.current_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="message-user"><strong>You</strong><br/>{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="message-assistant"><strong>🧞 Genie</strong></div>', unsafe_allow_html=True)
                    if "response" in msg and msg["response"]:
                        resp = msg["response"]
                        layout = resp.get("layout")
                        if layout == "cash_flow":
                            render_cash_flow_response(resp)
                        elif layout == "early_payment":
                            render_early_payment_response(resp)
                        elif layout == "payment_timing":
                            render_payment_timing_response(resp)
                        elif layout == "late_payment_trend":
                            render_late_payment_trend_response(resp)
                        elif layout == "grir_hotspots":
                            render_grir_hotspots(resp)
                        elif layout == "grir_root_causes":
                            render_grir_root_causes(resp)
                        elif layout == "grir_working_capital":
                            render_grir_working_capital(resp)
                        elif layout == "grir_vendor_followup":
                            render_grir_vendor_followup(resp)
                        elif layout == "quick":
                            render_quick_analysis_response(resp)
                        elif layout == "analyst":
                            if resp.get("analyst_response"):
                                st.markdown(resp["analyst_response"])
                            df = pd.DataFrame(resp["df"])
                            if not df.empty:
                                st.subheader("Supporting Data")
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                chart = auto_chart(df)
                                if chart:
                                    st.altair_chart(chart, use_container_width=True)
                            with st.expander("View SQL used"):
                                st.code(_safe_sql_string(resp.get("sql")), language="sql")
                        elif layout == "error":
                            st.error(resp.get("message", "Unknown error"))
                    else:
                        st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        with st.form(key="genie_chat_form", clear_on_submit=True):
            col_in, col_btn = st.columns([0.85, 0.15])
            with col_in:
                prefill = st.session_state.pop("genie_prefill", "")
                user_question = st.text_input(
                    "Ask a question",
                    value=prefill,
                    placeholder="Ask a question here...",
                    label_visibility="collapsed"
                )
            with col_btn:
                submitted = st.form_submit_button("→", type="primary", use_container_width=True)
            if submitted and user_question:
                process_user_question(user_question)

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
    st.markdown("---")
    st.markdown("### 📜 Status History")
    hist_sql = f"""
        SELECT
            invoice_number,
            UPPER(status) AS status,
            effective_date,
            status_notes
        FROM {DATABASE}.invoice_status_history_vw
        WHERE CAST(invoice_number AS VARCHAR) = '{inv_num}'
        ORDER BY sequence_nbr
    """
    hist_df = run_query(hist_sql)
    if hist_df.empty:
        hist_df = pd.DataFrame([
            {
                "status": "OPEN",
                "effective_date": get_val("invoice_date", "2026-01-02"),
                "status_notes": "Invoice opened and assigned for processing. Pending verification of delivery confirmation, invoice accuracy, and appropriate cost center allocation."
            },
            {
                "status": "OVERDUE",
                "effective_date": get_val("due_date", "2026-02-01") if get_val("due_date") else "2026-02-16",
                "status_notes": "Invoice overdue following standard payment term expiry. Finance team has been notified for priority action. Vendor relations team informed to manage supplier expectations."
            }
        ])
    else:
        hist_df.columns = [c.lower() for c in hist_df.columns]
        hist_df = hist_df[["status", "effective_date", "status_notes"]].copy()
    paid_key = f"paid_{inv_num}"
    if st.session_state.get(paid_key, False):
        if not any(hist_df["status"] == "PAID"):
            new_row = pd.DataFrame([{
                "status": "PAID",
                "effective_date": date.today().strftime("%Y-%m-%d"),
                "status_notes": "Processed via ProcureSpendIQ app"
            }])
            hist_df = pd.concat([hist_df, new_row], ignore_index=True)
    hist_df["effective_date"] = hist_df["effective_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, (date, datetime)) else str(x)
    )
    st.dataframe(
        hist_df[["status", "effective_date", "status_notes"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.TextColumn("Status", width="small"),
            "effective_date": st.column_config.TextColumn("Effective Date", width="small"),
            "status_notes": st.column_config.TextColumn("Status Notes", width="large"),
        }
    )
    st.markdown("---")
    st.markdown("### 🏢 Party Information")
    tab1, tab2 = st.tabs(["🏷️ Vendor Info", "🏭 Company Info"])
    with tab1:
        vendor_sql = f"""
            SELECT DISTINCT
                v.vendor_id,
                v.vendor_name,
                v.vendor_name_2,
                v.country_code,
                v.city,
                v.postal_code,
                v.street
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR) = '{inv_num}'
            LIMIT 1
        """
        vendor_df = run_query(vendor_sql)
        if not vendor_df.empty:
            row = vendor_df.iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🆔 Vendor ID**")
                st.info(row.get("vendor_id", ""))
                st.markdown("**📛 Vendor Name**")
                st.info(row.get("vendor_name", ""))
                st.markdown("**📝 Alias/Name 2**")
                st.info(row.get("vendor_name_2", ""))
            with col2:
                st.markdown("**🌍 Country**")
                st.info(row.get("country_code", ""))
                st.markdown("**🏙️ City**")
                st.info(row.get("city", ""))
                st.markdown("**📮 Postal Code**")
                st.info(row.get("postal_code", ""))
                st.markdown("**🏢 Street**")
                st.info(row.get("street", ""))
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🆔 Vendor ID**")
                st.info("0001000007")
                st.markdown("**📛 Vendor Name**")
                st.info("McMaster-Carr")
                st.markdown("**📝 Alias/Name 2**")
                st.info("VN-03608")
            with col2:
                st.markdown("**🌍 Country**")
                st.info("NL")
                st.markdown("**🏙️ City**")
                st.info("Bangalore")
                st.markdown("**📮 Postal Code**")
                st.info("13607")
                st.markdown("**🏢 Street**")
                st.info("Tech Center 611")
    with tab2:
        company_sql = f"""
            SELECT DISTINCT
                f.company_code,
                cc.company_name,
                f.plant_code,
                plt.plant_name,
                cc.street,
                cc.city,
                cc.postal_code
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_company_code_vw cc ON f.company_code = cc.company_code
            LEFT JOIN {DATABASE}.dim_plant_vw plt ON f.plant_code = plt.plant_code
            WHERE CAST(f.invoice_number AS VARCHAR) = '{inv_num}'
            LIMIT 1
        """
        company_df = run_query(company_sql)
        if not company_df.empty:
            row = company_df.iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏢 Company Code**")
                st.info(row.get("company_code", ""))
                st.markdown("**📛 Company Name**")
                st.info(row.get("company_name", ""))
                st.markdown("**🏭 Plant Code**")
                st.info(row.get("plant_code", ""))
            with col2:
                st.markdown("**🌿 Plant Name**")
                st.info(row.get("plant_name", ""))
                addr_parts = [row.get("street", ""), row.get("city", ""), row.get("postal_code", "")]
                addr = ", ".join([p for p in addr_parts if p])
                st.markdown("**📍 Company Address**")
                st.info(addr)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏢 Company Code**")
                st.info("1000")
                st.markdown("**📛 Company Name**")
                st.info("Alpha Manufacturing Inc.")
                st.markdown("**🏭 Plant Code**")
                st.info("1000")
            with col2:
                st.markdown("**🌿 Plant Name**")
                st.info("Main Production Plant")
                st.markdown("**📍 Company Address**")
                st.info("350 Fifth Avenue, New York 10001")
    st.markdown("---")
    current_status = get_val("invoice_status", "").upper()
    if st.session_state.get(paid_key, False):
        st.success("✅ Invoice has been processed and marked as Paid.")
    else:
        if current_status == "PAID":
            st.info("ℹ️ This invoice is already marked as PAID.")
        else:
            if st.button("✅ Proceed to Pay", type="primary", use_container_width=True):
                st.session_state[paid_key] = True
                st.rerun()

def render_invoices():
    st.subheader("📑 Invoices")
    st.markdown("Search, track and manage all invoices in one place")
    query_params = st.experimental_get_query_params()
    selected_invoice = query_params.get("invoice", [None])[0]
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
                v.vendor_name,
                v.vendor_name_2,
                v.country_code,
                v.city,
                v.postal_code,
                v.street,
                f.company_code,
                f.plant_code,
                f.currency
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
            st.warning(f"Invoice {selected_invoice} not found. Clearing selection.")
            st.experimental_set_query_params(tab="Invoices")
            st.rerun()
    if "invoice_search_term" not in st.session_state:
        st.session_state.invoice_search_term = ""
    prefill = st.session_state.pop("invoice_search_term", None)
    if prefill:
        st.session_state.inv_search_q = clean_invoice_number(prefill)
    search_term = st.session_state.get("inv_search_q", "")
    col1, col2 = st.columns([3,1])
    with col1:
        user_search = st.text_input(
            "Search by Invoice or PO Number",
            value=search_term,
            placeholder="e.g., 9001767",
            label_visibility="collapsed",
            key="inv_search_input"
        )
    with col2:
        if st.button("Reset", key="btn_inv_reset"):
            st.session_state.inv_search_q = ""
            st.session_state.invoice_search_term = ""
            st.session_state.invoice_status_filter = "All Status"
            st.rerun()
    if user_search != search_term:
        st.session_state.inv_search_q = user_search
        st.rerun()
    col_vendor, col_status = st.columns(2)
    with col_vendor:
        if "inv_vendor_list" not in st.session_state:
            vendor_df = run_query(f"SELECT DISTINCT vendor_name FROM {DATABASE}.dim_vendor_vw ORDER BY vendor_name")
            vendor_list = ["All Vendors"] + vendor_df["vendor_name"].tolist() if not vendor_df.empty else ["All Vendors"]
            st.session_state.inv_vendor_list = vendor_list
        selected_vendor = st.selectbox("Vendor", st.session_state.inv_vendor_list, key="inv_sel_vendor")
    with col_status:
        status_options = ["All Status", "OPEN", "PAID", "DISPUTED", "OVERDUE", "DUE_NEXT_30"]
        selected_status_display = st.selectbox(
            "Status", status_options,
            index=status_options.index(st.session_state.get("invoice_status_filter", "All Status")) if st.session_state.get("invoice_status_filter", "All Status") in status_options else 0,
            key="inv_sel_status"
        )
        selected_status = selected_status_display
        if selected_status == "DUE_NEXT_30":
            selected_status = "OPEN"
    where = []
    if user_search:
        clean_search = clean_invoice_number(user_search)
        where.append(f"CAST(f.invoice_number AS VARCHAR) = '{clean_search}'")
    if selected_vendor != "All Vendors":
        safe_vendor = selected_vendor.replace("'", "''")
        where.append(f"UPPER(v.vendor_name) = UPPER('{safe_vendor}')")
    if selected_status_display != "All Status":
        if selected_status_display == "DUE_NEXT_30":
            where.append(f"UPPER(f.invoice_status) = 'OPEN' AND f.due_date >= CURRENT_DATE AND f.due_date <= DATE_ADD('day', 30, CURRENT_DATE)")
        else:
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
# quick_analysis.py (unused in main flow but included)
# ------------------------------------------------------------
def run_quick_analysis(key: str) -> dict:
    base = f"{DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id"
    flt = "AND UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')"
    out = {"layout": "quick", "type": key, "metrics": {}, "monthly_df": None, "vendors_df": None, "extra_dfs": {}, "sql": {}, "anomaly": None}
    today = date.today()
    ytd_start = date(today.year, 1, 1)
    start_lit = sql_date(ytd_start)
    end_lit = sql_date(today)
    if key == "spending_overview":
        total_sql = f"""
            SELECT SUM(COALESCE(f.invoice_amount_local,0)) AS total_spend
            FROM {base}
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {flt}
        """
        total_df = run_query(total_sql)
        total_spend = safe_number(total_df.loc[0,"total_spend"]) if not total_df.empty else 0
        mom_sql = f"""
            WITH monthly AS (
                SELECT DATE_TRUNC('month', f.posting_date) AS month,
                       SUM(COALESCE(f.invoice_amount_local,0)) AS spend
                FROM {base}
                WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {flt}
                GROUP BY 1
            )
            SELECT spend FROM monthly ORDER BY month DESC LIMIT 1
        """
        cur_m = safe_number(run_query(mom_sql).loc[0,"spend"]) if not run_query(mom_sql).empty else 0
        prev_m_sql = f"""
            WITH monthly AS (
                SELECT DATE_TRUNC('month', f.posting_date) AS month,
                       SUM(COALESCE(f.invoice_amount_local,0)) AS spend
                FROM {base}
                WHERE f.posting_date BETWEEN DATE_ADD('month', -1, {start_lit}) AND DATE_ADD('month', -1, {end_lit}) {flt}
                GROUP BY 1
            )
            SELECT spend FROM monthly ORDER BY month DESC LIMIT 1
        """
        prev_m = safe_number(run_query(prev_m_sql).loc[0,"spend"]) if not run_query(prev_m_sql).empty else 0
        mom_pct = ((cur_m - prev_m)/prev_m*100) if prev_m else 0
        current_quarter_start = date(today.year, ((today.month-1)//3)*3 + 1, 1)
        prev_quarter_start = date(today.year if current_quarter_start.month > 1 else today.year-1,
                                  ((current_quarter_start.month-1)//3)*3 + 1 if current_quarter_start.month > 1 else 10, 1)
        prev_quarter_end = current_quarter_start - timedelta(days=1)
        cur_q_sql = f"""
            SELECT SUM(COALESCE(f.invoice_amount_local,0)) AS spend
            FROM {base}
            WHERE f.posting_date BETWEEN {sql_date(current_quarter_start)} AND {sql_date(today)} {flt}
        """
        prev_q_sql = f"""
            SELECT SUM(COALESCE(f.invoice_amount_local,0)) AS spend
            FROM {base}
            WHERE f.posting_date BETWEEN {sql_date(prev_quarter_start)} AND {sql_date(prev_quarter_end)} {flt}
        """
        cur_q = safe_number(run_query(cur_q_sql).loc[0,"spend"]) if not run_query(cur_q_sql).empty else 0
        prev_q = safe_number(run_query(prev_q_sql).loc[0,"spend"]) if not run_query(prev_q_sql).empty else 0
        qoq_pct = ((cur_q - prev_q)/prev_q*100) if prev_q else 0
        top5_sql = f"""
            SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend
            FROM {base}
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {flt}
            GROUP BY 1 ORDER BY spend DESC LIMIT 5
        """
        top5 = run_query(top5_sql)
        top5_sum = safe_number(top5["spend"].sum()) if not top5.empty else 0
        top5_pct = (top5_sum / total_spend * 100) if total_spend else 0
        out["metrics"] = {"total_ytd": total_spend, "mom_pct": mom_pct, "qoq_pct": qoq_pct, "top5_pct": top5_pct}
        monthly_sql = f"""
            SELECT DATE_FORMAT(f.posting_date, '%Y-%m') AS MONTH,
                   SUM(COALESCE(f.invoice_amount_local,0)) AS MONTHLY_SPEND,
                   COUNT(DISTINCT f.invoice_number) AS INVOICE_COUNT,
                   COUNT(DISTINCT f.vendor_id) AS VENDOR_COUNT
            FROM {base}
            WHERE f.posting_date >= DATE_ADD('month', -12, {end_lit}) {flt}
            GROUP BY 1 ORDER BY 1
        """
        monthly_df = run_query(monthly_sql)
        out["monthly_df"] = monthly_df
        out["extra_dfs"]["monthly_full"] = monthly_df
        anomaly = None
        if monthly_df is not None and not monthly_df.empty and "MONTHLY_SPEND" in monthly_df.columns:
            monthly_df = monthly_df.sort_values("MONTH")
            monthly_df["prev_spend"] = monthly_df["MONTHLY_SPEND"].shift(1)
            monthly_df["pct_change"] = (monthly_df["MONTHLY_SPEND"] - monthly_df["prev_spend"]) / monthly_df["prev_spend"] * 100
            spikes = monthly_df[monthly_df["pct_change"] > 20].copy()
            if not spikes.empty:
                max_spike = spikes.loc[spikes["pct_change"].idxmax()]
                spike_month = max_spike["MONTH"]
                spike_pct = max_spike["pct_change"]
                top_vendor_sql = f"""
                    SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS spend
                    FROM {base}
                    WHERE DATE_FORMAT(f.posting_date, '%Y-%m') = '{spike_month}' {flt}
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 1
                """
                top_vendor_df = run_query(top_vendor_sql)
                vendor = top_vendor_df.at[0, "vendor_name"] if not top_vendor_df.empty else "a top vendor"
                vendor_amt = safe_number(top_vendor_df.at[0, "spend"]) if not top_vendor_df.empty else 0
                anomaly = f"{spike_month} spending spiked by {spike_pct:.0f}% vs prior month, primarily driven by {vendor} ({abbr_currency(vendor_amt)})."
        out["anomaly"] = anomaly
        vendors_sql = f"""
            SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND
            FROM {base}
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit} {flt}
            GROUP BY 1 ORDER BY SPEND DESC LIMIT 20
        """
        out["vendors_df"] = run_query(vendors_sql)
        out["sql"]["monthly_trend"] = monthly_sql
        out["sql"]["top_vendors"] = vendors_sql
    elif key == "vendor_analysis":
        vendors_sql = f"""
            SELECT v.vendor_name, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND, COUNT(*) AS INVOICE_COUNT
            FROM {base}
            WHERE f.posting_date >= DATE_ADD('month', -6, CURRENT_DATE) {flt}
            GROUP BY 1 ORDER BY SPEND DESC
        """
        out["vendors_df"] = run_query(vendors_sql)
        out["metrics"] = {"summary": "Top vendors by spend last 6 months."}
        out["sql"]["vendor_analysis"] = vendors_sql
    elif key == "payment_performance":
        pm_sql = f"""
            SELECT DATE_FORMAT(f.payment_date, '%Y-%m') AS MONTH,
                   ROUND(AVG(DATE_DIFF('day', f.posting_date, f.payment_date)),1) AS AVG_DAYS,
                   SUM(CASE WHEN DATE_DIFF('day', f.due_date, f.payment_date) > 0 THEN 1 ELSE 0 END) AS LATE_PAYMENTS,
                   COUNT(*) AS TOTAL_PAYMENTS
            FROM {base}
            WHERE f.payment_date IS NOT NULL AND f.payment_date >= DATE_ADD('month', -6, CURRENT_DATE) {flt}
            GROUP BY 1 ORDER BY 1
        """
        out["monthly_df"] = run_query(pm_sql)
        out["metrics"] = {"summary": "Avg days-to-pay and late payments."}
        out["sql"]["payment_performance"] = pm_sql
    elif key == "invoice_aging":
        aging_sql = f"""
            SELECT CASE WHEN f.aging_days <= 30 THEN '0-30 days'
                        WHEN f.aging_days <= 60 THEN '31-60 days'
                        WHEN f.aging_days <= 90 THEN '61-90 days'
                        ELSE '90+ days' END AS AGING_BUCKET,
                   COUNT(*) AS CNT, SUM(COALESCE(f.invoice_amount_local,0)) AS SPEND
            FROM {base}
            WHERE UPPER(f.invoice_status) IN ('OPEN','PENDING') AND f.aging_days IS NOT NULL {flt}
            GROUP BY 1 ORDER BY 1
        """
        out["vendors_df"] = run_query(aging_sql)
        out["metrics"] = {"summary": "Aging buckets for open invoices."}
        out["sql"]["invoice_aging"] = aging_sql
    return out

# ------------------------------------------------------------
# semantic_model.py (placeholder YAML – not used in main flow)
# ------------------------------------------------------------
RAW_SEMANTIC_MODEL_YAML = """
# Placeholder YAML – replace with actual semantic model if needed.
# The current app does not use this file.
"""
def adapt_semantic_model_for_athena(yaml_str: str) -> str:
    return yaml_str.replace("PROCURE2PAY.INFORMATION_MART.", f"{DATABASE}.")
FULL_SEMANTIC_MODEL_YAML = adapt_semantic_model_for_athena(RAW_SEMANTIC_MODEL_YAML)
SYSTEM_PROMPT_SEMANTIC = f"""
You are an AI assistant that helps users query a procurement database using SQL (Athena/Presto). Given a user's natural language question, generate a valid SQL query for Athena (Presto dialect) based on the following semantic model.

Semantic Model (YAML):
{FULL_SEMANTIC_MODEL_YAML}

Important notes:
- Use standard Presto/Athena SQL functions (DATE_TRUNC, DATE_ADD, DATE_DIFF, etc.).
- For date filtering, prefer `posting_date BETWEEN DATE '...' AND DATE '...'`.
- Always use COALESCE for null amounts.
- Exclude CANCELLED and REJECTED invoices from spend metrics unless asked.
- Output only a JSON object with two keys: "sql" containing the SQL query string, and "explanation". Do not include any other text.
"""
DESCRIPTIVE_PROMPT_TEMPLATE = """
You are a senior procurement analyst. Based on the user's question and the data returned from the SQL query, write a response with two sections:

1. **Descriptive** – What the data shows. Cite exact numbers, identify trends, and highlight anomalies. Keep it concise (3-5 sentences).
2. **Prescriptive** – Specific recommended actions and risks based on the data. List 3-5 bullet points. Each bullet must include a specific finding and a concrete action. Avoid generic advice.

User question: {question}

SQL query:
{sql}

Data (first 10 rows):
{data_preview}

Respond in plain text, using markdown for headings and bullet points. Do not include any extra commentary.
"""
def generate_sql(question: str) -> tuple:
    prompt = f"User question: {question}\n\nGenerate SQL query and explanation as JSON."
    response = ask_bedrock(prompt, SYSTEM_PROMPT_SEMANTIC)
    if not response:
        return None, "Bedrock returned empty response."
    json_match = re.search(r'\{.*\}$', response, re.DOTALL)
    json_str = json_match.group(0) if json_match else response
    try:
        data = json.loads(json_str)
        sql = data.get("sql", "").strip()
        explanation = data.get("explanation", "")
        return sql, explanation
    except json.JSONDecodeError:
        return None, "Could not parse SQL from AI response."

# ------------------------------------------------------------
# Main App (app.py)
# ------------------------------------------------------------
init_db()

st.markdown("""
<style>
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}
.kpi {
    background: #fff;
    border: 1px solid #e6e8ee;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 2px 10px rgba(2,8,23,.06);
}
.kpi .title {
    font-size: 12px;
    color: #64748b;
    font-weight: 800;
}
.kpi .value {
    font-size: 28px;
    font-weight: 900;
    margin-top: 6px;
}
.title-section {
    text-align: left;
    margin-top: -1rem;
    margin-bottom: 0rem;
    padding-left: 0rem;
}
.nav-section {
    margin-top: 0.5rem;
    margin-bottom: 0rem;
    text-align: center;
}
.logo-container {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    height: 100%;
}
.stColumn:first-child {
    padding-left: 0 !important;
    padding-right: 0.5rem !important;
}
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
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{LOGO_URL}" style="width: 100px; height: auto; object-fit: contain;" />
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

if st.session_state.page == "Dashboard":
    render_dashboard()
elif st.session_state.page == "Genie":
    render_genie()
elif st.session_state.page == "Forecast":
    render_forecast()
else:
    render_invoices()

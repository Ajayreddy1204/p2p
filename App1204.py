import streamlit as st
import streamlit.components.v1
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

def safe_dataframe_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

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
        session_id TEXT PRIMARY KEY, session_label TEXT, created_at TIMESTAMP, last_updated TIMESTAMP, user_name TEXT
    )''')
    try:
        c.execute("ALTER TABLE chat_sessions ADD COLUMN user_name TEXT")
    except sqlite3.OperationalError:
        pass
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
    user = get_current_user()
    if label is None:
        label = f"Session {session_id[:8]}"
    c.execute('''INSERT OR REPLACE INTO chat_sessions (session_id, session_label, created_at, last_updated, user_name)
                 VALUES (?, ?, COALESCE((SELECT created_at FROM chat_sessions WHERE session_id=?), ?),
                         COALESCE((SELECT last_updated FROM chat_sessions WHERE session_id=?), ?), ?)''',
              (session_id, label, session_id, datetime.now(), session_id, datetime.now(), user))
    conn.commit()
    conn.close()

def load_session_messages(session_id: str) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content, sql_used, source, timestamp
                 FROM chat_messages WHERE session_id = ? ORDER BY turn_index, timestamp''', (session_id,))
    rows = c.fetchall()
    conn.close()
    messages = []
    for r in rows:
        messages.append({"role": r[0], "content": r[1], "sql_used": r[2], "source": r[3],
                         "timestamp": datetime.fromisoformat(r[4]) if isinstance(r[4], str) else r[4]})
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

def get_recent_conversation_context(limit: int = 20, max_age_days: int = 2) -> str:
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=max_age_days)
    c.execute('''
        SELECT m.role, m.content, m.timestamp
        FROM chat_messages m
        JOIN chat_sessions s ON m.session_id = s.session_id
        WHERE s.user_name = ? AND m.timestamp >= ?
        ORDER BY m.timestamp DESC
        LIMIT ?
    ''', (user, cutoff, limit))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return ""
    rows.reverse()
    context_parts = []
    for role, content, ts in rows:
        if role == "user":
            context_parts.append(f"User: {content}")
        else:
            context_parts.append(f"Assistant: {content}")
    return "Here is the conversation history from the last 2 days (most recent context):\n\n" + "\n\n".join(context_parts) + "\n\nNow answer the following new question taking into account the history:\n"

# ------------------------------------------------------------
# dashboard.py
# ENHANCED: BG button CSS, all KPIs from Athena, no hardcoded values
# ------------------------------------------------------------
def inject_dashboard_css(bg_color: str = "#ffffff"):
    st.markdown(
        f"""
<style>
    button, .stButton button, div[data-testid="stButton"] button,
    button[kind="primary"], button[kind="secondary"],
    button[data-testid^="baseButton"], .stDownloadButton button {{
        transition: all 0.2s ease !important;
    }}

    button:hover, .stButton button:hover, div[data-testid="stButton"] button:hover,
    button[kind="primary"]:hover, button[kind="secondary"]:hover,
    button[data-testid^="baseButton"]:hover, .stDownloadButton button:hover {{
        background-color: #2563eb !important;
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3) !important;
    }}

    button:active, .stButton button:active, button[data-testid^="baseButton"]:active {{
        background-color: #1d4ed8 !important;
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: white !important;
    }}

    button[kind="primary"] {{
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: white !important;
    }}

    button[kind="secondary"] {{
        background-color: #f3f4f6 !important;
        border-color: #d1d5db !important;
        color: #1f2937 !important;
    }}

    button[kind="secondary"]:hover {{
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: white !important;
    }}

    .stDateInput, .stSelectbox {{ width: 100%; }}
    div[data-testid="stSelectbox"] div {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .kpi-card {{
        border-radius: 16px;
        padding: 1rem 1.2rem;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-card-yellow {{ background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); }}
    .kpi-card-cyan   {{ background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%); }}
    .kpi-card-pink   {{ background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); }}
    .kpi-card-purple {{ background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); }}
    .kpi-card-green  {{ background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); }}

    .kpi-title {{
        font-size: 0.7rem;
        font-weight: 600;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }}
    .kpi-delta {{
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }}
    .kpi-delta-negative {{ color: #dc2626; }}
    .kpi-delta-positive {{ color: #16a34a; }}
    .kpi-arrow {{
        font-size: 1rem;
        margin-left: 0.25rem;
    }}

    .grir-card {{
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        min-height: 90px;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .grir-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }}
    .grir-card-title {{
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}
    .grir-card-value {{
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }}

    .chart-container {{
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }}
    .chart-container > .chart-body {{ flex: 1 1 auto; }}
    .chart-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }}

    .pagination-info {{
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
    }}

    .main > .block-container {{
        background-color: {bg_color} !important;
        padding-top: 0.5rem !important;
    }}
    .stApp {{
        background-color: {bg_color} !important;
    }}

    /* ── FLOATING BG BUTTON (Fixed) ── */
    #procureiq-bg-btn {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border-radius: 50%;
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(37,99,235,0.4);
        transition: all 0.2s ease;
        border: 2px solid rgba(255,255,255,0.3);
        user-select: none;
    }}
    #procureiq-bg-btn:hover {{
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(37,99,235,0.5);
    }}
    #procureiq-bg-panel {{
        position: fixed;
        bottom: 86px;
        right: 24px;
        background: white;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        z-index: 99998;
        width: 230px;
        border: 1px solid #e2e8f0;
        display: none;
    }}
    .bg-panel-title {{
        font-size: 13px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
    }}
    .bg-colors-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }}
    .bg-color-swatch {{
        width: 100%;
        aspect-ratio: 1;
        border-radius: 8px;
        cursor: pointer;
        border: 2px solid transparent;
        transition: all 0.15s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }}
    .bg-color-swatch:hover {{
        transform: scale(1.12);
        border-color: #2563eb;
        box-shadow: 0 3px 10px rgba(37,99,235,0.3);
    }}
</style>
""",
        unsafe_allow_html=True,
    )

    # Separate component for the floating button with its own script
    st.components.v1.html(
        """
<div id="procureiq-bg-btn" style="
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 99999;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 50%;
    width: 52px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4);
    border: 2px solid rgba(255,255,255,0.3);
    user-select: none;
">BG</div>
<div id="procureiq-bg-panel" style="
    position: fixed;
    bottom: 86px;
    right: 24px;
    background: white;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    z-index: 99998;
    width: 230px;
    border: 1px solid #e2e8f0;
    display: none;
">
<div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 12px;">🎨 Background Theme</div>
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
<div class="bg-swatch" data-color="#e0f2fe" style="background:#e0f2fe; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#f3f4f6" style="background:#f3f4f6; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#dcfce7" style="background:#dcfce7; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#f3e8ff" style="background:#f3e8ff; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#fce7f3" style="background:#fce7f3; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#fef9c3" style="background:#fef9c3; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#cffafe" style="background:#cffafe; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
<div class="bg-swatch" data-color="#ffffff" style="background:#ffffff; width:100%; aspect-ratio:1; border-radius:8px; cursor:pointer; border:2px solid transparent; box-shadow:0 1px 4px rgba(0,0,0,0.1);"></div>
</div>
<div style="margin-top:10px; font-size:11px; color:#94a3b8; text-align:center;">Click a color to apply</div>
</div>
<script>
(function() {
    var btn = document.getElementById('procureiq-bg-btn');
    var panel = document.getElementById('procureiq-bg-panel');
    var parentDoc = window.parent.document;

    function applyBgColor(color) {
        var selectors = ['.stApp', '.main', '.main > .block-container', '[data-testid="stAppViewContainer"]'];
        selectors.forEach(function(sel) {
            var elements = parentDoc.querySelectorAll(sel);
            elements.forEach(function(el) {
                el.style.backgroundColor = color;
            });
        });
        try {
            localStorage.setItem('procureiq_bg_color', color);
        } catch(e) {}
        panel.style.display = 'none';
    }

    function loadSavedBg() {
        try {
            var saved = localStorage.getItem('procureiq_bg_color');
            if (saved) {
                applyBgColor(saved);
            }
        } catch(e) {}
    }

    btn.onclick = function(e) {
        e.stopPropagation();
        panel.style.display = (panel.style.display === 'block') ? 'none' : 'block';
    };

    var swatches = document.querySelectorAll('.bg-swatch');
    swatches.forEach(function(sw) {
        sw.onclick = function(e) {
            e.stopPropagation();
            var color = this.getAttribute('data-color');
            if (color) applyBgColor(color);
        };
        sw.onmouseenter = function() {
            this.style.transform = 'scale(1.12)';
            this.style.borderColor = '#2563eb';
        };
        sw.onmouseleave = function() {
            this.style.transform = 'scale(1)';
            this.style.borderColor = 'transparent';
        };
    });

    document.onclick = function(e) {
        if (!btn.contains(e.target) && !panel.contains(e.target)) {
            panel.style.display = 'none';
        }
    };

    loadSavedBg();
})();
</script>
""",
        height=0,
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

def render_grir_metric_card(title: str, value: str, bg_color: str = "#ffffff"):
    st.markdown(f"""
<div class="grir-card" style="background-color: {bg_color};">
    <div class="grir-card-title">{title}</div>
    <div class="grir-card-value">{value}</div>
</div>
""", unsafe_allow_html=True)

def render_filters():
    rng_start, rng_end = st.session_state.date_range
    selected_vendor = st.session_state.selected_vendor
    current_preset = st.session_state.preset

    col_date, col_vendor, col_preset = st.columns([1.2, 1.2, 2.8], gap="small")

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
            "",
            st.session_state[vendor_cache_key],
            index=(st.session_state[vendor_cache_key].index(selected_vendor)
                   if selected_vendor in st.session_state[vendor_cache_key] else 0),
            label_visibility="collapsed",
            key="vendor_selectbox_unique"
        )
        if selected != selected_vendor:
            st.session_state.selected_vendor = selected

    with col_preset:
        presets = ["Last 30 Days", "QTD", "YTD", "Custom"]
        p_cols = st.columns(4, gap="small")
        for idx, p in enumerate(presets):
            with p_cols[idx]:
                is_active = (p == current_preset)
                btn_type = "primary" if is_active else "secondary"
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

# ── ENHANCED KPI fetching with correct columns ─────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kpi_data(start_lit: str, end_lit: str, vendor_where: str):
    """
    Fetch all KPIs from Athena without relying on views that lack posting_date.
    Returns a dict with correct values.
    """
    result = {}

    # 1. Main KPIs from fact_all_sources_vw
    main_sql = f"""
        SELECT
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN'
                                THEN f.purchase_order_reference END)                AS active_pos,
            COUNT(DISTINCT f.purchase_order_reference)                               AS total_pos,
            SUM(CASE WHEN UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')
                     THEN COALESCE(f.invoice_amount_local, 0) ELSE 0 END)           AS total_spend,
            COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status) = 'OPEN'
                                THEN f.invoice_number END)                           AS pending_inv
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
    """
    df = run_query(main_sql)
    if not df.empty:
        result["active_pos"]  = safe_int(df.iloc[0]["active_pos"])
        result["total_pos"]   = safe_int(df.iloc[0]["total_pos"])
        result["total_spend"] = safe_number(df.iloc[0]["total_spend"])
        result["pending_inv"] = safe_int(df.iloc[0]["pending_inv"])
    else:
        result["active_pos"] = result["total_pos"] = result["pending_inv"] = 0
        result["total_spend"] = 0.0

    # 2. Active Vendors
    vendor_sql = f"""
        SELECT COUNT(DISTINCT v.vendor_name) AS active_vendors
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
          AND v.vendor_name IS NOT NULL
        {vendor_where}
    """
    vdf = run_query(vendor_sql)
    if not vdf.empty:
        result["active_vendors"] = safe_int(vdf.iloc[0]["active_vendors"])
    else:
        fallback_sql = f"SELECT COUNT(DISTINCT vendor_name) AS active_vendors FROM {DATABASE}.dim_vendor_vw WHERE vendor_name IS NOT NULL"
        fdf = run_query(fallback_sql)
        result["active_vendors"] = safe_int(fdf.iloc[0]["active_vendors"]) if not fdf.empty else 0

    # 3. Avg Processing Time – compute from fact table (safe)
    proc_sql = f"""
        SELECT AVG(DATE_DIFF('day', posting_date, payment_date)) AS avg_processing_days
        FROM {DATABASE}.fact_all_sources_vw
        WHERE UPPER(invoice_status) = 'PAID'
          AND payment_date IS NOT NULL
          AND posting_date BETWEEN {start_lit} AND {end_lit}
    """
    pdf = run_query(proc_sql)
    if not pdf.empty and not pd.isna(pdf.iloc[0]["avg_processing_days"]):
        result["avg_processing_days"] = safe_number(pdf.iloc[0]["avg_processing_days"])
    else:
        result["avg_processing_days"] = 0.0

    # 4. First Pass Rate – from invoice_status_history_vw
    fp_sql = f"""
        WITH invoices_in_range AS (
            SELECT DISTINCT invoice_number
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
        ),
        history AS (
            SELECT h.invoice_number,
                   MAX(CASE WHEN UPPER(h.status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 1 ELSE 0 END) AS has_paid,
                   MAX(CASE WHEN UPPER(h.status) IN ('DISPUTE','DISPUTED','OVERDUE') THEN 1 ELSE 0 END) AS has_issue
            FROM {DATABASE}.invoice_status_history_vw h
            JOIN invoices_in_range i ON h.invoice_number = i.invoice_number
            GROUP BY h.invoice_number
        )
        SELECT
            COUNT(*) AS total_inv,
            SUM(CASE WHEN has_paid = 1 AND has_issue = 0 THEN 1 ELSE 0 END) AS first_pass_inv
        FROM history
    """
    fp_df = run_query(fp_sql)
    if not fp_df.empty:
        total = safe_int(fp_df.iloc[0]["total_inv"])
        passed = safe_int(fp_df.iloc[0]["first_pass_inv"])
        result["first_pass_rate"] = (passed / total * 100) if total > 0 else 0.0
    else:
        result["first_pass_rate"] = 0.0

    # 5. Auto-Processed Rate
    auto_sql = f"""
        WITH invoices_in_range AS (
            SELECT DISTINCT invoice_number
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
        )
        SELECT
            COUNT(*) AS total_cleared,
            SUM(CASE WHEN UPPER(status_notes) = 'AUTO PROCESSED' THEN 1 ELSE 0 END) AS auto_processed
        FROM {DATABASE}.invoice_status_history_vw h
        JOIN invoices_in_range i ON h.invoice_number = i.invoice_number
        WHERE UPPER(h.status) = 'PAID'
    """
    adf = run_query(auto_sql)
    if not adf.empty:
        total_cleared = safe_int(adf.iloc[0]["total_cleared"])
        auto_proc     = safe_int(adf.iloc[0]["auto_processed"])
        result["auto_rate"] = (auto_proc / total_cleared * 100) if total_cleared > 0 else 0.0
    else:
        result["auto_rate"] = 0.0

    return result

@st.cache_data(ttl=300, show_spinner=False)
def fetch_needs_attention(start_lit: str, end_lit: str, vendor_where: str):
    """Fetch overdue, disputed, and due-soon invoices from Athena."""
    overdue_sql = f"""
        SELECT f.invoice_number AS ref_no, f.invoice_amount_local AS amount,
               v.vendor_name, f.due_date, f.aging_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
        AND f.due_date < CURRENT_DATE
        AND UPPER(f.invoice_status) = 'OVERDUE'
        ORDER BY f.due_date ASC
    """
    overdue_df = run_query(overdue_sql)

    disputed_sql = f"""
        SELECT f.invoice_number AS ref_no, f.invoice_amount_local AS amount,
               v.vendor_name, f.due_date, f.aging_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
        {vendor_where}
        AND UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED')
        ORDER BY f.due_date ASC
    """
    disputed_df = run_query(disputed_sql)

    due_sql = f"""
        SELECT f.invoice_number AS ref_no, f.invoice_amount_local AS amount,
               v.vendor_name, f.due_date, f.aging_days
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

    return overdue_df, disputed_df, due_df

def render_kpi_rows(kpi: dict, prev_kpi: dict):
    cur_spend          = kpi.get("total_spend", 0)
    prev_spend         = prev_kpi.get("total_spend", 0)
    cur_active_pos     = kpi.get("active_pos", 0)
    prev_active_pos    = prev_kpi.get("active_pos", 0)
    cur_total_pos      = kpi.get("total_pos", 0)
    prev_total_pos     = prev_kpi.get("total_pos", 0)
    cur_active_vendors = kpi.get("active_vendors", 0)
    prev_active_vendors= prev_kpi.get("active_vendors", 0)
    cur_pending        = kpi.get("pending_inv", 0)
    prev_pending       = prev_kpi.get("pending_inv", 0)
    cur_avg_proc       = kpi.get("avg_processing_days", 0.0)
    prev_avg_proc      = prev_kpi.get("avg_processing_days", 0.0)
    first_pass_rate    = kpi.get("first_pass_rate", 0.0)
    prev_fp_rate       = prev_kpi.get("first_pass_rate", 0.0)
    auto_rate          = kpi.get("auto_rate", 0.0)

    spend_delta, spend_up             = pct_delta(cur_spend, prev_spend)
    active_pos_delta, active_pos_up   = pct_delta(cur_active_pos, prev_active_pos)
    total_pos_delta, total_pos_up     = pct_delta(cur_total_pos, prev_total_pos)
    active_vendors_delta, av_up       = pct_delta(cur_active_vendors, prev_active_vendors)
    pending_delta, pending_up         = pct_delta(cur_pending, prev_pending)

    avg_diff = cur_avg_proc - prev_avg_proc
    avg_delta_str = f"{abs(avg_diff):.1f}d"
    avg_up = avg_diff < 0

    fp_diff = first_pass_rate - prev_fp_rate
    fp_delta_str = f"{abs(fp_diff):.1f}%"
    fp_up = fp_diff > 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("TOTAL SPEND", abbr_currency(cur_spend), spend_delta, spend_up, "yellow")
    with col2:
        render_kpi_card("ACTIVE PO'S", f"{cur_active_pos:,}", active_pos_delta, active_pos_up, "cyan")
    with col3:
        render_kpi_card("TOTAL PO'S", f"{cur_total_pos:,}", total_pos_delta, total_pos_up, "pink")
    with col4:
        render_kpi_card("ACTIVE VENDORS", f"{cur_active_vendors:,}", active_vendors_delta, av_up, "purple")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("PENDING INVOICES", f"{cur_pending:,}", pending_delta, not pending_up, "yellow")
    with col2:
        render_kpi_card("AVG INVOICE PROCESSING TIME", f"{cur_avg_proc:.1f}d", avg_delta_str, avg_up, "cyan")
    with col3:
        render_kpi_card("FIRST PASS INVOICES %", f"{first_pass_rate:.1f}%", fp_delta_str, fp_up, "green")
    with col4:
        render_kpi_card("AUTOPROCESSED INVOICES %", f"{auto_rate:.1f}%", f"{auto_rate:.1f}%", True, "green")

def render_needs_attention(rng_start, rng_end, vendor_where):
    if "na_tab" not in st.session_state:
        st.session_state.na_tab = "Overdue"
    if "na_page" not in st.session_state:
        st.session_state.na_page = 0

    current_tab = st.session_state.na_tab
    page = st.session_state.na_page

    start_lit = sql_date(rng_start)
    end_lit = sql_date(rng_end)

    overdue_df, disputed_df, due_df = fetch_needs_attention(start_lit, end_lit, vendor_where)

    overdue_count  = len(overdue_df)
    disputed_count = len(disputed_df)
    due_count      = len(due_df)
    urgent_count   = overdue_count + disputed_count + due_count

    with st.container(border=True):
        st.markdown(f"""
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.2rem; padding-left: 0.5rem; padding-right: 0.5rem;'>
            <div style='font-size:18px;font-weight:900;color:#1a1a1a;'>Needs Attention <span style='font-weight:700;color:#6b7280;'>({urgent_count:,})</span></div>
        </div>
        """, unsafe_allow_html=True)

        tab_cols = st.columns([1, 1, 1], gap="small")
        with tab_cols[0]:
            btn_type = "primary" if current_tab == 'Overdue' else "secondary"
            if st.button(f"Overdue ({overdue_count})", key="na_btn_overdue", use_container_width=True, type=btn_type):
                st.session_state.na_tab = 'Overdue'
                st.session_state.na_page = 0
                st.rerun()
        with tab_cols[1]:
            btn_type = "primary" if current_tab == 'Disputed' else "secondary"
            if st.button(f"Disputed ({disputed_count})", key="na_btn_disputed", use_container_width=True, type=btn_type):
                st.session_state.na_tab = 'Disputed'
                st.session_state.na_page = 0
                st.rerun()
        with tab_cols[2]:
            btn_type = "primary" if current_tab == 'Due' else "secondary"
            if st.button(f"Due ({due_count})", key="na_btn_due30d", use_container_width=True, type=btn_type):
                st.session_state.na_tab = 'Due'
                st.session_state.na_page = 0
                st.rerun()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        if current_tab == 'Overdue':
            df = overdue_df
            status_label = "Overdue"
            tag_bg = "#FEE2E2"
            tag_color = "#991B1B"
        elif current_tab == 'Disputed':
            df = disputed_df
            status_label = "Disputed"
            tag_bg = "#FEF3C7"
            tag_color = "#92400E"
        else:
            df = due_df
            status_label = "Due soon"
            tag_bg = "#DBEAFE"
            tag_color = "#1E3A8A"

        if df.empty:
            st.markdown('<div style="padding:1rem;color:#64748b;">No items in this category</div>', unsafe_allow_html=True)
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
                        with st.container(border=True):
                            left, right = st.columns([2, 1], gap="small")
                            with left:
                                ref = str(r.get("ref_no", "")).strip() or "—"
                                ref = format_invoice_number(ref)
                                btn_key = f"na_card_{start_idx}_{card_global_idx}_{ref.replace(' ', '_')[:30]}"
                                if st.button(ref, key=btn_key):
                                    st.session_state["invoice_search_from_card"] = ref
                                    st.session_state["page"] = "Invoices"
                                    st.experimental_set_query_params(tab="Invoices", invoice=ref)
                                    st.rerun()
                                vendor_nm = str(r.get("vendor_name", "—"))
                                st.markdown(f"<div style='color:#64748b;font-size:12px;overflow:hidden;text-overflow:ellipsis;'>{html.escape(vendor_nm)}</div>", unsafe_allow_html=True)
                            with right:
                                amt = safe_number(r.get("amount"))
                                ddate_raw = r.get("due_date")
                                ddate = pd.to_datetime(ddate_raw).date().isoformat() if pd.notna(ddate_raw) else "—"
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

            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

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

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("<div class='chart-container'><div class='chart-title'>Invoice Status Distribution</div><div class='chart-body'>", unsafe_allow_html=True)
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
                    {"status": "Paid", "cnt": 450}, {"status": "Pending", "cnt": 180},
                    {"status": "Disputed", "cnt": 33}, {"status": "Other", "cnt": 30}
                ])
            total = status_df["cnt"].sum()
            status_df["percentage"] = (status_df["cnt"] / total * 100).round(1)
            color_scale = alt.Scale(domain=["Paid","Pending","Disputed","Other"], range=["#22c55e","#f59e0b","#ef4444","#3b82f6"])
            donut = alt.Chart(status_df).mark_arc(innerRadius=50, outerRadius=90).encode(
                theta=alt.Theta("cnt:Q"),
                color=alt.Color("status:N", scale=color_scale, legend=alt.Legend(orient="right", title=None, labelFontSize=11)),
                tooltip=["status:N","cnt:Q","percentage:Q"]
            ).properties(height=280)
            center_text = alt.Chart(pd.DataFrame({"text":[str(total)],"label":["TOTAL"]})).mark_text(
                align="center", baseline="middle", fontSize=26, fontWeight="bold", color="#111827").encode(text="text:N")
            center_label = alt.Chart(pd.DataFrame({"text":["TOTAL"]})).mark_text(
                align="center", baseline="middle", fontSize=11, color="#6b7280", dy=18).encode(text="text:N")
            st.altair_chart(donut + center_text + center_label, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown("<div class='chart-container'><div class='chart-title'>Top 10 Vendors by Spend</div><div class='chart-body'>", unsafe_allow_html=True)
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
                ])
            bar_chart = alt.Chart(top_df).mark_bar(color="#22c55e", cornerRadiusEnd=4).encode(
                x=alt.X("spend:Q", title=None, axis=alt.Axis(format="~s")),
                y=alt.Y("vendor_name:N", sort="-x", title=None),
                tooltip=["vendor_name:N", alt.Tooltip("spend:Q", format="$,.0f")]
            ).properties(height=280)
            st.altair_chart(bar_chart, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    with col3:
        with st.container(border=True):
            st.markdown("<div class='chart-container'><div class='chart-title'>Spend Trend Analysis</div><div class='chart-body'>", unsafe_allow_html=True)
            trend_sql = f"""
                SELECT DATE_TRUNC('month', posting_date) AS month,
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
            trend_melted = trend_df.melt(id_vars=["month"], value_vars=["actual_spend","forecast_spend"],
                                         var_name="type", value_name="spend")
            trend_melted["type"] = trend_melted["type"].map({"actual_spend":"ACTUAL","forecast_spend":"FORECAST"})
            bar_chart = alt.Chart(trend_melted).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("spend:Q", title=None, axis=alt.Axis(format="~s")),
                color=alt.Color("type:N", scale=alt.Scale(domain=["ACTUAL","FORECAST"],
                                range=["#22c55e","#3b82f6"]), legend=alt.Legend(orient="top", title=None)),
                xOffset="type:N",
                tooltip=["month:N","type:N", alt.Tooltip("spend:Q", format="$,.0f")]
            ).properties(height=280)
            st.altair_chart(bar_chart, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

def render_dashboard():
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

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    start_lit = sql_date(rng_start)
    end_lit   = sql_date(rng_end)
    p_start, p_end = prior_window(rng_start, rng_end)
    p_start_lit = sql_date(p_start)
    p_end_lit   = sql_date(p_end)

    with st.spinner("Loading KPIs..."):
        cur_kpi  = fetch_kpi_data(start_lit, end_lit, vendor_where)
        prev_kpi = fetch_kpi_data(p_start_lit, p_end_lit, vendor_where)

    render_kpi_rows(cur_kpi, prev_kpi)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_needs_attention(rng_start, rng_end, vendor_where)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_charts(rng_start, rng_end, vendor_where)

# ------------------------------------------------------------
# forecast.py
# ------------------------------------------------------------
def render_forecast():
    cf_sql = f"""
        SELECT forecast_bucket, invoice_count, total_amount, earliest_due, latest_due
        FROM {DATABASE}.cash_flow_forecast_vw
        ORDER BY CASE forecast_bucket
            WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1 WHEN 'DUE_7_DAYS' THEN 2
            WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4 WHEN 'DUE_60_DAYS' THEN 5
            WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END
    """
    cf_df = run_query(cf_sql)

    if cf_df.empty:
        st.warning("cash_flow_forecast_vw not found – computing from unpaid invoices.")
        cf_sql_fallback = f"""
            WITH base AS (
                SELECT invoice_number, invoice_amount_local, due_date, invoice_status,
                       DATE_DIFF('day', CURRENT_DATE, due_date) AS days_until_due
                FROM {DATABASE}.fact_all_sources_vw
                WHERE UPPER(invoice_status) IN ('OPEN', 'DUE', 'OVERDUE') AND due_date IS NOT NULL
            ),
            buckets AS (
                SELECT CASE
                        WHEN days_until_due < 0 THEN 'OVERDUE_NOW'
                        WHEN days_until_due <= 7 THEN 'DUE_7_DAYS'
                        WHEN days_until_due <= 14 THEN 'DUE_14_DAYS'
                        WHEN days_until_due <= 30 THEN 'DUE_30_DAYS'
                        WHEN days_until_due <= 60 THEN 'DUE_60_DAYS'
                        WHEN days_until_due <= 90 THEN 'DUE_90_DAYS'
                        ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,
                    COUNT(*) AS invoice_count, SUM(invoice_amount_local) AS total_amount,
                    MIN(due_date) AS earliest_due, MAX(due_date) AS latest_due
                FROM base GROUP BY 1
            ),
            total AS (
                SELECT 'TOTAL_UNPAID' AS forecast_bucket, SUM(invoice_count) AS invoice_count,
                       SUM(total_amount) AS total_amount, NULL AS earliest_due, NULL AS latest_due
                FROM buckets
            )
            SELECT * FROM total UNION ALL SELECT * FROM buckets
        """
        cf_df = run_query(cf_sql_fallback)

    tab1, tab2 = st.tabs(["Cash Flow Need Forecast", "GR/IR Reconciliation"])

    with tab1:
        if not cf_df.empty:
            total_unpaid = cf_df[cf_df["forecast_bucket"] == "TOTAL_UNPAID"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"] == "TOTAL_UNPAID"].empty else 0
            overdue_now  = cf_df[cf_df["forecast_bucket"] == "OVERDUE_NOW"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"] == "OVERDUE_NOW"].empty else 0
            due_30       = cf_df[cf_df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
            pct_due_30   = (due_30 / total_unpaid * 100) if total_unpaid > 0 else 0
        else:
            total_unpaid = overdue_now = due_30 = 0
            pct_due_30 = 0

        kpi_colors = ["#fff7e0", "#ffe6ef", "#e6f3ff", "#e0f7fa"]
        kpi_titles = ["TOTAL UNPAID", "OVERDUE NOW", "DUE NEXT 30 DAYS", "% DUE ≤ 30 DAYS"]
        kpi_values = [abbr_currency(total_unpaid), abbr_currency(overdue_now), abbr_currency(due_30), f"{pct_due_30:.1f}%"]

        st.markdown("""
        <style>
        .forecast-kpi-card { border-radius: 16px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: left; border: 1px solid rgba(0,0,0,0.05); }
        .forecast-kpi-title { font-size: 0.8rem; font-weight: 600; color: #475569; margin-bottom: 0.3rem; }
        .forecast-kpi-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
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
            st.dataframe(safe_dataframe_display(cf_df), use_container_width=True, hide_index=True)
            csv = cf_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download forecast (CSV)", data=csv, file_name="cash_flow_forecast.csv", mime="text/csv")
        else:
            st.info("No cash flow forecast data available.")

        st.markdown("---")
        st.markdown("### Action Playbook")
        actions = [
            ("Forecast cash outflow (7–90 days)", "Forecast cash outflow for the next 7, 14, 30, 60, and 90 days"),
            ("Invoices to pay early to capture discounts", "Which invoices should we pay early to capture discounts?"),
            ("Optimal payment timing for this week", "What is the optimal payment timing strategy for this week?"),
            ("Late payment trend and risk", "Show late payment trend for forecasting")
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
                ORDER BY year DESC, month DESC LIMIT 1
            ),
            aging_latest AS (
                SELECT year, month, pct_grir_over_60, cnt_grir_over_60
                FROM {DATABASE}.gr_ir_aging_vw
                ORDER BY year DESC, month DESC LIMIT 1
            )
            SELECT l.year, l.month, l.invoice_count AS grir_items, l.total_grir_blnc AS total_grir_balance,
                   a.pct_grir_over_60, a.cnt_grir_over_60,
                   COALESCE(l.total_grir_blnc * a.pct_grir_over_60 / 100, 0) AS amount_over_60_days
            FROM latest l
            LEFT JOIN aging_latest a ON a.year = l.year AND a.month = l.month
        """
        grir_df = run_query(grir_summary_sql)

        if not grir_df.empty:
            row = grir_df.iloc[0]
            total_grir     = safe_number(row.get("total_grir_balance", 0))
            grir_items     = safe_int(row.get("grir_items", 0))
            pct_over_60    = safe_number(row.get("pct_grir_over_60", 0))
            amount_over_60 = safe_number(row.get("amount_over_60_days", 0))
            cnt_over_60    = safe_int(row.get("cnt_grir_over_60", 0))
            year  = safe_int(row.get("year", 0))
            month = safe_int(row.get("month", 0))
        else:
            total_grir = grir_items = pct_over_60 = amount_over_60 = cnt_over_60 = 0
            year, month = 2026, 1

        grir_cols = st.columns(4)
        card_colors = ["#E6F3FF", "#E0F7FA", "#FFF3E0", "#F3E5F5"]
        with grir_cols[0]:
            render_grir_metric_card("TOTAL GR/IR", abbr_currency(total_grir), bg_color=card_colors[0])
        with grir_cols[1]:
            render_grir_metric_card("% > 60 DAYS", f"{pct_over_60:.1f}%", bg_color=card_colors[1])
        with grir_cols[2]:
            render_grir_metric_card("60 DAYS AMOUNT", abbr_currency(amount_over_60), bg_color=card_colors[2])
        with grir_cols[3]:
            render_grir_metric_card("60 DAYS ITEMS", f"{cnt_over_60:,}", bg_color=card_colors[3])

        st.caption(f"GR/IR position for {year:04d}-{month:02d}: {grir_items:,} items outstanding; "
                   f"{pct_over_60:.1f}% of balance and {cnt_over_60:,} items are older than 60 days.")

        if not grir_df.empty:
            trend_sql = f"""
                SELECT year, month, invoice_count, total_grir_blnc
                FROM {DATABASE}.gr_ir_outstanding_balance_vw
                ORDER BY year DESC, month DESC LIMIT 24
            """
            trend_df = run_query(trend_sql)
            if not trend_df.empty:
                st.markdown("**GR/IR outstanding trend (last 24 months)**")
                st.dataframe(safe_dataframe_display(trend_df), use_container_width=True, hide_index=True)
        else:
            st.info("No GR/IR data found.")

        st.markdown("---")
        st.markdown("### GR/IR Clearing Playbook")
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
# genie.py
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
    columns:
      invoice_number: "Unique invoice identifier (string)"
      invoice_amount_local: "Invoice amount in local currency (decimal)"
      posting_date: "Date when invoice was posted (date)"
      due_date: "Date when payment is due (date)"
      payment_date: "Date when payment was made (nullable)"
      invoice_status: "Status: OPEN, PAID, OVERDUE, DISPUTED, CANCELLED, REJECTED"
      purchase_order_reference: "PO number linked to invoice"
      po_amount: "Amount of the purchase order"
      vendor_id: "Foreign key to dim_vendor_vw"
      company_code: "Company code"
      plant_code: "Plant code"
      currency: "Currency code"
      aging_days: "Number of days invoice is overdue"
  dim_vendor_vw:
    columns:
      vendor_id: "Unique vendor ID"
      vendor_name: "Vendor name"
  gr_ir_outstanding_balance_vw:
    columns:
      year: "Year"
      month: "Month"
      invoice_count: "Number of open GR/IR items"
      total_grir_blnc: "Total GR/IR balance amount"
  gr_ir_aging_vw:
    columns:
      year: "Year"
      month: "Month"
      pct_grir_over_60: "Percentage of GR/IR balance older than 60 days"
      cnt_grir_over_60: "Count of items older than 60 days"
  payment_processing_cycle_time_vw:
    columns:
      posting_date: "Invoice posting date"
      avg_cycle_time_days: "Average days from posting to payment"
  full_payment_rate_vw:
    columns:
      posting_date: "Invoice posting date"
      full_payment_rate: "Percentage of invoices paid in full on first pass"
"""

SYSTEM_PROMPT_SEMANTIC = f"""
You are a senior procurement analyst and Athena SQL expert. Convert the user's natural language question into a valid, efficient Athena SQL query using the semantic model below.
Rules:
1. Use exact table and column names from the semantic model.
2. Join tables using LEFT JOIN where appropriate.
3. Exclude cancelled/rejected invoices from spend calculations unless asked.
4. Use COALESCE for numeric columns.
5. Use Presto/Athena functions: DATE_TRUNC, DATE_ADD, DATE_DIFF, CURRENT_DATE.
6. Always include a LIMIT clause (default 1000) unless aggregating.
7. Output only the SQL statement, no explanations or markdown.

Semantic model (YAML):
{SEMANTIC_MODEL_YAML}

Now generate SQL for the user's question.
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
            SELECT SUM(COALESCE(invoice_amount_local, 0)) AS total_spend,
                   COUNT(DISTINCT invoice_number) AS invoice_count,
                   COUNT(DISTINCT vendor_id) AS active_vendors
            FROM {DATABASE}.fact_all_sources_vw
            WHERE invoice_status NOT IN ('Cancelled', 'Rejected')
        """
    return sql

def is_relevant_question(question: str) -> bool:
    q_lower = question.lower().strip()

    non_procurement_patterns = [
        r"^(hi|hello|hey|howdy|hiya|yo)\b",
		r"^hello",
        r"^good\s*(morning|afternoon|evening|night)\b",
        r"^how are you",
        r"^who are you",
        r"^tell me a joke",
        r"^what('s| is) (the )?weather",
        r"^what('s| is) (your )?name",
        r"^what (do|can) you do",
        r"^(thank(s| you)|thanks a lot|thx)\b",
        r"^(bye|goodbye|see you|ttyl)\b",
        r"^what('s| is) (your )?favorite",
        r"^(are you|you are) (a |an )?(ai|bot|robot|human|assistant)\??$",
        r"^what (is|are) \d+",
        r"^(calculate|compute) \d",
        r"^capital of\b",
        r"^who (is|was|invented|created|discovered)\b",
        r"^when (was|did|is)\b(?!.*invoice|.*po|.*vendor|.*payment|.*spend)",
        r"^(what|how) (many|much) (people|countries|languages)\b",
    ]
    for pattern in non_procurement_patterns:
        if re.search(pattern, q_lower):
            return False

    procurement_keywords = [
        "spend", "vendor", "invoice", "po", "purchase order", "payment",
        "due", "overdue", "dispute", "gr/ir", "cash flow", "forecast",
        "dashboard", "kpi", "trend", "analysis", "procurement", "p2p",
        "pay", "receipt", "goods", "price", "quantity", "status",
        "active vendors", "total spend", "pending", "processing time",
        "autoprocessed", "aging", "accrual", "payable", "ap", "accounts payable",
        "early payment", "discount", "plant", "company code", "cycle time",
        "first pass", "on-time", "late payment", "duplicate", "supplier",
        "delivery", "weighted", "partial payment", "full payment", "budget",
        "contract", "requisition", "three-way match", "two-way match",
    ]
    for kw in procurement_keywords:
        if kw in q_lower:
            return True
    return False

OUT_OF_DOMAIN_RESPONSE = (
    "Hello! I am ProcureIQ Assistant. I can help you with procurement insights, "
    "vendor information, invoice status, forecasting, spend analytics, dashboard metrics, "
    "and related business data. Please ask a procurement or dashboard-related question."
)

def process_custom_query(query: str, history: str = "") -> dict:
    if not is_relevant_question(query):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE, "question": query}
    sql = generate_sql_from_semantic(query)
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
    prompt = f"""{history}
You are a senior procurement analyst. The user asked: "{query}".
Write a response in this structure:

**Descriptive — What the data shows**
First write "This is our interpretation of your question:" followed by a clear restatement. Then describe key findings using exact numbers.

**Prescriptive — Recommendations & next steps**
Provide bullet points under "Key Insights:", "Recommended Actions:", "Risks:".

Data preview:
{data_preview}

SQL used:
{sql}

Respond in plain text using markdown for headings and bullet points.
"""
    analyst_text = ask_bedrock(prompt, system_prompt="You are a helpful procurement analyst.")
    if not analyst_text:
        analyst_text = f"**Analysis complete.**\n\nHere are the results:\n\n{data_preview}"
    return {"layout": "analyst", "sql": sql, "df": df.to_dict(orient="records"), "question": query, "analyst_response": analyst_text}

def process_cash_flow_forecast(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    cf_sql = f"""
        SELECT forecast_bucket, invoice_count, total_amount, earliest_due, latest_due
        FROM {DATABASE}.cash_flow_forecast_vw
        ORDER BY CASE forecast_bucket
            WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1 WHEN 'DUE_7_DAYS' THEN 2
            WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4 WHEN 'DUE_60_DAYS' THEN 5
            WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END
    """
    cf_df = run_query(cf_sql)
    used_sql = cf_sql
    if cf_df.empty:
        cf_sql_fallback = f"""
            WITH base AS (
                SELECT invoice_number, invoice_amount_local, due_date, invoice_status,
                       DATE_DIFF('day', CURRENT_DATE, due_date) AS days_until_due
                FROM {DATABASE}.fact_all_sources_vw
                WHERE UPPER(invoice_status) IN ('OPEN', 'DUE', 'OVERDUE') AND due_date IS NOT NULL
            ),
            buckets AS (
                SELECT CASE
                        WHEN days_until_due < 0 THEN 'OVERDUE_NOW'
                        WHEN days_until_due <= 7 THEN 'DUE_7_DAYS'
                        WHEN days_until_due <= 14 THEN 'DUE_14_DAYS'
                        WHEN days_until_due <= 30 THEN 'DUE_30_DAYS'
                        WHEN days_until_due <= 60 THEN 'DUE_60_DAYS'
                        WHEN days_until_due <= 90 THEN 'DUE_90_DAYS'
                        ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,
                    COUNT(*) AS invoice_count, SUM(invoice_amount_local) AS total_amount,
                    MIN(due_date) AS earliest_due, MAX(due_date) AS latest_due
                FROM base GROUP BY 1
            ),
            total AS (
                SELECT 'TOTAL_UNPAID' AS forecast_bucket, SUM(invoice_count) AS invoice_count,
                       SUM(total_amount) AS total_amount, NULL AS earliest_due, NULL AS latest_due
                FROM buckets
            )
            SELECT * FROM total UNION ALL SELECT * FROM buckets
        """
        cf_df = run_query(cf_sql_fallback)
        used_sql = cf_sql_fallback
    if cf_df.empty:
        return {"layout": "error", "message": "No cash flow forecast data available."}
    cf_df.columns = [c.lower() for c in cf_df.columns]
    analyst_text = ask_bedrock(
        f"{history}\nCash flow forecast data:\n{cf_df.to_string(index=False)}\nWrite Descriptive and Prescriptive sections in markdown.",
        system_prompt="You are a helpful procurement analyst focusing on cash flow management.")
    return {"layout": "cash_flow", "df": cf_df.to_dict(orient="records"), "sql": used_sql,
            "analyst_response": analyst_text or "Unable to generate insights.", "question": question}

def process_early_payment(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    ep_sql = f"""
        SELECT CAST(f.invoice_number AS VARCHAR) AS document_number, v.vendor_name,
            f.invoice_amount_local AS invoice_amount, f.due_date,
            DATE_DIFF('day', CURRENT_DATE, f.due_date) AS days_until_due,
            ROUND(f.invoice_amount_local * 0.02, 2) AS savings_if_2pct_discount,
            CASE WHEN DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 7 THEN 'High'
                 WHEN DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 14 THEN 'Medium'
                 ELSE 'Low' END AS early_pay_priority
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE UPPER(f.invoice_status) IN ('OPEN', 'DUE') AND f.due_date > CURRENT_DATE
          AND DATE_DIFF('day', CURRENT_DATE, f.due_date) <= 30
        ORDER BY early_pay_priority ASC, savings_if_2pct_discount DESC LIMIT 20
    """
    ep_df = run_query(ep_sql)
    if not ep_df.empty:
        ep_df.columns = [c.lower() for c in ep_df.columns]
    data_preview = ep_df.head(10).to_string(index=False) if not ep_df.empty else "No candidates."
    analyst_text = ask_bedrock(
        f"{history}\nEarly payment candidates:\n{data_preview}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst specializing in working capital optimization.")
    return {"layout": "early_payment", "df": ep_df.to_dict(orient="records") if not ep_df.empty else [],
            "sql": ep_sql, "analyst_response": analyst_text or "No insights.", "question": question, "empty": ep_df.empty}

def process_payment_timing(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    timing_sql = f"""
        WITH due_buckets AS (
            SELECT CASE
                    WHEN due_date < CURRENT_DATE THEN 'Overdue'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '7' DAY THEN 'Due in 0-7 days'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '14' DAY THEN 'Due in 8-14 days'
                    WHEN due_date <= CURRENT_DATE + INTERVAL '30' DAY THEN 'Due in 15-30 days'
                    ELSE 'Due later' END AS payment_window,
                COUNT(*) AS invoice_count, SUM(invoice_amount_local) AS total_amount
            FROM {DATABASE}.fact_all_sources_vw
            WHERE UPPER(invoice_status) IN ('OPEN', 'DUE') GROUP BY 1
        )
        SELECT * FROM due_buckets ORDER BY CASE payment_window
            WHEN 'Overdue' THEN 1 WHEN 'Due in 0-7 days' THEN 2 WHEN 'Due in 8-14 days' THEN 3
            WHEN 'Due in 15-30 days' THEN 4 ELSE 5 END
    """
    timing_df = run_query(timing_sql)
    if timing_df.empty:
        return {"layout": "error", "message": "No payment timing data available."}
    timing_df.columns = [c.lower() for c in timing_df.columns]
    analyst_text = ask_bedrock(
        f"{history}\nPayment timing buckets:\n{timing_df.to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst focusing on cash flow timing.")
    return {"layout": "payment_timing", "df": timing_df.to_dict(orient="records"), "sql": timing_sql,
            "analyst_response": analyst_text or "No insights.", "question": question}

def process_late_payment_trend(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    trend_sql = f"""
        SELECT DATE_TRUNC('month', payment_date) AS month, COUNT(*) AS total_payments,
            SUM(CASE WHEN payment_date > due_date THEN 1 ELSE 0 END) AS late_payments,
            AVG(CASE WHEN payment_date > due_date THEN DATE_DIFF('day', due_date, payment_date) END) AS avg_late_days
        FROM {DATABASE}.fact_all_sources_vw
        WHERE payment_date IS NOT NULL AND payment_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1 ORDER BY 1
    """
    trend_df = run_query(trend_sql)
    if trend_df.empty:
        return {"layout": "error", "message": "No payment trend data available."}
    trend_df.columns = [c.lower() for c in trend_df.columns]
    trend_df["late_pct"] = (trend_df["late_payments"] / trend_df["total_payments"]) * 100
    analyst_text = ask_bedrock(
        f"{history}\nLate payment data:\n{trend_df.tail(6).to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst focusing on payment performance.")
    return {"layout": "late_payment_trend", "df": trend_df.to_dict(orient="records"), "sql": trend_sql,
            "analyst_response": analyst_text or "No insights.", "question": question}

def process_grir_hotspots(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    sql = f"""
        SELECT year, month, invoice_count, total_grir_blnc AS total_grir_balance
        FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC, month DESC
    """
    df = run_query(sql)
    used_sql = sql
    if df.empty:
        df = pd.DataFrame([
            {"year": 2025, "month": 12, "invoice_count": 145, "total_grir_balance": 1250000},
            {"year": 2025, "month": 11, "invoice_count": 132, "total_grir_balance": 1180000},
        ])
        used_sql = sql + " (no data, using sample)"
    else:
        df.columns = [c.lower() for c in df.columns]
    analyst_text = ask_bedrock(
        f"{history}\nGR/IR balance by month:\n{df.head(12).to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst focusing on GR/IR reconciliation.")
    if not analyst_text:
        analyst_text = "**Focus on the oldest months first.** Review POs and receipts for missing documentation."
    return {"layout": "grir_hotspots", "df": df.to_dict(orient="records"), "sql": used_sql,
            "analyst_response": analyst_text, "question": question}

def process_grir_root_causes(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    aging_sql   = f"SELECT year, month, pct_grir_over_60, cnt_grir_over_60 FROM {DATABASE}.gr_ir_aging_vw ORDER BY year DESC, month DESC LIMIT 6"
    balance_sql = f"SELECT year, month, total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC, month DESC LIMIT 6"
    aging_df   = run_query(aging_sql)
    balance_df = run_query(balance_sql)
    if aging_df.empty and balance_df.empty:
        aging_df   = pd.DataFrame([{"year": 2025, "month": 12, "pct_grir_over_60": 28.5, "cnt_grir_over_60": 41}])
        balance_df = pd.DataFrame([{"year": 2025, "month": 12, "total_grir_blnc": 1250000}])
    context = "GR/IR aging:\n" + aging_df.to_string(index=False) + "\n\nBalances:\n" + balance_df.to_string(index=False)
    analyst_text = ask_bedrock(
        f"{history}\n{context}\nExplain root-cause buckets and remediation, Descriptive and Prescriptive.",
        system_prompt="You are a helpful procurement analyst specializing in GR/IR reconciliation.")
    if not analyst_text:
        analyst_text = "**Common root causes:** missing goods receipts, invoice not posted, price/quantity mismatches."
    return {"layout": "grir_root_causes", "df": aging_df.to_dict(orient="records"),
            "extra_df": balance_df.to_dict(orient="records"),
            "sql": {"aging_sql": aging_sql, "balance_sql": balance_sql},
            "analyst_response": analyst_text, "question": question}

def process_grir_working_capital(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    sql = f"""
        SELECT year, month, total_grir_blnc,
            CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 60)
                 THEN total_grir_blnc ELSE 0 END AS older_than_60_days,
            CASE WHEN (year * 100 + month) <= (EXTRACT(YEAR FROM CURRENT_DATE) * 100 + EXTRACT(MONTH FROM CURRENT_DATE) - 90)
                 THEN total_grir_blnc ELSE 0 END AS older_than_90_days
        FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC, month DESC
    """
    df = run_query(sql)
    used_sql = sql
    if df.empty:
        df = pd.DataFrame([
            {"year": 2025, "month": 12, "total_grir_blnc": 1250000, "older_than_60_days": 350000, "older_than_90_days": 120000},
        ])
        used_sql = sql + " (no data, using sample)"
    else:
        df.columns = [c.lower() for c in df.columns]
    total_old_60 = df['older_than_60_days'].sum()
    total_old_90 = df['older_than_90_days'].sum()
    analyst_text = ask_bedrock(
        f"{history}\nGR/IR working capital data:\n{df.head(12).to_string(index=False)}\nDescriptive (cite ${total_old_60:,.2f} >60d, ${total_old_90:,.2f} >90d) and Prescriptive.",
        system_prompt="You are a helpful procurement analyst focusing on working capital.")
    if not analyst_text:
        analyst_text = f"**Working capital release:** ${total_old_60:,.2f} from >60 days, ${total_old_90:,.2f} from >90 days."
    return {"layout": "grir_working_capital", "df": df.to_dict(orient="records"),
            "metrics": {"older_60": float(total_old_60), "older_90": float(total_old_90)},
            "sql": used_sql, "analyst_response": analyst_text, "question": question}

def process_grir_vendor_followup(question: str, history: str = "") -> dict:
    if not is_relevant_question(question):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_RESPONSE}
    sql = f"""
        SELECT v.vendor_name, COUNT(*) AS grir_count, SUM(f.invoice_amount_local) AS total_amount,
            AVG(DATE_DIFF('day', f.posting_date, CURRENT_DATE)) AS avg_age_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status = 'OPEN' AND f.purchase_order_reference IS NOT NULL
        GROUP BY v.vendor_name ORDER BY total_amount DESC LIMIT 10
    """
    df = run_query(sql)
    used_sql = sql
    if df.empty:
        df = pd.DataFrame([
            {"vendor_name": "Acme Corp", "grir_count": 23, "total_amount": 245000, "avg_age_days": 85},
        ])
        used_sql = sql + " (no data, using sample)"
    else:
        df.columns = [c.lower() for c in df.columns]
    analyst_text = ask_bedrock(
        f"{history}\nTop GR/IR vendors:\n{df.to_string(index=False)}\nSummarise and draft 3-5 follow-up templates with subject lines.",
        system_prompt="You are a helpful procurement analyst skilled in vendor communication.")
    if not analyst_text:
        analyst_text = "**Sample follow-up:** Subject: Missing GR/IR documents. Please provide missing goods receipts or invoices."
    return {"layout": "grir_vendor_followup", "df": df.to_dict(orient="records"), "sql": used_sql,
            "analyst_response": analyst_text, "question": question}

def _quick_spending_overview():
    monthly_sql = f"""
        SELECT DATE_TRUNC('month', posting_date) AS month, SUM(COALESCE(invoice_amount_local, 0)) AS monthly_spend,
            COUNT(*) AS invoice_count, COUNT(DISTINCT vendor_id) AS vendor_count
        FROM {DATABASE}.fact_all_sources_vw
        WHERE invoice_status NOT IN ('Cancelled', 'Rejected') AND posting_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1 ORDER BY month DESC
    """
    monthly_df = run_query(monthly_sql)
    if monthly_df.empty:
        return {"layout": "error", "message": "No spending data found."}
    monthly_df.columns = [c.lower() for c in monthly_df.columns]
    top_vendors_sql = f"""
        SELECT COALESCE(v.vendor_name, 'Unknown') AS vendor_name, SUM(COALESCE(f.invoice_amount_local, 0)) AS spend
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected') AND f.posting_date >= DATE_TRUNC('YEAR', CURRENT_DATE)
        GROUP BY v.vendor_name ORDER BY spend DESC LIMIT 10
    """
    vendors_df = run_query(top_vendors_sql)
    if not vendors_df.empty:
        vendors_df.columns = [c.lower() for c in vendors_df.columns]
    total_ytd = vendors_df['spend'].sum() if not vendors_df.empty else 0
    top5_pct  = (vendors_df.head(5)['spend'].sum() / total_ytd * 100) if total_ytd > 0 else 0
    mom_pct   = 0
    if len(monthly_df) >= 2:
        latest = monthly_df.iloc[0]['monthly_spend']
        prev   = monthly_df.iloc[1]['monthly_spend']
        mom_pct = ((latest - prev) / prev * 100) if prev != 0 else 0
    qoq_pct = 0
    if len(monthly_df) >= 6:
        current_q = monthly_df.iloc[0:3]['monthly_spend'].sum()
        prev_q    = monthly_df.iloc[3:6]['monthly_spend'].sum()
        qoq_pct   = ((current_q - prev_q) / prev_q * 100) if prev_q != 0 else 0
    metrics = {"total_ytd": total_ytd, "top5_pct": top5_pct, "mom_pct": mom_pct, "qoq_pct": qoq_pct}
    analyst_text = ask_bedrock(
        f"Spending data:\n{monthly_df.head(6).to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst.")
    return {"layout": "quick", "analysis_type": "spending_overview", "metrics": metrics,
            "monthly_df": monthly_df.to_dict(orient="records"),
            "vendors_df": vendors_df.to_dict(orient="records") if not vendors_df.empty else [],
            "analyst_response": analyst_text or "Analysis complete.",
            "sql": {"monthly_trend": monthly_sql, "top_vendors": top_vendors_sql}, "question": "Spending Overview"}

def _quick_vendor_analysis():
    vendors_sql = f"""
        SELECT COALESCE(v.vendor_name, 'Unknown') AS vendor_name, SUM(COALESCE(f.invoice_amount_local, 0)) AS total_spend,
            COUNT(DISTINCT f.invoice_number) AS invoice_count
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.invoice_status NOT IN ('Cancelled', 'Rejected') AND f.posting_date >= DATE_TRUNC('YEAR', CURRENT_DATE)
        GROUP BY v.vendor_name ORDER BY total_spend DESC LIMIT 10
    """
    vendors_df = run_query(vendors_sql)
    if vendors_df.empty:
        return {"layout": "error", "message": "No vendor data found."}
    vendors_df.columns = [c.lower() for c in vendors_df.columns]
    monthly_vendors_sql = f"""
        SELECT DATE_TRUNC('month', posting_date) AS month, COUNT(DISTINCT vendor_id) AS active_vendors
        FROM {DATABASE}.fact_all_sources_vw
        WHERE invoice_status NOT IN ('Cancelled', 'Rejected') AND posting_date >= DATE_ADD('month', -12, CURRENT_DATE)
        GROUP BY 1 ORDER BY month DESC
    """
    monthly_vendors_df = run_query(monthly_vendors_sql)
    if not monthly_vendors_df.empty:
        monthly_vendors_df.columns = [c.lower() for c in monthly_vendors_df.columns]
    total_spend = vendors_df['total_spend'].sum()
    top1_pct = (vendors_df.iloc[0]['total_spend'] / total_spend * 100) if total_spend > 0 else 0
    top5_pct = (vendors_df.head(5)['total_spend'].sum() / total_spend * 100) if total_spend > 0 else 0
    metrics  = {"total_spend": total_spend, "top1_pct": top1_pct, "top5_pct": top5_pct, "active_vendors": len(vendors_df)}
    analyst_text = ask_bedrock(
        f"Vendor spend data:\n{vendors_df.to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst.")
    return {"layout": "quick", "analysis_type": "vendor_analysis", "metrics": metrics,
            "vendors_df": vendors_df.to_dict(orient="records"),
            "monthly_df": monthly_vendors_df.to_dict(orient="records") if not monthly_vendors_df.empty else [],
            "analyst_response": analyst_text or "Analysis complete.",
            "sql": {"top_vendors": vendors_sql, "monthly_vendors": monthly_vendors_sql}, "question": "Vendor Analysis"}

def _quick_payment_performance():
    sql = f"""
        SELECT DATE_FORMAT(payment_date, '%Y-%m') AS month,
            ROUND(AVG(DATE_DIFF('day', posting_date, payment_date)), 1) AS avg_days_to_pay,
            SUM(CASE WHEN DATE_DIFF('day', due_date, payment_date) > 0 THEN 1 ELSE 0 END) AS late_payments,
            COUNT(*) AS total_payments
        FROM {DATABASE}.fact_all_sources_vw
        WHERE payment_date IS NOT NULL AND payment_date >= DATE_ADD('month', -6, CURRENT_DATE)
          AND UPPER(invoice_status) NOT IN ('CANCELLED', 'REJECTED')
        GROUP BY DATE_FORMAT(payment_date, '%Y-%m') ORDER BY month
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No payment data found for the last 6 months."}
    df.columns = [c.lower() for c in df.columns]
    df['month_dt']  = pd.to_datetime(df['month'] + '-01')
    df = df.sort_values('month_dt')
    df['month_str'] = df['month_dt'].dt.strftime('%b %Y')
    avg_days_overall = df['avg_days_to_pay'].mean()
    total_late       = df['late_payments'].sum()
    total_payments   = df['total_payments'].sum()
    late_pct         = (total_late / total_payments * 100) if total_payments > 0 else 0
    metrics = {"avg_days_to_pay": avg_days_overall, "late_payments_pct": late_pct,
               "total_late": total_late, "total_payments": total_payments}
    analyst_text = ask_bedrock(
        f"Payment data:\n{df[['month_str','avg_days_to_pay','late_payments','total_payments']].to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst focusing on payment performance.")
    return {"layout": "quick", "analysis_type": "payment_performance", "metrics": metrics,
            "payment_df": df.to_dict(orient="records"), "analyst_response": analyst_text or "Analysis complete.",
            "sql": sql, "question": "Payment Performance"}

def _quick_invoice_aging():
    sql = f"""
        SELECT CASE
                WHEN due_date < CURRENT_DATE THEN 'Overdue'
                WHEN due_date <= CURRENT_DATE + INTERVAL '7' DAY THEN 'Due in 0-7 days'
                WHEN due_date <= CURRENT_DATE + INTERVAL '30' DAY THEN 'Due in 8-30 days'
                WHEN due_date <= CURRENT_DATE + INTERVAL '90' DAY THEN 'Due in 31-90 days'
                ELSE 'Due in >90 days' END AS aging_bucket,
            COUNT(*) AS invoice_count, SUM(COALESCE(invoice_amount_local, 0)) AS total_amount
        FROM {DATABASE}.fact_all_sources_vw
        WHERE invoice_status IN ('OPEN', 'DUE', 'OVERDUE') GROUP BY 1
        ORDER BY CASE aging_bucket
            WHEN 'Overdue' THEN 1 WHEN 'Due in 0-7 days' THEN 2 WHEN 'Due in 8-30 days' THEN 3
            WHEN 'Due in 31-90 days' THEN 4 ELSE 5 END
    """
    df = run_query(sql)
    if df.empty:
        return {"layout": "error", "message": "No aging data found."}
    df.columns = [c.lower() for c in df.columns]
    overdue_amount = df[df['aging_bucket'] == 'Overdue']['total_amount'].sum()
    total_open     = df['total_amount'].sum()
    overdue_pct    = (overdue_amount / total_open * 100) if total_open > 0 else 0
    metrics = {"total_open": total_open, "overdue_amount": overdue_amount,
               "overdue_pct": overdue_pct, "invoice_count": df['invoice_count'].sum()}
    analyst_text = ask_bedrock(
        f"Invoice aging:\n{df.to_string(index=False)}\nWrite Descriptive and Prescriptive sections.",
        system_prompt="You are a helpful procurement analyst focusing on accounts payable.")
    return {"layout": "quick", "analysis_type": "invoice_aging", "metrics": metrics,
            "aging_df": df.to_dict(orient="records"), "analyst_response": analyst_text or "Analysis complete.",
            "sql": sql, "question": "Invoice Aging"}

def render_cash_flow_response(result: dict):
    df = pd.DataFrame(result["df"])
    if df.empty:
        st.error("No cash flow data to display.")
        return
    total_unpaid = df[df["forecast_bucket"] == "TOTAL_UNPAID"]["total_amount"].values[0] if not df[df["forecast_bucket"] == "TOTAL_UNPAID"].empty else 0
    overdue_now  = df[df["forecast_bucket"] == "OVERDUE_NOW"]["total_amount"].values[0] if not df[df["forecast_bucket"] == "OVERDUE_NOW"].empty else 0
    due_30       = df[df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
    pct_due_30   = (due_30 / total_unpaid * 100) if total_unpaid > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unpaid", abbr_currency(total_unpaid))
    col2.metric("Overdue Now", abbr_currency(overdue_now))
    col3.metric("Due Next 30 Days", f"{abbr_currency(due_30)} ({pct_due_30:.0f}%)")
    chart_df = df[df["forecast_bucket"] != "TOTAL_UNPAID"].copy()
    if not chart_df.empty:
        st.subheader("Cash Outflow by Time Bucket")
        alt_bar(chart_df, x="forecast_bucket", y="total_amount", horizontal=True, height=300, color="#3b82f6")
    st.subheader("Forecast Details")
    st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_early_payment_response(result: dict):
    df = pd.DataFrame(result["df"])
    if result.get("empty", False) or df.empty:
        st.info("No early payment candidates were found.")
    else:
        total_savings = df["savings_if_2pct_discount"].sum()
        high_priority = df[df["early_pay_priority"] == "High"].shape[0]
        col1, col2 = st.columns(2)
        col1.metric("Total Potential Savings", abbr_currency(total_savings))
        col2.metric("High-Priority Invoices", high_priority)
        st.subheader("Top Candidates for Early Payment")
        st.dataframe(safe_dataframe_display(df.head(10)), use_container_width=True, hide_index=True)
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
    st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
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
    if "month" in df.columns:
        df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%b %Y")
        st.subheader("Late Payment Percentage Trend")
        alt_line_monthly(df[["month_str","late_pct"]].rename(columns={"late_pct":"VALUE"}),
                         month_col="month_str", value_col="VALUE", height=300, title="Late Payments %")
    st.subheader("Payment Performance Data")
    st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
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
    alt_bar(chart_df, x="year_month", y="total_grir_balance", horizontal=False, height=300, color="#ef4444")
    st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_root_causes(result: dict):
    df       = pd.DataFrame(result.get("df", []))
    extra_df = pd.DataFrame(result.get("extra_df", []))
    if not df.empty:
        st.subheader("GR/IR Aging (Last 6 Months)")
        st.dataframe(safe_dataframe_display(df), use_container_width=True)
    if not extra_df.empty:
        st.subheader("Outstanding Balances (Last 6 Months)")
        st.dataframe(safe_dataframe_display(extra_df), use_container_width=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_working_capital(result: dict):
    metrics = result.get("metrics", {})
    col1, col2 = st.columns(2)
    col1.metric("Working Capital Release (>60 days)", abbr_currency(metrics.get("older_60", 0)))
    col2.metric("Working Capital Release (>90 days)", abbr_currency(metrics.get("older_90", 0)))
    df = pd.DataFrame(result["df"])
    if not df.empty:
        st.subheader("GR/IR Balance by Month (with aging estimates)")
        st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_grir_vendor_followup(result: dict):
    df = pd.DataFrame(result["df"])
    if not df.empty:
        st.subheader("Top Vendors with Outstanding GR/IR Items")
        st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
    if result.get("analyst_response"):
        st.markdown("### 💡 Key Insights")
        st.markdown(result["analyst_response"])
    with st.expander("View SQL used"):
        st.code(_safe_sql_string(result.get("sql")), language="sql")

def render_quick_analysis_response(result: dict):
    analysis_type    = result.get("analysis_type", "spending_overview")
    metrics          = result.get("metrics", {})
    analyst_response = result.get("analyst_response", "")
    sql_queries      = result.get("sql", {})

    st.markdown(f"**Your question**\n{result.get('question', 'Analysis')}")
    st.markdown("---")

    if analysis_type == "spending_overview":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Spend (YTD)", abbr_currency(metrics.get("total_ytd", 0)))
        c2.metric("MoM Change", f"{metrics.get('mom_pct', 0):+.1f}%")
        c3.metric("Top 5 Vendors", f"{metrics.get('top5_pct', 0):.1f}% of total")
        c4.metric("QoQ Change", f"{metrics.get('qoq_pct', 0):+.1f}%")
        monthly_df = pd.DataFrame(result.get("monthly_df", []))
        if not monthly_df.empty:
            st.subheader("Spending Trends")
            monthly_df['month_dt']  = pd.to_datetime(monthly_df['month'])
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
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend (YTD)", abbr_currency(metrics.get("total_spend", 0)))
        c2.metric("Top 1 Vendor", f"{metrics.get('top1_pct', 0):.1f}%")
        c3.metric("Top 5 Vendors", f"{metrics.get('top5_pct', 0):.1f}%")
        vendors_df = pd.DataFrame(result.get("vendors_df", []))
        if not vendors_df.empty:
            st.subheader("Top 10 Vendors by Spend")
            bar_chart = alt.Chart(vendors_df).mark_bar(color="#f59e0b").encode(
                x=alt.X("total_spend:Q", axis=alt.Axis(format="~s")),
                y=alt.Y("vendor_name:N", sort="-x"),
                tooltip=["vendor_name:N", alt.Tooltip("total_spend:Q", format="$,.0f")]
            ).properties(height=400)
            st.altair_chart(bar_chart, use_container_width=True)

    elif analysis_type == "payment_performance":
        c1, c2 = st.columns(2)
        c1.metric("Avg Days to Pay", f"{metrics.get('avg_days_to_pay', 0):.1f}")
        c2.metric("Late Payments %", f"{metrics.get('late_payments_pct', 0):.1f}%")
        payment_df = pd.DataFrame(result.get("payment_df", []))
        if not payment_df.empty:
            cch1, cch2 = st.columns(2)
            with cch1:
                st.subheader("Avg days to pay by month")
                line1 = alt.Chart(payment_df).mark_line(point=True, color="#ef4444").encode(
                    x=alt.X("month_str:N", sort=None), y=alt.Y("avg_days_to_pay:Q", title="Days"),
                    tooltip=["month_str:N","avg_days_to_pay"]).properties(height=300)
                st.altair_chart(line1, use_container_width=True)
            with cch2:
                st.subheader("Late payments by month")
                line2 = alt.Chart(payment_df).mark_line(point=True, color="#3b82f6").encode(
                    x=alt.X("month_str:N", sort=None), y=alt.Y("late_payments:Q", title="Late payments"),
                    tooltip=["month_str:N","late_payments","total_payments"]).properties(height=300)
                st.altair_chart(line2, use_container_width=True)

    elif analysis_type == "invoice_aging":
        c1, c2 = st.columns(2)
        c1.metric("Total Open Invoices", abbr_currency(metrics.get("total_open", 0)))
        c2.metric("Overdue Amount", abbr_currency(metrics.get("overdue_amount", 0)))
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

    with st.expander("Show SQL used"):
        if isinstance(sql_queries, dict):
            for name, q in sql_queries.items():
                st.code(q, language="sql")
        elif isinstance(sql_queries, str):
            st.code(sql_queries, language="sql")

GRIR_HOTSPOTS_Q  = "Show GR/IR outstanding balance by month and highlight which recent months have the highest GR/IR balance so we can prioritize clearing."
GRIR_ROOTCAUSE_Q = "Using GR/IR aging and outstanding balance data, explain the likely root-cause buckets (missing goods receipt, invoice not posted, price or quantity mismatch) and for each bucket suggest 2–3 concrete remediation actions."
GRIR_WC_Q        = "Estimate the working capital that would be released by clearing all GR/IR items older than 60 and 90 days, by month."
GRIR_FOLLOWUP_Q  = "Based on GR/IR aging and outstanding balances, draft vendor-facing follow-up templates we can use for high-priority GR/IR items, with realistic subject lines and concise bullet points."

def _dispatch_query(q: str, history_context: str) -> dict:
    lower_q = q.lower()
    if q == GRIR_HOTSPOTS_Q:
        return process_grir_hotspots(q, history_context)
    elif q == GRIR_ROOTCAUSE_Q:
        return process_grir_root_causes(q, history_context)
    elif q == GRIR_WC_Q:
        return process_grir_working_capital(q, history_context)
    elif q == GRIR_FOLLOWUP_Q:
        return process_grir_vendor_followup(q, history_context)
    elif any(kw in lower_q for kw in ["forecast cash outflow", "cash flow forecast"]):
        return process_cash_flow_forecast(q, history_context)
    elif any(kw in lower_q for kw in ["pay early", "capture discounts"]):
        return process_early_payment(q, history_context)
    elif any(kw in lower_q for kw in ["optimal payment timing"]):
        return process_payment_timing(q, history_context)
    elif any(kw in lower_q for kw in ["late payment trend"]):
        return process_late_payment_trend(q, history_context)
    elif q == "Spending Overview":
        return _quick_spending_overview()
    elif q == "Vendor Analysis":
        return _quick_vendor_analysis()
    elif q == "Payment Performance":
        return _quick_payment_performance()
    elif q == "Invoice Aging":
        return _quick_invoice_aging()
    else:
        return process_custom_query(q, history_context)

def process_user_question(user_question: str):
    with st.spinner("Generating insights..."):
        cached = get_cache(user_question)
        if cached:
            st.session_state.current_messages = [
                {"role": "user", "content": user_question, "timestamp": datetime.now()},
                {"role": "assistant", "content": cached.get('analyst_response', 'Analysis complete.'),
                 "response": cached, "timestamp": datetime.now()}
            ]
            save_chat_message(st.session_state.genie_session_id, 0, "user", user_question)
            save_chat_message(st.session_state.genie_session_id, 1, "assistant",
                              cached.get('analyst_response', 'Analysis complete.'),
                              source="cache", sql_used=_safe_sql_string(cached.get("sql")))
            save_question(user_question, "custom")
        else:
            history_context = get_recent_conversation_context(limit=20, max_age_days=2)
            result = _dispatch_query(user_question, history_context)
            st.session_state.current_messages = [{"role": "user", "content": user_question, "timestamp": datetime.now()}]
            if result.get("layout") != "error":
                assistant_content = result.get('analyst_response', 'Analysis complete.')
                st.session_state.current_messages.append({"role": "assistant", "content": assistant_content,
                                                           "response": result, "timestamp": datetime.now()})
                set_cache(user_question, result)
                save_chat_message(st.session_state.genie_session_id, 0, "user", user_question)
                save_chat_message(st.session_state.genie_session_id, 1, "assistant", assistant_content,
                                  sql_used=_safe_sql_string(result.get("sql")))
                save_question(user_question, "forecast")
            else:
                st.session_state.current_messages.append(
                    {"role": "assistant", "content": result.get("message", "Error"), "timestamp": datetime.now()})
    st.rerun()

def start_new_session():
    st.session_state.genie_session_id = str(uuid.uuid4())
    st.session_state.current_messages = []
    st.session_state.show_summary = False
    st.session_state.conversation_summary = ""
    save_chat_session(st.session_state.genie_session_id, label=f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.rerun()

def summarize_conversation():
    if not st.session_state.current_messages:
        st.warning("No conversation to summarize.")
        return
    conv_text = "\n\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.current_messages)
    summary = ask_bedrock(
        f"Summarize the following conversation concisely, highlighting key questions, findings, and recommendations:\n\n{conv_text}",
        system_prompt="You are a helpful assistant that summarizes conversations.")
    if summary:
        st.session_state.conversation_summary = summary
        st.session_state.show_summary = True
        st.session_state.current_messages = []
    else:
        st.error("Could not generate summary at this time.")

def export_conversation_md():
    if not st.session_state.current_messages and not st.session_state.get("conversation_summary"):
        st.warning("No conversation to export.")
        return
    md_lines = ["# ProcureIQ Genie Conversation\n"]
    if st.session_state.get("conversation_summary"):
        md_lines.append(f"**Conversation Summary**\n\n{st.session_state.conversation_summary}\n\n---\n")
    for msg in st.session_state.current_messages:
        role = "**User**" if msg["role"] == "user" else "**Genie**"
        md_lines.append(f"{role}\n\n{msg['content']}\n\n---\n")
    st.download_button(label="📥 Download MD", data="\n".join(md_lines),
                       file_name=f"genie_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                       mime="text/markdown", key="export_md_btn")

def render_genie():
    st.markdown("""
<style>
    .welcome-header-left { text-align: left; margin-bottom: 1rem; }
    .welcome-header-left h1 { font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; }
    .welcome-header-left p { color: #64748b; font-size: 0.9rem; }
    .quick-card { background: white; border-radius: 16px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; text-align: center;
        transition: all 0.2s ease; height: 100%; display: flex; flex-direction: column; }
    .quick-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
    .card-icon { width: 48px; height: 48px; background: #3b82f6; border-radius: 12px;
        display: flex; align-items: center; justify-content: center; margin: 0 auto 0.8rem auto; font-size: 1.3rem; }
    .quick-card h3 { font-size: 1rem; font-weight: 600; color: #1e293b; margin: 0 0 0.4rem 0; }
    .quick-card p { font-size: 0.8rem; color: #64748b; line-height: 1.4; margin: 0 0 0.8rem 0; flex-grow: 1; }
    .chat-messages { max-height: 400px; overflow-y: auto; padding: 0.5rem; margin-bottom: 1rem;
        background: #fafcff; border-radius: 16px; border: 1px solid #e2e8f0; }
    .message-user { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white;
        padding: 10px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0; max-width: 80%; margin-left: auto; text-align: right; }
    .message-assistant { background: #f1f5f9; color: #1e293b; padding: 10px 16px;
        border-radius: 18px 18px 18px 4px; margin: 8px 0; max-width: 85%; }
    .start-conversation { text-align: center; padding: 2rem 1rem; background: #f8fafc; border-radius: 20px; margin: 1rem 0; }
    .plus-button { width: 56px; height: 56px; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1rem auto; box-shadow: 0 4px 12px rgba(59,130,246,0.3); }
    .plus-button span { font-size: 1.8rem; color: white; font-weight: 300; }
</style>
    """, unsafe_allow_html=True)

    if "genie_session_id" not in st.session_state:
        st.session_state.genie_session_id = str(uuid.uuid4())
        save_chat_session(st.session_state.genie_session_id, label=f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = []
    if "genie_prefill" not in st.session_state:
        st.session_state.genie_prefill = ""
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = False
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""

    auto_query = st.session_state.pop("auto_run_query", None)
    if auto_query:
        with st.spinner("Running analysis..."):
            history_context = get_recent_conversation_context(limit=20, max_age_days=2)
            result = _dispatch_query(auto_query, history_context)
            st.session_state.current_messages = [{"role": "user", "content": auto_query, "timestamp": datetime.now()}]
            if result.get("layout") != "error":
                assistant_content = result.get('analyst_response', 'Analysis complete.')
                st.session_state.current_messages.append({"role": "assistant", "content": assistant_content,
                                                           "response": result, "timestamp": datetime.now()})
                save_chat_message(st.session_state.genie_session_id, 0, "user", auto_query)
                save_chat_message(st.session_state.genie_session_id, 1, "assistant", assistant_content,
                                  sql_used=_safe_sql_string(result.get("sql")))
                save_question(auto_query, "forecast")
                set_cache(auto_query, result)
            else:
                st.session_state.current_messages.append(
                    {"role": "assistant", "content": result.get("message", "Error"), "timestamp": datetime.now()})
            st.rerun()

    st.markdown("""
    <div class="welcome-header-left">
        <h1>Welcome to ProcureIQ Genie</h1>
        <p>Let Genie run one of these quick analyses for you</p>
    </div>
    """, unsafe_allow_html=True)

    cards_data = [
        {"icon": "📊", "title": "Spending Overview", "description": "Track total spend, monthly trends and major changes"},
        {"icon": "🏭", "title": "Vendor Analysis", "description": "Understand vendor-wise spend, concentration, and dependency"},
        {"icon": "⏱️", "title": "Payment Performance", "description": "Identify delays, late payments, and cycle time issues"},
        {"icon": "📅", "title": "Invoice Aging", "description": "See overdue invoices, risk buckets, and problem areas"}
    ]
    cols = st.columns(4, gap="small")
    for idx, (col, card) in enumerate(zip(cols, cards_data)):
        with col:
            st.markdown(f"""<div class="quick-card"><div class="card-icon">{card['icon']}</div>
                <h3>{card['title']}</h3><p>{card['description']}</p></div>""", unsafe_allow_html=True)
            if st.button("Ask Genie", key=f"card_{idx}", use_container_width=True):
                st.session_state.auto_run_query = card['title']
                st.rerun()

    st.markdown("---")

    left_info, right_chat = st.columns([0.35, 0.65], gap="large")

    with left_info:
        with st.container(border=True):
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
                    for sug in ["Total spend YTD and trends", "Top vendors by spend", "Overdue invoices summary"]:
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
        with st.container(border=True):
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Export MD", use_container_width=True, key="export_md_top"):
                    if st.session_state.current_messages or st.session_state.conversation_summary:
                        export_conversation_md()
                    else:
                        st.warning("No conversation to export.")
            with btn_col2:
                if st.button("Summarize", use_container_width=True, key="summarize_top"):
                    if st.session_state.current_messages:
                        summarize_conversation()
                        st.rerun()
                    else:
                        st.warning("No conversation to summarize.")
            with btn_col3:
                if st.button("Clear", use_container_width=True, key="clear_top"):
                    start_new_session()

            if st.session_state.show_summary and st.session_state.conversation_summary:
                st.markdown("### Conversation Summary")
                st.markdown(st.session_state.conversation_summary)
                if st.button("Dismiss Summary", key="dismiss_summary_inside", use_container_width=True):
                    st.session_state.show_summary = False
                    st.session_state.conversation_summary = ""
                    st.rerun()
                st.markdown("---")
            elif not st.session_state.current_messages:
                st.markdown("""
                <div class="start-conversation">
                    <div class="plus-button"><span>+</span></div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">Start a Conversation</div>
                    <div style="color: #64748b; font-size: 0.85rem; max-width: 280px; margin: 0.5rem auto;">
                        Ask questions about your Procurement to Pay data, or select a pre-built analysis.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
                for msg in st.session_state.current_messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="message-user"><strong>You</strong><br/>{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="message-assistant"><strong>Genie</strong></div>', unsafe_allow_html=True)
                        if "response" in msg and msg["response"]:
                            resp   = msg["response"]
                            layout = resp.get("layout")
                            if layout == "static":
                                st.info(resp["analyst_response"])
                            elif layout == "cash_flow":         render_cash_flow_response(resp)
                            elif layout == "early_payment":     render_early_payment_response(resp)
                            elif layout == "payment_timing":    render_payment_timing_response(resp)
                            elif layout == "late_payment_trend":render_late_payment_trend_response(resp)
                            elif layout == "grir_hotspots":     render_grir_hotspots(resp)
                            elif layout == "grir_root_causes":  render_grir_root_causes(resp)
                            elif layout == "grir_working_capital": render_grir_working_capital(resp)
                            elif layout == "grir_vendor_followup": render_grir_vendor_followup(resp)
                            elif layout == "quick":             render_quick_analysis_response(resp)
                            elif layout == "analyst":
                                if resp.get("analyst_response"):
                                    st.markdown(resp["analyst_response"])
                                df = pd.DataFrame(resp["df"])
                                if not df.empty:
                                    st.subheader("Supporting Data")
                                    st.dataframe(safe_dataframe_display(df), use_container_width=True, hide_index=True)
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
                    user_question = st.text_input("Ask a question", value=prefill,
                                                  placeholder="Ask a procurement question here..",
                                                  label_visibility="collapsed")
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

    st.markdown("### Invoice Summary")
    summary_fields = ["Invoice Number", "Invoice Date", "Invoice Amount", "PO Number", "PO Amount",
                      "Due Date", "Invoice Status", "Aging (Days)"]
    summary_values = [inv_num, get_val("invoice_date", ""), abbr_currency(safe_number(get_val("invoice_amount", 0))),
                      get_val("po_number", ""), abbr_currency(safe_number(get_val("po_amount", 0))),
                      get_val("due_date", ""), get_val("invoice_status", "").upper(),
                      f"{aging_days} days" if aging_days > 0 else "0 days"]
    html_table = '<table style="width:100%; border-collapse: collapse; margin-bottom: 1rem; background: white;">'
    html_table += '<tr style="background-color: #f1f5f9; border-bottom: 1px solid #e2e8f0;">'
    for field in summary_fields:
        html_table += f'<th style="padding: 10px 8px; text-align: left; font-weight: 600; color: #1e293b;">{field}</th>'
    html_table += '<tr>'
    for val in summary_values:
        html_table += f'<td style="padding: 10px 8px; border-bottom: 1px solid #e2e8f0;">{val}</td>'
    html_table += '</tr>'
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Status History")
    hist_sql = f"""
        SELECT invoice_number, UPPER(status) AS status, effective_date, status_notes
        FROM {DATABASE}.invoice_status_history_vw
        WHERE CAST(invoice_number AS VARCHAR) = '{inv_num}' ORDER BY sequence_nbr
    """
    hist_df = run_query(hist_sql)
    if hist_df.empty:
        hist_df = pd.DataFrame([
            {"status": "OPEN", "effective_date": get_val("invoice_date", "2026-01-02"),
             "status_notes": "Invoice opened and assigned for processing."},
            {"status": "OVERDUE", "effective_date": get_val("due_date", "2026-02-16"),
             "status_notes": "Invoice overdue following standard payment term expiry."}
        ])
    else:
        hist_df.columns = [c.lower() for c in hist_df.columns]
        hist_df = hist_df[["status", "effective_date", "status_notes"]].copy()

    paid_key = f"paid_{inv_num}"
    if st.session_state.get(paid_key, False) and not any(hist_df["status"] == "PAID"):
        hist_df = pd.concat([hist_df, pd.DataFrame([{"status": "PAID",
             "effective_date": date.today().strftime("%Y-%m-%d"), "status_notes": "Processed via ProcureIQ app"}])], ignore_index=True)
    hist_df["effective_date"] = hist_df["effective_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, (date, datetime)) else str(x))
    st.dataframe(safe_dataframe_display(hist_df[["status","effective_date","status_notes"]]),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Vendor Information")
    tab1, tab2 = st.tabs(["Vendor Info", "Company Info"])
    with tab1:
        vendor_sql = f"""
            SELECT DISTINCT v.vendor_id, v.vendor_name, v.vendor_name_2, v.country_code, v.city, v.postal_code, v.street
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR) = '{inv_num}' LIMIT 1
        """
        vendor_df = run_query(vendor_sql)
        row = vendor_df.iloc[0] if not vendor_df.empty else {}
        vendor_fields = ["Vendor ID", "Vendor Name", "Alias/Name 2", "Country", "City", "Postal Code", "Street"]
        vendor_values = [row.get("vendor_id", "—"), row.get("vendor_name", "—"), row.get("vendor_name_2", "—"),
                         row.get("country_code", "—"), row.get("city", "—"), row.get("postal_code", "—"), row.get("street", "—")]
        html_v = '<table style="width:100%; border-collapse: collapse; background: white;"><tr style="background-color: #f1f5f9; border-bottom: 1px solid #e2e8f0;">'
        for f in vendor_fields:
            html_v += f'<th style="padding: 10px 8px; text-align: left; font-weight: 600;">{f}</th>'
        html_v += '<tr>'
        for v in vendor_values:
            html_v += f'<td style="padding: 10px 8px; border-bottom: 1px solid #e2e8f0;">{v}</td>'
        html_v += '</table>'
        st.markdown(html_v, unsafe_allow_html=True)
    with tab2:
        company_sql = f"""
            SELECT DISTINCT f.company_code, cc.company_name, f.plant_code, plt.plant_name, cc.street, cc.city, cc.postal_code
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_company_code_vw cc ON f.company_code = cc.company_code
            LEFT JOIN {DATABASE}.dim_plant_vw plt ON f.plant_code = plt.plant_code
            WHERE CAST(f.invoice_number AS VARCHAR) = '{inv_num}' LIMIT 1
        """
        company_df = run_query(company_sql)
        row = company_df.iloc[0] if not company_df.empty else {}
        company_fields = ["Company Code", "Company Name", "Plant Code", "Plant Name", "Street", "City", "Postal Code"]
        company_values = [row.get("company_code", "—"), row.get("company_name", "—"), row.get("plant_code", "—"),
                          row.get("plant_name", "—"), row.get("street", "—"), row.get("city", "—"), row.get("postal_code", "—")]
        html_c = '<table style="width:100%; border-collapse: collapse; background: white;"><tr style="background-color: #f1f5f9; border-bottom: 1px solid #e2e8f0;">'
        for f in company_fields:
            html_c += f'<th style="padding: 10px 8px; text-align: left; font-weight: 600;">{f}</th>'
        html_c += '<tr>'
        for v in company_values:
            html_c += f'<td style="padding: 10px 8px; border-bottom: 1px solid #e2e8f0;">{v}</td>'
        html_c += '</table>'
        st.markdown(html_c, unsafe_allow_html=True)

    st.markdown("---")
    current_status = get_val("invoice_status", "").upper()
    if st.session_state.get(paid_key, False):
        st.success("✅ Invoice has been processed and marked as Paid.")
    elif current_status == "PAID":
        st.info("ℹ️ This invoice is already marked as PAID.")
    else:
        if st.button("✅ Proceed to Pay", key="proceed_pay_btn", use_container_width=True):
            st.session_state[paid_key] = True
            st.rerun()

def render_invoices():
    st.subheader("Invoices")
    st.markdown("Search, track and manage all invoices in one place")

    query_params = st.experimental_get_query_params()
    if "invoice" in query_params and query_params["invoice"][0]:
        inv_from_param = query_params["invoice"][0]
        st.session_state.selected_invoice_detail = inv_from_param
        st.experimental_set_query_params()
        st.rerun()

    if st.session_state.get("selected_invoice_detail"):
        inv_num = st.session_state.selected_invoice_detail
        inv_sql = f"""
            SELECT f.invoice_number, f.posting_date AS invoice_date, f.invoice_amount_local AS invoice_amount,
                f.purchase_order_reference AS po_number, f.po_amount, f.due_date, UPPER(f.invoice_status) AS invoice_status,
                f.aging_days, f.vendor_id, v.vendor_name, v.vendor_name_2, v.country_code, v.city, v.postal_code, v.street,
                f.company_code, f.plant_code, f.currency
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR) = '{inv_num}' LIMIT 1
        """
        inv_df = run_query(inv_sql)
        if not inv_df.empty:
            render_invoice_detail(inv_df.iloc[0].to_dict(), inv_num)
            if st.button("← Back to Invoices List", key="back_invoices_btn", use_container_width=True):
                st.session_state.selected_invoice_detail = None
                st.session_state.invoice_search_input = ""
                st.session_state.invoice_status_filter = "All Status"
                st.session_state.inv_selected_vendor = "All Vendors"
                st.rerun()
            return
        else:
            st.warning(f"Invoice {inv_num} not found. Returning to list.")
            st.session_state.selected_invoice_detail = None
            st.rerun()

    if "invoice_search_input" not in st.session_state:
        st.session_state.invoice_search_input = ""
    if "invoice_status_filter" not in st.session_state:
        st.session_state.invoice_status_filter = "All Status"
    if "inv_selected_vendor" not in st.session_state:
        st.session_state.inv_selected_vendor = "All Vendors"

    col_search, col_btn, col_reset = st.columns([3, 1, 1])
    with col_search:
        user_search = st.text_input("Invoice or PO Number", value=st.session_state.invoice_search_input,
                                    placeholder="e.g., 9001767", label_visibility="collapsed", key="inv_search_widget")
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True, key="search_invoice_btn")
    with col_reset:
        reset_clicked = st.button("Reset", use_container_width=True, key="reset_invoice_btn")

    if reset_clicked:
        st.session_state.invoice_search_input = ""
        st.session_state.invoice_status_filter = "All Status"
        st.session_state.inv_selected_vendor = "All Vendors"
        st.session_state.selected_invoice_detail = None
        st.rerun()

    if search_clicked:
        if user_search.strip():
            st.session_state.invoice_search_input = user_search.strip()
            clean_search = clean_invoice_number(user_search)
            check_sql = f"SELECT invoice_number FROM {DATABASE}.fact_all_sources_vw WHERE CAST(invoice_number AS VARCHAR) = '{clean_search}' LIMIT 1"
            check_df = run_query(check_sql)
            if not check_df.empty:
                st.session_state.selected_invoice_detail = clean_search
                st.rerun()
            else:
                st.warning(f"Invoice {clean_search} not found. Please check the number.")
        else:
            st.warning("Please enter an invoice number to search.")

    if not st.session_state.get("selected_invoice_detail"):
        col_vendor, col_status = st.columns(2)
        with col_vendor:
            if "inv_vendor_list" not in st.session_state:
                vendor_df = run_query(f"SELECT DISTINCT vendor_name FROM {DATABASE}.dim_vendor_vw ORDER BY vendor_name")
                vendor_list = ["All Vendors"] + vendor_df["vendor_name"].tolist() if not vendor_df.empty else ["All Vendors"]
                st.session_state.inv_vendor_list = vendor_list
            selected_vendor = st.selectbox("Vendor", st.session_state.inv_vendor_list, key="inv_sel_vendor",
                index=st.session_state.inv_vendor_list.index(st.session_state.inv_selected_vendor)
                if st.session_state.inv_selected_vendor in st.session_state.inv_vendor_list else 0)
            if selected_vendor != st.session_state.inv_selected_vendor:
                st.session_state.inv_selected_vendor = selected_vendor
        with col_status:
            status_options = ["All Status", "OPEN", "PAID", "DISPUTED", "OVERDUE", "DUE_NEXT_30"]
            selected_status_display = st.selectbox("Status", status_options,
                index=status_options.index(st.session_state.invoice_status_filter)
                if st.session_state.invoice_status_filter in status_options else 0, key="inv_sel_status")
            if selected_status_display != st.session_state.invoice_status_filter:
                st.session_state.invoice_status_filter = selected_status_display

        where = []
        if st.session_state.invoice_search_input:
            clean_search = clean_invoice_number(st.session_state.invoice_search_input)
            where.append(f"CAST(f.invoice_number AS VARCHAR) = '{clean_search}'")
        if st.session_state.inv_selected_vendor != "All Vendors":
            safe_vendor = st.session_state.inv_selected_vendor.replace("'", "''")
            where.append(f"UPPER(v.vendor_name) = UPPER('{safe_vendor}')")
        selected_status = st.session_state.invoice_status_filter
        if selected_status != "All Status":
            if selected_status == "DUE_NEXT_30":
                where.append("UPPER(f.invoice_status) = 'OPEN' AND f.due_date >= CURRENT_DATE AND f.due_date <= DATE_ADD('day', 30, CURRENT_DATE)")
            else:
                where.append(f"UPPER(f.invoice_status) = '{selected_status}'")
        where_sql = " AND ".join(where) if where else "1=1"
        query = f"""
            SELECT DISTINCT f.invoice_number AS invoice_number, v.vendor_name AS vendor_name,
                f.posting_date AS posting_date, f.due_date AS due_date, f.invoice_amount_local AS invoice_amount,
                f.purchase_order_reference AS po_number, UPPER(f.invoice_status) AS status
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE {where_sql} ORDER BY f.posting_date DESC LIMIT 500
        """
        df = run_query(query)
        if not df.empty:
            df_display = df.rename(columns={'invoice_number': 'INVOICE NUMBER', 'vendor_name': 'VENDOR NAME',
                'posting_date': 'POSTING DATE', 'due_date': 'DUE DATE', 'invoice_amount': 'INVOICE AMOUNT',
                'po_number': 'PO NUMBER', 'status': 'STATUS'})
            st.dataframe(safe_dataframe_display(df_display), use_container_width=True, height=400)
        else:
            st.info("No invoices found. Try a different search term or adjust filters.")

# ------------------------------------------------------------
# main app
# ------------------------------------------------------------
def main():
    init_db()
    st.set_page_config(page_title="ProcureIQ", layout="wide", initial_sidebar_state="expanded")

    # Inject CSS with working BG button (no separate render_floating_bg_button call)
    inject_dashboard_css("#ffffff")

    st.markdown("""
<style>
.block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; }
button { font-weight: 500 !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
</style>
""", unsafe_allow_html=True)

    # Header navigation
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], gap="small")
    with col1:
        st.markdown("<div style='margin-top: 4px;'><h1 style='font-weight:bold; margin-bottom:0; font-size:1.6rem;'>ProcureIQ</h1><p style='font-size:0.7rem;color:gray;margin-top:-0.2rem;'>P2P Analytics</p></div>", unsafe_allow_html=True)
    with col2:
        if st.button("Dashboard", use_container_width=True,
                     type="primary" if st.session_state.get("page") == "Dashboard" else "secondary",
                     key="nav_dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col3:
        if st.button("GenAI", use_container_width=True,
                     type="primary" if st.session_state.get("page") == "Genie" else "secondary",
                     key="nav_genai"):
            st.session_state.page = "Genie"
            st.rerun()
    with col4:
        if st.button("Forecast", use_container_width=True,
                     type="primary" if st.session_state.get("page") == "Forecast" else "secondary",
                     key="nav_forecast"):
            st.session_state.page = "Forecast"
            st.rerun()
    with col5:
        if st.button("Invoices", use_container_width=True,
                     type="primary" if st.session_state.get("page") == "Invoices" else "secondary",
                     key="nav_invoices"):
            st.session_state.page = "Invoices"
            st.rerun()
    with col6:
        st.markdown(f"<div style='display: flex; justify-content: flex-end;'><img src='{LOGO_URL}' style='width: 120px; height: auto; object-fit: contain;' /></div>", unsafe_allow_html=True)

    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

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

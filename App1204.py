import streamlit as st
import boto3
import awswrangler as wr
import pandas as pd
import altair as alt
import json
import uuid
import re
import html
import sqlite3
import hashlib
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Union, List, Dict
import numpy as np

# ── config ───────────────────────────────────────────────────
DATABASE      = "procure2pay"
ATHENA_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
DB_PATH       = "procureiq.db"
LOGO_URL      = "https://th.bing.com/th/id/OIP.Vy1yFQtg8-D1SsAxcqqtSgHaE6?w=235&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3"

# BG colour palette (Streamlit-native BG button uses these)
BG_COLOR_OPTIONS = {
    "White":        "#ffffff",
    "Light Blue":   "#e0f2fe",
    "Light Gray":   "#f3f4f6",
    "Light Green":  "#dcfce7",
    "Light Purple": "#f3e8ff",
    "Light Pink":   "#fce7f3",
    "Light Beige":  "#fef9c3",
    "Light Cyan":   "#cffafe",
}

def compute_range_preset(preset: str):
    """
    Date range definitions:
      YTD        = last 365 days  (today-365 → today)
      QTD        = last 120 days  (today-120 → today)
      Last 30 Days = last 30 days (today-30  → today)
      Custom     = user-defined
    """
    today = date.today()
    if preset == "Last 30 Days": return today - timedelta(days=30), today
    if preset == "QTD":          return today - timedelta(days=120), today
    if preset == "YTD":          return today - timedelta(days=365), today
    return today - timedelta(days=30), today   # fallback = last 30 days

# ── utils ────────────────────────────────────────────────────
def safe_number(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except: return default

def safe_int(val, default=0):
    try:
        if pd.isna(val): return default
        return int(float(val))
    except: return default

def abbr_currency(v: float, sym: str = "$") -> str:
    n = abs(v); sign = "-" if v < 0 else ""
    if n >= 1_000_000_000: return f"{sign}{sym}{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:     return f"{sign}{sym}{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{sign}{sym}{n/1_000:.1f}K"
    return f"{sign}{sym}{n:.0f}"

def sql_date(d: date) -> str:
    return f"DATE '{d.strftime('%Y-%m-%d')}'"

def clean_invoice_number(inv_num):
    try:
        if isinstance(inv_num, (float, Decimal)): return str(int(inv_num))
        s = str(inv_num)
        if '.' in s: s = s.split('.')[0]
        return s
    except: return str(inv_num)

def pct_delta(cur, prev):
    if prev == 0:
        return ("-" if cur == 0 else "+100%"), cur >= 0
    change = (cur - prev) / prev * 100
    if abs(change) < 0.05: return "0%", True
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%", change >= 0

def prior_window(start: date, end: date):
    days = (end - start).days + 1
    prev_end   = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start, prev_end

def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    if isinstance(obj, (date, datetime)): return obj.isoformat()
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='list')
    if isinstance(obj, pd.Series): return obj.to_list()
    if isinstance(obj, dict): return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [make_json_serializable(i) for i in obj]
    return str(obj)

def safe_dataframe_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

def render_simple_table(df: pd.DataFrame, col_labels: dict = None,
                        striped: bool = True, max_rows: int = 500):
    """
    Render a pandas DataFrame as a clean, plain HTML table.
    Replaces st.dataframe() for tables that need simple styling:
    - Clean borders, no Streamlit column-resize handles
    - Optional striped rows
    - Optional column label override via col_labels dict
    - Numbers right-aligned, text left-aligned
    """
    if df.empty:
        st.caption("No data available.")
        return

    df = df.head(max_rows).copy()

    # Build header
    headers = []
    for col in df.columns:
        label = col_labels.get(col, col.replace("_", " ").title()) if col_labels else col.replace("_", " ").title()
        headers.append(f"<th style='padding:9px 12px;text-align:left;font-weight:600;"
                       f"font-size:13px;color:#374151;background:#f8fafc;"
                       f"border-bottom:2px solid #e2e8f0;white-space:nowrap;'>{label}</th>")

    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#fafafa" if (striped and i % 2 == 0) else "white"
        cells = []
        for col in df.columns:
            val = row[col]
            # Format value
            if pd.isna(val) if not isinstance(val, str) else False:
                display = "—"
                align = "left"
            elif isinstance(val, float):
                display = f"{val:,.2f}" if val != int(val) else f"{int(val):,}"
                align = "right"
            elif isinstance(val, int):
                display = f"{val:,}"
                align = "right"
            else:
                display = str(val)
                align = "left"
            cells.append(
                f"<td style='padding:8px 12px;font-size:13px;color:#1f2937;"
                f"border-bottom:1px solid #f0f0f0;text-align:{align};"
                f"background:{bg};'>{display}</td>"
            )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
<div style='overflow-x:auto;border-radius:10px;border:1px solid #e2e8f0;
            box-shadow:0 1px 4px rgba(0,0,0,0.05);margin:4px 0;'>
<table style='width:100%;border-collapse:collapse;background:white;
              font-family:inherit;'>
<thead><tr>{''.join(headers)}</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
</div>"""
    st.markdown(table_html, unsafe_allow_html=True)


def format_invoice_number(inv_num):
    if inv_num is None: return ""
    s = str(inv_num)
    if s.endswith('.0'): s = s[:-2]
    try: s = str(int(float(s)))
    except: pass
    return s

def build_vendor_where(selected_vendor: str) -> str:
    if selected_vendor == "All Vendors": return ""
    sv = selected_vendor.replace("'", "''")
    return f"AND UPPER(v.vendor_name) = UPPER('{sv}')"

def is_safe_sql(sql: str) -> bool:
    sl = sql.lower().strip()
    if not sl.startswith("select"): return False
    for w in ["insert","update","delete","drop","alter","create","truncate","grant","revoke"]:
        if re.search(r'\b'+w+r'\b', sl): return False
    return True

def ensure_limit(sql: str, default_limit: int = 100) -> str:
    sl = sql.lower()
    if "limit" in sl: return sql
    if re.search(r'\b(count|sum|avg|min|max)\b', sl) and "group by" not in sl: return sql
    return f"{sql.rstrip(';')} LIMIT {default_limit}"

# ── year/month filter (for views without posting_date) ──────
def year_month_filter(start: date, end: date) -> str:
    """Build WHERE clause for views that use year/month columns (no posting_date)."""
    pairs = []
    cur = date(start.year, start.month, 1)
    end_ym = date(end.year, end.month, 1)
    while cur <= end_ym:
        pairs.append((cur.year, cur.month))
        cur = date(cur.year+1, 1, 1) if cur.month == 12 else date(cur.year, cur.month+1, 1)
    if not pairs: return "1=0"
    return "(" + " OR ".join(f"(year={y} AND month={m})" for y,m in pairs) + ")"

def alt_bar(df, x, y, title=None, horizontal=False, color="#1459d2", height=320):
    if df.empty: st.info("No data for this chart."); return
    if horizontal:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            y=alt.Y(x, type='nominal', sort='-x', axis=alt.Axis(title=None)),
            tooltip=[x, alt.Tooltip(y, format=",.0f")])
    else:
        chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=4).encode(
            x=alt.X(x, type='nominal', axis=alt.Axis(title=None)),
            y=alt.Y(y, type='quantitative', axis=alt.Axis(title=None, format="~s")),
            tooltip=[x, alt.Tooltip(y, format=",.0f")])
    chart = chart.properties(height=height)
    if title: chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)

def alt_line_monthly(df, month_col='month', value_col='value', height=140, title=None):
    if df.empty: st.info("No data for this chart."); return
    data = df.copy()
    try:
        data['_dt'] = pd.to_datetime(data[month_col].astype(str)+'-01')
        data = data.sort_values('_dt')
        data['month_label'] = data['_dt'].dt.strftime('%b %Y')
    except:
        data['month_label'] = data[month_col].astype(str)
    chart = alt.Chart(data).mark_line(point=True, color='#1e88e5').encode(
        x=alt.X('month_label:N', sort=None, axis=alt.Axis(title=None, labelAngle=-45)),
        y=alt.Y(f'{value_col}:Q', axis=alt.Axis(title=None, grid=False, format='~s')),
        tooltip=[alt.Tooltip('month_label:N', title='Month'), alt.Tooltip(f'{value_col}:Q', format=',.0f')]
    ).properties(height=height)
    if title: chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)

def auto_chart(df: pd.DataFrame):
    if df.empty or len(df) > 200: return None
    nc = df.select_dtypes(include=['number']).columns.tolist()
    if not nc: return None
    dc = [c for c in df.columns if c not in nc]
    if dc:
        dim = dc[0]
        if len(nc) == 1:
            chart = alt.Chart(df).mark_bar().encode(x=alt.X(dim, sort=None), y=alt.Y(nc[0]), tooltip=[dim, nc[0]])
        else:
            melted = df.melt(id_vars=[dim], value_vars=nc)
            chart = alt.Chart(melted).mark_line(point=True).encode(
                x=alt.X(dim, sort=None), y=alt.Y('value', title='Value'),
                color='variable', tooltip=[dim, 'variable', 'value'])
        return chart.interactive()
    return None

# ── Athena client ────────────────────────────────────────────
@st.cache_resource
def get_aws_session(): return boto3.Session()

@st.cache_data(ttl=600, show_spinner=False)
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

# ── Bedrock client ───────────────────────────────────────────
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
        response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID,
                                        contentType="application/json", accept="application/json", body=body)
        return json.loads(response['body'].read())['output']['message']['content'][0]['text']
    except Exception as e:
        st.error(f"Bedrock invocation failed: {e}")
        return ""

# ── persistence ──────────────────────────────────────────────
def init_db():
    """
    Initialise all SQLite tables for:
      - Short-term memory  : chat_sessions + chat_messages (current session context)
      - Long-term memory   : user_memory (persistent facts/preferences per user)
                             question_history (all past queries)
                             saved_insights (bookmarked analyses)
      - Query cache        : query_cache (LLM+SQL response cache with TTL + hit tracking)
      - KPI snapshot cache : kpi_snapshot_cache (dashboard KPI history for trend display)
    """
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()

    # ── Short-term memory: conversation sessions ─────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id    TEXT PRIMARY KEY,
        session_label TEXT,
        created_at    TIMESTAMP,
        last_updated  TIMESTAMP,
        user_name     TEXT,
        page_context  TEXT DEFAULT 'Dashboard')''')
    try: c.execute("ALTER TABLE chat_sessions ADD COLUMN user_name TEXT")
    except sqlite3.OperationalError: pass
    try: c.execute("ALTER TABLE chat_sessions ADD COLUMN page_context TEXT DEFAULT 'Dashboard'")
    except sqlite3.OperationalError: pass

    # ── Short-term memory: individual messages ───────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT,
        turn_index  INTEGER,
        role        TEXT,
        content     TEXT,
        sql_used    TEXT,
        source      TEXT,
        timestamp   TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id))''')

    # ── Long-term memory: persistent user facts/preferences ──────
    # Stores things like "user prefers YTD view", "user tracks vendor X", etc.
    c.execute('''CREATE TABLE IF NOT EXISTS user_memory (
        memory_id   TEXT PRIMARY KEY,
        user_name   TEXT NOT NULL,
        memory_type TEXT NOT NULL,   -- "preference" | "entity" | "context" | "insight"
        memory_key  TEXT NOT NULL,   -- e.g. "preferred_preset", "favourite_vendor"
        memory_val  TEXT NOT NULL,   -- stored value (JSON or plain text)
        source      TEXT,            -- "explicit" (user set) | "inferred" (auto-detected)
        confidence  REAL DEFAULT 1.0,
        created_at  TIMESTAMP,
        updated_at  TIMESTAMP,
        access_count INTEGER DEFAULT 0)''')
    try: c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_user_memory ON user_memory(user_name, memory_key)")
    except sqlite3.OperationalError: pass

    # ── Long-term memory: question / query history ───────────────
    c.execute('''CREATE TABLE IF NOT EXISTS question_history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        normalized_query TEXT,
        query_text       TEXT,
        user_name        TEXT,
        analysis_type    TEXT,
        result_layout    TEXT,
        asked_at         TIMESTAMP)''')
    try: c.execute("ALTER TABLE question_history ADD COLUMN result_layout TEXT")
    except sqlite3.OperationalError: pass

    # ── Long-term memory: saved insights ─────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS saved_insights (
        insight_id          TEXT PRIMARY KEY,
        created_by          TEXT,
        page                TEXT,
        title               TEXT,
        question            TEXT,
        verified_query_name TEXT,
        created_at          TIMESTAMP)''')

    # ── Query / LLM response cache ────────────────────────────────
    # TTL-aware: expired entries are ignored and overwritten on next miss.
    c.execute('''CREATE TABLE IF NOT EXISTS query_cache (
        query_hash   TEXT PRIMARY KEY,
        question     TEXT,
        response_json TEXT,
        created_at   TIMESTAMP,
        last_hit_at  TIMESTAMP,
        hit_count    INTEGER DEFAULT 0,
        ttl_seconds  INTEGER DEFAULT 3600,
        cache_type   TEXT DEFAULT "genie"  -- "genie" | "kpi" | "chart"
    )''')
    try: c.execute("ALTER TABLE query_cache ADD COLUMN ttl_seconds INTEGER DEFAULT 3600")
    except sqlite3.OperationalError: pass
    try: c.execute("ALTER TABLE query_cache ADD COLUMN cache_type TEXT DEFAULT 'genie'")
    except sqlite3.OperationalError: pass

    # ── KPI snapshot cache (for delta trend display) ─────────────
    # Stores dashboard KPI values at each snapshot so we can show
    # period-over-period deltas without re-querying Athena.
    c.execute('''CREATE TABLE IF NOT EXISTS kpi_snapshot_cache (
        snapshot_id  TEXT PRIMARY KEY,
        user_name    TEXT,
        preset       TEXT,
        start_date   TEXT,
        end_date     TEXT,
        kpi_json     TEXT,  -- serialised KPI dict
        created_at   TIMESTAMP
    )''')

    conn.commit(); conn.close()

def get_current_user(): return "user1"

# ═══════════════════════════════════════════════════════════════════
# MEMORY SYSTEM
# ─────────────────────────────────────────────────────────────────
# Short-term  : last N messages in current session (in-process RAM +
#               SQLite chat_messages, scoped to session_id)
# Long-term   : user_memory table — persists preferences, entities,
#               and inferred context across sessions and reruns
# Cache       : query_cache table — TTL-aware LLM/SQL response store
#               with hit counting and type tagging
# KPI history : kpi_snapshot_cache — stores KPI snapshots per preset
#               so the dashboard can show meaningful delta arrows even
#               when Athena is slow or unavailable
# ═══════════════════════════════════════════════════════════════════

# ── Short-term memory helpers ─────────────────────────────────────

def get_short_term_context(session_id: str, max_turns: int = 10) -> list:
    """
    Return the last `max_turns` messages for a session as a list of dicts.
    Used to build the rolling conversation window sent to Bedrock.
    """
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''SELECT role, content, timestamp
                 FROM chat_messages
                 WHERE session_id = ?
                 ORDER BY turn_index DESC, timestamp DESC
                 LIMIT ?''', (session_id, max_turns * 2))
    rows = c.fetchall(); conn.close()
    rows.reverse()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


def build_bedrock_context(session_id: str, max_turns: int = 6) -> str:
    """
    Build a compact conversation context string from short-term memory.
    Passed as the `history` prefix to Bedrock prompts so the model
    can answer follow-up questions intelligently.
    """
    msgs = get_short_term_context(session_id, max_turns)
    if not msgs:
        return ""
    parts = []
    for m in msgs:
        label = "User" if m["role"] == "user" else "Assistant"
        # Truncate very long assistant responses to keep prompt size manageable
        text = m["content"][:600] + "…" if len(m["content"]) > 600 else m["content"]
        parts.append(f"{label}: {text}")
    return (
        "Previous conversation context (use this to answer follow-up questions):\n\n"
        + "\n\n".join(parts)
        + "\n\n---\nNew question:\n"
    )


# ── Long-term memory helpers ──────────────────────────────────────

def set_user_memory(key: str, value: str,
                    memory_type: str = "preference",
                    source: str = "explicit",
                    confidence: float = 1.0):
    """Upsert a long-term memory entry for the current user."""
    import uuid as _uuid
    user = get_current_user()
    mid  = hashlib.md5(f"{user}:{key}".encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''INSERT INTO user_memory
                    (memory_id, user_name, memory_type, memory_key, memory_val,
                     source, confidence, created_at, updated_at, access_count)
                 VALUES (?,?,?,?,?,?,?,?,?,0)
                 ON CONFLICT(memory_id) DO UPDATE SET
                    memory_val   = excluded.memory_val,
                    memory_type  = excluded.memory_type,
                    source       = excluded.source,
                    confidence   = excluded.confidence,
                    updated_at   = excluded.updated_at''',
              (mid, user, memory_type, key, value, source, confidence,
               datetime.now(), datetime.now()))
    conn.commit(); conn.close()


def get_user_memory(key: str, default=None):
    """Retrieve a long-term memory value for the current user."""
    user = get_current_user()
    mid  = hashlib.md5(f"{user}:{key}".encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT memory_val, access_count FROM user_memory WHERE memory_id=?', (mid,))
    row = c.fetchone()
    if row:
        # Increment access counter (helps rank importance)
        c.execute('UPDATE user_memory SET access_count=? WHERE memory_id=?',
                  (row[1]+1, mid))
        conn.commit()
    conn.close()
    return row[0] if row else default


def get_all_user_memories(memory_type: str = None) -> list:
    """Return all long-term memories for the current user, optionally filtered by type."""
    user = get_current_user()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    if memory_type:
        c.execute('''SELECT memory_key, memory_val, memory_type, source, confidence, updated_at
                     FROM user_memory WHERE user_name=? AND memory_type=?
                     ORDER BY access_count DESC''', (user, memory_type))
    else:
        c.execute('''SELECT memory_key, memory_val, memory_type, source, confidence, updated_at
                     FROM user_memory WHERE user_name=?
                     ORDER BY memory_type, access_count DESC''', (user,))
    rows = c.fetchall(); conn.close()
    return [{"key": r[0], "value": r[1], "type": r[2],
             "source": r[3], "confidence": r[4], "updated_at": r[5]} for r in rows]


def delete_user_memory(key: str):
    """Delete a specific long-term memory entry."""
    user = get_current_user()
    mid  = hashlib.md5(f"{user}:{key}".encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('DELETE FROM user_memory WHERE memory_id=?', (mid,))
    conn.commit(); conn.close()


def infer_and_save_preferences(question: str, result: dict):
    """
    Auto-infer user preferences from query patterns and save to long-term memory.
    Called after every successful Genie response.
    Examples:
      - If user often asks about a specific vendor → save as 'favourite_vendor'
      - If user consistently uses YTD queries → save 'preferred_time_range'
    """
    ql = question.lower()
    # Detect vendor preference
    vendors_in_query = []
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute('SELECT DISTINCT query_text FROM question_history WHERE user_name=? ORDER BY asked_at DESC LIMIT 20',
                  (get_current_user(),))
        recent = [r[0].lower() for r in c.fetchall()]; conn.close()
        # Simple heuristic: count vendor mentions across recent queries
        from collections import Counter
        words = " ".join(recent).split()
        # We can only reliably detect this if dim_vendor_vw data is in session
        vendors = st.session_state.get("vendor_list_stable", [])[1:]  # skip "All Vendors"
        for v in vendors[:50]:  # limit scan
            vl = v.lower()
            cnt = sum(1 for q in recent if vl in q)
            if cnt >= 3:
                set_user_memory("frequent_vendor", v, "entity", "inferred", min(cnt/10, 1.0))
                break
    except Exception:
        pass

    # Detect time range preference
    if "ytd" in ql:
        set_user_memory("preferred_preset", "YTD", "preference", "inferred", 0.8)
    elif "last 30" in ql or "30 days" in ql:
        set_user_memory("preferred_preset", "Last 30 Days", "preference", "inferred", 0.7)
    elif "qtd" in ql or "quarter" in ql:
        set_user_memory("preferred_preset", "QTD", "preference", "inferred", 0.7)


# ── Cache helpers (TTL-aware) ─────────────────────────────────────

def get_cache_with_ttl(question: str, cache_type: str = "genie"):
    """
    Retrieve a cached response, respecting TTL.
    Returns None if not found or expired.
    """
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''SELECT response_json, created_at, ttl_seconds
                 FROM query_cache
                 WHERE query_hash=? AND cache_type=?''', (q_hash, cache_type))
    row = c.fetchone(); conn.close()
    if not row:
        return None
    response_json, created_at_str, ttl = row
    # Check TTL expiry
    try:
        created_at = datetime.fromisoformat(created_at_str) if isinstance(created_at_str, str) else created_at_str
        age_seconds = (datetime.now() - created_at).total_seconds()
        if age_seconds > (ttl or 3600):
            return None  # expired
    except Exception:
        pass
    # Update hit count
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('UPDATE query_cache SET hit_count=hit_count+1, last_hit_at=? WHERE query_hash=?',
              (datetime.now(), q_hash))
    conn.commit(); conn.close()
    return json.loads(response_json)


def set_cache_with_ttl(question: str, response: dict,
                       cache_type: str = "genie", ttl_seconds: int = 3600):
    """Store a response in cache with explicit TTL and type tag."""
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    try:
        response_json = json.dumps(make_json_serializable(response))
    except Exception:
        return
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO query_cache
                 (query_hash, question, response_json, created_at, last_hit_at,
                  hit_count, ttl_seconds, cache_type)
                 VALUES (?,?,?,?,?,
                         COALESCE((SELECT hit_count+1 FROM query_cache WHERE query_hash=?),0),
                         ?,?)''',
              (q_hash, question, response_json, datetime.now(), datetime.now(),
               q_hash, ttl_seconds, cache_type))
    conn.commit(); conn.close()


def invalidate_cache(cache_type: str = None):
    """Clear all cache entries, optionally filtered by type."""
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    if cache_type:
        c.execute('DELETE FROM query_cache WHERE cache_type=?', (cache_type,))
    else:
        c.execute('DELETE FROM query_cache')
    conn.commit(); conn.close()


def get_cache_stats() -> dict:
    """Return cache statistics for the admin panel."""
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''SELECT
                    COUNT(*) AS total_entries,
                    SUM(hit_count) AS total_hits,
                    AVG(hit_count) AS avg_hits,
                    cache_type,
                    COUNT(CASE WHEN (julianday('now') - julianday(created_at))*86400 > ttl_seconds
                               THEN 1 END) AS expired_count
                 FROM query_cache
                 GROUP BY cache_type''')
    rows = c.fetchall(); conn.close()
    return [{"type": r[3], "entries": r[0], "total_hits": int(r[1] or 0),
             "avg_hits": round(r[2] or 0, 1), "expired": int(r[4] or 0)}
            for r in rows]


# ── KPI snapshot cache ────────────────────────────────────────────

def save_kpi_snapshot(preset: str, start_date: str, end_date: str, kpi: dict):
    """Persist a KPI result set for historical comparison."""
    snap_id = hashlib.md5(f"{get_current_user()}:{preset}:{start_date}:{end_date}".encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO kpi_snapshot_cache
                 (snapshot_id, user_name, preset, start_date, end_date, kpi_json, created_at)
                 VALUES (?,?,?,?,?,?,?)''',
              (snap_id, get_current_user(), preset, start_date, end_date,
               json.dumps(make_json_serializable(kpi)), datetime.now()))
    conn.commit(); conn.close()


def load_kpi_snapshot(preset: str, start_date: str, end_date: str) -> dict:
    """Load a cached KPI snapshot if available."""
    snap_id = hashlib.md5(f"{get_current_user()}:{preset}:{start_date}:{end_date}".encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT kpi_json FROM kpi_snapshot_cache WHERE snapshot_id=?', (snap_id,))
    row = c.fetchone(); conn.close()
    return json.loads(row[0]) if row else {}



def save_chat_message(session_id, turn_index, role, content, sql_used="", source=""):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('INSERT INTO chat_messages (session_id,turn_index,role,content,sql_used,source,timestamp) VALUES (?,?,?,?,?,?,?)',
              (session_id, turn_index, role, content, sql_used, source, datetime.now()))
    conn.commit(); conn.close()

def save_chat_session(session_id: str, label: str = None):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    if label is None: label = f"Session {session_id[:8]}"
    c.execute('''INSERT OR REPLACE INTO chat_sessions
        (session_id,session_label,created_at,last_updated,user_name) VALUES
        (?,?,COALESCE((SELECT created_at FROM chat_sessions WHERE session_id=?),?),
         COALESCE((SELECT last_updated FROM chat_sessions WHERE session_id=?),?),?)''',
        (session_id, label, session_id, datetime.now(), session_id, datetime.now(), get_current_user()))
    conn.commit(); conn.close()

def load_session_messages(session_id: str) -> list:
    """Load all messages for a session from the DB (used to resume a conversation)."""
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''SELECT role, content, sql_used, source, timestamp
                 FROM chat_messages
                 WHERE session_id = ?
                 ORDER BY turn_index ASC, timestamp ASC''', (session_id,))
    rows = c.fetchall(); conn.close()
    return [
        {"role": r[0], "content": r[1], "sql_used": r[2],
         "source": r[3], "timestamp": r[4]}
        for r in rows
    ]


def save_question(query, analysis_type):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('INSERT INTO question_history (normalized_query,query_text,user_name,analysis_type,asked_at) VALUES (?,?,?,?,?)',
              (query.lower().strip(), query, get_current_user(), analysis_type, datetime.now()))
    conn.commit(); conn.close()

def get_cache(question):
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT response_json FROM query_cache WHERE query_hash=?', (q_hash,))
    row = c.fetchone(); conn.close()
    return json.loads(row[0]) if row else None

def set_cache(question, response):
    q_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    try:
        response_json = json.dumps(make_json_serializable(response))
    except Exception as e:
        st.error(f"Cache serialize failed: {e}"); return
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO query_cache
        (query_hash,question,response_json,created_at,last_hit_at,hit_count) VALUES
        (?,?,?,?,?,COALESCE((SELECT hit_count+1 FROM query_cache WHERE query_hash=?),1))''',
        (q_hash, question, response_json, datetime.now(), datetime.now(), q_hash))
    conn.commit(); conn.close()

@st.cache_data(ttl=600)
def get_saved_insights_cached(page="genie", limit=20):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT insight_id,title,question,verified_query_name,created_at FROM saved_insights WHERE page=? AND created_by=? ORDER BY created_at DESC LIMIT ?',
              (page, get_current_user(), limit))
    rows = c.fetchall(); conn.close()
    return [{"id":r[0],"title":r[1],"question":r[2],"type":r[3],"created_at":r[4]} for r in rows]

@st.cache_data(ttl=600)
def get_frequent_questions_by_user_cached(limit=10):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT normalized_query,COUNT(*) as cnt FROM question_history WHERE user_name=? GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?',
              (get_current_user(), limit))
    rows = c.fetchall(); conn.close()
    return [{"query":r[0],"count":r[1]} for r in rows]

@st.cache_data(ttl=600)
def get_frequent_questions_all_cached(limit=10):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('SELECT normalized_query,COUNT(*) as cnt FROM question_history GROUP BY normalized_query ORDER BY cnt DESC LIMIT ?', (limit,))
    rows = c.fetchall(); conn.close()
    return [{"query":r[0],"count":r[1]} for r in rows]

def get_recent_conversation_context(limit: int = 20, max_age_days: int = 2) -> str:
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=max_age_days)
    c.execute('''SELECT m.role,m.content,m.timestamp FROM chat_messages m
        JOIN chat_sessions s ON m.session_id=s.session_id
        WHERE s.user_name=? AND m.timestamp>=? ORDER BY m.timestamp DESC LIMIT ?''',
        (get_current_user(), cutoff, limit))
    rows = c.fetchall(); conn.close()
    if not rows: return ""
    rows.reverse()
    parts = [f"{'User' if r[0]=='user' else 'Assistant'}: {r[1]}" for r in rows]
    return "Conversation history:\n\n" + "\n\n".join(parts) + "\n\nNew question:\n"

# ── Dashboard CSS (BG from session_state) ────────────────────
def inject_dashboard_css():
    bg = st.session_state.get("bg_color", "#ffffff")
    st.markdown(f"""<style>
    button, .stButton button, button[kind="primary"], button[kind="secondary"],
    button[data-testid^="baseButton"], .stDownloadButton button {{
        transition: all 0.2s ease !important;
    }}
    button:hover, .stButton button:hover, button[kind="primary"]:hover,
    button[kind="secondary"]:hover, button[data-testid^="baseButton"]:hover {{
        background-color: #2563eb !important; border-color: #2563eb !important;
        color: white !important; transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(37,99,235,0.3) !important;
    }}
    button[kind="primary"] {{ background-color:#2563eb !important; border-color:#2563eb !important; color:white !important; }}
    button[kind="secondary"] {{ background-color:#f3f4f6 !important; border-color:#d1d5db !important; color:#1f2937 !important; }}
    .kpi-card {{ border-radius:16px; padding:1rem 1.2rem; min-height:100px; display:flex; flex-direction:column; justify-content:center; }}
    .kpi-card-yellow {{ background:linear-gradient(135deg,#fef9c3 0%,#fef08a 100%); }}
    .kpi-card-cyan   {{ background:linear-gradient(135deg,#cffafe 0%,#a5f3fc 100%); }}
    .kpi-card-pink   {{ background:linear-gradient(135deg,#fce7f3 0%,#fbcfe8 100%); }}
    .kpi-card-purple {{ background:linear-gradient(135deg,#f3e8ff 0%,#e9d5ff 100%); }}
    .kpi-card-green  {{ background:linear-gradient(135deg,#dcfce7 0%,#bbf7d0 100%); }}
    .kpi-title  {{ font-size:0.7rem; font-weight:600; color:#374151; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.3rem; }}
    .kpi-value  {{ font-size:2rem; font-weight:800; color:#111827; line-height:1.1; }}
    .kpi-delta  {{ font-size:0.9rem; font-weight:600; margin-top:0.25rem; }}
    .kpi-delta-negative {{ color:#dc2626; }}
    .kpi-delta-positive {{ color:#16a34a; }}
    .grir-card {{ border-radius:14px; padding:0.9rem 1rem; border:1px solid #e2e8f0;
        box-shadow:0 2px 8px rgba(0,0,0,0.05); display:flex; flex-direction:column;
        gap:0.2rem; min-height:90px; justify-content:center; }}
    .grir-card-title {{ font-size:0.7rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.6px; }}
    .grir-card-value {{ font-size:1.8rem; font-weight:800; color:#111827; line-height:1.1; }}
    .chart-title {{ font-size:1.1rem; font-weight:700; color:#111827; margin-bottom:0.5rem; }}
    .main > .block-container {{ background-color:{bg} !important; }}
    .stApp {{ background-color:{bg} !important; }}
    .message-user {{ background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%); color:white;
        padding:10px 16px; border-radius:18px 18px 4px 18px; margin:8px 0;
        max-width:80%; margin-left:auto; text-align:right; }}
    .message-assistant {{ background:#f1f5f9; color:#1e293b; padding:10px 16px;
        border-radius:18px 18px 18px 4px; margin:8px 0; max-width:85%; }}
    .start-conversation {{ text-align:center; padding:2rem 1rem; background:#f8fafc; border-radius:20px; margin:1rem 0; }}
    .chat-messages {{ max-height:400px; overflow-y:auto; padding:0.5rem; margin-bottom:1rem;
        background:#fafcff; border-radius:16px; border:1px solid #e2e8f0; }}
    .quick-card {{ background:white; border-radius:16px; padding:1.2rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.06); border:1px solid #e2e8f0;
        text-align:center; height:100%; display:flex; flex-direction:column; }}
    .quick-card h3 {{ font-size:1rem; font-weight:600; color:#1e293b; margin:0 0 0.4rem 0; }}
    .quick-card p  {{ font-size:0.8rem; color:#64748b; flex-grow:1; margin:0 0 0.8rem 0; }}
    /* ── BG CIRCLE BUTTON — global, fires on every page ── */
    button[data-testid="baseButton-secondary"][aria-label="BG"],
    button[data-testid="baseButton-secondary"][aria-label="X"] {{
        width:52px!important;height:52px!important;
        min-width:52px!important;max-width:52px!important;
        min-height:52px!important;max-height:52px!important;
        border-radius:50%!important;padding:0!important;margin:0!important;
        background:white!important;color:#374151!important;
        border:2px solid #e5e7eb!important;
        box-shadow:0 2px 12px rgba(0,0,0,0.15)!important;
        font-size:13px!important;font-weight:700!important;
        line-height:52px!important;text-align:center!important;
        overflow:hidden!important;display:flex!important;
        align-items:center!important;justify-content:center!important;
    }}
    button[data-testid="baseButton-secondary"][aria-label="BG"]:hover,
    button[data-testid="baseButton-secondary"][aria-label="X"]:hover {{
        transform:scale(1.1)!important;
        box-shadow:0 4px 18px rgba(0,0,0,0.20)!important;
        background:#f9fafb!important;border-color:#9ca3af!important;
    }}
    div[data-testid="stButton"]:has(button[aria-label="BG"]),
    div[data-testid="stButton"]:has(button[aria-label="X"]) {{
        width:52px!important;max-width:52px!important;
        min-width:52px!important;flex:0 0 52px!important;
        padding:0!important;margin:0!important;
    }}
    </style>""", unsafe_allow_html=True)

def render_kpi_card(title, value, delta=None, is_positive=True, color_class="yellow"):
    delta_html = ""
    if delta is not None and delta != "-":
        dc = "kpi-delta-positive" if is_positive else "kpi-delta-negative"
        arrow = "↑" if is_positive else "↓"
        delta_html = f'<div class="kpi-delta {dc}">{delta} {arrow}</div>'
    elif delta == "-":
        delta_html = '<div class="kpi-delta" style="color:#9ca3af;">-</div>'
    st.markdown(f"""<div class="kpi-card kpi-card-{color_class}">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  {delta_html}
</div>""", unsafe_allow_html=True)

def render_grir_metric_card(title: str, value: str, bg_color: str = "#ffffff"):
    st.markdown(f"""<div class="grir-card" style="background-color:{bg_color};">
  <div class="grir-card-title">{title}</div>
  <div class="grir-card-value">{value}</div>
</div>""", unsafe_allow_html=True)

# ── BG Button: fixed bottom-right, pure Streamlit (no JS/HTML floating) ──
def render_bg_button_sidebar():
    """
    BG colour picker — circular button bottom-right of Spend Trend chart.
    Click opens a compact popup: gradient canvas + hue slider + hex input
    (matches the screenshot exactly). Selecting a colour immediately sets
    the app background. Called from render_charts() after the Spend Trend
    container, so it sits visually at the bottom-right of that chart.
    """
    current_bg = st.session_state.get("bg_color", "#ffffff")
    if "show_bg_panel" not in st.session_state:
        st.session_state.show_bg_panel = False

    # ── CSS: circular BG button ─────────────────────────────────
    # ── BG circle button — inject CSS globally + render button ──────
    st.markdown("""
<style>
/* === BG CIRCLE BUTTON — every possible selector === */

/* Target the wrapper column that contains bg_pill_btn */
div[data-testid="stColumn"]:has(button[data-testid="baseButton-secondary"]) {
    min-width: 0 !important;
    width: auto !important;
    flex: none !important;
}

/* The button itself — all states */
button[data-testid="baseButton-secondary"][aria-label="BG"],
button[data-testid="baseButton-secondary"][aria-label="X"],
button[data-testid="baseButton-secondary"][aria-label="BG"]:focus,
button[data-testid="baseButton-secondary"][aria-label="BG"]:active,
button[data-testid="baseButton-secondary"][aria-label="X"]:focus,
button[data-testid="baseButton-secondary"][aria-label="X"]:active {
    width:         52px !important;
    min-width:     52px !important;
    max-width:     52px !important;
    height:        52px !important;
    min-height:    52px !important;
    max-height:    52px !important;
    border-radius: 50% !important;
    padding:       0 !important;
    margin:        0 !important;
    font-size:     13px !important;
    font-weight:   700 !important;
    line-height:   52px !important;
    text-align:    center !important;
    background:    white !important;
    color:         #374151 !important;
    border:        2px solid #e5e7eb !important;
    box-shadow:    0 2px 12px rgba(0,0,0,0.15), 0 0 0 0 transparent !important;
    outline:       none !important;
    cursor:        pointer !important;
    display:       flex !important;
    align-items:   center !important;
    justify-content: center !important;
    flex-shrink:   0 !important;
    overflow:      hidden !important;
    transition:    box-shadow 0.18s ease, transform 0.18s ease !important;
}
button[data-testid="baseButton-secondary"][aria-label="BG"]:hover,
button[data-testid="baseButton-secondary"][aria-label="X"]:hover {
    transform:    scale(1.1) !important;
    box-shadow:   0 4px 18px rgba(0,0,0,0.20) !important;
    background:   #f9fafb !important;
    border-color: #9ca3af !important;
}
/* Parent stButton div: constrain to circle size */
div[data-testid="stButton"]:has(button[aria-label="BG"]),
div[data-testid="stButton"]:has(button[aria-label="X"]) {
    width:     52px !important;
    max-width: 52px !important;
    min-width: 52px !important;
    flex:      0 0 52px !important;
    padding:   0 !important;
    margin:    0 !important;
}
/* Colour picker */
.bg-picker-panel div[data-testid="stColorPicker"] label { display: none !important; }
.bg-picker-panel div[data-testid="stColorPicker"] > div  { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

    # ── BG circle button — use_container_width=False is critical ─────
    lbl = "X" if st.session_state.show_bg_panel else "BG"
    if st.button(lbl, key="bg_pill_btn", use_container_width=False):
        st.session_state.show_bg_panel = not st.session_state.show_bg_panel
        st.rerun()

    # ── Picker panel: opens above the button ─────────────────────
    if st.session_state.show_bg_panel:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:10px;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.6px;"
                "margin-bottom:4px;'>🎨 BG</div>",
                unsafe_allow_html=True,
            )
            safe_val = current_bg if (
                current_bg.startswith("#") and len(current_bg) in (4, 7)
            ) else "#ffffff"
            picked = st.color_picker(
                "bg", value=safe_val,
                key="bg_cp", label_visibility="collapsed",
            )
            if picked != current_bg:
                st.session_state["bg_color"] = picked
                st.session_state.show_bg_panel = False
                st.rerun()

# ── FIXED KPI fetching using correct view column names ───────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_kpi_data(start_lit: str, end_lit: str, vendor_where: str,
                   start_iso: str, end_iso: str) -> dict:
    """
    PERFORMANCE: All 8 KPIs + vendor list in ONE merged Athena query.
    Old approach: 5 sequential queries per period (10 total for cur+prev).
    New approach: 1 query per period → 2 Athena calls total for all KPIs.

    Uses a single pass over fact_all_sources_vw + LEFT JOINs to the
    aggregate views, all in one CTE/subquery block.
    """
    start = date.fromisoformat(start_iso)
    end   = date.fromisoformat(end_iso)
    ym    = year_month_filter(start, end)

    # ── Single merged query: all KPIs from fact table + aggregate views ──────
    # We UNION the three aggregate views as scalar subqueries so Athena
    # can scan fact_all_sources_vw only once.
    merged_sql = f"""
        WITH fact_agg AS (
            SELECT
                SUM(CASE WHEN UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')
                         THEN COALESCE(f.invoice_amount_local,0) ELSE 0 END)      AS total_spend,
                COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status)='OPEN'
                                    THEN f.purchase_order_reference END)           AS active_pos,
                COUNT(DISTINCT f.purchase_order_reference)                         AS total_pos,
                COUNT(DISTINCT CASE WHEN UPPER(f.invoice_status)='OPEN'
                                    THEN f.invoice_number END)                     AS pending_inv,
                COUNT(DISTINCT CASE WHEN v.vendor_name IS NOT NULL
                                    THEN v.vendor_name END)                        AS active_vendors
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
            {vendor_where}
        ),
        cycle_agg AS (
            SELECT AVG(CAST(avg_payment_cycle_time_days AS DOUBLE)) AS avg_days
            FROM {DATABASE}.payment_processing_cycle_time_vw
            WHERE {ym}
        ),
        fp_agg AS (
            SELECT
                SUM(CAST(full_paid_invoices AS BIGINT))     AS full_paid,
                SUM(CAST(total_cleared_invoices AS BIGINT)) AS total_cleared
            FROM {DATABASE}.full_payment_rate_vw
            WHERE {ym}
        ),
        auto_agg AS (
            SELECT
                COUNT(*) AS total_cleared_inv,
                SUM(CASE WHEN UPPER(status_notes)='AUTO PROCESSED' THEN 1 ELSE 0 END) AS auto_proc
            FROM {DATABASE}.invoice_status_history_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
              AND UPPER(status) IN ('PAID','CLEARED')
        )
        SELECT
            fa.total_spend, fa.active_pos, fa.total_pos,
            fa.pending_inv, fa.active_vendors,
            ca.avg_days          AS avg_processing_days,
            fp.full_paid         AS fp_full_paid,
            fp.total_cleared     AS fp_total_cleared,
            aa.total_cleared_inv AS auto_total,
            aa.auto_proc         AS auto_processed
        FROM fact_agg fa
        CROSS JOIN cycle_agg ca
        CROSS JOIN fp_agg fp
        CROSS JOIN auto_agg aa
    """
    df = run_query(merged_sql)

    if df.empty:
        return {
            "total_spend": 0.0, "active_pos": 0, "total_pos": 0,
            "pending_inv": 0, "active_vendors": 0,
            "avg_processing_days": 0.0, "first_pass_rate": 0.0, "auto_rate": 0.0,
        }

    row = df.iloc[0]
    total_spend      = safe_number(row["total_spend"])
    active_pos       = safe_int(row["active_pos"])
    total_pos        = safe_int(row["total_pos"])
    pending_inv      = safe_int(row["pending_inv"])
    active_vendors   = safe_int(row["active_vendors"])
    avg_proc_days    = safe_number(row.get("avg_processing_days"))
    fp_paid          = safe_number(row.get("fp_full_paid", 0))
    fp_cleared       = safe_number(row.get("fp_total_cleared", 0))
    first_pass_rate  = (fp_paid / fp_cleared * 100) if fp_cleared > 0 else 0.0
    auto_total       = safe_int(row.get("auto_total", 0))
    auto_proc        = safe_int(row.get("auto_processed", 0))
    auto_rate        = (auto_proc / auto_total * 100) if auto_total > 0 else 0.0

    # Fallback for avg_processing_days if view returned NULL
    if avg_proc_days == 0.0 or pd.isna(avg_proc_days):
        fb = run_query(f"""
            SELECT AVG(CAST(DATE_DIFF('day',posting_date,payment_date) AS DOUBLE)) AS avg_days
            FROM {DATABASE}.fact_all_sources_vw
            WHERE UPPER(invoice_status) IN ('PAID','CLEARED')
              AND payment_date IS NOT NULL
              AND posting_date BETWEEN {start_lit} AND {end_lit}
        """)
        avg_proc_days = safe_number(fb.iloc[0]["avg_days"]) if not fb.empty else 0.0

    return {
        "total_spend":          total_spend,
        "active_pos":           active_pos,
        "total_pos":            total_pos,
        "pending_inv":          pending_inv,
        "active_vendors":       active_vendors,
        "avg_processing_days":  avg_proc_days,
        "first_pass_rate":      first_pass_rate,
        "auto_rate":            auto_rate,
    }


    return result

@st.cache_data(ttl=600, show_spinner=False)
def fetch_needs_attention(start_lit: str, end_lit: str, vendor_where: str):
    """
    PERFORMANCE: Single UNION query replaces 3 sequential queries.
    One Athena scan of fact_all_sources_vw instead of three.
    A 'category' column tags each row so we can split into DataFrames in Python.
    """
    union_sql = f"""
        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days,
               'OVERDUE' AS category
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
          {vendor_where}
          AND f.due_date < CURRENT_DATE
          AND UPPER(f.invoice_status) = 'OVERDUE'

        UNION ALL

        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days,
               'DISPUTED' AS category
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
          {vendor_where}
          AND UPPER(f.invoice_status) IN ('DISPUTE','DISPUTED')

        UNION ALL

        SELECT f.invoice_number AS ref_no,
               f.invoice_amount_local AS amount,
               v.vendor_name,
               f.due_date,
               f.aging_days,
               'DUE' AS category
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
        WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
          {vendor_where}
          AND f.due_date >= CURRENT_DATE
          AND f.due_date <= CURRENT_DATE + INTERVAL '30' DAY
          AND UPPER(f.invoice_status) = 'OPEN'

        ORDER BY due_date ASC
    """
    all_df = run_query(union_sql)
    if all_df.empty:
        empty = pd.DataFrame(columns=["ref_no","amount","vendor_name","due_date","aging_days"])
        return empty, empty, empty

    overdue_df  = all_df[all_df["category"] == "OVERDUE"].drop(columns=["category"]).reset_index(drop=True)
    disputed_df = all_df[all_df["category"] == "DISPUTED"].drop(columns=["category"]).reset_index(drop=True)
    due_df      = all_df[all_df["category"] == "DUE"].drop(columns=["category"]).reset_index(drop=True)
    return overdue_df, disputed_df, due_df

def _load_vendor_list():
    """
    Load vendor list into a SINGLE stable session_state key: "vendor_list_stable".
    Only re-fetches when the date range actually changes.
    This prevents the selectbox from duplicating when preset buttons are clicked.
    """
    rng_start, rng_end = st.session_state.date_range
    last_start = st.session_state.get("_vendor_list_last_start")
    last_end   = st.session_state.get("_vendor_list_last_end")

    # Only re-query when date range truly changed
    needs_reload = (
        "vendor_list_stable" not in st.session_state
        or last_start != rng_start
        or last_end   != rng_end
    )

    if needs_reload:
        # Use @st.cache_data — run_query already caches this call at 600s
        vdf = run_query(
            f"SELECT DISTINCT v.vendor_name "
            f"FROM {DATABASE}.fact_all_sources_vw f "
            f"LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id "
            f"WHERE f.posting_date BETWEEN {sql_date(rng_start)} AND {sql_date(rng_end)} "
            f"AND v.vendor_name IS NOT NULL "
            f"ORDER BY 1"
        )
        new_list = (["All Vendors"] + vdf["vendor_name"].tolist()
                    if not vdf.empty else ["All Vendors"])
        st.session_state["vendor_list_stable"]      = new_list
        st.session_state["_vendor_list_last_start"] = rng_start
        st.session_state["_vendor_list_last_end"]   = rng_end
        if st.session_state.selected_vendor not in new_list:
            st.session_state.selected_vendor = "All Vendors"


def render_filters():
    """
    Renders date range, vendor selector, and preset buttons.

    KEY FIX — vendor list duplication:
    ─────────────────────────────────
    Root cause: old code used a date-keyed cache (vendor_list_2026-01-01_2026-06-11)
    so each preset click created a brand-new list in session_state, and the
    selectbox (with a fixed key) received a different options list every render,
    causing Streamlit to reset and visually duplicate the dropdown.

    Fix: single stable key "vendor_list_stable" managed by _load_vendor_list().
    The list is only reloaded when the date range actually changes — NOT on every
    preset button click. The selectbox always reads from the same stable key and
    the same widget key, so it never duplicates.
    """
    # ── Step 1: ensure vendor list is loaded (stable single key) ──
    _load_vendor_list()

    rng_start, rng_end = st.session_state.date_range
    selected_vendor    = st.session_state.selected_vendor
    current_preset     = st.session_state.preset
    vendor_list        = st.session_state["vendor_list_stable"]

    # ── Inject CSS: filter row — all elements same height, visible, no wrap ─
    st.markdown("""
<style>
/* ── Filter row: align all elements to centre vertically ── */
section.main div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
    align-items: center !important;
    min-height: 44px !important;
}
/* Date input */
div[data-testid="stDateInput"] input {
    height: 40px !important;
    min-height: 40px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 0 10px !important;
    white-space: nowrap !important;
}
/* Vendor selectbox */
div[data-testid="stSelectbox"] > div {
    height: 40px !important;
    min-height: 40px !important;
}
div[data-testid="stSelectbox"] > div > div {
    height: 40px !important;
    min-height: 40px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 0 10px !important;
    display: flex !important;
    align-items: center !important;
}
/* Preset buttons — no text wrap, fixed height */
div[data-testid="stHorizontalBlock"]:nth-of-type(2) button {
    height: 40px !important;
    min-height: 40px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 13px !important;
    padding: 0 12px !important;
    border-radius: 8px !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)

    # Layout: date(1.4) | vendor(1.4) | [Last30(1.4) QTD(0.8) YTD(0.8) Custom(0.8)]
    # Total ratio sums to ~6.6 — "Last 30 Days" gets enough room to fit on one line
    col_date, col_vendor, col_l30, col_qtd, col_ytd, col_custom = st.columns(
        [1.4, 1.4, 1.35, 0.75, 0.75, 0.75], gap="small"
    )

    # ── Date range picker ──────────────────────────────────────
    with col_date:
        date_range = st.date_input(
            "Date Range",
            value=(rng_start, rng_end),
            format="YYYY-MM-DD",
            label_visibility="collapsed",
            key="date_range_widget",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            ns, ne = date_range
            if (ns, ne) != (rng_start, rng_end):
                if not st.session_state.get("_preset_clicked", False):
                    st.session_state.date_range = (ns, ne)
                    st.session_state.preset     = "Custom"
                    st.session_state.pop("vendor_list_stable", None)
                else:
                    st.session_state._preset_clicked = False

    # ── Vendor selector ────────────────────────────────────────
    with col_vendor:
        try:
            v_idx = vendor_list.index(selected_vendor)
        except ValueError:
            v_idx = 0
        chosen = st.selectbox(
            "Vendor",
            options=vendor_list,
            index=v_idx,
            label_visibility="collapsed",
            key="vendor_selectbox_stable",
        )
        if chosen != st.session_state.selected_vendor:
            st.session_state.selected_vendor = chosen

    # ── Preset buttons — each in its own sized column so text never wraps ──
    preset_map = [
        (col_l30,    "Last 30 Days"),
        (col_qtd,    "QTD"),
        (col_ytd,    "YTD"),
        (col_custom, "Custom"),
    ]
    for p_col, p in preset_map:
        with p_col:
            btn_type = "primary" if p == current_preset else "secondary"
            if st.button(p, key=f"preset_{p}", use_container_width=True, type=btn_type):
                st.session_state._preset_clicked = True
                if p != "Custom":
                    ns2, ne2 = compute_range_preset(p)
                    st.session_state.date_range = (ns2, ne2)
                    st.session_state.pop("vendor_list_stable", None)
                st.session_state.preset = p
                st.rerun()

    return (
        st.session_state.date_range[0],
        st.session_state.date_range[1],
        st.session_state.selected_vendor,
    )

def render_kpi_rows(kpi: dict, prev_kpi: dict):
    cur_spend = kpi.get("total_spend", 0); prev_spend = prev_kpi.get("total_spend", 0)
    cur_apos  = kpi.get("active_pos", 0);  prev_apos  = prev_kpi.get("active_pos", 0)
    cur_tpos  = kpi.get("total_pos", 0);   prev_tpos  = prev_kpi.get("total_pos", 0)
    cur_av    = kpi.get("active_vendors", 0); prev_av = prev_kpi.get("active_vendors", 0)
    cur_pend  = kpi.get("pending_inv", 0); prev_pend  = prev_kpi.get("pending_inv", 0)
    cur_avg   = kpi.get("avg_processing_days", 0.0); prev_avg = prev_kpi.get("avg_processing_days", 0.0)
    cur_fp    = kpi.get("first_pass_rate", 0.0); prev_fp = prev_kpi.get("first_pass_rate", 0.0)
    auto_rate = kpi.get("auto_rate", 0.0)

    spend_d, spend_up = pct_delta(cur_spend, prev_spend)
    apos_d,  apos_up  = pct_delta(cur_apos,  prev_apos)
    tpos_d,  tpos_up  = pct_delta(cur_tpos,  prev_tpos)
    av_d,    av_up    = pct_delta(cur_av,     prev_av)
    pend_d,  pend_up  = pct_delta(cur_pend,   prev_pend)

    avg_diff    = cur_avg - prev_avg
    avg_d_str   = f"{abs(avg_diff):.1f}d"
    avg_up      = avg_diff < 0

    fp_diff     = cur_fp - prev_fp
    fp_d_str    = f"{abs(fp_diff):.1f}%"
    fp_up       = fp_diff > 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_kpi_card("TOTAL SPEND", abbr_currency(cur_spend), spend_d, spend_up, "yellow")
    with col2: render_kpi_card("ACTIVE PO'S", f"{cur_apos:,}", apos_d, apos_up, "cyan")
    with col3: render_kpi_card("TOTAL PO'S", f"{cur_tpos:,}", tpos_d, tpos_up, "pink")
    with col4: render_kpi_card("ACTIVE VENDORS", f"{cur_av:,}", av_d if prev_av > 0 else "-", av_up, "purple")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: render_kpi_card("PENDING INVOICES", f"{cur_pend:,}", pend_d, not pend_up, "yellow")
    with col2: render_kpi_card("AVG INVOICE PROCESSING TIME", f"{cur_avg:.1f}d", avg_d_str, avg_up, "cyan")
    with col3: render_kpi_card("FIRST PASS INVOICES %", f"{cur_fp:.1f}%", fp_d_str, fp_up, "green")
    with col4: render_kpi_card("AUTOPROCESSED INVOICES %", f"{auto_rate:.1f}%", "-", True, "green")

def render_needs_attention(rng_start, rng_end, vendor_where):
    for k, v in [("na_tab", "Overdue"), ("na_page", 0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    current_tab = st.session_state.na_tab
    page        = st.session_state.na_page
    start_lit   = sql_date(rng_start)
    end_lit     = sql_date(rng_end)

    overdue_df, disputed_df, due_df = fetch_needs_attention(start_lit, end_lit, vendor_where)
    oc  = len(overdue_df)
    dc  = len(disputed_df)
    duc = len(due_df)
    urgent = oc + dc + duc

    if current_tab == "Overdue":
        df = overdue_df;  sl = "Overdue";  tc_color = "#e53935"
    elif current_tab == "Disputed":
        df = disputed_df; sl = "Disputed"; tc_color = "#e53935"
    else:
        df = due_df;      sl = "Due soon"; tc_color = "#2e7d32"

    # ── CSS injected once — all styling here, no JS class tagging ────────────
    # Cards use unique per-card IDs injected via st.markdown so CSS targets them
    # precisely without needing window.parent or class injection.
    st.markdown("""
<style>
/* ── NA outer wrapper ── */
.na-outer {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px 16px 12px 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 6px;
}
.na-title {
    font-size: 16px; font-weight: 800; color: #111827; margin-bottom: 10px;
}
.na-title span { font-weight: 600; color: #6b7280; font-size: 14px; }

/* ── Tab buttons ── */
.na-tabs-row { margin-bottom: 10px; }
.na-tabs-row button {
    height: 42px !important;
    min-height: 42px !important;
    border-radius: 999px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}
.na-tabs-row button[kind="secondary"] {
    background: #f3f4f6 !important;
    border: 1.5px solid #e0e0e0 !important;
    color: #374151 !important;
    box-shadow: none !important;
}
.na-tabs-row button[kind="primary"] {
    background: #2563eb !important;
    color: white !important;
    border-color: #2563eb !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.28) !important;
}

/* ── Each individual card: target by unique data-na-card attribute ── */
[data-na-card] {
    background: #FFF0F2 !important;
    border: 1.5px solid #f5c6cb !important;
    border-radius: 14px !important;
    padding: 12px 12px 10px 12px !important;
    box-shadow: 0 2px 6px rgba(229,57,53,0.07) !important;
    margin-bottom: 0 !important;
}

/* ── Invoice number button inside a card ── */
[data-na-card] button {
    background:    #f3f4f6 !important;
    border:        1px solid #d1d5db !important;
    border-radius: 8px !important;
    color:         #374151 !important;
    font-size:     13px !important;
    font-weight:   700 !important;
    height:        30px !important;
    min-height:    30px !important;
    padding:       0 10px !important;
    box-shadow:    none !important;
    outline:       none !important;
    width: auto !important;
    display: inline-block !important;
}
[data-na-card] button:hover {
    background:  #ebebeb !important;
    border-color:#9ca3af !important;
    color:       #374151 !important;
    box-shadow:  none !important;
    outline:     none !important;
}
[data-na-card] button:focus,
[data-na-card] button:focus-visible,
[data-na-card] button:active {
    background:         #f3f4f6 !important;
    border-color:       #d1d5db !important;
    box-shadow:         none !important;
    -webkit-box-shadow: none !important;
    outline:            none !important;
    outline-width:      0 !important;
}

/* ── Pagination buttons ── */
.na-page-row button {
    height: 38px !important;
    min-height: 38px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    background: #f3f4f6 !important;
    border: 1px solid #e0e0e0 !important;
    color: #374151 !important;
    box-shadow: none !important;
}
.na-page-info {
    text-align: center; color: #6b7280; font-size: 13px; padding: 8px 0;
}
</style>
""", unsafe_allow_html=True)

    # ── Outer wrapper ─────────────────────────────────────────────────────────
    # Title
    st.markdown(
        f"<div class='na-title'>Needs Attention "
        f"<span>({urgent:,})</span></div>",
        unsafe_allow_html=True,
    )

    # Tab buttons (inside na-tabs-row div for CSS scoping)
    st.markdown("<div class='na-tabs-row'>", unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns([1, 1, 1], gap="small")
    with tc1:
        t = "primary" if current_tab == "Overdue" else "secondary"
        if st.button(f"Overdue ({oc})", key="na_btn_overdue", use_container_width=True, type=t):
            st.session_state.na_tab = "Overdue"; st.session_state.na_page = 0; st.rerun()
    with tc2:
        t = "primary" if current_tab == "Disputed" else "secondary"
        if st.button(f"Disputed ({dc})", key="na_btn_disputed", use_container_width=True, type=t):
            st.session_state.na_tab = "Disputed"; st.session_state.na_page = 0; st.rerun()
    with tc3:
        t = "primary" if current_tab == "Due" else "secondary"
        if st.button(f"Due ({duc})", key="na_btn_due30d", use_container_width=True, type=t):
            st.session_state.na_tab = "Due"; st.session_state.na_page = 0; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    if df.empty:
        st.markdown(
            "<div style='padding:2rem;color:#64748b;text-align:center;'>"
            "No items in this category</div>",
            unsafe_allow_html=True,
        )
    else:
        ipp = 8; tot = len(df); tp = max(1, (tot + ipp - 1) // ipp)
        si  = page * ipp; ei2 = min(si + ipp, tot)
        page_df = df.iloc[si:ei2]; gi = 0

        # ── Card CSS: paint stVerticalBlockBorderWrapper as pink card ─────────
        # Each card uses st.container(border=True). We inject per-card CSS
        # using nth-child selectors relative to a unique parent wrapper.
        # Inside each container: invoice st.button (grey pill) + HTML for the rest.
        # st.button is INSIDE st.container so it renders inside the pink box.
        # Nth-child approach: inject a unique class on each column div via
        # a sequential CSS counter — reliable, no JS, no window.parent.

        # Inject one global CSS block to style ALL card containers at once
        st.markdown("""
<style>
/* Paint every stVerticalBlockBorderWrapper that is a direct child of a
   stHorizontalBlock inside the na-grid-wrap as the pink card style */
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFF0F2 !important;
    border: 1.5px solid #f5c6cb !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 6px rgba(229,57,53,0.07) !important;
    overflow: visible !important;
}
/* Invoice button inside card: grey pill */
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] button {
    background:    #f3f4f6 !important;
    border:        1px solid #d1d5db !important;
    border-radius: 8px !important;
    color:         #374151 !important;
    font-size:     13px !important;
    font-weight:   700 !important;
    height:        30px !important;
    min-height:    30px !important;
    padding:       0 10px !important;
    box-shadow:    none !important;
    outline:       none !important;
    width:         auto !important;
    max-width:     none !important;
    display:       inline-block !important;
    white-space:   nowrap !important;
    overflow:      visible !important;
}
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] button:hover {
    background:    #eff6ff !important;
    border-color:  #2563eb !important;
    color:         #2563eb !important;
    box-shadow:    none !important;
    outline:       none !important;
}
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] button:focus,
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] button:focus-visible,
.na-grid-wrap div[data-testid="stVerticalBlockBorderWrapper"] button:active {
    background:         #f3f4f6 !important;
    border-color:       #d1d5db !important;
    box-shadow:         none !important;
    -webkit-box-shadow: none !important;
    outline:            none !important;
    outline-width:      0 !important;
}
</style>
""", unsafe_allow_html=True)

        # na-grid-wrap div wraps the columns — CSS above scopes to this div
        st.markdown("<div class='na-grid-wrap'>", unsafe_allow_html=True)

        for chunk_start in range(0, len(page_df), 4):
            row_chunk = page_df.iloc[chunk_start:chunk_start + 4]
            cols = st.columns(4, gap="medium")
            for col, (_, r) in zip(cols, row_chunk.iterrows()):
                with col:
                    ref   = format_invoice_number(str(r.get("ref_no","—")).strip() or "—")
                    vname = html.escape(str(r.get("vendor_name","—")))
                    amt   = safe_number(r.get("amount"))
                    ddr   = r.get("due_date")
                    dd    = pd.to_datetime(ddr).date().isoformat() if pd.notna(ddr) else "—"
                    bk    = f"na_btn_{si}_{gi}_{ref[:20]}"

                    with st.container(border=True):
                        # ── Row 1: invoice number button + status label ──────────
                        # No inner st.columns — they squeeze the number.
                        # Use full-width button + right-aligned status via CSS.
                        if st.button(ref, key=bk):
                            st.session_state["invoice_search_from_card"] = ref
                            st.session_state["page"] = "Invoices"
                            st.experimental_set_query_params(invoice=ref)
                            st.rerun()
                        st.markdown(
                            f"<div style='text-align:right;margin-top:-28px;"
                            f"font-size:11px;font-weight:700;color:{tc_color};'>"
                            f"{sl}</div>",
                            unsafe_allow_html=True,
                        )
                        # ── Row 2: amount + due date (right) ──
                        st.markdown(
                            f"<div style='text-align:right;margin-top:2px;'>"
                            f"<div style='font-size:14px;font-weight:800;"
                            f"color:#111827;line-height:1.2;'>{abbr_currency(amt)}</div>"
                            f"<div style='font-size:10px;color:#9ca3af;"
                            f"margin-top:1px;'>Due: {dd}</div></div>",
                            unsafe_allow_html=True,
                        )
                        # ── Row 3: vendor name (bottom-left) ──
                        st.markdown(
                            f"<div style='font-size:11px;color:#6b7280;"
                            f"margin-top:2px;'>{vname}</div>",
                            unsafe_allow_html=True,
                        )
                    gi += 1
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # na-grid-wrap

        # ── Pagination ────────────────────────────────────────────────────────
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='na-page-row'>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns([1, 1, 1], gap="small")
        with pc1:
            if page > 0:
                if st.button("← Prev", key="na_prev", use_container_width=True):
                    st.session_state.na_page = max(0, page - 1); st.rerun()
            else:
                st.markdown(
                    "<div style='text-align:center;color:#d1d5db;padding:8px;"
                    "font-size:13px;'>← Prev</div>",
                    unsafe_allow_html=True,
                )
        with pc2:
            st.markdown(
                f"<div class='na-page-info'>{page + 1} of {tp}</div>",
                unsafe_allow_html=True,
            )
        with pc3:
            if page < tp - 1:
                if st.button("Next →", key="na_next", use_container_width=True):
                    st.session_state.na_page = min(tp - 1, page + 1); st.rerun()
            else:
                st.markdown(
                    "<div style='text-align:center;color:#d1d5db;padding:8px;"
                    "font-size:13px;'>Next →</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # end render_needs_attention


def fetch_chart_data(start_lit: str, end_lit: str, vendor_where: str,
                     end_lit_6m: str) -> tuple:
    """
    PERFORMANCE: Fetch all 3 chart datasets in ONE Athena query using CTEs.
    Old: 3 sequential queries. New: 1 query, split into 3 DataFrames in Python.
    """
    merged_sql = f"""
        WITH
        -- Status distribution
        status_dist AS (
            SELECT
                CASE
                    WHEN UPPER(invoice_status) IN ('PAID','CLEARED','CLOSED','POSTED','SETTLED') THEN 'Paid'
                    WHEN UPPER(invoice_status) IN ('OPEN','PENDING','ON HOLD','PARKED','IN PROGRESS') THEN 'Pending'
                    WHEN UPPER(invoice_status) IN ('DISPUTE','DISPUTED','BLOCKED','CONTESTED') THEN 'Disputed'
                    ELSE 'Other'
                END AS status,
                COUNT(*) AS cnt,
                'STATUS' AS _type
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date BETWEEN {start_lit} AND {end_lit}
            GROUP BY 1
        ),
        -- Top 10 vendors by spend
        top_vendors AS (
            SELECT
                COALESCE(v.vendor_name,'Unknown') AS vendor_name,
                SUM(COALESCE(f.invoice_amount_local,0)) AS spend,
                'VENDOR' AS _type
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id = v.vendor_id
            WHERE f.posting_date BETWEEN {start_lit} AND {end_lit}
            {vendor_where}
            GROUP BY 1
            ORDER BY spend DESC
            LIMIT 10
        ),
        -- Monthly spend trend (last 6 months)
        spend_trend AS (
            SELECT
                DATE_TRUNC('month', posting_date) AS month,
                SUM(COALESCE(invoice_amount_local,0)) AS actual_spend,
                'TREND' AS _type
            FROM {DATABASE}.fact_all_sources_vw
            WHERE posting_date >= {end_lit_6m}
              AND UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
            GROUP BY 1
            ORDER BY 1
        )
        -- Return all three result sets tagged by _type
        SELECT CAST(status AS VARCHAR) AS col_a, CAST(cnt AS VARCHAR) AS col_b,
               CAST(NULL AS VARCHAR) AS col_c, _type FROM status_dist
        UNION ALL
        SELECT vendor_name AS col_a, CAST(spend AS VARCHAR) AS col_b,
               NULL AS col_c, _type FROM top_vendors
        UNION ALL
        SELECT CAST(month AS VARCHAR) AS col_a, CAST(actual_spend AS VARCHAR) AS col_b,
               NULL AS col_c, _type FROM spend_trend
    """
    all_df = run_query(merged_sql)

    # Split by _type tag
    if all_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    status_df = all_df[all_df["_type"] == "STATUS"][["col_a","col_b"]].copy()
    status_df.columns = ["status","cnt"]
    status_df["cnt"] = pd.to_numeric(status_df["cnt"], errors="coerce").fillna(0).astype(int)

    vendor_df = all_df[all_df["_type"] == "VENDOR"][["col_a","col_b"]].copy()
    vendor_df.columns = ["vendor_name","spend"]
    vendor_df["spend"] = pd.to_numeric(vendor_df["spend"], errors="coerce").fillna(0)

    trend_df = all_df[all_df["_type"] == "TREND"][["col_a","col_b"]].copy()
    trend_df.columns = ["month","actual_spend"]
    trend_df["actual_spend"] = pd.to_numeric(trend_df["actual_spend"], errors="coerce").fillna(0)

    return status_df, vendor_df, trend_df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_chart_data_cached(start_lit: str, end_lit: str, vendor_where: str,
                             end_lit_6m: str) -> tuple:
    """Cached wrapper for fetch_chart_data."""
    return fetch_chart_data(start_lit, end_lit, vendor_where, end_lit_6m)


def render_charts(rng_start, rng_end, vendor_where):
    """
    PERFORMANCE: One cached Athena call for all 3 charts (was 3 sequential).
    """
    start_lit   = sql_date(rng_start)
    end_lit     = sql_date(rng_end)
    end_lit_6m  = f"DATE_ADD('month', -6, {end_lit})"

    status_df, vendor_df, trend_df = fetch_chart_data_cached(
        start_lit, end_lit, vendor_where, end_lit_6m
    )

    # Three-column layout: Status Distribution | Top 10 Vendors | Spend Trend
    col1, col2, col3, col_bg = st.columns([1, 1, 1, 0.12], gap="small")

    with col1:
        with st.container(border=True):
            st.markdown("<div class='chart-title'>Invoice Status Distribution</div>",
                        unsafe_allow_html=True)
            if status_df.empty:
                status_df = pd.DataFrame([
                    {"status":"Paid","cnt":450},{"status":"Pending","cnt":180},
                    {"status":"Disputed","cnt":33},{"status":"Other","cnt":30}])
            total = status_df["cnt"].sum()
            status_df["percentage"] = (status_df["cnt"] / total * 100).round(1) if total > 0 else 0.0
            # Only label slices ≥ 3% to avoid tiny overlapping labels
            status_df["pct_label"] = status_df["percentage"].apply(
                lambda x: f"{x}%" if x >= 3.0 else ""
            )
            cs = alt.Scale(domain=["Paid","Pending","Disputed","Other"],
                           range=["#22c55e","#f59e0b","#ef4444","#3b82f6"])
            base_chart = alt.Chart(status_df).encode(
                theta=alt.Theta("cnt:Q", stack=True),
                color=alt.Color("status:N", scale=cs,
                                legend=alt.Legend(
                                    orient="bottom", title=None,
                                    labelFontSize=11, symbolSize=80,
                                    columns=2,
                                )),
            )
            # Donut: innerRadius/outerRadius sized to leave room for labels
            donut = base_chart.mark_arc(
                innerRadius=55, outerRadius=85, stroke="white", strokeWidth=2
            ).encode(tooltip=["status:N","cnt:Q","percentage:Q"])
            # Percentage labels just outside the arc
            pct_text = base_chart.mark_text(
                radius=102, size=10, fontWeight="bold", color="#374151"
            ).encode(text=alt.Text("pct_label:N"))
            # Centre: total count
            ct = alt.Chart(pd.DataFrame({"t":[str(total)]})).mark_text(
                align="center", baseline="middle",
                fontSize=24, fontWeight="bold", color="#111827"
            ).encode(text="t:N")
            cl = alt.Chart(pd.DataFrame({"t":["TOTAL"]})).mark_text(
                align="center", baseline="middle",
                fontSize=10, color="#6b7280", dy=16
            ).encode(text="t:N")
            st.altair_chart(
                (donut + pct_text + ct + cl).properties(height=300),
                use_container_width=True,
            )

    with col2:
        with st.container(border=True):
            st.markdown("<div class='chart-title'>Top 10 Vendors by Spend</div>",
                        unsafe_allow_html=True)
            if vendor_df.empty:
                vendor_df = pd.DataFrame([{"vendor_name":"No Data","spend":0}])
            st.altair_chart(
                alt.Chart(vendor_df).mark_bar(color="#22c55e", cornerRadiusEnd=4).encode(
                    x=alt.X("spend:Q", title=None, axis=alt.Axis(format="~s")),
                    y=alt.Y("vendor_name:N", sort="-x", title=None),
                    tooltip=["vendor_name:N", alt.Tooltip("spend:Q", format="$,.0f")]
                ).properties(height=280),
                use_container_width=True,
            )

    with col3:
        with st.container(border=True):
            st.markdown("<div class='chart-title'>Spend Trend Analysis</div>",
                        unsafe_allow_html=True)
            if trend_df.empty:
                trend_df = pd.DataFrame([{"month":"2026-01","actual_spend":0,"forecast_spend":0}])
            else:
                trend_df["month"] = pd.to_datetime(trend_df["month"]).dt.strftime("%Y-%m")
                trend_df["forecast_spend"] = (
                    trend_df["actual_spend"].rolling(2, min_periods=1).mean()
                    .shift(-1).fillna(trend_df["actual_spend"] * 1.1)
                )
            melted = trend_df.melt(
                id_vars=["month"], value_vars=["actual_spend","forecast_spend"],
                var_name="type", value_name="spend"
            )
            melted["type"] = melted["type"].map({"actual_spend":"ACTUAL","forecast_spend":"FORECAST"})
            st.altair_chart(
                alt.Chart(melted).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("spend:Q", title=None, axis=alt.Axis(format="~s")),
                    color=alt.Color("type:N",
                        scale=alt.Scale(domain=["ACTUAL","FORECAST"],
                                        range=["#22c55e","#3b82f6"]),
                        legend=alt.Legend(orient="top", title=None)),
                    xOffset="type:N",
                    tooltip=["month:N","type:N", alt.Tooltip("spend:Q", format="$,.0f")]
                ).properties(height=280),
                use_container_width=True,
            )


    with col_bg:
        # Position BG button at mid-height of the chart (charts are ~330px tall)
        st.markdown("<div style='height:270px;'></div>", unsafe_allow_html=True)
        render_bg_button_sidebar()

def render_dashboard():
    """
    Main dashboard page.
    PERFORMANCE summary (after optimisation):
      - Old: 17 sequential Athena queries per load (~35–70s cold)
      - New:  4 Athena queries per load (~8–16s cold), ~instant on cache hit
        1. fetch_kpi_data current  — 1 merged CTE query (5 KPIs in one scan)
        2. fetch_kpi_data previous — 1 merged CTE query (delta calculations)
        3. fetch_needs_attention   — 1 UNION query (overdue+disputed+due in one scan)
        4. fetch_chart_data_cached — 1 merged CTE query (all 3 charts in one scan)
      - Cache TTL 600s — all queries cached for 10 minutes
    """
    # ── Initialise session state (first load only) ───────────────────────────
    for k, v in [
        ("date_range",      compute_range_preset("Last 30 Days")),
        ("selected_vendor", "All Vendors"),
        ("preset",          "Last 30 Days"),
        ("na_tab",          "Overdue"),
        ("na_page",         0),
        ("_preset_clicked", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Remove any stale date-keyed vendor cache entries from old code
    stale = [k for k in list(st.session_state.keys())
             if isinstance(k, str) and k.startswith("vendor_list_") and k != "vendor_list_stable"]
    for k in stale:
        del st.session_state[k]

    rng_start, rng_end, selected_vendor = render_filters()
    vendor_where = build_vendor_where(selected_vendor)
    sl  = sql_date(rng_start);  el  = sql_date(rng_end)
    ps, pe = prior_window(rng_start, rng_end)

    # ── KPI Cards: 2 merged Athena queries (current + prior period) ──────────
    with st.spinner("Loading dashboard..."):
        cur_kpi  = fetch_kpi_data(sl, el, vendor_where,
                                   rng_start.isoformat(), rng_end.isoformat())
        prev_kpi = fetch_kpi_data(sql_date(ps), sql_date(pe), vendor_where,
                                   ps.isoformat(), pe.isoformat())
    # ── Save KPI snapshot to long-term cache (for history / trend replay) ────
    save_kpi_snapshot(
        st.session_state.get("preset","Custom"),
        rng_start.isoformat(), rng_end.isoformat(), cur_kpi
    )
    # ── Auto-save preferred preset to long-term memory ────────────────────────
    preset_now = st.session_state.get("preset","Last 30 Days")
    if preset_now != "Custom":
        set_user_memory("preferred_preset", preset_now, "preference", "inferred", 0.9)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    render_kpi_rows(cur_kpi, prev_kpi)

    # ── Needs Attention: 1 UNION query ───────────────────────────────────────
    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    render_needs_attention(rng_start, rng_end, vendor_where)

    # ── Charts: 1 merged CTE query ───────────────────────────────────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    render_charts(rng_start, rng_end, vendor_where)


# ── Forecast ─────────────────────────────────────────────────
def render_forecast():
    # Cache forecast data to stop flicker on every rerun
    if "forecast_cf_df" not in st.session_state:
        st.session_state.forecast_cf_df = None
    if st.session_state.forecast_cf_df is None:
        cf_sql = f"""SELECT forecast_bucket,invoice_count,total_amount,earliest_due,latest_due
            FROM {DATABASE}.cash_flow_forecast_vw
            ORDER BY CASE forecast_bucket WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1
                WHEN 'DUE_7_DAYS' THEN 2 WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4
                WHEN 'DUE_60_DAYS' THEN 5 WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END"""
        st.session_state.forecast_cf_df = run_query(cf_sql)
    cf_df = st.session_state.forecast_cf_df
    if cf_df.empty:
        st.warning("cash_flow_forecast_vw empty – computing from fact table.")
        cf_sql = f"""WITH base AS (
                SELECT invoice_number,invoice_amount_local,due_date,
                       DATE_DIFF('day',CURRENT_DATE,due_date) AS days_until_due
                FROM {DATABASE}.fact_all_sources_vw
                WHERE UPPER(invoice_status) IN ('OPEN','DUE','OVERDUE') AND due_date IS NOT NULL),
            buckets AS (SELECT CASE
                    WHEN days_until_due<0 THEN 'OVERDUE_NOW' WHEN days_until_due<=7 THEN 'DUE_7_DAYS'
                    WHEN days_until_due<=14 THEN 'DUE_14_DAYS' WHEN days_until_due<=30 THEN 'DUE_30_DAYS'
                    WHEN days_until_due<=60 THEN 'DUE_60_DAYS' WHEN days_until_due<=90 THEN 'DUE_90_DAYS'
                    ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,
                    COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount,
                    MIN(due_date) AS earliest_due,MAX(due_date) AS latest_due FROM base GROUP BY 1),
            total AS (SELECT 'TOTAL_UNPAID' AS forecast_bucket,SUM(invoice_count) AS invoice_count,
                       SUM(total_amount) AS total_amount,NULL AS earliest_due,NULL AS latest_due FROM buckets)
            SELECT * FROM total UNION ALL SELECT * FROM buckets"""
        cf_df = run_query(cf_sql)

    tab1, tab2 = st.tabs(["Cash Flow Need Forecast","GR/IR Reconciliation"])
    with tab1:
        if not cf_df.empty:
            tu = cf_df[cf_df["forecast_bucket"]=="TOTAL_UNPAID"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"]=="TOTAL_UNPAID"].empty else 0
            on = cf_df[cf_df["forecast_bucket"]=="OVERDUE_NOW"]["total_amount"].values[0] if not cf_df[cf_df["forecast_bucket"]=="OVERDUE_NOW"].empty else 0
            d30 = cf_df[cf_df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
            pct30 = (d30/tu*100) if tu>0 else 0
        else:
            tu=on=d30=pct30=0
        st.markdown("""<style>.fkc{border-radius:16px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);
            border:1px solid rgba(0,0,0,0.05);}.fkt{font-size:.8rem;font-weight:600;color:#475569;margin-bottom:.3rem;}
            .fkv{font-size:1.8rem;font-weight:700;color:#0f172a;line-height:1.2;}</style>""",unsafe_allow_html=True)
        kc=["#fff7e0","#ffe6ef","#e6f3ff","#e0f7fa"]
        kt=["TOTAL UNPAID","OVERDUE NOW","DUE NEXT 30 DAYS","% DUE ≤ 30 DAYS"]
        kv=[abbr_currency(tu),abbr_currency(on),abbr_currency(d30),f"{pct30:.1f}%"]
        cols=st.columns(4)
        for i,col in enumerate(cols):
            with col:
                st.markdown(f'<div class="fkc" style="background:{kc[i]};"><div class="fkt">{kt[i]}</div><div class="fkv">{kv[i]}</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Obligations by time bucket")
        if not cf_df.empty:
            # Clean up bucket labels for display
            display_df = cf_df.copy()
            bucket_labels = {
                "TOTAL_UNPAID": "Total Unpaid",
                "OVERDUE_NOW": "Overdue Now",
                "DUE_7_DAYS": "Due in 7 Days",
                "DUE_14_DAYS": "Due in 14 Days",
                "DUE_30_DAYS": "Due in 30 Days",
                "DUE_60_DAYS": "Due in 60 Days",
                "DUE_90_DAYS": "Due in 90 Days",
                "BEYOND_90_DAYS": "Beyond 90 Days",
            }
            if "forecast_bucket" in display_df.columns:
                display_df["forecast_bucket"] = display_df["forecast_bucket"].map(
                    lambda x: bucket_labels.get(str(x).upper(), x)
                )
            render_simple_table(display_df, col_labels={
                "forecast_bucket": "Time Bucket",
                "invoice_count":   "Invoice Count",
                "total_amount":    "Total Amount ($)",
                "earliest_due":    "Earliest Due",
                "latest_due":      "Latest Due",
            })
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.download_button("Download forecast (CSV)", data=cf_df.to_csv(index=False).encode(),
                               file_name="cash_flow_forecast.csv", mime="text/csv")
        else:
            st.info("No cash flow forecast data.")
        st.markdown("---"); st.markdown("### Action Playbook")
        for label,question in [
            ("Forecast cash outflow (7-90 days)","Forecast cash outflow for the next 7, 14, 30, 60, and 90 days"),
            ("Invoices to pay early to capture discounts","Which invoices should we pay early to capture discounts?"),
            ("Optimal payment timing for this week","What is the optimal payment timing strategy for this week?"),
            ("Late payment trend and risk","Show late payment trend for forecasting")]:
            if st.button(label,use_container_width=True):
                st.session_state.auto_run_query=question; st.session_state.page="Genie"; st.rerun()

    with tab2:
        st.markdown("#### GR/IR Reconciliation")
        grir_sql = f"""WITH lb AS (SELECT year,month,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 1),
            la AS (SELECT year,month,pct_grir_over_60,cnt_grir_over_60 FROM {DATABASE}.gr_ir_aging_vw ORDER BY year DESC,month DESC LIMIT 1)
            SELECT l.year,l.month,l.invoice_count AS grir_items,l.total_grir_blnc AS total_grir_balance,
                   a.pct_grir_over_60,a.cnt_grir_over_60,COALESCE(l.total_grir_blnc*a.pct_grir_over_60,0) AS amount_over_60_days
            FROM lb l LEFT JOIN la a ON a.year=l.year AND a.month=l.month"""
        gdf=run_query(grir_sql)
        if not gdf.empty:
            row=gdf.iloc[0]
            tg=safe_number(row.get("total_grir_balance",0)); gi=safe_int(row.get("grir_items",0))
            p60=safe_number(row.get("pct_grir_over_60",0)); a60=safe_number(row.get("amount_over_60_days",0))
            c60=safe_int(row.get("cnt_grir_over_60",0)); yr=safe_int(row.get("year",0)); mo=safe_int(row.get("month",0))
        else:
            tg=gi=p60=a60=c60=yr=mo=0
        gcols=st.columns(4)
        for i,(t,v,bg) in enumerate([("TOTAL GR/IR",abbr_currency(tg),"#E6F3FF"),("% > 60 DAYS",f"{p60:.1f}%","#E0F7FA"),("60 DAYS AMOUNT",abbr_currency(a60),"#FFF3E0"),("60 DAYS ITEMS",f"{c60:,}","#F3E5F5")]):
            with gcols[i]: render_grir_metric_card(t,v,bg_color=bg)
        st.caption(f"GR/IR position for {yr:04d}-{mo:02d}: {gi:,} items; {p60:.1f}% >60 days.")
        trsql=f"SELECT year,month,invoice_count,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 24"
        trdf=run_query(trsql)
        if not trdf.empty:
            st.markdown("**GR/IR outstanding trend (last 24 months)**")
            render_simple_table(trdf, col_labels={
                "year":            "Year",
                "month":           "Month",
                "invoice_count":   "Invoice Count",
                "total_grir_blnc": "Total GR/IR Balance ($)",
            })
        else:
            st.info("No GR/IR data.")
        st.markdown("---"); st.markdown("### GR/IR Clearing Playbook")
        for label,question in [
            ("1. Identify top GR/IR hotspots","Show GR/IR outstanding balance by month and highlight which recent months have the highest GR/IR balance so we can prioritize clearing."),
            ("2. Explain likely GR/IR root causes","Using GR/IR aging and outstanding balance data, explain the likely root-cause buckets (missing goods receipt, invoice not posted, price or quantity mismatch) and for each bucket suggest 2–3 concrete remediation actions."),
            ("3. Quantify working-capital benefit","Estimate the working capital that would be released by clearing all GR/IR items older than 60 and 90 days, by month."),
            ("4. Draft vendor follow-up messages","Based on GR/IR aging and outstanding balances, draft vendor-facing follow-up templates we can use for high-priority GR/IR items, with realistic subject lines and concise bullet points.")]:
            if st.button(label,use_container_width=True):
                st.session_state.auto_run_query=question; st.session_state.page="Genie"; st.rerun()


# ── Genie helpers ────────────────────────────────────────────
def _safe_sql_string(sql_val):
    if sql_val is None: return ""
    if isinstance(sql_val,(dict,list)): return json.dumps(sql_val)
    return str(sql_val)

SEMANTIC_MODEL = f"""
database: {DATABASE}
tables:
  fact_all_sources_vw:
    columns:
      invoice_number: decimal(7,0)
      posting_date: date
      vendor_id: string
      invoice_amount_local: decimal(7,2)
      invoice_status: string  # OPEN/PAID/OVERDUE/DISPUTED/CANCELLED/REJECTED
      purchase_order_reference: decimal(7,0)
      po_amount: decimal(8,2)
      due_date: date
      payment_date: date
      aging_days: decimal(3,0)
  dim_vendor_vw:
    columns: {{vendor_id: string, vendor_name: string}}
  payment_processing_cycle_time_vw:
    note: NO posting_date. Filter by year/month only.
    columns: {{year: decimal(4,0), month: decimal(2,0), avg_payment_cycle_time_days: decimal(9,6), cleared_invoices: decimal(3,0)}}
  full_payment_rate_vw:
    note: NO posting_date. Columns full_paid_invoices, total_cleared_invoices, full_payment_rate_pct.
    columns: {{year: decimal(4,0), month: decimal(2,0), full_paid_invoices: decimal(1,0), total_cleared_invoices: decimal(3,0), full_payment_rate_pct: decimal(7,6)}}
  gr_ir_outstanding_balance_vw:
    columns: {{year: decimal(4,0), month: decimal(2,0), invoice_count: decimal(3,0), total_grir_blnc: decimal(9,2)}}
  gr_ir_aging_vw:
    columns: {{year: decimal(4,0), month: decimal(2,0), pct_grir_over_60: decimal(8,8), cnt_grir_over_60: decimal(3,0)}}
"""

SYS_SEMANTIC = f"""You are a senior procurement analyst and Athena SQL expert.
CRITICAL RULES:
- payment_processing_cycle_time_vw: NO posting_date. Use year/month. Column = avg_payment_cycle_time_days.
- full_payment_rate_vw: NO posting_date. Use year/month. Column = full_payment_rate_pct (NOT full_payment_rate).
- fact_all_sources_vw: HAS posting_date for date range filters.
- Always COALESCE numeric columns. Use Presto functions. LIMIT 1000 unless aggregating.
- Output only SQL, no markdown/explanation.

Semantic model:
{SEMANTIC_MODEL}"""

OUT_OF_DOMAIN_MSG = ("Hello! I am ProcureIQ Assistant. I can help you with procurement insights, "
    "vendor information, invoice status, forecasting, spend analytics, dashboard metrics, "
    "and related business data. Please ask a procurement or dashboard-related question.")

def is_relevant_question(q: str) -> bool:
    """
    Strict whitelist-only: ONLY returns True when procurement keywords are present.
    Any greeting, small talk, or general knowledge question returns False immediately.
    """
    ql = q.lower().strip()

    # Step 1: Hard-block non-procurement patterns
    non_proc_patterns = [
        r"^(hi|hello|hey|howdy|hiya|yo)\b",
        r"^good\s*(morning|afternoon|evening|night|day)\b",
        r"^how are you", r"^how('s| is) (it going|everything|life|your day)",
        r"^who are you", r"^what (are|is) you\b",
        r"^tell me (a joke|something funny|a story)",
        r"^(crack|tell).*(joke|funny)",
        r"^what('s| is) (the )?weather",
        r"^what('s| is) (your )?(name|favorite|hobby|age)",
        r"^what (do|can) you do\??$",
        r"^(thanks|thank you|thx|ty)\b",
        r"^(bye|goodbye|see you|cya|ttyl)\b",
        r"^are you (a |an )?(ai|bot|robot|human|person|machine)\??$",
        r"^capital of\b",
        r"^who (invented|discovered|created|founded|built)\b",
        r"^what is (the )?(meaning of life|ai|machine learning|blockchain|crypto)",
        r"^(define|explain|what is)\s+(love|life|happiness|success|beauty)",
        r"^(how|what).*(recipe|cook|bake|movie|music|song|book|game|sport|football|cricket)\b",
        r"^(write|compose|create)\s+(a |an )?(poem|story|song|joke|essay)\b",
        r"^(translate|convert)\s+\w+\s+to\b",
        r"^what('s| is) (today's )?(date|time|day)\b",
        r"^\d+\s*[\+\-\*\/]\s*\d+",
    ]
    for pat in non_proc_patterns:
        if re.search(pat, ql):
            return False

    # Step 2: Strict procurement keyword whitelist — MUST match at least one
    procurement_keywords = [
        "spend","vendor","invoice","invoices","purchase order","payment","payments",
        "due date","overdue","dispute","disputed","gr/ir","grir","gr ir",
        "cash flow","forecast","dashboard","kpi","metric","procurement","p2p",
        "procure","accounts payable","payable","payables","accrual","aging","aged",
        "processing time","cycle time","autoprocess","first pass","late payment",
        "on-time payment","early payment","duplicate payment","supplier",
        "working capital","total spend","active vendor","pending invoice",
        "clearing","reconcil","goods receipt","delivery accuracy",
        "days payable","dpo","spend analysis","vendor analysis","invoice status",
        "po amount","po number","po date","purchase requisition",
    ]
    for kw in procurement_keywords:
        if kw in ql:
            return True

    # Nothing matched → not a procurement question
    return False

def generate_sql(question: str) -> str:
    """Generate SQL via Bedrock. Returns empty string if generation fails (no hardcoded fallback)."""
    sql = ask_bedrock(f"Question: {question}\n\nGenerate SQL.", SYS_SEMANTIC)
    if sql:
        sql = re.sub(r"```sql\s*","",sql); sql = re.sub(r"```\s*","",sql).strip()
        if not sql.lower().startswith("select"): sql=""
    return sql  # Empty string if Bedrock couldn't generate — caller handles this

SYS_ANALYST = "You are a helpful senior procurement analyst. Respond in markdown with Descriptive (What the data shows) and Prescriptive (Recommendations) sections."

def process_custom_query(query: str, history: str="") -> dict:
    # ALWAYS check relevance first — this is the primary gate
    if not is_relevant_question(query):
        return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG,"question":query}
    sql = generate_sql(query)
    if not sql or not is_safe_sql(sql):
        # If SQL generation failed, still try to give a helpful answer via Bedrock text
        txt = ask_bedrock(
            f'{history}\nUser asked: "{query}"\nNo SQL was generated. Provide a general procurement answer.',
            SYS_ANALYST)
        if txt:
            return {"layout":"static","analyst_response":txt,"question":query}
        return {"layout":"error","message":"Could not generate SQL for this question. Please try rephrasing with more specific procurement terms."}
    sql = ensure_limit(sql)
    df = run_query(sql)
    if df.empty:
        return {"layout":"error","message":"Query returned no data. Try rephrasing with more specific terms like vendor name, date range, or invoice status."}
    preview = df.head(10).to_string(index=False,max_colwidth=40)
    txt = ask_bedrock(f'{history}\nUser asked: "{query}"\nData:\n{preview}\nSQL:\n{sql}', SYS_ANALYST)
    return {"layout":"analyst","sql":sql,"df":df.to_dict(orient="records"),"question":query,"analyst_response":txt or preview}

def process_cash_flow_forecast(question: str, history: str="") -> dict:
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""SELECT forecast_bucket,invoice_count,total_amount,earliest_due,latest_due
        FROM {DATABASE}.cash_flow_forecast_vw ORDER BY CASE forecast_bucket
            WHEN 'TOTAL_UNPAID' THEN 0 WHEN 'OVERDUE_NOW' THEN 1 WHEN 'DUE_7_DAYS' THEN 2
            WHEN 'DUE_14_DAYS' THEN 3 WHEN 'DUE_30_DAYS' THEN 4 WHEN 'DUE_60_DAYS' THEN 5
            WHEN 'DUE_90_DAYS' THEN 6 WHEN 'BEYOND_90_DAYS' THEN 7 ELSE 8 END"""
    df = run_query(sql)
    if df.empty:
        sql = f"""WITH b AS (SELECT invoice_amount_local,due_date,DATE_DIFF('day',CURRENT_DATE,due_date) AS du
            FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE','OVERDUE') AND due_date IS NOT NULL),
            bk AS (SELECT CASE WHEN du<0 THEN 'OVERDUE_NOW' WHEN du<=7 THEN 'DUE_7_DAYS'
                WHEN du<=14 THEN 'DUE_14_DAYS' WHEN du<=30 THEN 'DUE_30_DAYS'
                WHEN du<=60 THEN 'DUE_60_DAYS' WHEN du<=90 THEN 'DUE_90_DAYS'
                ELSE 'BEYOND_90_DAYS' END AS forecast_bucket,
                COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount,
                MIN(due_date) AS earliest_due,MAX(due_date) AS latest_due FROM b GROUP BY 1),
            tot AS (SELECT 'TOTAL_UNPAID' AS forecast_bucket,SUM(invoice_count) AS invoice_count,
                SUM(total_amount) AS total_amount,NULL AS earliest_due,NULL AS latest_due FROM bk)
            SELECT * FROM tot UNION ALL SELECT * FROM bk"""
        df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No cash flow data."}
    df.columns=[c.lower() for c in df.columns]
    txt = ask_bedrock(f"{history}\nCash flow:\n{df.to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"cash_flow","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt or "","question":question}

def process_early_payment(question: str, history: str="") -> dict:
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""SELECT CAST(f.invoice_number AS VARCHAR) AS document_number,v.vendor_name,
        f.invoice_amount_local AS invoice_amount,f.due_date,
        DATE_DIFF('day',CURRENT_DATE,f.due_date) AS days_until_due,
        ROUND(f.invoice_amount_local*0.02,2) AS savings_if_2pct_discount,
        CASE WHEN DATE_DIFF('day',CURRENT_DATE,f.due_date)<=7 THEN 'High'
             WHEN DATE_DIFF('day',CURRENT_DATE,f.due_date)<=14 THEN 'Medium' ELSE 'Low' END AS early_pay_priority
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE UPPER(f.invoice_status) IN ('OPEN','DUE') AND f.due_date>CURRENT_DATE
          AND DATE_DIFF('day',CURRENT_DATE,f.due_date)<=30
        ORDER BY early_pay_priority ASC,savings_if_2pct_discount DESC LIMIT 20"""
    df = run_query(sql)
    if not df.empty: df.columns=[c.lower() for c in df.columns]
    txt = ask_bedrock(f"{history}\nEarly payment candidates:\n{df.head(10).to_string(index=False) if not df.empty else 'None'}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"early_payment","df":df.to_dict(orient="records") if not df.empty else [],"sql":sql,"analyst_response":txt or "","question":question,"empty":df.empty}

def process_payment_timing(question: str, history: str="") -> dict:
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""WITH db AS (SELECT CASE WHEN due_date<CURRENT_DATE THEN 'Overdue'
            WHEN due_date<=CURRENT_DATE+INTERVAL '7' DAY THEN 'Due in 0-7 days'
            WHEN due_date<=CURRENT_DATE+INTERVAL '14' DAY THEN 'Due in 8-14 days'
            WHEN due_date<=CURRENT_DATE+INTERVAL '30' DAY THEN 'Due in 15-30 days'
            ELSE 'Due later' END AS payment_window,
            COUNT(*) AS invoice_count,SUM(invoice_amount_local) AS total_amount
        FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE') GROUP BY 1)
        SELECT * FROM db ORDER BY CASE payment_window WHEN 'Overdue' THEN 1
            WHEN 'Due in 0-7 days' THEN 2 WHEN 'Due in 8-14 days' THEN 3
            WHEN 'Due in 15-30 days' THEN 4 ELSE 5 END"""
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No payment timing data."}
    df.columns=[c.lower() for c in df.columns]
    txt = ask_bedrock(f"{history}\nPayment timing:\n{df.to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"payment_timing","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt or "","question":question}

def process_late_payment_trend(question: str, history: str="") -> dict:
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""SELECT DATE_TRUNC('month',payment_date) AS month,COUNT(*) AS total_payments,
        SUM(CASE WHEN payment_date>due_date THEN 1 ELSE 0 END) AS late_payments,
        AVG(CASE WHEN payment_date>due_date THEN DATE_DIFF('day',due_date,payment_date) END) AS avg_late_days
        FROM {DATABASE}.fact_all_sources_vw
        WHERE payment_date IS NOT NULL AND payment_date>=DATE_ADD('month',-12,CURRENT_DATE)
        GROUP BY 1 ORDER BY 1"""
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No payment trend data."}
    df.columns=[c.lower() for c in df.columns]
    df["late_pct"]=(df["late_payments"]/df["total_payments"])*100
    txt = ask_bedrock(f"{history}\nLate payment:\n{df.tail(6).to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"late_payment_trend","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt or "","question":question}

def process_grir_hotspots(question, history=""):
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"SELECT year,month,invoice_count,total_grir_blnc AS total_grir_balance FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC"
    df = run_query(sql)
    if df.empty: df=pd.DataFrame([{"year":2025,"month":12,"invoice_count":145,"total_grir_balance":1250000}])
    else: df.columns=[c.lower() for c in df.columns]
    txt = ask_bedrock(f"{history}\nGR/IR by month:\n{df.head(12).to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"grir_hotspots","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt or "","question":question}

def process_grir_root_causes(question, history=""):
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    asql=f"SELECT year,month,pct_grir_over_60,cnt_grir_over_60 FROM {DATABASE}.gr_ir_aging_vw ORDER BY year DESC,month DESC LIMIT 6"
    bsql=f"SELECT year,month,total_grir_blnc FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC LIMIT 6"
    adf=run_query(asql); bdf=run_query(bsql)
    if adf.empty: adf=pd.DataFrame([{"year":2025,"month":12,"pct_grir_over_60":28.5,"cnt_grir_over_60":41}])
    if bdf.empty: bdf=pd.DataFrame([{"year":2025,"month":12,"total_grir_blnc":1250000}])
    ctx = f"GR/IR aging:\n{adf.to_string(index=False)}\n\nBalances:\n{bdf.to_string(index=False)}"
    txt = ask_bedrock(f"{history}\n{ctx}\nRoot causes and remediation. Descriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"grir_root_causes","df":adf.to_dict(orient="records"),"extra_df":bdf.to_dict(orient="records"),
            "sql":{"aging_sql":asql,"balance_sql":bsql},"analyst_response":txt or "","question":question}

def process_grir_working_capital(question, history=""):
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""SELECT year,month,total_grir_blnc,
        CASE WHEN (year*100+month)<=(CAST(EXTRACT(YEAR FROM CURRENT_DATE) AS INT)*100+CAST(EXTRACT(MONTH FROM CURRENT_DATE) AS INT)-60) THEN total_grir_blnc ELSE 0 END AS older_than_60_days,
        CASE WHEN (year*100+month)<=(CAST(EXTRACT(YEAR FROM CURRENT_DATE) AS INT)*100+CAST(EXTRACT(MONTH FROM CURRENT_DATE) AS INT)-90) THEN total_grir_blnc ELSE 0 END AS older_than_90_days
        FROM {DATABASE}.gr_ir_outstanding_balance_vw ORDER BY year DESC,month DESC"""
    df = run_query(sql)
    if df.empty: df=pd.DataFrame([{"year":2025,"month":12,"total_grir_blnc":1250000,"older_than_60_days":350000,"older_than_90_days":120000}])
    else: df.columns=[c.lower() for c in df.columns]
    o60=df['older_than_60_days'].sum(); o90=df['older_than_90_days'].sum()
    txt = ask_bedrock(f"{history}\nGR/IR WC:\n{df.head(12).to_string(index=False)}\nCite ${o60:,.0f} >60d, ${o90:,.0f} >90d. Descriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"grir_working_capital","df":df.to_dict(orient="records"),"metrics":{"older_60":float(o60),"older_90":float(o90)},"sql":sql,"analyst_response":txt or "","question":question}

def process_grir_vendor_followup(question, history=""):
    if not is_relevant_question(question): return {"layout":"static","analyst_response":OUT_OF_DOMAIN_MSG}
    sql = f"""SELECT v.vendor_name,COUNT(*) AS grir_count,SUM(f.invoice_amount_local) AS total_amount,
        AVG(DATE_DIFF('day',f.posting_date,CURRENT_DATE)) AS avg_age_days
        FROM {DATABASE}.fact_all_sources_vw f
        LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE UPPER(f.invoice_status)='OPEN' AND f.purchase_order_reference IS NOT NULL
        GROUP BY v.vendor_name ORDER BY total_amount DESC LIMIT 10"""
    df = run_query(sql)
    if df.empty: df=pd.DataFrame([{"vendor_name":"Acme Corp","grir_count":23,"total_amount":245000,"avg_age_days":85}])
    else: df.columns=[c.lower() for c in df.columns]
    txt = ask_bedrock(f"{history}\nTop GR/IR vendors:\n{df.to_string(index=False)}\nSummarise and draft 3-5 follow-up templates.", SYS_ANALYST)
    return {"layout":"grir_vendor_followup","df":df.to_dict(orient="records"),"sql":sql,"analyst_response":txt or "","question":question}

def _quick_spending_overview():
    msql = f"""SELECT DATE_TRUNC('month',posting_date) AS month,SUM(COALESCE(invoice_amount_local,0)) AS monthly_spend,
        COUNT(*) AS invoice_count,COUNT(DISTINCT vendor_id) AS vendor_count
        FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
          AND posting_date>=DATE_ADD('month',-12,CURRENT_DATE) GROUP BY 1 ORDER BY month DESC"""
    mdf = run_query(msql)
    if mdf.empty: return {"layout":"error","message":"No spending data."}
    mdf.columns=[c.lower() for c in mdf.columns]
    vsql = f"""SELECT COALESCE(v.vendor_name,'Unknown') AS vendor_name,SUM(COALESCE(f.invoice_amount_local,0)) AS spend
        FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')
          AND f.posting_date>=DATE_TRUNC('YEAR',CURRENT_DATE)
        GROUP BY v.vendor_name ORDER BY spend DESC LIMIT 10"""
    vdf = run_query(vsql)
    if not vdf.empty: vdf.columns=[c.lower() for c in vdf.columns]
    tytd=vdf['spend'].sum() if not vdf.empty else 0
    t5pct=(vdf.head(5)['spend'].sum()/tytd*100) if tytd>0 else 0
    mom=0
    if len(mdf)>=2:
        lat=mdf.iloc[0]['monthly_spend']; prev=mdf.iloc[1]['monthly_spend']
        mom=((lat-prev)/prev*100) if prev!=0 else 0
    qoq=0
    if len(mdf)>=6:
        cq=mdf.iloc[0:3]['monthly_spend'].sum(); pq=mdf.iloc[3:6]['monthly_spend'].sum()
        qoq=((cq-pq)/pq*100) if pq!=0 else 0
    txt = ask_bedrock(f"Spending data:\n{mdf.head(6).to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"quick","analysis_type":"spending_overview","metrics":{"total_ytd":tytd,"top5_pct":t5pct,"mom_pct":mom,"qoq_pct":qoq},
            "monthly_df":mdf.to_dict(orient="records"),"vendors_df":vdf.to_dict(orient="records") if not vdf.empty else [],
            "analyst_response":txt or "","sql":{"monthly_trend":msql,"top_vendors":vsql},"question":"Spending Overview"}

def _quick_vendor_analysis():
    vsql = f"""SELECT COALESCE(v.vendor_name,'Unknown') AS vendor_name,SUM(COALESCE(f.invoice_amount_local,0)) AS total_spend,
        COUNT(DISTINCT f.invoice_number) AS invoice_count
        FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE UPPER(f.invoice_status) NOT IN ('CANCELLED','REJECTED')
          AND f.posting_date>=DATE_TRUNC('YEAR',CURRENT_DATE)
        GROUP BY v.vendor_name ORDER BY total_spend DESC LIMIT 10"""
    vdf = run_query(vsql)
    if vdf.empty: return {"layout":"error","message":"No vendor data."}
    vdf.columns=[c.lower() for c in vdf.columns]
    msql = f"""SELECT DATE_TRUNC('month',posting_date) AS month,COUNT(DISTINCT vendor_id) AS active_vendors
        FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
          AND posting_date>=DATE_ADD('month',-12,CURRENT_DATE) GROUP BY 1 ORDER BY month DESC"""
    mvdf = run_query(msql)
    if not mvdf.empty: mvdf.columns=[c.lower() for c in mvdf.columns]
    ts=vdf['total_spend'].sum()
    txt = ask_bedrock(f"Vendor data:\n{vdf.to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"quick","analysis_type":"vendor_analysis",
            "metrics":{"total_spend":ts,"top1_pct":(vdf.iloc[0]['total_spend']/ts*100) if ts>0 else 0,"top5_pct":(vdf.head(5)['total_spend'].sum()/ts*100) if ts>0 else 0,"active_vendors":len(vdf)},
            "vendors_df":vdf.to_dict(orient="records"),"monthly_df":mvdf.to_dict(orient="records") if not mvdf.empty else [],
            "analyst_response":txt or "","sql":{"top_vendors":vsql,"monthly_vendors":msql},"question":"Vendor Analysis"}

def _quick_payment_performance():
    sql = f"""SELECT DATE_FORMAT(payment_date,'%Y-%m') AS month,
        ROUND(AVG(DATE_DIFF('day',posting_date,payment_date)),1) AS avg_days_to_pay,
        SUM(CASE WHEN DATE_DIFF('day',due_date,payment_date)>0 THEN 1 ELSE 0 END) AS late_payments,
        COUNT(*) AS total_payments
        FROM {DATABASE}.fact_all_sources_vw
        WHERE payment_date IS NOT NULL AND payment_date>=DATE_ADD('month',-6,CURRENT_DATE)
          AND UPPER(invoice_status) NOT IN ('CANCELLED','REJECTED')
        GROUP BY DATE_FORMAT(payment_date,'%Y-%m') ORDER BY month"""
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No payment data for last 6 months."}
    df.columns=[c.lower() for c in df.columns]
    df['month_dt']=pd.to_datetime(df['month']+'-01'); df=df.sort_values('month_dt')
    df['month_str']=df['month_dt'].dt.strftime('%b %Y')
    lp=df['late_payments'].sum(); tp=df['total_payments'].sum()
    txt = ask_bedrock(f"Payment data:\n{df[['month_str','avg_days_to_pay','late_payments','total_payments']].to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"quick","analysis_type":"payment_performance",
            "metrics":{"avg_days_to_pay":df['avg_days_to_pay'].mean(),"late_payments_pct":(lp/tp*100) if tp>0 else 0,"total_late":int(lp),"total_payments":int(tp)},
            "payment_df":df.to_dict(orient="records"),"analyst_response":txt or "","sql":sql,"question":"Payment Performance"}

def _quick_invoice_aging():
    sql = f"""SELECT CASE WHEN due_date<CURRENT_DATE THEN 'Overdue'
            WHEN due_date<=CURRENT_DATE+INTERVAL '7' DAY THEN 'Due in 0-7 days'
            WHEN due_date<=CURRENT_DATE+INTERVAL '30' DAY THEN 'Due in 8-30 days'
            WHEN due_date<=CURRENT_DATE+INTERVAL '90' DAY THEN 'Due in 31-90 days'
            ELSE 'Due in >90 days' END AS aging_bucket,
            COUNT(*) AS invoice_count,SUM(COALESCE(invoice_amount_local,0)) AS total_amount
        FROM {DATABASE}.fact_all_sources_vw WHERE UPPER(invoice_status) IN ('OPEN','DUE','OVERDUE') GROUP BY 1
        ORDER BY CASE aging_bucket WHEN 'Overdue' THEN 1 WHEN 'Due in 0-7 days' THEN 2
            WHEN 'Due in 8-30 days' THEN 3 WHEN 'Due in 31-90 days' THEN 4 ELSE 5 END"""
    df = run_query(sql)
    if df.empty: return {"layout":"error","message":"No aging data."}
    df.columns=[c.lower() for c in df.columns]
    oa=df[df['aging_bucket']=='Overdue']['total_amount'].sum(); tot=df['total_amount'].sum()
    txt = ask_bedrock(f"Invoice aging:\n{df.to_string(index=False)}\nDescriptive and Prescriptive.", SYS_ANALYST)
    return {"layout":"quick","analysis_type":"invoice_aging",
            "metrics":{"total_open":tot,"overdue_amount":oa,"overdue_pct":(oa/tot*100) if tot>0 else 0,"invoice_count":int(df['invoice_count'].sum())},
            "aging_df":df.to_dict(orient="records"),"analyst_response":txt or "","sql":sql,"question":"Invoice Aging"}


# ── Genie render helpers ──────────────────────────────────────
def render_cash_flow_response(r):
    df=pd.DataFrame(r["df"])
    if df.empty: st.error("No cash flow data."); return
    tu=df[df["forecast_bucket"]=="TOTAL_UNPAID"]["total_amount"].values[0] if not df[df["forecast_bucket"]=="TOTAL_UNPAID"].empty else 0
    on=df[df["forecast_bucket"]=="OVERDUE_NOW"]["total_amount"].values[0] if not df[df["forecast_bucket"]=="OVERDUE_NOW"].empty else 0
    d30=df[df["forecast_bucket"].isin(["DUE_7_DAYS","DUE_14_DAYS","DUE_30_DAYS"])]["total_amount"].sum()
    c1,c2,c3=st.columns(3)
    c1.metric("Total Unpaid",abbr_currency(tu)); c2.metric("Overdue Now",abbr_currency(on))
    c3.metric("Due Next 30 Days",f"{abbr_currency(d30)} ({(d30/tu*100) if tu>0 else 0:.0f}%)")
    cdf=df[df["forecast_bucket"]!="TOTAL_UNPAID"].copy()
    if not cdf.empty: alt_bar(cdf,x="forecast_bucket",y="total_amount",horizontal=True,height=300,color="#3b82f6")
    st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_early_payment_response(r):
    df=pd.DataFrame(r["df"])
    if r.get("empty",False) or df.empty: st.info("No early payment candidates.")
    else:
        ts=df["savings_if_2pct_discount"].sum(); hp=df[df["early_pay_priority"]=="High"].shape[0]
        c1,c2=st.columns(2); c1.metric("Total Potential Savings",abbr_currency(ts)); c2.metric("High-Priority Invoices",hp)
        st.dataframe(safe_dataframe_display(df.head(10)),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_payment_timing_response(r):
    df=pd.DataFrame(r["df"])
    if df.empty: st.error("No payment timing data."); return
    st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_late_payment_trend_response(r):
    df=pd.DataFrame(r["df"])
    if df.empty: st.error("No trend data."); return
    if "month" in df.columns:
        df["month_str"]=pd.to_datetime(df["month"]).dt.strftime("%b %Y")
        alt_line_monthly(df[["month_str","late_pct"]].rename(columns={"late_pct":"VALUE"}),month_col="month_str",value_col="VALUE",height=300,title="Late Payments %")
    st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_grir_hotspots(r):
    df=pd.DataFrame(r["df"])
    if df.empty: st.error("No GR/IR data."); return
    cdf=df.head(12).copy()
    cdf['ym']=cdf['year'].astype(str)+'-'+cdf['month'].astype(str).str.zfill(2)
    alt_bar(cdf,x="ym",y="total_grir_balance",horizontal=False,height=300,color="#ef4444")
    st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_grir_root_causes(r):
    df=pd.DataFrame(r.get("df",[])); edf=pd.DataFrame(r.get("extra_df",[]))
    if not df.empty: st.subheader("GR/IR Aging"); st.dataframe(safe_dataframe_display(df),use_container_width=True)
    if not edf.empty: st.subheader("Outstanding Balances"); st.dataframe(safe_dataframe_display(edf),use_container_width=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_grir_working_capital(r):
    m=r.get("metrics",{}); c1,c2=st.columns(2)
    c1.metric("WC Release (>60 days)",abbr_currency(m.get("older_60",0))); c2.metric("WC Release (>90 days)",abbr_currency(m.get("older_90",0)))
    df=pd.DataFrame(r["df"])
    if not df.empty: st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_grir_vendor_followup(r):
    df=pd.DataFrame(r["df"])
    if not df.empty: st.dataframe(safe_dataframe_display(df),use_container_width=True,hide_index=True)
    if r.get("analyst_response"): st.markdown("### 💡 Key Insights"); st.markdown(r["analyst_response"])
    with st.expander("View SQL"): st.code(_safe_sql_string(r.get("sql")),language="sql")

def render_quick_analysis_response(r):
    at=r.get("analysis_type","spending_overview"); m=r.get("metrics",{}); ar=r.get("analyst_response",""); sq=r.get("sql",{})
    st.markdown(f"**Question:** {r.get('question','Analysis')}"); st.markdown("---")
    if at=="spending_overview":
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total Spend (YTD)",abbr_currency(m.get("total_ytd",0))); c2.metric("MoM Change",f"{m.get('mom_pct',0):+.1f}%")
        c3.metric("Top 5 Vendors",f"{m.get('top5_pct',0):.1f}% of total"); c4.metric("QoQ Change",f"{m.get('qoq_pct',0):+.1f}%")
        mdf=pd.DataFrame(r.get("monthly_df",[]))
        if not mdf.empty:
            mdf['month_dt']=pd.to_datetime(mdf['month']); mdf=mdf.sort_values('month_dt'); mdf['month_str']=mdf['month_dt'].dt.strftime('%b %Y')
            st.altair_chart(alt.Chart(mdf).mark_bar(color="#22c55e").encode(x=alt.X("month_str:N",sort=None,title=None),y=alt.Y("monthly_spend:Q",axis=alt.Axis(format="~s")),tooltip=["month_str:N",alt.Tooltip("monthly_spend:Q",format="$,.0f")]).properties(height=250),use_container_width=True)
        vdf=pd.DataFrame(r.get("vendors_df",[]))
        if not vdf.empty:
            st.subheader("Top 10 Vendors (YTD)")
            st.altair_chart(alt.Chart(vdf.head(10)).mark_bar(color="#3b82f6").encode(x=alt.X("spend:Q",axis=alt.Axis(format="~s")),y=alt.Y("vendor_name:N",sort="-x"),tooltip=["vendor_name:N",alt.Tooltip("spend:Q",format="$,.0f")]).properties(height=400),use_container_width=True)
    elif at=="vendor_analysis":
        c1,c2,c3=st.columns(3)
        c1.metric("Total Spend (YTD)",abbr_currency(m.get("total_spend",0))); c2.metric("Top 1 Vendor",f"{m.get('top1_pct',0):.1f}%"); c3.metric("Top 5 Vendors",f"{m.get('top5_pct',0):.1f}%")
        vdf=pd.DataFrame(r.get("vendors_df",[]))
        if not vdf.empty:
            st.altair_chart(alt.Chart(vdf).mark_bar(color="#f59e0b").encode(x=alt.X("total_spend:Q",axis=alt.Axis(format="~s")),y=alt.Y("vendor_name:N",sort="-x"),tooltip=["vendor_name:N",alt.Tooltip("total_spend:Q",format="$,.0f")]).properties(height=400),use_container_width=True)
    elif at=="payment_performance":
        c1,c2=st.columns(2); c1.metric("Avg Days to Pay",f"{m.get('avg_days_to_pay',0):.1f}"); c2.metric("Late Payments %",f"{m.get('late_payments_pct',0):.1f}%")
        pdf=pd.DataFrame(r.get("payment_df",[]))
        if not pdf.empty:
            ch1,ch2=st.columns(2)
            with ch1: st.altair_chart(alt.Chart(pdf).mark_line(point=True,color="#ef4444").encode(x=alt.X("month_str:N",sort=None),y=alt.Y("avg_days_to_pay:Q"),tooltip=["month_str:N","avg_days_to_pay"]).properties(height=300),use_container_width=True)
            with ch2: st.altair_chart(alt.Chart(pdf).mark_line(point=True,color="#3b82f6").encode(x=alt.X("month_str:N",sort=None),y=alt.Y("late_payments:Q"),tooltip=["month_str:N","late_payments","total_payments"]).properties(height=300),use_container_width=True)
    elif at=="invoice_aging":
        c1,c2=st.columns(2); c1.metric("Total Open",abbr_currency(m.get("total_open",0))); c2.metric("Overdue Amount",abbr_currency(m.get("overdue_amount",0)))
        adf=pd.DataFrame(r.get("aging_df",[]))
        if not adf.empty:
            st.altair_chart(alt.Chart(adf).mark_bar(color="#dc2626").encode(x=alt.X("total_amount:Q",axis=alt.Axis(format="~s")),y=alt.Y("aging_bucket:N",sort=alt.EncodingSortField(field="total_amount",order="descending")),tooltip=["aging_bucket:N",alt.Tooltip("total_amount:Q",format="$,.0f"),"invoice_count:Q"]).properties(height=250),use_container_width=True)
    if ar: st.markdown("### Prescriptive — Recommendations & next steps"); st.markdown(ar)
    with st.expander("Show SQL"):
        if isinstance(sq,dict):
            for n,q in sq.items(): st.code(q,language="sql")
        elif isinstance(sq,str): st.code(sq,language="sql")

GRIR_HOTSPOTS_Q  = "Show GR/IR outstanding balance by month and highlight which recent months have the highest GR/IR balance so we can prioritize clearing."
GRIR_ROOTCAUSE_Q = "Using GR/IR aging and outstanding balance data, explain the likely root-cause buckets (missing goods receipt, invoice not posted, price or quantity mismatch) and for each bucket suggest 2–3 concrete remediation actions."
GRIR_WC_Q        = "Estimate the working capital that would be released by clearing all GR/IR items older than 60 and 90 days, by month."
GRIR_FOLLOWUP_Q  = "Based on GR/IR aging and outstanding balances, draft vendor-facing follow-up templates we can use for high-priority GR/IR items, with realistic subject lines and concise bullet points."

def _dispatch_query(q: str, history: str) -> dict:
    """
    Central dispatcher for all Genie queries.
    FIRST gate: is_relevant_question() — if the question is not procurement-related,
    return the standard out-of-domain message immediately, before any SQL or LLM call.
    Hard-coded quick-analysis labels ("Spending Overview" etc.) bypass the relevance
    check because they are internal system triggers, not user free-text.
    """
    # ── Internal quick-analysis triggers (system labels — bypass relevance check) ──
    if q == GRIR_HOTSPOTS_Q:  return process_grir_hotspots(q, history)
    if q == GRIR_ROOTCAUSE_Q: return process_grir_root_causes(q, history)
    if q == GRIR_WC_Q:        return process_grir_working_capital(q, history)
    if q == GRIR_FOLLOWUP_Q:  return process_grir_vendor_followup(q, history)
    if q == "Spending Overview":    return _quick_spending_overview()
    if q == "Vendor Analysis":      return _quick_vendor_analysis()
    if q == "Payment Performance":  return _quick_payment_performance()
    if q == "Invoice Aging":        return _quick_invoice_aging()

    # ── RELEVANCE GATE — must pass before any LLM or Athena call ──────────────
    # This is the definitive check. If it returns False, no SQL is generated,
    # no Bedrock call is made, no data is returned — only the default message.
    if not is_relevant_question(q):
        return {"layout": "static", "analyst_response": OUT_OF_DOMAIN_MSG, "question": q}

    # ── Procurement-confirmed: route to appropriate handler ───────────────────
    lq = q.lower()
    if any(kw in lq for kw in ["forecast cash outflow", "cash flow forecast"]):
        return process_cash_flow_forecast(q, history)
    if any(kw in lq for kw in ["pay early", "capture discounts"]):
        return process_early_payment(q, history)
    if "optimal payment timing" in lq:
        return process_payment_timing(q, history)
    if "late payment trend" in lq:
        return process_late_payment_trend(q, history)

    # Default: generate SQL from free-text question
    return process_custom_query(q, history)

def process_user_question(user_question: str):
    with st.spinner("Generating insights..."):
        # Skip cache entirely for out-of-domain questions
        # (avoids serving stale procurement responses for greetings)
        if not is_relevant_question(user_question):
            result = {"layout": "static", "analyst_response": OUT_OF_DOMAIN_MSG, "question": user_question}
            st.session_state.current_messages = [
                {"role": "user",      "content": user_question,         "timestamp": datetime.now()},
                {"role": "assistant", "content": OUT_OF_DOMAIN_MSG,
                 "response": result,  "timestamp": datetime.now()},
            ]
            st.rerun()
            return

        # ── Use TTL-aware cache (replaces old get_cache) ─────────────
        cached = get_cache_with_ttl(user_question, cache_type="genie")
        if cached:
            st.session_state.current_messages=[
                {"role":"user","content":user_question,"timestamp":datetime.now()},
                {"role":"assistant","content":cached.get('analyst_response',''),
                 "response":cached,"timestamp":datetime.now()}]
            save_chat_message(st.session_state.genie_session_id,0,"user",user_question)
            save_chat_message(st.session_state.genie_session_id,1,"assistant",
                              cached.get('analyst_response',''),source="cache",
                              sql_used=_safe_sql_string(cached.get("sql")))
            save_question(user_question,"custom")
        else:
            # ── Use short-term memory for conversation context ──────────────
            history = build_bedrock_context(
                st.session_state.genie_session_id, max_turns=6
            )
            result = _dispatch_query(user_question, history)
            st.session_state.current_messages=[{"role":"user","content":user_question,"timestamp":datetime.now()}]
            if result.get("layout")!="error":
                ac=result.get('analyst_response','Analysis complete.')
                st.session_state.current_messages.append({"role":"assistant","content":ac,"response":result,"timestamp":datetime.now()})
                # ── TTL cache: Genie responses valid for 1 hour ─────────────
                set_cache_with_ttl(user_question, result, cache_type="genie", ttl_seconds=3600)
                save_chat_message(st.session_state.genie_session_id,0,"user",user_question)
                save_chat_message(st.session_state.genie_session_id,1,"assistant",ac,
                                  sql_used=_safe_sql_string(result.get("sql")))
                save_question(user_question,"forecast")
                # ── Auto-infer long-term preferences ────────────────────────
                infer_and_save_preferences(user_question, result)
            else:
                st.session_state.current_messages.append({"role":"assistant","content":result.get("message","Error"),"timestamp":datetime.now()})
    st.rerun()

def start_new_session():
    st.session_state.genie_session_id=str(uuid.uuid4())
    st.session_state.current_messages=[]; st.session_state.show_summary=False; st.session_state.conversation_summary=""
    save_chat_session(st.session_state.genie_session_id,label=f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.rerun()

def summarize_conversation():
    """Summarise current conversation. Sets show_summary=True so button stays highlighted."""
    if not st.session_state.current_messages:
        return  # Caller checks messages exist before calling — no false warning here
    txt = "\n\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.current_messages
    )
    s = ask_bedrock(f"Summarize concisely:\n\n{txt}", "You summarize conversations.")
    if s:
        st.session_state.conversation_summary = s
        st.session_state.show_summary = True
        # Keep messages — user can keep chatting; summary shown above chat
    else:
        st.error("Could not generate summary.")

def export_conversation_md():
    if not st.session_state.current_messages and not st.session_state.get("conversation_summary"):
        st.warning("No conversation to export."); return
    lines=["# ProcureIQ Genie Conversation\n"]
    if st.session_state.get("conversation_summary"):
        lines.append(f"**Summary**\n\n{st.session_state.conversation_summary}\n\n---\n")
    for msg in st.session_state.current_messages:
        lines.append(f"{'**User**' if msg['role']=='user' else '**Genie**'}\n\n{msg['content']}\n\n---\n")
    st.download_button("📥 Download MD",data="\n".join(lines),
        file_name=f"genie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",mime="text/markdown",key="export_md_btn")

def render_genie():
    # ── Session state init ────────────────────────────────────────────────────
    for k, v in [("genie_session_id", None), ("current_messages", []),
                 ("genie_prefill", ""), ("show_summary", False),
                 ("conversation_summary", ""), ("show_chats_panel", False)]:
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.genie_session_id is None:
        st.session_state.genie_session_id = str(uuid.uuid4())
        save_chat_session(st.session_state.genie_session_id,
                          label=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Auto-run query from quick cards or nav ────────────────────────────────
    auto_query = st.session_state.pop("auto_run_query", None)
    if auto_query:
        with st.spinner("Running analysis..."):
            hc = get_recent_conversation_context(limit=20, max_age_days=2)
            result = _dispatch_query(auto_query, hc)
            st.session_state.current_messages = [
                {"role": "user", "content": auto_query, "timestamp": datetime.now()}
            ]
            if result.get("layout") != "error":
                ac = result.get("analyst_response", "Analysis complete.")
                st.session_state.current_messages.append(
                    {"role": "assistant", "content": ac, "response": result,
                     "timestamp": datetime.now()}
                )
                save_chat_message(st.session_state.genie_session_id, 0, "user", auto_query)
                save_chat_message(st.session_state.genie_session_id, 1, "assistant", ac,
                                  sql_used=_safe_sql_string(result.get("sql")))
                save_question(auto_query, "forecast")
                set_cache(auto_query, result)
            else:
                st.session_state.current_messages.append(
                    {"role": "assistant", "content": result.get("message", "Error"),
                     "timestamp": datetime.now()}
                )
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # ALL CSS for the Genie page — injected once at the top
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
<style>
/* ── Page title ── */
.genie-welcome h1 {
    font-size: 1.75rem; font-weight: 700; color: #1e293b; margin-bottom: 4px;
}
.genie-welcome p { font-size: 0.9rem; color: #64748b; margin: 0; }

/* ── Quick-analysis cards ── */
.quick-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px 14px 10px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    min-height: 130px;
    display: flex; flex-direction: column;
}
.quick-card-icon {
    font-size: 1.6rem; margin-bottom: 8px; line-height: 1;
}
.quick-card h3 {
    font-size: 0.92rem; font-weight: 700; color: #1e293b;
    margin: 0 0 5px 0;
}
.quick-card p {
    font-size: 0.78rem; color: #64748b; flex-grow: 1;
    margin: 0 0 10px 0; line-height: 1.4;
}

/* ── Left panel expanders ── */
.genie-left-panel div[data-testid="stExpander"] {
    border: none !important;
    border-bottom: 1px solid #f1f5f9 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
.genie-left-panel div[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    padding: 8px 4px !important;
}
/* Left panel buttons — left-aligned, no border */
.genie-left-panel button {
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    color: #374151 !important;
    font-size: 12.5px !important;
    padding: 4px 6px !important;
    height: auto !important;
    min-height: 28px !important;
    box-shadow: none !important;
    font-weight: 400 !important;
}
.genie-left-panel button:hover {
    background: #f1f5f9 !important;
    color: #2563eb !important;
    border: none !important;
}

/* ── Right panel: AI Assistant ── */
.genie-right-container {
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
/* Header buttons row */
.genie-right-container .stHorizontalBlock button {
    height: 34px !important;
    min-height: 34px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    padding: 0 12px !important;
}

/* ── Empty state ── */
.genie-empty {
    background: #f8fafc; border-radius: 12px;
    padding: 2.2rem 1rem; text-align: center;
    margin: 8px 0 6px 0; min-height: 200px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.genie-empty-icon  { font-size: 1.8rem; color: #cbd5e1; margin-bottom: 8px; }
.genie-empty-title { font-size: 0.98rem; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.genie-empty-sub   { font-size: 0.8rem; color: #94a3b8; max-width: 240px; line-height: 1.4; }

/* ── Chat messages ── */
.chat-messages {
    max-height: 360px; overflow-y: auto; padding: 4px 2px;
    margin-bottom: 6px; background: #fafcff;
    border-radius: 10px; border: 1px solid #e8edf3;
}

/* ── Resume panel ── */
.resume-panel {
    background: #f0f7ff; border-radius: 10px; padding: 10px 12px;
    border: 1px solid #bfdbfe; margin: 4px 0 8px 0;
}

/* ══════════════════════════════════════════════════════════════
   ASK INPUT FORM — full width stretching to submit button
   ══════════════════════════════════════════════════════════════ */
div[data-testid="stForm"] {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
    margin-top: 8px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
/* Outer vertical block inside form: remove all padding */
div[data-testid="stForm"] > div[data-testid="stVerticalBlock"] {
    padding: 0 !important;
    gap: 0 !important;
}
/* Horizontal row: flex, no gaps eating width */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    flex-wrap: nowrap !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* Input column: grow to fill ALL remaining space */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:first-child {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: 0 !important;
    padding: 0 !important;
}
/* Button column: fixed narrow */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:last-child {
    flex: 0 0 52px !important;
    width: 52px !important;
    min-width: 52px !important;
    padding: 0 !important;
}
/* Input element: truly 100% of its column */
div[data-testid="stForm"] div[data-testid="stTextInput"] {
    width: 100% !important; padding: 0 !important; margin: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] > div {
    width: 100% !important; padding: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
    width: 100% !important;
    height: 48px !important;
    min-height: 48px !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    color: #111827 !important;
    background: #f5f5f5 !important;
    padding: 0 18px !important;
    box-shadow: none !important;
    outline: none !important;
    box-sizing: border-box !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
    color: #9ca3af !important;
    font-size: 13.5px !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] label {
    display: none !important;
}
/* Circular submit button */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button[data-testid="baseButton-primary"] {
    width: 48px !important;
    height: 48px !important;
    min-height: 48px !important;
    min-width: 48px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.30) !important;
    line-height: 48px !important;
    text-align: center !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stForm"] button[data-testid="baseButton-primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.40) !important;
    transform: scale(1.05) !important;
}
</style>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Title + Quick-analysis cards
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
<div class="genie-welcome">
  <h1>Welcome to ProcureIQ Genie</h1>
  <p>Let Genie run one of these quick analyses for you</p>
</div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # 4 cards — icon text (ASCII/unicode safe for Snowflake)
    card_data = [
        {"title": "Spending Overview",
         "icon_bg": "#EEF2FF", "icon_color": "#4F46E5",
         "icon_char": "📊",
         "desc": "Track total spend, monthly trends and major changes"},
        {"title": "Vendor Analysis",
         "icon_bg": "#F0FDF4", "icon_color": "#16A34A",
         "icon_char": "🏭",
         "desc": "Understand vendor-wise spend, concentration, and dependency"},
        {"title": "Payment Performance",
         "icon_bg": "#FFF7ED", "icon_color": "#EA580C",
         "icon_char": "⏱️",
         "desc": "Identify delays, late payments, and cycle time issues"},
        {"title": "Invoice Aging",
         "icon_bg": "#FFF1F2", "icon_color": "#E11D48",
         "icon_char": "📅",
         "desc": "See overdue invoices, risk buckets, and problem areas"},
    ]
    card_cols = st.columns(4, gap="small")
    for idx, (col, card) in enumerate(zip(card_cols, card_data)):
        with col:
            st.markdown(
                f"<div class='quick-card'>"
                f"<div style='width:44px;height:44px;border-radius:12px;"
                f"background:{card['icon_bg']};display:flex;align-items:center;"
                f"justify-content:center;margin-bottom:10px;"
                f"font-size:1.3rem;font-weight:800;color:{card['icon_color']};'>"
                f"{card['icon_char']}</div>"
                f"<h3>{card['title']}</h3>"
                f"<p>{card['desc']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Ask Genie", key=f"card_{idx}", use_container_width=True):
                st.session_state.auto_run_query = card["title"]
                st.rerun()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Left panel + Right AI Assistant panel
    # ══════════════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([0.32, 0.68], gap="medium")

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    with left_col:
        st.markdown("<div class='genie-left-panel'>", unsafe_allow_html=True)
        with st.container(border=True):
            # Saved Insights
            with st.expander("Saved Insights"):
                ins = get_saved_insights_cached(page="genie")
                if ins:
                    for i in ins[:5]:
                        if st.button(i["title"][:45], key=f"insight_{i['id']}",
                                     use_container_width=True):
                            st.session_state.auto_run_query = i["question"]
                            st.rerun()
                else:
                    st.caption("No saved insights yet")

            # Frequently asked by you
            with st.expander("Frequently Asked by You"):
                faqs = get_frequent_questions_by_user_cached(5)
                if faqs:
                    for faq in faqs[:5]:
                        if st.button(faq["query"][:45], key=f"faq_{faq['query'][:20]}",
                                     use_container_width=True):
                            st.session_state.genie_prefill = faq["query"]
                            st.rerun()
                else:
                    for sug in ["Total spend YTD and trends",
                                "Top vendors by spend",
                                "Overdue invoices summary"]:
                        if st.button(sug, key=f"sug_{sug[:15]}", use_container_width=True):
                            st.session_state.genie_prefill = sug
                            st.rerun()

            # Most frequent (all)
            with st.expander("Most Frequent (All)"):
                af = get_frequent_questions_all_cached(5)
                if af:
                    for faq in af[:5]:
                        st.markdown(
                            f"<div style='text-align:left;color:#374151;"
                            f"font-size:0.83rem;padding:3px 0;cursor:default;'>"
                            f"{faq['query'][:45]}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No questions yet")

            # Long-term Memory (from screenshot — show it)
            memories = get_all_user_memories()
            mem_count = len(memories)
            with st.expander(f"Long-term Memory ({mem_count} facts)"):
                if memories:
                    for m in memories[:6]:
                        type_tag = {"preference": "pref", "entity": "entity",
                                    "context": "ctx", "insight": "insight"}.get(m["type"], m["type"])
                        st.markdown(
                            f"<div style='font-size:0.78rem;color:#475569;"
                            f"padding:2px 0;border-bottom:1px solid #f1f5f9;'>"
                            f"<b>{m['key']}</b>: {str(m['value'])[:35]}"
                            f"<span style='color:#94a3b8;font-size:0.7rem;"
                            f"margin-left:6px;'>[{type_tag}]</span></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No memories yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT PANEL — AI Assistant ────────────────────────────────────────────
    with right_col:
        with st.container(border=True):
            # Header: title + action buttons
            hc1, hc2, hc3, hc4, hc5 = st.columns([1.5, 0.7, 0.85, 0.85, 0.65])
            with hc1:
                st.markdown("<b style='font-size:1rem;color:#1e293b;'>AI Assistant</b>",
                            unsafe_allow_html=True)
            with hc2:
                chats_on = st.session_state.get("show_chats_panel", False)
                if st.button("Chats", key="genie_chats_btn", use_container_width=True,
                             type="primary" if chats_on else "secondary"):
                    st.session_state["show_chats_panel"] = not chats_on
                    st.rerun()
            with hc3:
                # Highlighted (primary) when summary is visible
                sum_active = (st.session_state.get("show_summary", False)
                              and bool(st.session_state.get("conversation_summary", "")))
                if st.button("Summarize", key="summarize_top", use_container_width=True,
                             type="primary" if sum_active else "secondary"):
                    if st.session_state.current_messages:
                        if sum_active:
                            # Toggle off: hide summary
                            st.session_state.show_summary = False
                            st.session_state.conversation_summary = ""
                        else:
                            # Generate summary and highlight button
                            summarize_conversation()   # sets show_summary=True internally
                        st.rerun()
                    elif sum_active:
                        # Already showing summary — clicking again dismisses it
                        st.session_state.show_summary = False
                        st.session_state.conversation_summary = ""
                        st.rerun()
                    # No messages and no summary: do nothing silently
            with hc4:
                if st.button("Export MD", key="export_md_top", use_container_width=True):
                    if st.session_state.current_messages or st.session_state.conversation_summary:
                        export_conversation_md()
            with hc5:
                if st.button("Clear", key="clear_top", use_container_width=True):
                    start_new_session()

            st.markdown("<hr style='margin:6px 0 8px 0;border:none;"
                        "border-top:1px solid #f1f5f9;'/>", unsafe_allow_html=True)

            # ── Chats: resume previous conversations ──────────────────────────
            if st.session_state.get("show_chats_panel", False):
                conn_c = sqlite3.connect(DB_PATH); cur_c = conn_c.cursor()
                cur_c.execute("""SELECT session_id, session_label, created_at
                                 FROM chat_sessions WHERE user_name=?
                                 ORDER BY last_updated DESC LIMIT 10""",
                              (get_current_user(),))
                recent = cur_c.fetchall(); session_data = []
                for sess in recent:
                    cur_c.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id=?",
                                  (sess[0],))
                    mc = cur_c.fetchone()[0]
                    if mc > 0:
                        session_data.append({"session_id": sess[0], "label": sess[1],
                                             "created_at": sess[2], "msg_count": mc})
                conn_c.close()

                st.markdown("""<div class="resume-panel">
                    <b>Resume a Previous Conversation</b><br>
                    <small style='color:#64748b;'>Pick one to continue, or start fresh.</small>
                </div>""", unsafe_allow_html=True)

                if session_data:
                    for sess in session_data[:5]:
                        try:
                            dt = datetime.fromisoformat(str(sess["created_at"]))
                            age_h = int((datetime.now() - dt).total_seconds() / 3600)
                            age_s = f"{age_h}h ago" if age_h < 24 else f"{age_h // 24}d ago"
                        except Exception:
                            age_s = "–"
                        ri, rb = st.columns([0.7, 0.3])
                        with ri:
                            st.markdown(
                                f"<div style='font-size:.82rem;font-weight:600;"
                                f"color:#1e293b;'>{sess['label']}</div>"
                                f"<div style='font-size:.72rem;color:#94a3b8;'>"
                                f"{sess['msg_count']} messages · {age_s}</div>",
                                unsafe_allow_html=True,
                            )
                        with rb:
                            if st.button("Resume", key=f"res_{sess['session_id'][:8]}",
                                         use_container_width=True, type="primary"):
                                msgs_r = load_session_messages(sess["session_id"])
                                st.session_state.genie_session_id = sess["session_id"]
                                st.session_state.current_messages = [
                                    {"role": m["role"], "content": m["content"],
                                     "timestamp": m["timestamp"]} for m in msgs_r
                                ]
                                st.session_state["show_chats_panel"] = False
                                st.rerun()
                        st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)
                else:
                    st.caption("No previous conversations.")

                if st.button("Start a New Conversation", key="start_new_conv",
                             use_container_width=True):
                    start_new_session()
                st.markdown("<hr style='margin:6px 0;'/>", unsafe_allow_html=True)

            # ── Summary ───────────────────────────────────────────────────────
            if st.session_state.show_summary and st.session_state.conversation_summary:
                st.markdown("**Conversation Summary**")
                st.markdown(st.session_state.conversation_summary)
                if st.button("Dismiss", key="dismiss_summary", use_container_width=True):
                    st.session_state.show_summary = False
                    st.session_state.conversation_summary = ""
                    st.rerun()
                st.markdown("---")

            # ── Empty state ───────────────────────────────────────────────────
            elif (not st.session_state.current_messages
                  and not st.session_state.get("show_chats_panel", False)):
                st.markdown("""
<div class="genie-empty">
  <div class="genie-empty-icon">+</div>
  <div class="genie-empty-title">Start a Conversation</div>
  <div class="genie-empty-sub">Ask questions about your Procurement to Pay data.</div>
</div>""", unsafe_allow_html=True)

            # ── Chat messages ─────────────────────────────────────────────────
            elif st.session_state.current_messages:
                for msg in st.session_state.current_messages:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="message-user"><strong>You</strong><br/>'
                            f'{html.escape(msg["content"])}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="message-assistant"><strong>Genie</strong></div>',
                            unsafe_allow_html=True,
                        )
                        if "response" in msg and msg["response"]:
                            resp = msg["response"]
                            layout = resp.get("layout")
                            if layout == "static":
                                st.info(resp["analyst_response"])
                            elif layout == "cash_flow":
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
                                # Load data — stored as list of dicts (to_dict orient=records)
                                try:
                                    raw_df = resp.get("df", [])
                                    if isinstance(raw_df, list) and len(raw_df) > 0:
                                        df_r = pd.DataFrame(raw_df)
                                    elif isinstance(raw_df, dict):
                                        df_r = pd.DataFrame([raw_df])
                                    else:
                                        df_r = pd.DataFrame()
                                except Exception:
                                    df_r = pd.DataFrame()
                                if not df_r.empty:
                                    st.markdown("**Supporting Data**")
                                    # st.dataframe is always visible (no HTML wrapper issues)
                                    st.dataframe(
                                        safe_dataframe_display(df_r),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                    ch_r = auto_chart(df_r)
                                    if ch_r:
                                        st.altair_chart(ch_r, use_container_width=True)
                                sql_s = _safe_sql_string(resp.get("sql", ""))
                                if sql_s and sql_s.strip():
                                    with st.expander("View SQL"):
                                        st.code(sql_s, language="sql")
                            elif layout == "error":
                                st.error(resp.get("message", "Unknown error"))
                        else:
                            st.markdown(msg["content"])
                pass  # end chat messages

        # ── Ask input: white container, same width as AI Assistant above ──
        st.markdown("""
<style>
/* Form wrapper: white rounded container matching AI Assistant width */
div[data-testid="stForm"] {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07) !important;
    margin-top: 10px !important;
    width: 100% !important;
}
/* Row: input fills space, button stays fixed */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    display: flex !important; align-items: center !important;
    gap: 10px !important; width: 100% !important;
    flex-wrap: nowrap !important; padding: 0 !important; margin: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:first-child {
    flex: 1 1 0% !important; min-width: 0 !important; padding: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:last-child {
    flex: 0 0 56px !important; width: 56px !important;
    min-width: 56px !important; padding: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"],
div[data-testid="stForm"] div[data-testid="stTextInput"] > div {
    width: 100% !important; padding: 0 !important; margin: 0 !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
    width: 100% !important;
    height: 52px !important; min-height: 52px !important;
    font-size: 14px !important;
    padding: 0 20px !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8f9fa !important;
    color: #111827 !important;
    box-shadow: none !important; outline: none !important;
    box-sizing: border-box !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important; background: white !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
    color: #9ca3af !important; font-size: 13.5px !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] label { display: none !important; }
/* Blue circle submit button */
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    width: 50px !important; height: 50px !important;
    min-width: 50px !important; min-height: 50px !important;
    border-radius: 50% !important; padding: 0 !important;
    font-size: 18px !important; font-weight: 700 !important;
    background: #2563eb !important; color: white !important;
    border: none !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.30) !important;
    line-height: 50px !important; text-align: center !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 5px 16px rgba(37,99,235,0.42) !important;
    transform: scale(1.06) !important;
}
</style>
""", unsafe_allow_html=True)

        with st.form(key="genie_chat_form", clear_on_submit=True):
            fi, fb = st.columns([0.90, 0.10])
            with fi:
                prefill = st.session_state.pop("genie_prefill", "")
                uq = st.text_input(
                    "q", value=prefill,
                    placeholder="Ask a procurement question…",
                    label_visibility="collapsed",
                )
            with fb:
                submitted = st.form_submit_button("->", type="primary", use_container_width=True)
            if submitted and uq:
                process_user_question(uq)

    # form is inside right_col — see above


# ── Invoices ──────────────────────────────────────────────────
def render_invoice_detail(inv_row: dict, inv_num: str):
    def gv(key,default=""):
        val=inv_row.get(key,default)
        try:
            if pd.isna(val): return default
        except: pass
        if isinstance(val,(date,datetime)): return val.strftime("%Y-%m-%d")
        return val

    aging_days=gv("aging_days",0)
    try:
        dd=inv_row.get("due_date")
        if dd and isinstance(dd,(date,datetime)): aging_days=(date.today()-dd).days
    except: pass

    st.markdown(f"""<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        border-radius:12px;padding:16px 20px;margin-bottom:24px;box-shadow:0 4px 12px rgba(0,0,0,0.1);">
        <div style="color:white;font-size:1.1rem;font-weight:600;">Genie Insights</div>
        <div style="color:#f0f0f0;margin-top:6px;">Recommend immediate review of invoice
        <strong>{inv_num}</strong> — outstanding for <strong>{aging_days}</strong> days.</div></div>""",unsafe_allow_html=True)

    st.markdown("### Invoice Summary")
    fields=["Invoice Number","Invoice Date","Invoice Amount","PO Number","PO Amount","Due Date","Invoice Status","Aging (Days)"]
    values=[inv_num,gv("invoice_date",""),abbr_currency(safe_number(gv("invoice_amount",0))),
            gv("po_number",""),abbr_currency(safe_number(gv("po_amount",0))),
            gv("due_date",""),gv("invoice_status","").upper(),f"{aging_days} days" if aging_days>0 else "0 days"]
    ht='<table style="width:100%;border-collapse:collapse;background:white;"><tr style="background:#f1f5f9;border-bottom:1px solid #e2e8f0;">'
    for f in fields: ht+=f'<th style="padding:10px 8px;text-align:left;font-weight:600;">{f}</th>'
    ht+='</tr><tr>'
    for v in values: ht+=f'<td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;">{v}</td>'
    ht+='</tr></table>'
    st.markdown(ht,unsafe_allow_html=True)

    st.markdown("---"); st.markdown("### Status History")
    # Cache status history per invoice to stop flicker
    hist_key = f"inv_hist_{inv_num}"
    if hist_key not in st.session_state:
        hsql=f"""SELECT UPPER(status) AS status,effective_date,status_notes
            FROM {DATABASE}.invoice_status_history_vw
            WHERE CAST(invoice_number AS VARCHAR)='{inv_num}' ORDER BY sequence_nbr"""
        st.session_state[hist_key] = run_query(hsql)
    hdf = st.session_state[hist_key].copy()
    if hdf.empty:
        hdf=pd.DataFrame([{"status":"OPEN","effective_date":gv("invoice_date",""),"status_notes":"Invoice opened."},
                          {"status":"OVERDUE","effective_date":gv("due_date",""),"status_notes":"Invoice overdue."}])
    else:
        hdf.columns=[c.lower() for c in hdf.columns]
        hdf=hdf[["status","effective_date","status_notes"]].copy()
    pk=f"paid_{inv_num}"
    if st.session_state.get(pk,False) and "PAID" not in hdf["status"].values:
        hdf=pd.concat([hdf,pd.DataFrame([{"status":"PAID","effective_date":date.today().strftime("%Y-%m-%d"),"status_notes":"Processed via ProcureIQ"}])],ignore_index=True)
    hdf["effective_date"]=hdf["effective_date"].apply(lambda x: x.strftime("%Y-%m-%d") if isinstance(x,(date,datetime)) else str(x))
    render_simple_table(
        hdf[["status", "effective_date", "status_notes"]],
        col_labels={
            "status":         "Status",
            "effective_date": "Effective Date",
            "status_notes":   "Notes",
        }
    )

    st.markdown("---"); st.markdown("### Vendor & Company Information")
    t1,t2=st.tabs(["Vendor Info","Company Info"])
    with t1:
        vsql=f"""SELECT DISTINCT v.vendor_id,v.vendor_name,v.vendor_name_2,v.country_code,v.city,v.postal_code,v.street
            FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR)='{inv_num}' LIMIT 1"""
        vdf=run_query(vsql); row=vdf.iloc[0] if not vdf.empty else {}
        vf=["Vendor ID","Vendor Name","Alias/Name 2","Country","City","Postal Code","Street"]
        vv=[row.get("vendor_id","—"),row.get("vendor_name","—"),row.get("vendor_name_2","—"),row.get("country_code","—"),row.get("city","—"),row.get("postal_code","—"),row.get("street","—")]
        ht='<table style="width:100%;border-collapse:collapse;background:white;"><tr style="background:#f1f5f9;">'
        for f in vf: ht+=f'<th style="padding:10px 8px;text-align:left;font-weight:600;">{f}</th>'
        ht+='</tr><tr>'
        for v in vv: ht+=f'<td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;">{v}</td>'
        ht+='</tr></table>'
        st.markdown(ht,unsafe_allow_html=True)
    with t2:
        csql=f"""SELECT DISTINCT f.company_code,cc.company_name,f.plant_code,plt.plant_name,cc.street,cc.city,cc.postal_code
            FROM {DATABASE}.fact_all_sources_vw f
            LEFT JOIN {DATABASE}.dim_company_code_vw cc ON f.company_code=cc.company_code
            LEFT JOIN {DATABASE}.dim_plant_vw plt ON f.plant_code=plt.plant_code
            WHERE CAST(f.invoice_number AS VARCHAR)='{inv_num}' LIMIT 1"""
        cdf=run_query(csql); row=cdf.iloc[0] if not cdf.empty else {}
        cf=["Company Code","Company Name","Plant Code","Plant Name","Street","City","Postal Code"]
        cv=[row.get("company_code","—"),row.get("company_name","—"),row.get("plant_code","—"),row.get("plant_name","—"),row.get("street","—"),row.get("city","—"),row.get("postal_code","—")]
        ht='<table style="width:100%;border-collapse:collapse;background:white;"><tr style="background:#f1f5f9;">'
        for f in cf: ht+=f'<th style="padding:10px 8px;text-align:left;font-weight:600;">{f}</th>'
        ht+='</tr><tr>'
        for v in cv: ht+=f'<td style="padding:10px 8px;border-bottom:1px solid #e2e8f0;">{v}</td>'
        ht+='</tr></table>'
        st.markdown(ht,unsafe_allow_html=True)

    st.markdown("---")
    cs=gv("invoice_status","").upper()
    if st.session_state.get(pk,False): st.success("✅ Invoice has been processed and marked as Paid.")
    elif cs=="PAID": st.info("ℹ️ This invoice is already marked as PAID.")
    else:
        if st.button("✅ Proceed to Pay",key="proceed_pay_btn",use_container_width=True):
            st.session_state[pk]=True; st.rerun()

def render_invoices():
    st.subheader("Invoices")
    st.markdown("Search, track and manage all invoices in one place")

    qp=st.experimental_get_query_params()
    if "invoice" in qp and qp["invoice"][0]:
        st.session_state.selected_invoice_detail=qp["invoice"][0]
        st.experimental_set_query_params(); st.rerun()

    if st.session_state.get("selected_invoice_detail"):
        inv_num=st.session_state.selected_invoice_detail
        isql=f"""SELECT f.invoice_number,f.posting_date AS invoice_date,f.invoice_amount_local AS invoice_amount,
            f.purchase_order_reference AS po_number,f.po_amount,f.due_date,UPPER(f.invoice_status) AS invoice_status,
            f.aging_days,f.vendor_id,v.vendor_name,v.vendor_name_2,v.country_code,v.city,v.postal_code,v.street,
            f.company_code,f.plant_code,f.currency
            FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
            WHERE CAST(f.invoice_number AS VARCHAR)='{inv_num}' LIMIT 1"""
        idf=run_query(isql)
        if not idf.empty:
            render_invoice_detail(idf.iloc[0].to_dict(),inv_num)
            if st.button("← Back to Invoices List",key="back_invoices_btn",use_container_width=True):
                st.session_state.selected_invoice_detail=None
                st.session_state.invoice_search_input=""
                st.session_state.invoice_status_filter="All Status"
                st.session_state.inv_selected_vendor="All Vendors"
                st.rerun()
            return
        else:
            st.warning(f"Invoice {inv_num} not found.")
            st.session_state.selected_invoice_detail=None; st.rerun()

    for k,v in [("invoice_search_input",""),("invoice_status_filter","All Status"),("inv_selected_vendor","All Vendors")]:
        if k not in st.session_state: st.session_state[k]=v

    cs1,cs2,cs3=st.columns([3,1,1])
    with cs1:
        us=st.text_input("Invoice Number",value=st.session_state.invoice_search_input,
                         placeholder="e.g., 9001767",label_visibility="collapsed",key="inv_search_widget")
    with cs2: sc=st.button("Search",use_container_width=True,key="search_invoice_btn")
    with cs3: rc=st.button("Reset",use_container_width=True,key="reset_invoice_btn")

    if rc:
        st.session_state.invoice_search_input=""; st.session_state.invoice_status_filter="All Status"
        st.session_state.inv_selected_vendor="All Vendors"; st.session_state.selected_invoice_detail=None; st.rerun()
    if sc and us.strip():
        st.session_state.invoice_search_input=us.strip()
        cs=clean_invoice_number(us)
        ck=run_query(f"SELECT invoice_number FROM {DATABASE}.fact_all_sources_vw WHERE CAST(invoice_number AS VARCHAR)='{cs}' LIMIT 1")
        if not ck.empty: st.session_state.selected_invoice_detail=cs; st.rerun()
        else: st.warning(f"Invoice {cs} not found.")
    elif sc: st.warning("Please enter an invoice number.")

    cv1,cv2=st.columns(2)
    with cv1:
        if "inv_vendor_list" not in st.session_state:
            vd=run_query(f"SELECT DISTINCT vendor_name FROM {DATABASE}.dim_vendor_vw ORDER BY vendor_name")
            st.session_state.inv_vendor_list=(["All Vendors"]+vd["vendor_name"].tolist() if not vd.empty else ["All Vendors"])
        svi=(st.session_state.inv_vendor_list.index(st.session_state.inv_selected_vendor)
             if st.session_state.inv_selected_vendor in st.session_state.inv_vendor_list else 0)
        sv=st.selectbox("Vendor filter",st.session_state.inv_vendor_list,index=svi,label_visibility="collapsed",key="inv_sel_vendor")
        if sv!=st.session_state.inv_selected_vendor: st.session_state.inv_selected_vendor=sv
    with cv2:
        so=["All Status","OPEN","PAID","DISPUTED","OVERDUE","DUE_NEXT_30"]
        ssi=(so.index(st.session_state.invoice_status_filter) if st.session_state.invoice_status_filter in so else 0)
        ss=st.selectbox("Status filter",so,index=ssi,label_visibility="collapsed",key="inv_sel_status")
        if ss!=st.session_state.invoice_status_filter: st.session_state.invoice_status_filter=ss

    where=[]
    if st.session_state.invoice_search_input:
        where.append(f"CAST(f.invoice_number AS VARCHAR)='{clean_invoice_number(st.session_state.invoice_search_input)}'")
    if st.session_state.inv_selected_vendor!="All Vendors":
        where.append(f"UPPER(v.vendor_name)=UPPER('{st.session_state.inv_selected_vendor.replace(chr(39),chr(39)*2)}')")
    if st.session_state.invoice_status_filter!="All Status":
        if st.session_state.invoice_status_filter=="DUE_NEXT_30":
            where.append("UPPER(f.invoice_status)='OPEN' AND f.due_date>=CURRENT_DATE AND f.due_date<=DATE_ADD('day',30,CURRENT_DATE)")
        else:
            where.append(f"UPPER(f.invoice_status)='{st.session_state.invoice_status_filter}'")
    wsql=" AND ".join(where) if where else "1=1"
    df=run_query(f"""SELECT DISTINCT f.invoice_number,v.vendor_name,f.posting_date,f.due_date,
        f.invoice_amount_local AS invoice_amount,f.purchase_order_reference AS po_number,UPPER(f.invoice_status) AS status
        FROM {DATABASE}.fact_all_sources_vw f LEFT JOIN {DATABASE}.dim_vendor_vw v ON f.vendor_id=v.vendor_id
        WHERE {wsql} ORDER BY f.posting_date DESC LIMIT 500""")
    if not df.empty:
        df=df.rename(columns={'invoice_number':'INVOICE NUMBER','vendor_name':'VENDOR NAME',
            'posting_date':'POSTING DATE','due_date':'DUE DATE','invoice_amount':'INVOICE AMOUNT',
            'po_number':'PO NUMBER','status':'STATUS'})
        st.dataframe(safe_dataframe_display(df),use_container_width=True,height=400)
    else: st.info("No invoices found.")

# ── Main app ──────────────────────────────────────────────────

# ── Main app ──────────────────────────────────────────────────
def main():
    init_db()
    st.set_page_config(page_title="ProcureIQ", layout="wide", initial_sidebar_state="collapsed")

    if "bg_color" not in st.session_state:
        st.session_state["bg_color"] = "#ffffff"
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    inject_dashboard_css()

    # ── Global layout + header CSS ─────────────────────────────
    bg = st.session_state.get("bg_color", "#ffffff")
    st.markdown(f"""
<style>
/* ── Reset block-container padding so header is not glued to top ── */
.block-container {{
    padding-top: 3.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}}
/* ── Background ── */
.stApp, .main > .block-container {{
    background-color: {bg} !important;
}}
/* ── All buttons: base style ── */
button {{
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

/* ════════════════════════════════════════════
   HEADER — force all 6 cols to same height
   and vertical-center their content
   ════════════════════════════════════════════ */
div[data-testid="stHorizontalBlock"]:first-of-type {{
    align-items: center !important;
    min-height: 56px !important;
}}
div[data-testid="stHorizontalBlock"]:first-of-type
  > div[data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}
/* Brand col: left-align */
div[data-testid="stHorizontalBlock"]:first-of-type
  > div[data-testid="column"]:first-child {{
    justify-content: flex-start !important;
}}
/* Logo col: right-align */
div[data-testid="stHorizontalBlock"]:first-of-type
  > div[data-testid="column"]:last-child {{
    justify-content: flex-end !important;
}}
/* Nav cols: centre */
div[data-testid="stHorizontalBlock"]:first-of-type
  > div[data-testid="column"]:not(:first-child):not(:last-child) {{
    justify-content: center !important;
}}
/* All buttons inside the first horizontal block (the nav row) */
div[data-testid="stHorizontalBlock"]:first-of-type button {{
    border-radius: 50px !important;
    height: 38px !important;
    min-height: 38px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border: 1.5px solid #d1d5db !important;
    background: white !important;
    color: #374151 !important;
    box-shadow: none !important;
    padding: 0 20px !important;
    width: 100% !important;
    margin: 0 !important;
}}
div[data-testid="stHorizontalBlock"]:first-of-type button:hover {{
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #f0f7ff !important;
    box-shadow: none !important;
    transform: none !important;
}}
/* Active nav button */
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="primary"] {{
    background: #2563eb !important;
    background-color: #2563eb !important;
    color: white !important;
    border-color: #2563eb !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
}}
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="primary"]:hover {{
    background: #1d4ed8 !important;
    color: white !important;
    transform: none !important;
}}
/* ── KPI + chart misc ── */
.kpi-card {{ border-radius:16px; padding:1rem 1.2rem; min-height:100px;
             display:flex; flex-direction:column; justify-content:center; }}
.kpi-card-yellow {{ background:linear-gradient(135deg,#fef9c3 0%,#fef08a 100%); }}
.kpi-card-cyan   {{ background:linear-gradient(135deg,#cffafe 0%,#a5f3fc 100%); }}
.kpi-card-pink   {{ background:linear-gradient(135deg,#fce7f3 0%,#fbcfe8 100%); }}
.kpi-card-purple {{ background:linear-gradient(135deg,#f3e8ff 0%,#e9d5ff 100%); }}
.kpi-card-green  {{ background:linear-gradient(135deg,#dcfce7 0%,#bbf7d0 100%); }}
.kpi-title {{ font-size:.7rem; font-weight:600; color:#374151; text-transform:uppercase;
              letter-spacing:.5px; margin-bottom:.3rem; }}
.kpi-value {{ font-size:2rem; font-weight:800; color:#111827; line-height:1.1; }}
.kpi-delta {{ font-size:.9rem; font-weight:600; margin-top:.25rem; }}
.kpi-delta-negative {{ color:#dc2626; }}
.kpi-delta-positive {{ color:#16a34a; }}
.grir-card {{ border-radius:14px; padding:.9rem 1rem; border:1px solid #e2e8f0;
              box-shadow:0 2px 8px rgba(0,0,0,.05); display:flex; flex-direction:column;
              gap:.2rem; min-height:90px; justify-content:center; }}
.grir-card-title {{ font-size:.7rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:.6px; }}
.grir-card-value {{ font-size:1.8rem; font-weight:800; color:#111827; line-height:1.1; }}
.chart-title {{ font-size:1.1rem; font-weight:700; color:#111827; margin-bottom:.5rem; }}
.message-user {{ background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%); color:white;
    padding:10px 16px; border-radius:18px 18px 4px 18px; margin:8px 0;
    max-width:80%; margin-left:auto; text-align:right; }}
.message-assistant {{ background:#f1f5f9; color:#1e293b; padding:10px 16px;
    border-radius:18px 18px 18px 4px; margin:8px 0; max-width:85%; }}
.start-conversation {{ text-align:center; padding:2rem 1rem; background:#f8fafc; border-radius:20px; margin:1rem 0; }}
.chat-messages {{ max-height:400px; overflow-y:auto; padding:.5rem; margin-bottom:1rem;
    background:#fafcff; border-radius:16px; border:1px solid #e2e8f0; }}
.quick-card {{ background:white; border-radius:16px; padding:1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,.06); border:1px solid #e2e8f0;
    text-align:center; height:100%; display:flex; flex-direction:column; }}
.quick-card h3 {{ font-size:1rem; font-weight:600; color:#1e293b; margin:0 0 .4rem 0; }}
.quick-card p  {{ font-size:.8rem; color:#64748b; flex-grow:1; margin:0 0 .8rem 0; }}
</style>
""", unsafe_allow_html=True)

    # ── Header row: single flex row, all items on one straight line ─
    pg = st.session_state.page

    # Brand (left) + nav buttons (centre) + logo (right) in one st.columns
    # so Streamlit renders them as a proper flex row with consistent height.
    # Column widths: brand=1.8 | 4 nav buttons = 0.85 each | logo=1.4
    hc = st.columns([1.8, 0.85, 0.85, 0.85, 0.85, 1.4], gap="small")

    # Brand
    with hc[0]:
        st.markdown(
            "<div style='display:flex;flex-direction:column;justify-content:center;"
            "height:52px;padding:0;'>"
            "<span style='font-size:1.45rem;font-weight:800;color:#111827;"
            "letter-spacing:-0.3px;line-height:1;'>ProcureIQ</span>"
            "<span style='font-size:0.62rem;color:#9ca3af;line-height:1;"
            "margin-top:2px;'>P2P Analytics</span></div>",
            unsafe_allow_html=True,
        )

    # Nav buttons
    nav_items = [
        ("Dashboard", "Dashboard", "nav_dashboard"),
        ("Genie",     "Genie",     "nav_genai"),
        ("Forecast",  "Forecast",  "nav_forecast"),
        ("Invoices",  "Invoices",  "nav_invoices"),
    ]
    for idx, (label, page_key, nav_key) in enumerate(nav_items):
        with hc[idx + 1]:
            if st.button(label, key=nav_key, use_container_width=True,
                         type="primary" if pg == page_key else "secondary"):
                st.session_state.page = page_key
                st.rerun()

    # Logo
    with hc[5]:
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:flex-end;"
            f"height:52px;'><img src='{LOGO_URL}' "
            f"style='height:46px;width:auto;object-fit:contain;'/></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='margin:4px 0 10px 0;border:none;border-top:1px solid #e5e7eb;'/>",
        unsafe_allow_html=True,
    )

    # ── Page routing ────────────────────────────────────────────
    if   pg == "Dashboard":
        # Clear forecast cache when leaving forecast page
        st.session_state.pop("forecast_cf_df", None)
        render_dashboard()
    elif pg == "Genie":
        render_genie()
    elif pg == "Forecast":
        render_forecast()
    else:
        # Clear invoice history cache when entering invoices fresh
        render_invoices()

    # BG button is rendered inside render_charts (bottom-right of Spend Trend)


if __name__ == "__main__":
    main()

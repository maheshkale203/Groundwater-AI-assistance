import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
import os
from datetime import datetime

# Language detection & translation
from deep_translator import GoogleTranslator
from langdetect import detect

# optional NLP / OpenAI
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# -------------------- Config --------------------
st.set_page_config(page_title="Ingres - INGRES Groundwater Assistant", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    .user-message {
        background: linear-gradient(135deg, #DCF8C6, #C8E6C9);
        text-align: right;
        margin-left: auto;
        color: #2E7D32;
    }
    .assistant-message {
        background: linear-gradient(135deg, #FFFFFF, #F5F5F5);
        text-align: left;
        margin-right: auto;
        color: #424242;
        border-left: 4px solid #4CAF50;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTitle {
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
    }
    .stMarkdown h1 {
        color: #2E7D32;
    }
    .stCaption {
        color: #666;
        font-style: italic;
    }
    .stSuccess {
        background-color: #E8F5E8;
        color: #2E7D32;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning {
        background-color: #FFF3E0;
        color: #E65100;
        border-radius: 8px;
        padding: 10px;
    }
    .stError {
        background-color: #FFEBEE;
        color: #C62828;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Default dataset path (use your uploaded path)
DEFAULT_DATA_PATH = "dataset.csv"
                     # You can change this to your actual file name


YEAR_RE = re.compile(r"(19|20)\d{2}")
PRE_RE = re.compile(r"pre[-_ ]?monsoon", re.IGNORECASE)
POST_RE = re.compile(r"post[-_ ]?monsoon", re.IGNORECASE)

# -------------------- Utilities --------------------

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_text(text, dest_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=dest_lang).translate(text)
    except Exception:
        return text


def safe_read(path_or_buffer):
    if path_or_buffer is None:
        return None
    try:
        if hasattr(path_or_buffer, "read"):
            return pd.read_csv(path_or_buffer, encoding="latin1", low_memory=False)
        return pd.read_csv(path_or_buffer, encoding="latin1", low_memory=False)
    except Exception:
        try:
            if hasattr(path_or_buffer, "read"):
                path_or_buffer.seek(0)
            return pd.read_excel(path_or_buffer)
        except Exception as e:
            st.sidebar.error(f"Failed to read dataset: {e}")
            return None


def get_col(df, options):
    """Return first matching column name from options that exists in df, else None."""
    for c in options:
        if c in df.columns:
            return c
    # try case-insensitive fallback
    for col in df.columns:
        for c in options:
            if c.lower() == col.lower():
                return col
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")


def normalize_text(s):
    if s is None:
        return ""
    if pd.isna(s):
        return ""
    # remove parenthetical LGD codes, trim and lowercase
    return re.sub(r"\s*\([^\)]*\)", "", str(s)).strip().lower()

# -------------------- Column detection helper --------------------

def auto_detect_geo_columns(df):
    # possibilities (ordered)
    col_map = {
        "state": ["State_Name_With_LGD_Code", "State", "state_name", "State_Name"],
        "district": ["District_Name_With_LGD_Code", "District", "district_name", "District_Name"],
        "block": ["Block_Name_With_LGD_Code", "Block", "block_name"],
        "village": ["Village", "Village_Name", "village_name"],
        "gp": ["GP_Name_With_LGD_Code", "GP", "Gram_Panchayat"],
        "site": ["Site_Name", "Well_ID", "Well_Name", "Well"]
    }
    detected = {}
    for key, opts in col_map.items():
        detected[key] = get_col(df, opts)
    return detected

# -------------------- Convert wide to long --------------------

def detect_monsoon_columns(df):
    monsoon_map = {}
    for c in df.columns:
        low = c.lower()
        m = YEAR_RE.search(low)
        if not m:
            continue
        yr = int(m.group())
        if PRE_RE.search(low):
            monsoon_map.setdefault(yr, {})["pre"] = c
        if POST_RE.search(low):
            monsoon_map.setdefault(yr, {})["post"] = c
    return dict(sorted(monsoon_map.items()))


def melt_wide_to_long(df, monsoon_map, district_col, site_col):
    rows = []
    if not monsoon_map:
        return pd.DataFrame(rows)
    for idx, r in df.iterrows():
        district = r.get(district_col, "") if district_col and district_col in df.columns else ""
        site = r.get(site_col, f"well_{idx}") if site_col and site_col in df.columns else f"well_{idx}"
        for year, parts in monsoon_map.items():
            pre_col = parts.get("pre")
            post_col = parts.get("post")
            if pre_col and pre_col in df.columns:
                rows.append({"district": district, "well_id": site, "year": year, "season": "pre", "value_raw": r.get(pre_col)})
            if post_col and post_col in df.columns:
                rows.append({"district": district, "well_id": site, "year": year, "season": "post", "value_raw": r.get(post_col)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["value"] = to_numeric_safe(long["value_raw"])
    long["district_norm"] = long["district"].astype(str).apply(normalize_text)
    long["well_id"] = long["well_id"].astype(str)
    return long.drop(columns=["value_raw"])

# -------------------- Simple matching helpers --------------------

def find_match(q, items):
    """Return first item string from items that is found as substring in q (case-insensitive).
    items are expected to be normalized strings (lowercase, no parenthesis).
    """
    if not q or not items:
        return None
    ql = q.lower()
    for it in items:
        if not it:
            continue
        # exact substring
        if it in ql:
            return it
        # partial match: check first 4 chars (if item longer)
        if len(it) >= 4 and it[:4] in ql:
            return it
        # token prefixes
        parts = it.split()
        for p in parts:
            if len(p) >= 4 and p[:4] in ql:
                return it
    return None


def extract_years(q):
    if not q:
        return []
    return sorted({int(m.group()) for m in YEAR_RE.finditer(q)})

# -------------------- Local NLP parser --------------------
# Note: accepts lists of known locations (normalized) and returns the parsed fields.
def nlp_parse_local(query, state_list, district_list, block_list, village_list):
    q = normalize_text(query)
    return {
        "state": find_match(q, state_list),
        "district": find_match(q, district_list),
        "block": find_match(q, block_list),
        "village": find_match(q, village_list),
        "years": extract_years(query),
        "season": "pre" if "pre" in q else "post" if "post" in q else "both",
        "intent": "trend" if any(w in q for w in ["trend","graph","plot","over time","compare"]) else "single"
    }


# -------------------- LLM + query parser (new) --------------------
def parse_user_query(query, df, district_col_geo, district_list, state_list, block_list, village_list, client=None, provider="Gemini"):
    """
    Returns a dictionary:
    {
        'intent': 'chat' | 'single' | 'trend',
        'district': str | None,
        'years': list,
        'season': 'pre'|'post'|'both'
    }
    """
    q_en = translate_text(query, dest_lang="en")

    parsed = None
    if client:
        try:
            if provider == "Gemini":
                system_prompt = (
                    "You are a JSON parser for a groundwater chatbot. "
                    "Return only JSON with fields: intent ('chat','single','trend'), district (str|null), "
                    "years (list), season ('pre','post','both'). Known districts: "
                    + ", ".join(map(str, district_list[:200]))
                )
                user_prompt = f"Question: {q_en}\nReturn only JSON."
                
                full_prompt = system_prompt + "\n" + user_prompt
                
                resp = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=full_prompt
                )
                
                import json, re
                text = resp.text.strip()
            elif provider == "OpenAI":
                system_prompt = (
                    "You are a JSON parser for a groundwater chatbot. "
                    "Return only JSON with fields: intent ('chat','single','trend'), district (str|null), "
                    "years (list), season ('pre','post','both'). Known districts: "
                    + ", ".join(map(str, district_list[:200]))
                )
                user_prompt = f"Question: {q_en}\nReturn only JSON."
                
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                
                import json, re
                text = resp.choices[0].message.content.strip()
            
            try:
                parsed = json.loads(text)
            except:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
        except Exception:
            parsed = None

    # fallback to local NLP if LLM fails
    if not parsed:
        parsed = nlp_parse_local(q_en, state_list, district_list, block_list, village_list)

    # Ensure casual chat questions are handled
    casual = ["hi","hello","how are you","bye","thanks","name","who are you"]
    if any(c in q_en.lower() for c in casual):
        parsed['intent'] = 'chat'
        
    # Standardize empty lists/None fields
    if not parsed.get("years"):
        parsed["years"] = []
    if parsed.get("district") == "null":
        parsed["district"] = None

    return parsed, q_en


# -------------------- UI / App --------------------
# Sidebar removed - data loads automatically from default file

# Language detection is automatic

# Load dataset from default path
df_raw = None
if os.path.exists(DEFAULT_DATA_PATH):
    df_raw = safe_read(DEFAULT_DATA_PATH)

if df_raw is None:
    st.error("No dataset loaded. Please ensure 'dataset.csv' exists in the directory.")
    st.stop()

# Auto-detect geo columns and monsoon columns
geo_cols = auto_detect_geo_columns(df_raw)
monsoon_map = detect_monsoon_columns(df_raw)

# Resolve required column names (fall back to detected or common defaults)
state_col = geo_cols.get("state")
district_col_geo = geo_cols.get("district")
block_col = geo_cols.get("block")
village_col = geo_cols.get("village")
site_col = geo_cols.get("site") or geo_cols.get("gp")  # site/Well/GP fallback

# If any of essential geo columns missing, try some reasonable defaults
if district_col_geo is None:
    for c in df_raw.columns:
        if "district" in c.lower():
            district_col_geo = c
            break

if site_col is None:
    for c in df_raw.columns:
        if any(k in c.lower() for k in ["site","well","bore","site_name"]):
            site_col = c
            break

# Build distinct lists (safe)
def safe_unique_list(df, col):
    if col and col in df.columns:
        return df[col].dropna().astype(str).unique().tolist()
    return []

# Normalize lists and keep mapping normalized->original
def normalize_list_and_map(lst):
    mp = {}
    for it in lst:
        norm = normalize_text(it)
        if norm not in mp:
            mp[norm] = it
    return mp

state_map = normalize_list_and_map(safe_unique_list(df_raw, state_col))
district_map = normalize_list_and_map(safe_unique_list(df_raw, district_col_geo))
block_map = normalize_list_and_map(safe_unique_list(df_raw, block_col))
village_map = normalize_list_and_map(safe_unique_list(df_raw, village_col))

# normalized lists
state_list = list(state_map.keys())
district_list = list(district_map.keys())
block_list = list(block_map.keys())
village_list = list(village_map.keys())

# UI title & messages
st.markdown("### 🌊 Welcome to GroundWater assistant ")
llm_provider = "Groq"  # Default to Groq

client = None
# ---------------------------------------------------------
# API Setup 
# ---------------------------------------------------------
llm_provider = "Groq"  

client = None
llm_provider = "Groq"

client = None
if GROQ_AVAILABLE:
    try:
        # We explicitly ask for the word "GROQ_API_KEY" as a string
        my_secret_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=my_secret_key)
    except Exception as e:
        client = None
        st.error(f"Error connecting to Groq: {e}")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render previous messages
for i, m in enumerate(st.session_state["messages"]):
    if m["role"] == "user":
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f'<div class="chat-message user-message">🧑‍💬 {m["content"]}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Edit", key=f"edit_{i}", help="Edit this message"):
                st.session_state['editing'] = i
    else:
        # 1. Render the text response
        st.markdown(f'<div class="chat-message assistant-message">🤖 {m["content"]}</div>', unsafe_allow_html=True)
        
        # 2. NEW: If this message contains saved trend data, render the tabs natively right here!
        if "chart_data" in m:
            df_chart = pd.read_json(m["chart_data"]) # read saved data back
            display_name = m.get("display_name", "Selected Location")
            
            df_pre = df_chart[df_chart["season"] == "pre"]
            df_post = df_chart[df_chart["season"] == "post"]
            
            tab1, tab2 = st.tabs(["📊 Simple View (District Average)", "🔬 Advanced View (Well Variance)"])
            
            with tab1:
                st.caption(f"**Quick Understanding:** Average water level depth across all wells in {display_name}.")
                if not df_pre.empty:
                    simple_pre = alt.Chart(df_pre).mark_bar(opacity=0.8, color="#ff9999").encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("mean(value):Q", title="Avg Depth (mbgl)", scale=alt.Scale(reverse=True)),
                        tooltip=[alt.Tooltip("year:O"), alt.Tooltip("mean(value):Q", format=".2f")]
                    ).properties(title=f"Pre-monsoon Average — {display_name}", height=300)
                    st.altair_chart(simple_pre, use_container_width=True)
                
                if not df_post.empty:
                    simple_post = alt.Chart(df_post).mark_bar(opacity=0.8, color="#66b3ff").encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("mean(value):Q", title="Avg Depth (mbgl)", scale=alt.Scale(reverse=True)),
                        tooltip=[alt.Tooltip("year:O"), alt.Tooltip("mean(value):Q", format=".2f")]
                    ).properties(title=f"Post-monsoon Average — {display_name}", height=300)
                    st.altair_chart(simple_post, use_container_width=True)
            
            with tab2:
                st.caption("**Deep Dive:** Statistical distribution breakdown showing individual well variance limits.")
                if not df_pre.empty:
                    adv_pre = alt.Chart(df_pre).mark_boxplot(extent='min-max', size=35).encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("value:Q", title="Water Level Depth (mbgl)", scale=alt.Scale(reverse=True, zero=False)),
                        color=alt.Color("year:N", legend=None)
                    ).properties(title=f"Pre-monsoon Well Distribution — {display_name}", height=300)
                    st.altair_chart(adv_pre, use_container_width=True)
                
                if not df_post.empty:
                    adv_post = alt.Chart(df_post).mark_boxplot(extent='min-max', size=35).encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("value:Q", title="Water Level Depth (mbgl)", scale=alt.Scale(reverse=True, zero=False)),
                        color=alt.Color("year:N", legend=None)
                    ).properties(title=f"Post-monsoon Well Distribution — {display_name}", height=300)
                    st.altair_chart(adv_post, use_container_width=True)

# Edit functionality
if 'editing' in st.session_state:
    edit_idx = st.session_state['editing']
    current_msg = st.session_state["messages"][edit_idx]["content"]
    st.markdown("---")
    with st.form(key="edit_form"):
        st.markdown("**Edit your message:**")
        new_msg = st.text_input("Edit your message:", value=current_msg)
        submitted = st.form_submit_button("💾 Save Edit")
    
    if submitted:
        st.session_state["messages"][edit_idx]["content"] = new_msg
        # Remove messages after the edited one and re-process
        st.session_state["messages"] = st.session_state["messages"][:edit_idx+1]
        st.session_state['pending_query'] = new_msg
        del st.session_state['editing']
        st.rerun()
    
    # Cancel button outside form
    if st.button("❌ Cancel Edit"):
        del st.session_state['editing']
        st.rerun()

st.markdown("---")
query = st.chat_input("Ask your question:")

pending = st.session_state.get('pending_query')
if query or pending:
    if pending:
        q = pending
        del st.session_state['pending_query']
    else:
        q = query.strip()

    if not q:
        st.warning("Please type a question.")
        st.rerun()

    # Store user msg
    st.session_state["messages"].append({"role": "user", "content": q})
    
    with st.spinner("Processing your query..."):
        # --- Determine Target Language for Reply ---
        target_lang = detect_language(q)


        # Parse
        parsed, q_en = parse_user_query(q, df_raw, district_col_geo, district_list, state_list, block_list, village_list, client, llm_provider)
        intent = parsed.get("intent", "single")
        
       # =============================
        # CHAT INTENT (General Questions)
        # =============================
        if intent == 'chat':
            if client:
                try:
                    system_prompt = (
                        "You are Ingres, a specialized AI assistant expert in the INGRES (INDIA-Groundwater Resource Estimation System) website and groundwater data analysis. "
                        "Your primary role is to provide comprehensive information about the INGRES website and analyze water level data from CSV files. "
                        "CRITICAL INSTRUCTION: You MUST detect the language of the user's question and reply in that EXACT same language (e.g., if they ask in Hindi, reply in Hindi. If Bengali, reply in Bengali). "
                        "Structure responses with clear headings and bullet points."
                    )
                    # Notice we use 'q' here (the user's exact input), NOT 'q_en'
                    user_prompt = f"Question: {q}\n\nProvide a detailed response."

                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.6
                    )
                    reply_text = resp.choices[0].message.content.strip()

                    # REMOVED: The GoogleTranslator step is completely gone! 
                    # Groq will output the correct language natively.

                except Exception as e:
                    st.error(f"🚨 Groq API Error Detail: {str(e)}") 
                    reply_text = "I'm sorry, I encountered an error while processing your question. Please try again."
            else:
                reply_text = "Hello! I am Ingres. What would you like to know about INGRES or your groundwater data?"
            
            st.session_state["messages"].append({"role": "assistant", "content": reply_text})
            st.rerun()

        # =============================
        # District inference block
        # =============================
        if not parsed.get("district"):
            inferred = None
            v = parsed.get("village")
            if v and village_col:
                try:
                    matched = df_raw[
                        df_raw[village_col]
                        .astype(str)
                        .str.lower()
                        .str.contains(str(v).lower(), na=False)
                    ]
                    if not matched.empty and district_col_geo in matched.columns:
                        inferred = matched.iloc[0][district_col_geo]
                except Exception:
                    inferred = None

            if inferred:
                parsed["district"] = normalize_text(inferred)
            else:
                # Use target_lang for error message
                error_msg_en = "Please include a district name or a location I can infer the district from."
                error_msg = translate_text(error_msg_en, dest_lang=target_lang)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )
                st.rerun()

        district_parsed_norm = parsed.get("district")

        # =============================
        # Long DF Preparation
        # =============================
        long_df = melt_wide_to_long(df_raw, monsoon_map, district_col_geo, site_col)

        if long_df.empty:
            error_msg_en = "No monsoon data columns (e.g., Pre-Monsoon 2020) were found in the dataset."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
            st.rerun()

        # District filter
        if district_parsed_norm:
            df_filtered = long_df[
                long_df["district_norm"].str.contains(district_parsed_norm, na=False)
            ]
        else:
            df_filtered = pd.DataFrame() # Should not happen due to check above, but safe.

        # Approx match fallback
        if df_filtered.empty and district_parsed_norm:
            tokens = re.findall(r"[A-Za-z]{3,}", str(district_parsed_norm or ""))
            for t in tokens:
                cand = long_df[long_df["district_norm"].str.contains(t, na=False)]
                if not cand.empty:
                    df_filtered = cand
                    break

        if df_filtered.empty:
            error_msg_en = f"No data found for district '{parsed.get('district')}'."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )
            st.rerun()

        # =============================
        # Trend Graph Block
        # =============================
        if intent == "trend":
            display_name = district_map.get(district_parsed_norm, parsed.get("district"))

            response_msg_en = f"Here is the trend data analysis for **{display_name}**. I have built both a clean bar summary and an advanced box-plot variance breakdown for you below."
            response_msg = translate_text(response_msg_en, dest_lang=target_lang)
            
            # Save data inside the message dictionary so it survives st.rerun()!
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": response_msg,
                "chart_data": df_filtered.to_json(), # Convert df to safe JSON string format
                "display_name": display_name
            })
            
            st.rerun()

        # =============================
        # Single Year / Value Block
        # =============================
        
        years_parsed = parsed.get("years", [])

        target_year = (
            years_parsed[0]
            if years_parsed
            else max(df_filtered["year"].unique())
            if not df_filtered.empty
            else None
        )

        if not target_year:
            error_msg_en = "Please include a year in your query, or try a 'trend' question."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )
            st.rerun()

        df_year = df_filtered[df_filtered["year"] == target_year]

        if df_year.empty:
            error_msg_en = f"No data for {district_map.get(district_parsed_norm)} in {target_year}."
            error_msg = translate_text(error_msg_en, dest_lang=target_lang)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )
            st.rerun()

        # =============================
        # Pivot + averages
        # =============================
        pivot = (
            df_year.pivot_table(
                index="well_id",
                columns="season",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )

        pre_avg = pivot["pre"].mean() if "pre" in pivot.columns else np.nan
        post_avg = pivot["post"].mean() if "post" in pivot.columns else np.nan

        data_context = f"District: {district_map.get(district_parsed_norm)}, Year: {target_year}, Pre-Monsoon Avg: {pre_avg:.2f}, Post-Monsoon Avg: {post_avg:.2f}."

       # =============================
        # LLM ANSWER (Groq)
        # =============================
        if client:
            try:
                system_prompt = (
                    "You are Ingres, a specialized AI assistant expert in the INGRES (INDIA-Groundwater Resource Estimation System) website and groundwater data analysis. "
                    "Analyze and provide insights from uploaded water level CSV data files. "
                    "CRITICAL INSTRUCTION: You MUST write your entire analysis in the exact same language the user used to ask the question (e.g., Hindi, Bengali, English). "
                    "Structure responses with clear headings and bullet points. "
                    "For data analysis, structure the response as follows: "
                    "- Overview: Provide a brief overview of the groundwater levels for the specified location and year. "
                    "- Key Points: Detail Pre-Monsoon and Post-Monsoon observations. "
                    "- Statewide Trends & Long Term Change. "
                    "- Example Table formatted in Markdown."
                )
                user_prompt = (
                    # We pass 'q' (original language) to let the model know what language to speak, 
                    # but we also pass the data context so it knows the numbers.
                    f"User's Question: {q}\n\nData Context (in English): {data_context}\n\n"
                    "Provide a detailed, helpful response as a groundwater expert in the user's language."
                )

                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                reply_text = resp.choices[0].message.content.strip()

            except Exception as e:
                reply_text = f"An error occurred while generating a detailed response. The core data is: {data_context}"

        else:
            district_name = district_map.get(district_parsed_norm, district_parsed_norm)
            reply_text = f"Here is the data for {district_name} in {target_year}: Pre-Monsoon {pre_avg:.2f}, Post-Monsoon {post_avg:.2f}."

        # We append the message exactly ONCE. Groq already handled the language!
        st.session_state["messages"].append({"role": "assistant", "content": reply_text})


        # Show pivot
        if not pivot.empty:
            st.caption(f"Raw data for {district_map.get(district_parsed_norm)} in {target_year}:")
            st.dataframe(pivot)

        st.rerun()

# =============================
# Clear Button
# =============================
if st.button("🗑️ Clear chat"):
    st.session_state["messages"] = []
    st.rerun()

st.caption(f"App last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
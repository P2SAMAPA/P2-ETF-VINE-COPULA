"""
Streamlit Dashboard — T‑COPULA: Daily, Global, Shrinking tabs.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant T‑COPULA", page_icon="📐", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("t_copula_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        path = hf_hub_download(repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
                               repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def safe_pct(val):
    try:
        return f"{float(val)*100:.2f}%"
    except:
        return "N/A"

def render_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data available.")
        return
    top_picks = mode_data.get('top_picks', [])
    universes = mode_data.get('universes', {})
    if top_picks:
        p = top_picks[0]
        ticker = p.get('ticker', 'N/A')
        raw_ret = p.get('expected_return_raw', 0)
        adj_score = p.get('t_copula_adj_score', 0)
        es95 = p.get('es_95', 0)
        dof = p.get('dof', 'N/A')
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size: 1.2rem; opacity: 0.8;">📐 {mode_name} TOP PICK</div>
            <div class="hero-ticker">{ticker}</div>
            <div>Adj Score: {adj_score:.4f}</div>
            <div>Raw Return: {safe_pct(raw_ret)} | ES₉₅: {safe_pct(es95)} | dof: {dof}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Top 3 Picks")
        rows = [{"Ticker": p['ticker'], "Adj Score": f"{p.get('t_copula_adj_score',0):.4f}",
                 "Raw Return": safe_pct(p.get('expected_return_raw',0)),
                 "ES₉₅": safe_pct(p.get('es_95',0)),
                 "dof": f"{p.get('dof','')}"} for p in top_picks]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### All ETFs")
        all_rows = [{"Ticker": t, "Adj Score": f"{d.get('t_copula_adj_score',0):.4f}",
                     "Raw Return": safe_pct(d.get('expected_return_raw',0)),
                     "ES₉₅": safe_pct(d.get('es_95',0))} for t, d in universes.items()]
        df_all = pd.DataFrame(all_rows).sort_values("Adj Score", ascending=False)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    ticker = shrinking_data['ticker']
    conviction = shrinking_data['conviction']
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 SHRINKING CONSENSUS</div>
        <div class="hero-ticker">{ticker}</div>
        <div>{conviction:.0f}% conviction across {shrinking_data['num_windows']} windows</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = [{"Window": f"{w['window_start']}-{w['window_end']}", "ETF": w['ticker'],
                 "Exp Return": safe_pct(w.get('expected_return',0))} for w in shrinking_data.get('windows', [])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">📐 P2Quant T‑COPULA</div>', unsafe_allow_html=True)
st.markdown('<div>Student‑t Copula – Tail‑Dependence‑Aware ETF Ranking</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        st.info(f"No data for {key}.")
        continue
    with tab:
        sub_daily, sub_global, sub_shrink = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with sub_daily:
            render_mode_tab(uni.get('daily'), "Daily")
        with sub_global:
            render_mode_tab(uni.get('global'), "Global")
        with sub_shrink:
            render_shrinking_tab(uni.get('shrinking'))

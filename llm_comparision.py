
# literature_review_streamlit_app.py
# Streamlit app: Interactive Literature Review on LLMs
# Run: streamlit run literature_review_streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

st.set_page_config(
    page_title="LLM Literature Review — Interactive",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# ---- Style & Utilities ----
# ---------------------------
PRIMARY_ACCENT = "#6C63FF"

CUSTOM_CSS = f"""
<style>
.block-container {{padding-top: 1rem;}}
.card {{
    background: radial-gradient(100% 100% at 50% 0%, rgba(255,255,255,0.70), rgba(255,255,255,0.50));
    border: 1px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.12);
    border-radius: 16px;
    padding: 1.1rem 1rem;
    margin-bottom: 1rem;
}}
.kbd {{
    background: #111827;
    color: #f9fafb;
    border-radius: 6px;
    padding: 2px 6px;
    font-size: 0.8rem;
    font-weight: 600;
}}
.chip {{
    display: inline-block;
    padding: 4px 10px;
    margin: 2px 6px 2px 0;
    border-radius: 999px;
    border: 1px solid rgba(107,114,128,0.3);
    background: rgba(99,102,241,0.08);
    color: #111827;
    font-size: 0.85rem;
}}
.small {{
    font-size: 0.88rem;
    color: #374151;
}}
.hint {{
    background: rgba(34,197,94,0.08);
    border: 1px dashed rgba(34,197,94,0.45);
    padding: 8px 12px;
    border-radius: 12px;
}}
h1 span.badge {{
    background: {PRIMARY_ACCENT};
    color: white;
    padding: 6px 10px;
    border-radius: 10px;
    font-size: 0.7em;
    margin-left: 8px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    except Exception:
        return text

def md_card(md: str):
    st.markdown(f'<div class="card">{md}</div>', unsafe_allow_html=True)

def section_header(title: str, badge: str = None):
    if badge:
        st.markdown(f"# {title} <span class='badge'>{badge}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"# {title}")

# ---------------------------
# ---- Sidebar Controls -----
# ---------------------------
st.sidebar.title("🔎 Explorer")
nav = st.sidebar.radio("Jump to:", [
    "1. Introduction",
    "Methodology",
    "Results",
    "Discussion",
    "Overview",
    "2.1 Evolution of LLM Architectures",
    "2.2 Reasoning-Focused Training",
    "2.3 Multimodal Integration",
    "2.4 Long-Context Models",
    "2.5 Benchmarking Practices",
    "2.6 Open vs. Proprietary Trends",
    "References & Exports"
], index=0)

st.sidebar.markdown("---")
search = st.sidebar.text_input("Search & highlight text", placeholder="e.g., Mixture-of-Experts")
st.sidebar.markdown("Use <span class='kbd'>Ctrl</span> + <span class='kbd'>K</span> to quickly search on-page.", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Theme")
compact = st.sidebar.toggle("Compact spacing", value=False)

# ---------------------------
# ---- Page: Introduction ---
# ---------------------------
if nav == "1. Introduction":
    section_header("1. Introduction")
    intro_text = """
Since 2023, LLM development has accelerated due to innovations in architecture (dense Transformers, Mixture-of-Experts, reasoning-tuned models), scaling strategies, and multi-modal integration. Competitive pressures have driven significant improvements in:

• **Reasoning capabilities** (e.g., AIME, GPQA benchmarks),  
• **Code synthesis and repository repair** (LiveCodeBench, SWE-bench),  
• **Contextual memory** (up to 2M tokens in commercial APIs),  
• **Open-weight availability** for enterprise deployment and fine-tuning.

This paper systematically compares leading LLM families, focusing on both technical specifications and empirical performance.
"""
    md_card(highlight_text(intro_text, search))

    st.markdown("### Key Themes")
    chip_row = """
<span class="chip">Reasoning — AIME/GPQA</span>
<span class="chip">Code — LiveCodeBench/SWE-bench</span>
<span class="chip">Context — up to 2M tokens</span>
<span class="chip">Openness — enterprise fine-tuning</span>
"""
    md_card(chip_row)

    with st.expander("Quick visual — emphasis across themes (illustrative)"):
        toy = pd.DataFrame({
            "Theme": ["Reasoning", "Code", "Context", "Openness"],
            "Emphasis (rel.)": [92, 88, 85, 80]
        })
        fig = px.bar(toy, x="Theme", y="Emphasis (rel.)")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ---- Page: Methodology ----
# ---------------------------
if nav == "Methodology":
    section_header("Methodology")
    md_card(highlight_text("""
**2.1 Model Selection**  
We included models from organizations with either frontier-level benchmarks or wide enterprise adoption.

**Proprietary:** OpenAI, Google DeepMind, Anthropic, xAI, Microsoft, Alibaba, DeepSeek, Baidu, Reka.  
**Open-weight:** Meta, Mistral, Databricks, NVIDIA.
    """, search))

    st.markdown("### Explorer — Organizations & Licenses (toy)")
    orgs = pd.DataFrame([
        {"Organization": "OpenAI", "Family": "Proprietary", "Modalities": "T,V,A", "Notes": "Frontier benchmarks"},
        {"Organization": "Google DeepMind", "Family": "Proprietary", "Modalities": "T,I,A,V", "Notes": "Gemini multi-modal"},
        {"Organization": "Anthropic", "Family": "Proprietary", "Modalities": "T,V", "Notes": "Safety emphasis"},
        {"Organization": "xAI", "Family": "Proprietary", "Modalities": "T,V,A", "Notes": "Grok-3 long context"},
        {"Organization": "Microsoft", "Family": "Proprietary", "Modalities": "T,V,A", "Notes": "Phi / orchestration"},
        {"Organization": "Alibaba", "Family": "Proprietary", "Modalities": "T,V", "Notes": "Qwen commercial APIs"},
        {"Organization": "DeepSeek", "Family": "Proprietary", "Modalities": "T,V", "Notes": "Process RL (R1)"},
        {"Organization": "Baidu", "Family": "Proprietary", "Modalities": "T,I,A,V", "Notes": "ERNIE family"},
        {"Organization": "Reka", "Family": "Proprietary", "Modalities": "T,I,A,V", "Notes": "Reka Core"},
        {"Organization": "Meta", "Family": "Open-weight", "Modalities": "T,V", "Notes": "Llama 3 ecosystem"},
        {"Organization": "Mistral", "Family": "Open-weight", "Modalities": "T,V", "Notes": "Mixtral / Codestral"},
        {"Organization": "Databricks", "Family": "Open-weight", "Modalities": "T", "Notes": "DBRX (MoE)"},
        {"Organization": "NVIDIA", "Family": "Open-weight", "Modalities": "T,V,A", "Notes": "Nemotron / NIMs"},
    ])

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        fam = st.selectbox("Family", ["Both", "Proprietary", "Open-weight"], index=0)
    with c2:
        mods = st.multiselect("Filter by modality", ["T","V","A","I"])
    with c3:
        st.caption("Use these filters to preview which orgs/models are in scope.")

    filt = orgs.copy()
    if fam != "Both":
        filt = filt[filt["Family"] == fam]
    if mods:
        def has_all(mstr):
            return all(m in (mstr or "") for m in mods)
        filt = filt[filt["Modalities"].apply(has_all)]

    st.dataframe(filt, use_container_width=True, hide_index=True)

    st.markdown("### 2.2 Evaluation Criteria")
    md_card(highlight_text("""
We compare models on:  
- **Architecture** — dense vs MoE; reasoning-tuned; multimodal integration.  
- **Modalities** — text, vision, audio, video.  
- **Context window** — effective and max capacity.  
- **Licensing** — open-weight vs closed API.  
- **Benchmarks**:
  - *Reasoning:* AIME 2024/2025, GPQA.  
  - *Coding:* LiveCodeBench, SWE-bench Verified.  
  - *Knowledge:* MMLU-Pro, MMMU.  
  - *Long-context:* LOFT-128k, max token capacity.
    """, search))

    with st.expander("Interactive checklist (what we will score)"):
        cols = st.columns(4)
        with cols[0]:
            st.checkbox("Architecture", value=True)
            st.checkbox("Reasoning (AIME/GPQA)", value=True)
        with cols[1]:
            st.checkbox("Coding (LiveCode/SWE)", value=True)
            st.checkbox("Knowledge (MMLU-Pro/MMMU)", value=True)
        with cols[2]:
            st.checkbox("Modalities", value=True)
            st.checkbox("Context window", value=True)
        with cols[3]:
            st.checkbox("Licensing", value=True)
            st.checkbox("Long-context (LOFT/Max tokens)", value=True)
        st.caption("This preview mirrors the scoring dimensions used later in Results.")

    st.markdown("### 2.3 Data Sources")
    md_card(highlight_text("""
Performance data is sourced from vendor evaluations, public leaderboards, and technical reports.  
For consistency, we **flag whether test-time compute optimizations** (e.g., majority voting, tool use) were applied.
    """, search))

    src = pd.DataFrame([
        {"Source": "Vendor model cards / tech reports", "Pros": "Closest to model details", "Caveats": "May use test-time tricks"},
        {"Source": "Public leaderboards", "Pros": "Comparable across models", "Caveats": "Different rules / stale data"},
        {"Source": "Independent eval posts", "Pros": "Reproducible, scripts", "Caveats": "Limited coverage"},
    ])
    st.table(src)

# ---------------------------
# ---- Page: Results --------
# ---------------------------
if nav == "Results":
    section_header("Results")
    # ----- 3.1 Technical Overview -----
    st.markdown("### 3.1 Technical Overview")
    tech_cols = ["Family","2025 Flagship","Architecture","Modalities","Context (tokens)","License"]
    tech_rows = [
        ["OpenAI","o4-mini / o3","reasoning-tuned Transformer","T/V/A","undisclosed (≥256k in evals)","closed"],
        ["Google","Gemini 2.5 Pro / Flash","undisclosed","T/I/A/V","1–2M","closed"],
        ["Anthropic","Claude 3.5 Sonnet","Transformer","T/V","200k","closed"],
        ["xAI","Grok-3","reasoning-tuned","T/V","1M","closed"],
        ["Meta","Llama 3.1 405B","dense","T","128k","open"],
        ["Mistral","Large 2","dense","T (Pixtral for V)","128k","API"],
        ["Databricks","DBRX","MoE","T","32–128k","open"],
        ["NVIDIA","Nemotron-4 340B","dense","T","4k","open"],
        ["Microsoft","Phi-3.5 MoE","MoE","T/V","128k","closed"],
        ["Alibaba","Qwen2.5-Max","MoE","T/V","varies","mixed"],
        ["DeepSeek","DeepSeek-R1","reasoning-tuned","T","varies","mixed"],
        ["Baidu","ERNIE 4.5/X1","MoE","T/I/A/V","varies","mixed"],
        ["Reka","Reka Core","frontier multimodal","T/I/A/V","varies","closed"],
    ]
    tech_df = pd.DataFrame(tech_rows, columns=tech_cols)

    def parse_ctx(s: str):
        if not isinstance(s,str): return None
        ss = s.lower().replace(" ", "")
        if "undisclosed" in ss or "varies" in ss: return None
        ss = ss.replace("–","-").replace("—","-")
        if "-" in ss:
            right = ss.split("-")[-1]
        else:
            right = ss
        try:
            if right.endswith("m"):
                return float(right.replace("m","")) * 1_000_000
            if right.endswith("k"):
                return float(right.replace("k","")) * 1_000
            return float(re.sub(r"[^0-9.]","", right))
        except:
            return None

    tech_df["Context (num)"] = tech_df["Context (tokens)"].apply(parse_ctx)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        lic = st.selectbox("License filter", ["All","open","closed","mixed","API"], index=0)
    with c2:
        modality_filter = st.multiselect("Modalities", ["T","V","A","I"])
    with c3:
        arch_filter = st.multiselect("Architecture", sorted(tech_df["Architecture"].unique().tolist()))

    filt = tech_df.copy()
    if lic != "All":
        filt = filt[filt["License"].str.lower()==lic.lower()]
    if modality_filter:
        def has_all(mods):
            mods = (mods or "").upper()
            return all(m in mods for m in modality_filter)
        filt = filt[filt["Modalities"].apply(has_all)]
    if arch_filter:
        filt = filt[filt["Architecture"].isin(arch_filter)]

    st.dataframe(filt[tech_cols], use_container_width=True, hide_index=True)

    with st.expander("Context window — approximate upper-bound (illustrative)"):
        ctx_plot = filt.dropna(subset=["Context (num)"])
        if not ctx_plot.empty:
            fig = px.bar(ctx_plot.sort_values("Context (num)", ascending=False),
                         x="Family", y="Context (num)",
                         hover_data=["2025 Flagship","Context (tokens)"])
            fig.update_layout(height=360, yaxis_title="Tokens (approx)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No numeric context values available for current filters.")

    # ----- 3.2 Benchmark Performance -----
    st.markdown("### 3.2 Benchmark Performance")

    tab_reason, tab_coding, tab_knowledge, tab_long = st.tabs(["Reasoning","Coding","Knowledge","Long-Context"])

    with tab_reason:
        reason_rows = [
            ["OpenAI o4-mini","AIME’25","pass@1",99.5,"Python-enabled"],
            ["OpenAI o3","AIME’25","pass@1",98.4,"Python-enabled"],
            ["Google Gemini 2.5 Pro","AIME’25","SOTA",None,"Reported without test-time voting"],
            ["xAI Grok-3","AIME’25","pass@1",93.3,""],
            ["xAI Grok-3","GPQA","score",84.6,""],
        ]
        rcols = ["Model","Benchmark","Metric","Value","Notes"]
        rdf = pd.DataFrame(reason_rows, columns=rcols)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        rplot = rdf.dropna(subset=["Value"])
        if not rplot.empty:
            fig = px.bar(rplot, x="Model", y="Value", color="Benchmark", barmode="group")
            fig.update_layout(height=360, yaxis_title="Score / Pass@1 (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_coding:
        coding_rows = [
            ["OpenAI o-series","LiveCodeBench","leader (recent slices)",None,"Top groups"],
            ["Google Gemini 2.5","LiveCodeBench","leader (recent slices)",None,"Top groups"],
            ["DeepSeek-R1","LiveCodeBench","competitive",None,""],
            ["Claude 3.5 Sonnet","SWE-bench Verified","Success %",49.0,"Minimal tools; repo-level fixing"],
        ]
        ccols = ["Model","Benchmark","Metric","Value","Notes"]
        cdf = pd.DataFrame(coding_rows, columns=ccols)
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        cplot = cdf.dropna(subset=["Value"])
        if not cplot.empty:
            fig = px.bar(cplot, x="Model", y="Value", color="Benchmark")
            fig.update_layout(height=320, yaxis_title="Success / Score (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_knowledge:
        know_rows = [
            ["Gemini 2.5 Pro Deep Think","MMMU","Score",84.0,""],
            ["Proprietary leaders","MMLU-Pro",">=83%",None,"Aggregate statement"],
            ["Llama 3.1 405B","MMLU-Pro","competitive",None,""],
        ]
        kcols = ["Model","Benchmark","Metric","Value","Notes"]
        kdf = pd.DataFrame(know_rows, columns=kcols)
        st.dataframe(kdf, use_container_width=True, hide_index=True)
        kplot = kdf.dropna(subset=["Value"])
        if not kplot.empty:
            fig = px.bar(kplot, x="Model", y="Value", color="Benchmark")
            fig.update_layout(height=320, yaxis_title="Score (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_long:
        long_rows = [
            ["Google Gemini (1.5/2.5)","Max context",2_000_000,""],
            ["xAI Grok-3","Max context",1_000_000,""],
            ["Anthropic Claude 3.5","Max context",200_000,""],
            ["Meta Llama 3.1 / Mistral Large 2","Max context",128_000,""],
        ]
        lcols = ["Model","Metric","Tokens","Notes"]
        ldf = pd.DataFrame(long_rows, columns=lcols)
        st.dataframe(ldf, use_container_width=True, hide_index=True)
        fig = px.bar(ldf, x="Model", y="Tokens")
        fig.update_layout(height=320, yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: Values reflect the dataset you provided. Some entries are qualitative or lack exact numbers; plots show only numeric values when available.")

# ---------------------------
# ---- Page: Overview -------
# ---------------------------
if nav == "Overview":
    section_header("Literature Review", "Interactive")
    colA, colB = st.columns([1.4, 1])
    with colA:
        md_card("""
**Scope.** This interactive review synthesizes core strands of recent LLM literature:
- Architectural evolution (dense vs. Mixture-of-Experts),
- Reasoning-focused training,
- Multimodal integration,
- Long-context modeling,
- Benchmarking practices and pitfalls,
- Open vs. proprietary ecosystem dynamics.

Use the sidebar to navigate sections, highlight keywords, and explore timelines and comparisons.
        """)
        md_card(f"""
<span class="hint">💡 Tip:</span> Try the <span class="chip">Timeline</span> and <span class="chip">Benchmark map</span> widgets below to get a quick, visual overview.
        """)
    with colB:
        md_card("""
**Quick Links**
- [1] Introduction  
- [2.1] Evolution of LLM Architectures  
- [2.2] Reasoning-Focused Training  
- [2.3] Multimodal Integration  
- [2.4] Long-Context Models  
- [2.5] Benchmarking Practices  
- [2.6] Open vs. Proprietary Trends  
        """)

    st.subheader("📅 Timeline — Models & Milestones")
    timeline = pd.DataFrame({
        "Item": [
            "Transformer (Attention Is All You Need)",
            "GPT-3",
            "Switch Transformer (MoE)",
            "CLIP",
            "Flamingo",
            "Press et al. (Long-Context)",
            "GPT-4 family (dense, multi-modal variants)",
            "Reasoning Tuning (CoT/process RL) grows",
            "GPT-4o / Gemini 1.5 multimodal",
            "DeepSeek-R1 (process RL)",
            "Gemini 2.5 (no test-time inflate reporting)"
        ],
        "Type": ["Paper", "Model", "Paper/Model", "Paper/Model", "Paper/Model", "Paper", "Model", "Practice", "Model", "Model", "Model"],
        "Year": [2017, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2025, 2025]
    })
    yr_min, yr_max = st.slider("Filter by year", min_value=2017, max_value=2025, value=(2019, 2025))
    filt = timeline[(timeline["Year"] >= yr_min) & (timeline["Year"] <= yr_max)]
    fig = px.scatter(
        filt, x="Year", y=["Type"], color="Type", hover_name="Item",
        size_max=18
    )
    fig.update_layout(height=380, xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧭 Benchmark Map (toy example)")
    bm = pd.DataFrame({
        "Benchmark": ["AIME", "GPQA", "MMLU-Pro", "MMMU", "LiveCodeBench", "SWE-bench Verified"],
        "Domain": ["Math", "Science", "General", "Multimodal", "Coding", "Coding"],
        "Focus": ["Reasoning", "Reasoning", "Knowledge", "Vision+Text", "Code gen", "Repo-level"]
    })
    fig2 = px.treemap(bm, path=["Domain", "Benchmark"], values=[1]*len(bm), hover_data=["Focus"])
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# ---- Section 2.1 ----------
# ---------------------------
if nav == "2.1 Evolution of LLM Architectures":
    section_header("2.1 Evolution of LLM Architectures")
    text = """
**Dense decoder-only Transformers.** Since *Vaswani et al., 2017*, decoder-only dense Transformers have dominated,
with all parameters active per token. Exemplars include **GPT-3 (Brown et al., 2020)** and later families (**GPT-4**, **Claude 3.x**, **Llama 3**).
Pros: stable scaling laws and predictable performance. Cons: high inference cost due to full-parameter activation.

**Mixture-of-Experts (MoE).** Re-popularized by **Switch Transformer (Fedus et al., 2021)**, MoE routes tokens
to a sparse subset of expert FFNs per layer, reducing per-token FLOPs while increasing total capacity.
Industry adoptions include **DBRX**, **Qwen2.5-Max**, **Phi-3.5-MoE**—often yielding improved cost-performance.
"""
    md_card(highlight_text(text, search))

    st.markdown("### ⚙️ Dense vs. MoE — Cost/Capacity Sketch (toy)")
    df = pd.DataFrame({
        "Paradigm": ["Dense", "Dense", "MoE", "MoE"],
        "Axis": ["Capacity", "Per-token cost", "Capacity", "Per-token cost"],
        "Score": [7, 7, 10, 4]
    })
    fig = px.bar(df, x="Paradigm", y="Score", color="Axis", barmode="group")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ---- Section 2.2 ----------
# ---------------------------
if nav == "2.2 Reasoning-Focused Training":
    section_header("2.2 Reasoning-Focused Training")
    text = """
Recent work emphasizes **reasoning tuning**: chain-of-thought prompting, **reinforcement learning from process feedback**
(e.g., *OpenAI o-series*, **DeepSeek-R1**), and configurable **“slow thinking”** modes (e.g., **xAI Grok-3**).
Empirically (e.g., *Zelikman et al., 2024*), these techniques raise scores on **AIME**, **GPQA**, etc., albeit often with **higher latency**.
"""
    md_card(highlight_text(text, search))

    with st.expander("Try a toy ablation — sampling budget vs. pass@1 lift"):
        toy = pd.DataFrame({
            "Test-time Samples": [1, 2, 4, 8, 16, 32],
            "Pass@1 (relative)": [1.00, 1.04, 1.08, 1.11, 1.13, 1.14],
        })
        fig = px.line(toy, x="Test-time Samples", y="Pass@1 (relative)")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Illustrative: gains saturate as sampling increases; latency and cost rise.")

# ---------------------------
# ---- Section 2.3 ----------
# ---------------------------
if nav == "2.3 Multimodal Integration":
    section_header("2.3 Multimodal Integration")
    text = """
Following **CLIP (Radford et al., 2021)** and **Flamingo (Alayrac et al., 2022)**, frontier models (e.g., **Gemini**, **GPT-4o**, **Claude 3.5**, **Reka Core**)
**natively integrate** text, image, audio, and video. Literature reports that cross-attention fusion improves **vision-language reasoning** (e.g., *Li et al., 2023*),
but with **significant training cost** and data curation overhead.
"""
    md_card(highlight_text(text, search))

    with st.expander("Widget: Fusion depth vs. VL score (toy)"):
        toy = pd.DataFrame({
            "Fusion Layers": [0, 2, 4, 6, 8],
            "VL Score (rel.)": [0.88, 0.93, 0.97, 0.99, 1.00]
        })
        fig = px.area(toy, x="Fusion Layers", y="VL Score (rel.)")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ---- Section 2.4 ----------
# ---------------------------
if nav == "2.4 Long-Context Models":
    section_header("2.4 Long-Context Models")
    text = """
Advances in **positional encodings** and **memory mechanisms** (e.g., *Press et al., 2022; Dao et al., 2023*) enable **context windows** up to **~2M tokens**
(e.g., **Gemini 1.5/2.5**, **Grok-3**). **RAG** benefits most when retrieval targets **task-relevant** passages; overly long contexts can dilute signal
(*Khandelwal et al., 2023*).
"""
    md_card(highlight_text(text, search))

    col1, col2 = st.columns(2)
    with col1:
        toy = pd.DataFrame({
            "Context (tokens)": [8e3, 32e3, 128e3, 512e3, 2e6],
            "Latency (ms)": [120, 180, 350, 900, 2800]
        })
        fig = px.line(toy, x="Context (tokens)", y="Latency (ms)", markers=True)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        toy2 = pd.DataFrame({
            "Context Relevancy (%)": [20, 40, 60, 80, 100],
            "Answer Quality (rel.)": [0.70, 0.82, 0.93, 1.00, 0.99]
        })
        fig2 = px.scatter(toy2, x="Context Relevancy (%)", y="Answer Quality (rel.)", trendline="ols")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Illustrative: beyond the task-relevant sweet spot, more context ≠ better answers.")

# ---------------------------
# ---- Section 2.5 ----------
# ---------------------------
if nav == "2.5 Benchmarking Practices":
    section_header("2.5 Benchmarking Practices")
    text = """
Common benchmarks: **AIME**, **GPQA** (math/science reasoning), **MMLU-Pro**, **MMMU** (general & multimodal),
**LiveCodeBench**, **SWE-bench Verified** (coding). However, **test-time compute inflation**—majority voting, repeated sampling—
can distort comparisons (*Reuel et al., 2025*). Some reports (e.g., **Gemini 2.5**) avoid such optimizations for transparency.
"""
    md_card(highlight_text(text, search))

    with st.expander("Interactive: What happens if we allow 32 samples? (toy)"):
        s = st.slider("Samples", 1, 64, 1, help="Simulated test-time sampling budget")
        base = 0.70
        lift = min(0.20, 0.06 * (1 - pow(0.85, s)))  # saturating toy curve
        st.metric("Reported Score (toy)", f"{(base + lift):.2f}", delta=f"+{lift*100:.1f} pts")

# ---------------------------
# ---- Section 2.6 ----------
# ---------------------------
if nav == "2.6 Open vs. Proprietary Trends":
    section_header("2.6 Open vs. Proprietary Trends")
    text = """
Governance literature (e.g., *Henderson et al., 2024; Bommasani et al., 2024*) notes a bifurcation:
**Proprietary leaders** (OpenAI, Google, Anthropic, xAI) tend to hold **SOTA** on many benchmarks,
while **open-weight** contenders (Meta, Databricks, Mistral, NVIDIA, Alibaba/Qwen) prioritize **transparency, research access,**
and **local deployment**—often trading absolute peak scores for openness and controllability.
"""
    md_card(highlight_text(text, search))

    with st.expander("Compare priorities (illustrative radar)"):
        radar = pd.DataFrame({
            "Axis": ["SOTA scores", "Openness", "Local deploy", "Speed of updates", "Ecosystem"],
            "Proprietary": [95, 30, 40, 85, 90],
            "Open-weight": [85, 95, 90, 70, 80],
        })
        rf = radar.melt(id_vars="Axis", var_name="Family", value_name="Score")
        fig = px.line_polar(rf, r="Score", theta="Axis", color="Family", line_close=True)
        fig.update_traces(fill='toself', opacity=0.6)
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ---- Page: Discussion -----
# ---------------------------
if nav == "Discussion":
    section_header("Discussion")
    md_card(highlight_text("""
Our analysis shows:

**Reasoning.** OpenAI and Google dominate high-stakes reasoning tasks, with xAI offering competitive alternatives.

**Coding.** No single leader — Claude 3.5 leads in repo-fix, while Gemini and OpenAI lead in synthetic coding benchmarks.

**Open-weight leadership.** Meta’s Llama 3.1 405B and Databricks’ DBRX provide high performance with customization freedom.

**Long-context breakthroughs.** Gemini remains unmatched in context size, benefiting large RAG and agent workflows.

**Architectural diversity.** MoE adoption is rising (DBRX, Qwen2.5-Max, Phi-3.5 MoE) for efficiency; reasoning-tuned lines (o-series, Grok-3, DeepSeek-R1) show clear benchmark gains.
    """, search))

    with st.expander("At-a-glance — who leads where? (illustrative)"):
        lead = pd.DataFrame({
            "Axis": ["Reasoning","Coding","Open-weight","Long-context","Efficiency (MoE)"],
            "Leader": ["OpenAI/Google","Claude 3.5 / Gemini / OpenAI","Meta / Databricks","Gemini","DBRX / Qwen2.5-Max / Phi-3.5-MoE"],
        })
        st.table(lead)
        toy = pd.DataFrame({
            "Category": ["Reasoning","Coding","Open-weight","Long-context","MoE momentum"],
            "Relative lead (toy)": [95, 85, 88, 99, 80]
        })
        fig = px.bar(toy, x="Category", y="Relative lead (toy)")
        fig.update_layout(height=320, yaxis_title="Relative index")
        st.plotly_chart(fig, use_container_width=True)

    md_card("""
<span class="hint">💡 Implication:</span> Choose models by **use-case fit**:  
- Mission-critical reasoning → prioritize **o-series / Gemini**;  
- Repo-level code repair → **Claude 3.5 Sonnet**;  
- Private customization → **Llama 3.1 405B / DBRX**;  
- Long-context RAG/agents → **Gemini**;  
- Inference efficiency at scale → **MoE** families (DBRX, Qwen2.5-Max, Phi-3.5-MoE).
    """)

# ---------------------------
# ---- References & Export --
# ---------------------------
if nav == "References & Exports":
    section_header("References & Exports")
    ref_md = """
- Vaswani et al., 2017 — Attention Is All You Need  
- Brown et al., 2020 — Language Models are Few-Shot Learners (GPT-3)  
- Fedus et al., 2021 — Switch Transformer (MoE)  
- Radford et al., 2021 — CLIP  
- Alayrac et al., 2022 — Flamingo  
- Press et al., 2022 — Train Short, Test Long  
- Dao et al., 2023 — FlashAttention / long-context advances  
- Zelikman et al., 2024 — Process supervision for reasoning  
- Hendrycks et al., 2021 — MMLU  
- Khandelwal et al., 2023 — Retrieval/context dilution observations  
- Reuel et al., 2025 — Benchmark inflation critique  
- Selected model families: GPT-4/4o, Claude 3.x/3.5, Gemini 1.5/2.5, Llama 3, DBRX, Qwen2.5, Phi-3.5-MoE, Grok-3, Reka Core
"""
    md_card(ref_md)

    st.markdown("### Export")
    full_text = """
## Literature Review (LLMs)

### 1. Introduction
Since 2023, LLM development has accelerated due to innovations in architecture (dense Transformers, Mixture-of-Experts, reasoning-tuned models), scaling strategies, and multi-modal integration. Competitive pressures have driven significant improvements in reasoning (AIME, GPQA), coding (LiveCodeBench, SWE-bench), context (up to 2M tokens), and open-weight availability. This paper compares leading LLM families across specs and empirical performance.

### Methodology (2.1–2.3)
Model selection across proprietary and open-weight families; evaluation criteria covering architecture, modalities, context, licensing, and benchmarks; data sources with flags for test-time compute.

### Results (3.1–3.2)
Technical overview table; benchmark tabs for Reasoning, Coding, Knowledge, and Long-Context.

### 2.x Literature Review
Architectures (dense vs MoE), reasoning-tuned training, multimodal integration, long-context advances, benchmarking practices, open vs proprietary trends.

### Discussion
Leaders by domain, open-weight strengths, long-context breakthroughs, and the rise of MoE and reasoning-tuned lines.
"""
    st.download_button("⬇️ Download summary as Markdown", data=full_text, file_name="llm_literature_review_summary.md", mime="text/markdown")

    st.caption(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")

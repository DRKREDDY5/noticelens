"""NoticeLens Streamlit product UI over the frozen Phase 5 RAG core."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="NoticeLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)

from noticelens.notice_input import extract_pdf_text  # noqa: E402
from noticelens.phase5 import create_live_core, load_phase5_secrets  # noqa: E402
from noticelens.phase5_1 import verify_phase51_frozen_inputs  # noqa: E402
from noticelens.streamlit_ui import (  # noqa: E402
    DEFAULT_ANALYSIS_QUESTION,
    SUGGESTED_QUESTIONS,
    ProductSnapshot,
    SampleNotice,
    ask_notice,
    build_evidence_cards,
    build_trace_rows,
    evidence_status,
    extract_uploaded_notice,
    load_product_snapshot,
    load_sample_notices,
    notice_detail_rows,
    safe_error_message,
)


APP_CSS = """
<style>
:root {
  --nl-bg: #08111A;
  --nl-sidebar: #0C1620;
  --nl-card: #111D28;
  --nl-elevated: #172735;
  --nl-primary: #49D7C7;
  --nl-secondary: #75A7FF;
  --nl-success: #7CE38B;
  --nl-warning: #F2C75C;
  --nl-danger: #FF7373;
  --nl-text: #F6F8FA;
  --nl-muted: #94A3B8;
}
.stApp { background: var(--nl-bg); color: var(--nl-text); }
[data-testid="stSidebar"] { background: var(--nl-sidebar); border-right: 1px solid #203241; }
[data-testid="stHeader"] { background: rgba(8, 17, 26, 0.88); }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3, h4, p, label, [data-testid="stCaptionContainer"] { color: var(--nl-text); }
.nl-kicker { color: var(--nl-primary); font-size: .78rem; font-weight: 800; letter-spacing: .18em; }
.nl-hero {
  padding: 2.1rem 2.2rem;
  border-radius: 24px;
  border: 1px solid rgba(73, 215, 199, .24);
  background: linear-gradient(135deg, rgba(23,39,53,.98), rgba(12,22,32,.96));
  box-shadow: 0 24px 70px rgba(0,0,0,.25);
  margin-bottom: 1.25rem;
}
.nl-hero h1 { margin: .2rem 0 .35rem; font-size: clamp(2.25rem, 5vw, 4.25rem); letter-spacing: -.05em; }
.nl-hero .tagline { font-size: clamp(1.05rem, 2vw, 1.45rem); color: var(--nl-text); margin: 0 0 .6rem; }
.nl-hero .support { color: var(--nl-muted); max-width: 760px; margin: 0; }
.nl-card {
  padding: 1.1rem 1.2rem;
  border: 1px solid #203746;
  background: var(--nl-card);
  border-radius: 16px;
  min-width: 0;
}
.nl-card-title { color: var(--nl-muted); font-size: .72rem; font-weight: 800; letter-spacing: .12em; margin-bottom: .35rem; }
.nl-card-value { color: var(--nl-text); font-size: 1rem; overflow-wrap: anywhere; }
.nl-status { display: inline-flex; align-items: center; gap: .5rem; padding: .42rem .72rem; border-radius: 999px; font-size: .75rem; font-weight: 850; letter-spacing: .08em; }
.nl-grounded { color: var(--nl-success); background: rgba(124,227,139,.12); border: 1px solid rgba(124,227,139,.28); }
.nl-insufficient { color: var(--nl-warning); background: rgba(242,199,92,.12); border: 1px solid rgba(242,199,92,.28); }
.nl-unclear { color: var(--nl-danger); background: rgba(255,115,115,.12); border: 1px solid rgba(255,115,115,.28); }
.nl-section-label { color: var(--nl-primary); font-size: .75rem; font-weight: 850; letter-spacing: .12em; margin-top: 1.2rem; }
.nl-architecture { padding: 1rem; border-radius: 14px; background: var(--nl-card); border: 1px solid #203746; color: var(--nl-muted); line-height: 1.8; text-align: center; overflow-wrap: anywhere; }
.nl-architecture strong { color: var(--nl-text); }
[data-testid="stMetric"] { background: var(--nl-card); border: 1px solid #203746; padding: .85rem 1rem; border-radius: 14px; min-width: 0; }
[data-testid="stMetricValue"] { color: var(--nl-text); }
[data-testid="stFileUploader"] { background: var(--nl-card); border-radius: 16px; padding: .4rem; }
[data-testid="stDataFrame"] { border: 1px solid #203746; border-radius: 14px; overflow: auto; max-width: 100%; }
[data-testid="stChatMessage"] { background: rgba(17,29,40,.74); border: 1px solid #203746; border-radius: 16px; padding: .35rem .65rem; }
[data-baseweb="tab-list"] { gap: .5rem; }
[data-baseweb="tab"] { background: var(--nl-card); border-radius: 12px 12px 0 0; padding-inline: 1rem; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--nl-primary); border-bottom-color: var(--nl-primary); }
.stButton > button, .stLinkButton > a { border-radius: 12px; border-color: rgba(73,215,199,.5); }
.stButton > button[kind="primary"] { background: var(--nl-primary); color: #041313; font-weight: 800; border: 0; }
a { color: var(--nl-secondary) !important; }
@media (max-width: 640px) {
  .block-container { padding: 1rem .8rem 5rem; max-width: 100%; overflow-x: hidden; }
  .nl-hero { padding: 1.35rem 1.1rem; border-radius: 18px; }
  .nl-hero h1 { font-size: 2.25rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .65rem; }
  [data-testid="column"] { min-width: min(100%, 19rem) !important; width: 100% !important; flex: 1 1 100% !important; }
  [data-testid="stChatInput"] { width: 100%; }
  [data-baseweb="tab-list"] { overflow-x: auto; }
  .nl-architecture { text-align: left; }
}
</style>
"""


@st.cache_data(show_spinner=False)
def public_snapshot() -> ProductSnapshot:
    return load_product_snapshot(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def public_samples() -> tuple[SampleNotice, ...]:
    return load_sample_notices(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def live_core() -> Any:
    """Build the approved read-only RAG core without caching secrets separately."""

    verify_phase51_frozen_inputs(PROJECT_ROOT)
    snapshot = load_product_snapshot(PROJECT_ROOT)
    secrets = load_phase5_secrets(project_root=PROJECT_ROOT)
    core, selection = create_live_core(project_root=PROJECT_ROOT, secrets=secrets)
    if selection.selected_model != snapshot.final_config["generation_model"]:
        raise RuntimeError("The live generation model differs from the frozen product configuration")
    return core


def _initialize_state() -> None:
    defaults = {
        "active_notice": None,
        "active_label": None,
        "analysis_run": None,
        "latest_run": None,
        "latest_query": None,
        "chat_messages": [],
        "ui_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _status_badge(status: str) -> None:
    css_class = {
        "GROUNDED": "nl-grounded",
        "INSUFFICIENT EVIDENCE": "nl-insufficient",
        "NOTICE IDENTITY UNCLEAR": "nl-unclear",
    }[status]
    st.markdown(f'<span class="nl-status {css_class}">{status}</span>', unsafe_allow_html=True)


def _render_notice_details(run: Any) -> None:
    st.markdown('<div class="nl-section-label">NOTICE DETAILS</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, notice_detail_rows(run)):
        with column:
            st.metric(label, value)


def _render_evidence(run: Any, *, compact: bool = False) -> None:
    try:
        cards = build_evidence_cards(run)
    except Exception as error:
        st.warning(safe_error_message(error))
        return
    if not compact:
        st.markdown("### Official IRS evidence")
    if not cards:
        st.caption("No official guidance citation was available for this response.")
        return
    for index, card in enumerate(cards, start=1):
        label = f"Evidence {index} · {card.notice_code} · {card.heading}"
        with st.expander(label, expanded=False):
            st.caption(card.source_title)
            st.markdown(f"**Heading:** {' › '.join(card.heading_path)}")
            st.markdown(f"> {card.evidence_excerpt}")
            st.link_button("Open official IRS source", card.source_url)


def _render_response(run: Any) -> None:
    status = evidence_status(run)
    _status_badge(status)
    st.markdown("#### Notice identified")
    st.markdown(f"### {run.identity.notice_code or 'Not confidently identified'}")

    st.markdown('<div class="nl-section-label">WHAT THIS NOTICE IS ABOUT</div>', unsafe_allow_html=True)
    st.write(run.response.answer)

    notice_claims = [claim.text for claim in run.response.claims if claim.evidence_type == "notice"]
    guidance_claims = [claim.text for claim in run.response.claims if claim.evidence_type == "guidance"]

    st.markdown('<div class="nl-section-label">WHAT YOUR NOTICE STATES</div>', unsafe_allow_html=True)
    if notice_claims:
        for claim in notice_claims:
            st.write(claim)
    else:
        st.caption("No additional notice-specific field was needed for this explanation.")

    st.markdown('<div class="nl-section-label">WHAT IRS GUIDANCE SAYS</div>', unsafe_allow_html=True)
    if guidance_claims:
        for claim in guidance_claims:
            st.write(claim)
    else:
        st.caption("No supported IRS guidance claim was available for this response.")

    _render_notice_details(run)
    _render_evidence(run)


def _run_analysis(*, mode: str, sample: SampleNotice | None, upload: Any | None) -> None:
    st.session_state.ui_error = None
    try:
        with st.status("Analyzing notice", expanded=True) as status:
            status.write("Preparing the frozen NoticeLens RAG core")
            core = live_core()
            status.write("Preparing the frozen NoticeLens RAG core — complete")
            status.write("Reading notice")
            if mode == "Try an official sample":
                if sample is None:
                    raise ValueError("No sample was selected")
                extracted = extract_pdf_text(sample.path)
                active_label = sample.notice_code
            else:
                if upload is None:
                    raise ValueError("No PDF was uploaded")
                extracted = extract_uploaded_notice(upload.getvalue(), upload.name)
                active_label = "Uploaded notice"
            status.write("Reading notice — complete")
            status.write("Identifying notice · Retrieving IRS guidance · Generating evidence-backed explanation")
            run = core.run_extracted(extracted, DEFAULT_ANALYSIS_QUESTION)
            status.write("Identifying notice — complete")
            if run.documents:
                status.write("Retrieving IRS guidance — complete")
            if run.response.status == "answered":
                status.write("Generating evidence-backed explanation — complete")
            elif run.response.status == "refused":
                status.write("Generation skipped — insufficient verified evidence")
            else:
                status.write("Retrieval and generation skipped — notice identity unclear")
            status.update(label="Notice analysis complete", state="complete", expanded=False)
        st.session_state.active_notice = extracted
        st.session_state.active_label = active_label
        st.session_state.analysis_run = run
        st.session_state.latest_run = run
        st.session_state.latest_query = DEFAULT_ANALYSIS_QUESTION
        st.session_state.chat_messages = []
    except Exception as error:
        st.session_state.active_notice = None
        st.session_state.active_label = None
        st.session_state.analysis_run = None
        st.session_state.latest_run = None
        st.session_state.latest_query = None
        st.session_state.chat_messages = []
        st.session_state.ui_error = safe_error_message(error)


def _run_chat(question: str) -> None:
    active_notice = st.session_state.active_notice
    if active_notice is None:
        return
    st.session_state.chat_messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Retrieving IRS guidance and generating a grounded response…"):
            run = ask_notice(live_core(), active_notice, question)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": run.response.answer, "run": run}
        )
        st.session_state.latest_run = run
        st.session_state.latest_query = question
    except Exception as error:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": safe_error_message(error), "error": True}
        )


def _render_sidebar(snapshot: ProductSnapshot) -> None:
    with st.sidebar:
        st.markdown('<div class="nl-kicker">NOTICELENS</div>', unsafe_allow_html=True)
        st.caption("Evidence-grounded IRS notice intelligence")
        st.divider()
        st.metric("Corpus", f"{snapshot.corpus_documents} IRS guidance documents")
        st.metric("Sample notices", snapshot.sample_notices)
        st.metric("Final chunking", "Heading-aware")
        st.metric("Faithfulness", f"{snapshot.faithfulness:.0%}")
        st.divider()
        st.caption("NoticeLens is not a tax advisor.")


def _render_analyze_tab(samples: tuple[SampleNotice, ...]) -> None:
    st.markdown(
        """
        <section class="nl-hero">
          <div class="nl-kicker">DOCUMENT INTELLIGENCE FOR IRS NOTICES</div>
          <h1>NOTICELENS</h1>
          <p class="tagline">Understand the IRS notice in front of you — with evidence.</p>
          <p class="support">Turn a confusing IRS notice into a clear, source-backed explanation.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info("NoticeLens is not a tax advisor. It explains what the notice states, what official IRS guidance says, and what evidence supports the explanation.")

    mode = st.radio(
        "Notice source",
        ("Try an official sample", "Upload a notice"),
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_sample: SampleNotice | None = None
    upload = None
    if mode == "Try an official sample":
        selected_label = st.selectbox("Official IRS sample", [sample.label for sample in samples])
        selected_sample = next(sample for sample in samples if sample.label == selected_label)
        st.link_button("View sample source on IRS.gov", selected_sample.source_url)
    else:
        st.warning(
            "IRS notices may contain sensitive information. For this project demo, use official samples or appropriately redacted notices."
        )
        st.caption("Uploaded notices are processed temporarily for this session and are not permanently stored. PII filtering is heuristic, not comprehensive.")
        upload = st.file_uploader("Upload a PDF notice", type=["pdf"], accept_multiple_files=False)

    if st.button(
        "Analyze notice",
        type="primary",
        width="stretch",
        disabled=mode == "Upload a notice" and upload is None,
    ):
        _run_analysis(mode=mode, sample=selected_sample, upload=upload)

    if st.session_state.ui_error:
        st.error(st.session_state.ui_error)
    if st.session_state.analysis_run is not None:
        st.divider()
        _render_response(st.session_state.analysis_run)


def _render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("run") is not None:
                _status_badge(evidence_status(message["run"]))
                _render_evidence(message["run"], compact=True)


def _render_chat_tab() -> None:
    if st.session_state.active_notice is None or st.session_state.analysis_run is None:
        st.info("Analyze a notice first to ask evidence-grounded questions.")
        return
    if st.session_state.analysis_run.identity.status != "identified":
        st.warning("Notice identity must be clear before evidence-grounded questions can be asked.")
        return
    st.markdown(f"### Ask about {st.session_state.analysis_run.identity.notice_code or 'this notice'}")
    st.caption("Every answer runs through the same notice-filtered retrieval and grounded-generation graph.")
    _render_chat_history()

    st.markdown("**Suggested questions**")
    columns = st.columns(2)
    pending: str | None = None
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with columns[index % 2]:
            if st.button(question, key=f"suggestion_{index}", width="stretch"):
                pending = question
    typed = st.chat_input("Ask an evidence-grounded question about this notice")
    if typed:
        pending = typed
    if pending:
        _run_chat(pending)
        st.rerun()


def _configuration_rows(snapshot: ProductSnapshot) -> list[dict[str, str]]:
    config = snapshot.final_config
    return [
        {"Setting": "Embedding", "Value": str(config["embedding_model"])},
        {"Setting": "Embedding dimension", "Value": str(config["embedding_dimension"])},
        {"Setting": "Chunking", "Value": "Heading-aware"},
        {"Setting": "Vector store", "Value": "Pinecone"},
        {"Setting": "Retrieval", "Value": "Dense cosine similarity"},
        {"Setting": "Top K", "Value": str(config["top_k"])},
        {"Setting": "Generation", "Value": str(config["generation_model"])},
    ]


def _render_trace(run: Any) -> None:
    st.markdown("### Retrieval trace")
    try:
        rows = build_trace_rows(run)
    except Exception as error:
        st.warning(safe_error_message(error))
        return
    if not rows:
        st.caption("No guidance chunks were retrieved for this response.")
        return
    table = [
        {
            "Rank": row["Rank"],
            "Notice code": row["Notice code"],
            "Heading": row["Heading"],
            "Similarity score": f'{row["Similarity"]:.6f}',
            "Source": row["Source"],
        }
        for row in rows
    ]
    st.dataframe(table, width="stretch", hide_index=True, height=240)
    for row in rows:
        with st.expander(f'Rank {row["Rank"]} · {row["Notice code"]} · {row["Heading"]}'):
            st.text(row["Preview"])
            st.link_button("Open official IRS source", row["Source URL"], key=f'trace_{row["Chunk ID"]}')


def _render_xray_tab(snapshot: ProductSnapshot) -> None:
    st.markdown("## RAG X-Ray")
    current_notice = (
        st.session_state.latest_run.identity.notice_code
        if st.session_state.latest_run is not None
        else None
    )
    current_columns = st.columns(2)
    current_columns[0].metric("Current notice", current_notice or "No notice analyzed")
    current_columns[1].metric("Query", st.session_state.latest_query or "No query yet")

    st.markdown("### RAG configuration")
    st.dataframe(_configuration_rows(snapshot), width="stretch", hide_index=True)
    st.markdown(
        """
        <div class="nl-architecture">
          <strong>PDF notice</strong> → deterministic identity and fields →
          <strong>Qwen3 embedding</strong> → exact notice-code filter →
          <strong>Pinecone heading-aware dense retrieval</strong> →
          <strong>LangGraph evidence gates</strong> → grounded Qwen3 answer → claim-level IRS citations
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Python · LangChain · LangGraph · Nebius Token Factory · Pinecone · Streamlit · Jupyter")

    if st.session_state.latest_run is not None:
        _render_trace(st.session_state.latest_run)

    st.markdown("### How we chose the retriever")
    notice_columns = st.columns(3)
    notice_columns[0].metric("Dense notice P@1", f"{snapshot.notice_dense_p1:.0%}")
    notice_columns[1].metric("Notice MRR", f"{snapshot.notice_dense_mrr:.3f}")
    notice_columns[2].metric("Notice Hit@5", f"{snapshot.notice_dense_hit5:.0%}")
    section_columns = st.columns(3)
    section_columns[0].metric("Fixed-size section P@1", f"{snapshot.fixed_section_p1:.2%}")
    section_columns[1].metric("Heading-aware section P@1", f"{snapshot.heading_section_p1:.2%}")
    section_columns[2].metric("Absolute improvement", f"+{snapshot.heading_gain_percentage_points:.2f} pp")

    st.markdown("#### Retrieval ablation")
    st.dataframe(
        [
            {"Retriever": label, "Section Precision@1": f"{value:.2%}"}
            for label, value in snapshot.ablation_p1.items()
        ],
        width="stretch",
        hide_index=True,
    )
    st.write(
        "BM25, hybrid search, and reranking were tested. None produced a meaningful Section Precision@1 improvement over heading-aware dense retrieval, so the simpler retriever was retained."
    )
    st.caption("This experiment is specific to the frozen NoticeLens corpus and benchmark; it is not proof that one retrieval technique is universally better.")

    st.markdown("### Quality metrics")
    quality_columns = st.columns(3)
    quality_columns[0].metric("Faithfulness", f"{snapshot.faithfulness:.0%}")
    quality_columns[1].metric("Correct refusal", f"{snapshot.refusal_rate:.0%}")
    quality_columns[2].metric("Latency target", f"<{snapshot.latency_target_seconds:.0f}s")
    latency_columns = st.columns(2)
    latency_columns[0].metric("Warm end-to-end median", f"{snapshot.warm_median_seconds:.3f}s")
    latency_columns[1].metric("Warm end-to-end p95", f"{snapshot.warm_p95_seconds:.3f}s")
    st.warning("LATENCY TARGET NOT MET. The measured warm p95 is 22.050 seconds against a strict target below 6 seconds.")

    st.markdown("### Jupyter lab")
    st.code("notebooks/noticelens_rag_lab.ipynb", language=None)
    st.caption("The full experiment notebook remains a separate local artifact and is not embedded in this application.")


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    _initialize_state()
    try:
        snapshot = public_snapshot()
        samples = public_samples()
    except Exception as error:
        st.error(safe_error_message(error))
        return
    _render_sidebar(snapshot)
    analyze_tab, chat_tab, xray_tab = st.tabs(("Analyze Notice", "Ask NoticeLens", "RAG X-Ray"))
    with analyze_tab:
        _render_analyze_tab(samples)
    with chat_tab:
        _render_chat_tab()
    with xray_tab:
        _render_xray_tab(snapshot)


if __name__ == "__main__":
    main()

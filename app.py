"""AgriAssist-RX: Streamlit research interface.

Developed by Muhammad Usman Shah. Reuses the existing six-route LangGraph agent.
"""

from __future__ import annotations

import hashlib
import html
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="AgriAssist-RX | Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent_csv import app as agent  # noqa: E402

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0B1514; --panel:#12221E; --line:#294039; --muted:#9CAE9F; --ivory:#EFF4E8; --lime:#B7E478; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
    background: radial-gradient(ellipse 70% 42% at 82% -4%, rgba(87,134,79,.17), transparent 78%),
                radial-gradient(ellipse 50% 40% at 9% 52%, rgba(28,95,81,.08), transparent 82%),
                var(--bg);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; }
.block-container { max-width:1410px; padding:1.8rem 2.7rem 3.5rem; }
[data-testid="stSidebar"] { background:#0D1A17; border-right:1px solid #2A3D35; }
[data-testid="stSidebar"] > div:first-child { padding-top:2rem; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; letter-spacing:-.038em !important; }
button, input, textarea { font-family:'DM Sans',sans-serif !important; }
[data-testid="stSidebar"] hr {border-color:#2A4237;}
/* Brand and sidebar */
.brand-lockup {display:flex;align-items:center;gap:12px;margin:.1rem 0 2.1rem;}
.brand-icon {display:grid;place-items:center;width:46px;height:46px;border-radius:14px;background:#B7E478;color:#13251E;font-size:25px;font-weight:800;}
.brand-title {font:800 1.14rem 'Manrope',sans-serif;letter-spacing:-.04em;color:#F3F6EE;}
.brand-sub {font-size:.66rem;letter-spacing:.16em;color:#91AB9B;margin-top:3px;}
.side-section {font-size:.68rem;letter-spacing:.17em;color:#8BA394;font-weight:800;margin:2rem 0 .7rem;}
.side-nav {display:flex;align-items:center;gap:12px;border:1px solid transparent;border-radius:11px;padding:10px 11px;margin:.22rem 0;color:#C5D4C8;font-size:.88rem;}
.side-nav.active {background:#1B342A;border-color:#315343;color:#E8F5DF;font-weight:700;}
.side-icon {display:inline-block;width:20px;text-align:center;opacity:.9;}
.side-credit {background:#182D25;border:1px solid #2C493B;border-radius:14px;padding:16px;margin-top:1.5rem;}
.side-credit small {color:#A2B6A6;letter-spacing:.11em;font-size:.66rem;}
.side-credit strong {color:#F1F6E9;display:block;margin-top:7px;font-size:.92rem;}
/* Header / hero */
.topline {display:flex;align-items:center;justify-content:space-between;margin:.1rem 0 1.3rem;}
.topline .crumb {color:#91A89B;font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;}
.topline .status {color:#CCE9BB;font-size:.72rem;border:1px solid #355A44;background:#183126;padding:7px 11px;border-radius:99px;}
.hero {position:relative;overflow:hidden;min-height:260px;border-radius:24px;border:1px solid #365146;
 background:linear-gradient(115deg,#1D382F 0%,#1A332A 46%,#203E31 100%);padding:38px 42px;margin-bottom:1.3rem;}
.hero:before {content:"";position:absolute;width:480px;height:480px;right:-88px;top:-105px;border:1px solid rgba(191,229,160,.19);border-radius:50%;box-shadow:0 0 0 48px rgba(191,229,160,.035),0 0 0 98px rgba(191,229,160,.035),0 0 0 150px rgba(191,229,160,.03);}
.hero:after {content:"";position:absolute;right:80px;top:45px;width:155px;height:155px;border-radius:60% 0 60% 0;transform:rotate(-26deg);background:linear-gradient(145deg,rgba(190,238,139,.27),rgba(120,179,114,.04));border:1px solid rgba(195,236,165,.25);}
.hero .eyebrow {font-size:.68rem;letter-spacing:.2em;color:#BDE997;font-weight:800;position:relative;z-index:1;margin-bottom:14px;}
.hero h1 {font:800 clamp(2rem,3.4vw,3.6rem)/1.15 'Manrope',sans-serif !important;margin:0;max-width:750px;color:#F5F8EE;position:relative;z-index:1;letter-spacing:-.06em !important;}
.hero h1 em {font-style:normal;color:#B7E478;}
.hero p {max-width:610px;position:relative;z-index:1;color:#C6D8C8;font-size:.97rem;line-height:1.75;margin-top:18px;margin-bottom:0;}
/* Summary bar */
.summary {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:.25rem 0 1.7rem;}
.summary-item {background:#14241E;border:1px solid #2C4337;border-radius:14px;padding:17px 20px;min-height:86px;}
.summary-label {font-size:.69rem;color:#94AD9D;letter-spacing:.12em;text-transform:uppercase;font-weight:700;margin-bottom:7px;}
.summary-value {font:750 1rem 'Manrope',sans-serif;color:#E9F4E6;}
.summary-value i {color:#B7E478;font-style:normal;}
/* workspace */
.workspace-title {font:750 1.34rem 'Manrope',sans-serif;color:#EAF2E7;letter-spacing:-.04em;margin:1.8rem 0 .3rem;}
.workspace-sub {color:#9FB2A5;font-size:.86rem;margin-bottom:1.25rem;}
.stTabs [data-baseweb="tab-list"] {background:#101F1B;border:1px solid #2B4237;border-radius:13px;padding:5px;gap:6px;margin-bottom:1.1rem;}
.stTabs [data-baseweb="tab"] {border-radius:9px;padding:10px 20px;color:#A9BCAD;font-weight:700;font-size:.89rem;}
.stTabs [aria-selected="true"] {background:#2B4B37 !important;color:#EAF8E0 !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {display:none;}
.panel-header {border-bottom:1px solid #2B4138;padding-bottom:1rem;margin:.45rem 0 1rem;}
.panel-header strong {font:750 1.17rem 'Manrope',sans-serif;color:#ECF4E9;}
.panel-header p {color:#9DB3A5;font-size:.86rem;margin:7px 0 0;}
/* interactive controls */
[data-testid="stChatMessage"] {background:#14251F;border:1px solid #2D493C;border-radius:15px;padding:1rem 1.15rem;margin-bottom:.78rem;}
[data-testid="stChatInput"] {background:#172922;border-radius:16px;border:1px solid #42674E;}
[data-testid="stFileUploader"] {background:#14251E;border:1px dashed #53765B;border-radius:17px;padding:.85rem;}
[data-testid="stFileUploader"] section {background:transparent;}
.stButton > button {border-radius:11px;min-height:43px;font-weight:700;}
.stButton > button[kind="primary"] {background:#B7E478;color:#102019;border:1px solid #B7E478;}
.stButton > button[kind="primary"]:hover {background:#C9F38D;color:#102019;border-color:#C9F38D;}
[data-testid="stProgress"] > div > div > div {background:#B7E478;}
[data-testid="stAlert"] {border-radius:12px;}
.suggestion {border:1px solid #31493A;border-radius:13px;padding:14px 16px;background:#15261F;color:#B9CBBE;font-size:.86rem;line-height:1.7;}
.empty-state {background:linear-gradient(120deg,#14251E,#14241F);border:1px solid #2D4639;border-radius:15px;padding:27px;margin-bottom:14px;}
.empty-state strong {display:block;color:#EFF6E9;font:750 1rem 'Manrope',sans-serif;margin-bottom:6px;}
.empty-state span {color:#A1B3A7;font-size:.88rem;}
.score-note {border-radius:12px;background:#1A3025;border:1px solid #365541;padding:12px 15px;color:#C2DDC2;font-size:.82rem;margin:12px 0;}
.bottom-credit {color:#849E8F;border-top:1px solid #2B4237;padding-top:1.2rem;margin-top:3rem;font-size:.79rem;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;}
.bottom-credit strong {color:#CBEBBE;}
@media(max-width:900px){.block-container{padding:1.4rem 1.2rem 2.2rem;}.hero{padding:28px 24px;min-height:230px;}.hero:after{opacity:.35}.summary{grid-template-columns:1fr;}.summary-item{min-height:65px;}}
@media(max-width:600px){.hero h1{font-size:2rem !important}.hero p{font-size:.87rem}.topline .status{display:none}.summary-item{padding:12px 15px}.bottom-credit{display:block}}
</style>
""",
    unsafe_allow_html=True,
)


def run_agent(question: str, image_path: str = "") -> dict:
    return agent.invoke({
        "question": question,
        "selected_tool": "",
        "calculation": "none",
        "arg1": None,
        "arg2": None,
        "answer": "",
        "sources": [],
        "supported": False,
        "image_path": image_path,
        "vision_prediction": {},
    })


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Evidence sources · {len(sources)}"):
        for source in sources:
            if isinstance(source, dict):
                name = html.escape(str(source.get("source", "Unknown document")))
                page = html.escape(str(source.get("page", "?")))
                sid = html.escape(str(source.get("id", "?")))
                st.markdown(f"**[{sid}] {name}**  · PDF page {page}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        route = message.get("tool")
        if route:
            st.caption(f"Agent route · {route}")
        render_sources(message.get("sources") or [])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vision_result" not in st.session_state:
    st.session_state.vision_result = None
if "vision_file_key" not in st.session_state:
    st.session_state.vision_file_key = None

with st.sidebar:
    st.markdown('''<div class="brand-lockup"><div class="brand-icon">✦</div><div><div class="brand-title">AgriAssist-RX</div><div class="brand-sub">AGRICULTURAL INTELLIGENCE</div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="side-section">YOUR WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav active"><span class="side-icon">◈</span> AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">◎</span> Wheat image screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">CAPABILITIES</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">▤</span> Document Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">☼</span> Live weather</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">∑</span> Farm calculations</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">▥</span> CSV analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav"><span class="side-icon">◉</span> Image classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">CONVERSATION</div>', unsafe_allow_html=True)
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<div class="side-credit"><small>DEVELOPED BY</small><strong>Muhammad Usman Shah</strong></div>', unsafe_allow_html=True)
    st.caption("Research prototype · Local execution")

st.markdown('<div class="topline"><div class="crumb">Workspace / Overview</div><div class="status">● &nbsp; Local research application</div></div>', unsafe_allow_html=True)
st.markdown('''<section class="hero"><div class="eyebrow">AGRICULTURAL AI · RESEARCH PLATFORM</div><h1>Better insights for<br><em>smarter agriculture.</em></h1><p>Meet AgriAssist-RX. Explore crop knowledge, local weather, farm calculations and data insights, or screen wheat photographs — all in one research workspace.</p></section>''', unsafe_allow_html=True)
st.markdown('''<div class="summary"><div class="summary-item"><div class="summary-label">Intelligence layer</div><div class="summary-value"><i>06</i> agent routes</div></div><div class="summary-item"><div class="summary-label">Image screening</div><div class="summary-value"><i>15</i> wheat conditions</div></div><div class="summary-item"><div class="summary-label">System architecture</div><div class="summary-value">LangGraph + RAG</div></div></div>''', unsafe_allow_html=True)
st.markdown('<div class="workspace-title">Your workspace</div><div class="workspace-sub">Choose a tool below to begin. Results are generated by your local agent and connected services.</div>', unsafe_allow_html=True)

chat_tab, image_tab = st.tabs(["◈  AI Assistant", "◎  Wheat image analysis"])

with chat_tab:
    st.markdown('<div class="panel-header"><strong>Ask AgriAssist</strong><p>Natural-language questions are routed to the right agricultural tool.</p></div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown('<div class="empty-state"><strong>What can I help you explore?</strong><span>Ask a crop question, check weather, calculate agricultural inputs or analyze CSV data.</span></div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown('<div class="suggestion"><b>Crop knowledge</b><br>What are the symptoms of wheat stem rust?</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="suggestion"><b>Weather intelligence</b><br>What is the weather in Lahore today?</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask an agricultural question…")
    if prompt and prompt.strip():
        prompt = prompt.strip()
        sent = {"role": "user", "content": prompt}
        st.session_state.messages.append(sent)
        render_message(sent)
        with st.chat_message("assistant"):
            with st.spinner("Working on your request…"):
                try:
                    outcome = run_agent(prompt)
                    answer = str(outcome.get("answer") or "No answer was returned.")
                    route = str(outcome.get("selected_tool") or "unknown")
                    sources = outcome.get("sources") or []
                except Exception as exc:
                    answer = "The request could not be completed. Please check the local setup and try again."
                    route, sources = "error", []
                    st.error(str(exc))
            st.markdown(answer)
            st.caption(f"Agent route · {route}")
            render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "tool": route, "sources": sources})

with image_tab:
    st.markdown('<div class="panel-header"><strong>Wheat condition screening</strong><p>Upload a photograph to review three ranked predictions from MobileNetV3-Small.</p></div>', unsafe_allow_html=True)
    upload_col, result_col = st.columns([1.05, .95], gap="large")
    with upload_col:
        st.markdown("**01 &nbsp; / &nbsp; Upload photograph**")
        uploaded = st.file_uploader(
            "Select a wheat photograph",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
            help="Maximum 25 MB. The trained model checkpoint must be available on this computer.",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            file_key = hashlib.sha256(image_bytes).hexdigest()
            if file_key != st.session_state.vision_file_key:
                st.session_state.vision_file_key = file_key
                st.session_state.vision_result = None
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                st.error("File exceeds the 25 MB limit.")
            else:
                st.image(image_bytes, caption=uploaded.name, use_container_width=True)
                if st.button("Analyze photograph", type="primary", use_container_width=True):
                    temp_path = None
                    try:
                        suffix = Path(uploaded.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as temporary:
                            temporary.write(image_bytes)
                            temp_path = Path(temporary.name)
                        with st.spinner("Running wheat screening model…"):
                            st.session_state.vision_result = run_agent(
                                f"Analyze this wheat image: {temp_path}", image_path=str(temp_path)
                            )
                    except Exception as exc:
                        st.session_state.vision_result = {"selected_tool": "error", "answer": str(exc)}
                    finally:
                        if temp_path is not None:
                            temp_path.unlink(missing_ok=True)
        else:
            st.session_state.vision_file_key = None
            st.session_state.vision_result = None
    with result_col:
        st.markdown("**02 &nbsp; / &nbsp; Model output**")
        result = st.session_state.vision_result
        if not result:
            st.markdown('<div class="empty-state"><strong>No analysis yet</strong><span>Upload a wheat photograph and select Analyze photograph to see model predictions here.</span></div>', unsafe_allow_html=True)
        else:
            prediction = result.get("vision_prediction") or {}
            top_three = prediction.get("predictions") or []
            route = result.get("selected_tool", "unknown")
            if route == "vision" and top_three:
                st.success("Screening completed")
                top_label = str(top_three[0].get("condition", "Unknown")).replace("_", " ").title()
                st.markdown(f"### {top_label}")
                st.caption("Highest-ranked model output — not a confirmed diagnosis")
                st.divider()
                for n, item in enumerate(top_three, 1):
                    label = str(item.get("condition", "Unknown")).replace("_", " ").title()
                    score = min(max(float(item.get("model_score", 0)), 0), 1)
                    st.markdown(f"**{n:02d} &nbsp; {label}**")
                    st.progress(score)
                    st.caption(f"Model score: {score * 100:.1f}%")
                if prediction.get("review_recommended", False):
                    st.warning("Review recommended: the top score is low or close to the second-ranked score.")
                else:
                    st.info("The top class is separated under the provisional screening rule.")
                st.markdown('<div class="score-note">This model is experimental. Scores are not calibrated probabilities, and unrelated photos can receive a wheat-condition prediction. Verify results before treatment decisions.</div>', unsafe_allow_html=True)
            else:
                st.error(str(result.get("answer") or "Image analysis could not be completed."))
    st.caption("Model evaluation is based on an internal grouped test split, not independent field validation.")

st.markdown('<div class="bottom-credit"><span>© 2026 AgriAssist-RX &nbsp;·&nbsp; Developed by <strong>Muhammad Usman Shah</strong></span><span>Research prototype &nbsp; / &nbsp; RAG · LangGraph · Computer Vision</span></div>', unsafe_allow_html=True)

"""
AutoStream Social-to-Lead Agent — Streamlit UI

Cyberpunk-themed chat interface that exposes:
    - Real-time intent classification badges
    - Progressive lead capture stepper
    - RAG source viewer for product answers
    - LangGraph state visualization
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import streamlit as st
import time
from agent import AutoStreamAgent

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoStream Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — Cyberpunk Dark Theme
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Dark Theme */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
    color: #e0e0e8;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
    border-right: 1px solid rgba(0, 240, 255, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #c0c0d0;
}

/* Chat input */
.stChatInput {
    border-color: rgba(0, 240, 255, 0.3) !important;
}

.stChatInput > div {
    background: rgba(15, 15, 30, 0.9) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    border-radius: 12px !important;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid transparent;
}

.stChatMessage:hover {
    transform: scale(1.005);
    border-color: rgba(0, 240, 255, 0.1);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
}

/* User message */
.stChatMessage[data-testid="stChatMessage-user"] {
    background: linear-gradient(135deg, rgba(0, 180, 255, 0.08), rgba(0, 220, 255, 0.04)) !important;
}

/* Assistant message */
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background: linear-gradient(135deg, rgba(180, 0, 255, 0.06), rgba(120, 0, 255, 0.03)) !important;
}

/* Intent Badge Styles */
.intent-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-align: center;
    margin: 4px 0;
    transition: all 0.3s ease;
}

.intent-greeting {
    background: rgba(0, 180, 255, 0.15);
    color: #00d4ff;
    border: 1px solid rgba(0, 180, 255, 0.3);
    box-shadow: 0 0 15px rgba(0, 180, 255, 0.1);
}

.intent-inquiry {
    background: rgba(255, 200, 0, 0.12);
    color: #ffc800;
    border: 1px solid rgba(255, 200, 0, 0.3);
    box-shadow: 0 0 15px rgba(255, 200, 0, 0.1);
}

.intent-high {
    background: rgba(255, 0, 128, 0.15);
    color: #ff4090;
    border: 1px solid rgba(255, 0, 128, 0.3);
    box-shadow: 0 0 15px rgba(255, 0, 128, 0.15);
    animation: pulse-glow 2s ease-in-out infinite;
}

.intent-lead {
    background: rgba(0, 255, 128, 0.12);
    color: #00ff80;
    border: 1px solid rgba(0, 255, 128, 0.3);
    box-shadow: 0 0 15px rgba(0, 255, 128, 0.1);
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(255, 0, 128, 0.15); }
    50% { box-shadow: 0 0 25px rgba(255, 0, 128, 0.35); }
}

/* Stepper */
.step-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    transition: all 0.3s ease;
}

.step-pending {
    background: rgba(255, 255, 255, 0.03);
    color: #555570;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.step-active {
    background: rgba(0, 240, 255, 0.08);
    color: #00f0ff;
    border: 1px solid rgba(0, 240, 255, 0.2);
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.08);
}

.step-done {
    background: rgba(0, 255, 128, 0.08);
    color: #00ff80;
    border: 1px solid rgba(0, 255, 128, 0.2);
}

/* Title styling */
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #b040ff, #ff4090);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}

.app-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #606080;
    margin-bottom: 20px;
}

/* Source viewer */
.source-box {
    background: rgba(0, 240, 255, 0.04);
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #80ffcc;
    margin-top: 8px;
    line-height: 1.6;
}

/* Lead captured celebration */
.lead-captured {
    background: linear-gradient(135deg, rgba(0, 255, 128, 0.08), rgba(0, 200, 255, 0.05));
    border: 1px solid rgba(0, 255, 128, 0.25);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
    animation: celebrate 0.6s ease-out;
}

@keyframes celebrate {
    0% { transform: scale(0.95); opacity: 0.5; }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); opacity: 1; }
}

/* Terminal output */
.terminal-output {
    background: #0a0a0a;
    border: 1px solid rgba(0, 255, 128, 0.2);
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #00ff80;
    line-height: 1.8;
    margin: 10px 0;
    box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.5);
}

/* Sidebar section headers */
.sidebar-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #606080;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.metric-label {
    color: #505068;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #c0c0e0;
    font-size: 14px;
    margin-top: 2px;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #00d4ff !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = AutoStreamAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intents" not in st.session_state:
    st.session_state.intents = []


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def get_intent_badge(intent: str) -> str:
    badges = {
        "greeting": '<div class="intent-badge intent-greeting">👋 Greeting</div>',
        "product_inquiry": '<div class="intent-badge intent-inquiry">🔍 Product Inquiry</div>',
        "high_intent": '<div class="intent-badge intent-high">🔥 High Intent</div>',
        "lead_info": '<div class="intent-badge intent-lead">📋 Lead Collection</div>',
    }
    return badges.get(intent, '<div class="intent-badge intent-inquiry">—</div>')


def get_step_html(label: str, value: str, status: str) -> str:
    if status == "done":
        icon = "✅"
        css_class = "step-done"
    elif status == "active":
        icon = "⏳"
        css_class = "step-active"
    else:
        icon = "○"
        css_class = "step-pending"

    display_value = f" — {value}" if value else ""
    return f'<div class="step-item {css_class}">{icon} {label}{display_value}</div>'


# ─────────────────────────────────────────────────────────────
# Sidebar: LangGraph Nerve Center
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="app-title">AutoStream</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Social-to-Lead Agentic Workflow</div>', unsafe_allow_html=True)

    agent_state = st.session_state.agent.get_state()

    # Intent Radar
    st.markdown('<div class="sidebar-header">Intent Classification</div>', unsafe_allow_html=True)
    current_intent = agent_state.get("intent")
    if current_intent:
        st.markdown(get_intent_badge(current_intent), unsafe_allow_html=True)
    else:
        st.markdown('<div class="intent-badge" style="background:rgba(255,255,255,0.03);color:#404060;border:1px solid rgba(255,255,255,0.05);">Awaiting Input</div>', unsafe_allow_html=True)

    # Lead Capture Progress
    st.markdown('<div class="sidebar-header">Lead Capture Progress</div>', unsafe_allow_html=True)

    name = agent_state.get("name")
    email = agent_state.get("email")
    platform = agent_state.get("platform")
    captured = agent_state.get("lead_captured", False)

    # Determine step statuses
    if captured:
        name_s, email_s, plat_s = "done", "done", "done"
    elif platform:
        name_s, email_s, plat_s = "done", "done", "done"
    elif email:
        name_s, email_s, plat_s = "done", "done", "active"
    elif name:
        name_s, email_s, plat_s = "done", "active", "pending"
    else:
        name_s, email_s, plat_s = "pending", "pending", "pending"
        if current_intent in ("high_intent", "lead_info"):
            name_s = "active"

    st.markdown(get_step_html("Name", name, name_s), unsafe_allow_html=True)
    st.markdown(get_step_html("Email", email, email_s), unsafe_allow_html=True)
    st.markdown(get_step_html("Platform", platform, plat_s), unsafe_allow_html=True)

    # Captured celebration
    if captured:
        st.markdown("""
        <div class="lead-captured">
            <div style="font-size:32px;margin-bottom:8px;">🎯</div>
            <div style="color:#00ff80;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;">
                LEAD CAPTURED
            </div>
            <div style="color:#608080;font-size:11px;margin-top:4px;">
                Sent to CRM
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Graph Architecture
    st.markdown('<div class="sidebar-header">LangGraph Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card" style="font-size:11px;line-height:1.8;color:#707090;">
        <span style="color:#00d4ff;">START</span> → classify_intent<br>
        &nbsp;&nbsp;→ <span style="color:#00d4ff;">greeting</span> → template<br>
        &nbsp;&nbsp;→ <span style="color:#ffc800;">inquiry</span> → RAG + LLM<br>
        &nbsp;&nbsp;→ <span style="color:#ff4090;">high_intent</span> → collect<br>
        &nbsp;&nbsp;→ <span style="color:#00ff80;">lead_info</span> → extract → capture<br>
        All → <span style="color:#00d4ff;">END</span>
    </div>
    """, unsafe_allow_html=True)

    # Conversation stats
    st.markdown('<div class="sidebar-header">Session Stats</div>', unsafe_allow_html=True)
    n_msgs = len(st.session_state.messages)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Messages</div>
        <div class="metric-value">{n_msgs}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="app-title" style="font-size:22px;margin-top:10px;">💬 AutoStream Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-Powered Sales Agent • LangGraph + Gemini</div>', unsafe_allow_html=True)

# Display message history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show RAG sources for assistant product inquiry responses
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📡 View RAG Sources"):
                for j, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""<div class="source-box">
                        <span style="color:#00d4ff;">Source {j}</span> — <span style="color:#ffc800;">{src}</span>
                    </div>""", unsafe_allow_html=True)

        # Show lead capture terminal for the capture message
        if msg["role"] == "assistant" and msg.get("lead_captured"):
            st.markdown(f"""<div class="terminal-output">
> mock_lead_capture() INITIATED...<br>
> PAYLOAD: {{"name": "{msg.get('lead_name','')}", "email": "{msg.get('lead_email','')}", "platform": "{msg.get('lead_platform','')}" }}<br>
> STATUS: <span style="color:#00ff80;">200 OK — SENT TO CRM ✓</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Chat Input Handler
# ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Message AutoStream..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner(""):
            # Show routing animation
            status_placeholder = st.empty()
            status_placeholder.markdown(
                '<div style="color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:12px;'
                'animation:pulse-glow 1s ease-in-out infinite;">'
                '⚡ Agent routing...</div>',
                unsafe_allow_html=True
            )

            response = st.session_state.agent.run(prompt)
            status_placeholder.empty()

        st.markdown(response)

        # Get updated state
        agent_state = st.session_state.agent.get_state()

        # Build message metadata
        msg_data = {
            "role": "assistant",
            "content": response,
            "sources": agent_state.get("rag_sources"),
        }

        # If lead was just captured, add terminal data
        if agent_state.get("lead_captured") and not any(
            m.get("lead_captured") for m in st.session_state.messages
        ):
            msg_data["lead_captured"] = True
            msg_data["lead_name"] = agent_state.get("name", "")
            msg_data["lead_email"] = agent_state.get("email", "")
            msg_data["lead_platform"] = agent_state.get("platform", "")

            # Show terminal flash
            st.markdown(f"""<div class="terminal-output">
> mock_lead_capture() INITIATED...<br>
> PAYLOAD: {{"name": "{agent_state.get('name','')}", "email": "{agent_state.get('email','')}", "platform": "{agent_state.get('platform','')}" }}<br>
> STATUS: <span style="color:#00ff80;">200 OK — SENT TO CRM ✓</span>
            </div>""", unsafe_allow_html=True)

        # Show RAG sources if present
        if agent_state.get("rag_sources"):
            with st.expander("📡 View RAG Sources"):
                for j, src in enumerate(agent_state["rag_sources"], 1):
                    st.markdown(f"""<div class="source-box">
                        <span style="color:#00d4ff;">Source {j}</span> — <span style="color:#ffc800;">{src}</span>
                    </div>""", unsafe_allow_html=True)

        st.session_state.messages.append(msg_data)

    st.rerun()

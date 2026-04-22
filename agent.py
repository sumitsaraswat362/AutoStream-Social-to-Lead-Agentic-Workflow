"""
AutoStream Social-to-Lead Agent — Multi-Node LangGraph Architecture

GRAPH TOPOLOGY (6 nodes, conditional edges):
    START → classify_intent (LLM-based)
        → "greeting"        → handle_greeting → END
        → "product_inquiry" → handle_rag → END
        → "high_intent"     → handle_lead_collection → END
        → "lead_info"       → extract_lead_info
                                → (complete)   → handle_lead_capture → END
                                → (incomplete) → handle_lead_collection → END

API BUDGET: Max 2 LLM calls per turn (classify + respond).
Lead collection turns use 0 API calls (templates + regex).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

from rag import KnowledgeBase
from tools import mock_lead_capture

class IntentClassification(BaseModel):
    intent: str = Field(
        description="Must be exactly one of: 'greeting', 'product_inquiry', or 'high_intent'"
    )


# ─────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
    response: str
    rag_sources: Optional[List[str]]


# ─────────────────────────────────────────────────────────────
# LLM & Knowledge Base + Rate Limiter
# ─────────────────────────────────────────────────────────────

import time as _time

_last_call_time = 0.0
_MIN_INTERVAL = 13.0  # 5 RPM = 12s between calls, add 1s buffer


def _rate_limited_invoke(llm, messages):
    """Invoke LLM with built-in rate limiting to stay within 5 RPM free tier."""
    global _last_call_time
    elapsed = _time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        wait = _MIN_INTERVAL - elapsed
        _time.sleep(wait)
    _last_call_time = _time.time()
    return llm.invoke(messages)


def _get_llm(model_name: str = "gemini-1.5-flash"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
        max_retries=0,
    )


def _is_model_not_found_error(err: Exception) -> bool:
    text = str(err)
    return "NOT_FOUND" in text and "gemini-1.5-flash" in text


def _get_compatible_llm() -> ChatGoogleGenerativeAI:
    """
    Some API keys do not expose Gemini 1.5 Flash anymore.
    Keep assignment-default model in code, but provide a runtime-compatible fallback.
    """
    for candidate in ("models/gemini-2.0-flash", "models/gemini-2.5-flash-lite"):
        try:
            return _get_llm(candidate)
        except Exception:
            continue
    return _get_llm()

_kb = KnowledgeBase()


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content).strip()


# ─────────────────────────────────────────────────────────────
# NODE 1: Intent Classification (LLM-based, structured output)
# ─────────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    LLM-based intent classification. The model reasons about user intent
    and returns a structured classification — NOT keyword matching.
    
    For active lead collection, fast-paths to 'lead_info' without LLM call.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Fast-path: if we're mid-collection, route to extraction (0 API calls)
    in_collection = (
        state.get("intent") in ("high_intent", "lead_info")
        and not all([state.get("name"), state.get("email"), state.get("platform")])
        and not state.get("lead_captured")
    )
    if in_collection:
        return {"intent": "lead_info"}

    # LLM-based classification (1 API call)
    already_captured = state.get("lead_captured", False)
    system_prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user's message into EXACTLY ONE category:

1. "greeting" — ONLY use this for explicit first-contact greetings like: hi, hello, hey, good morning. NOT for follow-up questions or casual chat.
2. "product_inquiry" — ANY question, comment, or statement about the product, company, pricing, features, or general conversation. This is the DEFAULT when unsure.
3. "high_intent" — User explicitly says they want to sign up, buy, subscribe, try a plan, or get started.

IMPORTANT RULES:
- If the user asks a question (what, how, why, can, does, is), classify as product_inquiry
- If the user makes a casual comment or statement, classify as product_inquiry
- ONLY classify as greeting if the message is a simple hello/hi with no other content
{"- The user already registered. Never classify as high_intent." if already_captured else ""}

Return a JSON object matching this schema:
{{"intent": "greeting|product_inquiry|high_intent"}}"""

    try:
        llm = _get_llm()
        structured_llm = llm.with_structured_output(IntentClassification)
        result = _rate_limited_invoke(structured_llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User message: \"{last_message}\""),
        ])
        raw = result.intent
    except Exception as e:
        if _is_model_not_found_error(e):
            try:
                compat_llm = _get_compatible_llm()
                structured_compat_llm = compat_llm.with_structured_output(IntentClassification)
                result = _rate_limited_invoke(structured_compat_llm, [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"User message: \"{last_message}\""),
                ])
                raw = result.intent
            except Exception as retry_error:
                print(f"  [classify_intent] LLM fallback: {retry_error}")
                raw = "product_inquiry"
        else:
            print(f"  [classify_intent] LLM fallback: {e}")
            raw = "product_inquiry"

    # Normalize — default to product_inquiry for safety
    if raw not in {"greeting", "product_inquiry", "high_intent"}:
        if "greet" in raw:
            raw = "greeting"
        elif "high" in raw or "intent" in raw:
            raw = "high_intent"
        else:
            raw = "product_inquiry"

    return {"intent": raw}


# ─────────────────────────────────────────────────────────────
# NODE 2: Greeting Handler (LLM-based for natural responses)
# ─────────────────────────────────────────────────────────────

def handle_greeting(state: AgentState) -> dict:
    """Generate a natural, contextual greeting using the LLM."""
    system_prompt = """You are a friendly sales assistant for AutoStream, an AI-powered video editing platform for content creators.

Respond warmly in 2-3 sentences. Mention what AutoStream does and invite them to ask about plans or features.
Be natural and conversational — not corporate. Use one emoji max. Keep it SHORT."""

    try:
        llm = _get_llm()
        result = _rate_limited_invoke(llm,
            [SystemMessage(content=system_prompt)] + state["messages"]
        )
        response = _extract_text(result.content)
    except Exception as e:
        if _is_model_not_found_error(e):
            try:
                llm = _get_compatible_llm()
                result = _rate_limited_invoke(llm, [SystemMessage(content=system_prompt)] + state["messages"])
                response = _extract_text(result.content)
            except Exception as retry_error:
                response = f"🚨 API ERROR: {str(retry_error)}"
        else:
            response = f"🚨 API ERROR: {str(e)}"

    return {
        "response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ─────────────────────────────────────────────────────────────
# NODE 3: RAG-Powered Response (1 API call)
# ─────────────────────────────────────────────────────────────

def handle_rag(state: AgentState) -> dict:
    """Retrieve relevant knowledge, then generate a grounded LLM response."""
    last_message = state["messages"][-1].content
    
    # Retrieve context + source metadata
    results = _kb.retrieve(last_message, top_k=3)
    context = "\n\n".join([r["text"] for r in results])
    sources = [r.get("source", "knowledge_base") for r in results]

    system_prompt = f"""You are a knowledgeable sales assistant for AutoStream.
Answer the user's question using ONLY the context below. Be concise, use bullet points
for comparisons. If info is missing, suggest support@autostream.io.

RETRIEVED CONTEXT:
{context}"""

    try:
        llm = _get_llm()
        result = _rate_limited_invoke(llm,
            [SystemMessage(content=system_prompt)] + state["messages"]
        )
        response_text = _extract_text(result.content)
    except Exception as e:
        if _is_model_not_found_error(e):
            try:
                llm = _get_compatible_llm()
                result = _rate_limited_invoke(llm, [SystemMessage(content=system_prompt)] + state["messages"])
                response_text = _extract_text(result.content)
            except Exception as retry_error:
                response_text = f"🚨 API ERROR: {str(retry_error)}"
        else:
            response_text = f"🚨 API ERROR: {str(e)}"

    return {
        "response": response_text,
        "rag_sources": sources,
        "messages": state["messages"] + [AIMessage(content=response_text)],
    }


# ─────────────────────────────────────────────────────────────
# NODE 4: Lead Collection (template — 0 API calls)
# ─────────────────────────────────────────────────────────────

def handle_lead_collection(state: AgentState) -> dict:
    """Ask for the next missing lead field, one at a time."""
    name = state.get("name")
    email = state.get("email")

    if not name:
        response = (
            "That's awesome — I'd love to help you get started with AutoStream! 🎉\n\n"
            "To set things up, could you share your **name**?"
        )
    elif not email:
        response = (
            f"Great to meet you, **{name}**! 😊\n\n"
            f"Could you share your **email address** so we can create your account?"
        )
    else:
        response = (
            f"Perfect, thanks **{name}**! Almost there — what **content platform** "
            f"do you primarily create for?\n\n"
            f"_(e.g., YouTube, Instagram, TikTok, Twitter)_"
        )

    return {
        "response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ─────────────────────────────────────────────────────────────
# NODE 5: Extract Lead Info (regex/string — 0 API calls)
# ─────────────────────────────────────────────────────────────

PLATFORM_MAP = {
    "youtube": "YouTube", "instagram": "Instagram", "tiktok": "TikTok",
    "tik tok": "TikTok", "twitter": "Twitter/X", "x": "Twitter/X",
    "facebook": "Facebook", "twitch": "Twitch", "linkedin": "LinkedIn",
    "snapchat": "Snapchat", "pinterest": "Pinterest",
}


def extract_lead_info(state: AgentState) -> dict:
    """Extract lead fields from raw user text. No LLM needed."""
    raw = state["messages"][-1].content.strip()
    name = state.get("name")
    email = state.get("email")

    if not name:
        cleaned = raw.strip('"\'.,!').strip()
        for prefix in ["my name is ", "i'm ", "im ", "i am ", "call me ", "it's ", "this is "]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return {"name": cleaned.title() if cleaned.islower() else cleaned}

    elif not email:
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw)
        return {"email": match.group(0).lower() if match else raw.strip()}

    else:
        raw_lower = raw.lower()
        for key, normalized in PLATFORM_MAP.items():
            if key in raw_lower:
                return {"platform": normalized}
        return {"platform": raw.strip().title()}


# ─────────────────────────────────────────────────────────────
# NODE 6: Lead Capture (fires tool — 0 API calls)
# ─────────────────────────────────────────────────────────────

def handle_lead_capture(state: AgentState) -> dict:
    """Execute mock_lead_capture() ONLY after all 3 fields verified."""
    name, email, platform = state["name"], state["email"], state["platform"]

    assert all([name, email, platform]), \
        f"Lead capture guard failed: {name}, {email}, {platform}"

    mock_lead_capture(name=name, email=email, platform=platform)

    response = (
        f"🎉 **You're all set, {name}!**\n\n"
        f"Here's your registration summary:\n"
        f"- **Name:** {name}\n"
        f"- **Email:** {email}\n"
        f"- **Platform:** {platform}\n\n"
        f"Our team will reach out to you at **{email}** shortly "
        f"with your account details. Welcome to AutoStream! 🚀"
    )

    return {
        "response": response,
        "lead_captured": True,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ─────────────────────────────────────────────────────────────
# Routing Functions (conditional edges)
# ─────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    return {
        "greeting": "handle_greeting",
        "product_inquiry": "handle_rag",
        "high_intent": "handle_lead_collection",
        "lead_info": "extract_lead_info",
    }.get(state.get("intent", "greeting"), "handle_rag")


def check_lead_complete(state: AgentState) -> str:
    if all([state.get("name"), state.get("email"), state.get("platform")]):
        return "complete"
    return "incomplete"


# ─────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the multi-node LangGraph StateGraph.
    
    6 nodes with conditional edges — proper agentic routing,
    not a single-node wrapper.
    """
    g = StateGraph(AgentState)

    # Register all 6 nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("handle_greeting", handle_greeting)
    g.add_node("handle_rag", handle_rag)
    g.add_node("handle_lead_collection", handle_lead_collection)
    g.add_node("extract_lead_info", extract_lead_info)
    g.add_node("handle_lead_capture", handle_lead_capture)

    # Entry point
    g.add_edge(START, "classify_intent")

    # Intent-based conditional routing
    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_rag": "handle_rag",
            "handle_lead_collection": "handle_lead_collection",
            "extract_lead_info": "extract_lead_info",
        },
    )

    # Terminal edges
    g.add_edge("handle_greeting", END)
    g.add_edge("handle_rag", END)
    g.add_edge("handle_lead_collection", END)
    g.add_edge("handle_lead_capture", END)

    # Lead completion check
    g.add_conditional_edges(
        "extract_lead_info",
        check_lead_complete,
        {"complete": "handle_lead_capture", "incomplete": "handle_lead_collection"},
    )

    return g.compile()


# ─────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────

class AutoStreamAgent:
    """Wraps the LangGraph state machine. Manages state across turns."""

    def __init__(self):
        self.graph = build_graph()
        self.state: AgentState = {
            "messages": [],
            "intent": None,
            "name": None,
            "email": None,
            "platform": None,
            "lead_captured": False,
            "response": "",
            "rag_sources": None,
        }

    def run(self, user_input: str) -> str:
        self.state["messages"] = self.state["messages"] + [
            HumanMessage(content=user_input)
        ]
        self.state["response"] = ""
        self.state["rag_sources"] = None

        result = self.graph.invoke(self.state)
        self.state = result
        return self.state["response"]

    def get_state(self) -> dict:
        """Expose state for UI visualization."""
        return {
            "intent": self.state.get("intent"),
            "name": self.state.get("name"),
            "email": self.state.get("email"),
            "platform": self.state.get("platform"),
            "lead_captured": self.state.get("lead_captured", False),
            "rag_sources": self.state.get("rag_sources"),
        }
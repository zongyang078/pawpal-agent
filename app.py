"""
PawPal+ Agent — Streamlit Chat Interface

A chat-based UI where users interact with the PawPal+ AI Agent
using natural language. The agent manages pet care schedules,
answers care questions, and provides proactive safety advice.
"""

import os

import streamlit as st

from agent import PawPalAgent
from pawpal_system import Owner


# --- Page config ---
st.set_page_config(page_title="PawPal+ Agent", page_icon="🐾", layout="centered")
st.title("🐾 PawPal+ Agent")
st.caption("AI-powered pet care assistant — chat naturally to manage your pets")


# --- Session state initialization ---
if "owner" not in st.session_state:
    # Try to load from saved data
    loaded = Owner.load_from_json()
    if loaded:
        st.session_state.owner = loaded
    else:
        st.session_state.owner = Owner(name="Pet Parent")

if "agent" not in st.session_state:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    model = "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o-mini"

    st.session_state.agent = PawPalAgent(
        owner=st.session_state.owner,
        api_key=api_key,
        api_provider=provider,
        model=model,
        use_llm=api_key is not None,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


agent: PawPalAgent = st.session_state.agent
owner: Owner = st.session_state.owner


# --- Sidebar: Status and settings ---
with st.sidebar:
    st.subheader("Status")

    # LLM mode indicator
    if agent.use_llm:
        st.success(f"LLM mode: {agent.api_provider} ({agent.model})")
    else:
        st.info("Rule-based mode (no API key set)")
        st.caption(
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable "
            "to enable LLM-powered reasoning."
        )

    st.divider()

    # Owner name
    owner_name = st.text_input("Your name", value=owner.name)
    if owner_name != owner.name:
        owner.name = owner_name

    st.divider()

    # Current pets summary
    st.subheader("Your Pets")
    if owner.pets:
        for pet in owner.pets:
            pending = len(pet.get_pending_tasks())
            emoji = {"dog": "🐕", "cat": "🐈", "bird": "🐦", "hamster": "🐹"}.get(
                pet.species, "🐾"
            )
            st.write(f"{emoji} **{pet.name}** ({pet.species}) — {pending} pending")
    else:
        st.caption("No pets yet. Tell the agent about your pet!")

    st.divider()

    # Agent log summary
    st.subheader("Agent Log")
    summary = agent.logger.get_summary()
    st.caption(
        f"Interactions: {summary['total_interactions']} | "
        f"Tool calls: {summary.get('total_tool_calls', 0)} | "
        f"Guardrail triggers: {summary.get('guardrail_triggers', 0)}"
    )

    if st.button("Save Logs"):
        filepath = agent.logger.save_to_file()
        st.success(f"Saved to {filepath}")

    if st.button("Save Pet Data"):
        owner.save_to_json()
        st.success("Saved to data.json")


# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show tool calls in expander for assistant messages
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            with st.expander("Agent reasoning", expanded=False):
                for tc in msg["tool_calls"]:
                    st.code(f"Tool: {tc['name']}\nArgs: {tc.get('args', {})}\nResult: {tc.get('result', '')[:200]}")

        # Show confidence and warnings
        if msg["role"] == "assistant" and msg.get("confidence"):
            confidence = msg["confidence"]
            if confidence < 0.4:
                st.caption(f"⚠️ Low confidence ({confidence:.0%})")
            elif confidence < 0.7:
                st.caption(f"Confidence: {confidence:.0%}")

        if msg["role"] == "assistant" and msg.get("warnings"):
            for w in msg["warnings"]:
                st.warning(w)


# --- Welcome message ---
if not st.session_state.messages:
    welcome = (
        "Welcome to PawPal+ Agent! I'm your AI-powered pet care assistant. "
        "Here are some things you can try:\n\n"
        "- **Add a pet**: \"Add Mochi, a golden retriever\"\n"
        "- **Schedule tasks**: \"Schedule a walk for Mochi at 7:30am daily\"\n"
        "- **Check schedule**: \"What's on today's schedule?\"\n"
        "- **Ask care questions**: \"How often should I bathe my dog?\"\n"
        "- **Mark tasks done**: \"Finished Mochi's morning walk\"\n"
        "- **Find time slots**: \"When's the next free 30-minute slot?\"\n"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    with st.chat_message("assistant"):
        st.markdown(welcome)


# --- Chat input ---
if user_input := st.chat_input("Ask me anything about pet care..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process through agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.process(user_input)

        st.markdown(response.message)

        # Show tool calls
        if response.tool_calls_made:
            with st.expander("Agent reasoning", expanded=False):
                for tc in response.tool_calls_made:
                    st.code(
                        f"Tool: {tc['name']}\n"
                        f"Args: {tc.get('args', {})}\n"
                        f"Result: {tc.get('result', '')[:200]}"
                    )

        # Show confidence
        if response.confidence < 0.4:
            st.caption(f"⚠️ Low confidence ({response.confidence:.0%})")

        # Show guardrail warnings
        for w in response.guardrail_warnings:
            st.warning(w)

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.message,
        "tool_calls": response.tool_calls_made,
        "confidence": response.confidence,
        "warnings": response.guardrail_warnings,
    })

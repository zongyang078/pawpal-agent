"""
PawPal+ Agent — Streamlit Chat Interface

A chat-based UI where users interact with the PawPal+ AI Agent
using natural language. The agent manages pet care schedules,
answers care questions, and provides proactive safety advice.
"""


import streamlit as st

from agent import PawPalAgent
from pawpal_system import Owner

# --- Page config ---
st.set_page_config(page_title="PawPal+ Agent", page_icon="🐾", layout="centered")
st.title("🐾 PawPal+ Agent")
st.caption("AI-powered pet care assistant — chat naturally to manage your pets")


# --- Session state initialization ---
SESSION_KEYS = ("owner", "agent", "messages")

if "owner" not in st.session_state:
    # Try to load from saved data
    loaded = Owner.load_from_json()
    if loaded:
        st.session_state.owner = loaded
    else:
        st.session_state.owner = Owner(name="Pet Parent")

if "agent" not in st.session_state:
    # Provider and key come from the environment; no key means rule-based mode.
    st.session_state.agent = PawPalAgent(owner=st.session_state.owner)

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

    st.divider()

    # The agent lives in session state, so a rerun keeps the instance that was
    # built when the session started -- edits to retrieval or the guardrails do
    # not take effect until it is rebuilt. Streamlit's own "Rerun" does not do
    # that; this does. Pets survive, because they are reloaded from disk.
    if st.button("Reset session", help="Rebuild the agent from the current code"):
        owner.save_to_json()
        for key in SESSION_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Clears chat history and reloads the code. Pets are kept.")


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
        if msg["role"] == "assistant" and msg.get("confidence") is not None:
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
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = agent.process(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.message,
        "tool_calls": response.tool_calls_made,
        "confidence": response.confidence,
        "warnings": response.guardrail_warnings,
    })

    # Rerun so the sidebar reflects what this turn changed. Streamlit runs the
    # script top to bottom, so the sidebar is drawn before this handler ever
    # executes -- without the rerun, a pet added now only appears in the
    # sidebar after the *next* message. The history loop above redraws both
    # messages, so nothing is lost by not rendering them inline here.
    st.rerun()

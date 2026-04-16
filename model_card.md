# PawPal+ Agent — Model Card

## Base Project

This project extends **PawPal+** (Module 2 Show), originally a Streamlit pet care management system with four OOP classes (Task, Pet, Owner, Scheduler), priority-based scheduling algorithms, conflict detection, recurring task automation, and JSON persistence. The original project demonstrated OOP design, algorithm implementation, and Streamlit UI development.

## What Changed

The extension adds an **agentic AI layer** that transforms PawPal+ from a form-based CRUD app into a conversational AI assistant. Key additions: a ReAct-loop Agent with intent detection and tool calling, a pet care knowledge base with 14 external documents and TF-IDF retrieval including stop word filtering, suffix stemming, title boosting, and species relevance scoring (RAG), safety guardrails (toxic food detection, emergency override, medical disclaimers, confidence scoring), interaction logging, and a chat-based Streamlit interface. The system supports both LLM-powered (OpenAI gpt-4o-mini / Anthropic Claude) and rule-based operation modes.

## Limitations and Biases

**Knowledge base coverage is limited.** The knowledge base has 14 external documents (loaded from the `knowledge/` directory) covering common dog, cat, bird, and hamster care topics. It lacks coverage for exotic pets, breed-specific needs, regional veterinary practices, and many medical conditions. Users may receive "no relevant information" for perfectly valid questions. Documents can be expanded by adding new `.txt` files to the directory.

**English-only.** The intent detection, knowledge base, and guardrails all operate on English text. Non-English inputs will likely fail silently — the system may misidentify intent or miss safety-critical keywords.

**Rule-based mode has low recall on ambiguous inputs.** The keyword-matching intent detector works well for clear, direct commands but struggles with indirect requests ("Can you help with Mochi?" or "I'm worried about my cat") that a human would understand immediately.

**Species bias toward dogs and cats.** The knowledge base has significantly more content about dogs (5 documents) and cats (5 documents) compared to birds (1) and hamsters (1). Users with less common pets will receive lower quality advice.

**No real-time medical knowledge.** The system has no access to current veterinary research, drug interactions, or outbreak information. All medical advice is based on static, general-purpose text.

## Potential Misuse and Mitigations

**Risk: Relying on AI for medical decisions.** A user might trust the Agent's health advice and delay seeing a vet. **Mitigation:** The guardrails add medical disclaimers to any health-related response, and emergency keywords trigger an immediate vet referral that overrides the normal response.

**Risk: Toxic food recommendation.** If the LLM hallucinates or the rule-based system misinterprets a query, it could suggest feeding something harmful. **Mitigation:** The post-response guardrail scans for known toxic foods by species and flags any that aren't already marked as dangerous in the response.

**Risk: False sense of completeness.** Users might assume the knowledge base covers everything about pet care. **Mitigation:** Every knowledge base response includes a disclaimer. The confidence score is displayed in the UI — low scores alert users that the system isn't sure.

## Testing and Reliability Observations

**What surprised me:** The toxic food guardrail caught a real edge case during testing — when I asked "Can my dog eat grapes?", the knowledge base correctly mentioned grapes in the feeding guidelines document (which says to avoid them), but the response included the word "grapes" in a context that could be ambiguous. The guardrail's context-window check (looking for "toxic" or "avoid" near the word) correctly identified that the response already warned against grapes, so it didn't double-flag. This kind of nuanced checking is exactly what deterministic guardrails are good at.

**What didn't work well initially:** The confidence scoring heuristic is crude — it mostly checks for error strings and result length. A more robust approach would use the LLM itself to rate confidence or track historical accuracy per tool. The current scoring can assign high confidence to a completely wrong but long response.

**Interesting failure mode:** In rule-based mode, "I finished walking Mochi" correctly detects intent as "complete_task" and extracts "Mochi" as the pet name, but it matches the task by checking if any word from the task description appears in the message. This means "finished walking" matches "Morning walk" because of "walk" — which works, but it would also match "I want to walk to the store" if that message somehow got classified as a completion intent.

## AI Collaboration During This Project

**Helpful suggestion:** When designing the guardrails module, I asked Claude to help me think through safety edge cases. It suggested the "context-window check" for toxic foods — instead of just flagging any mention of a toxic food, checking whether the surrounding text already contains warning language. This was a much better approach than my initial plan of blindly flagging every mention, which would have triggered false positives on the knowledge base's own warnings.

**Flawed suggestion:** When I initially asked Claude to help design the intent detection system, it suggested using sentence embeddings with cosine similarity against a bank of example utterances per intent. While technically more accurate, this would have added a dependency on a sentence-transformer model (~400MB download) and significantly increased startup time — overkill for a system with 8 intent categories where keyword matching gets the job done for the rule-based fallback mode. I chose the simpler approach and reserved sophisticated NLU for the LLM mode.

## What This Project Says About Me as an AI Engineer

This project demonstrates my ability to design and implement a complete AI system end-to-end: from OOP backend design to agentic orchestration, from safety engineering to systematic testing. I focused on practical engineering decisions — choosing TF-IDF over embeddings for a small knowledge base, building deterministic guardrails for safety-critical checks, and designing a graceful fallback system that works without API access. These are the kinds of trade-offs that matter in production AI systems, where reliability and maintainability outweigh algorithmic sophistication.

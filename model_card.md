# PawPal+ Agent — Model Card

## Overview

PawPal+ Agent is a pet care scheduling assistant with a natural language
interface. It combines a deterministic scheduling backend with an LLM-driven
tool-calling loop, a small TF-IDF retrieval layer over pet care documents, and a
rule-based safety layer that runs before and after every response.

The system is a thin orchestration layer over its own tools. It does not
fine-tune, train, or host a model — it calls the OpenAI or Anthropic API, and
falls back to keyword-based dispatch when no API key is configured.

| | |
|---|---|
| Reasoning | OpenAI (`gpt-4o-mini`) or Anthropic (`claude-sonnet-4-20250514`), configurable |
| Fallback | Deterministic keyword intent detection, no API required |
| Tools | 8, exposed as JSON Schema function definitions |
| Retrieval | Hand-rolled TF-IDF over 14 documents, no embeddings |
| Safety | 3 keyword tables (18 toxic substances, 15 emergency, 16 referral) |

## Intended Use

Helping a pet owner keep track of routine care for common household pets — dogs,
cats, birds, and hamsters. Concretely: registering pets, scheduling and
completing recurring tasks, surfacing scheduling conflicts, finding open time
slots, and answering general husbandry questions about feeding, grooming,
exercise, and vaccination timing.

It is built as a demonstration of agentic system design. It is not a product and
has not been deployed to real users.

## Out of Scope

- **Veterinary diagnosis or treatment.** The system cannot assess a specific
  animal. Health responses carry a disclaimer, and messages matching emergency
  keywords are answered only with a referral to an emergency vet.
- **Dosing, drug interactions, or medication advice** of any kind.
- **Exotic species.** The corpus covers four species; anything else falls back
  to generic text or returns no result.
- **Multi-user or concurrent use.** State is a single JSON file rewritten in
  full on every change, with no locking.

## Limitations

**Knowledge base coverage is narrow.** Fourteen documents covering common dog,
cat, bird, and hamster topics. No breed-specific guidance, regional veterinary
practice, or medical conditions. Valid questions can return "no relevant
information".

**Species coverage is unbalanced.** Dogs have 5 documents and cats 5, against 1
each for birds and hamsters. Advice quality degrades accordingly.

**English only.** Intent detection, retrieval, and every guardrail keyword table
operate on English text. Non-English input fails silently — it will misroute
intent and, more seriously, miss safety keywords entirely.

**Rule-based mode has low recall on indirect phrasing.** Keyword matching
handles direct commands well but misses requests like "Can you help with
Mochi?" or "I'm worried about my cat". LLM mode handles these; the fallback
does not.

**No real-time knowledge.** All content is static text. No access to current
veterinary research, recalls, or outbreak information.

**Retrieval has no chunking or vector scoring.** Whole documents are the
retrieval unit and whole documents are returned, scored by a raw TF-IDF weighted
sum with no length normalisation. This is adequate at 14 documents and would not
be at 140.

## Safety Design and Known Gaps

Three deterministic checks run around every response. They are keyword tables
rather than model-based classifiers, chosen because safety checks should be
fast, always-on, and produce the same answer every time.

| Check | When | Effect |
|---|---|---|
| Emergency | Before tools run | Replaces the response with a vet referral |
| Vet referral | Around the response | Appends a medical disclaimer |
| Toxic food | After the response | Appends a safety warning if an unwarned toxic item appears |

**Gaps found during testing, not yet fixed.** These are pinned as strict `xfail`
tests so they cannot be forgotten:

1. **A medical keyword suppresses the answer entirely.** The pre-flight check is
   called with an empty response string, so a vet-referral keyword makes the
   modified response non-empty and the agent mistakes it for an emergency
   override. Asking "my dog has a lump, how often should I feed him?" returns
   the disclaimer and nothing else — at a reported confidence of 1.0.
2. **Keyword tables match bare substrings.** "plump" contains "lump", and
   "swallowed his food" contains "swallowed", so both trigger false positives.
3. **The toxic food check can be talked out of firing.** Its context window
   treats the words "not" and "safe" as evidence that a warning is already
   present, so "Chocolate is a safe treat for your dog" passes unflagged.
4. **Retrieval is dead on small corpora.** IDF is `log(N / (1 + df))`, which is
   at most zero when a term appears in nearly every document, so a corpus of one
   or two documents never returns a hit.

Gaps 1 and 3 are the ones that matter: both cause the safety layer to fail
quietly rather than loudly.

## Evaluation

**There is no quantitative evaluation yet.** No measured intent accuracy, no
retrieval recall, no guardrail false-positive rate. This is the most significant
gap in the project, and the claims above about retrieval and guardrail quality
should be read as design intent rather than measured behaviour.

What does exist is a deterministic test suite: 110 tests across 7 modules at 74%
line coverage, with the domain layer at 98% and guardrails at 96%. The LLM
request paths are entirely uncovered — they have no fake client to test against.

## Misuse Risks

**Substituting the agent for a vet.** An owner might act on health advice and
delay real care. Mitigated by disclaimers on medical topics and the emergency
override — but see gap 1 above, which currently makes that override fire too
eagerly.

**Acting on a hallucinated food recommendation.** In LLM mode the model could
suggest something harmful. Mitigated by the post-response toxic food scan, which
knows 18 substances across 4 species — but see gap 3, which lets some phrasings
through.

**Assuming the knowledge base is complete.** Every retrieval response carries a
scope disclaimer, and a confidence score is shown in the UI so low-certainty
answers are visible.

## Origins

Built on a course starter project (a Streamlit CRUD app with four OOP classes),
which supplied roughly 200 lines. The agent, tool layer, retrieval, guardrails,
logging, CLI, and test suite were added afterwards. Commit history preserves the
attribution.

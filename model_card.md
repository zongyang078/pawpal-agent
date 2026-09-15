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

**Gaps found during testing, since fixed.** Three of the four defects that
restructuring the test suite surfaced were in this layer, and two of them made
the safety layer fail quietly rather than loudly:

1. **A medical keyword suppressed the answer entirely.** The pre-flight stage
   ran the full pipeline against an empty response, so a vet-referral keyword
   produced a non-empty modified response that the agent read as an emergency
   override — the user got the disclaimer alone, at confidence 1.0. Pre-flight
   now runs the emergency check only.
2. **Keyword tables matched bare substrings** ("plump" → "lump"). Now anchored
   on word boundaries.
3. **The toxic food check could be talked out of firing** — it counted "not"
   and "safe" as existing warnings, passing "Chocolate is a safe treat for your
   dog". Warning language is now an explicit pattern, and every mention of an
   item must carry its own warning.
4. **Retrieval returned nothing on corpora below three documents**, because
   unsmoothed IDF went non-positive. Now smoothed.

Each has a regression test named after the failure it prevents.

**The remaining limit is the approach itself.** Defect 2 showed it: "swallowed"
is what a dog does at every meal, so no boundary anchoring rescues "swallowed
his food quickly" — it needed a hand-written exclusion for ordinary food. A
keyword table cannot express much more nuance than that. Emergency detection
therefore trades recall for precision in a way a trained classifier would not
have to, and the tables remain English-only.

## Evaluation

**There is no quantitative evaluation yet.** No measured intent accuracy, no
retrieval recall, no guardrail false-positive rate. This is the most significant
gap in the project, and the claims above about retrieval and guardrail quality
should be read as design intent rather than measured behaviour.

What does exist is a deterministic test suite: 115 tests across 7 modules at 74%
line coverage, with the domain layer at 98% and guardrails at 96%. The LLM
request paths are entirely uncovered — they have no fake client to test against.

## Misuse Risks

**Substituting the agent for a vet.** An owner might act on health advice and
delay real care. Mitigated by a disclaimer on medical topics and by the
emergency override, which replaces the response outright rather than annotating
it. Detection is keyword-based, so an emergency described in wording the tables
do not cover is answered as an ordinary question.

**Acting on a hallucinated food recommendation.** In LLM mode the model could
suggest something harmful. Mitigated by the post-response toxic food scan, which
knows 18 substances across 4 species and requires every mention to be warned
about. It covers only those 18 substances, spelled in English.

**Assuming the knowledge base is complete.** Every retrieval response carries a
scope disclaimer, and a confidence score is shown in the UI so low-certainty
answers are visible.

## Origins

Built on a course starter project (a Streamlit CRUD app with four OOP classes),
which supplied roughly 200 lines. The agent, tool layer, retrieval, guardrails,
logging, CLI, and test suite were added afterwards. Commit history preserves the
attribution.

# PawPal+ Agent

A pet care scheduling assistant you talk to in plain language. It manages pets
and recurring care tasks, spots scheduling conflicts, finds open time slots, and
answers husbandry questions from a local knowledge base — with a deterministic
safety layer wrapped around every response.

```
$ python cli.py ask "Add Mochi, a dog"
Added Mochi the dog.

$ python cli.py ask "Schedule a walk for Mochi at 07:30 daily"
Added task 'Morning walk' for Mochi at 07:30.

$ python cli.py schedule
Today's schedule (2026-09-14):
  1. 🔴 07:30 - Morning walk (30min, daily, pending) [Mochi]

$ python cli.py ask --trace "How often should I bathe my dog?"
  trace:
    search_care_info({'query': 'How often should I bathe my dog?'})
    confidence: 0.70
Here is what I found in the pet care knowledge base:

  [GROOMING] Dog grooming basics
  Brushing frequency depends on coat type: short coats weekly [...]

$ python cli.py ask "My dog is not breathing!"
  ! Emergency keyword detected: 'not breathing'
This sounds like it could be a pet emergency. Please contact your
veterinarian or an emergency animal hospital immediately. [...]
```

## Architecture

![System Architecture](assets/architecture.svg)

The agent is an orchestration layer over its own tools. A turn runs as:
intent detection → pre-flight guardrails → tool planning and execution →
post-flight guardrails → confidence scoring → logging.

| Module | Role |
|---|---|
| [`agent.py`](agent.py) | Reasoning loop. Plans and executes tools, self-checks the result. Two modes: LLM function calling, or keyword dispatch when no API key is set. |
| [`tools.py`](tools.py) | Eight tools as JSON Schema function definitions: `add_pet`, `add_task`, `complete_task`, `get_schedule`, `get_pet_tasks`, `detect_conflicts`, `suggest_time_slot`, `search_care_info`. |
| [`knowledge_base.py`](knowledge_base.py) | Retrieval over 14 documents in [`knowledge/`](knowledge/). Hand-rolled TF-IDF with stop words, suffix stemming, 3× title boost, and species weighting. Drop in a `.txt` file to extend it. |
| [`guardrails.py`](guardrails.py) | Safety checks before and after each response: emergency override, medical disclaimer, toxic food scan, confidence score. |
| [`logger.py`](logger.py) | Records the full chain per interaction — input, intent, tool calls with arguments and results, guardrail outcome, response. Exports to JSON. |
| [`pawpal_system.py`](pawpal_system.py) | Domain layer: `Task`, `Pet`, `Owner`, `Scheduler`. Priority scheduling, recurrence, conflict detection, duration-aware slot finding, JSON persistence. |
| [`cli.py`](cli.py) | Terminal entry point. |
| [`app.py`](app.py) | Streamlit chat UI, with tool-call and confidence transparency. |

The LLM layer targets OpenAI (`gpt-4o-mini`) and Anthropic
(`claude-sonnet-4-20250514`) through their native tool-calling APIs. Both are
optional: with no key configured the agent runs entirely on keyword dispatch, so
the system works offline and the test suite never touches the network.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable LLM mode.

```bash
python cli.py seed                    # write a sample owner, 2 pets, 6 tasks
python cli.py schedule                # today's plan, plus any conflicts
python cli.py pets                    # registered pets and pending counts
python cli.py ask "..." [--trace]     # one question, optionally with the tool trace
python cli.py chat [--trace]          # interactive session
python cli.py --data demo.json ...    # use a different state file

streamlit run app.py                  # chat UI
```

## Testing

```bash
python -m pytest tests/ -q
```

115 tests across 7 modules, no network access required:

| Module | Tests | Under test |
|---|---|---|
| [`test_domain.py`](tests/test_domain.py) | 33 | Task, Pet, Owner, Scheduler, slot finder, persistence |
| [`test_agent.py`](tests/test_agent.py) | 19 | Intent detection, rule-based flows, guardrail regressions |
| [`test_guardrails.py`](tests/test_guardrails.py) | 16 | Toxic food, emergency, referral, confidence |
| [`test_tools.py`](tests/test_tools.py) | 14 | Tool schemas and dispatch |
| [`test_knowledge_base.py`](tests/test_knowledge_base.py) | 14 | Retrieval, IDF weighting, corpus loading |
| [`test_cli.py`](tests/test_cli.py) | 11 | Every subcommand, exit codes, stream routing |
| [`test_logger.py`](tests/test_logger.py) | 8 | Recording, summary, JSON export |

Line coverage is 74%:

```
pawpal_system.py   98%    knowledge_base.py  96%    guardrails.py  96%
logger.py          88%    tools.py           84%    cli.py         82%
agent.py           55%    app.py              0%
```

The gap in `agent.py` is the two LLM request paths, which have no fake client to
test against. That is the next thing to fix, and it is the reason the LLM mode
is described here as implemented rather than as verified.

## Design decisions

**Agentic orchestration over RAG-only.** The scheduling backend already had
real capabilities — priority sorting, recurrence, conflict detection, slot
finding. Wrapping those as tools lets the model compose them, and adding a
capability means registering a tool rather than changing a prompt. A
retrieval-only design could answer questions but could not *do* anything.

**Dual mode, LLM and rule-based.** The fallback is not a toy: it handles the
eight intent categories directly, which makes the system runnable with no
credentials and makes the whole test suite hermetic. It buys reproducibility at
the cost of recall on indirect phrasing — a trade-off that shows up plainly in
the model card's limitations.

**TF-IDF over embeddings.** At 14 documents, a scored keyword match with title
boosting and species weighting is fast, interpretable, and dependency-free. The
honest version of this decision: it is right at this corpus size and would be
wrong at 10× it, where the lack of chunking and vector scoring would start to
cost real accuracy.

**Keyword guardrails over an LLM judge.** Safety checks should be fast,
always-on, and identical every time. A table lookup for "chocolate" cannot be
prompted out of firing. The cost is brittleness at the edges, and the defects
below are exactly that cost showing up.

## Defects found and fixed

Restructuring the test suite surfaced four defects, three of them in the safety
layer. Each now has a regression test named after the failure it prevents.

1. **A medical keyword suppressed the answer entirely.** The pre-flight check
   ran the whole guardrail pipeline against an empty response string, so a
   vet-referral keyword made the modified response non-empty and the agent read
   it as an emergency override. "My dog has a lump, how often should I feed
   him?" returned the disclaimer and nothing else, at a reported confidence of
   1.0. The pre-flight stage now runs only the emergency check — the one check
   that should stop a turn before any tool does work.

2. **Keyword tables matched bare substrings.** "plump" contains "lump", so a
   hamster weight question read as a possible tumour. All three tables are now
   anchored on word boundaries.

3. **The toxic food check could be talked out of firing.** Its context window
   counted the words "not" and "safe" as evidence that a warning was already
   present, so "Chocolate is a safe treat for your dog" passed clean. Warning
   language is now an explicit pattern, and *every* mention of a toxic item has
   to carry its own warning — warning once at the top of a long retrieval answer
   no longer licenses an unwarned mention further down.

4. **Retrieval was dead on small corpora.** IDF was `log(N / (1 + df))`, which
   is ≤ 0 once a term appears in nearly every document, and `search()` drops
   anything scoring ≤ 0 — so a corpus of one or two documents could never
   return a hit. Now smoothed to `log((1 + N) / (1 + df)) + 1`. The earlier
   suite missed this because it only ever exercised the full 14-document corpus.

Defect 2 also exposed the ceiling of keyword matching: "swallowed" is what a dog
does at every meal, so no amount of boundary anchoring fixes "swallowed his food
quickly". That keyword now carries a negative lookahead for ordinary food, which
is about as much nuance as a keyword table can express.

## Still outstanding

Two near-duplicate provider methods that should be one loop behind an adapter;
tool results returned as strings, so the agent cannot branch on failure; no
timeouts or retries on API calls; magic numbers in place of configuration; and
no quantitative evaluation of intent accuracy, retrieval recall, or guardrail
false-positive rate. The last is the biggest gap — see the model card's
Evaluation section.

## Screenshots

| | |
|---|---|
| ![Chat](assets/demo_chat.png) | ![Schedule](assets/demo_schedule.png) |
| ![Retrieval](assets/demo_rag.png) | ![Emergency](assets/demo_emergency.png) |

## Project structure

```
pawpal-agent/
├── agent.py              # reasoning loop, intent detection
├── tools.py              # tool schemas and dispatch
├── knowledge_base.py     # TF-IDF retrieval
├── guardrails.py         # safety checks
├── logger.py             # interaction logging
├── pawpal_system.py      # domain layer
├── cli.py                # terminal entry point
├── app.py                # Streamlit chat UI
├── knowledge/            # 14 care documents (.txt)
├── tests/                # 115 tests across 7 modules
├── assets/               # architecture diagram, screenshots
├── model_card.md         # intended use, limitations, safety gaps
└── requirements.txt
```

## Origins

Built on a course starter project — a Streamlit CRUD app with four OOP classes,
roughly 200 lines. Everything after that (agent, tools, retrieval, guardrails,
logging, CLI, tests) was added on top. Commit history preserves the attribution.

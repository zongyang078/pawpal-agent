#!/usr/bin/env bash
#
# End-to-end check against a real provider. Deliberately NOT in CI: it needs a
# key and spends money, and the point of the test suite is that it does not.
# What this covers that the suite cannot is the one thing a fake client can
# never prove -- that the adapter's wire format is what the vendor accepts.
#
#     ./scripts/smoke_live.sh [anthropic|openai]
#
set -u
PROVIDER="${1:-anthropic}"
DATA=$(mktemp -t pawpal-live-XXXX.json); rm -f "$DATA"
trap 'rm -f "$DATA"' EXIT

run() { python cli.py --data "$DATA" --provider "$PROVIDER" ask --trace "$1"; }

echo "=============== 1. key and provider ==============="
echo "expect: mode: $PROVIDER:<model>, and no 'call failed'"
run "hi"

echo; echo "=============== 2. one tool call ==============="
echo "expect: add_pet({'name': 'Mochi', 'species': 'dog'})"
run "Add Mochi, a dog"

echo; echo "=============== 3. several tools in one turn ==============="
echo "expect: add_pet and get_schedule, and an answer drawing on both"
run "Add Luna the cat, then show me today's schedule"

echo; echo "=============== 4. guardrail pre-empts the provider ==============="
echo "expect: the canned referral, and NO tool calls at all"
run "My dog had a seizure!"

echo; echo "=============== 5. retrieval ==============="
echo "expect: search_care_info, answered from the local corpus"
run "How often should I bathe my dog?"

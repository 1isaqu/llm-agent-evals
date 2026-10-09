# The reliability ledger

[docs/reliability-ledger.md](reliability-ledger.md) (and the HTML twin next to it) is a status board of failure classes of
the agent: what went wrong, what fixed it, which scenario or test keeps it fixed, and what risk
remains. It is generated from `ledger/entries.json`.

```
llm-agent-evals ledger verify                    # every reference resolves, rules hold
llm-agent-evals ledger generate --out docs/reliability-ledger.md --html docs/reliability-ledger.html
```

## Entry format

```json
{
  "id": "L-05",
  "class": "Runaway tool loop",
  "status": "closed",
  "finding": "What went wrong, or could.",
  "fix": "What closed it.",
  "evidence": [
    {"kind": "scenario", "ref": "S19"},
    {"kind": "test", "ref": "tests/test_guards.py::test_an_a_b_a_b_spiral_declares_a_loop"},
    {"kind": "module", "ref": "src/llm_agent_evals/guards.py"}
  ],
  "residual_risk": "What can still go wrong.",
  "gap": "Only for partial entries: what is not covered."
}
```

| Status | Rule |
|---|---|
| `closed` | A fix exists and at least one `scenario` or `test` guards it. A `module` reference alone is not enough: it shows where a fix lives, not that anything still guards it. |
| `partial` | Mitigated, with a `gap` that says what is not covered. |
| `open` | Known and not fixed. |

Hazards that are flagged and deliberately not fixed go in the separate `landmines` list.

## What `verify` checks

- every `scenario` reference is the id of a scenario in `scenarios/`;
- every `test` reference is `path::name`, the file exists and defines a function or method of that
  name (found by parsing, so a test module that cannot be imported is still checked);
- every `module` reference is an existing file;
- ids are unique across entries and landmines, statuses are known, closed entries cite a test or a
  scenario, partial entries name their gap.

`ledger generate` runs the scripted scenarios and annotates each scenario reference with how it
currently behaves (`ok`, or `detected` for a self-test). The output holds no dates, so a test can
regenerate both documents and compare them with the committed ones.

## Why a generator instead of a page

A hand-edited ledger goes stale as soon as a test is renamed. Here, renaming a test or deleting a
scenario fails `ledger verify`, and editing the data without regenerating fails the drift test.

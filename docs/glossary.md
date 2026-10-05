# Glossary

Domain terms and the field names that carry them. The domain is consumer-debt collections, kept
neutral: no company, country, currency rules or vendor schema appears anywhere in this repository.

## Terms

| Term | Meaning |
|---|---|
| Agreement | A negotiated payment plan for one or more debts. One row per installment in `agreements`. |
| Generated agreement | An agreement created in the window, including those that broke later. Breaking is a later outcome and does not erase the generation. |
| Installment | One scheduled payment. `installment_no = 0` is the first installment (the down payment). |
| First-installment value | Sum of the first installments of generated agreements. The base for every risk percentage. |
| Payment slip | The document issued for an installment. "Slips issued" counts generated first installments. |
| Paid within grace | A slip paid on or before its due date plus `grace_days`. |
| Slip effectiveness | Slips paid within grace divided by slips issued. Not conversion. |
| Outreach attempt | One contact attempt by an agent (`contact_events`). |
| Answer | An attempt whose result is flagged `answered`. |
| Right-party contact (CPC) | An attempt whose result is flagged `answered` and `right_party`. A count, never a percentage. |
| Answer rate | `answer_count / attempt_count`. Never called CPC. |
| CPC rate | `cpc_count / answer_count`. |
| Conversion | `agreement_count / cpc_count`: generated agreements over right-party contacts. Exists only per agent. |
| Paid per CPC | `slips_paid / cpc_count`. A different metric, never called conversion. |
| Average ticket | `agreement_value / agreement_count`. |
| Exception | An agreement whose value is pending approval (`PENDING_EXCEPTION`). |
| Rejected | An agreement refused by a supervisor. |
| Broken / auto-broken | An agreement that stopped being paid, recorded manually or by the system. |
| Portfolio | A named book of debts (`debt_attributes.portfolio_label`). |
| Product line | One of two separate databases, `auto` and `consumer`. A session can also cover `all`. |
| Composite risk | The maximum of the three risk axes (exception, broken, rejected). Never the sum. |
| Risk axis | One status dimension as a share of the first-installment universe. |
| Universe | Generated + exception + rejected first-installment value of the window. The common denominator. |
| Anomaly | An axis outside 0 to 100 percent. Cannot happen with clean data. |
| Plan stage | Where an approved agreement is in its payment plan: early, middle, late or settled. |
| Daily pace | How today's generation compares with a normal day, by hour. |
| Window | An inclusive date range. The session window is a display filter, not a limit on tools. |
| Anchor | An explicit date for a relative period ("last quarter"), computed by code and given to the model. |
| Memoised call | A repeated identical tool call answered from the first result. |
| Egress guard | The check that runs on tool results before they can reach a model. |
| Scenario | One evaluation case: a question, data, a scripted model and expectations. |
| Golden scenario | A scenario that must pass cleanly. |
| Self-test | A scenario that scripts a mistake on purpose and declares the violation the harness must report. |
| Violation | One failed check, with a layer and a code (`layer:code`). |
| Ledger | A status board of failure classes, their fixes and the evidence that guards them. |

## Field names

| Field | Meaning |
|---|---|
| `attempt_count` | outreach attempts |
| `answer_count` | attempts that were answered |
| `cpc_count` | right-party contacts |
| `agreement_count` | generated agreements (agreement grain) |
| `agreement_count_by_contract` | generated agreements counted per contract (contract grain, totals only) |
| `agreement_value` | value of all installments of generated agreements |
| `first_installment_value` | value of first installments of generated agreements |
| `avg_ticket` | `agreement_value / agreement_count` |
| `slips_issued` | slips for generated first installments |
| `slips_paid` | those paid within grace |
| `exception_count`, `exception_value` | exception agreements and their first-installment value |
| `conversion_pct` | `agreement_count / cpc_count * 100` |
| `answer_rate_pct` | `answer_count / attempt_count * 100` |
| `cpc_rate_pct` | `cpc_count / answer_count * 100` |
| `paid_per_cpc_pct` | `slips_paid / cpc_count * 100` |
| `slip_effectiveness_pct` | `slips_paid / slips_issued * 100` |
| `composite_risk_pct` | the maximum of `exception_pct`, `broken_pct`, `rejected_pct` |
| `risk_level` | `low`, `medium` or `high`, from the configured cut-offs |
| `reference_date` | last day of the window the figure refers to |
| `product_line` | `auto`, `consumer` or `all` |
| `agent_id`, `agent_name` | the agent's public key (`agents.login`) and display name, both invented |

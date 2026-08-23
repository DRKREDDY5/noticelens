# Phase 4A fixed-chunk section failure analysis

This is the frozen fixed-chunk, notice-filtered section baseline. It reused the 350 Phase 3 vectors and did not build heading-aware chunks.

## Contract and thresholds

- Section correctness: exact notice code and exact normalized full heading path.
- Meaningful later heading-aware gain: at least 10 absolute percentage points.
- Section null branch: fixed P@1 at least 95.0% and later gain under 5 points.
- The null decision is not made in Phase 4A because the heading-aware comparator has not been built.

## Aggregate results

| Scope | n | Section P@1 | Section MRR | Section Hit@5 |
|---|---:|---:|---:|---:|
| Overall | 15 | 80.0% | 0.8689 | 100.0% |
| Naive | 10 | 80.0% | 0.8833 | 100.0% |
| Expert | 5 | 80.0% | 0.8400 | 100.0% |
| Family: balance_collection | 4 | 75.0% | 0.8750 | 100.0% |
| Family: installment_agreement | 1 | 100.0% | 1.0000 | 100.0% |
| Family: levy_cdp | 1 | 100.0% | 1.0000 | 100.0% |
| Family: non_filer | 5 | 80.0% | 0.8667 | 100.0% |
| Family: penalty_estimated_tax | 1 | 100.0% | 1.0000 | 100.0% |
| Family: underreporter_deficiency | 3 | 66.7% | 0.7333 | 100.0% |

## Rank-1 failures

### S03 — CP503

- Question: How soon do I need to pay this second reminder so more penalties and interest don't build up?
- Expected path: `Frequently asked questions > How much time do I have?`
- Expected evidence: pay the entire balance by the due date shown on your notice
- Rank 1 paths: `[["You may want to"], ["Frequently asked questions"], ["Frequently asked questions", "What is the notice telling me?"], ["Frequently asked questions", "What do I have to do?"]]`
- Rank 1 score/preview: 0.636513 — information](https://www.irs.gov/payments/online-account-for-individuals) pertaining to your tax account. - Learn more about your [payment options](https://www.irs.gov/payments) and how to make a [payment arrangement](https://www.irs.gov/payments/payment-plans-installment-agreements). - [Request an appeal](https://www.irs.gov/appeals/preparing-a-request-for-
- First correct rank: 2
- Top candidates: r1:miss, r2:match, r3:miss, r4:miss, r5:miss

### S07 — CP2000 series

- Question: After responding, which tax records should be reviewed or corrected to prevent the same mismatch from happening again?
- Expected path: `You may want to > Check and correct your records`
- Expected evidence: Correct your copy of your tax return. Keep it and the notice for your records.
- Rank 1 paths: `[["Why you received this notice"], ["What you need to do"]]`
- Rank 1 score/preview: 0.672760 — # Understanding your CP2000 series notice Source: https://www.irs.gov/individuals/understanding-your-cp2000-series-notice The CP2000 notice series includes: CP2000, CP2000A, CP2000B, CP2000C, CP2000D and CP2000E. Learn what your notice is about and what to do. ## Why you received this notice The income or payment information we received from third parties, s
- First correct rank: 5
- Top candidates: r1:miss, r2:miss, r3:miss, r4:miss, r5:match

### S11 — CP59

- Question: I mailed the missing return recently. Does the page say I need to send another copy right away?
- Expected path: `Frequently asked questions > What should I do if I've just filed my tax return?`
- Expected evidence: You don't have to do anything if you filed your tax return within the last eight weeks.
- Rank 1 paths: `[["What this notice is about"], ["What you need to do"], ["You may want to"]]`
- Rank 1 score/preview: 0.556709 — # Understanding your CP59 notice Source: https://www.irs.gov/individuals/understanding-your-cp59-notice ## What this notice is about We have no record that you filed your prior year personal tax return. ## What you need to do File your personal tax return immediately or explain to us why you don't need to file. Note: If you received an IRS-issued identity pr
- First correct rank: 3
- Top candidates: r1:miss, r2:miss, r3:match, r4:miss, r5:miss

## Predeclared five hardest

Ordering rule: no Section Hit@5 first; then first correct rank descending; then correct-vs-incorrect similarity margin ascending; then question ID.

- S07: “After responding, which tax records should be reviewed or corrected to prevent the same mismatch from happening again?” — first correct rank=5, Hit@5=1, margin=-0.08876418999999991
- S11: “I mailed the missing return recently. Does the page say I need to send another copy right away?” — first correct rank=3, Hit@5=1, margin=-0.02963250900000003
- S03: “How soon do I need to pay this second reminder so more penalties and interest don't build up?” — first correct rank=2, Hit@5=1, margin=-0.0009613039999999407
- S01: “I agree that I owe the balance, but I can't pay all of it right now. What option does the page give me?” — first correct rank=1, Hit@5=1, margin=0.0015906700000000162
- S02: “I think the changes to my account are wrong. How does the page say I should challenge them?” — first correct rank=1, Hit@5=1, margin=0.030159055999999962

## Interpretation

Fixed-chunk Section P@1 leaves 20.0 absolute percentage points to perfect on this benchmark.
This is the only retrieval evidence collected here; no heading-aware, hybrid, reranked, generated, or agentic system was implemented.

## Integrity confirmation

- Frozen Phase 1–3 hashes passed before and after retrieval.
- The existing namespace contained the same exact 350 IDs before and after.
- Document embeddings, index creation, upserts, updates, and deletes were all zero.
- API credentials were neither written nor logged.

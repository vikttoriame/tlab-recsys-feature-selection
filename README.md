# RecSys feature selection: an applicant research project

**Prepared for the T-Lab selection task, October 2026.** This is a candidate project, not employment at T-Bank.

The experiment asks whether a limited automatic search over behavioral features improves next-product-view ranking compared with a substantive hand-designed CatBoost baseline. It uses the public REES46 marketplace event stream, including user actions, product categories and prices.

## Read first

- [Final report in Russian (PDF)](output/pdf/tlab_final_report.pdf)
- [Original Russian submission guide](README_RU.md)
- [Experiment protocol](PROTOCOL.md)
- [Detailed results](RESULTS.md)

## Main finding

**The primary comparison did not establish a clear improvement from automatic feature selection.** This is an experimental result, not a claim of a successful production recommender or a business uplift.

| Variant | Features | NDCG@20 | Recall@20 |
|---|---:|---:|---:|
| Manual baseline | 108 | 0.40524 | 56.27% |
| Automatic windows | 120 | 0.40549 | 56.26% |
| Automatic combinations | 120 | 0.40567 | 56.38% |
| Both families (primary method) | 132 | 0.40505 | 56.15% |
| Size-matched random control | 132 | 0.40547 | 56.33% |

The primary difference in NDCG@20 is **-0.00018**, with a paired user-cluster 95% interval of **[-0.00106, +0.00066]**. An interval containing zero does not demonstrate equivalence or non-inferiority.

## Experiment in brief

- Complete October 2019 histories for 5,000 deterministically selected users: 71,429 events.
- Chronological train / validation / test separation; final test: October 25-30, 2019, with 10,384 prediction moments from 965 users.
- The same 200-candidate retrieval procedure for all variants; the true target is not inserted into the candidate list. Candidate coverage is 72.05%.
- A manual baseline with 105 numeric and three categorical features.
- A fixed feature grammar generates 112 window aggregates and 48 combinations; validation selects up to 12 per family.
- Five variants, three training seeds each, with matched CatBoostRanker settings and user-cluster bootstrap intervals.

## My contribution and AI assistance

Viktoriia Merkulova proposed the research motivation: account for the user's current price interest, category context and the meaning of different actions. The limitations and interpretation were discussed jointly with AI assistance. ChatGPT/Codex supported the protocol, implementation, checks, computation and preparation of the materials. The feature-search algorithm inside the experiment is separate from the use of AI development tools.

This project demonstrates a documented research question and analysis process; it is not presented as independently authored production ML engineering or professional localization experience.

## Verify the saved results

Python 3.12 is recommended. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python verify.py --outputs-only
```

This checks saved predictions, metrics, confidence intervals and feature-selection diagnostics without retraining or downloading the original dataset.

## Reproduce the experiment

The portfolio package omits raw event tables, daily catalog snapshots, trained models and feature caches. It retains source code, saved outputs, compact provenance records, the protocol and report. Use a NEW destination outside this repository:

```bash
python run_pipeline.py --output ../tlab_reproduced --download
```

Full reproduction downloads the October source twice (approximately 1.74 GB per pass). It uses CPU, not GPU; 16 GB RAM is recommended. The original `test_gate.json` contains the original frozen hashes, including models and data not bundled here. Therefore, use `--outputs-only` for verification of this portfolio package; a complete new run creates its own data, models and gate.

## Repository contents

- Root `*.py`: extraction, candidate retrieval, feature generation, selection, training, evaluation and verification.
- `results/`: saved metrics, predictions, diagnostics and verification records.
- `data/*.json`: compact extraction and selection provenance; no raw event tables.
- `output/pdf/`: the original final report.
- `assets/fonts/`: fonts and their original license for report reproduction.
- `SHA256SUMS`: checksums of this prepared portfolio package, regenerated after adding this README.

The experiment code and measured results are preserved from the original submission. The original Russian guide refers to both submission archives; this repository corresponds to the code-and-results edition.

## Sources

- REES46: https://data.rees46.com/
- CatBoost ranking objectives: https://catboost.ai/docs/en/concepts/loss-functions-ranking

The experiment covers one marketplace and one month. It does not measure true user budgets, sales, revenue or generalization to all recommendation tasks.

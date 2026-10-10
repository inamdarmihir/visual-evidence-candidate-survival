# Results (verified, user-run on a Colab Tesla T4)

Everything below is copied from files in `results/computer_science_primary_1024/` (run id `full-primary_1024-a5fda5219b`).
Nothing here is estimated, simulated or taken from another machine. **Only one domain has been run.** The
cross-domain section of the notebook (hr / physics / energy) has **not** been run yet; no cross-domain numbers exist.

## Setup (what was actually measured)
- Dataset `vidore/vidore_v3_computer_science` @ `d5cc75883d92e294f0c0fc2662551c9708a06ebc`; evaluated only `language == english AND query_generator == human`: **110 queries, 545 positive judgments** (451 score‑1 "critically relevant", 94 score‑2 "fully relevant"), 1,360 pages from two OpenStax textbooks. The LLM-merged `answer` and the `markdown` columns are never used.
- Model `Hcompany/NeoMME-260M-Retriever` @ `481083153ac22931afbd614e672212bffbcb666e`, float32, no autocast, pages resized to longest side **1024 px** (a resource-bounded preset; the checkpoint default is 2048, so this is *not* a reproduction of published scores). One forward pass gives the dense (1024‑d) and the multivector (128‑d per token) embeddings; every page has exactly 834 vectors.
- Qdrant v1.19.2 (pinned binary, SHA256-verified), exact dense search, then server-side rerank of the top‑B with MAX_SIM; B ∈ {10, 25, 50, 100}.
- Hardware: Tesla T4, 13.6 GB host RAM. Peak VRAM allocated 2.26 GB, peak host RSS 3.71 GB, Qdrant dir 0.64 GB.
- Wall clock: load data 46 s, embed 1,360 pages 328 s (0.19 s/page), Qdrant indexing 299 s, rerank experiment (all B and k, 110 queries) 20 s, optional full-corpus late reference 5 s.
- Checks that passed: Qdrant-vs-torch MeanMaxSim max abs diff 1.36e‑7; all 1,360 page and 110 query cache checksums OK; re-embed diff 0.0; the exact dense ranking matched its reference.

## Headline (top‑10, macro mean over 110 queries, 95% bootstrap CI over whole queries, 2,000 replicates, seed 42)

| B | candidate recall | pages lost before rerank | pages lost by rerank | pages surviving to top‑10 | top‑10 recall after rerank | Δ vs dense order [95% CI] |
|---|---|---|---|---|---|---|
| 10  | 0.583 [0.520, 0.646] | 51.4% | 0.0%  | 48.6% | 0.583 | +0.000 (identical in all 110 queries) |
| 25  | 0.709 [0.657, 0.763] | 36.3% | 9.0%  | 54.7% | 0.646 | +0.063 [+0.030, +0.103] |
| 50  | 0.801 [0.753, 0.848] | 25.1% | 17.1% | 57.8% | 0.678 | +0.095 [+0.052, +0.141] |
| 100 | 0.893 [0.858, 0.926] | 14.9% | 25.1% | 60.0% | 0.715 | +0.132 [+0.084, +0.181] |

- Dense order alone, top‑10 recall 0.583; unrestricted full-corpus late reference 0.723 [0.672, 0.777]. Reranking the top‑100 gives 0.715; the paired difference to the full-corpus reference is −0.009 [−0.027, +0.011] (not distinguishable from 0).
- Top‑5 recall: dense 0.459, B=100 0.601, full-corpus late 0.611. nDCG@10 (gain 2^score−1): dense 0.575, B=100 0.725, full late 0.730; Δ(B=100 − dense) = +0.150 [+0.100, +0.202].
- Coverage (every annotated page of a query in the top‑10): dense 0.309, B=100 0.364, full late 0.400. These coverage differences were not significant in the paired analysis; do not present them as gains.
- Candidate-stage coverage (all annotated pages inside the B candidates): 0.309 / 0.391 / 0.509 / 0.673 for B = 10 / 25 / 50 / 100.

## Takeaway (this domain only)
The dense candidate stage limits what the rerank can recover: no change at B=10, and top‑10 recall gains of +0.063 / +0.095 / +0.132 at B=25 / 50 / 100. Reranking the top‑100 is not distinguishable from full-corpus late interaction here. This is one domain, one resolution, one model; it is not evidence for other settings.

## Limits that must travel with the numbers
1. **Top‑k ceiling.** Median query has 4 annotated pages but 10 queries have more than 10 (max 21). Best possible top‑10 macro recall is ≈ 0.974 and best possible coverage ≈ 0.909, so part of the "lost by rerank" share is unavoidable.
2. **Judgments are incomplete.** Unjudged pages are not known negatives; recall is a lower bound on true evidence recall.
3. **Scope:** one domain (two OpenStax books), one model, one resolution (1024 px), 110 queries, uniform 1700×2200 page images (no natural document-length variation). It does not show that reranking helps in general, and nothing here measures answer correctness.
4. Small strata (e.g., one numerical query, four chart queries) are descriptive only.
5. The "candidate loss / reranking loss" labels describe where an annotated page drops out; they are diagnostics, not causal explanations.

## Files
`FINDINGS.md` (auto-generated readout), `final_metrics.csv`, `candidate_metrics.csv`, `paired_differences.csv`, `query_strata.csv`, `per_query_metrics.csv`, `page_level_traces.csv`, `storage_and_runtime.csv`, `run_manifest.json`, `parity.json`, `plots/*.png`, `trace_viewer.html`.
The viewer fetches page images on demand from the pinned dataset revision; page images are third-party material and are not stored in this repo (see `docs/ATTRIBUTION.md`).

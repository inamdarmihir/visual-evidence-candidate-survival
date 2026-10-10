# Prompt for drafting the blog post

Copy everything below the line into your writing assistant, then edit. Facts are limited to what is in `RESULTS.md`.
If you run the cross-domain section of the notebook, paste its three CSVs where marked; otherwise leave those lines out.

---

You are helping me write a technical blog post (about 900–1,200 words, plain and precise, no hype) for a Hugging Face community audience that builds retrieval systems.

**Working title ideas:** "How many relevant pages survive a dense → late-interaction cascade?" / "Choosing the candidate budget B for visual-document reranking".

**The question.** A fast dense retriever keeps only B candidates for a more expensive late-interaction reranker. If a human-annotated relevant page is not in the top‑B, the reranker can never recover it. How much evidence is lost at the candidate stage, how much more is lost at the final top‑10 cut, and how big does B have to be before reranking approaches scoring the whole corpus?

**Method (state exactly).** ViDoRe v3 *computer_science* (pinned revision d5cc7588…), English, human-written queries only (110 queries, 545 judged pages, 1,360 pages from two OpenStax textbooks). Model NeoMME‑260M‑Retriever, float32 on a free Colab Tesla T4, pages at 1024 px longest side (a resource-bounded preset, not the model's 2048 default). One forward pass gives a 1024‑d dense vector and 128‑d per-token multivectors. Qdrant v1.19.2: exact dense search, then server-side MAX_SIM rerank of the top‑B, B ∈ {10, 25, 50, 100}. Qdrant scores match a torch MeanMaxSim reference to 1.4e‑7. Macro means over queries with 95% bootstrap CIs over whole queries (2,000 replicates, seed 42). Never used the LLM-merged `answer` column.

**Results to report (use these numbers only; they are in RESULTS.md).**
- Candidate recall of annotated pages: 0.583 / 0.709 / 0.801 / 0.893 for B = 10 / 25 / 50 / 100 (pages lost before reranking: 51.4% → 14.9%).
- Top‑10 recall after reranking: 0.583 / 0.646 / 0.678 / 0.715 vs 0.583 for the dense order. Paired gains over dense: +0.063 [0.030, 0.103], +0.095 [0.052, 0.141], +0.132 [0.084, 0.181] for B = 25 / 50 / 100; at B=10 reranking only reorders the same 10 pages and recall is identical.
- Reranking only the top‑100 gives 0.715 vs 0.723 for an unrestricted full-corpus late-interaction ranking; the paired difference (−0.009 [−0.027, +0.011]) is not distinguishable from zero in this setting.
- The share of annotated pages lost *by the reranker* (in the candidates but below top‑10) grows with B (0% → 25.1%) because more candidates compete for ten slots. Part of that is a ceiling effect: 10 of 110 queries have more than 10 annotated pages, capping top‑10 recall at ≈ 0.974.
- Coverage (all annotated pages of a query in the top‑10) is low (0.31 → 0.36) and its gains were not statistically significant; say so.
- Cost on the T4: embedding 1,360 pages took 328 s, Qdrant indexing 299 s, the whole rerank experiment (110 queries × 4 budgets × 2 cutoffs) 20 s; 2.3 GB peak VRAM.
- Include one figure from `results/computer_science_primary_1024/plots/` (final_topk_before_after_rerank.png and candidate_recall_coverage_vs_B.png) and one worked example from the trace viewer (query 106 page 762: a relevant page at dense rank 10 that reranking moves to rank 19 at B=25, i.e., a "reranking loss"). Thumbnails come from the pinned dataset; credit OpenStax.

**Required honesty section (do not soften).** One domain, one model, one resolution, 110 queries; judgments are incomplete so unjudged pages are not negatives and recall is a lower bound; the top‑k ceiling; no answer-quality claim; 1024 px is not a reproduction of published scores; not a claim of novelty; no claim that reranking always helps. Say the 2048 px preset and Drive-resume path have not been run.

**Structure.** 1) Why candidate survival matters (3 sentences). 2) Setup in a compact table. 3) The funnel result with the table above. 4) Where evidence is lost: candidate stage vs reranking stage. 5) A practical reading: if you need ≥ 0.7 top‑10 recall here, B ≈ 100 matched the full-corpus reference on this data. 6) Limits. 7) Reproduce: link to the repo, the Colab notebook, the pinned revisions. 8) Sources and attribution (ViDoRe V3 arXiv:2601.08620, NeoMME arXiv:2609.01657, OpenStax, Qdrant v1.19.2).

**Rules for you.** Do not invent numbers, datasets, baselines or citations; if a number is not listed above, leave a `[TODO]` marker. Do not call this a benchmark, a state-of-the-art result or a general finding. Use the word "annotated pages", not "ground truth".

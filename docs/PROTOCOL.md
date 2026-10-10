# Protocol

**Question.** When a fast dense retriever keeps only B in {10,25,50,100} candidates for a late-interaction reranker, what happens to human-annotated relevant pages, and where are they lost: before the reranker (candidate loss) or inside it (reranking loss)?

**Data.** `vidore/vidore_v3_computer_science` @ `d5cc75883d92e294f0c0fc2662551c9708a06ebc`, configs corpus / queries / qrels / documents_metadata, split test. Evaluated: `language == "english" AND query_generator == "human"` (110 queries; 545 positive judgments: 451 score-1, 94 score-2). SDG queries and translations are never evaluated. The `answer` and `markdown` columns are never used. No queries, labels, negatives or pages are generated.

**Model.** `Hcompany/NeoMME-260M-Retriever` @ `481083153ac22931afbd614e672212bffbcb666e`, one forward pass per input yielding dense (1024-d) and multivector (128-d per valid position) embeddings. float32, no autocast. Page images: aspect preserved, longest side capped at 1024 px (`max_side`, a deviation from the model default 2048), no crop.

**Mask policy.** Only attention-mask padding is trimmed. All other positions (document/image markers, patch rows, row markers, query `<mask>` expansion tokens) are kept; the valid-position index and input ids are persisted.

**Retrieval.** Qdrant v1.19.2 local server. Dense: exact search (`exact=True`) top-100 once; candidate sets for each B are nested prefixes. Rerank: server-side prefetch (dense, exact, limit B) then named `late` vector MAX_SIM; score divided by the valid query-token count = MeanMaxSim. The returned set is checked against the saved dense prefix; on mismatch an ID-restricted filter over the saved candidates is used and recorded. Dense ranking is verified against a blockwise cosine reference; late scores against `sentence_transformers.util.mean_maxsim` with masks (tolerance 1e-4, ties explicit at 1e-5).

**Metrics.** Candidate recall = |G∩C|/|G|; coverage = 1[G⊆C]; the same at final top-5/10 after dense order and after reranking; nDCG@5/10 with gain 2^score-1, unjudged pages gain 0, ties broken by score desc then corpus_id asc. Score-1 diagnostics only over queries having a score-1 page; score-2 "availability" only over queries having a score-2 page. Macro means over queries, micro totals separately. Paired bootstrap over whole query IDs, 2000 replicates, seed 42, 95% percentile intervals. Strata (query types, content types) are multi-label and overlapping; n is reported for each.

**Optional.** Unrestricted full-corpus late-interaction reference (blockwise exact, resumable). If skipped, dependent metrics are "not computed", never zero.

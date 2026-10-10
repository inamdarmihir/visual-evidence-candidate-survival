# Limitations

- One domain (two OpenStax computer-science textbooks), one model, one resource-bounded resolution preset (1024 px, not the 2048 px default and not a reproduction of published scores).
- 110 independent human-origin English queries. Page judgments, query-type rows and translations are not independent observations; translated copies are excluded.
- Judgments are incomplete: unjudged pages are not established negatives, so recall figures are lower bounds on "relevant retrieved".
- Score 1 = critically relevant (needs additional information); score 2 = page can contain the complete answer. All-annotated-page coverage is NOT answer completeness and not proof every page is necessary. Downstream answer correctness is not measured.
- The `answer` column is LLM-merged and unused. The one numerical query (n=1) supports no claims.
- 3 qrel rows carry content_type "N/A (If relevance score=0)" despite positive scores; kept as published.
- Dense search is exact (isolates model/candidate loss); this is not an ANN benchmark.
- All pages in this corpus are the same pixel size, so real document-token counts are uniform at a given preset; document-side padding is exercised by explicit extra masked padding in the parity check, not by naturally varying page lengths.
- A flat or negative reranking result is a valid outcome. No claim of novelty, generalisation, or that reranking helps.
- Smoke mode is a qrel-aware engineering fixture; never compare it with full-corpus numbers.

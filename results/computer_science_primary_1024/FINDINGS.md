## Readout (n = 110 human-origin English queries; macro means; 95% bootstrap CI over whole queries)

Annotated pages = pages with positive human qrels. Final cutoff shown: top-10. Unjudged pages are not negatives; coverage is not answer completeness.

| B | candidate recall | candidate coverage | pages lost before rerank | pages lost by rerank (in candidates, below top-10) | pages surviving to top-10 |
|---|---|---|---|---|---|
| 10 | 0.583 [0.520, 0.646] | 0.309 [0.227, 0.400] | 51.4% | 0.0% | 48.6% |
| 25 | 0.709 [0.657, 0.763] | 0.391 [0.300, 0.482] | 36.3% | 9.0% | 54.7% |
| 50 | 0.801 [0.753, 0.848] | 0.509 [0.418, 0.600] | 25.1% | 17.1% | 57.8% |
| 100 | 0.893 [0.858, 0.926] | 0.673 [0.591, 0.764] | 14.9% | 25.1% | 60.0% |

Dense order alone, top-10: recall 0.583 [0.520, 0.646], coverage 0.309.

### Does reranking change top-10 recall vs. the dense order? (paired, same queries)

| B | mean diff in recall | 95% CI | queries better / worse / equal | reading |
|---|---|---|---|---|
| 10 | +0.000 | [+0.000, +0.000] | 0 / 0 / 110 | identical in every query (reranking only reorders the same B pages) |
| 25 | +0.063 | [+0.030, +0.103] | 36 / 6 / 68 | reranking higher |
| 50 | +0.095 | [+0.052, +0.141] | 41 / 7 / 62 | reranking higher |
| 100 | +0.132 | [+0.084, +0.181] | 49 / 7 / 54 | reranking higher |

Unrestricted full-corpus late-interaction reference, top-10: recall 0.723 [0.672, 0.777], coverage 0.400. This is a reference ranking, not a gold oracle.

Scope: one domain (two OpenStax books), one model, 1024 px preset unless noted, 110 independent queries. A flat result is a valid result; no general claim about reranking is supported.

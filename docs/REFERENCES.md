# References (titles and authors checked on arxiv.org/abs/<id>; read the papers before attributing specific claims to them)

**Data and model used in this repo**
- Loison, Macé, Edy, et al. *ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.* arXiv:2601.08620.
- Lac, Wu. *NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference.* arXiv:2609.01657.

**Background: late interaction and multi-stage retrieval**
- Khattab, Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* arXiv:2004.12832. (MaxSim late interaction.)
- Santhanam, Khattab, et al. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* arXiv:2112.01488.
- Santhanam, Khattab, et al. *PLAID: An Efficient Engine for Late Interaction Retrieval.* arXiv:2205.09707. (Candidate generation before full late scoring.)
- Faysse, Sibille, et al. *ColPali: Efficient Document Retrieval with Vision Language Models.* arXiv:2407.01449. (Late interaction on page images; origin of the ViDoRe benchmark line.)
- Dhulipala, Hadian, et al. *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings.* arXiv:2405.19504. (Reducing multi-vector search to single-vector candidate search.)
- Nogueira, Yang, et al. *Multi-Stage Document Ranking with BERT.* arXiv:1910.14424. (Cascade ranking: later stages only see earlier stages' candidates.)

**Tools:** Qdrant v1.19.2 (https://github.com/qdrant/qdrant/releases/tag/v1.19.2), qdrant-client 1.19.1, transformers 5.19.0, sentence-transformers 6.1.0 (`util.mean_maxsim`), datasets 5.1.0; OpenStax source books (see `docs/ATTRIBUTION.md`).

This repo does not claim novelty: candidate budgets for cascades are an established concern; the contribution is a pinned, reproducible measurement on one domain.

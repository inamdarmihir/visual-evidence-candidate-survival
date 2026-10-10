# Sources (each was opened and checked against the primary page)

| What | Source | What it was used for |
|---|---|---|
| ViDoRe V3 paper | arXiv:2601.08620, *ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios* (Loison, Macé, Edy, et al.) https://arxiv.org/abs/2601.08620 | benchmark definition; cite for the data |
| Dataset card | https://huggingface.co/datasets/vidore/vidore_v3_computer_science (pinned `d5cc75883d92e294f0c0fc2662551c9708a06ebc`), license CC BY 4.0 for annotations; qrels score 1 = *Critically Relevant*, 2 = *Fully Relevant*; source-document licenses are given per document in `documents_metadata["license"]` | schema, score meaning, licensing |
| Sibling domains (not yet run) | `vidore/vidore_v3_hr` `0cdf0979…`, `…_physics` `a0de276f…`, `…_energy` `caec06d3…`, `…_pharmaceuticals` `3abd4aa8…` (full SHAs in `src/vesk/config.py`) | optional cross-domain section |
| Model | https://huggingface.co/Hcompany/NeoMME-260M-Retriever (pinned `481083153ac22931afbd614e672212bffbcb666e`), Apache 2.0; NeoMME report arXiv:2609.01657 (Lac, Wu) https://arxiv.org/abs/2609.01657 | embeddings and scoring contract |
| Transformers | https://huggingface.co/docs/transformers/main/model_doc/neomme and the pinned v5.19.0 source (`max_side` is an image-processor kwarg passed via `processor_kwargs`; checkpoint default 2048; image processor needs torchvision) | processor usage |
| Qdrant server | https://github.com/qdrant/qdrant/releases/tag/v1.19.2 (asset SHA256 digests as published by GitHub; Linux x86_64 gnu `34a788a0…`) | pinned binary |
| qdrant-client 1.19.1 | schema inspected locally: `VectorParams.memory=Memory.COLD`, `Prefetch`, `SearchParams(exact)`, `MultiVectorConfig(MAX_SIM)` | collection design |
| Source texts | OpenStax *Introduction to Computer Science* and *Introduction to Python Programming* https://openstax.org/details/books/introduction-computer-science , https://openstax.org/details/books/introduction-python-programming | page images (not redistributed here) |
| sentence-transformers 6.1.0 | `sentence_transformers.util.mean_maxsim` | scoring reference for parity |

No result in this repo comes from any source other than the user-run notebook output in `results/`.

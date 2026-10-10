# Attribution

- **Annotations / qrels / metadata:** `vidore/vidore_v3_computer_science` (ViDoRe v3), CC BY 4.0. Cite: ViDoRe V3 (arXiv:2601.08620).
- **Corpus page images:** scanned pages of two OpenStax textbooks, *Introduction to Computer Science* and *Introduction to Python Programming* (https://openstax.org/details/books/introduction-computer-science, https://openstax.org/details/books/introduction-python-programming). The pinned `documents_metadata` rows state the OpenStax content is CC BY 4.0. Individual images may carry their own credited exceptions; this kit does not redistribute images and does not flatten third-party material to the annotation license. Page images are fetched from the source on demand.
- **Model:** `Hcompany/NeoMME-260M-Retriever` (Apache 2.0), H Company (arXiv:2609.01657). Model license is not a license for textbook content. Weights are not redistributed.
- **Qdrant** v1.19.2 (Apache 2.0) and qdrant-client; **transformers**, **sentence-transformers** (`mean_maxsim` is used as the scoring reference), **datasets** (Apache 2.0).
- Kit code: MIT (original). The scoring contract follows the NeoMME model card example.

# Visual Evidence Candidate-Survival Kit

Diagnostic: what happens to human-annotated relevant pages when a dense retriever passes only B ∈ {10,25,50,100}
candidates to a late-interaction reranker? Dataset `vidore/vidore_v3_computer_science` (pinned revision, 110 human English
queries), model `Hcompany/NeoMME-260M-Retriever` (pinned revision, one forward pass → dense + multivector),
Qdrant v1.19.2 local server. See `docs/PROTOCOL.md` and `docs/LIMITATIONS.md`.

## Run on Google Colab (free T4)
1. Open `notebooks/visual_evidence_candidate_survival_t4.ipynb`, Runtime → Change runtime type → T4 GPU.
2. Run the cells top to bottom with `MODE = "smoke"` (8 queries, 256 pages: engineering check only).
3. If the install cell tells you to restart the runtime, do so and run again from the top.
4. Set `MODE = "full"` and run again (1,360 pages, 110 queries). Every stage resumes from its checkpoints.

The notebook is self-contained: it writes the `vesk` package (shown in `src/vesk`) from embedded source.
CLI (non-Colab, GPU-enabled torch + torchvision already installed): `PYTHONPATH=src python -m vesk.cli smoke --root ./work`.
Tests (real pinned data, network needed): `pytest tests`.

## Results

Verified results from a user-run full computer_science run on a Colab Tesla T4 are in [`RESULTS.md`](RESULTS.md) and
`results/computer_science_primary_1024/`. Sources are listed in [`docs/SOURCES.md`](docs/SOURCES.md); a drafting prompt
for a write-up is in [`docs/BLOG_PROMPT.md`](docs/BLOG_PROMPT.md). The cross-domain section of the notebook has not been run yet.

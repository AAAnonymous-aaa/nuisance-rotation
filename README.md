# Nuisance removal is a rotation problem

Code for the paper "Nuisance removal is a rotation problem". Every experiment
runs from the repository root on a single 6 GB GPU (RTX 4050 laptop) and a
desktop CPU (Intel i5-13420H).

## Install

    pip install -r requirements.txt

The SPARC autoencoder is not distributed on PyPI; install the released `sparc`
package so that `sparc.model.get_sae_model_class` is importable.

Two checkpoints are not redistributed here and are expected at fixed paths: the
OpenAI CLIP ViT-L/14 weights at `clip_weights/open_clip_pytorch_model.bin`, and
the SPARC autoencoder at `model/` (`msae_checkpoint.pth` and
`run_config.json`). Both can be overridden with `--clip-weights`,
`--checkpoint-dir` or the `CLIP_WEIGHTS` variable. The other encoders are
fetched by `open_clip` and `torch.hub` on first use and cached afterwards; on a
machine without network, pre-populate those caches and set SURVEY_OFFLINE=1.

## How to run

Build a grid, run the survey on it, then the measurement and method scripts.
Everything is driven from the root, and the later scripts read the JSON reports
that the earlier ones write under `outputs/`:

    python dataset_generator.py dbpedia14.parquet -o survey_grid --layout-grid 6
    python backbone_survey.py --grid survey_grid --report-json outputs/backbone_survey.json
    python measure/strength_sweep.py
    python train_projection.py
    python method/fair_baselines.py
    python tables/make_tables.py
    python figures/make_figures.py

The remaining grids are built from public sources by `domains/*.py` (scan grids
downloaded through HuggingFace, arXiv first pages through the arXiv API,
identity documents from the filtered MIDV-500 distribution, synthetic shapes)
and by `topic_layout_probe.py` (topic grids over DBpedia-14, 20 Newsgroups and
FLORES-200). The identity grid is the filtered MIDV-500 distribution
(MIDV-500-filter), obtained from the Kaggle mirror
`cheickahmedcoulibaly/midv500-filtered`, kept as `archive.zip` and unpacked by
`domains/midv_from_zip.py`. Grids land under
`dataset/`, except the document grid the paper
reports, which is `survey_grid/`. A survey report has to keep the name the later
scripts expect: `outputs/backbone_survey.json`, `shape_survey.json`,
`funsd_scan_survey.json`, `docvqa_scan_survey.json`,
`real_scan_wide_survey.json`, `midv_full_survey.json`, `midv_survey.json` (for
`dataset/midv_capture`) and `domain_sae_survey.json`.

Feature extraction dominates the cost: a CLIP ViT-L/14 pass is about twenty
images per second here, features are cached in `outputs/cache/`, and peak GPU
memory stays around 2.4 GB with one encoder resident.

## Key scripts

    dataset_generator.py     renders a text corpus into a document grid
    topic_layout_probe.py    renders topic grids, probes topic against layout
    backbone_survey.py       content/nuisance geometry of one grid, per encoder
    factor_eval.py           geometry primitives: variance split, IDF, pairs, recall
    seed_variance.py         resampling error bars over a survey
    extract_fingerprints.py  builds the CLIP and SPARC retrieval database
    train_projection.py      fits the rotation, PCA on content-paired differences
    train_domain_sae.py      trains the in-domain sparse autoencoder
    domains/*.py             one grid builder per image domain
    measure/*.py             the bound, the diagonal frontier and their controls
    method/*.py              baselines (LEACE, INLP, PCA removal), rank choice,
                             retrieval intervals, downstream tasks
    autoencoder/*.py         SAE training at scale, top-k budget, selection rules
    tables/, figures/        rebuild the paper's tables and figures from outputs/
    checks/*.py              the language controls behind the script-locality claim

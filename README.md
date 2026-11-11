# BaPR — submission artifacts

This package contains only the implementations, fixed data splits, selected
weights and measurements used by the current BaPR manuscript. Historical
experiments, abandoned mechanisms and monitoring scripts are excluded.

## Download

- [Complete submission artifacts (code, fixed data, selected weights, caches and results)](https://github.com/weanpeter1-hub/BaPR/releases/tag/v1.0)
- [`BaPR-source.zip`](BaPR-source.zip): lightweight source, configurations and documentation. Extract this archive to obtain the directory structure below. For model evaluation, use the complete release archive.

## Contents

| Path | Purpose |
| --- | --- |
| `src/` | Base training, shared batch prompts, dual views and correction |
| `configs/` | Exact table configurations, selected seeds and stage records |
| `data/` | Fixed query manifests, filtered relations and frozen feature caches |
| `checkpoints/` | Selected LoRA/alignment weights and correction heads |
| `results/` | Final table/plot inputs; published baseline values are attributed |
| `scripts/` | Evaluation, correction refitting and figure/table regeneration |

## Setup

Use Python 3.10. The observed runtime versions are pinned in `requirements.txt`;
server environment snapshots are recorded in `environment/`. Install the CUDA
PyTorch build appropriate for the GPU. Model execution used int8 loading with
BF16 autocast; correction heads use FP32. Obtain Llama-2-7B from its official
distribution under its license. Neither Llama weights nor duplicate public
multimodal feature archives are bundled. Data access is described in
[`data/README.md`](data/README.md).

```bash
python -m pip install -r requirements.txt
python scripts/report_tables.py
python scripts/evaluate_heads.py --dataset all
python scripts/plot_ablation.py
python scripts/plot_residual.py
python scripts/plot_preservation_gate.py
```

`report_tables.py` regenerates tables from recorded measurements. `evaluate_heads.py`
executes the saved correction heads on frozen native test features; this is not
a fresh LLM forward pass. Outputs go to `generated/`.

To refit a specified correction condition using its original seed:

```bash
python scripts/train_head.py --dataset pog_dense --row full
```

The LLM training and native evaluation commands are in
[`REPRODUCE.md`](REPRODUCE.md). The package contains no automatic GPU selection,
remote execution or unattended search.

Packaging checks are summarized in `results/packaging_checks.json`. The saved
head evaluation matched 24 table entries and 73 parameter points. Full LLM
inference and complete training were not rerun during packaging.

## Reading the results

`configs/experiments.json` maps every displayed row to its checkpoint and frozen
base cache. Dense and Spotify main-table heads differ from their component
ablation heads; they are kept as separate configurations. POG's Repeat condition
retains the adapted parent selected at step zero. Gate removal reuses the same
head without refitting. Baseline scores quoted from Bundle-MLLM Table 2 are
published reference results, not reruns provided by this package.

Checkpoints within the recorded runs were selected on validation. Some displayed
configurations and runs were chosen retrospectively after test comparisons, and
the test set had earlier research exposure. Exact artifact reproduction does not
turn those choices into an untouched-test evaluation. The recorded results and
configuration mapping retain this distinction.

See `THIRD_PARTY.md` for code/data attribution and access terms.

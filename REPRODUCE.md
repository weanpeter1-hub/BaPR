# Reproduction guide

Run commands from the package root. Dataset names are `pog`, `pog_dense`,
`spotify`, and `spotify_sparse`. Outputs are written to `generated/` or a new
runtime; the bundled checkpoints are inputs.

## 1. Tables and figures — no GPU or external downloads

```bash
python scripts/report_tables.py
python scripts/evaluate_heads.py --dataset all
python scripts/evaluate_parameters.py
python scripts/plot_ablation.py
python scripts/plot_residual.py
python scripts/plot_preservation_gate.py
```

The table generator reads measurements. The two evaluators execute the saved
heads on frozen features. They require PyTorch but not a loaded language model.
The plotting commands use Matplotlib. `results/artifact_evaluation.json` records
the packaging-time CPU evaluation: the correct-answer counts matched all 24
main-table and ablation entries. This check does not rerun language-model
inference or training.

| Manuscript result | Configuration / measurements | Reproduction entry |
| --- | --- | --- |
| Main comparison | `configs/experiments.json`, row `main` | `evaluate_heads.py`, `report_tables.py` |
| Prompt adaptation | `results/adaptation.json` | `native_evaluate.py --condition before/after` |
| Dual-view ablation | rows `without_dual` and `full` | `evaluate_heads.py`, `plot_ablation.py` |
| Correction ablations | rows `full`, `without_relations`, `without_preservation`, `without_gate` | `evaluate_heads.py`, `plot_ablation.py` |
| Parameter analysis | `results/parameter_data.json` | `evaluate_parameters.py`, `plot_residual.py`, `plot_preservation_gate.py` |
| Text input | `results/token_comparison.csv`, `results/text_tokens.json` | `count_tokens.py`, `report_tables.py` |

Parameter-curve means and sample standard deviations use the three heads listed
at each point. They are not the single-run main-table scores. Baseline comparison
values retain their published source in `results/published_baselines.json`.

## 2. Fresh language-model inference

Obtain the public item assets and Llama-2-7B as described in `data/README.md`.
The data root must contain a subdirectory for each dataset, with
`content_feature.pt`, `description_feature.pt`, `item_cf_feature.pt`, `ui_full.txt`,
and `item_info.json`. The package supplies the filtered bundle graph and rebuilt
bundle–item vectors; do not substitute the original unfiltered vectors.

```bash
python scripts/prepare_runtime.py --dataset pog \
  --data-root /path/to/BundleConstruction --llama /path/to/llama-2-7b-hf
CUDA_VISIBLE_DEVICES=0 python scripts/native_evaluate.py --dataset pog --condition before
CUDA_VISIBLE_DEVICES=0 python scripts/native_evaluate.py --dataset pog --condition after
CUDA_VISIBLE_DEVICES=0 python scripts/native_evaluate.py --dataset pog --condition dual --head-row main
CUDA_VISIBLE_DEVICES=0 python scripts/native_evaluate.py --dataset pog --condition dual --head-row full
CUDA_VISIBLE_DEVICES=0 python scripts/native_evaluate.py --dataset pog --condition repeat
```

Use the same commands with each other dataset. Both adaptation conditions use
four-query prompts. `--condition before` loads the selected single-query
multimodal model; `after` loads the batch-adapted model. `dual` and `repeat` load
their respective selected continuation models and apply the specified saved
head. For POG, the selected Repeat branch is the adapted parent at step zero.

`--batch 1`, `--batch 2`, and `--batch 4` change prompt grouping while preserving
the recorded evaluation questions. Native inference needs a CUDA GPU and the
int8/BF16 execution dependencies. The selected runs used RTX PRO 6000 Blackwell;
device and software changes can affect numerical results. These new native CLI
entry points were assembled for this package; a fresh end-to-end LLM run was not
performed during packaging.

## 3. Refit a correction head

```bash
python scripts/train_head.py --dataset pog --row full --device cuda
python scripts/train_head.py --dataset pog_dense --row full
python scripts/train_head.py --dataset spotify --row without_relations
```

The row determines the original head seed, residual bound, feature mask and
preservation objective. Training uses the bundled correction cache and fits its
own normalizer. Validation selects the saved checkpoint. Test labels are not
read by this command. The dense full-system row uses group-balanced preservation
and therefore also reads the adapted parent's correction-pool predictions.
Gate removal applies the same full head with a zero switching margin; it does
not retrain a separate head.

These are refits, not guaranteed bitwise recreations of historical optimization.
Use the bundled selected heads for the recorded inference result. To train on
newly extracted features instead, pass `--cache /path/to/case` containing
`train.pt` and `validation.pt`; balanced preservation additionally accepts
`--parent-cache /path/to/parent_case/train.pt`.

## 4. Base training and the selected Repeat/Dual pair

The preceding `prepare_runtime.py` command creates a new training layout. To
rerun every base stage, execute the following in order:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_base.py --dataset pog --stage relations
CUDA_VISIBLE_DEVICES=0 python scripts/train_base.py --dataset pog --stage text
CUDA_VISIBLE_DEVICES=0 python scripts/train_base.py --dataset pog --stage multimodal
CUDA_VISIBLE_DEVICES=0 python scripts/train_base.py --dataset pog --stage adaptation
CUDA_VISIBLE_DEVICES=0 python scripts/train_pair.py --dataset pog
```

The commands use the supplied reserved/base bundle IDs, original dynamic
training recipe and validation questions. Relation pretraining writes new
vectors only inside the runtime. The text and multimodal stages dynamically
construct completion questions from the same 1,024 base bundles. Batch
adaptation consumes the supplied epoch manifests. The paired continuation uses
the recorded 512-question subset, 64 updates and shared parent, with only the
second-view arrangement changing between branches.

`train_pair.py` runs the single LLM seed in the dataset's final configuration;
`--seed` explicitly selects another seed. It retains the original three-head,
residual-bound-1.5 validation procedure used during continuation selection,
and writes every evaluated continuation model and correction/validation cache.
Final residual-bound and preservation settings are applied separately using
`train_head.py`. Thus the continuation-selection heads and the final display
heads are not conflated. The command does not evaluate the test split.

To skip earlier training, run only `--stage adaptation` on a prepared runtime;
it starts from the bundled single-query multimodal model. Running only
`train_pair.py` starts from the bundled batch-adapted model. For freshly trained
weights, `native_evaluate.py` accepts `--checkpoint /path/to/base` and
`--head-checkpoint /path/to/head.pt`.

Stage selections and seeds are recorded in `configs/stages/` and
`configs/experiments.json`. Full training was not rerun as part of artifact
assembly; fresh training may select different checkpoints. There is no claim
that the archived retrospective choice of displayed configurations is itself
a new validation-only selection experiment.

## 5. Text-token accounting

```bash
python scripts/count_tokens.py --data-root /path/to/BundleConstruction \
  --tokenizer /path/to/llama-2-7b-hf
```

This CPU-only command instantiates the prompt construction logic and tokenizer,
without loading LLM weights. It counts shared prompts at B=1/2/4 and the original
Bundle-MLLM prompt template on the same item content. Counts include BOS and
exclude padding and continuous multimodal embeddings. External-template counts
are measured here; they are not token figures quoted from the baseline paper.

## Package boundaries

The artifact is an implementation-and-results package, not a dump of research
history. It includes the selected weights, fixed splits and the heads needed for
the final parameter plots. It excludes unrelated trials, credentials, monitoring
jobs and the licensed Llama backbone. Historical provenance fields in saved
weights are not runtime dependencies. Public raw item assets are obtained through
the linked upstream distribution.

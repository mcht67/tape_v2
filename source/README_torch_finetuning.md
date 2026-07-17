# Adapting the TF pipeline to PyTorch fine-tuning

## Update: MultiTaskHead now mirrors TemporalCNN exactly (operates on spatial_embeddings)

The head was originally built on the pretrained encoder's *pooled* embedding
(one vector per clip), which meant `event_logits`/`framewise_polyphony_*`
couldn't be supported (no time axis left). It's now rebuilt to match
`model.py`'s `TemporalCNN` 1:1: it owns the same shared conv trunk
(`conv1`/`bn1`/`dropout1`/`conv2`/`bn2`) and consumes the frame-wise
`spatial_embeddings` sequence, so **all six objectives** are supported,
including the frame-wise ones.

This needed a small wiring change in `torch_models.py` so `spatial_embeddings`
reaches the head instead of `pooled_embeddings` — see `torch_models_patched.py`
(a patched copy of your uploaded `torch_models.py`; diff summary below).

| Backbone | Change needed | What I did |
|---|---|---|
| `BirdSetEfficientNet` | Code change (forward always used `pooled_features`) | Route to `spatial_embeddings` when `head.takes_spatial_embeddings` is set |
| `BirdSetBirdMAE` | Code change (forward always used the full sequence, even for single-vector heads — that was a latent bug) | Now computes both, routes by the same flag |
| `BirdSetAudioProtoPNet` | Had a `pooling=None` option that already does this | Added the same flag check for robustness so you don't have to remember to set `pooling=None` |
| `BirdSetAST` | Had `pooling=True` by default | Same flag check added; head now overrides the config default |
| `BirdSetWav2Vec2` | Had a `pooling=None` option that already does this | Same flag check added |

`MultiTaskHead.takes_spatial_embeddings = True` is the flag each wrapper now
checks. If you build other custom heads later that want the pooled vector
instead, they just don't set that attribute and nothing changes for them.

`MultiTaskHead.forward()` accepts either `(batch, time, freq, embedding)` or
`(batch, time, embedding)` — a freq axis, if present, is averaged out first,
exactly like `TemporalCNN`'s `tf.reduce_mean(inputs, axis=2)`.

## Files

| File | Role |
|---|---|
| `torch_multitask_head.py` | `MultiTaskHead` — mirrors `TemporalCNN`, operates on `spatial_embeddings`. Import it, or fold it into `torch_models.py`. |
| `torch_models_patched.py` | Your `torch_models.py` with the five wrapper-forward patches described above. Diff it against your original before overwriting. |
| `torch_losses.py` | Torch port of `losses.py`: `DynamicWeightedLoss`, `create_losses_from_objectives_torch`, `LossWeightScheduler` (plain class, called manually each epoch instead of via a Keras callback). Re-exports `compute_species_count_class_weights` from `losses.py` unchanged — it's plain numpy + `dataset.iter()`, no TF ops. |
| `torch_logging.py` | `TorchSummaryWriterLogger` — manual-loop equivalent of `CustomSummaryWriterCallback`, and `ModelAndHistorySaverTorch` — equivalent of `ModelAndHistorySaver`. Reuses `CustomSummaryWriter`, `plot_confusion_matrix_sklearn`, `build_confusion_matrix_specs`, `get_log_paths` from `logs.py` unchanged. |
| `torch_train.py` | Full training script, replacing your current one. |

## What logs identically to the TF pipeline

Same TensorBoard tag scheme, in the same `log_dir`:
- `loss_weights/{obj}` — current loss weight per objective
- `{obj}_loss` under `train/` and `validation/` subdirs — unweighted (base) loss
- Every other `logs` key (now including metrics from `metrics.py`, e.g. `val_{obj}_accuracy`, `val_{obj}_mae`, `val_{obj}_qwk`, ...) under `train/`/`validation/`
- `best/{metric_key}` and `best/{metric_key}_epoch` — staircase curves, same "higher is better" rule (`accuracy`, `f1`, `auc`)
- `Confusion_Matrix/{obj}` (single-objective), `Confusion_Matrix/{obj}/aggregated` + `.../per_species` (species objectives)
- `F1_per_species/*`, `F1_per_degree/*`, `F1_macro/*`, `F1_breakdown/*` (species objectives only)
- Hparams tab, populated with the same `Params` object and best-value metrics at the end of training
- `train_history.json` in the checkpoint dir, replayed into TensorBoard on resume (same as `_replay_history_to_tensorboard`)

## What's different / not ported

- **DVCLive, standard-TensorBoard-callback, model-graph tracing, remote rsync** from `CustomSummaryWriterCallback` aren't reproduced — none of that is TF- or Keras-specific to reimplement, but they weren't asked for. Say the word if you want them too (DVCLive in particular is a 10-line addition).
- **Metrics**: instead of porting the streaming `tf.keras.metrics.Metric` subclasses in `logs.py` (`RegressionCountPrecision` etc., which track macro precision/recall per count-class across batches), the torch script computes metrics once per epoch from the full validation-set predictions via `metrics.compute_polyphony_metrics` (already framework-agnostic, reused unchanged). You get `accuracy`, `off_by_one_accuracy`, `macro_f1`, `weighted_f1`, `qwk`, `mae`, `rmse`, `pearson_r` per objective instead of the TF pipeline's `accuracy`/`precision`/`recall`/`f1` — overlapping but not identical metric sets. Say the word if you need exact parity here.
- **Objectives supported by `MultiTaskHead`**: now all six from `model.py` — `polyphony_reg`, `polyphony_class`, `event_logits`, `framewise_polyphony_reg`, `framewise_polyphony_class`, `species_polyphony_reg`, `species_polyphony_class` — since it operates on `spatial_embeddings` (see the update section above).
- **Checkpointing**: unified into a single `torch.save({model_state_dict, optimizer_state_dict, epoch, loss_weights}, ...)` per epoch instead of the TF pipeline's separate full-model `.keras` / weights-only `.h5` distinction — PyTorch doesn't have that split.
- **Resume config key**: only `cfg.train.load_checkpoint_path` is honored (pointing at one of the `epoch_XXX.pt` / `best.pt` files); `load_model_path`/`load_history_path` distinction from `train.py` was collapsed since it's a TF-specific artifact split.

## Assumptions worth double-checking against your actual `params.yaml`

- `cfg.objectives` keys are exactly the four names `MultiTaskHead` supports (add more branches to `torch_multitask_head.py` / `torch_losses.py` / `cm_spec_type_for_objective` in `torch_train.py` if you use others).
- The dataset split used for torch training has an `audio` column (raw audio) *and* the objective label columns already present/derivable via `add_labels`/`get_birdset_id2label`, same as `train.py`'s dataset — since the torch pipeline runs the pretrained encoder live (true fine-tuning) rather than on precomputed embeddings, it needs raw audio, not the precomputed embedding columns `train.py` consumes.
- `utils.config.Params` / `utils.dataset.add_labels` / `get_birdset_id2label` behave the same regardless of caller (TF vs torch) — they weren't uploaded so I'm calling them exactly as `train.py` does and trusting that contract holds.

## Trying it

```bash
python torch_train.py   # reads params.yaml, same as train.py
tensorboard --logdir <cfg.path.train_output>/<run_dir>/logs
```

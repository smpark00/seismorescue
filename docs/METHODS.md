# Methods

## Benchmark: synthetic WWSSN films with exact ground truth

[`scripts/render_helicorder.py`](../scripts/render_helicorder.py) turns a real broadband record into a helicorder image together with per-line ground truth.

- **Geometry: two modes.**
  - `film` mode (main benchmark): 1 h per line at 1 px/s, 38 px line spacing, 8 h per record. This is 38 s of record per line spacing; real WWSSN long-period film has 40–47 s.
  - The first benchmark (`all` / `active` modes): 15 min per line at 4 px/s and 48 px spacing, which is 12 s per spacing.
  - The seconds-per-spacing ratio fixes the slopes and the number of crossings. Scan resolution does not change it, because the line spacing scales with dpi too.
- **Signal.**
  - IU/II LHZ records ([`scripts/fetch_film.py`](../scripts/fetch_film.py)), band-passed to 0.01–0.2 Hz.
  - The record starts 1–4 h before the P arrival, so the earthquake lands on line 2–5 and the other lines carry the station's real background noise, microseism included.
  - "Gain" scales the earthquake line's peak deflection to 1–8 line spacings.
- **Appearance.**
  - Photographic fading: exposure ∝ 1 / pen speed, with a floor of 0.1, so steep strokes are faint.
  - Minute marks, paper texture, and per-line ink layers that are added together.
- **Ground truth.** Saved per line in `gt_g<gain>.npz`:
  - `pen`: the pen centreline, including time marks.
  - `sig`: the signal only.
  - `mark`: the time marks.
  - `inst`: instance masks.
- **Splits.** Always by earthquake (event date): the same event never appears in training and test.

## Classical digitizer (proposers and judges)

- **A, greedy tracker.** Follows the nearest ink with a motion gate.
- **B, joint tracker** ([`scripts/track_joint.py`](../scripts/track_joint.py)).
  - Tracks all lines together and assigns ink runs to lines.
  - Through gaps and merges it coasts on a damped local polynomial fit, then reacquires.
  - Speckles are removed by a minimum ink area.
- **Disagreement events and judges** ([`scripts/arbiter.py`](../scripts/arbiter.py), [`scripts/cv_judges.py`](../scripts/cv_judges.py)).
  - Where A and B differ by more than 6 px, a judge picks one.
  - Judges: a rule, or logistic regression on physical features weighted by the pixels at stake. The decisive feature was baseline centring: a band-passed trace oscillates about its own zero line.
- **Active-line fix** ([`scripts/active_guard.py`](../scripts/active_guard.py)).
  - Erase the ink under B's quiet lines. The median row of the remaining ink marks the active line's baseline.
  - Re-track that line greedily on the cleaned ink (proposer C).
- **Muse Spark judge** ([`scripts/exp_spark_event_judge.py`](../scripts/exp_spark_event_judge.py)).
  - For the least confident events, Spark sees both candidates drawn on the crop and answers twice with the order swapped (single-turn requests).

## Learned extraction

### Row-anchor sequence model ([`scripts/ml_rowanchor.py`](../scripts/ml_rowanchor.py))
- **Target-conditioned input.**
  - A crop of ±10.5 line spacings centred on the target line's layout baseline, at half resolution (19 px per spacing, 2 s per column).
  - Channel 1: local ink darkness, normalised by the image's typical trace darkness.
  - Channel 2: a start hint, a Gaussian at the target's row in the first column. The classical trackers get the same start information.
- **Network** (884,281 parameters).
  - A small CNN pools 400 rows down to 25.
  - The per-column features feed a bidirectional 4-layer Transformer along time, with tokens of 4 s.
  - The head classifies each column into 200 row bins (4 px each at 38 px spacing), fusing the upsampled context with the 200-row CNN features.
  - Training uses Gaussian soft labels (σ = 1 bin).
- **Decoding.**
  - Viterbi over columns. Emissions are α·log p. Transitions are the empirical bin-step distribution of the training ground truth, with the start prior at the hint.
  - Both Viterbi and the argmax ablation are refined by a local expectation within ±2 bins.
  - α was chosen on validation (α = 4).
- **Training composites.** Other training images are stacked above and below the target with the line grid continued, so crops are full of lines as on a real 24-line record.
- **Budget.** 52 min on an Apple M5 Pro GPU (MPS), with the best checkpoint chosen on validation groups.

### SAM 2.1 fine-tune ([`scripts/ml_sam.py`](../scripts/ml_sam.py))
- **Model.**
  - The single-image path of SAM 2.1 hiera-tiny: Hiera trunk, FPN neck, prompt encoder and two-way mask decoder.
  - It is re-implemented in plain PyTorch under the official parameter names. The official checkpoint loads with strict key and shape checks, and the zero-shot IoU on test shapes is 0.991–0.996.
- **Task.**
  - A window of full height × 256 columns, resized to 1024 × 1024.
  - The first window gets a point prompt at the start position. Later windows get 4 points on the path decoded so far.
  - The stitched mask probabilities are decoded by Viterbi into one row per column, with smoothness λ chosen on validation (1e-5).
- **Training.** LoRA on the trunk and a trainable mask decoder; BCE + Dice loss; 4,326 steps, 45 min on MPS.

### Repair models, a negative result ([`scripts/ml_repair.py`](../scripts/ml_repair.py))
The input is B's per-line series, as positions relative to the target's baseline, plus existence masks for up to ±7 neighbouring lines. The output is the target's true series.
- **Pointer Transformer:** a convolutional stem and a bidirectional Transformer; at each column a pointer selects which B line carries the target, plus a residual.
- **Oracle-supervised pointer:** the same model, with its pointer trained on the ground-truth choice.
- **Own-line-only ablation.**
- **Spectrogram U-Net:** STFT of the target's own series, a 2D U-Net, then the inverse STFT.

## Real paired evaluation ([`scripts/real_paired_eval.py`](../scripts/real_paired_eval.py))
- **Scans.**
  - MAT LP-Z (EarthScope film-chip product, 2.17 px/s).
  - KIP LP-Z (3.85 px/s, local copy, not redistributed).
  - Both are deskewed, laid out by row profile, and rescaled so the line spacing is 38 px.
- **Reference.**
  - The co-located HGLP digital record (1 sample/s, unclipped) has its instrument response removed, and the WWSSN LP response is applied. That response is critically damped with To = 15 s and Tg = 100 s: zeros at 0 (×3), poles at −2π/15 (×2) and −2π/100 (×2).
  - The result is band-passed to 0.0025–0.08 Hz.
- **Alignment.**
  - Line model: t = T0 + period·k + (x − xa)/pps.
  - It is fitted once by dynamic programming on the scan and stored before any method is scored. The fitted period is 3588–3589 s.
  - A global amplitude scale converts line spacings to ground motion.
- **Metrics per line category** (quake, neighbours, other active, quiet):
  - `r_band`: correlation.
  - `in_0.1S`: share of samples within 0.1 line spacing.
  - `lost_0.5S`: share of samples more than 0.5 spacing off.
  - Chance level for |r| is about 0.06–0.07.

## Seismologist agent with Muse Spark ([`scripts/seismologist_agent.py`](../scripts/seismologist_agent.py), [`scripts/seismo_tools.py`](../scripts/seismo_tools.py))
- **Model.** Spark (`muse-spark-1.3-contributor`) uses function calling over tools: event catalogue lookup with TauP (iasp91) travel times, STA/LTA, an AIC picker, peak finding, a line-shift similarity check, and plots.
- **Tasks.** Identify the event, pick the P arrival and the time of the surface-wave maximum on digitized series, and flag spans the digitization got wrong.
- **Calls.** Multi-turn function calling on the Responses API; the full conversation history is sent each turn.
- **Data hygiene.** Ground-truth files are kept apart from what the agent can read.
- **Evaluation.** The agent is compared with no-LLM baselines built without peeking at the test items (see [RESULTS.md](RESULTS.md)).

# Noise Robustness Analysis Execution Plan (Accuracy vs SNR)

## Objective
Evaluate how the trained model’s classification accuracy degrades under additive Gaussian sensor noise at different Signal-to-Noise Ratios (SNRs), and generate the figure titled **“Robustness to Sensor Noise”** (Accuracy vs SNR).

---

## 1) Gaussian Noise Injection at Target SNRs

### 1.1 Where to inject noise
- **Default (recommended for reproducibility):** inject noise **at the model input tensor** (i.e., after any dataset preprocessing/normalization), right before `model(x)`.
- **Optional (physical-sensor SNR):** inject noise **in raw sensor units** before normalization (requires hooking earlier in the dataset pipeline).

This plan assumes **noise at model input** unless explicitly switched.

### 1.2 SNR definition and noise scaling
Let an input window be `x` (per sample) with shape typically `[C, T]` (or `[T, C]`), and `x_noisy = x + n`.

- Choose a target `SNR_dB` (e.g., `40, 30, 20, 10, 5, 0, -5` dB).
- Convert to linear:
  - `SNR_lin = 10^(SNR_dB / 10)`
- Compute signal power (per-sample by default):
  - `P_s = mean(x^2)` over all elements of the sample (optionally after de-meaning).
- Set noise power:
  - `P_n = P_s / SNR_lin`
- Noise standard deviation:
  - `sigma = sqrt(P_n)`
- Sample zero-mean Gaussian noise:
  - `n ~ Normal(0, sigma^2)` with the same shape as `x`
- Add noise:
  - `x_noisy = x + n`

### 1.3 Power estimation options
Provide a switch to control how `P_s` (and thus `sigma`) is computed:
- `snr_scope=per_sample` (default): one `sigma` per sample using power over all channels/time.
- `snr_scope=per_channel`: compute `sigma` per channel using power over time; preserves channel-wise SNR.

### 1.4 De-meaning (optional)
If inputs contain a DC offset, power can be inflated. Optionally compute power on:
- `x_centered = x - mean(x)` (mean over `(C,T)` or over `T` per channel)

---

## 2) Accuracy Evaluation for Each SNR

### 2.1 Evaluation loop (per SNR level)
For each `SNR_dB` in the configured list:
1. Load checkpoint (`--ckpt ...`) and set `model.eval()`.
2. Disable gradients (`torch.no_grad()`).
3. Iterate the evaluation dataloader (same split/fold protocol as standard reporting).
4. For each batch:
   - Obtain `x` and `y`.
   - If `SNR_dB` is `inf` / `None`: use clean `x`.
   - Else: build `x_noisy` using the SNR method above.
   - Forward pass and compute predictions.
   - Accumulate `correct` and `total`.
5. Compute top-1 accuracy:
   - `acc = correct / total`
6. Record `{snr_db, acc, n_samples, ckpt, seed, repeat_id, snr_scope, demean, timestamp}`.

### 2.2 Noise randomness and repeats
Because noise is stochastic, use multiple runs per SNR:
- Add `--repeats N` (e.g., `10`) and `--seed`.
- For each SNR, evaluate `N` times with different noise realizations (e.g., by offsetting the RNG seed per repeat).
- Report:
  - `mean_accuracy` and `std_accuracy` across repeats (optionally 95% CI).

### 2.3 Sanity checks
- Confirm **clean accuracy** (no noise or high SNR like 40 dB) matches the usual evaluation within rounding.
- Keep dataloader order fixed (no shuffle) to isolate the effect of noise.

---

## 3) Plot “Accuracy vs SNR” and Generate the Figure

### 3.1 Plot specification
- X-axis: `SNR (dB)` (sorted; choose increasing or decreasing consistently).
- Y-axis: `Accuracy (%)`.
- Plot: line + markers.
- If `repeats > 1`: add error bars (`±1 std` or CI).
- Title: `Robustness to Sensor Noise`

### 3.2 Outputs
- Save the plot into the chosen figure directory (e.g., `--figure-dir ./figures/demo`):
  - `Robustness_to_Sensor_Noise.png` (optionally also `.pdf`)
- Save raw results next to the figure for traceability:
  - `noise_robustness_results.json` (and/or `.csv`)
  - Include full CLI args and metadata (checkpoint path, seeds, SNR list, repeats, snr_scope, demean).

---

## 4) Robustness Considerations (Recommended Defaults)
- **Noise domain:** inject noise at the model input tensor (post-normalization) for reproducibility; optionally support pre-normalization injection if physical-sensor SNR is required.
- **Repeatability:** fix `--seed`; use `--repeats` to estimate variability.
- **Compute locality:** generate noise on the same device as `x` (CPU/GPU) to avoid overhead and ensure correctness.
- **Optional metrics:** accuracy is the primary curve; optionally log balanced accuracy / macro-F1 (especially under class imbalance) without changing the main figure.
- **Clipping/saturation:** do not clip after noise unless the pipeline already clips or you want to emulate sensor saturation.

---

## Proposed Default Configuration
- `SNRs (dB)`: `40, 30, 20, 10, 5, 0, -5`
- `snr_scope`: `per_sample`
- `demean`: `true`
- `repeats`: `10`
- `seed`: `42`


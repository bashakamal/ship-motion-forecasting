"""
Ship Motion Forecasting — Stakeholder Edition (Simplified)
==========================================================
Three tabs only, built for explaining results to stakeholders:

  TAB 1 — Prediction      : pick context window (120s, 240s ...) + future horizon
                            (next 3s / 6s / 10s ...). Shows a qualitative graph
                            (history -> predicted future, peaks labelled, roll AND
                            pitch in degrees) plus a quantitative table.
  TAB 2 — Statistics      : Peak / RMS / H1/3 with MAPE, MAE, MSE (as in Prediction.py).
  TAB 3 — Data Analytics  : predicted statistics across horizons, with peak roll
                            plotted on a clearly-labelled graph.

NATO STANAG content has been removed.
Engine: TimesFM 2.5 zero-shot (same as the original app).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ship Motion Forecasting", page_icon="🚢", layout="wide")
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
for key, default in [("ready", False), ("data", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def remove_outliers(series, z_thresh=3.0):
    s = series.copy().astype(float)
    mu, sigma = s.mean(), s.std()
    bad = np.abs(s - mu) > z_thresh * sigma
    s[bad] = np.nan
    s = s.interpolate(method="linear").ffill().bfill()
    return s, int(bad.sum())

def butter_lp(data, cutoff=2.0, fs=20.0, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, data).astype(np.float32)

def butter_bp(data, low, high, fs=20.0, order=4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")
    return filtfilt(b, a, data).astype(np.float32)

# Statistical descriptors — same definitions as Prediction.py
def compute_peak(x): return float(np.max(np.abs(x)))
def compute_rms(x):  return float(np.sqrt(np.mean(np.array(x, dtype=float) ** 2)))
def compute_h13(x):
    arr = np.abs(np.asarray(x, dtype=float))
    peaks, _ = find_peaks(arr)
    if len(peaks) == 0:
        s = np.sort(arr)[::-1]
        n = max(1, len(s) // 3)
        return float(np.mean(s[:n]))
    s = np.sort(arr[peaks])[::-1]
    n = max(1, len(s) // 3)
    return float(np.mean(s[:n]))

def stat_fn(x, metric):
    return {"Peak": compute_peak, "RMS": compute_rms, "H1/3": compute_h13}[metric](x)

@st.cache_resource(show_spinner="Loading TimesFM model (~800 MB, once only)...")
def load_model():
    from timesfm import TimesFM_2p5_200M_torch
    return TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

def tfm_predict(model, signal, cut, ctx_len, horizon):
    """Forecast `horizon` samples after `cut`, using `ctx_len` samples of history.
    Returns (actual_future_or_None, prediction)."""
    from timesfm import ForecastConfig
    model.compile(ForecastConfig(max_context=ctx_len, max_horizon=horizon))
    ctx = signal[cut - ctx_len:cut].astype(np.float32)
    local_mean = ctx.mean()
    forecast, _ = model.forecast(horizon=horizon, inputs=[ctx - local_mean])
    pred = (forecast[0] + local_mean).astype(np.float32)
    actual = (signal[cut:cut + horizon].astype(np.float32)
              if cut + horizon <= len(signal) else None)
    return actual, pred

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚢 Ship Motion Forecasting")
    st.caption("TimesFM Zero-Shot · Stakeholder view")
    st.divider()

    uploaded = st.file_uploader("Upload IMU CSV", type=["csv"],
                                help="Required: timestamp (Unix s), roll_deg, pitch_deg")

    st.subheader("Evaluation settings")
    st.caption("Used only for the Statistics tab (needs ground truth).")
    n_windows = st.slider("Evaluation windows", 3, 10, 5)

    run_btn = st.button("Load & prepare data", type="primary", use_container_width=True)

    if st.session_state.ready:
        if st.button("Clear / upload new file", use_container_width=True):
            st.session_state.ready = False
            st.session_state.data = {}
            st.rerun()

    st.divider()
    st.caption("Columns: `timestamp` or `time_sec`, `roll_deg`, `pitch_deg`. "
               "Optional: `yaw_deg`, `gx`, `gy`, `gz`.")

# ──────────────────────────────────────────────────────────────────────────────
# WELCOME
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.ready and not run_btn:
    st.markdown("## Welcome")
    if uploaded is None:
        st.info("Upload an IMU CSV in the sidebar, then click **Load & prepare data**.")
    else:
        st.info("File uploaded. Click **Load & prepare data** in the sidebar.")
    st.markdown("""
**Three tabs, built for stakeholders:**

1. **🔮 Prediction** — choose how much history to use (context window) and how far
   ahead to look (next 3s / 6s / 10s ...). See the future roll and pitch on a graph,
   plus the numbers.
2. **📊 Statistics** — accuracy of the predicted Peak, RMS and H1/3, reported as
   MAPE, MAE and MSE.
3. **📈 Data Analytics** — how predicted statistics (including peak roll) change
   as the horizon grows.

**Data format:**
```
timestamp,roll_deg,pitch_deg
1764676000.00,-5.845,-3.808
1764676000.05,-5.539,-4.089
```
""")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# DATA PREP — runs when button clicked
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    if uploaded is None:
        st.error("Please upload a CSV file first."); st.stop()

    progress = st.progress(0, text="Loading data...")
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    ts_col = next((c for c in ["timestamp", "time_sec", "time"] if c in df_raw.columns), None)
    if ts_col is None:
        st.error("No timestamp column found."); st.stop()
    for col in ["roll_deg", "pitch_deg"]:
        if col not in df_raw.columns:
            st.error(f"Missing required column: {col}"); st.stop()

    df_raw["time_sec"] = df_raw[ts_col] - df_raw[ts_col].iloc[0]
    DUR_MIN = df_raw["time_sec"].iloc[-1] / 60
    ts = df_raw[ts_col].values.astype(np.float64)
    DT_MEAN = np.diff(ts).mean()
    FS_RAW = 1.0 / DT_MEAN

    # Auto target rate: 10 Hz native or 20 Hz standardised
    TARGET_HZ = 10.0 if (1.0 / DT_MEAN) < 15.0 else 20.0

    progress.progress(20, text=f"Resampling to {TARGET_HZ:.0f} Hz...")
    IMU_COLS = ["roll_deg", "pitch_deg", "yaw_deg", "gx", "gy", "gz"]
    t0, t1 = ts[0], ts[-1]
    t_uniform = np.arange(t0, t1, 1.0 / TARGET_HZ)
    df_rs = pd.DataFrame({"timestamp": t_uniform, "time_sec": t_uniform - t0})
    for col in IMU_COLS:
        if col in df_raw.columns:
            df_rs[col] = np.interp(t_uniform, ts, df_raw[col].values.astype(float))
    FS = TARGET_HZ

    progress.progress(45, text="Cleaning signal...")
    df = df_rs.copy()
    outlier_log = {}
    for col in ["roll_deg", "pitch_deg"]:
        df[col], n_out = remove_outliers(df[col])
        outlier_log[col] = n_out

    # Filter + roll decomposition (slow sway + fast wave) — improves roll forecasts
    df["pitch_filt"] = butter_lp(df["pitch_deg"].values, 2.0, FS)
    df["roll_slow"]  = butter_lp(df["roll_deg"].values, 0.05, FS)
    df["roll_fast"]  = butter_bp(df["roll_deg"].values, 0.05, 2.0, FS)
    df["roll_filt"]  = butter_lp(df["roll_deg"].values, 2.0, FS)

    df_clean = pd.DataFrame({
        "time_sec":  df["time_sec"].values,
        "roll_deg":  df["roll_filt"].values,
        "roll_slow": df["roll_slow"].values,
        "roll_fast": df["roll_fast"].values,
        "pitch_deg": df["pitch_filt"].values,
    })

    progress.progress(100, text="Ready."); progress.empty()

    st.session_state.data = dict(
        df_raw=df_raw, df_rs=df_rs, df_clean=df_clean,
        FS=FS, TARGET_HZ=TARGET_HZ, FS_RAW=FS_RAW, DUR_MIN=DUR_MIN,
        DT_MEAN=DT_MEAN, outlier_log=outlier_log,
        n_windows=n_windows, filename=uploaded.name,
    )
    st.session_state.ready = True

# ──────────────────────────────────────────────────────────────────────────────
# UNPACK
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.ready:
    st.stop()

d = st.session_state.data
df_raw = d["df_raw"]; df_clean = d["df_clean"]
FS = d["FS"]; TARGET_HZ = d["TARGET_HZ"]; DUR_MIN = d["DUR_MIN"]
n_windows = d["n_windows"]
roll_raw  = df_clean["roll_deg"].values.astype(np.float32)
roll_slow = df_clean["roll_slow"].values.astype(np.float32)
roll_fast = df_clean["roll_fast"].values.astype(np.float32)
pitch_raw = df_clean["pitch_deg"].values.astype(np.float32)
CTX_360 = int(360 * FS)
model = load_model()

st.caption(f"File: **{d['filename']}**  ·  {DUR_MIN:.1f} min  ·  {TARGET_HZ:.0f} Hz  ·  {len(df_clean):,} samples")

tab_pred, tab_stats, tab_analytics = st.tabs([
    "🔮 Prediction",
    "📊 Statistics",
    "📈 Data Analytics",
])

# Shared horizon options (wider set 3–120s)
HORIZON_OPTIONS = [3, 6, 10, 20, 30, 60, 120]
CONTEXT_OPTIONS = [60, 120, 180, 240, 360]
HORIZON_COLORS  = ["#185FA5", "#D85A30", "#3B6D11", "#534AB7",
                   "#BA7517", "#0D7A7A", "#993C1D"]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION  (qualitative graph + quantitative table)
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Future Prediction")
    st.caption("Uses the most recent slice of your recording as context and forecasts "
               "forward. The most recent data point is **NOW**; everything to its right "
               "is the predicted future.")

    c1, c2 = st.columns(2)
    with c1:
        ctx_sec = st.select_slider(
            "Context window — how much past data to use",
            options=CONTEXT_OPTIONS, value=120,
            format_func=lambda x: f"{x}s  ({x // 60} min of history)", key="p_ctx")
    with c2:
        horizons = st.multiselect(
            "Future horizon — how far ahead to predict",
            options=HORIZON_OPTIONS, default=[3, 6, 10],
            format_func=lambda x: f"next {x}s", key="p_hor")

    if not horizons:
        st.warning("Select at least one future horizon.")
        st.stop()

    horizons = sorted(horizons)
    ctx = int(ctx_sec * FS)
    cut = len(df_clean)

    if cut < ctx + 10:
        st.error(f"Not enough data for a {ctx_sec}s context. Upload a longer recording.")
        st.stop()

    ctx_pitch = pitch_raw[cut - ctx:cut]
    ctx_roll  = roll_raw[cut - ctx:cut]
    t_ctx = np.arange(len(ctx_pitch)) / FS

    results = {}
    bar = st.progress(0, text="Forecasting...")
    for ki, h_sec in enumerate(horizons):
        h_samps = int(h_sec * FS)
        _, p_p  = tfm_predict(model, pitch_raw, cut, ctx, h_samps)
        _, p_rf = tfm_predict(model, roll_fast, cut, ctx, h_samps)
        c360    = min(CTX_360, cut)
        _, p_rs = tfm_predict(model, roll_slow, cut, c360, h_samps)
        p_roll  = p_rf + p_rs
        t_pred  = ctx_sec + np.arange(h_samps) / FS
        results[h_sec] = dict(p_pitch=p_p, p_roll=p_roll, t_pred=t_pred, h_samps=h_samps)
        bar.progress((ki + 1) / len(horizons), text=f"Predicted next {h_sec}s...")
    bar.empty()

    # ── QUALITATIVE GRAPH ─────────────────────────────────────────────────────
    st.markdown("### Qualitative view — predicted motion")
    st.caption("Gray = recorded past · black dashed = NOW · coloured dashed = predicted "
               "future · ★ marks the predicted peak (largest |angle|) of the longest horizon.")

    for sig_label, ctx_sig, pred_key in [("Roll", ctx_roll, "p_roll"),
                                         ("Pitch", ctx_pitch, "p_pitch")]:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(t_ctx, ctx_sig, color="gray", lw=1.0,
                label=f"Recorded (last {ctx_sec}s)", zorder=3)
        ax.axvline(ctx_sec, color="black", lw=1.5, ls="--", zorder=4, label="NOW")
        ax.axvspan(ctx_sec, ctx_sec + max(horizons), alpha=0.04, color="#185FA5")

        for i, h_sec in enumerate(horizons):
            r = results[h_sec]
            ax.plot(r["t_pred"], r[pred_key],
                    color=HORIZON_COLORS[i % len(HORIZON_COLORS)],
                    lw=1.8, ls="--", label=f"Predict next {h_sec}s", zorder=5)

        # mark peak of the longest horizon for a clear stakeholder takeaway
        long_h = horizons[-1]
        rr = results[long_h]
        pk_i = int(np.argmax(np.abs(rr[pred_key])))
        pk_t = rr["t_pred"][pk_i]; pk_v = rr[pred_key][pk_i]
        ax.scatter([pk_t], [pk_v], marker="*", s=220, color="#D81B60", zorder=6,
                   label=f"Predicted peak {abs(pk_v):.2f}°")
        ax.annotate(f"peak {abs(pk_v):.2f}°", (pk_t, pk_v),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="#D81B60", fontweight="bold")

        ax.set_xlabel("Time (s)   [0 = start of context window]")
        ax.set_ylabel(f"{sig_label} angle (deg)")
        ax.set_title(f"{sig_label} — recorded history + predicted future "
                     f"(context = last {ctx_sec}s)")
        ax.legend(fontsize=8, loc="upper left",
                  ncol=min(4, len(horizons) + 2))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── QUANTITATIVE TABLE ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Quantitative view — the numbers")
    st.caption("For each horizon: the predicted peak and RMS of roll and pitch, "
               "and whether the sea is expected to get rougher or calmer vs the "
               "matching slice of recent history.")

    q_rows = []
    for h_sec in horizons:
        r = results[h_sec]; h_samps = r["h_samps"]
        for sig_label, ctx_sig, pred_key in [("Roll", ctx_roll, "p_roll"),
                                             ("Pitch", ctx_pitch, "p_pitch")]:
            pred_w = r[pred_key]
            inp_w  = ctx_sig[-h_samps:] if h_samps <= len(ctx_sig) else ctx_sig
            pk_in, pk_pr = compute_peak(inp_w), compute_peak(pred_w)
            rms_in, rms_pr = compute_rms(inp_w), compute_rms(pred_w)
            chg = (rms_pr - rms_in) / (rms_in + 1e-9) * 100
            trend = "calming" if chg < -3 else "rougher" if chg > 3 else "stable"
            q_rows.append({
                "Horizon": f"next {h_sec}s", "Signal": sig_label,
                "Peak now (°)": round(pk_in, 3), "Peak predicted (°)": round(pk_pr, 3),
                "RMS now (°)": round(rms_in, 3), "RMS predicted (°)": round(rms_pr, 3),
                "RMS change (%)": round(chg, 1), "Trend": trend,
            })
    q_df = pd.DataFrame(q_rows)

    def color_trend(v):
        return ("color: green" if v == "calming"
                else "color: red" if v == "rougher" else "")
    st.dataframe(q_df.style.map(color_trend, subset=["Trend"]),
                 use_container_width=True, hide_index=True)

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    dl_rows = []
    for h_sec in horizons:
        r = results[h_sec]
        for tp, pp, pr in zip(r["t_pred"], r["p_pitch"], r["p_roll"]):
            dl_rows.append({"horizon_sec": h_sec,
                            "time_from_now_s": round(tp - ctx_sec, 3),
                            "pred_roll_deg": round(float(pr), 4),
                            "pred_pitch_deg": round(float(pp), 4)})
    st.download_button("Download predictions (CSV)",
                       data=pd.DataFrame(dl_rows).to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv", key="dl_pred")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICS  (Peak / RMS / H1/3 with MAPE, MAE, MSE)
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("Statistical Accuracy — Peak · RMS · H1/3")
    st.caption("Same descriptors as the offline study. For each horizon the model "
               "predicts the Peak, RMS and H1/3 of the next window; we compare against "
               "the true future and report **MAPE, MAE, MSE** across evaluation windows.")

    stat_ctx_sec = st.select_slider(
        "Context window for evaluation",
        options=CONTEXT_OPTIONS, value=120,
        format_func=lambda x: f"{x}s", key="s_ctx")
    stat_horizons = st.multiselect(
        "Horizons to evaluate", options=HORIZON_OPTIONS, default=[3, 6, 10, 30],
        format_func=lambda x: f"{x}s", key="s_hor")

    if not stat_horizons:
        st.warning("Select at least one horizon.")
        st.stop()

    stat_horizons = sorted(stat_horizons)
    s_ctx = int(stat_ctx_sec * FS)
    max_h = int(max(stat_horizons) * FS)

    if len(df_clean) < s_ctx + max_h + 10:
        st.error("Not enough data for these settings. Reduce context/horizon or upload "
                 "a longer recording."); st.stop()

    # evaluation cut points with room for context behind and horizon ahead
    cut_points = np.linspace(s_ctx + max_h, len(df_clean) - max_h,
                             n_windows, dtype=int)

    STAT_METRICS = ["Peak", "RMS", "H1/3"]
    rows = []
    bar = st.progress(0, text="Evaluating...")
    total = len(stat_horizons)
    for hi, h_sec in enumerate(stat_horizons):
        h_samps = int(h_sec * FS)
        # collect true & predicted descriptor values across windows
        acc = {sig: {m: {"true": [], "pred": []} for m in STAT_METRICS}
               for sig in ["Roll", "Pitch"]}
        for cut in cut_points:
            a_p, p_p   = tfm_predict(model, pitch_raw, cut, s_ctx, h_samps)
            a_rf, p_rf = tfm_predict(model, roll_fast, cut, s_ctx, h_samps)
            c360       = min(CTX_360, cut)
            a_rs, p_rs = tfm_predict(model, roll_slow, cut, c360, h_samps)
            a_roll = roll_raw[cut:cut + h_samps]
            p_roll = p_rf + p_rs
            for sig, a_sig, p_sig in [("Roll", a_roll, p_roll), ("Pitch", a_p, p_p)]:
                for m in STAT_METRICS:
                    acc[sig][m]["true"].append(stat_fn(a_sig, m))
                    acc[sig][m]["pred"].append(stat_fn(p_sig, m))
        # reduce to MAPE / MAE / MSE
        eps = 1e-8
        for sig in ["Roll", "Pitch"]:
            for m in STAT_METRICS:
                t = np.array(acc[sig][m]["true"])
                p = np.array(acc[sig][m]["pred"])
                mape = float(np.mean(np.abs((p - t) / (np.abs(t) + eps))) * 100)
                mae  = float(mean_absolute_error(t, p))
                mse  = float(mean_squared_error(t, p))
                rows.append({"Signal": sig, "Horizon": h_sec, "Metric": m,
                             "True_mean": round(float(t.mean()), 4),
                             "Pred_mean": round(float(p.mean()), 4),
                             "MAPE_%": round(mape, 2),
                             "MAE": round(mae, 4),
                             "MSE": round(mse, 5)})
        bar.progress((hi + 1) / total, text=f"Evaluated {h_sec}s...")
    bar.empty()

    stat_df = pd.DataFrame(rows)
    st.session_state.data["stat_df"] = stat_df  # share with analytics tab

    for sig in ["Roll", "Pitch"]:
        st.markdown(f"**{sig} motion**")
        sub = stat_df[stat_df["Signal"] == sig].drop(columns=["Signal"]).copy()
        sub["Horizon"] = sub["Horizon"].apply(lambda x: f"{x}s")
        sub = sub.rename(columns={"True_mean": "True", "Pred_mean": "Predicted",
                                  "MAPE_%": "MAPE (%)"})
        st.dataframe(sub, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Overall accuracy (averaged over all selected horizons)")
    ov = (stat_df.groupby(["Signal", "Metric"])[["MAPE_%", "MAE", "MSE"]]
          .mean().round(4).reset_index()
          .rename(columns={"MAPE_%": "MAPE (%)"}))
    st.dataframe(ov, use_container_width=True, hide_index=True)

    st.download_button("Download statistics (CSV)",
                       data=stat_df.to_csv(index=False),
                       file_name="statistics.csv", mime="text/csv", key="dl_stats")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA ANALYTICS  (predicted stats across horizons, peak roll graph)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Data Analytics")
    st.caption("How the predicted statistics evolve as the forecast horizon grows. "
               "Run the **Statistics** tab first so the analytics have data to chart.")

    stat_df = st.session_state.data.get("stat_df")
    if stat_df is None or stat_df.empty:
        st.info("No statistics yet — open the **📊 Statistics** tab and run an evaluation, "
                "then come back here.")
        st.stop()

    horizons_a = sorted(stat_df["Horizon"].unique())
    x = np.arange(len(horizons_a))

    # ── Peak roll across horizons — the headline graph ────────────────────────
    st.markdown("### Predicted **Peak Roll** by horizon")
    st.caption("The single most operationally important number — the largest roll angle "
               "the vessel is expected to reach. Bars = predicted peak; line = true peak.")

    roll_peak = stat_df[(stat_df["Signal"] == "Roll") & (stat_df["Metric"] == "Peak")] \
        .set_index("Horizon").reindex(horizons_a)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, roll_peak["Pred_mean"].values, width=0.55, color="#185FA5",
           alpha=0.85, label="Predicted peak roll")
    ax.plot(x, roll_peak["True_mean"].values, "o-", color="#D81B60", lw=2,
            markersize=7, label="True peak roll")
    for xi, (pv, tv) in enumerate(zip(roll_peak["Pred_mean"].values,
                                      roll_peak["True_mean"].values)):
        ax.annotate(f"{pv:.2f}°", (xi, pv), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color="#185FA5",
                    fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}s" for h in horizons_a])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Peak roll angle (deg)")
    ax.set_title("Predicted vs true peak roll across horizons")
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Peak / RMS / H1/3 for roll & pitch ────────────────────────────────────
    st.divider()
    st.markdown("### Predicted descriptors — Roll & Pitch")
    colors_met = {"Peak": "#D85A30", "RMS": "#185FA5", "H1/3": "#3B6D11"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, sig in zip(axes, ["Roll", "Pitch"]):
        sub = stat_df[stat_df["Signal"] == sig]
        w = 0.25
        for j, m in enumerate(["Peak", "RMS", "H1/3"]):
            mm = sub[sub["Metric"] == m].set_index("Horizon").reindex(horizons_a)
            ax.bar(x + j * w, mm["Pred_mean"].values, w, label=m,
                   color=colors_met[m], alpha=0.85)
        ax.set_xticks(x + w); ax.set_xticklabels([f"{h}s" for h in horizons_a])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Predicted amplitude (deg)")
        ax.set_title(f"{sig} — predicted Peak / RMS / H1/3")
        ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Accuracy (MAPE) across horizons ───────────────────────────────────────
    st.divider()
    st.markdown("### Prediction error (MAPE) by horizon")
    st.caption("Lower is better. Shows how accuracy degrades as we forecast further ahead.")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, sig in zip(axes, ["Roll", "Pitch"]):
        sub = stat_df[stat_df["Signal"] == sig]
        w = 0.25
        for j, m in enumerate(["Peak", "RMS", "H1/3"]):
            mm = sub[sub["Metric"] == m].set_index("Horizon").reindex(horizons_a)
            ax.bar(x + j * w, mm["MAPE_%"].values, w, label=m,
                   color=colors_met[m], alpha=0.85)
        ax.set_xticks(x + w); ax.set_xticklabels([f"{h}s" for h in horizons_a])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{sig} — prediction error by horizon")
        ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

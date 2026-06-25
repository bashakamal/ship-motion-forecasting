"""
Ship Motion Forecasting — Streamlit Web App
Converted from ship_motion_final_Dr_Suresh.ipynb
Upload IMU CSV → preprocessing → TimesFM zero-shot forecast → results """

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ship Motion Forecasting",
    page_icon="🚢",
    layout="wide",
)

plt.rcParams.update({
    "figure.dpi":        120,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})

# ── Session state init — persists results across reruns ──────────────────────
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = {}

# ── Helpers — copy-pasted directly from notebook ──────────────────────────────
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

def compute_peak(x): return float(np.max(np.abs(x)))
def compute_rms(x):  return float(np.sqrt(np.mean(np.array(x) ** 2)))
def compute_h13(x):
    arr = np.abs(x)
    peaks, _ = find_peaks(arr, distance=3)
    n_third  = max(1, len(arr) // 3)
    if len(peaks) < 3:
        s = np.sort(arr)[::-1]
        return float(np.mean(s[:n_third]))
    s = np.sort(arr[peaks])[::-1]
    return float(np.mean(s[:max(1, len(peaks) // 3)]))

def stat_fn(x, metric):
    if metric == "Peak": return compute_peak(x)
    if metric == "RMS":  return compute_rms(x)
    if metric == "H1/3": return compute_h13(x)

# ── TimesFM model — cached so it loads once per session ──────────────────────
@st.cache_resource(show_spinner="Loading TimesFM model (~800 MB, once only)...")
def load_model():
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    m = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    return m

def tfm_predict(model, signal, cut, ctx_len, horizon):
    from timesfm import ForecastConfig
    model.compile(ForecastConfig(max_context=ctx_len, max_horizon=horizon))
    context    = signal[cut - ctx_len : cut].astype(np.float32)
    actual     = signal[cut : cut + horizon].astype(np.float32)
    local_mean = context.mean()
    forecast, _ = model.forecast(horizon=horizon, inputs=[context - local_mean])
    pred = (forecast[0] + local_mean).astype(np.float32)
    return actual, pred

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚢 Ship Motion Forecasting")
    st.caption("TimesFM Zero-Shot · IMU Pipeline")
    st.divider()

    uploaded = st.file_uploader(
        "Upload IMU CSV",
        type=["csv"],
        help="Required columns: timestamp (Unix s), roll_deg, pitch_deg",
    )

    st.subheader("Forecast settings")
    ctx_sec = st.selectbox(
        "Context window (history used)",
        [60, 120, 180, 240, 360],
        index=1,
        format_func=lambda x: f"{x}s - {x//60} min of history",
    )
    horizon_sec = st.selectbox(
        "Forecast horizon (how far ahead)",
        [3, 10, 20, 30, 60, 120],
        index=3,
        format_func=lambda x: f"{x}s ahead",
    )
    n_windows = st.slider("Evaluation windows", 3, 10, 5)
    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Columns needed: `timestamp` (Unix s) or `time_sec`, "
        "`roll_deg`, `pitch_deg`. "
        "Optional: `yaw_deg`, `gz`, `gx`, `gy`, `ax`, `ay`, `az`."
    )

# ── Main area ─────────────────────────────────────────────────────────────────
# If results already computed in this session, skip welcome and pipeline
if not st.session_state.results_ready:
    if uploaded is None:
        st.markdown("## Welcome")
        st.info("Upload an IMU CSV file in the sidebar and click **Run pipeline** to begin.")
        st.markdown("""
**What this app does:**

1. Checks timestamp uniformity and resamples to exact 20 Hz
2. Removes sensor outliers and applies Butterworth filtering
3. Decomposes roll into slow sway + fast wave components
4. Runs TimesFM zero-shot forecasting at your chosen horizon
5. Reports MAE, RMSE, and statistical accuracy (Peak / RMS / H1/3)
6. Generates a downloadable summary report

**Data format:**
```
timestamp,roll_deg,pitch_deg,yaw_deg,gz,...
1764676000.00,-5.845,-3.808,176.87,...
1764676000.05,-5.539,-4.089,176.74,...
```
        """)
        st.stop()

    if not run_btn:
        st.info("File uploaded. Adjust settings in the sidebar then click **Run pipeline**.")
        st.stop()


# ── If results already in session_state, restore and jump to tabs ─────────────
if st.session_state.results_ready:
    d = st.session_state.pipeline_data
    df_raw          = d["df_raw"];   df_rs  = d["df_rs"];  df = d["df"]
    df_clean        = d["df_clean"]
    FS              = d["FS"];       FS_RS  = d["FS_RS"];  FS_ORIG = d["FS_ORIG"]
    DT_MEAN         = d["DT_MEAN"];  DT_STD = d["DT_STD"]; CV = d["CV"]
    DUR_MIN         = d["DUR_MIN"];  ts     = d["ts"];      dt = d["dt"]
    FS_RAW          = d.get("FS_RAW", 1.0/DT_MEAN)
    outlier_log     = d["outlier_log"]
    decomp_residual = d["decomp_residual"]
    decomp_corr     = d["decomp_corr"]
    ctx_sum         = d["ctx_sum"]
    BEST_CTX_SEC    = d["BEST_CTX_SEC"];  BEST_CTX = d["BEST_CTX"]
    hor_df          = d["hor_df"];        hor_sum  = d["hor_sum"]
    HORIZONS_SEC    = d["HORIZONS_SEC"];  store_hor = d["store_hor"]
    stat_df         = d["stat_df"]
    roll_raw        = d["roll_raw"];      roll_slow = d["roll_slow"]
    roll_fast       = d["roll_fast"];     pitch_raw = d["pitch_raw"]
    CTX_120         = d["CTX_120"];       CTX_360   = d["CTX_360"]
    CONTEXT_LENGTHS_SEC = d["CONTEXT_LENGTHS_SEC"]
    n_windows       = d["n_windows"]
    N_WINDOWS       = n_windows
    STAT_METRICS    = d.get("STAT_METRICS", ["Peak", "RMS", "H1/3"])
    model           = load_model()
else:
    pass  # fall through to pipeline below

if not st.session_state.results_ready:
# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

    progress = st.progress(0, text="Loading data...")

    # ── Load CSV ──────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # Detect timestamp column
    ts_col = next((c for c in ["timestamp", "time_sec", "time"] if c in df_raw.columns), None)
    if ts_col is None:
        st.error("No timestamp column found. Need `timestamp`, `time_sec`, or `time`.")
        st.stop()
    for col in ["roll_deg", "pitch_deg"]:
        if col not in df_raw.columns:
            st.error(f"Missing required column: `{col}`")
            st.stop()

    df_raw["time_sec"] = df_raw[ts_col] - df_raw[ts_col].iloc[0]
    FS_RAW   = len(df_raw) / df_raw["time_sec"].iloc[-1]
    DUR_MIN  = df_raw["time_sec"].iloc[-1] / 60

    progress.progress(5, text="Checking timestamps...")

    # ── Step 1: Timestamp quality ─────────────────────────────────────────────────
    ts  = df_raw[ts_col].values.astype(np.float64)
    dt  = np.diff(ts)
    n   = len(dt)
    DT_MEAN = dt.mean()
    DT_STD  = dt.std()
    CV      = DT_STD / DT_MEAN
    FS_ORIG = 1.0 / DT_MEAN   # estimated original sampling rate

    progress.progress(10, text="Resampling to 20 Hz...")

    # ── Step 2: Resample to 20 Hz ─────────────────────────────────────────────────
    TARGET_HZ = 20.0
    IMU_CHANNELS = ["roll_deg", "pitch_deg", "yaw_deg", "ax", "ay", "az", "gx", "gy", "gz"]

    t0, t1   = ts[0], ts[-1]
    t_uniform = np.arange(t0, t1, 1.0 / TARGET_HZ)

    df_rs = pd.DataFrame({"timestamp": t_uniform, "time_sec": t_uniform - t0})
    for col in IMU_CHANNELS:
        if col in df_raw.columns:
            df_rs[col] = np.interp(t_uniform, ts, df_raw[col].values.astype(float))

    FS    = TARGET_HZ
    FS_RS = TARGET_HZ

    progress.progress(20, text="Removing outliers and filtering...")

    # ── Step 3: Outlier removal ───────────────────────────────────────────────────
    df = df_rs.copy()
    if "yaw_deg" in df.columns:
        df["yaw_unwrap"] = np.unwrap(df["yaw_deg"].values, period=360)

    outlier_log = {}
    for col in ["roll_deg", "pitch_deg", "gz"]:
        if col in df.columns:
            df[col], n_out = remove_outliers(df[col])
            outlier_log[col] = n_out

    # ── Step 4: Butterworth filter + roll decomposition ──────────────────────────
    df["pitch_filt"] = butter_lp(df["pitch_deg"].values, cutoff=2.0, fs=FS)
    df["roll_slow"]  = butter_lp(df["roll_deg"].values,  cutoff=0.05, fs=FS)
    df["roll_fast"]  = butter_bp(df["roll_deg"].values,  low=0.05, high=2.0, fs=FS)
    df["roll_filt"]  = butter_lp(df["roll_deg"].values,  cutoff=2.0, fs=FS)

    decomp_residual = float(np.abs(df["roll_deg"].values - (df["roll_slow"] + df["roll_fast"])).mean())
    decomp_corr     = float(np.corrcoef(df["roll_filt"].values, df["roll_slow"] + df["roll_fast"])[0, 1])

    CLEAN_COLS = ["time_sec","roll_filt","roll_slow","roll_fast","pitch_filt"] + \
                 [c for c in ["gz","gx","gy","ax","ay","az"] if c in df.columns]
    df_clean = df[CLEAN_COLS].copy()
    df_clean.columns = ["time_sec","roll_deg","roll_slow","roll_fast","pitch_deg"] + \
                       [c for c in ["gz","gx","gy","ax","ay","az"] if c in df.columns]

    progress.progress(30, text="Loading TimesFM model...")

    # ── Step 5: Load TimesFM ──────────────────────────────────────────────────────
    from timesfm import ForecastConfig
    model  = load_model()
    CTX_120 = int(120 * FS)
    CTX_360 = int(360 * FS)

    roll_raw  = df_clean["roll_deg"].values.astype(np.float32)
    roll_slow = df_clean["roll_slow"].values.astype(np.float32)
    roll_fast = df_clean["roll_fast"].values.astype(np.float32)
    pitch_raw = df_clean["pitch_deg"].values.astype(np.float32)

    MAX_CTX = CTX_360
    MAX_HOR = int(max(120, horizon_sec) * FS)

    cut_points = np.linspace(
        MAX_CTX + MAX_HOR,
        len(df_clean) - MAX_HOR,
        n_windows, dtype=int
    )

    progress.progress(35, text="Running context window study...")

    # ── Step 6: Context window study ──────────────────────────────────────────────
    CONTEXT_LENGTHS_SEC = [60, 120, 180, 240, 360]
    HORIZON_5A          = int(120 * FS)
    ctx_results         = []

    for ctx_s in CONTEXT_LENGTHS_SEC:
        ctx_len = int(ctx_s * FS)
        for i, cut in enumerate(cut_points):
            a_p,  p_p  = tfm_predict(model, pitch_raw, cut, ctx_len, HORIZON_5A)
            a_rf, p_rf = tfm_predict(model, roll_fast,  cut, ctx_len, HORIZON_5A)
            a_rs, p_rs = tfm_predict(model, roll_slow,  cut, CTX_360, HORIZON_5A)
            a_roll = roll_raw[cut : cut + HORIZON_5A]
            p_roll = p_rf + p_rs
            ctx_results.append({
                "ctx_sec": ctx_s, "window": i + 1,
                "mae_pitch": mean_absolute_error(a_p, p_p),
                "mae_roll":  mean_absolute_error(a_roll, p_roll),
            })

    ctx_df  = pd.DataFrame(ctx_results)
    ctx_sum = ctx_df.groupby("ctx_sec").agg(
        pitch_mean=("mae_pitch","mean"), pitch_std=("mae_pitch","std"),
        roll_mean =("mae_roll", "mean"), roll_std =("mae_roll", "std"),
    ).reset_index()

    BEST_CTX_SEC = int(ctx_sum.iloc[
        (ctx_sum["pitch_mean"] + ctx_sum["roll_mean"]).argmin()
    ]["ctx_sec"])
    BEST_CTX = int(BEST_CTX_SEC * FS)

    progress.progress(55, text=f"Running horizon study (ctx={BEST_CTX_SEC}s)...")

    # ── Step 7: Horizon study ─────────────────────────────────────────────────────
    HORIZONS_SEC = [3, 10, 20, 30, 60, 120]
    H_USER       = horizon_sec
    hor_results  = []
    store_hor    = {h: [] for h in HORIZONS_SEC}

    for h_sec in HORIZONS_SEC:
        h_samps = int(h_sec * FS)
        for i, cut in enumerate(cut_points):
            a_p,  p_p  = tfm_predict(model, pitch_raw, cut, BEST_CTX, h_samps)
            ctx_p      = pitch_raw[cut - h_samps : cut].astype(np.float32)
            a_rf, p_rf = tfm_predict(model, roll_fast, cut, BEST_CTX, h_samps)
            a_rs, p_rs = tfm_predict(model, roll_slow, cut, CTX_360,  h_samps)
            a_roll     = roll_raw[cut : cut + h_samps]
            p_roll     = p_rf + p_rs
            ctx_roll   = roll_raw[cut - h_samps : cut].astype(np.float32)
            hor_results.append({
                "horizon_sec": h_sec, "window": i + 1,
                "mae_pitch":  mean_absolute_error(a_p, p_p),
                "mae_roll":   mean_absolute_error(a_roll, p_roll),
                "rmse_pitch": float(np.sqrt(mean_squared_error(a_p, p_p))),
                "rmse_roll":  float(np.sqrt(mean_squared_error(a_roll, p_roll))),
            })
            store_hor[h_sec].append(dict(
                a_pitch=a_p,  p_pitch=p_p,  ctx_pitch=ctx_p,
                a_roll=a_roll, p_roll=p_roll, ctx_roll=ctx_roll,
            ))

    hor_df  = pd.DataFrame(hor_results)
    hor_sum = hor_df.groupby("horizon_sec").agg(
        pitch_mean=("mae_pitch","mean"), pitch_std=("mae_pitch","std"),
        roll_mean =("mae_roll", "mean"), roll_std =("mae_roll", "std"),
        pitch_rmse=("rmse_pitch","mean"), roll_rmse=("rmse_roll","mean"),
    ).reset_index()

    progress.progress(85, text="Computing statistical metrics...")

    # ── Step 8: Statistical metrics ───────────────────────────────────────────────
    STAT_METRICS = ["Peak", "RMS", "H1/3"]
    all_stat_rows = []

    for sig_label, key_ctx, key_a, key_p in [
        ("Pitch","ctx_pitch","a_pitch","p_pitch"),
        ("Roll", "ctx_roll", "a_roll", "p_roll"),
    ]:
        for h_sec in HORIZONS_SEC:
            wins = store_hor[h_sec]
            for metric in STAT_METRICS:
                inp_v  = [stat_fn(w[key_ctx], metric) for w in wins]
                tru_v  = [stat_fn(w[key_a],   metric) for w in wins]
                prd_v  = [stat_fn(w[key_p],   metric) for w in wins]
                abspct = [abs(p-t)/(t+1e-9)*100 for p,t in zip(prd_v, tru_v)]
                all_stat_rows.append({
                    "Signal": sig_label, "Horizon_sec": h_sec, "Metric": metric,
                    "Input_mean":  round(np.mean(inp_v), 4),
                    "Pred_mean":   round(np.mean(prd_v), 4),
                    "True_mean":   round(np.mean(tru_v), 4),
                    "AbsErr_pct":  round(np.mean(abspct), 2),
                    "Std_pct":     round(np.std(abspct),  2),
                })

    stat_df = pd.DataFrame(all_stat_rows)

    progress.progress(100, text="Done.")
    progress.empty()

    # ── Store all results in session state ───────────────────────────────────────
    st.session_state.results_ready = True
    N_WINDOWS = n_windows
    st.session_state.pipeline_data = {
        "df_raw": df_raw, "df_rs": df_rs, "df": df, "df_clean": df_clean,
        "FS": FS, "FS_RS": FS_RS, "FS_ORIG": FS_ORIG, "FS_RAW": FS_RAW,
        "DT_MEAN": DT_MEAN, "DT_STD": DT_STD, "CV": CV,
        "DUR_MIN": DUR_MIN, "ts": ts, "dt": dt,
        "outlier_log": outlier_log,
        "decomp_residual": decomp_residual, "decomp_corr": decomp_corr,
        "ctx_sum": ctx_sum, "BEST_CTX_SEC": BEST_CTX_SEC, "BEST_CTX": BEST_CTX,
        "hor_df": hor_df, "hor_sum": hor_sum, "HORIZONS_SEC": HORIZONS_SEC,
        "store_hor": store_hor, "stat_df": stat_df,
        "roll_raw": roll_raw, "roll_slow": roll_slow,
        "roll_fast": roll_fast, "pitch_raw": pitch_raw,
        "CTX_120": CTX_120, "CTX_360": CTX_360,
        "CONTEXT_LENGTHS_SEC": CONTEXT_LENGTHS_SEC,
        "n_windows": n_windows, "filename": uploaded.name,
        "STAT_METRICS": ["Peak", "RMS", "H1/3"],
    }

    # ══════════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY — load from session state if already computed
    # ══════════════════════════════════════════════════════════════════════════════

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Prediction",
    "📊 Data Quality",
    "🔧 Preprocessing",
    "📈 Forecast (Evaluation)",
    "📐 Statistics (Evaluation)",
    "📋 Report",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    st.subheader("Live Prediction")
    st.caption(
        "Uses the most recent data in your file as context and predicts forward into the future. "
        "No ground truth is available — this is the actual forward forecast."
    )

    lp_col1, lp_col2 = st.columns(2)
    with lp_col1:
        st.markdown("**Context window — how much history to use**")
        lp_ctx_sec = st.select_slider(
            "Context length (s)",
            options=[60, 120, 180, 240, 360],
            value=120,
            format_func=lambda x: f"{x}s  ({x//60} min of past data)",
            key="lp_ctx",
        )
    with lp_col2:
        st.markdown("**Prediction horizons — select how far ahead to predict**")
        lp_horizons = st.multiselect(
            "Horizons (can select multiple)",
            options=[3, 5, 10, 15, 20, 30, 60, 90, 120],
            default=[3, 10, 20, 30],
            format_func=lambda x: f"{x}s",
            key="lp_horizons",
        )

    if not lp_horizons:
        st.warning("Select at least one horizon above.")
    else:
        lp_horizons_sorted = sorted(lp_horizons)
        lp_ctx     = int(lp_ctx_sec * FS)
        lp_cut     = len(df_clean)

        if lp_cut < lp_ctx + 10:
            st.error(f"Not enough data for {lp_ctx_sec}s context. Upload a longer recording.")
        else:
            ctx_pitch_live = pitch_raw[lp_cut - lp_ctx : lp_cut]
            ctx_roll_live  = roll_raw [lp_cut - lp_ctx : lp_cut]
            t_ctx = np.arange(len(ctx_pitch_live)) / FS

            lp_results = {}
            lp_bar = st.progress(0, text="Running live predictions...")
            for ki, h_sec in enumerate(lp_horizons_sorted):
                h_samps = int(h_sec * FS)
                _, p_p  = tfm_predict(model, pitch_raw, lp_cut, lp_ctx, h_samps)
                _, p_rf = tfm_predict(model, roll_fast, lp_cut, lp_ctx, h_samps)
                ctx_360 = min(CTX_360, lp_cut)
                _, p_rs = tfm_predict(model, roll_slow, lp_cut, ctx_360, h_samps)
                p_roll  = p_rf + p_rs
                t_pred  = lp_ctx_sec + np.arange(h_samps) / FS
                lp_results[h_sec] = {"p_pitch": p_p, "p_roll": p_roll,
                                     "t_pred": t_pred, "h_samps": h_samps}
                lp_bar.progress((ki+1)/len(lp_horizons_sorted),
                                text=f"Predicted {h_sec}s horizon...")
            lp_bar.empty()

            HORIZON_COLORS = ["#185FA5","#D85A30","#3B6D11","#534AB7",
                               "#BA7517","#0D7A7A","#993C1D","#444444","#5C1F8A"]

            # ── A: Time series ─────────────────────────────────────────────
            st.divider()
            st.markdown("### A — Time Series Prediction")
            st.caption(
                "Gray = recorded context (actual past). "
                "Coloured dashed = predicted future (each colour = one horizon). "
                "Black dashed line = NOW."
            )

            for sig_label, ctx_sig, pred_key in [
                ("Pitch", ctx_pitch_live, "p_pitch"),
                ("Roll",  ctx_roll_live,  "p_roll"),
            ]:
                fig, ax = plt.subplots(figsize=(15, 4))
                ax.plot(t_ctx, ctx_sig, color="gray", lw=1.0,
                        label=f"Recorded (last {lp_ctx_sec}s)", zorder=3)
                ax.axvline(lp_ctx_sec, color="black", lw=1.5, ls="--", zorder=4,
                           label="NOW")
                ax.fill_between(t_ctx, ctx_sig.min(), ctx_sig.max(),
                                alpha=0.04, color="gray")
                max_h = max(lp_horizons_sorted)
                ax.axvspan(lp_ctx_sec, lp_ctx_sec + max_h, alpha=0.04, color="#185FA5")

                for i, h_sec in enumerate(lp_horizons_sorted):
                    r = lp_results[h_sec]
                    ax.plot(r["t_pred"], r[pred_key],
                            color=HORIZON_COLORS[i % len(HORIZON_COLORS)],
                            lw=1.8, ls="--", label=f"Predict next {h_sec}s", zorder=5)

                ax.set_xlabel("Time (s)  [0 = start of context window]")
                ax.set_ylabel(f"{sig_label} (deg)")
                ax.set_title(
                    f"{sig_label} - Actual context + Forward prediction - Context = last {lp_ctx_sec}s"
                )
                ax.legend(fontsize=8, loc="upper left",
                          ncol=min(4, len(lp_horizons_sorted)+1))
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── B: Statistical table ───────────────────────────────────────
            st.divider()
            st.markdown("### B — Statistical Prediction (Peak / RMS / H1/3)")
            st.caption(
                "Input = stats of the last [horizon] seconds of recorded data — the sea state that just happened. "
                "Predicted = stats of the next [horizon] seconds — what the model forecasts. "
                "Trend = whether sea is expected to get calmer, rougher, or stay stable."
            )

            stat_rows_live = []
            for sig_label, ctx_sig, pred_key in [
                ("Pitch", ctx_pitch_live, "p_pitch"),
                ("Roll",  ctx_roll_live,  "p_roll"),
            ]:
                for h_sec in lp_horizons_sorted:
                    r       = lp_results[h_sec]
                    pred_w  = r[pred_key]
                    h_samps = r["h_samps"]
                    inp_w   = ctx_sig[-h_samps:] if h_samps <= len(ctx_sig) else ctx_sig

                    for metric in ["Peak", "RMS", "H1/3"]:
                        inp_val  = stat_fn(inp_w,  metric)
                        pred_val = stat_fn(pred_w, metric)
                        hor_match = hor_sum[hor_sum["horizon_sec"] == h_sec]
                        if len(hor_match) > 0:
                            ec = "pitch_mean" if sig_label=="Pitch" else "roll_mean"
                            emae = float(hor_match[ec].values[0])
                            acc  = f"±{emae:.2f}°"
                        else:
                            nearest = hor_sum.iloc[
                                (hor_sum["horizon_sec"]-h_sec).abs().argmin()]
                            ec   = "pitch_mean" if sig_label=="Pitch" else "roll_mean"
                            emae = float(nearest[ec])
                            acc  = f"±{emae:.2f}° (approx)"

                        pct_chg = (pred_val - inp_val) / (inp_val + 1e-9) * 100
                        trend   = ("calming" if pct_chg < -3
                                   else "rougher" if pct_chg > 3 else "stable")

                        stat_rows_live.append({
                            "Signal":   sig_label,
                            "Horizon":  f"{h_sec}s",
                            "Metric":   metric,
                            "Input (past Ns)":     round(inp_val,  3),
                            "Predicted (next Ns)": round(pred_val, 3),
                            "Change (%)":          round(pct_chg,  1),
                            "Trend":               trend,
                            "Model accuracy":      acc,
                        })

            stat_live_df = pd.DataFrame(stat_rows_live)

            for sig in ["Pitch", "Roll"]:
                st.markdown(f"**{sig} motion**")
                sub = stat_live_df[stat_live_df["Signal"]==sig].drop(columns=["Signal"])
                def color_trend(val):
                    if val == "calming": return "color: green"
                    if val == "rougher": return "color: red"
                    return ""
                st.dataframe(
                    sub.style.map(color_trend, subset=["Trend"]),
                    use_container_width=True, hide_index=True
                )

            # ── C: Bar chart ───────────────────────────────────────────────
            st.divider()
            st.markdown("### C — Predicted Statistics by Horizon")
            st.caption("How the predicted Peak, RMS, H1/3 change as horizon increases.")
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            colors_met = {"Peak":"#D85A30","RMS":"#185FA5","H1/3":"#3B6D11"}
            for ax, sig_label in zip(axes, ["Pitch","Roll"]):
                sub = stat_live_df[stat_live_df["Signal"]==sig_label]
                x   = np.arange(len(lp_horizons_sorted))
                w   = 0.25
                for j, metric in enumerate(["Peak","RMS","H1/3"]):
                    m    = sub[sub["Metric"]==metric]
                    vals = []
                    for h in lp_horizons_sorted:
                        row = m[m["Horizon"]==f"{h}s"]
                        vals.append(float(row["Predicted (next Ns)"].values[0])
                                    if len(row) > 0 else 0)
                    ax.bar(x + j*w, vals, w, label=metric,
                           color=colors_met[metric], alpha=0.85)
                ax.set_xticks(x + w)
                ax.set_xticklabels([f"{h}s" for h in lp_horizons_sorted])
                ax.set_xlabel("Forecast Horizon")
                ax.set_ylabel("Predicted amplitude (deg)")
                ax.set_title(f"{sig_label} - Predicted Stats by Horizon")
                ax.legend(fontsize=9)
            plt.suptitle(
                "Live Prediction — Statistical Summary "
                "All values are predictions (no ground truth available)",
                fontsize=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ── D: NATO check ──────────────────────────────────────────────
            st.divider()
            st.markdown("### D — NATO STANAG 4154 Check on Predicted Motion")
            st.caption(
                "Roll RMS limit = 2.5°  ·  Pitch RMS limit = 1.5°  ·  "
                "Based on predicted motion for each horizon."
            )
            nato_rows = []
            for h_sec in lp_horizons_sorted:
                r          = lp_results[h_sec]
                pred_r_rms = compute_rms(r["p_roll"])
                pred_p_rms = compute_rms(r["p_pitch"])
                roll_ok    = pred_r_rms < 2.5
                pitch_ok   = pred_p_rms < 1.5
                nato_rows.append({
                    "Horizon":           f"{h_sec}s",
                    "Pred Roll RMS (°)": round(pred_r_rms, 3),
                    "Roll  (lim 2.5°)":  "SAFE"    if roll_ok  else "EXCEEDS",
                    "Pred Pitch RMS (°)":round(pred_p_rms, 3),
                    "Pitch (lim 1.5°)":  "SAFE"    if pitch_ok else "EXCEEDS",
                    "Helicopter ops":    "GO"       if (roll_ok and pitch_ok) else "HOLD",
                })
            nato_df = pd.DataFrame(nato_rows)
            def colour_nato(val):
                if val in ("SAFE","GO"):     return "background-color:#d1fae5;color:#065f46"
                if val in ("EXCEEDS","HOLD"):return "background-color:#fee2e2;color:#991b1b"
                return ""
            st.dataframe(
                nato_df.style.map(colour_nato,
                    subset=["Roll  (lim 2.5°)","Pitch (lim 1.5°)","Helicopter ops"]),
                use_container_width=True, hide_index=True
            )
            st.info(
                "GO = predicted sea state within NATO limits. "
                "HOLD = model predicts conditions will exceed safety threshold at this horizon."
            )

            # ── E: Download ────────────────────────────────────────────────
            st.divider()
            st.markdown("### Download Predictions")
            dl_rows = []
            for h_sec in lp_horizons_sorted:
                r = lp_results[h_sec]
                for i, (tp, pp, pr) in enumerate(
                    zip(r["t_pred"], r["p_pitch"], r["p_roll"])
                ):
                    dl_rows.append({
                        "horizon_sec":     h_sec,
                        "time_from_now_s": round(tp - lp_ctx_sec, 3),
                        "pred_pitch_deg":  round(float(pp), 4),
                        "pred_roll_deg":   round(float(pr), 4),
                    })
            dl_df = pd.DataFrame(dl_rows)
            st.download_button(
                "Download all predictions (CSV)",
                data=dl_df.to_csv(index=False),
                file_name="live_predictions.csv",
            key="dl_live",
                mime="text/csv",
            )
            st.caption(
                "Columns: horizon_sec = which horizon · "
                "time_from_now_s = seconds after last data point · "
                "pred_pitch_deg / pred_roll_deg = predicted angle."
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Quality
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Data Quality")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw samples",      f"{len(df_raw):,}")
    c2.metric("Resampled (20 Hz)",f"{len(df_rs):,}")
    c3.metric("Duration",         f"{DUR_MIN:.1f} min")
    c4.metric("Original rate",    f"{FS_RAW:.2f} Hz")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean dt",  f"{DT_MEAN:.5f} s")
    c6.metric("Std dt",   f"{DT_STD:.5f} s")
    c7.metric("CV (jitter)", f"{CV:.4f}", delta="OK" if CV < 0.05 else "RESAMPLED", delta_color="normal" if CV < 0.05 else "inverse")
    c8.metric("Max gap",  f"{dt.max():.4f} s")

    if CV > 0.05:
        st.warning(f"CV = {CV:.4f} - significant jitter detected. Resampling to exact 20 Hz was applied automatically.")
    else:
        st.success(f"CV = {CV:.4f} - timestamps near-uniform.")

    st.divider()
    st.subheader("Signal statistics")
    STAT_COLS = [c for c in ["roll_deg","pitch_deg","yaw_deg","gz"] if c in df_raw.columns]
    stat_table = pd.DataFrame([{
        "Signal": c,
        "Mean":  round(df_raw[c].mean(), 3),
        "Std":   round(df_raw[c].std(),  3),
        "Min":   round(df_raw[c].min(),  3),
        "Max":   round(df_raw[c].max(),  3),
        "Skew":  round(sp_skew(df_raw[c]), 3),
        "Kurt":  round(sp_kurtosis(df_raw[c]), 3),
    } for c in STAT_COLS])
    st.dataframe(stat_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Timestamp distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(dt, bins=200, color="#185FA5", alpha=0.8, edgecolor="none")
    axes[0].axvline(0.05, color="tomato", lw=1.5, ls="--", label="Target 0.05 s")
    axes[0].set_xlabel("Inter-sample interval (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of time intervals")
    axes[0].legend()
    t_mid = (ts[:-1] + ts[1:]) / 2 - ts[0]
    axes[1].plot(t_mid, dt, color="#185FA5", lw=0.4, alpha=0.6)
    axes[1].axhline(0.05, color="tomato", lw=1.2, ls="--", label="Target 0.05 s")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Inter-sample interval (s)")
    axes[1].set_title("Time intervals over recording")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Resampling validation")
    val_rows = []
    for col in ["roll_deg","pitch_deg"]:
        if col not in df_raw.columns: continue
        o, r = df_raw[col].values, df_rs[col].values
        for stat_name, fo, fr in [("Mean",o.mean(),r.mean()),("Std",o.std(),r.std()),("Min",o.min(),r.min()),("Max",o.max(),r.max())]:
            delta = abs(fo - fr)
            val_rows.append({"Column":col,"Stat":stat_name,"Original":round(fo,4),"Resampled":round(fr,4),"Delta":round(delta,4)})
    st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dominant wave frequencies (PSD)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    FS_ORIG = 1.0 / DT_MEAN
    for row, col in enumerate(["roll_deg","pitch_deg"]):
        if col not in df_raw.columns: continue
        s_orig = df_raw[col].values - df_raw[col].mean()
        f_o, p_o = welch(s_orig, fs=FS_ORIG, nperseg=4096, noverlap=2048)
        s_rs = df_rs[col].values - df_rs[col].mean()
        f_r, p_r = welch(s_rs, fs=FS_RS, nperseg=4096, noverlap=2048)
        for ax, f, p, lbl, color in [
            (axes[row,0], f_o, p_o, "Original", "#185FA5"),
            (axes[row,1], f_r, p_r, "Resampled (20 Hz)", "#D85A30"),
        ]:
            ax.semilogy(f, p, color=color, lw=0.9)
            ax.set_xlim([0, 2.0])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (deg²/Hz)")
            ax.set_title(f"{col} - {lbl}")
            mask = (f > 0.005) & (f < 2.0)
            for pi in np.argsort(p[mask])[::-1][:3]:
                fi = f[mask][pi]
                ax.axvline(fi, color="tomato", lw=1.2, ls="--", alpha=0.8)
                ax.text(fi+0.01, p[mask][pi], f"{fi:.3f}Hz\n({1/fi:.1f}s)", fontsize=7, color="tomato")
    plt.suptitle("PSD — Before vs After Resampling", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Preprocessing")

    c1, c2, c3 = st.columns(3)
    for col, (widget, n) in zip(["roll_deg","pitch_deg","gz"], [(c1,"roll_deg"),(c2,"pitch_deg"),(c3,"gz")]):
        if col in outlier_log:
            widget.metric(f"{col} outliers removed", outlier_log[col],
                          delta=f"{100*outlier_log[col]/len(df_rs):.2f}% of data")

    st.metric("Roll decomposition residual", f"{decomp_residual:.5f} deg")
    st.metric("Decomposition correlation",   f"{decomp_corr:.6f}")

    st.divider()
    st.subheader("Raw vs cleaned — first 120 s")
    t_rs   = df["time_sec"].values
    t_raw2 = df_raw["timestamp"].values - df_raw["timestamp"].values[0]
    view_rs  = t_rs   < 120
    view_raw = t_raw2 < 120

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    for row, (col, col_filt, label) in enumerate([
        ("roll_deg",  "roll_filt",  "Roll (deg)"),
        ("pitch_deg", "pitch_filt", "Pitch (deg)"),
        ("gz",        "gz",         "gz (rad/s)"),
    ]):
        if col in df_raw.columns:
            raw_v       = df_raw[col].values[view_raw]
            t_raw_view  = t_raw2[view_raw]
        else:
            raw_v       = df[col].values[view_rs] if col in df.columns else np.array([0])
            t_raw_view  = t_rs[view_rs]
        clean_v = df[col_filt].values[view_rs] if col_filt in df.columns else df[col].values[view_rs] if col in df.columns else np.array([0])
        t_c     = t_rs[view_rs]
        axes[row,0].plot(t_raw_view, raw_v,   color="lightsteelblue", lw=0.7, label="Raw")
        axes[row,0].plot(t_c,        clean_v, color="#185FA5",        lw=1.3, label="Cleaned")
        axes[row,0].set_ylabel(label)
        axes[row,0].legend(fontsize=8)
        if len(raw_v) > 4 and len(clean_v) > 4:
            fr, pr = welch(raw_v   - raw_v.mean(),   fs=FS, nperseg=min(512,len(raw_v)//2))
            fc, pc = welch(clean_v - clean_v.mean(), fs=FS, nperseg=min(512,len(clean_v)//2))
            axes[row,1].semilogy(fr, pr, color="lightsteelblue", lw=0.8, label="Raw")
            axes[row,1].semilogy(fc, pc, color="#185FA5",        lw=1.3, label="Cleaned")
            axes[row,1].axvline(2.0, color="tomato", lw=1.2, ls="--", label="2 Hz cutoff")
            axes[row,1].set_xlim([0, FS/2])
            axes[row,1].legend(fontsize=8)
            axes[row,1].set_ylabel("PSD")
    axes[0,0].set_title("Time Domain — First 120 s")
    axes[0,1].set_title("PSD — Before vs After Filter")
    axes[-1,0].set_xlabel("Time (s)")
    axes[-1,1].set_xlabel("Frequency (Hz)")
    plt.suptitle("Preprocessing: Before vs After", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Roll decomposition — first 120 s")
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    v = view_rs
    axes[0].plot(t_rs[v], df["roll_deg"].values[v],  color="gray",    lw=0.8, label="Raw roll")
    axes[1].plot(t_rs[v], df["roll_slow"].values[v], color="#BA7517", lw=1.5, label="Slow (<0.05 Hz, ~107s)")
    axes[2].plot(t_rs[v], df["roll_fast"].values[v], color="#185FA5", lw=1.0, label="Fast (0.05–2 Hz, ~2.1s)")
    axes[3].plot(t_rs[v], df["roll_filt"].values[v], color="#534AB7", lw=1.0, label="Filtered total")
    for ax in axes:
        ax.legend(fontsize=9); ax.set_ylabel("deg")
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Roll Decomposition — Slow Sway + Fast Wave", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Forecast Results")

    st.info(f"Best context window: **{BEST_CTX_SEC}s** (selected automatically)")

    st.subheader("MAE by horizon")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col_m, col_s, label, color) in zip(axes, [
        ("pitch_mean","pitch_std","Pitch MAE (deg)","#3B6D11"),
        ("roll_mean", "roll_std", "Roll MAE (deg)", "#185FA5"),
    ]):
        ax.errorbar(hor_sum["horizon_sec"], hor_sum[col_m],
                    yerr=hor_sum[col_s], marker="o", color=color,
                    lw=1.8, capsize=4, capthick=1.5)
        ax.set_xlabel("Forecast Horizon (s)")
        ax.set_ylabel(label)
        ax.set_title(label.replace(" (deg)","") + " — MAE vs Horizon")
        ax.set_xticks(HORIZONS_SEC)
    plt.suptitle(f"TimesFM Zero-Shot - MAE vs Forecast Horizon\nContext={BEST_CTX_SEC}s | {n_windows} evaluation windows | Error bars = ±1 std", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("MAE vs context window")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (col_m, col_s, label, color) in zip(axes, [
        ("pitch_mean","pitch_std","Pitch MAE (deg)","#3B6D11"),
        ("roll_mean", "roll_std", "Roll MAE (deg)", "#185FA5"),
    ]):
        ax.errorbar(ctx_sum["ctx_sec"], ctx_sum[col_m],
                    yerr=ctx_sum[col_s], marker="s", color=color, lw=1.8, capsize=4)
        ax.axvline(BEST_CTX_SEC, color="tomato", ls="--", lw=1.2, label=f"Best={BEST_CTX_SEC}s")
        ax.set_xlabel("Context Window (s)")
        ax.set_ylabel(label)
        ax.set_title(label.replace(" (deg)","") + " — MAE vs Context")
        ax.set_xticks(CONTEXT_LENGTHS_SEC)
        ax.legend()
    plt.suptitle("Context Window Study — How Much History Helps?", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader(f"Actual vs Predicted - {horizon_sec}s horizon")
    COLORS = {"actual":"#185FA5","pred":"#D85A30"}
    if horizon_sec in store_hor:
        wins_h = store_hor[horizon_sec]
        for sig_label, key_a, key_p, col_mae in [
            ("Pitch","a_pitch","p_pitch","mae_pitch"),
            ("Roll", "a_roll", "p_roll", "mae_roll"),
        ]:
            sub   = hor_df[hor_df["horizon_sec"] == horizon_sec]
            mid_w = int((sub[col_mae] - sub[col_mae].median()).abs().argmin())
            if mid_w >= len(wins_h): mid_w = 0
            s     = wins_h[mid_w]
            t_h   = np.arange(len(s[key_a])) / FS
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(t_h, s[key_a], color=COLORS["actual"], lw=1.2, label="Actual")
            ax.plot(t_h, s[key_p], color=COLORS["pred"],   lw=1.2, ls="--", label="Predicted")
            mae_val = mean_absolute_error(s[key_a], s[key_p])
            ax.set_title(f"{sig_label} - horizon {horizon_sec}s | MAE = {mae_val:.3f}°")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("deg")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.subheader("Final results table")
    st.caption(
        "pitch_mean = average pitch MAE across windows · "
        "pitch_std = variation in that MAE across windows · "
        "pitch_rmse = root mean square error (larger errors penalised more)"
    )
    hor_display = hor_sum.rename(columns={
        "horizon_sec" : "Horizon (s)",
        "pitch_mean"  : "Pitch MAE avg (°)",
        "pitch_std"   : "Pitch MAE std (°)",
        "roll_mean"   : "Roll MAE avg (°)",
        "roll_std"    : "Roll MAE std (°)",
        "pitch_rmse"  : "Pitch RMSE (°)",
        "roll_rmse"   : "Roll RMSE (°)",
    }).round(4)
    st.dataframe(hor_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Statistics (Peak / RMS / H1/3)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Statistical Prediction Accuracy — Peak, RMS, H1/3")
    st.caption("Abs% Error = mean of |Predicted − True| / True × 100, per window then averaged")

    for sig in ["Pitch","Roll"]:
        st.markdown(f"**{sig} motion**")
        sub = stat_df[stat_df["Signal"] == sig].copy()
        sub = sub.drop(columns=["Signal"])
        sub["AbsErr_pct"] = sub["AbsErr_pct"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(sub, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"3-second horizon summary")
    three_s = stat_df[stat_df["Horizon_sec"] == 3][["Signal","Metric","Input_mean","Pred_mean","True_mean","AbsErr_pct"]].copy()
    three_s.columns = ["Signal","Metric","Input","Predicted","True","Abs% Error"]
    three_s["Abs% Error"] = three_s["Abs% Error"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(three_s, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Abs% error bar chart by horizon")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors_met = {"Peak":"#D85A30","RMS":"#185FA5","H1/3":"#3B6D11"}
    for ax, sig_label in zip(axes, ["Pitch","Roll"]):
        sub = stat_df[stat_df["Signal"] == sig_label]
        x   = np.arange(len(HORIZONS_SEC))
        w   = 0.25
        for j, metric in enumerate(STAT_METRICS):
            m = sub[sub["Metric"] == metric].set_index("Horizon_sec")
            if len(m) == 0: continue
            vals = m.loc[HORIZONS_SEC,"AbsErr_pct"].values if all(h in m.index for h in HORIZONS_SEC) else np.zeros(len(HORIZONS_SEC))
            errs = m.loc[HORIZONS_SEC,"Std_pct"].values    if all(h in m.index for h in HORIZONS_SEC) else np.zeros(len(HORIZONS_SEC))
            ax.bar(x + j*w, vals, w, label=metric, color=colors_met[metric],
                   alpha=0.85, yerr=errs, capsize=3, error_kw={"lw":1.0})
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"{h}s" for h in HORIZONS_SEC])
        ax.set_xlabel("Forecast Horizon (s)")
        ax.set_ylabel("Abs% Error")
        ax.set_title(f"{sig_label} - Prediction Error by Horizon")
        ax.legend(fontsize=8)
    plt.suptitle("Statistical Accuracy — Peak, RMS, H1/3\nError bars = ±1 std across evaluation windows", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("NATO STANAG 4154 — Helicopter operation check")
    st.caption(
        "This RMS is the roughness of the actual sea — computed from the real IMU signal, "
        "not from prediction errors. It answers: is the sea calm enough for helicopter landing today?"
    )
    roll_rms  = float(np.sqrt(np.mean(df_clean["roll_deg"].values ** 2)))
    pitch_rms = float(np.sqrt(np.mean(df_clean["pitch_deg"].values ** 2)))
    cc1, cc2 = st.columns(2)
    with cc1:
        status = "SAFE" if roll_rms < 2.5 else "EXCEEDS LIMIT"
        color  = "green" if roll_rms < 2.5 else "red"
        st.metric("Roll RMS", f"{roll_rms:.3f}°", delta=f"Limit 2.5° - {status}")
        st.progress(min(1.0, roll_rms / 2.5))
    with cc2:
        status2 = "SAFE" if pitch_rms < 1.5 else "EXCEEDS LIMIT"
        st.metric("Pitch RMS", f"{pitch_rms:.3f}°", delta=f"Limit 1.5° - {status2}")
        st.progress(min(1.0, pitch_rms / 1.5))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Report + Downloads
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Pipeline Summary Report")

    dur_s = df_raw["time_sec"].iloc[-1]

    report_text = f"""
SHIP MOTION FORECASTING — PIPELINE SUMMARY REPORT
Method: TimesFM Zero-Shot Foundation Model
{"="*65}

DATASET
-------
File               : {uploaded.name}
Raw samples        : {len(df_raw):,}
Recording duration : {dur_s:.0f} s  ({DUR_MIN:.1f} min)
Channels available : {", ".join([c for c in ["roll_deg","pitch_deg","yaw_deg","gz"] if c in df_raw.columns])}

STEP 1 — TIMESTAMP QUALITY
--------------------------
Original sampling rate : {FS_RAW:.3f} Hz  (target: 20.0 Hz)
Timing jitter (CV)     : {CV:.4f}  (above 0.05 = resampling required)
Verdict                : {"Resampling was required" if CV > 0.05 else "Near-uniform — resampling applied as precaution"}

STEP 2 — RESAMPLING
-------------------
Resampled to        : 20.0 Hz (0.05 s uniform intervals)
Resampled samples   : {len(df_rs):,}
Duration preserved  : {df_rs["time_sec"].iloc[-1]:.1f} s

STEP 3 — PREPROCESSING
-----------------------
Outliers removed    : {", ".join([f"{v} {k}" for k,v in outlier_log.items()])}
Filter              : Butterworth 4th order LP at 2 Hz
Roll decomposition  : roll_slow (<0.05 Hz) + roll_fast (0.05–2 Hz)
Decomposition MAE   : {decomp_residual:.5f} deg
Decomposition corr  : {decomp_corr:.6f}

STEP 4 — TIMESFM FORECASTING
-----------------------------
Model               : TimesFM 2.5 (200M parameters, zero-shot)
Best context window : {BEST_CTX_SEC} s  (from context study)
Evaluation windows  : {n_windows}
Horizons tested     : {", ".join(str(h)+"s" for h in HORIZONS_SEC)}

MAE RESULTS (degrees)
{"Horizon":>9s}  {"Pitch MAE":>12s}  {"Pitch RMSE":>12s}  {"Roll MAE":>10s}  {"Roll RMSE":>10s}
{"-"*58} """
    for _, row in hor_sum.iterrows():
        report_text += (
            f"{int(row['horizon_sec']):>8d}s  "
            f"{row['pitch_mean']:>8.4f} deg    "
            f"{row['pitch_rmse']:>8.4f} deg  "
            f"{row['roll_mean']:>8.4f} deg  "
            f"{row['roll_rmse']:>8.4f} deg\n"
        )

    roll_rms2  = float(np.sqrt(np.mean(df_clean["roll_deg"].values**2)))
    pitch_rms2 = float(np.sqrt(np.mean(df_clean["pitch_deg"].values**2)))

    report_text += f"""
NATO STANAG 4154 — HELICOPTER OPERATION CHECK
----------------------------------------------
Roll RMS   : {roll_rms2:.3f} deg  (limit 2.5 deg — {"SAFE" if roll_rms2 < 2.5 else "EXCEEDS LIMIT"})
Pitch RMS  : {pitch_rms2:.3f} deg  (limit 1.5 deg — {"SAFE" if pitch_rms2 < 1.5 else "EXCEEDS LIMIT"})

KEY FINDINGS
------------
1. Resampling essential: original CV={CV:.3f} indicates severe jitter
2. Roll has two timescales: slow sway (~107s) + fast wave (~2.1s)
   Makes roll harder to predict than pitch at long horizons
3. Best accuracy at short horizons (3-10s)
4. RMS predicted within 7-14% error across all horizons
5. Peak under-predicted at long horizons — apply safety margin for operations
{"="*65} """

    st.text_area("Report", report_text, height=400)

    st.divider()
    st.subheader("Download All Results")

    # Build ZIP in memory — single button, no page refresh
    import zipfile, io as _io
    _zip_buf = _io.BytesIO()
    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
        _zf.writestr("ship_motion_report.txt",      report_text)
        _zf.writestr("imu_cleaned_resampled.csv",   df_clean.to_csv(index=False))
        _zf.writestr("results_horizon_study.csv",   hor_df.to_csv(index=False))
        _zf.writestr("results_stats_all.csv",       stat_df.to_csv(index=False))
    _zip_buf.seek(0)

    st.download_button(
        label="Download all results as ZIP",
        data=_zip_buf.getvalue(),
        file_name="ship_motion_results.zip",
        mime="application/zip",
        key="dl_all_zip",
        use_container_width=True,
        type="primary",
    )
    st.caption(
        "ZIP contains: ship_motion_report.txt · imu_cleaned_resampled.csv · "
        "results_horizon_study.csv · results_stats_all.csv"
    )

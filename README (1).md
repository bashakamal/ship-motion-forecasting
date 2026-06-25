# Ship Motion Forecasting — Streamlit App

Zero-shot IMU wave motion forecasting using TimesFM.  
Upload a CSV → preprocessing → prediction → Peak / RMS / H1/3 results.

---

## Deploy in 5 steps (free)

### Step 1 — Create a GitHub repository

1. Go to https://github.com/new
2. Name it `ship-motion-forecasting`
3. Set to **Public**
4. Click **Create repository**

### Step 2 — Upload these three files

In your new repository, upload:
- `app.py`
- `requirements.txt`
- `README.md`

(Drag and drop all three into the GitHub file browser, then click **Commit changes**)

### Step 3 — Connect to Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Choose your repository: `ship-motion-forecasting`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**

### Step 4 — Wait for deployment (~5 minutes)

Streamlit installs all packages from `requirements.txt` automatically.  
The TimesFM model (~800 MB) downloads on first run — this takes ~3 minutes.  
After that it is cached and subsequent runs are fast.

### Step 5 — Share the link

Your app will be live at:
```
https://[your-github-username]-ship-motion-forecasting-app-[hash].streamlit.app
```

Copy this link and share it. Anyone can access it from a browser or mobile.

---

## If Streamlit Cloud runs out of RAM (TimesFM is ~800 MB)

Use **Hugging Face Spaces** instead — it gives 2 GB free RAM:

1. Go to https://huggingface.co/new-space
2. Name: `ship-motion-forecasting`
3. SDK: **Streamlit**
4. Visibility: Public
5. Upload `app.py` and `requirements.txt`
6. Click **Create Space**

Your app URL will be:
```
https://huggingface.co/spaces/[your-username]/ship-motion-forecasting
```

---

## Required CSV format

```
timestamp,roll_deg,pitch_deg,yaw_deg,gz,gx,gy,ax,ay,az
1764676000.00,-5.845,-3.808,176.87,-0.033,-0.080,0.056,0,0,0
1764676000.05,-5.539,-4.089,176.74,-0.035,-0.085,0.106,0,0,0
```

**Required columns:** `timestamp` (Unix seconds) or `time_sec`, `roll_deg`, `pitch_deg`  
**Optional columns:** `yaw_deg`, `gz`, `gx`, `gy`, `ax`, `ay`, `az`

---

## What the app does

| Tab | Content |
|-----|---------|
| Data Quality | Timestamp check, resampling validation, PSD frequency analysis |
| Preprocessing | Raw vs cleaned plots, roll decomposition (slow sway + fast wave) |
| Forecast | MAE by horizon, context window study, actual vs predicted charts |
| Statistics | Peak / RMS / H1/3 accuracy table (Input / Predicted / True / Abs%) |
| Report | Auto-generated summary + download buttons for all CSV results |

---

## Local testing

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

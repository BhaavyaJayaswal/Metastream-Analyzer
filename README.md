# Metastream Analyzer setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the App

```
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Features

### Model Analyzer
- Upload game ZIP files containing VMAF stream data
- Train ML models (Random Forest, Gradient Boosting, XGBoost) to predict throughput
- Configure hyperparameters or run automated grid search tuning
- Save trained models for later use

### Stream Analyzer
- Upload a game ZIP and select a VMAF quality pair
- Load a previously saved model
- Analyze the stream with throughput predictions and error metrics
- Interactive charts with error region highlighting and zoom inspector

## Data Format

Game ZIP files should contain:
```
tvmaf-{N}_tinterval-1/analytix/video_metrics.pkl.gz
```
where `N` is the VMAF level (50, 60, 70, 80, or 90).

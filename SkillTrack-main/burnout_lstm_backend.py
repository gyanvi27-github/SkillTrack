"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BURNOUT PREDICTION LSTM — BACKEND ENGINE                          ║
║          SkillTrack Student Wellness Intelligence System                    ║
║                                                                              ║
║  WHAT THIS FILE DOES:                                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. Loads the synthetic 1500-row dataset (student_mood_checkins.csv)         ║
║  2. Engineers rolling features (3-day / 7-day averages, trend slopes)        ║
║  3. Builds sliding 7-day sequences  →  input shape (N, 7, features)          ║
║  4. Trains a stacked LSTM with class-weighted binary cross-entropy            ║
║  5. Outputs per-student CONFIDENCE SCORES (0.0 – 1.0) for next-week burnout  ║
║  6. Saves artefacts the dashboard & Twilio alerter can consume:              ║
║       • burnout_lstm_model.keras   — trained model                           ║
║       • burnout_scaler.pkl         — MinMaxScaler (needed for live data)     ║
║       • burnout_predictions.json   — latest risk scores per student          ║
║       • burnout_report.csv         — human-readable report                   ║
║                                                                              ║
║  USAGE:                                                                      ║
║    pip install tensorflow scikit-learn pandas numpy joblib                   ║
║    python burnout_lstm_backend.py                                            ║
║                                                                              ║
║  TWILIO INTEGRATION (uncomment section at bottom):                           ║
║    pip install twilio                                                        ║
║    Set TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, COUNSELOR_PHONE env vars       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "student_mood_checkins.csv"   # path to dataset
SEQ_LEN         = 7                             # days the LSTM looks back
BURNOUT_THRESHOLD = 0.50                        # confidence >= this → alert
HIGH_RISK_THRESHOLD = 0.70                      # >= this → HIGH RISK badge
TEST_SPLIT      = 0.20                          # fraction of students held out
EPOCHS          = 60
BATCH_SIZE      = 32
PATIENCE        = 8                             # early stopping patience

# Base features fed into LSTM (will be scaled)
BASE_FEATURES = [
    "mood_score",
    "sleep_hours",
    "study_hours",
    "stress_level",
    "social_interactions",
    "assignment_load",
    "skipped_class",
]

# Engineered rolling features (added automatically)
ROLLING_FEATURES = [
    "mood_3d_avg",
    "stress_3d_avg",
    "mood_7d_avg",
    "mood_slope_3d",      # linear trend: negative = declining mood
    "sleep_deficit_flag", # 1 if sleep < 5h
]

ALL_FEATURES = BASE_FEATURES + ROLLING_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & ENGINEER FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["student_id", "day_number"]).reset_index(drop=True)

    enriched = []
    for sid, grp in df.groupby("student_id"):
        grp = grp.copy().reset_index(drop=True)
        m = grp["mood_score"]
        s = grp["stress_level"]
        sl = grp["sleep_hours"]

        # Rolling averages
        grp["mood_3d_avg"]  = m.rolling(3, min_periods=1).mean()
        grp["stress_3d_avg"]= s.rolling(3, min_periods=1).mean()
        grp["mood_7d_avg"]  = m.rolling(7, min_periods=1).mean()

        # 3-day linear slope of mood  (positive = improving, negative = declining)
        slopes = []
        for i in range(len(grp)):
            window = m.iloc[max(0, i-2):i+1].values
            if len(window) >= 2:
                slope = np.polyfit(range(len(window)), window, 1)[0]
            else:
                slope = 0.0
            slopes.append(slope)
        grp["mood_slope_3d"] = slopes

        # Sleep deficit flag
        grp["sleep_deficit_flag"] = (sl < 5.0).astype(int)

        enriched.append(grp)

    return pd.concat(enriched, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD SEQUENCES
# ──────────────────────────────────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame, scaler=None, fit_scaler: bool = True):
    """
    Returns:
        X      — shape (N, SEQ_LEN, len(ALL_FEATURES))
        y      — shape (N,)  binary burnout labels
        groups — shape (N,)  student_id per sequence
        scaler — fitted MinMaxScaler
    """
    from sklearn.preprocessing import MinMaxScaler

    df_scaled = df.copy()
    if fit_scaler:
        scaler = MinMaxScaler()
        df_scaled[ALL_FEATURES] = scaler.fit_transform(df[ALL_FEATURES])
    else:
        df_scaled[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    X_list, y_list, groups = [], [], []
    for sid, grp in df_scaled.groupby("student_id"):
        grp = grp.reset_index(drop=True)
        arr = grp[ALL_FEATURES].values
        lbl = grp["burnout_risk"].values
        for i in range(SEQ_LEN, len(arr)):
            X_list.append(arr[i - SEQ_LEN:i])
            y_list.append(lbl[i])
            groups.append(sid)

    return np.array(X_list), np.array(y_list), np.array(groups), scaler


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD LSTM MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_model(seq_len: int, n_features: int):
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from keras.regularizers import l2


    model = Sequential([
        # ── Layer 1: Bidirectional LSTM — captures both forward and backward
        #    temporal patterns in the 7-day mood window
        Bidirectional(
            LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
            input_shape=(seq_len, n_features)
        ),
        Dropout(0.35),

        # ── Layer 2: LSTM — distils the sequence into a fixed-size state
        LSTM(32, kernel_regularizer=l2(1e-4)),
        Dropout(0.25),
        BatchNormalization(),

        # ── Dense head — two layers for non-linear classification
        Dense(24, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.15),
        Dense(1, activation="sigmoid")   # output = P(burnout in next window)
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy",
         keras.metrics.AUC(name="auc"),
         keras.metrics.Precision(name="precision"),
         keras.metrics.Recall(name="recall")]
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN
# ──────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    from sklearn.model_selection import GroupShuffleSplit
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n[1/5] Engineering features …")
    X, y, groups, scaler = build_sequences(df, fit_scaler=True)
    print(f"      Sequences: {X.shape}  |  Burnout ratio: {y.mean()*100:.1f}%")

    print("[2/5] Splitting by student (no data leakage) …")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Class weights to handle imbalance
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    class_weight = {0: 1.0, 1: neg / max(pos, 1)}
    print(f"      Train: {len(y_tr)} seq | Test: {len(y_te)} seq")
    print(f"      Class weight → burnout: {class_weight[1]:.2f}x")

    print("[3/5] Building LSTM …")
    model = build_model(SEQ_LEN, X.shape[2])
    model.summary(print_fn=lambda x: print("     ", x))

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=PATIENCE,
                      restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)
    ]

    print("[4/5] Training …")
    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/5] Evaluating on held-out students …")
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  confusion_matrix)
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= BURNOUT_THRESHOLD).astype(int)

    print(classification_report(y_te, y_pred, target_names=["No Risk","Burnout"]))
    auc = roc_auc_score(y_te, y_prob)
    print(f"  ROC-AUC : {auc:.4f}")
    cm = confusion_matrix(y_te, y_pred)
    print(f"  Confusion Matrix:\n{cm}")

    return model, scaler, history, auc


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — CONFIDENCE SCORE ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def compute_confidence_scores(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    For each student, takes their LAST SEQ_LEN days as a sequence and
    returns a confidence score (0.0–1.0) representing the predicted
    probability of burnout in the NEXT 5–7 days.

    Returns a DataFrame with one row per student:
        student_id | name | archetype | confidence_score |
        risk_level | days_to_burnout_est | last_mood | trend
    """
    results = []

    # Scale full dataset with the fitted scaler
    df_scaled = df.copy()
    df_scaled[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    for sid, grp in df_scaled.groupby("student_id"):
        grp_raw  = df[df.student_id == sid].sort_values("day_number")
        grp_sc   = grp.sort_values("day_number")

        if len(grp_sc) < SEQ_LEN:
            continue   # not enough history

        # Last 7 days as input sequence
        seq = grp_sc[ALL_FEATURES].values[-SEQ_LEN:]
        seq = seq.reshape(1, SEQ_LEN, len(ALL_FEATURES))

        confidence = float(model.predict(seq, verbose=0)[0][0])

        # Estimate days to burnout from trend
        recent_mood = grp_raw["mood_score"].values[-7:]
        slope = np.polyfit(range(len(recent_mood)), recent_mood, 1)[0]

        if confidence >= HIGH_RISK_THRESHOLD:
            risk_level = "HIGH"
        elif confidence >= BURNOUT_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Rough estimate: if slope is negative, days until mood hits 3.5
        current_mood = grp_raw["mood_score"].values[-1]
        if slope < -0.05 and current_mood > 3.5:
            days_est = int((current_mood - 3.5) / abs(slope))
            days_est = max(1, min(days_est, 14))
        elif confidence >= BURNOUT_THRESHOLD:
            days_est = 5
        else:
            days_est = None

        results.append({
            "student_id"         : int(sid),
            "name"               : grp_raw["name"].iloc[0],
            "archetype"          : grp_raw["archetype"].iloc[0] if "archetype" in grp_raw.columns else "unknown",
            "confidence_score"   : round(confidence, 4),
            "risk_level"         : risk_level,
            "days_to_burnout_est": days_est,
            "last_mood"          : round(float(grp_raw["mood_score"].values[-1]), 2),
            "avg_mood_7d"        : round(float(grp_raw["mood_score"].values[-7:].mean()), 2),
            "avg_stress_7d"      : round(float(grp_raw["stress_level"].values[-7:].mean()), 2),
            "avg_sleep_7d"       : round(float(grp_raw["sleep_hours"].values[-7:].mean()), 2),
            "mood_slope_7d"      : round(float(slope), 4),
            "trend"              : "declining" if slope < -0.05 else
                                   "improving" if slope > 0.05 else "stable",
            "alert_counselor"    : confidence >= BURNOUT_THRESHOLD,
        })

    return pd.DataFrame(results).sort_values("confidence_score", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — TWILIO ALERT (uncomment & configure to enable)
# ──────────────────────────────────────────────────────────────────────────────
def send_twilio_alert(student_row: dict):
    """
    Sends an SMS/WhatsApp alert to the counselor via Twilio.

    Setup:
        pip install twilio
        export TWILIO_SID="ACxxxxxxxxxxxx"
        export TWILIO_TOKEN="your_auth_token"
        export TWILIO_FROM="+1415XXXXXXX"      # your Twilio number
        export COUNSELOR_PHONE="+91XXXXXXXXXX"  # counselor's number
    """
    # ── UNCOMMENT THE BLOCK BELOW AFTER SETTING ENV VARS ──────────────────
    # from twilio.rest import Client
    #
    # sid      = os.environ["TWILIO_SID"]
    # token    = os.environ["TWILIO_TOKEN"]
    # from_num = os.environ["TWILIO_FROM"]
    # to_num   = os.environ["COUNSELOR_PHONE"]
    #
    # body = (
    #     f"🚨 SkillTrack Burnout Alert\n"
    #     f"Student  : {student_row['name']} (ID {student_row['student_id']})\n"
    #     f"Risk     : {student_row['risk_level']}  "
    #     f"({student_row['confidence_score']*100:.1f}% confidence)\n"
    #     f"Est. days: {student_row.get('days_to_burnout_est', 'N/A')}\n"
    #     f"Mood ↓   : {student_row['avg_mood_7d']:.1f}/10 avg (7d)\n"
    #     f"Trend    : {student_row['trend']}\n"
    #     f"Action   : Please check in with this student today."
    # )
    #
    # client = Client(sid, token)
    # msg = client.messages.create(body=body, from_=from_num, to=to_num)
    # print(f"  Twilio SMS sent → {msg.sid}")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  [Twilio stub] Would alert counselor for {student_row['name']} "
          f"({student_row['risk_level']} | {student_row['confidence_score']*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SKILLTRACK — BURNOUT LSTM BACKEND")
    print("=" * 70)

    # ── Load + feature engineering
    print(f"\n Loading dataset: {CSV_PATH}")
    df = load_and_engineer(CSV_PATH)
    print(f"  {len(df):,} rows | {df.student_id.nunique()} students | "
          f"{df.burnout_risk.mean()*100:.1f}% burnout days")

    # ── Train
    model, scaler, history, auc = train(df)

    # ── Save model & scaler
    model.save("burnout_lstm_model.keras")
    joblib.dump(scaler, "burnout_scaler.pkl")
    print("\n  Saved: burnout_lstm_model.keras")
    print("  Saved: burnout_scaler.pkl")

    # ── Confidence scores for all students
    print("\n Computing per-student confidence scores …")
    scores_df = compute_confidence_scores(df, model, scaler)

    # ── Save JSON for dashboard
    def convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return str(obj)

    # Clean all string fields to remove characters that break JSON
    def clean_str(val):
        if isinstance(val, str):
            val = val.encode('ascii', errors='ignore').decode('ascii')
            val = ''.join(c for c in val if ord(c) >= 32)
        return val

    scores_json = scores_df.to_dict(orient="records")
    scores_json = [
        {k: clean_str(v) for k, v in row.items()}
        for row in scores_json
    ] 
    import tempfile, os

    payload = json.dumps({
    "model_auc" : round(auc, 4),
    "threshold" : BURNOUT_THRESHOLD,
    "seq_len"   : SEQ_LEN,
    "generated" : pd.Timestamp.now().isoformat(),
    "students"  : scores_json
    }, indent=2, default=convert)

# Verify it's valid before saving
    json.loads(payload)  # will raise if broken

# Write atomically
    tmp = "burnout_predictions.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, "burnout_predictions.json")
    print("  Saved: burnout_predictions.json")
    

    # ── Save CSV report
    scores_df.to_csv("burnout_report.csv", index=False)
    print("  Saved: burnout_report.csv")

    # ── Print summary table
    print("\n" + "=" * 70)
    print("  CONFIDENCE SCORE REPORT")
    print("=" * 70)
    print(f"{'ID':>4}  {'Name':<14} {'Score':>7}  {'Risk':<8}  "
              f"{'Days':>5}  {'Trend':<10}  {'Mood':>5}")
    print("-" * 70)
    for _, r in scores_df.iterrows():
        bar  = "█" * int(r.confidence_score * 20)
        days = str(r.days_to_burnout_est) if r.days_to_burnout_est else "—"
        flag = "⚠ " if r.risk_level == "HIGH" else ("! " if r.risk_level == "MEDIUM" else "  ")
        print(f"{flag}{int(r.student_id):>3}  {r['name']:<14} "
              f"{r.confidence_score*100:>5.1f}%  {r.risk_level:<8}  "
              f"{days:>5}  {r.trend:<10}  {r.last_mood:>5}")

    # ── Twilio alerts for at-risk students
    at_risk = scores_df[scores_df.alert_counselor == True]
    print(f"\n  Students requiring counselor alert: {len(at_risk)}")
    for _, row in at_risk.iterrows():
        send_twilio_alert(row.to_dict())

    print("\n  Pipeline complete.")
    print("=" * 70)
    return scores_df


if __name__ == "__main__":
    scores = main()


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER — use this in your dashboard/Flask API
# ──────────────────────────────────────────────────────────────────────────────
def load_and_predict(csv_path: str = CSV_PATH):
    """
    Load pre-trained model + scaler and return fresh confidence scores.
    Call this from your Flask/FastAPI route instead of re-training each time.

    Example:
        from burnout_lstm_backend import load_and_predict
        scores_df = load_and_predict()
        return scores_df.to_dict(orient="records")
    """
    import keras
    model = keras.models.load_model("burnout_lstm_model.keras")
    scaler = joblib.load("burnout_scaler.pkl")
    df     = load_and_engineer(csv_path)
    return compute_confidence_scores(df, model, scaler)

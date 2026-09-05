"""
predict.py — Batch LSTM inference microservice
Called ONCE by burnout_backend.php with ALL student sequences together.
Model loads once, predicts all, exits. No repeated startup overhead.

INPUT  (stdin):  JSON object:
    {
        "sequences": {
            "1": [[f1,f2,...], [f1,f2,...], ...],
            "2": [[...], ...],
            ...
        }
    }

OUTPUT (stdout): JSON object:
    {"1": 0.7821, "2": 0.3412, ...}
"""

import os
# Suppress ALL TensorFlow/oneDNN logs before importing
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL']    = '3'

import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'burnout_lstm_model.keras')

def main():
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("predict.py: No input received\n")
        sys.exit(1)

    try:
        payload   = json.loads(raw)
        sequences = payload['sequences']
    except Exception as e:
        sys.stderr.write(f"predict.py: Input parse error: {e}\n")
        sys.exit(1)

    if not sequences:
        print('{}', end='')
        return

    # Load model once
    try:
        import keras
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('keras').setLevel(logging.ERROR)
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        sys.stderr.write(f"predict.py: Model load error: {e}\n")
        sys.exit(1)

    # Build batch array shape (N, SEQ_LEN, n_features)
    student_ids = list(sequences.keys())
    try:
        batch = np.array([sequences[sid] for sid in student_ids], dtype=np.float32)
    except Exception as e:
        sys.stderr.write(f"predict.py: Array build error: {e}\n")
        sys.exit(1)

    # Single predict call for all students
    try:
        scores = model.predict(batch, verbose=0).flatten().tolist()
    except Exception as e:
        sys.stderr.write(f"predict.py: Prediction error: {e}\n")
        sys.exit(1)

    result = {sid: round(float(score), 6) for sid, score in zip(student_ids, scores)}
    print(json.dumps(result), end='')

if __name__ == '__main__':
    main()

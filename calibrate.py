import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load your brains
scaler = joblib.load('scaler.save')
encoder = load_model('encoder.h5')
xgb = joblib.load('xgb_model.save')

print("--- STARTING UNIT CALIBRATION SCAN ---")

# We will test different "Normal" Pressures to see what makes the model happy
test_pressures = [4.2, 42.0, 60.0, 420.0, 0.42] # Bar, Head, PSI, kPa, Normalized
test_flows = [150.0, 0.5, 1.0, 9000.0]          # L/min, Normalized, m3/h, L/hr

for p in test_pressures:
    for f in test_flows:
        # Build the vector exactly like the app does
        raw = np.array([[p, f]])
        scaled = scaler.transform(raw)
        model_input = np.concatenate([scaled, scaled, scaled], axis=1)
        latent = encoder.predict(model_input, verbose=0).flatten()[:16].reshape(1,16)
        final_input = np.concatenate([model_input, latent], axis=1)
        
        prob = xgb.predict_proba(final_input)[0][1]
        
        status = "🔴 LEAK" if prob > 0.5 else "🟢 NORMAL"
        print(f"Input [P={p}, F={f}] --> Prob: {prob:.4f} {status}")

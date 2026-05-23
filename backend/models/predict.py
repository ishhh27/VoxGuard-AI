import joblib
import numpy as np


model = joblib.load(
    "deepfake_model.pkl"
)


def predict_voice(features):

    features = np.array(features).reshape(1, -1)

    probabilities = model.predict_proba(
        features
    )[0]

    real_confidence = (
        probabilities[0] * 100
    )

    fake_confidence = (
        probabilities[1] * 100
    )

    confidence = round(
        max(
            real_confidence,
            fake_confidence
        ),
        2
    )

    # Smart threshold logic
    if confidence < 65:

        label = "SUSPICIOUS"

    elif fake_confidence > real_confidence:

        label = "AI-GENERATED"

    else:

        label = "REAL"
        
    return {
        "prediction": label,
        "confidence": confidence,
        "variance": round(
            float(np.var(features)),
            2
        )
    }
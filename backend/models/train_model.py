import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from core.audio_features import (
    extract_mfcc_features
)

X = []
y = []

REAL_DIRS = [
    "dataset/real",
    "deepfake_audio_dataset_jay15k/real"
]

FAKE_DIRS = [
    "dataset/fake",
    "deepfake_audio_dataset_jay15k/fake"
]


# REAL AUDIO
for real_dir in REAL_DIRS:

    for file in os.listdir(real_dir):

        path = f"{real_dir}/{file}"

        try:

            features = extract_mfcc_features(
                path
            )

            X.append(features)

            y.append(0)

        except Exception:

            print(f"Skipping: {file}")

# FAKE AUDIO
for fake_dir in FAKE_DIRS:

    for file in os.listdir(fake_dir):

        path = f"{fake_dir}/{file}"

        try:

            features = extract_mfcc_features(
                path
            )

            X.append(features)

            y.append(1)

        except Exception:

            print(f"Skipping: {file}")


X = np.array(X)
y = np.array(y)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
print("Total samples:", len(X))
model.fit(X, y)

joblib.dump(
    model,
    "deepfake_model.pkl"
)

print("Model trained successfully 😄🔥")
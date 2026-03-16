from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import random
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv("dataset/DSL-StrongPasswordData.csv")

# Prepare dataset for ML

X = data.iloc[:,3:].values
y = data["subject"].values

# Train Random Forest model

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)

users = sorted(data["subject"].unique())


@app.route("/")
def index():
    return render_template("index.html", users=users)


# MAIN ANALYSIS ROUTE (for graphs and metrics)
@app.route("/analyze", methods=["POST"])
def analyze():

    user = request.json["user"]

    genuine = data[data["subject"] == user]
    imposters = data[data["subject"] != user]

    X_genuine = genuine.iloc[:,3:].values
    X_imposter = imposters.iloc[:len(genuine),3:].values

    train_genuine = X_genuine[:200]
    test_genuine = X_genuine[200:]

    profile = np.mean(train_genuine, axis=0)

    dist_genuine = np.linalg.norm(test_genuine - profile, axis=1)
    dist_imposter = np.linalg.norm(X_imposter - profile, axis=1)

    threshold = np.mean(dist_genuine) + 2*np.std(dist_genuine)

    # labels
    y_true = np.concatenate((np.ones(len(dist_genuine)), np.zeros(len(dist_imposter))))
    scores = np.concatenate((dist_genuine, dist_imposter))

    predictions = scores <= threshold

    accuracy = np.mean(predictions == y_true)

    FAR = np.sum((scores <= threshold) & (y_true == 0)) / np.sum(y_true == 0)
    FRR = np.sum((scores > threshold) & (y_true == 1)) / np.sum(y_true == 1)

    fpr, tpr, _ = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)

    pattern = np.mean(train_genuine, axis=0)

    return jsonify({
        "genuine": dist_genuine.tolist(),
        "imposter": dist_imposter.tolist(),
        "roc_x": fpr.tolist(),
        "roc_y": tpr.tolist(),
        "pattern": pattern.tolist(),
        "accuracy": float(accuracy),
        "far": float(FAR),
        "frr": float(FRR),
        "auc": float(roc_auc)
    })


# LIVE SIMULATION
@app.route("/simulate", methods=["POST"])
def simulate():

    user = request.json["user"]

    # Separate genuine and imposters
    genuine = data[data["subject"] == user]
    imposters = data[data["subject"] != user]

    # Randomly choose genuine or imposter sample
    if random.random() > 0.5:
        sample_row = genuine.sample(1)
        actual = "Genuine"
    else:
        sample_row = imposters.sample(1)
        actual = "Imposter"

    # Extract feature vector
    sample_features = sample_row.iloc[:, 3:].values

    # Predict user using Random Forest
    predicted_user = model.predict(sample_features)[0]

    # Authentication decision
    if predicted_user == user:
        prediction = "Genuine"
    else:
        prediction = "Imposter"

    # Confidence score from RF probabilities
    probabilities = model.predict_proba(sample_features)[0]
    confidence = float(np.max(probabilities))

    return jsonify({
        "prediction": prediction,
        "actual": actual,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run()

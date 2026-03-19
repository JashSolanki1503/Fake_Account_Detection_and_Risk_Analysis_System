from src.load_models import load_models

knn_model, log_model, kmeans_model, anomaly_model, scaler, pca = load_models()


def analyze_account(input_data):

    # Supervised prediction
    prob = log_model.predict_proba(input_data)[0][1]
    prediction = knn_model.predict(input_data)[0]

    # Unsupervised preprocessing
    X_scaled = scaler.transform(input_data)
    X_pca = pca.transform(X_scaled)

    # Cluster prediction
    cluster = kmeans_model.predict(X_pca)[0]

    # Anomaly detection
    anomaly = anomaly_model.predict(X_pca)[0]

    return {
        "fake_probability": prob,
        "prediction": prediction,
        "cluster": cluster,
        "anomaly": anomaly
    }
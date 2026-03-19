import joblib

def load_models():
    log_model = joblib.load("models/log_model.pkl")
    knn_model = joblib.load("models/knn_pipline.pkl")
    kmeans_model = joblib.load("models/KMeans.pkl")
    anomaly_model = joblib.load("models/iso_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    return knn_model, log_model, kmeans_model, anomaly_model, scaler, pca
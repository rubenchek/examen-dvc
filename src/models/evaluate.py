import pandas as pd
import pickle
import os
import logging
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Dossier contenant les données
DATA_FOLDER = "data"
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, "processed_data")
MODEL_FOLDER = "models"
MODEL_FILE = "random_forest_model.pkl"
METRICS_FOLDER = "metrics"

# Initialisation du logger
logger = logging.getLogger(__name__)

def main():
    logger.info("Début de la phase d'évaluation du model")

    # Charger les datasets
    X_test_scaled, y_test = load_test_data()
    
    # Charger le modèle
    model = get_model(MODEL_FOLDER, MODEL_FILE)
    
    # Faire des prédictions
    predictions = model_predict(model, X_test_scaled)
    
    # Évaluer le modèle
    metrics = evaluate_model(y_test,predictions)
    
    #Sauvegardes:
    save_metrics(metrics)   
    save_predictions(predictions)

def load_test_data():
    logger.info(f"Récupération du dataset de test")
    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'y_test.csv')).squeeze()
    return X_test_scaled, y_test

def get_model(MODEL_FOLDER, MODEL):
    logger.info(f"Récupération du model {MODEL}")
    path = os.path.join(MODEL_FOLDER, MODEL)
    
    with open(path, "rb") as f:
        model = pickle.load(f)
        
    return model

def model_predict(model, X_test_scaled):
    logger.info("Prédiction du model sur le dataset test")
    
    # Entraîner le modèle
    prediction = model.predict(X_test_scaled)
    
    return prediction

def evaluate_model(y_test,predictions):
    logger.info("Évaluation du modèle")

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {"mse": mse,
            "rmse":rmse,
            "mae":mae,
            "r2":r2}
    
    logger.info(f"📊 Métriques du modèle : {metrics}")
    
    return metrics
    
def save_metrics(metrics):
    logger.info("Sauvegarde des metrics")
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    metrics_path = os.path.join(METRICS_FOLDER, "scores.json")
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Métriques sauvegardées dans {metrics_path}")

def save_predictions(predictions):
    """Sauvegarde les prédictions dans un fichier CSV."""
    logger.info("Sauvegarde des prédictions")
    
    os.makedirs(DATA_FOLDER, exist_ok=True)
    output_path = os.path.join(DATA_FOLDER, "prediction.csv")
    
    df_preds = pd.DataFrame(predictions, columns=["prediction"])
    df_preds.to_csv(output_path, index=False)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
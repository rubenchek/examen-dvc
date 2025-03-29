import pandas as pd
import pickle
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# Dossier contenant les données
DATA_FOLDER = "data/processed_data/"
MODEL_FOLDER = "models/"

# Initialisation du logger
logger = logging.getLogger(__name__)

def main():
    logger.info("Début de la phase d'entrainement du model")

    # Charger les datasets
    X_train_scaled = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze()
    
    best_params = get_best_params(MODEL_FOLDER)
    
    rf = train_model(X_train_scaled,y_train,best_params)

    save_model(rf, X_train_scaled,y_train)

def get_best_params(MODEL_FOLDER):
    logger.info(f"Récupération des meilleurs paramètres")
    path = os.path.join(MODEL_FOLDER, "best_rf_params.pkl")
    
    with open(path, "rb") as f:
        best_params = pickle.load(f)
        
    return best_params

def train_model(X_train_scaled, y_train, best_params):
    logger.info("Entrainement du modèle ")
    
    # Créer le modèle avec les meilleurs paramètres
    rf = RandomForestRegressor(**best_params, random_state=42)
    
    # Entraîner le modèle
    rf.fit(X_train_scaled, y_train)
    
    return rf

def save_model(rf, X_train_scaled,y_train):
    logger.info(f"Sauvegarde du modèle entraîné - score d'entrainement : {rf.score(X_train_scaled, y_train)}")
    
    model_path = os.path.join(MODEL_FOLDER, "random_forest_model.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    
    logger.info(f"Modèle sauvegardé à {model_path}")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
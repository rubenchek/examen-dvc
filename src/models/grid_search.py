import pandas as pd
import pickle
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# Paramètres du modèle
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [200, 250, 300, 350],
    "max_depth": [15, 20, 25],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [2, 3, 4, 5]
}

# Dossier contenant les données
DATA_FOLDER = "data/processed_data/"
MODEL_FOLDER = "models/"

# Initialisation du logger
logger = logging.getLogger(__name__)

def main():
    logger.info("Début de l'entraînement : GridSearchCV avec RandomForest")

    # Charger les datasets
    X_train_scaled = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze()
    
    best_params = grid_search(X_train_scaled, y_train)
    
    # 💾 Sauvegarde des meilleurs paramètres
    save_best_params(best_params)

def grid_search(X_train_scaled,y_train):
    logger.info("Lancement du GridSearchCV...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", 
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_params = grid_search.best_params_
    logger.info(f"Meilleurs paramètres trouvés : {best_params}")
    
    return best_params
    
def save_best_params(best_params):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    path = os.path.join(MODEL_FOLDER, "best_rf_params.pkl")
    
    with open(path, "wb") as f:
        pickle.dump(best_params, f)
    logger.info(f"Paramètres sauvegardés dans {path}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import logging
from pathlib import Path
import joblib
import os

print(joblib.__version__)

data_folderpath = 'data/processed_data/'

def main():
    logger = logging.getLogger(__name__)
    logger.info('Normalizing data')
    
    # Charger les datasets
    X_train = pd.read_csv(os.path.join(data_folderpath, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_folderpath, 'X_test.csv'))
    
    # Normalisation Min-Max
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarde des datasets normalisés
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(data_folderpath, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(data_folderpath, 'X_test_scaled.csv'), index=False)

    logger.info('Normalization complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
# import des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import os


def main():
    logger = logging.getLogger(__name__)
    logger.info('creating train and test data sets from raw data')

    # Définition des chemins d'accès
    input_filepath = 'data/raw_data/raw.csv'
    output_filepath ='data/processed_data'
    
    # Assurer que le dossier de sortie existe
    os.makedirs(output_filepath, exist_ok=True)
    
    process_data(input_filepath, output_filepath)


    
def process_data(input_filepath, output_filepath):
    df = import_dataset(input_filepath)
    df.drop('date', axis=1, inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

def split_data(df):
    X = df.drop('silica_concentrate',axis=1)
    y = df['silica_concentrate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
    
def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        # if check_existing_file(output_filepath):
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
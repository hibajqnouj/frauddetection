import pandas as pd
import numpy as np
import joblib
import sys

# Imports Scikit-learn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# CONFIGURATION ET CHARGEMENT


def create_advanced_features(df):
    """
    Crée des features avancées (DOIT ÊTRE IDENTIQUE À L'APP)
    """
    df = df.copy()
   
    # RATIOS CRITIQUES 
    df['ratio_to_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['ratio_to_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
   
    #  ERREURS DE BALANCE 
    df['error_orig'] = np.abs((df['oldbalanceOrg'] - df['newbalanceOrig']) - df['amount'])
    df['error_dest'] = np.abs((df['newbalanceDest'] - df['oldbalanceDest']) - df['amount'])
    df['error_orig_norm'] = df['error_orig'] / (df['amount'] + 1)
    df['error_dest_norm'] = df['error_dest'] / (df['amount'] + 1)
   
    #  INDICATEURS 
    df['orig_emptied'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
    df['dest_was_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    threshold_high = df['amount'].quantile(0.90)
    df['amount_very_high'] = (df['amount'] > threshold_high).astype(int)
   
    #  FEATURES TEMPORELLES 
    df['hour'] = df['step'] % 24
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['day_of_week'] = (df['step'] // 24) % 7
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
   
    # LOGS ET DELTAS 
    df['amount_log'] = np.log1p(df['amount'])
    df['oldbalanceOrg_log'] = np.log1p(df['oldbalanceOrg'])
    df['oldbalanceDest_log'] = np.log1p(df['oldbalanceDest'])
    df['delta_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['delta_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
   
    return df

if __name__ == "__main__":
    print("="*60)
    print(" DÉMARRAGE DU SCRIPT D'ENTRAÎNEMENT")
    print("="*60)

    #  Chargement des données
    csv_file = 'CHDD.csv' # <--- VÉRIFIEZ LE NOM DE VOTRE FICHIER ICI
    try:
        print(f" Chargement de {csv_file}...")
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f" ERREUR: Le fichier '{csv_file}' est introuvable.")
        print("   Assurez-vous qu'il est dans le dossier ou modifiez le nom dans le script.")
        sys.exit(1)

    # Feature Engineering
    print(" Création des features avancées...")
    df = create_advanced_features(df)

    # Échantillonnage (Pour aller plus vite et éviter les problèmes de mémoire)
    sample_size = min(100000, len(df))
    print(f"  Échantillonnage de {sample_size} lignes...")
    df_sample = df.sample(n=sample_size, random_state=42)

    # Préparation des colonnes
    features_base = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    features_advanced = ['ratio_to_orig', 'ratio_to_dest', 'error_orig_norm', 'error_dest_norm',
                         'orig_emptied', 'dest_was_zero', 'amount_very_high', 'is_night', 'is_weekend',
                         'amount_log', 'oldbalanceOrg_log', 'oldbalanceDest_log', 'delta_orig', 'delta_dest']
   
    all_features = features_base + features_advanced
    X = df_sample[all_features].copy()
    y = df_sample['isFraud'].copy()

    # Nettoyage rapide des infinis/NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    num_cols = [col for col in all_features if col != 'type']
    cat_cols = ['type']

    # 5. Pipeline de préprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

    #  Définition des modèles
    true_contamination = y.mean()
    contamination_adjusted = min(true_contamination * 1.5, 0.01)
    if contamination_adjusted == 0: contamination_adjusted = 0.001 # Sécurité
   
    print(f" Contamination ajustée: {contamination_adjusted:.5f}")

    # On définit uniquement LOF car c'est celui utilisé par l'app (pour gagner du temps)
   
    pipeline_lof = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('pca', PCA(n_components=10, random_state=42)),
        ('model', LocalOutlierFactor(
            n_neighbors=30,
            contamination=contamination_adjusted,
            novelty=True,
            metric='manhattan',
            n_jobs=-1
        ))
    ])

    # Entraînement
    X_train = X[y == 0] # On entraîne sur les transactions normales
    print(f"  Entraînement du modèle LOF sur {len(X_train)} transactions normales...")
   
    pipeline_lof.fit(X_train)
    print(" Modèle LOF entraîné avec succès.")

    #  Sauvegarde des fichiers critiques
    print("\n" + "="*60)
    print(" SAUVEGARDE DES FICHIERS")
    print("="*60)

    # Sauvegarde du modèle LOF
    print(" Ecriture de 'model_lof_optimized.pkl'...")
    joblib.dump(pipeline_lof, "model_lof_optimized.pkl")

    # Sauvegarde des infos de features
    feature_info = {
        'all_features': all_features,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'contamination': contamination_adjusted
    }
    print("Ecriture de 'feature_info.pkl'...")
    joblib.dump(feature_info, "feature_info.pkl")

    print("\n TERMINÉ ! Les fichiers sont générés et compatibles.")
    print(" Vous pouvez maintenant lancer : streamlit run app.py")

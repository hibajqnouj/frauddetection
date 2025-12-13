import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import joblib
from sklearn.preprocessing import OneHotEncoder
sns.set(style="whitegrid")
df = pd.read_csv('CHDD.csv')
print("\n" + "="*60)
print("📊 EXPLORATION INITIALE")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nProportion fraudes: {df['isFraud'].mean():.4%}")
print(f"Nombre de fraudes: {df['isFraud'].sum()}")
print(f"\nTypes de transactions:\n{df['type'].value_counts()}")
print("\n" + "="*60)
print("🔧 FEATURE ENGINEERING AVANCÉ")
print("="*60)

def create_advanced_features(df):
    """
    Crée des features avancées pour améliorer la détection
    """
    df = df.copy()
    
    # ----- 1. RATIOS CRITIQUES -----
    # Ratio montant / solde origine (fraude si > 0.8)
    df['ratio_to_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # Ratio montant / solde destination
    df['ratio_to_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    # ----- 2. ERREURS DE BALANCE (INCOHÉRENCES) -----
    # Pour origine : newBalance devrait être = oldBalance - amount
    df['error_orig'] = np.abs(
        (df['oldbalanceOrg'] - df['newbalanceOrig']) - df['amount']
    )
    
    # Pour destination : newBalance devrait être = oldBalance + amount
    df['error_dest'] = np.abs(
        (df['newbalanceDest'] - df['oldbalanceDest']) - df['amount']
    )
    
    # Erreur normalisée
    df['error_orig_norm'] = df['error_orig'] / (df['amount'] + 1)
    df['error_dest_norm'] = df['error_dest'] / (df['amount'] + 1)
    
    # ----- 3. INDICATEURS DE COMPORTEMENT SUSPECT -----
    # Compte origine vidé complètement
    df['orig_emptied'] = (df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)
    df['orig_emptied'] = df['orig_emptied'].astype(int)
    
    # Destination initialement vide
    df['dest_was_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Montant très élevé (> 90e percentile)
    threshold_high = df['amount'].quantile(0.90)
    df['amount_very_high'] = (df['amount'] > threshold_high).astype(int)
    
    # ----- 4. FEATURES TEMPORELLES -----
    # Transactions nocturnes (plus suspectes)
    df['hour'] = df['step'] % 24
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Jour de la semaine
    df['day_of_week'] = (df['step'] // 24) % 7
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ----- 5. FEATURES DE BALANCE -----
    # Log des montants (pour normalisation)
    df['amount_log'] = np.log1p(df['amount'])
    df['oldbalanceOrg_log'] = np.log1p(df['oldbalanceOrg'])
    df['oldbalanceDest_log'] = np.log1p(df['oldbalanceDest'])
    
    # Différences de balance
    df['delta_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['delta_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    return df

# Application du feature engineering
df = create_advanced_features(df)

print("✅ Features créées:")
new_features = ['ratio_to_orig', 'ratio_to_dest', 'error_orig', 'error_dest', 
                'error_orig_norm', 'error_dest_norm', 'orig_emptied', 
                'dest_was_zero', 'amount_very_high', 'is_night', 'is_weekend',
                'amount_log', 'oldbalanceOrg_log', 'oldbalanceDest_log',
                'delta_orig', 'delta_dest']
for feat in new_features:
    print(f"  • {feat}")
print("\n" + "="*60)
print("📦 PRÉPARATION DES DONNÉES")
print("="*60)

# Échantillonnage stratifié pour garder la proportion de fraudes
from sklearn.model_selection import train_test_split

# Garder plus de données (100k au lieu de 80k)
df_sample = df.sample(n=min(100000, len(df)), random_state=42)

print(f"Taille échantillon: {len(df_sample)}")
print(f"Fraudes dans échantillon: {df_sample['isFraud'].sum()} ({df_sample['isFraud'].mean():.4%})")

# Sélection des features
features_base = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                 'oldbalanceDest', 'newbalanceDest']

features_advanced = ['ratio_to_orig', 'ratio_to_dest', 
                     'error_orig_norm', 'error_dest_norm',
                     'orig_emptied', 'dest_was_zero', 'amount_very_high',
                     'is_night', 'is_weekend',
                     'amount_log', 'oldbalanceOrg_log', 'oldbalanceDest_log',
                     'delta_orig', 'delta_dest']

# Combinaison features de base + avancées
all_features = features_base + features_advanced

X = df_sample[all_features].copy()
y = df_sample['isFraud'].copy()

# Gestion valeurs infinies/NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\nNombre total de features: {len(all_features)}")
num_cols = [col for col in all_features if col != 'type']
cat_cols = ['type']

# Utiliser RobustScaler au lieu de StandardScaler (résistant aux outliers)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())  # Plus robuste aux outliers
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

print("✅ Preprocessing configuré (RobustScaler + OneHotEncoder)")
print("\n" + "="*60)
print("🤖 CONFIGURATION DES MODÈLES")
print("="*60)

# Contamination ajustée à la vraie proportion
true_contamination = y.mean()
contamination_adjusted = min(true_contamination * 1.5, 0.01)  # 1.5x la vraie proportion

print(f"Contamination réelle: {true_contamination:.4f}")
print(f"Contamination ajustée: {contamination_adjusted:.4f}")

# ----- ISOLATION FOREST (optimisé) -----
pipeline_iso = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=10, random_state=42)),
    ('model', IsolationForest(
        n_estimators=500,  # Augmenté
        contamination=contamination_adjusted,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

# ----- LOCAL OUTLIER FACTOR (optimisé) -----
pipeline_lof = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=10, random_state=42)),
    ('model', LocalOutlierFactor(
        n_neighbors=30,  # Augmenté pour plus de stabilité
        contamination=contamination_adjusted,
        novelty=True,
        metric='manhattan',  # Souvent meilleur que euclidean
        n_jobs=-1
    ))
])

# ----- ONE-CLASS SVM (optimisé) -----
pipeline_svm = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=10, random_state=42)),
    ('model', OneClassSVM(
        kernel='rbf',
        gamma='scale',  # Auto-ajusté
        nu=contamination_adjusted
    ))
])

# ----- ELLIPTIC ENVELOPE (optimisé) -----
pipeline_ell = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=5, random_state=42)),  # Peu de composantes
    ('model', EllipticEnvelope(
        contamination=contamination_adjusted,
        support_fraction=0.95,
        random_state=42
    ))
])

print("✅ 4 modèles configurés avec features avancées")
print("\n" + "="*60)
print("🎓 ENTRAÎNEMENT DES MODÈLES")
print("="*60)

X_train = X[y == 0]  # Uniquement transactions normales
X_test = X
y_test = y

print(f"Transactions normales (train): {len(X_train)}")
print(f"Transactions test: {len(X_test)}")

# Entraînement
print("\n⏳ Entraînement en cours...")

pipeline_iso.fit(X_train)
print("  ✅ Isolation Forest")

pipeline_lof.fit(X_train)
print("  ✅ Local Outlier Factor")

pipeline_svm.fit(X_train)
print("  ✅ One-Class SVM")

pipeline_ell.fit(X_train)
print("  ✅ Elliptic Envelope")

# Prédictions
pred_iso = np.where(pipeline_iso.predict(X_test) == -1, 1, 0)
pred_lof = np.where(pipeline_lof.predict(X_test) == -1, 1, 0)
pred_svm = np.where(pipeline_svm.predict(X_test) == -1, 1, 0)
pred_ell = np.where(pipeline_ell.predict(X_test) == -1, 1, 0)

print("✅ Prédictions effectuées")
print("\n" + "="*60)
print("📊 ÉVALUATION DES PERFORMANCES")
print("="*60)

models_preds = {
    "Isolation Forest": pred_iso,
    "Local Outlier Factor": pred_lof,
    "One-Class SVM": pred_svm,
    "Elliptic Envelope": pred_ell
}

results = []
true = y_test.values

for name, preds in models_preds.items():
    tp = int(((true == 1) & (preds == 1)).sum())
    tn = int(((true == 0) & (preds == 0)).sum())
    fp = int(((true == 0) & (preds == 1)).sum())
    fn = int(((true == 1) & (preds == 0)).sum())
    
    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    results.append({
        "Model": name,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 6),
        "Total_Pred": int(preds.sum())
    })

metrics_df = pd.DataFrame(results).set_index("Model")
print("\n" + "="*60)
print(metrics_df.to_string())
print("="*60)

# Comparaison avant/après
print("\n" + "="*60)
print("📈 COMPARAISON ANCIEN vs NOUVEAU (LOF)")
print("="*60)
print("ANCIEN LOF (features basiques):")
print("  • Recall:    21.7%")
print("  • Precision: 18.3%")
print("  • F1-Score:  19.8%")
print("\nNOUVEAU LOF (features avancées):")
lof_metrics = metrics_df.loc["Local Outlier Factor"]
print(f"  • Recall:    {lof_metrics['Recall']*100:.1f}%")
print(f"  • Precision: {lof_metrics['Precision']*100:.1f}%")
print(f"  • F1-Score:  {lof_metrics['F1']*100:.1f}%")
print("\n" + "="*60)
print("📊 GÉNÉRATION DES VISUALISATIONS")
print("="*60)

# 1. Matrice de confusion pour le meilleur modèle
best_model_name = metrics_df['F1'].idxmax()
best_preds = models_preds[best_model_name]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Matrices de confusion
for idx, (name, preds) in enumerate(models_preds.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name}\nF1={metrics_df.loc[name, "F1"]:.3f}')
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')

plt.tight_layout()
plt.savefig('confusion_matrices_optimized.png', dpi=150, bbox_inches='tight')
print("✅ Matrices de confusion sauvegardées")

# 2. Comparaison des métriques
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df[['Precision', 'Recall', 'F1']].plot(kind='bar', ax=ax)
ax.set_title('Comparaison des performances (Features Avancées)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xlabel('Modèle')
ax.legend(['Précision', 'Rappel', 'F1-Score'])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('metrics_comparison_optimized.png', dpi=150, bbox_inches='tight')
print("✅ Comparaison des métriques sauvegardée")

# 3. PCA avec prédictions du meilleur modèle
X_proc = preprocessor.fit_transform(X)
pca_vis = PCA(n_components=2, random_state=42)
X_pca = pca_vis.fit_transform(X_proc)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Vraies classes
axes[0].scatter(X_pca[true==0, 0], X_pca[true==0, 1], 
                c='blue', s=1, alpha=0.3, label='Normal')
axes[0].scatter(X_pca[true==1, 0], X_pca[true==1, 1], 
                c='red', s=10, alpha=0.8, label='Fraude')
axes[0].set_title('PCA - Vraies classes')
axes[0].legend()

# Prédictions meilleur modèle
axes[1].scatter(X_pca[best_preds==0, 0], X_pca[best_preds==0, 1], 
                c='blue', s=1, alpha=0.3, label='Normal (prédit)')
axes[1].scatter(X_pca[best_preds==1, 0], X_pca[best_preds==1, 1], 
                c='orange', s=10, alpha=0.8, label='Fraude (prédit)')
axes[1].set_title(f'PCA - Prédictions {best_model_name}')
axes[1].legend()

plt.tight_layout()
plt.savefig('pca_visualization_optimized.png', dpi=150, bbox_inches='tight')
print("✅ Visualisation PCA sauvegardée")

plt.show()
print("\n" + "="*60)
print("💾 SAUVEGARDE DU MEILLEUR MODÈLE")
print("="*60)

print(f"Meilleur modèle: {best_model_name} (F1={metrics_df.loc[best_model_name, 'F1']:.4f})")

# Sauvegarder le meilleur modèle
best_pipeline = {
    "Isolation Forest": pipeline_iso,
    "Local Outlier Factor": pipeline_lof,
    "One-Class SVM": pipeline_svm,
    "Elliptic Envelope": pipeline_ell
}[best_model_name]

with open("model_best_optimized.pkl", "wb") as f:
    joblib.dump(best_pipeline, f)

print(f"✅ Modèle sauvegardé: model_best_optimized.pkl")

# Sauvegarder aussi LOF pour compatibilité
with open("model_lof_optimized.pkl", "wb") as f:
    joblib.dump(pipeline_lof, f)

print(f"✅ LOF sauvegardé: model_lof_optimized.pkl")

# Sauvegarder les noms de features pour Streamlit
feature_info = {
    'all_features': all_features,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'contamination': contamination_adjusted
}

with open("feature_info.pkl", "wb") as f:
    joblib.dump(feature_info, f)

print(f"✅ Info features sauvegardée: feature_info.pkl")

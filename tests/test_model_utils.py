import pytest
import pandas as pd
import numpy as np
from src.model_utils import preprocess_data, calculate_custom_metric, optimize_threshold

def test_preprocess_data():
    # Création d'un DataFrame de test avec des valeurs manquantes
    test_data = pd.DataFrame({
        'numeric_col': [1.0, np.nan, 3.0, 4.0],
        'categorical_col': ['A', None, 'B', 'C']
    })
    
    # Prétraitement des données
    processed_df = preprocess_data(test_data)
    
    # Vérifications
    assert processed_df['numeric_col'].isna().sum() == 0, "Il reste des valeurs manquantes numériques"
    assert processed_df['categorical_col'].isna().sum() == 0, "Il reste des valeurs manquantes catégorielles"
    assert processed_df['numeric_col'].median() == processed_df['numeric_col'][1], "La valeur manquante n'a pas été remplacée par la médiane"
    assert processed_df['categorical_col'][1] == 'Unknown', "La valeur manquante catégorielle n'a pas été remplacée par 'Unknown'"

def test_calculate_custom_metric():
    # Cas de test avec différents scénarios
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    # Calcul de la métrique
    score = calculate_custom_metric(y_true, y_pred)
    
    # Un FP (coût = 1) et un FN (coût = 10)
    expected_score = (10 + 1) / 4  # (10*1 + 1*1) / 4
    
    assert score == expected_score, f"Le score calculé ({score}) ne correspond pas au score attendu ({expected_score})"

def test_optimize_threshold():
    # Création de données de test
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    
    # Optimisation du seuil
    best_threshold, best_score = optimize_threshold(y_true, y_proba)
    
    # Vérifications
    assert 0 <= best_threshold <= 1, "Le seuil optimal doit être entre 0 et 1"
    assert isinstance(best_score, float), "Le score doit être un nombre à virgule flottante"
    
    # Test avec le seuil optimal
    y_pred = (y_proba >= best_threshold).astype(int)
    score = calculate_custom_metric(y_true, y_pred)
    
    assert score == best_score, "Le score avec le seuil optimal ne correspond pas au meilleur score trouvé"

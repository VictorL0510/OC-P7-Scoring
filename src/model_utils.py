import pandas as pd
import numpy as np
from typing import Tuple, Union, List

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données pour le modèle de scoring.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes
        
    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    # Copie pour éviter de modifier les données originales
    df_processed = df.copy()
    
    # Gestion des valeurs manquantes numériques
    numeric_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
    
    # Gestion des valeurs manquantes catégorielles
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    df_processed[categorical_columns] = df_processed[categorical_columns].fillna('Unknown')
    
    return df_processed

def calculate_custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule la métrique personnalisée (coût FN = 10 × coût FP).
    
    Args:
        y_true (np.ndarray): Vraies étiquettes
        y_pred (np.ndarray): Prédictions
        
    Returns:
        float: Score de la métrique personnalisée
    """
    # Calcul des faux positifs et faux négatifs
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calcul du coût total (FN coûte 10 fois plus que FP)
    total_cost = (10 * fn + fp) / len(y_true)
    
    return total_cost

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Optimise le seuil de classification selon les critères métier.
    
    Args:
        y_true (np.ndarray): Vraies étiquettes
        y_proba (np.ndarray): Probabilités prédites
        
    Returns:
        Tuple[float, float]: Seuil optimal et score correspondant
    """
    thresholds = np.linspace(0, 1, 100)
    best_score = float('inf')
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = calculate_custom_metric(y_true, y_pred)
        
        if score < best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

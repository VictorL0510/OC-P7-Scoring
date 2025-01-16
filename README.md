# Projet de Scoring Crédit

Ce projet implémente un modèle de scoring pour prédire la probabilité de défaut de paiement des clients pour une société de crédit.

## Description

Le projet utilise des techniques de Machine Learning pour créer un modèle de classification qui prédit si un client est susceptible de rembourser son crédit ou non. Le modèle prend en compte différentes caractéristiques du client et de sa demande de prêt pour faire cette prédiction.

## Fonctionnalités

- Modèle de scoring optimisé avec GridSearchCV
- API de prédiction
- Interface de test avec Streamlit
- Suivi des expériences avec MLflow
- Tests unitaires avec Pytest
- Intégration continue avec Github Actions
- Monitoring de data drift avec Evidently

## Installation

```bash
pip install -r requirements.txt
```

## Structure du Projet

- `notebooks/` : Notebooks Jupyter pour l'analyse et le développement
- `api/` : Code source de l'API de prédiction
- `streamlit/` : Interface utilisateur Streamlit
- `tests/` : Tests unitaires
- `models/` : Modèles entraînés et métriques
- `data/` : Données d'entraînement et de test

## Utilisation

[Instructions à venir pour l'utilisation de l'API et de l'interface Streamlit]

## MLflow

Le projet utilise MLflow pour le tracking des expériences et le registre des modèles.

## Tests

```bash
pytest tests/
```

## Auteur

Victor Lesaffre

## Licence

Ce projet fait partie du parcours Data Scientist d'OpenClassrooms.

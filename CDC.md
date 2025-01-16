Préparation et Développement du Modèle :

Gérer le déséquilibre des classes dans les données
Implémenter la cross-validation et l'optimisation des hyperparamètres avec GridSearchCV
Créer un score métier personnalisé (coût FN = 10 × coût FP)
Optimiser le seuil de classification selon les critères métier
Maintenir un suivi des métriques techniques (AUC, accuracy)

MLOps - Tracking et Gestion des Modèles :

Configurer le tracking d'expérimentations avec MLFlow dans le notebook
Mettre en place et lancer l'interface web MLFlow UI
Implémenter le stockage centralisé des modèles avec MLFlow Model Registry
Tester le serving MLFlow

Gestion du Code et CI/CD :

Mettre en place la gestion de version avec Git
Créer et configurer un dépôt Github
Développer des tests unitaires avec Pytest
Configurer Github Actions pour :

L'exécution automatique des tests
Le déploiement continu de l'API sur le cloud



Développement et Déploiement API :

Créer une API de prédiction qui renvoie :

La probabilité de défaut
La classe (accepté/refusé) basée sur le seuil optimisé


Déployer l'API sur une plateforme cloud gratuite
Développer une interface de test locale (Notebook ou Streamlit)

Monitoring de Production :

Configurer Evidently pour la détection de Data Drift
Analyser le drift entre données d'entraînement (application_train) et de test (application_test)
Générer et analyser le rapport HTML Evidently sur les features principales
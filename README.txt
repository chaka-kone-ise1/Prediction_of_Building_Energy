# README: Projet IML - Prédiction de l'énergie des bâtiments

## 📊 Description du projet
Ce projet, intitulé **"Prediction of Building Energy"**, se concentre sur l'analyse et la prédiction des consommations énergétiques des bâtiments à l'aide de techniques avancées de machine learning et d'analyse de données. Il a été réalisé dans le cadre du cours supervisé par Madame Mously DIAW.

### Objectifs principaux :
- Analyser les tendances énergétiques des bâtiments à partir de données existantes.
- Créer des modèles prédictifs pour anticiper la consommation énergétique.
- Identifier les facteurs clés qui influencent la consommation.

## 📚 Structure du projet
Le projet est organisé comme suit :

### Fichiers et répertoires principaux :
- **`data/`** : Contient les ensembles de données utilisés pour l'analyse.
- **`notebooks/`** : Contient les notebooks de travail, y compris ce fichier principal.
- **`scripts/`** *(optionnel)* : Scripts Python pour des tâches automatisées.

### Sections du notebook :
1. **Importation des packages** : Configuration et importation des bibliothèques nécessaires.
2. **Définition du répertoire de travail** : Initialisation des chemins d'accès aux fichiers.
3. **Exploration des données (EDA)** : Analyse descriptive des données avec visualisations interactives.
4. **Traitement des données** : Gestion des valeurs manquantes et création des variables prédictives.
5. **Modélisation machine learning** : Implémentation de modèles (notamment XGBoost).
6. **Evaluation des modèles** : Comparaison des performances des modèles prédictifs.
7. **Visualisation des résultats** : Graphiques et résultats clés.

## 🔧 Installation et dépendances
Pour exécuter ce projet, les outils et bibliothèques suivants sont requis :

### Environnement Python :
- **Python 3.8+**

### Bibliothèques principales :
- `pandas` : Manipulation des données
- `numpy` : Opérations numériques
- `matplotlib` & `seaborn` : Visualisations statiques
- `plotly` : Visualisations interactives
- `xgboost` : Modélisation prédictive
- `missingno` : Gestion des données manquantes

### Installation :
Utilisez la commande suivante pour installer toutes les dépendances :
```bash
pip install -r requirements.txt
```

## 🔄 Exécution du notebook
1. Clonez le dépôt ou téléchargez le notebook.
2. Assurez-vous que toutes les dépendances sont installées.
3. Placez les fichiers de données dans le répertoire **`data/`**.
4. Lancez le notebook en utilisant Jupyter Notebook ou Jupyter Lab :
   ```bash
   jupyter notebook notebooks/notebook_principal.ipynb
   ```

## 🌐 Visualisations et résultats
Le projet inclut plusieurs visualisations clés :
- **Distribution des consommations énergétiques** : Comprendre les variations dans les données.
- **Corrélations** : Identifier les relations entre les variables.
- **Importance des variables** : Évaluer les variables clés pour les prédictions.

Les résultats préliminaires indiquent une précision de modèle satisfaisante avec XGBoost, avec des graphiques des résidus et des évaluations numériques.

## 👤 Contributions
### Membres de l'équipe :
- **Papa Abdourahmane CISSE**
- **Mouhamed TRAORE**
- **Chaka KONE**
- **Moussa Diakhité**

### Supervision :
- **Madame Mously DIAW**

### Comment contribuer :
Si vous souhaitez contribuer :
1. Forkez ce répôt.
2. Créez une nouvelle branche : `git checkout -b feature/AmazingFeature`.
3. Soumettez une pull request.

## 📚 Licence
Ce projet est sous licence [MIT](LICENSE).


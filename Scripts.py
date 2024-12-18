# LES PACKAGES 
#-------------

from pathlib import Path
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from pivottablejs import pivot_ui
from scipy.stats import chi2_contingency
import xgboost
from xgboost import XGBRegressor
#from ydata_profiling import ProfileReport
from yellowbrick.regressor import ResidualsPlot

# Définition du repertoire de travail
#--------------------------------------

HOME = Path.cwd().parent
print(f"Home directory: {HOME}")

# définir le répertoire des données
#----------------------------------

DATA = Path(HOME, "data")
print(f"Data directory: {DATA}")

# Importation des données
#------------------------

data=pd.read_csv(Path(DATA, "Building_Energy.csv"), sep=",")

# Première Mission : Réalisation d'une Analyse Exploratoire
#----------------------------------------------------------
# 1. Examen des données
#----------------------

data.info()

print('Nombre de doublons détecté : ',data.duplicated().sum())

plt.figure(figsize=(18, 6))

null = data.isnull().sum(axis=0).sort_values() / len(data) * 100
null_prop = data.isnull().sum(axis=0).sum()/len(data)/len(data.columns)*100

sns.barplot(x=null.index, y=null.values)
plt.ylabel("% de données manquantes", fontsize=13)
plt.xlabel("Variables", fontsize=13)
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.title("Pourcentage de valeurs manquantes pour chaque variable", fontsize=15)
plt.legend(['{:.2f}% global de données manquantes'.format(null_prop)], fontsize=13)
plt.show()

# 2. Nettoyage des données
#-------------------------

data["BuildingType"].unique()

fil_data = data[data['BuildingType'].isin(['Nonresidential WA', 'SPS-District K-12', 'Campus', 'NonResidential', 'Nonresidential COS'])]
print(fil_data['BuildingType'].unique())
print('Dimensions de la base filtrée : ',fil_data.shape)

fil_data.head()

# Remplacement des NA de SecondLargestPropertyUseType, SecondLargestPropertyUseTypeGFA, ThirdLargestPropertyUseType et ThirdLargestPropertyUseTypeGFA

fil_data["SecondLargestPropertyUseType"].fillna("Non Concerné", inplace=True)
fil_data["SecondLargestPropertyUseTypeGFA"].fillna(0, inplace=True)
fil_data["ThirdLargestPropertyUseType"].fillna("Non Concerné", inplace=True)
fil_data["ThirdLargestPropertyUseTypeGFA"].fillna(0, inplace=True)

# Création d'un DataFrame contenant la proportion de valeurs manquantes par variable
df = fil_data.isnull().sum()*100/len(fil_data)
df

# Recupérons les variables avec un taux de valeurs manquantes de plus de 50%
threshold = 50
high_NA_vars = df[df > threshold].index.tolist()
high_NA_vars

# Recupérons les variables dont toutes les modalités sont uniques
unique_modalities = fil_data.apply(lambda x: x.nunique())

unique_variables = unique_modalities[unique_modalities == len(fil_data)].index.tolist()

print("Variables dont toutes les modalités sont uniques:")
print(unique_variables)

# Recupérons les variable ayant une modalité
single_modal_vars = fil_data.apply(lambda col: col.nunique() == 1)

# Extraire les noms des colonnes avec une seule modalité
single_modal_vars = single_modal_vars[single_modal_vars].index.tolist()

print("Variables avec une seule modalité :")
print(single_modal_vars)

# Les variables à supprimer
var_ID = ['Address','TaxParcelIdentificationNumber','ZipCode','PropertyName']
vars_to_delete = high_NA_vars+single_modal_vars+unique_variables+var_ID
fil_data.drop(columns = vars_to_delete, inplace=True)

fil_data.shape

plt.figure(figsize=(18, 6))

null = fil_data.isnull().sum(axis=0).sort_values() / len(fil_data) * 100
null_prop = fil_data.isnull().sum(axis=0).sum()/len(fil_data)/len(fil_data.columns)*100

sns.barplot(x=null.index, y=null.values)
plt.ylabel("% de données manquantes", fontsize=13)
plt.xlabel("Variables", fontsize=13)
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.title("Pourcentage de valeurs manquantes pour chaque variable", fontsize=15)
plt.legend(['{:.2f}% global de données manquantes'.format(null_prop)], fontsize=13)
plt.show()

num_columns = list(fil_data.select_dtypes(include=['float64', 'int64']).columns) # Variables numériques
obj_columns = list(fil_data.select_dtypes(include=['object', 'bool']).columns) # Variables catégorielles
obj_columns,num_columns

fil_data["CouncilDistrictCode"] = fil_data["CouncilDistrictCode"].apply(str)
num_columns = list(fil_data.select_dtypes(include=['float64', 'int64']).columns)
obj_columns = list(fil_data.select_dtypes(include=['object', 'bool']).columns)
obj_columns,num_columns

# DataFrame renseignant sur le nombre de valeurs manquantes et de valeurs négatives par variable
data_dict = {'Variable': [], 'Missing Values': [], 'Negative Values': []}
for label in num_columns:
    missing_values = fil_data[label].isna().sum()
    negative_values = sum(fil_data[label] < 0.0)
    data_dict['Variable'].append(label)
    data_dict['Missing Values'].append(missing_values)
    data_dict['Negative Values'].append(negative_values)
result_df = pd.DataFrame(data_dict)
print(result_df)

(fil_data['GHGEmissionsIntensity'] == 1000*fil_data['TotalGHGEmissions']/fil_data['PropertyGFATotal']).all()

neg_cols = [col for col in num_columns if col != 'Longitude' and (fil_data[col] < 0).any()]
medians = fil_data[neg_cols].median()
for col in neg_cols:
    mask = (fil_data[col] < 0)
    fil_data.loc[mask, col] = medians[col]

    data_dict = {'Variable': [], 'Missing Values': [], 'Negative Values': []}
for label in neg_cols:
    missing_values = fil_data[label].isna().sum()
    negative_values = sum(fil_data[label] < 0.0)
    data_dict['Variable'].append(label)
    data_dict['Missing Values'].append(missing_values)
    data_dict['Negative Values'].append(negative_values)
result_df = pd.DataFrame(data_dict)
print(result_df)

numeric_cols_2 = [col for col in num_columns if col != 'ENERGYSTARScore']
for col in numeric_cols_2:
    median_value = fil_data[col].median()
    fil_data[col].fillna(median_value, inplace=True)

    plt.figure(figsize=(18, 6))

null = fil_data.isnull().sum(axis=0).sort_values() / len(fil_data) * 100
null_prop = fil_data.isnull().sum(axis=0).sum()/len(fil_data)/len(fil_data.columns)*100

sns.barplot(x=null.index, y=null.values)
plt.ylabel("% de données manquantes", fontsize=13)
plt.xlabel("Variables", fontsize=13)
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.title("Pourcentage de valeurs manquantes pour chaque variable", fontsize=15)
plt.legend(['{:.2f}% global de données manquantes'.format(null_prop)], fontsize=13)
plt.show()

fil_data= fil_data.dropna(subset=[col for col in fil_data.columns if col != 'ENERGYSTARScore'], how='any')

plt.figure(figsize=(18, 6))

null = fil_data.isnull().sum(axis=0).sort_values() / len(fil_data) * 100
null_prop = fil_data.isnull().sum(axis=0).sum()/len(fil_data)/len(fil_data.columns)*100

sns.barplot(x=null.index, y=null.values)
plt.ylabel("% de données manquantes", fontsize=13)
plt.xlabel("Variables", fontsize=13)
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.title("Pourcentage de valeurs manquantes pour chaque variable", fontsize=15)
plt.legend(['{:.2f}% global de données manquantes'.format(null_prop)], fontsize=13)
plt.show()

# 3. Traitement des valeurs aberantes
#------------------------------------

# Création d'un objet contenant toutes les variables liées à l'énergie
energy = ['GHGEmissionsIntensity','TotalGHGEmissions','NaturalGas(kBtu)','Electricity(kBtu)','SteamUse(kBtu)','SiteEnergyUseWN(kBtu)','SiteEnergyUse(kBtu)','SourceEUIWN(kBtu/sf)','SourceEUI(kBtu/sf)',
          'SiteEUIWN(kBtu/sf)','SiteEUI(kBtu/sf)']
plt.style.use('ggplot')
fig = plt.figure(1, figsize=(20, 15))
for i, label in enumerate(energy) :
    plt.subplot(4, 3, i + 1)
    sns.distplot(fil_data[label], bins=int(1 + np.log2(len(fil_data))))
plt.show()

#Boites à moustaches

fig = plt.figure(1, figsize=(20, 15))
for i,label in enumerate(energy) :
    plt.subplot(4, 3, i + 1)
    sns.boxplot(fil_data[label], orient="h")
plt.show()

# Transformation au log

plt.style.use('ggplot')
fig = plt.figure(1, figsize=(20, 15))
for i, label in enumerate(energy) :
    plt.subplot(4, 3, i + 1)
    sns.distplot(fil_data[label].apply(lambda x : np.log(1 + x)), bins=int(1 + np.log2(len(fil_data))))
plt.show()

fig = plt.figure(1, figsize=(20, 15))
for i,label in enumerate(energy) :
    plt.subplot(4, 3, i + 1)
    sns.boxplot(fil_data[label].apply(lambda x : np.log(1 + x)), orient="h")
plt.show()

fil_data_2 = fil_data

for label in energy :
    std_label = fil_data_2[label].apply(lambda x : np.log(1 + x)).std()
    mean_label = fil_data_2[label].apply(lambda x : np.log(1 + x)).mean()
    fil_data_2 = fil_data_2[(fil_data[label].apply(lambda x : np.log(1 + x))< mean_label + 3*std_label)]
    len(fil_data_2)- len(fil_data)

# 4. Analyse des variables catégorielles
# #-------------------------------------

obj_columns

# Le nombre de modalité par variable

fil_data_2[obj_columns].apply(lambda col: col.nunique())   

# Transformons les modalités des variables catégorielles en majuscule pour éviter un double compte
for cat in fil_data_2[obj_columns]:
    fil_data_2[cat] = fil_data_2[cat].apply(lambda x: str(x).upper())
fil_data_2[obj_columns]

# Redeterminons le nombre de modalités par variables
fil_data_2[obj_columns].apply(lambda col: col.nunique())

fil_data_2['Neighborhood'].unique()

fil_data_2['Neighborhood'] = fil_data_2['Neighborhood'].replace('DELRIDGE NEIGHBORHOODS', 'DELRIDGE')
fil_data_2['Neighborhood'].unique()
fil_data_2['PrimaryPropertyType'].unique()
fil_data_2['LargestPropertyUseType'].unique()
fil_data_2['ListOfAllPropertyUseTypes'].unique()

def compter_elements_uniques(liste):
    elements = [elem.strip() for elem in liste.split(',')]
    return len(set(elements))

# Créer une nouvelle colonne contenant le nombre d'éléments uniques dans chaque liste
fil_data_2['NumberOfPropertyUseTypes'] = fil_data_2['ListOfAllPropertyUseTypes'].apply(compter_elements_uniques)

fil_data_2['NumberOfPropertyUseTypes'].head()

# Sélectionnez uniquement les colonnes catégorielles
data_cat = fil_data_2.select_dtypes(include=['object', 'category'])

# Créez une fonction pour calculer le V de Cramer
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)  * 2) / (n - 1)
    kcorr = k - ((k - 1) * 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Créez un DataFrame pour stocker les résultats du V de Cramer
cramers_v_results = pd.DataFrame(index=data_cat.columns, columns=data_cat.columns)

# Calculer le V de Cramer pour chaque paire de variables catégorielles
for col1 in data_cat.columns:
    for col2 in data_cat.columns:
        if col1 != col2:
            v_cramer = cramers_v(data_cat[col1], data_cat[col2])
            cramers_v_results.loc[col1, col2] = v_cramer

# Convertir les valeurs de V de Cramer en nombres flottants
cramers_v_results = cramers_v_results.astype(float)

# Créer un heatmap des valeurs de V de Cramer
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_results, annot=True, cmap="YlGnBu")
plt.title("Heatmap des valeurs de V de Cramer entre les variables catégorielles")
plt.show()

fil_data_2['PrimaryPropertyType'].unique()
fil_data_2.loc[fil_data_2['PrimaryPropertyType'] == 'LOW-RISE MULTIFAMILY', 'ListOfAllPropertyUseTypes']
fil_data_2 = fil_data_2[fil_data_2['ListOfAllPropertyUseTypes'] != 'MULTIFAMILY HOUSING']

# Nouvelles classes en anglais et en majuscules
new_classes = {
    'EDUCATION': ['K-12 SCHOOL', 'UNIVERSITY'],
    'HEALTH AND MEDICAL': ['SENIOR CARE COMMUNITY', 'HOSPITAL', 'MEDICAL OFFICE', 'LABORATORY'],
    'OFFICE AND BUSINESS': ['OFFICE', 'LARGE OFFICE', 'SMALL- AND MID-SIZED OFFICE'],
    'STORAGE': ['SELF-STORAGE FACILITY', 'WAREHOUSE', 'REFRIGERATED WAREHOUSE', 'DISTRIBUTION CENTER'],
    'RETAIL AND PERSONAL SERVICES': ['RETAIL STORE'],
    'HOTEL': ['HOTEL'],
    'RESIDENTIAL AND HOUSING': ['LOW-RISE MULTIFAMILY', 'RESIDENCE HALL'],
    'FOOD AND BEVERAGE': ['RESTAURANT', 'SUPERMARKET / GROCERY STORE'],
    'MIXED USE': ['MIXED USE PROPERTY'],
    'OTHER SERVICES AND FACILITIES': ['OTHER', 'WORSHIP FACILITY']
}

# Fonction pour mapper les classes
def map_class(category):
    for key, values in new_classes.items():
        if category in values:
            return key
    return 'UNDEFINED'

# Création de la PrimaryProperty en fonction de la PrimaryPropertyType
fil_data_2['PrimaryProperty'] = fil_data_2['PrimaryPropertyType'].apply(map_class)

fil_data_2['PrimaryProperty'].unique()

fil_data_2['SecondLargestPropertyUseType'].unique()

new_classes = {
    'EDUCATION': ['K-12 SCHOOL', 'VOCATIONAL SCHOOL', 'COLLEGE/UNIVERSITY', 'ADULT EDUCATION', 'OTHER - EDUCATION', 'PRE-SCHOOL/DAYCARE'],
    'RETAIL AND PERSONAL SERVICES': ['RETAIL STORE', 'PERSONAL SERVICES (HEALTH/BEAUTY, DRY CLEANING, ETC)', 'CONVENIENCE STORE WITHOUT GAS STATION'],
    'ENTERTAINMENT AND LEISURE': ['OTHER - ENTERTAINMENT/PUBLIC ASSEMBLY', 'MOVIE THEATER', 'SWIMMING POOL', 'PERFORMING ARTS', 'BAR/NIGHTCLUB', 'SOCIAL/MEETING HALL'],
    'FOOD AND BEVERAGE': ['RESTAURANT', 'SUPERMARKET/GROCERY STORE', 'FOOD SALES', 'FOOD SERVICE', 'OTHER - RESTAURANT/BAR'],
    'HEALTH AND MEDICAL': ['MEDICAL OFFICE', 'LABORATORY', 'FITNESS CENTER/HEALTH CLUB/GYM'],
    'OFFICE AND BUSINESS': ['OFFICE', 'DATA CENTER', 'MANUFACTURING/INDUSTRIAL PLANT', 'BANK BRANCH', 'REPAIR SERVICES (VEHICLE, SHOE, LOCKSMITH, ETC)'],
    'RESIDENTIAL AND HOUSING': ['RESIDENCE HALL/DORMITORY', 'MULTIFAMILY HOUSING', 'OTHER - LODGING/RESIDENTIAL'],
    'STORAGE': ['SELF-STORAGE FACILITY', 'REFRIGERATED WAREHOUSE', 'NON-REFRIGERATED WAREHOUSE', 'DISTRIBUTION CENTER'],
    'OTHER SERVICES AND FACILITIES': ['OTHER', 'OTHER - SERVICES', 'OTHER - RECREATION', 'OTHER - PUBLIC SERVICES', 'ENCLOSED MALL', 'AUTOMOBILE DEALERSHIP', 'WORSHIP FACILITY'],
    'NOT CONCERNED': ['NOT CONCERNED'],
    'HOTEL': ['HOTEL'],
    'PARKING': ['PARKING']
}

# Fonction pour mapper les classes
def map_class(category):
    for key, values in new_classes.items():
        if category in values:
            return key
    return 'UNDEFINED'

# Création de la SecondLargest en fonction de la SecondLargestPropertyUseType
fil_data_2['SecondLargest'] = fil_data_2['SecondLargestPropertyUseType'].apply(map_class)

fil_data_2['ThirdLargestPropertyUseType'].unique()

# Nouvelles classes en anglais et en majuscules
new_classes = {
    'EDUCATION': ['K-12 SCHOOL', 'VOCATIONAL SCHOOL', 'COLLEGE/UNIVERSITY', 'ADULT EDUCATION', 'OTHER - EDUCATION', 'PRE-SCHOOL/DAYCARE'],
    'RETAIL AND PERSONAL SERVICES': ['RETAIL STORE', 'PERSONAL SERVICES (HEALTH/BEAUTY, DRY CLEANING, ETC)', 'CONVENIENCE STORE WITHOUT GAS STATION'],
    'ENTERTAINMENT AND LEISURE': ['SWIMMING POOL', 'OTHER - ENTERTAINMENT/PUBLIC ASSEMBLY', 'SOCIAL/MEETING HALL', 'BAR/NIGHTCLUB'],
    'FOOD AND BEVERAGE': ['RESTAURANT', 'FAST FOOD RESTAURANT', 'FOOD SERVICE', 'SUPERMARKET/GROCERY STORE'],
    'HEALTH AND MEDICAL': ['MEDICAL OFFICE', 'OTHER/SPECIALTY HOSPITAL', 'LABORATORY', 'FITNESS CENTER/HEALTH CLUB/GYM'],
    'OFFICE AND BUSINESS': ['OFFICE', 'DATA CENTER', 'FINANCIAL OFFICE', 'BANK BRANCH', 'MANUFACTURING/INDUSTRIAL PLANT'],
    'RESIDENTIAL AND HOUSING': ['MULTIFAMILY HOUSING'],
    'STORAGE': ['REFRIGERATED WAREHOUSE', 'NON-REFRIGERATED WAREHOUSE', 'DISTRIBUTION CENTER', 'SELF-STORAGE FACILITY'],
    'OTHER SERVICES AND FACILITIES': ['OTHER - SERVICES', 'OTHER - UTILITY', 'WORSHIP FACILITY', 'OTHER - RECREATION', 'OTHER - TECHNOLOGY/SCIENCE', 'OTHER', 'OTHER - RESTAURANT/BAR'],
    'HOTEL': ['HOTEL'],
    'PARKING': ['PARKING'],
    'NOT CONCERNED': ['NOT CONCERNED']
}

# Fonction pour mapper les classes
def map_class(category):
    for key, values in new_classes.items():
        if category in values:
            return key
    return 'UNDEFINED'

# Création de la ThirdLargest en fonction de la ThirdLargestPropertyUseType
fil_data_2['ThirdLargest'] = fil_data_2['ThirdLargestPropertyUseType'].apply(map_class)
fil_data_2['ThirdLargest'].unique()

# Description de la variable SiteEnergyUse(kBtu) 
#------------------------------------------------

# Min Max de SiteEnergyUse(kBtu)
min_site_energy_use = fil_data_2['SiteEnergyUse(kBtu)'].min()
max_site_energy_use = fil_data_2['SiteEnergyUse(kBtu)'].max()

print("Le minimum en consommation de SiteEnergyUse(kBtu) est :", min_site_energy_use)
print("le maximum en consommation de SiteEnergyUse(kBtu) est :", max_site_energy_use)

fil_data_2[fil_data_2['SiteEnergyUse(kBtu)']==0.0].shape
fil_data_2.loc[fil_data_2['SiteEnergyUse(kBtu)'] == 0.0, ['Electricity(kBtu)', 'NaturalGas(kBtu)', 'SteamUse(kBtu)']]

# Mise à jour de 'SiteEnergyUse(kBtu)' pour les lignes qui satisfont la condition
fil_data_2.loc[fil_data_2['SiteEnergyUse(kBtu)'] == 0.0, 'SiteEnergyUse(kBtu)'] = fil_data_2.loc[fil_data_2['SiteEnergyUse(kBtu)'] == 0.0, ['Electricity(kBtu)', 'NaturalGas(kBtu)', 'SteamUse(kBtu)']].sum(axis=1)
fil_data_2.loc[fil_data_2['SiteEnergyUse(kBtu)'] == 0.0, ['Electricity(kBtu)', 'NaturalGas(kBtu)', 'SteamUse(kBtu)']]
print("Nous avons a présent ", fil_data_2.loc[fil_data_2['SiteEnergyUse(kBtu)'] == 0.0, ['Electricity(kBtu)', 'NaturalGas(kBtu)', 'SteamUse(kBtu)']].shape[0], "batiments dont SiteEnergyUse est nulle")
fil_data_2 = fil_data_2[fil_data_2['SiteEnergyUse(kBtu)']!=0.0]

# Min Max de SiteEnergyUse(kBtu)
min_site_energy_use = fil_data_2['SiteEnergyUse(kBtu)'].min()
max_site_energy_use = fil_data_2['SiteEnergyUse(kBtu)'].max()

print("Le minimum en consommation de SiteEnergyUse(kBtu) est :", min_site_energy_use)
print("Le maximum en consommation de SiteEnergyUse(kBtu) est :", max_site_energy_use)

# Statistique descriptive sur Electricity(kBtu), NaturalGas(kBtu) et SteamUse(kBtu)
#----------------------------------------------------------------------------------

# Part de l'electricité et du gaz naturel dans la consommation

consommation_totale = fil_data_2['SiteEnergyUse(kBtu)'].sum()

consommation_electricite = fil_data_2['Electricity(kBtu)'].sum()
consommation_gaz_naturel = fil_data_2['NaturalGas(kBtu)'].sum()
consommation_steam = fil_data_2['SteamUse(kBtu)'].sum()

pourcentage_electricite = (consommation_electricite / consommation_totale) * 100
pourcentage_gaz_naturel = (consommation_gaz_naturel / consommation_totale) * 100
pourcentage_steam = (consommation_steam / consommation_totale) * 100


print("La part en pourcentage de la consommation d'électricité dans la consommation totale est : {:.5f}".format(pourcentage_electricite))
print("La part en pourcentage de la consommation de gaz naturel dans la consommation totale est :{:.5f}".format(pourcentage_gaz_naturel))
print("La part en pourcentage d'utilisation de vapeur dans la consommation totale est :{:.5f}".format(pourcentage_steam))

# Statistique descriptive sur le nombre d'année d'ancienneté des batiments

fil_data_2['NombreAnnees'] = 2016 - fil_data_2['YearBuilt']
min_nombre_annee = fil_data_2['NombreAnnees'].min()
max_nombre_annee = fil_data_2['NombreAnnees'].max()

print("Le minimum de nombre d'année d'ancienneté des batiments est de :", min_nombre_annee, "an(s)")
print("le maximum de nombre d'année d'ancienneté des batiments est de :", max_nombre_annee, "an(s)") 

# Analyse de la liaison entre SiteEnergyUse et Electricity(kBtu), NaturalGas(kBtu) et SteamUse(kBtu)
#---------------------------------------------------------------------------------------------------

y = fil_data_2['SiteEnergyUse(kBtu)']
x = fil_data_2['Electricity(kBtu)'] + fil_data_2['NaturalGas(kBtu)']+ fil_data_2['SteamUse(kBtu)']

# Tracé
sns.scatterplot(x=x, y=y)
plt.plot(y, y, color='blue', linestyle='-')  # Tracé de la droite y=x en rouge en pointillés
plt.xlabel('Consommation Totale')
plt.ylabel('Somme des autres consommations')
plt.title('')
plt.show()

# Liaison entre l'intensité d'émission et la consommation par surface
#--------------------------------------------------------------------

y = fil_data_2['GHGEmissionsIntensity']
x = 1000*fil_data_2['TotalGHGEmissions']/ fil_data_2['PropertyGFATotal']

# Tracé
sns.scatterplot(x=x, y=y)
plt.plot(y, y, color='blue', linestyle='--')  # Tracé de la droite y=x en rouge en pointillés
plt.xlabel('1000*TotalGHGEmissions / PropertyGFATotal')
plt.ylabel('GHG Emissions Intensity')
plt.title('')
plt.show()

# Croisement entre la variable SiteEnergyUse(kBtu) et la variable superficie des batiments
#-----------------------------------------------------------------------------------------

superficie = fil_data_2["PropertyGFABuilding(s)"]
consommation_energie = fil_data_2["SiteEnergyUse(kBtu)"]

# Crée un nuage de points
plt.scatter(superficie, consommation_energie)
plt.title("Consommation d'énergie en fonction de la superficie des bâtiments")
plt.xlabel("Superficie")
plt.ylabel("Consommation d'énergie (kWh)")

# Affiche le nuage de points
plt.show()

# Transformation au Log

superficie = fil_data_2["PropertyGFABuilding(s)"]
consommation_energie = fil_data_2["SiteEnergyUse(kBtu)"]

# Calcul du logarithme de la consommation d'énergie
log_consommation_energie = np.log(consommation_energie)
# Crée un nuage de points
plt.scatter(np.log(superficie), log_consommation_energie)
plt.title("Consommation d'énergie en fonction de la superficie des bâtiments")
plt.xlabel("Log-Superficie total de plancher brut")
plt.ylabel("Log-Consommation d'énergie (kBtu)")

# Affiche le nuage de points
plt.show()

# Croisement entre la variable SiteEnergyUse(kBtu) et la variable nombre d'étages
#--------------------------------------------------------------------------------

means_by_floor = fil_data_2.groupby('NumberofFloors')['SiteEnergyUse(kBtu)'].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.barplot(x=fil_data_2['NumberofFloors'], y=fil_data_2['SiteEnergyUse(kBtu)'], data=means_by_floor)
plt.xlabel('Number of Floors')
plt.ylabel('Consommation d\'énergie (kBtu)')
plt.xticks(rotation=90)
plt.show()

# Croisement entre nombre d'année d'ancienneté des batiments et SiteEnergyUse
#----------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

# Supprimer les lignes avec des valeurs manquantes pour éviter des problèmes de calcul
data_for_regression = fil_data_2[['NombreAnnees', 'SiteEnergyUse(kBtu)']].dropna()

# Régression linéaire
X = data_for_regression[['NombreAnnees']]
y = data_for_regression['SiteEnergyUse(kBtu)']

# Créer le modèle de régression linéaire
model = LinearRegression()

# Adapter le modèle aux données
model.fit(X, y)

# Prédictions
predictions = model.predict(X)

# Représentation graphique en nuage de points avec le logarithme de la consommation d'énergie
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(x='NombreAnnees', y=np.log1p(data_for_regression['SiteEnergyUse(kBtu)']), data=data_for_regression, ax=ax)
sns.lineplot(x=data_for_regression['NombreAnnees'], y=np.log1p(predictions), color='red', label='Régression linéaire', ax=ax)

ax.set(xlabel='Nombre d\'années d\'ancienneté', ylabel='la Consommation d\'énergie')
plt.title('Régression Linéaire entre Nombre d\'années et la Consommation d\'énergie')
plt.legend()
plt.show()

# Corrélation entre la variable SiteEnergyUse(kBtu) et la variable superficie des batiments

from scipy.stats import pearsonr

# Supprimer les lignes avec des valeurs manquantes pour éviter des problèmes de calcul
data_for_correlation = fil_data_2[['NombreAnnees', 'SiteEnergyUse(kBtu)']].dropna()

# Effectuer le test de corrélation de Pearson
correlation_coefficient, p_value = pearsonr(data_for_correlation['NombreAnnees'], np.log1p(data_for_correlation['SiteEnergyUse(kBtu)']))

# Afficher le coefficient de corrélation et la valeur p
print("Valeur p:", p_value)
print("Coefficient de corrélation:", correlation_coefficient)

# Matrice de correlation

# correlation plot
plt.figure(figsize=(14, 14))
corr =fil_data_2.select_dtypes(include=[int, float]).corr(method="pearson")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, cmap="YlGnBu", annot=True, square=True,
            fmt='.2f',
            mask=mask,
            cbar=True, vmin=-1, vmax=1);



var = [
    'SiteEUI(kBtu/sf)', 
    'SiteEUIWN(kBtu/sf)',
    'SourceEUI(kBtu/sf)',
    'SourceEUIWN(kBtu/sf)',
    'GHGEmissionsIntensity',
    'NaturalGas(kBtu)',
    'Electricity(kBtu)',
    'SteamUse(kBtu)',
    'Longitude',
    'Latitude',
    'PropertyGFATotal'
] + [
    'DefaultData', 
    'Electricity(kWh)',
    'NaturalGas(therms)',
    'ComplianceStatus',
    'ListOfAllPropertyUseTypes',
    'PrimaryPropertyType',
    'LargestPropertyUseType',
    'SecondLargestPropertyUseType',
    'ThirdLargestPropertyUseType',
    'SiteEnergyUseWN(kBtu)'
]
var
fil_data_2.drop(var,inplace = True, axis= 1)

# Enregistrement de la Base

fil_data_2.to_csv(Path(DATA, "2016_Building_Energy_2.csv"), index=False)

# Deuxième Mission : Tester différents modèles de prédiction afin de répondre au mieux à la problématique
#--------------------------------------------------------------------------------------------------------

data = pd.read_csv(Path(DATA, "2016_Building_Energy_2.csv"))

# Sélectionnez les colonnes catégorielles
df_cat = data.select_dtypes(include=['object'])

# Sélectionnez les colonnes numériques
df_num = data.select_dtypes(exclude=['object'])

# Concaténez les DataFrames dans le bon ordre
data = pd.concat([df_cat, df_num], axis=1)
data = data.iloc[:, data.columns != "ENERGYSTARScore"]
data.info()

X_index = data.columns != "SiteEnergyUse(kBtu)"  # Récupérer les colonnes différentes de "SiteEnergyUse(kBtu)"
y_index = data.columns == "SiteEnergyUse(kBtu)"  # Récupérer la colonne "SiteEnergyUse(kBtu)"

# Séparer les caractéristiques (X) et la cible (y) en utilisant iloc
X = data.iloc[:, X_index]
y = np.log1p(data.iloc[:, y_index])

num_columns =  ['PropertyGFAParking',
  'PropertyGFABuilding(s)',
  'LargestPropertyUseTypeGFA',
  'SecondLargestPropertyUseTypeGFA',
  'ThirdLargestPropertyUseTypeGFA',
  'TotalGHGEmissions',
                'YearBuilt',
                'NumberofBuildings',
                'NumberofFloors',
                'NumberOfPropertyUseTypes',
                'NombreAnnees'
                           
]
obj_columns = list(data.select_dtypes(include=['object', 'bool']).columns)
obj_columns,num_columns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from numpy import log

# Définition des étapes de transformation pour les variables qualitatives et numériques
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('log1p', FunctionTransformer(func=np.log1p))
])

# Création du préprocesseur en utilisant ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, obj_columns),
        ('num', numeric_transformer, num_columns)
    ],remainder='passthrough')

from sklearn import linear_model

import statsmodels.formula.api as smf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.diagnostic import het_white , normal_ad

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

def get_all_performances(value_train: tuple,
                         values_test: tuple,
                         metrics: list,
                        ) -> pd.DataFrame:
    test_perfs = []
    train_perfs = []
    metric_names = []
    for metric_func in metrics:
        metric_name = metric_func.__name__
        metric_names.append(metric_name)
        train_perfs.append(metric_func(*value_train))
        test_perfs.append(metric_func(*values_test))
    perfs = {"metric": metric_names, "train": train_perfs, "test": test_perfs,}
    return pd.DataFrame(perfs)
METRICS = [metrics.r2_score,
           metrics.mean_squared_error,
           metrics.mean_absolute_percentage_error,
           metrics.max_error, metrics.mean_absolute_error,
          ]

def train_model(model, x_train, y_train, x_test, y_test):
    # On entraîne ce modèle sur les données d'entrainement
    model.fit(x_train, y_train)
    
    # On récupère l'erreur de norme 2 sur le jeu de données train
    error_train = np.mean((model.predict(x_train).reshape(len(X_train),1) - y_train) ** 2)
    
    # On récupère l'erreur de norme 2 sur le jeu de données test
    error_test = np.mean((model.predict(x_test).reshape(len(X_test),1) - y_test) ** 2)

    # On obtient l'erreur quadratique ci-dessous
    print(f"Model error: {round(error_test, 5)}")
    return {"estimator": model, "error_train": error_train, "error_test": error_test}

# Regression linéaire
#--------------------

reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ("regressor", linear_model.LinearRegression()),
                          ]
                   )

reg_pipe

lr_results = train_model(model=reg_pipe,
                       x_train=X_train, y_train=y_train,
                       x_test=X_test, y_test=y_test)

lr = lr_results["estimator"]
get_all_performances(value_train=(y_train, lr.predict(X_train)),
                     values_test=(y_test, lr.predict(X_test)),
                     metrics=METRICS
                    )

# Regression linéaire : Optimisation des hyperparamètres

from sklearn.model_selection import GridSearchCV
# définition de Pipeline de régression avec Pipeline (c'est à nous de données les noms de chaque étape du workflow)
reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ("regressor", linear_model.LinearRegression()),
                          ]
                   )
# ou via make_pipeline (la seule différence est que make_pipeline génère automatiquement des noms pour les étapes).
# reg_pipe = make_pipeline(StandardScaler(), linear_model.LinearRegression())
reg_pipe

# Train and evaluate ridge regression
lr_results = train_model(model=reg_pipe,
                       x_train=X_train, y_train=y_train,
                       x_test=X_test, y_test=y_test)

lr_pipe = lr_results["estimator"]
param_grid = {
    "regressor__fit_intercept": [True, False],
}

grid_search = GridSearchCV(reg_pipe, param_grid, cv=10, scoring="r2", return_train_score=True)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best params: {best_params}")

grid_cv_results = grid_search.cv_results_
lr_best_model = grid_search.best_estimator_
grid_search.best_estimator_[-1].coef_.shape

# ElasticNet
# ---------- 

en_pipe = Pipeline(steps=[
                           ('preprocessor', preprocessor),
                           ("regressor", linear_model.ElasticNet()),
                          ]
                   )
en_pipe
en_results = train_model(model=en_pipe,
                       x_train=X_train, y_train=y_train,
                       x_test=X_test, y_test=y_test)

# ElasticNet : Optimisation des hyper paramètres
#-----------------------------------------------

# définition de Pipeline de régression avec Pipeline (c'est à nous de données les noms de chaque étape du workflow)
en_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ("regressor", linear_model.ElasticNet()),
                          ]
                   )
en_pipe
# Train and evaluate ridge regression
en_results = train_model(model=en_pipe,
                       x_train=X_train, y_train=y_train,
                       x_test=X_test, y_test=y_test)

eln_pipe = en_results["estimator"]

param_grid = {
    'regressor__alpha': [0.1, 0.5, 1.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.9],
    'regressor__fit_intercept': [True, False],
    'regressor__max_iter': [1000, 2000, 3000],
    'regressor__tol': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(en_pipe, param_grid, cv=10, scoring="r2", return_train_score=True)
grid_search
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best params: {best_params}")
grid_cv_results = grid_search.cv_results_

eln_best_model = grid_search.best_estimator_

# Importance des variables

encoded_categorical_column_names = preprocessor.named_transformers_['cat']\
    .named_steps['onehot'].get_feature_names_out(input_features=obj_columns)

# Récupérer les noms des colonnes pour les variables numériques
numeric_column_names = num_columns

# Concaténer les noms des colonnes catégorielles encodées et les noms des colonnes numériques
all_column_names = ['Intercept'] + list(encoded_categorical_column_names) + list(numeric_column_names)
encoded_column_names = reg_pipe.named_steps['preprocessor'].transformers_[0][1]\
    .named_steps['onehot'].get_feature_names_out(input_features=obj_columns)

    df_feature_importance = pd.DataFrame(grid_search.best_estimator_[-1].coef_.reshape(63,1), columns=["coef"], index=all_column_names)
print(f"Shape: {df_feature_importance.shape}")
(df_feature_importance
 .sort_values("coef", key=lambda v: abs(v), ascending=True)
 .plot(kind="barh", figsize=(10, 7))
)
plt.title("ElasticNet model")
plt.axvline(x=0, color='.6')
plt.subplots_adjust(left=.3);

X_index_eln = ['ThirdLargest','ThirdLargestPropertyUseTypeGFA','PrimaryProperty'
               ,'NumberOfPropertyUseTypes','PropertyGFABuilding(s)',
               'PropertyGFAParking','Neighborhood','NumberofBuildings']  # Récupérer les colonnes différentes de "SiteEnergyUse(kBtu)"


# Séparer les caractéristiques (X) et la cible (y) en utilisant iloc
X_train_eln = X_train.loc[:, X_index_eln]

# Séparer les caractéristiques (X) et la cible (y) en utilisant iloc
X_test_eln = X_test.loc[:, X_index_eln]

num_columns_2 = ['ThirdLargestPropertyUseTypeGFA',
               'NumberOfPropertyUseTypes','PropertyGFABuilding(s)',
               'PropertyGFAParking','NumberofBuildings']
obj_columns_2 = ['PrimaryProperty','ThirdLargest','Neighborhood']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('log1p', FunctionTransformer(func=np.log1p))
])

# Création du préprocesseur en utilisant ColumnTransformer
preprocessor_2 = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, obj_columns_2),
        ('num', numeric_transformer, num_columns_2)
    ],remainder='passthrough')

en_pipe_2 = Pipeline(steps=[('preprocessor_2', preprocessor_2),
                           ("regressor", linear_model.ElasticNet()),
                          ]
                   )
en_pipe_2
en_results_2 = train_model(model=en_pipe_2,
                       x_train=X_train_eln, y_train=y_train,
                       x_test=X_test_eln, y_test=y_test)
en_estimator_2 = en_results_2["estimator"]

get_all_performances(value_train=(y_train, en_estimator_2.predict(X_train_eln)),
                     values_test=(y_test, en_estimator_2.predict(X_test_eln)),
                     metrics=METRICS
                    )

param_grid = {
    'regressor__alpha': [0.1, 0.5, 1.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.9],
    'regressor__fit_intercept': [True, False],
    'regressor__max_iter': [1000, 2000, 3000],
    'regressor__tol': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(en_pipe_2, param_grid, cv=10, scoring="r2", return_train_score=True)
grid_search
grid_search.fit(X_train_eln, y_train)

best_params = grid_search.best_params_
print(f"Best params: {best_params}")


eln_best_model = grid_search.best_estimator_
eln_best_model.fit(X_train_eln, y_train)

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, eln_best_model.predict(X_train_eln)),
                     values_test=(y_test, eln_best_model.predict(X_test_eln)),
                     metrics=METRICS)

# Decision Tree 
#--------------

from sklearn.tree import DecisionTreeRegressor
dt_reg = Pipeline(steps=[('preprocessor', preprocessor),
                           ("regressor", DecisionTreeRegressor()),
                          ]
                   )
dt_reg
dt_results = train_model(model=dt_reg,
                       x_train=X_train, y_train=y_train,
                       x_test=X_test, y_test=y_test)

dt_estimator = dt_results["estimator"]
dt_estimator
# get performances in train & test
get_all_performances(value_train=(y_train, dt_estimator.predict(X_train)),
                     values_test=(y_test, dt_estimator.predict(X_test)),
                     metrics=METRICS
                    )

# Decision Tree :  Optimisation des hyper paramètres
#---------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Créer un pipeline avec le modèle Decision Tree
dt_pipe = Pipeline(steps=[('preprocessor', preprocessor),
    ("regressor", DecisionTreeRegressor())
])

# Définir la grille des paramètres à rechercher
param_grid = {
    'regressor__max_depth': [None, 5, 10, 15],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['auto', 'sqrt', 'log2']
}

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(estimator=dt_pipe, param_grid=param_grid, cv=5, scoring='r2')

# Exécuter la recherche sur la grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
best_params = grid_search.best_params_
print("Meilleurs paramètres:", best_params)

dt_best_model = grid_search.best_estimator_
dt_best_model.fit(X_train, y_train)

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, dt_best_model.predict(X_train)),
                     values_test=(y_test, dt_best_model.predict(X_test)),
                     metrics=METRICS)

# Random forest 
#--------------

from sklearn.ensemble import RandomForestRegressor
rf_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("regressor", RandomForestRegressor())
])

# Entraîner le modèle Random Forest
rf_results = train_model(model=rf_reg,
                         x_train=X_train, y_train=y_train,
                         x_test=X_test, y_test=y_test)

# Obtenir l'estimateur du modèle Random Forest
rf_estimator = rf_results["estimator"]

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, rf_estimator.predict(X_train)),
                     values_test=(y_test, rf_estimator.predict(X_test)),
                     metrics=METRICS)

# Random forest : Optimisation des hyper paramètres
#--------------------------------------------------

rf_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("regressor", RandomForestRegressor())
])

# Définir la grille des paramètres à rechercher
param_grid = {
    'regressor__n_estimators': [200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [5, 10],
}

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=5, scoring='r2')

# Exécuter la recherche sur la grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
best_params = grid_search.best_params_
print("Meilleurs paramètres:", best_params)

# Entraîner le modèle Random Forest avec les meilleurs paramètres trouvés
rf_best_model = grid_search.best_estimator_
rf_best_model.fit(X_train, y_train)

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, rf_best_model.predict(X_train)),
                     values_test=(y_test, rf_best_model.predict(X_test)),
                     metrics=METRICS)

# XGBoost
#--------

xgb_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("regressor", XGBRegressor())
])

# Entraîner le modèle XGBoost
xgb_results = train_model(model=xgb_reg,
                          x_train=X_train, y_train=y_train,
                          x_test=X_test, y_test=y_test)

# Obtenir l'estimateur du modèle XGBoost
xgb_estimator = xgb_results["estimator"]

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, xgb_estimator.predict(X_train)),
                     values_test=(y_test, xgb_estimator.predict(X_test)),
                     metrics=METRICS)

# XGBoost : Optimisation des hyper paramètres
#--------------------------------------------

xgb_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("regressor", XGBRegressor())
])

# Définir la grille des paramètres à rechercher
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__min_child_weight': [1, 3, 5]
}

# Créer l'objet GridSearchCV
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, scoring='r2')

# Exécuter la recherche sur la grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
best_params = grid_search.best_params_
print("Meilleurs paramètres:", best_params)


# Entraîner le modèle XGBoost avec les meilleurs paramètres trouvés
xgb_best_model = grid_search.best_estimator_
xgb_best_model.fit(X_train, y_train)

# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, xgb_best_model.predict(X_train)),
                     values_test=(y_test, xgb_best_model.predict(X_test)),
                     metrics=METRICS)

# Comparaison des modèles
#------------------------

models = [ xgb_best_model, rf_best_model, lr_best_model, eln_best_model, dt_best_model]
model_names = [ 'XGBoost', 'Random Forest', 'Linear Regression', 'ElasticNet', 'Decision Tree']

for model, model_name in zip(models, model_names):
    print("Model:", model_name)
    print(get_all_performances(value_train=(y_train, model.predict(X_train)),
                                values_test=(y_test, model.predict(X_test)),
                                metrics=METRICS))
    print("\n")

# Enregistrement du modèle
# ------------------------    

import pickle

# Sauvegarder le modèle dans un fichier
with open('model.pkl', 'wb') as file:
    pickle.dump(xgb_best_model, file)
    import joblib

# Sauvegarder le modèle dans un fichier
joblib.dump(xgb_best_model, 'model.joblib')

# Evaluons l'intérêt de EnergyStar_score pour la prediction de la consommation d'énergie
#---------------------------------------------------------------------------------------

data_2 = pd.read_csv(Path(DATA, "2016_Building_Energy_2.csv"))

df_cat = data_2.select_dtypes(include=['object'])

# Sélectionnez les colonnes numériques
df_num = data_2.select_dtypes(exclude=['object'])

# Concaténez les DataFrames dans le bon ordre
data_2 = pd.concat([df_cat, df_num], axis=1)
X_index = data_2.columns != "SiteEnergyUse(kBtu)"  # Récupérer les colonnes différentes de "SiteEnergyUse(kBtu)"
y_index = data_2.columns == "SiteEnergyUse(kBtu)"  # Récupérer la colonne "SiteEnergyUse(kBtu)"

# Séparer les caractéristiques (X) et la cible (y) en utilisant iloc
X = data_2.iloc[:, X_index]
y = np.log1p(data_2.iloc[:, y_index])

num_columns =  ['PropertyGFAParking',
  'PropertyGFABuilding(s)',
  'LargestPropertyUseTypeGFA',
  'SecondLargestPropertyUseTypeGFA',
  'ThirdLargestPropertyUseTypeGFA',
  'TotalGHGEmissions',
                'YearBuilt',
                'NumberofBuildings',
                'NumberofFloors',
                'NumberOfPropertyUseTypes',
                'NombreAnnees'
                           
]
obj_columns = list(data_2.select_dtypes(include=['object', 'bool']).columns)
obj_columns,num_columns

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.impute import KNNImputer

# Ajout d'un imputeur KNN pour la variable EnergyStarScore
imputer_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))  # Vous pouvez ajuster le nombre de voisins selon votre besoin
])

# Ajout de l'imputation KNN dans le préprocesseur existant
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, obj_columns),
        ('num', numeric_transformer, num_columns),
        ('impute', imputer_transformer, ['ENERGYSTARScore'])  # Traitement spécifique pour EnergyStarScore
    ]
)

# Création du pipeline final avec le préprocesseur
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
final_pipeline

# Regression linéaire
#--------------------

from sklearn.feature_selection import RFE

rfe_pipe = Pipeline([
    ("preprocess", final_pipeline),
    ("feature_selection", RFE(estimator=linear_model.LinearRegression(), n_features_to_select=27)),
    ("regressor", linear_model.LinearRegression())
])
rfe_results = train_model(model=rfe_pipe, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
rfe_pipe = rfe_results["estimator"]
# Afficher les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, rfe_pipe.predict(X_train)),
                     values_test=(y_test, rfe_pipe.predict(X_test)),
                     metrics=METRICS)

# XGBoost 
#--------

from xgboost import XGBRegressor  # Importez XGBRegressor


rfe_pipe_xgb = Pipeline([
    ("preprocess", final_pipeline),
    ("feature_selection", RFE(estimator=XGBRegressor(), n_features_to_select=10)),
    ("regressor", XGBRegressor())
])


rfe_results_xgb = train_model(model=rfe_pipe_xgb, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
rfe_pipe_xgb = rfe_results_xgb["estimator"]

# Affichez les performances sur les ensembles d'entraînement et de test
get_all_performances(value_train=(y_train, rfe_pipe_xgb.predict(X_train)),
                     values_test=(y_test, rfe_pipe_xgb.predict(X_test)),
                     metrics=METRICS)


param_grid = {
    'feature_selection__n_features_to_select': [10,15,20],  # Nombre de caractéristiques à sélectionner
    'regressor__n_estimators': [100, 200, 300],  # Nombre d'estimateurs dans XGBoost
}

# Initialisez GridSearchCV avec votre pipeline et la grille de paramètres
grid_search = GridSearchCV(estimator=rfe_pipe_xgb, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Effectuez la recherche sur grille
grid_search.fit(X_train, y_train)

# Obtenez le meilleur modèle trouvé
best_model_xbgrfe = grid_search.best_estimator_

get_all_performances(value_train=(y_train, best_model_xbgrfe.predict(X_train)),
                     values_test=(y_test, best_model_xbgrfe.predict(X_test)),
                     metrics=METRICS)

# Decision Tree
#--------------

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


dt_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("regressor", DecisionTreeRegressor())
])

rfe_dt_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', RFE(estimator=DecisionTreeRegressor(), n_features_to_select=10)),
    ("regressor", DecisionTreeRegressor())
])

rfe_results = train_model(model=rfe_dt_reg, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
rfe_estimator = rfe_results["estimator"]

get_all_performances(value_train=(y_train, rfe_estimator.predict(X_train)),
                     values_test=(y_test, rfe_estimator.predict(X_test)),
                     metrics=METRICS)

param_grid = {
    'feature_selection__n_features_to_select': [10, 15, 20],  # Nombre de caractéristiques à sélectionner
    'regressor__max_depth': [None, 10, 20],  # Profondeur maximale de l'arbre de décision
    'regressor__min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud  # Nombre minimum d'échantillons requis pour être à un nœud feuille
}

# Initialisez GridSearchCV avec votre pipeline et la grille de paramètres
grid_search = GridSearchCV(estimator=rfe_dt_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Effectuez la recherche sur grille
grid_search.fit(X_train, y_train)

# Obtenez le meilleur modèle trouvé
best_model_rfe_dt_reg = grid_search.best_estimator_

# Comparaison des modèles
#------------------------

models = [ rfe_pipe_xgb, rfe_pipe, rfe_dt_reg]
model_names = [ 'XGBoost',  'Linear Regression', 'Decision Tree']

for model, model_name in zip(models, model_names):
    print("Model:", model_name)
    print(get_all_performances(value_train=(y_train, model.predict(X_train)),
                                values_test=(y_test, model.predict(X_test)),
                                metrics=METRICS))
    print("\n")

# Enregistrement du modèle
# ------------------------

import pickle

# Sauvegarder le modèle dans un fichier
with open('model2.pkl', 'wb') as file:
    pickle.dump(rfe_pipe_xgb, file)    
import joblib

# Sauvegarder le modèle dans un fichier
joblib.dump(rfe_pipe_xgb, 'model2.joblib')    

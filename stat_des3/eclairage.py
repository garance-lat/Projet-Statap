"""import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def caracteristiques():
    path="/home/onyxia/work/projet-python/"
    df = pd.DataFrame()
    for year in range(2005, 2023):
        file_name1 = f"caracteristiques_{year}.csv"
        df_year = pd.read_csv(path + file_name1, sep=',', low_memory=False)
        df_year['an'] = year  # Adding the year column directly during file reading
        df = pd.concat([df, df_year])
    return df

df = caracteristiques()
# Make sure 'Num_Acc' and 'lum' are present in the 'caracteristiques' files
df = df[['Num_Acc', 'lum', 'an']]  # Keep only the necessary columns

def lieux():
    path="/home/onyxia/work/projet-python/"
    df = pd.DataFrame()
    for year in range(2005, 2023):
        file_name1 = f"lieux_{year}.csv"
        df_year = pd.read_csv(path + file_name1, sep=',', low_memory=False)
        df = pd.concat([df, df_year])
    return df

df2 = lieux()
# Make sure 'Num_Acc' is present in the 'lieux' files
df2 = df2[['Num_Acc']]  # Keep only the necessary columns
df_merge = pd.merge(df, df2, on='Num_Acc')

# Ensure 'lum' column is string and strip whitespaces
df_merge['lum'] = df_merge['lum'].astype(str).str.strip()

# Normalize values in 'lum' column
df_merge['lum'] = df_merge['lum'].replace({"1.0": "1", "2.0": "2", "3.0": "3", "4.0": "4", "5.0": "5"})

# Group data by year and lighting condition, then count occurrences
grouped_data = df_merge.groupby(['an', 'lum']).size().unstack().fillna(0)
grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

# Group data by year and lighting condition, then count occurrences
grouped_data = df_merge.groupby(['an', 'lum']).size().unstack().fillna(0)
# Calculate the total number of accidents per year
total_accidents_per_year = grouped_data.sum(axis=1)
grouped_data_percentage = grouped_data.div(total_accidents_per_year, axis=0) * 100

# Group data by year and lighting condition, then count occurrences
grouped_data = df_merge.groupby(['an', 'lum']).size().unstack().fillna(0)

# Calculate the total number of accidents per year
total_accidents_per_year = grouped_data.sum(axis=1)

# Sort the conditions within each year so the largest percentage is at the bottom of the bar
sorted_grouped_data = grouped_data.apply(lambda x: x.sort_values(ascending=False), axis=1)
grouped_data_percentage = sorted_grouped_data.div(total_accidents_per_year, axis=0) * 100

# Create a larger figure for better readability
fig, ax = plt.subplots(figsize=(18, 10))

# Define a colormap
colors = cm.get_cmap('viridis', len(grouped_data.columns))

# Create a stacked bar chart with sorted data and apply the colormap
grouped_data_percentage.plot(kind='bar', stacked=True, color=[colors(i) for i in range(len(grouped_data.columns))], ax=ax)

# Add labels and title with increased font size
plt.title('Conditions d\'éclairage lors des accidents routiers', fontsize=15)
plt.xlabel('Année', fontsize=12)
plt.ylabel('Pourcentage', fontsize=12)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Manually define legend labels with increased font size
legend_labels = [
    '1 – Plein jour',
    '2 – Crépuscule ou aube',
    '3 – Nuit sans éclairage public',
    '4 – Nuit avec éclairage public non allumé',
    '5 – Nuit avec éclairage public allumé'
]

ax.legend(legend_labels, title='Condition d\'éclairage', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.04, 1), loc='upper left')

# Adjust layout for tight fit and larger legend
plt.tight_layout(rect=[0, 0, 0.85, 1])



threshold = 0.1  # Seuil ajusté pour éviter d'afficher les pourcentages trop petits

# Calculer la position cumulée pour l'annotation des pourcentages
cumulative_heights = grouped_data_percentage.cumsum(axis=1)

# Annoter les pourcentages dans les barres
for i in range(len(grouped_data_percentage)):
    for j in range(len(grouped_data_percentage.columns)):
        value = grouped_data_percentage.iloc[i, j]
        if value > threshold:
            # Positionner le texte au milieu de la portion de la barre
            center = (cumulative_heights.iloc[i, j] - value) + (value / 2)
            # S'assurer que la valeur est supérieure à 0 avant d'ajouter l'étiquette
            if value > 0:
                ax.text(i, center, f'{value:.1f}%', ha='center', va='center', color='white', fontsize=7, fontweight='bold', rotation = 45)



# Save the figure
plt.savefig('/home/onyxia/work/projet-python/gagz/statap', dpi=300)"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

path="/home/onyxia/projet-python/"
file_name1="dataset_complet_part_1.csv"
file_name2="dataset_complet_part_2.csv"
df1= pd.read_csv(path+file_name1, sep=',',low_memory=False)
df2= pd.read_csv(path+file_name2, sep=',',low_memory=False)
df=pd.concat([df1,df2])
df['grav'] = df['grav'].replace({'1':1,'2':2,'3':3,'4':4})
df=df[df['grav']!='grav']

# Make sure 'Num_Acc' and 'lum' are present in the 'caracteristiques' files
df = df[['Num_Acc', 'lum', 'an','grav']]  # Keep only the necessary columns

# Ensure 'lum' column is string and strip whitespaces
df['lum'] = df['lum'].astype(str).str.strip()

# Normalize values in 'lum' column
df['lum'] = df['lum'].replace({"1.0": "1", "2.0": "2", "3.0": "3", "4.0": "4", "5.0": "5"})
legend_labels = ['1 : Plein jour','2 : Crépuscule ou aube','3 : Nuit sans éclairage public','4 : Nuit avec éclairage public non allumé','5 : Nuit avec éclairage public allumé']


for gr in (2,3,4):
    df2=df[df['grav']==gr]
    df2 = df2[df2['lum'].notna()]
    df2=df2[df2['lum']!='-1.0']
# Group data by year and lighting condition, then count occurrences
    print(df2['lum'].unique())
    grouped_data = df2.groupby(['an', 'lum']).size().unstack().fillna(0)
    grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100
    print(grouped_data.columns)


# Create a larger figure for better readability
    fig, ax = plt.subplots(figsize=(18, 10))

# Define a colormap
    colors =plt.get_cmap('viridis', len(grouped_data.columns))

# Create a stacked bar chart with sorted data and apply the colormap
    grouped_data_percentage.plot(kind='bar', stacked=True, color=[colors(i) for i in range(len(grouped_data.columns))], ax=ax)

# Add labels and title with increased font size
    plt.title('Conditions d\'éclairage lors des accidents routiers pour la gravité ' +str(gr), fontsize=15)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Pourcentage', fontsize=12)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

# Manually define legend labels with increased font size
    ax.legend(legend_labels, title='Condition d\'éclairage', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.04, 1), loc='upper left')

# Adjust layout for tight fit and larger legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])



    threshold = 0.1  # Seuil ajusté pour éviter d'afficher les pourcentages trop petits

# Calculer la position cumulée pour l'annotation des pourcentages
    cumulative_heights = grouped_data_percentage.cumsum(axis=1)

# Annoter les pourcentages dans les barres
    for i in range(len(grouped_data_percentage)):
        for j in range(len(grouped_data_percentage.columns)):
            value = grouped_data_percentage.iloc[i, j]
            if value > threshold:
            # Positionner le texte au milieu de la portion de la barre
                center = (cumulative_heights.iloc[i, j] - value) + (value / 2)
            # S'assurer que la valeur est supérieure à 0 avant d'ajouter l'étiquette
                if value > 0:
                    ax.text(i, center, f'{value:.1f}%', ha='center', va='center', color='white', fontsize=7, fontweight='bold', rotation = 45)



# Save the figure
    plt.savefig('/home/onyxia/projet-python/gagz/statap '+str(gr), dpi=300)











































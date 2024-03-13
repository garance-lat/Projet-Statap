import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path="/home/onyxia/projet-python/"
def dataset_per_year(year):
    file_name_characteristics = f"caracteristiques_{year}.csv"
    df_characteristics = pd.read_csv(path + file_name_characteristics, sep=',', low_memory=False)

    file_name_locations = f"lieux_{year}.csv"
    df_locations = pd.read_csv(path + file_name_locations, sep=',', low_memory=False)
    df = pd.merge(df_characteristics, df_locations, on='Num_Acc')

    file_name_users = f"usagers_{year}.csv"
    df_users = pd.read_csv(path + file_name_users, sep=',', low_memory=False)
    df_users = df_users[df_users['grav'] != 'grav']

    if int(year) < 2019:
        df_users['grav'] = df_users['grav'].replace({2: 4, 4: 2, '2': 4, '4': 2, '1': 1, '3': 3}).infer_objects(copy=False)
        df_users['grav'] = pd.to_numeric(df_users['grav'], errors='coerce')  # Convert to numeric
    if int(year)>2018:
        df_users['grav'] = df_users['grav'].replace({'4': 4, '2': 2, '1': 1, '3': 3}).infer_objects(copy=False)
        df_users['grav'] = df_users['grav'].replace({2: 4, 4: 2})
        df_users['grav'] = pd.to_numeric(df_users['grav'], errors='coerce')  # Convert to numeric

    grav_df = df_users.groupby('Num_Acc')['grav'].max().reset_index()
    df = pd.merge(df, grav_df, on='Num_Acc')
    df = df.drop_duplicates()

    return df
df=dataset_per_year(2011)
df['grav'] = df['grav'].replace({'1':1,'2':2,'3':3,'4':4})
df=df[df['grav']!='grav']


# Graphique selon le type de route

df = df[df['catr'] < 5]
df=df.sort_values(by='catr')
df['catr'] = df['catr'].replace({1: 'Autoroute', 3: 'Route Nationale',2:'Route Départementale',4:'Voie communale'})
graph_df = df.groupby(['grav', 'catr']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='catr', values='accident_count')
graph_df = graph_df.transpose()

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns, graph_df.loc[gravity_type], label=f'Gravity {gravity_type}')

ax=graph_df.plot(kind='bar', stacked=False, figsize=(10, 6))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.xlabel('Type de route')
plt.ylabel('Nombre d accidents')
plt.title('Nombre d accidents par type de route selon la gravité')
plt.legend()
plt.savefig(path+'partie II/vérifier 2010/'+'type de route 2011.png')
plt.show()

# En agglomération ou hors?

df['agg'] = df['agg'].astype(float)
df = df[df['agg'].notna()]
graph_df = df.groupby(['grav', 'agg']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='agg', values='accident_count')

graph_df = graph_df.transpose()
ax=graph_df.plot(kind='bar', stacked=True, figsize=(10, 6))

ax.set_xticks(range(len(graph_df.index)))
ax.set_xticklabels(['Hors agglomération', 'En agglomération'], rotation=0)
plt.ylabel('Nombre d accidents')
ax.set_xlabel('')
plt.title('Nombre d accidents selon si c était en agglomération ou non')
plt.legend()
plt.savefig(path+'partie II/vérifier 2010/'+'agg2011.png')
plt.show()


df=dataset_per_year(2010)
df['grav'] = df['grav'].replace({'1':1,'2':2,'3':3,'4':4})
df=df[df['grav']!='grav']

# Par mois

df['mois']=df['mois'].replace({'1':'01','2':'02','3':'03','4':'04','5':'05','6':'06','7':'07','8':'08','9':'09',1.0:'01',2.0:'02',3.0:'03',4.0:'04',5.0:'05',6.0:'06',7.0:'07',8.0:'08',9.0:'09',10.0:'10',11.0:'11',12.0:'12','1.0':'01','2.0':'02','3.0':'03','4.0':'04','5.0':'05','6.0':'06','7.0':'07','8.0':'08','9.0':'09','10.0':'10','11.0':'11','12.0':'12'})
#df=df[df['mois']!='mois']
graph_df = df.groupby(['grav', 'mois']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='mois', values='accident_count')

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns, graph_df.loc[gravity_type], label=f'Gravity {gravity_type}', marker='o', linestyle='-')

plt.xticks(df['mois'].unique().astype(str), ['Janvier', 'Février','Mars', 'Avril','Mai', 'Juin', 'Juillet', 'Août','Septembre','Octobre', 'Novembre', 'Décembre'], fontsize=8)
plt.ylabel('Nombre d accidents')
plt.xlabel('Mois')
plt.title('Nombre d accidents selon le mois en 2010')
plt.legend()
plt.savefig(path+'partie II/vérifier 2010/'+'mois 2011.png')
plt.show()


#Par heure

#df['hrmn'] = pd.to_numeric(df['hrmn'].str.replace(':', ''), errors='coerce')

graph_df = df.groupby(['grav', 'hrmn']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='hrmn', values='accident_count')

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns.astype(str).str.zfill(4).str[:2] + ':' + graph_df.columns.astype(str).str.zfill(4).str[2:],
             graph_df.loc[gravity_type], label=f'Gravity {gravity_type}')

plt.ylabel('Nombre d accidents')
plt.xlabel('Heure')
plt.title('Nombre d accidents selon l heure de la journée en 2011')


plt.xticks(rotation=45, ha='right')
num_columns = graph_df.shape[1]
specific_hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
tick_positions = [int(i*300*num_columns/2359)for i in range (8)]
plt.xticks(tick_positions, specific_hours, rotation=45, ha='right')


plt.legend()
plt.tight_layout() 
plt.savefig(path + 'partie II/vérifier 2010/' + 'heure 2011.png')
plt.show()









import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path="/home/onyxia/projet-python/"
file_name1="dataset_complet_part_1.csv"
file_name2="dataset_complet_part_2.csv"
df1= pd.read_csv(path+file_name1, sep=',',low_memory=False)
df2= pd.read_csv(path+file_name2, sep=',',low_memory=False)
df=pd.concat([df1,df2])
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
plt.savefig(path+'soline/'+'type de route.png')
plt.show()


#Graphique selon la vitesse max de la route - Useless car trop de valeurs non renseignées
"""
df = df[df['vma'].notna()]
allowed_values = ['30', '50', '70', '80', '90', '110', '130']
df.drop(df[~df['vma'].isin(allowed_values)].index, inplace=True)
df['vma']=df['vma'].replace({'30':30,'50':50,'70':70,'80':80,'90':90,'110':110,'130':130})


graph_df = df.groupby(['grav', 'vma']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='vma', values='accident_count')
graph_df = graph_df.transpose()

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns, graph_df.loc[gravity_type], label=f'Gravity {gravity_type}')

ax=graph_df.plot(kind='bar', stacked=False, figsize=(10, 6))

plt.xlabel('Vitesse max de la route')
plt.ylabel('Nombre d accidents')
plt.title('Nombre d accidents selon la vitesse max de la route ')
plt.legend()
plt.savefig(path+'soline/'+'vitesse max.png')
plt.show()"""

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
plt.savefig(path+'soline/'+'agg.png')
plt.show()

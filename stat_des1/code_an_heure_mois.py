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
"""
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
plt.title('Nombre d accidents selon le mois')
plt.legend()
plt.savefig(path+'capucine/'+'mois.png')
plt.show()



# Par an

df['an']=df['an'].replace({5.0:2005.0,6.0:2006.0,7.0:2007.0,8.0:2008.0,9.0:2009.0,10.0:2010.0,11.0:2011.0,12.0:2012.0,12.0:2012.0,13.0:2013.0,14.0:2014.0,15.0:2015.0,16.0:2016.0,17.0:2017.0,18.0:2018.0})

graph_df = df.groupby(['grav', 'an']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='an', values='accident_count')

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns, graph_df.loc[gravity_type], label=f'Gravity {gravity_type}', marker='o', linestyle='-')

plt.ylabel('Nombre d accidents')
plt.xlabel('An')
plt.title('Nombre d accidents selon l année')
plt.legend()
plt.savefig(path+'capucine/'+'an.png')
plt.show()"""


#Par heure

df['hrmn'] = pd.to_numeric(df['hrmn'].str.replace(':', ''), errors='coerce')

graph_df = df.groupby(['grav', 'hrmn']).size().reset_index(name='accident_count')
graph_df = graph_df.pivot(index='grav', columns='hrmn', values='accident_count')

plt.figure(figsize=(10, 6))
for gravity_type in graph_df.index:
    plt.plot(graph_df.columns.astype(str).str.zfill(4).str[:2] + ':' + graph_df.columns.astype(str).str.zfill(4).str[2:],
             graph_df.loc[gravity_type], label=f'Gravity {gravity_type}')

plt.ylabel('Nombre d accidents')
plt.xlabel('Heure')
plt.title('Nombre d accidents selon l heure de la journée')


plt.xticks(rotation=45, ha='right')
num_columns = graph_df.shape[1]
specific_hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
tick_positions = [int(i*300*num_columns/2359)for i in range (8)]
plt.xticks(tick_positions, specific_hours, rotation=45, ha='right')


plt.legend()
plt.tight_layout() 
plt.savefig(path + 'capucine/' + 'heure.png')
plt.show()






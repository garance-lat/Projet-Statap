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










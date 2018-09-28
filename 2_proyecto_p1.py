# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:34:40 2018

@author: Esteban Ortiz Tirado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import string
#%%
from eotg import eotg
#%%
data = pd.read_csv('../Data/datos_proyecto1.csv')

#%%repote de calidad de los datos
dqr= eotg.dqr(data)

#%%
Al_por_Carrera = pd.DataFrame(data.groupby(['Clave de la carrera'])['Matricula'].count())
Al_por_Plantel = pd.DataFrame(data.groupby(['Plantel'])['Matricula'].count())

#%% Eliminar columnas con valores unicos = 1
data = data.drop(columns=['Entidad','Periodo'])
#%% Normalizar texto
def remove_whitespaces(x):
    try:
        x = ''.join(x.split())
    except:
        pass
    return x
def remove_punctuation(x):
    try:
        x = ''.join(c for c in x if c not in string.punctuation)
    except:
        pass
    return x
data['Carrera Profesional Tecnico -Bachiller'] = data['Carrera Profesional Tecnico -Bachiller'].apply(remove_whitespaces).apply(remove_punctuation)
#%% Nuevo repote de calidad de los datos
dqr= eotg.dqr(data)

#%%
plt.figure(figsize=(12,6))
plt.bar(Al_por_Carrera.index,Al_por_Carrera.Matricula)
plt.xlabel('Carrera')
plt.ylabel('Numero de Alumnos')
plt.title('Alumnos Inscritos por Carrera')
plt.show()

plt.figure(figsize=(12,6))
plt.bar(Al_por_Plantel.index,Al_por_Plantel.Matricula)
plt.xlabel('Plantel')
plt.ylabel('Numero de Alumnos')
plt.title('Alumnos Inscritos por Plantel')
plt.show()





#%%Seleccionar variables categoricas, solo las columnas con tipo de valor int64
indx = np.array(data.dtypes=='object')
col_names = list(data.columns.values[indx])
data_obj = data[col_names]

#%%
mireporte2 = eotg.dqr(data_obj)

#%% Seleccionar columnas con menos valores unicos que 25, para esas convertirlas a dummy
indx = mireporte2.Unique_Values<25
col_names_unique = np.array(col_names)[indx]
data_obj_uni = data_obj[col_names_unique]

#%% Aplicar generacion de dummies a toda la tabla y juntarla en un solo DF
data_dummy = pd.get_dummies(data_obj_uni[col_names_unique[0]], prefix=col_names_unique[0]) #dummies del primer campo

for col in col_names_unique[1:]:
    tmp = pd.get_dummies(data_obj_uni[col], prefix=col) #genera los dummies de toda la lista
    data_dummy = data_dummy.join(tmp) #pegar las tablas temporales a data_dummy
del tmp #elimina tmp para no tener variables extras

#%% Aplicacion de indices de similitud
D1 = sc.pdist(data_dummy,'matching')
D1 = sc.squareform(D1)

#%% Clustering
from sklearn.cluster import KMeans
#%% Grafica de Codo
n_grupos=75
inercias = np.zeros(n_grupos)  
for k in np.arange(n_grupos)+1:
    model = KMeans(n_clusters=k,init='random')
    model = model.fit(data_dummy)
    inercias[k-1] = model.inertia_

plt.plot(np.arange(1,n_grupos+1),inercias)
plt.xlabel('Numero de grupos')
plt.ylabel('Inercia Global')
plt.show()

#%% Perfiles de alumnos, suponiendo que hay 5 diferentes tipos
model = KMeans(n_clusters=5,init='random')
model = model.fit(data_dummy)
centroides = model.cluster_centers_
plt.plot(centroides[:,0:53].transpose())
plt.title('Perfiles')

#%%
Ypredict = model.predict(data_dummy)
#%%
Perfiles = pd.DataFrame(columns=('Perfil','Matricula'))
Perfiles.Perfil = Ypredict
Perfiles.Matricula = data_dummy.index
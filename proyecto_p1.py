# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc

from myLib import mylib
import string
#%%definición de funciones 
def count_values(x,parameter):
    counter=0
    size=len(x)
    for i in range (size):
        if(x[i]==parameter):
            counter =counter+1  
    return counter 

#%%importacion de datos
 '../Data/datos_proyecto1.csv'
data = pd.read_csv( '../Data/datos_proyecto1.csv', encoding ='latin1')
data = data.iloc[:,0:7]

#%%repote de calidad de los datos
dqr= mylib.dqr(data)
#%%eliminación de datos nulos
data=data.dropna()
#%%lEstadísticas de los datos

plantel = (data['Plantel'])
modelo = data['Modelo educativo']
clave_c= data['Clave de la carrera']
carrea = data['Carrera Profesional Tecnico -Bachiller']
matricula = data['Matricula']
periodo = data['Periodo']
count_values(plantel,'Guadalajara II')



                 


 #%% conteo de valores únicos de cada columa del dataframe original
plantel_c=pd.value_counts(data.Plantel) 
modelo_c= pd.value_counts(data['Modelo educativo'])
clave_c= pd.value_counts(data['Clave de la carrera'])
carrera_c= pd.value_counts(data['Carrera Profesional Tecnico -Bachiller'])
matricula_C = pd.value_counts(data['Matricula'])
periodo_c = pd.value_counts(data['Periodo'])


modelo_C= pd.value_counts(data.Plantel)


    
    





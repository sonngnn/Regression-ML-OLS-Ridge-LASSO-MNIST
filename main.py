#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

# Fonctions utilisées dans le code
def l(u, v):
    return (u - v)**2

def r(beta0, beta):
    return 0

def f(x, beta0, beta):
    return beta0 + beta*x

def R(X_data, Y, beta0, beta):
    
    somme = 0
    
    for i in range(len(X_data)):
        somme += l(f(X_data[i], beta0, beta), Y[i]) + r(beta0, beta)
        
    return 1/len(X_data) * somme


#data = np.load('data1.npy')
data = np.load('data2.npy')

X_data = data[0,:]

X = np.ones((len(X_data), 2))

for i in range(len(X)):
    X[i, 1] = X_data[i]

Y = data[1,:]


# Calcul des coefficients optimaux beta0 et beta
X_T = np.transpose(X)
vect = np.linalg.inv(X_T @ X) @ X_T @ Y 

beta0 = vect[0]
beta = vect[1]


# Risque empirique
R = R(X_data, Y, beta0, beta)
print('Risque empirique obtenu:', R)


# Tracé de nos courbes et nuage de points
x = np.linspace(min(X_data), max(X_data), 100)
y = f(x, beta0, beta)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f')
plt.scatter(X_data, Y, color='red', label='Données (X, Y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression linéaire')
plt.legend()
plt.grid()
plt.show()


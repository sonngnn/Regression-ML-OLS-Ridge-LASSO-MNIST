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
    # Évaluation du polynôme en x
    return beta0 + np.dot(beta, np.array([x**i for i in range(1, len(beta) + 1)]))

def R(X_data, Y, beta0, beta):
    # Calcul du risque empirique
    somme = 0
    for i in range(len(X_data)):
        somme += l(f(X_data[i], beta0, beta), Y[i]) + r(beta0, beta)
    return 1 / len(X_data) * somme

# Charger les données
data = np.load('data2.npy')
X_data = data[0, :]
Y = data[1, :]

# Ajuster l'ordre du modèle
q = 10  # Choisir ici l'ordre du modèle polynomial
X_phi = np.ones((len(X_data), q + 1))  # Matrice avec des colonnes pour chaque puissance de X

for i in range(1, q + 1):
    X_phi[:, i] = X_data**i
    

# Calcul des coefficients optimaux beta0 et beta
X_phi_T = np.transpose(X_phi)
n = len(X_phi)
lambd = 15  # Paramètre 
I = np.eye(X_phi_T.shape[0])

vect = np.linalg.inv(X_phi_T @ X_phi + n * lambd * I) @ X_phi_T @ Y

beta0 = vect[0]
beta = vect[1:]

# Calcul du risque empirique
empirical_risk = R(X_data, Y, beta0, beta)
print('Risque empirique obtenu:', empirical_risk)

# Tracé de nos courbes et nuage de points
x = np.linspace(min(X_data), max(X_data), 100)
y = np.array([f(val, beta0, beta) for val in x])

plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'Modèle polynomial d\'ordre {q}')
plt.scatter(X_data, Y, color='red', label='Données (X, Y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Régression polynomial d\'ordre {q}')
plt.legend()
plt.grid()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge

# Fonctions pour calculer le modèle linéaire et les erreurs
def f(x, beta0, beta):
    return beta0 + np.dot(beta, x)

def R(X_data, Y, beta0, beta):
    somme = 0
    for i in range(len(X_data)):
        somme += (f(X_data[i], beta0, beta) - Y[i])**2
    return 1 / len(X_data) * somme

# Charger les données
data = np.load('data3.npy')
X_data = data[0, :]
Y = data[1, :]

# Ajouter une colonne de biais (X0 = 1) pour OLS et Ridge
X = np.ones((len(X_data), 2))
X[:, 1] = X_data

# OLS
X_T = np.transpose(X)
vect_OLS = np.linalg.inv(X_T @ X) @ X_T @ Y
beta0_OLS = vect_OLS[0]
beta_OLS = vect_OLS[1:]
error_OLS = R(X[:, 1:], Y, beta0_OLS, beta_OLS)

# Ridge
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X[:, 1:], Y)
beta0_ridge = ridge.intercept_
beta_ridge = ridge.coef_
error_ridge = R(X[:, 1:], Y, beta0_ridge, beta_ridge)

# LASSO
lasso = Lasso(alpha=0.1, fit_intercept=True, max_iter=10000)
lasso.fit(X[:, 1:], Y)
beta0_lasso = lasso.intercept_
beta_lasso = lasso.coef_
error_lasso = R(X[:, 1:], Y, beta0_lasso, beta_lasso)

# Affichage des résultats
x_plot = np.linspace(min(X_data), max(X_data), 100)
y_OLS = beta0_OLS + beta_OLS * x_plot
y_ridge = beta0_ridge + beta_ridge * x_plot
y_lasso = beta0_lasso + beta_lasso * x_plot

plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y, color='red', label='Données (X, Y)')
plt.plot(x_plot, y_OLS, label='OLS', linestyle='--')
plt.plot(x_plot, y_ridge, label='Ridge', linestyle='-.')
plt.plot(x_plot, y_lasso, label='LASSO', linestyle=':')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparaison des modèles OLS, Ridge et LASSO')
plt.legend()
plt.grid()
plt.show()

print("Erreurs d'apprentissage:")
print(f"OLS: {error_OLS}")
print(f"Ridge: {error_ridge}")
print(f"LASSO: {error_lasso}")
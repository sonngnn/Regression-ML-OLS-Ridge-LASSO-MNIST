#%% Code de base
###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554)
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 
###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=4
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  plt.title("Label = %i" % lb[i])


#%% Question 1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du classifieur par régression logistique avec régularisation ℓ2
log_reg = LogisticRegression(
    penalty='l2',        # Régularisation ℓ2
    multi_class='multinomial', 
    solver='saga',       
    max_iter=1000,       # Plus d'itérations pour convergence
    random_state=0,
    verbose=1
)
X_train_small = X_train_scaled[:500]
y_train_small = y_train[:500]
log_reg.fit(X_train_small, y_train_small)

# Prédictions sur les données de test
y_pred = log_reg.predict(X_test_scaled)

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Matrice de confusion - Régression logistique avec régularisation ℓ2")
plt.show()

#%% Question 2
# Dimensions de l'image (28x28 pixels pour MNIST)
image_shape = (28, 28)

# Coefficients du modèle (β)
coefficients = log_reg.coef_

# Création des images pour chaque classe
plt.figure(figsize=(12, 12))
for i in range(coefficients.shape[0]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(coefficients[i].reshape(image_shape), cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title(f"Classe {i}")
    plt.axis('off')

plt.suptitle("Visualisation des coefficients β pour chaque classe", fontsize=16)
plt.tight_layout()
plt.show()

#%% Question 3
# Réentraîner avec régularisation ℓ1
log_reg_l1 = LogisticRegression(
    penalty='l1',        # Régularisation ℓ1
    multi_class='multinomial',  
    solver='saga',       
    max_iter=1000,       # Plus d'itérations pour convergence
    random_state=0
)
X_train_small = X_train_scaled[:500]
y_train_small = y_train[:500]
log_reg_l1.fit(X_train_small, y_train_small)

# Coefficients pour ℓ1
coefficients_l1 = log_reg_l1.coef_

# Comparer avec ℓ2
plt.figure(figsize=(15, 10))
classes = np.arange(10)

for i, cls in enumerate(classes):
    # ℓ2 coefficients
    plt.subplot(2, 10, i + 1)
    plt.imshow(coefficients[cls].reshape(28, 28), cmap='RdBu', aspect='auto')
    plt.title(f"Classe {cls} (ℓ2)")
    plt.axis('off')
    
    # ℓ1 coefficients
    plt.subplot(2, 10, i + 11)
    plt.imshow(coefficients_l1[cls].reshape(28, 28), cmap='RdBu', aspect='auto')
    plt.title(f"Classe {cls} (ℓ1)")
    plt.axis('off')

plt.suptitle("Comparaison des coefficients ℓ2 vs ℓ1")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


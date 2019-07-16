import torch
from torch.nn import Linear, NLLLoss, LogSoftmax, Sequential
from torch.optim import Adam
from scipy.io import wavfile
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from kymatio import Scattering1D
from kymatio.datasets import fetch_fsdd

# Configuration du pipeline
# La longueur du signal
T = 2**13
J = 8
Q = 12

# Petite constante à ajouter aux coefficients de diffusion avant de calculer le logarithme. 
# Ceci évite des valeurs très grandes lorsque les coefficients de diffusion sont très proches de zéro.
log_eps = 1e-6

use_cuda = torch.cuda.is_available()
torch.manual_seed(42)

# Chargement des données
info_data = fetch_fsdd()
files = info_data['files']
path_dataset = info_data['path_dataset']

# Les signaux audio ( x_all ), les étiquettes ( y_all ) et déterminent si le signal est dans le train ou dans l'ensemble de test ( sous-ensemble ).
x_all = torch.zeros(len(files), T, dtype=torch.float32)
y_all = torch.zeros(len(files), dtype=torch.int64)
subset = torch.zeros(len(files), dtype=torch.int64)

for k, f in enumerate(files):
    basename = f.split('.')[0]

    # Get label (0-9) of recording.
    y = int(basename.split('_')[0])

    # Index larger than 5 gets assigned to training set.
    if int(basename.split('_')[2]) >= 5:
        subset[k] = 0
    else:
        subset[k] = 1

    # Load the audio signal and normalize it.
    _, x = wavfile.read(os.path.join(path_dataset, f))
    x = np.asarray(x, dtype='float')
    x /= np.max(np.abs(x))

    # Convert from NumPy array to PyTorch Tensor.
    x = torch.from_numpy(x)

    # If it's too long, truncate it.
    if x.numel() > T:
        x = x[:T]

    # If it's too short, zero-pad it.
    start = (T - x.numel()) // 2

    x_all[k,start:start + x.numel()] = x
    y_all[k] = y
    
# Transformation de diffusion de journal
scattering = Scattering1D(J, T, Q)

if use_cuda:
    scattering.cuda()
    x_all = x_all.cuda()
    y_all = y_all.cuda()
    
# Calculer la transformée de diffusion pour tous les signaux du jeu de données.    
Sx_all = scattering.forward(x_all)    
    
# Supprimer les coefficients de diffusion d'ordre zéro, qui sont toujours placés dans le premier canal du tenseur de diffusion.    
Sx_all = Sx_all[:,1:,:]    
    
Sx_all = torch.log(torch.abs(Sx_all) + log_eps)
  
# Effectuer une moyenne sur la dernière dimension (le temps) pour obtenir une représentation invariante par décalage dans le temps.  
Sx_all = torch.mean(Sx_all, dim=-1)  
  
# Former le classificateur  
# Extraire les données d'apprentissage (celles pour lesquelles le sous-ensemble = 0 ) et les étiquettes associées.  
Sx_tr, y_tr = Sx_all[subset == 0], y_all[subset == 0]  

# Normaliser les données pour avoir un zéro moyen et une variance d'unité.
mu_tr = Sx_tr.mean(dim=0)
std_tr = Sx_tr.std(dim=0)
Sx_tr = (Sx_tr - mu_tr) / std_tr

# Définir un modèle de régression logistique à l'aide de PyTorch, le former en utilisant Adam avec une perte de log-vraisemblance négative.
num_input = Sx_tr.shape[-1]
num_classes = y_tr.cpu().unique().numel()
model = Sequential(Linear(num_input, num_classes), LogSoftmax(dim=1))
optimizer = Adam(model.parameters())
criterion = NLLLoss()
  
if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# Paramètres pour la procédure d'optimisation
# Number of signals to use in each gradient descent step (batch).
batch_size = 32
# Number of epochs.
num_epochs = 50
# Learning rate for Adam.
lr = 1e-4    
    
# Nombre total de lots.
nsamples = Sx_tr.shape[0]
nbatches = nsamples // batch_size

for e in range(num_epochs):
    # Randomly permute the data. If necessary, transfer the permutation to the GPU.
    perm = torch.randperm(nsamples)
    if use_cuda:
        perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        model.zero_grad()
        resp = model.forward(Sx_tr[idx])
        loss = criterion(resp, y_tr[idx])
        loss.backward()
        optimizer.step()

    # Calculate the response of the training data at the end of this epoch and the average loss.
    resp = model.forward(Sx_tr)
    avg_loss = criterion(resp, y_tr)

    # Try predicting the classes of the signals in the training set and compute the accuracy.
    y_hat = resp.argmax(dim=1)
    accuracy = (y_tr == y_hat).float().mean()

    print('Epoch {}, average loss = {:1.3f}, accuracy = {:1.3f}'.format(e, avg_loss, accuracy))

    
# Tester le réseau
# Extraire les données de test (celles pour lesquelles le sous-ensemble = 1 ) et les étiquettes associées.
Sx_te, y_te = Sx_all[subset == 1], y_all[subset == 1]

# Utiliser la moyenne et l'écart type calculés sur les données d'apprentissage pour normaliser les données de test.
Sx_te = (Sx_te - mu_tr) / std_tr

# Calculer la réponse du classificateur sur les données de test et la perte résultante.
resp = model.forward(Sx_te)
avg_loss = criterion(resp, y_te)

# Try predicting the labels of the signals in the test data and compute the accuracy.
y_hat = resp.argmax(dim=1)
accu = (y_te == y_hat).float().mean()

print('TEST, average loss = {:1.3f}, accuracy = {:1.3f}'.format(avg_loss, accu))

# Traçage de la précision de la classification sous forme de matrice de confusion
predicted_categories = y_hat.cpu().numpy()
actual_categories = y_te.cpu().numpy()

confusion = confusion_matrix(actual_categories, predicted_categories)
plt.figure()
plt.imshow(confusion)
tick_locs = np.arange(10)
ticks = ['{}'.format(i) for i in range(1, 11)]
plt.xticks(tick_locs, ticks)
plt.yticks(tick_locs, ticks)
plt.ylabel("True number")
plt.xlabel("Predicted number")
plt.show()
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV para carregar a imagem
import os

def calcula_lbp(imagem, P, R, metodo):
    # Calcular o LBP
    lbp = local_binary_pattern(imagem, P, R, metodo)

    # Calcular o histograma LBP
    n_bins = int(lbp.max() + 1)
    hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist
  

def kfoldcv(indices, k):
    
    size = len(indices)
    subset_size = round(size / k)
    
    subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test: 
                train.append(subset)
        kfolds.append((train,test))
        
    return kfolds


# Parâmetros LBP
P = 8  # Número de pontos vizinhos
R = 2  # Raio
metodo = 'nri_uniform'

# Diretórios
diretorios = ['base/fold1', 'base/fold2', 'base/fold3']

hist = []
tag = [1,2,3,4,5,6,7,8,9,10]

for diretorio in diretorios:
    lista_arquivos = os.listdir(diretorio)
    for arquivo in lista_arquivos:
        caminho_imagem = os.path.join(diretorio, arquivo)  # Construir caminho completo
        # Carregar a imagem em escala de cinza
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        hist.append(calcula_lbp(imagem, P, R, metodo))


results = kfoldcv(hist, 3)

print(len(hist))
print(len(results))

from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV para carregar a imagem

# Carregar a imagem em escala de cinza
caminho_imagem = 'base/0001.png'
imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

# Parâmetros LBP
P = 8  # Número de pontos vizinhos
R = 2  # Raio
metodo = 'nri_uniform'

# Calcular o LBP
lbp = local_binary_pattern(imagem, P, R, metodo)

# Calcular o histograma LBP
n_bins = int(lbp.max() + 1)
hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

# Visualizar a imagem original e o LBP
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.imshow(imagem, cmap='gray')
ax1.set_title('Imagem Original')
ax1.axis('off')

ax2.imshow(lbp, cmap='gray')
ax2.set_title('LBP')
ax2.axis('off')

# Visualizar o histograma
ax3.bar(bins[:-1], hist, width=1)
ax3.set_xlim(0, n_bins)
ax3.set_title('Histograma LBP')
ax3.set_xlabel('Valores LBP')
ax3.set_ylabel('Frequência Normalizada')

plt.tight_layout()
plt.show()

# # Criar uma nova figura apenas para o histograma
# fig_hist, ax_hist = plt.subplots()
# ax_hist.bar(bins[:-1], hist, width=1)
# ax_hist.set_xlim(0, n_bins)
# ax_hist.set_title('Histograma LBP')
# ax_hist.set_xlabel('Valores LBP')
# ax_hist.set_ylabel('Frequência Normalizada')

# # Salvar o histograma
# plt.savefig('histograma.png', format='png')
# plt.close(fig_hist)

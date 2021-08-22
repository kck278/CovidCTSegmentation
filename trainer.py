import matplotlib.pyplot as plt
import numpy as np
from skimage import io

img = io.imread('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/lung/lung_001.png', as_gray=True)
a = np.asarray(img)
print(a.shape)
plt.imshow(a, interpolation='nearest', cmap='gray')
plt.show()

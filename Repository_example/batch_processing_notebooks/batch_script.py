import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flim_image_processing.module import sum
import matplotlib.pyplot as plt
import numpy as np

print(sum(1, 2))
plt.imshow(np.random.rand(10, 10))
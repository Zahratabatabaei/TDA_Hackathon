import numpy as np
import gudhi as gd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import tifffile as tiff
from tifffile import TiffFile
import os
import pandas as pd
from ripser import ripser
from persim import plot_diagrams,PersistenceImager
# import barcode
import persim
np.random.seed(42)

#%%

file_path = r"D:\Delifood\Milk\codes\Images\Mayonnaise_CLSM_RT_001.tif"
images = imread(file_path)
images_data = images

Train_set = []
for i in range(0,20):
    min_val = np.min(images_data[i])
    max_val = np.max(images_data[i])
    image_uint8 = ((images_data[i]- min_val) / (max_val - min_val) * 255).astype(np.uint8)
    Train_set.append(image_uint8)
    # Display the image
    # plt.imshow(images_data[i], cmap='gray')
    # plt.title(f'image_{[i]}')
    # plt.show()
    # print(i)    

#%%
H1_all_train = [] 
H0_train_all = []
p = 0
for image in Train_set:
  
    cubical_complex = gd.CubicalComplex(
        dimensions=image.shape, 
        top_dimensional_cells=image.flatten()
    )
    
    # Compute persistent homology
    persistence = cubical_complex.persistence()
    # Extract H0  and H1 
    # H0_train = cubical_complex.persistence_intervals_in_dimension(0)
    H1_train = cubical_complex.persistence_intervals_in_dimension(1)
    H1_all_train. append(H1_train)
    # H0_train_all.append(H0_train)
    # import pdb; pdb.set_trace()
    # Plot persistence diagram
    gd.plot_persistence_diagram(persistence)
    p += 1
    plt.title(f"Persistence Diagram image_{p}")
    plt.show()
#%% 
def betti_curve(persistence_intervals, resolution=100):
    
    # Extract birth and death times
    birth, death = persistence_intervals[:, 0], persistence_intervals[:, 1]
    death[death == np.inf] = max(birth) + 1  # Handle infinite persistence 
    """Treat infinitely persistent features as alive until just beyond the last birth."""

    # filtration values
    x_vals = np.linspace(min(birth), max(death), resolution)
    y_vals = np.zeros_like(x_vals)

    for b, d in zip(birth, death):
        y_vals += (x_vals >= b) & (x_vals < d)  # Count active features

    return x_vals, y_vals

betti_curves_train = []
for i in range(len(Train_set)):
    x_betti, y_betti = betti_curve(H1_all_train[i], resolution=50)
    #You can play with the resolution:
        # Depends on the data we can choose the resolution amount but as a hint:
            # Quick visualization	50–100
            # Smooth plots	100–300
            # ML / feature extraction (50-100)

            
            
    # plt.figure(figsize=(6, 3))
    # plt.plot(x_betti, y_betti, label=f'Betti Curve (H1) - Image {i}')
    # plt.title("Betti Curve from Cubical Complex")
    # plt.xlabel("Filtration Value")
    # plt.ylabel("Betti Number")
    # plt.legend()
    # plt.show()
    betti_curves_train.append(y_betti)



#%% Plotting the curves


betti_matrix = np.array(betti_curves_train)

# mean and std
mean_betti = np.mean(betti_matrix, axis=0)
std_betti = np.std(betti_matrix, axis=0)


plt.figure(figsize=(8, 4))
plt.plot(x_betti, mean_betti, label='Mean Betti Curve (H1)', color='blue')
plt.fill_between(x_betti, mean_betti - std_betti, mean_betti + std_betti,
                  alpha=0.3, color='blue', label='±1 Std Dev')
plt.xlabel("Filtration Value")
plt.ylabel("Average Betti Number")
plt.title(f"Average Betti Curve over Dataset (H1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


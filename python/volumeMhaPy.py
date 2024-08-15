import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the image data
imagePath = '/home/tunok/Work/mcDataIO_main/tests/output/doseTest.mha'
itkImage = sitk.ReadImage(imagePath)
npImage = sitk.GetArrayFromImage(itkImage)

# Normalize the image data
npImage = npImage.astype(np.float32)  # Ensure floating point for division
max_dose = np.max(npImage)
if max_dose != 0:
    npImage /= max_dose

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinate arrays for 3D plotting
x, y, z = np.meshgrid(np.arange(npImage.shape[1]),
                      np.arange(npImage.shape[0]),
                      np.arange(npImage.shape[2]))

# Plotting using scatter
ax.scatter(x, y, z, c=npImage.flatten(), cmap='gray_r', marker='o', alpha=0.1)

# Setting the viewing angle (adjust as necessary)
ax.view_init(elev=30, azim=30)

# Add color bar and labels
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Normalized Dose')
ax.set_xlabel('X Dimension')
ax.set_ylabel('Y Dimension')
ax.set_zlabel('Z Dimension')

# Show plot
plt.show()

# Save the plot to a file
# Specify the path and file name
plt.savefig('/home/tunok/Work/mcDataIO_main/tests/output/dose.png')
plt.close()

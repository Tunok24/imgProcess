import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def apply_window_level(np_image, window, level):
    """
    Adjust the image contrast using the specified window and level.

    Parameters:
    - np_image: NumPy array of the CT image.
    - window: The width of the intensity window.
    - level: The center of the intensity window.

    Returns:
    - Adjusted NumPy array with values scaled to [0, 255].
    """
    lower_bound = level - window / 2
    upper_bound = level + window / 2
    windowed_image = np.clip(np_image, lower_bound, upper_bound)
    windowed_image = np.interp(
        windowed_image, (lower_bound, upper_bound), (0, 255))
    return windowed_image.astype(np.uint8)


def noiseImage(image, k):
    """
    Calculate the local noise image using a local standard deviation calculation.

    Parameters:
        image: 2D numpy array representing the image slice.
        k: The radius of the neighborhood used for calculating the standard deviation.

    Returns:
        2D numpy array representing the local noise image.
    """
    # Convert the numpy array to a SimpleITK image
    sitkImage = sitk.GetImageFromArray(image)

    # Use Gaussian smoothing to approximate local mean
    meanFilter = sitk.SmoothingRecursiveGaussianImageFilter()
    meanFilter.SetSigma(k)
    meanImage = meanFilter.Execute(sitkImage)

    # Calculate the squared image
    squaredImage = sitk.Square(sitkImage)

    # Calculate mean of squared image
    meanSquaredImage = meanFilter.Execute(squaredImage)

    # Calculate the standard deviation image: sqrt(E[X^2] - (E[X])^2)
    stdDevImage = sitk.Sqrt(meanSquaredImage - sitk.Square(meanImage))

    # Convert the SimpleITK image back to a numpy array
    noiseNumpyImage = sitk.GetArrayFromImage(stdDevImage)

    return noiseNumpyImage

# Function to save an image or plot


def save_figure(path, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()
    print(f"Saved image as: {path}")


# READ IN IMAGE IN ITK_IMAGE
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstTest/totalSlices.mha'
outputPath = '/home/tunok/Work/imgProcess_main/tests/images/'
itkImage = sitk.ReadImage(imagePath)
npImage = sitk.GetArrayFromImage(itkImage)

npImageA = sitk.GetArrayFromImage(sitk.ReadImage(
    '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/floodSlices.mha'))
npImageB = sitk.GetArrayFromImage(sitk.ReadImage(
    '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/totalSlices.mha'))
npImageC = sitk.GetArrayFromImage(sitk.ReadImage(
    '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/scatterSlices.mha'))
npImageD = sitk.GetArrayFromImage(sitk.ReadImage(
    '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/tertiarySlices.mha'))

###################################################################
############# Check image shape directly from npImage #############
# print("Shape of npImage:", npImage.shape)

# with open(imagePath, 'rb') as file:
#     for _ in range(10):
#         line = file.readline().decode('latin1')
#         print(line.strip())
#         if line.strip() == "ElementDataFile = LOCAL":
#             break
############# Check image shape directly from npImage #############
###################################################################

# SELECT IMAGE SLICE, COORDINATES
sliceIndex = 0  # for isoharold prost reconstructed volume, take slice 55
rowIndex = int(npImage.shape[1]/2)
colIndex = int(npImage.shape[2]/2)

# SHOW IMAGE SLICE
# NPIMAGE(SLICE, ROW, COLUMN)
imageSlice = npImage[sliceIndex, :, :]

volumeSlice = npImage[:, sliceIndex, :]

#########################################################################################
#################################### For Dose Volume ####################################
# # Flatten the 3D array to a 1D array
# flattened_array = npImage.flatten()

# sizeX = npImage.shape[2]
# sizeY = npImage.shape[1]
# sizeZ = npImage.shape[0]

# print("Shape of npImage x:", sizeX)
# print("Shape of npImage y:", sizeY)
# print("Shape of npImage z:", sizeZ)

# # Reshape the flattened array back to 3D array with new dimensions - this is for Dose Volume
# reshaped_array = flattened_array.reshape(sizeZ, sizeY, sizeX)
# # doseSlice = npImage[:, :, sliceIndex]

# # Access a specific slice as needed
# doseSlice = reshaped_array[:, :, sliceIndex]
#################################### For Dose Volume ####################################
#########################################################################################

# Adjust window/level values
window = 0.1  # Example window value
level = 0.01  # Example level value
adjustedSlice = apply_window_level(imageSlice, window, level)

print("Extracting and saving volume/image slice")
img = plt.imshow(imageSlice, cmap='gray')
plt.title(f'Slice Index: {sliceIndex}')
plt.colorbar(img, label='Signal Level')  # Add a color bar with a label
plt.axis('off')
plt.xlabel('X Label')
plt.ylabel('Y Label')
# Draw a horizontal line on the nth row in blue color
plt.axhline(y=rowIndex, color='orange', linestyle='dashdot')
# Draw a vertical line on the mth column in blue color
plt.axvline(x=colIndex, color='b', linestyle='dashdot')
plt.savefig(outputPath + 'totalBoostedImage.png')
plt.close()
print('Saved image as:' + outputPath + 'totalBoostedImage.png')


# PLOTTING A ROW OR A COLUMN
rowDataA = npImageA[sliceIndex, rowIndex, :]
rowDataB = npImageB[sliceIndex, rowIndex, :]
rowDataC = npImageC[sliceIndex, rowIndex, :]
rowDataD = npImageD[sliceIndex, rowIndex, :]

# Determine the min and max intensity across all profiles for consistent y-axis scaling
min_intensity = min(np.min(rowDataA), np.min(rowDataB),
                    np.min(rowDataC), np.min(rowDataD))
max_intensity = max(np.max(rowDataA), np.max(rowDataB),
                    np.max(rowDataC), np.max(rowDataD))

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(rowDataA, color='b', label='Flood')
plt.plot(rowDataB, color='r', label='Total')
plt.plot(rowDataC, color='g', label='Scatter')
plt.plot(rowDataD, color='m', label='Tertiary')

# Add title, labels, legend, and set the y-axis limit
plt.title(f'Row Data at Slice {sliceIndex}, Column {colIndex}')
plt.xlabel('Col Index')
plt.ylabel('Intensity Value')
plt.ylim(min_intensity, max_intensity)
plt.legend()

# Save the plot to a file
plt.savefig(outputPath + 'rowData_comparison.png')
plt.close()
print('Saved image as:' + outputPath + 'rowData_comparison.png')


# PLOTTING A COLUMN FROM MULTIPLE IMAGES
colDataA = npImageA[sliceIndex, :, colIndex]
colDataB = npImageB[sliceIndex, :, colIndex]
colDataC = npImageC[sliceIndex, :, colIndex]
colDataD = npImageD[sliceIndex, :, colIndex]

# Determine the min and max intensity across all profiles for consistent y-axis scaling
min_intensity = min(np.min(colDataA), np.min(colDataB),
                    np.min(colDataC), np.min(colDataD))
max_intensity = max(np.max(colDataA), np.max(colDataB),
                    np.max(colDataC), np.max(colDataD))

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(colDataA, color='b', label='Flood')
plt.plot(colDataB, color='r', label='Total')
plt.plot(colDataC, color='g', label='Scatter')
plt.plot(colDataD, color='m', label='Tertiary')

# Add title, labels, legend, and set the y-axis limit
plt.title(f'Column Data at Slice {sliceIndex}, Column {colIndex}')
plt.xlabel('Row Index')
plt.ylabel('Intensity Value')
plt.ylim(min_intensity, max_intensity)
plt.legend()

# Save the plot to a file
plt.savefig(outputPath + 'colData_comparison.png')
plt.close()
print('Saved image as:' + outputPath + 'colData_comparison.png')

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import imageio


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
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstNew/primaryBoostedImage.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProst/totalBoostedImage.mha'
imagePath = '/home/tunok/Work/mcDataIO_main/tests/output/IsoHaroldProstUniform/totalBoostedImage300000000Projection9_2.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstUniform/totalBoostedImage.mha'
imagePath = '/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage20000000Projection1.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidNew/tertiaryUnboostedImage.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstTest/totalBoostedImage.mha'
imagePath = '/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage20000000Projection1.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/scatterEst.mha'
imagePath = '/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/totalBoostedVolume.mha'
itkImage = sitk.ReadImage(imagePath)
npImage = sitk.GetArrayFromImage(itkImage)

with open(imagePath, 'rb') as file:
    for _ in range(10):
        line = file.readline().decode('latin1')
        print(line.strip())
        if line.strip() == "ElementDataFile = LOCAL":
            break

# SELECT IMAGE SLICE, COORDINATES
sliceIndex = 100  # for isoharold prost reconstructed volume, take slice 55
rowIndex = int(npImage.shape[1]/2)
rowRange = 70
colIndex = int(npImage.shape[2]*3/4)
colRange = 50

# SHOW IMAGE SLICE
# NPIMAGE(SLICE, ROW, COLUMN)
imageSlice = npImage[sliceIndex, :, :]
volumeSlice = npImage[:, sliceIndex, :]


# Adjust window/level values
window = 0.1  # Example window value
level = 0.01  # Example level value
adjustedSlice = apply_window_level(volumeSlice, window, level)

print("Extracting and saving volume/image slice")
img = plt.imshow(volumeSlice, cmap='gray')
plt.title(f'Slice Index: {sliceIndex}')
plt.colorbar(img, label='Signal Level')  # Add a color bar with a label
plt.axis('off')
plt.xlabel('X Label')
plt.ylabel('Y Label')
# Draw a horizontal line on the nth row in blue color
# plt.axhline(y=rowIndex, color='orange', linestyle='dashdot')
# plt.axhline(y=rowIndex+rowRange, color='b', linestyle='dashdot')
# Draw a vertical line on the mth column in blue color
# plt.axvline(x=colIndex, color='b', linestyle='dashdot')
# plt.axvline(x=colIndex+colRange, color='b', linestyle='dashdot')

plt.savefig('/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/totalBoostedVolume.png')
plt.close()
print("Saved image as: ... /Examples/simulate_and_reconstruct/totalBoostedVolume.png")

# PLOTTING A ROW OR A COLUMN
rowData = npImage[sliceIndex, rowIndex, :]
min_intensity = np.min(rowData)
max_intensity = np.max(rowData)
# Now plot the row data
plt.plot(rowData, color='orange')
plt.title(f'Row Data at Slice {sliceIndex}, Row {rowIndex}')
plt.xlabel('Col Index')
plt.ylabel('Intensity Value')
plt.ylim(min_intensity, max_intensity)  # Set the y-axis range
plt.savefig(
    '/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/rowData.png')
plt.close()
print("Saved image as: ... /Examples/simulate_and_reconstruct/rowData.png")

# PLOTTING A ROW OR A COLUMN
colData = npImage[sliceIndex, :, colIndex]
min_intensity = np.min(colData)
max_intensity = np.max(colData)
# Now plot the row data
plt.plot(colData, color='b')
plt.title(f'Col Data at Slice {sliceIndex}, Col {colIndex}')
plt.xlabel('Row Index')
plt.ylabel('Intensity Value')
plt.ylim(min_intensity, max_intensity)  # Set the y-axis range
plt.savefig(
    '/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/colData.png')
plt.close()
print("Saved image as: ... /Examples/simulate_and_reconstruct/colData.png")


# Compute Local Noise
k = 2   # Window size
local_noise = noiseImage(volumeSlice, k)
plt.figure(figsize=(10, 8))  # Optional: adjust the figure size as needed
# Save the AxesImage returned by imshow
img = plt.imshow(local_noise, cmap='viridis')
plt.title(f'Slice Index: {sliceIndex}')
plt.colorbar(img, label='Noise Level')  # Add a color bar with a label
plt.axis('off')  # Turn off the axis numbers and ticks
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.savefig(
    '/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/Noise.png')
plt.close()
print("Saved image as: ... /Examples/simulate_and_reconstruct/Noise.png")


# # Calculate the average value of the image slice
# averageValue = np.mean(imageSlice)
# print("Average value of the image slice:", averageValue)

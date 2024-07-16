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


# Read and Load the .mha image file and convert in to numpy array.
imagePath = '/home/tunok/Work/mcDataIO_main/tests/output/IsoHaroldProst/totalBoostedImage250000000Projection9.mha'
imagePath = '/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidNew/totalBoostedImage.mha'
itkImage = sitk.ReadImage(imagePath)
npImage = sitk.GetArrayFromImage(itkImage)

# Look at the first 13 line i.e. the header of the image file
with open(imagePath, 'rb') as file:
    for _ in range(13):
        line = file.readline().decode('latin1')
        print(line.strip())
        if line.strip() == "ElementDataFile = LOCAL":
            break

# SELECT IMAGE SLICE, COORDINATES
sliceIndex = 100
rowIndex = int(npImage.shape[1]/2)
rowRange = 70
colIndex = int(npImage.shape[2]/2)
colRange = 50

# Get image slice
imageSlice = npImage[sliceIndex, :, :]  # for projection
volumeSlice = npImage[:, sliceIndex, :]  # for reconstructed volumes


# Adjust window/level values
window = 0.1  # Example window value
level = 0.01  # Example level value
adjustedSlice = apply_window_level(volumeSlice, window, level)

# Apply Fourier Transform
f_transform = np.fft.fftshift(np.fft.fft2(imageSlice))
f_magnitude = np.abs(f_transform)

# Log to enhance visibility of features
f_log_magnitude = np.log(f_magnitude + 1)

# Use a smaller constant with log1p for potentially better detail visibility
f_log_magnitude = np.log1p(f_magnitude)

# Normalize the Fourier magnitude
normalized_f_magnitude = f_magnitude / np.max(f_magnitude)

# Apply logarithmic scaling to the normalized magnitude
f_log_magnitude = np.log1p(normalized_f_magnitude)


# Plotting
plt.figure(figsize=(12, 6))  # Set the figure size (width, height in inches)

# Subplot 1: Original Image
plt.subplot(1, 2, 1)  # (rows, columns, panel number)
plt.imshow(imageSlice, cmap='gray')
plt.title('Original Image')
plt.colorbar()

# Subplot 2: Fourier Transformed Image
plt.subplot(1, 2, 2)
plt.imshow(f_log_magnitude, cmap='viridis')
plt.title('Fourier Transform Magnitude Spectrum')
# Adjust the upper limit to 10% of the max value
plt.clim(0, f_log_magnitude.max() * 0.1)
plt.colorbar()

# Save figure
plt.savefig('/home/tunok/Work/imgProcess_main/tests/others/scatterFourier.png')
plt.close()
print("Saved image as: ... /imgProcess_main/tests/others/scatterFourier.png")

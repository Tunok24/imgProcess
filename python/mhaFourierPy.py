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
imagePath = '/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstTest/totalSlices.mha'
outputPath = '/home/tunok/Work/imgProcess_main/tests/images/'
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
sliceIndex = 1  # for isoharold prost reconstructed volume, take slice 55
rowIndex = int(npImage.shape[1]/2)
colIndex = int(npImage.shape[2]/2)

# NPIMAGE(SLICE, ROW, COLUMN)
imageSlice = npImage[sliceIndex, :, :]
volumeSlice = npImage[:, sliceIndex, :]


# Adjust window/level values
window = 0.1  # Example window value
level = 0.01  # Example level value
adjustedSlice = apply_window_level(imageSlice, window, level)

# Apply Fourier Transform
# np.fft.fftshift function then shifts the zero-frequency component to the center of the spectrum
fTransform = np.fft.fftshift(np.fft.fft2(imageSlice))
fMagnitude = np.abs(fTransform)
fLogMagnitude = np.log(fMagnitude + 1)

# To perform the inverse Fourier transform:
recoveredImage = np.fft.ifft2(np.fft.ifftshift(fTransform))
# Take the absolute value to handle complex numbers
recoveredImage = np.abs(recoveredImage)


# Plotting
plt.figure(figsize=(12, 6))  # Set the figure size (width, height in inches)

# Subplot 1: Original Image
plt.subplot(1, 2, 1)  # (rows, columns, panel number)
plt.imshow(imageSlice, cmap='gray')
plt.title('Original Image')
plt.colorbar()

# Subplot 2: Fourier Transformed Image
plt.subplot(1, 2, 2)
plt.imshow(recoveredImage, cmap='gray')
plt.title('Fourier Transform Magnitude Spectrum')
# Adjust the upper limit to 10% of the max value
# plt.clim(0, fLogMagnitude.max() * 0.1)
plt.colorbar()


# Save figure
plt.tight_layout()  # Adjust layout to not overlap subplots
plt.savefig(outputPath + 'fourierTransformedImage.png')
plt.close()
print('Saved image as:' + outputPath + 'fourierTransformedImage.png')

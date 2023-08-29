#include <vector>
#include <string>
#include <math.h>
#include <yaml-cpp/yaml.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include "itkLinearInterpolateImageFunction.h"
#include <itkFlipImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include "itkDivideImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <itkImageDuplicator.h>
#include "itkImageRegionIterator.h"
#include "itkStatisticsImageFilter.h"
#include <itkBoxImageFilter.h>
#include <rtkThreeDCircularProjectionGeometry.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include <itkMultiplyImageFilter.h>

// Define the image types
typedef itk::Image<float, 3> ImageType; // Assuming your images are 3D and of type float
typedef itk::Image<float, 2> ImageType2D;
using Image1DType = itk::Image<float, 1>;
typedef itk::ImageFileWriter<ImageType> WriterType;

/*void ReadImageProperties(std::string filename)
{
    typedef itk::ImageFileReader<ImageType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);
    reader->Update();

    ImageType::Pointer image = reader->GetOutput();

    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    ImageType::SpacingType spacing = image->GetSpacing();

    std::cout << "Size: " << size[0] << ", " << size[1] << ", " << size[2] << std::endl;
    std::cout << "Spacing: " << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << std::endl;
}/**/

int GetImageDimension(const std::string &filename)
{
    // Create a GenericImageIO object that can handle any type of image
    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(
        filename.c_str(), itk::ImageIOFactory::ReadMode);

    if (!imageIO)
    {
        std::cerr << "Could not CreateImageIO for: " << filename << std::endl;
        return -1;
    }

    // Use the object to read the image file and get its meta-data information
    imageIO->SetFileName(filename);
    imageIO->ReadImageInformation();

    // Return the dimension of the image
    return imageIO->GetNumberOfDimensions();
}

/*float *ReadMHA(const std::string &filename, unsigned int &width, unsigned int &height, unsigned int &numProjections)
{
    int imageDim = GetImageDimension(filename);
    std::cout << "Reading MHA Image File... " << std::endl;
    std::cout << "Number of Dimension: " << imageDim << std::endl;
    // Define the image type and create a reader
    using ImageType = itk::Image<float, 3>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);
    reader->Update();

    // Get the image from the reader
    ImageType::Pointer image = reader->GetOutput();
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();

    // Copy the dimensions into the output parameters
    width = size[0];
    height = size[1];
    numProjections = size[2];

    // Copy the data into a new array
    float *arrayData = new float[width * height * numProjections];
    itk::ImageRegionConstIterator<ImageType> imageIt(image, region);
    size_t i = 0;
    while (!imageIt.IsAtEnd())
    {
        arrayData[i++] = imageIt.Get();
        ++imageIt;
    }

    return arrayData;
}/**/

ImageType::Pointer ReadMHA(const std::string &filename)
{
    // Define the image type and create a reader
    using ImageType = itk::Image<float, 3>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);

    // Update the reader to read the image
    // The reader will automatically allocate the necessary memory
    reader->Update();

    // Get image from the reader
    ImageType::Pointer image = reader->GetOutput();

    // Return the image
    return image;
}

// This is the function for 2D BSpline interpolation function. Outputs the 3DImage MHA file containing the interpolated values.

ImageType::Pointer ScatterEstimation(ImageType::Pointer scatterImage3D)
{
    // Get the size of the 3D image
    itk::Size<3> size3D = scatterImage3D->GetLargestPossibleRegion().GetSize();

    // Create the 3D scatter estimate image outside the loop to prevent it being created multiple times
    ImageType::Pointer scatterEstimate = ImageType::New();
    ImageType::IndexType start3D;
    start3D.Fill(0);
    ImageType::RegionType region3D(start3D, size3D);
    scatterEstimate->SetRegions(region3D);
    scatterEstimate->Allocate();

    // Loop through each slice in the 3D image
    for (unsigned int z = 0; z < size3D[2]; ++z)
    {
        // Create a new 2D scatter image for this slice
        ImageType2D::Pointer scatterImage2D = ImageType2D::New();
        ImageType2D::IndexType start2D;
        start2D.Fill(0);
        ImageType2D::SizeType size2D;
        size2D[0] = size3D[0]; // Width
        size2D[1] = size3D[1]; // Height
        ImageType2D::RegionType region2D(start2D, size2D);
        scatterImage2D->SetRegions(region2D);
        scatterImage2D->Allocate();

        // Copy the current slice from the 3D image to the 2D image
        itk::ImageRegionIterator<ImageType> it3D(scatterImage3D, scatterImage3D->GetLargestPossibleRegion());
        itk::ImageRegionIterator<ImageType2D> it2D(scatterImage2D, scatterImage2D->GetLargestPossibleRegion());
        for (it3D.GoToBegin(); !it3D.IsAtEnd(); ++it3D)
        {
            if (it3D.GetIndex()[2] == z)
            {
                it2D.Set(it3D.Get());
                ++it2D;
            }
        }

        // Create the interpolator
        typedef itk::BSplineInterpolateImageFunction<ImageType2D, double, double> InterpolatorType;
        InterpolatorType::Pointer interpolator = InterpolatorType::New();
        interpolator->SetSplineOrder(3);
        interpolator->SetInputImage(scatterImage2D);

        // Interpolate and store the scatter estimates
        itk::ImageRegionIterator<ImageType> itScatter(scatterEstimate, scatterEstimate->GetLargestPossibleRegion());
        for (itScatter.GoToBegin(); !itScatter.IsAtEnd(); ++itScatter)
        {
            if (itScatter.GetIndex()[2] == z)
            {
                ImageType::IndexType index3D = itScatter.GetIndex();
                ImageType2D::IndexType index2D;
                index2D[0] = index3D[0];
                index2D[1] = index3D[1];
                itScatter.Set(interpolator->EvaluateAtContinuousIndex(index2D));
            }
        }
    }

    return scatterEstimate;
}

typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussFilterType;
void LateralSmoothing(ImageType::Pointer &scatterEstimate, double variance)
{
    GaussFilterType::Pointer filter = GaussFilterType::New();
    filter->SetInput(scatterEstimate);

    GaussFilterType::ArrayType varianceArray;
    varianceArray[0] = variance; // x-direction variance
    varianceArray[1] = variance; // y-direction variance
    varianceArray[2] = 0;        // z-direction variance (no smoothing)
    filter->SetVariance(varianceArray);

    try
    {
        filter->Update();
    }
    catch (itk::ExceptionObject &error)
    {
        std::cerr << "Error: " << error << std::endl;
        return; // Handle error appropriately
    }

    typedef itk::ImageDuplicator<ImageType> DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(filter->GetOutput());
    duplicator->Update();

    scatterEstimate = duplicator->GetOutput();
}

// Function to extract a row or column from a 2D ITK Image
// 'direction' is 0 for row, 1 for column.
// 'index' is the index of the row or column to extract.
/*std::vector<float> extractLine(itk::Image<float, 2>::Pointer image, unsigned direction, unsigned index)
{
    std::vector<float> lineValues;

    // Get the size of the image
    auto size = image->GetLargestPossibleRegion().GetSize();

    // Check if index is valid
    if ((direction == 0 && index >= size[1]) || // If extracting a row
        (direction == 1 && index >= size[0]))   // If extracting a column
    {
        std::cerr << "Invalid index." << std::endl;
        return lineValues; // Return an empty vector
    }

    // Iterate over the image
    itk::ImageRegionIterator<itk::Image<float, 2>> iterator(image, image->GetLargestPossibleRegion());

    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
        auto idx = iterator.GetIndex();

        // If we're extracting a row and this pixel is in the correct row
        if (direction == 0 && idx[1] == index)
        {
            lineValues.push_back(iterator.Get());
        }
        // Or if we're extracting a column and this pixel is in the correct column
        else if (direction == 1 && idx[0] == index)
        {
            lineValues.push_back(iterator.Get());
        }
    }

    return lineValues;
}/**/

/*double interpolateAtPoint(const std::vector<float> &columnData, float y)
{
    // Define the image type using float pixels and 2 dimensions
    using ImageType = itk::Image<float, 1>;
    using ImagePointer = ImageType::Pointer;
    using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double, double>;

    // Create a new 1D ITK image for the column data
    auto columnImage = ImageType::New();
    itk::Size<1> size = {columnData.size()};
    columnImage->SetRegions(size);
    columnImage->Allocate();

    // Copy the column data to the 1D ITK image
    itk::ImageRegionIterator<ImageType> it(columnImage, columnImage->GetLargestPossibleRegion());
    for (unsigned int i = 0; !it.IsAtEnd(); ++i, ++it)
    {
        it.Set(columnData[i]);
    }

    // Set up the interpolator
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3); // Set the order of the spline, e.g. 3 for cubic
    interpolator->SetInputImage(columnImage);

    // Create the point for the y-coordinate
    ImageType::PointType point;
    point[0] = y;

    // Check if the point is inside the image
    if (interpolator->IsInsideBuffer(point))
    {
        // Evaluate the interpolated pixel value
        double interpolatedValue = interpolator->Evaluate(point);
        return interpolatedValue;
    }
    else
    {
        throw std::invalid_argument("Point is outside the image!");
    }
}/**/

/*void InterpolateColumns(itk::Image<float, 3>::Pointer scatterImage3D, itk::ImageRegion<3> exclusionRegion)
{
    // Get the size of the 3D image
    itk::Size<3> size3D = scatterImage3D->GetLargestPossibleRegion().GetSize();

    // Loop through each slice in the 3D image
    for (unsigned int z = 0; z < size3D[2]; ++z)
    {
        // Loop through each column in the 2D slice
        for (unsigned int x = 0; x < size3D[0]; ++x)
        {
            std::vector<float> columnData;

            // Loop through each row in the column
            for (unsigned int y = 0; y < size3D[1]; ++y)
            {
                itk::Image<float, 3>::IndexType index3D;
                index3D[0] = x;
                index3D[1] = y;
                index3D[2] = z;

                // If the current index is outside the exclusion region, add it to the column data
                if (!exclusionRegion.IsInside(index3D))
                {
                    columnData.push_back(scatterImage3D->GetPixel(index3D));
                }
            }

            // Now loop again through the rows, but this time to fill in the missing data in the exclusion region
            for (unsigned int y = exclusionRegion.GetIndex()[1]; y < exclusionRegion.GetIndex()[1] + exclusionRegion.GetSize()[1]; ++y)
            {
                try
                {
                    // Interpolate the missing pixel value
                    float interpolatedValue = interpolateAtPoint(columnData, y);

                    // Replace the pixel value in the original 3D image
                    itk::Image<float, 3>::IndexType index3D;
                    index3D[0] = x;
                    index3D[1] = y;
                    index3D[2] = z;
                    scatterImage3D->SetPixel(index3D, interpolatedValue);
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << e.what() << std::endl;
                }
            }
        }
    }
}/**/

// Causal recursive filter
void CausalRecursiveFilter(ImageType::Pointer inputImage, float theta)
{
    typedef itk::ImageRegionIterator<ImageType> IteratorType;

    itk::Size<3> size = inputImage->GetLargestPossibleRegion().GetSize();

    for (unsigned int x = 0; x < size[0]; ++x)
    {
        for (unsigned int y = 0; y < size[1]; ++y)
        {
            double previousValue = 0.0;
            for (unsigned int z = 0; z < size[2]; ++z)
            {
                ImageType::IndexType index = {x, y, z};
                double currentValue = inputImage->GetPixel(index);
                double outputValue = theta * currentValue + (1 - theta) * previousValue;
                inputImage->SetPixel(index, outputValue);
                previousValue = outputValue;
            }
        }
    }
}

// Non-causal, forward-backward filter
ImageType::Pointer NonCausalFilter(ImageType::Pointer inputImage, int kernelWidth)
{
    typedef itk::BoxImageFilter<ImageType, ImageType> MeanFilterType;

    MeanFilterType::Pointer meanFilter = MeanFilterType::New();
    MeanFilterType::SizeType radius;
    radius.Fill(0);
    radius[2] = kernelWidth / 2; // Kernel width in z-direction
    meanFilter->SetRadius(radius);
    meanFilter->SetInput(inputImage);
    meanFilter->Update();

    return meanFilter->GetOutput();
}

double calculateSNR(itk::Image<float, 3>::Pointer image3D, itk::ImageRegion<3> noiseRegion, unsigned int slice)
{
    typedef itk::Image<float, 2> ImageType2D;
    ImageType2D::Pointer image2D = ImageType2D::New();

    ImageType2D::RegionType region2D;
    ImageType2D::IndexType start;
    ImageType2D::SizeType size;

    start[0] = noiseRegion.GetIndex()[0];
    start[1] = noiseRegion.GetIndex()[1];
    size[0] = noiseRegion.GetSize()[0];
    size[1] = noiseRegion.GetSize()[1];

    region2D.SetSize(size);
    region2D.SetIndex(start);
    image2D->SetRegions(region2D);
    image2D->Allocate();

    typedef itk::ImageRegionIterator<ImageType2D> IteratorType2D;
    IteratorType2D it2D(image2D, image2D->GetRequestedRegion());

    itk::Image<float, 3>::IndexType start3D = noiseRegion.GetIndex();
    start3D[2] = slice;
    itk::Image<float, 3>::SizeType size3D = noiseRegion.GetSize();
    size3D[2] = 0;
    itk::Image<float, 3>::RegionType desiredRegion(start3D, size3D);

    typedef itk::ImageRegionConstIterator<itk::Image<float, 3>> IteratorType3D;
    IteratorType3D it3D(image3D, desiredRegion);

    for (it3D.GoToBegin(), it2D.GoToBegin(); !it3D.IsAtEnd(); ++it3D, ++it2D)
    {
        it2D.Set(it3D.Get());
    }

    double noiseMean = 0.0;
    double noiseStdDev = 0.0;
    unsigned int noiseCount = 0;

    // Calculate mean of noise region
    for (it2D.GoToBegin(); !it2D.IsAtEnd(); ++it2D)
    {
        noiseMean += it2D.Get();
        ++noiseCount;
    }
    noiseMean /= noiseCount;

    // Calculate standard deviation of noise region
    for (it2D.GoToBegin(); !it2D.IsAtEnd(); ++it2D)
    {
        double val = it2D.Get() - noiseMean;
        noiseStdDev += val * val;
    }
    noiseStdDev = std::sqrt(noiseStdDev / (noiseCount - 1));

    return noiseMean / noiseStdDev; // Return SNR
}

double calculateCNR(ImageType::Pointer image,
                    ImageType::RegionType signalRegion1, ImageType::RegionType signalRegion2,
                    ImageType::RegionType noiseRegion)
{
    // Create iterators for the signal and noise regions
    itk::ImageRegionIterator<ImageType> signalIterator1(image, signalRegion1);
    itk::ImageRegionIterator<ImageType> signalIterator2(image, signalRegion2);
    itk::ImageRegionIterator<ImageType> noiseIterator(image, noiseRegion);

    double signalMean1 = 0.0;
    double signalMean2 = 0.0;
    double noiseStdDev = 0.0;
    unsigned int signalCount1 = 0;
    unsigned int signalCount2 = 0;
    unsigned int noiseCount = 0;

    // Calculate mean of signal regions
    for (signalIterator1.GoToBegin(); !signalIterator1.IsAtEnd(); ++signalIterator1)
    {
        signalMean1 += signalIterator1.Get();
        ++signalCount1;
    }
    signalMean1 /= signalCount1;

    for (signalIterator2.GoToBegin(); !signalIterator2.IsAtEnd(); ++signalIterator2)
    {
        signalMean2 += signalIterator2.Get();
        ++signalCount2;
    }
    signalMean2 /= signalCount2;

    // Calculate standard deviation of noise region
    for (noiseIterator.GoToBegin(); !noiseIterator.IsAtEnd(); ++noiseIterator)
    {
        double val = noiseIterator.Get() - ((signalMean1 + signalMean2) / 2);
        noiseStdDev += val * val;
        ++noiseCount;
    }
    noiseStdDev = std::sqrt(noiseStdDev / (noiseCount - 1));

    return std::abs(signalMean1 - signalMean2) / noiseStdDev; // Return CNR
}

double calculateNoise(ImageType::Pointer image, ImageType::RegionType noiseRegion)
{
    itk::ImageRegionIterator<ImageType> noiseIterator(image, noiseRegion);

    double noiseMean = 0.0;
    double noiseStdDev = 0.0;
    unsigned int noiseCount = 0;

    // Calculate mean of noise region
    for (noiseIterator.GoToBegin(); !noiseIterator.IsAtEnd(); ++noiseIterator)
    {
        noiseMean += noiseIterator.Get();
        ++noiseCount;
    }
    noiseMean /= noiseCount;

    // Calculate standard deviation of noise region
    for (noiseIterator.GoToBegin(); !noiseIterator.IsAtEnd(); ++noiseIterator)
    {
        double val = noiseIterator.Get() - noiseMean;
        noiseStdDev += val * val;
    }
    noiseStdDev = std::sqrt(noiseStdDev / (noiseCount - 1));

    return noiseStdDev; // Return standard deviation as a measure of noise
}

double CalculateMean(itk::Image<float, 3>::Pointer image)
{
    using ImageType = itk::Image<float, 3>;
    using StatisticsImageFilterType = itk::StatisticsImageFilter<ImageType>;

    StatisticsImageFilterType::Pointer statsFilter = StatisticsImageFilterType::New();
    statsFilter->SetInput(image);
    statsFilter->Update();

    return statsFilter->GetMean();
}

double CalculateStandardDeviation(itk::Image<float, 3>::Pointer image)
{
    using ImageType = itk::Image<float, 3>;
    using StatisticsImageFilterType = itk::StatisticsImageFilter<ImageType>;

    StatisticsImageFilterType::Pointer statsFilter = StatisticsImageFilterType::New();
    statsFilter->SetInput(image);
    statsFilter->Update();

    return statsFilter->GetSigma();
}

ImageType::Pointer ConcatenateImages(ImageType::Pointer image1, ImageType::Pointer image2, ImageType::SizeType concatSize)
{
    ImageType::RegionType region1 = image1->GetLargestPossibleRegion();
    ImageType::RegionType region2 = image2->GetLargestPossibleRegion();
    ImageType::SizeType size1 = region1.GetSize();
    ImageType::SizeType size2 = region2.GetSize();

    long totalNumberOfPixels1 = size1[0] * size1[1] * size1[2];
    long totalNumberOfPixels2 = size2[0] * size2[1] * size2[2];

    std::vector<float> concatVector;
    std::vector<float> pixelValues1(image1->GetBufferPointer(), image1->GetBufferPointer() + totalNumberOfPixels1);
    std::vector<float> pixelValues2(image2->GetBufferPointer(), image2->GetBufferPointer() + totalNumberOfPixels2);

    concatVector.insert(concatVector.end(), pixelValues1.begin(), pixelValues1.end());
    concatVector.insert(concatVector.end(), pixelValues2.begin(), pixelValues2.end());

    // Create an image.
    ImageType::Pointer concatImage = ImageType::New();

    // Define the region.
    ImageType::RegionType region;
    region.SetSize(concatSize);

    // Set the region and allocate memory for the image.
    concatImage->SetRegions(region);
    concatImage->Allocate();

    // Copy the data from the vector to the image.
    std::copy(concatVector.begin(), concatVector.end(), concatImage->GetBufferPointer());

    return concatImage;
}

ImageType::Pointer ConcatenateMultipleImages(std::vector<std::string> filenames)
{
    // Assume that the images have the same size.
    ImageType::Pointer image1 = ReadMHA(filenames[0]);
    ImageType::SizeType concatSize = image1->GetLargestPossibleRegion().GetSize();

    // Keep track of the current concatenated image.
    ImageType::Pointer currentImage = image1;

    for (int i = 1; i < filenames.size(); i++)
    {
        // Read the next image.
        ImageType::Pointer image2 = ReadMHA(filenames[i]);

        // Concatenate the current image with the next image.
        ImageType::SizeType newSize;
        newSize[0] = concatSize[0];
        newSize[1] = concatSize[1];
        newSize[2] = concatSize[2] + image2->GetLargestPossibleRegion().GetSize()[2];
        currentImage = ConcatenateImages(currentImage, image2, newSize);

        // Update the size for the next concatenation.
        concatSize = newSize;
    }

    return currentImage;
}

using BSplineInterpolatorType = itk::BSplineInterpolateImageFunction<Image1DType>;
using LinearInterpolatorType = itk::LinearInterpolateImageFunction<Image1DType>;

// Function to interpolate a 1D image at a specified position
float interpolate1D(Image1DType::Pointer image, Image1DType::IndexType idx, int interpolationOrder)
{
    if (interpolationOrder == 1)
    {
        LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
        interpolator->SetInputImage(image);
        return interpolator->EvaluateAtIndex(idx);
    }
    else
    {
        BSplineInterpolatorType::Pointer interpolator = BSplineInterpolatorType::New();
        interpolator->SetSplineOrder(interpolationOrder);
        interpolator->SetInputImage(image);
        return interpolator->EvaluateAtIndex(idx);
    }
}

/*void estimateUnknownRows(ImageType::Pointer image, int startUnknownRow, int endUnknownRow, int interpolationOrder)
{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            Image1DType::Pointer columnImage = Image1DType::New();
            Image1DType::RegionType region1D;
            Image1DType::IndexType start1D;
            Image1DType::SizeType size1D;

            start1D[0] = 0;
            size1D[0] = imageSize[1] - (endUnknownRow - startUnknownRow + 1); // Exclude the unknown rows
            region1D.SetSize(size1D);
            region1D.SetIndex(start1D);
            columnImage->SetRegions(region1D);
            columnImage->Allocate();

            // Copy the known values to the 1D image
            int dstIdx = 0;
            for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
            {
                if (rowIdx < startUnknownRow || rowIdx > endUnknownRow)
                {
                    ImageType::IndexType srcIdx = {{colIdx, rowIdx, sliceIdx}};
                    Image1DType::IndexType dstIdx1D = {{dstIdx}};
                    columnImage->SetPixel(dstIdx1D, image->GetPixel(srcIdx));
                    ++dstIdx;
                }
            }

            // Interpolate the unknown values
            for (int rowIdx = startUnknownRow; rowIdx <= endUnknownRow; ++rowIdx)
            {
                Image1DType::IndexType idx1D = {{rowIdx - startUnknownRow}};
                float interpolatedValue = interpolate1D(columnImage, idx1D, interpolationOrder);

                // Insert the interpolated value back into the original image
                ImageType::IndexType idx3D = {{colIdx, rowIdx, sliceIdx}};
                image->SetPixel(idx3D, interpolatedValue);
            }
        }
    }
} /**/

std::vector<float> estimateUnknownValues(const std::vector<float> &values, const std::vector<int> &unknownIndices, int order)
{
    // Collect known data
    std::vector<float> knownValues;
    std::vector<int> knownIndices;
    for (int i = 0; i < values.size(); ++i)
    {
        if (std::find(unknownIndices.begin(), unknownIndices.end(), i) == unknownIndices.end())
        {
            knownValues.push_back(values[i]);
            knownIndices.push_back(i);
        }
    }

    // Build the Vandermonde matrix
    Eigen::MatrixXf X(knownIndices.size(), order + 1);
    for (int i = 0; i < knownIndices.size(); ++i)
    {
        for (int j = 0; j <= order; ++j)
        {
            X(i, j) = std::pow(knownIndices[i], j);
        }
    }

    // Build the y vector
    Eigen::VectorXf y(knownValues.size());
    for (int i = 0; i < knownValues.size(); ++i)
    {
        y[i] = knownValues[i];
    }

    // Solve for the polynomial coefficients
    Eigen::VectorXf coeffs = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    // Estimate the unknown values
    std::vector<float> estimatedValues = values;
    for (int i : unknownIndices)
    {
        float estimate = 0.0f;
        for (int j = 0; j <= order; ++j)
        {
            estimate += coeffs[j] * std::pow(i, j);
        }
        estimatedValues[i] = estimate;
    }

    return estimatedValues;
}

using IndexType = ImageType::IndexType;

void estimateUnknownRows(ImageType::Pointer image, int startUnknownRow, int endUnknownRow, int order)
{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            // Collect the known and unknown values for this column
            std::vector<float> values(imageSize[1]);
            std::vector<int> unknownIndices;
            for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
            {
                IndexType index = {{colIdx, rowIdx, sliceIdx}};
                values[rowIdx] = image->GetPixel(index);
                if (rowIdx >= startUnknownRow && rowIdx <= endUnknownRow)
                {
                    unknownIndices.push_back(rowIdx);
                }
            }

            // Estimate the unknown values
            std::vector<float> estimatedValues = estimateUnknownValues(values, unknownIndices, order);

            // Set the estimated values
            for (int rowIdx = startUnknownRow; rowIdx <= endUnknownRow; ++rowIdx)
            {
                IndexType index = {{colIdx, rowIdx, sliceIdx}};
                image->SetPixel(index, estimatedValues[rowIdx]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::cout << "Checkpoint 1" << std::endl;
    if (argc < 1)
    {
        std::cerr << "Usage: " << argv[0] << " <string> <integer>\n";
        return 1;
    }

    bool boosted = false; // Default: Don't boost

    // Loop through command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "boosted")
        {
            boosted = true;
        }
    }

    /////////////////////////////////////////////////////////
    ///////////////////// READ CT Params ////////////////////
    std::string pathToConfig = "/home/tunok/Work/mcDataIO_main/tests/output/simInfo.txt";
    YAML::Node configFile = YAML::LoadFile(pathToConfig);
    int nPhotons = configFile["nPhotons"].as<int>();
    float SAD = configFile["SAD"].as<float>();
    float SDD = configFile["SDD"].as<float>();
    float SCD = configFile["SCD"].as<float>();
    float SPD = configFile["SPD"].as<float>();
    float PixelPitch = configFile["PixelPitch"].as<float>();
    int nProjections = configFile["nProjections"].as<int>();

    std::cout << "nPhotons: " << nPhotons << std::endl;
    std::cout << "SAD: " << SAD << std::endl;
    std::cout << "SDD: " << SDD << std::endl;
    std::cout << "SCD: " << SCD << std::endl;
    std::cout << "SPD: " << SPD << std::endl;
    std::cout << "PixelPitch: " << PixelPitch << std::endl;
    std::cout << "nProjections: " << nProjections << std::endl; /**/
    ///////////////////// READ CT Params ////////////////////
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    //////////////////// READ IMAGE FILES ///////////////////

    ImageType::Pointer totalImage;
    ImageType::Pointer scatterImage;
    ImageType::Pointer floodImage;

    ////////////////////////////////////////////////////////
    ///////////////////// CONCATENATE //////////////////////
    ImageType::SizeType concatSize;
    concatSize[0] = 375; // size in x-direction
    concatSize[1] = 375; // size in y-direction
    concatSize[2] = 401; // size in z-direction

    std::cout << "Concatenating Images ... " << std::endl;

    if (boosted)
    {
        std::vector<std::string> totalBoostedImageFilenames;
        totalBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage100000000Projection5.mha");
        totalBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage100000000Projection6.mha");
        totalBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage100000000Projection7.mha");
        totalBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage100000000Projection8.mha");

        std::vector<std::string> scatterBoostedImageFilenames;
        scatterBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage100000000Projection5.mha");
        scatterBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage100000000Projection6.mha");
        scatterBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage100000000Projection7.mha");
        scatterBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage100000000Projection8.mha");

        std::vector<std::string> floodBoostedImageFilenames;
        floodBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6,7,8/floodBoostedImage300000000Projection5.mha");
        floodBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6,7,8/floodBoostedImage300000000Projection6.mha");
        floodBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6,7,8/floodBoostedImage300000000Projection7.mha");
        floodBoostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6,7,8/floodBoostedImage300000000Projection8.mha");

        totalImage = ConcatenateMultipleImages(totalBoostedImageFilenames);
        scatterImage = ConcatenateMultipleImages(scatterBoostedImageFilenames);
        floodImage = ConcatenateMultipleImages(floodBoostedImageFilenames);
    }
    else
    {
        std::vector<std::string> totalUnboostedImageFilenames;
        totalUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalUnboostedImage100000000Projection5.mha");
        totalUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/totalUnboostedImage100000000Projection6.mha");
        // totalUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/totalUnboostedImage100000000Projection7.mha");
        // totalUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/totalUnboostedImage100000000Projection8.mha");

        std::vector<std::string> scatterUnboostedImageFilenames;
        scatterUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterUnboostedImage100000000Projection5.mha");
        scatterUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/scatterUnboostedImage100000000Projection6.mha");
        // scatterUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/scatterUnboostedImage100000000Projection7.mha");
        // scatterUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/scatterUnboostedImage100000000Projection8.mha");

        std::vector<std::string> floodUnboostedImageFilenames;
        floodUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6/floodUnboostedImage300000000Projection5.mha");
        floodUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6/floodUnboostedImage300000000Projection6.mha");
        // floodUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/floodUnboostedImage100000000Projection7.mha");
        // floodUnboostedImageFilenames.push_back("/home/tunok/Work/mcDataIO_main/tests/output/Set100000000Projection5,6,7,8/floodUnboostedImage100000000Projection8.mha");

        totalImage = ConcatenateMultipleImages(totalUnboostedImageFilenames);
        scatterImage = ConcatenateMultipleImages(scatterUnboostedImageFilenames);
        floodImage = ConcatenateMultipleImages(floodUnboostedImageFilenames);
    }

    // Define the writer type
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();

    if (boosted)
    {
        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/totalBoosted.mha");
        writer->SetInput(totalImage);
        writer->Update();
        std::cout << "Writing File Out: totalBoosted.mha" << std::endl;

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterBoosted.mha");
        writer->SetInput(scatterImage);
        writer->Update();
        std::cout << "Writing File Out: scatterBoosted.mha" << std::endl;

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/floodBoosted.mha");
        writer->SetInput(floodImage);
        writer->Update();
        std::cout << "Writing File Out: floodBoosted.mha" << std::endl;
    }
    else
    {
        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/totalUnboosted.mha");
        writer->SetInput(totalImage);
        writer->Update();
        std::cout << "Writing File Out: totalUnboosted.mha" << std::endl;

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterUnboosted.mha");
        writer->SetInput(scatterImage);
        writer->Update();
        std::cout << "Writing File Out: scatterUnboosted.mha" << std::endl;

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/floodUnboosted.mha");
        writer->SetInput(floodImage);
        writer->Update();
        std::cout << "Writing File Out: floodUnboosted.mha" << std::endl;
    }
    ///////////////////// CONCATENATE //////////////////////
    ////////////////////////////////////////////////////////

    //////////////////// READ IMAGE FILES ///////////////////
    /////////////////////////////////////////////////////////
    if (false) // this is to execute the image process section or to calculate CNR values
    {
        /////////////////////////////////////////////////////////
        //////////////////// SCATTER ESTIMATE ///////////////////
        // Define the ROI region (that is a misrepresentation region of the scatter signal i.e. we want to exclude this region
        // Create a copy of the scatter image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;
        DuplicatorType::Pointer duplicator = DuplicatorType::New();
        duplicator->SetInputImage(scatterImage);
        duplicator->Update();
        ImageType::Pointer scatterEst = duplicator->GetOutput();

        // If boosting, then estimate scatter
        if (boosted)
        {
            // Estimate the values in the unknown region
            std::cout << "Interpolating ... " << std::endl;
            // Define the start and end rows of the unknown region
            int startUnknownRow = 100;
            int endUnknownRow = 200;
            int interpolationOrder = 4;
            estimateUnknownRows(scatterEst, startUnknownRow, endUnknownRow, interpolationOrder);
            std::cout << "Done interpolating ... " << std::endl;

            // Lateral Smoothing of ScatterEstimation
            std::cout << "Lateral Smoothing Scatter ... " << std::endl;
            double variance = 5;
            LateralSmoothing(scatterEst, variance);

            // Further Scatter Estimate by doing PROJECTION-TO-PROJECTION-SMOOTHING by theta_filter
            std::cout << "Projection-To-Projection Smoothing Scatter ... " << std::endl;
            float theta_filt = 1.0f; // 0 - <1: Causal 1 - 40: Non-causal
            unsigned int width = 2;  // Choose your filter width
            CausalRecursiveFilter(scatterEst, theta_filt);
            // ImageType::Pointer scatterEst = NonCausalFilter(scatterLatSmooth, width); /**/
            writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterEst.mha");
            writer->SetInput(scatterEst);
            writer->Update();
            std::cout << "Writing File Out: scatterEst.mha" << std::endl;
        }
        //////////////////// SCATTER ESTIMATE ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// IMAGE CORRECTION ///////////////////

        std::cout << "Scatter Correcting Original Images ... " << std::endl;
        //// Correcting Scatter Image by Scatter Estimate ////
        using SubtractFilterType = itk::SubtractImageFilter<ImageType>;
        SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
        subtractFilter->SetInput1(totalImage);
        subtractFilter->SetInput2(scatterImage);
        subtractFilter->Update();
        ImageType::Pointer primaryImage = subtractFilter->GetOutput(); // This "primaryImage" is the purely primary
        primaryImage->DisconnectPipeline();

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        subtractFilter->SetInput1(scatterImage);
        subtractFilter->SetInput2(scatterEst);
        subtractFilter->Update();
        ImageType::Pointer scatterCorrectedScatter = subtractFilter->GetOutput();
        scatterCorrectedScatter->DisconnectPipeline();

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
        subtractFilter->SetInput1(totalImage);
        subtractFilter->SetInput2(scatterEst);
        subtractFilter->Update();
        ImageType::Pointer primaryEst = subtractFilter->GetOutput(); // This "primaryEst" is the estimated primary
        primaryEst->DisconnectPipeline();

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        //// Correcting Total Image by Flood Image  ////
        std::cout << "Flood Correcting Scatter Corrected Images ... " << std::endl;
        using DivideFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
        DivideFilterType::Pointer divideFilter = DivideFilterType::New();
        divideFilter->SetInput1(primaryImage);
        divideFilter->SetInput2(floodImage);
        divideFilter->Update();
        primaryImage = divideFilter->GetOutput();
        primaryImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryEst);
        divideFilter->Update();
        primaryEst = divideFilter->GetOutput();
        primaryEst->DisconnectPipeline();

        /////////// ReScale an image if needed ///////////
        std::cout << "Rescaling Corrected Images ... " << std::endl;
        // typedef for the RescaleIntensityImageFilter
        using FilterType = itk::MultiplyImageFilter<ImageType, ImageType, ImageType>;
        auto multiplyConstantFilter = FilterType::New();
        multiplyConstantFilter->SetInput(primaryImage);
        multiplyConstantFilter->SetConstant(5);
        multiplyConstantFilter->Update();
        primaryImage = multiplyConstantFilter->GetOutput();
        primaryImage->DisconnectPipeline();

        multiplyConstantFilter->SetInput(primaryEst);
        multiplyConstantFilter->Update();
        primaryEst = multiplyConstantFilter->GetOutput();
        primaryEst->DisconnectPipeline();
        //////////////////// IMAGE CORRECTION ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        std::cout << "Setting size and origin ... " << std::endl;
        // Change size
        ImageType::SizeType size;
        size[0] = 250;                  // New size in the x direction
        size[1] = 250;                  // New size in the y direction
        size[2] = 401;                  // New size in the z direction
        primaryImage->SetRegions(size); // when doing scatter correction
        primaryEst->SetRegions(size);

        // Change spacing
        ImageType::SpacingType spacing;
        spacing[0] = 1;                    // New spacing in the x direction
        spacing[1] = 1;                    // New spacing in the y direction
        spacing[2] = 1;                    // New spacing in the z direction
        primaryImage->SetSpacing(spacing); // when doing scatter correction
        primaryEst->SetSpacing(spacing);

        // Change origin
        ImageType::PointType newOrigin;
        newOrigin[0] = -((size[0] / 2) - (spacing[0] / 2)); // new x origin
        newOrigin[1] = -((size[1] / 2) - (spacing[1] / 2)); // new y origin
        newOrigin[2] = 0;                                   // new z origin
        primaryImage->SetOrigin(newOrigin);                 // when doing scatter correction
        primaryEst->SetOrigin(newOrigin);
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// WRITE FILES OUT ////////////////////

        if (boosted)
        {
            writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/primaryBoostedImage.mha");
            writer->SetInput(primaryImage);
            writer->Update();
            std::cout << "Writing File Out: primaryBoostedImage.mha" << std::endl;

            writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/primaryEst.mha");
            writer->SetInput(primaryEst);
            writer->Update();
            std::cout << "Writing File Out: primaryEst.mha" << std::endl;
            writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterCorrectedScatter.mha");
            writer->SetInput(scatterCorrectedScatter);
            writer->Update();
            std::cout << "Writing File Out: scatterCorrectedScatter.mha" << std::endl;
        }
        else
        {
            writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/primaryUnboostedImage.mha");
            writer->SetInput(primaryImage);
            writer->Update();
            std::cout << "Writing File Out: primaryUnboostedImage.mha" << std::endl;
        }

        //////////////////// WRITE FILES OUT ////////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////////////// GEOMETRY XML //////////////////////

        // Set the parameters
        std::cout << "Writing Geometry ... " << std::endl;
        using GeometryType = rtk::ThreeDCircularProjectionGeometry;
        GeometryType::Pointer geometry = GeometryType::New();

        double sid = 1000.0;                // source to isocenter distance
        double sdd = 1500.0;                // source to detector distance
        double start_angle = 0.0;           // start angle
        double stop_angle = 360.0;          // stop angle
        unsigned int num_projections = 400; // number of projections

        double angle_step = (stop_angle - start_angle) / (num_projections);
        for (unsigned int i = 0; i <= num_projections; i++)
        {
            double angle = start_angle + i * angle_step;
            geometry->AddProjection(sid, sdd, -angle, 0, 0, 0, 0, 0, 0); // Positive angle flips the object front to back
        }

        rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter;
        xmlWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
        xmlWriter->SetFilename("/home/tunok/Work/imgProcess_main/tests/Geometry.xml");
        xmlWriter->SetObject(geometry);
        xmlWriter->WriteFile();
        std::cout << "Written File Out: Geometry.xml" << std::endl;
    }

    if (false)
    {

        /////////////////////////////////////////////////////////
        //////////////////// WRITE VALUES OUT ///////////////////
        // Read Image file: Good Corrected Reconstructed Volume
        ImageType::Pointer goodVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/primaryBoostedVolume.mha");
        // Read Image file: Bad Corrected Reconstructed Volume
        ImageType::Pointer badVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/primaryUnboostedVolume.mha");

        // Caculating Mean
        /*double meanOriginal = CalculateMean(scatterImage);
        double meanEstimate = CalculateMean(scatterEst);
        double meanCorrected = CalculateMean(scatterCorrectedScatterImage);
        std::cout << "Scatter Means of Original, Estimate and Corrected Images are: " << meanOriginal << ", " << meanEstimate << ", " << meanCorrected << "." << std::endl;

        // Calculating Standard Deviation
        double stdOriginal = CalculateStandardDeviation(scatterImage);
        double stdEstimate = CalculateStandardDeviation(scatterEst);
        double stdCorrected = CalculateStandardDeviation(scatterCorrectedScatterImage);
        std::cout << "Scatter Stds of Original, Estimate and Corrected Images are: " << stdOriginal << ", " << stdEstimate << ", " << stdCorrected << "." << std::endl;/**/

        /*
        ImageType::SizeType size = goodReconVolume->GetLargestPossibleRegion().GetSize();
        std::cout << "Image dimensions: "
                  << size[0] << " x "
                  << size[1] << " x "
                  << size[2] << std::endl;

        size = badReconVolume->GetLargestPossibleRegion().GetSize();
        std::cout << "Image dimensions: "
                  << size[0] << " x "
                  << size[1] << " x "
                  << size[2] << std::endl;

        ///////////// Calculating SNR //////////////
        // int imageDim = GetImageDimension("/home/tunok/Work/mcDataIO_main/tests/output/scatterImage.mha");
        // std::cout << "Number of Dimension: " << imageDim << std::endl;
        // Let 's assume we' re interested in the 1st slice, 192nd Row and 192nd Column(depth = 0)
        const unsigned int sliceNumber = 36;
        const unsigned int rowIndex = 191; // indexing starts from 0
        const unsigned int columnIndex = 191;
        // Define the noise region
        ImageType::IndexType start3D;
        start3D[0] = 45;          // example, needs to be defined by the user we are aiming around the point (50, 73, 36)
        start3D[1] = 68;          // example, needs to be defined by the user
        start3D[2] = sliceNumber; // we will set the z-index when we call the function

        ImageType::SizeType size3D;
        size3D[0] = 10; // example, needs to be defined by the user
        size3D[1] = 10; // example, needs to be defined by the user
        size3D[2] = 1;  // single slice

        ImageType::RegionType noiseRegion;
        noiseRegion.SetSize(size3D);
        noiseRegion.SetIndex(start3D);

        double snr = calculateSNR(goodReconVolume, noiseRegion, sliceNumber);
        std::cout << "The SNR for good Image is: " << snr << std::endl;
        snr = calculateSNR(badReconVolume, noiseRegion, sliceNumber);
        std::cout << "The SNR for bad Image is: " << snr << std::endl;
        /**/
        ///////////// Calculating SNR //////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////// Calculating CNR ////////////////////////////////////////

        /////////////////////////////////////// good Region ///////////////////////////////////////
        double cnrGood;
        double cnrBad;
        if (true)
        {
            ImageType::SizeType size = {20, 1, 20};

            // Create the position vector of the midpoint of the regions
            std::vector<float> midPointRegion1 = {180, 210, 130};
            std::vector<float> midPointRegion2 = {180, 210, 195};     // 190 - signal region inside the good region
            std::vector<float> midPointRegionNoise = {180, 210, 195}; //{180, 212, 195}; // midPointRegion2
            std::cout << "x: " << midPointRegion1[0] << ", y: " << midPointRegion1[1] << ", z: " << midPointRegion1[2] << std::endl;

            // signalRegion1 around (50, 73, 36), singalRegion2 around (95, 20, 36), noiseRegion around (20, 20, 36)
            ImageType::IndexType startRegion1;
            startRegion1[0] = midPointRegion1[0] - (size[0] / 2); // starting x coordinate
            startRegion1[1] = midPointRegion1[1];                 // starting y coordinate
            startRegion1[2] = midPointRegion1[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegion1 x: " << startRegion1[0] << ", y: " << startRegion1[1] << ", z: " << startRegion1[2] << std::endl;

            ImageType::IndexType startRegion2;
            startRegion2[0] = midPointRegion2[0] - (size[0] / 2); // starting x coordinate
            startRegion2[1] = midPointRegion2[1];                 // starting y coordinate
            startRegion2[2] = midPointRegion2[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegion2 x: " << startRegion2[0] << ", y: " << startRegion2[1] << ", z: " << startRegion2[2] << std::endl;

            ImageType::IndexType startRegionNoise;
            startRegionNoise[0] = midPointRegionNoise[0] - (size[0] / 2); // starting x coordinate
            startRegionNoise[1] = midPointRegionNoise[1];                 // starting y coordinate
            startRegionNoise[2] = midPointRegionNoise[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegionNoise x: " << startRegionNoise[0] << ", y: " << startRegionNoise[1] << ", z: " << startRegionNoise[2] << std::endl;

            ImageType::RegionType region1;
            region1.SetIndex(startRegion1);
            region1.SetSize(size);

            ImageType::RegionType region2;
            region2.SetIndex(startRegion2);
            region2.SetSize(size);

            ImageType::RegionType regionNoise;
            regionNoise.SetIndex(startRegionNoise);
            regionNoise.SetSize(size);

            cnrGood = calculateCNR(goodVolume, region1, region2, regionNoise);
            std::cout << "The CNR of the Good Image is: " << cnrGood << std::endl;
        }
        /////////////////////////////////////// Good Region ///////////////////////////////////////

        /////////////////////////////////////// Bad Region ///////////////////////////////////////
        // Assuming the size, shape and range of the regions in both good and bad volumes will be the same.
        if (true)
        {

            ImageType::SizeType size = {20, 1, 20};

            // Create the position vector of the midpoint of the regions
            std::vector<float> midPointRegion1 = {180, 210, 130};
            std::vector<float> midPointRegion2 = {180, 210, 250};     // 190 - signal region inside the boosted region
            std::vector<float> midPointRegionNoise = {180, 210, 250}; //{180, 212, 195}; // midPointRegion2
            std::cout << "x: " << midPointRegion1[0] << ", y: " << midPointRegion1[1] << ", z: " << midPointRegion1[2] << std::endl;

            // signalRegion1 around (50, 73, 36), singalRegion2 around (95, 20, 36), noiseRegion around (20, 20, 36)
            ImageType::IndexType startRegion1;
            startRegion1[0] = midPointRegion1[0] - (size[0] / 2); // starting x coordinate
            startRegion1[1] = midPointRegion1[1];                 // starting y coordinate
            startRegion1[2] = midPointRegion1[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegion1 x: " << startRegion1[0] << ", y: " << startRegion1[1] << ", z: " << startRegion1[2] << std::endl;

            ImageType::IndexType startRegion2;
            startRegion2[0] = midPointRegion2[0] - (size[0] / 2); // starting x coordinate
            startRegion2[1] = midPointRegion2[1];                 // starting y coordinate
            startRegion2[2] = midPointRegion2[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegion2 x: " << startRegion2[0] << ", y: " << startRegion2[1] << ", z: " << startRegion2[2] << std::endl;

            ImageType::IndexType startRegionNoise;
            startRegionNoise[0] = midPointRegionNoise[0] - (size[0] / 2); // starting x coordinate
            startRegionNoise[1] = midPointRegionNoise[1];                 // starting y coordinate
            startRegionNoise[2] = midPointRegionNoise[2] - (size[2] / 2); // starting z coordinate
            std::cout << "startRegionNoise x: " << startRegionNoise[0] << ", y: " << startRegionNoise[1] << ", z: " << startRegionNoise[2] << std::endl;

            ImageType::RegionType region1;
            region1.SetIndex(startRegion1);
            region1.SetSize(size);

            ImageType::RegionType region2;
            region2.SetIndex(startRegion2);
            region2.SetSize(size);

            ImageType::RegionType regionNoise;
            regionNoise.SetIndex(startRegionNoise);
            regionNoise.SetSize(size);

            cnrBad = calculateCNR(badVolume, region1, region2, regionNoise);
            std::cout << "The CNR of the Bad Image is: " << cnrBad << std::endl;
        }
        if (cnrBad != 0.0)
        { // Avoid division by zero
            double improvementPercentage = (cnrGood - cnrBad) / cnrBad * 100.0;
            std::cout << "Improvement in CNR is " << improvementPercentage << "%" << std::endl;
            ;
        }
        else
        {
            std::cout << "Cannot calculate improvement as the original CNR is zero." << std::endl;
            ;
        }

        /////////////////////////////////////// Calculating CNR ////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// Calculating NOISE ///////////////////////////////////////

        // Define the region you want to assess the noise
        ImageType::RegionType noiseRegion;
        ImageType::SizeType size = {20, 1, 20};

        // Create the position vector of the midpoint of the regions
        std::vector<float> midPoint = {180, 210, 130};
        std::cout << "x: " << midPoint[0] << ", y: " << midPoint[1] << ", z: " << midPoint[2] << std::endl;

        // signalRegion1 around (50, 73, 36), singalRegion2 around (95, 20, 36), noiseRegion around (20, 20, 36)
        ImageType::IndexType startRegionNoise;
        startRegionNoise[0] = midPoint[0] - (size[0] / 2); // starting x coordinate
        startRegionNoise[1] = midPoint[1];                 // starting y coordinate
        startRegionNoise[2] = midPoint[2] - (size[2] / 2); // starting z coordinate
        std::cout << "startRegionNoise x: " << startRegionNoise[0] << ", y: " << startRegionNoise[1] << ", z: " << startRegionNoise[2] << std::endl;

        noiseRegion.SetSize(size);
        noiseRegion.SetIndex(startRegionNoise);

        // Assuming 'image' is already defined as an ITK image pointer
        double noise = calculateNoise(goodVolume, noiseRegion);

        std::cout << "Noise in the specified region is: " << noise << std::endl;

        ////////////////////////////////////// Calculating NOISE ///////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////// WRITE VALUES OUT ///////////////////
        /////////////////////////////////////////////////////////
    }
    std::cout
        << "Press ENTER to continue... " << std::flush;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    return 0;
}

// running this program should output the string: "Image dimensions: 250 x 250 x 401" (as of August 9th, 2023)
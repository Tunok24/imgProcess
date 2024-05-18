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
#include "itkAddImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <itkExtractImageFilter.h>
#include <itkImageDuplicator.h>
#include "itkImageRegionIterator.h"
#include "itkImage.h"
#include "itkNeighborhoodIterator.h"
#include "itkStatisticsImageFilter.h"
#include <itkBoxImageFilter.h>
#include <rtkThreeDCircularProjectionGeometry.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include <itkMultiplyImageFilter.h>

// Define the image types
typedef itk::Image<float, 3> ImageType; // Assuming images are 3D and of type float
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

/* Median Filter: Reduces noise while preserving edges by replacing each pixel's value with the
median value of its neighborhood.
using MedianFilterType = itk::MedianImageFilter<ImageType, ImageType>;
MedianFilterType::Pointer medianFilter = MedianFilterType::New();
MedianFilterType::InputSizeType radius;
radius.Fill(1); // Set radius size
medianFilter->SetRadius(radius);
medianFilter->SetInput(image);
medianFilter->Update();
ImageType::Pointer denoisedImage = medianFilter->GetOutput();*/

/* Smoothing Recursive Gaussian Filter: Smooths the image by applying a Gaussian blur, effective for Gaussian noise.
using GaussianFilterType = itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>;
GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
gaussianFilter->SetSigma(1.0); // Set the standard deviation of the kernel
gaussianFilter->SetInput(image);
gaussianFilter->Update();*/

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

// AddImages: adds the two images pixelwise
ImageType::Pointer AddImages(ImageType::Pointer image1, const std::string &file2)
{
    ImageType::Pointer image2 = ReadMHA(file2);

    // Check if dimensions are the same
    if (image1->GetLargestPossibleRegion().GetSize() != image2->GetLargestPossibleRegion().GetSize())
    {
        throw std::runtime_error("Images do not have the same dimensions.");
    }

    typedef itk::AddImageFilter<ImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(image1);
    addFilter->SetInput2(image2);
    addFilter->Update();

    ImageType::Pointer outputImage = addFilter->GetOutput();
    outputImage->SetSpacing(image1->GetSpacing());
    outputImage->SetOrigin(image1->GetOrigin());
    outputImage->SetDirection(image1->GetDirection());

    return outputImage;
}

// void AddMultipleImages(const std::vector<std::string> &filePaths, const std::string &outputFile)
// {
//     if (filePaths.size() < 2)
//     {
//         throw std::runtime_error("At least two image files are required.");
//     }

//     ImageType::Pointer currentImage = ReadMHA(filePaths[0]);

//     for (size_t i = 1; i < filePaths.size(); ++i)
//     {
//         currentImage = AddImages(currentImage, filePaths[i]);
//     }

//     typedef itk::ImageFileWriter<ImageType> WriterType;
//     WriterType::Pointer writer = WriterType::New();
//     writer->SetFileName(outputFile);
//     writer->SetInput(currentImage);
//     writer->Update();
// }

void AddMultipleImages(const std::vector<std::string> &filePaths, const std::string &outputFile)
{
    if (filePaths.empty())
    {
        throw std::runtime_error("No image files provided.");
    }

    ImageType::Pointer currentImage = ReadMHA(filePaths[0]);

    for (size_t i = 1; i < filePaths.size(); ++i)
    {
        currentImage = AddImages(currentImage, filePaths[i]);
    }

    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outputFile);
    writer->SetInput(currentImage);
    writer->Update();
}

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

ImageType2D::Pointer calculateLocalSNR(ImageType::Pointer image3D, int sliceIndex)
{
    // Define a filter to extract the 2D slice
    typedef itk::ExtractImageFilter<ImageType, ImageType2D> FilterType;
    FilterType::Pointer filter = FilterType::New();

    ImageType::RegionType inputRegion = image3D->GetLargestPossibleRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    size[2] = 0;

    ImageType::IndexType start = inputRegion.GetIndex();
    start[2] = sliceIndex;

    ImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);

    filter->SetExtractionRegion(desiredRegion);
    filter->SetInput(image3D);

    // Set the direction collapse strategy
    filter->SetDirectionCollapseToSubmatrix(); // or filter->SetDirectionCollapseToIdentity();

    filter->Update();
    ImageType2D::Pointer image2D = filter->GetOutput();

    // Create a new 2D image to store the local SNR values
    ImageType2D::Pointer snrImage = ImageType2D::New();
    snrImage->SetRegions(image2D->GetLargestPossibleRegion());
    snrImage->SetSpacing(image2D->GetSpacing());
    snrImage->SetOrigin(image2D->GetOrigin());
    snrImage->SetDirection(image2D->GetDirection());
    snrImage->Allocate();

    // Create 3x3 neighborhood iterator for the original image
    itk::Size<2> radius;
    radius.Fill(1);
    itk::NeighborhoodIterator<ImageType2D> nit(radius, image2D, image2D->GetLargestPossibleRegion());

    // Create an iterator for the new SNR image
    itk::ImageRegionIterator<ImageType2D> sit(snrImage, snrImage->GetLargestPossibleRegion());

    // Calculate local SNR
    for (nit.GoToBegin(), sit.GoToBegin(); !nit.IsAtEnd() && !sit.IsAtEnd(); ++nit, ++sit)
    {
        float sum = 0.0;
        float sumSq = 0.0;

        for (unsigned int i = 0; i < nit.Size(); ++i)
        {
            float val = nit.GetPixel(i);
            sum += val;
            sumSq += val * val;
        }

        float mean = sum / nit.Size();
        float variance = (sumSq - sum * mean) / (nit.Size() - 1);
        float stddev = std::sqrt(variance);

        float localSNR = (stddev != 0.0) ? (mean / stddev) : 0.0;

        sit.Set(localSNR);
    }

    return snrImage;
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

ImageType::Pointer ExtractFirstNSlices(const ImageType::Pointer &inputImage, unsigned int numberOfSlices)
{
    // Define the region to extract
    ImageType::RegionType inputRegion = inputImage->GetLargestPossibleRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    size[2] = numberOfSlices; // Set the size in the z direction to the desired number of slices

    ImageType::IndexType start = inputRegion.GetIndex();
    start[2] = 0; // Start from the first slice

    // Define the desired region
    ImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);

    // Set up the extraction filter
    typedef itk::ExtractImageFilter<ImageType, ImageType> FilterType;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetExtractionRegion(desiredRegion);
#if ITK_VERSION_MAJOR >= 4
    filter->SetDirectionCollapseToIdentity(); // This line is only needed for ITK >= 4
#endif
    filter->SetInput(inputImage);

    // Apply the filter
    filter->Update();

    // The result is a new image with only the first numberOfSlices slices
    return filter->GetOutput();
}

/**
 * @brief Estimates unknown values in a dataset using polynomial interpolation.
 *
 * This function takes a vector of values and a vector of indices that indicate which values are unknown
 * and need to be estimated. The known values are used to construct a Vandermonde matrix for a least squares
 * fitting of a polynomial of a specified order. The polynomial is then used to estimate the unknown values.
 *
 * @param values A std::vector<float> containing the known and placeholder values for the unknowns in the dataset.
 * @param unknownIndices A std::vector<int> containing the indices in the 'values' vector that need to be estimated.
 * @param order The order of the polynomial to be fitted to the known values.
 * @return A std::vector<float> containing the original known values and the newly estimated values for the unknowns.
 *
 * @note This function uses the Eigen library to solve the least squares problem and compute the polynomial coefficients.
 *       It assumes that 'values' contains at least 'order + 1' known values to construct a solvable system.
 *       The 'unknownIndices' should be within the range of 'values' vector indices.
 */

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

// Function to process each row of the 3D input image within the ROI
void estimateUnknownRows(ImageType::Pointer inputImage, ImageType::Pointer maskImage, int fitOrder)
{
    ImageType::RegionType region = inputImage->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();

    // Iterators for input and mask images
    itk::ImageRegionIteratorWithIndex<ImageType> inputIt(inputImage, region);
    itk::ImageRegionIteratorWithIndex<ImageType> maskIt(maskImage, region);

    // Iterate over each slice (z-direction)
    for (unsigned int z = 0; z < size[2]; ++z)
    {
        // Iterate over each row (y-direction)
        for (unsigned int y = 0; y < size[1]; ++y)
        {
            std::vector<float> rowValues;
            std::vector<int> unknownIndices;

            // Collect row values and determine unknown indices based on the mask
            for (unsigned int x = 0; x < size[0]; ++x)
            {
                ImageType::IndexType idx = {{x, y, z}};
                inputIt.SetIndex(idx);
                maskIt.SetIndex(idx);

                rowValues.push_back(inputIt.Get());
                if (maskIt.Get() == 1)
                { // If the mask indicates an unknown value
                    unknownIndices.push_back(x);
                }
            }

            // Estimate the unknown values in the row
            if (!unknownIndices.empty())
            {
                std::vector<float> estimatedRow = estimateUnknownValues(rowValues, unknownIndices, fitOrder);

                // Write the estimated values back to the input image
                for (int i : unknownIndices)
                {
                    ImageType::IndexType idx = {{static_cast<unsigned int>(i), y, z}};
                    inputImage->SetPixel(idx, estimatedRow[i]);
                }
            }
        }
    }
}

// Function to process each column of the 3D input image within the ROI
void estimateUnknownColumns(ImageType::Pointer inputImage, ImageType::Pointer maskImage, int fitOrder)
{
    ImageType::RegionType region = inputImage->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();

    // Iterators for input and mask images
    itk::ImageRegionIteratorWithIndex<ImageType> inputIt(inputImage, region);
    itk::ImageRegionIteratorWithIndex<ImageType> maskIt(maskImage, region);

    // Iterate over each slice (z-direction)
    for (unsigned int z = 0; z < size[2]; ++z)
    {
        // Iterate over each column (x-direction)
        for (unsigned int x = 0; x < size[0]; ++x)
        {
            std::vector<float> columnValues;
            std::vector<int> unknownIndices;

            // Collect column values and determine unknown indices based on the mask
            for (unsigned int y = 0; y < size[1]; ++y)
            {
                ImageType::IndexType idx = {{x, y, z}};
                inputIt.SetIndex(idx);
                maskIt.SetIndex(idx);

                columnValues.push_back(inputIt.Get());
                if (maskIt.Get() == 1)
                { // If the mask indicates an unknown value
                    unknownIndices.push_back(y);
                }
            }

            // Estimate the unknown values in the column
            if (!unknownIndices.empty())
            {
                std::vector<float> estimatedColumn = estimateUnknownValues(columnValues, unknownIndices, fitOrder);

                // Write the estimated values back to the input image
                for (int i : unknownIndices)
                {
                    ImageType::IndexType idx = {{x, static_cast<unsigned int>(i), z}};
                    inputImage->SetPixel(idx, estimatedColumn[i]);
                }
            }
        }
    }
}

// // Function to add and average two images
// ImageType::Pointer addAndAverage(ImageType::Pointer image1, ImageType::Pointer image2)
// {
//     typedef itk::AddImageFilter<ImageType> AddImageFilterType;
//     typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideImageFilterType;

//     // Add the two images
//     AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
//     addFilter->SetInput1(image1);
//     addFilter->SetInput2(image2);
//     addFilter->Update();

//     // Divide the result by 2 to average
//     DivideImageFilterType::Pointer divideFilter = DivideImageFilterType::New();
//     divideFilter->SetInput(addFilter->GetOutput());
//     divideFilter->SetConstant(2.0);
//     divideFilter->Update();

//     return divideFilter->GetOutput();
// }

// void estimateROI(ImageType::Pointer inputImage, ImageType::Pointer maskImage, int fitOrder)
// {
//     typedef itk::ImageDuplicator<ImageType> DuplicatorType;

//     // Duplicate the inputImage for row-wise estimation
//     DuplicatorType::Pointer duplicator = DuplicatorType::New();
//     duplicator->SetInputImage(inputImage);
//     duplicator->Update();
//     ImageType::Pointer rowEstimate = duplicator->GetOutput();

//     // Estimate unknown values row-wise
//     estimateUnknownRows(rowEstimate, maskImage, fitOrder);

//     // Duplicate the inputImage again for column-wise estimation
//     duplicator->Update();
//     ImageType::Pointer colEstimate = duplicator->GetOutput();

//     // Estimate unknown values column-wise
//     estimateUnknownColumns(colEstimate, maskImage, fitOrder);

//     // Add and average the row and column estimates
//     inputImage = addAndAverage(rowEstimate, colEstimate);
//     // ImageType::Pointer finalEstimate = addAndAverage(rowEstimate, colEstimate);

//     // // Create an iterator to modify only the ROI in the inputImage
//     // itk::ImageRegionIteratorWithIndex<ImageType> inputIt(inputImage, inputImage->GetLargestPossibleRegion());
//     // itk::ImageRegionIteratorWithIndex<ImageType> finalIt(finalEstimate, finalEstimate->GetLargestPossibleRegion());
//     // itk::ImageRegionIteratorWithIndex<ImageType> maskIt(maskImage, maskImage->GetLargestPossibleRegion());

//     // for (inputIt.GoToBegin(), finalIt.GoToBegin(), maskIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++finalIt, ++maskIt)
//     // {
//     //     if (maskIt.Get() == 1)
//     //     { // If the current pixel is within the ROI
//     //         inputIt.Set(finalIt.Get());
//     //     }
//     // }
// }

// Function to add and average two images
ImageType3D::Pointer addAndAverage(ImageType3D::Pointer image1, ImageType3D::Pointer image2)
{
    typedef itk::AddImageFilter<ImageType3D> AddImageFilterType;
    typedef itk::DivideImageFilter<ImageType3D, ImageType3D, ImageType3D> DivideImageFilterType;

    // Add the two images
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(image1);
    addFilter->SetInput2(image2);
    addFilter->Update();

    // Divide the result by 2 to average
    DivideImageFilterType::Pointer divideFilter = DivideImageFilterType::New();
    divideFilter->SetInput(addFilter->GetOutput());
    divideFilter->SetConstant(2.0);
    divideFilter->Update();

    return divideFilter->GetOutput();
}

// Wrapper function to estimate the ROI by averaging row and column estimates
void estimateROI(ImageType3D::Pointer inputImage, ImageType3D::Pointer maskImage, int fitOrder)
{
    typedef itk::ImageDuplicator<ImageType3D> DuplicatorType;

    // Duplicate the inputImage for row-wise estimation
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(inputImage);
    duplicator->Update();
    ImageType3D::Pointer rowEstimate = duplicator->GetOutput();

    // Estimate unknown values row-wise
    estimateUnknownRows(rowEstimate, maskImage, fitOrder);

    // Duplicate the inputImage again for column-wise estimation
    duplicator->Update();
    ImageType3D::Pointer colEstimate = duplicator->GetOutput();

    // Estimate unknown values column-wise
    estimateUnknownColumns(colEstimate, maskImage, fitOrder);

    // Add and average the row and column estimates
    ImageType3D::Pointer finalEstimate = addAndAverage(rowEstimate, colEstimate);

    // Create an iterator to modify only the ROI in the inputImage
    itk::ImageRegionIteratorWithIndex<ImageType3D> inputIt(inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<ImageType3D> finalIt(finalEstimate, finalEstimate->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<ImageType3D> maskIt(maskImage, maskImage->GetLargestPossibleRegion());

    for (inputIt.GoToBegin(), finalIt.GoToBegin(), maskIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++finalIt, ++maskIt)
    {
        if (maskIt.Get() == 1)
        { // If the current pixel is within the ROI
            inputIt.Set(finalIt.Get());
        }
    }
}

// Define a struct to hold the collimator data
struct CollimatorOpening // Not the same collimator object "CollimatorInfo" being used for the simulation
{
    float x, y, width, height;
};

// Assuming definitions for CollimatorOpening, Detector, and CTGeometry are provided as before

ImageType::Pointer CreateCollimatorMask(
    const std::vector<CollimatorOpening> &collimatorOpenings, int pixelsX, int pixelsY,
    float SDD, float SCD, float sourceX, float sourceY, float detectorPixelPitch)
{
    // Create a 3D ITK image
    ImageType::Pointer maskImage = ImageType::New();

    // Defining the size of the 3D image
    ImageType::SizeType size;
    size[0] = pixelsX;                   // size along X
    size[1] = pixelsY;                   // size along Y
    size[2] = collimatorOpenings.size(); // size along Z

    std::cout << "Dimension of Collimator Mask: " << size[0] << ", " << size[1] << ", " << size[2] << "." << std::endl;

    // Setting the regions of the image
    ImageType::RegionType region;
    region.SetSize(size);

    // Allocating memory for the image
    maskImage->SetRegions(region);
    maskImage->Allocate();
    maskImage->FillBuffer(0); // Initializing with zeros

    // For each collimator opening, projecting to detector and create a slice
    for (int z = 0; z < collimatorOpenings.size(); ++z)
    {
        const auto &opening = collimatorOpenings[z];

        // Projecting the collimator corners onto the detector plane
        float collimatorDistanceRatio = SDD / SCD;
        float projectedWidth = opening.width * collimatorDistanceRatio;
        float projectedHeight = opening.height * collimatorDistanceRatio;
        float projectedX = sourceX + (opening.x - sourceX) * collimatorDistanceRatio;
        float projectedY = sourceY + (opening.y - sourceY) * collimatorDistanceRatio;

        // Defining the projected collimator corners on the detector
        float cut = 10;
        float startX = ((pixelsX * detectorPixelPitch) / 2.0) + projectedX + cut;
        float endX = startX + projectedWidth - cut * 2;
        float startY = ((pixelsY * detectorPixelPitch) / 2.0) - projectedY + cut;
        float endY = startY + projectedHeight - cut * 2;

        // Iterate through the pixels of the detector
        for (int y = 0; y < pixelsY; ++y)
        {
            for (int x = 0; x < pixelsX; ++x)
            {
                float pixelCenterX = (x + 0.5) * detectorPixelPitch;
                float pixelCenterY = (y + 0.5) * detectorPixelPitch;

                if (pixelCenterX >= startX && pixelCenterX <= endX &&
                    pixelCenterY >= startY && pixelCenterY <= endY)
                {
                    ImageType::IndexType pixelIndex = {{x, y, z}};
                    maskImage->SetPixel(pixelIndex, 1);
                }
            }
        }
    }

    return maskImage;
}

void ApplyMaskAndAdd(
    const ImageType::Pointer collimatorMask,
    const ImageType::Pointer boostedImage,
    ImageType::Pointer unboostedImage)
{
    // Check if all images are of the same size
    if (collimatorMask->GetLargestPossibleRegion().GetSize() != boostedImage->GetLargestPossibleRegion().GetSize() ||
        boostedImage->GetLargestPossibleRegion().GetSize() != unboostedImage->GetLargestPossibleRegion().GetSize())
    {
        throw std::runtime_error("Collimator Mask and Images must be of the same size");
    }

    // Define iterators for the mask, the boosted image, and the unboosted image
    itk::ImageRegionIterator<ImageType> maskIterator(collimatorMask, collimatorMask->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> boostIterator(boostedImage, boostedImage->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> unboostIterator(unboostedImage, unboostedImage->GetLargestPossibleRegion());

    // Iterate through the mask image
    while (!maskIterator.IsAtEnd())
    {
        // Check the mask value
        if (maskIterator.Get() == 1)
        {
            // If the mask value is 1, add the boosted image value to the unboosted image
            ImageType::PixelType boostedValue = boostIterator.Get();
            ImageType::PixelType unboostedValue = unboostIterator.Get();
            boostedValue = static_cast<ImageType::PixelType>(unboostedValue + boostedValue);
            unboostIterator.Set(boostedValue);
        }

        // Move to the next pixel in each image
        ++maskIterator;
        ++boostIterator;
        ++unboostIterator;
    }
}

bool ConvertMHAtoNRRD(const std::string &inputFileName)
{
    std::string outputFileName = inputFileName;
    size_t lastdot = outputFileName.find_last_of(".");
    if (lastdot != std::string::npos)
        outputFileName = outputFileName.substr(0, lastdot) + ".nrrd";

    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(inputFileName);

    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outputFileName);

    writer->SetInput(reader->GetOutput());

    try
    {
        writer->Update();
    }
    catch (const itk::ExceptionObject &error)
    {
        std::cerr << "Error converting file: " << error << std::endl;
        return false;
    }

    return true;
}

// ./imgProcess         := loads, processes and prepares the "unboosted" mha 3D image file(s) for reconstruction
// ./imgProcess boosted := loads, processes and prepares the "boosted" mha 3D image file(s) for reconstruction
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
    //////////////////// READ IMAGE FILES ///////////////////
    // SCATTER IMAGE
    std::string inputDir = "/home/tunok/Work/mcDataIO_main/tests/output/";
    std::string outputDir = "/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidTest/";

    std::string pathToConfig = inputDir + "simInfo.txt";
    YAML::Node configFile = YAML::LoadFile(pathToConfig);

    int nPhotons = configFile["nPhotons"].as<int>();
    float SAD = configFile["SAD"].as<float>();
    float SDD = configFile["SDD"].as<float>();
    float SCD = configFile["SCD"].as<float>();
    float SPD = configFile["SPD"].as<float>();
    std::vector<int> detectorSize = configFile["DetectorSz"].as<std::vector<int>>();
    float PixelPitch = configFile["PixelPitch"].as<float>();
    int nProjections = configFile["nProjections"].as<int>();

    // Read collimator openings
    std::cout << "Reading Collimator Openings ..." << std::endl;
    std::vector<CollimatorOpening> collimatorOpenings;
    for (YAML::const_iterator it = configFile.begin(); it != configFile.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        if (key.rfind("CollimatorforAngle ", 0) == 0)
        {
            std::vector<float> openingParams = it->second.as<std::vector<float>>();
            if (openingParams.size() == 4)
            {
                CollimatorOpening opening;
                opening.x = openingParams[0];
                opening.y = openingParams[1];
                opening.width = openingParams[2];
                opening.height = openingParams[3];
                collimatorOpenings.push_back(opening);
            }
        }
    }
    std::cout << "Done Reading Collimator Openings." << std::endl;

    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();

    // Create the Collimator Mask
    float sourceX = 0;
    float sourceY = 0;
    ImageType::Pointer collimatorMask = CreateCollimatorMask(collimatorOpenings, detectorSize[0], detectorSize[1], SDD, SCD, sourceX, sourceY, PixelPitch);
    writer->SetFileName(outputDir + "collimatorMask.mha");
    writer->SetInput(collimatorMask);
    writer->Update();
    std::cout << "Writing File Out:" + outputDir + "collimatorMask.mha" << std::endl;

    ImageType::Pointer scatterBoostedImage;
    ImageType::Pointer scatterUnboostedImage;

    std::vector<std::string> filePaths = {
        inputDir + "scatterBoostedImage20000000Projection1.mha"}; //, inputDir + "scatterBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "scatterBoostedImage.mha");
    scatterBoostedImage = ReadMHA(outputDir + "scatterBoostedImage.mha");
    std::cout << "Read Scatter Boosted Image" << std::endl;

    typedef itk::ImageDuplicator<ImageType> DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(scatterBoostedImage);
    duplicator->Update();
    ImageType::Pointer scatterEst = duplicator->GetOutput();

    // If boosting, then estimate scatter
    // Estimate the values in the unknown region
    std::cout << "Scatter estimating ... " << std::endl;
    int fitOrder = 2;
    estimateROI(scatterEst, collimatorMask, fitOrder);

    writer->SetFileName(outputDir + "scatterEst.mha");
    writer->SetInput(scatterEst);
    writer->Update();
    std::cout << "Written File Out: " + outputDir + "scatterEst.mha" << std::endl;

    //////////////////// READ IMAGE FILES ///////////////////
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    //////////////////// SCATTER ESTIMATE ///////////////////
    // Estimate the values in the unknown region

    /* Lateral Smoothing of ScatterEstimation
std::cout << "Lateral Smoothing Scatter ... " << std::endl;
double variance = 7;
LateralSmoothing(scatterImage, variance);

// Further Scatter Estimate by doing PROJECTION-TO-PROJECTION-SMOOTHING by theta_filter
std::cout << "Projection-To-Projection Smoothing Scatter ... " << std::endl;
float theta_filt = 1.0f; // 0 - <1: Causal 1 - 40: Non-causal
unsigned int width = 2;  // Choose your filter width
CausalRecursiveFilter(scatterImage, theta_filt);
// ImageType::Pointer scatterEst = NonCausalFilter(scatterLatSmooth, width); /**/
    //////////////////// WRITE FILES OUT ////////////////////
    /////////////////////////////////////////////////////////
}
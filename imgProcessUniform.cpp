#include <vector>
#include <string>
#include <math.h>
#include <yaml-cpp/yaml.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include "itkMeanImageFilter.h"
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
typedef itk::Image<float, 1> ImageType1D;

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

void AddMultipleImages(const std::vector<std::string> &filePaths, const std::string &outputFile)
{
    if (filePaths.size() < 2)
    {
        throw std::runtime_error("At least two image files are required.");
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

using BSplineInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType1D>;
using LinearInterpolatorType = itk::LinearInterpolateImageFunction<ImageType1D>;

// Function to interpolate a 1D image at a specified position
float interpolate1D(ImageType1D::Pointer image, ImageType1D::IndexType idx, int interpolationOrder)
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

// void estimateUnknownRows(ImageType::Pointer image, int startUnknownRow, int endUnknownRow, int order)
// {
//     ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
//     for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
//     {
//         for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
//         {
//             // Collect the known and unknown values for this column
//             std::vector<float> values(imageSize[1]);
//             std::vector<int> unknownIndices;
//             for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
//             {
//                 IndexType index = {{colIdx, rowIdx, sliceId×}};
//                 values[rowIdx] = image->GetPixel(index);
//                 if (rowldx >= startUnknownRow && rowIdx <= endUnknownRow)
//                 {
//                     unknownIndices.push_ back(rowIdx);
//                 }
//             }
//             // Estimate the unknown values
//             std::vector<float> estimatedValues = estimateUnknownValues(values, unknownIndices, order);
//             // Set the estimated values
//             for (int rowIdx = startUnknownRow; rowId <= endUnknownRow; ++rowIdx)
//             {
//                 IndexType index = {{colIdx, rowIdx, sliceIdx}};
//                 image->SetPixel(index, estimatedValues[rowIdx]);
//             }
//         }
//     }
// }

void estimateUnknownRows(ImageType::Pointer testImage, ImageType::Pointer maskImage, int order)
{
    ImageType::SizeType imageSize = testImage->GetLargestPossibleRegion().GetSize();

    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            std::vector<float> columnValues(imageSize[1], 0); // Initialize with value 0
            std::vector<int> unknownIndices;

            // Collect the known and unknown values for this column
            for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
            {
                ImageType::IndexType index = {{colIdx, rowIdx, sliceIdx}};
                columnValues[rowIdx] = testImage->GetPixel(index);
                if (maskImage->GetPixel(index) == 1)
                {
                    unknownIndices.push_back(rowIdx);
                }
            }

            // Skip columns that do not have any unknown values
            if (unknownIndices.empty())
                continue;

            // Estimate the unknown values
            std::vector<float> estimatedValues = estimateUnknownValues(columnValues, unknownIndices, order);

            // Set the estimated values back into the testImage
            for (size_t idx = 0; idx < unknownIndices.size(); ++idx)
            {
                int rowIdx = unknownIndices[idx];
                ImageType::IndexType index = {{colIdx, rowIdx, sliceIdx}};
                testImage->SetPixel(index, estimatedValues[idx]);
            }
        }
    }
}

void estimateUnknownColumns(ImageType::Pointer image, int startUnknownCol, int endUnknownCol, int order)
{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
        {
            // Collect the known and unknown values for this row
            std::vector<float> values(imageSize[0]);
            std::vector<int> unknownIndices;
            for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
            {
                IndexType index = {{colIdx, rowIdx, sliceIdx}};
                values[colIdx] = image->GetPixel(index);
                if (colIdx >= startUnknownCol && colIdx <= endUnknownCol)
                {
                    unknownIndices.push_back(colIdx);
                }
            }

            // Estimate the unknown values
            std::vector<float> estimatedValues = estimateUnknownValues(values, unknownIndices, order);

            // Set the estimated values
            for (int colIdx = startUnknownCol; colIdx <= endUnknownCol; ++colIdx)
            {
                IndexType index = {{colIdx, rowIdx, sliceIdx}};
                image->SetPixel(index, estimatedValues[colIdx]);
            }
        }
    }
}

void estimateROI(ImageType::Pointer image, int startUnknownRow, int endUnknownRow, int startUnknownCol, int endUnknownCol, int order, int sliceIdx)
{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    // Estimate unknown columns for the specified slice
    for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
    {
        std::vector<float> values(imageSize[0]);
        std::vector<int> unknownIndices;
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            IndexType index = {{colIdx, rowIdx, sliceIdx}};
            values[colIdx] = image->GetPixel(index);
            if (colIdx >= startUnknownCol && colIdx <= endUnknownCol)
            {
                unknownIndices.push_back(colIdx);
            }
        }

        std::vector<float> colEstimatedValues = estimateUnknownValues(values, unknownIndices, order);
        for (int colIdx = startUnknownCol; colIdx <= endUnknownCol; ++colIdx)
        {
            IndexType index = {{colIdx, rowIdx, sliceIdx}};
            image->SetPixel(index, colEstimatedValues[colIdx]);
        }
    }

    // Estimate unknown rows for the specified slice
    for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
    {
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

        std::vector<float> rowEstimatedValues = estimateUnknownValues(values, unknownIndices, order);
        for (int rowIdx = startUnknownRow; rowIdx <= endUnknownRow; ++rowIdx)
        {
            IndexType index = {{colIdx, rowIdx, sliceIdx}};
            float averagedValue = (image->GetPixel(index) + rowEstimatedValues[rowIdx]) / 2;
            image->SetPixel(index, averagedValue);
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
        float cut = 0;
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

    std::string inputDir = "/home/tunok/Work/mcDataIO_main/tests/output/IsoHaroldProstUniform/";
    std::string outputDir = "/home/tunok/Work/imgProcess_main/tests/IsoHaroldProstUniform/";

    /////////////////////////////////////////////////////////
    ///////////////////// READ CT Params ////////////////////
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

    // Define the writer type
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();

    using WriterType2D = itk::ImageFileWriter<ImageType2D>;
    WriterType2D::Pointer writer2D = WriterType2D::New();

    // Create the Collimator Mask
    float sourceX = 0;
    float sourceY = 0;
    // ImageType::Pointer collimatorMask = CreateCollimatorMask(collimatorOpenings, detectorSize[0], detectorSize[1], SDD, SCD, sourceX, sourceY, PixelPitch);
    writer->SetFileName(outputDir + "collimatorMask.mha");
    // writer->SetInput(collimatorMask);
    // writer->Update();
    std::cout << "Writing File Out:" + outputDir + "collimatorMask.mha" << std::endl;

    std::cout << "nPhotons: " << nPhotons << std::endl;
    std::cout << "SAD: " << SAD << std::endl;
    std::cout << "SDD: " << SDD << std::endl;
    std::cout << "SCD: " << SCD << std::endl;
    std::cout << "SPD: " << SPD << std::endl;
    std::cout << "PixelPitch: " << PixelPitch << std::endl;
    std::cout << "Detector Size: [" << detectorSize[0] << ", " << detectorSize[1] << "]" << std::endl;
    std::cout << "nProjections: " << nProjections << std::endl; /**/
    ///////////////////// READ CT Params ////////////////////
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    ///////////////////// GEOMETRY XML //////////////////////

    // Set the parameters
    std::cout << "Writing Geometry ... " << std::endl;
    using GeometryType = rtk::ThreeDCircularProjectionGeometry;
    GeometryType::Pointer geometry = GeometryType::New();

    double sid = SAD;          // source to isocenter distance
    double sdd = SDD;          // source to detector distance
    double start_angle = 0.0;  // start angle
    double stop_angle = 360.0; // stop angle
    std::cout << "nProjections: " << nProjections << std::endl;
    unsigned int num_projections = nProjections - 1; // number of projections
    std::cout << "num_Projections: " << num_projections << std::endl;

    double angle_step = (stop_angle - start_angle) / (num_projections);
    for (unsigned int i = 0; i <= num_projections; i++)
    {
        double angle = start_angle + i * angle_step;
        geometry->AddProjection(sid, sdd, angle, 0, 0, 0, 0, 0, 0); // Positive angle flips the object front to back
    }

    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter;
    xmlWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
    xmlWriter->SetFilename(outputDir + "Geometry.xml");
    xmlWriter->SetObject(geometry);
    xmlWriter->WriteFile();
    std::cout << "Written File Out: " + outputDir + "Geometry.xml" << std::endl;

    ///////////////////// GEOMETRY XML //////////////////////
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    //////////////////// READ IMAGE FILES ///////////////////

    ImageType::Pointer totalBoostedImage; // primary + scatter + tertiary
    ImageType::Pointer scatterBoostedImage;
    ImageType::Pointer floodBoostedImage;
    ImageType::Pointer tertiaryBoostedImage;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read totalBoosted ///////////////////////////////////////////////////

    std::vector<std::string> filePaths = {
        inputDir + "totalBoostedImage300000000Projection9.mha",
        inputDir + "totalBoostedImage300000000Projection9_2.mha",
        inputDir + "totalBoostedImage300000000Projection9_3.mha",
        inputDir + "totalBoostedImage90000000Projection9.mha"}; //, inputDir + "totalBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "totalBoostedImage.mha");
    totalBoostedImage = ReadMHA(outputDir + "totalBoostedImage.mha");
    std::cout << "Read Total Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read scatterBoosted /////////////////////////////////////////////////

    filePaths = {
        inputDir + "scatterBoostedImage300000000Projection9.mha",
        inputDir + "scatterBoostedImage300000000Projection9_2.mha",
        inputDir + "scatterBoostedImage300000000Projection9_3.mha",
        inputDir + "scatterBoostedImage90000000Projection9.mha"}; //, inputDir + "scatterBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "scatterBoostedImage.mha");
    scatterBoostedImage = ReadMHA(outputDir + "scatterBoostedImage.mha");
    std::cout << "Read Scatter Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read tertiaryBoosted ////////////////////////////////////////////////

    filePaths = {
        inputDir + "tertiaryBoostedImage300000000Projection9.mha",
        inputDir + "tertiaryBoostedImage300000000Projection9_2.mha",
        inputDir + "tertiaryBoostedImage300000000Projection9_3.mha",
        inputDir + "tertiaryBoostedImage90000000Projection9.mha"}; //, inputDir + "tertiaryBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "tertiaryBoostedImage.mha");
    tertiaryBoostedImage = ReadMHA(outputDir + "tertiaryBoostedImage.mha");
    std::cout << "Read Tertiary Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "floodBoostedImage300000000Projection9.mha",
        inputDir + "floodBoostedImage300000000Projection9_2.mha",
        inputDir + "floodBoostedImage300000000Projection9_3.mha",
        inputDir + "floodBoostedImage90000000Projection9.mha"}; //, inputDir + "floodBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "floodBoostedImage.mha");
    floodBoostedImage = ReadMHA(outputDir + "floodBoostedImage.mha");
    std::cout << "Read Flood Boosted Image" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (true) // this is to execute the image process section or to calculate CNR values
    {
        /////////////////////////////////////////////////////////
        //////////////////// SCATTER ESTIMATE ///////////////////
        // Define the ROI region (that is a misrepresentation region of the scatter signal i.e. we want to exclude this region
        // Create a copy of the scatter image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;
        DuplicatorType::Pointer duplicator = DuplicatorType::New();
        duplicator->SetInputImage(scatterBoostedImage);
        duplicator->Update();
        ImageType::Pointer scatterEst = duplicator->GetOutput();

        // If boosting, then estimate scatter
        // Estimate the values in the unknown region
        std::cout << "Interpolating ... " << std::endl;
        int interpolationOrder = 2;
        // estimateUnknownRows(scatterEst, collimatorMask, interpolationOrder);

        writer->SetFileName(outputDir + "scatterEst.mha");
        writer->SetInput(scatterEst);
        writer->Update();
        std::cout << "Written File Out: " + outputDir + "scatterEst.mha" << std::endl;
        //////////////////// SCATTER ESTIMATE ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// IMAGE CORRECTION ///////////////////

        std::cout << "Scatter Correcting Original Images ... " << std::endl;
        //// Get PrimaryScatter, PrimaryTertiary and Primary Images ////
        /////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// BOOSTED //////////////////////////////////////
        using SubtractFilterType = itk::SubtractImageFilter<ImageType>;
        SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
        subtractFilter->SetInput1(totalBoostedImage);
        subtractFilter->SetInput2(scatterBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryTertiaryBoostedImage = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
        primaryTertiaryBoostedImage->DisconnectPipeline();

        subtractFilter->SetInput1(totalBoostedImage);
        subtractFilter->SetInput2(tertiaryBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryScatterBoostedImage = subtractFilter->GetOutput(); // total - tertiary = nonTertiary = primary + scatter
        primaryScatterBoostedImage->DisconnectPipeline();

        subtractFilter->SetInput1(primaryTertiaryBoostedImage);
        subtractFilter->SetInput2(tertiaryBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryBoostedImage = subtractFilter->GetOutput(); // This "primaryImage" is the purely primary
        primaryBoostedImage->DisconnectPipeline();

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////// Correcting Total Image by Flood Image  ////////////////////////////////
        //////////////////////////////// Boosted ////////////////////////////////
        std::cout << "Flood Correcting Scatter Corrected Images ... " << std::endl;
        using DivideFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
        DivideFilterType::Pointer divideFilter = DivideFilterType::New();
        divideFilter->SetInput1(primaryBoostedImage);
        divideFilter->SetInput2(floodBoostedImage);
        divideFilter->Update();
        primaryBoostedImage = divideFilter->GetOutput();
        primaryBoostedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryTertiaryBoostedImage);
        divideFilter->SetInput2(floodBoostedImage);
        divideFilter->Update();
        primaryTertiaryBoostedImage = divideFilter->GetOutput();
        primaryTertiaryBoostedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryScatterBoostedImage);
        divideFilter->SetInput2(floodBoostedImage);
        divideFilter->Update();
        primaryScatterBoostedImage = divideFilter->GetOutput();
        primaryScatterBoostedImage->DisconnectPipeline();

        divideFilter->SetInput1(totalBoostedImage);
        divideFilter->SetInput2(floodBoostedImage);
        divideFilter->Update();
        totalBoostedImage = divideFilter->GetOutput();
        totalBoostedImage->DisconnectPipeline();

        ///////////////////////////////////////////////////////////////////////////
        /////////////////////// ReScale an image if needed ////////////////////////
        /*std::cout << "Rescaling Corrected Images ... " << std::endl;
        // typedef for the RescaleIntensityImageFilter
        using FilterType = itk::MultiplyImageFilter<ImageType, ImageType, ImageType>;
        auto multiplyConstantFilter = FilterType::New();
        multiplyConstantFilter->SetInput(primaryBoostedImage);
        multiplyConstantFilter->SetConstant(2);
        multiplyConstantFilter->Update();
        primaryBoostedImage = multiplyConstantFilter->GetOutput();
        primaryBoostedImage->DisconnectPipeline();

        multiplyConstantFilter->SetInput(primaryTertiaryBoostedImage);
        multiplyConstantFilter->SetConstant(2);
        multiplyConstantFilter->Update();
        primaryTertiaryBoostedImage = multiplyConstantFilter->GetOutput();
        primaryTertiaryBoostedImage->DisconnectPipeline();

        multiplyConstantFilter->SetInput(primaryScatterBoostedImage);
        multiplyConstantFilter->SetConstant(2);
        multiplyConstantFilter->Update();
        primaryScatterBoostedImage = multiplyConstantFilter->GetOutput();
        primaryScatterBoostedImage->DisconnectPipeline();

        multiplyConstantFilter->SetInput(totalBoostedImage);
        multiplyConstantFilter->SetConstant(2);
        multiplyConstantFilter->Update();
        primaryScatterBoostedImage = multiplyConstantFilter->GetOutput();
        primaryScatterBoostedImage->DisconnectPipeline();/**/

        ///////////////////////////////////////////////////////////////////////////
        ////////////////////////// MASK BOOSTED IMAGES ////////////////////////////
        std::cout << "Applying Mask and setting Boost Image." << std::endl;
        // ApplyMaskAndAdd(collimatorMask, primaryBoostedImage, primaryBoostedMaskedImage);
        // ApplyMaskAndAdd(collimatorMask, totalBoostedImage, totalBoostedMaskedImage);
        // ApplyMaskAndAdd(collimatorMask, primaryTertiaryBoostedImage, primaryTertiaryBoostedMaskedImage);
        // ApplyMaskAndAdd(collimatorMask, primaryScatterBoostedImage, primaryScatterBoostedMaskedImage);
        std::cout << "Done applying Mask." << std::endl;

        //////////////////// IMAGE CORRECTION ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        std::cout
            << "Setting size and origin ... " << std::endl;
        // Change size
        ImageType::SizeType size;
        size[0] = detectorSize[0]; // New size in the x direction in pixels
        size[1] = detectorSize[1]; // New size in the y direction in pixels
        size[2] = nProjections;    // New size in the z direction
        std::cout << "Detector size: " << size[0] << ", " << size[1] << "." << std::endl;
        primaryBoostedImage->SetRegions(size);
        totalBoostedImage->SetRegions(size);
        primaryTertiaryBoostedImage->SetRegions(size);
        primaryScatterBoostedImage->SetRegions(size);

        // Change spacing
        ImageType::SpacingType spacing;
        spacing[0] = PixelPitch; // New spacing in the x direction in mm
        spacing[1] = PixelPitch; // New spacing in the y direction in mm
        spacing[2] = 1;          // New spacing in the z direction
        primaryBoostedImage->SetSpacing(spacing);
        totalBoostedImage->SetSpacing(spacing);
        primaryTertiaryBoostedImage->SetSpacing(spacing);
        primaryScatterBoostedImage->SetSpacing(spacing);

        // Change origin
        ImageType::PointType newOrigin;

        newOrigin[0] = -((size[0] * PixelPitch / 2) - (spacing[0] / 2)); // new x origin in mm
        newOrigin[1] = -((size[1] * PixelPitch / 2) - (spacing[1] / 2)); //-((size[1] / 2) - (spacing[1] / 2)); // new y origin in mm
        newOrigin[2] = 1;                                                // new z origin
        std::cout << "newOrigin: " << newOrigin[0] << ", " << newOrigin[1] << "." << std::endl;
        primaryBoostedImage->SetOrigin(newOrigin);
        totalBoostedImage->SetOrigin(newOrigin);
        primaryTertiaryBoostedImage->SetOrigin(newOrigin);
        primaryScatterBoostedImage->SetOrigin(newOrigin);

        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// WRITE FILES OUT ////////////////////

        writer->SetFileName(outputDir + "primaryBoostedImage.mha");
        writer->SetInput(primaryBoostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryBoostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "totalBoostedImage.mha");
        writer->SetInput(totalBoostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "totalBoostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryTertiaryBoostedImage.mha");
        writer->SetInput(primaryTertiaryBoostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryTertiaryBoostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryScatterBoostedImage.mha");
        writer->SetInput(primaryScatterBoostedImage);
        writer->Update();
        std::cout << "Writing File Out:" + outputDir + "primaryScatterBoostedImage.mha" << std::endl;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //////////////////// WRITE FILES OUT ////////////////////
        /////////////////////////////////////////////////////////
    }

    if (false)
    {

        /////////////////////////////////////////////////////////
        //////////////////// WRITE VALUES OUT ///////////////////
        // Read Image file: Good Corrected Reconstructed Volume
        ImageType::Pointer goodVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/primaryBoostedVolumeBrainTumor0.6.mha");
        // Read Image file: Bad Corrected Reconstructed Volume
        ImageType::Pointer badVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/primaryUnboostedVolume.mha");

        // Calculate the local SNR and get the 2D image
        int sliceIndex = 105;
        ImageType2D::Pointer snrImage = calculateLocalSNR(goodVolume, sliceIndex);

        writer2D->SetFileName("/home/tunok/Work/imgProcess_main/tests/relSnrImage.mha");
        writer2D->SetInput(snrImage);
        writer2D->Update();
        std::cout << "Writing File Out: relSnrImage.mha" << std::endl;

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
        if (false)
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
        if (false)
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
    }

    ////////////////////////////////////// Calculating NOISE ///////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////// WRITE VALUES OUT ///////////////////
    /////////////////////////////////////////////////////////
    std::cout
        << "Press ENTER to continue... " << std::flush;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    return 0;
}

// running this program should output the string: "Image dimensions: 250 x 250 x 401" (as of August 9th, 2023)
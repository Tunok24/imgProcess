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

void PrintImageDimensions(ImageType::Pointer image)
{
    // Print size, spacing, origin
    typename ImageType::SizeType imgSize = image->GetLargestPossibleRegion().GetSize();
    const typename ImageType::SpacingType &imgSpacing = image->GetSpacing();
    const typename ImageType::PointType &imgOrigin = image->GetOrigin();

    std::cout << "Image Size: " << imgSize[0] << ", " << imgSize[1] << ", " << imgSize[2] << "." << std::endl;
    std::cout << "Image Spacing: " << imgSpacing[0] << ", " << imgSpacing[1] << ", " << imgSpacing[2] << "." << std::endl;
    std::cout << "Image Origin: " << imgOrigin[0] << ", " << imgOrigin[1] << ", " << imgOrigin[2] << "." << std::endl;
}

ImageType::Pointer ReadMHA(const std::string &filename)
{
    // Define the image type and create a reader
    // using ImageType = itk::Image<float, 3>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);

    reader->Update();

    ImageType::Pointer image = reader->GetOutput();

    return image;
}

void SetImageDimensions(ImageType::Pointer image, int sizeX, int sizeY, int nProjections, float PixelPitch)
{
    ImageType::SizeType size;
    size[0] = sizeX;
    size[1] = sizeY;
    size[2] = nProjections;

    ImageType::SpacingType spacing;
    spacing[0] = PixelPitch;
    spacing[1] = PixelPitch;
    spacing[2] = 1;

    ImageType::PointType origin;
    origin[0] = -((size[0] * PixelPitch / 2) - (spacing[0] / 2));
    origin[1] = -((size[1] * PixelPitch / 2) - (spacing[1] / 2));
    origin[2] = 1;

    image->SetRegions(size);
    image->SetSpacing(spacing);
    image->SetOrigin(origin);
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
        return;
    }

    typedef itk::ImageDuplicator<ImageType> DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(filter->GetOutput());
    duplicator->Update();

    scatterEstimate = duplicator->GetOutput();
}

// AddImages: adds the two images pixelwise
ImageType::Pointer AddImages(ImageType::Pointer image1, ImageType::Pointer image2)
{
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

ImageType::Pointer AddMultipleImages(const std::vector<std::string> &filePaths)
{
    ImageType::Pointer currentImage = ReadMHA(filePaths[0]);

    if (filePaths.size() < 2)
    {
        std::cout << ("At least two image files are required to add. Returning the provided single image.") << std::endl;
        return currentImage;
    }

    for (size_t i = 1; i < filePaths.size(); ++i)
    {
        ImageType::Pointer nextImage = ReadMHA(filePaths[i]);
        currentImage = AddImages(currentImage, nextImage);
    }

    return currentImage;
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

// Function to add and average two images
ImageType::Pointer addAndAverage(ImageType::Pointer image1, ImageType::Pointer image2)
{
    typedef itk::AddImageFilter<ImageType> AddImageFilterType;
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideImageFilterType;

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

    // Disconnect from the pipeline
    ImageType::Pointer resultImage = divideFilter->GetOutput();
    resultImage->DisconnectPipeline();

    return resultImage;
}

void estimateROI(ImageType::Pointer inputImage, ImageType::Pointer maskImage, int fitOrder)
{
    typedef itk::ImageDuplicator<ImageType> DuplicatorType;

    // Duplicate inputImage for row-wise estimation
    DuplicatorType::Pointer rowDuplicator = DuplicatorType::New();
    rowDuplicator->SetInputImage(inputImage);
    rowDuplicator->Update();
    ImageType::Pointer rowEstimate = rowDuplicator->GetOutput();
    rowEstimate->DisconnectPipeline(); // Disconnect to ensure independent manipulation
    estimateUnknownRows(rowEstimate, maskImage, fitOrder);

    // Duplicate inputImage for column-wise estimation
    DuplicatorType::Pointer colDuplicator = DuplicatorType::New();
    colDuplicator->SetInputImage(inputImage);
    colDuplicator->Update();
    ImageType::Pointer colEstimate = colDuplicator->GetOutput();
    colEstimate->DisconnectPipeline(); // Disconnect to ensure independent manipulation
    estimateUnknownColumns(colEstimate, maskImage, fitOrder);

    // Add and average row and column estimates
    ImageType::Pointer finalEstimate = addAndAverage(rowEstimate, colEstimate);

    // Update inputImage with the final estimate within the ROI
    itk::ImageRegionIteratorWithIndex<ImageType> inputIt(inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<ImageType> finalIt(finalEstimate, finalEstimate->GetLargestPossibleRegion());
    itk::ImageRegionIteratorWithIndex<ImageType> maskIt(maskImage, maskImage->GetLargestPossibleRegion());

    for (inputIt.GoToBegin(), finalIt.GoToBegin(), maskIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++finalIt, ++maskIt)
    {
        if (maskIt.Get() == 1)
        { // Only update pixels within the ROI
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
        float cut = 12;
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

ImageType::Pointer ApplyCollimatorMask(
    const ImageType::Pointer collimatorMask,
    const ImageType::Pointer inputImage)
{
    // Check if the mask and the input image are of the same size
    if (collimatorMask->GetLargestPossibleRegion().GetSize() !=
        inputImage->GetLargestPossibleRegion().GetSize())
    {
        throw std::runtime_error("Collimator Mask and the Input Image must be of the same size");
    }

    ImageType::Pointer outputImage = ImageType::New();
    outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
    outputImage->Allocate();
    outputImage->FillBuffer(0);
    outputImage->SetSpacing(inputImage->GetSpacing());
    outputImage->SetOrigin(inputImage->GetOrigin());

    // Define iterators for the mask and the input image
    itk::ImageRegionIterator<ImageType> maskIterator(collimatorMask, collimatorMask->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> inputIterator(inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

    // Iterate through the mask image
    while (!maskIterator.IsAtEnd())
    {
        // Check the mask value
        if (maskIterator.Get() == 1)
        {
            // If the mask value is 1, copy the input image value to the output image
            outputIterator.Set(inputIterator.Get());
        }

        // Move to the next pixel in each image
        ++maskIterator;
        ++inputIterator;
        ++outputIterator;
    }

    return outputImage;
}

ImageType::Pointer ApplyMask(
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

    // Get the ROI values
    using SubtractFilterType = itk::SubtractImageFilter<ImageType>;
    SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
    subtractFilter->SetInput1(boostedImage);
    subtractFilter->SetInput2(unboostedImage);
    subtractFilter->Update();
    ImageType::Pointer output = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
    output->DisconnectPipeline();

    output = ApplyCollimatorMask(collimatorMask, output);

    // // For debug
    // PrintImageDimensions(collimatorMask);
    // PrintImageDimensions(output);
    // PrintImageDimensions(unboostedImage);

    // Just add the Roi Image and the Unboosted Image
    typedef itk::AddImageFilter<ImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(output);
    addFilter->SetInput2(unboostedImage);
    addFilter->Update();
    output = addFilter->GetOutput();
    output->DisconnectPipeline();

    return output;
}

ImageType::Pointer extractSlices(ImageType::Pointer image, const std::vector<int> &indices, const std::string outputFilepath)
{
    // Get input image size and properties
    ImageType::RegionType inputRegion = image->GetLargestPossibleRegion();
    ImageType::SizeType inputSize = inputRegion.GetSize();

    // Prepare the output image size and region
    ImageType::SizeType outputSize = inputSize;
    outputSize[2] = indices.size(); // Set the number of slices to the number of indices

    ImageType::RegionType outputRegion;
    outputRegion.SetSize(outputSize);

    ImageType::Pointer outputImage = ImageType::New();
    outputImage->SetRegions(outputRegion);
    outputImage->Allocate();
    outputImage->FillBuffer(0);

    // Iterate over each index in indices to copy the slices
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= inputSize[2])
        {
            std::cout << "Invalid Index for extraction." << std::endl;
            continue; // Skip invalid indices
        }

        // Define the region of the current slice
        ImageType::IndexType start = {0, 0, indices[i]};
        ImageType::RegionType desiredSlice;
        ImageType::SizeType sliceSize = inputSize;
        sliceSize[2] = 1; // Single slice
        desiredSlice.SetSize(sliceSize);
        desiredSlice.SetIndex(start);

        // Extract the slice
        itk::ImageRegionConstIterator<ImageType> inputIt(image, desiredSlice);
        ImageType::IndexType outputStart = {0, 0, static_cast<int>(i)};
        ImageType::RegionType outputSliceRegion;
        outputSliceRegion.SetSize(sliceSize);
        outputSliceRegion.SetIndex(outputStart);

        itk::ImageRegionIterator<ImageType> outputIt(outputImage, outputSliceRegion);
        for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt)
        {
            outputIt.Set(inputIt.Get());
        }
    }

    // using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outputFilepath);
    writer->SetInput(outputImage);
    writer->Update();
    std::cout << "Writing File Out:" + outputFilepath << std::endl;

    return outputImage;
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

    std::string inputDir = "/home/tunok/Work/mcDataIO_main/tests/output/CylinderEllipsoid/";
    std::string outputDir = "/home/tunok/Work/imgProcess_main/tests/CylinderEllipsoidNew/";

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
    WriterType::Pointer writer = WriterType::New();

    // Create the Collimator Mask
    float sourceX = 0;
    float sourceY = 0;
    ImageType::Pointer collimatorMask = CreateCollimatorMask(collimatorOpenings, detectorSize[0], detectorSize[1], SDD, SCD, sourceX, sourceY, PixelPitch);
    SetImageDimensions(collimatorMask, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    writer->SetFileName(outputDir + "collimatorMask.mha");
    writer->SetInput(collimatorMask);
    writer->Update();
    std::cout << "Written File Out:" + outputDir + "collimatorMask.mha" << std::endl;

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

    ImageType::Pointer totalBoostedImage;
    ImageType::Pointer scatterBoostedImage;
    ImageType::Pointer floodBoostedImage;
    ImageType::Pointer tertiaryBoostedImage;

    ImageType::Pointer totalUnboostedImage; // primary + scatter + tertiary
    ImageType::Pointer scatterUnboostedImage;
    ImageType::Pointer floodUnboostedImage;
    ImageType::Pointer tertiaryUnboostedImage;

    ImageType::Pointer primaryTertiaryEstimatedImage;

    ImageType::Pointer doseBoostedImage;
    ImageType::Pointer doseUnboostedImage;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read totalBoosted ///////////////////////////////////////////////////

    std::vector<std::string> filePaths = {
        inputDir + "totalBoostedImage290000000Projection9.mha",
        inputDir + "totalBoostedImage350000000Projection9.mha",
        // inputDir + "totalBoostedImage200000000Projection9_2.mha",
        // inputDir + "totalBoostedImage200000000Projection9_3.mha",
        inputDir + "totalBoostedImage350000000Projection9_2.mha"}; //, inputDir + "totalBoostedImage200000000Projection9_3.mha"};
    totalBoostedImage = AddMultipleImages(filePaths);
    std::cout << "Read Total Boosted Image" << std::endl;

    writer->SetFileName(inputDir + "totalBoostedImage.mha");
    writer->SetInput(totalBoostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "totalBoostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read scatterBoosted /////////////////////////////////////////////////

    filePaths = {
        inputDir + "scatterBoostedImage290000000Projection9.mha",
        inputDir + "scatterBoostedImage350000000Projection9.mha",
        // inputDir + "scatterBoostedImage200000000Projection9_2.mha",
        // inputDir + "scatterBoostedImage200000000Projection9_3.mha",
        inputDir + "scatterBoostedImage350000000Projection9_2.mha"}; //, inputDir + "scatterBoostedImage200000000Projection9_3.mha"};
    scatterBoostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(scatterBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Scatter Boosted Image" << std::endl;

    writer->SetFileName(inputDir + "scatterBoostedImage.mha");
    writer->SetInput(scatterBoostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "scatterBoostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read tertiaryBoosted ////////////////////////////////////////////////

    filePaths = {
        inputDir + "tertiaryBoostedImage290000000Projection9.mha",
        inputDir + "tertiaryBoostedImage350000000Projection9.mha",
        // inputDir + "tertiaryBoostedImage200000000Projection9_2.mha",
        // inputDir + "tertiaryBoostedImage200000000Projection9_3.mha",
        inputDir + "tertiaryBoostedImage350000000Projection9_2.mha"}; //, inputDir + "tertiaryBoostedImage200000000Projection9_3.mha"};
    tertiaryBoostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(tertiaryBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Tertiary Boosted Image" << std::endl;

    writer->SetFileName(inputDir + "tertiaryBoostedImage.mha");
    writer->SetInput(tertiaryBoostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "tertiaryBoostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "floodBoostedImage290000000Projection9.mha",
        inputDir + "floodBoostedImage350000000Projection9.mha",
        // inputDir + "floodBoostedImage200000000Projection9_2.mha",
        // inputDir + "floodBoostedImage200000000Projection9_3.mha",
        inputDir + "floodBoostedImage350000000Projection9_2.mha"};
    floodBoostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(floodBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Flood Boosted Image" << std::endl;

    writer->SetFileName(inputDir + "floodBoostedImage.mha");
    writer->SetInput(floodBoostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "floodBoostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read totalUnboosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "totalUnboostedImage290000000Projection9.mha",
        inputDir + "totalUnboostedImage350000000Projection9.mha",
        // inputDir + "totalUnboostedImage200000000Projection9_2.mha",
        // inputDir + "totalUnboostedImage200000000Projection9_3.mha",
        inputDir + "totalUnboostedImage350000000Projection9_2.mha"};
    totalUnboostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(totalUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Total Unboosted Image" << std::endl;

    writer->SetFileName(inputDir + "totalUnboostedImage.mha");
    writer->SetInput(totalUnboostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "totalUnboostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read scatterBoosted /////////////////////////////////////////////////

    filePaths = {
        inputDir + "scatterUnboostedImage290000000Projection9.mha",
        inputDir + "scatterUnboostedImage350000000Projection9.mha",
        // inputDir + "scatterUnboostedImage200000000Projection9_2.mha",
        // inputDir + "scatterUnboostedImage200000000Projection9_3.mha",
        inputDir + "scatterUnboostedImage350000000Projection9_2.mha"};
    scatterUnboostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(scatterUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Scatter Unboosted Image" << std::endl;

    writer->SetFileName(inputDir + "scatterUnboostedImage.mha");
    writer->SetInput(scatterUnboostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "scatterUnboostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read tertiaryBoosted ////////////////////////////////////////////////

    filePaths = {
        inputDir + "tertiaryUnboostedImage290000000Projection9.mha",
        inputDir + "tertiaryUnboostedImage350000000Projection9.mha",
        // inputDir + "tertiaryUnboostedImage200000000Projection9_2.mha",
        // inputDir + "tertiaryUnboostedImage200000000Projection9_3.mha",
        inputDir + "tertiaryUnboostedImage350000000Projection9_2.mha"};
    tertiaryUnboostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(tertiaryUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Tertiary Unboosted Image" << std::endl;

    writer->SetFileName(inputDir + "tertiaryUnboostedImage.mha");
    writer->SetInput(tertiaryUnboostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "tertiaryUnboostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "floodUnboostedImage290000000Projection9.mha",
        inputDir + "floodUnboostedImage350000000Projection9.mha",
        // inputDir + "floodUnboostedImage200000000Projection9_2.mha",
        // inputDir + "floodUnboostedImage200000000Projection9_3.mha",
        inputDir + "floodUnboostedImage350000000Projection9_2.mha"};
    floodUnboostedImage = AddMultipleImages(filePaths);
    // SetImageDimensions(floodUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    std::cout << "Read Flood Unboosted Image" << std::endl;

    writer->SetFileName(inputDir + "floodUnboostedImage.mha");
    writer->SetInput(floodUnboostedImage);
    writer->Update();
    std::cout << "Written File Out:" + inputDir + "floodUnboostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    // filePaths = {
    //     inputDir + "doseUnboosted190000000Projection9.mha",
    //     inputDir + "doseUnboosted200000000Projection9.mha",
    //     inputDir + "doseUnboosted200000000Projection9_2.mha",
    //     inputDir + "doseUnboosted200000000Projection9_3.mha",
    //     inputDir + "doseUnboosted200000000Projection9_4.mha"};
    // doseUnboostedImage = AddMultipleImages(filePaths);
    // // SetImageDimensions(floodUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    // std::cout << "Read Dose Unboosted Image" << std::endl;

    // writer->SetFileName(inputDir + "doseUnboostedImage.mha");
    // writer->SetInput(doseUnboostedImage);
    // writer->Update();
    // std::cout << "Written File Out:" + inputDir + "doseUnboostedImage.mha" << std::endl;

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    // filePaths = {
    //     inputDir + "doseBoosted190000000Projection9.mha",
    //     inputDir + "doseBoosted200000000Projection9.mha",
    //     inputDir + "doseBoosted200000000Projection9_2.mha",
    //     inputDir + "doseBoosted200000000Projection9_3.mha",
    //     inputDir + "doseBoosted200000000Projection9_4.mha"};
    // doseBoostedImage = AddMultipleImages(filePaths);
    // // SetImageDimensions(floodUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
    // std::cout << "Read Dose Boosted Image" << std::endl;

    // writer->SetFileName(inputDir + "doseBoostedImage.mha");
    // writer->SetInput(doseBoostedImage);
    // writer->Update();
    // std::cout << "Written File Out:" + inputDir + "doseBoostedImage.mha" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// Add in the tertiary to the unboosted iamge /////////////////////////////////////
    // Add in the tertiary to the unboosted iamge
    typedef itk::AddImageFilter<ImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    // addFilter->SetInput1(totalUnboostedImage);
    // addFilter->SetInput2(tertiaryUnboostedImage);
    // addFilter->Update();
    // totalUnboostedImage = addFilter->GetOutput();
    // totalUnboostedImage->DisconnectPipeline();
    // this is done specially only because of the small issue (now resolved) in the simulation that propagated into the big simulation results

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// EXTRACT SLICES ///////////////////////////////////////
    // ImageType::Pointer totalBoostedImage = ReadMHA(inputDir + "totalBoostedImage.mha");
    // ImageType::Pointer floodBoostedImage = ReadMHA(inputDir + "floodBoostedImage.mha");
    // ImageType::Pointer scatterBoostedImage = ReadMHA(inputDir + "scatterBoostedImage.mha");
    // ImageType::Pointer tertiaryBoostedImage = ReadMHA(inputDir + "tertiaryBoostedImage.mha");
    // std::cout << "Read Total Boosted Image" << std::endl;
    // std::vector<int> extractIndices = {1, 125, 250, 300};

    // std::string outputFileName = outputDir + "totalSlices.mha";
    // ImageType::Pointer extractedImage = extractSlices(totalBoostedImage, extractIndices, outputFileName);

    // outputFileName = outputDir + "floodSlices.mha";
    // extractedImage = extractSlices(floodBoostedImage, extractIndices, outputFileName);

    // outputFileName = outputDir + "scatterSlices.mha";
    // extractedImage = extractSlices(scatterBoostedImage, extractIndices, outputFileName);

    // outputFileName = outputDir + "tertiarySlices.mha";
    // extractedImage = extractSlices(tertiaryBoostedImage, extractIndices, outputFileName);

    //////////////////////////////////////// EXTRACT SLICES ///////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (false) // this is to execute the image process section
    {
        /////////////////////////////////////////////////////////
        //////////////////// SCATTER ESTIMATE ///////////////////
        // Define the ROI region (that is a misrepresentation region of the scatter signal i.e. we want to exclude this region
        // Create a copy of the scatter image
        typedef itk::ImageDuplicator<ImageType> DuplicatorType;
        DuplicatorType::Pointer duplicator = DuplicatorType::New();
        using SubtractFilterType = itk::SubtractImageFilter<ImageType>;
        SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
        subtractFilter->SetInput1(scatterBoostedImage);
        subtractFilter->SetInput2(scatterUnboostedImage); // this is to replicate the
        subtractFilter->Update();
        ImageType::Pointer scatterEst = subtractFilter->GetOutput();
        scatterEst->DisconnectPipeline();

        // If boosting, then estimate scatter
        std::cout << "Scatter estimating ... " << std::endl;
        int fitOrder = 2;
        estimateROI(scatterEst, collimatorMask, fitOrder);

        writer->SetFileName(outputDir + "scatterEst.mha");
        writer->SetInput(scatterEst);
        writer->Update();
        std::cout << "Written File Out: " + outputDir + "scatterEst.mha" << std::endl;
        //////////////////// SCATTER ESTIMATE ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// IMAGE CORRECTION ///////////////////

        //// Get Primary, PrimaryScatter, PrimaryTertiary, PrimaryTertiaryEstimated Images ////
        /////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// BOOSTED //////////////////////////////////////
        subtractFilter->SetInput1(totalBoostedImage);
        subtractFilter->SetInput2(scatterBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryTertiaryBoostedImage = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
        primaryTertiaryBoostedImage->DisconnectPipeline();
        SetImageDimensions(primaryTertiaryBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        subtractFilter->SetInput1(totalBoostedImage);
        subtractFilter->SetInput2(tertiaryBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryScatterBoostedImage = subtractFilter->GetOutput(); // total - tertiary = nonTertiary = primary + scatter
        primaryScatterBoostedImage->DisconnectPipeline();
        SetImageDimensions(primaryScatterBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        subtractFilter->SetInput1(primaryTertiaryBoostedImage);
        subtractFilter->SetInput2(tertiaryBoostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryBoostedImage = subtractFilter->GetOutput(); // This "primaryImage" is the purely primary
        primaryBoostedImage->DisconnectPipeline();
        SetImageDimensions(primaryBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        subtractFilter->SetInput1(totalBoostedImage);
        subtractFilter->SetInput2(scatterEst);
        subtractFilter->Update();
        ImageType::Pointer primaryTertiaryEstimatedImage = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
        primaryTertiaryEstimatedImage->DisconnectPipeline();
        SetImageDimensions(primaryTertiaryEstimatedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        /////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////// UNBOOSTED /////////////////////////////////////

        subtractFilter->SetInput1(totalUnboostedImage);
        subtractFilter->SetInput2(scatterUnboostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryTertiaryUnboostedImage = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
        primaryTertiaryUnboostedImage->DisconnectPipeline();
        SetImageDimensions(primaryTertiaryUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        subtractFilter->SetInput1(totalUnboostedImage);
        subtractFilter->SetInput2(tertiaryUnboostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryScatterUnboostedImage = subtractFilter->GetOutput(); // total - tertiary = nonTertiary = primary + scatter
        primaryScatterUnboostedImage->DisconnectPipeline();
        SetImageDimensions(primaryScatterUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        subtractFilter->SetInput1(primaryTertiaryUnboostedImage);
        subtractFilter->SetInput2(tertiaryUnboostedImage);
        subtractFilter->Update();
        ImageType::Pointer primaryUnboostedImage = subtractFilter->GetOutput(); // This "primaryImage" is the purely primary
        primaryUnboostedImage->DisconnectPipeline();
        SetImageDimensions(primaryUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        ////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////
        /////////////////////// APPLY COLLIMATOR MASK ///////////////////////
        ImageType::Pointer totalMaskedImage = ApplyMask(collimatorMask, totalBoostedImage, totalUnboostedImage);
        ImageType::Pointer floodMaskedImage = ApplyMask(collimatorMask, floodBoostedImage, floodUnboostedImage);
        ImageType::Pointer primaryTertiaryMaskedImage = ApplyMask(collimatorMask, primaryTertiaryBoostedImage, primaryTertiaryUnboostedImage);
        ImageType::Pointer primaryTertiaryEstimatedMaskedImage = ApplyMask(collimatorMask, primaryTertiaryEstimatedImage, primaryTertiaryUnboostedImage);
        ImageType::Pointer primaryMaskedImage = ApplyMask(collimatorMask, primaryBoostedImage, primaryUnboostedImage);
        ImageType::Pointer primaryScatterMaskedImage = ApplyMask(collimatorMask, primaryScatterBoostedImage, primaryScatterUnboostedImage);

        SetImageDimensions(totalMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(floodMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryEstimatedMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryScatterMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////// Correcting Total Image by Flood Image  ////////////////////////////////
        //////////////////////////////// Boosted ////////////////////////////////
        std::cout << "Flood Correcting Images ... " << std::endl;
        using DivideFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
        DivideFilterType::Pointer divideFilter = DivideFilterType::New();
        divideFilter->SetInput1(primaryMaskedImage);
        divideFilter->SetInput2(floodMaskedImage);
        divideFilter->Update();
        primaryMaskedImage = divideFilter->GetOutput();
        primaryMaskedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryTertiaryMaskedImage);
        divideFilter->SetInput2(floodMaskedImage);
        divideFilter->Update();
        primaryTertiaryMaskedImage = divideFilter->GetOutput();
        primaryTertiaryMaskedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryTertiaryEstimatedMaskedImage);
        divideFilter->SetInput2(floodMaskedImage);
        divideFilter->Update();
        primaryTertiaryEstimatedMaskedImage = divideFilter->GetOutput();
        primaryTertiaryEstimatedMaskedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryScatterMaskedImage);
        divideFilter->SetInput2(floodMaskedImage);
        divideFilter->Update();
        primaryScatterMaskedImage = divideFilter->GetOutput();
        primaryScatterMaskedImage->DisconnectPipeline();

        divideFilter->SetInput1(totalMaskedImage);
        divideFilter->SetInput2(floodMaskedImage);
        divideFilter->Update();
        totalMaskedImage = divideFilter->GetOutput();
        totalMaskedImage->DisconnectPipeline();

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
        //////////////////////////////// Unboosted ////////////////////////////////

        divideFilter->SetInput1(primaryUnboostedImage);
        divideFilter->SetInput2(floodUnboostedImage);
        divideFilter->Update();
        primaryUnboostedImage = divideFilter->GetOutput();
        primaryUnboostedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryTertiaryUnboostedImage);
        divideFilter->SetInput2(floodUnboostedImage);
        divideFilter->Update();
        primaryTertiaryUnboostedImage = divideFilter->GetOutput();
        primaryTertiaryUnboostedImage->DisconnectPipeline();

        divideFilter->SetInput1(primaryScatterUnboostedImage);
        divideFilter->SetInput2(floodUnboostedImage);
        divideFilter->Update();
        primaryScatterUnboostedImage = divideFilter->GetOutput();
        primaryScatterUnboostedImage->DisconnectPipeline();

        divideFilter->SetInput1(totalUnboostedImage);
        divideFilter->SetInput2(floodUnboostedImage);
        divideFilter->Update();
        totalUnboostedImage = divideFilter->GetOutput();
        totalUnboostedImage->DisconnectPipeline();

        //////////////////// IMAGE CORRECTION ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        std::cout << "Setting size and origin ... " << std::endl;
        // Change size
        // ImageType::SizeType size;
        // size[0] = detectorSize[0]; // New size in the x direction in pixels
        // size[1] = detectorSize[1]; // New size in the y direction in pixels
        // size[2] = nProjections;    // New size in the z direction
        // std::cout << "Detector size: " << size[0] << ", " << size[1] << "." << std::endl;
        // primaryBoostedImage->SetRegions(size);
        // totalBoostedImage->SetRegions(size);
        // primaryTertiaryBoostedImage->SetRegions(size);
        // primaryScatterBoostedImage->SetRegions(size);
        // primaryMaskedImage->SetRegions(size);
        // totalMaskedImage->SetRegions(size);
        // primaryTertiaryMaskedImage->SetRegions(size);
        // primaryTertiaryEstimatedMaskedImage->SetRegions(size);
        // primaryScatterMaskedImage->SetRegions(size);

        // primaryUnboostedImage->SetRegions(size);
        // totalUnboostedImage->SetRegions(size);
        // primaryTertiaryUnboostedImage->SetRegions(size);
        // primaryScatterUnboostedImage->SetRegions(size);

        // // Change spacing
        // ImageType::SpacingType spacing;
        // spacing[0] = PixelPitch; // New spacing in the x direction in mm
        // spacing[1] = PixelPitch; // New spacing in the y direction in mm
        // spacing[2] = 1;          // New spacing in the z direction
        // primaryBoostedImage->SetSpacing(spacing);
        // totalBoostedImage->SetSpacing(spacing);
        // primaryTertiaryBoostedImage->SetSpacing(spacing);
        // primaryScatterBoostedImage->SetSpacing(spacing);
        // primaryMaskedImage->SetSpacing(spacing);
        // totalMaskedImage->SetSpacing(spacing);
        // primaryTertiaryMaskedImage->SetSpacing(spacing);
        // primaryTertiaryEstimatedMaskedImage->SetSpacing(spacing);
        // primaryScatterMaskedImage->SetSpacing(spacing);

        // primaryUnboostedImage->SetSpacing(spacing);
        // totalUnboostedImage->SetSpacing(spacing);
        // primaryTertiaryUnboostedImage->SetSpacing(spacing);
        // primaryScatterUnboostedImage->SetSpacing(spacing);

        // // Change origin
        // ImageType::PointType newOrigin;

        // newOrigin[0] = -((size[0] * PixelPitch / 2) - (spacing[0] / 2)); // new x origin in mm
        // newOrigin[1] = -((size[1] * PixelPitch / 2) - (spacing[1] / 2)); //-((size[1] / 2) - (spacing[1] / 2)); // new y origin in mm
        // newOrigin[2] = 1;                                                // new z origin
        // std::cout << "newOrigin: " << newOrigin[0] << ", " << newOrigin[1] << "." << std::endl;
        // primaryBoostedImage->SetOrigin(newOrigin);
        // totalBoostedImage->SetOrigin(newOrigin);
        // primaryTertiaryBoostedImage->SetOrigin(newOrigin);
        // primaryScatterBoostedImage->SetOrigin(newOrigin);
        // primaryMaskedImage->SetOrigin(newOrigin);
        // totalMaskedImage->SetOrigin(newOrigin);
        // primaryTertiaryMaskedImage->SetOrigin(newOrigin);
        // primaryTertiaryEstimatedMaskedImage->SetOrigin(newOrigin);
        // primaryScatterMaskedImage->SetOrigin(newOrigin);

        // primaryUnboostedImage->SetOrigin(newOrigin);
        // totalUnboostedImage->SetOrigin(newOrigin);
        // primaryTertiaryUnboostedImage->SetOrigin(newOrigin);
        // primaryScatterUnboostedImage->SetOrigin(newOrigin);
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        /////////////////////////////////////////////////////////

        SetImageDimensions(primaryBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(totalBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryScatterBoostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        SetImageDimensions(totalMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryEstimatedMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryScatterMaskedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

        SetImageDimensions(primaryUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(totalUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryTertiaryUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);
        SetImageDimensions(primaryScatterUnboostedImage, detectorSize[0], detectorSize[1], nProjections, PixelPitch);

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

        writer->SetFileName(outputDir + "totalMaskedImage.mha");
        writer->SetInput(totalMaskedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "totalMaskedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryTertiaryMaskedImage.mha");
        writer->SetInput(primaryTertiaryMaskedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryTertiaryMaskedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryTertiaryEstimatedMaskedImage.mha");
        writer->SetInput(primaryTertiaryEstimatedMaskedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryTertiaryEstimatedMaskedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryMaskedImage.mha");
        writer->SetInput(primaryMaskedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryMaskedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryScatterMaskedImage.mha");
        writer->SetInput(primaryScatterMaskedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryScatterMaskedImage.mha" << std::endl;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        writer->SetFileName(outputDir + "primaryUnboostedImage.mha");
        writer->SetInput(primaryUnboostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryUnboostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "totalUnboostedImage.mha");
        writer->SetInput(totalUnboostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "totalUnboostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryTertiaryUnboostedImage.mha");
        writer->SetInput(primaryTertiaryUnboostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryTertiaryUnboostedImage.mha" << std::endl;

        writer->SetFileName(outputDir + "primaryScatterUnboostedImage.mha");
        writer->SetInput(primaryScatterUnboostedImage);
        writer->Update();
        std::cout << "Writing File Out: " + outputDir + "primaryScatterUnboostedImage.mha" << std::endl;

        //////////////////// WRITE FILES OUT ////////////////////
        /////////////////////////////////////////////////////////
    }

    std::cout
        << "Press ENTER to continue... " << std::flush;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    return 0;
}
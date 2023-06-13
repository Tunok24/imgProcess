#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>
#include "itkDivideImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <vector>
#include <math.h>

// Define the image types
typedef itk::Image<float, 3> ImageType; // Assuming your images are 3D and of type float
typedef itk::Image<float, 2> ImageType2D;
typedef itk::Image<float, 3> ImageType3D;

void ReadImageProperties(std::string filename)
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
}

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

float *ReadMHA(const std::string &filename, unsigned int &width, unsigned int &height, unsigned int &numProjections)
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
}

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

    // Get the image from the reader
    ImageType::Pointer image = reader->GetOutput();

    // Return the image
    return image;
}

// This is the function for 2D surface interpolation
ImageType3D::Pointer ScatterCorrection(ImageType3D::Pointer scatterImage3D)
{
    // Extract the first slice from the 3D image
    itk::Size<3> size = scatterImage3D->GetLargestPossibleRegion().GetSize();
    ImageType2D::Pointer scatterImage2D = ImageType2D::New();
    ImageType2D::IndexType start2D;
    start2D.Fill(0);
    ImageType2D::SizeType size2D;
    size2D[0] = size[0]; // Width
    size2D[1] = size[1]; // Height
    ImageType2D::RegionType region2D(start2D, size2D);
    scatterImage2D->SetRegions(region2D);
    scatterImage2D->Allocate();

    itk::ImageRegionIterator<ImageType3D> it3D(scatterImage3D, scatterImage3D->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType2D> it2D(scatterImage2D, scatterImage2D->GetLargestPossibleRegion());
    for (it3D.GoToBegin(), it2D.GoToBegin(); !it3D.IsAtEnd() && !it2D.IsAtEnd(); ++it3D, ++it2D)
    {
        it2D.Set(it3D.Get());
    }

    // Create the interpolator
    typedef itk::BSplineInterpolateImageFunction<ImageType2D, double, double> InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3);
    interpolator->SetInputImage(scatterImage2D);

    // Create an image to store the interpolated scatter estimates
    ImageType3D::Pointer scatterEstimate = ImageType3D::New();
    ImageType3D::IndexType start3D;
    start3D.Fill(0);
    size[2] = 1; // depth is 1
    ImageType3D::RegionType region3D(start3D, size);
    scatterEstimate->SetRegions(region3D);
    scatterEstimate->Allocate();

    // Interpolate and store the scatter estimates
    itk::ImageRegionIterator<ImageType3D> itScatter(scatterEstimate, scatterEstimate->GetLargestPossibleRegion());
    for (itScatter.GoToBegin(); !itScatter.IsAtEnd(); ++itScatter)
    {
        ImageType3D::IndexType index3D = itScatter.GetIndex();
        ImageType2D::IndexType index2D;
        index2D[0] = index3D[0];
        index2D[1] = index3D[1];
        itScatter.Set(interpolator->EvaluateAtContinuousIndex(index2D));
    }

    return scatterEstimate;
}

// Function to extract a row or column from a 2D ITK Image
// 'direction' is 0 for row, 1 for column.
// 'index' is the index of the row or column to extract.
std::vector<float> extractLine(itk::Image<float, 2>::Pointer image, unsigned direction, unsigned index)
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
}

double interpolateAtPoint(const itk::Image<float, 2>::ConstPointer &image, itk::Image<float, 2>::PointType &point)
{
    // Define the image type using float pixels and 2 dimensions
    using ImageType = itk::Image<float, 2>;
    using ConstImagePointer = ImageType::ConstPointer;
    using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double, double>;

    // Set up the interpolator
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3); // Set the order of the spline, e.g. 3 for cubic
    interpolator->SetInputImage(image);

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

    /*EXAMPLE: Assume 'image' is an ImageType::Pointer and 'point' is an ImageType::PointType
    try {
        double value = interpolateAtPoint(image, point);
        std::cout << "Interpolated value: " << value << std::endl;
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
    }
    */
}

// Calculate SNR for given image and noise region
double calculateSNR(itk::Image<float, 2>::Pointer image, itk::ImageRegion<2> noiseRegion)
{
    // Create an iterator for the noise region
    itk::ImageRegionIterator<itk::Image<float, 2>> noiseIterator(image, noiseRegion);

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

    return noiseMean / noiseStdDev; // Return SNR
}

// Calculate CNR for given image and two signal regions
double calculateCNR(itk::Image<float, 2>::Pointer image,
                    itk::ImageRegion<2> signalRegion1, itk::ImageRegion<2> signalRegion2,
                    itk::ImageRegion<2> noiseRegion)
{
    // Create iterators for the signal and noise regions
    itk::ImageRegionIterator<itk::Image<float, 2>> signalIterator1(image, signalRegion1);
    itk::ImageRegionIterator<itk::Image<float, 2>> signalIterator2(image, signalRegion2);
    itk::ImageRegionIterator<itk::Image<float, 2>> noiseIterator(image, noiseRegion);

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

ImageType::Pointer floodCorrectedImage(const ImageType::Pointer &floodImage, const ImageType::Pointer &projectedImage)
{
    using ImageType = itk::Image<float, 3>;
    // Check if images are of the same size
    if (floodImage->GetLargestPossibleRegion() != projectedImage->GetLargestPossibleRegion())
    {
        throw std::invalid_argument("Both images should have the same dimensions for flood correction");
    }

    // Create a filter for division
    using DivideImageFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
    DivideImageFilterType::Pointer divideImageFilter = DivideImageFilterType::New();

    divideImageFilter->SetInput1(projectedImage);
    divideImageFilter->SetInput2(floodImage);

    try
    {
        divideImageFilter->Update();
    }
    catch (itk::ExceptionObject &error)
    {
        std::cerr << "Error: " << error << std::endl;
        return nullptr; // or handle exception in a different way
    }

    // Now, let's handle potential Inf and NaN values
    using IteratorType = itk::ImageRegionIteratorWithIndex<ImageType>;
    IteratorType it(divideImageFilter->GetOutput(), divideImageFilter->GetOutput()->GetLargestPossibleRegion());

    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        if (std::isinf(it.Get()) || std::isnan(it.Get()))
        {
            it.Set(0);
        }
    }

    return divideImageFilter->GetOutput();
}

int main()
{
    // Loading Scatter Image as 3D Image
    ImageType3D::Pointer scatterImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/scatterImage.mha");
    // Loading Total Image as 3D Image
    ImageType3D::Pointer totalImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/totalImage.mha");
    // Loading Flood Image as 3D Image
    ImageType3D::Pointer floodImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/floodImage.mha");
    // ScatterCorrection returns the Scatter Estimate by doing BSpline 2D Interpolation of the Scatter Image
    ImageType3D::Pointer scatterEst = ScatterCorrection(scatterImage);

    // We will do flood field correction After Scatter Correction

    //// Correcting Total Image by Flood Image //// Assumption - they are of identical dimension.
    using SubtractFilterType = itk::SubtractImageFilter<ImageType3D>;
    SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();

    subtractFilter->SetInput1(totalImage);
    subtractFilter->SetInput2(scatterEst);
    subtractFilter->Update();

    ImageType3D::Pointer scatterCorrectedImage = subtractFilter->GetOutput();

    // The scatterCorrectedImage now holds the result of the pixel-wise subtraction.

    //// Correcting Total Image by Flood Image  ////
    using DivideFilterType = itk::DivideImageFilter<ImageType3D, ImageType3D, ImageType3D>;
    DivideFilterType::Pointer divideFilter = DivideFilterType::New();

    divideFilter->SetInput1(scatterCorrectedImage);
    divideFilter->SetInput2(floodImage);
    divideFilter->Update();

    ImageType3D::Pointer correctedImage = divideFilter->GetOutput();

    /*
    ImageType::SizeType size = scatterImage->GetLargestPossibleRegion().GetSize();

    std::cout << "Image dimensions: "
              << size[0] << " x "
              << size[1] << " x "
              << size[2] << std::endl;

    int imageDim = GetImageDimension("/home/tunok/Work/mcDataIO_main/tests/output/scatterImage.mha");
    std::cout << "Number of Dimension: " << imageDim << std::endl;

    // Let's assume we're interested in the 1st slice (depth = 0)
    const unsigned int sliceNumber = 0;

    // Print the 192nd row and 192nd column of the chosen slice (192 is about 1/4th of height=768)
    const unsigned int rowIndex = 191; // indexing starts from 0
    const unsigned int columnIndex = 191;

    ImageType::IndexType pixelIndex;

    // Print the 192nd row
    pixelIndex[2] = sliceNumber; // Depth
    pixelIndex[1] = rowIndex;    // Row
    for (unsigned int i = 0; i < size[0]; ++i)
    {
        pixelIndex[0] = i; // Column
        std::cout << scatterImage->GetPixel(pixelIndex) << " ";
    }
    std::cout << std::endl;

    // Print the 192nd column
    pixelIndex[2] = sliceNumber; // Depth
    pixelIndex[0] = columnIndex; // Column
    for (unsigned int i = 0; i < size[1]; ++i)
    {
        pixelIndex[1] = i; // Row
        std::cout << scatterImage->GetPixel(pixelIndex) << " ";
    }
    std::cout << std::endl;
/**/

    // Define the writer type
    using WriterType = itk::ImageFileWriter<ImageType3D>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterCorrectedImage.mha");
    writer->SetInput(scatterCorrectedImage);
    writer->Update();
    return 0;
}

// running this program should output the string: "Image dimensions: 512 x 768 x 73" (as of june 6th, 2023)
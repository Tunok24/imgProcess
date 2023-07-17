#include <vector>
#include <math.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkFlipImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>
#include "itkDivideImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkStatisticsImageFilter.h"
#include <itkBoxImageFilter.h>
#include <rtkThreeDCircularProjectionGeometry.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include <itkMultiplyImageFilter.h>
#include <rtkForwardProjectionImageFilter.h>
#include <rtkConstantImageSource.h>
#include <rtkFDKConeBeamReconstructionFilter.h>

// Define the image types
typedef itk::Image<float, 3> ImageType; // Assuming your images are 3D and of type float
typedef itk::Image<float, 2> ImageType2D;

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

// This is the function for 2D BSpline interpolation function. Outputs the 3DImage MHA file containing the interpolated values.

ImageType::Pointer ScatterCorrection(ImageType::Pointer scatterImage3D)
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

// Causal recursive filter
ImageType::Pointer CausalRecursiveFilter(ImageType::Pointer inputImage, float theta)
{
    typedef itk::ImageRegionIterator<ImageType> IteratorType;

    ImageType::Pointer outputImage = ImageType::New();
    outputImage->CopyInformation(inputImage);
    outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
    outputImage->Allocate();
    outputImage->FillBuffer(0);

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
                outputImage->SetPixel(index, outputValue);
                previousValue = outputValue;
            }
        }
    }

    return outputImage;
}

// Non-causal, forward-backward filter
/*ImageType::Pointer ForwardBackwardFilter(ImageType::Pointer image, unsigned int width)
{
    // Apply forward filter
    ImageType::Pointer forward = CausalRecursiveFilter(image, 1.0f / width);

    // Reverse the image
    typedef itk::FlipImageFilter<ImageType> FlipImageFilterType;
    FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();
    FlipImageFilterType::FlipAxesArrayType flipAxes;
    flipAxes[0] = false;
    flipAxes[1] = false;
    flipAxes[2] = true; // Flip in Z direction
    flipFilter->SetFlipAxes(flipAxes);
    flipFilter->SetInput(forward);
    flipFilter->Update();
    ImageType::Pointer reversed = flipFilter->GetOutput();

    // Apply backward filter
    ImageType::Pointer backward = CausalRecursiveFilter(reversed, 1.0f / width);

    // Reverse the image back
    flipFilter->SetInput(backward);
    flipFilter->Update();

    return flipFilter->GetOutput();
}/**/

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

int main()
{
    /////////////////////////////////////////////////////////
    //////////////////// READ IMAGE FILES ///////////////////
    bool boosted = true; // if false, processes the unboosted images
    if (true)            // this is to execute the image process section or not (for now)
    {
        ImageType::Pointer totalImage;
        ImageType::Pointer scatterImage;
        ImageType::Pointer floodImage;
        if (!boosted)
        {
            // TOTAL IMAGE
            totalImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/totalUnboostedImage500000000.mha");
            // SCATTER IMAGE
            scatterImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/scatterUnboostedImage500000000.mha");
            // FLOOD IMAGE
            floodImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/floodUnboostedImage500000000.mha");
        }
        else
        {
            // TOTAL BOOSTED IMAGE
            totalImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/totalBoostedImage500000000.mha");
            // SCATTER BOOSTED IMAGE
            scatterImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/scatterBoostedImage500000000.mha");
            // FLOOD BOOSTED IMAGE
            floodImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/floodBoostedImage500000000.mha");
        }

        /**/
        //////////////////// READ IMAGE FILES ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// SCATTER ESTIMATE ///////////////////

        /*// Scatter Estimate by doing BSPLINE 2D INTERPOLATION of each individual slice
        ImageType::Pointer scatterEst = ScatterCorrection(scatterImage);

        // Further Scatter Estimate by doing PROJECTION-TO-PROJECTION-SMOOTHING by theta_filter
        float theta_filt = 1.0f; // Choose your filter parameter
        unsigned int width = 2;  // Choose your filter width
        ImageType::Pointer smoothedScatterEstimate = CausalRecursiveFilter(scatterEst, theta_filt);
        // ImageType::Pointer smoothedScatterEstimate = NonCausalFilter(scatterEst, width);/**/
        //////////////////// SCATTER ESTIMATE ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// IMAGE CORRECTION ///////////////////

        //// Correcting Scatter Image by Scatter Estimate ////
        using SubtractFilterType = itk::SubtractImageFilter<ImageType>;
        SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();
        subtractFilter->SetInput1(totalImage);
        subtractFilter->SetInput2(scatterImage);
        subtractFilter->Update();
        ImageType::Pointer scatterCorrectedImage = subtractFilter->GetOutput(); // This "scatterCorrectedImage" is the estimated primary

        //// Correcting Original Scatter Image by subtracting Smoothed Scatter Estimate
        /*subtractFilter->SetInput1(scatterImage);
        subtractFilter->SetInput2(smoothedScatterEstimate);
        subtractFilter->Update();
        ImageType::Pointer scatterCorrectedScatterImage = subtractFilter->GetOutput();/**/

        //// Correcting Total Image by Flood Image  ////
        using DivideFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
        DivideFilterType::Pointer divideFilter = DivideFilterType::New();
        // divideFilter->SetInput1(totalImage); // when not doing scatter correction
        divideFilter->SetInput1(scatterCorrectedImage); // when doing scatter correction
        divideFilter->SetInput2(floodImage);
        divideFilter->Update();
        // ImageType::Pointer floodCorrectedImage = divideFilter->GetOutput(); // when not doing scatter correction
        ImageType::Pointer correctedImage = divideFilter->GetOutput(); // when doing scatter correction/**/

        /////////// ReScale an image if needed ///////////
        // typedef for the RescaleIntensityImageFilter
        using FilterType = itk::MultiplyImageFilter<ImageType, ImageType, ImageType>;

        // Create and setup the filter
        auto multiplyConstantFilter = FilterType::New();
        multiplyConstantFilter->SetInput(correctedImage);
        multiplyConstantFilter->SetConstant(10);
        multiplyConstantFilter->Update();

        ImageType::Pointer rescaledImage = multiplyConstantFilter->GetOutput();

        //////////////////// IMAGE CORRECTION ///////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////// CHANGE METADATA BEFORE WRITING ////////////

        // Change size
        ImageType::SizeType size;
        size[0] = 250;                    // New size in the x direction
        size[1] = 250;                    // New size in the y direction
        size[2] = 201;                    // New size in the z direction
        correctedImage->SetRegions(size); // when doing scatter correction
        // floodCorrectedImage->SetRegions(size);

        // Change spacing
        ImageType::SpacingType spacing;
        spacing[0] = 1;                      // New spacing in the x direction
        spacing[1] = 1;                      // New spacing in the y direction
        spacing[2] = 1;                      // New spacing in the z direction
        correctedImage->SetSpacing(spacing); // when doing scatter correction
        // floodCorrectedImage->SetSpacing(spacing);

        // Change origin
        ImageType::PointType newOrigin;
        newOrigin[0] = -((size[0] / 2) - (spacing[0] / 2)); // new x origin
        newOrigin[1] = -((size[1] / 2) - (spacing[1] / 2)); // new y origin
        newOrigin[2] = 0;                                   // new z origin
        correctedImage->SetOrigin(newOrigin);               // when doing scatter correction
        // floodCorrectedImage->SetOrigin(newOrigin);
        /**/
        ///////////// CHANGE METADATA BEFORE WRITING ////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        //////////////////// WRITE FILES OUT ////////////////////
        // Define the writer type
        using WriterType = itk::ImageFileWriter<ImageType>;
        WriterType::Pointer writer = WriterType::New();

        /*writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterEst.mha");
        writer->SetInput(scatterEst);
        writer->Update();
        std::cout << "Writing File Out: scatterEst.mha" << std::endl;

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/smoothedScatterEstimate.mha");
        writer->SetInput(smoothedScatterEstimate);
        writer->Update();
        std::cout << "Writing File Out: smoothedScatterEstimate.mha" << std::endl;/**/

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/primaryImage500000000.mha");
        writer->SetInput(scatterCorrectedImage);
        writer->Update();
        std::cout << "Writing File Out: primaryImage500000000.mha" << std::endl;
        /*
                writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterCorrectedScatterImage5000000.mha");
                writer->SetInput(scatterCorrectedScatterImage);
                writer->Update();
                std::cout << "Writing File Out: scatterCorrectedScatterImage.mha" << std::endl;/**/

        writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/rescaledImage500000000.mha");
        writer->SetInput(correctedImage);
        // writer->SetInput(floodCorrectedImage);
        writer->Update();
        std::cout << "Writing File Out: rescaledImage500000000.mha" << std::endl;
        /**/
        //////////////////// WRITE FILES OUT ////////////////////
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        ///////////////////// GEOMETRY XML //////////////////////

        // Set the parameters
        using GeometryType = rtk::ThreeDCircularProjectionGeometry;
        GeometryType::Pointer geometry = GeometryType::New();

        double sid = 1000.0;                // source to isocenter distance
        double sdd = 1500.0;                // source to detector distance
        double start_angle = 0.0;           // start angle
        double stop_angle = 360.0;          // stop angle
        unsigned int num_projections = 200; // number of projections

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

        ////////////////////////////////////////////////////////
        ///////////////////// RECONSTRUCT ////////////////////// // Cannot reconstuct anything as of yet.
        // Set up the FDK reconstruction filter
        // Create an image filled with zeroes that will hold the reconstructed volume
        if (false)
        { /*
             using ConstantImageSourceType = rtk::ConstantImageSource<ImageType>;
             ConstantImageSourceType::Pointer source = ConstantImageSourceType::New();
             ImageType::SizeType reconSize;
             reconSize[0] = 100; // adjust as needed
             reconSize[1] = 100; // adjust as needed
             reconSize[2] = 100; // adjust as needed
             source->SetSize(reconSize);
             source->SetSpacing(1.0);                // adjust as needed
             source->SetOrigin(-reconSize[0] / 2.0); // adjust as needed
             source->SetConstant(0.0);
             source->Update();

             // Set up FDK reconstruction filter
             using FDKFilterType = rtk::FDKConeBeamReconstructionFilter<ImageType, ImageType>;
             FDKFilterType::Pointer fdk = FDKFilterType::New();
             fdk->SetInput(source->GetOutput());
             fdk->SetInput(1, correctedImage);
             fdk->SetGeometry(geometry);
             fdk->Update();

             // The resulting reconstructed image can be obtained by calling
             ImageType::Pointer reconstructedVolume = fdk->GetOutput();

             // Save the reconstructed volume
             using WriterType = itk::ImageFileWriter<ImageType>;
             WriterType::Pointer reconWriter = WriterType::New();
             reconWriter->SetFileName("/home/tunok/Work/imgProcess_main/tests/reconImage.mha");
             reconWriter->SetInput(reconstructedVolume);
             try
             {
                 reconWriter->Update();
             }
             catch (itk::ExceptionObject &e)
             {
                 std::cerr << "Error: " << e << std::endl;
                 return EXIT_FAILURE;
             }
             std::cout << "Writing File Out: reconImage.mha" << std::endl;/**/
        }
        ///////////////////// RECONSTRUCT //////////////////////
        ////////////////////////////////////////////////////////
    }

    /**/

    if (false)
    {

        /////////////////////////////////////////////////////////
        //////////////////// WRITE VALUES OUT ///////////////////
        // TOTAL IMAGE
        ImageType::Pointer totalBoostedImage = ReadMHA("/home/tunok/Work/imgProcess_main/tests/correctedUnboostedImage500000000.mha");
        ImageType::Pointer totalUnboostedImage = ReadMHA("/home/tunok/Work/imgProcess_main/tests/correctedUnboostedImage500000000.mha");
        // Read Image file: Boosted Corrected Reconstructed Volume
        // ImageType::Pointer boostedReconVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/boostedCorrectedVolume.mha");
        // Read Image file: Unboosted Corrected Reconstructed Volume
        // ImageType::Pointer unboostedReconVolume = ReadMHA("/home/tunok/Work/Fresco-21.1.0-CustomLinuxBuild/Examples/simulate_and_reconstruct/unboostedCorrectedVolume.mha");/**/

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
        ImageType::SizeType size = boostedReconVolume->GetLargestPossibleRegion().GetSize();
        std::cout << "Image dimensions: "
                  << size[0] << " x "
                  << size[1] << " x "
                  << size[2] << std::endl;

        size = unboostedReconVolume->GetLargestPossibleRegion().GetSize();
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

        double snr = calculateSNR(boostedReconVolume, noiseRegion, sliceNumber);
        std::cout << "The SNR for boosted Image is: " << snr << std::endl;
        snr = calculateSNR(unboostedReconVolume, noiseRegion, sliceNumber);
        std::cout << "The SNR for unboosted Image is: " << snr << std::endl;
        /**/
        ///////////// Calculating SNR //////////////

        ///////////// Calculating CNR //////////////

        // work with 10x10 regions
        ImageType::SizeType size;
        size[0] = 10; // size in x direction
        size[1] = 10; // size in y direction
        size[2] = 1;  // size in z direction

        // signalRegion1 around (50, 73, 36), singalRegion2 around (95, 20, 36), noiseRegion around (20, 20, 36)
        ImageType::IndexType startRegion1;
        startRegion1[0] = 90;  // starting x coordinate
        startRegion1[1] = 110; // starting y coordinate
        startRegion1[2] = 0;   // starting z coordinate

        ImageType::IndexType startRegion2;
        startRegion2[0] = 90; // starting x coordinate
        startRegion2[1] = 80; // starting y coordinate
        startRegion2[2] = 0;  // starting z coordinate

        ImageType::IndexType startRegionNoise;
        startRegionNoise[0] = 15; // starting x coordinate
        startRegionNoise[1] = 50; // starting y coordinate
        startRegionNoise[2] = 0;  // starting z coordinate

        ImageType::RegionType region1;
        region1.SetIndex(startRegion1);
        region1.SetSize(size);

        ImageType::RegionType region2;
        region2.SetIndex(startRegion2);
        region2.SetSize(size);

        ImageType::RegionType regionNoise;
        regionNoise.SetIndex(startRegionNoise);
        regionNoise.SetSize(size);

        double cnr = calculateCNR(totalUnboostedImage, region1, region2, regionNoise);
        std::cout << "The CNR of the Unboosted Image is: " << cnr << std::endl;

        cnr = calculateCNR(totalBoostedImage, region1, region2, regionNoise);
        std::cout << "The CNR of the Boosted Image is: " << cnr << std::endl;
    }
    ///////////// Calculating CNR //////////////
    //////////////////// WRITE VALUES OUT ///////////////////
    /////////////////////////////////////////////////////////

    std::cout
        << "Press ENTER to continue... " << std::flush;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    return 0;
}

// running this program should output the string: "Image dimensions: 192 x 192 x 121" (as of june 6th, 2023)
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
#include <itkMultiplyImageFilter.h>

// Define the image types
typedef itk::Image<float, 3> ImageType; // Assuming images are 3D and of type float
typedef itk::Image<float, 2> ImageType2D;
using Image1DType = itk::Image<float, 1>;
typedef itk::ImageFileWriter<ImageType> WriterType;

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

    // Create an output image and allocate memory for it
    ImageType::Pointer outputImage = ImageType::New();
    outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
    outputImage->Allocate();
    outputImage->FillBuffer(0); // Initialize all pixels to 0

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

ImageType::Pointer ApplyBoostMask(
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
    std::string inputDir = "/home/tunok/Work/mcDataIO_main/tests/output/CylinderEllipsoid/";
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

    /////////////////////////////////////////////////////////
    //////////////////// READ IMAGE FILES ///////////////////

    ImageType::Pointer totalBoostedImage; // primary + scatter + tertiary
    ImageType::Pointer scatterBoostedImage;
    ImageType::Pointer floodBoostedImage;
    ImageType::Pointer tertiaryBoostedImage;

    ImageType::Pointer totalUnboostedImage; // primary + scatter + tertiary
    ImageType::Pointer scatterUnboostedImage;
    ImageType::Pointer floodUnboostedImage;
    ImageType::Pointer tertiaryUnboostedImage;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read totalBoosted ///////////////////////////////////////////////////

    std::vector<std::string> filePaths = {
        inputDir + "totalBoostedImage350000000Projection9.mha",
        inputDir + "totalBoostedImage350000000Projection9_2.mha",
        inputDir + "totalBoostedImage290000000Projection9.mha"}; //, inputDir + "totalBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "totalBoostedImage.mha");
    totalBoostedImage = ReadMHA(outputDir + "totalBoostedImage.mha");
    std::cout << "Read Total Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read scatterBoosted /////////////////////////////////////////////////

    filePaths = {
        inputDir + "scatterBoostedImage350000000Projection9.mha",
        inputDir + "scatterBoostedImage350000000Projection9_2.mha",
        inputDir + "scatterBoostedImage290000000Projection9.mha"}; //, inputDir + "scatterBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "scatterBoostedImage.mha");
    scatterBoostedImage = ReadMHA(outputDir + "scatterBoostedImage.mha");
    std::cout << "Read Scatter Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read tertiaryBoosted ////////////////////////////////////////////////

    filePaths = {
        inputDir + "tertiaryBoostedImage350000000Projection9.mha",
        inputDir + "tertiaryBoostedImage350000000Projection9_2.mha",
        inputDir + "tertiaryBoostedImage290000000Projection9.mha"}; //, inputDir + "tertiaryBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "tertiaryBoostedImage.mha");
    tertiaryBoostedImage = ReadMHA(outputDir + "tertiaryBoostedImage.mha");
    std::cout << "Read Tertiary Boosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "floodBoostedImage350000000Projection9.mha",
        inputDir + "floodBoostedImage350000000Projection9_2.mha",
        inputDir + "floodBoostedImage290000000Projection9.mha"}; //, inputDir + "floodBoostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "floodBoostedImage.mha");
    floodBoostedImage = ReadMHA(outputDir + "floodBoostedImage.mha");
    std::cout << "Read Flood Boosted Image" << std::endl;

    filePaths = {
        inputDir + "totalUnboostedImage350000000Projection9.mha",
        inputDir + "totalUnboostedImage350000000Projection9_2.mha",
        inputDir + "totalUnboostedImage290000000Projection9.mha"}; //, inputDir + "totalUnboostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "totalUnboostedImage.mha");
    totalUnboostedImage = ReadMHA(outputDir + "totalUnboostedImage.mha");
    std::cout << "Read Total Unboosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read scatterBoosted /////////////////////////////////////////////////

    filePaths = {
        inputDir + "scatterUnboostedImage350000000Projection9.mha",
        inputDir + "scatterUnboostedImage350000000Projection9_2.mha",
        inputDir + "scatterUnboostedImage290000000Projection9.mha"}; //, inputDir + "scatterUnboostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "scatterUnboostedImage.mha");
    scatterUnboostedImage = ReadMHA(outputDir + "scatterUnboostedImage.mha");
    std::cout << "Read Scatter Unboosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Read tertiaryBoosted ////////////////////////////////////////////////

    filePaths = {
        inputDir + "tertiaryUnboostedImage350000000Projection9.mha",
        inputDir + "tertiaryUnboostedImage350000000Projection9_2.mha",
        inputDir + "tertiaryUnboostedImage290000000Projection9.mha"}; //, inputDir + "tertiaryUnboostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "tertiaryUnboostedImage.mha");
    tertiaryUnboostedImage = ReadMHA(outputDir + "tertiaryUnboostedImage.mha");
    std::cout << "Read Tertiary Unboosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Read floodBoosted /////////////////////////////////////////////////
    filePaths = {
        inputDir + "floodUnboostedImage350000000Projection9.mha",
        inputDir + "floodUnboostedImage350000000Projection9_2.mha",
        inputDir + "floodUnboostedImage290000000Projection9.mha"}; //, inputDir + "floodUnboostedImage200000000Projection9_3.mha"};
    AddMultipleImages(filePaths, outputDir + "floodUnboostedImage.mha");
    floodUnboostedImage = ReadMHA(outputDir + "floodUnboostedImage.mha");
    std::cout << "Read Flood Unboosted Image" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// Add in the tertiary to the unboosted iamge /////////////////////////////////////
    // Add in the tertiary to the unboosted iamge
    typedef itk::AddImageFilter<ImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(totalUnboostedImage);
    addFilter->SetInput2(tertiaryUnboostedImage);
    addFilter->Update();
    totalUnboostedImage = addFilter->GetOutput();
    totalUnboostedImage->DisconnectPipeline();

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

    ///////////////////////////////////// UNBOOSTED /////////////////////////////////////

    subtractFilter->SetInput1(totalUnboostedImage);
    subtractFilter->SetInput2(scatterUnboostedImage);
    subtractFilter->Update();
    ImageType::Pointer primaryTertiaryUnboostedImage = subtractFilter->GetOutput(); // total - scatter = nonScatter = primary + tertiary
    primaryTertiaryUnboostedImage->DisconnectPipeline();

    subtractFilter->SetInput1(totalUnboostedImage);
    subtractFilter->SetInput2(tertiaryUnboostedImage);
    subtractFilter->Update();
    ImageType::Pointer primaryScatterUnboostedImage = subtractFilter->GetOutput(); // total - tertiary = nonTertiary = primary + scatter
    primaryScatterUnboostedImage->DisconnectPipeline();

    subtractFilter->SetInput1(primaryTertiaryUnboostedImage);
    subtractFilter->SetInput2(tertiaryUnboostedImage);
    subtractFilter->Update();
    ImageType::Pointer primaryUnboostedImage = subtractFilter->GetOutput(); // This "primaryImage" is the purely primary
    primaryUnboostedImage->DisconnectPipeline();

    /////////////////////////////////////////////////////////////////////
    /////////////////////// APPLY COLLIMATOR MASK ///////////////////////

    ImageType::Pointer totalMaskedImage = ApplyBoostMask(collimatorMask, totalBoostedImage, totalUnboostedImage);
    ImageType::Pointer floodMaskedImage = ApplyBoostMask(collimatorMask, floodBoostedImage, floodUnboostedImage);
    ImageType::Pointer primaryTertiaryMaskedImage = ApplyBoostMask(collimatorMask, primaryTertiaryBoostedImage, primaryTertiaryUnboostedImage);
    ImageType::Pointer primaryMaskedImage = ApplyBoostMask(collimatorMask, primaryBoostedImage, primaryUnboostedImage);
    ImageType::Pointer primaryScatterMaskedImage = ApplyBoostMask(collimatorMask, primaryScatterBoostedImage, primaryScatterUnboostedImage);

    std::cout << "Flood Correcting Images ... " << std::endl;
    using DivideFilterType = itk::DivideImageFilter<ImageType, ImageType, ImageType>;
    DivideFilterType::Pointer divideFilter = DivideFilterType::New();
    divideFilter->SetInput1(totalMaskedImage);
    divideFilter->SetInput2(floodMaskedImage);
    divideFilter->Update();
    totalMaskedImage = divideFilter->GetOutput();
    totalMaskedImage->DisconnectPipeline();

    divideFilter->SetInput1(totalBoostedImage);
    divideFilter->SetInput2(floodBoostedImage);
    divideFilter->Update();
    totalBoostedImage = divideFilter->GetOutput();
    totalBoostedImage->DisconnectPipeline();

    divideFilter->SetInput1(primaryTertiaryMaskedImage);
    divideFilter->SetInput2(floodMaskedImage);
    divideFilter->Update();
    primaryTertiaryMaskedImage = divideFilter->GetOutput();
    primaryTertiaryMaskedImage->DisconnectPipeline();

    divideFilter->SetInput1(primaryMaskedImage);
    divideFilter->SetInput2(floodMaskedImage);
    divideFilter->Update();
    primaryMaskedImage = divideFilter->GetOutput();
    primaryMaskedImage->DisconnectPipeline();

    divideFilter->SetInput1(primaryScatterMaskedImage);
    divideFilter->SetInput2(floodMaskedImage);
    divideFilter->Update();
    primaryScatterMaskedImage = divideFilter->GetOutput();
    primaryScatterMaskedImage->DisconnectPipeline();

    /////////////////////// APPLY COLLIMATOR MASK ///////////////////////
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    //////////////////////// SET SIZE AND SHAPE /////////////////////////

    std::cout << "Setting size and origin ... " << std::endl;
    // Change size
    ImageType::SizeType size;
    size[0] = detectorSize[0]; // New size in the x direction in pixels
    size[1] = detectorSize[1]; // New size in the y direction in pixels
    size[2] = nProjections;    // New size in the z direction
    std::cout << "Detector size: " << size[0] << ", " << size[1] << "." << std::endl;
    totalBoostedImage->SetRegions(size);
    totalMaskedImage->SetRegions(size);
    primaryTertiaryMaskedImage->SetRegions(size);
    primaryMaskedImage->SetRegions(size);
    primaryScatterMaskedImage->SetRegions(size);

    ImageType::SpacingType spacing;
    spacing[0] = PixelPitch; // New spacing in the x direction in mm
    spacing[1] = PixelPitch; // New spacing in the y direction in mm
    spacing[2] = 1;          // New spacing in the z direction
    totalBoostedImage->SetSpacing(spacing);
    totalMaskedImage->SetSpacing(spacing);
    primaryTertiaryMaskedImage->SetSpacing(spacing);
    primaryMaskedImage->SetSpacing(spacing);
    primaryScatterMaskedImage->SetSpacing(spacing);

    ImageType::PointType newOrigin;
    newOrigin[0] = -((size[0] * PixelPitch / 2) - (spacing[0] / 2)); // new x origin in mm
    newOrigin[1] = -((size[1] * PixelPitch / 2) - (spacing[1] / 2)); //-((size[1] / 2) - (spacing[1] / 2)); // new y origin in mm
    newOrigin[2] = 1;                                                // new z origin
    std::cout << "newOrigin: " << newOrigin[0] << ", " << newOrigin[1] << "." << std::endl;
    totalBoostedImage->SetOrigin(newOrigin);
    totalMaskedImage->SetOrigin(newOrigin);
    primaryTertiaryMaskedImage->SetOrigin(newOrigin);
    primaryMaskedImage->SetOrigin(newOrigin);
    primaryScatterMaskedImage->SetOrigin(newOrigin);
    //////////////////////// SET SIZE AND SHAPE /////////////////////////
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    ////////////////////////// WRITE OUT FILES //////////////////////////

    writer->SetFileName(outputDir + "totalBoostedImage.mha");
    writer->SetInput(totalBoostedImage);
    writer->Update();
    std::cout << "Writing File Out: " + outputDir + "totalBoostedImage.mha" << std::endl;

    writer->SetFileName(outputDir + "totalMaskedImage.mha");
    writer->SetInput(totalMaskedImage);
    writer->Update();
    std::cout << "Writing File Out: " + outputDir + "totalMaskedImage.mha" << std::endl;

    writer->SetFileName(outputDir + "primaryTertiaryMaskedImage.mha");
    writer->SetInput(primaryTertiaryMaskedImage);
    writer->Update();
    std::cout << "Writing File Out: " + outputDir + "primaryTertiaryMaskedImage.mha" << std::endl;

    writer->SetFileName(outputDir + "primaryMaskedImage.mha");
    writer->SetInput(primaryMaskedImage);
    writer->Update();
    std::cout << "Writing File Out: " + outputDir + "primaryMaskedImage.mha" << std::endl;

    writer->SetFileName(outputDir + "primaryScatterMaskedImage.mha");
    writer->SetInput(primaryScatterMaskedImage);
    writer->Update();
    std::cout << "Writing File Out: " + outputDir + "primaryScatterMaskedImage.mha" << std::endl;

    ////////////////////////// WRITE OUT FILES //////////////////////////
    /////////////////////////////////////////////////////////////////////
}
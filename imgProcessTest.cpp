#include <vector>
#include <string>
#include <math.h>
#include <itkImage.h>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include "itkLinearInterpolateImageFunction.h"
#include <itkFlipImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include "itkExtractImageFilter.h"
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
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

    // Get the image from the reader
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
ImageType::Pointer LateralSmoothing(ImageType::Pointer scatterEstimate, double variance)
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
        return nullptr; // Return or handle error appropriately
    }

    return filter->GetOutput();
}

// Function to extract a row or column from a 2D ITK Image
// 'direction' is 0 for row, 1 for column.
// 'index' is the index of the row or column to extract.

double interpolateAtPoint(const std::vector<float> &columnData, float y)
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
}

void InterpolateColumns(itk::Image<float, 3>::Pointer scatterImage3D, itk::ImageRegion<3> exclusionRegion)
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

ImageType::Pointer ConcatenateImages(ImageType::Pointer image1, ImageType::Pointer image2, ImageType::SizeType concatSize, std::string filename)
{
    ImageType::RegionType region1 = image1->GetLargestPossibleRegion();
    ImageType::RegionType region2 = image2->GetLargestPossibleRegion();
    ImageType::SizeType size1 = region1.GetSize();
    ImageType::SizeType size2 = region2.GetSize();
    long totalNumberOfPixels1 = size1[0] * size1[1] * size1[2];
    long totalNumberOfPixels2 = size2[0] * size2[1] * size2[2];

    std::vector<float> concatVector;
    std::vector<float> pixelValues1(image1->GetBufferPointer(), image1->GetBufferPointer() + totalNumberOfPixels1);
    concatVector.insert(concatVector.end(), pixelValues1.begin(), pixelValues1.end());
    std::vector<float> pixelValues2(image2->GetBufferPointer(), image2->GetBufferPointer() + totalNumberOfPixels2);
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

    // Write out the concatenated image
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(concatImage);
    writer->Update();
    std::cout << "Written File Out: " << filename << std::endl;

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
        newSize[1] = concatSize[1] + image2->GetLargestPossibleRegion().GetSize()[1];
        newSize[2] = concatSize[2];
        currentImage = ConcatenateImages(currentImage, image2, newSize, "concatImage" + std::to_string(i) + ".mha");

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
} /* works incrrectly */

/*void estimateUnknownRows(ImageType::Pointer image, int startUnknownRow, int endUnknownRow)
{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            // Prepare column data
            std::vector<float> columnData;
            for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
            {
                if (rowIdx < startUnknownRow || rowIdx > endUnknownRow)
                {
                    ImageType::IndexType idx3D = {{colIdx, rowIdx, sliceIdx}};
                    columnData.push_back(image->GetPixel(idx3D));
                }
            }

            // Interpolate the unknown values
            for (int rowIdx = startUnknownRow; rowIdx <= endUnknownRow; ++rowIdx)
            {
                // Normalize the current row index to the size of the known data
                float normalizedRowIndex = static_cast<float>(rowIdx - startUnknownRow) *
                                           (columnData.size() - 1) /
                                           (imageSize[1] - (endUnknownRow - startUnknownRow + 1));

                // Estimate value at the current position using interpolation
                float interpolatedValue = interpolateAtPoint(columnData, normalizedRowIndex);

                // Insert the interpolated value back into the original image
                ImageType::IndexType idx3D = {{colIdx, rowIdx, sliceIdx}};
                image->SetPixel(idx3D, interpolatedValue);
            }
        }
    }
}/* works incorrectly */

/*void estimateUnknownRows(ImageType::Pointer image, int startUnknownRow, int endUnknownRow)

{
    ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    for (int sliceIdx = 0; sliceIdx < imageSize[2]; ++sliceIdx)
    {
        for (int colIdx = 0; colIdx < imageSize[0]; ++colIdx)
        {
            // Prepare column data
            std::vector<float> columnData;
            for (int rowIdx = 0; rowIdx < imageSize[1]; ++rowIdx)
            {
                if (rowIdx < startUnknownRow || rowIdx > endUnknownRow)
                {
                    ImageType::IndexType idx3D = {{colIdx, rowIdx, sliceIdx}};
                    columnData.push_back(image->GetPixel(idx3D));
                }
            }

            // Interpolate the unknown values
            for (int rowIdx = startUnknownRow; rowIdx <= endUnknownRow; ++rowIdx)
            {
                // Normalize the current row index to the size of the known data
                float normalizedRowIndex = static_cast<float>(rowIdx - startUnknownRow) *
                                           (columnData.size() - 1) /
                                           (imageSize[1] - (endUnknownRow - startUnknownRow + 1));

                // Estimate value at the current position using interpolation
                float interpolatedValue = interpolateAtPoint(columnData, normalizedRowIndex);

                // Insert the interpolated value back into the original image
                ImageType::IndexType idx3D = {{colIdx, rowIdx, sliceIdx}};
                image->SetPixel(idx3D, interpolatedValue);
            }
        }
    }
}/**/

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
    ImageType::Pointer scatterImage = ReadMHA("/home/tunok/Work/mcDataIO_main/tests/output/Set300000000Projection5,6/scatterBoostedImage300000000Projection5.mha");
    using ExtractFilterType = itk::ExtractImageFilter<ImageType, ImageType>;
    // Get the size of the original image
    ImageType::RegionType inputRegion = scatterImage->GetLargestPossibleRegion();
    ImageType::SizeType size = inputRegion.GetSize();

    // Create a region that includes all columns and rows, but only the first 3 slices
    ImageType::IndexType start = inputRegion.GetIndex();
    size[2] = 4; // Number of slices
    ImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);

    // Create the extract filter
    ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
    extractFilter->SetExtractionRegion(desiredRegion);
    extractFilter->SetInput(scatterImage);
    // Execute the filter
    extractFilter->Update();

    // Get the output (the first 3 slices of the image)
    ImageType::Pointer sampleScatter = extractFilter->GetOutput();
    ///////////////////// CONCATENATE //////////////////////
    ////////////////////////////////////////////////////////

    //////////////////// READ IMAGE FILES ///////////////////
    /////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////
    //////////////////// SCATTER ESTIMATE ///////////////////
    ImageType::Pointer interpolatedScatterImage = sampleScatter;

    // Estimate the values in the unknown region
    std::cout << "Interpolating ... " << std::endl;
    // Define the start and end rows of the unknown region
    int startUnknownRow = 100;
    int endUnknownRow = 200;
    int interpolationOrder = 2;
    estimateUnknownRows(interpolatedScatterImage, startUnknownRow, endUnknownRow, interpolationOrder);
    std::cout << "Done interpolating ... " << std::endl;

    // Lateral Smoothing of ScatterEstimation
    std::cout << "Lateral Smoothing Scatter ... " << std::endl;
    double variance = 7;
    ImageType::Pointer scatterLatSmooth = LateralSmoothing(interpolatedScatterImage, variance);

    // Further Scatter Estimate by doing PROJECTION-TO-PROJECTION-SMOOTHING by theta_filter
    std::cout << "Projection-To-Projection Smoothing Scatter ... " << std::endl;
    float theta_filt = 1.0f; // 0 - <1: Causal 1 - 40: Non-causal
    unsigned int width = 2;  // Choose your filter width
    ImageType::Pointer scatterEst = CausalRecursiveFilter(scatterLatSmooth, theta_filt);
    // ImageType::Pointer scatterEst = NonCausalFilter(scatterLatSmooth, width); /**/

    // Writing Files out
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName("/home/tunok/Work/imgProcess_main/tests/scatterEst.mha");
    writer->SetInput(scatterEst);
    writer->Update();
    std::cout << "Writing File Out: scatterEst.mha" << std::endl;
    //////////////////// WRITE FILES OUT ////////////////////
    /////////////////////////////////////////////////////////
}
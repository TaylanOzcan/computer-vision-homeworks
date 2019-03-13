/*Before trying to construct hybrid images, it is suggested that you
implement myFilter() and then debug it using proj1_test_filtering.cpp */

// Comp408 - Assignment 1
// Özgür Taylan Özcan

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat myFilter(Mat, Mat, int);
Mat hybrid_image_visualize(Mat);
Mat DFT_Spectrum(Mat);

enum border { Border_Replicate, Border_Reflect, Border_Constant };

Mat myFilter(Mat im, Mat filter, int borderType = Border_Constant)
{
	/*This function is intended to behave like the built in function filter2D()

	Your function should work for color images. Simply filter each color
	channel independently.

	Your function should work for filters of any width and height
	combination, as long as the width and height are odd(e.g. 1, 7, 9).This
	restriction makes it unambigious which pixel in the filter is the center
	pixel.

	Boundary handling can be tricky.The filter can't be centered on pixels
	at the image boundary without parts of the filter being out of bounds.
	There are several options to deal with boundaries. Your code should be
	able to handle the border types defined above as the following enum types:
	* Border_Replicate:     aaaaaa|abcdefgh|hhhhhhh
	* Border_Reflect:       fedcba|abcdefgh|hgfedcb
	* Border_Constant:      iiiiii|abcdefgh|iiiiiii  with 'i=0'
	(image boundaries are denoted with '|')

	*/

	Mat outI;

	// Get width, height and number of channels of the original image
	int im_width = im.size().width;
	int im_height = im.size().height;
	int im_channels = im.channels();

	// Get width and height of the filter given
	int filter_width = filter.size().width;
	int filter_height = filter.size().height;

	// Calculate padding sizes
	Mat padded;
	int padding_x = (filter_width - 1) / 2;
	int padding_y = (filter_height - 1) / 2;

	// Select the border type according to given parameter (borderType)
	int border_type;
	switch (borderType) {
	case Border_Replicate:
		border_type = BORDER_REPLICATE; break;
	case Border_Reflect:
		border_type = BORDER_REFLECT; break;
	case Border_Constant:
		border_type = BORDER_CONSTANT; break;
	default:
		break;
	}

	// Create the padded image
	copyMakeBorder(im, padded, padding_y, padding_y, padding_x, padding_x, border_type);

	// Create a template output image to fill later
	outI = Mat::zeros(im_height, im_width, im.type());

	// Split the padded image into its channels
	vector<Mat> padded_channels(im_channels);
	split(padded, padded_channels);

	// Split the template output image into its channels
	vector<Mat> channels(im_channels);
	split(outI, channels);

	// Traverse the channels of the padded image and apply filter on each pixel in each channel
	// Then place the filtered pixel into the relevant channel of the output image
	for (int column = 0; column < im_width; column++) {
		for (int row = 0; row < im_height; row++) {
			for (int channel = 0; channel < im_channels; channel++) {
				// Element-wise multiplication of the relevant part of the image and the filter
				Mat filtered = padded_channels[channel](cv::Rect(column, row, filter_width, filter_height)).mul(filter);
				// Sum the elements of the multiplied matrix and 
				// assign it to the relevant pixel in the relevant channel of the output image
				channels[channel].at<double>(row, column) = sum(filtered)[0];
			}
		}
	}

	// Merge the channels and place it into output image
	merge(channels, outI);

	return outI;
}


Mat hybrid_image_visualize(Mat hybrid_image)
{
	//visualize a hybrid image by progressively downsampling the image and
	//concatenating all of the images together.
	int scales = 5; //how many downsampled versions to create
	double scale_factor = 0.5; //how much to downsample each time
	int padding = 5; //how many pixels to pad.
	int original_height = hybrid_image.rows; // height of the image
	int num_colors = hybrid_image.channels(); //counting how many color channels the input has
	Mat output = hybrid_image;
	Mat cur_image = hybrid_image;

	for (int i = 2; i <= scales; i++)
	{
		//add padding
		hconcat(output, Mat::ones(original_height, padding, CV_8UC3), output);

		//dowsample image;
		resize(cur_image, cur_image, Size(0, 0), scale_factor, scale_factor, INTER_LINEAR);

		//pad the top and append to the output
		Mat tmp;
		vconcat(Mat::ones(original_height - cur_image.rows, cur_image.cols, CV_8UC3), cur_image, tmp);
		hconcat(output, tmp, output);
	}

	return output;
}

Mat DFT_Spectrum(Mat img)
{
	/*
	This function is intended to return the spectrum of an image in a displayable form. Displayable form
	means that once the complex DFT is calculated, the log magnitude needs to be determined from the real
	and imaginary parts. Furthermore the center of the resultant image needs to correspond to the origin of the spectrum.
	*/

	vector<Mat> im_channels(3);
	split(img, im_channels);
	img = im_channels[0];

	/////////////////////////////////////////////////////////////////////
	//STEP 1: pad the input image to optimal size using getOptimalDFTSize()

	Mat padded_img;
	// Calculate the vertical and horizontal paddings for dft
	int vertical_padding = getOptimalDFTSize(img.rows) - img.rows;
	int horizontal_padding = getOptimalDFTSize(img.cols) - img.cols;
	// Create the padded image for dft
	copyMakeBorder(img, padded_img, 0, vertical_padding, 0, horizontal_padding, BORDER_CONSTANT, Scalar::all(0));


	///////////////////////////////////////////////////////////////////
	//STEP 2:  Determine complex DFT of the image.
	// Use the function dft(src, dst, DFT_COMPLEX_OUTPUT) to return a complex Mat variable.
	// The first dimension represents the real part and second dimesion represents the complex part of the DFT

	Mat dft_real, dft_imaginary, complex;
	// Create the real part
	img.convertTo(dft_real, CV_32F);
	// Create a template for the imaginary part
	dft_imaginary = Mat::zeros(img.size(), CV_32F);
	// Put real and imaginary parts into an array
	Mat dft_complex[] = { dft_real, dft_imaginary };
	// Merge real and imaginary layers to get a complex matrix
	merge(dft_complex, 2, complex);
	// Apply dft and place the resulting complex matrix back into the same matrix
	dft(complex, complex, DFT_COMPLEX_OUTPUT);

	////////////////////////////////////////////////////////////////////
	//Step 3: compute the magnitude and switch to logarithmic scale
	//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	Mat magI;

	// Split the complex matrix into its layers
	split(complex, dft_complex);
	// Calculate the magnitude of the complex matrix
	magnitude(dft_complex[0], dft_complex[1], magI);
	// Switch the magnitude to logarithmic state and put the result into magI again
	log(magI + Scalar::all(1), magI);

	///////////////////////////////////////////////////////////////////
	// Step 4:
	/* For visualization purposes the quadrants of the spectrum are rearranged so that the
	origin (zero, zero) corresponds to the image center. To achieve this swap the top left
	quadrant with bottom right quadrant, and swap the top right quadrant with bottom left quadrant
	*/

	//crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// Find the horizontal and vertical centers of the magnitude matrix
	int width = magI.cols, height = magI.rows;
	int center_x = width / 2, center_y = height / 2;

	// Take all 4 quadrants of the magnitude matrix
	Mat quadrant0 = magI({ 0, center_y }, { 0, center_x });
	Mat quadrant1 = magI({ center_y , height }, { 0, center_x });
	Mat quadrant2 = magI({ 0, center_y }, { center_x, width });
	Mat quadrant3 = magI({ center_y, height }, { center_x, width });

	// Switch quadrants (0 with 3 and 1 with 2)
	Mat temp_quadrant;
	quadrant3.copyTo(temp_quadrant);
	quadrant0.copyTo(quadrant3);
	temp_quadrant.copyTo(quadrant0);
	quadrant2.copyTo(temp_quadrant);
	quadrant1.copyTo(quadrant2);
	temp_quadrant.copyTo(quadrant1);

	// Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	normalize(magI, magI, 0, 1, CV_MINMAX);
	return magI;
}


int main()
{
	//Read images
	Mat image1 = imread("../data/dog.bmp");
	if (!image1.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat image2 = imread("../data/cat.bmp");
	if (!image2.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image1.convertTo(image1, CV_64FC3); //CV_64FC3 64-bit Floating point 3 channels
	image2.convertTo(image2, CV_64FC3);


	/*Several additional test cases are provided for you, but feel free to make
	your own(you'll need to align the images in a photo editor such as
	Photoshop).The hybrid images will differ depending on which image you
	assign as image1(which will provide the low frequencies) and which image
	you asign as image2(which will provide the high frequencies) */


	//========================================================================
	//							   PART 1
	//========================================================================

	// IMPLEMENT THE FUNCTION myFilter(Mat,Mat,int)
	// THIS FUNCTION TAKES THREE ARGUMENTS. FIRST ARGUMENT IS THE MAT IMAGE,
	// SECOND ARGUMENT IS THE MAT FILTER AND THE THIRD ARGUMENT SPECIFIES THE
	// PADDING TYPE

	//========================================================================
	//							   PART2
	//========================================================================
	////  FILTERING AND HYBRID IMAGE CONSTRUCTION  ////

	int cutoff_frequency = 7;
	/*This is the standard deviation, in pixels, of the
	Gaussian blur that will remove the high frequencies from one image and
	remove the low frequencies from another image (by subtracting a blurred
	version from the original version). You will want to tune this for every
	image pair to get the best results.*/

	Mat filter = getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency, CV_64F);
	filter = filter * filter.t();



	// YOUR CODE BELOW.
	// Use myFilter() to create low_frequencies of image 1. The easiest
	// way to create high frequencies of image 2 is to subtract a blurred
	// version of image2 from the original version of image2. Combine the
	// low frequencies and high frequencies to create 'hybrid_image'


	Mat low_freq_img;

	Mat high_freq_img;

	Mat hybrid_image;

	// Apply low-pass filter on the first image
	low_freq_img = myFilter(image1, filter);
	// Apply high-pass filter on the second image (by subtracting low-pass filter from original image)
	cv::subtract(image2, myFilter(image2, filter), high_freq_img);
	// Add filtered images to get the hybrid image
	cv::add(low_freq_img, high_freq_img, hybrid_image);


	////  Visualize and save outputs  ////
	//add a scalar to high frequency image because it is centered around zero and is mostly black
	high_freq_img = high_freq_img + Scalar(0.5, 0.5, 0.5) * 255;
	//Convert the resulting images type to the 8 bit unsigned integer matrix with 3 channels
	high_freq_img.convertTo(high_freq_img, CV_8UC3);
	low_freq_img.convertTo(low_freq_img, CV_8UC3);
	hybrid_image.convertTo(hybrid_image, CV_8UC3);

	Mat vis = hybrid_image_visualize(hybrid_image);

	imshow("Low frequencies", low_freq_img); waitKey(0);
	imshow("High frequencies", high_freq_img);	waitKey(0);
	imshow("Hybrid image", vis); waitKey(0);


	imwrite("low_frequencies.jpg", low_freq_img);
	imwrite("high_frequencies.jpg", high_freq_img);
	imwrite("hybrid_image.jpg", hybrid_image);
	imwrite("hybrid_image_scales.jpg", vis);

	//============================================================================
	//							PART 3
	//============================================================================
	//In this part determine the DFT of just one channel of image1 and image2, as well
	// as the DFT of the low frequency image and high frequency image.

	//Complete the code for DFT_Spectrum() method

	Mat img1_DFT = DFT_Spectrum(image1);
	imshow("Image 1 DFT", img1_DFT); waitKey(0);
	imwrite("Image1_DFT.jpg", img1_DFT * 255);

	low_freq_img.convertTo(low_freq_img, CV_64FC3);
	Mat low_freq_DFT = DFT_Spectrum(low_freq_img);
	imshow("Low Frequencies DFT", low_freq_DFT); waitKey(0);
	imwrite("Low_Freq_DFT.jpg", low_freq_DFT * 255);

	Mat img2_DFT = DFT_Spectrum(image2);
	imshow("Image 2 DFT", img2_DFT); waitKey(0);
	imwrite("Image2_DFT.jpg", img2_DFT * 255);

	high_freq_img.convertTo(high_freq_img, CV_64FC3);
	Mat high_freq_DFT = DFT_Spectrum(high_freq_img);
	imshow("High Frequencies DFT", high_freq_DFT); waitKey(0);
	imwrite("High_Freq_DFT.jpg", high_freq_DFT * 255);
}
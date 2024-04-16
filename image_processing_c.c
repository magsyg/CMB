#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

typedef struct {
	float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage *convertToAccurateImage(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	int dim = image->x * image->y;
	imageAccurate->data = (AccuratePixel*)malloc(dim * sizeof(AccuratePixel));
	
	#pragma omp parallel 
	{
		int thread_id = omp_get_thread_num();
		int n_threads = omp_get_num_threads();
		int tilePart = (dim)/n_threads;
		int tileStart = tilePart*thread_id;
		int top = ((tileStart+tilePart)<dim)?((tileStart+tilePart)):dim;
		for(int i = tileStart; i < top; i++) {
			imageAccurate->data[i].red = (float) image->data[i].red;
			imageAccurate->data[i].green = (float) image->data[i].green;
			imageAccurate->data[i].blue = (float) image->data[i].blue;
		}
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
	return imageAccurate;
}
AccurateImage *convertToBlankAccurateImage(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	int dim = image->x * image->y;
	imageAccurate->data = (AccuratePixel*)malloc(dim * sizeof(AccuratePixel));

	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
	return imageAccurate;
}

PPMImage * convertToPPPMImage(AccurateImage *imageIn) {
    PPMImage *imageOut;
    imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	int dim = imageIn->x * imageIn->y;
    imageOut->data = (PPMPixel*)malloc(dim * sizeof(PPMPixel));

    imageOut->x = imageIn->x;
    imageOut->y = imageIn->y;

	#pragma omp parallel 
	{
		int thread_id = omp_get_thread_num();
		int n_threads = omp_get_num_threads();
		int tilePart = (dim)/n_threads;
		int tileStart = tilePart*thread_id;
		int top = ((tileStart+tilePart)<dim)?((tileStart+tilePart)):dim;

		for(int i = tileStart; i < top; i++) {
			imageOut->data[i].red = imageIn->data[i].red;
			imageOut->data[i].green = imageIn->data[i].green;
			imageOut->data[i].blue = imageIn->data[i].blue;
		}
	}
    return imageOut;
}

// blur one color channel
void blurCornersIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
	unsigned short corners[4][4] = {
		{
			0, size,
			0, imageIn->x
		}, 
		{
			imageIn->y-size, imageIn->y,
			0, imageIn->x
		}, 
		{
			0, imageIn->y,
			0, size,
		}, 
		{
			0, imageIn->y,
			imageIn->x-size, imageIn->x,
		}
	};

	
	for(unsigned short  i = 0; i < 4; i++) {	
	#pragma omp parallel 
	{
		unsigned short thread_id = omp_get_thread_num();
		unsigned short n_threads = omp_get_num_threads();
		short sY = corners[i][0], eY = corners[i][1];
		short sX = corners[i][2], eX = corners[i][3];
		if (i >= 2) {
			unsigned short tilePart = (eY-sY)/n_threads;
			sY = tilePart*thread_id + sY;
			eY = ((sY+tilePart)< corners[i][1])?((sY+tilePart)):(corners[i][1]);
		} else {
			unsigned short tilePart = (eX-sX)/n_threads;
			sX = tilePart*thread_id + sX;
			eX = ((sX+tilePart)< corners[i][3])?((sX+tilePart)):(corners[i][3]);
		}
		for(short senterY = sY; senterY < eY; senterY++) {
			unsigned short endY = (senterY+size < imageIn->y)? senterY+size+1:imageIn->y;
			unsigned short startY = ((senterY-size)>0)? senterY-size:0;
			for(short senterX = sX; senterX < eX; senterX++) {
					// For each pixel we compute the magic number
					float sumR = 0, sumG = 0, sumB = 0;
					unsigned short endX = (senterX+size < imageIn->x)? senterX+size+1:imageIn->x;
					unsigned short startX = ((senterX-size)>0)? senterX-size:0;
					float countIncluded = (endX-startX)*(endY-startY);
					for(unsigned short x = startX; x < endX; x++) {
						for(unsigned short  y = startY; y < endY; y++) {
							// Now we can begin
							int offsetOfThePixel = (imageIn->x * y + x);
							sumR += imageIn->data[offsetOfThePixel].red;
							sumG += imageIn->data[offsetOfThePixel].green;
							sumB += imageIn->data[offsetOfThePixel].blue;
							// Keep track of how many values we have included
						}

					}
					int offsetOfThePixel = (imageOut->x * senterY + senterX);
					imageOut->data[offsetOfThePixel].red = sumR/countIncluded;
					imageOut->data[offsetOfThePixel].green = sumG/countIncluded;
					imageOut->data[offsetOfThePixel].blue = sumB/countIncluded;
				}

			}
		}
	}
}
void blurIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
	
	// Iterate over each pixel
	float numElements = (2*size+1)*(2*size+1);
	numElements = pow(numElements,-1);
	blurCornersIteration(imageOut, imageIn, size);
	#pragma omp parallel 
	{
		unsigned short  thread_id = omp_get_thread_num();
		unsigned short  n_threads = omp_get_num_threads();
		unsigned short  tilePart = imageIn->y/n_threads;
		unsigned short  tileStart = tilePart*thread_id + size;
		unsigned short topY = ((tileStart+tilePart)< imageIn->y-size)?((tileStart+tilePart)):(imageIn->y-size);
		for(unsigned short senterY = tileStart; senterY < topY; senterY++) {
			int offsetOfThePixel = (imageIn->x * senterY + size);
			unsigned short  bottomY = senterY-size;
			unsigned short topY = senterY+size;
			float sumR = 0, sumG = 0, sumB = 0;
			for(unsigned short  x = 0; x <= (size+size); x++) {
				for(unsigned short y = bottomY; y <= topY; y++) {
					int offsetOfThePixel = (imageIn->x * y + x);
					sumR += imageIn->data[offsetOfThePixel].red;
					sumG += imageIn->data[offsetOfThePixel].green;
					sumB += imageIn->data[offsetOfThePixel].blue;
				}
			}
			imageOut->data[offsetOfThePixel].red = sumR*numElements;
			imageOut->data[offsetOfThePixel].green = sumG*numElements;
			imageOut->data[offsetOfThePixel].blue = sumB*numElements;
			int yRow = imageIn->x * bottomY;
			for(unsigned short senterX = size+1; senterX < imageIn->x-size; senterX++) {
				// For each pixel we compute the magic number
				offsetOfThePixel = (imageIn->x * senterY + senterX);
				unsigned short leftX = senterX-size-1;
				unsigned short  rightX = senterX+size;
				float sumAddR = 0, sumSubR = 0;
				float sumAddG = 0, sumSubG = 0;
				float sumAddB = 0, sumSubB = 0;
				int leftOffset = (yRow+ leftX);
				int rightOffset = (yRow + rightX);
				for(int y = bottomY; y <= topY; y++) {
					sumSubR += imageIn->data[leftOffset].red;
					sumAddR += imageIn->data[rightOffset].red;
					sumSubG += imageIn->data[leftOffset].green;
					sumAddG += imageIn->data[rightOffset].green;
					sumAddB += imageIn->data[rightOffset].blue;
					sumSubB += imageIn->data[leftOffset].blue;
					leftOffset+=imageIn->x;
					rightOffset+=imageIn->x;
				}
				sumR= sumR+sumAddR-sumSubR;
				sumG= sumG+sumAddG-sumSubG;
				sumB= sumB+sumAddB-sumSubB;

				imageOut->data[offsetOfThePixel].red = sumR*numElements;
				imageOut->data[offsetOfThePixel].green = sumG*numElements;
				imageOut->data[offsetOfThePixel].blue = sumB*numElements;
			}

		}
	}	
}


// Perform the final step, and return it as ppm.
PPMImage * imageDifference(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
	PPMImage *imageOut;
	
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	int dim = imageInSmall->x * imageInSmall->y;
	imageOut->data = (PPMPixel*)malloc(dim * sizeof(PPMPixel));
	
	imageOut->x = imageInSmall->x;
	imageOut->y = imageInSmall->y;
	
	#pragma omp parallel 
	{
		int thread_id = omp_get_thread_num();
		int n_threads = omp_get_num_threads();
		int tilePart = (dim)/n_threads;
		int tileStart = tilePart*thread_id;
		int top = ((tileStart+tilePart)<dim)?((tileStart+tilePart)):dim;
		for(int i = tileStart; i < top; i++) {
			float value = (imageInLarge->data[i].red - imageInSmall->data[i].red);

			if (value < -1.0) {
				value = 257.0+value;
			} else if (value > -1.0 && value < 0.0) {
				value = 0;
			}

			if(value > 255)
				imageOut->data[i].red = 255.0;
			else {
				imageOut->data[i].red = floor(value);
			}

			value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
			if (value < -1.0) {
				value = 257.0+value;
			} else if (value > -1.0 && value < 0.0) {
				value = 0;
			} 
			if(value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);

			value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
			if (value < -1.0) {
				value = 257.0+value;
			} else if (value > -1.0 && value < 0.0) {
				value = 0;
			}
			if(value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		}
	}
	return imageOut;
}



int main(int argc, char** argv) {
    // read image
    PPMImage *image;
    // select where to read the image from
    if(argc > 1) {
        // from file for debugging (with argument)
        image = readPPM("flower.ppm");
    } else {
        // from stdin for cmb
        image = readStreamPPM(stdin);
    }
	
	AccurateImage *imageAccurate = convertToAccurateImage(image);

	AccurateImage *imageAccurate1_tiny = convertToBlankAccurateImage(image);
	AccurateImage *imageAccurate2_tiny = convertToBlankAccurateImage(image);
	
	// Process the tiny case:
	int size = 2;
	blurIteration(imageAccurate2_tiny, imageAccurate, size);
	
	blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, size);
	blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, size);
	blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, size);
	blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, size);
	
	
	
	AccurateImage *imageAccurate1_small = convertToBlankAccurateImage(image);
	AccurateImage *imageAccurate2_small = convertToBlankAccurateImage(image);
	
	// Process the small case:

	size = 3;
	blurIteration(imageAccurate2_small, imageAccurate, size);
	blurIteration(imageAccurate1_small, imageAccurate2_small, size);
	blurIteration(imageAccurate2_small, imageAccurate1_small, size);
	blurIteration(imageAccurate1_small, imageAccurate2_small, size);
	blurIteration(imageAccurate2_small, imageAccurate1_small, size);


    // an intermediate step can be saved for debugging like this
//    writePPM("imageAccurate2_tiny.ppm", convertToPPPMImage(imageAccurate2_tiny));
	
	AccurateImage *imageAccurate1_medium = convertToBlankAccurateImage(image);
	AccurateImage *imageAccurate2_medium = convertToBlankAccurateImage(image);
	
	// Process the medium case:

	size = 5;
	blurIteration(imageAccurate2_medium, imageAccurate, size);
	blurIteration(imageAccurate1_medium, imageAccurate2_medium, size);
	blurIteration(imageAccurate2_medium, imageAccurate1_medium, size);
	blurIteration(imageAccurate1_medium, imageAccurate2_medium, size);
	blurIteration(imageAccurate2_medium, imageAccurate1_medium, size);

	
	AccurateImage *imageAccurate1_large = convertToBlankAccurateImage(image);
	AccurateImage *imageAccurate2_large = convertToBlankAccurateImage(image);
	
	// Do each color channel
	size = 8;
	blurIteration(imageAccurate2_large, imageAccurate, size);
	blurIteration(imageAccurate1_large, imageAccurate2_large, size);
	blurIteration(imageAccurate2_large, imageAccurate1_large, size);
	blurIteration(imageAccurate1_large, imageAccurate2_large, size);
	blurIteration(imageAccurate2_large, imageAccurate1_large, size);

	// calculate difference
	PPMImage *final_tiny = imageDifference(imageAccurate2_tiny, imageAccurate2_small);
    PPMImage *final_small = imageDifference(imageAccurate2_small, imageAccurate2_medium);
    PPMImage *final_medium = imageDifference(imageAccurate2_medium, imageAccurate2_large);
	// Save the images.
    if(argc > 1) {
        writePPM("flower_tiny.ppm", final_tiny);
        writePPM("flower_small.ppm", final_small);
        writePPM("flower_medium.ppm", final_medium);
    } else {
        writeStreamPPM(stdout, final_tiny);
        writeStreamPPM(stdout, final_small);
        writeStreamPPM(stdout, final_medium);
    }
	
}

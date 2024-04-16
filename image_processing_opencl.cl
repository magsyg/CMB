__kernel void naive_kernel(
        __global const float* restrict in_image,
        __global float* restrict out_image,
        const int width,
        const int height,
        const int size
)
{
    // global 2D NDRange sizes (size of the image)
    // point in currently being executed (each pixel)
    ushort senterY = get_global_id(0);
    
    int pixelPos = (width* senterY + size);
    unsigned short bottomY = senterY-size;
    unsigned short topY = senterY+size;
    

    if (bottomY < 0 ) bottomY = 0;
    if (topY > height ) topY = height;
    int numElements = (2*size+1)*(topY-bottomY);

	float3 sum = (float3) (0.0f, 0.0f, 0.0f);
	for(unsigned short  x = 0; x <= (size+size); x++) {
		for(unsigned short y = bottomY; y < topY; y++) {
			int offsetOfThePixel = (width * y + x);
			sum+=vload3(offsetOfThePixel, in_image);
		}
		}

		vstore3(sum / numElements, pixelPos, out_image);

		int yRow = width * bottomY;

		for(unsigned short senterX = size+1; senterX < width-size; senterX++) {
		// For each pixel we compute the magic number
		pixelPos = (width * senterY + senterX);

		unsigned short leftX = senterX-size-1;
		unsigned short  rightX = senterX+size;
		int leftOffset = (yRow+ leftX);
		int rightOffset = (yRow + rightX);

		for(int y = bottomY; y < topY; y++) {
			sum+=vload3(rightOffset, in_image);
			sum-=vload3(leftOffset, in_image);
			leftOffset+=width;
			rightOffset+=width;
		}
		vstore3(sum / numElements, pixelPos, out_image);

	}
	
   	/*
    ushort startY = ((senterY-size) > 0)? (senterY-size): 0;
  
    ushort endY = senterY+size+1;
    if (endY >= height) endY = height;
    int start = width * (startY-1);
    int pixelPos = (width * senterY);
    float3 sum = (float3) (0.0f, 0.0f, 0.0f);
    // Start accumulation
    
    for(ushort x = 0; x < size; x++) {
        int offsetOfThePixel = (start + x);
        for(ushort y = startY; y < endY; y++) {
            // Now we can begin
            offsetOfThePixel+=width;
            sum += vload3(offsetOfThePixel, in_image);
            // Keep track of how many values we have included
        }
    }
    // Fill accumulation

    vstore3(sum / ((size+1)*(endY-startY)), pixelPos, out_image);

    for(ushort x = 1; x < size+2; x++) {
  
        int offsetOfThePixel = (start + x + size);
        pixelPos++;
         
        for(ushort y = startY; y < endY; y++) {
            // Now we can begin
            offsetOfThePixel+=width;
            sum += vload3(offsetOfThePixel, in_image);
        }  
        vstore3(sum / ((size+1+x)*(endY-startY)), pixelPos, out_image);
    }
    
    // Full accumulation
    int count = (2*size+1)*(endY-startY);

    for(ushort x = size+2; x < width-size; x++) {
    	
        pixelPos++;
        int leftOffset = (start+x-size-1);
        int rightOffset = (start+x+size);
	
        for(ushort y = startY; y < endY; y++) {
            leftOffset+=width;
            rightOffset+=width;
            sum -= vload3(leftOffset, in_image);
            sum += vload3(rightOffset, in_image);
            
        }
        vstore3(sum/count, pixelPos, out_image);
	
    }

    // End of accumulation
    for(ushort x = width-size; x < width; x++) {
        pixelPos++;
        int leftOffset = (start+x-size+1);
        for(ushort y = startY; y < endY; y++) {
            leftOffset+=width;
            sum -= vload3(leftOffset, in_image);
        }
        vstore3(sum / ((width-x+size)*(endY-startY)), pixelPos, out_image);
    }
	*/
}

__kernel void naive_kernel(
        __global const float* restrict in_image,
        __global float* restrict out_image,
        const int width,
        const int height,
        const int size
)
{
    // global 2D NDRange sizes (size of the image)
    ushort num_cols = get_global_size(0);
    ushort num_rows = get_global_size(1);
    // point in currently being executed (each pixel)
    ushort senterY = get_global_id(1);
   
    ushort startY = ((senterY-size) > 0)? (senterY-size): 0;
  
    ushort endY = senterY+size+1;
    if (endY >= height) endY = height;//((senterY+size) < height) ? (start+size+1): height;
    int start = num_cols * (startY-1);
    int pixelPos = (num_cols * senterY);
    float3 sum = (float3) (0.0f, 0.0f, 0.0f);
    // Start accumulation
    
    for(ushort x = 0; x < size; x++) {
        int offsetOfThePixel = (start + x);
        for(ushort y = startY; y < endY; y++) {
            // Now we can begin
            offsetOfThePixel+=num_cols;
            sum += vload3(offsetOfThePixel, in_image);
            // Keep track of how many values we have included
        }
    }
   
    // Fill accumulation
    vstore3(sum / (size+1)*(endY-startY), pixelPos, out_image);
    
    for(ushort x = 1; x < size+1; x++) {
  
        int offsetOfThePixel = (start + x);
        pixelPos++;
         
        for(ushort y = startY; y < endY; y++) {
            // Now we can begin
            offsetOfThePixel+=num_cols;
            sum += vload3(offsetOfThePixel, in_image);
            // Keep track of how many values we have included
        }  
        vstore3(sum / (size+1+x)*(endY-startY), pixelPos, out_image);
    }

    // Full accumulation
    int count = (2*size+1)*(endY-startY);
    
    for(ushort x = 1; x < width-size; x++) {
    	
        pixelPos++;
        int leftOffset = (start - size+1);
        int rightOffset = (start + size);
        
        for(ushort y = startY; y < endY; y++) {
            leftOffset+=width;
            rightOffset+=width;
            sum -= vload3(leftOffset, in_image);
            sum += vload3(rightOffset, in_image);
            
        }
        vstore3(sum / count, pixelPos, out_image);

    }
    
    // End of accumulation
    for(ushort x = num_cols-size; x < num_cols; x++) {
        pixelPos++;
        int leftOffset = (start - size+1);
        for(ushort y = startY; y < endY; y++) {
            leftOffset+=num_cols;
            sum -= vload3(leftOffset, in_image);
        }
        vstore3(sum / (num_cols-x+size)*(endY-startY), pixelPos, out_image);
    }
}

#ifndef CUDA_HOG_H
#define CUDA_HOG_H

/** 
 * Normalize oriented gradient histogram using neighboring cells.
 */
__global__ NormalizeHogCells() {

}

/**
 * Computes oriented gradient histogram in cells (unnormalized).  Returns result
 * in array bins of dimension (width / dcell)x(height / dcell)x(nbins).
 */
__global__ HogCells(float *grad, float *bins, size_t width, size_t height, size_t dcell,
    size_t nbins, bool symmetric) {
}

/**
 * Compute gradient along each dimension by convolving with [-1, 0, 1] and 
 * [-1, 0, 1]^T.  Returns result in grad array of dim (width)x(height)x2.
 */
__global__ Gradient(uint8_t *image, float *grad, size_t width, size_t height) {
}

/**
 * Computes histogram of oriented gradients.
 */
void CudaHog(uint8_t * image, size_t width, size_t height, float * hog,
     size_t dcell, size_t dblock, size_t nbins, bool symmetric) {

}

#endif

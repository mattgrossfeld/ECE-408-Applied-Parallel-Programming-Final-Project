
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>
namespace mxnet
{
namespace op
{


#define TILE_SIZE 16
#define KERNEL_SIZE 5
#define MASK_WIDTH 24 * 12 * 7 * 7
__constant__ float Mask [MASK_WIDTH];


__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    extern __shared__ float shmem[];
    float* X_shared = shmem;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil((1.0 * W_out) / TILE_SIZE);

    // An example use of these macros:
    // float a = y4d(0,0,0,0)

    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int n, m, h, w, c, p, q;
    int tile_width = TILE_SIZE + K - 1;

    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.x;
    w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.y;
    
    float acc = 0.0;
    bool stopLoop = false;
    int firstStop;
    int secondStop;
    for (c = 0; c < C; c++) {
        __syncthreads();
        firstStop = min(h + tile_width - threadIdx.x, H);
        for (int i = h; i < firstStop; i += TILE_SIZE) {
            secondStop = min(w + tile_width - threadIdx.y, W);
            for (int j = w; j < secondStop; j += TILE_SIZE) {
                X_shared[(i - h + threadIdx.x) * tile_width + j - w + threadIdx.y] = x4d(n, c, i, j);
            }
        }
        __syncthreads();
        for (p = 0; p < K; p++) {
            #pragma unroll 25
            for (q = 0; q < (K - (K % 5)); q += 5) {
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q] * k4d(m, c, p, q);
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q + 1] * k4d(m, c, p, q + 1);
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q + 2] * k4d(m, c, p, q + 2);
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q + 3] * k4d(m, c, p, q + 3);
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q + 4] * k4d(m, c, p, q + 4);
            }
            for (q = (K - (K % 5)); q < K; q += 5) {
                acc = acc + X_shared[(threadIdx.x + p) * tile_width + threadIdx.y + q] * k4d(m, c, p, q);
            }
        }
    }
    if (h < H_out && w < W_out) {
      y4d(n, m, h, w) = acc; 
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    cudaMemcpyToSymbol(Mask, w.dptr_, KERNEL_SIZE * KERNEL_SIZE * M * C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t shmem_size = ((TILE_SIZE + K-1) * (TILE_SIZE + K-1)) * sizeof(float);


    const int Z = ceil((1.0 * W_out) / TILE_SIZE) * ceil((1.0* H_out)/ TILE_SIZE);

    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    //Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif

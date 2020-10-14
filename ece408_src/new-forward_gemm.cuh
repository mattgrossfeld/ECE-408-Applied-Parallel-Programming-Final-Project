
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define BLOCK_WIDTH 1024

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 32

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int b,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

  __shared__ float TileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float TileN[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x; int ty = threadIdx.y;
  int bx = blockIdx.x; int by = blockIdx.y;

  int r = by*TILE_WIDTH+ty; int c = bx*TILE_WIDTH+tx;

  int offset = b*numCRows*numCColumns;
  //int offset = 0;

  float pv = 0;
  for (int m=0; m<ceil(numAColumns/(1.0*TILE_WIDTH));m++){
    int row = m*TILE_WIDTH+threadIdx.y;
    int col = m*TILE_WIDTH+threadIdx.x;


    if (col < numAColumns)
        TileM[ty][tx] = A[r*numAColumns+col];
    else
        TileM[ty][tx] = 0.0;

    if (row < numAColumns)
        TileN[ty][tx] = B[row*numBColumns+c];
    else
        TileN[ty][tx] = 0.0;

    __syncthreads();
    for (int k=0;k<TILE_WIDTH;k++){
      pv += TileM[ty][k]*TileN[k][tx];
    }
   //__syncthreads();

  }

  if ((r < numCRows) && (c < numCColumns))
    C[offset + r*numCColumns+c] = pv;

}

__global__ void unroll_Kernel(float* x, float* x_unroll, int b, int C, int H, int W, int K){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;

    int H_out = H-K+1;
    int W_out = W-K+1;

    if (idx < C*K*K*H_out*W_out){

        int row = idx / (H_out*W_out);
        int col = idx % (H_out*W_out);

        int row_K = row/K;
        //int offset = C*H*W*b;
        int offset = C*H*W*b;

        x_unroll[idx] = x[offset + (row_K / K)*(H*W) + (col / W_out + row_K % K)*W + col % W_out + row%K];
    }
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
    float* x_unrolled;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int unrolled_size = C*K*K*H_out*W_out;

    cudaMalloc(&x_unrolled, sizeof(float)*unrolled_size);
    //int b = B;


    dim3 gridDim (ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    
    for (int b = B; b--; ) {
        unroll_Kernel<<<ceil(unrolled_size/(1.0*BLOCK_WIDTH)), BLOCK_WIDTH>>>(x.dptr_, x_unrolled, b, C, H, W, K);

        matrixMultiplyShared<<<gridDim, blockDim>>>(w.dptr_, x_unrolled, y.dptr_, b, K ,C*K*K, C*K*K, H_out*W_out ,M, H_out*W_out);

        //b--;
    }

    cudaFree(x_unrolled);

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

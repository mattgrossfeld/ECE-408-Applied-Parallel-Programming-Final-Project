
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define FUSION_TILE_WIDTH 16

#define MASK_WIDTH 14112

__constant__ float Mask[MASK_WIDTH];

__global__ void fusion_gemm_Kernel(float* x, float *y, int C, int H, int W, int K,
                                     int M, int H_out, int W_out){

  __shared__ float TileM[FUSION_TILE_WIDTH][FUSION_TILE_WIDTH];
  __shared__ float TileN[FUSION_TILE_WIDTH][FUSION_TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int b = blockIdx.z; //batch number
  int tx = threadIdx.x; int ty = threadIdx.y;

  int CKK = C*K*K;
  int outFeatureMapSize = H_out*W_out;

  int row_out = by*FUSION_TILE_WIDTH+ty; int col_out = bx*FUSION_TILE_WIDTH+tx;

  float pv = 0;
  for (int m=0; m<ceil(1.0*CKK/FUSION_TILE_WIDTH); m++){
    int row = m*FUSION_TILE_WIDTH+ty;
    int col = m*FUSION_TILE_WIDTH+tx;

    if (col < CKK)
        TileM[ty][tx] = Mask[row_out*CKK + col];
    else
        TileM[ty][tx] = 0.0;

    if (row < CKK){
        int c = row/(K*K);
        int h = col_out / W_out;
        int w = col_out % W_out;
        int p = (row / K) % K;
        int q = row % K;

        TileN[ty][tx] = x[b*C*H*W + c*H*W + (h+p)*W + w + q];
    }else
        TileN[ty][tx] = 0.0;

    __syncthreads();
    for (int k=0; k<FUSION_TILE_WIDTH; k++){
      pv += TileM[ty][k]*TileN[k][tx];
    }
   __syncthreads();

  }

  if ((row_out < M) && (col_out < outFeatureMapSize))
    y[b*M*outFeatureMapSize + row_out*outFeatureMapSize + col_out] = pv;
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

    cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float)*M*C*K*K);

    dim3 gridDim (ceil(1.0*H_out*W_out/FUSION_TILE_WIDTH), ceil(1.0*M/FUSION_TILE_WIDTH), B);
    dim3 blockDim(FUSION_TILE_WIDTH, FUSION_TILE_WIDTH, 1);

    fusion_gemm_Kernel<<<gridDim, blockDim>>>(x.dptr_, y.dptr_, C, H, W, K, M, H_out, W_out);

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

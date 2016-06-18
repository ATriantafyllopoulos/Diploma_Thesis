#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float3 operator+(const float3 &a, const float3 &b);
__device__ float3 operator/(const float3 &a, const float &b);
__device__ float3 operator*(const float3 &a, const float &b);
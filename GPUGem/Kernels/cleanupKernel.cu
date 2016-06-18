#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
Using void argument to make this function compatible with all types of pointers.
*/
cudaError_t cleanup(void** pt)
{
	cudaError_t cudaStatus = cudaFree(*pt);
	return cudaStatus;
}
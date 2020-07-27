#include "softmax_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>
#include <math.h>
#include <vector>


__global__ void calculate_exponent_and_sum(float* value, float* sum, float* Z, int Z_x_dim, int Z_y_dim){
    // Find unique ID of each thread row and thread column
	int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize max array to store maximum of each row
    float max[Z_y_dim] = {-INFINITY};
    // Loop over the row
	for (size_t i = 0; i < Z_x_dim; i++){
    // Make sure the index doesnt exceed the number of elements in matrix
    	if(thread_row * Z_x_dim + i < Z_x_dim * Z_y_dim){
      	// If it is greater, put if in the max for the corresponding row.
    		if(Z[thread_row * Z_x_dim + i] > max[thread_row]){
      			max[thread_row] = Z[thread_row * Z_x_dim + i];
      		}
    	}
  	}
  	// Get unique index id for each thread
  	int index = thread_row * Z_x_dim + thread_col;
  	// Make sure that the thread_col is not greater than the number of rows in matrix
  	if (thread_col < Z_y_dim){
    	// Calculate exponent by subtracting each value by the max of that row
  		value[index] = expf(Z[index] - max[thread_row]);
  	}
  	// Populate sum array
  	for(size_t i = 0; i < Z_x_dim; i++){
    	// Make sure that the row ID is not greater than the number of rows in matrix
    	if(thread_row < Z_y_dim){
      	// populate each rows sum
    		sum[thread_row] += value[thread_row * Z_x_dim + i];
    	}
  	}
}


 Matrix& SoftmaxActivation::Calculate_Exponent_and_Sum(Matrix& Z){
	
	value.allocateMemoryIfNotAllocated(Z.shape.x * Z.shape.y);

	dim3 block_size(128, 128);
	dim3 num_of_blocks( (Z.shape.x + block_size.x - 1)/ block_size.x,
						(Z.shape.y + block_size.y - 1)/ block_size.y);
	calculate_exponent_and_sum<<<num_of_blocks, block_size>>>(value.data_device.get(),
														sum.data_device.get(),
														Z.data_device.get(),
														Z.shape.x, Z.shape.y,
														);
}


__global__ void softmaxActivationForward(float* Z, float* A, float* value,
										 int Z_x_dim, int Z_y_dim) {

	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Z_x_dim * Z_y_dim) {
		A[index] = value[i];
	}
}
    	
  	
__global__ void softmaxActivationBackprop(float* Z, float* dA, float* dZ,
										  int* value, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		dZ[index] = dA[index] * value[i] * (1 - value[i]);
	}
}

SoftmaxActivation::SoftmaxActivation(std::string name) {
	this->name = name;
}

SoftmaxActivation::~SoftmaxActivation()
{ }

Matrix& SoftmaxActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	Calculate_Exponent_and_Sum(Z);

	cudaDeviceSynchronize();

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	value, Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

	return A;
}

Matrix& SoftmaxActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(), value
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

	return dZ;
}


#include "softmax_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>
#include <math.h>
#include <vector>


__global__ void softmaxActivationForward(float* input, float* A, float* max_num, float* row_sum, int Z_x_dim, int Z_y_dim) {
    
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (int i = 0; i < Z_y_dim; i++) {
		max_num[i] = -INFINITY;
	}

	if (idx < Z_y_dim ) {
		for (int i = 0; i < Z_x_dim; i++){ 
			if(max_num[idx] < input[idx * Z_x_dim + i] ){
				max_num[idx] = input[idx * Z_x_dim + i];
			}
			
		}
	}
	__syncthreads();

	if(idx < Z_x_dim){
		for(size_t i = 0; i < Z_x_dim; i++)
			A[idx * Z_y_dim + i] = expf(input[idx * Z_y_dim + i] - max_num[idx]);
	}

	if(idx < Z_x_dim){
		for(size_t i = 0; i < Z_x_dim; i++)
			row_sum[idx] += A[idx * Z_y_dim + i];
		
	}
	if(idx < Z_x_dim){
		for(size_t i = 0; i < Z_x_dim; i++)
			A[idx * Z_y_dim + i] = A[idx * Z_y_dim + i] / row_sum[idx];
	}


}
/*
void SoftmaxActivation::softmax_act(Matrix& Z){
	
	A.allocateMemoryIfNotAllocated(Z.shape.x * Z.shape.y);
	Shape shape_Y = Shape(Z.shape.y, 1);
	max_num.allocateMemoryIfNotAllocated(shape_Y);
	row_sum.allocateMemoryIfNotAllocated(shape_Y);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	softmax<<<num_of_blocks, block_size>>>(Z.data_device.get(),
											A.data_device.get(),
											max_num.data_device.get(),
											row_sum.data_device.get(),
											Z.shape.x, Z.shape.y
											);
}


__global__ void softmaxActivationForward(float* Z, float* A, float* A,
										 int Z_x_dim, int Z_y_dim) {

	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Z_x_dim * Z_y_dim) {
		A[index] = A[index];
	}
}
*/    	
  	
__global__ void softmaxActivationBackprop(float* Z, float* dA, float* dZ,
										  float* A, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		dZ[index] = dA[index] * A[index] * (1 - A[index]);
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
	Shape shape_Y = Shape(Z.shape.y, 1);
	max_num.allocateMemoryIfNotAllocated(shape_Y);
	row_sum.allocateMemoryIfNotAllocated(shape_Y);

	
	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(),
															A.data_device.get(),
															max_num.data_device.get(),
															row_sum.data_device.get(),
															Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

	return A;
}

Matrix& SoftmaxActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(), A.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

	return dZ;
}


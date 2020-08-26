#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include "softmax_activation.hh"

__global__ void softmax(float* input, float* buffer, float* max_num, float* row_sum, int Z_x_dim, int Z_y_dim) {
  	assert(input);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d \n", idx);
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
  	// assert(input_len >= 0);  Not needed
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
			buffer[idx * Z_y_dim + i] = expf(input[idx * Z_y_dim + i] - max_num[idx]);
	}

	if(idx < Z_x_dim){
		for(size_t i = 0; i < Z_x_dim; i++)
			row_sum[idx] += buffer[idx * Z_y_dim + i];
		
	}
	if(idx < Z_x_dim){
		for(size_t i = 0; i < Z_x_dim; i++)
			buffer[idx * Z_y_dim + i] = buffer[idx * Z_y_dim + i] / row_sum[idx];
	}


}

int main()
{
	
	float Z[25*25];
	int i ;
 	for( i = 0; i < 25*25; ++i){
		Z[i] = i+1;
    }

  	float buffer[25*25];

 	float *max_num = new float[25];
 	float *row_sum = new float[25];

 	float *Z_d, *buffer_d, *max_num_d, *row_sum_d;
 	cudaMalloc((void **)&Z_d, 25*25*sizeof(float));
 	cudaMalloc((void **)&buffer_d, 25*25*sizeof(float));
 	cudaMalloc((void **)&max_num_d, 25*sizeof(float));
 	cudaMalloc((void **)&row_sum_d, 25*sizeof(float));

 	cudaMemcpy(Z_d, Z, 25*25*sizeof(float), cudaMemcpyHostToDevice);
 	cudaMemcpy(buffer_d, buffer, 25*25*sizeof(float), cudaMemcpyHostToDevice);
 	cudaMemcpy(max_num_d, max_num, 25*sizeof(float), cudaMemcpyHostToDevice);
 	cudaMemcpy(row_sum_d, row_sum, 25*sizeof(float), cudaMemcpyHostToDevice);

 	dim3 block_size(64);
	dim3 num_of_blocks((25 * 25 + block_size.x - 1) / block_size.x);

 	softmax<<<num_of_blocks, block_size>>>(Z_d, buffer_d, max_num_d, row_sum_d, 25, 25);
	
	cudaMemcpy(buffer, buffer_d, 25*25*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_num, max_num_d, 25*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_sum, row_sum_d, 25*sizeof(float), cudaMemcpyDeviceToHost);
 	std::cout<<"buffer"<<std::endl;
 	
	for(i = 0; i < 25*25; i++){
    	std::cout<<buffer[i]<<std::endl;
	}
	std::cout<<"max num"<<std::endl;
 	
	for(i = 0; i < 25; i++){
    	std::cout<<max_num[i]<<std::endl;
	}
	std::cout<<"row sum"<<std::endl;
 	
	for(i = 0; i < 25; i++){
    	std::cout<<row_sum[i]<<std::endl;
	}

 	return 0;

}
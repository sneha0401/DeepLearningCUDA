#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include "softmax_activation.hh"

__global__ void softmax(float* input, float* buffer, float* max_num, int Z_x_dim, int Z_y_dim) {
  	assert(input);

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
  	// assert(input_len >= 0);  Not needed
	for (int i = 0; i < Z_y_dim; i++) {
		max_num[i] = -INFINITY;
	}

	if (row < Z_y_dim ) {
		for (int i = 0; i < Z_x_dim; i++) {
			if(max_num[i] < input[row * Z_x_dim + i] ){
				max_num[row] = input[row * Z_x_dim + i];
			}
		}
	}
}
int main()
{
	
	float Z[25][25];
	int i ,j;
 	for( i = 0; i < 25*25; ++i){
 		for( j =0; j < 25; ++j){
  			Z[i][j] = rand();
    	}
    }

  	float buffer[25][25];

 	for( i = 0; i < 25*25; ++i){
 		for( j =0; j < 25; ++j){
  			buffer[i][j] = rand();
    	}
    }

 	float *max_num = new float[25];

 	float *Z_d, *buffer_d, *max_num_d;
 	cudaMalloc((void **)&Z_d, 25*25*sizeof(float));
 	cudaMalloc((void **)&buffer_d, 25*25*sizeof(float));
 	cudaMalloc((void **)&max_num_d, 25*sizeof(float));

 	cudaMemcpy(Z_d, Z, 25*25*sizeof(float), cudaMemcpyHostToDevice);
 	cudaMemcpy(buffer_d, buffer, 25*25*sizeof(float), cudaMemcpyHostToDevice);
 	cudaMemcpy(max_num_d, max_num, 25*sizeof(float), cudaMemcpyHostToDevice);

 	dim3 block_size(64);
	dim3 num_of_blocks((25 * 25 + block_size.x - 1) / block_size.x);

 	softmax<<<num_of_blocks, block_size>>>(Z_d, buffer_d, max_num_d, 25, 25);
/*
 	for(i = 0; i < 10; i++){
 		for( j = 0; j < 3; ++j)
    		std::cout<<buffer[i][j]<<'\t';
	}
*/
 	cudaMemcpy(max_num, max_num_d, 25*sizeof(float), cudaMemcpyDeviceToHost);
 	for(i = 0; i < 25; i++){
 		std::cout<<max_num[i]<<std::endl;
 	}
 	return 0;

}
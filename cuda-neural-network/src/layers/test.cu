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
	
	float Z[25][25], i ,j;
 	for( i = 0; i < 25; ++i){
  		for( j = 0;  j < 25; ++j){
     		Z[i][j] = rand();
     	}
  	}

  	float buffer[25][25];
 	
 	for( i = 0; i < 25; ++i){
 		for( j = 0; j < 25; ++j)
    		std::cout<<Z[i][j]<<'\t';
   		std::cout<<'\n';
 	}
 	float *max_num = new float[25];

 	softmax<<<2, 20>>>(Z, buffer, max_num, 25, 25);
 	
 	for(i = 0; i < 10; i++){
 		for( j = 0; j < 3; ++j)
    		std::cout<<buffer[i][j]<<'\t';
	}
 	return 0;

}
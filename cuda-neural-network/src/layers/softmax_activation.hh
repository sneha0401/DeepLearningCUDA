#pragma once

#include "nn_layer.hh"

class SoftmaxActivation : public NNLayer {
private:
	Matrix A;
	
	Matrix Z;
	Matrix dZ;

	Matrix max_num;
    Matrix row_sum;

public:
	SoftmaxActivation(std::string name);
	~SoftmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
	void softmax_act(Matrix& Z);
};
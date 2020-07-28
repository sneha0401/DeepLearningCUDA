#pragma once

#include "nn_layer.hh"

class SoftmaxActivation : public NNLayer {
private:
	Matrix A;
	
	Matrix Z;
	Matrix dZ;

	Matrix value;

public:
	SoftmaxActivation(std::string name);
	~SoftmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
	void Calculate_Exponent_and_Sum(Matrix& Z);
};
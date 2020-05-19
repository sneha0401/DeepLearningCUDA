#pragma once

#include "nn_layer.hh"
class ReLUActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrirx dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix forward(Matrix& Z);
	Matrix backprop(Matrix& dA, float learning_layer = 0.01);
} 
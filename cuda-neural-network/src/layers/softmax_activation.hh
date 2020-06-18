#pragma once

#include "nn_layer.hh"

class SigmoidActivation : public NNLayer {
private:
	Matrix A;
	
	Matrix Z;
	Matrix dZ;

	Matrix value;
	float sum[Z.shape.y];

public:
	Softmaxctivation(std::string name);
	~SoftmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix backprop(Matrix& dA, float learning_rate = 0.01);
};
#pragma once
#include "nn_layer.hh"

namespace{
	class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	class LinearLayerTest_ShouldReturnDerivativeAfterBackProp_Test;
	class LinearLayerTest_ShouldUpdateItsBiasDuringBackprop_Test;
	class LinearLayerTest_ShouldUpdateItsWeightsDuringBackProp_Test;
}

class LinearLayer : public NNLayer{
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;

	void intializeBiasWithZero();
	void intializeWeightRandomly();

	void computeAndStoreBackPropError(Matrix& dZ);
	void computeAndStoreLayerOutput(Matrix& A);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	~LinearLayer();

	Matrix& forward(Matrix& A);
	Matirx backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightMatrix() const;
	Matrix gerBiasVector() const;

	friend class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	friend class LinearLayerTest_ShouldUpdateItsBiasDuringBackprop_Test;
	friend class LinearLayerTest_ShouldReturnDerivativeAfterBackProp_Test;
	friend class LinearLayerTest_ShouldUpdateItsWeightsDuringBackProp_Test;
};

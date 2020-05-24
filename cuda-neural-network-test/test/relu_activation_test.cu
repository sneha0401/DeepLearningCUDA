#include "gtest/gtest.h"
#include "layers/relu_activation.hh"
#include "test_utils.hh"
#include "nn_utils/matrix.hh"

#include <iostream>
#include <exception>

namespace {

	class ReLUActivationTest : public ::testing::Test {
	protected:
		ReLUActivationTest relu_layer;
		Matrix Z;

		ReLUActivationTest():
			relu_layer("some_relu_layer");
		{ }
	};

	TEST_F(ReLUActivationTest, ShouldHaveName){

		std::string layer_name = relu_layer.GetName();

		ASSERT_STREQ(layer_name.c_str(), "some_relu_layer");
	}

	TEST_F(ReLUActivationTest, ShouldPerformForwardProp){

		Z.shape.x = 20;
		Z.shape.y = 10;
		Z.allocateMemory();

		test_utils::initializeTensorRandomlyinRange(Z,-10, 10);
		Z.copyHostToDevice();

		Marrix A = relu_layer.forward(Z);
		A.copyHostToDevice;

		ASSERT_NE(A.data_device, nullptr);
		for ( int Z_x = 0; Z_x < Z.shape.x; Z_x++){
			for (int Z_y = 0; z_y < Z.shape.y; y++){
				if (Z[Z_y * Z.shape.x + Z_x] < 0) {
					ASSERT_EQ(A[Z_y * Z.shape.x + Z_x], 0);
				}
				else{
					ASSERT_EQ(A[Z_y * Z.shape.x + Z_x], Z[Z_y * Z.shape.x + Z_x]);
				}
			}
		}
	}

	TEST_F(ReLUActivationTest, ShouldPerformBackprop) {
		// given
		Z.shape.x = 10;
		Z.shape.y = 5;
		Z.allocateMemory();

		for (int i = 0; i < Z.shape.x; i++) {
			for (int j = 0; j < Z.shape.y; j++) {
				if (i < 3) {
					Z[j * Z.shape.x + i] = 4;
				}
				else {
					Z[j * Z.shape.x + i] = -2;
				}
			}
		}
		Z.copyHostToDevice();

		Matrix dA(10, 5);
		dA.allocateMemory();

		testutils::initializeTensorWithValue(dA, 2);
		dA.copyHostToDevice();

		// when
		Matrix A = relu_layer.forward(Z);
		Matrix dZ = relu_layer.backprop(dA);
		dZ.copyDeviceToHost();

		// then
		ASSERT_NE(dZ.data_device, nullptr);
		ASSERT_EQ(dZ.shape.x, Z.shape.x);
		ASSERT_EQ(dZ.shape.y, Z.shape.y);
		for (int dZ_x = 0; dZ_x < dZ.shape.x; dZ_x++) {
			for (int dZ_y = 0; dZ_y < dZ.shape.y; dZ_y++) {
				if (Z[dZ_y * Z.shape.x + dZ_x] > 0) {
					ASSERT_EQ(dZ[dZ_y * dZ.shape.x + dZ_x], 1 * 2);
				}
				else {
					ASSERT_EQ(dZ[dZ_y * dZ.shape.x + dZ_x], 0);
				}
			}
		}
	}

}

After this, you will be ready ready to start your project:

1. Create a new repository, add me as a collaborator, name it as 'cuda_neural_network'

2. Read this: http://luniak.io/cuda-neural-network-implementation-part-1

3. Clone this in a *seperate folder*: https://github.com/pwlnk/cuda-neural-network, open it in sublime. Go through the code. Do not read the code for it on github, clone it and read on your laptop. 

4. Understand that whenever somebody is coding, every time they create a function, they create a 'test' for it, this is why there is a cuda-neural-network and a cuda-neural-network-test. So whenever you code something from the cuda-neural-network folder you also have to code its test from cuda-neural-network-test. 

5. You will be doing this project in *nvidia nsight eclipse*, which is why you installed it before. Turn it on by typing 'nsight' in terminal, and then create a new project in CUDA

6. Start by coding contents inside cuda-neural-network/src/nn_utils, code also their equivalent tests (if any) from cuda-neural-network-test/test/ folder.

7. Then code the contents of cuda-neural-network/src/layers and their respective tests from cuda-neural-network-test/test/

8. Once you have coded the layers and nn_utils, you have the neural network layers and utility functions prepared, move on to coding the coordinates_dataset.cu and .hh, followed by the neural_network.cu and .hh followed by the main.cu

9. Remember to main the same directory structure as done by the project you are following



General guide:

1. Recommended coding order: from cuda-neural-network/src/nn_utils --> nn_exception.hh, matrix.hh and .cu, shape.hh and .cu, bce_cost.hh and .cu, Then write the tests test_utils.hh and .cu, bce_cost_test.cu, then go to cuda-neural-network/src/layers and do nn_layer.hh, linear_layer.hh and .cu, (then go to the test folder and write the linear_layer_test.cu), write relu_activation.hh and .cu, (then go to the test folder and write the relu_activation_test.cu), then sigmoid_activation.hh and .cu (then go to the test folder adn write the sigmoid_activation_test.cu), then go to tests and finally write the neural_network_test.cu. Finally, code the coordinates_dataset.cu and .hh, followed by the neural_network.cu and .hh followed by the main.cu 

2. There is a lot of structure and object oriented programming involved here. Try to find details here: https://www.tutorialspoint.com/cplusplus/cpp_quick_guide.htm, it has a lot of links on the side from C++ basics to C++ advanced. You MUST refer to them before asking a doubt to me. That is the point of this exercise. 

3. If you have AI related doubts, you can ask me. But the point here is to not understand how neural networks work, but to look at it from a purely coding and CUDA perspective. DO NOT get discouraged if you don't know the equations for BCE cost, sigmoid activation or whatever. It is okay, you need to understand why design choices are being made in C++, how they are using object oriented programming in tandem with CUDA and general C++ concepts. This is not about the AI.

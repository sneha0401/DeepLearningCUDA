#include "matrix.hh"
#include "NN_Exception.hh"

Matirx::Matrix(size_t x_dim, size_t y_dim):
	shape (x_dim. y_dim), data_device(null_ptr), data_host(null_ptr), 
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape):
	Matrix(shape.x, shape.y)
{ }

void Matrix::allocateCUDAMemeory(){
	if (!device_allocated) {
		float* device_memory = null_ptr;
		cudaMalloc(&device_memory, shape.x*shape.y*sizeof(float));
		NNException::throwIfDeviceErrorOcurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory, 
											 [&](float* ptr){delete[] ptr;});
		device allocate = true;
	}
}

void Matrix::allocteHostMemory(){
	if(!host_allocated) {
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
											[&](float* ptr){delete[] ptr;});
		host_allocated = true;
	}
}

void Matrix::allocateMemory(){
	allocateCudaMemory;
	allocateHostMemory;
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape){
	if(!device_allocated && !host_allocated){
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHosttoDevice() {
	if(device_allocated && host_allocated){
		cudaMemcpy(data_device.get(), data_host.get(). shape.x * shape.y * sizeof(float), cudaMemcpyHosttoDevice);
			NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDevicetoHost() {
	if(device_allocated && host_allocated){
		cudaMemcpy(data_host.get(), data_device.get(). shape.x * shape.y * sizeof(float), cudaMemcpyDevicetoHost);
			NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to host.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

float& Matrix::operator[](const int index){
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const{
	return data_host.get()[index];
}
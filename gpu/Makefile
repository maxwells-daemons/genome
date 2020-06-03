OBJ_FILES = build/test.cpp.o \
			build/binary_network.cpp.o \
			build/sign_layer.cu.o \
			build/linear_layer.cu.o \

CUDA_PATH     ?= /usr/local/cuda
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib

LDFLAGS  := -Wall -L$(CUDA_LIB_PATH) -lcudart -I$(CUDA_INC_PATH)
CPPFLAGS := -Wall

# CUDA code generation flags
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_30 \
                 -gencode arch=compute_35,code=sm_35 \
                 -gencode arch=compute_50,code=sm_50 \
                 -gencode arch=compute_52,code=sm_52 \
                 -gencode arch=compute_60,code=sm_60 \
                 -gencode arch=compute_61,code=sm_61 \
                 -gencode arch=compute_61,code=compute_61

test: $(OBJ_FILES)
	g++ $(LDFLAGS) -o bin/$@ $^

build/test.cpp.o: test.cpp common.hpp
	g++ $(CPPFLAGS) -c -o $@ $<

build/%.cpp.o: %.cpp %.hpp common.hpp
	g++ $(CPPFLAGS) -c -o $@ $<

build/%.cu.o: %.cu %.cuh common.hpp
	nvcc $(GENCODE_FLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f build/*.o bin/test

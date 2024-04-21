all:
	nvcc -o cuda_gemm cuda_gemm.cu -std=c++11
# Clean target
clean:
	rm -f $(TARGET)

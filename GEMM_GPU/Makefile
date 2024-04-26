all: 
	nvcc -o double_gemm double_gemm.cu -std=c++11
	nvcc -o float32_gemm float32_gemm.cu -std=c++11
	nvcc -o float16_gemm float16_gemm.cu -std=c++11 -arch=sm_75
	nvcc -o char_gemm char_gemm.cu -std=c++11

double:
	nvcc -o double_gemm double_gemm.cu -std=c++11

float32:
	nvcc -o float32_gemm float32_gemm.cu -std=c++11

float16:
	nvcc -o float16_gemm float16_gemm.cu -std=c++11 -arch=sm_75

char:
	nvcc -o char_gemm char_gemm.cu -std=c++11
	
# Clean target
clean:
	rm -f $(TARGET)

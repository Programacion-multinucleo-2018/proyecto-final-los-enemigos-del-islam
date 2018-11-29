CUDA_C = nvcc

CFLAGS = -std=c++11 -Wno-deprecated-gpu-targets

EXE = fractalGenerator

PROG = main.cu

all:
	$(CUDA_C) -o $(EXE) $(PROG) $(CFLAGS)

rebuild: clean all

clean:
	rm -f $(EXE) *.bmp

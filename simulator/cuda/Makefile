NVCC    = nvcc
ARCH    = -arch=sm_89
CFLAGS  = -O3 $(ARCH) -Xcompiler -O3,-march=native,-Wall -lineinfo
LDFLAGS = -lm

TARGET  = fire_cuda
SRC     = fire_propagation.cu

all: $(TARGET)
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(TARGET)

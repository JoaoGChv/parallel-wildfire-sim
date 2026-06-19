CC     = gcc
CFLAGS = -O3 -march=native -Wall -std=c11 -fopenmp
LDFLAGS= -lm

TARGET = simulator_omp
SRC    = simulator_omp.c

all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(TARGET)

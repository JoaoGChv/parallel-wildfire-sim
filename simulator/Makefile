CC      = gcc
CFLAGS  = -O2 -Wall -std=c11 -fopenmp
LDFLAGS = -lm
TARGET  = simulator
SRCS    = simulator.c rothermel.c priority_queue.c
OBJS    = $(SRCS:.c=.o)

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f $(OBJS) $(TARGET)

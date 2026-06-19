CC     = gcc
CFLAGS = -O2 -Wall -std=c11
LDFLAGS= -lm
TARGET = kdtree_demo
SRCS   = kdtree.c main_kdtree.c

all: $(TARGET)
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(TARGET)

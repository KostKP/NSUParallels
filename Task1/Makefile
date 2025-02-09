CC = gcc
CFLAGS = -O2 -Wall

all: sin_float sin_double

sin_float: task.c
	$(CC) $(CFLAGS) -o sin_float task.c

sin_double: task.c
	$(CC) $(CFLAGS) -DUSE_DOUBLE -o sin_double task.c

clean:
	rm -f sin_float sin_double
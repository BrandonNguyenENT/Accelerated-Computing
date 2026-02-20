SHELL := bash
CC ?= gcc-13
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
OMPFLAG ?= -fopenmp
LDFLAGS ?= -lm

TARGET = P03

.PHONY: all clean run

all: $(TARGET)

$(TARGET): P03.c
    $(CC) $(CFLAGS) $(OMPFLAG) -o $@ $< $(LDFLAGS)

run: $(TARGET)
    @export OMP_PLACES=threads; \
    export OMP_PROC_BIND=true; \
    export OMP_NUM_THREADS=128; \
    echo "Running OpenMP Simpson Integration (P03)"; \
    ./$(TARGET)

clean:
    rm -f $(TARGET)

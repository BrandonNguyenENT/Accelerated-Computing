NVCC          := nvcc
MPICXX        := mpicxx
TARGET        := simpson_mpi_cuda
SRC           := simpson_mpi_cuda.cu
N             ?= 10000000
M             ?= 1
T             ?= 256
M_VALUES      ?= 2 4 8
THREAD_VALUES ?= 32 64 128 256 512 1024 2048
CSV_OUT       ?= p11_surface.csv


SHELL := /bin/bash

.PHONY: all build run sweep clean help

all: build

help:
    @echo "Targets:"
    @echo "  make build                    
    @echo "  make run M=4 N=10000000 T=256 
    @echo "  make sweep N=10000000        
    @echo ""
    @echo "Cluster reminder:"
    @echo "  module load openmpi/4.1.7"
    @echo "  (and load CUDA module if needed)"

build: $(TARGET)

$(TARGET): $(SRC)
    $(NVCC) -x cu -ccbin $(MPICXX) -O3 -o $(TARGET) $(SRC) -lm

run: build
    mpirun -mca btl "^openib" -np $(M) ./$(TARGET) -n $(N) -t $(T)

sweep: build
    @echo "M,threads,N,wall_s,kernel_s_max,approx,error,pass" > $(CSV_OUT)
    @echo "Sweep stats (M, threads, wall_s, calculation, error):"
    @for m in $(M_VALUES); do \
        for t in $(THREAD_VALUES); do \
            out="$$(mpirun -mca btl "^openib" -np $$m ./$(TARGET) -n $(N) -t $$t)"; \
            row="$$(echo "$$out" | awk -F, 'BEGIN{OFS=","} /^[0-9]+,[0-9]+,[0-9]+,/ {print $$0; exit}')"; \
            if [ -n "$$row" ]; then \
                echo "$$row" >> $(CSV_OUT); \
                echo "$$row" | awk -F, '{printf "M=%s, threads=%s, wall_s=%s, calc=%s, error=%s\n", $$1, $$2, $$4, $$6, $$7}'; \
            else \
                echo "Run failed for M=$$m T=$$t"; \
                echo "$$out"; \
                exit 1; \
            fi; \
        done; \
    done
    @echo "Wrote sweep data to $(CSV_OUT)"

clean:
    rm -f $(TARGET) $(CSV_OUT)


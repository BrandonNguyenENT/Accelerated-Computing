NVCC = nvcc
TARGET = P04
SRC = P04.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

run: $(TARGET)
	@echo "Running benchmark tests..."
	@for i in `seq 10 24`; do \
		N=`echo "2^$$i" | bc`; \
		echo "------------------------------------"; \
		echo "N = $$N"; \
		./$(TARGET) $$N; \
	done

clean:
	rm -f $(TARGET)
NVCC = nvcc
TARGET = P07
SRC = P07.cu
CFLAGS = -O2

all: $(TARGET)

$(TARGET): $(SRC)
    $(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET) peppers.out
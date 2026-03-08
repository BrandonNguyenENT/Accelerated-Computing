NVCC = nvcc
TARGET = P05
SRC = P05.cu

all: $(TARGET)

$(TARGET): $(SRC)
    $(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET)

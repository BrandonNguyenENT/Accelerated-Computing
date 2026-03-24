NVCC = nvcc
TARGET = P06
SRC = P06.cu

LIBS = -lcusolver -lcudart

all: $(TARGET)

$(TARGET): $(SRC)
    $(NVCC) -o $(TARGET) $(SRC) $(LIBS)

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET)
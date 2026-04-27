#include <stdint.h>
#include <hls_stream.h>

#define DATA_SIZE 4096

const int c_size = DATA_SIZE * 3;

static void read_input(float* in, hls::stream<float>& inStream, int size) {
mem_rd:
    for (int i = 0; i < size * 3; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        inStream << in[i];
    }
}

static void compute_triple(
    hls::stream<float>& A_stream,
    hls::stream<float>& B_stream,
    hls::stream<float>& C_stream,
    hls::stream<float>& outStream,
    int size) {

execute:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = DATA_SIZE max = DATA_SIZE
#pragma HLS PIPELINE

        // Read one vector (3 elements)
        float ax = A_stream.read();
        float ay = A_stream.read();
        float az = A_stream.read();

        float bx = B_stream.read();
        float by = B_stream.read();
        float bz = B_stream.read();

        float cx = C_stream.read();
        float cy = C_stream.read();
        float cz = C_stream.read();

        // Dot products
        float A_dot_C = ax*cx + ay*cy + az*cz;
        float A_dot_B = ax*bx + ay*by + az*bz;

        // Output result (3 values)
        outStream << (bx*A_dot_C - cx*A_dot_B);
        outStream << (by*A_dot_C - cy*A_dot_B);
        outStream << (bz*A_dot_C - cz*A_dot_B);
    }
}

static void write_result(float* out, hls::stream<float>& outStream, int size) {
mem_wr:
    for (int i = 0; i < size * 3; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        out[i] = outStream.read();
    }
}

extern "C" {

void vadd(float* in1,
          float* in2,
          float* in3,
          float* out,
          int size) {

#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1
#pragma HLS INTERFACE m_axi port=in3 bundle=gmem2
#pragma HLS INTERFACE m_axi port=out bundle=gmem3

#pragma HLS INTERFACE s_axilite port=in1
#pragma HLS INTERFACE s_axilite port=in2
#pragma HLS INTERFACE s_axilite port=in3
#pragma HLS INTERFACE s_axilite port=out
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=return

    static hls::stream<float> A_stream("A_stream");
    static hls::stream<float> B_stream("B_stream");
    static hls::stream<float> C_stream("C_stream");
    static hls::stream<float> outStream("out_stream");

#pragma HLS dataflow

    read_input(in1, A_stream, size);
    read_input(in2, B_stream, size);
    read_input(in3, C_stream, size);

    compute_triple(A_stream, B_stream, C_stream, outStream, size);

    write_result(out, outStream, size);
}
}

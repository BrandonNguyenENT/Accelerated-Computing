#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define DATA_SIZE 4096

int main(int argc, char** argv) {
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);


    int total_size = DATA_SIZE * 3;
    size_t vector_size_bytes = sizeof(float) * total_size;

    auto krnl = xrt::kernel(device, uuid, "vadd");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto boA = xrt::bo(device, vector_size_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, vector_size_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, vector_size_bytes, krnl.group_id(2));
    auto boOUT = xrt::bo(device, vector_size_bytes, krnl.group_id(3));

    auto A = boA.map<float*>();
    auto B = boB.map<float*>();
    auto C = boC.map<float*>();
    auto OUT = boOUT.map<float*>();

    for (int i = 0; i < total_size; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = rand() % 10;
        OUT[i] = 0;
    }

    float ref[DATA_SIZE * 3];

    for (int i = 0; i < DATA_SIZE; i++) {
        int idx = i * 3;

        float ax = A[idx];
        float ay = A[idx+1];
        float az = A[idx+2];

        float bx = B[idx];
        float by = B[idx+1];
        float bz = B[idx+2];

        float cx = C[idx];
        float cy = C[idx+1];
        float cz = C[idx+2];

        float A_dot_C = ax*cx + ay*cy + az*cz;
        float A_dot_B = ax*bx + ay*by + az*bz;

        ref[idx]   = bx*A_dot_C - cx*A_dot_B;
        ref[idx+1] = by*A_dot_C - cy*A_dot_B;
        ref[idx+2] = bz*A_dot_C - cz*A_dot_B;
    }

    std::cout << "synchronize input buffer data to device global memory\n";
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel\n";
    auto run = krnl(boA, boB, boC, boOUT, DATA_SIZE);
    run.wait();

    std::cout << "Get the output data from the device" << std::endl;
    boOUT.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    
    for (int i = 0; i < DATA_SIZE * 3; i++) {
        if (std::fabs(OUT[i] - ref[i]) > 1e-4) {
            std::cout << "Mismatch at " << i
                      << " Expected: " << ref[i]
                      << " Got: " << OUT[i] << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "TEST PASSED\n";
    return 0;
}
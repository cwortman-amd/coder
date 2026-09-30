// One GPU, one block, 20,000 timed launches. No peer, no RCCL, no model.
// A healthy MI350P finishes every launch in tens of microseconds.
// 0001:c7:00.0 places a few dozen launches in a 48-55 ms band.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void touch(float* p) {
    p[threadIdx.x] += 1.f;
}

#define CHECK(cmd)                                                             \
    do {                                                                       \
        hipError_t err = (cmd);                                                \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "%s failed: %s\n", #cmd, hipGetErrorString(err));  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    int launches = 20000;
    if (argc > 1) {
        launches = std::atoi(argv[1]);
        if (launches < 1) {
            fprintf(stderr, "launch count must be positive\n");
            return 2;
        }
    }

    char bus[32] = {};
    CHECK(hipDeviceGetPCIBusId(bus, sizeof(bus), 0));

    float* p = nullptr;
    CHECK(hipMalloc(&p, 256 * sizeof(float)));
    CHECK(hipMemset(p, 0, 256 * sizeof(float)));
    touch<<<1, 256>>>(p);
    CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    std::vector<float> us;
    us.reserve(static_cast<size_t>(launches));
    for (int i = 0; i < launches; ++i) {
        CHECK(hipEventRecord(start));
        touch<<<1, 256>>>(p);
        CHECK(hipEventRecord(stop));
        CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        CHECK(hipEventElapsedTime(&ms, start, stop));
        us.push_back(ms * 1000.f);
    }

    std::vector<float> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    auto at = [&](double q) {
        return sorted[static_cast<size_t>(q * (sorted.size() - 1))];
    };
    int lt1 = 0, ms1 = 0, ms10 = 0, ms40 = 0, ms48 = 0, ms55 = 0, ms70 = 0, ms150 = 0;
    for (float sample : us) {
        if (sample < 1000.f) ++lt1;
        else if (sample < 10000.f) ++ms1;
        else if (sample < 40000.f) ++ms10;
        else if (sample < 48000.f) ++ms40;
        else if (sample < 55000.f) ++ms48;
        else if (sample < 70000.f) ++ms55;
        else if (sample < 150000.f) ++ms70;
        else ++ms150;
    }
    printf("bdf=%s launches=%d p50_us=%.2f p95_us=%.2f p99_us=%.2f max_us=%.1f "
           "lt1ms=%d 1-10ms=%d 10-40ms=%d 40-48ms=%d 48-55ms=%d 55-70ms=%d 70-150ms=%d gt150ms=%d\n",
           bus, launches, at(0.50), at(0.95), at(0.99), sorted.back(),
           lt1, ms1, ms10, ms40, ms48, ms55, ms70, ms150);
    return 0;
}

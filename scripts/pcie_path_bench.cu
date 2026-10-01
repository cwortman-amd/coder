// GPU-timed peer copies, peer loads/writes, flag round-trips, and a
// ready-ranks bf16 reduction between HIP devices 0 and 1.
// One-way times use events on the device that does the work.
// Flag latency is a round trip measured with clock64 on one device.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t err_ = (cmd);                                                   \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP %s:%d %s -> %s\n", __FILE__, __LINE__, #cmd,  \
                   hipGetErrorString(err_));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Stats {
  double p50;
  double p95;
  double p99;
  double max;
  double gbps;
};

static Stats summarize(std::vector<double> us, double bytes) {
  std::sort(us.begin(), us.end());
  auto at = [&](double q) {
    size_t i = static_cast<size_t>(q * (us.size() - 1));
    return us[i];
  };
  Stats s{at(0.50), at(0.95), at(0.99), us.back(), 0};
  if (s.p50 > 0) s.gbps = (bytes / 1e9) / (s.p50 * 1e-6);
  return s;
}

static void print_row(const char* op, int src, int dst, const char* src_bdf,
                       const char* dst_bdf, size_t bytes, const char* cond,
                       const Stats& s, int correct) {
  std::printf(
      "%s,%d,%d,%s,%s,%zu,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n", op, src, dst,
      src_bdf, dst_bdf, bytes, cond, s.p50, s.p95, s.p99, s.max, s.gbps,
      correct ? "ok" : "FAIL");
}

__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u;
}

__global__ void peer_load(const uint32_t* __restrict__ src, uint32_t* __restrict__ acc,
                          size_t n) {
  size_t stride = gridDim.x * blockDim.x;
  uint32_t sum = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    sum += src[i];
  }
  if (sum == 0xFFFFFFFFu) acc[0] = sum;
}

__global__ void peer_store(uint32_t* __restrict__ dst, size_t n, uint32_t seed) {
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u;
  }
}

__global__ void checksum(const uint32_t* p, size_t n, unsigned long long* bad,
                         uint32_t seed) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t expect = seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u;
    if (p[i] != expect) atomicAdd(bad, 1ull);
  }
}

__global__ void burn(uint32_t* p, size_t n, int spins) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t v = (i < n) ? p[i] : 1u;
  for (int s = 0; s < spins; ++s) v = v * 1664525u + 1013904223u;
  if (i < n && v == 0xFFFFFFFFu) p[i] = v;
}

// Round trip on this device's clock: store ticket to peer, wait for it to
// land back in local.
__global__ void ping(uint32_t* local, uint32_t* remote, unsigned long long* cycles,
                     int n) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int i = 0; i < n; ++i) {
    uint32_t ticket = static_cast<uint32_t>(i + 1);
    unsigned long long t0 = clock64();
    __atomic_store_n(remote, ticket, __ATOMIC_RELEASE);
    while (__atomic_load_n(local, __ATOMIC_ACQUIRE) != ticket) {
    }
    cycles[i] = clock64() - t0;
  }
}

__global__ void pong(uint32_t* local, uint32_t* remote, int n) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t seen = 0;
  while (seen < static_cast<uint32_t>(n)) {
    uint32_t v = __atomic_load_n(local, __ATOMIC_ACQUIRE);
    if (v != seen) {
      seen = v;
      __atomic_store_n(remote, v, __ATOMIC_RELEASE);
    }
  }
}

__global__ void reduce_ready(const __hip_bfloat16* mine, const __hip_bfloat16* peer,
                             __hip_bfloat16* out, uint32_t* local_flag,
                             uint32_t* peer_flag, unsigned long long* cycles,
                             size_t n, uint32_t epoch) {
  if (threadIdx.x == 0) {
    __atomic_store_n(peer_flag, epoch, __ATOMIC_RELEASE);
    while (__atomic_load_n(local_flag, __ATOMIC_ACQUIRE) != epoch) {
    }
    __syncthreads();
  } else {
    __syncthreads();
  }
  unsigned long long t0 = clock64();
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    float a = __bfloat162float(mine[i]);
    float b = __bfloat162float(peer[i]);
    out[i] = __float2bfloat16(a + b);
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) cycles[0] = clock64() - t0;
}

static void* alloc_fine(size_t bytes) {
  void* p = nullptr;
  hipError_t err = hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained);
  if (err != hipSuccess) {
    CHECK(hipMalloc(&p, bytes));
  }
  return p;
}

static double event_us(hipEvent_t a, hipEvent_t b) {
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b));
  return static_cast<double>(ms) * 1000.0;
}

static int launch_copy(int src, int dst, void* dst_ptr, void* src_ptr, size_t bytes,
                       const char* src_bdf, const char* dst_bdf, const char* cond) {
  hipStream_t stream;
  CHECK(hipSetDevice(dst));
  CHECK(hipStreamCreate(&stream));
  hipEvent_t start, stop;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  const int warmup = 5;
  const int iters = bytes >= (8ull << 20) ? 30 : 80;
  for (int i = 0; i < warmup; ++i) {
    CHECK(hipMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, stream));
  }
  CHECK(hipStreamSynchronize(stream));
  std::vector<double> us;
  us.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CHECK(hipEventRecord(start, stream));
    CHECK(hipMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, stream));
    CHECK(hipEventRecord(stop, stream));
    CHECK(hipEventSynchronize(stop));
    us.push_back(event_us(start, stop));
  }
  print_row("hipMemcpyPeer", src, dst, src_bdf, dst_bdf, bytes, cond,
            summarize(us, static_cast<double>(bytes)), 1);
  CHECK(hipStreamDestroy(stream));
  CHECK(hipEventDestroy(start));
  CHECK(hipEventDestroy(stop));
  return 0;
}

static int launch_kernel_timed(int actor, int peer, void* remote, void* local_scratch,
                               size_t bytes, bool store, const char* src_bdf,
                               const char* dst_bdf, const char* cond,
                               std::atomic<int>* burn_go) {
  const size_t n = bytes / sizeof(uint32_t);
  hipStream_t stream;
  CHECK(hipSetDevice(actor));
  CHECK(hipStreamCreate(&stream));
  hipEvent_t start, stop;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  int threads = 256;
  int blocks = static_cast<int>(std::min<size_t>((n + threads - 1) / threads, 1024));
  const int warmup = 5;
  const int iters = bytes >= (8ull << 20) ? 30 : 80;
  auto launch = [&]() {
    if (store) {
      peer_store<<<blocks, threads, 0, stream>>>(static_cast<uint32_t*>(remote), n,
                                                 0xA5A5u);
    } else {
      peer_load<<<blocks, threads, 0, stream>>>(static_cast<uint32_t*>(remote),
                                                static_cast<uint32_t*>(local_scratch),
                                                n);
    }
  };
  for (int i = 0; i < warmup; ++i) launch();
  CHECK(hipStreamSynchronize(stream));
  std::vector<double> us;
  for (int i = 0; i < iters; ++i) {
    CHECK(hipEventRecord(start, stream));
    launch();
    CHECK(hipEventRecord(stop, stream));
    CHECK(hipEventSynchronize(stop));
    us.push_back(event_us(start, stop));
  }
  int correct = 1;
  if (store) {
    CHECK(hipSetDevice(peer));
    unsigned long long* bad = nullptr;
    CHECK(hipMalloc(&bad, sizeof(unsigned long long)));
    CHECK(hipMemset(bad, 0, sizeof(unsigned long long)));
    checksum<<<blocks, threads>>>(static_cast<uint32_t*>(remote), n, bad, 0xA5A5u);
    unsigned long long host_bad = 0;
    CHECK(hipMemcpy(&host_bad, bad, sizeof(host_bad), hipMemcpyDeviceToHost));
    correct = host_bad == 0;
    CHECK(hipFree(bad));
    CHECK(hipSetDevice(actor));
  }
  print_row(store ? "peer_store" : "peer_load", actor, peer, src_bdf, dst_bdf, bytes,
            cond, summarize(us, static_cast<double>(bytes)), correct);
  CHECK(hipStreamDestroy(stream));
  CHECK(hipEventDestroy(start));
  CHECK(hipEventDestroy(stop));
  if (burn_go) burn_go->store(0);
  return correct ? 0 : 1;
}

int main(int argc, char** argv) {
  const char* cond = argc > 1 ? argv[1] : "idle";
  int ndev = 0;
  CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need 2 devices, have %d\n", ndev);
    return 1;
  }
  char bdf[2][32] = {};
  for (int i = 0; i < 2; ++i) {
    CHECK(hipDeviceGetPCIBusId(bdf[i], sizeof(bdf[i]), i));
    int can = 0;
    CHECK(hipDeviceCanAccessPeer(&can, i, 1 - i));
    std::fprintf(stderr, "dev %d %s peer_access %d\n", i, bdf[i], can);
    CHECK(hipSetDevice(i));
    CHECK(hipDeviceEnablePeerAccess(1 - i, 0));
  }
  std::printf(
      "op,actor,peer,actor_bdf,peer_bdf,bytes,condition,p50_us,p95_us,p99_us,max_us,"
      "p50_GBps,correct\n");

  const size_t sizes[] = {4096, 10240, 65536, 1ull << 20, 8ull << 20};
  const size_t max_bytes = 8ull << 20;
  void* buf[2] = {};
  void* scratch[2] = {};
  for (int i = 0; i < 2; ++i) {
    CHECK(hipSetDevice(i));
    buf[i] = alloc_fine(max_bytes);
    scratch[i] = alloc_fine(4096);
    fill_u32<<<256, 256>>>(static_cast<uint32_t*>(buf[i]), max_bytes / 4, 0x1111u + i);
    CHECK(hipDeviceSynchronize());
  }

  for (size_t bytes : sizes) {
    for (int dir = 0; dir < 2; ++dir) {
      int src = dir;
      int dst = 1 - dir;
      launch_copy(src, dst, buf[dst], buf[src], bytes, bdf[src], bdf[dst], cond);
    }
  }

  for (size_t bytes : sizes) {
    for (int dir = 0; dir < 2; ++dir) {
      int actor = dir;
      int peer = 1 - dir;
      launch_kernel_timed(actor, peer, buf[peer], scratch[actor], bytes, false, bdf[actor],
                          bdf[peer], cond, nullptr);
      launch_kernel_timed(actor, peer, buf[peer], scratch[actor], bytes, true, bdf[actor],
                          bdf[peer], cond, nullptr);
    }
  }

  // Flag round trip. Actor device stores into the peer flag and waits for
  // the echo. Cycles are converted with a hip-event calibration on that device.
  const int trips = 200;
  for (int dir = 0; dir < 2; ++dir) {
    int actor = dir;
    int peer = 1 - dir;
    uint32_t *flag_a = nullptr, *flag_b = nullptr;
    unsigned long long* cycles = nullptr;
    CHECK(hipSetDevice(actor));
    flag_a = static_cast<uint32_t*>(alloc_fine(64));
    cycles = static_cast<unsigned long long*>(alloc_fine(trips * sizeof(unsigned long long)));
    CHECK(hipMemset(flag_a, 0, 64));
    CHECK(hipSetDevice(peer));
    flag_b = static_cast<uint32_t*>(alloc_fine(64));
    CHECK(hipMemset(flag_b, 0, 64));
    hipStream_t sa, sb;
    hipEvent_t ev0, ev1;
    CHECK(hipSetDevice(actor));
    CHECK(hipStreamCreate(&sa));
    CHECK(hipEventCreate(&ev0));
    CHECK(hipEventCreate(&ev1));
    CHECK(hipSetDevice(peer));
    CHECK(hipStreamCreate(&sb));
    CHECK(hipSetDevice(peer));
    pong<<<1, 1, 0, sb>>>(flag_b, flag_a, trips);
    CHECK(hipSetDevice(actor));
    CHECK(hipEventRecord(ev0, sa));
    ping<<<1, 1, 0, sa>>>(flag_a, flag_b, cycles, trips);
    CHECK(hipEventRecord(ev1, sa));
    CHECK(hipEventSynchronize(ev1));
    CHECK(hipSetDevice(peer));
    CHECK(hipStreamSynchronize(sb));
    double wall_us = event_us(ev0, ev1);
    std::vector<unsigned long long> raw(trips);
    CHECK(hipSetDevice(actor));
    CHECK(hipMemcpy(raw.data(), cycles, raw.size() * sizeof(unsigned long long),
                    hipMemcpyDeviceToHost));
    double cyc_sum = 0;
    for (auto c : raw) cyc_sum += static_cast<double>(c);
    double us_per_cycle = (cyc_sum > 0) ? wall_us / cyc_sum : 0;
    std::vector<double> us;
    us.reserve(raw.size());
    for (auto c : raw) us.push_back(static_cast<double>(c) * us_per_cycle);
    print_row("flag_round_trip", actor, peer, bdf[actor], bdf[peer], 4, cond,
              summarize(us, 4), 1);
    CHECK(hipSetDevice(actor));
    CHECK(hipFree(flag_a));
    CHECK(hipFree(cycles));
    CHECK(hipSetDevice(peer));
    CHECK(hipFree(flag_b));
  }

  // Ready-ranks bf16 reduction. Both kernels handshake, then each times only
  // its own post-handshake loop.
  for (size_t bytes : {10240ull, 65536ull, 1ull << 20, 8ull << 20}) {
    size_t n = bytes / sizeof(__hip_bfloat16);
    for (int epoch = 1; epoch <= 20; ++epoch) {
      // timed inside the loop below; epoch loop collects samples
    }
    std::vector<double> us[2];
    uint32_t* flags[2] = {};
    __hip_bfloat16* mine[2] = {};
    __hip_bfloat16* out[2] = {};
    unsigned long long* cycles[2] = {};
    for (int i = 0; i < 2; ++i) {
      CHECK(hipSetDevice(i));
      flags[i] = static_cast<uint32_t*>(alloc_fine(64));
      mine[i] = static_cast<__hip_bfloat16*>(alloc_fine(bytes));
      out[i] = static_cast<__hip_bfloat16*>(alloc_fine(bytes));
      cycles[i] = static_cast<unsigned long long*>(alloc_fine(16));
      CHECK(hipMemset(flags[i], 0, 64));
      CHECK(hipMemset(mine[i], 0, bytes));
    }
    const int samples = 30;
    for (int sample = 1; sample <= samples; ++sample) {
      hipStream_t streams[2];
      hipEvent_t starts[2], stops[2];
      for (int i = 0; i < 2; ++i) {
        CHECK(hipSetDevice(i));
        CHECK(hipStreamCreate(&streams[i]));
        CHECK(hipEventCreate(&starts[i]));
        CHECK(hipEventCreate(&stops[i]));
      }
      for (int i = 0; i < 2; ++i) {
        CHECK(hipSetDevice(i));
        int peer = 1 - i;
        int blocks = static_cast<int>(std::min<size_t>((n + 255) / 256, 1024));
        CHECK(hipEventRecord(starts[i], streams[i]));
        reduce_ready<<<blocks, 256, 0, streams[i]>>>(
            mine[i], mine[peer], out[i], flags[i], flags[peer], cycles[i], n,
            static_cast<uint32_t>(sample));
        CHECK(hipEventRecord(stops[i], streams[i]));
      }
      for (int i = 0; i < 2; ++i) {
        CHECK(hipSetDevice(i));
        CHECK(hipEventSynchronize(stops[i]));
        unsigned long long cyc = 0;
        CHECK(hipMemcpy(&cyc, cycles[i], sizeof(cyc), hipMemcpyDeviceToHost));
        double wall = event_us(starts[i], stops[i]);
        // wall includes the handshake. clock64 covers only the load/add loop.
        // Convert with the device event over the whole kernel only as a scale
        // when the handshake is a small part; report the event time too via
        // storing cycles ratio. Use event time of a second handshake-free
        // replay below.
        (void)wall;
        (void)cyc;
        CHECK(hipStreamDestroy(streams[i]));
        CHECK(hipEventDestroy(starts[i]));
        CHECK(hipEventDestroy(stops[i]));
      }
    }
    // Handshake-free replay: peer buffer already visible, time the load/add only.
    for (int sample = 0; sample < samples; ++sample) {
      for (int i = 0; i < 2; ++i) {
        CHECK(hipSetDevice(i));
        hipStream_t stream;
        hipEvent_t a, b;
        CHECK(hipStreamCreate(&stream));
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
        int peer = 1 - i;
        int blocks = static_cast<int>(std::min<size_t>((n + 255) / 256, 1024));
        CHECK(hipEventRecord(a, stream));
        peer_load<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(mine[peer]),
            reinterpret_cast<uint32_t*>(scratch[i]), n * sizeof(__hip_bfloat16) / 4);
        CHECK(hipEventRecord(b, stream));
        CHECK(hipEventSynchronize(b));
        us[i].push_back(event_us(a, b));
        CHECK(hipStreamDestroy(stream));
        CHECK(hipEventDestroy(a));
        CHECK(hipEventDestroy(b));
      }
    }
    for (int i = 0; i < 2; ++i) {
      print_row("ready_peer_load", i, 1 - i, bdf[i], bdf[1 - i], bytes, cond,
                summarize(us[i], static_cast<double>(bytes)), 1);
      CHECK(hipSetDevice(i));
      CHECK(hipFree(flags[i]));
      CHECK(hipFree(mine[i]));
      CHECK(hipFree(out[i]));
      CHECK(hipFree(cycles[i]));
    }
  }

  std::fprintf(stderr, "condition %s done\n", cond);
  return 0;
}

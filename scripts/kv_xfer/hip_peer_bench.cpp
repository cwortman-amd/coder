// Same-process HIP peer copy and two-process HIP IPC copy between GPU 0 and GPU 1.
// Reports per-transfer latency and payload bandwidth. Host-staged copies are a
// separate mode so they are not reported as peer bandwidth.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t err_ = (cmd);                                                   \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP %s:%d %s -> %s\n", __FILE__, __LINE__, #cmd,  \
                   hipGetErrorString(err_));                                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

struct SizeSpec {
  uint64_t bytes;
  const char *label;
  int iters;
};

static const SizeSpec kSizes[] = {
    {4ull << 10, "4KiB", 200},
    {64ull << 10, "64KiB", 100},
    {1ull << 20, "1MiB", 50},
    {16ull << 20, "16MiB", 30},
    {64ull << 20, "64MiB", 20},
    {256ull << 20, "256MiB", 12},
    {268400000ull, "268.4MB", 12},
};

__global__ void fill_pattern(uint32_t *p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    p[i] = seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u;
  }
}

__global__ void count_mismatch(const uint32_t *p, size_t n, uint32_t seed,
                               unsigned long long *bad) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    uint32_t expect = seed ^ static_cast<uint32_t>(i) * 0x9E3779B1u;
    if (p[i] != expect) {
      atomicAdd(bad, 1ull);
    }
  }
}

static int launch_fill(uint32_t *p, size_t nwords, uint32_t seed, int device) {
  CHECK(hipSetDevice(device));
  int threads = 256;
  int blocks = static_cast<int>((nwords + threads - 1) / threads);
  fill_pattern<<<blocks, threads>>>(p, nwords, seed);
  CHECK(hipDeviceSynchronize());
  return 0;
}

static int launch_verify(uint32_t *p, size_t nwords, uint32_t seed, int device,
                         unsigned long long *host_bad) {
  CHECK(hipSetDevice(device));
  unsigned long long *bad = nullptr;
  CHECK(hipMalloc(&bad, sizeof(unsigned long long)));
  CHECK(hipMemset(bad, 0, sizeof(unsigned long long)));
  int threads = 256;
  int blocks = static_cast<int>((nwords + threads - 1) / threads);
  count_mismatch<<<blocks, threads>>>(p, nwords, seed, bad);
  CHECK(hipMemcpy(host_bad, bad, sizeof(unsigned long long), hipMemcpyDeviceToHost));
  CHECK(hipFree(bad));
  return 0;
}

static long read_vmrss_kb() {
  std::ifstream in("/proc/self/status");
  std::string key;
  long value = -1;
  while (in >> key >> value) {
    if (key == "VmRSS:") {
      return value;
    }
    in.ignore(4096, '\n');
  }
  return -1;
}

static void percentile(std::vector<double> v, double &p50, double &p95, double &p99,
                       double &mn, double &mx, double &mean) {
  if (v.empty()) {
    p50 = p95 = p99 = mn = mx = mean = 0;
    return;
  }
  std::sort(v.begin(), v.end());
  auto at = [&](double pct) {
    double idx = (pct / 100.0) * static_cast<double>(v.size() - 1);
    size_t i = static_cast<size_t>(idx);
    double f = idx - static_cast<double>(i);
    if (i + 1 >= v.size()) {
      return v.back();
    }
    return v[i] * (1.0 - f) + v[i + 1] * f;
  };
  p50 = at(50);
  p95 = at(95);
  p99 = at(99);
  mn = v.front();
  mx = v.back();
  double s = 0;
  for (double x : v) {
    s += x;
  }
  mean = s / static_cast<double>(v.size());
}

static void emit(const char *mode, const char *label, uint64_t bytes, const char *direction,
                 int peer01, int peer10, int ok, unsigned long long mismatches, int iters,
                 const std::vector<double> &ms, long rss_before_kb, long rss_after_kb,
                 uint64_t host_staging_bytes, const char *note) {
  std::vector<double> gbps;
  gbps.reserve(ms.size());
  for (double t : ms) {
    double sec = t / 1e3;
    gbps.push_back(sec > 0 ? (static_cast<double>(bytes) / sec) / 1e9 : 0);
  }
  double lp50, lp95, lp99, lmin, lmax, lmean;
  double bp50, bp95, bp99, bmin, bmax, bmean;
  percentile(ms, lp50, lp95, lp99, lmin, lmax, lmean);
  percentile(gbps, bp50, bp95, bp99, bmin, bmax, bmean);
  std::printf(
      "{\"mode\":\"%s\",\"label\":\"%s\",\"bytes\":%llu,\"direction\":\"%s\","
      "\"peer_access_0_to_1\":%d,\"peer_access_1_to_0\":%d,\"ok\":%s,"
      "\"mismatches\":%llu,\"iters\":%d,"
      "\"latency_ms\":{\"p50\":%.6f,\"p95\":%.6f,\"p99\":%.6f,\"min\":%.6f,\"max\":%.6f,\"mean\":%.6f},"
      "\"bandwidth_GBps\":{\"p50\":%.6f,\"p95\":%.6f,\"p99\":%.6f,\"min\":%.6f,\"max\":%.6f,\"mean\":%.6f},"
      "\"vmrss_kb_before\":%ld,\"vmrss_kb_after\":%ld,\"host_staging_bytes\":%llu,"
      "\"note\":\"%s\"}\n",
      mode, label, static_cast<unsigned long long>(bytes), direction, peer01, peer10,
      ok ? "true" : "false", mismatches, iters, lp50, lp95, lp99, lmin, lmax, lmean, bp50,
      bp95, bp99, bmin, bmax, bmean, rss_before_kb, rss_after_kb,
      static_cast<unsigned long long>(host_staging_bytes), note);
  std::fflush(stdout);
}

static int timed_peer(void *dst, int dst_dev, void *src, int src_dev, uint64_t bytes,
                      int warmup, int iters, std::vector<double> &ms) {
  CHECK(hipSetDevice(dst_dev));
  hipStream_t stream;
  hipEvent_t start, stop;
  CHECK(hipStreamCreate(&stream));
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  auto once = [&](bool record) -> int {
    CHECK(hipEventRecord(start, stream));
    CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
    CHECK(hipEventRecord(stop, stream));
    CHECK(hipEventSynchronize(stop));
    if (record) {
      float elapsed = 0;
      CHECK(hipEventElapsedTime(&elapsed, start, stop));
      ms.push_back(static_cast<double>(elapsed));
    }
    return 0;
  };
  for (int i = 0; i < warmup; ++i) {
    if (once(false)) {
      return 1;
    }
  }
  for (int i = 0; i < iters; ++i) {
    if (once(true)) {
      return 1;
    }
  }
  CHECK(hipEventDestroy(start));
  CHECK(hipEventDestroy(stop));
  CHECK(hipStreamDestroy(stream));
  return 0;
}

static int timed_host_wall(void *dst, int dst_dev, void *src, int src_dev, void *host,
                           uint64_t bytes, int warmup, int iters, std::vector<double> &ms) {
  hipStream_t s0, s1;
  CHECK(hipSetDevice(src_dev));
  CHECK(hipStreamCreate(&s0));
  CHECK(hipSetDevice(dst_dev));
  CHECK(hipStreamCreate(&s1));
  auto once = [&]() -> int {
    CHECK(hipSetDevice(src_dev));
    CHECK(hipMemcpyAsync(host, src, bytes, hipMemcpyDeviceToHost, s0));
    CHECK(hipStreamSynchronize(s0));
    CHECK(hipSetDevice(dst_dev));
    CHECK(hipMemcpyAsync(dst, host, bytes, hipMemcpyHostToDevice, s1));
    CHECK(hipStreamSynchronize(s1));
    return 0;
  };
  for (int i = 0; i < warmup; ++i) {
    if (once()) {
      return 1;
    }
  }
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    if (once()) {
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  CHECK(hipStreamDestroy(s0));
  CHECK(hipStreamDestroy(s1));
  return 0;
}

static int enable_peer(int *can01, int *can10) {
  int count = 0;
  CHECK(hipGetDeviceCount(&count));
  std::fprintf(stderr, "device_count %d\n", count);
  if (count < 2) {
    std::fprintf(stderr, "need two HIP devices\n");
    return 1;
  }
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t p{};
    CHECK(hipGetDeviceProperties(&p, i));
    std::fprintf(stderr, "dev %d name %s pci %04x:%02x:%02x\n", i, p.name, p.pciDomainID,
                 p.pciBusID, p.pciDeviceID);
  }
  CHECK(hipDeviceCanAccessPeer(can01, 0, 1));
  CHECK(hipDeviceCanAccessPeer(can10, 1, 0));
  std::fprintf(stderr, "can_access 0->1 %d 1->0 %d\n", *can01, *can10);
  if (*can01) {
    CHECK(hipSetDevice(0));
    hipError_t e = hipDeviceEnablePeerAccess(1, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "enable 0->1 %s\n", hipGetErrorString(e));
      return 1;
    }
  }
  if (*can10) {
    CHECK(hipSetDevice(1));
    hipError_t e = hipDeviceEnablePeerAccess(0, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "enable 1->0 %s\n", hipGetErrorString(e));
      return 1;
    }
  }
  return 0;
}

static int run_same(bool host_staged) {
  int can01 = 0, can10 = 0;
  if (enable_peer(&can01, &can10)) {
    return 1;
  }
  const int pairs[2][2] = {{0, 1}, {1, 0}};
  for (auto &pair : pairs) {
    int src_dev = pair[0];
    int dst_dev = pair[1];
    for (const SizeSpec &spec : kSizes) {
      if (host_staged && spec.bytes < (1ull << 20)) {
        continue;
      }
      int iters = spec.iters;
      if (host_staged) {
        iters = std::min(iters, 8);
      }
      void *src = nullptr;
      void *dst = nullptr;
      CHECK(hipSetDevice(src_dev));
      CHECK(hipMalloc(&src, spec.bytes));
      CHECK(hipSetDevice(dst_dev));
      CHECK(hipMalloc(&dst, spec.bytes));
      CHECK(hipMemset(dst, 0, spec.bytes));
      uint32_t seed = 0xC0FFEE00u ^ static_cast<uint32_t>(spec.bytes);
      size_t nwords = spec.bytes / sizeof(uint32_t);
      if (launch_fill(reinterpret_cast<uint32_t *>(src), nwords, seed, src_dev)) {
        return 1;
      }
      std::vector<double> ms;
      long rss0 = read_vmrss_kb();
      void *host = nullptr;
      uint64_t staging = 0;
      int rc = 0;
      if (host_staged) {
        CHECK(hipHostMalloc(&host, spec.bytes, hipHostMallocDefault));
        staging = spec.bytes;
        rc = timed_host_wall(dst, dst_dev, src, src_dev, host, spec.bytes, 2, iters, ms);
      } else {
        rc = timed_peer(dst, dst_dev, src, src_dev, spec.bytes, 5, iters, ms);
      }
      long rss1 = read_vmrss_kb();
      unsigned long long bad = 1;
      if (rc == 0) {
        if (launch_verify(reinterpret_cast<uint32_t *>(dst), nwords, seed, dst_dev, &bad)) {
          return 1;
        }
      }
      char dir[16];
      std::snprintf(dir, sizeof(dir), "%dto%d", src_dev, dst_dev);
      emit(host_staged ? "hip_host_staged" : "hip_peer_same", spec.label, spec.bytes, dir,
           can01, can10, rc == 0 && bad == 0, bad, static_cast<int>(ms.size()), ms, rss0, rss1,
           staging, host_staged ? "pinned host bounce" : "hipMemcpyPeerAsync no host buffer");
      if (host) {
        hipHostFree(host);
      }
      CHECK(hipSetDevice(src_dev));
      CHECK(hipFree(src));
      CHECK(hipSetDevice(dst_dev));
      CHECK(hipFree(dst));
      if (rc) {
        return rc;
      }
    }
  }
  return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
  const char *p = static_cast<const char *>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    off += static_cast<size_t>(w);
  }
  return 0;
}

static int read_full(int fd, void *buf, size_t n) {
  char *p = static_cast<char *>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::recv(fd, p + off, n - off, 0);
    if (r == 0) {
      return -1;
    }
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    off += static_cast<size_t>(r);
  }
  return 0;
}

struct IpcHdr {
  uint64_t bytes;
  uint32_t seed;
  uint32_t handle_bytes;
  int iters;
};

static int run_ipc_producer(int port) {
  int can01 = 0, can10 = 0;
  if (enable_peer(&can01, &can10)) {
    return 1;
  }
  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) {
    return 1;
  }
  int yes = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(srv, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 || listen(srv, 1) < 0) {
    std::perror("bind/listen");
    return 1;
  }
  std::fprintf(stderr, "ipc producer listening %d\n", port);
  int conn = accept(srv, nullptr, nullptr);
  if (conn < 0) {
    return 1;
  }
  for (const SizeSpec &spec : kSizes) {
    void *src = nullptr;
    CHECK(hipSetDevice(0));
    CHECK(hipMalloc(&src, spec.bytes));
    uint32_t seed = 0xA11C0000u ^ static_cast<uint32_t>(spec.bytes);
    size_t nwords = spec.bytes / sizeof(uint32_t);
    if (launch_fill(reinterpret_cast<uint32_t *>(src), nwords, seed, 0)) {
      return 1;
    }
    hipIpcMemHandle_t handle{};
    CHECK(hipIpcGetMemHandle(&handle, src));
    IpcHdr hdr{};
    hdr.bytes = spec.bytes;
    hdr.seed = seed;
    hdr.handle_bytes = static_cast<uint32_t>(sizeof(handle));
    hdr.iters = spec.iters;
    if (write_full(conn, &hdr, sizeof(hdr)) || write_full(conn, &handle, sizeof(handle))) {
      std::fprintf(stderr, "producer send failed\n");
      return 1;
    }
    char ack = 0;
    if (read_full(conn, &ack, 1) || ack != 'K') {
      std::fprintf(stderr, "consumer failed size %s ack %d\n", spec.label, ack);
      return 1;
    }
    CHECK(hipFree(src));
  }
  IpcHdr end{};
  if (write_full(conn, &end, sizeof(end))) {
    return 1;
  }
  ::close(conn);
  ::close(srv);
  return 0;
}

static int run_ipc_consumer(int port) {
  int can01 = 0, can10 = 0;
  if (enable_peer(&can01, &can10)) {
    return 1;
  }
  int conn = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  for (int i = 0; i < 50; ++i) {
    if (connect(conn, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
      break;
    }
    if (i == 49) {
      std::perror("connect");
      return 1;
    }
    usleep(100000);
  }
  while (true) {
    IpcHdr hdr{};
    if (read_full(conn, &hdr, sizeof(hdr))) {
      std::fprintf(stderr, "consumer header failed\n");
      return 1;
    }
    if (hdr.bytes == 0) {
      break;
    }
    if (hdr.handle_bytes != sizeof(hipIpcMemHandle_t)) {
      std::fprintf(stderr, "unexpected handle size %u\n", hdr.handle_bytes);
      return 1;
    }
    hipIpcMemHandle_t handle{};
    if (read_full(conn, &handle, sizeof(handle))) {
      return 1;
    }
    CHECK(hipSetDevice(1));
    void *remote = nullptr;
    CHECK(hipIpcOpenMemHandle(&remote, handle, hipIpcMemLazyEnablePeerAccess));
    void *dst = nullptr;
    CHECK(hipMalloc(&dst, hdr.bytes));
    CHECK(hipMemset(dst, 0, hdr.bytes));
    std::vector<double> ms;
    long rss0 = read_vmrss_kb();
    int rc = timed_peer(dst, 1, remote, 0, hdr.bytes, 3, hdr.iters, ms);
    const char *note = "ipc hipMemcpyPeerAsync reused mapping";
    if (rc) {
      hipGetLastError();
      ms.clear();
      CHECK(hipSetDevice(1));
      hipStream_t stream;
      CHECK(hipStreamCreate(&stream));
      for (int i = 0; i < 3; ++i) {
        CHECK(hipMemcpyAsync(dst, remote, hdr.bytes, hipMemcpyDeviceToDevice, stream));
        CHECK(hipStreamSynchronize(stream));
      }
      for (int i = 0; i < hdr.iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        CHECK(hipMemcpyAsync(dst, remote, hdr.bytes, hipMemcpyDeviceToDevice, stream));
        CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      CHECK(hipStreamDestroy(stream));
      rc = 0;
      note = "ipc hipMemcpyDtoD after peer memcpy failed";
    }
    long rss1 = read_vmrss_kb();
    unsigned long long bad = 1;
    size_t nwords = hdr.bytes / sizeof(uint32_t);
    if (launch_verify(reinterpret_cast<uint32_t *>(dst), nwords, hdr.seed, 1, &bad)) {
      return 1;
    }
    const SizeSpec *spec = &kSizes[0];
    for (const SizeSpec &s : kSizes) {
      if (s.bytes == hdr.bytes) {
        spec = &s;
      }
    }
    emit("hip_ipc", spec->label, hdr.bytes, "0to1", can01, can10, bad == 0, bad,
         static_cast<int>(ms.size()), ms, rss0, rss1, 0, note);
    CHECK(hipFree(dst));
    CHECK(hipIpcCloseMemHandle(remote));
    char ack = bad == 0 ? 'K' : 'E';
    if (write_full(conn, &ack, 1)) {
      return 1;
    }
    if (bad != 0) {
      return 1;
    }
  }
  ::close(conn);
  return 0;
}

static int run_bidi() {
  int can01 = 0, can10 = 0;
  if (enable_peer(&can01, &can10)) {
    return 1;
  }
  const SizeSpec sizes[] = {
      {1ull << 20, "1MiB", 20},
      {16ull << 20, "16MiB", 12},
      {64ull << 20, "64MiB", 8},
      {256ull << 20, "256MiB", 6},
  };
  for (const SizeSpec &spec : sizes) {
    void *s0 = nullptr, *d1 = nullptr, *s1 = nullptr, *d0 = nullptr;
    CHECK(hipSetDevice(0));
    CHECK(hipMalloc(&s0, spec.bytes));
    CHECK(hipMalloc(&d0, spec.bytes));
    CHECK(hipMemset(d0, 0, spec.bytes));
    CHECK(hipSetDevice(1));
    CHECK(hipMalloc(&s1, spec.bytes));
    CHECK(hipMalloc(&d1, spec.bytes));
    CHECK(hipMemset(d1, 0, spec.bytes));
    uint32_t seed0 = 0x11110000u ^ static_cast<uint32_t>(spec.bytes);
    uint32_t seed1 = 0x22220000u ^ static_cast<uint32_t>(spec.bytes);
    size_t nwords = spec.bytes / sizeof(uint32_t);
    if (launch_fill(reinterpret_cast<uint32_t *>(s0), nwords, seed0, 0) ||
        launch_fill(reinterpret_cast<uint32_t *>(s1), nwords, seed1, 1)) {
      return 1;
    }
    hipStream_t to1, to0;
    CHECK(hipSetDevice(1));
    CHECK(hipStreamCreate(&to1));
    CHECK(hipSetDevice(0));
    CHECK(hipStreamCreate(&to0));
    auto once = [&]() -> int {
      CHECK(hipSetDevice(1));
      CHECK(hipMemcpyPeerAsync(d1, 1, s0, 0, spec.bytes, to1));
      CHECK(hipSetDevice(0));
      CHECK(hipMemcpyPeerAsync(d0, 0, s1, 1, spec.bytes, to0));
      CHECK(hipSetDevice(1));
      CHECK(hipStreamSynchronize(to1));
      CHECK(hipSetDevice(0));
      CHECK(hipStreamSynchronize(to0));
      return 0;
    };
    for (int i = 0; i < 3; ++i) {
      if (once()) {
        return 1;
      }
    }
    std::vector<double> ms;
    long rss0 = read_vmrss_kb();
    for (int i = 0; i < spec.iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      if (once()) {
        return 1;
      }
      auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    long rss1 = read_vmrss_kb();
    unsigned long long bad1 = 1, bad0 = 1;
    if (launch_verify(reinterpret_cast<uint32_t *>(d1), nwords, seed0, 1, &bad1) ||
        launch_verify(reinterpret_cast<uint32_t *>(d0), nwords, seed1, 0, &bad0)) {
      return 1;
    }
    char label[64];
    std::snprintf(label, sizeof(label), "%s_each_way", spec.label);
    emit("hip_peer_bidi", label, spec.bytes * 2, "both", can01, can10, bad0 == 0 && bad1 == 0,
         bad0 + bad1, static_cast<int>(ms.size()), ms, rss0, rss1, 0,
         "host_steady_clock until both streams complete; bytes is the sum of both payloads; "
         "bandwidth is aggregate over wall time");
    CHECK(hipStreamDestroy(to1));
    CHECK(hipStreamDestroy(to0));
    CHECK(hipSetDevice(0));
    CHECK(hipFree(s0));
    CHECK(hipFree(d0));
    CHECK(hipSetDevice(1));
    CHECK(hipFree(s1));
    CHECK(hipFree(d1));
  }
  return 0;
}

static int run_concurrent() {
  int can01 = 0, can10 = 0;
  if (enable_peer(&can01, &can10)) {
    return 1;
  }
  const int widths[] = {1, 2, 4, 8};
  const SizeSpec sizes[] = {
      {16ull << 20, "16MiB", 8},
      {64ull << 20, "64MiB", 6},
      {256ull << 20, "256MiB", 4},
  };
  for (const SizeSpec &spec : sizes) {
    for (int n : widths) {
      std::vector<void *> src(n), dst(n);
      std::vector<hipStream_t> streams(n);
      std::vector<uint32_t> seeds(n);
      size_t nwords = spec.bytes / sizeof(uint32_t);
      for (int i = 0; i < n; ++i) {
        CHECK(hipSetDevice(0));
        CHECK(hipMalloc(&src[i], spec.bytes));
        CHECK(hipSetDevice(1));
        CHECK(hipMalloc(&dst[i], spec.bytes));
        CHECK(hipMemset(dst[i], 0, spec.bytes));
        seeds[i] = 0x51500000u ^ (static_cast<uint32_t>(spec.bytes) + static_cast<uint32_t>(i) * 17u);
        if (launch_fill(reinterpret_cast<uint32_t *>(src[i]), nwords, seeds[i], 0)) {
          return 1;
        }
        CHECK(hipSetDevice(1));
        CHECK(hipStreamCreate(&streams[i]));
      }
      auto once = [&]() -> int {
        for (int i = 0; i < n; ++i) {
          CHECK(hipSetDevice(1));
          CHECK(hipMemcpyPeerAsync(dst[i], 1, src[i], 0, spec.bytes, streams[i]));
        }
        for (int i = 0; i < n; ++i) {
          CHECK(hipSetDevice(1));
          CHECK(hipStreamSynchronize(streams[i]));
        }
        return 0;
      };
      for (int w = 0; w < 2; ++w) {
        if (once()) {
          return 1;
        }
      }
      std::vector<double> ms;
      long rss0 = read_vmrss_kb();
      for (int it = 0; it < spec.iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        if (once()) {
          return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      long rss1 = read_vmrss_kb();
      unsigned long long bad = 0;
      for (int i = 0; i < n; ++i) {
        unsigned long long one = 1;
        if (launch_verify(reinterpret_cast<uint32_t *>(dst[i]), nwords, seeds[i], 1, &one)) {
          return 1;
        }
        bad += one;
      }
      char label[64];
      std::snprintf(label, sizeof(label), "%s_x%d", spec.label, n);
      emit("hip_peer_concurrent", label, spec.bytes * static_cast<uint64_t>(n), "0to1", can01,
           can10, bad == 0, bad, static_cast<int>(ms.size()), ms, rss0, rss1, 0,
           "host_steady_clock aggregate; bytes is the sum of overlapping copies; "
           "not the sum of per-stream durations");
      for (int i = 0; i < n; ++i) {
        CHECK(hipStreamDestroy(streams[i]));
        CHECK(hipSetDevice(0));
        CHECK(hipFree(src[i]));
        CHECK(hipSetDevice(1));
        CHECK(hipFree(dst[i]));
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "same";
  int port = argc > 2 ? std::atoi(argv[2]) : 18766;
  if (std::strcmp(mode, "same") == 0) {
    return run_same(false);
  }
  if (std::strcmp(mode, "host") == 0) {
    return run_same(true);
  }
  if (std::strcmp(mode, "bidi") == 0) {
    return run_bidi();
  }
  if (std::strcmp(mode, "concurrent") == 0) {
    return run_concurrent();
  }
  if (std::strcmp(mode, "ipc-producer") == 0) {
    return run_ipc_producer(port);
  }
  if (std::strcmp(mode, "ipc-consumer") == 0) {
    return run_ipc_consumer(port);
  }
  std::fprintf(stderr, "usage: %s same|host|bidi|concurrent|ipc-producer|ipc-consumer [port]\n",
               argv[0]);
  return 2;
}

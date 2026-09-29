#include <hip/hip_runtime.h>

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t e = (call);                                                     \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e));           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

struct Header {
  uint64_t bytes;
  hipIpcMemHandle_t handle;
};

static int send_all(int fd, const void *buf, size_t n) {
  const char *p = static_cast<const char *>(buf);
  for (size_t off = 0; off < n;) {
    ssize_t ret = send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (ret <= 0) return 1;
    off += static_cast<size_t>(ret);
  }
  return 0;
}

static int recv_all(int fd, void *buf, size_t n) {
  char *p = static_cast<char *>(buf);
  for (size_t off = 0; off < n;) {
    ssize_t ret = recv(fd, p + off, n - off, 0);
    if (ret <= 0) return 1;
    off += static_cast<size_t>(ret);
  }
  return 0;
}

static int make_server(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ||
      listen(fd, 1)) {
    return -1;
  }
  int conn = accept(fd, nullptr, nullptr);
  close(fd);
  return conn;
}

static int make_client(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  for (int i = 0; i < 50; ++i) {
    if (!connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) return fd;
    usleep(100000);
  }
  return -1;
}

static double pct(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  double index = p / 100.0 * (values.size() - 1);
  size_t lower = static_cast<size_t>(index);
  double fraction = index - lower;
  if (lower + 1 == values.size()) return values[lower];
  return values[lower] * (1 - fraction) + values[lower + 1] * fraction;
}

static int producer(int port, int device, uint64_t bytes) {
  int fd = make_server(port);
  if (fd < 0) return 1;
  HIP_OK(hipSetDevice(device));
  void *src = nullptr;
  HIP_OK(hipMalloc(&src, bytes));
  HIP_OK(hipMemset(src, 0xa5, bytes));
  HIP_OK(hipDeviceSynchronize());
  Header header{bytes, {}};
  HIP_OK(hipIpcGetMemHandle(&header.handle, src));
  if (send_all(fd, &header, sizeof(header))) return 1;
  char done = 0;
  if (recv_all(fd, &done, 1)) return 1;
  HIP_OK(hipFree(src));
  close(fd);
  return done == 1 ? 0 : 1;
}

static int consumer(int port, int src_device, int dst_device, int iters) {
  int fd = make_client(port);
  if (fd < 0) return 1;
  Header header{};
  if (recv_all(fd, &header, sizeof(header))) return 1;
  HIP_OK(hipSetDevice(dst_device));
  auto setup_start = std::chrono::steady_clock::now();
  void *remote = nullptr;
  HIP_OK(hipIpcOpenMemHandle(&remote, header.handle, hipIpcMemLazyEnablePeerAccess));
  void *dst = nullptr;
  HIP_OK(hipMalloc(&dst, header.bytes));
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  auto setup_end = std::chrono::steady_clock::now();
  double registration_ms =
      std::chrono::duration<double, std::milli>(setup_end - setup_start).count();

  auto copy_once = [&]() {
    HIP_OK(hipMemcpyPeerAsync(dst, dst_device, remote, src_device, header.bytes, stream));
    HIP_OK(hipStreamSynchronize(stream));
    return 0;
  };

  HIP_OK(hipMemset(dst, 0, header.bytes));
  HIP_OK(hipDeviceSynchronize());
  auto cold_start = std::chrono::steady_clock::now();
  if (copy_once()) return 1;
  auto cold_end = std::chrono::steady_clock::now();
  double cold_ms =
      std::chrono::duration<double, std::milli>(cold_end - cold_start).count();

  for (int i = 0; i < 2; ++i) {
    if (copy_once()) return 1;
  }
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto start = std::chrono::steady_clock::now();
    if (copy_once()) return 1;
    auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::vector<unsigned char> host(header.bytes);
  HIP_OK(hipMemcpy(host.data(), dst, header.bytes, hipMemcpyDeviceToHost));
  uint64_t bad = 0;
  for (unsigned char value : host) bad += value != 0xa5;
  double p50 = pct(samples, 50), p95 = pct(samples, 95), p99 = pct(samples, 99);
  double mean = 0;
  for (double sample : samples) mean += sample;
  mean /= samples.size();
  std::printf(
      "{\"mode\":\"hip_ipc_matched\",\"bytes\":%llu,\"iters\":%d,\"ok\":%s,"
      "\"mismatches\":%llu,\"registration_ms\":%.4f,\"cold_transfer_ms\":%.4f,"
      "\"latency_ms\":{\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f,"
      "\"min\":%.4f,\"mean\":%.4f,\"max\":%.4f},"
      "\"GBps_at_p50\":%.4f,\"GBps_at_mean\":%.4f}\n",
      static_cast<unsigned long long>(header.bytes), iters, bad ? "false" : "true",
      static_cast<unsigned long long>(bad), registration_ms, cold_ms, p50, p95,
      p99, *std::min_element(samples.begin(), samples.end()), mean,
      *std::max_element(samples.begin(), samples.end()),
      (header.bytes / (p50 / 1e3)) / 1e9,
      (header.bytes / (mean / 1e3)) / 1e9);
  char done = bad ? 0 : 1;
  send_all(fd, &done, 1);
  HIP_OK(hipStreamDestroy(stream));
  HIP_OK(hipFree(dst));
  HIP_OK(hipIpcCloseMemHandle(remote));
  close(fd);
  return bad ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: %s producer|consumer port src_dev dst_dev bytes iters\n",
                 argv[0]);
    return 2;
  }
  int port = std::atoi(argv[2]);
  int src = std::atoi(argv[3]);
  int dst = std::atoi(argv[4]);
  uint64_t bytes = std::strtoull(argv[5], nullptr, 10);
  int iters = std::atoi(argv[6]);
  if (!std::strcmp(argv[1], "producer")) return producer(port, src, bytes);
  return consumer(port, src, dst, iters);
}

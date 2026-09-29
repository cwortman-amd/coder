#include <hip/hip_runtime.h>
#include <nixl.h>

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#define CHECK_HIP(x) do { auto e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)
#define CHECK_NIXL(x) do { auto s = (x); if (s != NIXL_SUCCESS) { \
  std::fprintf(stderr, "%s: %d\n", #x, static_cast<int>(s)); return 1; } } while (0)

static int send_blob(int fd, const void *data, size_t length) {
  uint64_t wire_length = length;
  if (send(fd, &wire_length, sizeof(wire_length), MSG_NOSIGNAL) != sizeof(wire_length)) return 1;
  const char *bytes = static_cast<const char *>(data);
  for (size_t sent = 0; sent < length;) {
    ssize_t count = send(fd, bytes + sent, length - sent, MSG_NOSIGNAL);
    if (count <= 0) return 1;
    sent += static_cast<size_t>(count);
  }
  return 0;
}

static std::string recv_blob(int fd) {
  uint64_t length = 0;
  if (recv(fd, &length, sizeof(length), MSG_WAITALL) != sizeof(length)) return {};
  std::string result(length, '\0');
  for (size_t received = 0; received < length;) {
    ssize_t count = recv(fd, result.data() + received, length - received, 0);
    if (count <= 0) return {};
    received += static_cast<size_t>(count);
  }
  return result;
}

static int server(int port) {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0), yes = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in address{AF_INET, htons(static_cast<uint16_t>(port)), {htonl(INADDR_LOOPBACK)}};
  if (bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ||
      listen(listen_fd, 1)) return -1;
  int fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);
  return fd;
}

static int client(int port) {
  sockaddr_in address{AF_INET, htons(static_cast<uint16_t>(port)), {htonl(INADDR_LOOPBACK)}};
  for (int attempt = 0; attempt < 50; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address))) return fd;
    close(fd);
    usleep(100000);
  }
  return -1;
}

static double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  double index = p / 100.0 * (values.size() - 1);
  size_t lower = static_cast<size_t>(index);
  double fraction = index - lower;
  return lower + 1 == values.size() ? values[lower]
      : values[lower] * (1 - fraction) + values[lower + 1] * fraction;
}

static nixl_xfer_dlist_t make_descs(uintptr_t base, size_t bytes, int device, int count) {
  nixl_xfer_dlist_t list(VRAM_SEG);
  size_t chunk = bytes / static_cast<size_t>(count);
  for (int i = 0; i < count; ++i) {
    size_t offset = static_cast<size_t>(i) * chunk;
    size_t length = i == count - 1 ? bytes - offset : chunk;
    list.addDesc(nixlBasicDesc(base + offset, length, device));
  }
  return list;
}

static int producer(int port, int device, size_t bytes, int regions) {
  CHECK_HIP(hipSetDevice(device));
  const size_t region_bytes = bytes / static_cast<size_t>(regions);
  std::vector<void *> sources(regions, nullptr);
  nixl_reg_dlist_t registration(VRAM_SEG);
  for (int region = 0; region < regions; ++region) {
    CHECK_HIP(hipMalloc(&sources[region], region_bytes));
    CHECK_HIP(hipMemset(sources[region], 0xa5, region_bytes));
    registration.addDesc(nixlBlobDesc(
        reinterpret_cast<uintptr_t>(sources[region]), region_bytes, device));
  }
  CHECK_HIP(hipDeviceSynchronize());

  nixlAgent agent("producer", nixlAgentConfig{});
  nixlBackendH *backend = nullptr;
  nixl_b_params_t params;
  CHECK_NIXL(agent.createBackend("HIP_IPC", params, backend));
  nixl_opt_args_t options;
  options.backends.push_back(backend);
  CHECK_NIXL(agent.registerMem(registration, &options));
  std::string metadata;
  CHECK_NIXL(agent.getLocalMD(metadata));

  int fd = server(port);
  if (fd < 0) return 1;
  struct RemoteBuffer { uintptr_t pointer; int device; };
  std::vector<RemoteBuffer> remotes(regions);
  for (int region = 0; region < regions; ++region) {
    remotes[region] = {reinterpret_cast<uintptr_t>(sources[region]), device};
  }
  if (send_blob(fd, remotes.data(), remotes.size() * sizeof(RemoteBuffer)) ||
      send_blob(fd, metadata.data(), metadata.size())) return 1;
  if (recv_blob(fd) != "done") return 1;
  CHECK_NIXL(agent.deregisterMem(registration, &options));
  for (void *source : sources) CHECK_HIP(hipFree(source));
  close(fd);
  return 0;
}

static int consumer(int port, int device, size_t bytes, int descriptors,
                     int iterations, int regions) {
  int fd = client(port);
  if (fd < 0) return 1;
  std::string pointer_blob = recv_blob(fd);
  std::string metadata = recv_blob(fd);
  struct RemoteBuffer { uintptr_t pointer; int device; };
  if (pointer_blob.size() != sizeof(RemoteBuffer) * static_cast<size_t>(regions) ||
      metadata.empty()) return 1;
  std::vector<RemoteBuffer> remotes(regions);
  std::memcpy(remotes.data(), pointer_blob.data(), pointer_blob.size());

  CHECK_HIP(hipSetDevice(device));
  const size_t region_bytes = bytes / static_cast<size_t>(regions);
  std::vector<void *> destinations(regions, nullptr);
  nixl_reg_dlist_t registration(VRAM_SEG);
  for (int region = 0; region < regions; ++region) {
    CHECK_HIP(hipMalloc(&destinations[region], region_bytes));
    registration.addDesc(nixlBlobDesc(
        reinterpret_cast<uintptr_t>(destinations[region]), region_bytes, device));
  }
  nixlAgent agent("consumer", nixlAgentConfig{});
  nixlBackendH *backend = nullptr;
  nixl_b_params_t params;
  CHECK_NIXL(agent.createBackend("HIP_IPC", params, backend));
  nixl_opt_args_t options;
  options.backends.push_back(backend);
  CHECK_NIXL(agent.registerMem(registration, &options));
  std::string remote_name;
  CHECK_NIXL(agent.loadRemoteMD(metadata, remote_name));

  const int per_region = descriptors / regions;
  nixl_xfer_dlist_t local(VRAM_SEG), remote(VRAM_SEG);
  for (int region = 0; region < regions; ++region) {
    auto local_part = make_descs(
        reinterpret_cast<uintptr_t>(destinations[region]), region_bytes, device, per_region);
    auto remote_part = make_descs(
        remotes[region].pointer, region_bytes, remotes[region].device, per_region);
    for (int index = 0; index < per_region; ++index) {
      local.addDesc(local_part[index]);
      remote.addDesc(remote_part[index]);
    }
  }
  nixlXferReqH *request = nullptr;
  CHECK_NIXL(agent.createXferReq(
      NIXL_READ, local, remote, remote_name, request, &options));

  auto transfer = [&]() -> int {
    nixl_status_t status = agent.postXferReq(request);
    while (status == NIXL_IN_PROG) status = agent.getXferStatus(request);
    return status == NIXL_SUCCESS ? 0 : 1;
  };
  auto cold_start = std::chrono::steady_clock::now();
  if (transfer()) return 1;
  double cold = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cold_start).count();
  for (int i = 0; i < 2; ++i) if (transfer()) return 1;
  std::vector<double> samples;
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    if (transfer()) return 1;
    samples.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
  }
  std::vector<unsigned char> host(region_bytes);
  size_t mismatches = 0;
  for (void *destination : destinations) {
    CHECK_HIP(hipMemcpy(host.data(), destination, region_bytes, hipMemcpyDeviceToHost));
    mismatches += static_cast<size_t>(std::count_if(
        host.begin(), host.end(), [](unsigned char value) { return value != 0xa5; }));
  }
  double p50 = percentile(samples, 50), p95 = percentile(samples, 95);
  double p99 = percentile(samples, 99);
  std::printf("{\"backend\":\"HIP_IPC\",\"bytes\":%zu,\"descriptors\":%d,"
              "\"regions\":%d,\"iters\":%d,\"ok\":%s,\"mismatches\":%zu,\"cold_ms\":%.4f,"
              "\"latency_ms\":{\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f},"
              "\"GBps_at_p50\":%.4f}\n",
              bytes, descriptors, regions, iterations, mismatches ? "false" : "true",
              mismatches, cold, p50, p95, p99, (bytes / (p50 / 1e3)) / 1e9);
  send_blob(fd, "done", 4);
  CHECK_NIXL(agent.releaseXferReq(request));
  CHECK_NIXL(agent.deregisterMem(registration, &options));
  for (void *destination : destinations) CHECK_HIP(hipFree(destination));
  close(fd);
  return mismatches ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc < 7 || argc > 8) {
    std::fprintf(stderr,
                 "usage: %s producer|consumer port device bytes descriptors iters [regions]\n",
                 argv[0]);
    return 2;
  }
  int port = std::atoi(argv[2]), device = std::atoi(argv[3]);
  size_t bytes = std::strtoull(argv[4], nullptr, 10);
  int descriptors = std::atoi(argv[5]), iterations = std::atoi(argv[6]);
  int regions = argc == 8 ? std::atoi(argv[7]) : 1;
  if (regions < 1 || descriptors < regions || descriptors % regions != 0 ||
      bytes % static_cast<size_t>(regions) != 0) {
    return 2;
  }
  return !std::strcmp(argv[1], "producer")
      ? producer(port, device, bytes, regions)
      : consumer(port, device, bytes, descriptors, iterations, regions);
}

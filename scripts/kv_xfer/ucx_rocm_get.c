/* Two-process UCP GET of HIP memory. The consumer pulls GPU src_dev into GPU dst_dev.
 * Buffers come from hipMalloc, then ucp_mem_map, so UCX does not have to allocate VRAM.
 */
#include <hip/hip_runtime.h>
#include <ucp/api/ucp.h>

#include <arpa/inet.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static ucp_worker_h g_worker;

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void hip_ok(hipError_t err, const char *what)
{
    if (err != hipSuccess) {
        fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(err));
        exit(1);
    }
}

static void ucs_ok(ucs_status_t st, const char *what)
{
    if (st != UCS_OK) {
        fprintf(stderr, "UCS %s: %s\n", what, ucs_status_string(st));
        exit(1);
    }
}

static void send_all(int fd, const void *buf, size_t n)
{
    const char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t w = send(fd, p + off, n - off, MSG_NOSIGNAL);
        if (w <= 0) {
            die("send");
        }
        off += (size_t)w;
    }
}

static void recv_all(int fd, void *buf, size_t n)
{
    char *p = buf;
    size_t off = 0;
    while (off < n) {
        ssize_t r = recv(fd, p + off, n - off, 0);
        if (r <= 0) {
            die("recv");
        }
        off += (size_t)r;
    }
}

static void send_blob(int fd, const void *buf, uint64_t n)
{
    uint64_t len = n;
    send_all(fd, &len, sizeof(len));
    if (n) {
        send_all(fd, buf, n);
    }
}

static void *recv_blob(int fd, uint64_t *n_out)
{
    uint64_t n = 0;
    recv_all(fd, &n, sizeof(n));
    void *buf = n ? malloc(n) : NULL;
    if (n && !buf) {
        die("malloc blob");
    }
    if (n) {
        recv_all(fd, buf, n);
    }
    *n_out = n;
    return buf;
}

static int listen_local(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) || listen(fd, 1)) {
        die("bind/listen");
    }
    int conn = accept(fd, NULL, NULL);
    if (conn < 0) {
        die("accept");
    }
    close(fd);
    return conn;
}

static int connect_local(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    for (int i = 0; i < 50; i++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            return fd;
        }
        usleep(100000);
    }
    die("connect");
    return -1;
}

static void init_ucp(ucp_context_h *ctx, ucp_worker_h *worker)
{
    ucp_params_t params;
    ucp_worker_params_t wparams;
    ucp_config_t *config;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_TAG;
    ucs_ok(ucp_config_read(NULL, NULL, &config), "config_read");
    ucs_ok(ucp_init(&params, config, ctx), "ucp_init");
    ucp_config_release(config);
    memset(&wparams, 0, sizeof(wparams));
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucs_ok(ucp_worker_create(*ctx, &wparams, worker), "worker_create");
    g_worker = *worker;
}

static void exchange_workers(int fd, ucp_worker_h local, ucp_address_t **remote, size_t *remote_len)
{
    ucp_address_t *addr = NULL;
    size_t addr_len = 0;
    ucs_ok(ucp_worker_get_address(local, &addr, &addr_len), "get_address");
    send_blob(fd, addr, addr_len);
    ucp_worker_release_address(local, addr);
    uint64_t n = 0;
    *remote = recv_blob(fd, &n);
    *remote_len = n;
}

static void wait_req(ucs_status_ptr_t req)
{
    if (UCS_PTR_IS_ERR(req)) {
        fprintf(stderr, "request error %s\n", ucs_status_string(UCS_PTR_STATUS(req)));
        exit(1);
    }
    if (UCS_PTR_IS_PTR(req)) {
        ucs_status_t st;
        do {
            ucp_worker_progress(g_worker);
            st = ucp_request_check_status(req);
        } while (st == UCS_INPROGRESS);
        ucp_request_free(req);
        ucs_ok(st, "request status");
    }
}

static double ms_since(struct timespec t0)
{
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double percentile(const double *values, int count, double p)
{
    double *copy = malloc((size_t)count * sizeof(*copy));
    if (!copy) {
        die("percentile malloc");
    }
    memcpy(copy, values, (size_t)count * sizeof(*copy));
    qsort(copy, (size_t)count, sizeof(*copy), cmp_double);
    double index = (p / 100.0) * (count - 1);
    int lower = (int)index;
    double fraction = index - lower;
    double value = copy[lower];
    if (lower + 1 < count) {
        value = value * (1.0 - fraction) + copy[lower + 1] * fraction;
    }
    free(copy);
    return value;
}

static int run_producer(int port, int dev, size_t nbytes, uint8_t pattern)
{
    int fd = listen_local(port);
    hip_ok(hipSetDevice(dev), "set device");
    void *buf = NULL;
    hip_ok(hipMalloc(&buf, nbytes), "malloc");
    hip_ok(hipMemset(buf, pattern, nbytes), "memset");
    hip_ok(hipDeviceSynchronize(), "sync fill");

    ucp_context_h ctx;
    ucp_worker_h worker;
    init_ucp(&ctx, &worker);

    ucp_mem_map_params_t mp;
    memset(&mp, 0, sizeof(mp));
    mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                    UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mp.address = buf;
    mp.length = nbytes;
    mp.memory_type = UCS_MEMORY_TYPE_ROCM;
    ucp_mem_h memh;
    ucs_ok(ucp_mem_map(ctx, &mp, &memh), "mem_map");
    void *rkey = NULL;
    size_t rkey_size = 0;
    ucs_ok(ucp_rkey_pack(ctx, memh, &rkey, &rkey_size), "rkey_pack");

    ucp_address_t *remote = NULL;
    size_t remote_len = 0;
    exchange_workers(fd, worker, &remote, &remote_len);
    uint64_t meta[2] = {(uint64_t)(uintptr_t)buf, nbytes};
    send_blob(fd, meta, sizeof(meta));
    send_blob(fd, rkey, rkey_size);
    fprintf(stderr, "producer gpu %d ptr %p bytes %zu rkey %zu\n", dev, buf, nbytes, rkey_size);

    /* Stay live so the consumer GET can complete. */
    char done = 0;
    while (!done) {
        ucp_worker_progress(worker);
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv = {0, 1000};
        if (select(fd + 1, &rfds, NULL, NULL, &tv) > 0) {
            recv_all(fd, &done, 1);
        }
    }
    fprintf(stderr, "producer done\n");
    ucp_rkey_buffer_release(rkey);
    ucp_mem_unmap(ctx, memh);
    hipFree(buf);
    ucp_worker_destroy(worker);
    ucp_cleanup(ctx);
    close(fd);
    free(remote);
    return 0;
}

static int run_consumer(int port, int dev, size_t nbytes, uint8_t pattern, int iters,
                        int descs)
{
    int fd = connect_local(port);
    hip_ok(hipSetDevice(dev), "set device");
    struct timespec registration_start;
    clock_gettime(CLOCK_MONOTONIC, &registration_start);
    void *buf = NULL;
    hip_ok(hipMalloc(&buf, nbytes), "malloc");
    hip_ok(hipMemset(buf, 0, nbytes), "clear");

    ucp_context_h ctx;
    ucp_worker_h worker;
    init_ucp(&ctx, &worker);
    ucp_mem_map_params_t mp;
    memset(&mp, 0, sizeof(mp));
    mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                    UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mp.address = buf;
    mp.length = nbytes;
    mp.memory_type = UCS_MEMORY_TYPE_ROCM;
    ucp_mem_h memh;
    ucs_ok(ucp_mem_map(ctx, &mp, &memh), "mem_map");
    double registration_ms = ms_since(registration_start);

    ucp_address_t *remote_addr = NULL;
    size_t remote_len = 0;
    exchange_workers(fd, worker, &remote_addr, &remote_len);
    uint64_t meta_n = 0;
    uint64_t *meta = recv_blob(fd, &meta_n);
    uint64_t rkey_n = 0;
    void *rkey_buf = recv_blob(fd, &rkey_n);
    uint64_t remote_ptr = meta[0];
    if (meta[1] != nbytes) {
        die("size mismatch");
    }

    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = remote_addr;
    ucp_ep_h ep;
    ucs_ok(ucp_ep_create(worker, &ep_params, &ep), "ep_create");
    ucp_rkey_h rkey;
    ucs_ok(ucp_ep_rkey_unpack(ep, rkey_buf, &rkey), "rkey_unpack");

    ucp_request_param_t req_param;
    memset(&req_param, 0, sizeof(req_param));
    if (descs < 1) {
        descs = 1;
    }
    if ((size_t)descs > nbytes) {
        descs = (int)nbytes;
    }
    ucs_status_ptr_t *requests = calloc((size_t)descs, sizeof(*requests));
    if (!requests) {
        die("requests calloc");
    }
    size_t chunk = nbytes / (size_t)descs;
    double transfer_once_ms;
    #define POST_TRANSFER() do {                                                \
        for (int d = 0; d < descs; d++) {                                      \
            size_t offset = (size_t)d * chunk;                                 \
            size_t length = (d == descs - 1) ? nbytes - offset : chunk;         \
            requests[d] = ucp_get_nbx(ep, (char *)buf + offset, length,         \
                                      remote_ptr + offset, rkey, &req_param);    \
        }                                                                       \
        for (int d = 0; d < descs; d++) {                                      \
            wait_req(requests[d]);                                              \
        }                                                                       \
        hip_ok(hipDeviceSynchronize(), "gpu sync");                            \
    } while (0)

    hip_ok(hipMemset(buf, 0, nbytes), "cold clear");
    hip_ok(hipDeviceSynchronize(), "cold clear sync");
    struct timespec cold_start;
    clock_gettime(CLOCK_MONOTONIC, &cold_start);
    POST_TRANSFER();
    transfer_once_ms = ms_since(cold_start);

    for (int i = 0; i < 2; i++) {
        hip_ok(hipMemset(buf, 0, nbytes), "warmup clear");
        hip_ok(hipDeviceSynchronize(), "warmup sync");
        POST_TRANSFER();
    }
    if (iters > 1000) {
        iters = 1000;
    }
    double *samples = calloc((size_t)iters, sizeof(*samples));
    if (!samples) {
        die("samples calloc");
    }
    for (int i = 0; i < iters; i++) {
        hip_ok(hipMemset(buf, 0, nbytes), "clear");
        hip_ok(hipDeviceSynchronize(), "clear sync");
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        POST_TRANSFER();
        samples[i] = ms_since(t0);
    }
    uint8_t *host = malloc(nbytes);
    if (!host) {
        die("host malloc");
    }
    hip_ok(hipMemcpy(host, buf, nbytes, hipMemcpyDeviceToHost), "d2h");
    size_t bad = 0;
    for (size_t i = 0; i < nbytes; i++) {
        if (host[i] != pattern) {
            bad++;
        }
    }
    double sum = 0, pmin = samples[0], pmax = samples[0];
    for (int i = 0; i < iters; i++) {
        sum += samples[i];
        if (samples[i] < pmin) {
            pmin = samples[i];
        }
        if (samples[i] > pmax) {
            pmax = samples[i];
        }
    }
    double mean = sum / iters;
    double gbps = (nbytes / (mean / 1e3)) / 1e9;
    double p50 = percentile(samples, iters, 50.0);
    double p95 = percentile(samples, iters, 95.0);
    double p99 = percentile(samples, iters, 99.0);
    printf("{\"bytes\":%zu,\"descriptors\":%d,\"iters\":%d,\"ok\":%s,\"mismatches\":%zu,"
           "\"registration_ms\":%.4f,\"cold_transfer_ms\":%.4f,"
           "\"latency_ms\":{\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f,"
           "\"min\":%.4f,\"mean\":%.4f,\"max\":%.4f},"
           "\"GBps_at_p50\":%.4f,\"GBps_at_mean\":%.4f,\"dst_dev\":%d}\n",
           nbytes, descs, iters, bad ? "false" : "true", bad, registration_ms,
           transfer_once_ms, p50, p95, p99, pmin, mean, pmax,
           (nbytes / (p50 / 1e3)) / 1e9, gbps, dev);
    fflush(stdout);
    char done = 1;
    send_all(fd, &done, 1);
    free(host);
    free(samples);
    free(requests);
    free(meta);
    free(rkey_buf);
    free(remote_addr);
    ucp_rkey_destroy(rkey);
    ucp_request_param_t close_param;
    memset(&close_param, 0, sizeof(close_param));
    wait_req(ucp_ep_close_nbx(ep, &close_param));
    ucp_mem_unmap(ctx, memh);
    hipFree(buf);
    ucp_worker_destroy(worker);
    ucp_cleanup(ctx);
    close(fd);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s producer|consumer port dev bytes iters [descriptors]\n",
                argv[0]);
        return 2;
    }
    const char *role = argv[1];
    int port = atoi(argv[2]);
    int dev = atoi(argv[3]);
    size_t nbytes = strtoull(argv[4], NULL, 10);
    int iters = atoi(argv[5]);
    int descs = argc > 6 ? atoi(argv[6]) : 1;
    unsigned maj = 0, minor = 0, rel = 0;
    ucp_get_version(&maj, &minor, &rel);
    fprintf(stderr, "ucp %u.%u.%u role %s dev %d bytes %zu\n", maj, minor, rel, role, dev, nbytes);
    if (strcmp(role, "producer") == 0) {
        return run_producer(port, dev, nbytes, 0xA5);
    }
    return run_consumer(port, dev, nbytes, 0xA5, iters, descs);
}

/// gst-nvmm-cpp tunable defaults.
///
/// All numbers that affect runtime behavior live here. Wire-format
/// constants (magic, version code, pool cap, cache line) live in
/// shm_protocol.h and must NOT be tuned per build — those are bytes
/// on disk shared between processes.
///
/// Build-time overrides via meson:
///   -Dpool_release_delay=N      consumer keeps the slot ref live for
///                               N frames after fetch (encoder pipeline
///                               depth). Default 4.
///   -Dpoll_interval_us=N        consumer/wait-loop poll interval, in
///                               microseconds. Lower = lower latency,
///                               higher CPU. Default 1000 (1 ms).
#pragma once

#include "config.h"   /* meson-generated overrides land here */

#ifdef __cplusplus

namespace nvmm {
namespace config {

/* ---- Element-instance defaults (also overridable per element via the
 * GObject properties shm-name, pool-size). ---- */
constexpr const char *default_shm_name   = "/nvmm_sink_0";
constexpr int         default_pool_size  = 8;
constexpr int         min_pool_size      = 4;
/* The max pool size is wire-format-bound (NVMM_POOL_SIZE_MAX in
 * shm_protocol.h), not tunable here. */

/* Producer socket path: derived from shm_name at runtime, prefixed with
 * this. Slashes in shm_name get flattened to underscores. */
constexpr const char *socket_path_prefix = "/tmp/nvmm";

/* ---- Timing knobs (the meson options injects -DNVMM_*) ---- */

/* Consumer keeps a ref to the slot it just fetched for this many frames
 * before releasing — covers downstream encoder pipeline depth so the
 * producer can't reclaim a slot the encoder is still reading. */
constexpr int pool_release_delay =
#ifdef NVMM_POOL_RELEASE_DELAY
    NVMM_POOL_RELEASE_DELAY;
#else
    4;
#endif

/* Poll interval for the consumer fetch wait loop and start() handshake
 * wait. Latency floor; lower burns more CPU. */
constexpr int poll_interval_us =
#ifdef NVMM_POLL_INTERVAL_US
    NVMM_POLL_INTERVAL_US;
#else
    1000;
#endif

/* Producer-ready wait timeout in consumer_start(), as a count of
 * poll_interval_us ticks. Default = 5000 ticks * 1ms = 5s. */
constexpr int start_wait_ticks = 5000;

/* Consumer fetch idle timeout, count of poll_interval_us ticks.
 * 0 = block forever. Default = 2000 ticks * 1ms = 2s before returning EOS. */
constexpr int fetch_idle_ticks =
#ifdef NVMM_FETCH_IDLE_TICKS
    NVMM_FETCH_IDLE_TICKS;
#else
    2000;
#endif

/* CAS retry budget when a consumer races a producer writer-locking the
 * current slot. Combined with busy_slot_backoff_us, default is
 * 1000 * 100us = 100ms before giving up. */
constexpr int cas_retry_limit       = 1000;
constexpr int busy_slot_backoff_us  = 100;

/* Producer accept-loop poll timeout, milliseconds. Bound on how long
 * stop() blocks waiting for the accept thread to notice. */
constexpr int accept_poll_ms = 200;

/* Backoff in accept_loop while waiting for set_caps to allocate the
 * pool, microseconds. */
constexpr int accept_pool_wait_us = 10000;

} /* namespace config */
} /* namespace nvmm */

#endif /* __cplusplus */

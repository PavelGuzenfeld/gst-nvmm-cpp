/// fd_ipc.h — SCM_RIGHTS file descriptor passing over unix domain sockets.
/// Used by nvmmsink (server) and nvmmappsrc (client) for GPU-copy IPC
/// (pool DMA-buf fd passing).
#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Send an array of file descriptors over a connected unix socket.
/// Returns 0 on success, -1 on error (check errno).
static inline int
nvmm_send_fds(int sock, const int *fds, int count)
{
    /* We send a dummy byte as the message payload — SCM_RIGHTS requires
       at least 1 byte of normal data alongside the ancillary data. */
    char dummy = 'F';
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    /* Ancillary data buffer for the fds */
    size_t cmsg_space = CMSG_SPACE(count * sizeof(int));
    char *cmsg_buf = (char *)alloca(cmsg_space);
    memset(cmsg_buf, 0, cmsg_space);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = cmsg_space;

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(count * sizeof(int));
    memcpy(CMSG_DATA(cmsg), fds, count * sizeof(int));

    ssize_t ret = sendmsg(sock, &msg, 0);
    return (ret >= 0) ? 0 : -1;
}

/// Receive an array of file descriptors from a connected unix socket.
/// `fds` must point to an array of at least `count` ints.
/// Returns 0 on success, -1 on error (check errno).
static inline int
nvmm_recv_fds(int sock, int *fds, int count)
{
    char dummy;
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };

    size_t cmsg_space = CMSG_SPACE(count * sizeof(int));
    char *cmsg_buf = (char *)alloca(cmsg_space);
    memset(cmsg_buf, 0, cmsg_space);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = cmsg_space;

    ssize_t ret = recvmsg(sock, &msg, 0);
    if (ret < 0) return -1;

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        errno = EPROTO;
        return -1;
    }

    int received_count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    if (received_count != count) {
        /* Close any received fds to avoid leaks */
        int *recv_fds = (int *)CMSG_DATA(cmsg);
        for (int i = 0; i < received_count; i++)
            close(recv_fds[i]);
        errno = EPROTO;
        return -1;
    }

    memcpy(fds, CMSG_DATA(cmsg), count * sizeof(int));
    return 0;
}

/// Create a unix domain socket server, bind and listen.
/// Returns the listening socket fd, or -1 on error.
static inline int
nvmm_server_listen(const char *path)
{
    /* Remove stale socket */
    unlink(path);

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    size_t path_len = strnlen(path, sizeof(addr.sun_path));
    if (path_len >= sizeof(addr.sun_path)) {
        close(sock);
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(addr.sun_path, path, path_len);  /* null terminator already set by memset */

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    if (listen(sock, 8) < 0) {
        close(sock);
        unlink(path);
        return -1;
    }

    return sock;
}

/// Connect to a unix domain socket server.
/// Returns the connected socket fd, or -1 on error.
static inline int
nvmm_client_connect(const char *path)
{
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    size_t path_len = strnlen(path, sizeof(addr.sun_path));
    if (path_len >= sizeof(addr.sun_path)) {
        close(sock);
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(addr.sun_path, path, path_len);  /* null terminator already set by memset */

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    return sock;
}

#ifdef __cplusplus
}
#endif

#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int
nvmm_send_fds(int sock, const int *fds, int count)
{
    /// SCM_RIGHTS needs at least one byte of normal data next to the fds.
    char dummy = 'F';
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

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(count * sizeof(int));
    memcpy(CMSG_DATA(cmsg), fds, count * sizeof(int));

    ssize_t ret = sendmsg(sock, &msg, 0);
    return (ret >= 0) ? 0 : -1;
}

/// On a count mismatch the received fds are closed, so none leak.
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
        int *recv_fds = (int *)CMSG_DATA(cmsg);
        for (int i = 0; i < received_count; i++)
            close(recv_fds[i]);
        errno = EPROTO;
        return -1;
    }

    memcpy(fds, CMSG_DATA(cmsg), count * sizeof(int));
    return 0;
}

/// Unlinks any stale socket at `path` first.
static inline int
nvmm_server_listen(const char *path)
{
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
    memcpy(addr.sun_path, path, path_len);

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
    memcpy(addr.sun_path, path, path_len);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    return sock;
}

#ifdef __cplusplus
}
#endif

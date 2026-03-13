// thread_compat.h
#ifndef THREAD_COMPAT_H
#define THREAD_COMPAT_H

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <stdlib.h>

typedef HANDLE thread_handle_t;

// Wrapper data structure for converting calling conventions
struct thread_wrapper_data {
    void *(*func)(void *);
    void *arg;
};

// Wrapper function to convert calling convention
static unsigned __stdcall thread_wrapper(void *data) {
    struct thread_wrapper_data *wd = (struct thread_wrapper_data *)data;
    void *(*f)(void *) = wd->func;
    void *a = wd->arg;
    free(wd);
    f(a);
    return 0;
}

static inline int thread_create(thread_handle_t *thread, void *(*func)(void *), void *arg) {
    struct thread_wrapper_data *wd = malloc(sizeof(*wd));
    if (!wd) return -1;
    wd->func = func;
    wd->arg = arg;
    *thread = (HANDLE)_beginthreadex(NULL, 0, thread_wrapper, wd, 0, NULL);
    return (*thread == 0) ? -1 : 0;
}

static inline int thread_join(thread_handle_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#else
#include <pthread.h>

typedef pthread_t thread_handle_t;

#define THREAD_CALL

static inline int thread_create(thread_handle_t *thread, void *(*func)(void *), void *arg) {
    return pthread_create(thread, NULL, func, arg);
}

static inline int thread_join(thread_handle_t thread) { return pthread_join(thread, NULL); }

#endif

#endif  // THREAD_COMPAT_H


// Copyright (c) Simon Fraser University & The Chinese University of Hong Kong. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

#include <cstdint>
#include <iostream>

#ifdef PMEM
#include "libpmem.h"
#include "libpmemobj.h"
#endif

static constexpr const uint32_t k_cache_line_size = 64;

static bool FileExists(const char *pool_path) {
    struct stat buffer;
    return (stat(pool_path, &buffer) == 0);
}

#ifdef PMEM
#define CREATE_MODE_RW (S_IWUSR | S_IRUSR)
#endif

#define LOG_FATAL(msg)        \
    std::cout << msg << "\n"; \
    exit(-1)

#define LOG(msg) std::cout << msg << "\n"

#define CAS(_p, _u, _v)                                               \
    (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                                 __ATOMIC_ACQUIRE))

// ADD and SUB return the value after add or sub
#define ADD(_p, _v) (__atomic_add_fetch(_p, _v, __ATOMIC_SEQ_CST))
#define SUB(_p, _v) (__atomic_sub_fetch(_p, _v, __ATOMIC_SEQ_CST))
#define LOAD(_p) (__atomic_load_n(_p, __ATOMIC_SEQ_CST))
#define STORE(_p, _v) (__atomic_store_n(_p, _v, __ATOMIC_SEQ_CST))

#define SIMD 1
#define SIMD_CMP8(src, key)                                             \
    do {                                                                \
        const __m256i key_data = _mm256_set1_epi8(key);                 \
        __m256i seg_data =                                              \
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)); \
        __m256i rv_mask = _mm256_cmpeq_epi8(seg_data, key_data);        \
        mask = _mm256_movemask_epi8(rv_mask);                           \
    } while (0)

#define SSE_CMP8(src, key)                                           \
    do {                                                             \
        const __m128i key_data = _mm_set1_epi8(key);                 \
        __m128i seg_data =                                           \
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
        __m128i rv_mask = _mm_cmpeq_epi8(seg_data, key_data);        \
        mask = _mm_movemask_epi8(rv_mask);                           \
    } while (0)

#define CHECK_BIT(var, pos) ((((var) & (1 << pos)) > 0) ? (1) : (0))

inline void mfence(void) { asm volatile("mfence" ::: "memory"); }

int msleep(uint64_t msec) {
    struct timespec ts;
    int res;

    if (msec < 0) {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

class timer {
public:
    timer() {
        total.tv_sec = total.tv_usec = 0;
        diff.tv_sec = diff.tv_usec = 0;
    }

    double duration() {
        double duration;

        duration = (total.tv_sec) * 1000000.0;  // sec to us
        duration += (total.tv_usec);            // us

        return duration * 1000.0;  // ns
    }

    void start() {
        gettimeofday(&t1, NULL);
    }

    void end() {
        gettimeofday(&t2, NULL);
        timersub(&t2, &t1, &diff);
        timeradd(&diff, &total, &total);
    }

    void reset() {
        total.tv_sec = total.tv_usec = 0;
        diff.tv_sec = diff.tv_usec = 0;
    }

    timeval t1, t2, diff;
    timeval total;
};

class nsTimer {
public:
    struct timespec t1, t2;
    long long diff, total, count, abnormal, normal;

    nsTimer() { reset(); }
    void start() { clock_gettime(CLOCK_MONOTONIC, &t1); }
    long long end(bool flag = false) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        diff = (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);
        total += diff;
        count++;
        if (diff > 10000000)
            abnormal++;
        if (diff < 10000)
            normal++;
        return diff;
    }
    long long op_count() { return count; }
    void reset() { diff = total = count = 0; }
    long long duration() { return total; } // ns
    double avg() { return double(total) / count; } // ns
    double abnormal_rate() { return double(abnormal) / count; }
    double normal_rate() { return double(normal) / count; }
};

#pragma once

#include <stdlib.h>

#include <cmath>
#include <cstdint>

#define fingerprint_mask 0xfff0000000000000
#define len_mask 0x000f000000000000
#define addr_mask 0x0000ffffffffffff
#define overflow_mask 0xfff0000000000000
#define overflow_fingerprint_mask 0xfe00000000000000
#define overflow_valid_mask 0x0100000000000000
#define overflow_position_mask 0x00f0000000000000

#define fingerprint_shift 50  // 64 - 12 - 2 (not use the last two bits)
#define addr_shift 0
#define len_shift 48                   // 64 - 12 - 4
#define overflow_fingerprint_shift 55  // 64 - 7 - 2 (not use the last two bits)
#define overflow_position_shift 52     // 64 - 7 - 1 - 4

// 8 => 3, 9 => 4, 16 => 4
int upper_log(int n) {
    int origin = n;
    int count = 0;
    if (n == 1) {
        return 0;
    }
    while (n > 1) {
        n >>= 1;
        count++;
    }
    if (origin != std::pow(2, count)) {
        count++;
    }
    return count;
}

/* private */
inline void set_ptr(uint64_t *ptr_ptr, uint64_t param, uint64_t mask, uint64_t shift) {
    uint64_t clear_param = (param << shift) & mask;
    uint64_t clear_ptr = *ptr_ptr & ~mask;
    *ptr_ptr = clear_ptr | clear_param;
}

inline uint64_t get_info(uint64_t ptr, uint64_t mask, uint64_t shift) {
    return (ptr & mask) >> shift;
}

/* public */
inline uint64_t get_overflow_position(uint64_t ptr) {
    return get_info(ptr, overflow_position_mask, overflow_position_shift);
}

inline void set_overflow_fingerprint_position(uint64_t *ptr_ptr, uint64_t hash, uint64_t pos) {
    set_ptr(ptr_ptr, hash, overflow_fingerprint_mask, overflow_fingerprint_shift);
    set_ptr(ptr_ptr, pos, overflow_position_mask, overflow_position_shift);
    *ptr_ptr = *ptr_ptr | overflow_valid_mask;
}

inline void clear_overflow(uint64_t *ptr_ptr) {
    *ptr_ptr = *ptr_ptr & ~overflow_mask;
}

inline bool check_overflow_valid(uint64_t ptr) {
    return (uint64_t)(ptr & overflow_valid_mask) != 0;
}

inline bool match_overflow_fingerprint(uint64_t ptr, uint64_t hash) {
    if (!check_overflow_valid(ptr))
        return false;
    return (ptr & overflow_fingerprint_mask) == ((hash << overflow_fingerprint_shift) & overflow_fingerprint_mask);
}

inline uint64_t get_len(uint64_t ptr) {
    return 1UL << get_info(ptr, len_mask, len_shift);  // length = 2 ^ len
}

inline uint64_t get_addr(uint64_t ptr) {
    return get_info(ptr, addr_mask, addr_shift);
}

inline void set_len_addr(uint64_t *ptr_ptr, uint64_t length, uint64_t address) {
    uint64_t len = upper_log(length);
    set_ptr(ptr_ptr, len, len_mask, len_shift);
    set_ptr(ptr_ptr, address, addr_mask, addr_shift);
}

inline bool match_fingerprint(uint64_t ptr, uint64_t hash) {
    if (ptr == 0)
        return false;
    return (ptr & fingerprint_mask) == ((hash << fingerprint_shift) & fingerprint_mask);
}

inline void set_fingerprint(uint64_t *ptr_ptr, uint64_t hash) {
    set_ptr(ptr_ptr, hash, fingerprint_mask, fingerprint_shift);
}
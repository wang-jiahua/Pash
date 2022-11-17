#pragma once

#include<stdlib.h>
#include <cstdint>

#define MASK_FP 0xfff0000000000000
#define MASK_LEN 0x000f000000000000
#define MASK_ADDR 0x0000ffffffffffff
#define MASK_OF 0xfff0000000000000
#define MASK_OF_FP 0xfe00000000000000
#define MASK_OF_VALID 0x0100000000000000
#define MASK_OF_POS 0x00f0000000000000

#define SHIFT_FP 50 // 64 - 12 - 2 (not use the last two bits)
#define SHIFT_ADDR 0
#define SHIFT_LEN 48 // 64 - 12 - 4
#define SHIFT_OF_FP 55 // 64 - 7 - 2 (not use the last two bits)
#define SHIFT_OF_POS 52 // 64 - 7 - 1 - 4

// 8 => 3, 9 => 4, 16 => 4
int upper_log(int n)
{
    int origin = n, count = 0;
	if (n == 1)
		return 0;
		
	while (n > 1) {
		n = n >> 1;
		count++;
	}

    if (origin != std::pow(2, count))
        count++;
	return count; 
}

/* private */
inline void set_compound_pointer(uint64_t *pointer_p, uint64_t param, uint64_t mask, uint64_t shift) {
    uint64_t clear_param = (param << shift ) & mask;
    uint64_t clear_pointer = *pointer_p & ~mask;
    *pointer_p = clear_pointer | clear_param;
}

inline uint64_t get_compound_pointer(uint64_t pointer, uint64_t mask, uint64_t shift) {
    return (pointer & mask) >> shift;
}


/* public */
inline uint64_t get_pointer_of_pos(uint64_t pointer) {
    return get_compound_pointer(pointer, MASK_OF_POS, SHIFT_OF_POS);
}

inline void set_pointer_of_fp_pos(uint64_t *pointer_p, uint64_t hash, uint64_t pos) {
    set_compound_pointer(pointer_p, hash, MASK_OF_FP, SHIFT_OF_FP);
    set_compound_pointer(pointer_p, pos, MASK_OF_POS, SHIFT_OF_POS);
    *pointer_p = *pointer_p | MASK_OF_VALID; // set valid bit
}

inline void clear_pointer_of(uint64_t *pointer_p) {
    *pointer_p = *pointer_p & ~MASK_OF;
}

inline bool check_pointer_of_valid(uint64_t pointer) {
    uint64_t fp = pointer & MASK_OF_VALID;
    return (fp != 0);
}

inline bool match_pointer_of_fp(uint64_t pointer, uint64_t hash) {
    if (!check_pointer_of_valid(pointer))
        return false;
    return (pointer & MASK_OF_FP) == ((hash << SHIFT_OF_FP) & MASK_OF_FP);
}

inline uint64_t get_pointer_len(uint64_t pointer) {
    uint64_t len = get_compound_pointer(pointer, MASK_LEN, SHIFT_LEN);
    return ((uint64_t)1 << len); // length = 2 ^ len
}

inline uint64_t get_pointer_addr(uint64_t pointer) {
    return get_compound_pointer(pointer, MASK_ADDR, SHIFT_ADDR);
}

inline void set_pointer_len_addr(uint64_t *pointer_p, uint64_t length, uint64_t address) {
    uint64_t len = upper_log(length);
    set_compound_pointer(pointer_p, len, MASK_LEN, SHIFT_LEN);
    set_compound_pointer(pointer_p, address, MASK_ADDR, SHIFT_ADDR);
}

inline bool match_pointer_fp(uint64_t pointer, uint64_t hash) {
    if (pointer == 0)
        return false;
    return (pointer & MASK_FP) == ((hash << SHIFT_FP) & MASK_FP);
}

inline void set_pointer_fp(uint64_t *pointer_p, uint64_t hash) {
    set_compound_pointer(pointer_p, hash, MASK_FP, SHIFT_FP);
}
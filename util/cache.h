#pragma once

#include "x86intrin.h"

#define CACHE_ALIGN 64
#define FLUSH_SIZE 256

#define asm_clwb(addr)                                      \
	asm volatile(".byte 0x66; xsaveopt %0"                  \
				 : "+m"(*(volatile char *)addr));
#define asm_clflush(addr)					                \
({								                            \
	__asm__ __volatile__ ("clflush %0" : : "m"(*addr));	    \
})

#define asm_mfence()			                          	\
({						                                    \
	__asm__ __volatile__ ("mfence");	                    \
})

static void flush(void* addr, size_t len) {
	char* end = (char*)(addr) + len;
	char* ptr = (char *)((unsigned long)addr &~(CACHE_ALIGN-1));
	for (; ptr < end; ptr += CACHE_ALIGN)
		asm_clwb(ptr);
	asm_mfence();
}

static void asyn_flush(void* addr, size_t len) {
	char* end = (char*)(addr) + len;
	char* ptr = (char *)((unsigned long)addr &~(CACHE_ALIGN-1));
	for (; ptr < end; ptr += CACHE_ALIGN)
		asm_clwb(ptr);
	// asm_mfence();
}

static void clear_cache() {
  int size = 256 * 1024 * 1024;
  char *garbage = new char[size];
  for (int i = 0; i < size; ++i)
    garbage[i] = i;
  for (int i = 100; i < size; ++i)
    garbage[i] += garbage[i - 100];
  flush(garbage, size);
  delete[] garbage;
}

inline void prefetch(const void *ptr) {
    typedef struct {
        char x[CACHE_ALIGN];
    } cacheline_t;
    asm volatile("prefetcht0 %0" : : "m"(*(const cacheline_t *)ptr));
}

inline void prefetch_more(const void *ptr, int size) {
    for (int i = 0; i < size / CACHE_ALIGN; i++) {
        prefetch(ptr + CACHE_ALIGN * i);
    }
}


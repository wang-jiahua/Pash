#pragma once
#include <fcntl.h>
#include <garbage_list.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <thread>

#include "../util/cache.h"
#include "../util/pair.h"
#include "../util/utils.h"
#include "x86intrin.h"

// #define eADR
#define PAGE_SIZE 4096

struct Region {
    char* start_addr_;
    char* curr_addr_;
    char* flush_addr_;
    char* end_addr_;
    bool valid;

    Region() : valid(false) {}

    void reset_region(char* start, char* end) {
        start_addr_ = start;
        curr_addr_ = start;
        flush_addr_ = start;
        end_addr_ = end;
        valid = true;
    }

    char* allocate_block(size_t block_size, bool set_zero = false) {
        assert(curr_addr_ + block_size <= end_addr_);
        char* allocate = curr_addr_;
        curr_addr_ += block_size;
        if (curr_addr_ >= end_addr_)
            valid = false;
        if (set_zero)
            memset((void*)allocate, 0, block_size);
        return allocate;
    }

    void batch_flush() {
        size_t diff = curr_addr_ - flush_addr_;
        if (diff >= FLUSH_SIZE) {
            int line_num = diff / FLUSH_SIZE;
            flush(flush_addr_, FLUSH_SIZE * line_num);
            flush_addr_ += (FLUSH_SIZE * line_num);
        }
    }
};

const static int free_list_number = 8;  // from 8byte to 1K
const static size_t power_two[free_list_number] = {8, 16, 32, 64, 128, 256, 512, 1024};
const static size_t mem_pool_size = 1024ul * 1024ul * 1024ul * 20ul;
static Region global_pm_pool;
static Region global_mem_pool;
__thread Region* pm_block_lists;

// space alignment
inline char* get_next_line(char* curr, size_t align) {
    uint64_t remain = (uint64_t)curr % align;
    if (!remain)
        return curr;
    return curr + align - remain;
}

// find right block list for input size
inline size_t get_list_index(size_t size) {
    for (int i = 0; i < free_list_number; i++)
        if (power_two[i] >= size)
            return i;

    assert(false);
}

struct AAllocator {
public:
    size_t pool_size_;
    size_t thread_num_;
    static AAllocator* instance_;

    AAllocator(const char* pool_name, size_t pool_size, size_t thread_num)
        : pool_size_(pool_size), thread_num_(thread_num + 1) {
        /* initialize pm pool */
        LOG(pool_name);
        if (!FileExists(pool_name)) {
            LOG_FATAL("[ALLOCATOR] pm file does not exist!");
        }
        int fd = open(pool_name, O_RDWR);
        if (fd < 0) {
            LOG_FATAL("[ALLOCATOR] failed to open nvm file!");
        }
        if (ftruncate(fd, pool_size) < 0) {
            LOG_FATAL("[ALLOCATOR] failed to truncate file!");
        }
        char* pm_pool = (char*)mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        global_pm_pool.reset_region(pm_pool, pm_pool + pool_size_);
        LOG("[ALLOCATOR] pm pool ready");

        /* initialize mem pool */
        char* mem_pool = new char[mem_pool_size];
        memset(mem_pool, 0, mem_pool_size);
        global_mem_pool.reset_region(mem_pool, mem_pool + mem_pool_size);
        LOG("[ALLOCATOR] mem pool ready");
    }

    static void Initialize(const char* pool_name, size_t pool_size, uint64_t thread_num) {
        pm_block_lists = new Region[free_list_number];
        instance_ = new AAllocator(pool_name, pool_size, thread_num);
        printf("[ALLOCATOR] pm pool opened at: %p\n", global_pm_pool.start_addr_);
        printf("[ALLOCATOR] mem pool opened at: %p\n", global_mem_pool.start_addr_);
    }

    void Initialize_thread(int thread_id) {
        if (!pm_block_lists)
            pm_block_lists = new Region[free_list_number];
    }

    void Store_thread_status(int thread_id) {}

    static AAllocator* Get() { return instance_; }

    static void* GetRoot(size_t size) {
        size_t idx = get_list_index(size);
        return allocate_block_from_region(pm_block_lists[idx], power_two[idx]);
    }

    static void Allocate(void** ptr, uint32_t alignment, size_t size, int (*alloc_constr)(void* ptr, void* arg), void* arg) {
        size_t idx = get_list_index(size);
        *ptr = (void*)allocate_block_from_region(pm_block_lists[idx], power_two[idx], true);

        alloc_constr(*ptr, arg);
    }

    static void* Allocate_without_proc(size_t size) {
        size_t idx = get_list_index(size);
        return (void*)allocate_block_from_region(pm_block_lists[idx], power_two[idx]);
    }

    static void ZAllocate(void** ptr, uint32_t alignment, size_t size) {
        size_t idx = get_list_index(size);
        *ptr = (void*)allocate_block_from_region(pm_block_lists[idx], power_two[idx], true);
    }

    static void* ZAllocate(size_t size) {
        size_t idx = get_list_index(size);
        return (void*)allocate_block_from_region(pm_block_lists[idx], power_two[idx], true);
    }

    // directly fetch and add from global mem pool for dram allocation
    static void DAllocate(void** ptr, uint32_t alignment, size_t size, int (*alloc_constr)(void* ptr, void* arg), void* arg) {
        char* curr_addr = __sync_fetch_and_add(&(global_mem_pool.curr_addr_), size + alignment);
        if (curr_addr + size + alignment >= global_mem_pool.end_addr_) {
            LOG_FATAL("[ALLOCATOR] global mem pool has no more space!");
        }
        *ptr = get_next_line(curr_addr, alignment);

        alloc_constr(*ptr, arg);
    }

    static void Reclaim(size_t size) {}

    static Value_t Prepare_value(Value_t value, size_t value_size) {
        size_t idx = get_list_index(value_size);
        uint64_t* value_addr = (uint64_t*)allocate_block_from_region(pm_block_lists[idx], power_two[idx]);

        uint64_t value_number = uint64_t(value);
        for (int i = 0; i < value_size / sizeof(uint64_t); i++) {
            value_addr[i] = value_number;
        }

        pm_block_lists[idx].batch_flush();
        return (Value_t)value_addr;
    }

    static char* Prepare_key(string_key* key) {
        size_t idx = get_list_index(key->length);
        char* key_addr = (char*)allocate_block_from_region(pm_block_lists[idx], power_two[idx]);

        memcpy(key_addr, key->key, key->length);

        pm_block_lists[idx].batch_flush();
        return key_addr;
    }

    static string_key* Prepare_string_key(string_key* key) {
        size_t key_size = sizeof(string_key) + key->length;
        size_t idx = get_list_index(key->length);
        char* key_addr = (char*)allocate_block_from_region(pm_block_lists[idx], power_two[idx]);

        memcpy(key_addr, key, key_size);

        pm_block_lists[idx].batch_flush();
        return (string_key*)key_addr;
    }

    static void Persist(void* ptr, size_t size) {
#ifdef eADR
        mfence();
#else
        flush(ptr, size);
#endif
    }

    static void Persist_flush(void* ptr, size_t size) {
        flush(ptr, size);
    }

    static void Persist_asyn_flush(void* ptr, size_t size) {
        asyn_flush(ptr, size);
    }

    static void NTWrite64(uint64_t* ptr, uint64_t val) {
        _mm_stream_si64((long long*)ptr, val);
    }

    static void NTWrite32(uint32_t* ptr, uint32_t val) {
        _mm_stream_si32((int*)ptr, val);
    }

    static uint64_t total_pm_alloc() {
        return global_pm_pool.curr_addr_ - global_pm_pool.start_addr_;
    }

private:
    static void fetch_page_from_center(Region& block_list, size_t page_size = PAGE_SIZE) {
        char* curr_addr = __sync_fetch_and_add(&(global_pm_pool.curr_addr_), page_size);
        if (curr_addr + page_size >= global_pm_pool.end_addr_) {
            LOG_FATAL("[ALLOCATOR] global pm pool has no more space!");
        }
        block_list.reset_region(curr_addr, curr_addr + page_size);
        assert((size_t)curr_addr % page_size == 0);
    }

    static char* allocate_block_from_region(Region& block_list, size_t block_size, bool set_zero = false) {
        if (!block_list.valid) {
            fetch_page_from_center(block_list);
        }
        return block_list.allocate_block(block_size, set_zero);
    }
};

AAllocator* AAllocator::instance_ = nullptr;

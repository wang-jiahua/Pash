// Copyright (c) Simon Fraser University & The Chinese University of Hong Kong.
// All rights reserved. Licensed under the MIT license.
#include <gflags/gflags.h>
#include <immintrin.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "../util/System.hpp"
#include "../util/key_generator.hpp"
#include "../util/uniform.hpp"
// #include "./pash/pash.h"
#include "./pash/pash_variable_key_value.h"
#include "Hash.h"
#include "libpmemobj.h"

uint64_t pm_count_records[4] = {0, 0, 0, 0};
const std::string pm_count_cmds[4] = {"MediaReads", "MediaWrites", "ReadRequests", "WriteRequests"};

std::string get_cmd(int cmd_index) { return "sudo ipmctl show -dimm -performance " + pm_count_cmds[cmd_index]; }

std::string get_flag(int cmd_index) { return pm_count_cmds[cmd_index] + "=0x"; }

static void pm_count(bool end = false, uint64_t operation_num = 20000000) {
    mfence();
    char line[1024];
    FILE *fp;
    for (int i = 0; i < 4; i++) {
        std::string curr_cmd = get_cmd(i);
        std::string curr_flag = get_flag(i);
        uint64_t count_num = 0;
        if ((fp = popen(curr_cmd.data(), "r")) == NULL) {
            printf("[PM COUNTER] error\n");
            exit(0);
        }
        while (fgets(line, sizeof(line) - 1, fp) != NULL) {
            std::string line_str = line;
            uint64_t start_pos = line_str.find(curr_flag);
            if (start_pos != std::string::npos) {
                char *tmp;
                std::string xxx = get_flag(i);
                uint64_t i = strtol(line_str.substr(start_pos + curr_flag.size(), line_str.size()).c_str(), &tmp, 16);
                count_num += i;
            }
        }
        pclose(fp);
        // check to print counting results since last time
        if (end) {
            // printf("[PM COUNTER] %s: total = %lu, average = %.3f\n",
            //    pm_count_cmds[i].c_str(), (count_num - pm_count_records[i]),
            //    double(count_num - pm_count_records[i]) / operation_num);
        }
        pm_count_records[i] = count_num;
    }
}

__thread nsTimer *clk;

__thread Key_t **cand_keys;
__thread Value_t **cand_values;
__thread void **cand_segments;
__thread int *cand_nums;
__thread int **cand_slots;

__thread Key_t **sec_cand_keys;
__thread Value_t **sec_cand_values;
__thread void **sec_cand_segments;
__thread int *sec_cand_nums;
__thread int **sec_cand_slots;

#ifdef LAT_COUNT
std::shared_mutex lat_mutex;
#endif

#define PMEM1

#ifdef PMEM1
std::string pool_name = "/mnt/pmem1/";
#else
std::string pool_name = "/mnt/pmem0/";
#endif

DEFINE_string(index, "dash-ex", "hash index: dash-ex/dash-lh/cceh/level");
DEFINE_string(k, "variable", "key type: fixed/variable");
DEFINE_string(distribution, "uniform", "workload distribution: uniform/skew");
DEFINE_uint64(i, 8192, "initial number of directory entries");
DEFINE_uint64(t, 1, "number of concurrent threads");
DEFINE_uint64(n, 0, "number of pre-load");
DEFINE_uint64(loadType, 0, "type of pre-load: 0(random)/1(range)");
DEFINE_uint64(p, 20000000, "number of insert/search/deletion operations");
DEFINE_uint64(u, 0, "number of update operations");
DEFINE_string(op, "full", "operation type: insert/pos/neg/delete/mixed/insert-load/skew-count");
DEFINE_double(r, 0, "read ratio for mixed workload: 0~1.0");
DEFINE_double(s, 1, "insert ratio for mixed workload: 0~1.0");
DEFINE_double(d, 0, "delete ratio for mixed workload: 0~1.0");
DEFINE_double(skew, 0.99, "skew factor of workload");
DEFINE_uint32(e, 0, "whether register epoch in application level: 0/1");
DEFINE_uint32(ms, 100, "miliseconds to sample the operations");
DEFINE_uint64(ps, 200ul, "size of memory pool (GB)");
DEFINE_uint64(ed, 1000, "frequency to enroll into the epoch");
DEFINE_uint64(vkl, 8, "length of the variable length key");
DEFINE_uint64(vvl, 8, "length of variable length value");
DEFINE_uint64(hb, 13, "hot bits");
DEFINE_uint64(bs, 4, "batch size");

uint64_t init_cap, thread_num, load_num, operation_num;
std::string operation;
std::string distribution;
std::string key_type;
std::string index_type;
int bar_a, bar_b, bar_c;
double read_ratio, insert_ratio, delete_ratio, skew_factor;
std::mutex mtx;
std::condition_variable cv;
bool finished = false;
bool open_epoch;
uint32_t msec;
struct timeval tv1, tv2, tv3;
size_t pool_size = 1024ul * 1024ul * 1024ul * 100ul;
key_generator_t *uniform_generator;
uint64_t EPOCH_DURATION;
uint64_t load_type = 0;
uint64_t key_length = 8;
uint64_t value_length = 8;
uint64_t kv_length;
uint64_t hot_bit = 13;
uint64_t hot_num = 1 << hot_bit;
uint64_t asso = 2;
uint64_t update_retry_time = 4;
uint64_t update_num = 0;
uint64_t batch_size = 4;

struct operation_record_t {
    uint64_t number;
    uint64_t dummy[7]; /*patch to a cacheline size, avoid false sharing*/
};

operation_record_t operation_record[1024];

struct Range {
    int thread_id_;
    uint64_t begin;
    uint64_t end;
    int length;  // in case of variable key length, use `length` to indicate the length of key
    void *workload;
    uint64_t random_num;
    struct timeval tv;
};

void set_affinity(uint32_t idx) {
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
#ifdef PMEM1
    CPU_SET((idx * 2 + 1) % 112, &my_set);
#else
    CPU_SET((idx * 2) % 112, &my_set);
#endif
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

void init_batch_utils() {
    // init thread local candidate slots
    cand_keys = new Key_t *[batch_size];
    cand_values = new Value_t *[batch_size];
    cand_nums = new int[batch_size];
    cand_segments = new void *[batch_size];
    cand_slots = new int *[batch_size];
    for (int i = 0; i < batch_size; i++) {
        cand_keys[i] = new Key_t[8];
        cand_values[i] = new Value_t[8];
        cand_nums[i] = -1;
        cand_slots[i] = new int[8];
    }
    sec_cand_keys = new Key_t *[batch_size];
    sec_cand_values = new Value_t *[batch_size];
    sec_cand_nums = new int[batch_size];
    sec_cand_segments = new void *[batch_size];
    sec_cand_slots = new int *[batch_size];
    for (int i = 0; i < batch_size; i++) {
        sec_cand_keys[i] = new Key_t[8];
        sec_cand_values[i] = new Value_t[8];
        sec_cand_nums[i] = -1;
        sec_cand_slots[i] = new int[8];
    }
}

template <class T>
Hash<T> *InitializeIndex(int init_cap) {
    Hash<T> *index;
    bool file_exist = false;
    gettimeofday(&tv1, NULL);
    std::cout << "Initialize Pash" << std::endl;
    std::string index_pool_name = pool_name + "pmem_zhash.data";
    if (FileExists(index_pool_name.c_str()))
        file_exist = true;
    AAllocator::Initialize(index_pool_name.c_str(), pool_size, thread_num);
    index = reinterpret_cast<Hash<T> *>(AAllocator::GetRoot(sizeof(zhash::ZHASH<T>)));
    new (index) zhash::ZHASH<T>(init_cap);
    return index;
}

// generate 8-byte numbers and store them in the memory_region
void generate_8B(void *memory_region, uint64_t generate_num, key_generator_t *key_generator, uint64_t mod_range = -1) {
    uint64_t *keys = reinterpret_cast<uint64_t *>(memory_region);
    for (uint64_t i = 0; i < generate_num; ++i) {
        keys[i] = key_generator->next_uint64();
        if (mod_range != -1) {
            keys[i] = keys[i] % mod_range + 1;
        }
    }
}

// generate 16-byte strings and store them in the memory_region
void generate_16B(void *memory_region, uint64_t generate_num, key_generator_t *key_generator, uint64_t mod_range = -1) {
    string_key *var_key;
    uint64_t *_key;
    uint64_t random_num;
    char *workload = reinterpret_cast<char *>(memory_region);
    int word_num = (key_length / 8) + (((key_length % 8) != 0) ? 1 : 0);
    _key = reinterpret_cast<uint64_t *>(malloc(word_num * sizeof(uint64_t)));
    for (uint64_t i = 0; i < generate_num; ++i) {
        var_key = reinterpret_cast<string_key *>(workload + i * (key_length + sizeof(string_key)));
        var_key->length = key_length;
        random_num = key_generator->next_uint64();
        if (mod_range != -1) {
            random_num = random_num % mod_range + 1;
        }
        for (int j = 0; j < word_num; ++j) {
            _key[j] = random_num;
        }
        memcpy(var_key->key, _key, key_length);
    }
}

template <class T>
void Load(int kv_num, Hash<T> *index, int length, void *workload) {
    std::cout << "Start load warm-up workload" << std::endl;
    if (kv_num == 0) {
        return;
    }
    std::string fixed("fixed");
    T *_worklod = reinterpret_cast<T *>(workload);
    T key;
    if constexpr (!std::is_pointer_v<T>) {
        for (uint64_t i = 0; i < kv_num; ++i) {
            index->Insert(_worklod[i], DEFAULT);
        }
    } else { /*genereate 16B key*/
        char *persist_workload = reinterpret_cast<char *>(workload);
        int string_key_size = sizeof(string_key) + length;
        for (uint64_t i = 0; i < kv_num; ++i) {
            key = reinterpret_cast<T>(persist_workload + i * string_key_size);
            index->Insert(key, DEFAULT);
        }
    }
    std::cout << "Finish loading " << kv_num << " keys" << std::endl;
}

inline void spin_wait() {
    SUB(&bar_b, 1);
    while (LOAD(&bar_a) == 1)
        ; /*spinning*/
}

inline void end_notify(Range *range) {
    gettimeofday(&range->tv, NULL);
    if (SUB(&bar_c, 1) == 0) {
        std::unique_lock<std::mutex> lck(mtx);
        finished = true;
        cv.notify_one();
    }
}

inline void end_sub() { SUB(&bar_c, 1); }

template <class T>
void concurr_insert_without_epoch(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
#if defined(BREAKDOWN) || defined(LAT_COUNT)
    clk = new nsTimer[10];
    long long lat;
#endif
#ifdef LAT_COUNT
    // need to change the return value of Insert() when split occurs
    int *insert_lat = new int[1500]();
    int *split_lat = new int[1500]();
#endif
#ifdef VALUE_LENGTH_VARIABLE
    init_batch_utils();
#endif
    T batch_keys[batch_size];
    nsTimer *batch_clks = new nsTimer[batch_size];
    spin_wait();
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        for (uint64_t i = begin; i < end; ++i) {
#if defined(BREAKDOWN) || defined(LAT_COUNT)
            if (clk)
                clk[1].start();
#endif
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = keys[i + j];
            else {
                if (i / batch_size * batch_size < begin) {
                    batch_clks[batch_offset].start();
                }
            }
            Value_t value = reinterpret_cast<Value_t>(keys[i] + i);
            int ret = index->Insert(keys[i], value, batch_offset, batch_keys, batch_clks);
            if (ret != 0) {
                printf("concurr_insert_without_epoch key: %p fail\n", keys[i]);
            }
#if defined(BREAKDOWN) || defined(LAT_COUNT)
            if (clk)
                lat = clk[1].end();
#endif
#ifdef LAT_COUNT
            if (ret == 0)
                insert_lat[lat / 1000]++;
            else if (ret == 1)
                split_lat[lat / 1000]++;
#endif
            batch_clks[batch_offset].end();
            mfence();
        }
    } else {
        T var_key;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = reinterpret_cast<T>(workload + string_key_size * (i + j));
            var_key = reinterpret_cast<T>(workload + string_key_size * i);
            Value_t value = reinterpret_cast<Value_t>(i);
            index->Insert(var_key, value, batch_offset, batch_keys);
            mfence();
        }
    }
    end_notify(range);
#ifdef LAT_COUNT
    lat_mutex.lock();
    for (int i = 0; i < 1500; i++)
        printf("thread %d insert_lat[%d]: %d\n", range->thread_id_, i, insert_lat[i]);
    for (int i = 0; i < 1500; i++)
        printf("thread %d split_lat[%d]: %d\n", range->thread_id_, i, split_lat[i]);
    lat_mutex.unlock();
#endif
    long long total_count = 0, total_dura = 0;
    for (int i = 0; i < batch_size; i++) {
        // printf("[BATCH LATENCY] Ops %d count %lld, mean time %lf ns\n", i, batch_clks[i].count, batch_clks[i].avg());
        total_count += batch_clks[i].count;
        total_dura += batch_clks[i].total;
    }
    // printf("[BATCH LATENCY] Average total count %lld, average mean time %lf ns\n", total_count, double(total_dura) / total_count);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_update_without_epoch(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
#ifdef VALUE_LENGTH_VARIABLE
    init_batch_utils();
#endif
    T batch_keys[batch_size];
    nsTimer *batch_clks = new nsTimer[batch_size];
    spin_wait();
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = keys[i + j];
            else {
                if (i / batch_size * batch_size < begin) {
                    batch_clks[batch_offset].start();
                }
            }
            Value_t value = reinterpret_cast<Value_t>(keys[i] + i);
            int ret = index->Update(keys[i], value, batch_offset, batch_keys, batch_clks);
            if (ret == 0) {
                printf("concurr_update_without_epoch key: %p success\n", keys[i]);
            }
            batch_clks[batch_offset].end();
            mfence();
        }
    } else {
        T var_key;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = reinterpret_cast<T>(workload + string_key_size * (i + j));
            var_key = reinterpret_cast<T>(workload + string_key_size * i);
            Value_t value = reinterpret_cast<Value_t>(i);
            index->Update(var_key, value, batch_offset, batch_keys);
            mfence();
        }
    }
    end_notify(range);
    long long total_count = 0, total_dura = 0;
    for (int i = 0; i < batch_size; i++) {
        // printf("[BATCH LATENCY] Ops %d count %lld, mean time %lf ns\n", i, batch_clks[i].count, batch_clks[i].avg());
        total_count += batch_clks[i].count;
        total_dura += batch_clks[i].total;
    }
    // printf("[BATCH LATENCY] Average total count %lld, average mean time %lf ns\n", total_count, double(total_dura) / total_count);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_insert(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T key;
    uint64_t repeat_key = 0;
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                index->Insert(keys[j], DEFAULT, true);
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                index->Insert(keys[i], DEFAULT, true);
            }
        }
    } else {
        T var_key;
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                var_key = reinterpret_cast<T>(workload + string_key_size * j);
                index->Insert(var_key, DEFAULT, true);
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                var_key = reinterpret_cast<T>(workload + string_key_size * i);
                index->Insert(var_key, DEFAULT, true);
            }
        }
    }
    end_notify(range);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

/*Insert benchmark for load factor test only */
template <class T>
void insert_load_factor(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T key;
    spin_wait();
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        for (uint64_t i = begin; i < end; ++i) {
            index->Insert(keys[i], DEFAULT);
            if (i % 100000 == 0) {
                index->getNumber();
            }
        }
    } else {
        T var_key;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        for (uint64_t i = begin; i < end; ++i) {
            var_key = reinterpret_cast<T>(workload + string_key_size * i);
            index->Insert(var_key, DEFAULT);
            if (i % 100000 == 0) {
                index->getNumber();
            }
        }
    }
    end_notify(range);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

/*In this search version, the thread also needs to do the record its */
template <class T>
void concurr_search_sample(Range *range, Hash<T> *index) {
    uint64_t thread_id = range->thread_id_;
    set_affinity(thread_id);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    operation_record[thread_id].number = 0;
    uint64_t begin = range->begin;
    uint64_t end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    uint64_t not_found = 0;
    Value_t value;
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                index->Get(keys[j], &value, true);
                operation_record[thread_id].number++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                index->Get(keys[i], &value, true);
                operation_record[thread_id].number++;
            }
        }
    } else {
        T var_key;
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                var_key = reinterpret_cast<T>(workload + string_key_size * j);
                index->Get(var_key, &value, true);
                operation_record[thread_id].number++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                var_key = reinterpret_cast<T>(workload + string_key_size * i);
                index->Get(var_key, &value, true);
                operation_record[thread_id].number++;
            }
        }
    }
    // std::cout << "not_found = " << not_found << std::endl;
    end_sub();
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_insert_sample(Range *range, Hash<T> *index) {
    uint64_t curr_index = range->thread_id_;
    set_affinity(curr_index);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    operation_record[curr_index].number = 0;
    uint64_t begin = range->begin;
    uint64_t end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T key;
    uint64_t not_found = 0;
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                index->Insert(keys[j], DEFAULT, true);
                operation_record[curr_index].number++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                index->Insert(keys[i], DEFAULT, true);
                operation_record[curr_index].number++;
            }
        }
    } else {
        T var_key;
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                var_key = reinterpret_cast<T>(workload + string_key_size * j);
                index->Insert(var_key, DEFAULT, true);
                operation_record[curr_index].number++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                var_key = reinterpret_cast<T>(workload + string_key_size * i);
                index->Insert(var_key, DEFAULT, true);
                operation_record[curr_index].number++;
            }
        }
    }
    end_sub();
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_search(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    uint64_t begin = range->begin;
    uint64_t end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    uint64_t not_found = 0;
    Value_t value;
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                if (!index->Get(keys[j], &value, true))
                    not_found++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                if (!index->Get(keys[i], &value, true))
                    not_found++;
            }
        }
    } else {
        T var_key;
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                var_key = reinterpret_cast<T>(workload + string_key_size * j);
                if (!index->Get(var_key, &value, true))
                    not_found++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                var_key = reinterpret_cast<T>(workload + string_key_size * i);
                if (!index->Get(var_key, &value, true))
                    not_found++;
            }
        }
    }
    end_notify(range);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_search_without_epoch(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    uint64_t not_found = 0;
    char *value = new char[value_length];
#ifdef BREAKDOWN
    clk = new nsTimer[10];
#endif
    clk = new nsTimer[10];
#ifdef VALUE_LENGTH_VARIABLE
    init_batch_utils();
#endif
    nsTimer *batch_clks = new nsTimer[batch_size];
    spin_wait();
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        T batch_keys[batch_size];
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0) {
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++) {
                    batch_keys[j] = keys[i + j];
                }
            } else {
                if (i / batch_size * batch_size < begin) {
                    batch_clks[batch_offset].start();
                }
            }
            if (!index->Get(keys[i], (Value_t *)value, batch_offset, batch_keys, batch_clks)) {
                not_found++;
            }
            batch_clks[batch_offset].end();
            mfence();
        }
    } else {
        T var_key;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        T batch_keys[batch_size];
        for (uint64_t i = begin; i < end; ++i) {
            var_key = reinterpret_cast<T>(workload + string_key_size * i);
            int batch_offset = i % batch_size;
            if (batch_offset == 0) {
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++) {
                    batch_keys[j] = reinterpret_cast<T>(workload + string_key_size * (i + j));
                }
            }
            if (!index->Get(var_key, (Value_t *)value, batch_offset, batch_keys)) {
                not_found++;
            }
            mfence();
        }
    }
    // std::cout << "not_found = " << not_found << std::endl;
    end_notify(range);
#ifdef BREAKDOWN
    printf("[BREAKDOWN] hash time: %lf us\n", double(clk[0].duration()) / (end - begin));
    printf("[BREAKDOWN] find segment time: %lf us\n", double(clk[1].duration()) / (end - begin));
    printf("[BREAKDOWN] scan slots time: %lf us\n", double(clk[2].duration()) / (end - begin));
#endif
    long long total_count = 0, total_dura = 0;
    for (int i = 0; i < batch_size; i++) {
        // printf("[BATCH LATENCY] Ops %d count %lld, mean time %lf ns\n", i, batch_clks[i].count, batch_clks[i].avg());
        total_count += batch_clks[i].count;
        total_dura += batch_clks[i].total;
    }
    // printf("[BATCH LATENCY] Average total count %lld, average mean time %lf ns\n", total_count, double(total_dura) / total_count);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_delete_without_epoch(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    uint64_t not_found = 0;
#ifdef VALUE_LENGTH_VARIABLE
    init_batch_utils();
#endif
    T batch_keys[batch_size];
    nsTimer *batch_clks = new nsTimer[batch_size];
    spin_wait();
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = keys[i + j];
            else {
                if (i / batch_size * batch_size < begin) {
                    batch_clks[batch_offset].start();
                }
            }
            if (!index->Delete(keys[i], batch_offset, batch_keys, batch_clks)) {
                printf("concurr_delete_without_epoch key: %p fail\n", keys[i]);
                not_found++;
            }
            batch_clks[batch_offset].end();
            mfence();
        }
    } else {
        T var_key;
        int string_key_size = sizeof(string_key) + range->length;
        for (uint64_t i = begin; i < end; ++i) {
            int batch_offset = i % batch_size;
            if (batch_offset == 0)
                for (unsigned j = 0; j < batch_size && (i + j) < end; j++)
                    batch_keys[j] = reinterpret_cast<T>(workload + string_key_size * (i + j));
            var_key = reinterpret_cast<T>(workload + string_key_size * i);
            if (!index->Delete(var_key, batch_offset, batch_keys)) {
                not_found++;
            }
        }
    }
    end_notify(range);
    long long total_count = 0, total_dura = 0;
    for (int i = 0; i < batch_size; i++) {
        // printf("[BATCH LATENCY] Ops %d count %lld, mean time %lf ns\n", i, batch_clks[i].count, batch_clks[i].avg());
        total_count += batch_clks[i].count;
        total_dura += batch_clks[i].total;
    }
    // printf("[BATCH LATENCY] Average total count %lld, average mean time %lf ns\n", total_count, double(total_dura) / total_count);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void concurr_delete(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    int begin = range->begin;
    int end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T key;
    uint64_t not_found = 0;
    if constexpr (!std::is_pointer_v<T>) {
        T *keys = reinterpret_cast<T *>(workload);
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                if (!index->Delete(keys[j], true))
                    not_found++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                if (!index->Delete(keys[i], true))
                    not_found++;
            }
        }
    } else {
        T var_key;
        uint64_t round = (end - begin) / EPOCH_DURATION;
        uint64_t i = 0;
        uint64_t string_key_size = sizeof(string_key) + range->length;
        spin_wait();
        while (i < round) {
            uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
            for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
                var_key = reinterpret_cast<T>(workload + string_key_size * j);
                if (!index->Delete(var_key, true))
                    not_found++;
            }
            ++i;
        }
        {
            for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
                var_key = reinterpret_cast<T>(workload + string_key_size * i);
                if (!index->Delete(var_key, true))
                    not_found++;
            }
        }
    }
    // std::cout << "not_found = " << not_found << std::endl;
    end_notify(range);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void mixed_without_epoch(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    uint64_t begin = range->begin;
    uint64_t end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T *keys = reinterpret_cast<T *>(range->workload);
    T key;
    int string_key_size = sizeof(string_key) + range->length;
    UniformRandom rng(range->random_num);
    uint32_t random;
    uint32_t not_found = 0;
    uint64_t ins_num = 0;
    uint32_t insert_sign = (uint32_t)(insert_ratio * 100);
    uint32_t read_sign = (uint32_t)(read_ratio * 100) + insert_sign;
    uint32_t delete_sign = (uint32_t)(delete_ratio * 100) + read_sign;
    char *value = new char[value_length];
    clk = new nsTimer[10];
#ifdef VALUE_LENGTH_VARIABLE
    init_batch_utils();
#endif
    T batch_keys[batch_size];
    spin_wait();
    for (uint64_t i = begin; i < end; ++i) {
        if constexpr (std::is_pointer_v<T>) { /* variable length*/
            key = reinterpret_cast<T>(workload + string_key_size * i);
        } else {
            key = keys[i];
        }
        // init batch
        int batch_offset = i % batch_size;
        if (batch_offset == 0) {
            for (unsigned j = 0; j < batch_size && (i + j) < end; j++) {
                if constexpr (std::is_pointer_v<T>) {
                    batch_keys[j] = reinterpret_cast<T>(workload + string_key_size * (i + j));
                } else {
                    batch_keys[j] = keys[i + j];
                }
            }
        }
        random = rng.next_uint32() % 100;
        if (random < insert_sign) { /*insert*/
            index->Insert(key, DEFAULT, batch_offset, batch_keys);
            ins_num++;
        } else if (random < read_sign) { /*get*/
            if (!index->Get(key, (Value_t *)value, batch_offset, batch_keys)) {
                not_found++;
            }
        } else { /*delete*/
            index->Delete(key, batch_offset, batch_keys);
        }
        mfence();
    }
    end_notify(range);
    // std::cout << "not_found = " << not_found << " ins/total = " << ins_num << "/" << end - begin << std::endl;
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void mixed(Range *range, Hash<T> *index) {
    set_affinity(range->thread_id_);
    AAllocator::Get()->Initialize_thread(range->thread_id_);
    uint64_t begin = range->begin;
    uint64_t end = range->end;
    char *workload = reinterpret_cast<char *>(range->workload);
    T *keys = reinterpret_cast<T *>(range->workload);
    T key;
    int string_key_size = sizeof(string_key) + range->length;
    UniformRandom rng(range->random_num);
    uint32_t random;
    uint64_t not_found = 0;
    uint32_t insert_sign = (uint32_t)(insert_ratio * 100);
    uint32_t read_sign = (uint32_t)(read_ratio * 100) + insert_sign;
    uint32_t delete_sign = (uint32_t)(delete_ratio * 100) + read_sign;
    uint64_t round = (end - begin) / EPOCH_DURATION;
    uint64_t i = 0;
    Value_t value;
    spin_wait();
    while (i < round) {
        uint64_t _end = begin + (i + 1) * EPOCH_DURATION;
        for (uint64_t j = begin + i * EPOCH_DURATION; j < _end; ++j) {
            if constexpr (std::is_pointer_v<T>) { /* variable length*/
                key = reinterpret_cast<T>(workload + string_key_size * j);
            } else {
                key = keys[j];
            }
            random = rng.next_uint32() % 100;
            if (random < insert_sign) { /*insert*/
                index->Insert(key, DEFAULT, true);
            } else if (random < read_sign) { /*get*/
                if (!index->Get(key, &value, true)) {
                    not_found++;
                }
            } else { /*delete*/
                index->Delete(key, true);
            }
        }
        ++i;
    }
    {
        for (i = begin + EPOCH_DURATION * round; i < end; ++i) {
            if constexpr (std::is_pointer_v<T>) { /* variable length*/
                key = reinterpret_cast<T>(workload + string_key_size * i);
            } else {
                key = keys[i];
            }
            random = rng.next_uint32() % 100;
            if (random < insert_sign) { /*insert*/
                index->Insert(key, DEFAULT, true);
            } else if (random < read_sign) { /*get*/
                if (!index->Get(key, &value, true)) {
                    not_found++;
                }
            } else { /*delete*/
                index->Delete(key, true);
            }
        }
    }
    // std::cout << "not_found = " << not_found << std::endl;
    /*the last thread notify the main thread to wake up*/
    end_notify(range);
    AAllocator::Get()->Store_thread_status(range->thread_id_);
}

template <class T>
void GeneralBench(Range *ranges, Hash<T> *index, int thread_num, uint64_t operation_num, std::string profile_name, void (*test_func)(Range *, Hash<T> *)) {
    // clear cache after warm up
    clear_cache();
    std::thread *thread_array[1024];
    profile_name = profile_name + std::to_string(thread_num);
    double duration;
    finished = false;
    bar_a = 1;
    bar_b = thread_num;
    bar_c = thread_num;
    std::cout << profile_name << " Begin" << std::endl;
    //  System::profile(profile_name, [&]() {
    for (uint64_t i = 0; i < thread_num; ++i) {
        thread_array[i] = new std::thread(*test_func, &ranges[i], index);
    }
    while (LOAD(&bar_b) != 0)
        ;                                   // Spin
    std::unique_lock<std::mutex> lck(mtx);  // get the lock of condition variable
    gettimeofday(&tv1, NULL);
    STORE(&bar_a, 0);  // start test
    while (!finished) {
        cv.wait(lck);  // go to sleep and wait for the wake-up from child threads
    }
    gettimeofday(&tv2, NULL);  // test end
    for (int i = 0; i < thread_num; ++i) {
        thread_array[i]->join();
        delete thread_array[i];
    }
    double longest = (double)(ranges[0].tv.tv_usec - tv1.tv_usec) / 1000000 + (double)(ranges[0].tv.tv_sec - tv1.tv_sec);
    double shortest = longest;
    duration = longest;
    for (int i = 1; i < thread_num; ++i) {
        double interval = (double)(ranges[i].tv.tv_usec - tv1.tv_usec) / 1000000 + (double)(ranges[i].tv.tv_sec - tv1.tv_sec);
        duration += interval;
        if (shortest > interval)
            shortest = interval;
        if (longest < interval)
            longest = interval;
    }
    // std::cout << "The time difference is " << longest - shortest << std::endl;
    // duration = duration / thread_num;
    // printf("%d threads, Time = %f s, throughput = %f ops/s, latency = %f us, fastest = %f, slowest = %f\n",
    //        thread_num, duration, operation_num / duration, (duration * 1000000) / operation_num, operation_num / shortest,
    //        operation_num / longest);
    std::cout << profile_name << " End" << std::endl;
}

template <class T>
void RecoveryBench(Range *ranges, Hash<T> *index, int thread_num, uint64_t operation_num, std::string profile_name) {
    std::thread *thread_array[1024];
    profile_name = profile_name + std::to_string(thread_num);
    double duration;
    finished = false;
    bar_a = 1;
    bar_b = thread_num;
    bar_c = thread_num;
    uint64_t *last_record = new uint64_t[thread_num];
    uint64_t *curr_record = new uint64_t[thread_num];
    memset(last_record, 0, sizeof(uint64_t) * thread_num);
    memset(curr_record, 0, sizeof(uint64_t) * thread_num);
    double seconds = (double)msec / 1000;
    std::cout << profile_name << " Begin" << std::endl;
    for (uint64_t i = 0; i < thread_num; ++i) {
        thread_array[i] = new std::thread(concurr_search_sample<T>, &ranges[i], index);
    }
    while (LOAD(&bar_b) != 0)
        ;  // Spin
    gettimeofday(&tv1, NULL);
    STORE(&bar_a, 0);  // start test
    /* Start to do the sampling and record in the file*/
    while (bar_c != 0) {
        msleep(msec);
        for (int i = 0; i < thread_num; ++i) {
            curr_record[i] = operation_record[i].number;
        }
        uint64_t operation_num = 0;
        for (int i = 0; i < thread_num; ++i) {
            operation_num += (curr_record[i] - last_record[i]);
        }
        double throughput = (double)operation_num / (double)1000000 / seconds;
        std::cout << throughput << std::endl; /*Mops/s*/
        memcpy(last_record, curr_record, sizeof(uint64_t) * thread_num);
    }
    gettimeofday(&tv2, NULL);  // test end
    for (int i = 0; i < thread_num; ++i) {
        thread_array[i]->join();
        delete thread_array[i];
    }
    duration = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
    printf("%d threads, Time = %f s, Total throughput = %f ops/s\n",
           thread_num, (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec),
           operation_num / duration);
    std::cout << profile_name << " End" << std::endl;
}

void *GenerateInsRangePosUniformWorkload(uint64_t load_num, uint64_t insert_num, uint64_t pos_search_num, uint64_t neg_search_num) {
    char *workload;
    void (*generate_NB)(void *memory_region, uint64_t generate_num, key_generator_t *key_generator, uint64_t mod_range);
    size_t item_size;
    if (key_type == "fixed") {
        generate_NB = generate_8B;
        item_size = sizeof(uint64_t);
    } else {
        generate_NB = generate_16B;
        item_size = key_length + sizeof(string_key);
    }
    uint64_t exist_num = load_num + insert_num;
    workload = (char *)malloc((exist_num + pos_search_num + neg_search_num) * item_size);
    key_generator_t *range_generator = new range_key_generator_t(1);
    // load and insert workload (range in 1 ~ exist_num)
    generate_NB(workload, exist_num, range_generator, -1);
    // pos search workload (uniform in 1 ~ exist_num)
    generate_NB(workload + exist_num * item_size, pos_search_num, uniform_generator, exist_num);
    // neg search workload (range in exist_num + 1 ~ exist_num + neg_search_num)
    generate_NB(workload + (exist_num + pos_search_num) * item_size, neg_search_num, range_generator, -1);
    std::cout << "Finish InsRangePosUniform workload Generation" << std::endl;
    return workload;
}

void *GenerateSkewWorkload(uint64_t load_num, uint64_t insert_num, uint64_t pos_search_num, uint64_t neg_search_num) {
    char *workload;
    void (*generate_NB)(void *memory_region, uint64_t generate_num, key_generator_t *key_generator, uint64_t mod_range);
    size_t item_size;
    if (key_type == "fixed") {
        generate_NB = generate_8B;
        item_size = sizeof(uint64_t);
    } else {
        generate_NB = generate_16B;
        item_size = key_length + sizeof(string_key);
    }
    workload = (char *)malloc((load_num + insert_num + pos_search_num + neg_search_num) * item_size);
    // load workload (range in 1 ~ load_num)
    key_generator_t *range_generator = new range_key_generator_t(1);
    generate_NB(workload, load_num, range_generator, -1);
    delete range_generator;
    if (update_num == 0) {
        // insert workload (skew in load_num + 1 ~ load_num + insert_num)
        key_generator_t *skew_generator = new zipfian_key_generator_t(load_num + 1, load_num + insert_num, skew_factor);
        generate_NB(workload + load_num * item_size, insert_num, skew_generator, -1);
        delete skew_generator;
        // pos search workload (skew in 1 ~ load_num)
        skew_generator = new zipfian_key_generator_t(1, load_num, skew_factor);
        generate_NB(workload + (load_num + insert_num) * item_size, pos_search_num, skew_generator, -1);
        delete skew_generator;
    } else {
        // insert workload (range in load_num + 1 ~ load_num + insert_num)
        key_generator_t *range_generator2 = new range_key_generator_t(load_num + 1);
        generate_NB(workload + load_num * item_size, insert_num, range_generator2, -1);
        delete range_generator2;
        // pos search workload (skew in 1 ~ load_num + insert_num)
        key_generator_t *skew_generator = new zipfian_key_generator_t(1, load_num + insert_num, skew_factor);
        generate_NB(workload + (load_num + insert_num) * item_size, pos_search_num, skew_generator, -1);
        delete skew_generator;
    }
    // neg search workload (skew in load_num + insert_num + 1 ~ load_num + insert_num * 2)
    key_generator_t *skew_generator = new zipfian_key_generator_t(load_num + insert_num + 1, load_num + insert_num * 2, skew_factor);
    generate_NB(workload + (load_num + insert_num + pos_search_num) * item_size, neg_search_num, skew_generator, -1);
    delete skew_generator;
    std::cout << "Finish Generation" << std::endl;
    return workload;
}

template <class T>
void Run() {
    if (operation == "skew-count") {
        void *workload = GenerateSkewWorkload(0, operation_num, 0, 0);
        uint64_t *keys = reinterpret_cast<uint64_t *>(workload);
        uint64_t count_array[operation_num] = {0};
        for (int i = 0; i < operation_num; i++) {
            count_array[keys[i] - 1]++;
        }
        for (int i = 0; i < operation_num; i++) {
            printf("%d, %lu\n", i + 1, count_array[i]);
        }
        return;
    }
    uniform_generator = new uniform_key_generator_t();
    Hash<T> *index = InitializeIndex<T>(init_cap);
    uint64_t generate_num = operation_num * 2 + load_num;
    /* Generate the workload and corresponding range array*/
    std::cout << "Generate workload" << std::endl;
    void *workload;
    if (distribution == "uniform") {
        if (update_num == 0)
            workload = GenerateInsRangePosUniformWorkload(load_num, operation_num, operation_num, operation_num);
        else
            workload = GenerateInsRangePosUniformWorkload(load_num, operation_num, update_num, update_num);
    } else {
        if (update_num == 0)
            workload = GenerateSkewWorkload(load_num, operation_num, operation_num, operation_num);
        else
            workload = GenerateSkewWorkload(load_num, operation_num, update_num, update_num);
    }
    void *insert_workload = workload;
    std::cout << "Finish Generate workload" << std::endl;
    std::cout << "load num = " << load_num << std::endl;
    // pm_count(false);
    Load<T>(load_num, index, key_length, insert_workload);
    // pm_count(true, load_num);
    index->getNumber();
    void *not_used_workload;
    void *not_used_insert_workload;
    if (key_type == "fixed") {
        uint64_t *keys = reinterpret_cast<uint64_t *>(workload);
        not_used_workload = reinterpret_cast<void *>(keys + load_num);
    } else {
        char *keys = reinterpret_cast<char *>(workload);
        not_used_workload = keys + (sizeof(string_key) + key_length) * load_num;
    }
    not_used_insert_workload = not_used_workload;
    /* Description of the workload*/
    srand((unsigned)time(NULL));
    Range *ranges;
    uint64_t chunk_size = operation_num / thread_num;
    ranges = reinterpret_cast<Range *>(malloc(thread_num * (sizeof(Range))));
    for (int i = 0; i < thread_num; ++i) {
        ranges[i].thread_id_ = i;
        ranges[i].random_num = rand();
        ranges[i].begin = i * chunk_size;
        ranges[i].end = (i + 1) * chunk_size;
        ranges[i].length = key_length;
        ranges[i].workload = not_used_workload;
    }
    ranges[thread_num - 1].end = operation_num;
    /* Benchmark Phase */
    if (operation == "insert") {
        std::cout << "Insert-only Benchmark" << std::endl;
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = not_used_insert_workload;
        }
        if (open_epoch) {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Insert", &concurr_insert);
        } else {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Insert", &concurr_insert_without_epoch);
        }
    } else if (operation == "pos") {
        if (!load_num) {
            std::cout << "Please first specify the # pre_load keys!" << std::endl;
            return;
        }
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = workload;
        }
        if (open_epoch) {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Pos_search", &concurr_search);
        } else {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Pos_search", &concurr_search_without_epoch);
        }
    } else if (operation == "neg") {
        if (!load_num) {
            std::cout << "Please first specify the # pre_load keys!" << std::endl;
            return;
        }
        if (open_epoch) {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Neg_search", &concurr_search);
        } else {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Neg_search", &concurr_search_without_epoch);
        }
    } else if (operation == "delete") {
        if (!load_num) {
            std::cout << "Please first specify the # pre_load keys!" << std::endl;
            return;
        }
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = workload;
        }
        if (open_epoch) {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Delete", &concurr_delete);
        } else {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Delete", &concurr_delete_without_epoch);
        }
    } else if (operation == "recovery") {
        std::cout << "Start the Recovery Benchmark" << std::endl;
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = not_used_workload;
        }
        RecoveryBench<T>(ranges, index, thread_num, operation_num, "Pos_search");
    } else if (operation == "insert-load") {
        std::cout << "Start the insert for load factor Benchmark" << std::endl;
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = not_used_insert_workload;
        }
        GeneralBench<T>(ranges, index, thread_num, operation_num, "Insert_load", &insert_load_factor);
    } else { /*do the benchmark for all single operations*/
        while (!index->readyForNextStage()) {
            asm("pause");
        }
        // printf("Index ready for next bench!\n");
        // std::cout << "Comprehensive Benchmark" << std::endl;
        std::cout << "insertion start" << std::endl;
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = not_used_insert_workload;
        }
        /* insert */
        // pm_count(false);
        if (open_epoch) {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Insert", &concurr_insert);
        } else {
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Insert", &concurr_insert_without_epoch);
        }
        // pm_count(true, operation_num);
        while (!index->readyForNextStage()) {
            asm("pause");
        }
        // printf("Index ready for next bench!\n");
        index->getNumber();
        if (update_num != 0)
            chunk_size = update_num / thread_num;
        /* update or mixed */
        // pm_count(false);
        // GeneralBench<T>(ranges, index, thread_num, operation_num, "Update", &concurr_update_without_epoch);
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].workload = not_used_workload;
        }
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].begin = operation_num + i * chunk_size;
            ranges[i].end = operation_num + (i + 1) * chunk_size;
        }
        if (update_num == 0) {
            ranges[thread_num - 1].end = 2 * operation_num;
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Update", &concurr_update_without_epoch);
        } else {
            ranges[thread_num - 1].end = operation_num + update_num;
            if (operation == "mixed")
                GeneralBench<T>(ranges, index, thread_num, update_num, "Mixed", &mixed_without_epoch);
            else
                GeneralBench<T>(ranges, index, thread_num, update_num, "Update", &concurr_update_without_epoch);
        }
        // pm_count(true, operation_num);
        while (!index->readyForNextStage()) {
            asm("pause");
        }
        // printf("Index ready for next bench!\n");
        index->getNumber();
        /* pos search */
        // pm_count(false);
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].begin = operation_num + i * chunk_size;
            ranges[i].end = operation_num + (i + 1) * chunk_size;
        }
        if (update_num == 0) {
            ranges[thread_num - 1].end = 2 * operation_num;
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Pos_search", &concurr_search_without_epoch);
        } else {
            ranges[thread_num - 1].end = operation_num + update_num;
            GeneralBench<T>(ranges, index, thread_num, update_num, "Pos_search", &concurr_search_without_epoch);
        }
        // pm_count(true, operation_num);
        index->getNumber();
        /* neg search */
        // pm_count(false);
        if (update_num == 0) {
            for (int i = 0; i < thread_num; ++i) {
                ranges[i].begin = operation_num * 2 + i * chunk_size;
                ranges[i].end = operation_num * 2 + (i + 1) * chunk_size;
            }
            ranges[thread_num - 1].end = 3 * operation_num;
            GeneralBench<T>(ranges, index, thread_num, operation_num, "Neg_search", &concurr_search_without_epoch);
        } else {
            for (int i = 0; i < thread_num; ++i) {
                ranges[i].begin = operation_num + update_num + i * chunk_size;
                ranges[i].end = operation_num + update_num + (i + 1) * chunk_size;
            }
            ranges[thread_num - 1].end = operation_num + 2 * update_num;
            GeneralBench<T>(ranges, index, thread_num, update_num, "Neg_search", &concurr_search_without_epoch);
        }
        // pm_count(true, operation_num);
        index->getNumber();
        /* delete */
        // pm_count(false);
        chunk_size = operation_num / thread_num;
        for (int i = 0; i < thread_num; ++i) {
            ranges[i].begin = i * chunk_size;
            ranges[i].end = (i + 1) * chunk_size;
        }
        ranges[thread_num - 1].end = operation_num;
        GeneralBench<T>(ranges, index, thread_num, operation_num, "Delete", &concurr_delete_without_epoch);
        // pm_count(true, operation_num);
        index->getNumber();
    }

    /*TODO Free the workload memory*/
}

bool check_ratio() {
    int read_portion = (int)(read_ratio * 100);
    int insert_portion = (int)(insert_ratio * 100);
    int delete_portion = (int)(delete_ratio * 100);
    if ((read_portion + insert_portion + delete_portion) != 100)
        return false;
    return true;
}

int main(int argc, char *argv[]) {
#ifdef VALUE_LENGTH_VARIABLE
    std::cout << "Variable Value Length!" << std::endl;
#endif
    set_affinity(0);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    init_cap = FLAGS_i;
    thread_num = FLAGS_t;
    load_num = FLAGS_n;
    operation_num = FLAGS_p;
    update_num = FLAGS_u;
    std::cout << "Update num = " << update_num << std::endl;
    if (update_num != 0)
        std::cout << "Multi load for update!" << std::endl;
    key_type = FLAGS_k;
    if (key_type != "fixed")
        std::cout << "Variable Key Length!" << std::endl;
    index_type = FLAGS_index;
    distribution = FLAGS_distribution;
    load_type = FLAGS_loadType;
    std::cout << "Distribution = " << distribution << std::endl;
    std::string fixed("fixed");
    operation = FLAGS_op;
    open_epoch = FLAGS_e;
    EPOCH_DURATION = FLAGS_ed;
    msec = FLAGS_ms;
    pool_size = FLAGS_ps * 1024ul * 1024ul * 1024ul; /*pool_size*/
    key_length = FLAGS_vkl;
    std::cout << "Variable key length = " << key_length << std::endl;
    value_length = FLAGS_vvl;
    std::cout << "Variable value length = " << value_length << std::endl;
    hot_bit = FLAGS_hb;
    hot_num = 1 << hot_bit;
    std::cout << "Hot bits = " << hot_bit << std::endl;
    batch_size = FLAGS_bs;
    std::cout << "Batch size = " << batch_size << std::endl;
    size_t kv_length = value_length;
#ifdef VARIABLE_KV
    kv_length = key_length + value_length + 2 * sizeof(int);
    std::cout << "Variable KV length = " << kv_length << std::endl;
#endif
    if (kv_length >= 128)
        update_retry_time = 2;
    if (kv_length >= 256)
        update_retry_time = 1;
    std::cout << "Update retry time = " << update_retry_time << std::endl;
    if (open_epoch)
        std::cout << "EPOCH registration in application level" << std::endl;
    read_ratio = FLAGS_r;
    insert_ratio = FLAGS_s;
    delete_ratio = FLAGS_d;
    skew_factor = FLAGS_skew;
    if (distribution == "skew")
        std::cout << "Skew theta = " << skew_factor << std::endl;
    if (operation == "mixed") {
        std::cout << "Search ratio = " << read_ratio << std::endl;
        std::cout << "Insert ratio = " << insert_ratio << std::endl;
        std::cout << "Delete ratio = " << delete_ratio << std::endl;
    }
    if (!check_ratio()) {
        std::cout << "The ratio is wrong!" << std::endl;
        return 0;
    }
    if (key_type.compare(fixed) == 0) {
        Run<uint64_t>();
    } else {
        std::cout << "Variable-length key = " << key_length << std::endl;
        Run<string_key *>();
    }
}

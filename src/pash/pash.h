#pragma once
#include <tbb/spin_rw_mutex.h>

#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../../util/hash.h"
#include "../../util/pair.h"
#include "../Hash.h"

#define INSERT_HTM
// #define SPLIT_LOCK
#define READ_HTM
// #define READ_LOCK
#define READ_RETRY_TIME 20

// #define VALUE_LENGTH_VARIABLE
extern uint64_t value_length;

#define INLOCK_UPDATE_RETRY_TIME 3
#define FREQ 64

extern __thread nsTimer *clk;
extern uint64_t update_retry_time;
extern uint64_t hot_num;
extern uint64_t hot_bit;
extern uint64_t asso;
__thread uint64_t request = 0;
extern uint64_t batch_size;

namespace zhash {
    template <class T>
    struct _Pair {
        T key;
        Value_t value;
    };

    constexpr size_t k_segment_bits = 2;
    constexpr size_t k_hash_suffix_mask = (1 << k_segment_bits) - 1;
    constexpr size_t k_segment_size = (1 << k_segment_bits) * 16 * 4;
    constexpr size_t k_metadata_space = 0;
    constexpr size_t k_num_bucket_per_segment = 4;
    constexpr size_t k_num_slot_per_bucket = 4;

    /* fingerprint and position in key_compound: 0-10 bits = fingerprint; 11 bit = valid; 12-15 bits = position; 16-63 bits = key */
    constexpr size_t k_fingerprint_bits = 11;
    constexpr size_t k_valid_bits = 1;
    constexpr size_t k_position_bits = 4;
    constexpr size_t k_fingerprint_shift = 64 - k_fingerprint_bits - 2;  // not use the last two bits
    constexpr size_t k_position_shift = 64 - k_fingerprint_bits - k_valid_bits - k_position_bits;
    constexpr size_t k_fingerprint_mask = 0xffe0000000000000;
    constexpr size_t k_valid_mask = 0x0010000000000000;
    constexpr size_t k_position_mask = 0x000f000000000000;
    constexpr size_t k_key_mask = 0x0000ffffffffffff;

    constexpr int status_dup_insert = -3;
    constexpr int status_seg_insert_ok = 0;
    constexpr int status_seg_insert_error = 1;
    constexpr int status_local_depth_changed = 2;
    constexpr int status_key_exist = 4;
    constexpr int status_find_in_sec_seg = 5;

    constexpr int value_type_default = 0;
    constexpr int value_type_hot = 1;
    constexpr int value_type_cold_large = 2;
    constexpr int value_type_cold_small = 3;

    constexpr size_t pri_sec_ratio = 32U;

    inline uint64_t get_position(uint64_t key_compound) {
        return (key_compound & k_position_mask) >> k_position_shift;
    }

    inline uint64_t get_position(void *key_compound) {
        return ((uint64_t)key_compound & k_position_mask) >> k_position_shift;
    }

    inline void set_fingerprint_position(uint64_t *key_compound, uint64_t hash, uint64_t position) {
        uint64_t key = *key_compound & k_key_mask;
        uint64_t fingerprint = (hash << k_fingerprint_shift) & k_fingerprint_mask;
        position = (position << k_position_shift) & k_position_mask;
        *key_compound = fingerprint | k_valid_mask | position | key;
    }

    inline void set_fingerprint_position(void **key_compound, uint64_t hash, uint64_t position) {
        uint64_t key = (uint64_t)*key_compound & k_key_mask;
        uint64_t fingerprint = (hash << k_fingerprint_shift) & k_fingerprint_mask;
        position = (position << k_position_shift) & k_position_mask;
        *key_compound = (void *)(fingerprint | k_valid_mask | position | key);
    }

    inline void clear_fingerprint_position(uint64_t *key_compound) {
        *key_compound = *key_compound & k_key_mask;
    }

    inline void clear_fingerprint_position(void **key_compound) {
        *key_compound = (void *)((uint64_t)*key_compound & k_key_mask);
    }

    inline bool check_valid(uint64_t key_compound) {
        return (key_compound & k_valid_mask) != 0;
    }

    inline bool check_valid(void *key_compound) {
        return ((uint64_t)key_compound & k_valid_mask) != 0;
    }

    inline bool match_fingerprint(uint64_t key_compound, uint64_t hash) {
        if (!check_valid(key_compound))
            return false;
        return (key_compound & k_fingerprint_mask) == ((hash << k_fingerprint_shift) & k_fingerprint_mask);
    }

    inline bool match_fingerprint(void *key_compound, uint64_t hash) {
        if (!check_valid(key_compound))
            return false;
        return ((uint64_t)key_compound & k_fingerprint_mask) == ((hash << k_fingerprint_shift) & k_fingerprint_mask);
    }

    inline bool match_key(uint64_t key_compound, uint64_t key_compare) {
        return (key_compound & k_key_mask) == (key_compare & k_key_mask);
    }

    inline bool match_key(void *key_compound, void *key_compare) {
        return ((uint64_t)key_compound & k_key_mask) == ((uint64_t)key_compare & k_key_mask);
    }

    inline uint64_t get_key(uint64_t key_compound) {
        return key_compound & k_key_mask;
    }

    inline uint64_t get_key(void *key_compound) {
        return (uint64_t)key_compound & k_key_mask;
    }

    inline void set_key(uint64_t *key_compound, uint64_t key) {
        uint64_t clear_key_compound = *key_compound & ~k_key_mask;
        uint64_t clear_key = key & k_key_mask;
        *key_compound = clear_key_compound | clear_key;
    }

    inline void set_key(void **key_compound, void *key) {
        uint64_t clear_key_compound = (uint64_t)*key_compound & ~k_key_mask;
        uint64_t clear_key = (uint64_t)key & k_key_mask;
        *key_compound = (void *)(clear_key_compound | clear_key);
    }

    inline void set_key(void **key_compound, uint64_t key) {
        uint64_t clear_key_compound = (uint64_t)*key_compound & ~k_key_mask;
        uint64_t clear_key = key & k_key_mask;
        *key_compound = (void *)(clear_key_compound | clear_key);
    }

    inline bool check_key_not_zero(uint64_t key_compound) {
        return (key_compound & k_key_mask) != 0;
    }

    inline bool check_key_not_zero(void *key_compound) {
        return ((uint64_t)key_compound & k_key_mask) != 0;
    }

    inline void clear_key(uint64_t *key_compound) {
        *key_compound = (*key_compound) & ~k_key_mask;
    }

    inline void clear_key(void **key_compound) {
        *key_compound = (void *)((uint64_t)(*key_compound) & ~k_key_mask);
    }

    // only use "key" bits (last 48 bits) to compute hash of a key
    inline size_t get_key_hash(const void *key, size_t len, size_t seed = 0xc70697UL) {
        uint64_t key_bits = get_key(*((uint64_t *)key));
        return h(&key_bits, len, seed);
    }

    /* metadata in segment addr: 0-5 bits = local_depth; 6 bit = lock */
    constexpr size_t k_depth_bits = 6;
    constexpr size_t k_depth_shift = 64 - k_depth_bits;
    constexpr size_t k_depth_mask = 0xfc00000000000000;
    constexpr size_t k_lock_mask = 0x0200000000000000;
    constexpr size_t k_addr_mask = 0x0000ffffffffffff;

    inline void *get_seg_addr(void *seg_ptr) {
        return (void *)((uint64_t)seg_ptr & k_addr_mask);
    }

    inline uint64_t get_local_depth(void *seg_ptr) {
        return ((uint64_t)seg_ptr & k_depth_mask) >> k_depth_shift;
    }

    inline void set_local_depth(void **seg_ptr_ptr, uint64_t depth) {
        uint64_t clear_old_addr = (uint64_t)*seg_ptr_ptr & ~k_depth_mask;
        *seg_ptr_ptr = (void *)(clear_old_addr | (depth << k_depth_shift));
    }

    inline bool get_seg_lock(void *seg_ptr) {
        return (uint64_t)seg_ptr & k_lock_mask;
    }

    inline void acquire_seg_lock(void **seg_ptr_ptr) {
        *seg_ptr_ptr = (void *)((uint64_t)*seg_ptr_ptr | k_lock_mask);
    }

    inline void acquire_seg_lock_with_cas(void **seg_ptr_ptr) {
        while (true) {
            if (!get_seg_lock(*seg_ptr_ptr)) {
                volatile void *old_seg_ptr = (void *)((uint64_t)*seg_ptr_ptr & ~k_lock_mask);
                volatile void *new_seg_ptr = (void *)((uint64_t)*seg_ptr_ptr | k_lock_mask);
                if (CAS(seg_ptr_ptr, &old_seg_ptr, new_seg_ptr))
                    break;
            }
            asm("pause");
        }
    }

    inline void release_seg_lock(void **seg_ptr_ptr) {
        *seg_ptr_ptr = (void *)((uint64_t)*seg_ptr_ptr & ~k_lock_mask);
    }

    inline void *construct_seg_ptr(void *seg_addr, uint64_t depth) {
        uint64_t clear_addr = (uint64_t)seg_addr & k_addr_mask;
        if (clear_addr == 0U) {
            printf("construct_seg_ptr seg_addr: %p depth: %lu\n", seg_addr, depth);
            assert(false);
        }
        return (void *)(clear_addr | (depth << k_depth_shift));
    }

    template <class T>
    struct Directory;

    template <class T>
    struct Segment {
        static const size_t k_num_slot_per_segment = k_segment_size / sizeof(_Pair<T>) - k_metadata_space;

        Segment(void) {
            std::cout << "Segment()" << std::endl;
            memset((void *)&pairs_[0], 255, sizeof(_Pair<T>) * k_num_slot_per_segment);
        }

        Segment(size_t depth) {
            memset((void *)&pairs_[0], 255, sizeof(_Pair<T>) * k_num_slot_per_segment);
        }

        static void New(void **seg_ptr_ptr, size_t depth) {
            auto seg_addr = reinterpret_cast<Segment *>(AAllocator::Allocate_without_proc(sizeof(Segment)));
            memset((void *)&seg_addr->pairs_[0], 0, sizeof(_Pair<T>) * k_num_slot_per_segment);
            *seg_ptr_ptr = construct_seg_ptr(seg_addr, depth);
        }

        ~Segment(void) { std::cout << "~Segment()" << std::endl; }

        int Insert(T, Value_t, size_t, size_t, int, void **, Directory<T> *);
        int sec_insert(T, Value_t, int, void **, Directory<T> *);
        int Update(Value_t, void **, Directory<T> *, int, bool);
        int get_slot_index(T, size_t);
        int get_sec_slot_index(T);
        void insert_for_split(T, Value_t, size_t);
        void rebalance(Segment<T> *sec_segment, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir);

        size_t acquire_lock(void **seg_ptr_ptr, void *dir_entries, size_t global_depth) {
            // get the first entry in the chunk
            char *first_dir_entry_addr = (char *)dir_entries;
            size_t chunk_size = pow(2, global_depth - get_local_depth(*seg_ptr_ptr));
            int dir_entry_index = ((char *)(seg_ptr_ptr)-first_dir_entry_addr) / sizeof(void *);
            if (dir_entry_index < 0) {
                printf("seg_ptr_ptr: %p first_dir_entry_addr: %p dir_entry_index: %d global_depth: %lu local_depth: %lu chunk_size: %lu\n", seg_ptr_ptr, first_dir_entry_addr, dir_entry_index, global_depth, get_local_depth(*seg_ptr_ptr), chunk_size);
                assert(false);
            }
            dir_entry_index = dir_entry_index - (dir_entry_index % chunk_size);
            // lock the first entry with cas
            acquire_seg_lock_with_cas((void **)(first_dir_entry_addr + dir_entry_index * sizeof(void *)));
            // lock the other entries without cas
            for (int i = 1; i < chunk_size; i++) {
                acquire_seg_lock((void **)(first_dir_entry_addr + (dir_entry_index + i) * sizeof(void *)));
            }
            return chunk_size;
        }

        void release_lock(void **seg_ptr_ptr, void *dir_entries, size_t chunk_size) {
            char *first_dir_entry_addr = (char *)dir_entries;
            int dir_entry_index = ((char *)(seg_ptr_ptr)-first_dir_entry_addr) / sizeof(void *);
            dir_entry_index = dir_entry_index - (dir_entry_index % chunk_size);
            // release locks in the opposite order
            for (int i = chunk_size - 1; i >= 0; i--) {
                release_seg_lock((void **)(first_dir_entry_addr + (dir_entry_index + i) * sizeof(void *)));
            }
            mfence();
        }

        bool check_lock(void *seg_ptr) { return get_seg_lock(seg_ptr); }

        _Pair<T> pairs_[k_num_slot_per_segment];
    };

    template <class T>
    struct Directory {
        size_t global_depth_;
        void *dir_entries_[0];

        static void New(void **dir_ptr_ptr, size_t capacity) {
            auto callback = [](void *ptr, void *arg) {
                auto value_ptr = reinterpret_cast<size_t *>(arg);
                auto dir_ptr = reinterpret_cast<Directory *>(ptr);
                dir_ptr->global_depth_ = static_cast<size_t>(log2(*value_ptr));
                return 0;
            };
            AAllocator::DAllocate(dir_ptr_ptr, k_cache_line_size,
                                  sizeof(Directory) + sizeof(uint64_t) * capacity,
                                  callback, reinterpret_cast<void *>(&capacity));
        }
    };

    template <class T>
    struct Hot_array {
        uint64_t hot_keys_[0];

        static void New(void **ha_ptr_ptr, size_t num) {
            auto callback = [](void *ptr, void *arg) {
                auto value_ptr = reinterpret_cast<size_t *>(arg);
                auto ha_ptr = reinterpret_cast<Hot_array *>(ptr);
                memset(ha_ptr->hot_keys_, 0, (*value_ptr) * sizeof(uint64_t));
                return 0;
            };
            AAllocator::DAllocate(ha_ptr_ptr, k_cache_line_size,
                                  sizeof(uint64_t) * num,
                                  callback, reinterpret_cast<void *>(&num));
        }
    };

    template <class T>
    struct Directory_Wrapper {
        static const size_t k_default_dir_size = 1024;
        Hot_array<T> *hot_arr_;
        Directory<T> *dir_;
        Directory<T> *sec_dir_;
        void *new_dir_;
        void *new_sec_dir_;
        size_t capacity_;
        bool lock_;
        bool fall_back_ = 0;

        static void New(Directory_Wrapper **dir_wrapper_ptr_ptr, size_t capacity) {
            auto callback = [](void *ptr, void *arg) {
                auto value_ptr = reinterpret_cast<std::pair<size_t, Directory<T> *> *>(arg);
                auto dir_wrapper_ptr = reinterpret_cast<Directory_Wrapper *>(ptr);
                dir_wrapper_ptr->capacity_ = value_ptr->first;
                dir_wrapper_ptr->dir_ = value_ptr->second;
                dir_wrapper_ptr->sec_dir_ = nullptr;
                dir_wrapper_ptr->new_dir_ = nullptr;
                dir_wrapper_ptr->lock_ = false;
                dir_wrapper_ptr->fall_back_ = 0;
                dir_wrapper_ptr = nullptr;
                return 0;
            };

            auto call_args = std::make_pair(capacity, nullptr);
            AAllocator::DAllocate((void **)dir_wrapper_ptr_ptr, k_cache_line_size, sizeof(Directory_Wrapper),
                                  callback, reinterpret_cast<void *>(&call_args));
        }

        ~Directory_Wrapper() { printf("~Directory_Wrapper()\n"); }

        void get_item_num() {
            size_t valid_key_num = 0;
            size_t seg_num = 0;
            Directory<T> *dir = dir_;
            void **dir_entries = dir->dir_entries_;
            void *seg_ptr;
            Segment<T> *segment;
            auto global_depth = dir->global_depth_;
            size_t depth_diff;
            std::unordered_map<uint64_t, int> exist;
            std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> location;
            std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> sec_location;
            for (uint64_t i = 1; i <= 20000000U; i++) {
                exist[i] = 0;
            }
            for (int dir_entry_index = 0; dir_entry_index < capacity_;) {
                seg_ptr = dir_entries[dir_entry_index];
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
                depth_diff = global_depth - get_local_depth(seg_ptr);
                for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
                    if (check_key_not_zero(segment->pairs_[slot].key)) {
                        // printf("get_item_num primary segment key %lu dir_entry_index %u slot %u\n", segment->pairs_[slot].key, dir_entry_index, slot);
                        ++valid_key_num;
                        uint64_t key = get_key(segment->pairs_[slot].key);
                        uint64_t key_hash = get_key_hash(&key, sizeof(T));
                        uint64_t dir_entry_index_1 = key_hash >> (64 - global_depth);
                        void *seg_ptr_1 = dir_entries[dir_entry_index_1];
                        Segment<T> *segment_1 = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr_1));
                        // if (segment_1 != segment) {
                        //     printf("\n>>> slot: %u seg computed %016p != seg read %016p\n\n", segment_1, segment);
                        //     printf("key: %012lx hash: %016lx\n", key, key_hash);
                        //     printf("dir_entry_index: %016lx seg_ptr: %016p seg read: %016p\n", dir_entry_index, seg_ptr, segment);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\ndir_entry_index: %016lx seg_ptr: %016lx seg computed: %016p\n", dir_entry_index_1, seg_ptr_1, segment_1);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment_1->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\n\n<<< seg computed %p != seg read %p\n", segment_1, segment);
                        // }
                        exist[key]++;
                        if (location.find(key) == location.end()) {
                            location[key] = std::vector<std::pair<int, int>>();
                        }
                        location[key].push_back(std::make_pair(dir_entry_index, slot));
                    }
                }
                seg_num++;
                dir_entry_index += pow(2, depth_diff);
            }
            Directory<T> *sec_dir = sec_dir_;
            void **sec_dir_entries = sec_dir->dir_entries_;
            void *sec_seg_ptr;
            Segment<T> *sec_segment;
            auto sec_global_depth = sec_dir->global_depth_;
            for (int sec_dir_entry_index = 0; sec_dir_entry_index < capacity_ / pri_sec_ratio;) {
                sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
                sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
                depth_diff = sec_global_depth - get_local_depth(sec_seg_ptr);
                // printf("%d %016p %p %lu %lu\n", dir_entry_index, sec_seg_ptr, sec_segment, get_local_depth(sec_seg_ptr), depth_diff);
                for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
                    if (check_key_not_zero(sec_segment->pairs_[slot].key)) {
                        // printf("get_item_num secondary segment key %lu sec_dir_entry_index %u slot %u\n", segment->pairs_[slot].key, sec_dir_entry_index, slot);
                        ++valid_key_num;
                        uint64_t key = get_key(sec_segment->pairs_[slot].key);
                        uint64_t key_hash = get_key_hash(&key, sizeof(T));
                        uint64_t sec_dir_entry_index_1 = key_hash >> (64 - sec_global_depth);
                        void *sec_seg_ptr_1 = sec_dir_entries[sec_dir_entry_index_1];
                        Segment<T> *sec_segment_1 = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr_1));
                        // if (sec_segment_1 != sec_segment) {
                        //     printf("\n>>> sec seg computed %016p != sec seg read %016p\n\n", sec_segment_1, sec_segment);
                        //     printf("key: %012lx hash: %016lx\n", key, key_hash);
                        //     printf("sec_dir_entry_index: %016lx sec_seg_ptr: %016p sec seg read: %016p\n", sec_dir_entry_index, sec_seg_ptr, sec_segment);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx sec_dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, sec_dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\nsec_dir_entry_index: %016lx sec_seg_ptr: %016p sec seg computed: %016p\n", sec_dir_entry_index_1, sec_seg_ptr_1, sec_segment_1);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment_1->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx sec_dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, sec_dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\n\n<<< sec seg computed %016p != sec seg read %016p\n", sec_segment_1, sec_segment);
                        // }
                        exist[key]++;
                        if (sec_location.find(key) == sec_location.end()) {
                            sec_location[key] = std::vector<std::pair<int, int>>();
                        }
                        sec_location[key].push_back(std::make_pair(sec_dir_entry_index, slot));
                    }
                }
                seg_num++;
                sec_dir_entry_index += 1;  // 🟣sec seg 永远只被 1 个 sec dir 指向，local depth == global depth
            }
            std::cout << "#items: " << valid_key_num << std::endl;
            if (valid_key_num > 5000U) {  // insert / update
                for (auto &i : exist) {
                    uint64_t key = i.first;
                    int count = i.second;
                    if (count == 0) {
                        // printf("\n>>> key: %012lx count: %d\n\n", key, count);
                        // uint64_t key_hash = get_key_hash(&key, sizeof(T));
                        // uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        // uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        // segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[dir_entry_index]));
                        // printf("hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu seg computed: %016p\n", key_hash, dir_entry_index, main_bucket_index, segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //     uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        // }
                        // uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        // sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
                        // printf("\nhash: %016lx sec_dir_entry_index: %016lx sec seg computed: %016p\n", key_hash, sec_dir_entry_index, sec_segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, key, key_hash, dir_entry_index);
                        // }
                        // printf("\n\n<<< key: %012lx count: %d\n", key, count);
                    } else if (count > 1) {
                        // printf("\n>>> key: %012lx count: %d\n\n", key, count);
                        // uint64_t key_hash = get_key_hash(&key, sizeof(T));
                        // uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        // uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        // segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[dir_entry_index]));
                        // printf("hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu seg computed: %016p\n", key_hash, dir_entry_index, main_bucket_index, segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //     uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        // }
                        // uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        // sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
                        // printf("\nhash: %016lx sec_dir_entry_index: %016lx sec seg computed: %016p\n", key_hash, sec_dir_entry_index, sec_segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, key, key_hash, dir_entry_index);
                        // }
                        // for (auto &dir_entry_index_slot : location[key]) {
                        //     uint64_t dir_entry_index = dir_entry_index_slot.first;
                        //     int slot = dir_entry_index_slot.second;
                        //     seg_ptr = dir_entries[dir_entry_index];
                        //     segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
                        //     depth_diff = global_depth - get_local_depth(seg_ptr);
                        //     printf("\ndir_entry_index: %016x slot: %02u seg_ptr: %016lx seg read: %016p depth_diff: %u\n", dir_entry_index, slot, seg_ptr, segment, depth_diff);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        // }
                        // for (auto &sec_dir_entry_index_slot : sec_location[key]) {
                        //     uint64_t sec_dir_entry_index = sec_dir_entry_index_slot.first;
                        //     int slot = sec_dir_entry_index_slot.second;
                        //     sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
                        //     sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
                        //     depth_diff = sec_global_depth - get_local_depth(sec_seg_ptr);
                        //     printf("\nsec_dir_entry_index: %016x slot: %02u sec_seg_ptr: %016lx sec seg read: %016p depth_diff: %u\n", sec_dir_entry_index, slot, sec_seg_ptr, sec_segment, depth_diff);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t sec_dir_entry_index = key_hash >> (64 - global_depth);
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx sec_dir_entry_index: %016lx\n", slot, key_compound, key, key_hash, sec_dir_entry_index);
                        //     }
                        // }
                        // printf("\n\n<<< key: %012lx count: %d\n", key, count);
                    }
                }
            } else {  // delete
                for (auto &i : exist) {
                    uint64_t key = i.first;
                    int count = i.second;
                    if (count != 0) {
                        // printf("\n>>> key: %012lx count: %d\n\n", key, count);
                        // uint64_t key_hash = get_key_hash(&key, sizeof(T));
                        // uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        // uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        // segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[dir_entry_index]));
                        // printf("hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu seg computed: %016p\n", key_hash, dir_entry_index, main_bucket_index, segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //     uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        // }
                        // uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        // sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
                        // printf("\nhash: %016lx sec_dir_entry_index: %016lx sec seg computed: %016p\n", key_hash, sec_dir_entry_index, sec_segment);
                        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //     uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //     uint64_t key = get_key(key_compound);
                        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //     uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //     printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, key, key_hash, dir_entry_index);
                        // }
                        // for (auto &dir_entry_index_slot : location[key]) {
                        //     uint64_t dir_entry_index = dir_entry_index_slot.first;
                        //     int slot = dir_entry_index_slot.second;
                        //     seg_ptr = dir_entries[dir_entry_index];
                        //     segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
                        //     depth_diff = global_depth - get_local_depth(seg_ptr);
                        //     printf("\ndir_entry_index: %016lx slot: %02u seg_ptr: %016p seg read: %016p depth_diff: %lu\n", dir_entry_index, slot, seg_ptr, segment, depth_diff);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        // }
                        // for (auto &sec_dir_entry_index_slot : sec_location[key]) {
                        //     uint64_t sec_dir_entry_index = sec_dir_entry_index_slot.first;
                        //     int slot = sec_dir_entry_index_slot.second;
                        //     sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
                        //     sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
                        //     depth_diff = sec_global_depth - get_local_depth(sec_seg_ptr);
                        //     printf("\nsec_dir_entry_index: %016lx slot: %02u sec_seg_ptr: %016p sec seg read: %016p depth_diff: %lu\n", sec_dir_entry_index, slot, sec_seg_ptr, sec_segment, depth_diff);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //         uint64_t key = get_key(key_compound);
                        //         uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
                        //         uint64_t sec_dir_entry_index = key_hash >> (64 - global_depth);
                        //         printf("slot: %02u key_compound: %016lx key: %012lx hash: %016lx sec_dir_entry_index: %016lx\n", slot, key_compound, key, key_hash, sec_dir_entry_index);
                        //     }
                        // }
                        // printf("\n\n<<< key: %012lx count: %d\n", key, count);
                    }
                }
            }
            std::cout << std::fixed << "load_factor: "
                      << (double)valid_key_num / (seg_num * ((1 << k_segment_bits) * 4 - k_metadata_space))
                      << std::endl;
        }

        bool Acquire(void) {
            bool unlocked = false;
            return CAS(&lock_, &unlocked, true);
        }

        bool Release(void) {
            bool locked = true;
            return CAS(&lock_, &locked, false);
        }

        bool acquire_fallback(void) {
            bool unlocked = false;
            return CAS(&fall_back_, &unlocked, true);
        }

        bool release_fallback(void) {
            bool locked = true;
            return CAS(&fall_back_, &locked, false);
        }
    };

    template <class T>
    class ZHASH : public Hash<T> {
    public:
        ZHASH(void);
        ZHASH(int);
        ~ZHASH();
        int Insert(T key, Value_t value, int batch_offset = -1, T *batch_keys = nullptr, nsTimer *clks = nullptr);
        bool Delete(T, int batch_offset = -1, T *batch_keys = nullptr, nsTimer *clks = nullptr);
        bool Get(T, Value_t *, int batch_offset = -1, T *batch_keys = nullptr, nsTimer *clks = nullptr);
        void *get_bucket_addr(T key);
        void Recovery(void);
        Segment<T> *Split(T key, Segment<T> *old_segment, uint64_t prev_local_depth,
                          Directory<T> *dir, void **seg_ptr_ptr, uint64_t dir_entry_index, size_t *old_seg_prefix, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir);
        void help_double_dir(T key, Directory<T> *new_dir, uint64_t prev_local_depth);
        void help_double_sec_dir(T key, Directory<T> *new_sec_dir, uint64_t prev_sec_local_depth);
        void double_dir();
        void update_dir(int dir_entry_index, void *old_seg_ptr, void **new_seg_ptr_ptr, Directory<T> *sa);
        void Lock();
        void Unlock();
        bool check_fallback();
        void lock_dir();
        void unlock_dir();
        bool check_lock_dir();
        void getNumber() {
            // print_hot();
            dir_wrapper_->get_item_num();
        }

        bool check_hot(T);
        bool check_hot_without_update(T);
        bool update_hot_keys(T);
        void set_hot();
        void print_hot();

        Directory_Wrapper<T> *dir_wrapper_;
    };

    template <class T>
    int Segment<T>::get_slot_index(T key, size_t main_bucket_index) {
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_key(pairs_[main_bucket_slot].key, key)) {
                return main_bucket_slot;
            }
        }
        // if not found, check the fingerprints in this bucket
        uint64_t key_hash = get_key_hash(&key, sizeof(key));
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_fingerprint(pairs_[main_bucket_slot].key, key_hash)) {
                unsigned overflow_bucket_slot = get_position(pairs_[main_bucket_slot].key);
                if (match_key(pairs_[overflow_bucket_slot].key, key)) {
                    return overflow_bucket_slot;
                }
            }
        }
        return -1;
    }

    template <class T>
    int Segment<T>::get_sec_slot_index(T key) {
        if (this == nullptr) {
            printf("Segment<T>::get_sec_slot_index this == nullptr\n");
            assert(false);
        }
        for (unsigned slot = 0; slot < k_num_slot_per_segment; slot++) {
            if (match_key(pairs_[slot].key, key)) {
                return slot;
            }
        }
        return -1;
    }

    template <class T>
    int Segment<T>::Update(Value_t value, void **seg_ptr_ptr, Directory<T> *dir, int slot, bool hot) {
        int type = value_type_default;  // determine update type
#ifdef VALUE_LENGTH_VARIABLE
        if (hot) {
            type = value_type_hot;  // hot key, in-place update
        } else {
            if (value_length >= 128)
                type = value_type_cold_large;  // cold large key, in-place + flush
            else
                type = value_type_cold_small;  // cold small key, copy on write
        }
#endif
        if (type == value_type_hot || type == value_type_cold_large) {
            int htm_status = 1;
            int inlock_status;
            size_t locking_chunk_size = 0;
            for (int j = 0; j < update_retry_time; ++j) {
                while (check_lock(*seg_ptr_ptr))  // check lock before start HTM
                    asm("pause");
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED) {
                    break;
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                locking_chunk_size = acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
                // htm in lock path
                for (int k = 0; k < INLOCK_UPDATE_RETRY_TIME; k++) {
                    inlock_status = _xbegin();
                    if (inlock_status == _XBEGIN_STARTED)
                        break;
                }
            } else if (check_lock(*seg_ptr_ptr)) {
                _xabort(6);
            }
            // in-place update
            uint64_t *value_addr = (uint64_t *)(pairs_[slot].value);
            for (int i = 0; i < value_length / sizeof(uint64_t); i++) {
                value_addr[i] = uint64_t(value);
            }
            if (htm_status == _XBEGIN_STARTED) {
                _xend();
            } else {
                if (inlock_status == _XBEGIN_STARTED)
                    _xend();
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            }
            // flush cold key
            if (type == value_type_cold_large)
                AAllocator::Persist_flush(value_addr, value_length);
        } else if (type == value_type_cold_small || type == value_type_default) {
            if (type == value_type_cold_small) {
                value = AAllocator::Prepare_value(value, value_length);
            }
        RE_UPDATE:
            int htm_status;
            size_t locking_chunk_size = 0;
            for (int i = 0; i < 2; ++i) {
                while (check_lock(*seg_ptr_ptr))
                    asm("pause");
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED)
                    break;
            }
            if (htm_status != _XBEGIN_STARTED) {
                locking_chunk_size = acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
            } else if (check_lock(*seg_ptr_ptr)) {
                _xend();
                goto RE_UPDATE;
            }
            pairs_[slot].value = value;
            if (htm_status != _XBEGIN_STARTED)
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            else
                _xend();
        }
        return status_dup_insert;
    }

    template <class T>
    int Segment<T>::Insert(T key, Value_t value, size_t main_bucket_index, size_t key_hash, int prev_local_depth, void **seg_ptr_ptr, Directory<T> *dir) {
        int ret = status_seg_insert_error;
#ifdef VALUE_LENGTH_VARIABLE
        value = AAllocator::Prepare_value(value, value_length);
#endif
#ifdef INSERT_HTM
        int htm_status;
        size_t locking_chunk_size = 0;
        for (int i = 0; i < 64; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED)
                break;
            asm("pause");
        }
        if (htm_status != _XBEGIN_STARTED) {
            locking_chunk_size = acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
            if (prev_local_depth != get_local_depth(*seg_ptr_ptr)) {
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
                return status_local_depth_changed;
            }
        } else if (check_lock(*seg_ptr_ptr) || prev_local_depth != get_local_depth(*seg_ptr_ptr)) {
            _xend();
            return status_local_depth_changed;
        }
#endif
        int slot = get_slot_index(key, main_bucket_index);
        if (slot != -1) {
#ifdef INSERT_HTM
            if (htm_status != _XBEGIN_STARTED)
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            else
                _xend();
#endif
            return status_key_exist;
        }
        int invalid_main_bucket_slot = -1;
        int overflow_bucket_slot = -1;
        for (unsigned i = 0; i < k_num_slot_per_segment; i++) {
            unsigned slot = (main_bucket_index * k_num_slot_per_bucket + i) % k_num_slot_per_segment;
            if (i < k_num_slot_per_bucket) {  // main bucket
                if (!check_key_not_zero(pairs_[slot].key)) {
                    pairs_[slot].value = value;
                    set_key((void **)&(pairs_[slot].key), key);
                    ret = status_seg_insert_ok;
                    // if constexpr (!std::is_pointer_v<T>) {
                    //     if (key == 1082446U) {
                    //         printf("Segment<T>::Insert key %lu slot %u\n", key, slot);
                    //     }
                    // }
                    break;
                } else {
                    if (invalid_main_bucket_slot == -1 && !check_valid(pairs_[slot].key)) {
                        // if constexpr (!std::is_pointer_v<T>) {
                        //     if (key == 1082446U || match_key(pairs_[slot].key, 1082446U)) {
                        //         printf("Segment<T>::Insert key %lu slot %u pairs_[slot].key %lu check_valid(pairs_[slot].key) %d\n", key, slot, pairs_[slot].key, check_valid(pairs_[slot].key));
                        //     }
                        // }
                        invalid_main_bucket_slot = slot;
                    }
                }
            } else {  // overflow bucket
                if (invalid_main_bucket_slot == -1)
                    break;
                if (!check_key_not_zero(pairs_[slot].key)) {
                    // if constexpr (!std::is_pointer_v<T>) {
                    //     if (key == 1082446U || match_key(pairs_[invalid_main_bucket_slot].key, 1082446U)) {
                    //         printf("Segment<T>::Insert before key %lu slot %u invalid_main_bucket_slot %u pairs_[invalid_main_bucket_slot].key %lu key_hash %lu\n", key, slot, invalid_main_bucket_slot, pairs_[invalid_main_bucket_slot].key, key_hash);
                    //     }
                    // }
                    overflow_bucket_slot = slot;
                    pairs_[slot].value = value;
                    set_key((void **)&(pairs_[slot].key), key);
                    set_fingerprint_position((void **)&(pairs_[invalid_main_bucket_slot].key), key_hash, slot);
                    ret = status_seg_insert_ok;
                    // if constexpr (!std::is_pointer_v<T>) {
                    //     if (key == 1082446U || match_key(pairs_[invalid_main_bucket_slot].key, 1082446U)) {
                    //         printf("Segment<T>::Insert after key %lu slot %u invalid_main_bucket_slot %u pairs_[invalid_main_bucket_slot].key %lu\n", key, slot, invalid_main_bucket_slot, pairs_[invalid_main_bucket_slot].key);
                    //     }
                    // }
                    break;
                }
            }
        }
#ifdef INSERT_HTM
        if (htm_status != _XBEGIN_STARTED)
            release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
        else
            _xend();
#endif
        if (overflow_bucket_slot != -1) {
            AAllocator::Persist_asyn_flush(&(pairs_[main_bucket_index * k_num_slot_per_bucket]), 64);
            AAllocator::Persist_asyn_flush(&(pairs_[overflow_bucket_slot / k_num_slot_per_bucket * k_num_slot_per_bucket]), 64);
        }
        return ret;
    }

    template <class T>
    int Segment<T>::sec_insert(T key, Value_t value, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir) {
        int ret = status_seg_insert_error;
#ifdef VALUE_LENGTH_VARIABLE
        value = AAllocator::Prepare_value(value, value_length);
#endif
#ifdef INSERT_HTM
        int htm_status;
        size_t locking_chunk_size = 0;
        for (int i = 0; i < 64; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED)
                break;
            asm("pause");
        }
        if (htm_status != _XBEGIN_STARTED) {
            locking_chunk_size = acquire_lock(sec_seg_ptr_ptr, sec_dir->dir_entries_, sec_dir->global_depth_);
            if (prev_sec_local_depth != get_local_depth(*sec_seg_ptr_ptr)) {
                release_lock(sec_seg_ptr_ptr, sec_dir->dir_entries_, locking_chunk_size);
                return status_local_depth_changed;
            }
        } else if (check_lock(*sec_seg_ptr_ptr) || prev_sec_local_depth != get_local_depth(*sec_seg_ptr_ptr)) {
            _xend();
            return status_local_depth_changed;
        }
#endif
        int slot = get_sec_slot_index(key);
        if (slot != -1) {
#ifdef INSERT_HTM
            if (htm_status != _XBEGIN_STARTED)
                release_lock(sec_seg_ptr_ptr, sec_dir->dir_entries_, locking_chunk_size);
            else
                _xend();
#endif
            return status_key_exist;
        }
        for (unsigned slot = 0; slot < k_num_slot_per_segment; slot++) {
            if (!check_key_not_zero(pairs_[slot].key)) {
                pairs_[slot].value = value;
                set_key((void **)&(pairs_[slot].key), key);
                ret = status_seg_insert_ok;
                // if constexpr (!std::is_pointer_v<T>) {
                //     if (key == 45982U) {
                //         printf("Segment<T>::sec_insert key %lu slot %u\n", key, slot);
                //     }
                // }
                break;
            }
        }
#ifdef INSERT_HTM
        if (htm_status != _XBEGIN_STARTED)
            release_lock(sec_seg_ptr_ptr, sec_dir->dir_entries_, locking_chunk_size);
        else
            _xend();
#endif
        return ret;
    }

    template <class T>
    void Segment<T>::rebalance(Segment<T> *sec_segment, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir) {
        _Pair<T> overflow_pairs[k_num_slot_per_segment];
        uint64_t overflow_hashs[k_num_slot_per_segment];
        size_t num_overflow_pair = 0;
        size_t pair_nums[4] = {0, 0, 0, 0};
        // bool a = false;
        // if constexpr (!std::is_pointer_v<T>) {
        //     for (unsigned i = 0; i < k_num_slot_per_segment; i++) {
        //         if (match_key(pairs_[i].key, 1082446U)) {
        //             a = true;
        //         }
        //     }
        // }
        // if (a) {
        //     printf("\n rebalance before find overflow pairs \n");
        //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
        //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
        //     }
        // }
        // find all overflow pairs
        for (unsigned i = 0; i < k_num_slot_per_segment; i++) {
            if (check_key_not_zero(pairs_[i].key)) {
                uint64_t key_hash = get_key_hash(&(pairs_[i].key), sizeof(T));
                if ((uint64_t)(i / k_num_bucket_per_segment) != (uint64_t)(key_hash & k_hash_suffix_mask)) {
                    // if (a) {
                    //     printf("key %lu i / k_num_bucket_per_segment %lu != key_hash & k_hash_suffix_mask %lu\n", pairs_[i].key, i / k_num_bucket_per_segment, key_hash & k_hash_suffix_mask);
                    // }
                    overflow_pairs[num_overflow_pair] = pairs_[i];
                    overflow_hashs[num_overflow_pair] = key_hash;
                    num_overflow_pair++;
                    clear_key((void **)&(pairs_[i].key));
                } else {
                    // if (a) {
                    //     printf("key %lu i / k_num_bucket_per_segment %lu == key_hash & k_hash_suffix_mask %lu\n", pairs_[i].key, i / k_num_bucket_per_segment, key_hash & k_hash_suffix_mask);
                    // }
                    pair_nums[i / k_num_bucket_per_segment]++;
                }
            }
        }
        // if (a) {
        //     printf("\n rebalance after find overflow pairs \n");
        //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
        //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
        //     }
        // }
        // try to bring overflow pairs back to their main buckets
        for (unsigned i = 0; i < num_overflow_pair; i++) {
            size_t main_bucket_index = overflow_hashs[i] & k_hash_suffix_mask;
            for (unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket; main_bucket_slot < (main_bucket_index + 1) * k_num_slot_per_bucket; main_bucket_slot++) {
                if (!check_key_not_zero(pairs_[main_bucket_slot].key)) {
                    pairs_[main_bucket_slot].value = overflow_pairs[i].value;
                    set_key((void **)&(pairs_[main_bucket_slot].key), overflow_pairs[i].key);
                    overflow_pairs[i].key = (T)0;
                    pair_nums[main_bucket_index]++;
                    break;
                }
            }
        }
        // if (a) {
        //     printf("\n rebalance after move overflow pairs to main buckets \n");
        //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
        //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
        //     }
        // }
        // bring the remaining overflow pairs to overflow buckets
        for (unsigned i = 0; i < num_overflow_pair; i++) {
            if (overflow_pairs[i].key != (T)0) {
                size_t main_bucket_index = overflow_hashs[i] & k_hash_suffix_mask;
                int overflow_bucket_slot = -1;
                int final_slot = -1;
                // find the most empty overflow bucket
                size_t most_empty_bucket = (main_bucket_index + 1) % k_num_bucket_per_segment;
                for (unsigned j = main_bucket_index + 2; j < main_bucket_index + 4; j++) {
                    if (pair_nums[j % k_num_bucket_per_segment] < pair_nums[most_empty_bucket])
                        most_empty_bucket = j % k_num_bucket_per_segment;
                }
                // select the first empty slot in the most empty bucket
                for (unsigned slot = most_empty_bucket * k_num_slot_per_bucket; slot < (most_empty_bucket + 1) * k_num_slot_per_bucket; slot++) {
                    if (!check_key_not_zero(pairs_[slot].key)) {
                        overflow_bucket_slot = slot;
                        break;
                    }
                }
                // assert(overflow_bucket_slot >= 0);
                if (overflow_bucket_slot < 0) {  // no empty slot in the most empty bucket
                    int ret = sec_segment->sec_insert((T)get_key(overflow_pairs[i].key), overflow_pairs[i].value, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
                    // if (a) {
                    //     printf("rebalance overflow_bucket_slot < 0 ret %d\n", ret);
                    //     printf("\n rebalance after move %lx to %d from %d \n", overflow_pairs[i].key, overflow_bucket_slot, final_slot);
                    //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
                    //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
                    //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
                    //     }
                    // }
                    continue;
                }
                // select the first invalid slot in main bucket
                for (unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket; main_bucket_slot < (main_bucket_index + 1) * k_num_slot_per_bucket; main_bucket_slot++) {
                    if (!check_valid(pairs_[main_bucket_slot].key)) {
                        final_slot = main_bucket_slot;
                        break;
                    }
                }
                // assert(final_slot >= 0);
                if (final_slot < 0) {
                    int ret = sec_segment->sec_insert((T)get_key(overflow_pairs[i].key), overflow_pairs[i].value, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
                    // if (a) {
                    //     printf("rebalance final_slot < 0 ret %d\n", ret);
                    //     printf("\n rebalance after move %lx to %d from %d \n", overflow_pairs[i].key, overflow_bucket_slot, final_slot);
                    //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
                    //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
                    //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
                    //     }
                    // }
                    continue;
                }
                // insert pair into the most empty overflow bucket
                pairs_[overflow_bucket_slot].value = overflow_pairs[i].value;
                set_key((void **)&(pairs_[overflow_bucket_slot].key), overflow_pairs[i].key);
                pair_nums[most_empty_bucket]++;
                // set fingerprint and position in main bucket
                set_fingerprint_position((void **)&(pairs_[final_slot].key), overflow_hashs[i], overflow_bucket_slot);
                // if (a) {
                //     printf("\n rebalance after move %lx to %d from %d \n", overflow_pairs[i].key, overflow_bucket_slot, final_slot);
                //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
                //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
                //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
                //     }
                // }
            }
        }
        // if (a) {
        //     printf("\n rebalance after move overflow pairs to overflow buckets \n");
        //     for (unsigned j = 0; j < k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(pairs_[j].key), sizeof(T));
        //         printf("%u %lx %lx %lu\n", j, pairs_[j].key, key_hash, key_hash & k_hash_suffix_mask);
        //     }
        // }
    }

    template <class T>
    void Segment<T>::insert_for_split(T key, Value_t value, size_t slot) {
        pairs_[slot].value = value;
        set_key((void **)&(pairs_[slot].key), key);
    }

    template <class T>
    ZHASH<T>::ZHASH(int init_cap) {
        Directory_Wrapper<T>::New(&dir_wrapper_, init_cap);
        Hot_array<T>::New((void **)(&dir_wrapper_->hot_arr_), hot_num);
        Directory<T>::New(&dir_wrapper_->new_dir_, init_cap);
        dir_wrapper_->dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
        Directory<T>::New(&dir_wrapper_->new_sec_dir_, init_cap / pri_sec_ratio);
        dir_wrapper_->sec_dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_sec_dir_);
        auto dir_entries = dir_wrapper_->dir_->dir_entries_;
        for (int dir_entry_index = 0; dir_entry_index < dir_wrapper_->capacity_; ++dir_entry_index) {
            Segment<T>::New(&dir_entries[dir_entry_index], dir_wrapper_->dir_->global_depth_);
        }
        auto sec_dir_entries = dir_wrapper_->sec_dir_->dir_entries_;
        for (int sec_dir_entry_index = 0; sec_dir_entry_index < dir_wrapper_->capacity_ / pri_sec_ratio; ++sec_dir_entry_index) {
            Segment<T>::New(&sec_dir_entries[sec_dir_entry_index], dir_wrapper_->sec_dir_->global_depth_);
        }
        printf("segment size: %ld\n", sizeof(Segment<T>));
        printf("segment slots size: %ld\n", sizeof(Segment<T>::pairs_));
        printf("pair size: %ld\n", sizeof(_Pair<T>));
    }

    template <class T>
    ZHASH<T>::ZHASH(void) { std::cout << "Reintialize Up for ZHASH" << std::endl; }

    template <class T>
    ZHASH<T>::~ZHASH(void) { std::cout << "~ZHASH()" << std::endl; }

    template <class T>
    void ZHASH<T>::Recovery(void) {
        if (dir_wrapper_ != nullptr) {
            dir_wrapper_->lock_ = 0;
            if (dir_wrapper_->dir_ == nullptr)
                return;
            auto dir_entries = dir_wrapper_->dir_->dir_entries_;
            size_t global_depth = dir_wrapper_->dir_->global_depth_;
            size_t local_depth, next_i, stride, i = 0;
            /* recover the directory */
            size_t seg_count = 0;
            while (i < dir_wrapper_->capacity_) {
                auto seg_ptr = reinterpret_cast<Segment<T> *>(dir_entries[i]);
                local_depth = get_local_depth(seg_ptr);
                stride = pow(2, global_depth - local_depth);
                next_i = i + stride;
                for (int j = next_i - 1; j > i; j--) {
                    seg_ptr = reinterpret_cast<Segment<T> *>(dir_entries[j]);
                    if (dir_entries[j] != dir_entries[i]) {
                        dir_entries[j] = dir_entries[i];
                    }
                }
                seg_count++;
                i = i + stride;
            }
        }
    }

    template <class T>
    bool ZHASH<T>::check_hot(T key) {
        size_t key_hash = h(&key, sizeof(key));
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; ++i) {
            if (hot_keys[i] == uint64_t(key))
                return true;
            else if (hot_keys[i] == 0) {
                update_hot_keys(key);
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ZHASH<T>::check_hot_without_update(T key) {
        size_t key_hash = h(&key, sizeof(key));
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; ++i) {
            if (hot_keys[i] == uint64_t(key))
                return true;
            else if (hot_keys[i] == 0) {
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ZHASH<T>::update_hot_keys(T key) {
        size_t key_hash = h(&key, sizeof(key));
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        bool found_key = false;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; i++) {
            if (hot_keys[i] == 0) {
                hot_keys[i] = uint64_t(key);
                found_key = true;
                break;
            }
            if (hot_keys[i] == uint64_t(key)) {
                found_key = true;
                if (i != idx) {
                    int htm_status = _xbegin();
                    if (htm_status == _XBEGIN_STARTED) {
                        hot_keys[i] = hot_keys[i - 1];
                        hot_keys[i - 1] = uint64_t(key);
                        _xend();
                    }
                }
                break;
            }
        }
        if (!found_key)
            hot_keys[idx + asso - 1] = uint64_t(key);
        return true;
    }

    template <class T>
    void ZHASH<T>::print_hot() {
        int top_100_count = 0;
        int top_10_count = 0;
        int top_1_count = 0;
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        for (int i = 0; i < uint64_t(hot_num); ++i) {
            if (hot_keys[i] != 0) {
                if (hot_keys[i] < uint64_t(hot_num))
                    top_100_count++;
                if (hot_keys[i] < uint64_t(hot_num) / 10)
                    top_10_count++;
                if (hot_keys[i] < uint64_t(hot_num) / 100)
                    top_1_count++;
            }
        }
        printf("Top 100% Hot number: %f\n", double(top_100_count) / uint64_t(hot_num));
        printf("Top 10% Hot number: %f\n", double(top_10_count) / uint64_t(hot_num) * 10);
        printf("Top 1% Hot number: %f\n", double(top_1_count) / uint64_t(hot_num) * 100);
    }

    template <class T>
    void ZHASH<T>::set_hot() {
        int global_depth = dir_wrapper_->dir_->global_depth_;
        uint64_t num_dir_entry = 1llu << global_depth;
        uint64_t num_dir_entry_per_hot_key = num_dir_entry / uint64_t(hot_num);
        Segment<T> *segment;
        Segment<T> *prev_segment = NULL;
        auto dir_entries = dir_wrapper_->dir_->dir_entries_;
        uint64_t curr_key;
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        printf("[set_hot] global depth: %d\n", global_depth);
        printf("[set_hot] directory entry num: %lu\n", num_dir_entry);
        printf("[set_hot] hot num: %lu\n", hot_num);
        printf("[set_hot] seg num: %lu\n", num_dir_entry_per_hot_key);
        for (int i = 0; i < uint64_t(hot_num); i += asso) {
            for (int j = i; j < i + asso; ++j)
                hot_keys[j] = 20000000;
            for (int j = i * num_dir_entry_per_hot_key; j < (i + asso) * num_dir_entry_per_hot_key; ++j) {
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[j]));
                if (segment == prev_segment)
                    continue;
                for (unsigned k = 0; k < Segment<T>::k_num_slot_per_segment; ++k) {
                    if (check_key_not_zero(segment->pairs_[k].key)) {
                        curr_key = get_key(segment->pairs_[k].key);
                        for (int p = i; p < i + asso; ++p) {
                            if (curr_key < hot_keys[p]) {
                                if (p != i)
                                    hot_keys[p - 1] = hot_keys[p];
                                hot_keys[p] = curr_key;
                            }
                        }
                    }
                }
                prev_segment = segment;
            }
        }
        print_hot();
    }

    template <class T>
    Segment<T> *ZHASH<T>::Split(T key, Segment<T> *old_segment, uint64_t prev_local_depth, Directory<T> *dir, void **seg_ptr_ptr, uint64_t dir_entry_index, size_t *old_seg_prefix, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir) {
        int prev_global_depth = dir->global_depth_;
        if (get_local_depth(*seg_ptr_ptr) >= prev_global_depth) {
            lock_dir();
            if (dir_wrapper_->dir_->global_depth_ == prev_global_depth) {
                printf("doubling number %lu global depth %d\n", uint64_t(key), prev_global_depth);
                double_dir();
            }
            unlock_dir();
            return NULL;
        }
        // split without doubling
        size_t locking_chunk_size = 0;
        size_t key_hash = get_key_hash(&key, sizeof(key));
        void *new_seg_ptr;
        void **new_seg_ptr_ptr = &new_seg_ptr;
        Segment<T>::New(new_seg_ptr_ptr, get_local_depth(*seg_ptr_ptr));
        Segment<T> *new_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*new_seg_ptr_ptr));
        int htm_status;
        for (int i = 0; i < 2; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED)
                break;
        }
        if (htm_status != _XBEGIN_STARTED) {
            locking_chunk_size = old_segment->acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
            if (get_local_depth(*seg_ptr_ptr) != prev_local_depth) {
                old_segment->release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
                return NULL;
            }
        } else if (old_segment->check_lock(*seg_ptr_ptr) || get_local_depth(*seg_ptr_ptr) != prev_local_depth ||
                   (check_lock_dir() && (Directory<T> *)dir_wrapper_->new_dir_ != dir)) {
            // Ensure that the segment is not locked by the fall back path of HTM
            // Ensure that the split of segment has not been finished by other threads
            // Ensure that the doubling doesn't happen after the split begins
            _xend();
            return NULL;
        }
        size_t common_prefix = dir_entry_index >> (prev_global_depth - prev_local_depth);
        size_t old_seg_prefix1 = common_prefix << 1;
        size_t new_seg_prefix = (common_prefix << 1) + 1;
        auto old_sec_dir = dir_wrapper_->sec_dir_;
        auto sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_sec_dir->global_depth_);
        auto sec_dir_entries = old_sec_dir->dir_entries_;
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
        // bool a = false;
        // if constexpr (!std::is_pointer_v<T>) {
        //     for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; i++) {
        //         if (match_key(sec_segment->pairs_[i].key, 1082446U)) {
        //             a = true;
        //         }
        //         if (match_key(old_segment->pairs_[i].key, 1082446U)) {
        //             a = true;
        //         }
        //     }
        // }
        // if (a) {
        //     printf("\n before move old to new \n");
        //     printf("\n secondary segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(sec_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, sec_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n old segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(old_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, old_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n new segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(new_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, new_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        // }
        for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
            uint64_t key_hash = get_key_hash(&(old_segment->pairs_[slot].key), sizeof(T));
            // clear all fingerprints because there is a overall modification afterward
            clear_fingerprint_position((void **)&(old_segment->pairs_[slot].key));
            if (check_key_not_zero(old_segment->pairs_[slot].key) &&
                (key_hash >> (8 * 8 - get_local_depth(*seg_ptr_ptr) - 1) == new_seg_prefix)) {
                // move only clear keys without fingerprint
                new_segment->insert_for_split((T)get_key(old_segment->pairs_[slot].key), old_segment->pairs_[slot].value, slot);
                clear_key((void **)&(old_segment->pairs_[slot].key));
            }
        }
        // insert key-value pairs in secondary segments into primary old segment and primary new segment
        // if (a) {
        //     printf("\n after move old to new \n");
        //     printf("\n secondary segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(sec_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, sec_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n old segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(old_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, old_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n new segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(new_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, new_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        // }
        for (unsigned k = 0; k < Segment<T>::k_num_slot_per_segment; ++k) {
            uint64_t key_hash = get_key_hash(&(sec_segment->pairs_[k].key), sizeof(T));
            if (check_key_not_zero(sec_segment->pairs_[k].key)) {
                auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
                if (hash_prefix == new_seg_prefix) {
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        if (!check_key_not_zero(new_segment->pairs_[i].key)) {
                            new_segment->insert_for_split((T)get_key(sec_segment->pairs_[k].key), sec_segment->pairs_[k].value, i);
                            clear_key((void **)&(sec_segment->pairs_[k].key));
                            break;
                        }
                    }
                } else if (hash_prefix == old_seg_prefix1) {
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        if (!check_key_not_zero(old_segment->pairs_[i].key)) {
                            old_segment->insert_for_split((T)get_key(sec_segment->pairs_[k].key), sec_segment->pairs_[k].value, i);
                            clear_key((void **)&(sec_segment->pairs_[k].key));
                            break;
                        }
                    }
                }
            }
        }
        // if (a) {
        //     printf("\n after move seg to old and new \n");
        //     printf("\n secondary segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(sec_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, sec_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n old segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(old_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, old_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n new segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(new_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, new_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        // }
        new_segment->rebalance(sec_segment, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
        old_segment->rebalance(sec_segment, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
        // if (a) {
        //     printf("\n after rebalance old and new \n");
        //     printf("\n secondary segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(sec_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, sec_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n old segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(old_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, old_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        //     printf("\n new segment \n");
        //     for (unsigned j = 0; j < Segment<T>::k_num_slot_per_segment; j++) {
        //         uint64_t key_hash = get_key_hash(&(new_segment->pairs_[j].key), sizeof(T));
        //         auto hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
        //         printf("%u %lx %lx %lx\n", j, new_segment->pairs_[j].key, key_hash, hash_prefix);
        //     }
        // }
        *old_seg_prefix = (key_hash >> (8 * sizeof(key_hash) - get_local_depth(*new_seg_ptr_ptr))) << 1;
        dir_entry_index = key_hash >> (8 * sizeof(key_hash) - dir->global_depth_);
        update_dir(dir_entry_index, *seg_ptr_ptr, new_seg_ptr_ptr, dir);
        if (htm_status != _XBEGIN_STARTED) {
            old_segment->release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
        } else {
            _xend();
        }
        return new_segment;
    }

    template <class T>
    void ZHASH<T>::help_double_dir(T key, Directory<T> *new_dir, uint64_t prev_local_depth) {
        size_t key_hash = get_key_hash(&key, sizeof(key));
        auto dir = dir_wrapper_->dir_;
        auto dir_entry_index = key_hash >> (8 * sizeof(key_hash) - dir->global_depth_);
        unsigned depth_diff = dir->global_depth_ - prev_local_depth;
        int chunk_size = pow(2, depth_diff);
        int dir_entry_index_start = dir_entry_index - (dir_entry_index % chunk_size);
        int dir_entry_index_end = dir_entry_index_start + chunk_size;
        dir_entry_index_start = dir_entry_index_start - (dir_entry_index_start % 4);
        dir_entry_index_end = dir_entry_index_end - (dir_entry_index_end % 4) + 4;
        int htm_status;
        bool new_dir_entry_not_null;
        for (int i = dir_entry_index_start; i < dir_entry_index_end; i += 4) {
            new_dir_entry_not_null = false;
            for (int j = 0; j < 8; ++j) {
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED)
                    break;
                else
                    asm("pause");
            }
            if (htm_status != _XBEGIN_STARTED) {
                Lock();
            } else if (check_fallback()) {
                _xend();
                while (check_fallback()) {
                    asm("pause");
                }
                i -= 4;
                continue;
            }
            for (int j = i; j < i + 4; j++) {
                if (new_dir->dir_entries_[2 * j] != NULL || new_dir->dir_entries_[2 * j + 1] != NULL) {
                    new_dir_entry_not_null = true;
                    break;
                }
            }
            if (!new_dir_entry_not_null) {
                for (int j = i; j < i + 4; j++) {
                    new_dir->dir_entries_[2 * j] = dir->dir_entries_[j];
                    new_dir->dir_entries_[2 * j + 1] = dir->dir_entries_[j];
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                Unlock();
            } else {
                _xend();
            }
        }
    }

    template <class T>
    void ZHASH<T>::help_double_sec_dir(T key, Directory<T> *new_sec_dir, uint64_t prev_sec_local_depth) {
        size_t key_hash = get_key_hash(&key, sizeof(key));
        auto sec_dir = dir_wrapper_->sec_dir_;
        auto sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - sec_dir->global_depth_);    // 255
        unsigned depth_diff = sec_dir->global_depth_ - prev_sec_local_depth;                       // 0 1 2 3
        int chunk_size = pow(2, depth_diff);                                                       // 1 2 4 8 -> 0 1 3 7
        int sec_dir_entry_index_start = sec_dir_entry_index - (sec_dir_entry_index % chunk_size);  // 255 254 252 248
        int sec_dir_entry_index_end = sec_dir_entry_index_start + chunk_size;                      // 256 256 256 256
        // sec_dir_entry_index_start = sec_dir_entry_index_start - (sec_dir_entry_index_start % 4);   // 252 252 252 248
        // sec_dir_entry_index_end = sec_dir_entry_index_end - (sec_dir_entry_index_end % 4);         // 256 256 256 256
        int htm_status;
        bool new_sec_dir_entry_not_null;
        // printf("thread ID: %u sec_dir_entry_index: %u global_depth: %u local_depth: %u start: %d end: %d\n", gettid(), sec_dir_entry_index, sec_dir->global_depth_, prev_sec_local_depth, sec_dir_entry_index_start, sec_dir_entry_index_end);
        // Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir->dir_entries_[sec_dir_entry_index]));
        // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
        //     uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
        //     uint64_t key = get_key(key_compound);
        //     uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
        //     uint64_t dir_entry_index = key_hash >> (64 - dir_wrapper_->sec_dir_->global_depth_);
        //     uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
        //     printf("%02u %016lx %012lx %016lx %016lx %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
        // }
        for (int i = sec_dir_entry_index_start; i < sec_dir_entry_index_end; i += 1) {
            new_sec_dir_entry_not_null = false;
            for (int j = 0; j < 8; ++j) {
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED)
                    break;
                else
                    asm("pause");
            }
            if (htm_status != _XBEGIN_STARTED) {
                Lock();
            } else if (check_fallback()) {
                _xend();
                while (check_fallback()) {
                    asm("pause");
                }
                i -= 1;
                continue;
            }
            for (int j = i; j < i + 1; j++) {
                if (new_sec_dir->dir_entries_[2 * j] != NULL || new_sec_dir->dir_entries_[2 * j + 1] != NULL) {
                    new_sec_dir_entry_not_null = true;
                    break;
                }
            }
            if (!new_sec_dir_entry_not_null) {
                for (int j = i; j < i + 1; j++) {
                    new_sec_dir->dir_entries_[2 * j] = construct_seg_ptr(get_seg_addr(sec_dir->dir_entries_[j]), get_local_depth(sec_dir->dir_entries_[j]) + 1);
                    Segment<T>::New(&(new_sec_dir->dir_entries_[2 * j + 1]), get_local_depth(sec_dir->dir_entries_[j]) + 1);
                    Segment<T> *old_sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(new_sec_dir->dir_entries_[2 * j]));
                    // printf("thread ID: %u sec_dir_entries[%d]: %016p new_sec_dir_entries[%d]: %016p new_sec_dir_entries[%d]: %016p\n", std::this_thread::get_id(), j, sec_dir->dir_entries_[j], 2 * j, new_sec_dir->dir_entries_[2 * j], 2 * j + 1, new_sec_dir->dir_entries_[2 * j + 1]);
                    Segment<T> *new_sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(new_sec_dir->dir_entries_[2 * j + 1]));
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        uint64_t key_hash = get_key_hash(&(old_sec_segment->pairs_[i].key), sizeof(T));
                        if (check_key_not_zero(old_sec_segment->pairs_[i].key) &&
                            (key_hash >> (8 * 8 - get_local_depth(sec_dir->dir_entries_[j]) - 1) == 2 * j + 1)) {
                            new_sec_segment->insert_for_split((T)get_key(old_sec_segment->pairs_[i].key), old_sec_segment->pairs_[i].value, i);
                            clear_key((void **)&(old_sec_segment->pairs_[i].key));
                        }
                    }
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                Unlock();
            } else {
                _xend();
            }
        }
    }

    template <class T>
    void ZHASH<T>::double_dir() {
        Directory<T> *dir = dir_wrapper_->dir_;
        void **dir_entries = dir->dir_entries_;
        void *new_seg_ptr;
        void **new_dir_ptr_ptr = &new_seg_ptr;
        Directory<T>::New(new_dir_ptr_ptr, 2 * dir_wrapper_->capacity_);
        Directory<T> *sec_dir = dir_wrapper_->sec_dir_;
        void **sec_dir_entries = sec_dir->dir_entries_;
        void *new_sec_dir_ptr;
        void **new_sec_dir_ptr_ptr = &new_sec_dir_ptr;
        Directory<T>::New(new_sec_dir_ptr_ptr, 2 * dir_wrapper_->capacity_ / pri_sec_ratio);
        mfence();
        uint64_t pm_alloc = AAllocator::total_pm_alloc();
        uint64_t dram_alloc = 2 * dir_wrapper_->capacity_ + 2 * dir_wrapper_->capacity_ / pri_sec_ratio;
        printf("At this doubling, DRAM: %lu, PM: %lu, DRAM/PM: %.3f\n", dram_alloc, pm_alloc, double(dram_alloc) / pm_alloc);
        dir_wrapper_->new_dir_ = *new_dir_ptr_ptr;
        dir_wrapper_->new_sec_dir_ = *new_sec_dir_ptr_ptr;
        // other threads can operate on dir_wrapper_->new_dir_
        mfence();
        auto new_dir = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
        auto new_dir_entries = new_dir->dir_entries_;
        int htm_status;
        for (unsigned i = 0; i < dir_wrapper_->capacity_; i += 4) {
            bool new_dir_entry_not_null = false;
            for (int i = 0; i < 8; ++i) {
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED)
                    break;
                else
                    asm("pause");
            }
            if (htm_status != _XBEGIN_STARTED) {
                Lock();
            } else if (check_fallback()) {
                _xend();
                while (check_fallback()) {
                    asm("pause");
                }
                i -= 4;
                continue;
            }
            for (int j = i; j < i + 4; ++j) {
                if (new_dir_entries[2 * j] != 0 || new_dir_entries[2 * j + 1] != 0) {
                    new_dir_entry_not_null = true;
                    break;
                }
            }
            if (!new_dir_entry_not_null) {
                for (int j = i; j < i + 4; ++j) {
                    new_dir_entries[2 * j] = dir_entries[j];
                    new_dir_entries[2 * j + 1] = dir_entries[j];
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                Unlock();
            } else {
                _xend();
            }
        }
        dir_wrapper_->dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
        auto new_sec_dir = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_sec_dir_);
        auto new_sec_dir_entries = new_sec_dir->dir_entries_;
        for (unsigned i = 0; i < dir_wrapper_->capacity_ / pri_sec_ratio; i += 1) {
            bool new_sec_dir_entry_not_null = false;
            for (int i = 0; i < 8; ++i) {
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED)
                    break;
                else
                    asm("pause");
            }
            if (htm_status != _XBEGIN_STARTED) {
                Lock();
            } else if (check_fallback()) {
                _xend();
                while (check_fallback()) {
                    asm("pause");
                }
                i -= 1;
                continue;
            }
            for (int j = i; j < i + 1; ++j) {
                if (get_seg_addr(new_sec_dir_entries[2 * j]) != nullptr || get_seg_addr(new_sec_dir_entries[2 * j + 1]) != nullptr) {
                    new_sec_dir_entry_not_null = true;
                    // printf("new_sec_dir_entries[%d]: %p new_sec_dir_entries[%d]: %p\n", 2 * j, new_sec_dir_entries[2 * j], 2 * j + 1, new_sec_dir_entries[2 * j + 1]);
                    break;
                }
            }
            if (!new_sec_dir_entry_not_null) {
                // printf("new_sec_dir_entries[%d] - new_sec_dir_entries[%d] is nullptr\n", 2 * i, 2 * i + 1);
                for (int j = i; j < i + 1; ++j) {
                    new_sec_dir_entries[2 * j] = construct_seg_ptr(get_seg_addr(sec_dir_entries[j]), get_local_depth(sec_dir_entries[j]) + 1);
                    Segment<T>::New(&(new_sec_dir_entries[2 * j + 1]), get_local_depth(sec_dir_entries[j]) + 1);
                    Segment<T> *old_sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(new_sec_dir_entries[2 * j]));
                    Segment<T> *new_sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(new_sec_dir_entries[2 * j + 1]));
                    // printf("new_sec_dir_entries[%u]: %p new_sec_dir_entries[%u]: %p\n", 2 * j, new_sec_dir_entries[2 * j], 2 * j + 1, new_sec_dir_entries[2 * j + 1]);
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        uint64_t key_hash = get_key_hash(&(old_sec_segment->pairs_[i].key), sizeof(T));
                        if (check_key_not_zero(old_sec_segment->pairs_[i].key) &&
                            (key_hash >> (8 * 8 - get_local_depth(sec_dir_entries[j]) - 1) == 2 * j + 1)) {
                            new_sec_segment->insert_for_split((T)get_key(old_sec_segment->pairs_[i].key), old_sec_segment->pairs_[i].value, i);
                            clear_key((void **)&(old_sec_segment->pairs_[i].key));
                        }
                    }
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                Unlock();
            } else {
                _xend();
            }
        }
        // for (unsigned i = 0; i < dir_wrapper_->capacity_ / pri_sec_ratio; i++) {
        //     printf("sec_dir_entries[%u]: %p\n", i, sec_dir_entries[i]);
        // }
        // for (unsigned i = 0; i < dir_wrapper_->capacity_ * 2 / pri_sec_ratio; i++) {
        //     printf("new_sec_dir_entries[%u]: %p\n", i, new_sec_dir_entries[i]);
        // }
        dir_wrapper_->sec_dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_sec_dir_);
        mfence();
        dir_wrapper_->capacity_ *= 2;
        mfence();
    }

    template <class T>
    void ZHASH<T>::lock_dir() {
        while (!dir_wrapper_->Acquire()) {
            asm("nop");
        }
    }

    template <class T>
    void ZHASH<T>::unlock_dir() {
        while (!dir_wrapper_->Release()) {
            asm("nop");
        }
    }

    template <class T>
    bool ZHASH<T>::check_lock_dir() { return dir_wrapper_->lock_; }

    template <class T>
    void ZHASH<T>::Lock() {
        while (!dir_wrapper_->acquire_fallback()) {
            asm("nop");
        }
    }

    template <class T>
    void ZHASH<T>::Unlock() {
        while (!dir_wrapper_->release_fallback()) {
            asm("nop");
        }
    }

    template <class T>
    bool ZHASH<T>::check_fallback() { return dir_wrapper_->fall_back_; }

    template <class T>
    void ZHASH<T>::update_dir(int dir_entry_index, void *old_seg_ptr, void **new_seg_ptr_ptr, Directory<T> *dir) {
        size_t old_local_depth = get_local_depth(old_seg_ptr);
        void **dir_entries = dir->dir_entries_;
        auto global_depth = dir->global_depth_;
        unsigned depth_diff = global_depth - old_local_depth;
        int chunk_size = pow(2, global_depth - old_local_depth);
        dir_entry_index = dir_entry_index - (dir_entry_index % chunk_size);
        int half_chunk_size = chunk_size / 2;
        for (int i = 0; i < half_chunk_size; ++i) {
            dir_entries[dir_entry_index + half_chunk_size + i] = (Segment<T> *)(*new_seg_ptr_ptr);
        }
        // update all local depths in the chunk
        for (int i = 0; i < chunk_size; ++i) {
            set_local_depth(&dir_entries[dir_entry_index + i], old_local_depth + 1);
        }
    }

    template <class T>
    int ZHASH<T>::Insert(T key, Value_t value, int batch_offset, T *batch_keys, nsTimer *clks) {
        if (batch_offset == 0) {
            if (clks)
                clks[0].start();
        }
        bool flush_split_seg = false;
        bool batch_process = false;
    STARTOVER:
        uint64_t key_hash = get_key_hash(&key, sizeof(key));
        auto main_bucket_index = key_hash & k_hash_suffix_mask;
    RETRY:
        Directory<T> *old_dir = dir_wrapper_->dir_;
        void **old_dir_entries = old_dir->dir_entries_;
        uint64_t old_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_dir->global_depth_);
        void **old_seg_ptr_ptr = &(old_dir_entries[old_dir_entry_index]);
        Segment<T> *old_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*old_seg_ptr_ptr));
        int old_prev_local_depth = get_local_depth(*old_seg_ptr_ptr);

        Directory<T> *old_sec_dir = dir_wrapper_->sec_dir_;
        void **old_sec_dir_entries = old_sec_dir->dir_entries_;
        uint64_t old_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_sec_dir->global_depth_);
        void **old_sec_seg_ptr_ptr = &(old_sec_dir_entries[old_sec_dir_entry_index]);
        Segment<T> *old_sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*old_sec_seg_ptr_ptr));
        int old_prev_sec_local_depth = get_local_depth(*old_sec_seg_ptr_ptr);

        int ret = 0;
        Directory<T> *new_dir;
        Directory<T> *new_sec_dir;

        Directory<T> *dir = old_dir;
        void **dir_entries = old_dir_entries;
        uint64_t dir_entry_index = old_dir_entry_index;
        void **seg_ptr_ptr = old_seg_ptr_ptr;
        Segment<T> *segment = old_segment;
        int prev_local_depth = old_prev_local_depth;

        Directory<T> *sec_dir = old_sec_dir;
        void **sec_dir_entries = old_sec_dir_entries;
        uint64_t sec_dir_entry_index = old_sec_dir_entry_index;
        void **sec_seg_ptr_ptr = old_sec_seg_ptr_ptr;
        Segment<T> *sec_segment = old_sec_segment;
        int prev_sec_local_depth = old_prev_sec_local_depth;

        // find correct directory when directory doubling is happening
        // if (check_lock_dir()) {
        //     new_dir = (Directory<T> *)dir_wrapper_->new_dir_;
        //     new_sec_dir = (Directory<T> *)dir_wrapper_->new_sec_dir_;
        //     bool dir_doubling = new_dir != old_dir;
        //     bool sec_dir_doubling = new_sec_dir != old_sec_dir;
        //     if (dir_doubling && sec_dir_doubling) {
        //         uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
        //         uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
        //         bool new_seg_allocated = get_seg_addr(new_dir->dir_entries_[new_dir_entry_index]) != nullptr;
        //         bool new_sec_seg_allocated = get_seg_addr(new_sec_dir->dir_entries_[new_sec_dir_entry_index]) != nullptr;
        //         if (new_seg_allocated && new_sec_seg_allocated) {
        //             dir = new_dir;
        //             dir_entries = dir->dir_entries_;
        //             dir_entry_index = new_dir_entry_index;
        //             seg_ptr_ptr = &(dir_entries[dir_entry_index]);
        //             segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
        //             // printf("when doubling new_dir_entries[%u]: %p dir_entries[%u]:%p segment: %p\n", new_dir_entry_index, new_dir->dir_entries_[new_dir_entry_index], dir_entry_index, dir_entries[dir_entry_index], segment);
        //             prev_local_depth = get_local_depth(*seg_ptr_ptr);

        //             sec_dir = new_sec_dir;
        //             sec_dir_entries = sec_dir->dir_entries_;
        //             sec_dir_entry_index = new_sec_dir_entry_index;
        //             sec_seg_ptr_ptr = &(sec_dir_entries[sec_dir_entry_index]);
        //             sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*sec_seg_ptr_ptr));
        //             // printf("when doubling new_sec_dir_entries[%u]: %p sec_dir_entries[%u]:%p sec_segment: %p\n", new_sec_dir_entry_index, new_sec_dir->dir_entries_[new_sec_dir_entry_index], sec_dir_entry_index, sec_dir_entries[sec_dir_entry_index], sec_segment);
        //             prev_sec_local_depth = get_local_depth(*sec_seg_ptr_ptr);
        //         } else if ((new_seg_allocated && !new_sec_seg_allocated) || (!new_seg_allocated && new_sec_seg_allocated)) {
        //             // goto RETRY;
        //         }
        //     } else if ((dir_doubling && !sec_dir_doubling) || (!dir_doubling && sec_dir_doubling)) {
        //         // goto RETRY;
        //     }
        // }
        if (check_lock_dir() && (new_dir = (Directory<T> *)(dir_wrapper_->new_dir_)) != old_dir) {
            uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
            bool new_seg_allocated = get_seg_addr(new_dir->dir_entries_[new_dir_entry_index]) != nullptr;
            if (new_seg_allocated) {
                dir = new_dir;
                dir_entries = dir->dir_entries_;
                dir_entry_index = new_dir_entry_index;
                seg_ptr_ptr = &(dir_entries[dir_entry_index]);
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
                // printf("when doubling new_dir_entries[%u]: %p dir_entries[%u]:%p segment: %p\n", new_dir_entry_index, new_dir->dir_entries_[new_dir_entry_index], dir_entry_index, dir_entries[dir_entry_index], segment);
                prev_local_depth = get_local_depth(*seg_ptr_ptr);
            }
        }
        if (check_lock_dir() && (new_sec_dir = (Directory<T> *)(dir_wrapper_->new_sec_dir_)) != old_sec_dir) {
            uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
            bool new_sec_seg_allocated = get_seg_addr(new_sec_dir->dir_entries_[new_sec_dir_entry_index]) != nullptr;
            if (new_sec_seg_allocated) {
                sec_dir = new_sec_dir;
                sec_dir_entries = sec_dir->dir_entries_;
                sec_dir_entry_index = new_sec_dir_entry_index;
                sec_seg_ptr_ptr = &(sec_dir_entries[sec_dir_entry_index]);
                sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*sec_seg_ptr_ptr));
                // printf("when doubling new_sec_dir_entries[%u]: %p sec_dir_entries[%u]:%p sec_segment: %p\n", new_sec_dir_entry_index, new_sec_dir->dir_entries_[new_sec_dir_entry_index], sec_dir_entry_index, sec_dir_entries[sec_dir_entry_index], sec_segment);
                prev_sec_local_depth = get_local_depth(*sec_seg_ptr_ptr);
            }
        }
        if (get_seg_addr(dir_entries[dir_entry_index]) != segment || get_seg_addr(sec_dir_entries[sec_dir_entry_index]) != sec_segment) {
            printf("depth and dentry mismatch\n");
            asm("pause");
            goto RETRY;
        }
        if (batch_offset == 0 && !batch_process) {
            void *bucket_addr_0 = &(segment->pairs_[main_bucket_index * k_num_slot_per_bucket]);
            prefetch(bucket_addr_0);
            for (int batch_index = 1; batch_index < batch_size; batch_index++) {
                if (clks)
                    clks[batch_index].start();
                void *bucket_addr = get_bucket_addr(batch_keys[batch_index]);
                prefetch(bucket_addr);
            }
            batch_process = true;
        }
    Check:
        int slot = segment->get_slot_index(key, main_bucket_index);
        if (slot != -1) {
            bool hot = false;
#ifdef VALUE_LENGTH_VARIABLE
            if (request % FREQ == 0) {
                update_hot_keys(key);
            }
            request++;
            hot = check_hot(key);
#endif
            ret = segment->Update(value, seg_ptr_ptr, dir, slot, hot);
            if (flush_split_seg) {
                AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            }
            return -1;
        }
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("can not find %lu in primary segment %p, try to find %lu in secondary segment %p\n", key, segment, key, sec_segment);
        //     }
        // }
        // can not find in primary segment, try to find in secondary segment
        if (sec_segment == nullptr) {
            printf("ZHASH<T>::Insert sec_segment == nullptr\n");
            assert(false);
        }
        slot = sec_segment->get_sec_slot_index(key);
        if (slot != -1) {  // found in secondary segment
            bool hot = false;
            ret = sec_segment->Update(value, sec_seg_ptr_ptr, sec_dir, slot, hot);
            if (flush_split_seg) {
                AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            }
            return -1;
        }
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("can not find %lu in secondary segment %p, try to insert %lu into primary segment %p\n", key, sec_segment, key, segment);
        //     }
        // }
        // can not find in secondary segment, try to insert into primary segment
        ret = segment->Insert(key, value, main_bucket_index, key_hash, prev_local_depth, seg_ptr_ptr, dir);
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("insert %lu into primary segment %p return %d\n", key, segment, ret);
        //     }
        // }
        if (ret == status_key_exist) {
            goto Check;
        }
        if (flush_split_seg) {
            AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            flush_split_seg = false;
        }
        if (ret == status_local_depth_changed) {
            goto STARTOVER;
        }
        if (ret == status_seg_insert_ok) {
            return ret;
        }
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("primary segment %p has no room to insert %lu, try to insert %lu into secondary segment %p\n", segment, key, key, sec_segment);
        //     }
        // }
        // primary segment has no room to insert, try to insert into secondary segment
        ret = sec_segment->sec_insert(key, value, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("insert %lu into secondary segment %p return %d\n", key, sec_segment, ret);
        //         for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
        //             uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
        //             uint64_t key = get_key(key_compound);
        //             uint64_t key_hash = get_key_hash(&key_compound, sizeof(T));
        //             uint64_t dir_entry_index = key_hash >> (64 - dir_wrapper_->sec_dir_->global_depth_);
        //             uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
        //             printf("%02u %016lx %012lx %016lx %016lx %02lu\n", slot, key_compound, key, key_hash, dir_entry_index, main_bucket_index);
        //         }
        //     }
        // }
        if (ret == status_key_exist) {
            goto Check;
        }
        if (flush_split_seg) {
            AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            flush_split_seg = false;
        }
        if (ret == status_local_depth_changed) {
            goto STARTOVER;
        }
        if (ret == status_seg_insert_ok) {
            return ret;
        }
        // if constexpr (!std::is_pointer_v<T>) {
        //     if (key == 45982U) {
        //         printf("secondary segment %p has no room to insert %lu, split primary segment %p\n", sec_segment, key, segment);
        //     }
        // }
        // secondary segment has no room to insert, split primary segment
        if (ret == status_seg_insert_error) {
#ifdef SPLIT_LOCK
            if (!segment->try_get_lock()) {
                goto RETRY;
            }
#endif
            Segment<T> *new_segment;
            size_t old_seg_prefix;
            if (check_lock_dir()) {  // other threads
                volatile Directory<T> *tmp_new_dir;
                volatile Directory<T> *tmp_new_sec_dir;
                // do {
                //     asm("pause");
                //     tmp_new_dir = (Directory<T> *)dir_wrapper_->new_dir_;
                //     tmp_new_sec_dir = (Directory<T> *)dir_wrapper_->new_sec_dir_;
                //     new_dir = (Directory<T> *)tmp_new_dir;
                //     new_sec_dir = (Directory<T> *)tmp_new_sec_dir;
                // } while (new_dir == old_dir && new_sec_dir == old_sec_dir && check_lock_dir());  // wait to allocate new directory
                // if (!check_lock_dir()) {
                //     goto RETRY;
                // }
                // if (new_dir != old_dir) {
                //     help_double_dir(key, new_dir, prev_local_depth);
                //     uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
                //     seg_ptr_ptr = &(new_dir->dir_entries_[new_dir_entry_index]);
                //     new_segment = Split(key, segment, prev_local_depth, new_dir, seg_ptr_ptr, new_dir_entry_index, &old_seg_prefix, prev_sec_local_depth, sec_seg_ptr_ptr, new_sec_dir);
                // }
                // if (new_sec_dir != old_sec_dir) {
                //     help_double_sec_dir(key, new_sec_dir, prev_sec_local_depth);
                //     uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
                //     sec_seg_ptr_ptr = &(new_sec_dir->dir_entries_[new_sec_dir_entry_index]);
                // }
                do {
                    asm("pause");
                    tmp_new_dir = (Directory<T> *)dir_wrapper_->new_dir_;
                    tmp_new_sec_dir = (Directory<T> *)dir_wrapper_->new_sec_dir_;
                    new_dir = (Directory<T> *)tmp_new_dir;
                    new_sec_dir = (Directory<T> *)tmp_new_sec_dir;
                } while ((new_dir == old_dir || new_sec_dir == old_sec_dir) && check_lock_dir());  // wait to allocate new directory
                if (new_dir != old_dir && new_sec_dir != old_sec_dir) {
                    help_double_dir(key, new_dir, prev_local_depth);
                    help_double_sec_dir(key, new_sec_dir, prev_sec_local_depth);
                    // update seg_ptr_ptr on new directory
                    uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
                    seg_ptr_ptr = &(new_dir->dir_entries_[new_dir_entry_index]);
                    uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
                    sec_seg_ptr_ptr = &(new_sec_dir->dir_entries_[new_sec_dir_entry_index]);
                    new_segment = Split(key, segment, prev_local_depth, new_dir, seg_ptr_ptr, new_dir_entry_index, &old_seg_prefix, prev_sec_local_depth, sec_seg_ptr_ptr, new_sec_dir);
                } else {
                    goto RETRY;
                }
            } else {  // main doubling thread
                // find correct directory when doubling has finished🟣一个线程不可能同时倍增和分裂，所有线程都可能走到这里
                new_dir = (Directory<T> *)dir_wrapper_->new_dir_;
                new_sec_dir = (Directory<T> *)dir_wrapper_->new_sec_dir_;
                bool dir_doubling = new_dir != old_dir;
                bool sec_dir_doubling = new_sec_dir != old_sec_dir;
                // if (dir_doubling && sec_dir_doubling) {
                //     uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
                //     uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
                //     bool new_seg_allocated = get_seg_addr(new_dir->dir_entries_[new_dir_entry_index]) != nullptr;
                //     bool new_sec_seg_allocated = get_seg_addr(new_sec_dir->dir_entries_[new_sec_dir_entry_index]) != nullptr;
                //     if (new_seg_allocated && new_sec_seg_allocated) {
                //         dir = new_dir;
                //         dir_entries = dir->dir_entries_;
                //         dir_entry_index = new_dir_entry_index;
                //         seg_ptr_ptr = &(dir_entries[dir_entry_index]);
                //         segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
                //         prev_local_depth = get_local_depth(*seg_ptr_ptr);

                //         sec_dir = new_sec_dir;
                //         sec_dir_entries = sec_dir->dir_entries_;
                //         sec_dir_entry_index = new_sec_dir_entry_index;
                //         sec_seg_ptr_ptr = &(sec_dir_entries[sec_dir_entry_index]);
                //         sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*sec_seg_ptr_ptr));
                //         prev_sec_local_depth = get_local_depth(*sec_seg_ptr_ptr);
                //     } else if ((new_seg_allocated && !new_sec_seg_allocated) || (!new_seg_allocated && new_sec_seg_allocated)) {
                //         // goto RETRY;
                //     }
                // } else if ((dir_doubling && !sec_dir_doubling) || (!dir_doubling && sec_dir_doubling)) {
                //     // goto RETRY;
                // }
                if (dir_doubling) {
                    uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
                    bool new_seg_allocated = get_seg_addr(new_dir->dir_entries_[new_dir_entry_index]) != nullptr;
                    if (new_seg_allocated) {
                        dir = new_dir;
                        dir_entries = dir->dir_entries_;
                        dir_entry_index = new_dir_entry_index;
                        seg_ptr_ptr = &(dir_entries[dir_entry_index]);
                        segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
                        prev_local_depth = get_local_depth(*seg_ptr_ptr);
                    }
                }
                if (sec_dir_doubling) {
                    uint64_t new_sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_sec_dir->global_depth_);
                    bool new_sec_seg_allocated = get_seg_addr(new_sec_dir->dir_entries_[new_sec_dir_entry_index]) != nullptr;
                    if (new_sec_seg_allocated) {
                        sec_dir = new_sec_dir;
                        sec_dir_entries = sec_dir->dir_entries_;
                        sec_dir_entry_index = new_sec_dir_entry_index;
                        sec_seg_ptr_ptr = &(sec_dir_entries[sec_dir_entry_index]);
                        sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*sec_seg_ptr_ptr));
                        prev_sec_local_depth = get_local_depth(*sec_seg_ptr_ptr);
                    }
                }
                new_segment = Split(key, segment, prev_local_depth, dir, seg_ptr_ptr, dir_entry_index, &old_seg_prefix, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
            }
            if (new_segment == NULL) {  // directory is doubled or the segment has been split by other thread
                goto RETRY;
            }
#ifdef SPLIT_LOCK
            segment->release_lock();
#endif
            if (old_seg_prefix == (key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr)))) {
                AAllocator::Persist_flush(new_segment, sizeof(Segment<T>));
            } else {
                AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            }
            flush_split_seg = true;
            goto RETRY;
        }
        return ret;
    }

    template <class T>
    bool ZHASH<T>::Delete(T key, int batch_offset, T *batch_keys, nsTimer *clks) {
        if (batch_offset == 0) {
            clks[0].start();
        }
        uint64_t key_hash = get_key_hash(&key, sizeof(key));
        auto main_bucket_index = key_hash & k_hash_suffix_mask;
        auto old_dir = dir_wrapper_->dir_;
        auto dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_dir->global_depth_);
        auto dir_entries = old_dir->dir_entries_;
        void *seg_ptr = dir_entries[dir_entry_index];
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
        if (batch_offset == 0) {
            void *bucket_addr_0 = &(segment->pairs_[main_bucket_index * k_num_slot_per_bucket]);
            prefetch(bucket_addr_0);
            for (int batch_index = 1; batch_index < batch_size; batch_index++) {
                clks[batch_index].start();
                void *bucket_addr = get_bucket_addr(batch_keys[batch_index]);
                prefetch(bucket_addr);
            }
        }
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if constexpr (!std::is_pointer_v<T>) {
                if (match_key(segment->pairs_[main_bucket_slot].key, key)) {
                    T old_key = segment->pairs_[main_bucket_slot].key;
                    T invalid_key = (T)(uint64_t)old_key & ~k_key_mask;
                    __sync_bool_compare_and_swap(&(segment->pairs_[main_bucket_slot].key), old_key, invalid_key);
                    AAllocator::Persist(&segment->pairs_[main_bucket_slot], sizeof(_Pair<T>));
                    return true;
                }
            }
        }
        // can not find in main bucket, check all fingerprints in main bucket
        if constexpr (!std::is_pointer_v<T>) {
            for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
                unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
                if (match_fingerprint(segment->pairs_[main_bucket_slot].key, key_hash)) {
                    unsigned overflow_bucket_slot = get_position(segment->pairs_[main_bucket_slot].key);
                    if (match_key(segment->pairs_[overflow_bucket_slot].key, key)) {
                        T old_key = segment->pairs_[overflow_bucket_slot].key;
                        T invalid_key = (T)(uint64_t)old_key & ~k_key_mask;
                        __sync_bool_compare_and_swap(&(segment->pairs_[overflow_bucket_slot].key), old_key, invalid_key);
                        AAllocator::Persist(&segment->pairs_[main_bucket_slot], sizeof(_Pair<T>));
                        return true;
                    }
                }
            }
        }
        // can not find in primary segment, try to find in secondary segment
        if constexpr (!std::is_pointer_v<T>) {
            auto old_sec_dir = dir_wrapper_->sec_dir_;
            auto sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_sec_dir->global_depth_);
            auto sec_dir_entries = old_sec_dir->dir_entries_;
            void *sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
            Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
            for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; i++) {
                if (match_key(sec_segment->pairs_[i].key, key)) {
                    T old_key = sec_segment->pairs_[i].key;
                    T invalid_key = (T)(uint64_t)old_key & ~k_key_mask;
                    __sync_bool_compare_and_swap(&(sec_segment->pairs_[i].key), old_key, invalid_key);
                    AAllocator::Persist(&sec_segment->pairs_[i], sizeof(_Pair<T>));
                    return true;
                }
            }
        }
        return false;
    }

    template <class T>
    void *ZHASH<T>::get_bucket_addr(T key) {
        uint64_t key_hash = get_key_hash(&key, sizeof(key));
        auto main_bucket_index = key_hash & k_hash_suffix_mask;
        auto old_dir = dir_wrapper_->dir_;
        auto dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_dir->global_depth_);
        auto dir_entries = old_dir->dir_entries_;
        void *seg_ptr = dir_entries[dir_entry_index];
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
        return &(segment->pairs_[main_bucket_index * k_num_slot_per_bucket]);
    }

    template <class T>
    bool ZHASH<T>::Get(T key, Value_t *value_, int batch_offset, T *batch_keys, nsTimer *clks) {
        if (batch_offset == 0) {
            if (clks)
                clks[0].start();
        }
        uint64_t key_hash = get_key_hash(&key, sizeof(key));
        auto main_bucket_index = key_hash & k_hash_suffix_mask;
    RETRY:
        auto old_dir = dir_wrapper_->dir_;
        auto dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_dir->global_depth_);
        auto dir_entries = old_dir->dir_entries_;
        void *seg_ptr = dir_entries[dir_entry_index];
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
        if (batch_offset == 0) {
            void *bucket_addr_0 = &(segment->pairs_[main_bucket_index * k_num_slot_per_bucket]);
            prefetch(bucket_addr_0);
            for (int batch_index = 1; batch_index < batch_size; batch_index++) {
                if (clks)
                    clks[batch_index].start();
                void *bucket_addr = get_bucket_addr(batch_keys[batch_index]);
                prefetch(bucket_addr);
            }
        }
#ifdef READ_HTM
        int htm_status;
        size_t locking_chunk_size = 0;
        for (int htm_retry = 0; htm_retry < READ_RETRY_TIME; ++htm_retry) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED)
                break;
        }
        if (htm_status != _XBEGIN_STARTED) {
            return false;
        }
#endif
#ifdef READ_LOCK
        size_t locking_chunk_size = segment->acquire_lock(&(dir_entries[dir_entry_index]), dir_wrapper_->dir_->dir_entries_, dir_wrapper_->dir_->global_depth_);
#endif
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_key(segment->pairs_[main_bucket_slot].key, key)) {
                auto value = segment->pairs_[main_bucket_slot].value;
#ifdef VALUE_LENGTH_VARIABLE
                memcpy(value_, value, value_length);
#else
                *value_ = value;
#endif
#ifdef READ_HTM
                if (htm_status == _XBEGIN_STARTED)
                    _xend();
#endif
#ifdef READ_LOCK
                segment->release_lock(&(dir_entries[dir_entry_index]), dir_wrapper_->dir_->dir_entries_, locking_chunk_size);
#endif
                return true;
            }
        }
        // can not find in main bucket, check all fingerprints in main bucket
        if constexpr (!std::is_pointer_v<T>) {
            for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
                unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
                if (match_fingerprint(segment->pairs_[main_bucket_slot].key, key_hash)) {
                    unsigned overflow_bucket_slot = get_position(segment->pairs_[main_bucket_slot].key);
                    if (match_key(segment->pairs_[overflow_bucket_slot].key, key)) {
                        auto value = segment->pairs_[overflow_bucket_slot].value;
#ifdef VALUE_LENGTH_VARIABLE
                        memcpy(value_, value, value_length);
#else
                        *value_ = value;
#endif
#ifdef READ_HTM
                        if (htm_status == _XBEGIN_STARTED)
                            _xend();
#endif
#ifdef READ_LOCK
                        segment->release_lock(&(dir_entries[dir_entry_index]), dir_wrapper_->dir_->dir_entries_, locking_chunk_size);
#endif
                        return true;
                    }
                }
            }
        }
        // can not find in primary segment, try to find in secondary segment
        auto old_sec_dir = dir_wrapper_->sec_dir_;
        auto sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_sec_dir->global_depth_);
        auto sec_dir_entries = old_sec_dir->dir_entries_;
        void *sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
        for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; i++) {
            if (match_key(sec_segment->pairs_[i].key, key)) {
                auto value = sec_segment->pairs_[i].value;
#ifdef VALUE_LENGTH_VARIABLE
                memcpy(value_, value, value_length);
#else
                *value_ = value;
#endif
#ifdef READ_HTM
                if (htm_status == _XBEGIN_STARTED)
                    _xend();
#endif
#ifdef READ_LOCK
                segment->release_lock(&(dir_entries[dir_entry_index]), dir_wrapper_->dir_->dir_entries_, locking_chunk_size);
#endif
                return true;
            }
        }
        // can not find in secondary segment
#ifdef READ_HTM
        if (htm_status == _XBEGIN_STARTED)
            _xend();
#endif
#ifdef READ_LOCK
        segment->release_lock(&(dir_entries[dir_entry_index]), dir_wrapper_->dir_->dir_entries_, locking_chunk_size);
#endif
        return false;
    }
}  // namespace zhash
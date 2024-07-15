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

#include "../../util/compound_pointer.h"
#include "../../util/hash.h"
#include "../../util/pair.h"
#include "../Hash.h"

#define INSERT_HTM
// #define SPLIT_LOCK
#define READ_HTM
#define READ_RETRY_TIME 20

#define VALUE_LENGTH_VARIABLE
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

extern __thread Key_t **cand_keys;
extern __thread Value_t **cand_values;
extern __thread void **cand_segments;
extern __thread int *cand_nums;
extern __thread int **cand_slots;

extern __thread Key_t **sec_cand_keys;
extern __thread Value_t **sec_cand_values;
extern __thread void **sec_cand_segments;
extern __thread int *sec_cand_nums;
extern __thread int **sec_cand_slots;

namespace zhash {
    struct _Pair {
        Key_t key;
        Value_t value;
    };

    const Key_t INVAL = 0;

    constexpr size_t k_segment_bits = 2;
    constexpr size_t k_hash_suffix_mask = (1 << k_segment_bits) - 1;
    constexpr size_t k_segment_size = (1 << k_segment_bits) * 16 * 4;
    constexpr size_t k_metadata_space = 0;
    constexpr size_t k_num_bucket_per_segment = 4;
    constexpr size_t k_num_slot_per_bucket = 4;

    /* metadata in segment addr: 0-5 bits = local_depth; 6 bit = lock */
    constexpr size_t k_depth_bits = 6;
    constexpr size_t k_depth_shift = 64 - k_depth_bits;
    constexpr size_t k_depth_mask = 0xfc00000000000000;
    constexpr size_t k_lock_mask = 0x0200000000000000;
    constexpr size_t k_addr_mask = 0x0000ffffffffffff;

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
        int time = 0;
        while (true) {
            if (!get_seg_lock(*seg_ptr_ptr)) {
                volatile void *old_value = (void *)((uint64_t)*seg_ptr_ptr & ~k_lock_mask);
                volatile void *new_value = (void *)((uint64_t)*seg_ptr_ptr | k_lock_mask);
                if (CAS(seg_ptr_ptr, &old_value, new_value))
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
        return (void *)(clear_addr | (depth << k_depth_shift));
    }

    inline bool var_compare(char *str1, char *str2, int len1, int len2) {
        if (len1 != len2) {
            return false;
        }
        return !memcmp(str1, str2, len1);
    }

    template <class T>
    inline bool match_key(Key_t slot_key, T key, uint64_t key_hash) {
        if constexpr (std::is_pointer_v<T>) {
            return match_fingerprint(slot_key, key_hash) && var_compare((char *)get_addr(slot_key), (char *)&(key->key), get_len(slot_key), key->length);
        } else {
            return slot_key == key;
        }
    }

    template <class T>
    inline void set_key(Key_t *slot_key_ptr, T key, uint64_t key_hash, char *key_addr) {
        if constexpr (std::is_pointer_v<T>) {
            set_fingerprint(slot_key_ptr, key_hash);
            set_len_addr(slot_key_ptr, key->length, (uint64_t)key_addr);
        } else {
            *slot_key_ptr = key;
        }
    }

    inline bool check_key_not_zero(Key_t slot_key) {
        return slot_key != INVAL;
    }

    inline void clear_key(Key_t *slot_key_ptr) {
        *slot_key_ptr = INVAL;
    }

    template <class T>
    struct Directory;

    template <class T>
    struct Segment {
        static const size_t k_num_slot_per_segment = k_segment_size / sizeof(_Pair) - k_metadata_space;

        Segment(void) { memset((void *)&pairs_[0], 255, sizeof(_Pair) * k_num_slot_per_segment); }

        Segment(size_t depth) { memset((void *)&pairs_[0], 255, sizeof(_Pair) * k_num_slot_per_segment); }

        static void New(void **seg_ptr_ptr, size_t depth) {
            auto seg_addr = reinterpret_cast<Segment *>(AAllocator::Allocate_without_proc(sizeof(Segment)));
            memset((void *)&seg_addr->pairs_[0], 0, sizeof(_Pair) * k_num_slot_per_segment);
            *seg_ptr_ptr = construct_seg_ptr(seg_addr, depth);
        }

        ~Segment(void) {}

        int Insert(T, Value_t, size_t, size_t, int, void **, Directory<T> *);
        int sec_insert(T, Value_t, size_t, int, void **, Directory<T> *);
        int Update(Value_t, void **, Directory<T> *, int, bool);
        int get_slot_index(T, size_t loc);
        int get_sec_slot_index(T);
        int get_slot_index_from_cands(T key, int cand_nums, int *cand_slots);
        void insert_for_split(Key_t, Value_t, size_t);
        void rebalance(Segment<T> *sec_segment, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir);

        size_t acquire_lock(void **seg_ptr_ptr, void *dir_entries, size_t global_depth) {
            // get the first entry in the chunk
            char *first_dir_entry_addr = (char *)dir_entries;
            size_t chunk_size = pow(2, global_depth - get_local_depth(*seg_ptr_ptr));
            int dir_entry_index = ((char *)(seg_ptr_ptr)-first_dir_entry_addr) / sizeof(void *);
            if (dir_entry_index < 0) {
                printf("dir_entry_index: %d\n", dir_entry_index);
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
            // release locks in opposite order
            for (int i = chunk_size - 1; i >= 0; i--) {
                release_seg_lock((void **)(first_dir_entry_addr + (dir_entry_index + i) * sizeof(void *)));
            }
            mfence();
        }

        bool check_lock(void *seg_ptr) { return get_seg_lock(seg_ptr); }

        _Pair pairs_[k_num_slot_per_segment];
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
            AAllocator::DAllocate(dir_ptr_ptr, k_cache_line_size, sizeof(Directory) + sizeof(uint64_t) * capacity, callback, reinterpret_cast<void *>(&capacity));
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
            AAllocator::DAllocate(ha_ptr_ptr, k_cache_line_size, sizeof(uint64_t) * num, callback, reinterpret_cast<void *>(&num));
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
            AAllocator::DAllocate((void **)dir_wrapper_ptr_ptr, k_cache_line_size, sizeof(Directory_Wrapper), callback, reinterpret_cast<void *>(&call_args));
        }

        ~Directory_Wrapper(void) {}

        void get_item_num() {
            size_t valid_key_num = 0;
            size_t seg_num = 0;
            Directory<T> *dir = dir_;
            void **dir_entries = dir->dir_entries_;
            void *seg_ptr;
            Segment<T> *segment;
            auto global_depth = dir->global_depth_;
            size_t depth_diff;
            // std::unordered_map<uint64_t, int> exist;
            // std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> location;
            // std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> sec_location;
            // for (uint64_t i = 1; i <= 20000000U; i++) {
            //     exist[i] = 0;
            // }
            for (int dir_entry_index = 0; dir_entry_index < capacity_;) {
                seg_ptr = dir_entries[dir_entry_index];
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
                depth_diff = global_depth - get_local_depth(seg_ptr);
                for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
                    if (check_key_not_zero(segment->pairs_[slot].key)) {
                        ++valid_key_num;
                        // uint64_t key = *(uint64_t *)get_addr(segment->pairs_[slot].key);
                        // uint64_t key_hash = h((void *)get_addr(segment->pairs_[slot].key), sizeof(key));
                        // uint64_t dir_entry_index_1 = key_hash >> (64 - global_depth);
                        // void *seg_ptr_1 = dir_entries[dir_entry_index_1];
                        // Segment<T> *segment_1 = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr_1));
                        // if (segment_1 != segment) {
                        //     printf("\n>>> seg computed %016p != seg read %016p\n\n", segment_1, segment);
                        //     printf("key: %012lx hash: %016lx\n", key, key_hash);
                        //     printf("dir_entry_index: %016lx seg_ptr: %016p seg read: %016p\n", dir_entry_index, seg_ptr, segment);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
                        //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
                        //         uint64_t len = get_len(key_compound);
                        //         uint64_t key_addr = get_addr(key_compound);
                        //         uint64_t value_compound = (uint64_t)(segment->pairs_[slot].value);
                        //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
                        //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
                        //         uint64_t overflow_position = get_overflow_position(value_compound);
                        //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
                        //         uint64_t key_hash = key_addr == 0U ? 0U : h((void *)key_addr, sizeof(key));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\ndir_entry_index: %016lx seg_ptr: %016lx seg computed: %016p\n", dir_entry_index_1, seg_ptr_1, segment_1);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(segment_1->pairs_[slot].key);
                        //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
                        //         uint64_t len = get_len(key_compound);
                        //         uint64_t key_addr = get_addr(key_compound);
                        //         uint64_t value_compound = (uint64_t)(segment_1->pairs_[slot].value);
                        //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
                        //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
                        //         uint64_t overflow_position = get_overflow_position(value_compound);
                        //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
                        //         uint64_t key_hash = key_addr == 0U ? 0U : h((void *)key_addr, sizeof(key));
                        //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
                        //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
                        //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index, main_bucket_index);
                        //     }
                        //     printf("\n\n<<< seg computed %p != seg read %p\n", segment_1, segment);
                        // }
                        // exist[key]++;
                        // if (location.find(key) == location.end()) {
                        //     location[key] = std::vector<std::pair<int, int>>();
                        // }
                        // location[key].push_back(std::make_pair(dir_entry_index, slot));
                    }
                }
                seg_num++;
                dir_entry_index += pow(2, depth_diff);
            }
            Directory<T> *sec_dir = sec_dir_;
            void **sec_dir_entries = sec_dir->dir_entries_;
            void *sec_seg_ptr;
            Segment<T> *sec_segment;
            uint64_t sec_global_depth = sec_dir->global_depth_;
            for (int sec_dir_entry_index = 0; sec_dir_entry_index < 8192 / pri_sec_ratio;) {
                sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
                sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
                depth_diff = sec_global_depth - get_local_depth(sec_seg_ptr);
                if (depth_diff != 0U) {
                    printf("depth_diff: %lu\n", depth_diff);
                }
                for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
                    if (check_key_not_zero(sec_segment->pairs_[slot].key)) {
                        ++valid_key_num;
                        // uint64_t key = *(uint64_t *)get_addr(sec_segment->pairs_[slot].key);
                        // uint64_t key_hash = h((void *)get_addr(sec_segment->pairs_[slot].key), sizeof(key));
                        // uint64_t sec_dir_entry_index_1 = key_hash >> (64 - sec_global_depth);
                        // void *sec_seg_ptr_1 = sec_dir_entries[sec_dir_entry_index_1];
                        // Segment<T> *sec_segment_1 = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr_1));
                        // if (sec_segment_1 != sec_segment) {
                        //     printf("\n>>> sec seg computed %016p != sec seg read %016p\n\n", sec_segment_1, sec_segment);
                        //     printf("key: %012lx hash: %016lx\n", key, key_hash);
                        //     printf("sec_dir_entry_index: %016lx sec_seg_ptr: %016p sec seg read: %016p\n", sec_dir_entry_index, sec_seg_ptr, sec_segment);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
                        //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
                        //         uint64_t len = get_len(key_compound);
                        //         uint64_t key_addr = get_addr(key_compound);
                        //         uint64_t value_compound = (uint64_t)(sec_segment->pairs_[slot].value);
                        //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
                        //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
                        //         uint64_t overflow_position = get_overflow_position(value_compound);
                        //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
                        //         uint64_t key_hash = 0U ? 0U : h((void *)key_addr, sizeof(key));
                        //         uint64_t dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index);
                        //     }
                        //     printf("\nsec_dir_entry_index: %016lx sec_seg_ptr: %016p sec seg computed: %016p\n", sec_dir_entry_index_1, sec_seg_ptr_1, sec_segment_1);
                        //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
                        //         uint64_t key_compound = (uint64_t)(sec_segment_1->pairs_[slot].key);
                        //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
                        //         uint64_t len = get_len(key_compound);
                        //         uint64_t key_addr = get_addr(key_compound);
                        //         uint64_t value_compound = (uint64_t)(sec_segment_1->pairs_[slot].value);
                        //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
                        //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
                        //         uint64_t overflow_position = get_overflow_position(value_compound);
                        //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
                        //         uint64_t key_hash = 0U ? 0U : h((void *)key_addr, sizeof(key));
                        //         uint64_t dir_entry_index = key_hash >> (64 - sec_global_depth);
                        //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index);
                        //     }
                        //     printf("\n\n<<< sec seg computed %016p != sec seg read %016p\n", sec_segment_1, sec_segment);
                        // }
                        // exist[key]++;
                        // if (sec_location.find(key) == sec_location.end()) {
                        //     sec_location[key] = std::vector<std::pair<int, int>>();
                        // }
                        // sec_location[key].push_back(std::make_pair(sec_dir_entry_index, slot));
                    }
                }
                seg_num++;
                sec_dir_entry_index += pow(2, depth_diff);
            }
            std::cout << "#items: " << valid_key_num << std::endl;
            // std::cout << std::fixed << "load_factor: "
            //           << (double)valid_key_num / (seg_num * ((1 << k_segment_bits) * 4 - k_metadata_space))
            //           << std::endl;
            // if (valid_key_num > 5000U) {
            //     for (auto &i : exist) {
            //         uint64_t key = i.first;
            //         int count = i.second;
            //         if (count == 0) {
            //             printf("\n>>> key: %016lx count: %d\n\n", key, count);
            //         } else if (count > 1) {
            //             printf("\n>>> key: %016lx count: %d\n\n", key, count);
            // uint64_t key_hash = h((void *)get_addr(segment->pairs_[slot].key), sizeof(key));
            // uint64_t dir_entry_index = key_hash >> (64 - global_depth);
            // uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
            // segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[dir_entry_index]));
            // printf("hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu seg computed: %016p\n", key_hash, dir_entry_index, main_bucket_index, segment);
            // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
            //     uint64_t key_compound = (uint64_t)(segment->pairs_[slot].key);
            //     uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
            //     uint64_t len = get_len(key_compound);
            //     uint64_t key_addr = get_addr(key_compound);
            //     uint64_t value_compound = (uint64_t)(segment->pairs_[slot].value);
            //     uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
            //     uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
            //     uint64_t overflow_position = get_overflow_position(value_compound);
            //     uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
            //     uint64_t key_hash = h((void *)get_addr(segment->pairs_[slot].key), sizeof(key));
            //     uint64_t dir_entry_index = key_hash >> (64 - global_depth);
            //     uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
            //     printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index, main_bucket_index);
            // }
            // uint64_t sec_dir_entry_index = key_hash >> (64 - sec_global_depth);
            // sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
            // printf("\nhash: %016lx sec_dir_entry_index: %016lx sec seg computed: %016p\n", key_hash, sec_dir_entry_index, sec_segment);
            // for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
            //     uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
            //     uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
            //     uint64_t len = get_len(key_compound);
            //     uint64_t key_addr = get_addr(key_compound);
            //     uint64_t value_compound = (uint64_t)(sec_segment->pairs_[slot].value);
            //     uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
            //     uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
            //     uint64_t overflow_position = get_overflow_position(value_compound);
            //     uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
            //     uint64_t key_hash = h((void *)get_addr(sec_segment->pairs_[slot].key), sizeof(key));
            //     uint64_t dir_entry_index = key_hash >> (64 - sec_global_depth);
            //     printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index);
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
            //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
            //         uint64_t len = get_len(key_compound);
            //         uint64_t key_addr = get_addr(key_compound);
            //         uint64_t value_compound = (uint64_t)(segment->pairs_[slot].value);
            //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
            //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
            //         uint64_t overflow_position = get_overflow_position(value_compound);
            //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
            //         uint64_t key_hash = h((void *)get_addr(segment->pairs_[slot].key), sizeof(key));
            //         uint64_t dir_entry_index = key_hash >> (64 - global_depth);
            //         uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
            //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx main_bucket_index: %02lu\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index, main_bucket_index);
            //     }
            // }
            // for (auto &sec_dir_entry_index_slot : sec_location[key]) {
            //     uint64_t sec_dir_entry_index = sec_dir_entry_index_slot.first;
            //     int slot = sec_dir_entry_index_slot.second;
            //     sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
            //     sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
            //     depth_diff = sec_global_depth - get_local_depth(sec_seg_ptr);
            //     printf("\nsec_dir_entry_index: %016x slot: %02u sec_seg_ptr: %016lx sec seg read: %016p depth_diff: %lu\n", sec_dir_entry_index, slot, sec_seg_ptr, sec_segment, depth_diff);
            //     for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
            //         uint64_t key_compound = (uint64_t)(sec_segment->pairs_[slot].key);
            //         uint64_t fingerprint = (key_compound & fingerprint_mask) >> 52;
            //         uint64_t len = get_len(key_compound);
            //         uint64_t key_addr = get_addr(key_compound);
            //         uint64_t value_compound = (uint64_t)(sec_segment->pairs_[slot].value);
            //         uint64_t overflow_fingerprint = (value_compound & overflow_fingerprint_mask) >> 57;
            //         uint64_t overflow_valid = (value_compound & overflow_valid_mask) >> 56;
            //         uint64_t overflow_position = get_overflow_position(value_compound);
            //         uint64_t key = key_addr == 0U ? 0U : *(uint64_t *)key_addr;
            //         uint64_t key_hash = h((void *)get_addr(sec_segment->pairs_[slot].key), sizeof(key));
            //         uint64_t dir_entry_index = key_hash >> (64 - sec_global_depth);
            //         printf("slot: %02u key_comp: %016lx fp: %03lx len: %01lx key_addr: %012lx value_comp: %016lx of_fp: %02lx of_valid: %01lx of_pos: %01lx key: %016lx hash: %016lx dir_entry_index: %016lx\n", slot, key_compound, fingerprint, len, key_addr, value_compound, overflow_fingerprint, overflow_valid, overflow_position, key, key_hash, dir_entry_index);
            //     }
            // }
            //             printf("\n\n<<< key: %012lx count: %d\n", key, count);
            //         }
            //     }
            // } else {
            //     for (auto &i : exist) {
            //         uint64_t key = i.first;
            //         int count = i.second;
            //         if (count != 0) {
            //             printf("\n>>> key: %lu count: %d\n\n", key, count);
            //         }
            //     }
            // }
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
        void *get_bucket_addr(T key, Segment<T> **batch_segment_ptr = nullptr);
        void *get_sec_bucket_addr(T key, Segment<T> **sec_batch_segment_ptr = nullptr);
        unsigned get_cands(T key, Segment<T> *segment, Key_t *cand_keys, Value_t *cand_values, int *cand_slots = nullptr);
        unsigned get_sec_cands(T key, Segment<T> *sec_segment, Key_t *sec_cand_keys, Value_t *sec_cand_values, int *sec_cand_slots = nullptr);
        void Recovery(void);
        Segment<T> *Split(T key, Segment<T> *old_segment, uint64_t prev_local_depth, Directory<T> *dir, void **seg_ptr_ptr, uint64_t dir_entry_index, size_t *old_seg_prefix, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir);
        void help_double_dir(T key, Directory<T> *new_dir, uint64_t prev_depth);
        void double_dir();
        void update_dir(int dir_entry_index, void *old_seg_ptr, void **new_seg_ptr_ptr, Directory<T> *dir);
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
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_key(pairs_[main_bucket_slot].key, key, key_hash)) {
                return main_bucket_slot;
            }
        }
        // if not found, check overflow fingerprints in main bucket
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_overflow_fingerprint((uint64_t)pairs_[main_bucket_slot].value, key_hash)) {
                unsigned overflow_bucket_slot = get_overflow_position((uint64_t)pairs_[main_bucket_slot].value);
                if (match_key(pairs_[overflow_bucket_slot].key, key, key_hash)) {
                    return overflow_bucket_slot;
                }
            }
        }
        return -1;
    }

    template <class T>
    int Segment<T>::get_sec_slot_index(T key) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        for (unsigned slot = 0; slot < k_num_slot_per_segment; slot++) {
            if (match_key(pairs_[slot].key, key, key_hash)) {
                return slot;
            }
        }
        return -1;
    }

    template <class T>
    int Segment<T>::get_slot_index_from_cands(T key, int cand_nums, int *cand_slots) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        for (unsigned i = 0; i < cand_nums; i++) {
            if (match_key(pairs_[cand_slots[i]].key, key, key_hash)) {
                return cand_slots[i];
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
            if (value_length >= 128) {
                type = value_type_cold_large;  // cold large key, in-place + flush
            } else {
                type = value_type_cold_small;  // cold small key, copy on write
            }
        }
#endif
        if (type == value_type_hot || type == value_type_cold_large) {
            int htm_status = 1;
            int inlock_status;
            size_t locking_chunk_size = 0;
            for (int j = 0; j < update_retry_time; ++j) {
                while (check_lock(*seg_ptr_ptr)) {  // check lock before start HTM
                    asm("pause");
                }
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
                    if (inlock_status == _XBEGIN_STARTED) {
                        break;
                    }
                }
            } else if (check_lock(*seg_ptr_ptr)) {
                _xabort(6);
            }
            // in-place update
            uint64_t *value_addr = (uint64_t *)get_addr((uint64_t)pairs_[slot].value);
            for (int i = 0; i < value_length / sizeof(uint64_t); i++) {
                value_addr[i] = uint64_t(value);
            }
            if (htm_status == _XBEGIN_STARTED) {
                _xend();
            } else {
                if (inlock_status == _XBEGIN_STARTED) {
                    _xend();
                }
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            }
            // flush cold key
            if (type == value_type_cold_large) {
                AAllocator::Persist_flush(value_addr, value_length);
            }
        } else if (type == value_type_cold_small || type == value_type_default) {
            if (type == value_type_cold_small) {
                value = AAllocator::Prepare_value(value, value_length);
            }
        RE_UPDATE:
            int htm_status;
            size_t locking_chunk_size = 0;
            for (int i = 0; i < 2; ++i) {
                while (check_lock(*seg_ptr_ptr)) {
                    asm("pause");
                }
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED) {
                    break;
                }
            }
            if (htm_status != _XBEGIN_STARTED) {
                locking_chunk_size = acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
            } else if (check_lock(*seg_ptr_ptr)) {
                _xend();
                goto RE_UPDATE;
            }
            set_len_addr((uint64_t *)&(pairs_[slot].value), value_length, (uint64_t)value);
            if (htm_status != _XBEGIN_STARTED) {
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            } else {
                _xend();
            }
        }
        return status_dup_insert;
    }

    template <class T>
    int Segment<T>::Insert(T key, Value_t value, size_t main_bucket_index, size_t key_hash, int prev_local_depth, void **seg_ptr_ptr, Directory<T> *dir) {
        int ret = status_seg_insert_error;
        char *key_addr;
#ifdef VALUE_LENGTH_VARIABLE
        value = AAllocator::Prepare_value(value, value_length);
        if constexpr (std::is_pointer_v<T>) {
            key_addr = AAllocator::Prepare_key(key);
        }
#endif
#ifdef INSERT_HTM
        int htm_status;
        size_t locking_chunk_size = 0;
        for (int i = 0; i < 64; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED) {
                break;
            }
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
            if (htm_status != _XBEGIN_STARTED) {
                release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
            } else {
                _xend();
            }
#endif
            return status_key_exist;
        }
        int invalid_main_bucket_slot = -1;
        int overflow_bucket_slot = -1;
        for (unsigned i = 0; i < k_num_slot_per_segment; i++) {
            unsigned slot = (main_bucket_index * k_num_slot_per_bucket + i) % k_num_slot_per_segment;
            if (i < k_num_slot_per_bucket) {  // main bucket
                if (!check_key_not_zero(pairs_[slot].key)) {
                    set_len_addr((uint64_t *)&(pairs_[slot].value), value_length, (uint64_t)value);
                    set_key(&(pairs_[slot].key), key, key_hash, key_addr);
                    ret = status_seg_insert_ok;
                    break;
                } else {
                    if (invalid_main_bucket_slot == -1 && !check_overflow_valid((uint64_t)pairs_[slot].value)) {
                        invalid_main_bucket_slot = slot;
                    }
                }
            } else {  // overflow bucket
                if (invalid_main_bucket_slot == -1) {
                    break;
                }
                if (!check_key_not_zero(pairs_[slot].key)) {
                    overflow_bucket_slot = slot;
                    set_len_addr((uint64_t *)&(pairs_[slot].value), value_length, (uint64_t)value);
                    set_key(&(pairs_[slot].key), key, key_hash, key_addr);
                    set_overflow_fingerprint_position((uint64_t *)&(pairs_[invalid_main_bucket_slot].value), key_hash, slot);
                    ret = status_seg_insert_ok;
                    break;
                }
            }
        }
#ifdef INSERT_HTM
        if (htm_status != _XBEGIN_STARTED) {
            release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
        } else {
            _xend();
        }
#endif
        if (overflow_bucket_slot != -1) {
            AAllocator::Persist_asyn_flush(&(pairs_[main_bucket_index * k_num_slot_per_bucket]), 64);
            AAllocator::Persist_asyn_flush(&(pairs_[overflow_bucket_slot / k_num_slot_per_bucket * k_num_slot_per_bucket]), 64);
        }
        return ret;
    }

    template <class T>
    int Segment<T>::sec_insert(T key, Value_t value, size_t key_hash, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir) {
        int ret = status_seg_insert_error;
        char *key_addr;
#ifdef VALUE_LENGTH_VARIABLE
        value = AAllocator::Prepare_value(value, value_length);
        if constexpr (std::is_pointer_v<T>) {
            key_addr = AAllocator::Prepare_key(key);
        }
#endif
#ifdef INSERT_HTM
        int htm_status;
        size_t locking_chunk_size = 0;
        for (int i = 0; i < 64; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED) {
                break;
            }
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
            else {
                _xend();
            }
#endif
            return status_key_exist;
        }
        for (unsigned slot = 0; slot < k_num_slot_per_segment; slot++) {
            if (!check_key_not_zero(pairs_[slot].key)) {
                set_len_addr((uint64_t *)&(pairs_[slot].value), value_length, (uint64_t)value);
                set_key(&(pairs_[slot].key), key, key_hash, key_addr);
                ret = status_seg_insert_ok;
                break;
            }
        }
#ifdef INSERT_HTM
        if (htm_status != _XBEGIN_STARTED) {
            release_lock(sec_seg_ptr_ptr, sec_dir->dir_entries_, locking_chunk_size);
        } else {
            _xend();
        }
#endif
        return ret;
    }

    template <class T>
    void Segment<T>::rebalance(Segment<T> *sec_segment, int prev_sec_local_depth, void **sec_seg_ptr_ptr, Directory<T> *sec_dir) {
        _Pair overflow_pairs[k_num_slot_per_segment];
        uint64_t overflow_hashs[k_num_slot_per_segment];
        size_t num_overflow_pair = 0;
        size_t pair_nums[4] = {0, 0, 0, 0};
        // find all overflow pairs
        for (unsigned i = 0; i < k_num_slot_per_segment; i++) {
            if (check_key_not_zero(pairs_[i].key)) {
                uint64_t key_hash;
                if constexpr (std::is_pointer_v<T>) {
                    key_hash = h((void *)get_addr(pairs_[i].key), get_len(pairs_[i].key));
                } else {
                    key_hash = h(&(pairs_[i].key), sizeof(Key_t));
                }
                if ((i / k_num_bucket_per_segment) != (key_hash & k_hash_suffix_mask)) {
                    overflow_pairs[num_overflow_pair] = pairs_[i];
                    overflow_hashs[num_overflow_pair] = key_hash;
                    num_overflow_pair++;
                    clear_key(&(pairs_[i].key));
                } else {
                    pair_nums[i / k_num_bucket_per_segment]++;
                }
            }
        }
        // try to bring overflow pairs back to their main buckets
        for (unsigned i = 0; i < num_overflow_pair; i++) {
            size_t main_bucket_index = overflow_hashs[i] & k_hash_suffix_mask;
            for (unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket; main_bucket_slot < (main_bucket_index + 1) * k_num_slot_per_bucket; main_bucket_slot++) {
                if (!check_key_not_zero(pairs_[main_bucket_slot].key)) {
                    pairs_[main_bucket_slot] = overflow_pairs[i];  // value in overflow_pairs is already cleaned without overflow info
                    overflow_pairs[i].key = INVAL;
                    pair_nums[main_bucket_index]++;
                    break;
                }
            }
        }
        // bring the remaining overflow pairs to overflow buckets
        for (unsigned i = 0; i < num_overflow_pair; i++) {
            if (overflow_pairs[i].key != INVAL) {
                size_t main_bucket_index = overflow_hashs[i] & k_hash_suffix_mask;
                int overflow_bucket_slot = -1;
                int final_slot = -1;
                // find the most empty outing bucket
                size_t most_empty_bucket = (main_bucket_index + 1) % k_num_bucket_per_segment;
                for (unsigned j = main_bucket_index + 2; j < main_bucket_index + 4; j++) {
                    if (pair_nums[j % k_num_bucket_per_segment] < pair_nums[most_empty_bucket]) {
                        most_empty_bucket = j % k_num_bucket_per_segment;
                    }
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
                    T key;
                    if constexpr (std::is_pointer_v<T>) {
                        // key = new typename std::remove_pointer<T>::type();
                        char *key_addr = (char *)get_addr(overflow_pairs[i].key);
                        uint64_t key_len = get_len(overflow_pairs[i].key);
                        char *mem = new char[sizeof(string_key) + key_len];
                        key = reinterpret_cast<string_key *>(mem);
                        key->length = key_len;
                        memcpy(key->key, key_addr, key_len);
                    } else {
                        key = get_addr(overflow_pairs[i].key);
                    }
                    int ret = sec_segment->sec_insert(key, overflow_pairs[i].value, overflow_hashs[i], prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
                    // if constexpr (std::is_pointer_v<T>) {
                    //     delete key;
                    // }
                    continue;
                }
                // select the first invalid slot in main bucket
                for (unsigned slot = main_bucket_index * k_num_slot_per_bucket; slot < (main_bucket_index + 1) * k_num_slot_per_bucket; slot++) {
                    if (!check_overflow_valid((uint64_t)pairs_[slot].value)) {
                        // set_overflow_fingerprint_position((uint64_t *)&(pairs_[slot].value), overflow_hashs[i], overflow_bucket_slot);
                        final_slot = slot;
                        break;
                    }
                }
                // assert(final_slot >= 0);
                if (final_slot < 0) {
                    T key;
                    if constexpr (std::is_pointer_v<T>) {
                        // key = new typename std::remove_pointer<T>::type();
                        char *key_addr = (char *)get_addr(overflow_pairs[i].key);
                        uint64_t key_len = get_len(overflow_pairs[i].key);
                        char *mem = new char[sizeof(string_key) + key_len];
                        key = reinterpret_cast<string_key *>(mem);
                        key->length = key_len;
                        memcpy(key->key, key_addr, key_len);
                    } else {
                        key = get_addr(overflow_pairs[i].key);
                    }
                    int ret = sec_segment->sec_insert(key, overflow_pairs[i].value, overflow_hashs[i], prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
                    // if constexpr (std::is_pointer_v<T>) {
                    //     delete key;
                    // }
                    continue;
                }
                // insert pair into the most empty overflow bucket
                pairs_[overflow_bucket_slot] = overflow_pairs[i];
                pair_nums[most_empty_bucket]++;
                // set fingerprint and position in main bucket
                set_overflow_fingerprint_position((uint64_t *)&(pairs_[final_slot].value), overflow_hashs[i], overflow_bucket_slot);
            }
        }
    }

    template <class T>
    void Segment<T>::insert_for_split(Key_t key, Value_t value, size_t slot) {
        pairs_[slot].value = value;
        pairs_[slot].key = key;
    }

    template <class T>
    ZHASH<T>::ZHASH(int init_cap) {
        Directory_Wrapper<T>::New(&dir_wrapper_, init_cap);
        Hot_array<T>::New((void **)(&dir_wrapper_->hot_arr_), hot_num);

        Directory<T>::New(&dir_wrapper_->new_dir_, init_cap);
        dir_wrapper_->dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
        auto dir_entries = dir_wrapper_->dir_->dir_entries_;
        for (int dir_entry_index = 0; dir_entry_index < dir_wrapper_->capacity_; ++dir_entry_index) {
            Segment<T>::New(&dir_entries[dir_entry_index], dir_wrapper_->dir_->global_depth_);
        }

        Directory<T>::New(&dir_wrapper_->new_sec_dir_, init_cap / pri_sec_ratio);
        dir_wrapper_->sec_dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_sec_dir_);
        auto sec_dir_entries = dir_wrapper_->sec_dir_->dir_entries_;
        for (int i = 0; i < dir_wrapper_->capacity_ / pri_sec_ratio; ++i) {
            Segment<T>::New(&sec_dir_entries[i], dir_wrapper_->sec_dir_->global_depth_);
        }
        printf("Segment size: %ld\n", sizeof(Segment<T>));
        printf("Segment slots size: %ld\n", sizeof(Segment<T>::pairs_));
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
        size_t key_hash;
        Key_t key_number;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
            uint64_t *key_addr = (uint64_t *)key->key;
            key_number = key_addr[0];
        } else {
            key_hash = h(&key, sizeof(key));
            key_number = key;
        }
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; ++i) {
            if (hot_keys[i] == key_number) {
                return true;
            } else if (hot_keys[i] == 0) {
                update_hot_keys(key);
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ZHASH<T>::check_hot_without_update(T key) {
        size_t key_hash;
        Key_t key_number;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
            uint64_t *key_addr = (uint64_t *)key->key;
            key_number = key_addr[0];
        } else {
            key_hash = h(&key, sizeof(key));
            key_number = key;
        }
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; ++i) {
            if (hot_keys[i] == key_number) {
                return true;
            } else if (hot_keys[i] == 0) {
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ZHASH<T>::update_hot_keys(T key) {
        size_t key_hash;
        Key_t key_number;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
            uint64_t *key_addr = (uint64_t *)key->key;
            key_number = key_addr[0];
        } else {
            key_hash = h(&key, sizeof(key));
            key_number = key;
        }
        int idx = key_hash >> (64 - hot_bit);
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        bool found_key = false;
        idx = idx - (idx % asso);
        for (int i = idx; i < idx + asso; i++) {
            if (hot_keys[i] == 0) {
                hot_keys[i] = key_number;
                found_key = true;
                break;
            }
            if (hot_keys[i] == key_number) {
                found_key = true;
                if (i != idx) {
                    int status = _xbegin();
                    if (status == _XBEGIN_STARTED) {
                        hot_keys[i] = hot_keys[i - 1];
                        hot_keys[i - 1] = key_number;
                        _xend();
                    }
                }
                break;
            }
        }
        if (!found_key) {
            hot_keys[idx + asso - 1] = key_number;
        }
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
                if (hot_keys[i] < uint64_t(hot_num)) {
                    top_100_count++;
                }
                if (hot_keys[i] < uint64_t(hot_num) / 10) {
                    top_10_count++;
                }
                if (hot_keys[i] < uint64_t(hot_num) / 100) {
                    top_1_count++;
                }
            }
        }
        printf("Top 100% Hot number:%f\n", double(top_100_count) / uint64_t(hot_num));
        printf("Top 10% Hot number:%f\n", double(top_10_count) / uint64_t(hot_num) * 10);
        printf("Top 1% Hot number:%f\n", double(top_1_count) / uint64_t(hot_num) * 100);
    }

    template <class T>
    void ZHASH<T>::set_hot() {
        int global_depth = dir_wrapper_->dir_->global_depth_;
        uint64_t num_dir_entry = 1llu << global_depth;
        uint64_t num_dir_entry_per_hot_key = num_dir_entry / uint64_t(hot_num);
        Segment<T> *segment;
        Segment<T> *prev_segment = NULL;
        Segment<T> **dir_entries = (Segment<T> **)dir_wrapper_->dir_->dir_entries_;
        uint64_t curr_key;
        uint64_t *hot_keys = dir_wrapper_->hot_arr_->hot_keys_;
        printf("[set_hot] global depth:%d\n", global_depth);
        printf("[set_hot] directory entry:%lu\n", num_dir_entry);
        printf("[set_hot] hot number:%lu\n", hot_num);
        printf("[set_hot] seg num:%lu\n", num_dir_entry_per_hot_key);
        for (int i = 0; i < uint64_t(hot_num); i += asso) {
            for (int j = i; j < i + asso; ++j) {
                hot_keys[j] = 20000000;
            }
            for (int j = i * num_dir_entry_per_hot_key; j < (i + asso) * num_dir_entry_per_hot_key; ++j) {
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entries[j]));
                if (segment == prev_segment) {
                    continue;
                }
                for (unsigned k = 0; k < Segment<T>::k_num_slot_per_segment; ++k) {
                    if (check_key_not_zero(segment->pairs_[k].key)) {
                        if constexpr (std::is_pointer_v<T>) {
                            uint64_t *key_addr = (uint64_t *)get_addr(segment->pairs_[k].key);
                            curr_key = key_addr[0];
                        } else {
                            curr_key = segment->pairs_[k].key;
                        }
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
                printf("doubling number %lx global depth %d\n", uint64_t(key), prev_global_depth);
                double_dir();
            }
            unlock_dir();
            return NULL;
        }
        size_t locking_chunk_size = 0;
        size_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        void *new_seg_ptr;
        void **new_seg_ptr_ptr = &new_seg_ptr;
        Segment<T>::New(new_seg_ptr_ptr, get_local_depth(*seg_ptr_ptr));
        Segment<T> *new_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*new_seg_ptr_ptr));
        int htm_status;
        for (int i = 0; i < 2; ++i) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED) {
                break;
            }
        }
        if (htm_status != _XBEGIN_STARTED) {
            locking_chunk_size = old_segment->acquire_lock(seg_ptr_ptr, dir->dir_entries_, dir->global_depth_);
            if (get_local_depth(*seg_ptr_ptr) != prev_local_depth) {
                old_segment->release_lock(seg_ptr_ptr, dir->dir_entries_, locking_chunk_size);
                return NULL;
            }
        } else if (old_segment->check_lock(*seg_ptr_ptr) || get_local_depth(*seg_ptr_ptr) != prev_local_depth || (check_lock_dir() && (Directory<T> *)dir_wrapper_->new_dir_ != dir)) {
            // Ensure that the segment is not locked by the fall back path of HTM
            // Ensure that the split of segment has not been finished by other threads
            // Ensure that the doubling doesn't happen after the split begins
            _xend();
            return NULL;
        }
        uint64_t common_prefix = dir_entry_index >> (prev_global_depth - prev_local_depth);
        uint64_t old_seg_prefix1 = common_prefix << 1;
        uint64_t new_seg_prefix = (common_prefix << 1) + 1;
        Directory<T> *old_sec_dir = dir_wrapper_->sec_dir_;
        uint64_t sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - old_sec_dir->global_depth_);
        void **sec_dir_entries = old_sec_dir->dir_entries_;
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_dir_entries[sec_dir_entry_index]));
        for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
            if (check_key_not_zero(old_segment->pairs_[slot].key)) {
                uint64_t key_hash;
                if constexpr (std::is_pointer_v<T>) {
                    key_hash = h((void *)get_addr(old_segment->pairs_[slot].key), get_len(old_segment->pairs_[slot].key));
                } else {
                    key_hash = h(&(old_segment->pairs_[slot].key), sizeof(Key_t));
                }
                // if constexpr (std::is_pointer_v<T>) {
                //     if (*(uint64_t *)get_addr(old_segment->pairs_[slot].key) == 0xda24) {
                //         printf("split old segment %p slot %u key %lx\n", sec_segment, slot, *(uint64_t *)get_addr(old_segment->pairs_[slot].key));
                //     }
                // }
                clear_overflow((uint64_t *)&(old_segment->pairs_[slot].value));
                uint64_t hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
                if (hash_prefix == new_seg_prefix) {
                    // move only clear key without fingerprints
                    new_segment->insert_for_split(old_segment->pairs_[slot].key, old_segment->pairs_[slot].value, slot);
                    clear_key(&(old_segment->pairs_[slot].key));
                }
            }
        }
        for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; ++slot) {
            if (check_key_not_zero(sec_segment->pairs_[slot].key)) {
                uint64_t key_hash;
                if constexpr (std::is_pointer_v<T>) {
                    key_hash = h((void *)get_addr(sec_segment->pairs_[slot].key), get_len(sec_segment->pairs_[slot].key));
                } else {
                    key_hash = h(&(sec_segment->pairs_[slot].key), sizeof(Key_t));
                }
                uint64_t hash_prefix = key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr) - 1);
                // if constexpr (std::is_pointer_v<T>) {
                //     if (*(uint64_t *)get_addr(sec_segment->pairs_[slot].key) == 0xda24) {
                //         printf("split secondary segment %p slot %u key %lx hash %lx hash_prefix %lx\n", sec_segment, slot, *(uint64_t *)get_addr(sec_segment->pairs_[slot].key), key_hash, hash_prefix);
                //     }
                // }
                if (hash_prefix == new_seg_prefix) {
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        if (!check_key_not_zero(new_segment->pairs_[i].key)) {
                            // if constexpr (std::is_pointer_v<T>) {
                            //     if (*(uint64_t *)get_addr(sec_segment->pairs_[slot].key) == 0xda24) {
                            //         printf("move secondary segment %p slot %u key %lx to new segment %p slot %u\n", sec_segment, slot, *(uint64_t *)get_addr(sec_segment->pairs_[slot].key), new_segment, i);
                            //     }
                            // }
                            new_segment->insert_for_split(sec_segment->pairs_[slot].key, sec_segment->pairs_[slot].value, i);
                            clear_key(&(sec_segment->pairs_[slot].key));
                            break;
                        }
                    }
                } else if (hash_prefix == old_seg_prefix1) {
                    for (unsigned i = 0; i < Segment<T>::k_num_slot_per_segment; ++i) {
                        if (!check_key_not_zero(old_segment->pairs_[i].key)) {
                            // if constexpr (std::is_pointer_v<T>) {
                            //     if (*(uint64_t *)get_addr(sec_segment->pairs_[slot].key) == 0xda24) {
                            //         printf("move secondary segment %p slot %u key %lx to old segment %p slot %u\n", sec_segment, slot, *(uint64_t *)get_addr(sec_segment->pairs_[slot].key), old_segment, i);
                            //     }
                            // }
                            old_segment->insert_for_split(sec_segment->pairs_[slot].key, sec_segment->pairs_[slot].value, i);
                            clear_key(&(sec_segment->pairs_[slot].key));
                            break;
                        }
                    }
                }
            }
        }
        new_segment->rebalance(sec_segment, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
        old_segment->rebalance(sec_segment, prev_sec_local_depth, sec_seg_ptr_ptr, sec_dir);
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
        size_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
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
                if (htm_status == _XBEGIN_STARTED) {
                    break;
                } else {
                    asm("pause");
                }
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
    void ZHASH<T>::double_dir() {
        Directory<T> *dir = dir_wrapper_->dir_;
        void **dir_entries = dir->dir_entries_;
        void *new_seg_ptr;
        void **new_dir_ptr_ptr = &new_seg_ptr;
        Directory<T>::New(new_dir_ptr_ptr, 2 * dir_wrapper_->capacity_);
        mfence();
        uint64_t pm_alloc = AAllocator::total_pm_alloc();
        uint64_t dram_alloc = 2 * dir_wrapper_->capacity_ + 8192U / pri_sec_ratio;
        printf("At this doubling, DRAM: %lu, PM: %lu, DRAM/PM: %.3f\n", dram_alloc, pm_alloc, double(dram_alloc) / pm_alloc);
        dir_wrapper_->new_dir_ = *new_dir_ptr_ptr;
        mfence();
        auto new_dir = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
        auto new_dir_entries = new_dir->dir_entries_;
        int htm_status;
        bool new_dir_entry_not_null;
        for (unsigned i = 0; i < dir_wrapper_->capacity_; i += 4) {
            new_dir_entry_not_null = false;
            for (int i = 0; i < 8; ++i) {
                htm_status = _xbegin();
                if (htm_status == _XBEGIN_STARTED) {
                    break;
                } else {
                    asm("pause");
                }
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
            if (!new_dir_entry_not_null)
                for (int j = i; j < i + 4; ++j) {
                    new_dir_entries[2 * j] = dir_entries[j];
                    new_dir_entries[2 * j + 1] = dir_entries[j];
                }

            if (htm_status != _XBEGIN_STARTED) {
                Unlock();
            } else {
                _xend();
            }
        }
        dir_wrapper_->dir_ = reinterpret_cast<Directory<T> *>(dir_wrapper_->new_dir_);
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
    bool ZHASH<T>::check_lock_dir() {
        return dir_wrapper_->lock_;
    }

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
        // update all the local depth in the chunk
        for (int i = 0; i < chunk_size; ++i) {
            set_local_depth(&dir_entries[dir_entry_index + i], old_local_depth + 1);
        }
    }

    template <class T>
    int ZHASH<T>::Insert(T key, Value_t value, int batch_offset, T *batch_keys, nsTimer *clks) {
        if (batch_offset == 0) {
            Segment<T> *batch_segments[batch_size];
            for (unsigned i = 0; i < batch_size; i++) {  // clear candidate slots
                cand_nums[i] = -1;
                batch_segments[i] = nullptr;
            }
            // determine segment addresses + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                if (batch_index == 0 || !check_hot_without_update(batch_keys[batch_index])) {
                    void *p = get_bucket_addr(batch_keys[batch_index], &batch_segments[batch_index]);
                    cand_segments[batch_index] = (void *)(batch_segments[batch_index]);
                    prefetch(p);
                }
            }
            // determine candidate slots + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                if (batch_segments[batch_index] != nullptr) {
                    cand_nums[batch_index] = get_cands(batch_keys[batch_index], batch_segments[batch_index], cand_keys[batch_index], cand_values[batch_index], cand_slots[batch_index]);
                    for (unsigned cand_index = 0; cand_index < cand_nums[batch_index]; cand_index++) {
                        if constexpr (std::is_pointer_v<T>) {
                            prefetch((void *)get_addr(cand_keys[batch_index][cand_index]));
                        }
                        prefetch((void *)get_addr((uint64_t)cand_values[batch_index][cand_index]));
                    }
                }
            }
            Segment<T> *batch_sec_segments[batch_size];
            for (unsigned i = 0; i < batch_size; i++) {  // clear candidate slots
                sec_cand_nums[i] = -1;
                batch_sec_segments[i] = nullptr;
            }
            // determine secondary segment addresses + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                if (batch_index == 0 || !check_hot_without_update(batch_keys[batch_index])) {
                    void *p = get_sec_bucket_addr(batch_keys[batch_index], &batch_sec_segments[batch_index]);
                    sec_cand_segments[batch_index] = (void *)(batch_sec_segments[batch_index]);
                    prefetch(p);
                }
            }
            // determine secondary candidate slots + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                if (batch_sec_segments[batch_index] != nullptr) {
                    sec_cand_nums[batch_index] = get_sec_cands(batch_keys[batch_index], batch_sec_segments[batch_index], sec_cand_keys[batch_index], sec_cand_values[batch_index], sec_cand_slots[batch_index]);
                    for (unsigned sec_cand_index = 0; sec_cand_index < sec_cand_nums[batch_index]; sec_cand_index++) {
                        if constexpr (std::is_pointer_v<T>) {
                            prefetch((void *)get_addr(sec_cand_keys[batch_index][sec_cand_index]));
                        }
                        prefetch((void *)get_addr((uint64_t)sec_cand_values[batch_index][sec_cand_index]));
                    }
                }
            }
        }
        bool flush_split_seg;
        bool recheck = false;
    STARTOVER:
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
    RETRY:
        Directory<T> *dir = dir_wrapper_->dir_;
        void **dir_entries = dir->dir_entries_;
        uint64_t dir_entry_index = key_hash >> (8 * sizeof(key_hash) - dir->global_depth_);
        void **seg_ptr_ptr = &(dir_entries[dir_entry_index]);
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
        uint64_t prev_local_depth = get_local_depth(*seg_ptr_ptr);

        Directory<T> *sec_dir = dir_wrapper_->sec_dir_;
        void **sec_dir_entries = sec_dir->dir_entries_;
        uint64_t sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - sec_dir->global_depth_);
        void **sec_seg_ptr_ptr = &(sec_dir_entries[sec_dir_entry_index]);
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*sec_seg_ptr_ptr));
        uint64_t prev_sec_local_depth = get_local_depth(*sec_seg_ptr_ptr);

        int ret = 0;
        Directory<T> *new_dir;
        // find correct directory when doubling is happening
        if (check_lock_dir() && (new_dir = (Directory<T> *)(dir_wrapper_->new_dir_)) != dir) {
            uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
            if (new_dir->dir_entries_[new_dir_entry_index] != NULL) {
                dir = new_dir;
                dir_entries = dir->dir_entries_;
                dir_entry_index = new_dir_entry_index;
                seg_ptr_ptr = &(dir_entries[dir_entry_index]);
                segment = reinterpret_cast<Segment<T> *>(get_seg_addr(*seg_ptr_ptr));
                prev_local_depth = get_local_depth(*seg_ptr_ptr);
            }
        }
        if (get_seg_addr(dir_entries[dir_entry_index]) != segment) {
            printf("depth and dentry mismatch\n");
            asm("pause");
            goto RETRY;
        }
    Check:
        int slot;
        if (recheck || !cand_nums || cand_nums[batch_offset] == -1 || cand_segments[batch_offset] != (void *)segment) {
            slot = segment->get_slot_index(key, main_bucket_index);
        } else {
            slot = segment->get_slot_index_from_cands(key, cand_nums[batch_offset], cand_slots[batch_offset]);
        }
        if (slot != -1) {  // found in primary segment
            bool hot = false;
#ifdef VALUE_LENGTH_VARIABLE
            if (request % FREQ == 0) {
                update_hot_keys(key);
            }
            request++;
            hot = check_hot(key);
#endif
            ret = segment->Update(value, seg_ptr_ptr, dir_wrapper_->dir_, slot, hot);
            if (flush_split_seg) {
                AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            }
            return -1;
        }
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("can not find %lx in primary segment %p, try to find %lx in secondary segment %p\n", *(uint64_t *)(key->key), segment, *(uint64_t *)(key->key), sec_segment);
        //     }
        // }
        // can not find in primary segment, try to find in secondary segment
        if (recheck || !sec_cand_nums || sec_cand_nums[batch_offset] == -1 || sec_cand_segments[batch_offset] != (void *)sec_segment) {
            slot = sec_segment->get_sec_slot_index(key);
        } else {
            slot = sec_segment->get_slot_index_from_cands(key, sec_cand_nums[batch_offset], sec_cand_slots[batch_offset]);
        }
        if (slot != -1) {  // found in secondary segment
            bool hot = false;
#ifdef VALUE_LENGTH_VARIABLE
            if (request % FREQ == 0) {
                update_hot_keys(key);
            }
            request++;
            hot = check_hot(key);
#endif
            ret = sec_segment->Update(value, sec_seg_ptr_ptr, dir_wrapper_->sec_dir_, slot, hot);
            if (flush_split_seg) {
                AAllocator::Persist_flush(segment, sizeof(Segment<T>));
            }
            return -1;
        }
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("can not find %lx in secondary segment %p, try to insert %lx into primary segment %p\n", *(uint64_t *)(key->key), sec_segment, *(uint64_t *)(key->key), segment);
        //     }
        // }
        // can not find in secondary segment, try to insert into primary segment
        ret = segment->Insert(key, value, main_bucket_index, key_hash, prev_local_depth, seg_ptr_ptr, dir_wrapper_->dir_);
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("insert %lx into primary segment %p return %d\n", *(uint64_t *)(key->key), segment, ret);
        //     }
        // }
        if (ret == status_key_exist) {
            recheck = true;
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
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("primary segment %p has no room to insert %lx, try to insert %lx into secondary segment %p\n", segment, *(uint64_t *)(key->key), *(uint64_t *)(key->key), sec_segment);
        //     }
        // }
        // primary segment has no room to insert, try to insert into secondary segment
        ret = sec_segment->sec_insert(key, value, key_hash, prev_sec_local_depth, sec_seg_ptr_ptr, dir_wrapper_->sec_dir_);
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("insert %lx into secondary segment %p return %d\n", *(uint64_t *)(key->key), sec_segment, ret);
        //     }
        // }
        if (ret == status_key_exist) {
            recheck = true;
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
        // if constexpr (std::is_pointer_v<T>) {
        //     if (*(uint64_t *)(key->key) == 0xda24) {
        //         printf("secondary segment %p has no room to insert %lx, split primary segment %p\n", sec_segment, *(uint64_t *)(key->key), segment);
        //     }
        // }
        // secondary segment has no room to insert, split primary segment
        if (ret == status_seg_insert_error) {
#ifdef SPLIT_LOCK
            if (!target->try_get_lock()) {
                goto RETRY;
            }
#endif
            Segment<T> *new_segment;
            size_t old_seg_prefix;
            if (check_lock_dir()) {  // directory is doubling
                volatile Directory<T> *tmp_new_dir;
                do {
                    asm("pause");
                    tmp_new_dir = (Directory<T> *)dir_wrapper_->new_dir_;
                    new_dir = (Directory<T> *)tmp_new_dir;
                } while (new_dir == dir_wrapper_->dir_ && check_lock_dir());  // wait to allocate new directory
                if (new_dir != dir_wrapper_->dir_) {
                    help_double_dir(key, new_dir, prev_local_depth);
                    uint64_t new_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - new_dir->global_depth_);
                    seg_ptr_ptr = &(new_dir->dir_entries_[new_dir_entry_index]);
                    new_segment = Split(key, segment, prev_local_depth, new_dir, seg_ptr_ptr, new_dir_entry_index, &old_seg_prefix, prev_sec_local_depth, sec_seg_ptr_ptr, dir_wrapper_->sec_dir_);
                } else {
                    goto RETRY;
                }
            } else {  // directory is not doubling
                new_segment = Split(key, segment, prev_local_depth, dir, seg_ptr_ptr, dir_entry_index, &old_seg_prefix, prev_sec_local_depth, sec_seg_ptr_ptr, dir_wrapper_->sec_dir_);
            }
            if (new_segment == NULL) {  // segment has been split by other thread
                goto RETRY;
            }
#ifdef SPLIT_LOCK
            target->release_lock();
#endif
            if (old_seg_prefix == key_hash >> (8 * sizeof(key_hash) - get_local_depth(*seg_ptr_ptr))) {
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
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
        Directory<T> *dir = dir_wrapper_->dir_;
        uint64_t dir_entry_index = key_hash >> (8 * sizeof(key_hash) - dir->global_depth_);
        void **dir_entries = dir->dir_entries_;
        void *seg_ptr = dir_entries[dir_entry_index];
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_key(segment->pairs_[main_bucket_slot].key, key, key_hash)) {
                Key_t old_key = segment->pairs_[main_bucket_slot].key;
                Key_t invalid_key = INVAL;
                __sync_bool_compare_and_swap(&(segment->pairs_[main_bucket_slot].key), old_key, invalid_key);
                AAllocator::Persist(&segment->pairs_[main_bucket_slot], sizeof(_Pair));
                return true;
            }
        }
        // cannot find in main bucket, check all fingerprints in main bucket
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_overflow_fingerprint((uint64_t)segment->pairs_[main_bucket_slot].value, key_hash)) {
                unsigned overflow_bucket_slot = get_overflow_position((uint64_t)segment->pairs_[main_bucket_slot].value);
                if (match_key(segment->pairs_[overflow_bucket_slot].key, key, key_hash)) {
                    Key_t old_key = segment->pairs_[overflow_bucket_slot].key;
                    Key_t invalid_key = INVAL;
                    __sync_bool_compare_and_swap(&(segment->pairs_[overflow_bucket_slot].key), old_key, invalid_key);
                    AAllocator::Persist(&segment->pairs_[main_bucket_slot], sizeof(_Pair));
                    return true;
                }
            }
        }
        // can not find in primary segment, try to find in secondary segment
        Directory<T> *sec_dir = dir_wrapper_->sec_dir_;
        uint64_t sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - sec_dir->global_depth_);
        void **sec_dir_entries = sec_dir->dir_entries_;
        void *sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
        for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
            if (match_key(sec_segment->pairs_[slot].key, key, key_hash)) {
                Key_t old_key = sec_segment->pairs_[slot].key;
                Key_t invalid_key = INVAL;
                __sync_bool_compare_and_swap(&(sec_segment->pairs_[slot].key), old_key, invalid_key);
                AAllocator::Persist(&sec_segment->pairs_[slot], sizeof(_Pair));
                return true;
            }
        }
        return false;
    }

    template <class T>
    void *ZHASH<T>::get_bucket_addr(T key, Segment<T> **batch_segment_ptr) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        uint64_t main_bucket_index = key_hash & k_hash_suffix_mask;
        Directory<T> *dir = dir_wrapper_->dir_;
        uint64_t dir_entry_index = key_hash >> (8 * sizeof(key_hash) - dir->global_depth_);
        void **dir_entries = dir->dir_entries_;
        void *seg_ptr = dir_entries[dir_entry_index];
        Segment<T> *segment = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_ptr));
        if (batch_segment_ptr != nullptr) {
            *batch_segment_ptr = segment;
        }
        return &(segment->pairs_[main_bucket_index * k_num_slot_per_bucket]);
    }

    template <class T>
    void *ZHASH<T>::get_sec_bucket_addr(T key, Segment<T> **sec_batch_segment_ptr) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        auto sec_dir = dir_wrapper_->sec_dir_;
        auto sec_dir_entry_index = key_hash >> (8 * sizeof(key_hash) - sec_dir->global_depth_);
        auto sec_dir_entries = sec_dir->dir_entries_;
        void *sec_seg_ptr = sec_dir_entries[sec_dir_entry_index];
        Segment<T> *sec_segment = reinterpret_cast<Segment<T> *>(get_seg_addr(sec_seg_ptr));
        if (sec_batch_segment_ptr != nullptr) {
            *sec_batch_segment_ptr = sec_segment;
        }
        return &(sec_segment->pairs_[0]);  // TODO
    }

    template <class T>
    unsigned ZHASH<T>::get_cands(T key, Segment<T> *segment, Key_t *cand_keys, Value_t *cand_values, int *cand_slots) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        auto main_bucket_index = key_hash & k_hash_suffix_mask;
        unsigned cand_num = 0;
        // main bucket key match
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if constexpr (std::is_pointer_v<T>) {
                if (match_fingerprint(segment->pairs_[main_bucket_slot].key, key_hash)) {
                    cand_keys[cand_num] = segment->pairs_[main_bucket_slot].key;
                    cand_values[cand_num] = segment->pairs_[main_bucket_slot].value;
                    if (cand_slots) {
                        cand_slots[cand_num] = main_bucket_slot;
                    }
                    cand_num++;
                }
            } else {
                if (match_key(segment->pairs_[main_bucket_slot].key, key, key_hash)) {  // find the certain right slot for non-pointer key
                    cand_keys[cand_num] = segment->pairs_[main_bucket_slot].key;
                    cand_values[cand_num] = segment->pairs_[main_bucket_slot].value;
                    if (cand_slots) {
                        cand_slots[cand_num] = main_bucket_slot;
                    }
                    cand_num++;
                }
            }
        }
        // overflow bucket key match
        for (unsigned i = 0; i < k_num_slot_per_bucket; i++) {
            unsigned main_bucket_slot = main_bucket_index * k_num_slot_per_bucket + i;
            if (match_overflow_fingerprint((uint64_t)segment->pairs_[main_bucket_slot].value, key_hash)) {
                unsigned overflow_bucket_slot = get_overflow_position((uint64_t)segment->pairs_[main_bucket_slot].value);
                if constexpr (std::is_pointer_v<T>) {
                    if (match_fingerprint(segment->pairs_[overflow_bucket_slot].key, key_hash)) {  // still need to check fingerprint in overflow bucket
                        cand_keys[cand_num] = segment->pairs_[overflow_bucket_slot].key;
                        cand_values[cand_num] = segment->pairs_[overflow_bucket_slot].value;
                        if (cand_slots) {
                            cand_slots[cand_num] = overflow_bucket_slot;
                        }
                        cand_num++;
                    }
                } else {
                    if (match_key(segment->pairs_[overflow_bucket_slot].key, key, key_hash)) {  // find the certain right slot for non-pointer key
                        cand_keys[cand_num] = segment->pairs_[overflow_bucket_slot].key;
                        cand_values[cand_num] = segment->pairs_[overflow_bucket_slot].value;
                        if (cand_slots) {
                            cand_slots[cand_num] = overflow_bucket_slot;
                        }
                        cand_num++;
                    }
                }
            }
        }
        return cand_num;
    }

    template <class T>
    unsigned ZHASH<T>::get_sec_cands(T key, Segment<T> *sec_segment, Key_t *sec_cand_keys, Value_t *sec_cand_values, int *sec_cand_slots) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
            key_hash = h(key->key, key->length);
        } else {
            key_hash = h(&key, sizeof(key));
        }
        unsigned sec_cand_num = 0;
        for (unsigned slot = 0; slot < Segment<T>::k_num_slot_per_segment; slot++) {
            if constexpr (std::is_pointer_v<T>) {
                if (match_fingerprint(sec_segment->pairs_[slot].key, key_hash)) {
                    sec_cand_keys[sec_cand_num] = sec_segment->pairs_[slot].key;
                    sec_cand_values[sec_cand_num] = sec_segment->pairs_[slot].value;
                    if (sec_cand_slots) {
                        sec_cand_slots[sec_cand_num] = slot;
                    }
                    sec_cand_num++;
                }
            } else {  // find the certain right slot for non-pointer key
                if (match_key(sec_segment->pairs_[slot].key, key, key_hash)) {
                    sec_cand_keys[sec_cand_num] = sec_segment->pairs_[slot].key;
                    sec_cand_values[sec_cand_num] = sec_segment->pairs_[slot].value;
                    if (sec_cand_slots) {
                        sec_cand_slots[sec_cand_num] = slot;
                    }
                    sec_cand_num++;
                }
            }
        }
        return sec_cand_num;
    }

    template <class T>
    bool ZHASH<T>::Get(T key, Value_t *value_, int batch_offset, T *batch_keys, nsTimer *clks) {
        if (batch_offset == 0) {  // determine segment addresses + prefetch
            Segment<T> *batch_segments[batch_size];
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                void *bucket_addr = get_bucket_addr(batch_keys[batch_index], &batch_segments[batch_index]);
                prefetch(bucket_addr);
            }
            // determine candidate slots + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                cand_nums[batch_index] = get_cands(batch_keys[batch_index], batch_segments[batch_index], cand_keys[batch_index], cand_values[batch_index]);
                for (unsigned cand_index = 0; cand_index < cand_nums[batch_index]; cand_index++) {
                    if constexpr (std::is_pointer_v<T>) {
                        prefetch((void *)get_addr(cand_keys[batch_index][cand_index]));
                    }
                    prefetch((void *)get_addr((uint64_t)cand_values[batch_index][cand_index]));
                }
            }
            Segment<T> *batch_sec_segments[batch_size];
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                void *bucket_addr = get_sec_bucket_addr(batch_keys[batch_index], &batch_sec_segments[batch_index]);
                prefetch(bucket_addr);
            }
            // determine secondary candidate slots + prefetch
            for (int batch_index = 0; batch_index < batch_size; batch_index++) {
                sec_cand_nums[batch_index] = get_sec_cands(batch_keys[batch_index], batch_sec_segments[batch_index], sec_cand_keys[batch_index], sec_cand_values[batch_index]);
                for (unsigned sec_cand_index = 0; sec_cand_index < sec_cand_nums[batch_index]; sec_cand_index++) {
                    if constexpr (std::is_pointer_v<T>) {
                        prefetch((void *)get_addr(sec_cand_keys[batch_index][sec_cand_index]));
                    }
                    prefetch((void *)get_addr((uint64_t)sec_cand_values[batch_index][sec_cand_index]));
                }
            }
        }
        // start real request process
        Segment<T> *segment;
        get_bucket_addr(key, &segment);
        if (cand_nums[batch_offset] == -1) {
            cand_nums[batch_offset] = get_cands(key, segment, cand_keys[batch_offset], cand_values[batch_offset]);
        }
        Segment<T> *sec_segment;
        get_sec_bucket_addr(key, &sec_segment);
        if (sec_cand_nums[batch_offset] == -1) {
            sec_cand_nums[batch_offset] = get_sec_cands(key, sec_segment, sec_cand_keys[batch_offset], sec_cand_values[batch_offset]);
        }
// start to check cand_nums in HTM
#ifdef READ_HTM
        int htm_status;
        for (int htm_retry = 0; htm_retry < READ_RETRY_TIME; ++htm_retry) {
            htm_status = _xbegin();
            if (htm_status == _XBEGIN_STARTED) {
                break;
            }
        }
        if (htm_status != _XBEGIN_STARTED) {
            return false;
        }
#endif
        for (unsigned cand_index = 0; cand_index < cand_nums[batch_offset]; cand_index++) {
            if constexpr (std::is_pointer_v<T>) {
                if (var_compare((char *)get_addr(cand_keys[batch_offset][cand_index]), (char *)&(key->key), get_len(cand_keys[batch_offset][cand_index]), key->length)) {
                    Value_t value = (Value_t)get_addr((uint64_t)cand_values[batch_offset][cand_index]);
#ifdef VALUE_LENGTH_VARIABLE
                    memcpy(value_, value, value_length);
#else
                    *value_ = value;
#endif
#ifdef READ_HTM
                    if (htm_status == _XBEGIN_STARTED) {
                        _xend();
                    }
#endif
                    return true;
                }
            } else {
                assert(cand_nums[batch_offset] <= 1);
                Value_t value = (Value_t)get_addr((uint64_t)cand_values[batch_offset][cand_index]);
#ifdef VALUE_LENGTH_VARIABLE
                memcpy(value_, value, value_length);
#else
                *value_ = value;
#endif
#ifdef READ_HTM
                if (htm_status == _XBEGIN_STARTED) {
                    _xend();
                }
#endif
                return true;
            }
        }
        // can not find in primary segment, try to find in secondary segment
        for (unsigned sec_cand_index = 0; sec_cand_index < sec_cand_nums[batch_offset]; sec_cand_index++) {
            if constexpr (std::is_pointer_v<T>) {
                if (var_compare((char *)get_addr(sec_cand_keys[batch_offset][sec_cand_index]), (char *)&(key->key), get_len(sec_cand_keys[batch_offset][sec_cand_index]), key->length)) {
                    Value_t value = (Value_t)get_addr((uint64_t)sec_cand_values[batch_offset][sec_cand_index]);
#ifdef VALUE_LENGTH_VARIABLE
                    memcpy(value_, value, value_length);
#else
                    *value_ = value;
#endif
#ifdef READ_HTM
                    if (htm_status == _XBEGIN_STARTED) {
                        _xend();
                    }
#endif
                    return true;
                }
            } else {
                assert(sec_cand_nums[batch_offset] <= 1);
                Value_t value = (Value_t)get_addr((uint64_t)sec_cand_values[batch_offset][sec_cand_index]);
#ifdef VALUE_LENGTH_VARIABLE
                memcpy(value_, value, value_length);
#else
                *value_ = value;
#endif
#ifdef READ_HTM
                if (htm_status == _XBEGIN_STARTED) {
                    _xend();
                }
#endif
                return true;
            }
        }
#ifdef READ_HTM
        if (htm_status == _XBEGIN_STARTED) {
            _xend();
        }
#endif
        return false;
    }
}  // namespace zhash
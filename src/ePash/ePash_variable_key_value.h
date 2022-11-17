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
#include "../../util/compound_pointer.h"
#include "../Hash.h"

#define INSERT_HTM
// #define READ_HTM
#define READ_RETRY_TIME 2

#define VALUE_LENGTH_VARIABLE
extern uint64_t value_length;

#define INLOCK_UPDATE_RETRY_TIME 3

extern __thread nsTimer *clk;

extern uint64_t update_retry_time;
extern uint64_t hot_num;
extern uint64_t hot_bit;
extern uint64_t asso;
__thread uint64_t request = 0;

namespace zhash {
struct _Pair {
  Key_t key;
  Value_t value;
};

const Key_t INVAL = 0;

// const size_t kCacheLineSize = 64;
// constexpr size_t kSegmentBits = 8;
constexpr size_t kSegmentBits = 2;
constexpr size_t kMask = (1 << kSegmentBits) - 1;
constexpr size_t kShift = kSegmentBits;
constexpr size_t kSegmentSize = (1 << kSegmentBits) * 16 * 4;
constexpr size_t kNumPairPerCacheLine = kCacheLineSize / 16;
constexpr size_t kNumCacheLine = 4;
constexpr size_t kMetadataSpace = 0;

constexpr size_t kNumBucket = 4;
constexpr size_t kNumSlotPerBucket = 4;

/* metadata in segment addr: 0-5 bits = local_depth; 6 bit = lock */
constexpr size_t kDepthBits = 6;
constexpr size_t kDepthShift = 64 - kDepthBits;
constexpr size_t kDepthMask =  0xfc00000000000000;
constexpr size_t kLockMask = 0x0200000000000000;
constexpr size_t kAddrMask = 0x0000ffffffffffff;

inline void* get_seg_addr(void* seg_p) {
  return (void*)((uint64_t)seg_p & kAddrMask);
}

inline void set_seg_addr(void** seg_pp, void* new_addr) {
  uint64_t clear_new_addr = (uint64_t)new_addr & kAddrMask;
  *seg_pp = (void*)((uint64_t)*seg_pp & ~kAddrMask | clear_new_addr);
}

inline uint64_t get_local_depth(void* seg_p) {
  return ((uint64_t)seg_p & kDepthMask) >> kDepthShift;
}

inline void set_local_depth(void** seg_pp, uint64_t depth) {
  uint64_t clear_old_addr = (uint64_t)*seg_pp & ~kDepthMask;
  *seg_pp = (void*)(clear_old_addr | (depth << kDepthShift));
}

inline bool get_seg_lock(void* seg_p) {
  return (uint64_t)seg_p & kLockMask;
}

inline void set_seg_lock(void** seg_pp) {
  *seg_pp = (void*)((uint64_t)*seg_pp | kLockMask);
}

inline void set_seg_lock_with_cas(void** seg_pp) {
  int time = 0;
  while (true) {
    if (!get_seg_lock(*seg_pp)) {
      volatile void* old_value = (void*)((uint64_t)*seg_pp & ~kLockMask);
      volatile void* new_value = (void*)((uint64_t)*seg_pp | kLockMask);
      if (CAS(seg_pp, &old_value, new_value))
        break;
    }
    asm("pause");
  }
}

inline void release_seg_lock(void** seg_pp) {
  *seg_pp = (void*)((uint64_t)*seg_pp & ~kLockMask);
}

inline void* construct_seg_ptr(void* seg_addr, uint64_t depth) {
  uint64_t clear_addr = (uint64_t)seg_addr & kAddrMask;
  return (void*)(clear_addr | (depth << kDepthShift));
}

uint64_t clflushCount;

inline bool var_compare(char *str1, char *str2, int len1, int len2) {
  if (len1 != len2) return false;
  return !memcmp(str1, str2, len1);
}

template <class T>
inline bool match_key(Key_t slot_key, T input_key, uint64_t input_hash) {
  if constexpr (std::is_pointer_v<T>) {
    if (match_pointer_fp(slot_key, input_hash) &&
        var_compare((char *)get_pointer_addr(slot_key), (char *)&(input_key->key), get_pointer_len(slot_key), input_key->length)) {
        return true;
    } else
      return false;
  } else {
    return (slot_key == input_key);
  }
}

template <class T>
inline void set_key(Key_t *slot_key_p, T input_key, uint64_t input_hash, char *key_addr) {
  if constexpr (std::is_pointer_v<T>) {
    set_pointer_fp(slot_key_p, input_hash);
    set_pointer_len_addr(slot_key_p, input_key->length, (uint64_t)key_addr);
  } else {
    *slot_key_p = input_key;
  }
}

inline bool check_key_valid(Key_t slot_key) {
  return (slot_key != INVAL);
}

inline void clear_key(Key_t *slot_key_p) {
  *slot_key_p = INVAL;
}

template <class T>
struct Seg_array;
template <class T>
struct Segment {
  static const size_t kNumSlot = kSegmentSize / sizeof(_Pair) - kMetadataSpace;

  Segment(void) {
    memset((void *)&_[0], 255, sizeof(_Pair) * kNumSlot);
  }

  Segment(size_t depth) {
    memset((void *)&_[0], 255, sizeof(_Pair) * kNumSlot);
  }

  static void New(void **seg, size_t depth) {
    auto seg_ptr = reinterpret_cast<Segment *>(AAllocator::Allocate_without_proc(sizeof(Segment)));
    memset((void *)&seg_ptr->_[0], 0, sizeof(_Pair) * kNumSlot);
    
    *seg = construct_seg_ptr(seg_ptr, depth);
  }

  ~Segment(void) {}

  int Insert(T, Value_t, size_t, size_t, int, void**, Seg_array<T>*);
  int Update(T, Value_t, size_t, size_t, int, void**, Seg_array<T>*, int, bool);
  int Uniqueness_check(T, size_t loc);
  int Insert4split(Key_t, Value_t, size_t);
  bool Put(T, Value_t, size_t);
  void Rebalance();

  size_t get_lock(void** seg_pp, void* sa, size_t global_depth) {
    // get the first entry in the chunk
    char* start_addr = (char*)sa;
    size_t chunk_size = pow(2, global_depth - get_local_depth(*seg_pp));
    int x = ((char*)(seg_pp) - start_addr) / sizeof(void*);
    if (x < 0) {
      printf("x: %d\n", x);
      assert(false);
    }
    x = x - (x % chunk_size);
    // lock the first entry with cas
    set_seg_lock_with_cas((void**)(start_addr + x * sizeof(void*)));
    // lock the other entries without cas
    for (int i = 1; i < chunk_size; i++) {
      set_seg_lock((void**)(start_addr + (x + i) * sizeof(void*)));
    }
    return chunk_size;
  }

  void release_lock(void** seg_pp, void* sa, size_t chunk_size) {
    char* start_addr = (char*)sa;
    int x = ((char*)(seg_pp) - start_addr) / sizeof(void*);
    x = x - (x % chunk_size);

    // release locks in the opposite order
    for (int i = chunk_size - 1; i >= 0; i--) {
      release_seg_lock((void**)(start_addr + (x + i) * sizeof(void*)));
    }
    mfence();
  }

  bool check_lock(void* seg_p) {
    return get_seg_lock(seg_p);
  }

  void get_rd_lock() {
    // mutex.lock_shared();
  }

  void release_rd_lock() {
    // mutex.unlock_shared();
  }

  // bool try_get_lock() {
  //   uint64_t temp = 0;
  //   return CAS(&seg_lock, &temp, 1);
  //   // return CAS(&sema, &temp, -1);
  // }

  bool try_get_rd_lock() {
    // return mutex.try_lock_shared();
    return true;
  }

  _Pair _[kNumSlot];
};

template <class T>
struct Seg_array {
  size_t global_depth;
  void* _[0];

  static void New(void **sa, size_t capacity) {
    auto callback = [](void *ptr, void *arg) {
      auto value_ptr = reinterpret_cast<size_t *>(arg);
      auto sa_ptr = reinterpret_cast<Seg_array *>(ptr);
      sa_ptr->global_depth = static_cast<size_t>(log2(*value_ptr));
      return 0;
    };
    AAllocator::DAllocate(sa, kCacheLineSize,
                          sizeof(Seg_array) + sizeof(uint64_t) * capacity,
                          callback, reinterpret_cast<void *>(&capacity));
  }
};

template <class T>
struct Hot_array {
  uint64_t _[0];

  static void New(void **sa, size_t num) {
    auto callback = [](void *ptr, void *arg) {
      auto value_ptr = reinterpret_cast<size_t *>(arg);
      auto ha_ptr = reinterpret_cast<Hot_array *>(ptr);
      memset(ha_ptr->_, 0, (*value_ptr) * sizeof(uint64_t));
      return 0;
    };
    AAllocator::DAllocate(sa, kCacheLineSize,
                          sizeof(uint64_t) * num,
                          callback, reinterpret_cast<void *>(&num));
  }
};

template <class T>
struct Directory {
  static const size_t kDefaultDirectorySize = 1024;
  Hot_array<T> *hot_arr;
  Seg_array<T> *sa;
  void *new_sa;
  size_t capacity;
  bool lock;
  bool fall_back = 0;
  uint64_t version = 0;
  int sema = 0;

  Directory(Seg_array<T> *_sa) {
    capacity = kDefaultDirectorySize;
    sa = _sa;
    new_sa = nullptr;
    lock = false;
    sema = 0;
    version = 0;
    fall_back = 0;
  }

  Directory(size_t size, Seg_array<T> *_sa) {
    capacity = size;
    sa = _sa;
    new_sa = nullptr;
    lock = false;
    sema = 0;
    version = 0;
    fall_back = 0;
  }

  static void New(Directory **dir, size_t capacity) {
    auto callback = [](void *ptr, void *arg) {
      auto value_ptr =
          reinterpret_cast<std::pair<size_t, Seg_array<T> *> *>(arg);
      auto dir_ptr = reinterpret_cast<Directory *>(ptr);
      dir_ptr->capacity = value_ptr->first;
      dir_ptr->sa = value_ptr->second;
      dir_ptr->new_sa = nullptr;
      dir_ptr->lock = false;
      dir_ptr->sema = 0;
      dir_ptr->version = 0;
      dir_ptr->fall_back = 0;
      dir_ptr = nullptr;
      return 0;
    };

    auto call_args = std::make_pair(capacity, nullptr);
    AAllocator::DAllocate((void **)dir, kCacheLineSize, sizeof(Directory),
                          callback, reinterpret_cast<void *>(&call_args));
  }

  ~Directory(void) {}

  void get_item_num() {
    size_t count = 0;
    size_t seg_num = 0;
    Seg_array<T> *seg = sa;
    void **dir_entry = seg->_;
    void *seg_p;
    Segment<T> *ss;
    auto global_depth = seg->global_depth;
    size_t depth_diff;
    for (int i = 0; i < capacity;) {
      seg_p = dir_entry[i];
      ss = reinterpret_cast<Segment<T> *>(get_seg_addr(seg_p));
      depth_diff = global_depth - get_local_depth(seg_p);

      for (unsigned i = 0; i < Segment<T>::kNumSlot; ++i) {
        if (check_key_valid(ss->_[i].key))
          ++count;
      }

      seg_num++;
      i += pow(2, depth_diff);
    }
    std::cout << "#items: " << count << std::endl;
    std::cout << std::fixed << "load_factor: "
              << (double)count / (seg_num * ((1 << kSegmentBits) * 4 - kMetadataSpace))
              << std::endl;
  }

  bool Acquire(void) {
    bool unlocked = false;
    return CAS(&lock, &unlocked, true);
  }

  bool Release(void) {
    bool locked = true;
    return CAS(&lock, &locked, false);
  }

  bool Acquire_fallback(void) {
    bool unlocked = false;
    return CAS(&fall_back, &unlocked, true);
  }

  bool Release_fallback(void) {
    bool locked = true;
    return CAS(&fall_back, &locked, false);
  }

  void SanityCheck(void *);
};

template <class T>
class ZHASH : public Hash<T> {
 public:
  ZHASH(void);
  ZHASH(int);
  ~ZHASH(void);
  int Insert(T key, Value_t value);
  bool Delete(T);
  bool Get(T, Value_t *);
  double Utilization(void);
  size_t Capacity(void);
  void Recovery(void);
  Segment<T> *Split(T key, Segment<T> *target, uint64_t prev_depth,
                    Seg_array<T> *sa, void** target_p, uint64_t x, size_t *target_pattern);
  void Help_Doubling(T key, Seg_array<T> *new_dir, Segment<T> *target,
                     uint64_t prev_depth);
  void Directory_Doubling();
  void Directory_Update(int x, void *s0, void **s1, Seg_array<T> *sa);
  void Lock();
  void Unlock();
  bool Checklock();
  void Lock_Directory();
  void Unlock_Directory();
  bool Checklock_Directory();
  void Swap(void **entry, void **new_seg);
  void getNumber() { 
    printHot();
    dir->get_item_num(); 
  }

  bool Hot_check(T);
  bool Hot_update(T);
  void setHot();
  void printHot();

  Directory<T> *dir;
  int seg_num;
  int restart;
};

template <class T>
int Segment<T>::Uniqueness_check(T key, size_t loc)
{
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }

  for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
    unsigned slot = loc * kNumSlotPerBucket + i;
    if (match_key(_[slot].key, key, key_hash)) {
      return slot;
    }
  }
  // if not found, check the overflow fingerprints in this bucket
  for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
    unsigned slot = loc * kNumSlotPerBucket + i;
    if (match_pointer_of_fp((uint64_t)_[slot].value, key_hash)) {
      unsigned pos = get_pointer_of_pos((uint64_t)_[slot].value);
      if (match_key(_[pos].key, key, key_hash)) {
        return pos;
      }
    }
  }
  return -1;
}

template <class T>
int Segment<T>::Update(T key, Value_t value, size_t loc, size_t key_hash,
                       int prev_depth, void** seg_pp, Seg_array<T>*sa, int slot, bool hot) {
  int ret = -3;

  // determine update type
  int type = 0;
  #ifdef VALUE_LENGTH_VARIABLE
  if (hot) {
    type = 1; // hot key, in-place update
  } else {
    if (value_length >= 128)
      type = 2; // cold large key, in-place + flush
    else
      type = 3; // cold small key, Cow
  }
  #endif

  if (type == 1 || type == 2) {
    int status = 1;
    int inlock_status;
    size_t locking_chunk_size = 0;

    for (int j = 0; j < update_retry_time; ++j) {
      while (check_lock(*seg_pp)) // first to check lock before start htm
        asm("pause");
      status = _xbegin();
      if (status == _XBEGIN_STARTED) {
        // if (clk) clk[0].start();
        break;
      }
    }
    if (status != _XBEGIN_STARTED) {
      // return -3;
      locking_chunk_size = get_lock(seg_pp, sa->_, sa->global_depth);
      
      // htm in lock path
      for (int k = 0; k < INLOCK_UPDATE_RETRY_TIME; k++) {
        inlock_status = _xbegin();
        if (inlock_status == _XBEGIN_STARTED)
          break;
      }
    } else if (check_lock(*seg_pp)) {
      _xabort(6);
    }

    // in-place update
    uint64_t *value_addr = (uint64_t *)get_pointer_addr((uint64_t)_[slot].value);
    for (int i = 0; i < value_length / sizeof(uint64_t); i++) {
      value_addr[i] = uint64_t(value);
    }

    if (status == _XBEGIN_STARTED) {
      // if (clk) clk[0].end();
      _xend();
    } else {
      if (inlock_status == _XBEGIN_STARTED)
        _xend();
      release_lock(seg_pp, sa->_, locking_chunk_size);
    }
      
    
    // flush for cold key
    if (type == 2)
      AAllocator::Persist_flush(value_addr, value_length);
  } else if (type == 3 || type == 0) {
    if (type == 3) {
      value = AAllocator::Prepare_value(value, value_length);
    }

RE_UPDATE:
    int status;
    size_t locking_chunk_size = 0;
    for (int i = 0; i < 2; ++i) {
      while (check_lock(*seg_pp))
        asm("pause");
      status = _xbegin();
      if (status == _XBEGIN_STARTED)
        break;
    }
    if (status != _XBEGIN_STARTED) {
      locking_chunk_size = get_lock(seg_pp, sa->_, sa->global_depth);
    } else if (check_lock(*seg_pp)) {
      _xend();
      goto RE_UPDATE;
    }

    /* Critical Section Start */
    set_pointer_len_addr((uint64_t *)&(_[slot].value), value_length, (uint64_t)value);
    /* Critical Section End */

    if (status != _XBEGIN_STARTED)
      release_lock(seg_pp, sa->_, locking_chunk_size);
    else
      _xend();
  }

  return ret;
}

template <class T>
int Segment<T>::Insert(T key, Value_t value, size_t loc, size_t key_hash,
                       int prev_depth, void** seg_pp, Seg_array<T>*sa) {
  int ret = 1;
  char *key_addr;
#ifdef VALUE_LENGTH_VARIABLE
  value = AAllocator::Prepare_value(value, value_length);
  if constexpr (std::is_pointer_v<T>)
    key_addr = AAllocator::Prepare_key(key);
#endif
#ifdef INSERT_HTM
  int status;
  size_t locking_chunk_size = 0;
  for (int i = 0; i < 64; ++i) {
    status = _xbegin();
    if (status == _XBEGIN_STARTED) break;
    asm("pause");
  }
  if (status != _XBEGIN_STARTED) {
    locking_chunk_size = get_lock(seg_pp, sa->_, sa->global_depth);
    if (prev_depth != get_local_depth(*seg_pp)) {
      release_lock(seg_pp, sa->_, locking_chunk_size);
      return 2;
    }
  } else if (check_lock(*seg_pp) || prev_depth != get_local_depth(*seg_pp)) {
    _xend();
    return 2;
  }
#endif
  int slot = Uniqueness_check(key, loc); 
  if (slot != -1)
  {
#ifdef INSERT_HTM
    if (status != _XBEGIN_STARTED)
        release_lock(seg_pp, sa->_, locking_chunk_size);
    else
        _xend();
#endif
    return 4;
  }

  int empty_fp_index = -1;
  int outing_index = -1;

  for (unsigned i = 0; i < kNumSlot; i++) { 
    unsigned slot = (loc * kNumSlotPerBucket + i) % kNumSlot;
    if (i < kNumSlotPerBucket) { // original bucket
      if (!check_key_valid(_[slot].key)) {
        set_pointer_len_addr((uint64_t *)&(_[slot].value), value_length, (uint64_t)value);
        set_key(&(_[slot].key), key, key_hash, key_addr);
        ret = 0;
        break;
      } else {
        if (empty_fp_index == -1 && !check_pointer_of_valid((uint64_t)_[slot].value)) {
          empty_fp_index = slot;
        }
      }
    } else { // outing bucket
      if (empty_fp_index == -1)
        break;
      if (!check_key_valid(_[slot].key)) {
        outing_index = slot;
        set_pointer_len_addr((uint64_t *)&(_[slot].value), value_length, (uint64_t)value);
        set_key(&(_[slot].key), key, key_hash, key_addr);
        set_pointer_of_fp_pos((uint64_t *)&(_[empty_fp_index].value), key_hash, slot);
        ret = 0;
        break;
      }
    }
  }


#ifdef INSERT_HTM
  if (status != _XBEGIN_STARTED)
    release_lock(seg_pp, sa->_, locking_chunk_size);
  else
    _xend();
#endif

  if (outing_index != -1) {
    AAllocator::Persist_asyn_flush(&(_[loc * kNumSlotPerBucket]), 64);
    AAllocator::Persist_asyn_flush(&(_[outing_index / kNumSlotPerBucket * kNumSlotPerBucket]), 64);
  }
  return ret;
}

template <class T>
void Segment<T>::Rebalance() {
  _Pair outing_list[kNumSlot];
  uint64_t outing_hash[kNumSlot];
  size_t outing_number = 0;
  size_t bucket_number[4] = {0, 0, 0, 0};

  // find all the outing items
  for (unsigned i = 0; i < kNumSlot; i++) {
    if (check_key_valid(_[i].key)) {
      uint64_t key_hash;
      if constexpr (std::is_pointer_v<T>) {
        key_hash = h((void *)get_pointer_addr(_[i].key), get_pointer_len(_[i].key));
      } else {
        key_hash = h(&(_[i].key), sizeof(Key_t));
      }
      if ((i / kNumBucket) != (key_hash & kMask)) {
        outing_list[outing_number] = _[i];
        outing_hash[outing_number] = key_hash;
        outing_number++;
        clear_key(&(_[i].key));
      } else {
        bucket_number[i / kNumBucket]++;
      }
    }
  }

  // fisrt round to try to set outing items back to their our bucket
  for (unsigned i = 0; i < outing_number; i++) {
    size_t bucket = outing_hash[i] & kMask;
    for (unsigned slot = bucket * kNumSlotPerBucket; slot < (bucket + 1) * kNumSlotPerBucket; slot++) {
      if (!check_key_valid(_[slot].key)) {
        _[slot] = outing_list[i]; // value in outing_list is clean without of info
        outing_list[i].key = INVAL;
        bucket_number[bucket]++;
        break;
      }
    }
  }

  // second round to set the remaining items to outing buckets
  for (unsigned i = 0; i < outing_number; i++) {
    if (outing_list[i].key != INVAL) {
      size_t bucket = outing_hash[i] & kMask;
      int pos = -1;
      // find the most empty outing bucket
      size_t empty_bucket = (bucket + 1) % kNumBucket;
      for (unsigned j = bucket + 2; j < bucket + 4; j++) {
        if (bucket_number[j % kNumBucket] < bucket_number[empty_bucket])
          empty_bucket = j % kNumBucket;
      }
      // insert the item in the found bucket
      for (unsigned slot = empty_bucket * kNumSlotPerBucket; slot < (empty_bucket + 1) * kNumSlotPerBucket; slot++) {
        if (!check_key_valid(_[slot].key)) {
          _[slot] = outing_list[i]; // value in outing_list is clean without of info
          bucket_number[empty_bucket]++;
          pos = slot;
          break;
        }
      }
      assert(pos >= 0);
      // find and set fingerprint slot in original bucket
      int fin_pos = -1;
      for (unsigned slot = bucket * kNumSlotPerBucket; slot < (bucket + 1) * kNumSlotPerBucket; slot++) {
        if (!check_pointer_of_valid((uint64_t)_[slot].value)) {
          set_pointer_of_fp_pos((uint64_t *)&(_[slot].value), outing_hash[i], pos);
          fin_pos = slot;
          break;
        }
      }
      assert(fin_pos >= 0);
    }
  }
}

template <class T>
int Segment<T>::Insert4split(Key_t key, Value_t value, size_t loc) {
  // insert into the previous location directly
  _[loc].value = value;
  _[loc].key = key;
  return 0;
}

template <class T>
ZHASH<T>::ZHASH(int initCap) {
  Directory<T>::New(&dir, initCap);
  Hot_array<T>::New((void **)(&dir->hot_arr), hot_num);
  Seg_array<T>::New(&dir->new_sa, initCap);
  dir->sa = reinterpret_cast<Seg_array<T> *>(dir->new_sa);
  // dir->new_sa = nullptr;
  auto dir_entry = dir->sa->_;
  for (int i = 0; i < dir->capacity; ++i) {
    Segment<T>::New(&dir_entry[i], dir->sa->global_depth);
    Segment<T> *curr_seg = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entry[i]));
  }

  seg_num = 0;
  restart = 0;

  printf("Segment size: %ld\n", sizeof(Segment<T>));
  printf("Segment slots: %ld\n", sizeof(Segment<T>::_));
}

template <class T>
ZHASH<T>::ZHASH(void) {
  std::cout << "Reintialize Up for ZHASH" << std::endl;
}

template <class T>
ZHASH<T>::~ZHASH(void) {}

template <class T>
void ZHASH<T>::Recovery(void) {
  if (dir != nullptr) {
    dir->lock = 0;
    if (dir->sa == nullptr) return;
    auto dir_entry = dir->sa->_;
    size_t global_depth = dir->sa->global_depth;
    size_t depth_cur, buddy, stride, i = 0;
    /*Recover the Directory*/
    size_t seg_count = 0;
    while (i < dir->capacity) {
      auto target = reinterpret_cast<Segment<T> *>(dir_entry[i]);
      depth_cur = get_local_depth(target);
      stride = pow(2, global_depth - depth_cur);
      buddy = i + stride;
      for (int j = buddy - 1; j > i; j--) {
        target = reinterpret_cast<Segment<T> *>(dir_entry[j]);
        if (dir_entry[j] != dir_entry[i]) {
          dir_entry[j] = dir_entry[i];
        }
      }
      seg_count++;
      i = i + stride;
    }
  }
}

template <class T>
bool ZHASH<T>::Hot_check(T key) {
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
  uint64_t *hot_arr = dir->hot_arr->_;

  idx = idx - (idx % asso);
  for (int i = idx; i < idx + asso; ++i) {
    if (hot_arr[i] == key_number)
      return true;
    else if (hot_arr[i] == 0) {
      Hot_update(key);
      return true;
    }
  }
  return false;
}

template <class T>
bool ZHASH<T>::Hot_update(T key) {
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
  uint64_t *hot_arr = dir->hot_arr->_;

  bool flag = false;
  idx = idx - (idx % asso);

  for (int i = idx; i < idx + asso; i++) {
    if (hot_arr[i] == 0) {
      hot_arr[i] = key_number;
      flag = true;
      break;
    }
    if (hot_arr[i] == key_number) {
      flag = true;
      if (i != idx) {
        int status = _xbegin();
        if (status == _XBEGIN_STARTED) {
          hot_arr[i] = hot_arr[i - 1];
          hot_arr[i - 1] = key_number;
          _xend();
        }
      }
      break;
    }
  }
  if (!flag)
    hot_arr[idx + asso - 1] = key_number;

  return true;
}

template <class T>
void ZHASH<T>::printHot() {
  int count1 = 0;
  int count2 = 0;
  int count3 = 0;
  uint64_t *hot_arr = dir->hot_arr->_;
  for (int i = 0; i < uint64_t(hot_num); ++i) {
    if (hot_arr[i] != 0) {
      if (hot_arr[i] < uint64_t(hot_num))
        count1++;
      if (hot_arr[i] < uint64_t(hot_num) / 10)
        count2++;
      if (hot_arr[i] < uint64_t(hot_num) / 100)
        count3++;     
    }
  }
  printf("Top 100% Hot number:%f\n", double(count1) / uint64_t(hot_num));
  printf("Top 10% Hot number:%f\n", double(count2) / uint64_t(hot_num) * 10);
  printf("Top 1% Hot number:%f\n", double(count3) / uint64_t(hot_num) * 100);
}

template <class T>
void ZHASH<T>::setHot() {
  int global_depth = dir->sa->global_depth;
  uint64_t dentry_num = (1llu << global_depth);
  uint64_t seg_num = dentry_num / uint64_t(hot_num);
  Segment<T> *ss;
  Segment<T> *prev_ss = NULL;
  Segment<T> **dir_entry = (Segment<T> **)dir->sa->_;

  uint64_t curr_key;
  uint64_t *hot_arr = dir->hot_arr->_;

  printf("[setHot] global depth:%d\n", global_depth);
  printf("[setHot] directory entry:%lu\n", dentry_num);
  printf("[setHot] hot number:%lu\n", hot_num);
  printf("[setHot] seg num:%lu\n", seg_num);
  for (int i = 0; i < uint64_t(hot_num); i += asso) {
    for (int j = i; j < i + asso; ++j)
      hot_arr[j] = 20000000;

    for (int j = i * seg_num; j < (i + asso) * seg_num; ++j) {
      ss = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entry[j]));
      if (ss == prev_ss)
        continue;
      for (unsigned k = 0; k < Segment<T>::kNumSlot; ++k) {
        if (check_key_valid(ss->_[k].key)) {
          if constexpr (std::is_pointer_v<T>) {
            uint64_t *key_addr = (uint64_t *)get_pointer_addr(ss->_[k].key);
            curr_key = key_addr[0];
          } else {
            curr_key = ss->_[k].key;
          }
          for (int p = i; p < i + asso; ++p) {
            if (curr_key < hot_arr[p]) {
              if (p != i)
                hot_arr[p-1] = hot_arr[p];
              hot_arr[p] = curr_key;
            }
          }
        }
      }
      prev_ss = ss;
    }
  }
  printHot();
}

template <class T>
void ZHASH<T>::Swap(void **entry, void **new_seg) {
  *entry = *new_seg;
  *new_seg = nullptr;
}

template <class T>
Segment<T> *ZHASH<T>::Split(T key, Segment<T> *target, uint64_t prev_depth,
                           Seg_array<T> *sa, void** target_p, uint64_t x, size_t *target_pattern) {
  size_t locking_chunk_size = 0;
  size_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }

  int prev_global_depth = sa->global_depth;
  if (get_local_depth(*target_p) < prev_global_depth) {
    void *ss;
    void **temp = &ss;
    target->New(temp, get_local_depth(*target_p));
    Segment<T> *split = reinterpret_cast<Segment<T> *>(get_seg_addr(*temp));

    int status;
    for (int i = 0; i < 2; ++i) {
      status = _xbegin();
      if (status == _XBEGIN_STARTED) break;
    }
    if (status != _XBEGIN_STARTED) {
      locking_chunk_size = target->get_lock(target_p, sa->_, sa->global_depth);
      if (get_local_depth(*target_p) != prev_depth) {
        target->release_lock(target_p, sa->_, locking_chunk_size);
        return NULL;
      }
    } else if (target->check_lock(*target_p) || get_local_depth(*target_p) != prev_depth ||
               (Checklock_Directory() && (Seg_array<T> *)dir->new_sa != sa)) {
      // Ensure that the segment is not locked by the fall back path of HTM
      // Ensure that the split of segment has not been finished by other threads
      // Ensure that the doubling doesn't happen after the split begins
      _xend();
      return NULL;
    }

    size_t pattern = x >> (prev_global_depth - prev_depth);
    size_t new_pattern = (pattern << 1) + 1;
    size_t old_pattern = pattern << 1;
    for (unsigned i = 0; i < Segment<T>::kNumSlot; ++i) {
      if (check_key_valid(target->_[i].key)) {
        uint64_t key_hash;
        if constexpr (std::is_pointer_v<T>) {
          key_hash = h((void *)get_pointer_addr(target->_[i].key), get_pointer_len(target->_[i].key));
        } else {
          key_hash = h(&(target->_[i].key), sizeof(Key_t));
        }
        size_t bucket = key_hash & kMask;

        // clear all the overflow info because there is a overall modification afterward
        // clear_fingerprint_and_position((void **)&(target->_[i].key));
        clear_pointer_of((uint64_t *)&(target->_[i].value));

        if (key_hash >> (8 * 8 - get_local_depth(*target_p) - 1) == new_pattern) {
          // move only clear key without fingerprints
          split->Insert4split(target->_[i].key, target->_[i].value, i);
          clear_key(&(target->_[i].key));
        }
      }
    }

    // rebalance the two segment after modification
    split->Rebalance();
    target->Rebalance();

    *target_pattern = (key_hash >> (8 * sizeof(key_hash) - get_local_depth(*temp))) << 1;
    
    // Directory update
    uint64_t x = (key_hash >> (8 * sizeof(key_hash) - sa->global_depth));
    Directory_Update(x, *target_p, temp, sa);

    // End of Directory Update
    if (status != _XBEGIN_STARTED) {
      target->release_lock(target_p, sa->_, locking_chunk_size);
    } else
      _xend();
    return split;
  } else {
    // printf("doubling number:%lu!\n", uint64_t(key));
    Lock_Directory();
    if (dir->sa->global_depth != prev_global_depth) {
      Unlock_Directory();
      return NULL;
    }
    printf("doubling number:%lu!\n", uint64_t(key));
    long long duration;
    Directory_Doubling();
    Unlock_Directory();
    return NULL;
  }
}

template <class T>
void ZHASH<T>::Help_Doubling(T key, Seg_array<T> *new_sa, Segment<T> *target,
                            uint64_t prev_depth) {
  size_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto sa = dir->sa;
  auto x = (key_hash >> (8 * sizeof(key_hash) - sa->global_depth));
  unsigned depth_diff = sa->global_depth - prev_depth;
  int chunk_size = pow(2, depth_diff);
  int start = x - (x % chunk_size);
  int end = start + chunk_size;
  start = start - (start % 4);
  end = end - (end % 4) + 4;

  int status;
  int flag;

  for (int i = start; i < end; i += 4) {
    flag = 0;
    for (int j = 0; j < 8; ++j) {
      status = _xbegin();
      if (status == _XBEGIN_STARTED)
        break;
      else
        asm("pause");
    }
    if (status != _XBEGIN_STARTED) {
      // printf("status:%d\n", status);
      Lock();
    } else if (Checklock()) {
      _xend();
      while (Checklock()) asm("pause");
      i -= 4;
      continue;
    }
    for (int j = i; j < i + 4; j++) {
      if (new_sa->_[2 * j] != NULL || new_sa->_[2 * j + 1] != NULL) flag = 1;
    }
    if (flag == 0) {
      for (int j = i; j < i + 4; j++)
        new_sa->_[2 * j + 1] = new_sa->_[2 * j] = sa->_[j];
    }
    if (status != _XBEGIN_STARTED) {
      Unlock();
    } else
      _xend();
  }
}

template <class T>
void ZHASH<T>::Directory_Doubling() {
  Seg_array<T> *sa = dir->sa;
  void **d = sa->_;
  auto global_depth = sa->global_depth;
  /* new segment array*/
  long long duration;

  void *ss;
  void **temp = &ss;
  Seg_array<T>::New(temp, 2 * dir->capacity);
  mfence();
  dir->new_sa = *temp;
  // ----------- other threads can operate on dir->new_sa
  mfence();

  auto new_seg_array = reinterpret_cast<Seg_array<T> *>(dir->new_sa);
  auto dd = new_seg_array->_;

  int status;
  int flag;
  for (unsigned i = 0; i < dir->capacity; i += 4) {
    flag = 0;
    for (int i = 0; i < 8; ++i) {
      status = _xbegin();
      if (status == _XBEGIN_STARTED)
        break;
      else
        asm("pause");
    }
    if (status != _XBEGIN_STARTED) {
      // printf("status:%d\n", status);
      // assert(false);
      Lock();
    } else if (Checklock()) {
      _xend();
      while (Checklock()) asm("pause");
      i -= 4;
      continue;
    }

    for (int j = i; j < i + 4; ++j) {
      if (dd[2 * j] != 0 || dd[2 * j + 1] != 0) {
        flag = 1;
        break;
      }
    }
    if (flag == 0)
      for (int j = i; j < i + 4; ++j) {
        dd[2 * j] = d[j];
        dd[2 * j + 1] = d[j];
      }

    if (status != _XBEGIN_STARTED) {
      Unlock();
    } else
      _xend();
  }

  dir->sa = reinterpret_cast<Seg_array<T> *>(dir->new_sa);
  mfence();
  dir->capacity *= 2;

  dir->version++;
  mfence();
}

template <class T>
void ZHASH<T>::Lock_Directory() {
  while (!dir->Acquire()) {
    asm("nop");
  }
}

template <class T>
void ZHASH<T>::Unlock_Directory() {
  while (!dir->Release()) {
    asm("nop");
  }
}

template <class T>
bool ZHASH<T>::Checklock_Directory() {
  return dir->lock;
}

template <class T>
void ZHASH<T>::Lock() {
  while (!dir->Acquire_fallback()) {
    asm("nop");
  }
}

template <class T>
void ZHASH<T>::Unlock() {
  while (!dir->Release_fallback()) {
    asm("nop");
  }
}

template <class T>
bool ZHASH<T>::Checklock() {
  return dir->fall_back;
}

template <class T>
void ZHASH<T>::Directory_Update(int x, void *s0, void **s1,
                               Seg_array<T> *sa) {
  size_t old_local_depth = get_local_depth(s0);
  void **dir_entry = sa->_;
  auto global_depth = sa->global_depth;
  unsigned depth_diff = global_depth - old_local_depth;

  int chunk_size = pow(2, global_depth - old_local_depth);
  x = x - (x % chunk_size);
  int base = chunk_size / 2;

  for (int i = 0; i < base; ++i) {
    dir_entry[x + base + i] = (Segment<T> *)(*s1);
  }
  // update all the local depth in the chunk
  for (int i = 0; i < chunk_size; ++i) {
    set_local_depth(&dir_entry[x + i], old_local_depth + 1);
  }
}

template <class T>
int ZHASH<T>::Insert(T key, Value_t value) {
  int flag;
  int retry_time = 0;
  int lock_retry = 0;
  int depth_retry = 0;

STARTOVER:
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto y = key_hash & kMask; // bucket index

RETRY:

  auto old_sa = dir->sa;
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  auto dir_entry = old_sa->_;

  Segment<T> *target = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entry[x]));
  void **target_p = &(dir_entry[x]);
  int prev_depth = get_local_depth(*target_p);
  int ret = 0;
  Seg_array<T> *new_sa;

  // Ensuring that find correct directory when doubling is happening
  if (Checklock_Directory() &&
      (new_sa = (Seg_array<T> *)(dir->new_sa)) != old_sa) {
    auto new_x = (key_hash >> (8 * sizeof(key_hash) - new_sa->global_depth));
    if (new_sa->_[new_x] != NULL) {
      old_sa = new_sa;
      x = new_x;
      dir_entry = old_sa->_;
      target = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entry[x]));
      target_p = &(dir_entry[x]);
      prev_depth = get_local_depth(*target_p);
    }
  }

  if (get_seg_addr(dir_entry[x]) != target) {
    printf("depth and dentry mismatch\n");
    asm("pause");
    goto RETRY;
  }
Check:
  int slot = target->Uniqueness_check(key, y);
  if (slot != -1) {
    bool hot = false;
    #ifdef VALUE_LENGTH_VARIABLE
    hot = Hot_check(key);
    #endif
    ret = target->Update(key, value, y, key_hash, prev_depth, target_p, dir->sa, slot, hot);
  } else {
    ret = target->Insert(key, value, y, key_hash, prev_depth, target_p, dir->sa);
  }
  if (ret == 4)
    goto Check;

  // Flush the splitted segment
  if (flag) {
    AAllocator::Persist_flush(target, sizeof(Segment<T>));
    flag = 0;
  }

  // Duplicate Insert
  if (ret == -3) return -1;
  // Split
  if (ret == 1) {
    Segment<T> *s;
    size_t target_pattern;
    if (Checklock_Directory()) {
      volatile Seg_array<T> *touch;
      do {
        asm("pause");
        touch = (Seg_array<T> *)dir->new_sa;
        new_sa = (Seg_array<T> *)touch;
      } while (new_sa == dir->sa && Checklock_Directory()); // wait to allocate new sa
      if (new_sa != dir->sa) {
        Help_Doubling(key, new_sa, target, prev_depth);

        // update target_p on new sa
        auto new_x = (key_hash >> (8 * sizeof(key_hash) - new_sa->global_depth));
        target_p = &(new_sa->_[new_x]);
        s = Split(key, target, prev_depth, new_sa, target_p, new_x, &target_pattern);
      } else
        goto RETRY;

    } else {
      // Ensuring that find correct directory when doubling has been finished
      if ((new_sa = (Seg_array<T> *)(dir->new_sa)) != old_sa) {
        auto new_x = (key_hash >> (8 * sizeof(key_hash) - new_sa->global_depth));
        if (new_sa->_[new_x] != NULL) {
          old_sa = new_sa;
          x = new_x;
          dir_entry = old_sa->_;
          target = reinterpret_cast<Segment<T> *>(get_seg_addr(dir_entry[x]));
          target_p = &(dir_entry[x]);
          prev_depth = get_local_depth(*target_p);
        }
      }
      s = Split(key, target, prev_depth, dir->sa, target_p, x, &target_pattern);
    }
    if (s == NULL) {
      retry_time++;
      // The segment has been splitted by other thread
      goto RETRY;
    }
    if (target_pattern == (key_hash >> (8 * sizeof(key_hash) - get_local_depth(*target_p))))
      AAllocator::Persist_flush(s, sizeof(Segment<T>));
    else
      AAllocator::Persist_flush(target, sizeof(Segment<T>));
    flag = 1;
    goto RETRY;
  } else if (ret == 2) {
    // assert(false);
    goto STARTOVER;
  }
  return ret;
}

template <class T>
bool ZHASH<T>::Delete(T key) {
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto y = key_hash & kMask;

RETRY:
  auto old_sa = dir->sa;
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  auto dir_entry = old_sa->_;
  void* target = dir_entry[x];
  Segment<T> *dir_ = reinterpret_cast<Segment<T> *>(get_seg_addr(target));

  for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
    unsigned slot = y * kNumSlotPerBucket + i;
    if constexpr (!std::is_pointer_v<T>) {
      if (match_key(dir_->_[slot].key, key, key_hash)) {
        Key_t old_key = dir_->_[slot].key;
        Key_t invalid_key = INVAL;
        __sync_bool_compare_and_swap(&(dir_->_[slot].key), old_key, invalid_key);
        AAllocator::Persist(&dir_->_[slot], sizeof(_Pair));
        return true;
      }
    }
  }

  // can not find target in original slots, check all the fingerprints in original bucket
  if constexpr (!std::is_pointer_v<T>) {
    for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
      unsigned slot = y * kNumSlotPerBucket + i;
      // if (match_fingerprint(dir_->_[slot].key, key_hash)) {
      if (match_pointer_of_fp((uint64_t)dir_->_[slot].value, key_hash)) {
        // unsigned target_slot = get_position(dir_->_[slot].key);
        unsigned target_slot = get_pointer_of_pos((uint64_t)dir_->_[slot].value);
        if (match_key(dir_->_[target_slot].key, key, key_hash)) {
          Key_t old_key = dir_->_[target_slot].key;
          Key_t invalid_key = INVAL;
          __sync_bool_compare_and_swap(&(dir_->_[target_slot].key), old_key, invalid_key);
          AAllocator::Persist(&dir_->_[slot], sizeof(_Pair));
          return true;
        }
      }
    }
  }

  return false;
}

template <class T>
bool ZHASH<T>::Get(T key, Value_t *value_) {
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto y = key_hash & kMask;
RETRY:
  auto old_sa = dir->sa;
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  auto dir_entry = old_sa->_;
  void* target = dir_entry[x];
  Segment<T> *dir_ = reinterpret_cast<Segment<T> *>(get_seg_addr(target));

  if (!dir_->try_get_rd_lock()) {
    goto RETRY;
  }

#ifdef READ_HTM
  int status;
  size_t locking_chunk_size = 0;
  for (int htm_retry = 0; htm_retry < READ_RETRY_TIME; ++htm_retry) {
    status = _xbegin();
    if (status == _XBEGIN_STARTED)
      break;
  }
  if (status != _XBEGIN_STARTED) {
    return false;
  }
#endif

  for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
    unsigned slot = y * kNumSlotPerBucket + i;
    if (match_key(dir_->_[slot].key, key, key_hash)) {
      Value_t value = (Value_t)get_pointer_addr((uint64_t)dir_->_[slot].value);
      dir_->release_rd_lock();
      #ifdef VALUE_LENGTH_VARIABLE
      memcpy(value_, value, value_length);
      #else
      *value_ = value;
      #endif
      #ifdef READ_HTM
      if (status == _XBEGIN_STARTED)
        _xend();
      #endif
      return true;
    }
  }

  // can not find target in original slots, check all the fingerprints in original bucket
  if constexpr (!std::is_pointer_v<T>) {
    for (unsigned i = 0; i < kNumSlotPerBucket; i++) {
      unsigned slot = y * kNumSlotPerBucket + i;
      // if (match_fingerprint(dir_->_[slot].key, key_hash)) {
      if (match_pointer_of_fp((uint64_t)dir_->_[slot].value, key_hash)) {
        // unsigned target_slot = get_position(dir_->_[slot].key);
        unsigned target_slot = get_pointer_of_pos((uint64_t)dir_->_[slot].value);
        if (match_key(dir_->_[target_slot].key, key, key_hash)) {
          Value_t value = (Value_t)get_pointer_addr((uint64_t)dir_->_[target_slot].value);
          dir_->release_rd_lock();
          #ifdef VALUE_LENGTH_VARIABLE
          memcpy(value_, value, value_length);
          #else
          *value_ = value;
          #endif
          #ifdef READ_HTM
          if (status == _XBEGIN_STARTED)
            _xend();
          #endif
          return true;
        }
      }
    }
  }
  
  dir_->release_rd_lock();
#ifdef READ_HTM
  if (status == _XBEGIN_STARTED)
    _xend();
#endif
  return false;
}
}  // namespace zhash
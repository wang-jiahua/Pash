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
// #include "./ePash/ePash_variable_key_value.h"
#include "./ePash/ePash.h"
#include "Hash.h"
#include "libpmemobj.h"

__thread nsTimer* clk;
#define PMEM1

#ifdef PMEM1
std::string pool_name = "/mnt/pmem1/";
#else
std::string pool_name = "/mnt/pmem0/";
#endif

DEFINE_string(k, "fixed", "the type of stored keys: fixed/variable");
DEFINE_string(distribution, "uniform", "The distribution of the workload: uniform/skew");
DEFINE_uint64(i, 8192, "the initial number of segments in extendible hashing");
DEFINE_uint64(t, 1, "the number of concurrent threads");
DEFINE_uint64(n, 0, "the number of pre-insertion load");
DEFINE_uint64(loadType, 0, "type of pre-load integers: random (0) - range (1)");
DEFINE_uint64(p, 20000000, "the number of operations(insert/search/deletion) to execute");
DEFINE_uint64(u, 0, "the number of update/mixed operations to execute");
DEFINE_string(op, "full", "which type of operation to execute:insert/pos/neg/delete/mixed");
DEFINE_double(r, 0, "read ratio for mixed workload:0~1.0");
DEFINE_double(s, 1, "insert ratio for mixed workload: 0~1.0");
DEFINE_double(d, 0, "delete ratio for mixed workload:0~1.0");
DEFINE_double(skew, 0.99, "skew factor of the workload");
DEFINE_uint32(ms, 100, "#miliseconds to sample the operations");
DEFINE_uint64(ps, 100ul, "The size of the memory pool (GB)");
DEFINE_uint64(vkl, 8, "The length of the variable length key");
DEFINE_uint64(vvl, 8, "The length of the variable length value");
DEFINE_uint64(hb, 13, "Hot bits");

uint64_t initCap, thread_num, load_num, operation_num;
std::string operation;
std::string distribution;
std::string key_type;
int bar_a, bar_b, bar_c;
double read_ratio, insert_ratio, delete_ratio, skew_factor;
std::mutex mtx;
std::condition_variable cv;
bool finished = false;
uint32_t msec;
struct timeval tv1, tv2, tv3;
size_t pool_size = 1024ul * 1024ul * 1024ul * 100ul;
key_generator_t *uniform_generator;
uint64_t load_type = 0;
uint64_t key_length = 8;
uint64_t value_length = 8;
uint64_t kv_length;
uint64_t hot_bit = 13;
uint64_t hot_num = 1 << hot_bit;
uint64_t asso = 2;
uint64_t update_retry_time = 4;
uint64_t update_num = 0;

struct operation_record_t {
  uint64_t number;
  uint64_t dummy[7]; /*patch to a cacheline size, avoid false sharing*/
};

operation_record_t operation_record[1024];

struct range {
  int index;
  uint64_t begin;
  uint64_t end;
  int length; /*if this is the variable length key, use this parameter to
                 indicate the length of the key*/
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

template <class T>
Hash<T> *InitializeIndex(int seg_num) {
  Hash<T> *eh;
  bool file_exist = false;
  gettimeofday(&tv1, NULL);
  
  std::cout << "Initialize ePash" << std::endl;
  std::string index_pool_name = pool_name + "pmem_zhash.data";
  if (FileExists(index_pool_name.c_str())) file_exist = true;
  AAllocator::Initialize(index_pool_name.c_str(), pool_size, thread_num);
  eh = reinterpret_cast<Hash<T> *>(AAllocator::GetRoot(sizeof(zhash::ZHASH<T>)));
  new (eh) zhash::ZHASH<T>(seg_num);

  return eh;
}

/*generate 8-byte number and store it in the memory_region*/
void generate_8B(void *memory_region, uint64_t generate_num, key_generator_t *key_generator, uint64_t mod_range = -1) {
  uint64_t *array = reinterpret_cast<uint64_t *>(memory_region);

  for (uint64_t i = 0; i < generate_num; ++i) {
    array[i] = key_generator->next_uint64();
    if (mod_range != -1)
      array[i] = array[i] % mod_range + 1;
  }
}

/*generate 16-byte string and store it in the memory_region*/
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
    if (mod_range != -1)
      random_num = random_num % mod_range + 1;
    for (int j = 0; j < word_num; ++j) {
      _key[j] = random_num;
    }
    memcpy(var_key->key, _key, key_length);
  }
}

template <class T>
void Load(int kv_num, Hash<T> *index, int length, void *workload) {
  std::cout << "Start load warm-up workload" << std::endl;
  if (kv_num == 0) return;
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

inline void end_notify(struct range *rg) {
  gettimeofday(&rg->tv, NULL);
  if (SUB(&bar_c, 1) == 0) {
    std::unique_lock<std::mutex> lck(mtx);
    finished = true;
    cv.notify_one();
  }
}

inline void end_sub() { SUB(&bar_c, 1); }

template <class T>
void concurr_insert(struct range *_range, Hash<T> *index) {
  set_affinity(_range->index);
  AAllocator::Get()->Initialize_thread(_range->index);
  int begin = _range->begin;
  int end = _range->end;
  char *workload = reinterpret_cast<char *>(_range->workload);
  T key;

  spin_wait();
  if constexpr (!std::is_pointer_v<T>) {
    T *key_array = reinterpret_cast<T *>(workload);
    for (uint64_t i = begin; i < end; ++i) {
      Value_t value = reinterpret_cast<Value_t>(key_array[i] + i);
      int ret = index->Insert(key_array[i], value);
    }
  } else {
    T var_key;
    uint64_t string_key_size = sizeof(string_key) + _range->length;
    for (uint64_t i = begin; i < end; ++i) {
      var_key = reinterpret_cast<T>(workload + string_key_size * i);
      Value_t value = reinterpret_cast<Value_t>(i);
      index->Insert(var_key, value);
      mfence();
    }
  }
  end_notify(_range);
}

template <class T>
void concurr_update(struct range *_range, Hash<T> *index) {
  set_affinity(_range->index);
  AAllocator::Get()->Initialize_thread(_range->index);
  int begin = _range->begin;
  int end = _range->end;
  char *workload = reinterpret_cast<char *>(_range->workload);
  T key;

  spin_wait();
  if constexpr (!std::is_pointer_v<T>) {
    T *key_array = reinterpret_cast<T *>(workload);
    for (uint64_t i = begin; i < end; ++i) {
      Value_t value = reinterpret_cast<Value_t>(key_array[i] + i);
      int ret = index->Update(key_array[i], value);
      mfence();
    }
  } else {
    T var_key;
    uint64_t string_key_size = sizeof(string_key) + _range->length;
    for (uint64_t i = begin; i < end; ++i) {
      var_key = reinterpret_cast<T>(workload + string_key_size * i);
      Value_t value = reinterpret_cast<Value_t>(i);
      index->Update(var_key, value);
      mfence();
    }
  }

  end_notify(_range);
}

template <class T>
void concurr_search(struct range *_range, Hash<T> *index) {
  set_affinity(_range->index);
  AAllocator::Get()->Initialize_thread(_range->index);
  int begin = _range->begin;
  int end = _range->end;
  char *workload = reinterpret_cast<char *>(_range->workload);
  T key;
  uint64_t not_found = 0;
  char* value = new char[value_length];
  clk = new nsTimer[10];
  spin_wait();

  if constexpr (!std::is_pointer_v<T>) {
    T *key_array = reinterpret_cast<T *>(workload);
    for (uint64_t i = begin; i < end; ++i) {
      if (index->Get(key_array[i], (Value_t*)value) == false) {
        not_found++;
      }
      mfence();
    }
  } else {
    T var_key;
    uint64_t string_key_size = sizeof(string_key) + _range->length;
    for (uint64_t i = begin; i < end; ++i) {
      var_key = reinterpret_cast<T>(workload + string_key_size * i);
      if (index->Get(var_key, (Value_t*)value) == false) {
        not_found++;
      }
      mfence();
    }
  }
  std::cout << "not_found = " << not_found << std::endl;
  end_notify(_range);
}

template <class T>
void concurr_delete(struct range *_range, Hash<T> *index) {
  set_affinity(_range->index);
  AAllocator::Get()->Initialize_thread(_range->index);
  int begin = _range->begin;
  int end = _range->end;
  char *workload = reinterpret_cast<char *>(_range->workload);
  T key;
  uint64_t not_found = 0;

  spin_wait();
  if constexpr (!std::is_pointer_v<T>) {
    T *key_array = reinterpret_cast<T *>(workload);
    for (uint64_t i = begin; i < end; ++i) {
      if (index->Delete(key_array[i]) == false) {
        not_found++;
      }
      mfence();
    }
  } else {
    T var_key;
    int string_key_size = sizeof(string_key) + _range->length;
    for (uint64_t i = begin; i < end; ++i) {
      var_key = reinterpret_cast<T>(workload + string_key_size * i);
      if (index->Delete(var_key) == false) {
        not_found++;
      }
    }
  }
  end_notify(_range);
}

template <class T>
void mixed(struct range *_range, Hash<T> *index) {
  set_affinity(_range->index);
  AAllocator::Get()->Initialize_thread(_range->index);
  uint64_t begin = _range->begin;
  uint64_t end = _range->end;
  char *workload = reinterpret_cast<char *>(_range->workload);
  T *key_array = reinterpret_cast<T *>(_range->workload);
  T key;
  int string_key_size = sizeof(string_key) + _range->length;

  UniformRandom rng(_range->random_num);
  uint32_t random;
  uint32_t not_found = 0;
  uint64_t ins_num = 0;

  uint32_t insert_sign = (uint32_t)(insert_ratio * 100);
  uint32_t read_sign = (uint32_t)(read_ratio * 100) + insert_sign;
  uint32_t delete_sign = (uint32_t)(delete_ratio * 100) + read_sign;
  char* value = new char[value_length];;
  clk = new nsTimer[10];
  spin_wait();

  for (uint64_t i = begin; i < end; ++i) {
    if constexpr (std::is_pointer_v<T>) { /* variable length*/
      key = reinterpret_cast<T>(workload + string_key_size * i);
    } else {
      key = key_array[i];
    }

    random = rng.next_uint32() % 100;
    if (random < insert_sign) { /*insert*/
      clk[0].start();
      index->Insert(key, DEFAULT);
      clk[0].end();
      ins_num++;
    } else if (random < read_sign) { /*get*/
      if (index->Get(key, (Value_t*)value) == false) {
        not_found++;
      }
    } else { /*delete*/
      index->Delete(key);
    }
    mfence();
  }
  end_notify(_range);
}

template <class T>
void GeneralBench(range *rarray, Hash<T> *index, int thread_num,
                  uint64_t operation_num, std::string profile_name,
                  void (*test_func)(struct range *, Hash<T> *)) {
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
  for (uint64_t i = 0; i < thread_num; ++i) {
    thread_array[i] = new std::thread(*test_func, &rarray[i], index);
  }

  while (LOAD(&bar_b) != 0)
    ;                                     // Spin
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
 
  double longest = (double)(rarray[0].tv.tv_usec - tv1.tv_usec) / 1000000 +
                   (double)(rarray[0].tv.tv_sec - tv1.tv_sec);
  double shortest = longest;
  duration = longest;

  for (int i = 1; i < thread_num; ++i) {
    double interval = (double)(rarray[i].tv.tv_usec - tv1.tv_usec) / 1000000 +
                      (double)(rarray[i].tv.tv_sec - tv1.tv_sec);
    duration += interval;
    if (shortest > interval) shortest = interval;
    if (longest < interval) longest = interval;
  }
  duration = duration / thread_num;
  printf(
      "%d threads, Time = %f s, throughput = %f "
      "ops/s, latency = %f us, fastest = %f, slowest = %f\n",
      thread_num, duration, operation_num / duration, (duration * 1000000) / operation_num, operation_num / shortest,
      operation_num / longest);
  std::cout << profile_name << " End" << std::endl;
}

void *GenerateInsRangePosUniformWorkload(uint64_t load_num, uint64_t ins_num, 
                                        uint64_t pos_num, uint64_t neg_num) {
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
    uint64_t exist_num = load_num + ins_num;
    workload = (char *)malloc((exist_num + pos_num + neg_num) * item_size);
    key_generator_t *range_generator = new range_key_generator_t(1);
    // load and insert workload (all range)
    generate_NB(workload, exist_num, range_generator, -1);
    // pos search workload (uniform in the range of 'load + insert')
    generate_NB(workload + exist_num * item_size, pos_num, uniform_generator, exist_num);
    // neg search workload (range behind exist's range)
    generate_NB(workload + (exist_num + pos_num) * item_size, neg_num, range_generator, -1);
    std::cout << "Finish InsRangePosUniform workload Generation" << std::endl;
  return workload;
}

void *GenerateSkewWorkload(uint64_t load_num, uint64_t ins_num,
                           uint64_t pos_num, uint64_t neg_num) {
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

  workload = (char *)malloc((load_num + ins_num + pos_num + neg_num) * item_size);    
  // load workload (range in 1 - n)
  key_generator_t *range_generator = new range_key_generator_t(1);
  generate_NB(workload, load_num, range_generator, -1);
  delete range_generator;

  if (update_num == 0) {
    // insert workload (skew in n+1 - n+p)
    key_generator_t *skew_generator = new zipfian_key_generator_t(load_num + 1, load_num + ins_num, skew_factor);
    generate_NB(workload + load_num * item_size, ins_num, skew_generator, -1);
    delete skew_generator;
    // pos search workload (skew in 1 - n)
    skew_generator = new zipfian_key_generator_t(1, load_num, skew_factor);
    generate_NB(workload + (load_num + ins_num) * item_size, pos_num, skew_generator, -1);
    delete skew_generator;
  } else {
    // insert workload (range in n+1 - n+p)
    key_generator_t *range_generator2 = new range_key_generator_t(load_num + 1);
    generate_NB(workload + load_num * item_size, ins_num, range_generator2, -1);
    delete range_generator2;
    // pos search workload (skew in 1 - n+p)
    key_generator_t *skew_generator = new zipfian_key_generator_t(1, load_num + ins_num, skew_factor);
    generate_NB(workload + (load_num + ins_num) * item_size, pos_num, skew_generator, -1);
    delete skew_generator;
  }

  // neg search workload (skew in n+p+1 - n+2p)
  key_generator_t *skew_generator = new zipfian_key_generator_t(load_num + ins_num + 1, load_num + ins_num * 2, skew_factor);
  generate_NB(workload + (load_num + ins_num + pos_num) * item_size, neg_num, skew_generator, -1);
  delete skew_generator;

  std::cout << "Finish Generation" << std::endl;
  
  return workload;
}

template <class T>
void Run() {
  /* Initialize Index for Finger_EH*/
  uniform_generator = new uniform_key_generator_t();
  Hash<T> *index = InitializeIndex<T>(initCap);
  uint64_t generate_num = operation_num * 2 + load_num;
  /* Generate the workload and corresponding range array*/
  std::cout << "Generate workload" << std::endl;
  void *workload;
  if (distribution == "uniform") {
    workload = GenerateInsRangePosUniformWorkload(load_num, operation_num, operation_num, operation_num);
  } else {
    if (update_num == 0)
      workload = GenerateSkewWorkload(load_num, operation_num, operation_num, operation_num);
    else
      workload = GenerateSkewWorkload(load_num, operation_num, update_num, update_num);
  }

  void *insert_workload = workload;
  std::cout << "Finish Generate workload" << std::endl;

  std::cout << "load num = " << load_num << std::endl;
  Load<T>(load_num, index, key_length, insert_workload);
  void *not_used_workload;
  void *not_used_insert_workload;

  if (key_type == "fixed") {
    uint64_t *key_array = reinterpret_cast<uint64_t *>(workload);
    not_used_workload = reinterpret_cast<void *>(key_array + load_num);
  } else {
    char *key_array = reinterpret_cast<char *>(workload);
    not_used_workload = key_array + (sizeof(string_key) + key_length) * load_num; 
  }
  not_used_insert_workload = not_used_workload;

  /* Description of the workload*/
  srand((unsigned)time(NULL));
  struct range *rarray;
  uint64_t chunk_size = operation_num / thread_num;
  rarray = reinterpret_cast<range *>(malloc(thread_num * (sizeof(range))));
  for (int i = 0; i < thread_num; ++i) {
    rarray[i].index = i;
    rarray[i].random_num = rand();
    rarray[i].begin = i * chunk_size;
    rarray[i].end = (i + 1) * chunk_size;
    rarray[i].length = key_length;
    rarray[i].workload = not_used_workload;
  }
  rarray[thread_num - 1].end = operation_num;

  /* Benchmark Phase */
  if (operation == "insert") {
    std::cout << "Insert-only Benchmark" << std::endl;
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].workload = not_used_insert_workload;
    }
    GeneralBench<T>(rarray, index, thread_num, operation_num, "Insert", &concurr_insert);
  } else if (operation == "pos") {
    if (!load_num) {
      std::cout << "Please first specify the # pre_load keys!" << std::endl;
      return;
    }
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].workload = workload;
    }
    GeneralBench<T>(rarray, index, thread_num, operation_num, "Pos_search", &concurr_search);
  } else if (operation == "neg") {
    if (!load_num) {
      std::cout << "Please first specify the # pre_load keys!" << std::endl;
      return;
    }
    GeneralBench<T>(rarray, index, thread_num, operation_num, "Neg_search", &concurr_search);
  } else if (operation == "delete") {
    if (!load_num) {
      std::cout << "Please first specify the # pre_load keys!" << std::endl;
      return;
    }
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].workload = workload;
    }
    GeneralBench<T>(rarray, index, thread_num, operation_num, "Delete", &concurr_delete);
  } else { /*do the benchmark for all single operations*/
    std::cout << "Comprehensive Benchmark" << std::endl;
    std::cout << "insertion start" << std::endl;
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].workload = not_used_insert_workload;
    }

    /* insert */
    GeneralBench<T>(rarray, index, thread_num, operation_num, "Insert", &concurr_insert);
    index->getNumber();

    /* update or mixed */
    if (update_num != 0)
      chunk_size = update_num / thread_num;
    
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].workload = not_used_workload;
    }

    for (int i = 0; i < thread_num; ++i) {
      rarray[i].begin = operation_num + i * chunk_size;
      rarray[i].end = operation_num + (i + 1) * chunk_size;
    }
    if (update_num == 0) {
      rarray[thread_num - 1].end = 2 * operation_num;
      GeneralBench<T>(rarray, index, thread_num, operation_num, "Update", &concurr_update);
    } else {
      rarray[thread_num - 1].end = operation_num + update_num;
      if (operation == "mixed")
        GeneralBench<T>(rarray, index, thread_num, update_num, "Mixed", &mixed);
      else
        GeneralBench<T>(rarray, index, thread_num, update_num, "Update", &concurr_update);

      return;
    }
    
    index->getNumber();

    /* pos search */
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].begin = operation_num + i * chunk_size;
      rarray[i].end = operation_num + (i + 1) * chunk_size;
    }
    rarray[thread_num - 1].end = 2 * operation_num;

    GeneralBench<T>(rarray, index, thread_num, operation_num, "Pos_search", &concurr_search);

    /* neg search */
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].begin = operation_num * 2 + i * chunk_size;
      rarray[i].end = operation_num * 2 + (i + 1) * chunk_size;
    }
    rarray[thread_num - 1].end = 3 * operation_num;

    GeneralBench<T>(rarray, index, thread_num, operation_num, "Neg_search", &concurr_search);

    /* delete */
    for (int i = 0; i < thread_num; ++i) {
      rarray[i].begin = i * chunk_size;
      rarray[i].end = (i + 1) * chunk_size;
    }
    rarray[thread_num - 1].end = operation_num;

    GeneralBench<T>(rarray, index, thread_num, operation_num, "Delete", &concurr_delete);
    index->getNumber();
  }

  /*TODO Free the workload memory*/
}

bool check_ratio() {
  int read_portion = (int)(read_ratio * 100);
  int insert_portion = (int)(insert_ratio * 100);
  int delete_portion = (int)(delete_ratio * 100);
  if ((read_portion + insert_portion + delete_portion) != 100) return false;
  return true;
}

int main(int argc, char *argv[]) {
  #ifdef VALUE_LENGTH_VARIABLE
  std::cout << "Variable Value Length!" << std::endl;
  #endif
  set_affinity(0);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  initCap = FLAGS_i;
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
  distribution = FLAGS_distribution;
  load_type = FLAGS_loadType;
  std::cout << "Distribution = " << distribution << std::endl;
  std::string fixed("fixed");
  operation = FLAGS_op;
  msec = FLAGS_ms;
  pool_size = FLAGS_ps * 1024ul * 1024ul * 1024ul; /*pool_size*/
  key_length = FLAGS_vkl;
  std::cout << "Variable key length = " << key_length << std::endl;
  value_length = FLAGS_vvl;
  std::cout << "Variable value length = " << value_length << std::endl;
  hot_bit = FLAGS_hb;
  hot_num = 1 << hot_bit;
  std::cout << "Hot bits = " << hot_bit << std::endl;
  size_t kv_length = value_length;
  if (kv_length >= 128)
    update_retry_time = 2;
  if (kv_length >= 256)
    update_retry_time = 1;
  std::cout << "Update retry time = " << update_retry_time << std::endl;

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

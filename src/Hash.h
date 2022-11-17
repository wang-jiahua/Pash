#ifndef HASH_INTERFACE_H_
#define HASH_INTERFACE_H_

#include "../util/pair.h"
#ifdef PMEM
#include <libpmemobj.h>
#endif
#include "allocator.h"

#define APPEND_ALLOCATE

/*
* Parent function of hash indexes
* Used to define the interface of the hash indexes
*/

template <class T>
class Hash {
 public:
  Hash(void) = default;
  ~Hash(void) = default;
  virtual int Insert(T, Value_t) = 0;
  virtual bool Delete(T) = 0;
  virtual bool Get(T, Value_t*) = 0;
  virtual int Update(T key, Value_t value) { return Insert(key, value); }
  virtual void Recovery() = 0;
  virtual void getNumber() = 0;
};

#endif  // _HASH_INTERFACE_H_

#ifndef HASH_INTERFACE_H_
#define HASH_INTERFACE_H_

#include "../util/pair.h"
#ifdef PMEM
#include <libpmemobj.h>
#endif
#include "allocator.h"

/*
 * Parent function of hash indexes
 * Used to define the interface of the hash indexes
 */

template <class T>
class Hash {
public:
    Hash(void) = default;
    ~Hash(void) = default;
    virtual int Insert(T, Value_t, int batch_offset = -1, T *batch_array = nullptr, nsTimer *clks = nullptr) = 0;
    virtual bool Delete(T, int batch_offset = -1, T *batch_array = nullptr, nsTimer *clks = nullptr) = 0;
    virtual bool Get(T, Value_t *, int batch_offset = -1, T *batch_array = nullptr, nsTimer *clks = nullptr) = 0;
    virtual void Recovery() = 0;
    virtual void getNumber() = 0;
    virtual bool readyForNextStage() { return true; }
    virtual int Update(T key, Value_t value, int batch_offset = -1, T *batch_array = nullptr, nsTimer *clks = nullptr) {
        return Insert(key, value, batch_offset, batch_array, clks);
    }
};

#endif  // _HASH_INTERFACE_H_

#ifndef TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_MKL_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_MKL_ALLOCATOR_H_

#include <cstdlib>
#include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"


#include <atomic>
#include <stack>
#include <vector>
#include <unordered_map>

namespace tensorflow {

class TensorPoolMklAllocator : public TensorPoolAllocator {
 public:
  TensorPoolMklAllocator();
  ~TensorPoolMklAllocator() override { delete large_size_allocator_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  static constexpr size_t kDefaultMaxLimit = 64LL << 30;
  static const bool kAllowGrowth = true;
  static constexpr const char* kName = "tensor_pool_mkl_allocator";

  inline bool LargeAlloc(size_t s) {
    return s > kLargeAllocationsThreshold;
  }

  inline bool IsLargeSizeAllocation(const void* ptr) const
    LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    return large_allocations_map_.find(ptr) != large_allocations_map_.end();
  }

  // AddLargeAllocMap and RemoveLargeAllocMap are always called with a lock held
  inline void AddLargeAllocMap(void* ptr, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (ptr != nullptr) {
      std::pair<void*, size_t> map_val(ptr, num_bytes);
      large_allocations_map_.insert(map_val);
    }
  }

  inline void RemoveLargeAllocMap(void* ptr) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    auto map_iter = large_allocations_map_.find(ptr);
    if (map_iter != large_allocations_map_.end()) {
      large_allocations_map_.erase(map_iter);
    } else {
      LOG(ERROR) << "tried to deallocate invalid pointer";
    }
    return;
  }

 private:
  Allocator* large_size_allocator_ = nullptr;              // owned by this class
  // Hash map to keep track of "BFC" allocations
  // We do not use BFC allocator for small allocations.
  std::unordered_map<const void*, size_t> large_allocations_map_
      GUARDED_BY(mutex_);

  mutable mutex mutex_;

  // Size in bytes that defines the upper-bound for "small" allocations.
  // Any allocation above this threshold is "large" allocation.
  int64 kLargeAllocationsThreshold;
};

}
#endif // TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_MKL_ALLOCATOR_H_
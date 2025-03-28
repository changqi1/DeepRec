#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/tensorpool_mkl_allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/env_var.h"
#include <sys/time.h>

#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {

namespace {
  constexpr int64 DEFAULT_TENSORPOOL_MKL_LARGE_SIZE = (512 << 10);
}

namespace {
void* SetLightHeader(void* p, size_t total_bytes, size_t header_size) {
  // LightHeader *KB max(sizeof(LightHeader)=8B, alignment)
  //   { | .....| checksum (4B) | header_size (4B)}
  auto user_ptr = (char*)p + header_size;
  new((char*)user_ptr - sizeof(LightHeader)) LightHeader(header_size);

  return user_ptr;
}

LightHeader* GetLightHeader(void* p) {
  auto light_header = (LightHeader*)((char*)p - sizeof(LightHeader));

  return (strcmp(light_header->checksum, CHECK_SUM.c_str()) == 0)
           ? light_header
           : nullptr;
}

Header* GetHeader(void* p) {
  auto header = (Header*)((char*) p - sizeof(Header));

  if (header->user_ptr != p) {
    auto light_header = GetLightHeader(p);
    LOG(FATAL) << "Memory corruption!"
               << ", p:" << p
               << ", p->header_size:" << light_header->header_size
               << ", p->checksum:" << light_header->checksum;
  }

  return header;
}
}

TensorPoolMklAllocator::TensorPoolMklAllocator()
    : TensorPoolAllocator() {
  Status s = ReadInt64FromEnvVar("TENSORPOOL_MKL_LARGE_SIZE",
    DEFAULT_TENSORPOOL_MKL_LARGE_SIZE,
    &kLargeAllocationsThreshold);
  uint64 max_mem_bytes = kDefaultMaxLimit;
  large_size_allocator_ =
    new BFCAllocator(sub_allocator_.get(), max_mem_bytes, kAllowGrowth, kName);
}

void* TensorPoolMklAllocator::AllocateRaw(size_t alignment,
    size_t num_bytes) {
  void* ret;
  if (SmallAlloc(num_bytes)) {
    auto header_size = std::max(sizeof(LightHeader), alignment);
    auto total = num_bytes + header_size;
    auto ptr = sub_allocator_->Alloc(alignment, total);
    ret = SetLightHeader(ptr, total, header_size);

    return ret;
  }

  if (LargeAlloc(num_bytes)) {
    VLOG(1) << "Large allocate " << num_bytes << " bytes.";
    mutex_lock l(mutex_);
    ret = large_size_allocator_->AllocateRaw(alignment, num_bytes);
    AddLargeAllocMap(ret, num_bytes);

    return ret;
  }

  if (unlikely(stats_)) {
    ret = BigAllocateStatistic(alignment, num_bytes);
  } else {
    ret = BigAllocate(alignment, num_bytes);
  }

  return ret;
}

void TensorPoolMklAllocator::DeallocateRaw(void* ptr) {
  VLOG(1) << "DeallocateRaw " << Name() << " "
          << (ptr ? RequestedSize(ptr) : 0);

  auto light_header = GetLightHeader(ptr);
  if (light_header != nullptr) {
    auto header_size = light_header->header_size;
    auto raw_ptr = (char*)ptr - header_size;
    // LightHeader not record allocation size
    // Free interface ignore the freed num_bytes
    sub_allocator_->Free(raw_ptr, 0);
    return;
  }

  if (IsLargeSizeAllocation(ptr)) {
    mutex_lock l(mutex_);
    RemoveLargeAllocMap(ptr);
    large_size_allocator_->DeallocateRaw(ptr);
    return;
  }

  auto header = GetHeader(ptr);
  BigDeallocate(header);
}
} // tensorflow

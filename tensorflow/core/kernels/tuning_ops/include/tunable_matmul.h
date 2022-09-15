#ifndef __TUNABLE_MATMUL_H
#define __TUNABLE_MATMUL_H
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <emmintrin.h>
#include "timer.h"
#include "sgemm_kernel.h"
#include "host_proxy_manager.h"

#include "tensorflow/core/lib/core/threadpool.h"

#include <mutex>
std::mutex mMutex;

namespace tensorflow {
namespace thread {
class ThreadPool;
}
}

void ShowLog(const std::string& msg);

#define CACHELINE_SIZE 64
#define MAX_GROUP_LIMIT 8

typedef float T;

typedef std::function<bool(std::vector<int> const &param)> VerifyFunc;

template<class T>
struct TuningParam {
    const std::string name;
    std::pair<int, int> min_max_index;
    std::vector<T> min_max_value;
    VerifyFunc verify_func;
};

struct PerfStat
{
  float avg_latency;
  float min_latency;
  float max_latency;
  float variance;
  int samples;
};

typedef void (*KERNEL_FIXMN)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K);
typedef void (*KERNEL_FIXN)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K);
typedef void (*KERNEL_FIXM)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K);
typedef void (*KERNEL_NOFIX)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K);

struct SmallKernels
{
  KERNEL_FIXMN kernel_fixmn_acc;
  KERNEL_FIXMN kernel_fixmn_nonacc;
  KERNEL_FIXM kernel_fixm_acc;
  KERNEL_FIXM kernel_fixm_nonacc;
  KERNEL_FIXN kernel_fixn_acc;
  KERNEL_FIXN kernel_fixn_nonacc;
  KERNEL_NOFIX kernel_nofix_acc;
  KERNEL_NOFIX kernel_nofix_nonacc;
};

struct MatmulSize
{
  // A: m*k; B: k*n; C: m*n
  int m;
  int n;
  int k;
  // leading dimension
  int lda;
  int ldb;
  int ldc;
  // block size (how many elements inside the block)
  int bm;
  int bn;
  int bk;
  // group number (totally how many groups along m/n/k dimension)
  int mgroups;
  int ngroups;
  int kgroups;
  // block number in each group along m/n/k dimension
  int mblocks_per_group;
  int nblocks_per_group;
  int kblocks_per_group;

  bool ta;
  bool tb;

  // tensorflow::thread::ThreadPool* thread_pool;
  const Eigen::ThreadPoolDevice* thread_pool;
};

// Inner implementation of matmul
typedef void (*INNER_MATMUL_FUNC)(const T *A, const T *B, T *C,
                                  const MatmulSize &mmsize, const SmallKernels &kernels);

struct MatmulConfig
{
  INNER_MATMUL_FUNC impl;

  SmallKernels kernels;

  MatmulSize mmsize;

  const T *A;

  const T *B;

  T *C;
};

#define B_G_M 0
#define B_G_N 1
#define B_G_K 2
#define B_G_M_NAME "bm_mg_pair"
#define B_G_N_NAME "bn_ng_pair"
#define B_G_K_NAME "bk_kg_pair"
#define IMPL 3
#define IMPL_NAME "impls"

#define B_M 0
#define B_N 1
#define B_K 2
#define G_M 3
#define G_N 4
#define G_K 5

#define SET_KERNELS(bm, bn)                                            \
    kernels.kernel_fixmn_acc = small_gemm_fixmn<(bm), (bn), true>;     \
    kernels.kernel_fixmn_nonacc = small_gemm_fixmn<(bm), (bn), false>; \
    kernels.kernel_fixm_acc = small_gemm_fixm<(bm), true>;             \
    kernels.kernel_fixm_nonacc = small_gemm_fixm<(bm), false>;         \
    kernels.kernel_fixn_acc = small_gemm_fixn<(bn), true>;             \
    kernels.kernel_fixn_nonacc = small_gemm_fixn<(bn), false>;

#define SET_KERNELS_ENUM_BN(bm)                       \
    if (bn == 32)                                     \
    {                                                 \
        SET_KERNELS((bm), 32);                        \
    }                                                 \
    else if (bn == 48)                                \
    {                                                 \
        SET_KERNELS((bm), 48);                        \
    }                                                 \
    else if (bn == 64)                                \
    {                                                 \
        SET_KERNELS((bm), 64);                        \
    }                                                 \
    else if (bn == 80)                                \
    {                                                 \
        SET_KERNELS((bm), 80);                        \
    }                                                 \
    else                                              \
    {                                                 \
        printf("Unsupported kernel for bn=%d\n", bn); \
        exit(-1);                                     \
    }

static T* transpose(const T *src, T *dst, int src_stride, int src_length, int src_ld, int dst_ld) {
//  int ret = libxsmm_trans(dst, src, src_stride, src_length, src_ld, dst_ld);
  for(int i = 0; i < src_length; i++){
    for(int j = 0; j < src_stride; j++){
      dst[j * dst_ld + i] = src[i * src_ld + j];
    }
  }
  return dst;
}

#define LOOP0 for (int kg = 0; kg < mmsize.kgroups; ++kg)
#define LOOP1 for (int ig = 0; ig < mmsize.mgroups; ++ig)
#define LOOP2 for (int jg = 0; jg < mmsize.ngroups; ++jg)
#define LOOP3 for (int i = 0; i < mmsize.mblocks_per_group; ++i)
#define LOOP4 for (int j = 0; j < mmsize.nblocks_per_group; ++j)
#define LOOP5 for (int k = 0; k < mmsize.kblocks_per_group; ++k)

#define LOOPE { task_pool_param.push_back(std::make_tuple(ig, jg, i, j)); }
#define LOOPE2 { task_pool_param.push_back(std::make_tuple(ig, jg)); }

// Equals to: //#pragma omp parallel for collapse(4)
#define PARALLEL_C4 _Pragma("omp parallel for collapse(4)")
#define PARALLEL_C2 _Pragma("omp parallel for collapse(2)")

#define FUNC_DEF_HEAD(name)                                                                             \
  static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) \
  {                                                                                                     \
    std::vector<std::tuple<int, int, int, int>> task_pool_param;                                        \
    task_pool_param.reserve(                                                                            \
      mmsize.mgroups * mmsize.ngroups * mmsize.mblocks_per_group * mmsize.nblocks_per_group);           \

#define FUNC_DEF_HEAD2(name)                                                                            \
  static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) \
  {                                                                                                     \
    std::vector<std::tuple<int, int>> task_pool_param;                                                  \
    task_pool_param.reserve(mmsize.mgroups * mmsize.ngroups);                                           \


#ifndef USE_LIBXSMM												 \

#define FUNC_CORE_CAL                                                                                            \
  {                    												 \
    T *fake_A = mmsize.ta ? new T[mmsize.bm * mmsize.bk] : nullptr;                                              \
    T *fake_B = mmsize.tb ? new T[mmsize.bk * mmsize.bn] : nullptr;                                              \
    int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;                                                 \
    int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;                                                 \
    int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;                                                 \
    if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k)                                                \
    {                                                                                                            \
      int lda = mmsize.ta ? mmsize.bk : mmsize.lda;                                                              \
      int ldb = mmsize.tb ? mmsize.bn : mmsize.ldb;                                                              \
      int realbm = mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);                               \
      int realbn = mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);                               \
      int realbk = mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);                               \
      const T *pa = mmsize.ta ? transpose(&A[k_off * mmsize.lda + i_off], fake_A, realbm, realbk, mmsize.lda, mmsize.bk) : &A[i_off * mmsize.lda + k_off]; \
      const T *pb = mmsize.tb ? transpose(&B[j_off * mmsize.ldb + k_off], fake_B, realbk, realbn, mmsize.ldb, mmsize.bn) : &B[k_off * mmsize.ldb + j_off]; \
      T *pc = &C[i_off * mmsize.ldc + j_off];                                                                    \
      if (realbm == mmsize.bm)                                                                                   \
      {                                                                                                          \
        if (realbn == mmsize.bn)                                                                                 \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixmn_acc(pa, pb, pc, lda, ldb, mmsize.ldc, realbk);                                  \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixmn_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc, realbk);                               \
          }                                                                                                      \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixm_acc(pa, pb, pc, lda, ldb, mmsize.ldc, realbn, realbk);                           \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixm_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc, realbn, realbk);                        \
          }                                                                                                      \
        }                                                                                                        \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        if (realbn == mmsize.bn)                                                                                 \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixn_acc(pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbk);                           \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixn_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbk);                        \
          }                                                                                                      \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_nofix_acc(pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk);                  \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_nofix_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk);               \
          }                                                                                                      \
        }                                                                                                        \
      }                                                                                                          \
    }                                                                                                            \
    if (mmsize.ta) delete[] fake_A;									 	 \
    if (mmsize.tb) delete[] fake_B;										 \
  }                                                                                                              \

#else

#define FUNC_CORE_CAL                                                                                            \
  {                    												 \
    int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;                                                 \
    int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;                                                 \
    int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;                                                 \
    if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k)                                                \
    {                                                                                                            \
      int realbm = mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);                               \
      int realbn = mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);                               \
      int realbk = mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);                               \
      T *pc = &C[i_off * mmsize.ldc + j_off];                                                                    \
      const T* pa = mmsize.ta? &A[k_off * mmsize.lda + i_off] : &A[i_off * mmsize.lda + k_off];			\
      const T* pb = mmsize.tb? &B[j_off * mmsize.ldb + k_off] : &B[k_off * mmsize.ldb + j_off];			\
      int lda = mmsize.ta ? mmsize.m : mmsize.lda;							\
      int ldb = mmsize.tb ? mmsize.k : mmsize.ldb;							\
      if (k_off != 0) {											\
      	small_gemm_libxsmm(mmsize.ta, mmsize.tb, pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk, true);	\
      } else {														\
      	small_gemm_libxsmm(mmsize.ta, mmsize.tb, pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk, false);	\
      }															\
    }                                                                                                            \
  }                                                                                                              \

#endif																\


#define FUNC_DEF_TAIL                                                       \
  int kg = 0;                                                               \
  auto _worker = [&](int64_t begin, int64_t end) -> void {                  \
    int ig, jg, i, j;                                                       \
    for(int index=begin; index < end; index++){                             \
      std::tie(ig, jg, i, j) = task_pool_param[index];                      \
      for (int k = 0; k < mmsize.kblocks_per_group; ++k){                   \
        FUNC_CORE_CAL                                                       \
      }                                                                     \
    }                                                                       \
  };                                                                        \
  auto cost = Eigen::TensorOpCost(4*128*128*2, 4*128*128*2, 128*128*2);     \
  for (; kg < mmsize.kgroups; ++kg){                                        \
    mmsize.thread_pool->parallelFor(task_pool_param.size(), cost, _worker); \
  }                                                                         \
}                                                                           \

#define FUNC_DEF_MID2                                                       \
  int kg = 0;                                                               \
  auto _worker = [&](int64_t begin, int64_t end) -> void {                  \
    int ig, jg;                                                             \
    for(int index=begin; index < end; index++){                             \
      std::tie(ig, jg) = task_pool_param[index];                            \


#define FUNC_DEF_TAIL2                                                      \
        FUNC_CORE_CAL                                                       \
    }                                                                       \
  };                                                                        \
  auto cost = Eigen::TensorOpCost(4*128*128*2, 4*128*128*2, 128*128*2);     \
  for (; kg < mmsize.kgroups; ++kg){                                        \
    mmsize.thread_pool->parallelFor(task_pool_param.size(), cost, _worker); \
  }                                                                         \
}                                                                           \

static void v1(const T* A, const T* B, T* C, const MatmulSize& mmsize,
               const SmallKernels& kernels) {
  std::vector<std::tuple<int, int, int, int>> task_pool_param;
  task_pool_param.reserve(mmsize.mgroups * mmsize.ngroups *
                          mmsize.mblocks_per_group * mmsize.nblocks_per_group);

  for (int ig = 0; ig < mmsize.mgroups; ++ig)
    for (int jg = 0; jg < mmsize.ngroups; ++jg)
      for (int i = 0; i < mmsize.mblocks_per_group; ++i)
        for (int j = 0; j < mmsize.nblocks_per_group; ++j) {
          task_pool_param.push_back(std::make_tuple(ig, jg, i, j));
        }

  int kg = 0;
  auto _worker = [&](int64_t begin, int64_t end) -> void {
    int ig, jg, i, j;
    for (int index = begin; index < end; index++) {
      std::tie(ig, jg, i, j) = task_pool_param[index];
      for (int k = 0; k < mmsize.kblocks_per_group; ++k) {
        {
          int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;
          int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;
          int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;
          if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k) {
            int realbm =
                mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);
            int realbn =
                mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);
            int realbk =
                mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);
            T* pc = &C[i_off * mmsize.ldc + j_off];
#ifndef USE_LIBXSMM
            T* fake_A = mmsize.ta ? new T[mmsize.bm * mmsize.bk] : nullptr;
            T* fake_B = mmsize.tb ? new T[mmsize.bk * mmsize.bn] : nullptr;
            int lda = mmsize.ta ? mmsize.bk : mmsize.lda;
            int ldb = mmsize.tb ? mmsize.bn : mmsize.ldb;
            const T* pa = mmsize.ta
                              ? transpose(&A[k_off * mmsize.lda + i_off],
                                          fake_A, realbm, realbk, mmsize.lda, mmsize.bk)
                              : &A[i_off * mmsize.lda + k_off];
            const T* pb = mmsize.tb
                              ? transpose(&B[j_off * mmsize.ldb + k_off],
                                          fake_B, realbk, realbn, mmsize.ldb, mmsize.bn)
                              : &B[k_off * mmsize.ldb + j_off];
            if (realbm == mmsize.bm) {
              if (realbn == mmsize.bn) {
                if (k_off != 0) {
                  kernels.kernel_fixmn_acc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                           realbk);
                } else {
                  kernels.kernel_fixmn_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                              realbk);
                }
              } else {
                if (k_off != 0) {
                  kernels.kernel_fixm_acc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                          realbn, realbk);
                } else {
                  kernels.kernel_fixm_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                             realbn, realbk);
                }
              }
            } else {
              if (realbn == mmsize.bn) {
                if (k_off != 0) {
                  kernels.kernel_fixn_acc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                          realbm, realbk);
                } else {
                  kernels.kernel_fixn_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                             realbm, realbk);
                }
              } else {
                if (k_off != 0) {
                  kernels.kernel_nofix_acc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                           realbm, realbn, realbk);
                } else {
                  kernels.kernel_nofix_nonacc(pa, pb, pc, lda, ldb, mmsize.ldc,
                                              realbm, realbn, realbk);
                }
              }
            }
            if (mmsize.ta) delete[] fake_A;
            if (mmsize.tb) delete[] fake_B;
#else
            const T* pa = mmsize.ta
                              ? &A[k_off * mmsize.lda + i_off] : &A[i_off * mmsize.lda + k_off];
            const T* pb = mmsize.tb
                              ? &B[j_off * mmsize.ldb + k_off] : &B[k_off * mmsize.ldb + j_off];
            int lda = mmsize.ta ? mmsize.m : mmsize.lda;
            int ldb = mmsize.tb ? mmsize.k : mmsize.ldb;
            if (k_off != 0) {
	    	small_gemm_libxsmm(mmsize.ta, mmsize.tb, pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk, true);
	    } else {
	    	small_gemm_libxsmm(mmsize.ta, mmsize.tb, pa, pb, pc, lda, ldb, mmsize.ldc, realbm, realbn, realbk, false);
            }

#endif
          }
        }
      }
    }
  };
  auto cost =
      Eigen::TensorOpCost(4 * 128 * 128 * 2, 4 * 128 * 128 * 2, 128 * 128 * 2);
  for (; kg < mmsize.kgroups; ++kg) {
    mmsize.thread_pool->parallelFor(task_pool_param.size(), cost, _worker);
  }
}                                                                   

  //  FUNC_DEF_HEAD(v1) LOOP1 LOOP2 LOOP3 LOOP4 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v2) LOOP1 LOOP2 LOOP4 LOOP3 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v3) LOOP1 LOOP3 LOOP2 LOOP4 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v4) LOOP1 LOOP3 LOOP4 LOOP2 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v5) LOOP1 LOOP4 LOOP2 LOOP3 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v6) LOOP1 LOOP4 LOOP3 LOOP2 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v7) LOOP2 LOOP1 LOOP3 LOOP4 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v8) LOOP2 LOOP1 LOOP4 LOOP3 LOOPE FUNC_DEF_TAIL
   FUNC_DEF_HEAD(v9) LOOP2 LOOP3 LOOP1 LOOP4 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v10) LOOP2 LOOP3 LOOP4 LOOP1 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v11) LOOP2 LOOP4 LOOP1 LOOP3 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v12) LOOP2 LOOP4 LOOP3 LOOP1 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v13) LOOP3 LOOP1 LOOP2 LOOP4 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v14) LOOP3 LOOP1 LOOP4 LOOP2 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v15) LOOP3 LOOP2 LOOP1 LOOP4 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v16) LOOP3 LOOP2 LOOP4 LOOP1 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v17) LOOP3 LOOP4 LOOP1 LOOP2 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v18) LOOP3 LOOP4 LOOP2 LOOP1 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v19) LOOP4 LOOP1 LOOP2 LOOP3 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v20) LOOP4 LOOP1 LOOP3 LOOP2 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v21) LOOP4 LOOP2 LOOP1 LOOP3 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v22) LOOP4 LOOP2 LOOP3 LOOP1 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v23) LOOP4 LOOP3 LOOP1 LOOP2 LOOPE FUNC_DEF_TAIL
  FUNC_DEF_HEAD(v24) LOOP4 LOOP3 LOOP2 LOOP1 LOOPE FUNC_DEF_TAIL
FUNC_DEF_HEAD2(v100) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v101) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP3 LOOP5 LOOP4 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v102) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v103) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP4 LOOP5 LOOP3 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v104) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP5 LOOP3 LOOP4 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v105) LOOP1 LOOP2 LOOPE2 FUNC_DEF_MID2 LOOP5 LOOP4 LOOP3 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v106) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v107) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP3 LOOP5 LOOP4 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v108) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v109) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP4 LOOP5 LOOP3 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v110) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP5 LOOP3 LOOP4 FUNC_DEF_TAIL2
FUNC_DEF_HEAD2(v111) LOOP2 LOOP1 LOOPE2 FUNC_DEF_MID2 LOOP5 LOOP4 LOOP3 FUNC_DEF_TAIL2


struct MatmulImpl
{
  std::string name;
  INNER_MATMUL_FUNC impl;
};

// 1. TunableMatmul tmm(M, N, K, lda, ldb, ldc, nthreads);
// 2. tmm.tune(); or tmm.load_config(filename); # After tune, can call tmm.save_config(filename)
// 3. tmm.compute(A, B, C);
//
class TunableMatmul
{
public:
  TunableMatmul(){}

  TunableMatmul(int M, int N, int K, int lda, int ldb, int ldc, int nthreads = -1)
  {
    SetParams(false, false, M, N, K, lda, ldb, ldc, nthreads);
  }

  TunableMatmul(bool transa, bool transb, int M, int N, int K, int lda, int ldb, int ldc, int nthreads = -1)
  {
    SetParams(transa, transb, M, N, K, lda, ldb, ldc, nthreads);
  }

  ~TunableMatmul(){
    for(auto& item : name_2_handle_){
      HostOSTProxyManager::Instance().GetProxy(item.second)->Stop();
      HostOSTProxyManager::Instance().ReleaseProxy(item.second);
    }
  }

  void SetParams(bool transa, bool transb, int M, int N, int K, int lda, int ldb, int ldc, int nthreads = -1){
    SetParams(transa, transb, M, N, K, lda, ldb, ldc, nullptr, nullptr, nullptr, nthreads);
  }

  void SetParams(bool transa, bool transb, int M, int N, int K, int lda, int ldb, int ldc, const T *A, const T *B, T *C, int nthreads = -1){
    mmconfig.mmsize.ta = transa;
    mmconfig.mmsize.tb = transb;
    mmconfig.mmsize.m = M;
    mmconfig.mmsize.n = N;
    mmconfig.mmsize.k = K;
    mmconfig.mmsize.lda = lda;
    mmconfig.mmsize.ldb = ldb;
    mmconfig.mmsize.ldc = ldc;
    mmconfig.mmsize.bm = -1;
    mmconfig.mmsize.bn = -1;
    mmconfig.mmsize.bk = -1;
    mmconfig.A = A;
    mmconfig.B = B;
    mmconfig.C = C;

    // TODO: Set thread number
    if (nthreads != -1)
    {
    }
  }

  PROXY_HANDLE GetHandleByName(const std::string handle_name){
    auto iter = name_2_handle_.find(handle_name);
    if (iter != name_2_handle_.end()){
      return iter->second;
    }
    auto proxy_handle = HostOSTProxyManager::Instance().CreateNewProxy(handle_name.c_str());
    name_2_handle_[handle_name] = proxy_handle;
    return proxy_handle;
  }

  void SetThreadPool(tensorflow::thread::ThreadPool* thread_pool){
    // mmconfig.mmsize.thread_pool = thread_pool;
    // _cpu_device = thread_pool;
  }

  void SetThreadPool(const Eigen::ThreadPoolDevice* thread_pool){
    mmconfig.mmsize.thread_pool = thread_pool;
    // _cpu_device = thread_pool;
  }

  void SetConditions(int iter_per_cycle = 0, int max_iters = 0){
    max_cycle_  = max_iters;
    max_per_cycle_ = iter_per_cycle;
  }

  std::vector<TuningParam<int>>& GetTuningSpace(){
    if(space.size() == 0){
      MatmulSize mmsize = mmconfig.mmsize;
      // Set block m/n/k param;
      std::vector<int> bm_list;
      std::vector<int> bn_list;
      std::vector<int> bk_list;
      std::vector<int> bm_mg_pair;
      std::vector<int> bn_ng_pair;
      std::vector<int> bk_kg_pair;

      prepare_bm(mmsize.m, bm_list);
      prepare_bn(mmsize.n, bn_list);
      prepare_bk(mmsize.k, bk_list);
      prepare_block_group_pair(mmsize.m, bm_list, bm_mg_pair);
      prepare_block_group_pair(mmsize.n, bn_list, bn_ng_pair);
      prepare_block_group_pair(mmsize.k, bk_list, bk_kg_pair);

      space.push_back({
        B_G_M_NAME, std::make_pair(0, bm_mg_pair.size() / 2 - 1), bm_mg_pair, nullptr});
      space.push_back({
        B_G_N_NAME, std::make_pair(0, bn_ng_pair.size() / 2 - 1), bn_ng_pair, nullptr});
      space.push_back({
        B_G_K_NAME, std::make_pair(0, bk_kg_pair.size() / 2 - 1), bk_kg_pair, nullptr});

      // Set Matmul implement func;
      std::vector<int> impls;
      int idx = 0;
      while (impl_list[idx].impl != nullptr)
      {
        impls.push_back(idx);
        idx += 1;
      }

      space.push_back({
        IMPL_NAME, std::make_pair(0, impls.size() - 1),
        impls, nullptr});
    }

    return space;
  }

  void tune(bool flush_b)
  {
    MatmulSize mmsize = mmconfig.mmsize;

    // Allocate buffer and prepare data for A and B
    T *a = (T *)aligned_alloc(64, mmsize.m * mmsize.lda * sizeof(T));
    T *b = (T *)aligned_alloc(64, mmsize.k * mmsize.ldb * sizeof(T));
    T *c = (T *)aligned_alloc(64, mmsize.m * mmsize.ldc * sizeof(T));

    for (int i = 0; i < mmsize.m * mmsize.lda; ++i)
    {
      a[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < mmsize.k * mmsize.ldb; ++i)
    {
      b[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    tune(flush_b, a, b, c);

    free(c);
    free(b);
    free(a);
  }

  void tune(bool flush_b, const T * a, const T * b, T * c)
  {
    MatmulSize mmsize = mmconfig.mmsize;
    float best = std::numeric_limits<float>::max();
    auto bm_compare = [a, b, c, flush_b, &best, this](INNER_MATMUL_FUNC impl,
                                                      const MatmulSize &mmsize,
                                                      const SmallKernels &kernels)
    {
      // benchmark and record the best
      PerfStat stat = benchmark(impl, mmsize, kernels, a, b, c, flush_b);
      if (stat.avg_latency < best)
      {
        best = stat.avg_latency;
        mmconfig.mmsize = mmsize;
        mmconfig.kernels = kernels;
        mmconfig.impl = impl;
      }

      printf("\t%p: avg=%f, max=%f, min=%f. BEST=%f\n", impl,
             stat.avg_latency, stat.max_latency, stat.min_latency, best);
    };

    enumerate_do(bm_compare);
  }

  void host_tune(bool flush_b)
  {
    MatmulSize mmsize = mmconfig.mmsize;
    SmallKernels kernels;

    // Allocate buffer and prepare data for A and B
    T *a = (T *)aligned_alloc(64, mmsize.m * mmsize.lda * sizeof(T));
    T *b = (T *)aligned_alloc(64, mmsize.k * mmsize.ldb * sizeof(T));
    T *c = (T *)aligned_alloc(64, mmsize.m * mmsize.ldc * sizeof(T));

    for (int i = 0; i < mmsize.m * mmsize.lda; ++i)
    {
      a[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < mmsize.k * mmsize.ldb; ++i)
    {
      b[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    host_tune(flush_b, a, b, c);

    free(c);
    free(b);
    free(a);
  }

  void host_tune(bool flush_b, const T * a, const T * b, T * c)
  {
    std::lock_guard<std::mutex> lk(mMutex);

    ShowLog("start of   void host_tune(bool flush_b, const T * a, const T * b, T * c)");
    total_cycle_++;
    total_per_cycle_ = 0;

    MatmulSize& mmsize = mmconfig.mmsize;
    SmallKernels kernels;

    if(space.size() == 0){
      GetTuningSpace();
    }

    std::string handle_name = "host_test-" + std::to_string(mmsize.m)
                              + "-" + std::to_string(mmsize.n)
                              + "-" + std::to_string(mmsize.k)
			      + "-" + std::to_string(mmsize.ta) + std::to_string(mmsize.tb);
    auto proxy_handle = GetHandleByName(handle_name);
    auto my_host_proxy = HostOSTProxyManager::Instance().GetProxy(proxy_handle);

    auto state = my_host_proxy->GetProxyState();

    if(state == HostProxy::State::UNINITIALIZED){
      
      float best = std::numeric_limits<float>::max();

      auto cust_condition = [&my_host_proxy, this](TuningContext &context){
        this->total_per_cycle_++;

        if (this->max_per_cycle_ != 0 &&
            this->total_per_cycle_ > this->max_per_cycle_){
          return true;
        }

        //todo(marvin) put it to outside.
        // if (this->max_cycle_ != 0 && this->total_cycle_ > this->max_cycle_){
        //   std::cout<<"marvin test -- max_cycle_"<<std::endl;
        //   std::cout<<" total_cycle_ == " << this->total_cycle_ <<std::endl;
        //   std::cout<<" max_cycle_ == " << this->max_cycle_ <<std::endl;
        //   my_host_proxy->Stop();
        // }

        return false;
      };

      auto bm_compare = [flush_b, &best, this,
                        &mmsize, &kernels](std::string const &name, std::vector<int> const &params)
      {
        // Set params
        mmsize.bm = this->space[B_G_M].min_max_value[2 * params[B_G_M]];
        mmsize.bn = this->space[B_G_N].min_max_value[2 * params[B_G_N]];
        mmsize.bk = this->space[B_G_K].min_max_value[2 * params[B_G_K]];
        mmsize.mgroups = this->space[B_G_M].min_max_value[2 * params[B_G_M] + 1];
        mmsize.ngroups = this->space[B_G_N].min_max_value[2 * params[B_G_N] + 1];
        mmsize.kgroups = this->space[B_G_K].min_max_value[2 * params[B_G_K] + 1];

        int impl_id = this->space[IMPL].min_max_value[params[IMPL]];
        const MatmulImpl& matmulImpl = impl_list[impl_id];

        // std::cout << "mmsize.bm = " << mmsize.bm << std::endl;
        // std::cout << "mmsize.bn = " << mmsize.bn << std::endl;
        // std::cout << "mmsize.bk = " << mmsize.bk << std::endl;
        // std::cout << "mmsize.mgroups = " << mmsize.mgroups << std::endl;
        // std::cout << "mmsize.ngroups = " << mmsize.ngroups << std::endl;
        // std::cout << "mmsize.kgroups = " << mmsize.kgroups << std::endl;
        // std::cout << "impl_id = " << impl_id << std::endl;

        // Verify the parameters
#ifdef USE_LIBXSMM
	// libxsmm with noblas (mnk)^1/3 <= 64
	if (mmsize.bm * mmsize.bn * mmsize.bk > 64*64*64)
            return std::numeric_limits<float>::max();
#endif
        for(auto _item : this->space){
          if(_item.verify_func == nullptr) continue;

          if(!_item.verify_func(params)){
            return std::numeric_limits<float>::max();
          }
        }

        int mblocks = (mmsize.m + mmsize.bm - 1) / mmsize.bm;
        int nblocks = (mmsize.n + mmsize.bn - 1) / mmsize.bn;
        int kblocks = (mmsize.k + mmsize.bk - 1) / mmsize.bk;
        // Update blocks per group
        mmsize.mblocks_per_group = (mblocks + mmsize.mgroups - 1) / mmsize.mgroups;
        mmsize.nblocks_per_group = (nblocks + mmsize.ngroups - 1) / mmsize.ngroups;
        mmsize.kblocks_per_group = (kblocks + mmsize.kgroups - 1) / mmsize.kgroups;

        // printf("Try bm,bn,bk=%d,%d,%d; mgroups,ngroups,kgroups=%d,%d,%d; blocks_per_group=%d,%d,%d\n",
        //   mmsize.bm, mmsize.bn, mmsize.bk,
        //   mmsize.mgroups, mmsize.ngroups, mmsize.kgroups,
        //   mmsize.mblocks_per_group, mmsize.nblocks_per_group, mmsize.kblocks_per_group);

        // Update kernel according to block size
        update_kernels(kernels, mmsize.bm, mmsize.bn);

        // benchmark and record the best
        PerfStat stat = benchmark(matmulImpl.impl, mmsize, kernels,
                                  this->mmconfig.A, this->mmconfig.B, this->mmconfig.C, flush_b);
        // Update the config.
        if (stat.avg_latency < best)
        {
          best = stat.avg_latency;
          mmconfig.mmsize = mmsize;
          mmconfig.kernels = kernels;
          mmconfig.impl = matmulImpl.impl;
        }

        return stat.avg_latency;
      };

      char *algo = "GA";
      int gens = 50;
      int pops = 20;
      my_host_proxy->SetAlgorithm(algo, gens, pops);

      for(auto param : space){
        my_host_proxy->SetParamter(param.name.c_str(), param.min_max_index.first, param.min_max_index.second);
      }
      my_host_proxy->SetEvaluateFunc(std::move(bm_compare));
      my_host_proxy->SetConditionFunc(std::move(cust_condition));
      my_host_proxy->Start();
      do {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } while(my_host_proxy->GetProxyState() == HostProxy::State::RUNNING);
    } else if (state == HostProxy::State::RUNNING ||
               state == HostProxy::State::SUSPENDED){
      my_host_proxy->Start();
      do {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "    while in state == HostProxy::State::RUNNING|SUSPENDED" << std::endl;
      } while(my_host_proxy->GetProxyState() == HostProxy::State::RUNNING);
    } else if (state == HostProxy::State::STOPPED){
      // auto middle_res = my_host_proxy->GetTunedResult();
      ShowLog("} else if (state == HostProxy::State::STOPPED){");
      load_config(my_host_proxy->GetTunedResult());
      ShowLog("load_config(my_host_proxy->GetTunedResult());");
      compute(a, b, c);
      ShowLog("compute(a, b, c);");
    } else {

    }
    
    ShowLog("end of   void host_tune(bool flush_b, const T * a, const T * b, T * c)");
  }

  // Check if the implementation is correct or not
  void verify()
  {
    MatmulSize mmsize = mmconfig.mmsize;

    // Allocate buffer and prepare data for A and B
    T *a = (T *)aligned_alloc(64, mmsize.m * mmsize.lda * sizeof(T));
    T *b = (T *)aligned_alloc(64, mmsize.k * mmsize.ldb * sizeof(T));
    T *c = (T *)aligned_alloc(64, mmsize.m * mmsize.ldc * sizeof(T));
    T *ref_c = (T *)aligned_alloc(64, mmsize.m * mmsize.ldc * sizeof(T));

    for (int i = 0; i < mmsize.m * mmsize.lda; ++i)
    {
      a[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < mmsize.k * mmsize.ldb; ++i)
    {
      b[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    // Ref impl.
    for (int i = 0; i < mmsize.m; ++i)
    {
      for (int j = 0; j < mmsize.n; ++j)
      {
        T sum = 0;
        for (int k = 0; k < mmsize.k; ++k)
        {
          sum += a[i * mmsize.lda + k] * b[k * mmsize.ldb + j];
        }
        ref_c[i * mmsize.ldc + j] = sum;
      }
    }

    // Compute and compare with the reference result
    auto compute_cmp = [a, b, c, ref_c, this](INNER_MATMUL_FUNC impl,
                                              const MatmulSize &mmsize,
                                              const SmallKernels &kernels)
    {
      memset(c, 0, mmsize.m * mmsize.ldc * sizeof(T));
      impl(a, b, c, mmsize, kernels);
      if (!is_same(c, ref_c, mmsize.m, mmsize.n, mmsize.ldc))
      {
        printf("\t%p: NOT correct\n", impl);
        exit(-1);
      }
      else
      {
        printf("\t%p: correct\n", impl);
      }
    };

    enumerate_do(compute_cmp);

    free(ref_c);
    free(c);
    free(b);
    free(a);
  }

  void compute(const T *A, const T *B, T *C)
  {
    if (mmconfig.impl)
    {
      mmconfig.impl(A, B, C, mmconfig.mmsize, mmconfig.kernels);
    }
    else
    {
      printf("TunableMatmul: Cannot find an implementation.\n");
      exit(-1);
    }
  }

  // Saved tuned config to a file
  bool save_config(const char *filepath)
  {
    bool ret = false;

    FILE *fp = fopen(filepath, "a");
    if (fp)
    {
      // Save size info
      const MatmulSize &mmsize = mmconfig.mmsize;
      fprintf(fp, "mnk=%d,%d,%d; ldabc=%d,%d,%d; bmnk=%d,%d,%d; mnkgroups=%d,%d,%d; ",
              mmsize.m, mmsize.n, mmsize.k, mmsize.lda, mmsize.ldb, mmsize.ldc,
              mmsize.bm, mmsize.bn, mmsize.bk, mmsize.mgroups, mmsize.ngroups, mmsize.kgroups);

      // Save the impl. function
      int idx = 0;
      while (impl_list[idx].impl != nullptr)
      {
        if (impl_list[idx].impl == mmconfig.impl)
        {
          fprintf(fp, "impl=%s\n", impl_list[idx].name.c_str());
          ret = true;
        }
        idx += 1;
      }

      fclose(fp);
    }

    return ret;
  }

  bool load_config(const char *filepath)
  {
    bool ret = false;
    FILE *fp = fopen(filepath, "r");

    if (fp)
    {
      int m, n, k, lda, ldb, ldc;
      int bm, bn, bk, mgroups, ngroups, kgroups;
      char impl_name[16] = {0};
      MatmulSize &mmsize = mmconfig.mmsize;

      int read = -1;
      while (fscanf(fp, "mnk=%d,%d,%d; ldabc=%d,%d,%d; bmnk=%d,%d,%d; mnkgroups=%d,%d,%d; impl=%15s\n",
                    &m, &n, &k, &lda, &ldb, &ldc,
                    &bm, &bn, &bk, &mgroups, &ngroups, &kgroups, impl_name) > 0)
      {
        if (m == mmsize.m && n == mmsize.n && k == mmsize.k &&
            lda == mmsize.lda && ldb == mmsize.ldb && ldc == mmsize.ldc)
        {
          mmsize.bm = bm;
          mmsize.bn = bn;
          mmsize.bk = bk;

          mmsize.mgroups = mgroups;
          mmsize.ngroups = ngroups;
          mmsize.kgroups = kgroups;

          int mblocks = (mmsize.m + bm - 1) / bm;
          int nblocks = (mmsize.n + bn - 1) / bn;
          int kblocks = (mmsize.k + bk - 1) / bk;

          // Update blocks per group
          mmsize.mblocks_per_group = (mblocks + mmsize.mgroups - 1) / mmsize.mgroups;
          mmsize.nblocks_per_group = (nblocks + mmsize.ngroups - 1) / mmsize.ngroups;
          mmsize.kblocks_per_group = (kblocks + mmsize.kgroups - 1) / mmsize.kgroups;

          // Set the small kernels
          if (mmsize.bm > 0 && mmsize.bn > 0)
          {
            update_kernels(mmconfig.kernels, mmsize.bm, mmsize.bn);
          }

          // Set the impl. function
          if (impl_name[0] != '\0')
          {
            int idx = 0;
            while (impl_list[idx].impl != nullptr)
            {
              if (impl_list[idx].name == impl_name)
              {
                fprintf(fp, "impl=%s\n", impl_list[idx].name.c_str());
                mmconfig.impl = impl_list[idx].impl;
                ret = true;
                break;
              }
              idx += 1;
            }
          }

          break;
        }
      }

      fclose(fp);
    }

    return ret;
  }

  bool load_config(std::map<std::string, int> rst)
  {
    MatmulSize &mmsize = mmconfig.mmsize;

    if(space.size() == 0){
      GetTuningSpace();
    }

    mmsize.bm = this->space[B_G_M].min_max_value[2 * rst[B_G_M_NAME]];
    mmsize.bn = this->space[B_G_N].min_max_value[2 * rst[B_G_N_NAME]];
    mmsize.bk = this->space[B_G_K].min_max_value[2 * rst[B_G_K_NAME]];
    mmsize.mgroups = this->space[B_G_M].min_max_value[2 * rst[B_G_M_NAME] + 1];
    mmsize.ngroups = this->space[B_G_N].min_max_value[2 * rst[B_G_N_NAME] + 1];
    mmsize.kgroups = this->space[B_G_K].min_max_value[2 * rst[B_G_K_NAME] + 1];

    int impl_id = this->space[IMPL].min_max_value[rst[IMPL_NAME]];
    const MatmulImpl& matmulImpl = impl_list[impl_id];

    int mblocks = (mmsize.m + mmsize.bm - 1) / mmsize.bm;
    int nblocks = (mmsize.n + mmsize.bn - 1) / mmsize.bn;
    int kblocks = (mmsize.k + mmsize.bk - 1) / mmsize.bk;

    // Update blocks per group
    mmsize.mblocks_per_group = (mblocks + mmsize.mgroups - 1) / mmsize.mgroups;
    mmsize.nblocks_per_group = (nblocks + mmsize.ngroups - 1) / mmsize.ngroups;
    mmsize.kblocks_per_group = (kblocks + mmsize.kgroups - 1) / mmsize.kgroups;

    // Set the small kernels
    if (mmsize.bm > 0 && mmsize.bn > 0)
    {
      update_kernels(mmconfig.kernels, mmsize.bm, mmsize.bn);
    }

    // Set the impl. function
    mmconfig.impl = matmulImpl.impl;

    ShowLog(" mmsize.bm=" + to_string(mmsize.bm)
          + " mmsize.bn=" + to_string(mmsize.bn)
          + " mmsize.bk=" + to_string(mmsize.bk)
          + " mmsize.mgroups=" + to_string(mmsize.mgroups)
          + " mmsize.ngroups=" + to_string(mmsize.ngroups)
          + " mmsize.kgroups=" + to_string(mmsize.kgroups)
          + " mmsize.mblocks=" + to_string(mblocks)
          + " mmsize.nblocks=" + to_string(nblocks)
          + " mmsize.kblocks=" + to_string(kblocks)
          + " mmsize.impl_id=" + to_string(impl_id));

    return true;
  }

  static void flush_cache(const T *buf, size_t size)
  {
// //#pragma omp parallel for
    for (size_t offset = 0; offset < size; offset += CACHELINE_SIZE / sizeof(T))
    {
      _mm_clflush(buf + offset);
    }
  }

private:
  void prepare_block_group_pair(int n, std::vector<int> &b_list, std::vector<int> &bg_pair) {
      for (auto block_item : b_list) {
          int nblocks = (n + block_item - 1) / block_item;
          int max_ngroups = nblocks < MAX_GROUP_LIMIT ? nblocks : MAX_GROUP_LIMIT;
          for (int i = 1; i <= max_ngroups; i++) {
              bg_pair.push_back(block_item);
              bg_pair.push_back(i);
          }
      }
  }

  void prepare_bm(int m, std::vector<int> &bm_list)
  {
    if (m < 32)
    {
      bm_list.push_back(m);
    }
    else if (m < 64)
    {
      bm_list.push_back(m);
      bm_list.push_back((m + 1) / 2);
    }
    else
    {
      bm_list.push_back(64);
      bm_list.push_back(48);
      bm_list.push_back(32);
    }
  }

  void prepare_bn(int n, std::vector<int> &bn_list)
  {
    prepare_bm(n, bn_list);
  }

  void prepare_bk(int k, std::vector<int> &bk_list)
  {
    // bk = 64, ...
    //int candidates[] = { 64, 96, 128, 160, 192, 224, 256, 384, 512 };
    int candidates[] = {64, 128, 256, 512};
    for (int i = 0; i < sizeof(candidates) / sizeof(int); ++i)
    {
      if (candidates[i] <= k)
      {
        bk_list.push_back(candidates[i]);
      }
      else
      {
        break;
      }
    }

    // bk = k, k/2, k/3, ...
    int divider = 1;
    do
    {
      int bk = (k + divider - 1) / divider;
      // do not try small values
      if (bk < 128)
      {
        break;
      }
      if (std::find(bk_list.begin(), bk_list.end(), bk) == bk_list.end())
      {
        bk_list.push_back(bk);
      }
      divider += 1;
    } while (true);

    // In case of small k
    if (bk_list.empty())
    {
      bk_list.push_back(k);
    }
  }

  // Get the split according to position
  bool get_split(std::vector<int> &bm_list, std::vector<int> &bn_list,
                 std::vector<int> &bk_list, int &bm, int &bn, int &bk, int position)
  {
    int size1 = bm_list.size();
    int size2 = bn_list.size();
    int size3 = bk_list.size();

    int idx3 = position % size3;
    int idx1_2 = position / size3;
    int idx2 = idx1_2 % size2;
    int idx1 = idx1_2 / size2;

    // The split is out of range
    if (idx1 >= size1)
    {
      return false;
    }

    bm = bm_list[idx1];
    bn = bn_list[idx2];
    bk = bk_list[idx3];

    return true;
  }

  bool get_next_partition(MatmulSize &mmsize, int bm, int bn, int bk)
  {
    int mblocks = (mmsize.m + bm - 1) / bm;
    int nblocks = (mmsize.n + bn - 1) / bn;
    int kblocks = (mmsize.k + bk - 1) / bk;

    // Previous has the same split, then try next partition/group
    if (mmsize.bm == bm && mmsize.bn == bn && mmsize.bk == bk)
    {
      int max_mgroups = mblocks < MAX_GROUP_LIMIT ? mblocks : MAX_GROUP_LIMIT;
      int max_ngroups = nblocks < MAX_GROUP_LIMIT ? nblocks : MAX_GROUP_LIMIT;
      int max_kgroups = kblocks < MAX_GROUP_LIMIT ? kblocks : MAX_GROUP_LIMIT;

      mmsize.kgroups += 1;
      if (mmsize.kgroups > max_kgroups)
      {
        mmsize.kgroups = 1;
        mmsize.ngroups += 1;
        if (mmsize.ngroups > max_ngroups)
        {
          mmsize.ngroups = 1;
          mmsize.mgroups += 1;
          if (mmsize.mgroups > max_mgroups)
          { // All partitions already enumerated
            mmsize.mgroups = 1;
            return false;
          }
        }
      }
    }
    else
    {
      mmsize.bm = bm;
      mmsize.bn = bn;
      mmsize.bk = bk;

      // A new split, use the first partition
      mmsize.mgroups = 1;
      mmsize.ngroups = 1;
      mmsize.kgroups = 1;
    }

    // Update blocks per group
    mmsize.mblocks_per_group = (mblocks + mmsize.mgroups - 1) / mmsize.mgroups;
    mmsize.nblocks_per_group = (nblocks + mmsize.ngroups - 1) / mmsize.ngroups;
    mmsize.kblocks_per_group = (kblocks + mmsize.kgroups - 1) / mmsize.kgroups;

    return true;
  }

  // Enumerate all the possible impl. and do something
  template <typename Lambda>
  void enumerate_do(const Lambda &do_func)
  {
    MatmulSize mmsize = mmconfig.mmsize;

    // split list
    std::vector<int> bm_list;
    std::vector<int> bn_list;
    std::vector<int> bk_list;
    prepare_bm(mmsize.m, bm_list);
    prepare_bn(mmsize.n, bn_list);
    prepare_bk(mmsize.k, bk_list);

    // Enumerate all splits
    int position = 0;
    int bm, bn, bk;
    float best = std::numeric_limits<float>::max();
    while (get_split(bm_list, bn_list, bk_list, bm, bn, bk, position++))
    {

      // Enumerate all partitions
      while (get_next_partition(mmsize, bm, bn, bk))
      {
        printf("Try bm,bn,bk=%d,%d,%d; mgroups,ngroups,kgroups=%d,%d,%d; blocks_per_group=%d,%d,%d\n",
               mmsize.bm, mmsize.bn, mmsize.bk,
               mmsize.mgroups, mmsize.ngroups, mmsize.kgroups,
               mmsize.mblocks_per_group, mmsize.nblocks_per_group, mmsize.kblocks_per_group);

        // Update kernel according to block size
        SmallKernels kernels;
        update_kernels(kernels, mmsize.bm, mmsize.bn);

        // Enumerate each impl.
        int idx = 0;
        while (impl_list[idx].impl != nullptr)
        {
          do_func(impl_list[idx].impl, mmsize, kernels);
          idx += 1;
        }
      }
    }
  }

  void update_kernels(SmallKernels &kernels, int bm, int bn)
  {
#ifndef USE_LIBXSMM
    if (bm == 32)
    {
        SET_KERNELS_ENUM_BN(32)
    }
    else if (bm == 48)
    {
        SET_KERNELS_ENUM_BN(48)
    }
    else if (bm == 64)
    {
        SET_KERNELS_ENUM_BN(64)
    }
    else if (bm == 80)
    {
        SET_KERNELS_ENUM_BN(80)
    }
    else if (bm < 32)
    {
        switch (bm)
        {
        case 1:
            SET_KERNELS_ENUM_BN(1);
            break;
        case 2:
            SET_KERNELS_ENUM_BN(2);
            break;
        case 3:
            SET_KERNELS_ENUM_BN(3);
            break;
        case 4:
            SET_KERNELS_ENUM_BN(4);
            break;
        case 5:
            SET_KERNELS_ENUM_BN(5);
            break;
        case 6:
            SET_KERNELS_ENUM_BN(6);
            break;
        case 7:
            SET_KERNELS_ENUM_BN(7);
            break;
        case 8:
            SET_KERNELS_ENUM_BN(8);
            break;
        case 9:
            SET_KERNELS_ENUM_BN(9);
            break;
        case 10:
            SET_KERNELS_ENUM_BN(10);
            break;
        case 11:
            SET_KERNELS_ENUM_BN(11);
            break;
        case 12:
            SET_KERNELS_ENUM_BN(12);
            break;
        case 13:
            SET_KERNELS_ENUM_BN(13);
            break;
        case 14:
            SET_KERNELS_ENUM_BN(14);
            break;
        case 15:
            SET_KERNELS_ENUM_BN(15);
            break;
        case 16:
            SET_KERNELS_ENUM_BN(16);
            break;
        case 17:
            SET_KERNELS_ENUM_BN(17);
            break;
        case 18:
            SET_KERNELS_ENUM_BN(18);
            break;
        case 19:
            SET_KERNELS_ENUM_BN(19);
            break;
        case 20:
            SET_KERNELS_ENUM_BN(20);
            break;
        case 21:
            SET_KERNELS_ENUM_BN(21);
            break;
        case 22:
            SET_KERNELS_ENUM_BN(22);
            break;
        case 23:
            SET_KERNELS_ENUM_BN(23);
            break;
        case 24:
            SET_KERNELS_ENUM_BN(24);
            break;
        case 25:
            SET_KERNELS_ENUM_BN(25);
            break;
        case 26:
            SET_KERNELS_ENUM_BN(26);
            break;
        case 27:
            SET_KERNELS_ENUM_BN(27);
            break;
        case 28:
            SET_KERNELS_ENUM_BN(28);
            break;
        case 29:
            SET_KERNELS_ENUM_BN(29);
            break;
        case 30:
            SET_KERNELS_ENUM_BN(30);
            break;
        case 31:
            SET_KERNELS_ENUM_BN(31);
            break;
        }
    }
    else
    {
        printf("Unsupported kernel for bm=%d\n", bm);
        exit(-1);
    }

    kernels.kernel_nofix_acc = small_gemm_nofix<true>;
    kernels.kernel_nofix_nonacc = small_gemm_nofix<false>;
#endif
  }

  PerfStat benchmark(INNER_MATMUL_FUNC func, const MatmulSize &mmsize,
                     const SmallKernels &kernels,
                     const T *A, const T *B, T *C,
                     bool flush_b)
  {
    const int warmup_loops = 0;
    const int benchmark_loops = 1;

    PerfStat perfStat;
    std::vector<float> latencies;
    latencies.reserve(benchmark_loops);

    // Warmup and benchmark
    for (int i = 0; i < warmup_loops + benchmark_loops; ++i)
    {
      Timer t;
      func(A, B, C, mmsize, kernels);
      if (i >= warmup_loops)
      {
        latencies.push_back(t.getTime());
      }
      if (flush_b) {
        flush_cache(B, mmsize.k * mmsize.ldb);
      }
    }

    // Stat the perf data
    perfStat.avg_latency = 0;
    perfStat.max_latency = 0;
    perfStat.min_latency = std::numeric_limits<float>::max();
    for (float latency : latencies)
    {
      if (latency > perfStat.max_latency)
        perfStat.max_latency = latency;
      if (latency < perfStat.min_latency)
        perfStat.min_latency = latency;
      perfStat.avg_latency += latency;
    }
    perfStat.avg_latency /= latencies.size();
    perfStat.samples = latencies.size();

    return perfStat;
  }

  static bool is_same(const T *data1, const T *data2, int rows, int cols, int stride)
  {
    bool is_same = true;

//#pragma omp parallel for
    for (int i = 0; i < rows; ++i)
    {
      int offset = i * stride;
      for (int j = 0; j < cols; ++j)
      {
        if (fabs(data1[offset] - data2[offset]) > 0.0001)
        {
          printf("[%d, %d] is different: %f vs. %f\n", i, j, data1[offset], data2[offset]);
          is_same = false;
        }
        offset += 1;
      }
    }

    return is_same;
  }

private:
  MatmulConfig mmconfig;
  std::map<string, PROXY_HANDLE> name_2_handle_;
  std::map<std::string, int> middle_res;
  std::vector<TuningParam<int>> space;
  int total_per_cycle_ = 0;
  int max_per_cycle_ = 0;

  int total_cycle_ = 0;
  int max_cycle_ = 0;
  static const MatmulImpl impl_list[];
  tensorflow::thread::ThreadPool* _cpu_device;
};

#endif

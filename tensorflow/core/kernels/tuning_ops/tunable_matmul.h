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

#define CACHELINE_SIZE 64
#define MAX_GROUP_LIMIT 8

typedef float T;

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
};

// Inner implementation of matmul
typedef void (*INNER_MATMUL_FUNC)(const T *A, const T *B, T *C,
                                  const MatmulSize &mmsize, const SmallKernels &kernels);

struct MatmulConfig
{
  INNER_MATMUL_FUNC impl;

  SmallKernels kernels;

  MatmulSize mmsize;
};

#define LOOP0 for (int kg = 0; kg < mmsize.kgroups; ++kg)
#define LOOP1 for (int ig = 0; ig < mmsize.mgroups; ++ig)
#define LOOP2 for (int jg = 0; jg < mmsize.ngroups; ++jg)
#define LOOP3 for (int i = 0; i < mmsize.mblocks_per_group; ++i)
#define LOOP4 for (int j = 0; j < mmsize.nblocks_per_group; ++j)
#define LOOP5 for (int k = 0; k < mmsize.kblocks_per_group; ++k)

// Equals to: #pragma omp parallel for collapse(4)
#define PARALLEL_C4 _Pragma("omp parallel for collapse(4)")
#define PARALLEL_C2 _Pragma("omp parallel for collapse(2)")

#define FUNC_DEF_HEAD(name)                                                                             \
  static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) \
  {                                                                                                     \
    for (int kg = 0; kg < mmsize.kgroups; ++kg)                                                         \
    {                                                                                                   \
      PARALLEL_C4

#define FUNC_DEF_HEAD2(name)                                                                            \
  static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) \
  {                                                                                                     \
    for (int kg = 0; kg < mmsize.kgroups; ++kg)                                                         \
    {                                                                                                   \
      PARALLEL_C2

#define FUNC_DEF_TAIL                                                                                            \
  {                                                                                                              \
    int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;                                                 \
    int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;                                                 \
    int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;                                                 \
    if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k)                                                \
    {                                                                                                            \
      int realbm = mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);                               \
      int realbn = mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);                               \
      int realbk = mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);                               \
      const T *pa = &A[i_off * mmsize.lda + k_off];                                                              \
      const T *pb = &B[k_off * mmsize.ldb + j_off];                                                              \
      T *pc = &C[i_off * mmsize.ldc + j_off];                                                                    \
      if (realbm == mmsize.bm)                                                                                   \
      {                                                                                                          \
        if (realbn == mmsize.bn)                                                                                 \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixmn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);                    \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixmn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);                 \
          }                                                                                                      \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixm_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);             \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixm_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);          \
          }                                                                                                      \
        }                                                                                                        \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        if (realbn == mmsize.bn)                                                                                 \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_fixn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);             \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_fixn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);          \
          }                                                                                                      \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          if (k_off != 0)                                                                                        \
          {                                                                                                      \
            kernels.kernel_nofix_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn, realbk);    \
          }                                                                                                      \
          else                                                                                                   \
          {                                                                                                      \
            kernels.kernel_nofix_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn, realbk); \
          }                                                                                                      \
        }                                                                                                        \
      }                                                                                                          \
    }                                                                                                            \
  }                                                                                                              \
  }                                                                                                              \
  }

static void v1(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels)
{
  for (int kg = 0; kg < mmsize.kgroups; ++kg)
  {
#pragma omp parallel for collapse(4)
    for (int ig = 0; ig < mmsize.mgroups; ++ig)
      for (int jg = 0; jg < mmsize.ngroups; ++jg)
        for (int i = 0; i < mmsize.mblocks_per_group; ++i)
          for (int j = 0; j < mmsize.nblocks_per_group; ++j)
          {
            for (int k = 0; k < mmsize.kblocks_per_group; ++k)
            {
              int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;
              int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;
              int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;
              if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k)
              {
                int realbm = mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);
                int realbn = mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);
                int realbk = mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);
                const T *pa = &A[i_off * mmsize.lda + k_off];
                const T *pb = &B[k_off * mmsize.ldb + j_off];
                T *pc = &C[i_off * mmsize.ldc + j_off];
                //printf("\trealbm,realbn,realbk=%d,%d,%d, ijk_off=%d,%d,%d\n", realbm, realbn, realbk, i_off, j_off, k_off);
                if (realbm == mmsize.bm)
                {
                  if (realbn == mmsize.bn)
                  {
                    if (k_off != 0)
                    {
                      kernels.kernel_fixmn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);
                    }
                    else
                    {
                      kernels.kernel_fixmn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);
                    }
                  }
                  else
                  {
                    if (k_off != 0)
                    {
                      kernels.kernel_fixm_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);
                    }
                    else
                    {
                      kernels.kernel_fixm_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);
                    }
                  }
                }
                else
                {
                  if (realbn == mmsize.bn)
                  {
                    if (k_off != 0)
                    {
                      kernels.kernel_fixn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);
                    }
                    else
                    {
                      kernels.kernel_fixn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);
                    }
                  }
                  else
                  {
                    if (k_off != 0)
                    {
                      kernels.kernel_nofix_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn, realbk);
                    }
                    else
                    {
                      kernels.kernel_nofix_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn, realbk);
                    }
                  }
                }
              }
            }
          }
  }
}

FUNC_DEF_HEAD(v1_invalid)
LOOP1 LOOP2 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v2) LOOP1 LOOP2 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v3) LOOP1 LOOP3 LOOP2 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v4) LOOP1 LOOP3 LOOP4 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v5) LOOP1 LOOP4 LOOP2 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v6) LOOP1 LOOP4 LOOP3 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v7) LOOP2 LOOP1 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v8) LOOP2 LOOP1 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v9) LOOP2 LOOP3 LOOP1 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v10) LOOP2 LOOP3 LOOP4 LOOP1 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v11) LOOP2 LOOP4 LOOP1 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v12) LOOP2 LOOP4 LOOP3 LOOP1 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v13) LOOP3 LOOP1 LOOP2 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v14) LOOP3 LOOP1 LOOP4 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v15) LOOP3 LOOP2 LOOP1 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v16) LOOP3 LOOP2 LOOP4 LOOP1 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v17) LOOP3 LOOP4 LOOP1 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v18) LOOP3 LOOP4 LOOP2 LOOP1 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v19) LOOP4 LOOP1 LOOP2 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v20) LOOP4 LOOP1 LOOP3 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v21) LOOP4 LOOP2 LOOP1 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v22) LOOP4 LOOP2 LOOP3 LOOP1 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v23) LOOP4 LOOP3 LOOP1 LOOP2 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD(v24) LOOP4 LOOP3 LOOP2 LOOP1 LOOP5 FUNC_DEF_TAIL

    FUNC_DEF_HEAD2(v100) LOOP1 LOOP2 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v101) LOOP1 LOOP2 LOOP3 LOOP5 LOOP4 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v102) LOOP1 LOOP2 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v103) LOOP1 LOOP2 LOOP4 LOOP5 LOOP3 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v104) LOOP1 LOOP2 LOOP5 LOOP3 LOOP4 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v105) LOOP1 LOOP2 LOOP5 LOOP4 LOOP3 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v106) LOOP2 LOOP1 LOOP3 LOOP4 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v107) LOOP2 LOOP1 LOOP3 LOOP5 LOOP4 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v108) LOOP2 LOOP1 LOOP4 LOOP3 LOOP5 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v109) LOOP2 LOOP1 LOOP4 LOOP5 LOOP3 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v110) LOOP2 LOOP1 LOOP5 LOOP3 LOOP4 FUNC_DEF_TAIL
    FUNC_DEF_HEAD2(v111) LOOP2 LOOP1 LOOP5 LOOP4 LOOP3 FUNC_DEF_TAIL

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
    SetParams(M, N, K, lda, ldb, ldc, nthreads);
  }

  void SetParams(int M, int N, int K, int lda, int ldb, int ldc, int nthreads = -1){
    mmconfig.mmsize.m = M;
    mmconfig.mmsize.n = N;
    mmconfig.mmsize.k = K;
    mmconfig.mmsize.lda = lda;
    mmconfig.mmsize.ldb = ldb;
    mmconfig.mmsize.ldc = ldc;
    mmconfig.mmsize.bm = -1;
    mmconfig.mmsize.bn = -1;
    mmconfig.mmsize.bk = -1;

    // TODO: Set thread number
    if (nthreads != -1)
    {
    }
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

    free(c);
    free(b);
    free(a);
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

    float best = std::numeric_limits<float>::max();

    auto bm_compare = [a, b, c, flush_b, &best, this,
                       &mmsize, &kernels](std::vector<int> const &params)
    {
      std::cout << "cur iteration's m/n/kgroups:";
      for (auto param : params) {
          std::cout << param << " ";
      }
      std::cout << " " << std::endl;
      // set params
      int mblocks = (mmsize.m + mmsize.bm - 1) / mmsize.bm;
      int nblocks = (mmsize.n + mmsize.bn - 1) / mmsize.bn;
      int kblocks = (mmsize.k + mmsize.bk - 1) / mmsize.bk;
      mmsize.mgroups = params[0];
      mmsize.ngroups = params[1];
      mmsize.kgroups = params[2];
      const MatmulImpl& matmulImpl = impl_list[3];
      const INNER_MATMUL_FUNC& impl = matmulImpl.impl;

      // Update blocks per group
      mmsize.mblocks_per_group = (mblocks + mmsize.mgroups - 1) / mmsize.mgroups;
      mmsize.nblocks_per_group = (nblocks + mmsize.ngroups - 1) / mmsize.ngroups;
      mmsize.kblocks_per_group = (kblocks + mmsize.kgroups - 1) / mmsize.kgroups;

      printf("Try bm,bn,bk=%d,%d,%d; mgroups,ngroups,kgroups=%d,%d,%d; blocks_per_group=%d,%d,%d\n",
        mmsize.bm, mmsize.bn, mmsize.bk,
        mmsize.mgroups, mmsize.ngroups, mmsize.kgroups,
        mmsize.mblocks_per_group, mmsize.nblocks_per_group, mmsize.kblocks_per_group);

      // Update kernel according to block size
      update_kernels(kernels, mmsize.bm, mmsize.bn);

      // benchmark and record the best
      
      // Enumerate each impl.
      int idx = 0;
      float best_per_cycle = std::numeric_limits<float>::max();
      while (impl_list[idx].impl != nullptr)
      {
        PerfStat stat = benchmark(impl_list[idx].impl, mmsize, kernels, a, b, c, flush_b);
        if (stat.avg_latency < best_per_cycle){
          best_per_cycle = stat.avg_latency;
        }
        if (stat.avg_latency < best)
        {
          best = stat.avg_latency;
          mmconfig.mmsize = mmsize;
          mmconfig.kernels = kernels;
          mmconfig.impl = impl_list[idx].impl;
        }
        idx += 1;
      }

      // printf("\t%s: avg=%f, max=%f, min=%f. BEST=%f\n", matmulImpl.name.c_str(),
      //        stat.avg_latency, stat.max_latency, stat.min_latency, best);
      return best_per_cycle;
    };

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
    while (get_split(bm_list, bn_list, bk_list, bm, bn, bk, position++))
    {
      std::string handle_name = "host_test" + std::to_string(position);
      auto proxy_handle = HostOSTProxyManager::Instance().CreateNewProxy(handle_name.c_str());
      auto my_host_proxy = HostOSTProxyManager::Instance().GetProxy(proxy_handle);

      mmsize.bm = bm;
      mmsize.bn = bn;
      mmsize.bk = bk;
      int mblocks = (mmsize.m + mmsize.bm - 1) / mmsize.bm;
      int nblocks = (mmsize.n + mmsize.bn - 1) / mmsize.bn;
      int kblocks = (mmsize.k + mmsize.bk - 1) / mmsize.bk;
      int impls = 36;

      int max_mgroups = mblocks < MAX_GROUP_LIMIT ? mblocks : MAX_GROUP_LIMIT;
      int max_ngroups = nblocks < MAX_GROUP_LIMIT ? nblocks : MAX_GROUP_LIMIT;
      int max_kgroups = kblocks < MAX_GROUP_LIMIT ? kblocks : MAX_GROUP_LIMIT;

      my_host_proxy->SetParamter("mgroups", 1, max_mgroups);
      my_host_proxy->SetParamter("ngroups", 1, max_ngroups);
      my_host_proxy->SetParamter("kgroups", 1, max_kgroups);
      // fixme(marvin): 不能设置超过三个Paramters?
      // my_host_proxy->SetParamter("impls", 1, impls);
      my_host_proxy->SetEvaluateFunc(bm_compare);
      my_host_proxy->Start();

      HostOSTProxyManager::Instance().ReleaseProxy(proxy_handle);
    }

    free(c);
    free(b);
    free(a);
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

  static void flush_cache(const T *buf, size_t size)
  {
#pragma omp parallel for
    for (size_t offset = 0; offset < size; offset += CACHELINE_SIZE / sizeof(T))
    {
      _mm_clflush(buf + offset);
    }
  }

private:
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

  void update_kernels(SmallKernels &kernels, int bm, int bn);

  PerfStat benchmark(INNER_MATMUL_FUNC func, const MatmulSize &mmsize,
                     const SmallKernels &kernels,
                     const T *A, const T *B, T *C,
                     bool flush_b)
  {
    const int warmup_loops = 1;
    const int benchmark_loops = 5;

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

#pragma omp parallel for
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

  static const MatmulImpl impl_list[];
};

#endif
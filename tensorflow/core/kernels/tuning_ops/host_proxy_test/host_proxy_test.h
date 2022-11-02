#include "sgemm_kernel.h"
#include "timer.h"
#include <vector>
#include <algorithm>

#define MAX_GROUP_LIMIT 8
#define CACHELINE_SIZE 64
typedef float T;

struct MatmulSize {
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

typedef void (*KERNEL_FIXMN)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K);
typedef void (*KERNEL_FIXN)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K);
typedef void (*KERNEL_FIXM)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K);
typedef void (*KERNEL_NOFIX)(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K);

struct SmallKernels {
    KERNEL_FIXMN kernel_fixmn_acc;
    KERNEL_FIXMN kernel_fixmn_nonacc;
    KERNEL_FIXM kernel_fixm_acc;
    KERNEL_FIXM kernel_fixm_nonacc;
    KERNEL_FIXN kernel_fixn_acc;
    KERNEL_FIXN kernel_fixn_nonacc;
    KERNEL_NOFIX kernel_nofix_acc;
    KERNEL_NOFIX kernel_nofix_nonacc;
};

struct PerfStat {
    float avg_latency;
    float min_latency;
    float max_latency;
    float variance;
    int samples;
};

typedef void (*INNER_MATMUL_FUNC)(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels);

#define SET_KERNELS(bm, bn)                                                                                            \
    kernels.kernel_fixmn_acc = small_gemm_fixmn<(bm), (bn), true>;                                                     \
    kernels.kernel_fixmn_nonacc = small_gemm_fixmn<(bm), (bn), false>;                                                 \
    kernels.kernel_fixm_acc = small_gemm_fixm<(bm), true>;                                                             \
    kernels.kernel_fixm_nonacc = small_gemm_fixm<(bm), false>;                                                         \
    kernels.kernel_fixn_acc = small_gemm_fixn<(bn), true>;                                                             \
    kernels.kernel_fixn_nonacc = small_gemm_fixn<(bn), false>;

#define SET_KERNELS_ENUM_BN(bm)                                                                                        \
    if (bn == 32) {                                                                                                    \
        SET_KERNELS((bm), 32);                                                                                         \
    } else if (bn == 48) {                                                                                             \
        SET_KERNELS((bm), 48);                                                                                         \
    } else if (bn == 64) {                                                                                             \
        SET_KERNELS((bm), 64);                                                                                         \
    } else if (bn == 80) {                                                                                             \
        SET_KERNELS((bm), 80);                                                                                         \
    } else {                                                                                                           \
        printf("Unsupported kernel for bn=%d\n", bn);                                                                  \
        exit(-1);                                                                                                      \
    }

#define LOOP0 for (int kg = 0; kg < mmsize.kgroups; ++kg)
#define LOOP1 for (int ig = 0; ig < mmsize.mgroups; ++ig)
#define LOOP2 for (int jg = 0; jg < mmsize.ngroups; ++jg)
#define LOOP3 for (int i = 0; i < mmsize.mblocks_per_group; ++i)
#define LOOP4 for (int j = 0; j < mmsize.nblocks_per_group; ++j)
#define LOOP5 for (int k = 0; k < mmsize.kblocks_per_group; ++k)

// Equals to: #pragma omp parallel for collapse(4)
#define PARALLEL_C4 _Pragma("omp parallel for collapse(4)")
#define PARALLEL_C2 _Pragma("omp parallel for collapse(2)")

#define FUNC_DEF_HEAD(name)                                                                                            \
    static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) {            \
        for (int kg = 0; kg < mmsize.kgroups; ++kg) {                                                                  \
            PARALLEL_C4

#define FUNC_DEF_HEAD2(name)                                                                                           \
    static void name(const T *A, const T *B, T *C, const MatmulSize &mmsize, const SmallKernels &kernels) {            \
        for (int kg = 0; kg < mmsize.kgroups; ++kg) {                                                                  \
            PARALLEL_C2

#define FUNC_DEF_TAIL                                                                                                  \
    {                                                                                                                  \
        int i_off = (ig * mmsize.mblocks_per_group + i) * mmsize.bm;                                                   \
        int j_off = (jg * mmsize.nblocks_per_group + j) * mmsize.bn;                                                   \
        int k_off = (kg * mmsize.kblocks_per_group + k) * mmsize.bk;                                                   \
        if (i_off < mmsize.m && j_off < mmsize.n && k_off < mmsize.k) {                                                \
            int realbm = mmsize.m - i_off >= mmsize.bm ? mmsize.bm : (mmsize.m - i_off);                               \
            int realbn = mmsize.n - j_off >= mmsize.bn ? mmsize.bn : (mmsize.n - j_off);                               \
            int realbk = mmsize.k - k_off >= mmsize.bk ? mmsize.bk : (mmsize.k - k_off);                               \
            const T *pa = &A[i_off * mmsize.lda + k_off];                                                              \
            const T *pb = &B[k_off * mmsize.ldb + j_off];                                                              \
            T *pc = &C[i_off * mmsize.ldc + j_off];                                                                    \
            if (realbm == mmsize.bm) {                                                                                 \
                if (realbn == mmsize.bn) {                                                                             \
                    if (k_off != 0) {                                                                                  \
                        kernels.kernel_fixmn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);              \
                    } else {                                                                                           \
                        kernels.kernel_fixmn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbk);           \
                    }                                                                                                  \
                } else {                                                                                               \
                    if (k_off != 0) {                                                                                  \
                        kernels.kernel_fixm_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);       \
                    } else {                                                                                           \
                        kernels.kernel_fixm_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbn, realbk);    \
                    }                                                                                                  \
                }                                                                                                      \
            } else {                                                                                                   \
                if (realbn == mmsize.bn) {                                                                             \
                    if (k_off != 0) {                                                                                  \
                        kernels.kernel_fixn_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);       \
                    } else {                                                                                           \
                        kernels.kernel_fixn_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbk);    \
                    }                                                                                                  \
                } else {                                                                                               \
                    if (k_off != 0) {                                                                                  \
                        kernels.kernel_nofix_acc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn,       \
                                                 realbk);                                                              \
                    } else {                                                                                           \
                        kernels.kernel_nofix_nonacc(pa, pb, pc, mmsize.lda, mmsize.ldb, mmsize.ldc, realbm, realbn,    \
                                                    realbk);                                                           \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    }                                                                                                                  \
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
struct MatmulImpl
{
  std::string name;
  INNER_MATMUL_FUNC impl;
};

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

const MatmulImpl impl_list[37] = {
        {"v1", v1},
        {"v2", v2},
        {"v3", v3},
        {"v4", v4},
        {"v5", v5},
        {"v6", v6},
        {"v7", v7},
        {"v8", v8},
        {"v9", v9},
        {"v10", v10},
        {"v11", v11},
        {"v12", v12},
        {"v13", v13},
        {"v14", v14},
        {"v15", v15},
        {"v16", v16},
        {"v17", v17},
        {"v18", v18},
        {"v19", v19},
        {"v20", v20},
        {"v21", v21},
        {"v22", v22},
        {"v23", v23},
        {"v24", v24},
        {"v100", v100},
        {"v101", v101},
        {"v102", v102},
        {"v103", v103},
        {"v104", v104},
        {"v105", v105},
        {"v106", v106},
        {"v107", v107},
        {"v108", v108},
        {"v109", v109},
        {"v110", v110},
        {"v111", v111},
        {"", nullptr},
    };

struct MatmulConfig {
  INNER_MATMUL_FUNC impl;
  std::string name_impl;
  SmallKernels kernels;
  MatmulSize mmsize;
  float best = std::numeric_limits<float>::max();
};

struct MatMul {
  float *mat_a;
  float *mat_b;
  float *mat_c;
  int m;
  int n;
  int k;
  std::vector<std::pair<int ,int>> bm_mg_pair;
  std::vector<std::pair<int ,int>> bn_ng_pair;
  std::vector<std::pair<int ,int>> bk_kg_pair;
};
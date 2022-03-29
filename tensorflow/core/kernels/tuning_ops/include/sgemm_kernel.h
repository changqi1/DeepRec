#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#include <cstdlib>
#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <immintrin.h>
#include <emmintrin.h>

#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

#define likely(x)       __builtin_expect((x), 1)
#define unlikely(x)     __builtin_expect((x), 0)

// workaround c++17 warning
// #if __GNUC__ && __cpp_if_constexpr < 201606
// #define 
// #endif

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        compile_time_for<i-1>::op(function, args...);
        function(std::integral_constant<int, i-1>{}, args...);
    }
};
template <>
struct compile_time_for<1> {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        function(std::integral_constant<int, 0>{}, args...);
    }
};
template <>
struct compile_time_for<0> {
    // 0 loops, do nothing
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
    }
};

// Get mask for last column
template <int EXPANDED_N, int col>
inline unsigned short get_mask(unsigned short mask) {
    // Not last column, return 0xffffff indicating load/store all 16 floats
    if  (col < EXPANDED_N / 16 - 1)
      return (unsigned short)0xffff;
    else
      return mask;
}

template <int EXPANDED_N>
inline unsigned short get_mask(int col, unsigned short mask) {
    // Not last column, return 0xffffff indicating load/store all 16 floats
    if (col < EXPANDED_N / 16 - 1)
      return (unsigned short)0xffff;
    else
      return mask;
}

//  ___________________________________
// |         |         |         |     |
// |  fixmn  |         |         |fixm |
// |         |         |         |     |
// |_________|_________|_________|_____|
// |         |         |         |     |
// |         |         |         |fixm |
// |         |         |         |     |
// |_________|_________|_________|_____|
// |         |         |         |     |
// |  fixn   |  fixn   |  fixn   |nofix|
// |_________|_________|_________|_____|


// Small GEMM implemented as load A first
namespace laf {

// Get maximum lines computing at the same time, #registers for C is #LINES * #COLS
template<int COLS>
 inline int get_max_lines() {
  return 31 / (COLS + 1);
}

template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
   const int COLS = N / 16;
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loada = [&va, A, m] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % LINES;
       const int col = i / LINES;
      if  (line == 0) {
        vb = _mm512_loadu_ps(ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<LINES>::op(loada, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loada = [&va, A, m] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % lines;
       const int col = i / lines;
      if  (line == 0) {
        vb = _mm512_loadu_ps(ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<lines>::op(loada, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
   const int COLS = N / 16;
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % LINES;
       const int col = i / LINES;
      if  (line == 0) {
        vb = _mm512_loadu_ps(ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<LINES>::op(loada, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % lines;
       const int col = i / lines;
      if  (line == 0) {
        vb = _mm512_loadu_ps(ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<lines>::op(loada, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// EXPANDED_N: expanded N to multiple of 16
// Similar with fixmn, unless the last column load/store with mask
template<int M, int EXPANDED_N, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
   const int COLS = EXPANDED_N / 16;
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, mask, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % LINES;
       const int col = i / LINES;
      if  (line == 0) {
        vb = _mm512_mask_loadu_ps(vb, get_mask<EXPANDED_N, col>(mask), ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<LINES>::op(loada, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, mask, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % lines;
       const int col = i / lines;
      if  (line == 0) {
        vb = _mm512_mask_loadu_ps(vb, get_mask<EXPANDED_N, col>(mask), ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<lines>::op(loada, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// M is not a fixed value
template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
   const int COLS = N / 16;
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int LINES = get_max_lines<COLS>();

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % LINES;
       const int col = i / LINES;
      if  (line == 0) {
        vb = _mm512_loadu_ps(ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<LINES>::op(loada, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // There are remain lines
  if (m < M) {
    int lines = M - m;

    // Load from C or set to 0
    if  (ACC) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        #pragma unroll
        for (int n = 0; n < N; n += 16) {
          vc[INDEX(i, n/16, N/16)] = _mm512_loadu_ps(ADDRESS(C, m + i, n, ldc));
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < lines * COLS; ++i) {
        vc[i] = _mm512_setzero_ps();
      }
    }

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        va[i] = _mm512_set1_ps(*ADDRESS(A, m + i, k, lda));
      }
      #pragma unroll
      for (int n = 0; n < N; n += 16) {
        __m512 vb = _mm512_loadu_ps(ADDRESS(B, k, n, ldb));
        #pragma unroll
        for (int i = 0; i < lines; ++i) {
          vc[INDEX(i, n/16, N/16)] = _mm512_fmadd_ps(va[i], vb, vc[INDEX(i, n/16, N/16)]);
        }
      }
    } // end k

    // Store to C
    #pragma unroll
    for (int i = 0; i < lines; ++i) {
      #pragma unroll
      for (int n = 0; n < N; n += 16) {
        _mm512_storeu_ps(ADDRESS(C, m + i, n, ldc), vc[INDEX(i, n/16, N/16)]);
      }
    }
  } // end if
}

// EXPANDED_N: expanded N to multiple of 16
template<int EXPANDED_N, bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
   const int COLS = EXPANDED_N / 16;
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
   const int LINES = get_max_lines<COLS>();

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loada = [&va, A, m, lda] (auto i, int k) {
      va[i] = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + i, k, lda)));
    };

    auto compute = [&va, &vb, &vc, B, ldb, mask, COLS] (auto i, int k) { // Compute in vertical order
       const int line = i % LINES;
       const int col = i / LINES;
      if  (line == 0) {
        vb = _mm512_mask_loadu_ps(vb, get_mask<EXPANDED_N, col>(mask), ADDRESS(B, k, col * 16, ldb));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va[line], vb, vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<LINES>::op(loada, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // There are remain lines
  if (m < M) {
    int lines = M - m;

    // Load from C or set to 0
    if  (ACC) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
          vc[INDEX(i, col, COLS)] = _mm512_mask_loadu_ps(vc[INDEX(i, col, COLS)], get_mask<EXPANDED_N>(col, mask), ADDRESS(C, m + i, col * 16, ldc));
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < lines * COLS; ++i) {
        vc[i] = _mm512_setzero_ps();
      }
    }

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        va[i] = _mm512_set1_ps(*ADDRESS(A, m + i, k, lda));
      }
      #pragma unroll
      for (int col = 0; col < COLS; ++col) {
        vb = _mm512_mask_loadu_ps(vb, get_mask<EXPANDED_N>(col, mask), ADDRESS(B, k, col * 16, ldb));
        #pragma unroll
        for (int i = 0; i < lines; ++i) {
          vc[INDEX(i, col, COLS)] = _mm512_fmadd_ps(va[i], vb, vc[INDEX(i, col, COLS)]);
        }
      }
    } // end k

    // Store to C
    #pragma unroll
    for (int i = 0; i < lines; ++i) {
      #pragma unroll
      for (int col = 0; col < COLS; ++col) {
        _mm512_mask_storeu_ps(ADDRESS(C, m + i, col * 16, ldc), get_mask<EXPANDED_N>(col, mask), vc[INDEX(i, col, COLS)]);
      }
    }
  } // end if
}

} // end namespace laf


namespace lbf {
  
// Get maximum lines computing at the same time, #registers for C is #LINES * #COLS
template<int COLS>
 inline int get_max_lines() {
  return 31 / COLS - 1;
}

// Small GEMM implemented as load B first
// M&N&K are fixed, and lda&ldb&ldc also fixed
template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
   const int COLS = N / 16;
  //assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loadb = [&vb, B] (auto i, int k) {
      vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k + 3 < K; k += 4) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<LINES * COLS>::op(compute, k);
      compile_time_for<COLS>::op(loadb, k+1);
      compile_time_for<LINES * COLS>::op(compute, k+1);
      compile_time_for<COLS>::op(loadb, k+2);
      compile_time_for<LINES * COLS>::op(compute, k+2);
      compile_time_for<COLS>::op(loadb, k+3);
      compile_time_for<LINES * COLS>::op(compute, k+3);
    }

    if  (K % 4) { // remain k
       const int remain = K % 4;
      if  (remain == 3) {
        compile_time_for<COLS>::op(loadb, K-3);
        compile_time_for<LINES * COLS>::op(compute, K-3);
        compile_time_for<COLS>::op(loadb, K-2);
        compile_time_for<LINES * COLS>::op(compute, K-2);
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
      if  (remain == 2) {
        compile_time_for<COLS>::op(loadb, K-2);
        compile_time_for<LINES * COLS>::op(compute, K-2);
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
      if  (remain == 1) {
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
    }

    // Store to C
    auto store = [&vc, &C, m, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loadb = [&vb, B] (auto i, int k) {
      vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k (manually unroll cause perf drop for gcc 8.3.1)
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// Small GEMM implemented as load B first
template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
   const int COLS = N / 16;
  //assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb] (auto i, int k) {
      vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k (manually unroll cause big perf drop for gcc 8.3.1)
    #pragma unroll(4)
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb] (auto i, int k) {
      vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    #pragma unroll(4)
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// EXPANDED_N: expanded N to multiple of 16
// Similar with fixmn, unless the last column load/store with mask
template<int M, int EXPANDED_N, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
   const int COLS = EXPANDED_N / 16;
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
   const int max_lines = get_max_lines<COLS>();
   const int loops = (M + max_lines - 1) / max_lines;
   const int LINES = (M + loops - 1) / loops;

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb, mask] (auto i, int k) {
      vb[i] = _mm512_mask_loadu_ps(vb[i], get_mask<EXPANDED_N, i>(mask), ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if  (M % LINES) {
     const int lines = M % LINES;

    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<lines * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<lines * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb, mask] (auto i, int k) {
      vb[i] = _mm512_mask_loadu_ps(vb[i], get_mask<EXPANDED_N, i>(mask), ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<lines * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// Small GEMM implemented as load B first
template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
   const int COLS = N / 16;
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
   const int LINES = get_max_lines<COLS>();

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_loadu_ps(ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb] (auto i, int k) {
      vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if (m < M) {
    const int lines = M - m;

    // Load from C or set to 0
    if  (ACC) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        #pragma unroll
        for (int j = 0; j < COLS; ++j) {
          vc[INDEX(i, j, COLS)] = _mm512_loadu_ps(ADDRESS(C, m + i, j * 16, ldc));
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < lines * COLS; ++i) {
        vc[i] = _mm512_setzero_ps();
      }
    }

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      #pragma unroll
      for (int i = 0; i < COLS; ++i) {
        vb[i] = _mm512_loadu_ps(ADDRESS(B, k, i * 16, ldb));
      }
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        __m512 va = _mm512_set1_ps(*ADDRESS(A, m + i, k, lda));
        #pragma unroll
        for (int j = 0; j < COLS; ++j) {
          vc[INDEX(i, j, COLS)] = _mm512_fmadd_ps(va, vb[j], vc[INDEX(i, j, COLS)]);
        }
      }
    } // end k

    // Store to C
    #pragma unroll
    for (int i = 0; i < lines; ++i) {
      #pragma unroll
      for (int j = 0; j < COLS; ++j) {
        _mm512_storeu_ps(ADDRESS(C, m + i, j * 16, ldc), vc[INDEX(i, j, COLS)]);
      }
    }
  }
}

// EXPANDED_N: expanded N to multiple of 16
template<int EXPANDED_N, bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
   const int COLS = EXPANDED_N / 16;
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
   const int LINES = get_max_lines<COLS>();

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if  (ACC) {
      auto loadc = [&vc, C, m, ldc, mask, COLS] (auto i) {
         const int line = i / COLS;
         const int col = i % COLS;
        vc[i] = _mm512_mask_loadu_ps(vc[i], get_mask<EXPANDED_N, col>(mask), ADDRESS(C, m + line, col * 16, ldc));
      };
      compile_time_for<LINES * COLS>::op(loadc);
    } else {
      auto set0 = [&vc] (auto i) {
        vc[i] = _mm512_setzero_ps();
      };
      compile_time_for<LINES * COLS>::op(set0);
    }

    auto loadb = [&vb, B, ldb, mask] (auto i, int k) {
      vb[i] = _mm512_mask_loadu_ps(vb[i], get_mask<EXPANDED_N, i>(mask), ADDRESS(B, k, i * 16, ldb));
    };

    auto compute = [&va, &vb, &vc, A, m, lda, COLS] (auto i, int k) {
       const int line = i / COLS;
       const int col = i % COLS;
      if  (col == 0) {
        va = _mm512_broadcastss_ps(_mm_load_ss(ADDRESS(A, m + line, k, lda)));
      }
      vc[INDEX(line, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(line, col, COLS)]);
    };

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      compile_time_for<COLS>::op(loadb, k);
      compile_time_for<LINES * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&vc, &C, m, ldc, mask, COLS] (auto i) {
       const int line = i / COLS;
       const int col = i % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if (m < M) {
    const int lines = M - m;

    // Load from C or set to 0
    if  (ACC) {
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
          vc[INDEX(i, col, COLS)] = _mm512_mask_loadu_ps(vc[INDEX(i, col, COLS)], get_mask<EXPANDED_N>(col, mask), ADDRESS(C, m + i, col * 16, ldc));
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < lines * COLS; ++i) {
        vc[i] = _mm512_setzero_ps();
      }
    }

    // Accumulate along k
    for (int k = 0; k < K; ++k) {
      #pragma unroll
      for (int col = 0; col < COLS; ++col) {
        vb[col] = _mm512_mask_loadu_ps(vb[col], get_mask<EXPANDED_N>(col, mask), ADDRESS(B, k, col * 16, ldb));
      }
      #pragma unroll
      for (int i = 0; i < lines; ++i) {
        __m512 va = _mm512_set1_ps(*ADDRESS(A, m + i, k, lda));
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
          vc[INDEX(i, col, COLS)] = _mm512_fmadd_ps(va, vb[col], vc[INDEX(i, col, COLS)]);
        }
      }
    } // end k

    // Store to C
    #pragma unroll
    for (int i = 0; i < lines; ++i) {
      #pragma unroll
      for (int col = 0; col < COLS; ++col) {
        _mm512_mask_storeu_ps(ADDRESS(C, m + i, col * 16, ldc), get_mask<EXPANDED_N>(col, mask), vc[INDEX(i, col, COLS)]);
      }
    }
  } // end if
} // end small_gemm_nofix

} // end namespace lbf


template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
   auto COLS = N / 16;

  if  (COLS <= 4) {
    lbf::small_gemm_fixmnk_fixldabc<M, N, K, lda, ldb, ldc, ACC>(A, B, C);
  } else {
    laf::small_gemm_fixmnk_fixldabc<M, N, K, lda, ldb, ldc, ACC>(A, B, C);
  }
}

template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
   auto COLS = N / 16;

  if  (COLS <= 4) {
    lbf::small_gemm_fixmn<M, N, ACC>(A, B, C, lda, ldb, ldc, K);
  } else {
    laf::small_gemm_fixmn<M, N, ACC>(A, B, C, lda, ldb, ldc, K);
  }
}

template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
   auto COLS = N / 16;

  if  (COLS <= 4) {
    lbf::small_gemm_fixn<N, ACC>(A, B, C, lda, ldb, ldc, M, K);
  } else {
    laf::small_gemm_fixn<N, ACC>(A, B, C, lda, ldb, ldc, M, K);
  }
}

template<int M, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
   const int max_supported_cols = 8;
  auto COLS = (N + 15) / 16;

  if (unlikely(N > max_supported_cols * 16)) {
    printf("Bigger N is not supported at %s:%d\n", __FILE__, __LINE__);
    exit(-1);
  }

  // TODO: to fix the ugly code?
  if (COLS <= 4) {
    if (N > (max_supported_cols - 1) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 0) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 2) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 1) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 3) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 2) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 4) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 3) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 5) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 4) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 6) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 5) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 7) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 6) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 8) * 16) {
      lbf::small_gemm_fixm<M, (max_supported_cols - 7) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    }
  } else {
    if (N > (max_supported_cols - 1) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 0) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 2) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 1) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 3) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 2) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 4) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 3) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 5) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 4) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 6) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 5) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 7) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 6) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    } else if (N > (max_supported_cols - 8) * 16) {
      laf::small_gemm_fixm<M, (max_supported_cols - 7) * 16, ACC>(A, B, C, lda, ldb, ldc, N, K);
    }
  }
}

template<bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
   const int max_supported_cols = 8;
  auto COLS = (N + 15) / 16;

  if (unlikely(N > max_supported_cols * 16)) {
    printf("Bigger N is not supported at %s:%d\n", __FILE__, __LINE__);
    exit(-1);
  }

  // TODO: to fix the ugly code?
  if (COLS <= 4) {
    if (N > (max_supported_cols - 1) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 0) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 2) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 1) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 3) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 2) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 4) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 3) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 5) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 4) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 6) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 5) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 7) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 6) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 8) * 16) {
      lbf::small_gemm_nofix<(max_supported_cols - 7) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    }
  } else {
    if (N > (max_supported_cols - 1) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 0) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 2) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 1) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 3) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 2) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 4) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 3) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 5) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 4) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 6) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 5) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 7) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 6) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    } else if (N > (max_supported_cols - 8) * 16) {
      laf::small_gemm_nofix<(max_supported_cols - 7) * 16, ACC>(A, B, C, lda, ldb, ldc, M, N, K);
    }
  }
}

#endif

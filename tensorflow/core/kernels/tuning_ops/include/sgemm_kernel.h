#include <cstdlib>
#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <immintrin.h>
#include <emmintrin.h>

#define USE_GEMMK

#ifndef USE_GEMMK


#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

#define likely(x)       __builtin_expect((x), 1)
#define unlikely(x)     __builtin_expect((x), 0)

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

// A class for compile time if
// For lower version GCC which does not support "if constexpr"
template <bool condition>
struct compile_time_if_cls {
    template <typename IfLambda, typename ElseLambda>
    inline static void doit(const IfLambda& if_f, const ElseLambda& else_f) {
        if_f();
    }
};
template <>
struct compile_time_if_cls<false> {
    template <typename IfLambda, typename ElseLambda>
    inline static void doit(const IfLambda& if_f, const ElseLambda& else_f) {
        else_f();
    }
};
template<bool condition, typename IfLambda, typename ElseLambda>
void compile_time_if(const IfLambda& if_f, const ElseLambda& else_f) {
    compile_time_if_cls<condition>::doit(if_f, else_f);
}
template<bool condition, typename IfLambda>
void compile_time_if(const IfLambda& if_f) {
    compile_time_if_cls<condition>::doit(if_f, []{});
}

// Get mask to load/store data
template <int EXPANDED_N, const int col>
inline unsigned short get_mask(const unsigned short mask) {
    // Not last column, return 0xffffff indicating load/store all 16 floats
    if constexpr (col < EXPANDED_N / 16 - 1)
      return (unsigned short)0xffff;
    else
      return mask;
}

template <int EXPANDED_N>
inline unsigned short get_mask(const int col, const unsigned short mask) {
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
constexpr inline int get_max_lines() {
  return 31 / (COLS + 1);
}

template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
  #define COLS (N / 16)
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if constexpr (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }

  #undef COLS
}

template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
  #define COLS (N / 16)
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if constexpr (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }

  #undef COLS
}

// EXPANDED_N: expanded N to multiple of 16
// Similar with fixmn, unless the last column load/store with mask
template<int M, int EXPANDED_N, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
  #define COLS (EXPANDED_N / 16)
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb, mask] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if constexpr (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb, mask] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }

  #undef COLS
}

// M is not a fixed value
template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
  #define COLS (N / 16)
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int LINES = get_max_lines<COLS>();

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // There are remain lines
  if (m < M) {
    int lines = M - m;

    // Load from C or set to 0
    if constexpr (ACC) {
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

  #undef COLS
}

// EXPANDED_N: expanded N to multiple of 16
template<int EXPANDED_N, bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
  #define COLS (EXPANDED_N / 16)
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int LINES = get_max_lines<COLS>();

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va[LINES];
  __m512 vb;
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, B, ldb, mask] (auto i, int k) { // Compute in vertical order
      constexpr const int line = (int)i % LINES;
      constexpr const int col = (int)i / LINES;
      if constexpr (line == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // There are remain lines
  if (m < M) {
    int lines = M - m;

    // Load from C or set to 0
    if constexpr (ACC) {
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

  #undef COLS
}

} // end namespace laf


namespace lbf {
  
// Get maximum lines computing at the same time, #registers for C is #LINES * #COLS
template<int COLS>
constexpr inline int get_max_lines() {
  return 31 / COLS - 1;
}

// Small GEMM implemented as load B first
// M&N&K are fixed, and lda&ldb&ldc also fixed
template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
  #define COLS (N / 16)
  //assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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

    if constexpr (K % 4) { // remain k
      constexpr const int remain = K % 4;
      if constexpr (remain == 3) {
        compile_time_for<COLS>::op(loadb, K-3);
        compile_time_for<LINES * COLS>::op(compute, K-3);
        compile_time_for<COLS>::op(loadb, K-2);
        compile_time_for<LINES * COLS>::op(compute, K-2);
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
      if constexpr (remain == 2) {
        compile_time_for<COLS>::op(loadb, K-2);
        compile_time_for<LINES * COLS>::op(compute, K-2);
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
      if constexpr (remain == 1) {
        compile_time_for<COLS>::op(loadb, K-1);
        compile_time_for<LINES * COLS>::op(compute, K-1);
      }
    }

    // Store to C
    auto store = [&vc, &C, m] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if constexpr (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }

  #undef COLS
}

// Small GEMM implemented as load B first
template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
  constexpr const int COLS = N / 16;
  //assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }
}

// EXPANDED_N: expanded N to multiple of 16
// Similar with fixmn, unless the last column load/store with mask
template<int M, int EXPANDED_N, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
  #define COLS (EXPANDED_N / 16)
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int max_lines = get_max_lines<COLS>();
  constexpr const int loops = (M + max_lines - 1) / max_lines;
  constexpr const int LINES = (M + loops - 1) / loops;

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if constexpr (M % LINES) {
    constexpr const int lines = M % LINES;

    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<lines * COLS>::op(store);
  }

  #undef COLS
}

// Small GEMM implemented as load B first
template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
  #define COLS (N / 16)
  assert(N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int LINES = get_max_lines<COLS>();

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if (m < M) {
    const int lines = M - m;

    // Load from C or set to 0
    if constexpr (ACC) {
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

  #undef COLS
}

// EXPANDED_N: expanded N to multiple of 16
template<int EXPANDED_N, bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
  #define COLS (EXPANDED_N / 16)
  assert(EXPANDED_N % 16 == 0);

  // How many lines of A are computed at the same time
  constexpr const int LINES = get_max_lines<COLS>();

  // How many float numbers in last column
  const int floats = (N % 16 == 0 ? 16 : N % 16);
  unsigned short mask = (1 << floats) - 1;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];

  int m = 0;
  for (; m + LINES <= M; m += LINES) {
    // Load from C or set to 0
    if constexpr (ACC) {
      auto loadc = [&vc, C, m, ldc, mask] (auto i) {
        constexpr const int line = decltype(i)::value / COLS;
        constexpr const int col = decltype(i)::value % COLS;
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

    auto compute = [&va, &vb, &vc, A, m, lda] (auto i, int k) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      if constexpr (col == 0) {
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
    auto store = [&vc, &C, m, ldc, mask] (auto i) {
      constexpr const int line = decltype(i)::value / COLS;
      constexpr const int col = decltype(i)::value % COLS;
      _mm512_mask_storeu_ps(ADDRESS(C, m + line, col * 16, ldc), get_mask<EXPANDED_N, col>(mask), vc[i]);
    };

    compile_time_for<LINES * COLS>::op(store);
  } // end m

  // Deal with remaining rows
  if (m < M) {
    const int lines = M - m;

    // Load from C or set to 0
    if constexpr (ACC) {
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
  
  #undef COLS
} // end small_gemm_nofix

} // end namespace lbf

#endif //ndef USE_GEMMK


template<int M, int N, int K, int lda, int ldb, int ldc, bool ACC>
void small_gemm_fixmnk_fixldabc(const float *A, const float *B, float *C) {
#ifndef USE_GEMMK
  constexpr const int COLS = N / 16;

  if constexpr (COLS <= 4) {
    lbf::small_gemm_fixmnk_fixldabc<M, N, K, lda, ldb, ldc, ACC>(A, B, C);
  } else {
    laf::small_gemm_fixmnk_fixldabc<M, N, K, lda, ldb, ldc, ACC>(A, B, C);
  }
#endif
}

template<int M, int N, bool ACC>
void small_gemm_fixmn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K) {
#ifndef USE_GEMMK
  constexpr const int COLS = N / 16;

  if constexpr (COLS <= 4) {
    lbf::small_gemm_fixmn<M, N, ACC>(A, B, C, lda, ldb, ldc, K);
  } else {
    laf::small_gemm_fixmn<M, N, ACC>(A, B, C, lda, ldb, ldc, K);
  }
#endif
}

template<int N, bool ACC>
void small_gemm_fixn(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int K) {
#ifndef USE_GEMMK
  constexpr const int COLS = N / 16;

  if constexpr (COLS <= 4) {
    lbf::small_gemm_fixn<N, ACC>(A, B, C, lda, ldb, ldc, M, K);
  } else {
    laf::small_gemm_fixn<N, ACC>(A, B, C, lda, ldb, ldc, M, K);
  }
#endif
}

template<int M, bool ACC>
void small_gemm_fixm(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int N, int K) {
#ifndef USE_GEMMK
  constexpr const int max_supported_cols = 8;
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
#endif
}

template<bool ACC>
void small_gemm_nofix(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K) {
#ifndef USE_GEMMK
  constexpr const int max_supported_cols = 8;
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
#endif
}

#ifdef USE_GEMMK

#include "libxsmm.h"
int gemmkernel;
int roll_back;

void small_gemm_libxsmm(bool transa, bool transb, const float *A, const float *B, float *C, int lda, int ldb, int ldc, int M, int N, int K, bool ACC) {
  float alpha = 1.0, beta = 0.0;
  if (ACC) {
        beta = 1.0;
  }
  char ta[] = "N";
  char tb[] = "N";
  if (transa)
        ta[0] = 'T';
  if (transb)
        tb[0] = 'T';
  if (gemmkernel == 0 || (M <= 64 && N <= 64 && K<=64)) {
  	libxsmm_sgemm(tb, ta, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
  } else if (gemmkernel == 1) {
	assert(0);
//  	dnnl_sgemm(ta[0], tb[0], M, N, K, alpha, A, lda, B, ldb,
//                    beta, C, ldc);
  } else {
  	mkldnn_sgemm(tb, ta, &N, &M, &K, &alpha, B, &ldb, A, &lda,
                    &beta, C, &ldc);
//  	dnnl::threadpool_interop::sgemm(ta[0], tb[0], M, N, K, alpha, A, lda, B, ldb,
//                    beta, C, ldc, nullptr);
  }
}
#endif

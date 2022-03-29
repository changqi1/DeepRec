#include <signal.h>
#include "tunable_matmul.h"

const MatmulImpl TunableMatmul::impl_list[] = {
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

void TunableMatmul::update_kernels(SmallKernels &kernels, int bm, int bn)
{
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
}

static TunableMatmul *tmm = nullptr;
#define TUNED_CONFIG_FILE "/tmp/tuned"

static void intHandler(int)
{
    tmm->save_config(TUNED_CONFIG_FILE);
    exit(-1);
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s M N K [flush_b=1/0] [tune=1/0] [host=1/0]\n", argv[0]);
        exit(-1);
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int lda = K;
    int ldb = N;
    int ldc = N;

    bool flush_b = false;
    if (argc > 4 && argv[4][0] == '1')
    {
        flush_b = true;
    }

    bool tune = false;
    if (argc > 5 && argv[5][0] == '1')
    {
        tune = true;
    }

    bool host = false;
    if (argc > 6 && argv[6][0] == '1')
    {
        host = true;
    }

    tmm = new TunableMatmul(M, N, K, lda, ldb, ldc);
    if (tune)
    {
        signal(SIGINT, intHandler);
        if (host){
            tmm->host_tune(flush_b);
        } else {
            tmm->tune(flush_b);
        }
        tmm->save_config(TUNED_CONFIG_FILE);
    }
    else
    {
        if (!tmm->load_config(TUNED_CONFIG_FILE))
        {
            printf("Cannot load matmul config.\n");
            exit(-1);
        }
    }

    // Allocate buffer and prepare data for A and B
    float *a = (float *)aligned_alloc(64, M * lda * sizeof(float));
    float *b = (float *)aligned_alloc(64, K * ldb * sizeof(float));
    float *c = (float *)aligned_alloc(64, M * ldc * sizeof(float));

    for (int i = 0; i < M * lda; ++i)
    {
        a[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < K * ldb; ++i)
    {
        b[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < 20; ++i)
    {
        Timer t;
        tmm->compute(a, b, c);
        printf("Time: %f ms\n", t.getTime());
        if (flush_b)
        {
            TunableMatmul::flush_cache(b, K * ldb);
        }
    }

    free(c);
    free(b);
    free(a);
    delete tmm;
}
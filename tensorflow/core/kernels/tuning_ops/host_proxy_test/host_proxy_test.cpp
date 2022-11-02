#include "host_proxy_test.h"
#include "host_proxy_manager.h"
#include <iostream>
#include <thread>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/syscall.h>
#define gettid() syscall(__NR_gettid)


std::map<std::string, MatmulConfig> config_list;
std::map<std::string, MatMul> mat_list;
bool flush_b = true;



void full_Mats(T *A, T *B,int m,int n,int k) {
    for (int i = 0; i < m * k; ++i) {
        A[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < k * n; ++i) {
        B[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }
}

void prepare_bm(int m, std::vector<int> &bm_list) {
    if (m < 32) {
        bm_list.push_back(m);
    } else if (m < 64) {
        bm_list.push_back(m);
        bm_list.push_back((m + 1) / 2);
    } else {
        bm_list.push_back(64);
        bm_list.push_back(48);
        bm_list.push_back(32);
    }
}

void prepare_bn(int n, std::vector<int> &bn_list) {
    prepare_bm(n, bn_list);
}

void prepare_bk(int k, std::vector<int> &bk_list) {
    // bk = 64, ...
    // int candidates[] = { 64, 96, 128, 160, 192, 224, 256, 384, 512 };
    int candidates[] = {64, 128, 256, 512};
    for (int i = 0; i < sizeof(candidates) / sizeof(int); ++i) {
        if (candidates[i] <= k) {
            bk_list.push_back(candidates[i]);
        } else {
            break;
        }
    }

    // bk = k, k/2, k/3, ...
    int divider = 1;
    do {
        int bk = (k + divider - 1) / divider;
        // do not try small values
        if (bk < 128) {
            break;
        }
        if (std::find(bk_list.begin(), bk_list.end(), bk) == bk_list.end()) {
            bk_list.push_back(bk);
        }
        divider += 1;
    } while (true);

    // In case of small k
    if (bk_list.empty()) {
        bk_list.push_back(k);
    }
}

void prepare_bk_kg_pair(int k, std::vector<int> &bk_list, std::vector<pair<int, int>> &bk_kg_pair) {
    for (auto cur_bk : bk_list) {
        int kblocks = (k + cur_bk - 1) / cur_bk;
        int max_kgroups = kblocks < MAX_GROUP_LIMIT ? kblocks : MAX_GROUP_LIMIT;
        for (int i = 1; i <= max_kgroups; i++) {
            pair<int, int> cur_p(cur_bk, i);
            bk_kg_pair.push_back(cur_p);
        }
    }
}

void prepare_bm_mg_pair(int m, std::vector<int> &bm_list, std::vector<pair<int, int>> &bm_mg_pair) {
    for (auto cur_bm : bm_list) {
        int mblocks = (m + cur_bm - 1) / cur_bm;
        int max_mgroups = mblocks < MAX_GROUP_LIMIT ? mblocks : MAX_GROUP_LIMIT;
        for (int i = 1; i <= max_mgroups; i++) {
            pair<int, int> cur_p(cur_bm, i);
            bm_mg_pair.push_back(cur_p);
        }
    }
}

void prepare_bn_ng_pair(int n, std::vector<int> &bn_list, std::vector<pair<int, int>> &bn_ng_pair) {
    for (auto cur_bn : bn_list) {
        int nblocks = (n + cur_bn - 1) / cur_bn;
        int max_ngroups = nblocks < MAX_GROUP_LIMIT ? nblocks : MAX_GROUP_LIMIT;
        for (int i = 1; i <= max_ngroups; i++) {
            pair<int, int> cur_p(cur_bn, i);
            bn_ng_pair.push_back(cur_p);
        }
    }
}

void update_kernels(SmallKernels &kernels, int bm, int bn) {
    if (bm == 32) {
        SET_KERNELS_ENUM_BN(32)
    } else if (bm == 48) {
        SET_KERNELS_ENUM_BN(48)
    } else if (bm == 64) {
        SET_KERNELS_ENUM_BN(64)
    } else if (bm == 80) {
        SET_KERNELS_ENUM_BN(80)
    } else if (bm < 32) {
        switch (bm) {
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
    } else {
        printf("Unsupported kernel for bm=%d\n", bm);
        exit(-1);
    }

    kernels.kernel_nofix_acc = small_gemm_nofix<true>;
    kernels.kernel_nofix_nonacc = small_gemm_nofix<false>;
}

static void flush_cache(const T *buf, size_t size) {
#pragma omp parallel for
    for (size_t offset = 0; offset < size; offset += CACHELINE_SIZE / sizeof(T)) {
        _mm_clflush(buf + offset);
    }
}

void public_update_kernels(SmallKernels &kernels_, int bm_, int bn_) {
    update_kernels(kernels_, bm_, bn_);
}

PerfStat benchmark(INNER_MATMUL_FUNC func, const MatmulSize &mmsize, const SmallKernels &kernels, const T *A,
                   const T *B, T *C, bool flush_b) {
    const int warmup_loops = 1;
    const int benchmark_loops = 5;

    PerfStat perfStat;
    std::vector<float> latencies;
    latencies.reserve(benchmark_loops);

    // Warmup and benchmark
    for (int i = 0; i < warmup_loops + benchmark_loops; ++i) {
        Timer t;
        func(A, B, C, mmsize, kernels);
        if (i >= warmup_loops) {
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
    for (float latency : latencies) {
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

PerfStat public_benchmark(INNER_MATMUL_FUNC func, const MatmulSize &mmsize, const SmallKernels &kernels, const T *A,
                          const T *B, T *C, bool flush_b) {
    PerfStat stat_ = benchmark(func, mmsize, kernels, A, B, C, flush_b);
    return stat_;
}

const float cust_evaluate(std::string const &name,std::vector<int> const &params) {
    SmallKernels var_kernels;
    MatmulSize var_mmsize;
    var_mmsize.m = mat_list[name].m;
    var_mmsize.n = var_mmsize.ldb = var_mmsize.ldc = mat_list[name].n;
    var_mmsize.k = var_mmsize.lda = mat_list[name].k;
    var_mmsize.bm = (int)mat_list[name].bm_mg_pair[(int)params[0]].first;
    var_mmsize.bn = (int)mat_list[name].bn_ng_pair[(int)params[1]].first;
    var_mmsize.bk = (int)mat_list[name].bk_kg_pair[(int)params[2]].first;
    var_mmsize.mgroups = (int)mat_list[name].bm_mg_pair[(int)params[0]].second;
    var_mmsize.ngroups = (int)mat_list[name].bn_ng_pair[(int)params[1]].second;
    var_mmsize.kgroups = (int)mat_list[name].bk_kg_pair[(int)params[2]].second;
    INNER_MATMUL_FUNC impl = impl_list[(int)params[3]].impl;

    std::cout << "cur thread:" << gettid() << " cur m:" << var_mmsize.m << ",cur n:" << var_mmsize.n << ",cur k:" << var_mmsize.k << " ";
    std::cout << "cur bm:" << var_mmsize.bm << ",cur mgroups:" << var_mmsize.mgroups << " ";
    std::cout << "cur bn:" << var_mmsize.bn << ",cur ngroups:" << var_mmsize.ngroups << " ";
    std::cout << "cur bk:" << var_mmsize.bk << ",cur kgroups:" << var_mmsize.kgroups << " ";
    std::cout << "cur impl:" << impl_list[(int)params[3]].name << std::endl;

    int mblocks = (var_mmsize.m + var_mmsize.bm - 1) / var_mmsize.bm;
    int nblocks = (var_mmsize.n + var_mmsize.bn - 1) / var_mmsize.bn;
    int kblocks = (var_mmsize.k + var_mmsize.bk - 1) / var_mmsize.bk;
    var_mmsize.mblocks_per_group = (mblocks + var_mmsize.mgroups - 1) / var_mmsize.mgroups;
    var_mmsize.nblocks_per_group = (nblocks + var_mmsize.ngroups - 1) / var_mmsize.ngroups;
    var_mmsize.kblocks_per_group = (kblocks + var_mmsize.kgroups - 1) / var_mmsize.kgroups;

    public_update_kernels(var_kernels, var_mmsize.bm, var_mmsize.bn);
    PerfStat stat = public_benchmark(impl, var_mmsize, var_kernels, mat_list[name].mat_a, mat_list[name].mat_b,mat_list[name].mat_c, flush_b);
    float avg_latency = stat.avg_latency;
    auto mmpair = config_list.find(name);
    if (mmpair == config_list.end())
         std::cout << "Error!:This proxy doesn't have its mmconfig"<<std::endl;
    auto mmconfig = mmpair->second;
    if (stat.avg_latency < mmconfig.best) {
        mmconfig.best = stat.avg_latency;
        mmconfig.mmsize = var_mmsize;
        mmconfig.kernels = var_kernels;
        mmconfig.impl = impl;
        mmconfig.name_impl = impl_list[(int)params[3]].name;
        config_list[name]=mmconfig;
    }
    return avg_latency;
}

const bool cust_condition(TuningContext &context) {
    auto cur_iter =context.cur_iter;
    auto cur_percent = context.tune_percentage;
    auto cur_best_params = context.cur_best_params;
    auto cur_best_fitness = context.cur_best_fitness;
    auto cur_tune_params = context.cur_tune_params;
    auto cur_tune_fitness = context.cur_tune_fitness;
    if (cur_percent==(float)0.05){
        std::cout<<"Now percentage arrive 5%"<<std::endl;
        return true;
    }
    return false;
}
void compute_tune(MatmulConfig mmconfig,const T *A, const T *B, T *C)
  {
    //Timer t;
    if (mmconfig.impl)
    {
        //impl_list[0].impl(A, B, C, mmconfig.mmsize, mmconfig.kernels);
        mmconfig.impl(A, B, C, mmconfig.mmsize, mmconfig.kernels);
    }
    else
    {
      printf("TunableMatmul: Cannot find an implementation.\n");
      exit(-1);
    }
    //printf("tuning-matmul Compute Time: %f ms\n", t.getTime());
  }

void compute_v0(MatmulConfig mmconfig,const T *A, const T *B, T *C)
  {
    //Timer t;
    if (mmconfig.impl)
    {
        impl_list[0].impl(A, B, C, mmconfig.mmsize, mmconfig.kernels);
        //mmconfig.impl(A, B, C, mmconfig.mmsize, mmconfig.kernels);
    }
    else
    {
      printf("TunableMatmul: Cannot find an implementation.\n");
      exit(-1);
    }
    //printf("tuning-matmul Compute Time: %f ms\n", t.getTime());
  }
void getCurRes(std::string resName, MatMul curMat,HostProxy::Ptr my_host_proxy) {
    TuningContext context = my_host_proxy->GetTuningContext();
    map<std::string, int> cur_res = my_host_proxy->GetTunedResult();

    if (cur_res["impl"]==-std::numeric_limits<int>::max()){
        std::cout<<"the first round of iteration has not been completed, so there is no best result at present!"<<std::endl;
        return;
    }
    std::cout << resName << " best params:";
    for (auto iter : cur_res) {
        if (iter.first=="impl")
            std::cout << iter.first << ":" << impl_list[(int)iter.second].name<< " ";
        else if (iter.first =="bmmg_pairs"){
            std::cout <<"bm:" << curMat.bm_mg_pair[(int)iter.second].first<< ","<<"mgroups:"<< curMat.bm_mg_pair[(int)iter.second].second<<"  ";
        }
        else if (iter.first =="bnng_pairs"){
            std::cout <<"bn:" << curMat.bn_ng_pair[(int)iter.second].first<< ","<<"ngroups:"<< curMat.bn_ng_pair[(int)iter.second].second<<"  ";
        }
        else if (iter.first =="bkkg_pairs"){
            std::cout <<"bk:" << curMat.bk_kg_pair[(int)iter.second].first<< ","<<"kgroups:"<< curMat.bk_kg_pair[(int)iter.second].second<<"  ";
        }
        else
            std::cout << iter.first << ":" << iter.second << " ";
    }
    std::cout << "\n" << resName << " best cost:" << my_host_proxy->GetBestCost() << std::endl;
}

void print_blocks(std::string block_name,vector<int> block_list){
    std::cout<<block_name<<":";
    for (auto bb: block_list){
        std::cout<<bb<<" ";
    }
    std::cout<<"\n";
}

void print_pairs(std::string block_name,vector<pair<int,int>> bg_pairs){
    std::cout<<block_name<<":";
    for (auto bb: bg_pairs){
        std::cout<<(int)bb.first<<" "<<(int)bb.second<<"|";
    }
    std::cout<<"\n";
}

void val_performance(std::string name, float bestFitness, const T *A, const T *B, T *C, int k, int n) {
    float avg_cost_af2 = 0;
    float sum_cost_af2 = 0;
    float avg_cost_af5 = 0;
    float sum_cost_af5 = 0;
    for (int i = 1; i <= 20; ++i) {
        Timer t_tune;
        // test
        //
        compute_tune(config_list[name], A, B, C);
        float cur_t = t_tune.getTime();
        if (i>=3) {
            if (i >= 6 ) {
                sum_cost_af5 += cur_t;
                avg_cost_af5 = sum_cost_af5 / (i - 5);
            }
            sum_cost_af2 += cur_t;
            avg_cost_af2 = sum_cost_af2 / (i - 2);
        }
        if (flush_b) {
            flush_cache(B, k * n);
        }
    }
    std::cout<<name<<" res:"<<std::endl;
    printf("   AVG Time our tuned af2: %f ms\n", avg_cost_af2);
    printf("   AVG Time our tuned af5: %f ms\n", avg_cost_af5);
    float tune_dis_val = avg_cost_af5 - bestFitness;
    printf("   diff between tune and val: %f ms\n", tune_dis_val);

}
int main(int argc, char *argv[]) {
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int m2 = atoi(argv[4]);
    int n2 = atoi(argv[5]);
    int k2 = atoi(argv[6]);
    int iter = atoi(argv[7]);
    int pops = atoi(argv[8]);
    char *algo = argv[9];
    std::cout << "m:" << m << ",n:" << n << ",k:" << k << "m2:" << m2 << ",n2:" << n2 << ",k2:" << k2 << "iter:" << iter << ",pops:" << pops
              << std::endl;
    std::vector<int> bk_list;
    std::vector<int> bm_list;
    std::vector<int> bn_list;

    std::vector<int> bk_list2;
    std::vector<int> bm_list2;
    std::vector<int> bn_list2;

    T *a = (T *)aligned_alloc(64, m * k * sizeof(T));
    T *b = (T *)aligned_alloc(64, k * n * sizeof(T));
    T *c = (T *)aligned_alloc(64, m * n * sizeof(T));

    T *a2 = (T *)aligned_alloc(64, m2 * k2 * sizeof(T));
    T *b2 = (T *)aligned_alloc(64, k2 * n2 * sizeof(T));
    T *c2 = (T *)aligned_alloc(64, m2 * n2 * sizeof(T));

    full_Mats(a, b, m, n, k);
    full_Mats(a2, b2, m2, n2, k2);
    Timer t_online;
    auto proxy_handle1 = HostOSTProxyManager::Instance().CreateNewProxy("m1n1k1");
    auto my_host_proxy1 = HostOSTProxyManager::Instance().GetProxy(proxy_handle1);
    auto proxy_handle2 = HostOSTProxyManager::Instance().CreateNewProxy("m2n2k2");
    auto my_host_proxy2 = HostOSTProxyManager::Instance().GetProxy(proxy_handle2);
    auto name1 = my_host_proxy1->GetName();
    auto name2 = my_host_proxy2->GetName();
    MatMul curMat1;
    MatMul curMat2;
    curMat1.mat_a = a;
    curMat1.mat_b = b;
    curMat1.mat_c = c;
    curMat1.m = m;
    curMat1.n = n;
    curMat1.k = k;

    curMat2.mat_a = a2;
    curMat2.mat_b = b2;
    curMat2.mat_c = c2;
    curMat2.m = m2;
    curMat2.n = n2;
    curMat2.k = k2;

    MatmulConfig mmconfig1;
    MatmulConfig mmconfig2;
    config_list[name1]= mmconfig1;
    config_list[name2]= mmconfig2;

    MatmulSize mmsize;
    MatmulSize mmsize2;
    mmsize.m = m;
    mmsize.n = mmsize.ldb = mmsize.ldc = n;
    mmsize.k = mmsize.lda = k;
    mmsize2.m = m2;
    mmsize2.n = mmsize2.ldb = mmsize2.ldc = n2;
    mmsize2.k = mmsize2.lda = k2;

    prepare_bm(mmsize.m, bm_list);
    prepare_bn(mmsize.n, bn_list);
    prepare_bk(mmsize.k, bk_list);
    prepare_bm(mmsize2.m, bm_list2);
    prepare_bn(mmsize2.n, bn_list2);
    prepare_bk(mmsize2.k, bk_list2);

    prepare_bk_kg_pair(mmsize.k, bk_list, curMat1.bk_kg_pair);
    prepare_bm_mg_pair(mmsize.m, bm_list, curMat1.bm_mg_pair);
    prepare_bn_ng_pair(mmsize.n, bn_list, curMat1.bn_ng_pair);
    prepare_bk_kg_pair(mmsize2.k, bk_list2, curMat2.bk_kg_pair);
    prepare_bm_mg_pair(mmsize2.m, bm_list2, curMat2.bm_mg_pair);
    prepare_bn_ng_pair(mmsize2.n, bn_list2, curMat2.bn_ng_pair);
    print_pairs("bm_g1",curMat1.bm_mg_pair);
    print_pairs("bn_g1",curMat1.bn_ng_pair);
    print_pairs("bk_g1",curMat1.bk_kg_pair);
    print_pairs("bm_g2",curMat2.bm_mg_pair);
    print_pairs("bn_g2",curMat2.bn_ng_pair);
    print_pairs("bk_g2",curMat2.bk_kg_pair);
    int max_bkkg_num = curMat1.bk_kg_pair.size();
    int max_bmmg_num = curMat1.bm_mg_pair.size();
    int max_bnng_num = curMat1.bn_ng_pair.size();
    int max_bkkg_num2 = curMat2.bk_kg_pair.size();
    int max_bmmg_num2 =  curMat2.bm_mg_pair.size();
    int max_bnng_num2 = curMat2.bn_ng_pair.size();

    mat_list[name1] = curMat1;
    mat_list[name2] = curMat2;
    my_host_proxy1->SetParamter("bmmg_pairs", 0, max_bmmg_num - 1);
    my_host_proxy1->SetParamter("bnng_pairs", 0, max_bnng_num - 1);
    my_host_proxy1->SetParamter("bkkg_pairs", 0, max_bkkg_num - 1);
    my_host_proxy1->SetParamter("impl", 0, 35);
    my_host_proxy1->SetAlgorithm(algo, iter, pops);
    my_host_proxy1->SetEvaluateFunc(cust_evaluate);
    my_host_proxy1->SetConditionFunc(cust_condition);

    my_host_proxy2->SetParamter("bmmg_pairs", 0, max_bmmg_num2 - 1);
    my_host_proxy2->SetParamter("bnng_pairs", 0, max_bnng_num2 - 1);
    my_host_proxy2->SetParamter("bkkg_pairs", 0, max_bkkg_num2 - 1);
    my_host_proxy2->SetParamter("impl", 0, 35);
    my_host_proxy2->SetAlgorithm(algo, iter, pops);
    my_host_proxy2->SetEvaluateFunc(cust_evaluate);
    my_host_proxy2->SetConditionFunc(cust_condition);

    my_host_proxy1->Start();
    my_host_proxy2->Start();
    auto state1 = my_host_proxy1->GetProxyState();
    auto state2 = my_host_proxy2->GetProxyState();
    do {
        if (state1 == HostProxy::State::STOPPED || state1 == HostProxy::State::SUSPENDED)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        state1 = my_host_proxy1->GetProxyState();
    } while (state1 == HostProxy::State::RUNNING);
    do {
        if (state2 == HostProxy::State::STOPPED || state2 == HostProxy::State::SUSPENDED)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        state2 = my_host_proxy2->GetProxyState();
    } while (state2 == HostProxy::State::RUNNING);

    map<std::string, int> middle_res1 = my_host_proxy1->GetTunedResult();
    getCurRes("Middleproxy1 result:", curMat1,my_host_proxy1);
    getCurRes("Middleproxy2 result:", curMat2,my_host_proxy2);
    my_host_proxy1->Start();
    my_host_proxy2->Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    my_host_proxy1->Suspend();
    my_host_proxy2->Suspend();
    getCurRes("Middleproxy1 result:", curMat1,my_host_proxy1);
    getCurRes("Middleproxy2 result:", curMat2,my_host_proxy2);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    my_host_proxy1->Start();
    my_host_proxy2->Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //if you want to stop the tune process before tune over,use my_host_proxy->Stop(); and comment out "do{...}" below
    do {
        state1 = my_host_proxy1->GetProxyState();
        state2 = my_host_proxy2->GetProxyState();
        if (state1 == HostProxy::State::STOPPED && state2 == HostProxy::State::STOPPED )
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (state1 == HostProxy::State::RUNNING || state2 == HostProxy::State::RUNNING);

    printf("Time of tune: %f ms\n", t_online.getTime());
    getCurRes("Final proxy1 result:",mat_list[name1],my_host_proxy1);
    getCurRes("Final proxy2 result:",mat_list[name2],my_host_proxy2);
    float bestFitness1 = my_host_proxy1->GetBestCost();
    float bestFitness2 = my_host_proxy2->GetBestCost();


    float *a_t = (float *)aligned_alloc(64, m * k * sizeof(float));
    float *b_t = (float *)aligned_alloc(64, k * n * sizeof(float));
    float *c_t = (float *)aligned_alloc(64, m * n * sizeof(float));
    float *a_t2 = (float *)aligned_alloc(64, m2 * k2 * sizeof(float));
    float *b_t2 = (float *)aligned_alloc(64, k2 * n2 * sizeof(float));
    float *c_t2 = (float *)aligned_alloc(64, m2 * n2 * sizeof(float));
    full_Mats(a_t, b_t, m, n, k);
    full_Mats(a_t2, b_t2, m2, n2, k2);
    std::cout << "mmconfig1: impl:" << config_list[name1].name_impl << ", bm:" << config_list[name1].mmsize.bm
            << ",bn:" << config_list[name1].mmsize.bn << ",bk:" <<config_list[name1].mmsize.bk << ",mgroups:" <<config_list[name1].mmsize.mgroups
            << ",ngroups:" << config_list[name1].mmsize.ngroups << ",kgroups:" <<config_list[name1].mmsize.kgroups << std::endl;
    std::cout << "mmconfig2: impl:" << config_list[name2].name_impl << ", bm:" << config_list[name2].mmsize.bm
              << ",bn:" << config_list[name2].mmsize.bn << ",bk:" <<config_list[name2].mmsize.bk << ",mgroups:" <<config_list[name2].mmsize.mgroups
              << ",ngroups:" << config_list[name2].mmsize.ngroups << ",kgroups:" <<config_list[name2].mmsize.kgroups << std::endl;
    for (int i = 0; i < m * k; ++i) {
        a_t[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }
    for (int i = 0; i < k * n; ++i) {
        b_t[i] = static_cast<float>(1.0f * rand() / RAND_MAX);
    }
    val_performance(name1, bestFitness1, a_t, b_t, c_t, k, n);
    val_performance(name2, bestFitness2, a_t2, b_t2, c_t2, k2, n2);
    if (proxy_handle1) {
        HostOSTProxyManager::Instance().ReleaseProxy(proxy_handle1);
        std::cout << "Released proxy1" << std::endl;
    }
    if (proxy_handle2) {
        HostOSTProxyManager::Instance().ReleaseProxy(proxy_handle2);
        std::cout << "Released proxy2" << std::endl;
    }
    std::cout << "Free mats" << std::endl;
    free(c);
    free(b);
    free(a);
    free(c2);
    free(b2);
    free(a2);
    free(c_t);
    free(b_t);
    free(a_t);
    free(c_t2);
    free(b_t2);
    free(a_t2);
    return 0;
}
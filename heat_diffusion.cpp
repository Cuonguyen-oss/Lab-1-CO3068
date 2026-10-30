
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef N_SIZE
#define N_SIZE 4000
#endif

constexpr long N = N_SIZE;
constexpr long W = N + 2;                
constexpr long TOTAL = W * W;            
constexpr double PAD_VALUE = 30.0;      

constexpr double K[3][3] = {{0.05, 0.10, 0.05},
                            {0.10, 0.40, 0.10},
                            {0.05, 0.10, 0.05}};

static inline long P(long i, long j) { return (i + 1) * W + (j + 1); }

static inline double conv_cell(const double* __restrict in, long p) {
    double s = 0.0;
    for (int u = -1; u <= 1; ++u)
        for (int v = -1; v <= 1; ++v)
            s += K[u + 1][v + 1] * in[p + u * W + v];
    return s;
}


static void conv_span(const double* __restrict in, double* __restrict out, long i, long j0, long j1) {
    const double* __restrict up = in + P(i - 1, 0);
    const double* __restrict mid = in + P(i, 0);
    const double* __restrict dn = in + P(i + 1, 0);
    double* __restrict o = out + P(i, 0);
#pragma omp simd
    for (long j = j0; j < j1; ++j)
        o[j] = 0.40 * mid[j] + 0.10 * (up[j] + dn[j] + mid[j - 1] + mid[j + 1]) +
               0.05 * (up[j - 1] + up[j + 1] + dn[j - 1] + dn[j + 1]);
}


static double* alloc_grid() {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, TOTAL * sizeof(double)) != 0) {
        std::fprintf(stderr, "Out of memory\n");
        std::exit(1);
    }
    return static_cast<double*>(ptr);
}


static void copy_grid(double* dst, const double* src) {
#pragma omp parallel for schedule(static)
    for (long r = 0; r < W; ++r) std::memcpy(dst + r * W, src + r * W, W * sizeof(double));
}


static bool load_csv(const std::string& path, double* grid) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (long i = 0; i < TOTAL; ++i) grid[i] = PAD_VALUE;  // halo = 30

    const char* p = buf.c_str();
    const char* end = p + buf.size();
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            while (p < end && !(std::isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.')) ++p;
            if (p >= end) {
                std::fprintf(stderr, "CSV has fewer than %ld x %ld values\n", N, N);
                return false;
            }
            char* next;
            grid[P(i, j)] = std::strtod(p, &next);
            p = next;
        }
    }
    return true;
}


static double* run_sequential(double* a, double* b, int iters) {
    double* cur = a;
    double* nxt = b;
    for (int it = 0; it < iters; ++it) {
        for (long i = 0; i < N; ++i)
            for (long j = 0; j < N; ++j) nxt[P(i, j)] = conv_cell(cur, P(i, j));
        std::swap(cur, nxt);
    }
    return cur; 
}


static double* run_par_a(double* a, double* b, int iters, int threads) {
    double* cur = a;
    double* nxt = b;
    for (int it = 0; it < iters; ++it) {
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
        for (long i = 0; i < N; ++i)
            for (long j = 0; j < N; ++j) nxt[P(i, j)] = conv_cell(cur, P(i, j));
        std::swap(cur, nxt);
    }
    return cur;
}


static double* run_par_c(double* a, double* b, int iters, int threads) {
#pragma omp parallel num_threads(threads) default(none) shared(a, b, iters)
    {
        double* cur = a;  
        double* nxt = b;
        for (int it = 0; it < iters; ++it) {
#pragma omp for schedule(static)
            for (long i = 0; i < N; ++i) conv_span(cur, nxt, i, 0, N);  
            std::swap(cur, nxt);
        }
    }
    return (iters % 2 == 0) ? a : b;
}


struct Stats {
    double sum, max;
};

static Stats grid_stats(const double* g) {
    double sum = 0.0, mx = -1e300;
#pragma omp parallel for collapse(2) reduction(+ : sum) reduction(max : mx)
    for (long i = 0; i < N; ++i)
        for (long j = 0; j < N; ++j) {
            const double v = g[P(i, j)];
            sum += v;
            mx = std::max(mx, v);
        }
    return {sum, mx};
}

static double max_abs_diff(const double* x, const double* y) {
    double d = 0.0;
#pragma omp parallel for collapse(2) reduction(max : d)
    for (long i = 0; i < N; ++i)
        for (long j = 0; j < N; ++j) d = std::max(d, std::fabs(x[P(i, j)] - y[P(i, j)]));
    return d;
}


static void dump_downsampled(const double* g, const std::string& path, long step) {
    std::ofstream f(path);
    for (long i = 0; i < N; i += step) {
        for (long j = 0; j < N; j += step) f << g[P(i, j)] << (j + step < N ? "," : "");
        f << "\n";
    }
}


using Runner = double* (*)(double*, double*, int, int);

struct Algo {
    const char* name;
    Runner fn;
};

int main(int argc, char** argv) {
    std::string csv = "heat_matrix.csv", out_csv = "results.csv", dump = "";
    int iters = 100, repeat = 1;
    std::vector<int> thread_list;

    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&]() -> std::string { return (a + 1 < argc) ? argv[++a] : ""; };
        if (s == "--csv") csv = next();
        else if (s == "--iters") iters = std::atoi(next().c_str());
        else if (s == "--repeat") repeat = std::max(1, std::atoi(next().c_str()));
        else if (s == "--out") out_csv = next();
        else if (s == "--dump") dump = next();
        else if (s == "--threads") {
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ',')) thread_list.push_back(std::atoi(tok.c_str()));
        } else {
            std::printf("Usage: %s [--csv file] [--iters 100] [--threads 1,2,4,8] [--repeat R]"
                        " [--out results.csv] [--dump final_downsampled.csv]\n", argv[0]);
            return 0;
        }
    }
    if (iters < 1) iters = 1;
    if (thread_list.empty()) {
        const int mx = omp_get_num_procs();
        for (int t = 1; t < mx; t *= 2) thread_list.push_back(t);
        thread_list.push_back(mx);
    }

    std::printf("Grid %ld x %ld (padded %ld x %ld), %d iterations, hardware threads: %d\n",
                N, N, W, W, iters, omp_get_num_procs());

    if (omp_get_num_procs() < 2)
        std::printf("WARNING: only 1 hardware thread visible (VM / container?). "
                    "No parallel speedup is possible - give the machine more CPUs.\n");

    double* init = alloc_grid();
    double* A = alloc_grid();
    double* B = alloc_grid();
    double* ref = alloc_grid();

    double t_load = omp_get_wtime();
    if (!load_csv(csv, init)) {
        std::fprintf(stderr, "Cannot read %s\n", csv.c_str());
        return 1;
    }
    std::printf("Loaded %s in %.2fs\n", csv.c_str(), omp_get_wtime() - t_load);

    const Stats s0 = grid_stats(init);
    std::printf("Initial : center=%.1f  max=%.1f  mean=%.4f\n", init[P(N / 2, N / 2)], s0.max,
                s0.sum / double(N * N));

 
    double t_seq = 1e300;
    for (int r = 0; r < repeat; ++r) {
        copy_grid(A, init);
        copy_grid(B, init);
        const double t0 = omp_get_wtime();
        double* res = run_sequential(A, B, iters);
        const double t1 = omp_get_wtime();
        t_seq = std::min(t_seq, t1 - t0);
        if (r == repeat - 1) copy_grid(ref, res);
    }
    const Stats sf = grid_stats(ref);
    std::printf("Final   : center=%.1f  max=%.1f  mean=%.4f\n", ref[P(N / 2, N / 2)], sf.max,
                sf.sum / double(N * N));
    std::printf("\n%-28s %8s %12s %10s %11s %12s\n", "Algorithm", "Threads", "Time (s)", "Speedup",
                "Efficiency", "MaxAbsErr");
    std::printf("%-28s %8d %12.4f %10.2f %10.1f%% %12.3g\n", "Sequential", 1, t_seq, 1.0, 100.0, 0.0);

    std::ofstream rf(out_csv);
    rf << "algorithm,threads,time_s,speedup,efficiency,max_abs_err\n";
    rf << "Sequential,1," << t_seq << ",1,1,0\n";

    const Algo algos[] = {{"DUMMY collapse+static", run_par_a},
                          {"PRO persistent+simd", run_par_c}};

    bool all_ok = true;
    for (const Algo& al : algos) {
        for (int T : thread_list) {
            double best = 1e300, err = 0.0;
            for (int r = 0; r < repeat; ++r) {
                copy_grid(A, init);
                copy_grid(B, init);
                const double t0 = omp_get_wtime();
                double* res = al.fn(A, B, iters, T);
                const double t1 = omp_get_wtime();
                best = std::min(best, t1 - t0);
                if (r == repeat - 1) {
                    err = max_abs_diff(res, ref);
                    if (!dump.empty() && T == thread_list.back() && al.fn == run_par_c)
                        dump_downsampled(res, dump, 10);
                }
            }
            const double sp = t_seq / best;
            std::printf("%-28s %8d %12.4f %10.2f %10.1f%% %12.3g%s\n", al.name, T, best, sp,
                        100.0 * sp / T, err, err > 1e-3 ? "  <-- MISMATCH" : "");
            rf << al.name << "," << T << "," << best << "," << sp << "," << sp / T << "," << err
               << "\n";
            if (err > 1e-3) all_ok = false;
        }
    }
    std::printf("\nVerification vs sequential: %s\n", all_ok ? "ALL PASSED" : "FAILED");
    std::printf("Results written to %s\n", out_csv.c_str());

    free(init); free(A); free(B); free(ref);
    return all_ok ? 0 : 2;
}

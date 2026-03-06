#include "../thermite.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

// Define the benchmark targets here
#define KERNEL_FUNC sin_cosf_vv
#define SCALAR_FUNC sinf

const M_PI_F = 3.14159265358979323846f;

// Two-level stringification macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// High-resolution monotonic timer for Windows (nanoseconds)
static inline uint64_t get_time_ns(void) {
    static uint64_t freq = 0;
    if (freq == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        freq = f.QuadPart;
    }
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (uint64_t)((count.QuadPart * 1000000000ULL) / freq);
}
#else
#include <time.h>

// High-resolution monotonic timer for POSIX (nanoseconds)
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif

// Xorshift32 algorithm: fast, minimal state, low instruction count
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main() {
    Thermite* vtable = (Thermite*)malloc(sizeof(Thermite));
    thermite_init_vtable(vtable, HighPerformance);

    int res = vtable->disable_denormals();

    printf("Using Thermite backend: %s\n", vtable->name);
    printf("Benchmarking `%s` against `%s` from math.h\n", STR(KERNEL_FUNC), STR(SCALAR_FUNC));

    size_t array_elements = 1024 * 32; // 32K elements, about 128KB, fits in L1 cache
    size_t alloc_size = array_elements * sizeof(float);

    // source_data holds pristine values so we can reset before each benchmark iteration
    float *source_data = (float *)_aligned_malloc(alloc_size, vtable->alignment);
    float *data = (float *)_aligned_malloc(alloc_size, vtable->alignment);
    float *data2 = (float *)_aligned_malloc(alloc_size, vtable->alignment);
    float *data3 = (float *)_aligned_malloc(alloc_size, vtable->alignment); // for sin_cos outputs

    if (!source_data || !data || !data2) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize RNG state and precompute normalization factor
    uint32_t rng_state = 2463534242U; // Arbitrary non-zero seed
    const float norm_factor = 1.0f / 4294967295.0f;

    // Populate the source array
    for(size_t i = 0; i < array_elements; i++) {
        source_data[i] = (float)xorshift32(&rng_state) * norm_factor;
    }

    int warmup_iterations = 10;
    int bench_iterations = 10000;

    printf("\n--- WARMUP (%d iterations) ---\n", warmup_iterations);
    for (int i = 0; i < warmup_iterations; i++) {
        memcpy(data, source_data, alloc_size);
        vtable->KERNEL_FUNC(array_elements, data, data, data3); // in-place operation

        memcpy(data2, source_data, alloc_size);
        for(size_t j = 0; j < array_elements; j++) {
            data2[j] = SCALAR_FUNC(data2[j]);
        }
    }
    printf("Warmup complete. Memory paged, caches hot, symbols resolved.\n");

    printf("\n--- BENCHMARK (%d iterations) ---\n", bench_iterations);

    // Benchmark: Thermite Kernel
    uint64_t thermite_total_ns = 0;
    for (int i = 0; i < bench_iterations; i++) {
        memcpy(data, source_data, alloc_size);

        uint64_t t0 = get_time_ns();
        vtable->KERNEL_FUNC(array_elements, data, data, data3); // in-place operation
        uint64_t t1 = get_time_ns();

        thermite_total_ns += (t1 - t0);
    }

    // Benchmark: Naive Scalar
    uint64_t scalar_total_ns = 0;
    for (int i = 0; i < bench_iterations; i++) {
        memcpy(data2, source_data, alloc_size);

        uint64_t t0 = get_time_ns();
        for(size_t j = 0; j < array_elements; j++) {
            data2[j] = SCALAR_FUNC(data2[j]);
        }
        uint64_t t1 = get_time_ns();

        scalar_total_ns += (t1 - t0);
    }

    // Math / Statistics Output
    double total_elements = (double)array_elements * bench_iterations;

    double thermite_ns_per_elem = (double)thermite_total_ns / total_elements;
    double scalar_ns_per_elem = (double)scalar_total_ns / total_elements;

    double thermite_melems = (total_elements / ((double)thermite_total_ns / 1e9)) / 1e6;
    double scalar_melems = (total_elements / ((double)scalar_total_ns / 1e9)) / 1e6;

    printf("Thermite Kernel : %8.3f ns/elem | %8.2f MElem/s\n", thermite_ns_per_elem, thermite_melems);
    printf("Naive Scalar    : %8.3f ns/elem | %8.2f MElem/s\n", scalar_ns_per_elem, scalar_melems);
    printf("Speedup         : %8.2fx\n", scalar_ns_per_elem / thermite_ns_per_elem);

    printf("\n--- VERIFICATION ---\n");
    // Print the first few elements to verify correctness and defeat dead-code elimination
    for(size_t i = 0; i < 4; i++) {
        printf("Source: %f -> Thermite: %f vs Scalar: %f\n", source_data[i], data[i], data2[i]);
    }
    printf("...\n");

    free(source_data);
    free(data);
    free(data2);

    if(res == SuccessWasEnabled) {
        vtable->enable_denormals();
    }

    free(vtable);

    return 0;
}
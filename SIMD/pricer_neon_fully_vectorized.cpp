//brew install sleef

//g++-15 -Ofast -march=armv8.4-a+simd -mcpu=apple-m4 -flto -funroll-loops \
  -std=c++17 -pipe -fomit-frame-pointer \
  pricer_neon_fully_vectorized.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lsleef -lm -o pricer
// # Run it
// ./pricer_neon_fully_vectorized

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <numeric>
#include <arm_neon.h>
#include <cstring>
// --- 1. INCLUDE THE SLEEF HEADER ---
#include <sleef.h>

/**
 * A simple 4-way parallel SIMD Pseudo-Random Number Generator (PRNG)
 * This is a xoshiro128+ generator, adapted for NEON.
 * It's fast and has good statistical properties.
 */

struct PRNG_NEON
{
    // Four parallel states, one for each "lane" of the SIMD register
    uint32x4_t s0, s1, s2, s3;

    // Seed the generator with 4 different 32-bit seeds
    PRNG_NEON(uint32_t seed)
    {
        // Build four independent 32-bit seeds
        uint32_t a = seed;
        uint32_t b = seed * 1664525u + 1013904223u;
        uint32_t c = seed ^ 0x9e3779b9u;
        uint32_t d = seed * 1103515245u + 12345u;

        // Spread them to create 4 different seeds per lane
        uint32_t s0_arr[4] = {a, a * 15485863u + 1, a * 32452843u + 2, a * 49979687u + 3};
        uint32_t s1_arr[4] = {b, b * 15485863u + 5, b * 32452843u + 6, b * 49979687u + 7};
        uint32_t s2_arr[4] = {c, c * 15485863u + 11, c * 32452843u + 12, c * 49979687u + 13};
        uint32_t s3_arr[4] = {d, d * 15485863u + 17, d * 32452843u + 18, d * 49979687u + 19};

        s0 = vld1q_u32(s0_arr);
        s1 = vld1q_u32(s1_arr);
        s2 = vld1q_u32(s2_arr);
        s3 = vld1q_u32(s3_arr);
    }

    // A helper for bitwise rotation
    inline uint32x4_t rotl(const uint32x4_t x, int k)
    {
        return vorrq_u32(vshlq_n_u32(x, k), vshrq_n_u32(x, 32 - k));
    }

    // Generate 4 parallel random 32-bit integers
    inline uint32x4_t next_u32()
    {
        uint32x4_t const result = rotl(vaddq_u32(s0, s3), 7);
        uint32x4_t const t = vshlq_n_u32(s1, 9);
        s2 = veorq_u32(s2, s0);
        s3 = veorq_u32(s3, s1);
        s1 = veorq_u32(s1, s2);
        s0 = veorq_u32(s0, s3);
        s2 = veorq_u32(s2, t);
        s3 = rotl(s3, 11);
        return result;
    }

    // Generate 4 parallel floats between [0, 1)
    inline float32x4_t next_float()
    {
        // Get 4 random 32-bit ints
        uint32x4_t u = next_u32();

        // Convert them to floats in [0, 1)
        // This is a standard bit-magic trick
        // 1. Mask to get 23 bits of randomness
        // 2. Add the bits for '1.0' (which is 0x3F800000)
        // 3. Subtract 1.0 to get the range [0, 1)
        float32x4_t f = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(u, vdupq_n_u32(0x007FFFFF)), vdupq_n_u32(0x3F800000)));
        f = vsubq_f32(f, vdupq_n_f32(1.0f));
        uint32x4_t zeros_mask = vceqq_f32(f, vdupq_n_f32(0.0f));
        float32x4_t tiny = vdupq_n_f32(1e-7f);
        f = vbslq_f32(zeros_mask, tiny, f);
        return f;
    }

    

    // Generate 4 parallel standard normal (Z) values
    // Using the (fast) Ziggurat approximation method
    // This is complex, so for this project, we'll use a simpler
    // Box-Muller transform, which requires 2 uniform numbers (u1, u2)
    // Z1 = sqrt(-2 * log(u1)) * cos(2 * PI * u2)
    // Z2 = sqrt(-2 * log(u1)) * sin(2 * PI * u2)
    // To keep it 4-wide, we'll just do 2 Box-Mullers
    inline float32x4_t next_normal()
    {
        const float32x4_t v_pi = vdupq_n_f32(3.1415926535f);
        const float32x4_t v_two_pi = vmulq_f32(v_pi, vdupq_n_f32(2.0f));
        const float32x4_t v_minus_two = vdupq_n_f32(-2.0f);

        // Generate 4 uniform random floats [0, 1)
        float32x4_t u1 = next_float();
        float32x4_t u2 = next_float();

        // --- Use SLEEF to get log and sin/cos ---
        // v_log_u1 = [log(u1[0]), log(u1[1]), log(u1[2]), log(u1[3])]
        const float32x4_t eps = vdupq_n_f32(1e-7f);
        u1 = vmaxq_f32(u1, eps);   // ensure u1 >= eps
        float32x4_t v_log_u1 = Sleef_logf4_u35(u1);
        float32x4_t v_rad = vmulq_f32(u2, v_two_pi);

        // sleef_sincosf4_u35 computes sin and cos at the same time!
        // Sleef_float4_2 v_sincos = Sleef_sincosf4_u35(v_rad);
        Sleef_float32x4_t_2 v_sincos = Sleef_sincosf4_u35(v_rad);
        // v_sincos.x holds the 4 sin results
        // v_sincos.y holds the 4 cos results

        // Common term: sqrt(-2 * log(u1))
        float32x4_t common = vsqrtq_f32(vmulq_f32(v_minus_two, v_log_u1));

        // We now have 8 normal numbers (4 from cos, 4 from sin)
        // We'll just use the 4 from the 'cos' result
        // Z = common * cos(rad)
        return vmulq_f32(common, v_sincos.y);
    }
};

void run_simulation_thread(
    int N_Thread,
    int thread_id,
    float S0, float K, float r, float sigma, float T,
    float &output_thread)
{
    // --- 2. SEED OUR NEW PRNG ---
    // Use std::random_device for a high-quality, non-deterministic seed
    PRNG_NEON rng(std::random_device{}() + thread_id);

    // Pre-calculate constants
    const float drift = (r - 0.5f * sigma * sigma) * T;
    const float vol_sqrt_T = sigma * std::sqrt(T);

    // Pre-load constants into NEON registers
    const float32x4_t v_S0 = vdupq_n_f32(S0);
    const float32x4_t v_K = vdupq_n_f32(K);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_drift = vdupq_n_f32(drift);
    const float32x4_t v_vol_sqrt_T = vdupq_n_f32(vol_sqrt_T);

    // Vector register to sum payoffs
    float32x4_t v_total_payoff = vdupq_n_f32(0.0f);

    // Process 4 simulations at a time
    int i = N_Thread;
    while (i > 3)
    {
        i -= 4;

        // --- 3. FULLY VECTORIZED CALCULATION ---

        // 1. Get 4 parallel Normal (Z) numbers
        float32x4_t v_Z = rng.next_normal();

        // 2. Calculate 4 parallel ST values
        // ST = S0 * exp(drift + vol_sqrt_T * Z)

        // term = drift + vol_sqrt_T * Z
        float32x4_t v_term = vmlaq_f32(v_drift, v_vol_sqrt_T, v_Z);

        // v_exp = exp(term) --- Using SLEEF
        float32x4_t v_exp = Sleef_expf4_u10(v_term);

        // v_ST = S0 * v_exp
        float32x4_t v_ST = vmulq_f32(v_S0, v_exp);

        // 3. Calculate 4 parallel payoffs
        // payoff = max(0, ST - K)
        float32x4_t v_payoff = vsubq_f32(v_ST, v_K);
        v_payoff = vmaxq_f32(v_zero, v_payoff);

        // 4. Add to total
        v_total_payoff = vaddq_f32(v_total_payoff, v_payoff);
    }

    // Sum the 4 lanes of the total payoff vector
    float total_thread_output = vaddvq_f32(v_total_payoff);

    // Process any remaining simulations (if N isn't divisible by 4)
    while (i > 0)
    {
        i--;
        // Use the scalar logic from pricer_neon.cpp
        // (Note: we could get 1 scalar number from rng, but this is simpler)
        std::normal_distribution<float> scalar_dist(0.0f, 1.0f);
        std::mt19937 scalar_gen(std::random_device{}() + i);
        float Z = scalar_dist(scalar_gen);
        float ST = S0 * std::exp(drift + vol_sqrt_T * Z);
        total_thread_output += std::max(0.0f, ST - K);
    }

    output_thread = total_thread_output;
}

// --- The main() function is IDENTICAL to pricer_neon.cpp ---
// (I've increased N to 100 million to see the speedup)
int main()
{
    const float S0 = 100.0f;
    const float K = 105.0f;
    const float r = 0.05f;
    const float sigma = 0.2f;
    const float T = 1.0f;
    const int N = 10e6; // 10million

    const int no_of_Avaliable_threads = std::thread::hardware_concurrency();
    const int num_pre_threads = N / no_of_Avaliable_threads;

    std::vector<std::thread> threads(no_of_Avaliable_threads);
    std::vector<float> partial_outputs(no_of_Avaliable_threads);

    std::cout << "--- Fully Vectorized (NEON + SLEEF) Monte Carlo ---" << std::endl;
    std::cout << "Total Simulations: " << N << std::endl;
    std::cout << "Using " << no_of_Avaliable_threads << " threads" << std::endl;
    std::cout << "Sims per thread:   " << num_pre_threads << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < no_of_Avaliable_threads; i++)
    {
        threads[i] = std::thread(run_simulation_thread, num_pre_threads, i, S0, K, r, sigma, T, std::ref(partial_outputs[i]));
    }

    for (std::thread &it : threads)
    {
        it.join();
    }

    float total_payoff = std::accumulate(partial_outputs.begin(), partial_outputs.end(), 0.0f);
    float mean_payoff = total_payoff / N;
    float option_price = mean_payoff * std::exp(-r * T);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Option Price:      " << option_price << std::endl;
    std::cout << "Time Elapsed:    " << duration.count() << " seconds" << std::endl;

    return 0;
}
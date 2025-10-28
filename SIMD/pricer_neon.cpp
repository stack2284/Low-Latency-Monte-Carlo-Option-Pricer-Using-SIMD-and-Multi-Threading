#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <random>
#include <array>
#include <numeric>

// Neon Mac M4 architecture
// 1. Include the ARM NEON header
#include <arm_neon.h>

void run_simulation_thread(

    int N_thread,
    int thread_id,
    float S0,
    float K,
    float r,
    float sigma,
    float T,
    float &output_thread)
{

    std ::mt19937 generator(std ::random_device{}() + thread_id);
    std ::normal_distribution<float> distribution_normal(0.0, 1.0);
    float total_thread_output = 0.0;

    // --- NEON ---
    // Precalculate scalar constants

    const float drift = (r - 0.5 * sigma * sigma) * T;
    const float vol_sqrt_t = sigma * std ::sqrt(T);

    // --- NEON ---
    // Preloading constants in vectors
    const float32x4_t v_K = vdupq_n_f32(K);
    // v_K = {k , k , k ,k }
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    // v_zero = {0. ,0 , 0 , 0 }

    int sim_cnt = N_thread;
    float ST_array[4];

    while (sim_cnt > 3)
    {
        sim_cnt -= 4;

        // ---SCALAR PART---
        //  We do this one-by-one because std::exp is scalar
        for (int i = 0; i < 4; i++)
        {
            float Z = distribution_normal(generator);
            float st = S0 * std ::exp((drift) + Z * vol_sqrt_t);
            ST_array[i] = st;
        }
        // VECTOR PART
        // Actual part where we use Simd

        float32x4_t v_ST = vld1q_f32(ST_array);
        float32x4_t v_payoff = vsubq_f32(v_ST, v_K);
        v_payoff = vmaxq_f32(v_zero, v_payoff);
        total_thread_output += vaddvq_f32(v_payoff);
    }
    while (sim_cnt > 0)
    {
        sim_cnt--;
        float Z = distribution_normal(generator);
        float ST = S0 * std::exp(drift + vol_sqrt_t * Z);
        total_thread_output += std::max(0.0f, ST - K);
    }
    output_thread = total_thread_output;

    return;
}

int main()
{
    // --- Switched to float and more simulations ---
    // neom uses 128 bit arch so double can use vector of size 2 while float uses 4
    const float S0 = 100.0f;
    const float K = 105.0f;
    const float r = 0.05f;
    const float sigma = 0.2f;
    const float T = 1.0f;
    const int N = 10e6;

    // ---hardware specs and threads---
    const int no_of_Avaliable_threads = std::thread::hardware_concurrency();
    const int num_pre_threads = N / no_of_Avaliable_threads;
    std::vector<std::thread> threads(no_of_Avaliable_threads);
    std::vector<float> partial_outputs(no_of_Avaliable_threads); // Changed to float

    std::cout << "--- Multi-Threaded + NEON SIMD Monte Carlo ---" << std::endl;
    std::cout << "Total Simulations: " << N << std::endl;
    std::cout << "Using " << no_of_Avaliable_threads << " threads" << std::endl;
    std::cout << "Sims per thread:   " << num_pre_threads << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < no_of_Avaliable_threads; i++)
    {
        threads[i] = std::thread(
            run_simulation_thread, 
            num_pre_threads, 
            i, 
            S0, 
            K, 
            r, 
            sigma, 
            T, 
            std::ref(partial_outputs[i])
        );
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
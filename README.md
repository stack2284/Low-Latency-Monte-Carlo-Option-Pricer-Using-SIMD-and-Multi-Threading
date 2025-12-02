# Low-Latency Monte Carlo Option Pricer

This project demonstrates the progressive optimization of a C++ Monte Carlo simulation for pricing a European call option. The goal is to show a "layers of optimization" approach, moving from a naive single-threaded baseline to a fully parallelized and vectorized (SIMD) implementation.

Preformance critical implementation acheiving upto 28x effiecny using sleef and 16x normally. 

---

##  Core Concepts Demonstrated

* **C++ Programming:** Clean, modern C++ (`std::c++17`).
* **Financial Modeling:** Implementation of the Black-Scholes model via Monte Carlo simulation.
* **Concurrency:** Parallelizing work across multiple CPU cores using `std::thread`.
* **CPU-Level Optimization (SIMD):** Using instruction-level parallelism to perform calculations on multiple data points at once.
* **ARM NEON Intrinsics:** Writing low-level CPU-specific code for an Apple M4 processor (ARMv9 architecture).

---

##  Project Structure

The project is broken into three distinct stages of optimization:

* **`/base`**:
    * `pricer.cpp`: The naive, single-threaded baseline implementation. It uses 64-bit `double`s and a simple `for` loop.
* **`/threaded`**:
    * `pricer_threaded.cpp`: The first optimization. This version divides the simulation work among all available CPU cores using `std::thread`. It still uses 64-bit `double`s.
* **`/SIMD`**:
    * `pricer_neon.cpp`: This version uses both multi-threading *and* ARM NEON SIMD intrinsics. To maximize SIMD throughput, the calculations are switched from 64-bit `double`s to 32-bit `float`s, allowing 4 paths to be processed per instruction instead of 2.
    * `pricer_neon_fully_vectorized.cpp`: This version uses sleef library and xoshiro128+ generator. 
---

## How to Build and Run

All versions should be compiled with full optimizations (`-O3`) to ensure a fair performance comparison.

### 1. Baseline (Single-Thread)

```bash
# Navigate to the base directory
cd base

# Compile with optimizations
# (Note: g++-15 was used in the test, your compiler name may vary)
g++ -O3 -std=c++17 -g pricer.cpp -o pricer

# Run the benchmark
./pricer
```

###2. Multi-Threaded

```bash
# Navigate to the threaded directory
cd threaded

# Compile with optimizations
g++ -O3 -std=c++17 -g pricer_threaded.cpp -o pricer_threaded

# Run the benchmark
./pricer_threaded
```

###3. Multi-Threaded + NEON (SIMD)
```bash
# Navigate to the SIMD directory
cd SIMD

# Compile with optimizations
g++ -O3 -std=c++17 -g pricer_neon.cpp -o pricer_neon

# Run the benchmark
./pricer_neon
```

###4. Multi-Threaded + NEON + sleef (SIMD)

```bash
cd SIMD
# please make sure here you have the right flags set 
g++-15 -Ofast -march=armv8.4-a+simd -mcpu=apple-m1 -flto -funroll-loops \
  -std=c++17 -pipe -fomit-frame-pointer \
  pricer_neon_fully_vectorized.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lsleef -lm -o pricer
./pricer_neon_fully_vectorized

```

## Performance Results

These benchmarks were run on a **MacBook Pro (Apple M4, 14 Cores)**, comparing the time to complete **10,000,000 simulations**.

| Version | Time Elapsed (s) | Speedup (vs. Baseline) |
| :--- | :--- | :--- |
| **1. Baseline (Single-Thread, `double`)** | `0.205796 s` | `1.00x` |
| **2. Multi-Threaded (14 Cores, `double`)** | `~0.03056 s` | `~6.73x` |
| **3. Threaded + NEON (14 Cores, `float`)** | `0.012407 s` | **`16.60x`** |
| **4. Threaded + sleef + NEON (14 Cores, `float`)** | `0.00719 s` | **`28.60x`** |

> **Note:** The 'Multi-Threaded' time is an estimate. The original test was run with 1M simulations (`0.003056s`) and the result was scaled by 10 to provide a fair comparison against the 10M-simulation benchmarks.

---

## Analysis of Speedup

* **Step 1 (Multi-Threading):** The **~6.7x speedup** comes from leveraging **thread-level parallelism**. The work is divided among 14 CPU cores, allowing them to run in parallel. The speedup is not a perfect 14x due to thread creation/management overhead and other system factors.

* **Step 2 (SIMD + Data Type):** The final leap to a **16.60x total speedup** (an additional ~2.5x over the threaded version) comes from two related optimizations:
    1.  **Data Type:** By switching from 64-bit `double`s to 32-bit `float`s, we halve the memory footprint for the calculations. This allows the 128-bit NEON registers to pack 4 values at once (`float32x4_t`) instead of just 2 (`float64x2_t`).
    2.  **Instruction-Level Parallelism:** Using NEON intrinsics (like `vld1q_f32`, `vsubq_f32`, `vmaxq_f32`, and `vaddvq_f32`) allows each thread to perform a single operation (like `max(0, payoff)`) on 4 separate simulation paths *at the same time*.

This project successfully demonstrates a 28.6x performance improvement by applying both concurrency and CPU-specific vectorization.

# --- Variables ---
# Define the C++ compiler. 
CXX = g++-15

# Define our compilation flags
# -O3 = High optimizations
# -std=c++17 = C++17 standard
# -g = Include debug symbols
CXXFLAGS = -O3 -std=c++17 -g

# --- Targets ---

# The 'all' target is the default.
# Type 'make all' or just 'make' to run this.
all: base/pricer threaded/pricer_threaded SIMD/pricer_neon

# Rule to build the 'base' executable
base/pricer: base/pricer.cpp
	$(CXX) $(CXXFLAGS) base/pricer.cpp -o base/pricer

# Rule to build the 'threaded' executable
threaded/pricer_threaded: threaded/pricer_threaded.cpp
	$(CXX) $(CXXFLAGS) threaded/pricer_threaded.cpp -o threaded/pricer_threaded

# Rule to build the 'SIMD' executable
SIMD/pricer_neon: SIMD/pricer_neon.cpp
	$(CXX) $(CXXFLAGS) SIMD/pricer_neon.cpp -o SIMD/pricer_neon

# The 'clean' target.
# Type 'make clean' to remove all built executables.
clean:
	rm -f base/pricer
	rm -f threaded/pricer_threaded
	rm -f SIMD/pricer_neon

# Phony targets aren't files. 'all' and 'clean' are just commands.
.PHONY: all clean
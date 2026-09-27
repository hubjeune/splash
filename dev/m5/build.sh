#!/bin/sh
# splash-m5 tools. Run from the repository root after `make`.
set -eu
mkdir -p build/m5
xcrun -sdk macosx clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev -Ibuild/engine \
  -mmacosx-version-min=26.4 -fobjc-arc dev/m5/kernel_bench.mm build/engine/libsplash.a \
  -framework Foundation -framework Metal -framework IOKit -o build/m5/kernel-bench
echo "built build/m5/kernel-bench"

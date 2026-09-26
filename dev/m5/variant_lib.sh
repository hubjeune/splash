#!/bin/sh
# splash-m5: build a variant metallib with extra Metal defines, recompiling only
# the named kernel sources and reusing build/metal's other production objects.
#   dev/m5/variant_lib.sh OUT.metallib "decode/attention_q8 prefill/attention_q8" -DNAME=VALUE ...
set -eu
out=$1; sources=$2; shift 2
dir=$(mktemp -d)
airs=$(find build/metal -name '*.air' | sort)
for src in $sources; do
  mkdir -p "$dir/$(dirname "$src")"
  xcrun -sdk macosx metal -std=metal4.0 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=26.4 \
    "$@" -c "runtime/metal/kernels/$src.metal" -o "$dir/$src.air"
  airs=$(printf '%s\n' $airs | grep -v "build/metal/$src.air\$")
  airs="$dir/$src.air $airs"  # first: metallib ignored a variant listed last
done
xcrun -sdk macosx metallib $airs -o "$out"
rm -rf "$dir"
echo "built $out"

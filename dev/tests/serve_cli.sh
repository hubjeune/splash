#!/bin/sh
# Check the serve-native parser without a model package or a Metal device.
# Every case here is rejected before model inspection, so no package or
# metallib is needed; the parser is the production binary's own, as the
# tuning CLI tests use theirs.
set -eu
binary=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-serve-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

# args: expected message, then trailing arguments after the fixed ones. A
# zero cache-disk quota is prepended so the first variable argument is the
# option under test, as the launcher emits it.
reject() {
    expected=$1
    shift
    status=0
    "$binary" serve-native "$work/target" "$work/draft" auto auto 0 "$@" \
        >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 64 ] || ! grep -Fq -- "$expected" "$work/err"; then
        echo "serve-native accepted or misclassified arguments: $*" >&2
        cat "$work/err" >&2
        exit 1
    fi
    test ! -s "$work/out"
}

# An out-of-range or non-integer duty cycle is rejected and fails closed.
bounds='--power must be an integer in [1, 100]'
for value in 0 101 -1 50.0 1e2 50x '' ' 50' '50 '; do
    reject "$bounds" --power "$value"
done
# A trailing option with no value or an unknown value is an argument error.
trailing='expected --kv-format int8|bf16 or --power 1..100'
reject "$trailing" --power
reject "$trailing" --kv-format
reject "$bounds" --power --kv-format
reject "$trailing" --sentinel 1
reject '--power may be given once' --power 25 --power 30
reject '--kv-format may be given once' --kv-format int8 --kv-format int8
reject '--kv-format requires int8 or bf16' --kv-format fp16
# A valid option parses, and an unknown trailing option proves the valid ones
# before it parsed, in either order and together.
reject "$trailing" --power 25 --sentinel
reject "$trailing" --power 25 --kv-format int8 --sentinel
reject "$trailing" --kv-format int8 --power 25 --sentinel
reject "$trailing" --kv-format bf16 --power 100 --sentinel
reject "$trailing" --power 25 --kv-format bf16 --sentinel
# Every valid duty cycle passes the option parser and fails at model
# resolution, which is as far as this test can go without a real package.
missing='TARGET_DIRECTORY must name an existing directory'
reject "$missing" --power 1
reject "$missing" --power 25
reject "$missing" --power 100
reject "$missing" --power 25 --kv-format bf16
reject "$missing" --kv-format bf16 --power 25
# The quota still parses ahead of either option.
status=0
"$binary" serve-native "$work/target" "$work/draft" auto auto 5G \
    >"$work/out" 2>"$work/err" || status=$?
if [ "$status" -ne 64 ] || ! grep -Fq -- 'MAX_CACHE_DISK_BYTES must be a nonnegative integer' "$work/err"; then
    echo "serve-native misclassified or accepted a size quota" >&2
    cat "$work/err" >&2
    exit 1
fi
echo 'serve-native CLI validation: PASS'

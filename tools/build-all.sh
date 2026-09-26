#!/bin/bash
# Build every example. Each pNN-* dir contains a standalone main.cpp
# that links against the shared vkmini.hpp helper.
set -u
cd "$(dirname "$0")/.."
ok=0; fail=0
for d in p*/; do
    [ -f "$d/main.cpp" ] || continue
    libs="-lvulkan -lglfw -lshaderc"
    grep -q "png.h" "$d/main.cpp" && libs="$libs -lpng"
    grep -q "<thread>" "$d/main.cpp" && libs="$libs -pthread"
    if g++ -std=c++23 -O2 "$d/main.cpp" -o "$d/app" $libs 2>"$d/.build.log"; then
        ok=$((ok+1))
    else
        fail=$((fail+1)); echo "FAIL: $d (see $d/.build.log)"
    fi
done
echo "built $ok, failed $fail"

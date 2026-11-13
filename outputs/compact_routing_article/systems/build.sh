#!/bin/bash
set -euo pipefail
systems_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_dir="$(cd "$systems_dir/../../.." && pwd)"
cd "$systems_dir"
clang++ -O3 -ffast-math -std=c++17 -DACCELERATE_NEW_LAPACK -dynamiclib \
  -framework Accelerate bucket_tail.cpp -o bucket_tail.dylib
clang++ -O3 -ffast-math -std=c++17 -DACCELERATE_NEW_LAPACK -dynamiclib \
  -framework Accelerate "$workspace_dir/outputs/compact_routing_followup/compiled_tail.cpp" \
  -o compiled_reference.dylib
clang++ -O3 -ffast-math -Wno-deprecated-declarations -std=c++17 -dynamiclib \
  -framework Accelerate quant_dense.cpp -o quant_dense.dylib

#!/usr/bin/env bash
# Build and test vsq on linux/amd64 (review item 3.1).
#
#   docker/test_amd64.sh     run C++ + Python tests twice:
#                              1) native container CPU (runtime dispatch check:
#                                 must fall back to scalar without SIGILL when
#                                 the CPU has no AVX2)
#                              2) qemu-x86_64 -cpu max (AVX2/FMA/F16C emulated:
#                                 the AVX2 kernels must be selected and agree
#                                 with scalar)
#
# On Apple Silicon the container itself is emulated and exposes no AVX2, hence
# run 2. On a real x86 host with AVX2 both runs use AVX2.
#
# qemu-user must be >= 8: qemu 7.2 (Debian 12) ignores VSIB index register
# ymm4 in AVX2 gathers (SIB index field 100 is decoded as "no index"), so every
# lane loads base[0]. NumPy's AVX2 argsort/argpartition (x86-simd-sort) uses
# such gathers and segfaults under it. A probe below rejects a broken qemu.
#
# The toolchain image is created once from debian:13 with `docker run` +
# `docker commit` (no `docker build`, which needs registry metadata access) and
# reused. The wheel is built with VSQ_PORTABLE=ON (x86-64 baseline), the
# configuration of the published wheels, so AVX2 is reached only via CPUID.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${TQ_BASE_IMAGE:-debian:13}"
IMAGE="vsq-amd64-test:${BASE//[:\/]/-}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "== creating toolchain image $IMAGE from $BASE"
  docker rm -f tq-amd64-setup >/dev/null 2>&1 || true
  docker run --platform linux/amd64 --name tq-amd64-setup "$BASE" bash -euo pipefail -c '
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      build-essential cmake ninja-build qemu-user python3 python3-dev python3-venv \
      binutils ca-certificates >/dev/null
    python3 -m venv /opt/venv
    /opt/venv/bin/pip install -q "scikit-build-core>=0.10" "pybind11>=2.13" numpy scipy pytest
    rm -rf /var/lib/apt/lists/*'
  docker commit tq-amd64-setup "$IMAGE" >/dev/null
  docker rm tq-amd64-setup >/dev/null
fi

docker run --rm --platform linux/amd64 -v "$ROOT:/src:ro" "$IMAGE" bash -euo pipefail -c '
  export PATH=/opt/venv/bin:$PATH
  mkdir -p /work/src
  tar -C /src --exclude=.venv --exclude=.git --exclude=build --exclude=build-tests \
      --exclude="*.so" --exclude=__pycache__ --exclude=origin_tmp -cf - . | tar -C /work/src -xf -
  cd /work/src

  echo "== container CPU flags"
  grep -o -m1 -w avx2 /proc/cpuinfo || echo "no avx2 in /proc/cpuinfo"
  qemu-x86_64 --version | head -1

  echo "== build wheel (portable x86-64 baseline)"
  pip install --no-build-isolation --no-deps -q . --config-settings=cmake.define.VSQ_PORTABLE=ON

  echo "== C++ core test (x86-64 baseline, no -mavx2, -Werror)"
  g++ -std=c++17 -O2 -march=x86-64 -Wall -Wextra -Werror -ffast-math -fno-finite-math-only \
      -Iinclude tests/cpp/test_core.cpp -o /tmp/test_core
  echo "vpermps instructions in binary: $(objdump -d /tmp/test_core | grep -c vpermps)"

  cd /tmp   # import the installed wheel, not the source tree
  echo "== run 1: native container CPU"
  /tmp/test_core
  python -c "import vsq; print(\"detected_isa:\", vsq.detected_isa())"
  python -m pytest -q /work/src/python/tests -p no:cacheprovider

  echo "== run 2: qemu-x86_64 -cpu max (AVX2 emulated)"
  # Preflight: vpgatherdd with index ymm4 must use the index (see header).
  cat > /tmp/vsib4.c <<"EOC"
#include <stdio.h>
int main(void) {
  static const int t[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  static const int idx[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  int out[8];
  __asm__ volatile("vmovdqu (%1), %%ymm4\n\t"
                   "vpcmpeqd %%ymm0, %%ymm0, %%ymm0\n\t"
                   "vpxor %%ymm1, %%ymm1, %%ymm1\n\t"
                   "vpgatherdd %%ymm0, (%2,%%ymm4,4), %%ymm1\n\t"
                   "vmovdqu %%ymm1, (%0)\n\t"
                   :: "r"(out), "r"(idx), "r"(t) : "memory", "xmm0", "xmm1", "xmm4");
  for (int i = 0; i < 8; i++)
    if (out[i] != t[idx[i]]) { printf("lane %d: got %d want %d\n", i, out[i], t[idx[i]]); return 1; }
  return 0;
}
EOC
  gcc -O1 /tmp/vsib4.c -o /tmp/vsib4
  qemu-x86_64 -cpu max /tmp/vsib4 __argv_placeholder__ || {
    echo "qemu-user AVX2 gather is broken (VSIB index ymm4); use qemu >= 8 (TQ_BASE_IMAGE=debian:13)" >&2
    exit 1; }
  qemu-x86_64 -cpu max /tmp/test_core
  # Under nested emulation (Docker binfmt + qemu-user) the first argument
  # after the program is dropped, so pass a placeholder; the venv is not
  # detected through qemu, so its site-packages go on PYTHONPATH.
  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  qpy() { PYTHONPATH="$SITE" qemu-x86_64 -cpu max "$(readlink -f "$(command -v python)")" __argv_placeholder__ "$@"; }
  qpy -c "import vsq; print(\"detected_isa:\", vsq.detected_isa())"
  qpy -m pytest -q /work/src/python/tests -p no:cacheprovider
'

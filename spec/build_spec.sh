#!/usr/bin/env bash
# Build the selected SPEC CPU 2026 benchmarks with memory-trace instrumentation
# and set up their run directories.
#   SPEC_DIR=~/spec/cpu2026 spec/build_spec.sh [size] [benchmarks...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SPEC_DIR=${SPEC_DIR:-$HOME/spec/cpu2026}
SIZE=${1:-refrate}; shift || true
BENCHES=${*:-$(cat "$ROOT/spec/benchmarks.txt" | grep -v '^#' | xargs)}
make -C "$ROOT/tracer" >/dev/null
cp "$ROOT/spec/memtrace-llvm.cfg" "$SPEC_DIR/config/memtrace-llvm.cfg"
# The vendor toolset links against libcrypt.so.1; modern distributions ship only
# libcrypt.so.2, which exports the same XCRYPT_2.0 symbols.
COMPAT="$ROOT/spec/.compat-lib"; mkdir -p "$COMPAT"
[ -e /usr/lib/libcrypt.so.1 ] || ln -sf /usr/lib/libcrypt.so.2 "$COMPAT/libcrypt.so.1"
export LD_LIBRARY_PATH="$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SPEC_DIR"
set +u; source ./shrc; set -u
runcpu --config=memtrace-llvm --define tracer="$ROOT/tracer/build" --action=build ${REBUILD:+--rebuild} $BENCHES
runcpu --config=memtrace-llvm --define tracer="$ROOT/tracer/build" --action=setup --size="$SIZE" $BENCHES

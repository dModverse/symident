#!/bin/sh
# Builds msolve with static GMP, MPFR and FLINT into
# $SYMIDENT_LIBS_CACHE/deps-msolve-$SYMIDENT_MSOLVE_VERSION; prints the prefix.
set -eu

CACHE=${SYMIDENT_LIBS_CACHE:?}
MSOLVE=${SYMIDENT_MSOLVE_VERSION:-0.10.1}
GMP=${SYMIDENT_GMP_VERSION:-6.3.0}
MPFR=${SYMIDENT_MPFR_VERSION:-4.2.1}
FLINT=${SYMIDENT_FLINT_VERSION:-3.3.1}
JOBS=${SYMIDENT_JOBS:-2}
# GMP 6.3 does not configure under C23
CC="${CC:-cc} -std=gnu17"
export CC

PREFIX="$CACHE/deps-msolve-$MSOLVE"
WORK="$CACHE/build-msolve-$MSOLVE"
rm -rf "$WORK" "$PREFIX/.msolve-complete"
mkdir -p "$WORK" "$PREFIX/bin"
cd "$WORK"

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"
  else wget -q -O "$2" "$1"; fi
}
step() { echo "==> $*" >&2; }

step "GMP $GMP"
fetch "https://ftp.gnu.org/gnu/gmp/gmp-$GMP.tar.xz" gmp.tar.xz
tar xf gmp.tar.xz
(cd "gmp-$GMP" && ./configure --prefix="$PREFIX" --disable-shared --enable-static \
   --with-pic >&2 && make -j"$JOBS" >&2 && make install >&2)

step "MPFR $MPFR"
fetch "https://ftp.gnu.org/gnu/mpfr/mpfr-$MPFR.tar.xz" mpfr.tar.xz
tar xf mpfr.tar.xz
(cd "mpfr-$MPFR" && ./configure --prefix="$PREFIX" --disable-shared --enable-static \
   --with-pic --with-gmp="$PREFIX" >&2 && make -j"$JOBS" >&2 && make install >&2)

step "FLINT $FLINT"
fetch "https://github.com/flintlib/flint/releases/download/v$FLINT/flint-$FLINT.tar.gz" flint.tar.gz
tar xzf flint.tar.gz
(cd "flint-$FLINT" && ./configure --prefix="$PREFIX" --disable-shared --enable-static \
   --with-pic --with-gmp="$PREFIX" --with-mpfr="$PREFIX" >&2 && make -j"$JOBS" >&2 &&
 make install >&2)

step "msolve $MSOLVE"
fetch "https://github.com/algebraic-solving/msolve/archive/refs/tags/v$MSOLVE.tar.gz" msolve.tar.gz
tar xzf msolve.tar.gz
cd "msolve-$MSOLVE"
cat > config.h <<EOF
#define VERSION "$MSOLVE"
#define HAVE_FLINT_NMOD_H 1
#define HAVE_TIMESPEC_GET 1
EOF
mkdir -p ompstub obj
cat > ompstub/omp.h <<'EOF'
#ifndef SYMIDENT_OMP_STUB_H
#define SYMIDENT_OMP_STUB_H
#include <time.h>
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline double omp_get_wtime(void) {
  struct timespec ts; timespec_get(&ts, TIME_UTC); return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif
EOF
# one translation unit per library; main.c includes libmsolve.c
for u in src/usolve/usolve.c src/fglm/fglm_core.c src/neogb/gb.c src/msolve/main.c; do
  $CC -O2 -DHAVE_CONFIG_H -I. -Isrc -Iompstub -I"$PREFIX/include" -w \
    -c "$u" -o "obj/$(basename "$u" .c).o"
done
$CC obj/main.o obj/gb.o obj/fglm_core.o obj/usolve.o -o "$PREFIX/bin/msolve" \
  "$PREFIX/lib/libflint.a" "$PREFIX/lib/libmpfr.a" "$PREFIX/lib/libgmp.a" -lm
strip "$PREFIX/bin/msolve" 2>/dev/null || true

step "check"
printf 'x, y\n65521\nx^2+y-7,\nx*y-6\n' > check.ms
"$PREFIX/bin/msolve" -f check.ms -o check.out -P 1 >&2
grep -q "65521" check.out

# keep the binary only; the marker flags a complete build
cd "$CACHE"
rm -rf "$WORK" "$PREFIX/include" "$PREFIX/lib" "$PREFIX/share"
touch "$PREFIX/.msolve-complete"
echo "$PREFIX"

// Exact GF(p) kernel for the observability analytic path of symmetryDetection().
// Builds the observability-identifiability matrix by a Taylor-mode construction
// over a finite field (Lie derivatives as truncated power series in time,
// parameter gradients as forward-mode dual numbers), reduces it, and supports
// the rational reconstruction used by the R orchestration. Primes are < 2^31 so
// residue products stay below 2^62 in uint64; the CRT product of four primes
// stays below 2^124 in unsigned __int128 (symRatRecon only; compilers without
// __int128 leave it out and the Python side reconstructs with arbitrary precision).

#include "kernel.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

using namespace Rcpp;

typedef uint64_t u64;
#if defined(__SIZEOF_INT128__) && !defined(SYMIDENT_NO_INT128)
#define SYMIDENT_HAVE_INT128 1
// __int128 (GCC/Clang extension) for the CRT of symRatRecon; __extension__ keeps
// -Wpedantic quiet without suppressing it globally.
__extension__ typedef __int128 i128;
__extension__ typedef unsigned __int128 u128;
#endif

// High 64 bits of the 128-bit product a*b.
inline u64 mulhi64(u64 a, u64 b) {
#if defined(SYMIDENT_HAVE_INT128)
  return (u64)(((u128)a * b) >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
  return __umulh(a, b);
#else
  u64 al = a & 0xffffffffULL, ah = a >> 32, bl = b & 0xffffffffULL, bh = b >> 32;
  u64 ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  u64 mid = (ll >> 32) + (lh & 0xffffffffULL) + (hl & 0xffffffffULL);
  return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
}

// floor(2^64 / p) for p > 1 not a power of two.
inline u64 barrett_m(u64 p) { return UINT64_MAX / p; }

namespace {

const int OP_CONST = 0, OP_ADD = 1, OP_MUL = 2, OP_INV = 3, OP_SQRT = 4, OP_ROOT = 5;

inline u64 addmod(u64 a, u64 b, u64 p) { u64 s = a + b; return s >= p ? s - p : s; }
inline u64 submod(u64 a, u64 b, u64 p) { return a >= b ? a - b : a + p - b; }
// Barrett reduction of any 64-bit x modulo a fixed p < 2^31: one 64x64->128 multiply
// instead of a division (m = floor(2^64 / p); the quotient estimate is low by at
// most 2, corrected by the subtractions).
struct Barrett {
  u64 p, m;
  explicit Barrett(u64 p_) : p(p_), m(barrett_m(p_)) {}
  inline u64 red(u64 x) const {
    u64 q = mulhi64(x, m);
    u64 r = x - q * p;
    while (r >= p) r -= p;
    return r;
  }
};

// Residues below 2^32 multiply within 64 bits (the primes are < 2^31), which a
// hardware division reduces; an unreduced argument is reduced first. Same value.
inline u64 mulmod(u64 a, u64 b, u64 p) {
  if (((a | b) >> 32) == 0) return (a * b) % p;
  return ((a % p) * (b % p)) % p;
}

u64 powmod(u64 a, u64 e, u64 p) {
  u64 r = 1;
  a %= p;
  while (e) {
    if (e & 1) r = mulmod(r, a, p);
    a = mulmod(a, a, p);
    e >>= 1;
  }
  return r;
}

inline u64 invmod(u64 a, u64 p) { return powmod(a % p, p - 2, p); }

// The q-th root of a modulo p for an odd q prime to p - 1, where x -> x^q is a bijection:
// a^(1/q mod (p - 1)); false when q shares a factor with p - 1.
bool rootmod(u64 a, u64 q, u64 p, u64& r) {
  long long m = (long long)(p - 1), x0 = 0, x1 = 1, aa = (long long)(q % (p - 1)), bb = m;
  while (aa != 0) {                          // extended Euclid on (q, p - 1)
    long long t = bb / aa, tmp = bb - t * aa; bb = aa; aa = tmp;
    tmp = x0 - t * x1; x0 = x1; x1 = tmp;
  }
  if (bb != 1) return false;
  long long e = x0 % m; if (e < 0) e += m;
  r = powmod(a % p, (u64)e, p);
  return true;
}

// The smaller square root of a modulo the odd prime p (Tonelli-Shanks); false for a
// non-residue.
bool sqrtmod(u64 a, u64 p, u64& r) {
  a %= p;
  if (a == 0) { r = 0; return true; }
  if (powmod(a, (p - 1) / 2, p) != 1) return false;
  u64 q = p - 1, s = 0;
  while ((q & 1) == 0) { q >>= 1; ++s; }
  u64 z = 2;
  while (powmod(z, (p - 1) / 2, p) != p - 1) ++z;
  u64 m = s, c = powmod(z, q, p), t = powmod(a, q, p), x = powmod(a, (q + 1) / 2, p);
  while (t != 1) {
    u64 i = 0, t2 = t;
    while (t2 != 1) { t2 = mulmod(t2, t2, p); ++i; }
    u64 b = c;
    for (u64 j = 0; j + 1 < m - i; ++j) b = mulmod(b, b, p);
    m = i; c = mulmod(b, b, p); t = mulmod(t, c, p); x = mulmod(x, b, p);
  }
  r = x <= p - x ? x : p - x;               // the canonical root, as in gfp.sqrt_mod
  return true;
}

// Reduce a (possibly negative) R integer to its residue in [0, p).
inline u64 red(long long x, u64 p) {
  long long v = (long long)x % (long long)p; if (v < 0) v += p; return (u64)v;
}

inline std::vector<int> to_ivec(const IntegerVector& v) {
  return std::vector<int>(v.begin(), v.end());
}

inline std::vector<std::string> to_svec(const CharacterVector& v) {
  std::vector<std::string> o(v.size());
  for (int i = 0; i < v.size(); ++i) o[i] = as<std::string>(v[i]);
  return o;
}

// Parse a signed decimal string to its residue modulo p (handles arbitrary
// length without overflow).
u64 parse_mod(const std::string& s, u64 p) {
  size_t i = 0;
  bool neg = false;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; ++i; }
  u64 r = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') continue;
    r = (r * 10 + (u64)(s[i] - '0')) % p;
  }
  return neg ? (p - r) % p : r;
}

// Residue of the rational constant num/den (decimal strings) modulo p.
inline u64 reduce_rational(const std::string& num, const std::string& den, u64 p) {
  return mulmod(parse_mod(num, p), invmod(parse_mod(den, p), p), p);
}

// Accumulate the dual product of duals X and Y into o (width w: value at 0,
// partial derivatives at 1..w-1).
inline void dual_mul_acc(u64* o, const u64* X, const u64* Y, int w, u64 p) {
  // all entries are residues below p < 2^31, so x*Y[c] + y*X[c] < 2^63 is reduced
  // once, by Barrett (the constant cached per thread and prime)
  static thread_local u64 cp = 0, cm = 0;
  if (p != cp) { cp = p; cm = barrett_m(p); }
  u64 xv = X[0], yv = Y[0];
  u64 x0 = xv * yv, q0 = mulhi64(x0, cm), r0 = x0 - q0 * p;
  while (r0 >= p) r0 -= p;
  o[0] = addmod(o[0], r0, p);
  for (int c = 1; c < w; ++c) {
    u64 x = xv * Y[c] + yv * X[c];
    u64 q = mulhi64(x, cm), r = x - q * p;
    while (r >= p) r -= p;
    o[c] = addmod(o[c], r, p);
  }
}

// ---- Multivariate truncated power series over K formal gap lengths ----
// A "polyseries" is a flat array of nMono monomial coefficients, each a width-w
// dual number [value, d/dz_1, ...]. Monomials are the multi-indices mu in N^K with
// total degree |mu| <= Mtot, ordered by increasing total degree. It holds the
// exact dependence of a propagated state on the (generic) inter-event gap lengths
// Delta t_i; the rank over GF(p) later treats each monomial as a separate row (a
// polynomial in the gaps vanishes for generic gaps iff every coefficient does).
struct PolyBasis {
  int K, Mtot, nMono, w;
  std::vector<std::vector<int> > idx;          // nMono multi-indices, degree-ordered
  std::vector<std::vector<int> > sumSlot;      // sumSlot[a][b] = slot(idx[a]+idx[b]) or -1
  std::vector<std::vector<std::pair<int,int> > > decomp;  // decomp[s] = {(a,b): a+b == s}
  std::map<std::vector<int>, int> slotOf;

  // slot of idx[a] with `axis` incremented by k (promoting a time-order into a
  // gap axis), or -1 if it overflows the total-degree budget
  int shiftSlot(int a, int axis, int k) const {
    std::vector<int> mu = idx[a];
    mu[axis] += k;
    int td = 0; for (int i = 0; i < K; ++i) td += mu[i];
    if (td > Mtot) return -1;
    std::map<std::vector<int>, int>::const_iterator it = slotOf.find(mu);
    return it == slotOf.end() ? -1 : it->second;
  }

  PolyBasis(int K_, int Mtot_, int w_) : K(K_), Mtot(Mtot_), w(w_) {
    std::vector<int> mu(K, 0);
    for (int d = 0; d <= Mtot; ++d) enumerate(d, 0, d, mu);
    nMono = (int)idx.size();
    for (int s = 0; s < nMono; ++s) slotOf[idx[s]] = s;
    sumSlot.assign(nMono, std::vector<int>(nMono, -1));
    decomp.assign(nMono, std::vector<std::pair<int,int> >());
    std::vector<int> sum(K);
    for (int a = 0; a < nMono; ++a)
      for (int b = 0; b < nMono; ++b) {
        int td = 0;
        for (int i = 0; i < K; ++i) { sum[i] = idx[a][i] + idx[b][i]; td += sum[i]; }
        if (td > Mtot) continue;
        int s = slotOf[sum];
        sumSlot[a][b] = s;
        decomp[s].push_back(std::make_pair(a, b));
      }
  }
  // recursively emit all K-compositions of `remaining` (total degree d) at axis pos
  void enumerate(int remaining, int pos, int d, std::vector<int>& mu) {
    if (pos == K - 1) { mu[pos] = remaining; idx.push_back(mu); return; }
    for (int v = 0; v <= remaining; ++v) {
      mu[pos] = v;
      enumerate(remaining - v, pos + 1, d, mu);
    }
  }
};

// out (nMono x w) += A * B, convolution over monomials with the dual product rule
inline void poly_mul_acc(u64* out, const u64* A, const u64* B,
                         const PolyBasis& pb, u64 p) {
  int w = pb.w;
  for (int a = 0; a < pb.nMono; ++a) {
    const u64* Aa = A + (size_t)a * w;
    bool nz = false;
    for (int c = 0; c < w; ++c) if (Aa[c]) { nz = true; break; }
    if (!nz) continue;
    const std::vector<int>& srow = pb.sumSlot[a];
    for (int b = 0; b < pb.nMono; ++b) {
      int s = srow[b];
      if (s < 0) continue;
      dual_mul_acc(out + (size_t)s * w, Aa, B + (size_t)b * w, w, p);
    }
  }
}

// B = 1 / A as a polyseries of duals (A's constant monomial must have nonzero
// value). Computed monomial by monomial in increasing total degree.
inline bool poly_inv(u64* B, const u64* A, const PolyBasis& pb, u64 p) {
  int w = pb.w;
  std::fill(B, B + (size_t)pb.nMono * w, (u64)0);
  u64 a0 = A[0] % p;                       // slot 0 is the degree-0 monomial
  if (a0 == 0) return false;
  u64 vi = invmod(a0, p), vi2 = mulmod(vi, vi, p);
  std::vector<u64> invA0(w, 0);
  invA0[0] = vi;
  for (int c = 1; c < w; ++c) invA0[c] = submod(0, mulmod(A[c], vi2, p), p);
  for (int c = 0; c < w; ++c) B[c] = invA0[c];
  std::vector<u64> s(w, 0);
  for (int m = 1; m < pb.nMono; ++m) {     // monomials are degree-ordered
    std::fill(s.begin(), s.end(), (u64)0);
    const std::vector<std::pair<int,int> >& dc = pb.decomp[m];
    for (size_t t = 0; t < dc.size(); ++t) {
      int a = dc[t].first, b = dc[t].second;
      if (a == 0) continue;                // exclude the A[0]*B[m] term
      dual_mul_acc(s.data(), A + (size_t)a * w, B + (size_t)b * w, w, p);
    }
    for (int c = 0; c < w; ++c) s[c] = submod(0, s[c], p);   // negate
    dual_mul_acc(B + (size_t)m * w, invA0.data(), s.data(), w, p);
  }
  return true;
}

// B = sqrt(A) for a series with a residue constant term, from B*B = A monomial by
// monomial: 2 B_0 B_m = A_m - sum of B_a B_b over a, b != 0.
inline bool poly_sqrt(u64* B, const u64* A, const PolyBasis& pb, u64 p) {
  int w = pb.w;
  std::fill(B, B + (size_t)pb.nMono * w, (u64)0);
  u64 r;
  if (!sqrtmod(A[0], p, r) || r == 0) return false;
  u64 inv2r = invmod(mulmod(2, r, p), p);
  B[0] = r;
  for (int c = 1; c < w; ++c) B[c] = mulmod(A[c], inv2r, p);
  std::vector<u64> two0(w, 0), inv0(w, 0), s(w, 0);
  for (int c = 0; c < w; ++c) two0[c] = mulmod(2, B[c], p);
  u64 vi = invmod(two0[0], p), vi2 = mulmod(vi, vi, p);
  inv0[0] = vi;
  for (int c = 1; c < w; ++c) inv0[c] = submod(0, mulmod(two0[c], vi2, p), p);
  for (int m = 1; m < pb.nMono; ++m) {
    for (int c = 0; c < w; ++c) s[c] = A[(size_t)m * w + c] % p;
    const std::vector<std::pair<int,int> >& dc = pb.decomp[m];
    std::vector<u64> acc(w, 0);
    for (size_t t = 0; t < dc.size(); ++t) {
      int a = dc[t].first, b = dc[t].second;
      if (a == 0 || b == 0) continue;
      dual_mul_acc(acc.data(), B + (size_t)a * w, B + (size_t)b * w, w, p);
    }
    for (int c = 0; c < w; ++c) s[c] = submod(s[c], acc[c], p);
    dual_mul_acc(B + (size_t)m * w, inv0.data(), s.data(), w, p);
  }
  return true;
}

// B = A^(1/q) for an odd q prime to p - 1, by Newton on B^q = A from the root of the
// constant term: B <- B - (B^q - A) / (q B^(q - 1)), doubling the exact degree each step.
inline bool poly_root(u64* B, const u64* A, u64 q, const PolyBasis& pb, u64 p) {
  int w = pb.w;
  size_t blk = (size_t)pb.nMono * w;
  std::fill(B, B + blk, (u64)0);
  u64 r;
  if (!rootmod(A[0], q, p, r) || r == 0) return false;
  B[0] = r;
  std::vector<u64> P(blk), Pq(blk), T(blk), D(blk), I(blk);
  for (int it = 0; it < 64; ++it) {
    std::fill(P.begin(), P.end(), (u64)0);       // P = B^(q - 1)
    P[0] = 1;
    for (u64 k = 0; k + 1 < q; ++k) {
      std::fill(T.begin(), T.end(), (u64)0);
      poly_mul_acc(T.data(), P.data(), B, pb, p);
      P = T;
    }
    std::fill(Pq.begin(), Pq.end(), (u64)0);      // Pq = B^q - A
    poly_mul_acc(Pq.data(), P.data(), B, pb, p);
    for (size_t c = 0; c < blk; ++c) Pq[c] = submod(Pq[c], A[c] % p, p);
    bool zero = true;
    for (size_t c = 0; c < blk; ++c) if (Pq[c]) { zero = false; break; }
    if (zero) return true;
    for (size_t c = 0; c < blk; ++c) D[c] = mulmod(P[c], q % p, p);
    if (!poly_inv(I.data(), D.data(), pb, p)) return false;
    std::fill(T.begin(), T.end(), (u64)0);
    poly_mul_acc(T.data(), Pq.data(), I.data(), pb, p);
    for (size_t c = 0; c < blk; ++c) B[c] = submod(B[c], T[c], p);
  }
  return false;
}

// In-place Gauss-Jordan over GF(p); returns the pivot columns.
std::vector<int> rref_mod(std::vector<std::vector<u64> >& A, u64 p) {
  std::vector<int> pivots;
  if (A.empty()) return pivots;
  const Barrett br(p);
  int nrows = (int)A.size(), ncols = (int)A[0].size(), r = 0;
  for (int c = 0; c < ncols && r < nrows; ++c) {
    int piv = -1;
    for (int i = r; i < nrows; ++i)
      if (A[i][c] % p != 0) { piv = i; break; }
    if (piv < 0) continue;
    std::swap(A[r], A[piv]);
    // the pivot row is zero left of c (earlier pivot columns eliminated, skipped
    // columns zero from row r down), so every update starts at c; all entries are
    // residues below p < 2^31, so a product fits 64 bits
    u64* pr = A[r].data();
    for (int j = c; j < ncols; ++j) pr[j] = br.red(pr[j] % p);
    u64 inv = invmod(pr[c], p);
    for (int j = c; j < ncols; ++j) pr[j] = br.red(pr[j] * inv);
    for (int i = 0; i < nrows; ++i) {
      if (i == r) continue;
      u64* ai = A[i].data();
      u64 f = ai[c] % p;
      if (f == 0) continue;
      u64 nf = p - f;                              // a - f*b = a + (p - f)*b mod p
      for (int j = c; j < ncols; ++j) {
        if (pr[j] == 0) continue;
        ai[j] = br.red(br.red(ai[j]) + nf * pr[j]);
      }
    }
    pivots.push_back(c);
    ++r;
  }
  return pivots;
}

// ---- Rank over GF(p)((eps)) of rows truncated at eps^N ----
// A series row holds nz entries, each a truncated power series: coefficient d of
// entry c at c*N + d. Rows are reduced with a pivot of minimal valuation over the
// whole remaining block, which keeps every entry exact modulo eps^N. The number of
// pivots found is a lower bound of the rank that grows to it with N.
inline int ser_val(const u64* a, int n) {
  for (int d = 0; d < n; ++d) if (a[d]) return d;
  return n;
}

// B = 1 / A modulo eps^n, A[0] != 0
inline void ser_inv(u64* B, const u64* A, int n, u64 p) {
  u64 a0 = invmod(A[0], p);
  B[0] = a0;
  for (int d = 1; d < n; ++d) {
    u64 s = 0;
    for (int j = 1; j <= d; ++j) s = addmod(s, mulmod(A[j], B[d - j], p), p);
    B[d] = mulmod(submod(0, s, p), a0, p);
  }
}

// In place: the first `rank` rows of A become the echelon rows. Columns with
// mask[c] == 0 are ignored.
// With `carry`, the columns outside the mask are updated as well. `prec` is each row's precision
// (all N when absent); a reduced row is exact to min(prec_i, prec_r), ties in valuation
// go to the more precise pivot. `cprec` receives the precision in the columns outside the mask.
int rref_series_mod(std::vector<std::vector<u64> >& A, int nz, int N, u64 p,
                    const std::vector<char>& mask, int threads = 1, bool carry = false,
                    std::vector<int>* prec = nullptr, std::vector<int>* cprec = nullptr) {
  int nr = (int)A.size(), r = 0;
  std::vector<char> used(nz, 0), piv(nz, 0);
  for (int c = 0; c < nz; ++c) if (!mask[c]) used[c] = 1;
  std::vector<int> P0, C0;
  if (!prec) { P0.assign(nr, N); prec = &P0; }
  if (!cprec) { C0.assign(nr, N); cprec = &C0; } else cprec->assign(prec->begin(), prec->end());
  std::vector<int>& P = *prec;
  std::vector<int>& CP = *cprec;
  std::vector<u64> uinv(N);
  std::vector<size_t> nzc;
  auto swapRows = [&](int i, int j) {
    std::swap(A[i], A[j]); std::swap(P[i], P[j]); std::swap(CP[i], CP[j]); };
  while (r < nr) {
    int bi = -1, bc = -1, bv = N;
    // a row with no entry left in the eligible columns moves behind the active block
    for (int i = r; i < nr && !(bv == 0 && P[bi] == N); ) {
      int rv = P[i], rc = -1;
      for (int c = 0; c < nz && rv > 0; ++c) {
        if (used[c]) continue;
        int v = ser_val(A[i].data() + (size_t)c * N, rv);
        if (v < rv) { rv = v; rc = c; }
      }
      if (rc < 0) { swapRows(i, nr - 1); --nr; continue; }
      if (rv < bv || (rv == bv && P[i] > P[bi])) { bv = rv; bi = i; bc = rc; }
      ++i;
    }
    if (bi < 0) break;
    swapRows(r, bi);
    int n = P[r] - bv;
    ser_inv(uinv.data(), A[r].data() + (size_t)bc * N + bv, n, p);
    // below the pivot, used columns are zero and column bc becomes zero
    nzc.clear();
    for (int c = 0; c < nz; ++c) {
      if (c == bc || piv[c] || (used[c] && !carry)) continue;
      if (ser_val(A[r].data() + (size_t)c * N, N) < N) nzc.push_back((size_t)c * N);
    }
    const u64* pr = A[r].data();
    const size_t off = (size_t)bc * N;
    const int Pr = P[r], CPr = CP[r];
    #pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(dynamic, 8)
    for (int i = r + 1; i < nr; ++i) {
      u64* row = A[i].data();
      const u64* b = row + off;
      if (ser_val(b, P[i]) >= P[i]) continue;
      int Pn = std::min(P[i], Pr), nq = Pn - bv;
      std::vector<u64> q(nq > 0 ? nq : 0);
      for (int d = 0; d < nq; ++d) {
        u64 s = 0;
        for (int j = 0; j <= d; ++j) s = addmod(s, mulmod(b[bv + j], uinv[d - j], p), p);
        q[d] = s;
      }
      int qn = nq;
      while (qn > 0 && !q[qn - 1]) --qn;
      for (size_t k = 0; k < nzc.size(); ++k) {
        const u64* rc = pr + nzc[k];
        u64* ic = row + nzc[k];
        for (int d = 0; d < N; ++d) {
          if (!rc[d]) continue;
          for (int e = 0; e < qn && d + e < N; ++e)
            if (q[e]) ic[d + e] = submod(ic[d + e], mulmod(q[e], rc[d], p), p);
        }
      }
      std::fill(row + off, row + off + N, (u64)0);
      int CPn = std::min(CP[i], std::min(CPr, nq));
      // coefficients past the precision are unknown: zero them
      if (Pn < N || CPn < N)
        for (int c = 0; c < nz; ++c) {
          int lim = (used[c] && !piv[c]) ? CPn : Pn;
          if (lim < N) std::fill(row + (size_t)c * N + std::max(lim, 0), row + (size_t)(c + 1) * N, (u64)0);
        }
      P[i] = Pn; CP[i] = CPn;
    }
    used[bc] = 1; piv[bc] = 1;
    ++r;
  }
  return r;
}

// ---- The series kernel at eps = 1 ----
// A Laurent series with absolute precision: coefficient of eps^(lo + i) is c[i],
// zero beyond c, and the series is known modulo eps^prec.
const int LS_EXACT = 1 << 20;
struct LSer { int lo, prec; std::vector<u64> c; };

inline int ls_val(const LSer& a) {
  for (size_t i = 0; i < a.c.size() && a.lo + (int)i < a.prec; ++i)
    if (a.c[i]) return a.lo + (int)i;
  return a.prec;
}

LSer ls_mul(const LSer& a, const LSer& b, u64 p) {
  int va = ls_val(a), vb = ls_val(b);
  LSer o; o.lo = a.lo + b.lo;
  long pa = (long)va + b.prec, pb = (long)vb + a.prec;
  o.prec = (int)std::min<long>(LS_EXACT, std::min(pa, pb));
  int len = (int)std::min<long>((long)o.prec - o.lo, (long)a.c.size() + b.c.size());
  o.c.assign(std::max(0, len), 0);
  for (size_t i = 0; i < a.c.size(); ++i) {
    if (!a.c[i]) continue;
    for (size_t j = 0; j < b.c.size() && (int)(i + j) < len; ++j)
      if (b.c[j]) o.c[i + j] = addmod(o.c[i + j], mulmod(a.c[i], b.c[j], p), p);
  }
  return o;
}

// a + s * b
LSer ls_axpy(const LSer& a, const LSer& b, u64 sgn, u64 p) {
  LSer o; o.lo = std::min(a.lo, b.lo); o.prec = std::min(a.prec, b.prec);
  int hi = std::max(a.lo + (int)a.c.size(), b.lo + (int)b.c.size());
  hi = std::min(hi, o.prec);
  o.c.assign(std::max(0, hi - o.lo), 0);
  for (size_t i = 0; i < a.c.size(); ++i) {
    int e = a.lo + (int)i - o.lo;
    if (e < (int)o.c.size()) o.c[e] = addmod(o.c[e], a.c[i], p);
  }
  for (size_t i = 0; i < b.c.size(); ++i) {
    int e = b.lo + (int)i - o.lo;
    if (e < (int)o.c.size()) o.c[e] = addmod(o.c[e], mulmod(sgn, b.c[i], p), p);
  }
  return o;
}

// Kernel of the series echelon rows U (rank r, entries mod eps^N) at eps = 1. Each
// kernel vector is solved by back substitution in Laurent series, shifted to start
// at eps^0, and accepted only if every entry is a polynomial whose known tail of
// zeros is at least as long as its degree. Returns false otherwise.
bool series_kernel_at_one(const std::vector<std::vector<u64> >& U, int r, int nz, int N,
                          u64 p, std::vector<std::vector<u64> >& K) {
  std::vector<int> pc(r, -1), pv(r, N);
  std::vector<char> isPiv(nz, 0);
  for (int t = 0; t < r; ++t) {
    // the pivot of row t: its entry of minimal valuation among columns no earlier
    // row pivots on, ties to the lowest column (as rref_series_mod chose it)
    for (int c = 0; c < nz; ++c) {
      if (isPiv[c]) continue;
      int v = ser_val(U[t].data() + (size_t)c * N, N);
      if (v < pv[t]) { pv[t] = v; pc[t] = c; }
    }
    if (pc[t] < 0) return false;
    isPiv[pc[t]] = 1;
  }
  K.clear();
  for (int f = 0; f < nz; ++f) {
    if (isPiv[f]) continue;
    std::vector<LSer> x(nz);
    for (int c = 0; c < nz; ++c) { x[c].lo = 0; x[c].prec = LS_EXACT; }
    x[f].c.assign(1, 1);
    for (int t = r - 1; t >= 0; --t) {
      LSer acc; acc.lo = 0; acc.prec = LS_EXACT;
      for (int c = 0; c < nz; ++c) {
        if (c == pc[t] || x[c].c.empty()) continue;
        LSer u; u.lo = 0; u.prec = N;
        u.c.assign(U[t].begin() + (size_t)c * N, U[t].begin() + (size_t)(c + 1) * N);
        acc = ls_axpy(acc, ls_mul(u, x[c], p), 1, p);
      }
      int v = pv[t], n = N - v;
      std::vector<u64> uinv(n);
      ser_inv(uinv.data(), U[t].data() + (size_t)pc[t] * N + v, n, p);
      LSer ui; ui.lo = -v; ui.prec = n - v; ui.c = uinv;
      LSer q = ls_mul(acc, ui, p);
      for (size_t i = 0; i < q.c.size(); ++i) q.c[i] = submod(0, q.c[i], p);
      x[pc[t]] = q;
    }
    int m = LS_EXACT;
    for (int c = 0; c < nz; ++c) if (!x[c].c.empty()) m = std::min(m, ls_val(x[c]));
    std::vector<u64> k1(nz, 0);
    for (int c = 0; c < nz; ++c) {
      const LSer& e = x[c];
      int deg = -1;
      for (size_t i = 0; i < e.c.size(); ++i)
        if (e.lo + (int)i < e.prec && e.c[i]) deg = e.lo + (int)i - m;
      int known = e.prec >= LS_EXACT ? LS_EXACT : e.prec - m;
      if (deg >= 0 && known < 2 * (deg + 1) + 1) return false;
      for (size_t i = 0; i < e.c.size(); ++i)
        if (e.lo + (int)i < e.prec) k1[c] = addmod(k1[c], e.c[i], p);
    }
    K.push_back(k1);
  }
  // the vectors at eps = 1 must stay independent
  std::vector<std::vector<u64> > KK = K;
  if ((int)rref_mod(KK, p).size() != (int)K.size()) return false;
  return true;
}

// Reduced rows whose nullspace is spanned by the vectors K.
void annihilator_rows(const std::vector<std::vector<u64> >& K, int nz, u64 p,
                      std::vector<std::vector<u64> >& R, std::vector<int>& piv) {
  std::vector<std::vector<u64> > A = K;
  std::vector<int> kp = rref_mod(A, p);
  std::vector<char> isP(nz, 0);
  for (size_t i = 0; i < kp.size(); ++i) isP[kp[i]] = 1;
  // annihilator basis: for each non-pivot column g of rref(K), the row
  // e_g - sum_i A[i][g] e_{kp[i]}
  std::vector<std::vector<u64> > rows;
  for (int g = 0; g < nz; ++g) {
    if (isP[g]) continue;
    std::vector<u64> row(nz, 0);
    row[g] = 1;
    for (size_t i = 0; i < kp.size(); ++i) row[kp[i]] = submod(0, A[i][g], p);
    rows.push_back(row);
  }
  piv = rref_mod(rows, p);
  rows.resize(piv.size());
  R = rows;
}

// Restrict the gap monomials to the line Delta_i = lineC[i] * eps: rows arrive in
// groups of nMono (one per monomial of one time-order row) and each group becomes a
// series row, monomial mu contributing lineC^mu at eps^|mu|.
void rows_to_line(const std::vector<std::vector<u64> >& rows, size_t from,
                  const PolyBasis& pb, const std::vector<u64>& lineC, int nz, int N,
                  u64 p, std::vector<std::vector<u64> >& srows, bool keepZero = false) {
  int nM = pb.nMono;
  std::vector<u64> coef(nM, 1);
  std::vector<int> deg(nM, 0);
  for (int mo = 0; mo < nM; ++mo)
    for (int i = 0; i < pb.K; ++i) {
      deg[mo] += pb.idx[mo][i];
      u64 ci = i < (int)lineC.size() ? lineC[i] : 1;
      coef[mo] = mulmod(coef[mo], powmod(ci, (u64)pb.idx[mo][i], p), p);
    }
  for (size_t g = from; g + nM <= rows.size(); g += nM) {
    std::vector<u64> s((size_t)nz * N, 0);
    bool any = false;
    for (int mo = 0; mo < nM; ++mo) {
      if (deg[mo] >= N || !coef[mo]) continue;
      const std::vector<u64>& row = rows[g + mo];
      for (int c = 0; c < nz; ++c) {
        if (!row[c]) continue;
        u64* e = s.data() + (size_t)c * N + deg[mo];
        *e = addmod(*e, mulmod(coef[mo], row[c], p), p);
        any = true;
      }
    }
    if (any || keepZero) srows.push_back(s);
  }
}

#ifdef SYMIDENT_HAVE_INT128
u128 u128_isqrt(u128 n) {
  if (n == 0) return 0;
  int bits = 0;
  for (u128 t = n; t; t >>= 1) ++bits;
  u128 x = (u128)1 << ((bits + 1) / 2);
  for (int it = 0; it < 300; ++it) {
    u128 y = (x + n / x) / 2;
    if (y >= x) break;
    x = y;
  }
  while (x > 0 && x * x > n) --x;
  while ((x + 1) * (x + 1) <= n) ++x;
  return x;
}

std::string i128_to_string(i128 v) {
  if (v == 0) return "0";
  bool neg = v < 0;
  u128 u = neg ? (u128)(-(v + 1)) + 1 : (u128)v;
  std::string s;
  while (u > 0) { s += (char)('0' + (int)(u % 10)); u /= 10; }
  if (neg) s += '-';
  std::reverse(s.begin(), s.end());
  return s;
}

// Recover n/d with x = n * d^{-1} (mod M), |n|, |d| <= sqrt(M/2); returns false
// if no such bounded rational exists.
bool rational_reconstruct(u128 x, u128 M, i128& num, i128& den) {
  x %= M;
  if (x == 0) { num = 0; den = 1; return true; }
  u128 bound = u128_isqrt(M / 2);
  i128 r0 = (i128)M, r1 = (i128)x, s0 = 0, s1 = 1;
  while (r1 > (i128)bound) {
    i128 q = r0 / r1;
    i128 r2 = r0 - q * r1; r0 = r1; r1 = r2;
    i128 s2 = s0 - q * s1; s0 = s1; s1 = s2;
  }
  if (s1 == 0) return false;
  num = r1; den = s1;
  if (den < 0) { num = -num; den = -den; }
  u128 an = (u128)(num < 0 ? -num : num);
  if (an > bound || (u128)den > bound) return false;
  return true;
}
#endif

// Evaluate the initial-condition tape at time order 0 only. icVal has one width-w
// dual vector per slot; leaves [0, nLeaves) are pre-filled with their value and
// dual, instruction i writes slot nLeaves + i. Returns false on a zero reciprocal.
bool eval_ic_order0(std::vector<std::vector<u64> >& icVal,
                    const std::vector<int>& icOp, const std::vector<int>& icA,
                    const std::vector<int>& icB, const std::vector<u64>& icCval,
                    int nLeaves, int w, u64 p) {
  int n = (int)icOp.size();
  for (int i = 0; i < n; ++i) {
    u64* o = icVal[nLeaves + i].data();
    switch (icOp[i]) {
      case OP_CONST:
        o[0] = icCval[i];
        break;
      case OP_ADD: {
        const u64* pa = icVal[icA[i]].data();
        const u64* pb = icVal[icB[i]].data();
        for (int c = 0; c < w; ++c) o[c] = addmod(pa[c], pb[c], p);
        break;
      }
      case OP_MUL: {
        const u64* pa = icVal[icA[i]].data();
        const u64* pb = icVal[icB[i]].data();
        for (int c = 0; c < w; ++c) o[c] = 0;
        dual_mul_acc(o, pa, pb, w, p);
        break;
      }
      case OP_INV: {
        const u64* pa = icVal[icA[i]].data();
        u64 a0 = pa[0] % p;
        if (a0 == 0) return false;
        u64 vi = invmod(a0, p), vi2 = mulmod(vi, vi, p);
        o[0] = vi;
        for (int c = 1; c < w; ++c) o[c] = submod(0, mulmod(pa[c], vi2, p), p);
        break;
      }
      case OP_SQRT: {
        // d sqrt(a) = da / (2 sqrt(a)); a non-residue has no point on the variety here
        const u64* pa = icVal[icA[i]].data();
        u64 r;
        if (!sqrtmod(pa[0], p, r) || r == 0) return false;
        u64 inv2r = invmod(mulmod(2, r, p), p);
        o[0] = r;
        for (int c = 1; c < w; ++c) o[c] = mulmod(pa[c], inv2r, p);
        break;
      }
      case OP_ROOT: {
        // d a^(1/q) = da / (q r^(q - 1)), q odd in icB
        const u64* pa = icVal[icA[i]].data();
        u64 q = (u64)icB[i], r;
        if (!rootmod(pa[0], q, p, r) || r == 0) return false;
        u64 inv = invmod(mulmod(q % p, powmod(r, q - 1, p), p), p);
        o[0] = r;
        for (int c = 1; c < w; ++c) o[c] = mulmod(pa[c], inv, p);
        break;
      }
    }
  }
  return true;
}

bool build_obs_rows(std::vector<std::vector<u64> >& val,
                    const std::vector<int>& op, const std::vector<int>& a,
                    const std::vector<int>& b, const std::vector<u64>& cval,
                    int instrBase, const std::vector<int>& stateSlots,
                    const std::vector<int>& fOut, const std::vector<int>& gOut,
                    int nz, int w, int Nt, u64 p,
                    std::vector<std::vector<u64> >& rows) {
  int nInstr = (int)op.size();
  int m = (int)stateSlots.size();
  std::vector<std::vector<u64> > invA0(nInstr);
  bool fail = false;
  for (int k = 0; k <= Nt && !fail; ++k) {
    for (int i = 0; i < nInstr; ++i) {
      u64* o = val[instrBase + i].data() + (size_t)k * w;
      switch (op[i]) {
        case OP_CONST:
          if (k == 0) o[0] = cval[i];
          break;
        case OP_ADD: {
          const u64* pa = val[a[i]].data() + (size_t)k * w;
          const u64* pb = val[b[i]].data() + (size_t)k * w;
          for (int c = 0; c < w; ++c) o[c] = addmod(pa[c], pb[c], p);
          break;
        }
        case OP_MUL: {
          const u64* pa = val[a[i]].data();
          const u64* pb = val[b[i]].data();
          for (int ii = 0; ii <= k; ++ii)
            dual_mul_acc(o, pa + (size_t)ii * w, pb + (size_t)(k - ii) * w, w, p);
          break;
        }
        case OP_INV: {
          const u64* pa = val[a[i]].data();
          if (k == 0) {
            u64 a0 = pa[0] % p;
            if (a0 == 0) { fail = true; break; }
            u64 vi = invmod(a0, p), vi2 = mulmod(vi, vi, p);
            invA0[i].assign(w, 0);
            invA0[i][0] = vi;
            for (int c = 1; c < w; ++c) invA0[i][c] = submod(0, mulmod(pa[c], vi2, p), p);
            for (int c = 0; c < w; ++c) o[c] = invA0[i][c];
          } else {
            std::vector<u64> s(w, 0);
            const u64* self = val[instrBase + i].data();
            for (int j = 1; j <= k; ++j)
              dual_mul_acc(s.data(), pa + (size_t)j * w,
                           self + (size_t)(k - j) * w, w, p);
            for (int c = 0; c < w; ++c) s[c] = submod(0, s[c], p);
            dual_mul_acc(o, invA0[i].data(), s.data(), w, p);
          }
          break;
        }
      }
      if (fail) break;
    }
    if (fail || k == Nt) break;
    u64 invk = invmod((u64)(k + 1), p);
    for (int i = 0; i < m; ++i) {
      const u64* fk = val[fOut[i]].data() + (size_t)k * w;
      u64* st = val[stateSlots[i]].data() + (size_t)(k + 1) * w;
      for (int c = 0; c < w; ++c) st[c] = mulmod(fk[c], invk, p);
    }
  }
  if (fail) return false;
  for (int gi = 0; gi < (int)gOut.size(); ++gi) {
    const u64* g = val[gOut[gi]].data();
    for (int k = 0; k <= Nt; ++k) {
      std::vector<u64> row(nz);
      for (int c = 0; c < nz; ++c) row[c] = g[(size_t)k * w + 1 + c];
      rows.push_back(row);
    }
  }
  return true;
}

// Polyseries version of build_obs_rows: every value is a width-(nMono*w) block (a
// truncated power series in the gap lengths, each monomial a dual). Runs to time
// order Nrun and emits z-gradient rows for orders 0..Ntemit and all gap monomials.
// `val` slots are pre-sized to (Nrun+1)*blk with states seeded at order 0.
bool build_obs_rows_poly(std::vector<std::vector<u64> >& val,
                         const std::vector<int>& op, const std::vector<int>& a,
                         const std::vector<int>& b, const std::vector<u64>& cval,
                         int instrBase, const std::vector<int>& stateSlots,
                         const std::vector<int>& fOut, const std::vector<int>& gOut,
                         const PolyBasis& pb, int Nrun, int Ntemit, u64 p,
                         std::vector<std::vector<u64> >& rows,
                         const u64* dtau = nullptr) {
  int nInstr = (int)op.size();
  int m = (int)stateSlots.size();
  int w = pb.w, nM = pb.nMono, blk = nM * w, nz = w - 1;
  std::vector<std::vector<u64> > invA0(nInstr);
  bool fail = false;
  // Lane ranges: [lo, hi) of the nonzero dual lanes of a value at one order and
  // monomial, found when first read (a value is final once an instruction reads it).
  // Products run over these ranges only and add unreduced: each product is below
  // p^2 < 2^62, the sum is kept below a multiple of p under 2^63 and reduced once.
  int nVal = (int)val.size();
  std::vector<std::vector<int> > rlo(nVal), rhi(nVal);
  auto rangeOf = [&](int v, int k, int mo, int& lo, int& hi) {
    if (rlo[v].empty()) { rlo[v].assign((size_t)(Nrun + 1) * nM, -1); rhi[v].assign((size_t)(Nrun + 1) * nM, 0); }
    size_t key = (size_t)k * nM + mo;
    if (rlo[v][key] < 0) {
      const u64* d = val[v].data() + (size_t)k * blk + (size_t)mo * w;
      int l = 1; while (l < w && d[l] == 0) ++l;
      int h = w; while (h > l && d[h - 1] == 0) --h;
      rlo[v][key] = l; rhi[v][key] = h;
    }
    lo = rlo[v][key]; hi = rhi[v][key];
  };
  const u64 LIM = (((u64)1 << 63) / p) * p;
  const Barrett br(p);
  std::vector<u64> acc(blk);
  // out += sum_{ii = j0..j1} A_ii * B_{k-ii} over the monomials, A and B tape values
  auto convAdd = [&](u64* out, int av, int bv, int j0, int j1, int k) {
    std::fill(acc.begin(), acc.end(), (u64)0);
    const u64* pa = val[av].data(); const u64* pbv = val[bv].data();
    for (int ii = j0; ii <= j1; ++ii) {
      for (int am = 0; am < nM; ++am) {
        const u64* Aa = pa + (size_t)ii * blk + (size_t)am * w;
        int la, ha; rangeOf(av, ii, am, la, ha);
        u64 x0 = Aa[0];
        if (x0 == 0 && la == ha) continue;
        const std::vector<int>& srow = pb.sumSlot[am];
        for (int bm = 0; bm < nM; ++bm) {
          int sl = srow[bm];
          if (sl < 0) continue;
          const u64* Bb = pbv + (size_t)(k - ii) * blk + (size_t)bm * w;
          int lb, hb; rangeOf(bv, k - ii, bm, lb, hb);
          u64 y0 = Bb[0];
          if (y0 == 0 && lb == hb) continue;
          u64* ac = acc.data() + (size_t)sl * w;
          if (x0 && y0) { u64 t = ac[0] + x0 * y0; ac[0] = t >= LIM ? t - LIM : t; }
          if (x0) for (int c = lb; c < hb; ++c) { u64 t = ac[c] + x0 * Bb[c]; ac[c] = t >= LIM ? t - LIM : t; }
          if (y0) for (int c = la; c < ha; ++c) { u64 t = ac[c] + y0 * Aa[c]; ac[c] = t >= LIM ? t - LIM : t; }
        }
      }
    }
    for (int c = 0; c < blk; ++c) if (acc[c]) out[c] = addmod(out[c], br.red(acc[c]), p);
  };
  for (int k = 0; k <= Nrun && !fail; ++k) {
    for (int i = 0; i < nInstr; ++i) {
      u64* o = val[instrBase + i].data() + (size_t)k * blk;
      switch (op[i]) {
        case OP_CONST:
          if (k == 0) o[0] = cval[i];       // value at monomial 0, dual 0
          break;
        case OP_ADD: {
          const u64* pa = val[a[i]].data() + (size_t)k * blk;
          const u64* pbb = val[b[i]].data() + (size_t)k * blk;
          for (int c = 0; c < blk; ++c) o[c] = addmod(pa[c], pbb[c], p);
          break;
        }
        case OP_MUL:
          convAdd(o, a[i], b[i], 0, k, k);
          break;
        case OP_INV: {
          const u64* pa = val[a[i]].data();
          if (k == 0) {
            invA0[i].assign(blk, 0);
            if (!poly_inv(invA0[i].data(), pa, pb, p)) { fail = true; break; }
            for (int c = 0; c < blk; ++c) o[c] = invA0[i][c];
          } else {
            std::vector<u64> s(blk, 0);
            convAdd(s.data(), a[i], instrBase + i, 1, k, k);
            for (int c = 0; c < blk; ++c) s[c] = submod(0, s[c], p);
            poly_mul_acc(o, invA0[i].data(), s.data(), pb, p);
          }
          break;
        }
      }
      if (fail) break;
    }
    if (fail || k == Nrun) break;
    u64 invk = invmod((u64)(k + 1), p);
    for (int i = 0; i < m; ++i) {
      const u64* fk = val[fOut[i]].data() + (size_t)k * blk;
      u64* st = val[stateSlots[i]].data() + (size_t)(k + 1) * blk;
      for (int c = 0; c < blk; ++c) st[c] = mulmod(fk[c], invk, p);
    }
  }
  if (fail) return false;
  for (int gi = 0; gi < (int)gOut.size(); ++gi) {
    const u64* g = val[gOut[gi]].data();
    for (int k = 0; k <= Ntemit; ++k)
      for (int mo = 0; mo < nM; ++mo) {
        std::vector<u64> row(nz);
        const u64* d = g + (size_t)k * blk + (size_t)mo * w;
        for (int c = 0; c < nz; ++c) row[c] = d[1 + c];
        // a segment anchored at a moving time tau: at fixed absolute time the local
        // coefficient y_k shifts by -(k+1) y_{k+1} dtau
        if (dtau) {
          u64 fk = mulmod((u64)(k + 1) % p, g[(size_t)(k + 1) * blk + (size_t)mo * w], p);
          if (fk)
            for (int c = 0; c < nz; ++c)
              row[c] = submod(row[c], mulmod(fk, dtau[c], p), p);
        }
        rows.push_back(row);
      }
  }
  return true;
}

// One condition's tape as thread-safe plain C++: instructions and slots as ints,
// constants and the IC map as decimal strings, an optional steady-state seed
// icSeed as raw ints. Constants are reduced per prime in build_one_condition, so
// one extraction serves every prime (the batch path relies on this).
struct CondRaw {
  std::vector<int> op, a, b, stateSlots, fOut, gOut, icLeaf, icOp, icA, icB, icOut;
  std::vector<std::string> cnum, cden, icCnum, icCden, icNum, icDen;
  bool hasIcTape, hasIcSeed;
  std::vector<std::vector<int> > icSeedRaw;
};

CondRaw extract_cond_raw(List tp, int nStates, int w) {
  CondRaw cd;
  cd.op = to_ivec(tp["op"]); cd.a = to_ivec(tp["a"]); cd.b = to_ivec(tp["b"]);
  cd.stateSlots = to_ivec(tp["state_slots"]);
  cd.fOut = to_ivec(tp["f_out"]); cd.gOut = to_ivec(tp["g_out"]);
  cd.icLeaf = to_ivec(tp["ic_leaf"]);
  cd.cnum = to_svec(tp["cnum"]); cd.cden = to_svec(tp["cden"]);
  cd.hasIcTape = tp.containsElementNamed("ic_op");
  cd.hasIcSeed = tp.containsElementNamed("ic_seed");
  if (cd.hasIcSeed) {
    IntegerMatrix icSeed = as<IntegerMatrix>(tp["ic_seed"]);
    cd.icSeedRaw.assign(nStates, std::vector<int>(w, 0));
    for (int i = 0; i < nStates; ++i)
      for (int c = 0; c < w; ++c) cd.icSeedRaw[i][c] = icSeed(i, c);
  } else if (cd.hasIcTape) {
    cd.icOp = to_ivec(tp["ic_op"]); cd.icA = to_ivec(tp["ic_a"]);
    cd.icB = to_ivec(tp["ic_b"]); cd.icOut = to_ivec(tp["ic_out"]);
    cd.icCnum = to_svec(tp["ic_cnum"]); cd.icCden = to_svec(tp["ic_cden"]);
  } else {
    cd.icNum = to_svec(tp["ic_num"]); cd.icDen = to_svec(tp["ic_den"]);
  }
  return cd;
}

// Build one condition's observability rows at a single point/prime. leafVal holds
// each leaf's value and dual already reduced mod p; the tape and IC constants are
// reduced from their strings here, so the same CondRaw serves any prime. Returns
// false on a vanishing denominator (tape reciprocal or reciprocal IC).
bool build_one_condition(const CondRaw& cd,
                         const std::vector<std::vector<u64> >& leafVal,
                         int nLeaves, int nStates, int nz, int w, int Nt, u64 p,
                         std::vector<std::vector<u64> >& outRows) {
  int nInstr = (int)cd.op.size();
  int instrBase = nLeaves + nStates;
  int S = instrBase + nInstr;

  std::vector<u64> cval(nInstr, 0);
  for (int i = 0; i < nInstr; ++i)
    if (cd.op[i] == OP_CONST)
      cval[i] = reduce_rational(cd.cnum[i], cd.cden[i], p);

  std::vector<std::vector<u64> > icVal;
  if (cd.hasIcTape && !cd.hasIcSeed) {
    int nIc = (int)cd.icOp.size();
    std::vector<u64> icCval(nIc, 0);
    for (int i = 0; i < nIc; ++i)
      if (cd.icOp[i] == OP_CONST)
        icCval[i] = reduce_rational(cd.icCnum[i], cd.icCden[i], p);
    icVal.assign(nLeaves + nIc, std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L)
      for (int c = 0; c < w; ++c) icVal[L][c] = leafVal[L][c];
    if (!eval_ic_order0(icVal, cd.icOp, cd.icA, cd.icB, icCval, nLeaves, w, p))
      return false;
  }

  std::vector<u64> icConst;
  if (!cd.hasIcTape && !cd.hasIcSeed) {
    icConst.assign(nStates, 0);
    for (int i = 0; i < nStates; ++i)
      icConst[i] = reduce_rational(cd.icNum[i], cd.icDen[i], p);
  }

  std::vector<std::vector<u64> > val(S, std::vector<u64>((size_t)(Nt + 1) * w, 0));
  for (int L = 0; L < nLeaves; ++L)
    for (int c = 0; c < w; ++c) val[L][c] = leafVal[L][c];
  for (int i = 0; i < (int)cd.stateSlots.size(); ++i) {
    int slot = cd.stateSlots[i];
    if (cd.hasIcSeed) {
      for (int c = 0; c < w; ++c) val[slot][c] = red(cd.icSeedRaw[i][c], p);
      continue;
    }
    int src = cd.hasIcTape ? cd.icOut[i] : (cd.icLeaf[i] >= 0 ? cd.icLeaf[i] : -1);
    if (src >= 0) {
      const std::vector<u64>& s = src < nLeaves ? leafVal[src] : icVal[src];
      for (int c = 0; c < w; ++c) val[slot][c] = s[c];
    } else {
      val[slot][0] = icConst[i];
    }
  }

  return build_obs_rows(val, cd.op, cd.a, cd.b, cval, instrBase, cd.stateSlots,
                        cd.fOut, cd.gOut, nz, w, Nt, p, outRows);
}

// build_one_condition over column blocks on `threads` threads: each block holds the
// value and its own dual lanes, the rows are assembled over all columns. A steady-state
// seed is laid out over all columns, so a seeded condition is built unblocked.
double chain_block_bytes();

bool build_one_condition_blocked(const CondRaw& cd,
                                 const std::vector<std::vector<u64> >& leafVal,
                                 int nLeaves, int nStates, int nz, int Nt, u64 p,
                                 std::vector<std::vector<u64> >& outRows, int threads) {
  // lanes per block: split over the threads, and bounded so that one block's Taylor
  // values stay within the budget of chain_block_bytes()
  double perLane = (double)(nLeaves + nStates + cd.op.size()) * (Nt + 1) * 8.0;
  int Bmem = std::max(1, (int)(chain_block_bytes() / perLane) - 1);
  int nbT = std::min(threads, nz / 16);
  int Bt = nbT >= 2 ? (nz + nbT - 1) / nbT : nz;
  int Bsz = std::min(Bt, Bmem);
  if (Bsz >= nz || cd.hasIcSeed)
    return build_one_condition(cd, leafVal, nLeaves, nStates, nz, nz + 1, Nt, p, outRows);
  int nb = (nz + Bsz - 1) / Bsz;
  std::vector<int> dualCol(nLeaves, -1);
  for (int L = 0; L < nLeaves; ++L)
    for (int c = 1; c <= nz; ++c)
      if (leafVal[L][c]) { dualCol[L] = c - 1; break; }
  int B = Bsz;
  std::vector<std::vector<std::vector<u64> > > rB(nb);
  std::vector<char> ok(nb, 1);
  int nThr = std::max(1, std::min(threads, nb));
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
  for (int b = 0; b < nb; ++b) {
    int s0 = b * B, s1 = std::min(nz, s0 + B), nzB = s1 - s0;
    if (nzB <= 0) continue;
    std::vector<std::vector<u64> > lv(nLeaves, std::vector<u64>(nzB + 1, 0));
    for (int L = 0; L < nLeaves; ++L) {
      lv[L][0] = leafVal[L][0];
      if (dualCol[L] >= s0 && dualCol[L] < s1) lv[L][1 + dualCol[L] - s0] = 1;
    }
    ok[b] = build_one_condition(cd, lv, nLeaves, nStates, nzB, nzB + 1, Nt, p, rB[b]);
  }
  for (int b = 0; b < nb; ++b)
    if (!ok[b] || rB[b].size() != rB[0].size()) return false;
  size_t r0 = outRows.size();
  outRows.resize(r0 + rB[0].size(), std::vector<u64>(nz, 0));
  for (int b = 0; b < nb; ++b) {
    int s0 = b * B;
    for (size_t i = 0; i < rB[b].size(); ++i)
      std::copy(rB[b][i].begin(), rB[b][i].end(), outRows[r0 + i].begin() + s0);
  }
  return true;
}

// One chain segment in thread-safe plain C++: the regime-substituted tape, its
// reduced constants, and the optional first-segment steady-state seed, IC tape
// and state-dose event map. Constants are reduced against the single call prime.
struct SegRaw {
  std::vector<int> op, a, b, stateSlots, fOut, gOut;
  std::vector<u64> cval;
  bool hasIcSeed, hasIcTape, hasEv;
  std::vector<std::vector<u64> > icSeed;
  std::vector<int> icOp, icA, icB, icOut;
  std::vector<u64> icCval;
  std::vector<int> evVarIdx, evMethod, evOp, evA, evB, evOut;
  std::vector<u64> evCval;
  bool hasTm;
  std::vector<int> tmOp, tmA, tmB, tmOut;
  std::vector<u64> tmCval;
};

SegRaw extract_seg_raw(List seg, int nStates, int w, u64 p) {
  SegRaw s;
  s.op = to_ivec(seg["op"]); s.a = to_ivec(seg["a"]); s.b = to_ivec(seg["b"]);
  s.stateSlots = to_ivec(seg["state_slots"]);
  s.fOut = to_ivec(seg["f_out"]); s.gOut = to_ivec(seg["g_out"]);
  std::vector<std::string> cnum = to_svec(seg["cnum"]), cden = to_svec(seg["cden"]);
  int nInstr = (int)s.op.size();
  s.cval.assign(nInstr, 0);
  for (int i = 0; i < nInstr; ++i)
    if (s.op[i] == OP_CONST)
      s.cval[i] = reduce_rational(cnum[i], cden[i], p);
  s.hasIcSeed = seg.containsElementNamed("ic_seed");
  if (s.hasIcSeed) {
    IntegerMatrix icSeed = as<IntegerMatrix>(seg["ic_seed"]);
    s.icSeed.assign(nStates, std::vector<u64>(w, 0));
    for (int i = 0; i < nStates; ++i)
      for (int c = 0; c < w; ++c) s.icSeed[i][c] = red(icSeed(i, c), p);
  }
  s.hasIcTape = seg.containsElementNamed("ic_op");
  if (s.hasIcTape) {
    s.icOp = to_ivec(seg["ic_op"]); s.icA = to_ivec(seg["ic_a"]);
    s.icB = to_ivec(seg["ic_b"]); s.icOut = to_ivec(seg["ic_out"]);
    std::vector<std::string> icCnum = to_svec(seg["ic_cnum"]), icCden = to_svec(seg["ic_cden"]);
    int nIc = (int)s.icOp.size();
    s.icCval.assign(nIc, 0);
    for (int i = 0; i < nIc; ++i)
      if (s.icOp[i] == OP_CONST)
        s.icCval[i] = reduce_rational(icCnum[i], icCden[i], p);
  }
  s.hasEv = seg.containsElementNamed("ev_var_idx");
  if (s.hasEv) {
    s.evVarIdx = to_ivec(seg["ev_var_idx"]); s.evMethod = to_ivec(seg["ev_method"]);
    s.evOp = to_ivec(seg["ev_op"]); s.evA = to_ivec(seg["ev_a"]); s.evB = to_ivec(seg["ev_b"]);
    s.evOut = to_ivec(seg["ev_out"]);
    std::vector<std::string> ecn = to_svec(seg["ev_cnum"]), ecd = to_svec(seg["ev_cden"]);
    int nEv = (int)s.evOp.size();
    s.evCval.assign(nEv, 0);
    for (int i = 0; i < nEv; ++i)
      if (s.evOp[i] == OP_CONST)
        s.evCval[i] = reduce_rational(ecn[i], ecd[i], p);
  }
  s.hasTm = seg.containsElementNamed("tm_op");
  if (s.hasTm) {
    s.tmOp = to_ivec(seg["tm_op"]); s.tmA = to_ivec(seg["tm_a"]); s.tmB = to_ivec(seg["tm_b"]);
    s.tmOut = to_ivec(seg["tm_out"]);
    std::vector<std::string> tcn = to_svec(seg["tm_cnum"]), tcd = to_svec(seg["tm_cden"]);
    int nTm = (int)s.tmOp.size();
    s.tmCval.assign(nTm, 0);
    for (int i = 0; i < nTm; ++i)
      if (s.tmOp[i] == OP_CONST)
        s.tmCval[i] = reduce_rational(tcn[i], tcd[i], p);
  }
  return s;
}

// A straight-line tape (CONST/ADD/MUL/INV) evaluated on polyseries blocks at time order
// 0: V holds one block per slot, leaves [0, nLeaves) filled. False on a zero reciprocal.
bool eval_tape_poly(std::vector<std::vector<u64> >& V, const std::vector<int>& op,
                    const std::vector<int>& a, const std::vector<int>& b,
                    const std::vector<u64>& cval, int nLeaves, const PolyBasis& pb,
                    u64 p) {
  size_t blk = (size_t)pb.nMono * pb.w;
  for (size_t i = 0; i < op.size(); ++i) {
    u64* o = V[nLeaves + i].data();
    std::fill(o, o + blk, (u64)0);
    switch (op[i]) {
      case OP_CONST: o[0] = cval[i]; break;
      case OP_ADD:
        for (size_t c = 0; c < blk; ++c) o[c] = addmod(V[a[i]][c], V[b[i]][c], p);
        break;
      case OP_MUL: poly_mul_acc(o, V[a[i]].data(), V[b[i]].data(), pb, p); break;
      case OP_SQRT: if (!poly_sqrt(o, V[a[i]].data(), pb, p)) return false; break;
      case OP_ROOT: if (!poly_root(o, V[a[i]].data(), (u64)b[i], pb, p)) return false; break;
      default: if (!poly_inv(o, V[a[i]].data(), pb, p)) return false;
    }
  }
  return true;
}

// Build one condition's chain (all segments) at a single point/prime, propagating
// the state across each gap as a formal power series and applying state-dose
// events on the carry. leafPt holds the leaf residues; the z-leaves seed a dual
// unit. Returns false on a vanishing denominator. Touches no R object.
bool build_one_chain(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                     int nz, int w, const std::vector<int>& dualCol,
                     const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                     std::vector<std::vector<u64> >& rows,
                     std::vector<std::vector<u64> >& srows,
                     const std::vector<std::vector<u64> >* ser = nullptr,
                     const std::vector<int>* lane = nullptr,
                     const std::vector<char>* movesIn = nullptr) {
  int instrBase = nLeaves + nStates;
  int nSeg = (int)segs.size();
  int K = nSeg - 1; if (K < 1) K = 1;
  // a continuation series in the leaves (coefficients of eps^1..eps^T) on one more axis,
  // read on the line with coefficient 1
  int serAxis = -1;
  if (ser) { serAxis = K; K += 1; }
  PolyBasis pb(K, Mtot, w);
  int nM = pb.nMono, blk = nM * w;

  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(blk, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)leafPt[L] % (long long)p; if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }
  if (ser) {
    std::vector<int> mu(K, 0);
    for (int L = 0; L < nLeaves; ++L)
      for (int j = 1; j <= (int)(*ser)[L].size() && j <= Mtot; ++j) {
        mu[serAxis] = j;
        leafVal[L][(size_t)pb.slotOf.at(mu) * w] = (*ser)[L][j - 1] % p;
      }
  }

  // each segment's left boundary time with its duals; the gaps between them set the
  // line Delta_i = lineC[i] * eps along which the rank is taken
  std::vector<std::vector<u64> > tau(nSeg, std::vector<u64>(w, 0));
  std::vector<char> moves(nSeg, 0);
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    if (!sg.hasTm) continue;
    std::vector<std::vector<u64> > tmVal(nLeaves + sg.tmOp.size(), std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L)
      for (int c = 0; c < w; ++c) tmVal[L][c] = leafVal[L][c];
    if (!eval_ic_order0(tmVal, sg.tmOp, sg.tmA, sg.tmB, sg.tmCval, nLeaves, w, p))
      return false;
    tau[sj] = tmVal[sg.tmOut[0]];
    for (int c = 1; c < w; ++c) if (tau[sj][c]) moves[sj] = 1;
  }
  // with a column block, which boundaries move is decided over all columns
  if (movesIn) moves = *movesIn;
  bool anyMoves = false;
  for (int sj = 0; sj < nSeg; ++sj) if (moves[sj]) anyMoves = true;
  std::vector<u64> lineC(K, 1);
  for (int i = 0; i + 1 < nSeg; ++i) {
    if (!segs[i].hasTm || !segs[i + 1].hasTm) continue;
    lineC[i] = submod(tau[i + 1][0], tau[i][0], p);
    if (!lineC[i]) return false;
  }
  // a moving boundary needs one time order more, for the time derivative
  // output jet just before a moving boundary, continued in the old regime
  std::vector<std::vector<u64> > leftJet;

  std::vector<std::vector<u64> > carry;
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& seg = segs[sj];
    int nInstr = (int)seg.op.size();
    int Sn = instrBase + nInstr;
    int NtS = Nt;
    int Nrun = (NtS > Mtot ? NtS : Mtot) + (anyMoves ? 1 : 0);
    std::vector<std::vector<u64> > val(Sn, std::vector<u64>((size_t)(Nrun + 1) * blk, 0));
    for (int L = 0; L < nLeaves; ++L)
      for (int c = 0; c < blk; ++c) val[L][c] = leafVal[L][c];

    std::vector<std::vector<u64> > icVal;
    bool useIcTape = (sj == 0) && !seg.hasIcSeed && seg.hasIcTape;
    if (useIcTape && ser) {
      icVal.assign(nLeaves + seg.icOp.size(), std::vector<u64>(blk, 0));
      for (int L = 0; L < nLeaves; ++L) icVal[L] = leafVal[L];
      if (!eval_tape_poly(icVal, seg.icOp, seg.icA, seg.icB, seg.icCval, nLeaves, pb, p))
        return false;
    } else if (useIcTape) {
      int nIc = (int)seg.icOp.size();
      icVal.assign(nLeaves + nIc, std::vector<u64>(w, 0));
      for (int L = 0; L < nLeaves; ++L)
        for (int c = 0; c < w; ++c) icVal[L][c] = leafVal[L][c];
      if (!eval_ic_order0(icVal, seg.icOp, seg.icA, seg.icB, seg.icCval, nLeaves, w, p))
        return false;
    }
    for (int i = 0; i < nStates; ++i) {
      u64* st = val[nLeaves + i].data();
      if (sj == 0 && seg.hasIcSeed) {
        st[0] = seg.icSeed[i][0];
        for (int c = 1; c < w; ++c) st[c] = seg.icSeed[i][lane ? 1 + (*lane)[c - 1] : c];
      } else if (useIcTape) {
        const std::vector<u64>& s = seg.icOut[i] < nLeaves ? leafVal[seg.icOut[i]]
                                                           : icVal[seg.icOut[i]];
        for (int c = 0; c < (ser ? blk : w); ++c) st[c] = s[c];
      } else {
        for (int c = 0; c < blk; ++c) st[c] = carry[i][c];
      }
    }
    size_t from = rows.size();
    if (!build_obs_rows_poly(val, seg.op, seg.a, seg.b, seg.cval, instrBase,
                             seg.stateSlots, seg.fOut, seg.gOut, pb, Nrun, NtS, p, rows,
                             moves[sj] ? tau[sj].data() + 1 : nullptr))
      return false;
    rows_to_line(rows, from, pb, lineC, nz, Mtot + 1, p, srows, lane != nullptr);
    // where the output is not analytic across a moving boundary, the boundary time is
    // itself read off the output: its gradient joins the rows
    if (sj > 0 && moves[sj]) {
      bool kink = leftJet.size() != seg.gOut.size();
      for (size_t gi = 0; gi < seg.gOut.size() && !kink; ++gi) {
        const u64* gr = val[seg.gOut[gi]].data();
        for (int k = 0; k <= NtS && !kink; ++k)
          for (int mo = 0; mo < nM; ++mo) {
            size_t at = (size_t)k * blk + (size_t)mo * w;
            if (gr[at] != leftJet[gi][at]) { kink = true; break; }
          }
      }
      if (kink) {
        std::vector<u64> row(tau[sj].begin() + 1, tau[sj].end());
        std::vector<u64> ser((size_t)nz * (Mtot + 1), 0);
        for (int c = 0; c < nz; ++c) ser[(size_t)c * (Mtot + 1)] = row[c];
        rows.push_back(row);
        srows.push_back(ser);
      }
    }
    if (sj < nSeg - 1) {
      int axis = sj;
      carry.assign(nStates, std::vector<u64>(blk, 0));
      for (int i = 0; i < nStates; ++i) {
        const u64* xv = val[seg.stateSlots[i]].data();
        for (int k = 0; k <= Mtot; ++k)
          for (int aMono = 0; aMono < nM; ++aMono) {
            int dst = pb.shiftSlot(aMono, axis, k);
            if (dst < 0) continue;
            const u64* src = xv + (size_t)k * blk + (size_t)aMono * w;
            u64* d = carry[i].data() + (size_t)dst * w;
            for (int c = 0; c < w; ++c) d[c] = addmod(d[c], src[c], p);
          }
      }
      // a gap whose length depends on the coordinates adds x'(Delta) dDelta
      if (moves[sj] || moves[sj + 1]) {
        std::vector<u64> dd(w, 0);
        for (int c = 1; c < w; ++c) dd[c] = submod(tau[sj + 1][c], tau[sj][c], p);
        for (int i = 0; i < nStates; ++i) {
          const u64* xv = val[seg.stateSlots[i]].data();
          for (int k = 0; k <= Mtot; ++k)
            for (int aMono = 0; aMono < nM; ++aMono) {
              int dst = pb.shiftSlot(aMono, axis, k);
              if (dst < 0) continue;
              u64 fk = mulmod((u64)(k + 1) % p,
                              xv[(size_t)(k + 1) * blk + (size_t)aMono * w], p);
              if (!fk) continue;
              u64* d = carry[i].data() + (size_t)dst * w;
              for (int c = 1; c < w; ++c)
                if (dd[c]) d[c] = addmod(d[c], mulmod(fk, dd[c], p), p);
            }
        }
      }
      if (moves[sj + 1]) {
        int NtL = Nt;
        std::vector<std::vector<u64> > valL(Sn, std::vector<u64>((size_t)(NtL + 1) * blk, 0));
        for (int L = 0; L < nLeaves; ++L)
          for (int c = 0; c < blk; ++c) valL[L][c] = leafVal[L][c];
        for (int i = 0; i < nStates; ++i)
          for (int c = 0; c < blk; ++c) valL[nLeaves + i][c] = carry[i][c];
        std::vector<std::vector<u64> > none;
        if (!build_obs_rows_poly(valL, seg.op, seg.a, seg.b, seg.cval, instrBase,
                                 seg.stateSlots, seg.fOut, seg.gOut, pb, NtL, -1, p, none))
          return false;
        leftJet.assign(seg.gOut.size(), std::vector<u64>());
        for (size_t gi = 0; gi < seg.gOut.size(); ++gi) leftJet[gi] = valL[seg.gOut[gi]];
      }
      const SegRaw& nxt = segs[sj + 1];
      if (nxt.hasEv && ser) {
        std::vector<std::vector<u64> > evVal(nLeaves + nxt.evOp.size(),
                                             std::vector<u64>(blk, 0));
        for (int L = 0; L < nLeaves; ++L) evVal[L] = leafVal[L];
        if (!eval_tape_poly(evVal, nxt.evOp, nxt.evA, nxt.evB, nxt.evCval, nLeaves, pb, p))
          return false;
        for (int e = 0; e < (int)nxt.evVarIdx.size(); ++e) {
          int s = nxt.evVarIdx[e], meth = nxt.evMethod[e];
          const std::vector<u64>& vv = evVal[nxt.evOut[e]];
          if (meth == 0) {
            carry[s] = vv;
          } else if (meth == 1) {
            for (int c = 0; c < blk; ++c) carry[s][c] = addmod(carry[s][c], vv[c], p);
          } else {
            std::vector<u64> scaled(blk, 0);
            poly_mul_acc(scaled.data(), carry[s].data(), vv.data(), pb, p);
            carry[s] = scaled;
          }
        }
      } else if (nxt.hasEv) {
        int nEv = (int)nxt.evOp.size();
        std::vector<std::vector<u64> > evVal(nLeaves + nEv, std::vector<u64>(w, 0));
        for (int L = 0; L < nLeaves; ++L)
          for (int c = 0; c < w; ++c) evVal[L][c] = leafVal[L][c];
        if (!eval_ic_order0(evVal, nxt.evOp, nxt.evA, nxt.evB, nxt.evCval, nLeaves, w, p))
          return false;
        for (int e = 0; e < (int)nxt.evVarIdx.size(); ++e) {
          int s = nxt.evVarIdx[e], meth = nxt.evMethod[e];
          const std::vector<u64>& vv = nxt.evOut[e] < nLeaves ? leafVal[nxt.evOut[e]]
                                                              : evVal[nxt.evOut[e]];
          u64* cv = carry[s].data();
          if (meth == 0) {
            std::fill(carry[s].begin(), carry[s].end(), (u64)0);
            for (int c = 0; c < w; ++c) cv[c] = vv[c] % p;
          } else if (meth == 1) {
            for (int c = 0; c < w; ++c) cv[c] = addmod(cv[c], vv[c], p);
          } else {
            std::vector<u64> scaled((size_t)nM * w, 0);
            for (int mo = 0; mo < nM; ++mo)
              dual_mul_acc(scaled.data() + (size_t)mo * w,
                           cv + (size_t)mo * w, vv.data(), w, p);
            carry[s] = scaled;
          }
        }
      }
    }
  }
  return true;
}

// Which segment boundaries move with the coordinates: the duals of each segment's start
// time over all columns.
bool chain_moves(const std::vector<SegRaw>& segs, int nLeaves, int w,
                 const std::vector<int>& dualCol, const std::vector<int>& leafPt, u64 p,
                 std::vector<char>& moves) {
  int nSeg = (int)segs.size();
  moves.assign(nSeg, 0);
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    if (!sg.hasTm) continue;
    std::vector<std::vector<u64> > tmVal(nLeaves + sg.tmOp.size(), std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L) {
      long long pv = (long long)leafPt[L] % (long long)p; if (pv < 0) pv += p;
      tmVal[L][0] = (u64)pv;
      if (dualCol[L] >= 0) tmVal[L][1 + dualCol[L]] = 1;
    }
    if (!eval_ic_order0(tmVal, sg.tmOp, sg.tmA, sg.tmB, sg.tmCval, nLeaves, w, p))
      return false;
    const std::vector<u64>& t = tmVal[sg.tmOut[0]];
    for (int c = 1; c < w; ++c) if (t[c]) moves[sj] = 1;
  }
  return true;
}

// Columns per block so that one block's Taylor values stay within `budget` bytes.
int chain_block_size(const std::vector<SegRaw>& segs, int nLeaves, int nStates, int nz,
                     int Nt, int Mtot, bool hasSer, double budget) {
  size_t Sn = 0;
  for (const SegRaw& sg : segs) Sn = std::max(Sn, (size_t)(nLeaves + nStates) + sg.op.size());
  int K = std::max(1, (int)segs.size() - 1) + (hasSer ? 1 : 0);
  double nMono = 1;                                    // binomial(K + Mtot, Mtot)
  for (int i = 1; i <= Mtot; ++i) nMono = nMono * (K + i) / i;
  double perLane = (double)Sn * (std::max(Nt, Mtot) + 2) * nMono * 8.0;
  int B = (int)(budget / perLane) - 1;
  return std::max(2, std::min(nz, B));
}

// Taylor values per column block, bytes; SYMIDENT_BLOCKMB overrides (MB)
double chain_block_bytes() {
  const char* e = std::getenv("SYMIDENT_BLOCKMB");
  return (e && *e) ? std::atof(e) * 1048576.0 : 256.0 * 1048576.0;
}

// Threads inside one chain when `outer` chains share `cores`; nesting is enabled for it.
int chain_inner_threads(int cores, int outer) {
  int o = std::max(1, std::min(cores, outer));
  int inner = std::max(1, cores / o);
#if defined(_OPENMP) && _OPENMP >= 200805
  if (inner > 1) omp_set_max_active_levels(2);
#endif
  return inner;
}

// build_one_chain over column blocks: each block holds the value and its own dual
// lanes, the rows are assembled over all columns. Rows and series rows keep the order
// of the unblocked call, zero series rows dropped.
bool build_chain_blocked(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                         int nz, const std::vector<int>& dualCol,
                         const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                         std::vector<std::vector<u64> >& rows,
                         std::vector<std::vector<u64> >& srows,
                         const std::vector<std::vector<u64> >* ser, int B, int threads) {
  if (B >= nz)
    return build_one_chain(segs, nLeaves, nStates, nz, nz + 1, dualCol, leafPt, Nt, Mtot,
                           p, rows, srows, ser);
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, nz + 1, dualCol, leafPt, p, moves)) return false;
  int nb = (nz + B - 1) / B, N = Mtot + 1;
  std::vector<std::vector<std::vector<u64> > > rB(nb), sB(nb);
  std::vector<char> ok(nb, 1);
  #pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(dynamic) \
          if (threads > 1)
  for (int b = 0; b < nb; ++b) {
    int s0 = b * B, s1 = std::min(nz, s0 + B), nzB = s1 - s0;
    std::vector<int> dc(nLeaves, -1), lane(nzB);
    for (int L = 0; L < nLeaves; ++L)
      if (dualCol[L] >= s0 && dualCol[L] < s1) dc[L] = dualCol[L] - s0;
    for (int j = 0; j < nzB; ++j) lane[j] = s0 + j;
    ok[b] = build_one_chain(segs, nLeaves, nStates, nzB, nzB + 1, dc, leafPt, Nt, Mtot, p,
                            rB[b], sB[b], ser, &lane, &moves);
  }
  for (int b = 0; b < nb; ++b)
    if (!ok[b] || rB[b].size() != rB[0].size() || sB[b].size() != sB[0].size()) return false;
  size_t r0 = rows.size();
  rows.resize(r0 + rB[0].size(), std::vector<u64>(nz, 0));
  for (int b = 0; b < nb; ++b) {
    int s0 = b * B;
    for (size_t i = 0; i < rB[b].size(); ++i)
      std::copy(rB[b][i].begin(), rB[b][i].end(), rows[r0 + i].begin() + s0);
  }
  for (size_t i = 0; i < sB[0].size(); ++i) {
    std::vector<u64> sr((size_t)nz * N, 0);
    bool any = false;
    for (int b = 0; b < nb; ++b) {
      int s0 = b * B, nzB = (int)(sB[b][i].size() / N);
      for (int c = 0; c < nzB; ++c)
        for (int d = 0; d < N; ++d) {
          u64 v = sB[b][i][(size_t)c * N + d];
          if (v) { sr[(size_t)(s0 + c) * N + d] = v; any = true; }
        }
    }
    if (any) srows.push_back(sr);
  }
  return true;
}

// Value of every output slot of a straight-line value tape (opcodes
// CONST/ADD/MUL/INV) at the integer leaves mod p. Returns false on a zero
// reciprocal.
[[maybe_unused]] bool seed_tape_value(const std::vector<int>& op, const std::vector<int>& a,
                     const std::vector<int>& b, const std::vector<u64>& cval,
                     const std::vector<u64>& leaf, int nLeaves, u64 p,
                     std::vector<u64>& val) {
  int n = (int)op.size();
  val.assign(nLeaves + n, 0);
  for (int L = 0; L < nLeaves; ++L) val[L] = leaf[L] % p;
  for (int i = 0; i < n; ++i) {
    int s = nLeaves + i;
    if (op[i] == OP_CONST) val[s] = cval[i] % p;
    else if (op[i] == OP_ADD) val[s] = addmod(val[a[i]], val[b[i]], p);
    else if (op[i] == OP_MUL) val[s] = mulmod(val[a[i]], val[b[i]], p);
    else { u64 av = val[a[i]] % p; if (av == 0) return false; val[s] = invmod(av, p); }
  }
  return true;
}

// Value and forward-mode duals (dual column j is d/d(leaf j), w = nLeaves + 1) of
// every output slot of a straight-line tape. Returns false on a zero reciprocal.
[[maybe_unused]] bool seed_tape_dual(const std::vector<int>& op, const std::vector<int>& a,
                    const std::vector<int>& b, const std::vector<u64>& cval,
                    const std::vector<u64>& leaf, int nLeaves, u64 p,
                    std::vector<std::vector<u64> >& val) {
  int n = (int)op.size(), w = nLeaves + 1;
  val.assign(nLeaves + n, std::vector<u64>(w, 0));
  for (int L = 0; L < nLeaves; ++L) { val[L][0] = leaf[L] % p; val[L][1 + L] = 1; }
  for (int i = 0; i < n; ++i) {
    int s = nLeaves + i;
    if (op[i] == OP_CONST) {
      val[s][0] = cval[i] % p;
    } else if (op[i] == OP_ADD) {
      const std::vector<u64>& A = val[a[i]]; const std::vector<u64>& B = val[b[i]];
      for (int c = 0; c < w; ++c) val[s][c] = addmod(A[c], B[c], p);
    } else if (op[i] == OP_MUL) {
      const std::vector<u64>& A = val[a[i]]; const std::vector<u64>& B = val[b[i]];
      val[s][0] = mulmod(A[0], B[0], p);
      for (int c = 1; c < w; ++c)
        val[s][c] = addmod(mulmod(A[0], B[c], p), mulmod(A[c], B[0], p), p);
    } else {
      const std::vector<u64>& A = val[a[i]];
      u64 a0 = A[0] % p; if (a0 == 0) return false;
      u64 vi = invmod(a0, p), vi2 = mulmod(vi, vi, p);
      val[s][0] = vi;
      for (int c = 1; c < w; ++c) val[s][c] = submod(0, mulmod(A[c], vi2, p), p);
    }
  }
  return true;
}

// Solve A X = B over GF(p), A n x n, B n x k. Returns false when A is singular.
[[maybe_unused]] bool seed_solve_mod(const std::vector<std::vector<u64> >& A,
                    const std::vector<std::vector<u64> >& B, u64 p,
                    std::vector<std::vector<u64> >& X) {
  int n = (int)A.size();
  int k = (n > 0 && !B.empty()) ? (int)B[0].size() : 0;
  std::vector<std::vector<u64> > M(n, std::vector<u64>(n + k, 0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) M[i][j] = A[i][j] % p;
    for (int j = 0; j < k; ++j) M[i][n + j] = B[i][j] % p;
  }
  for (int col = 0; col < n; ++col) {
    int piv = -1;
    for (int r = col; r < n; ++r) if (M[r][col] % p) { piv = r; break; }
    if (piv < 0) return false;
    std::swap(M[col], M[piv]);
    u64 inv = invmod(M[col][col] % p, p);
    for (int j = 0; j < n + k; ++j) M[col][j] = mulmod(M[col][j], inv, p);
    for (int r = 0; r < n; ++r) if (r != col && M[r][col] % p) {
      u64 f = M[r][col] % p;
      for (int j = 0; j < n + k; ++j)
        M[r][j] = submod(M[r][j], mulmod(f, M[col][j], p), p);
    }
  }
  X.assign(n, std::vector<u64>(k, 0));
  for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) X[i][j] = M[i][n + j];
  return true;
}

}  // namespace

// Multi-condition observability over a shared coordinate space. Each element of
// `tapes` is one condition on the same slot layout: leaves [0, nLeaves), states
// [nLeaves, nLeaves + nStates), then its own instructions. Initial values are a
// leaf slot (icLeaf >= 0) or a rational constant icNum/icDen. Leaves and duals are
// shared, so the stacked rows give the intersection nullspace. Conditions are built
// in parallel over `cores` threads. ok is FALSE on a vanishing denominator.
List symObsNullMulti(List tapes, int nLeaves, int nStates,
                     IntegerVector zSlots, IntegerVector point,
                     double pIn, int Nt, int cores = 1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1;
  int T = tapes.size();

  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;

  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p;
    if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }

  // serial pre-pass into plain C++: the parallel region must touch no R object
  std::vector<CondRaw> td(T);
  for (int t = 0; t < T; ++t) td[t] = extract_cond_raw(tapes[t], nStates, w);

  // parallel per-condition build: each thread owns its row block; a vanishing
  // denominator in any condition marks failure (no early return inside OpenMP).
  std::vector<std::vector<std::vector<u64> > > perRows(T);
  std::vector<char> okFlag(T, 1);
  int inner = chain_inner_threads(cores, T);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t)
    if (!build_one_condition_blocked(td[t], leafVal, nLeaves, nStates, nz, Nt, p,
                                     perRows[t], inner))
      okFlag[t] = 0;

  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);

  std::vector<std::vector<u64> > rows;
  for (int t = 0; t < T; ++t)
    for (size_t i = 0; i < perRows[t].size(); ++i)
      rows.push_back(std::move(perRows[t][i]));

  std::vector<int> pivots = rref_mod(rows, p);
  int rank = (int)pivots.size();
  IntegerMatrix R(rank, nz);
  for (int i = 0; i < rank; ++i)
    for (int c = 0; c < nz; ++c) R(i, c) = (int)rows[i][c];
  IntegerVector piv(rank);
  for (int i = 0; i < rank; ++i) piv[i] = pivots[i];

  return List::create(_["ok"] = true, _["R"] = R, _["pivots"] = piv,
                      _["rank"] = rank, _["dim"] = nz);
}

// Rank of the stacked single-segment rows of `tapes` up to each Lie order 0..Nt, by an
// incremental echelon in order-major row order. The rows up to order k do not depend
// on Nt, so ranks[k] equals the rank of symObsNullMulti at order k.
List symObsRankProfile(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                       IntegerVector point, double pIn, int Nt, int cores = 1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1;
  int T = tapes.size();
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p;
    if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }
  std::vector<CondRaw> td(T);
  for (int t = 0; t < T; ++t) td[t] = extract_cond_raw(tapes[t], nStates, w);
  std::vector<std::vector<std::vector<u64> > > perRows(T);
  std::vector<char> okFlag(T, 1);
  int inner = chain_inner_threads(cores, T);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t)
    if (!build_one_condition_blocked(td[t], leafVal, nLeaves, nStates, nz, Nt, p,
                                     perRows[t], inner))
      okFlag[t] = 0;
  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);

  // rows of a condition are ordered by output, then order
  std::vector<std::vector<u64> > basis;
  std::vector<int> pivCol;
  IntegerVector ranks(Nt + 1);
  for (int k = 0; k <= Nt; ++k) {
    for (int t = 0; t < T; ++t) {
      int nOut = (int)(perRows[t].size() / (size_t)(Nt + 1));
      for (int gi = 0; gi < nOut; ++gi) {
        std::vector<u64>& r = perRows[t][(size_t)gi * (Nt + 1) + k];
        for (size_t b = 0; b < basis.size(); ++b) {
          u64 f = r[pivCol[b]] % p;
          if (!f) continue;
          u64 nf = p - f;
          const std::vector<u64>& bb = basis[b];
          for (int c = pivCol[b]; c < nz; ++c)
            if (bb[c]) r[c] = addmod(r[c] % p, mulmod(nf, bb[c], p), p);
        }
        int pc = -1;
        for (int c = 0; c < nz; ++c) if (r[c] % p) { pc = c; break; }
        if (pc < 0) continue;
        u64 inv = invmod(r[pc] % p, p);
        for (int c = pc; c < nz; ++c) r[c] = mulmod(r[c] % p, inv, p);
        basis.push_back(std::move(r));
        pivCol.push_back(pc);
      }
    }
    ranks[k] = (int)basis.size();
  }
  return List::create(_["ok"] = true, _["ranks"] = ranks);
}

// Rank per Lie order of one chain at gap order 0 from one jet: the rows of every segment
// up to order Nt, reduced in order-major row order. A chain with a moving boundary
// returns ok = FALSE.
List symObsChainRankProfile(List chain, int nLeaves, int nStates, IntegerVector zSlots,
                            IntegerVector point, double pIn, int Nt, int threads = 1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1;
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<int> leafPt(nLeaves);
  for (int L = 0; L < nLeaves; ++L) leafPt[L] = point[L];
  int nSeg = chain.size();
  std::vector<SegRaw> segs;
  segs.reserve(nSeg);
  for (int sj = 0; sj < nSeg; ++sj) segs.push_back(extract_seg_raw(chain[sj], nStates, w, p));
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, w, dualCol, leafPt, p, moves))
    return List::create(_["ok"] = false);
  for (int sj = 0; sj < nSeg; ++sj)
    if (moves[sj]) return List::create(_["ok"] = false);
  int B = chain_block_size(segs, nLeaves, nStates, nz, Nt, 0, false, chain_block_bytes());
  std::vector<std::vector<u64> > rows, srows;
  if (!build_chain_blocked(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, 0, p, rows,
                           srows, nullptr, B, threads))
    return List::create(_["ok"] = false);
  // rows of a segment are ordered by output, then order
  std::vector<int> tag;
  tag.reserve(rows.size());
  for (int sj = 0; sj < nSeg; ++sj) {
    for (size_t gi = 0; gi < segs[sj].gOut.size(); ++gi)
      for (int k = 0; k <= Nt; ++k) tag.push_back(k);
  }
  if (tag.size() != rows.size()) return List::create(_["ok"] = false);
  std::vector<std::vector<int> > byOrder(Nt + 1);
  for (size_t i = 0; i < rows.size(); ++i) byOrder[tag[i]].push_back((int)i);
  std::vector<std::vector<u64> > basis;
  std::vector<int> pivCol;
  IntegerVector ranks(Nt + 1);
  for (int k = 0; k <= Nt; ++k) {
    for (size_t q = 0; q < byOrder[k].size(); ++q) {
      std::vector<u64>& r = rows[byOrder[k][q]];
      for (size_t b = 0; b < basis.size(); ++b) {
        u64 f = r[pivCol[b]] % p;
        if (!f) continue;
        u64 nf = p - f;
        const std::vector<u64>& bb = basis[b];
        for (int c = pivCol[b]; c < nz; ++c)
          if (bb[c]) r[c] = addmod(r[c] % p, mulmod(nf, bb[c], p), p);
      }
      int pc = -1;
      for (int c = 0; c < nz; ++c) if (r[c] % p) { pc = c; break; }
      if (pc < 0) continue;
      u64 inv = invmod(r[pc] % p, p);
      for (int c = pc; c < nz; ++c) r[c] = mulmod(r[c] % p, inv, p);
      basis.push_back(std::move(r));
      pivCol.push_back(pc);
    }
    ranks[k] = (int)basis.size();
  }
  return List::create(_["ok"] = true, _["ranks"] = ranks);
}

// Whether the direction `dir` (residues over the nz columns) annihilates every row of
// the single-segment tapes up to order Nt: the jets have one dual lane along it.
List symObsDirectional(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                       IntegerVector point, double pIn, int Nt, NumericVector dir,
                       int cores = 1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size();
  int T = tapes.size();
  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(2, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p;
    if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
  }
  for (int c = 0; c < nz; ++c) {
    long long dv = (long long)dir[c] % (long long)p;
    if (dv < 0) dv += p;
    leafVal[zSlots[c]][1] = (u64)dv;
  }
  std::vector<CondRaw> td(T);
  for (int t = 0; t < T; ++t) td[t] = extract_cond_raw(tapes[t], nStates, 2);
  std::vector<std::vector<std::vector<u64> > > perRows(T);
  std::vector<char> okFlag(T, 1);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t)
    if (!build_one_condition(td[t], leafVal, nLeaves, nStates, 1, 2, Nt, p, perRows[t]))
      okFlag[t] = 0;
  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);
  bool zero = true;
  for (int t = 0; t < T && zero; ++t)
    for (size_t i = 0; i < perRows[t].size(); ++i)
      if (perRows[t][i][0] % p) { zero = false; break; }
  return List::create(_["ok"] = true, _["zero"] = zero);
}

// Batched single-segment observability: the condition tapes at nB (point, prime)
// pairs (rows of `points`, entries of `primes`), one OpenMP task per pair, for the
// reconstruction's sample bank. Returns nB results shaped like symObsNullMulti; a
// vanishing denominator yields ok=FALSE.
List symObsNullBatch(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerMatrix points, NumericVector primes, int Nt,
                     int cores = 1) {
  int nz = zSlots.size(), w = nz + 1;
  int T = tapes.size();
  int nB = points.nrow();

  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;

  // copy out of R for the parallel region; constants are reduced per prime inside
  // the build, so one extraction serves every pair
  std::vector<int> pts((size_t)nB * nLeaves);
  for (int bi = 0; bi < nB; ++bi)
    for (int L = 0; L < nLeaves; ++L) pts[(size_t)bi * nLeaves + L] = points(bi, L);
  std::vector<u64> pr(nB);
  for (int bi = 0; bi < nB; ++bi) pr[bi] = (u64)primes[bi];

  std::vector<CondRaw> td(T);
  for (int t = 0; t < T; ++t) td[t] = extract_cond_raw(tapes[t], nStates, w);

  std::vector<char> okFlag(nB, 1);
  std::vector<std::vector<std::vector<u64> > > redRows(nB);
  std::vector<std::vector<int> > redPiv(nB);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int bi = 0; bi < nB; ++bi) {
    u64 p = pr[bi];
    std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L) {
      long long pv = (long long)pts[(size_t)bi * nLeaves + L] % (long long)p; if (pv < 0) pv += p;
      leafVal[L][0] = (u64)pv;
      if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
    }
    std::vector<std::vector<u64> > rows;
    bool ok = true;
    for (int t = 0; t < T && ok; ++t)
      if (!build_one_condition_blocked(td[t], leafVal, nLeaves, nStates, nz, Nt, p, rows, 1))
        ok = false;
    if (!ok) { okFlag[bi] = 0; continue; }
    std::vector<int> pivots = rref_mod(rows, p);
    int rank = (int)pivots.size();
    redRows[bi].assign(rank, std::vector<u64>(nz));
    for (int i = 0; i < rank; ++i)
      for (int c = 0; c < nz; ++c) redRows[bi][i][c] = rows[i][c];
    redPiv[bi] = pivots;
  }

  List out(nB);
  for (int bi = 0; bi < nB; ++bi) {
    if (!okFlag[bi]) { out[bi] = List::create(_["ok"] = false); continue; }
    int rank = (int)redPiv[bi].size();
    IntegerMatrix R(rank, nz);
    for (int i = 0; i < rank; ++i)
      for (int c = 0; c < nz; ++c) R(i, c) = (int)redRows[bi][i][c];
    IntegerVector piv(rank);
    for (int i = 0; i < rank; ++i) piv[i] = redPiv[bi][i];
    out[bi] = List::create(_["ok"] = true, _["R"] = R, _["pivots"] = piv,
                           _["rank"] = rank, _["dim"] = nz);
  }
  return out;
}

// Series echelon rows as an R integer matrix (rank x nz*N).
static IntegerMatrix series_matrix(const std::vector<std::vector<u64> >& S, int rank, int width) {
  IntegerMatrix M(rank, width);
  for (int i = 0; i < rank; ++i)
    for (int j = 0; j < width; ++j) M(i, j) = (int)S[i][j];
  return M;
}

// Where the series rank falls below the rank of the stacked coefficient rows, some
// kernel direction moves with the gap lengths. If the series kernel is polynomial in
// eps it is replaced by its value at eps = 1, the actual gaps: R, pivots and rank then
// describe that kernel, and `atOne` says so.
static void kernel_at_one_or_keep(const std::vector<std::vector<u64> >& ser, int rankS,
                                  int nz, int N, u64 p,
                                  std::vector<std::vector<u64> >& R, std::vector<int>& piv,
                                  bool& atOne) {
  atOne = false;
  if (rankS >= (int)piv.size()) return;
  std::vector<std::vector<u64> > K;
  if (!series_kernel_at_one(ser, rankS, nz, N, p, K)) return;
  annihilator_rows(K, nz, p, R, piv);
  atOne = true;
}

static List chain_result(const std::vector<std::vector<u64> >& R, const std::vector<int>& piv,
                         int nz, int rankS, const std::vector<std::vector<u64> >& ser,
                         int N, bool atOne) {
  int rank = (int)piv.size();
  IntegerMatrix Rm(rank, nz);
  for (int i = 0; i < rank; ++i)
    for (int c = 0; c < nz; ++c) Rm(i, c) = (int)R[i][c];
  IntegerVector pv(piv.begin(), piv.end());
  return List::create(_["ok"] = true, _["R"] = Rm, _["pivots"] = pv,
                      _["rank"] = rank, _["dim"] = nz, _["rank_s"] = rankS,
                      _["S"] = series_matrix(ser, rankS, nz * N), _["N"] = N,
                      _["at_one"] = atOne);
}

// Multi-segment observability with exact generic timing across events. `chains`
// holds one ordered segment list per condition, each segment a regime-substituted
// tape; the first may hold an IC seed. The nSeg-1 gaps are formal lengths Delta t_i:
// the state crosses a gap by promoting its local-time Taylor coefficients into that
// gap's axis. Each segment's output jet, to total gap-degree Mtot, contributes
// z-gradient rows per gap monomial; a direction is non-identifiable iff every
// monomial annihilates it, so the stacked rows are reduced once over GF(p).
List symObsNullChain(List chains, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int Mtot, int cores = 1,
                     IntegerVector NtChain = IntegerVector::create(),
                     SEXP pointSer = R_NilValue) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1;
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  int T = chains.size();

  // serial pre-pass into plain C++: the parallel build must touch no R object
  std::vector<int> leafPt(nLeaves);
  for (int L = 0; L < nLeaves; ++L) leafPt[L] = point[L];
  // a chain whose rank saturated at a lower Lie order is built only to that order:
  // its rows at Nt add nothing, and the jet cost grows with the order
  std::vector<int> ntOf(T, Nt);
  if (NtChain.size() == T)
    for (int t = 0; t < T; ++t) ntOf[t] = std::min(Nt, (int)NtChain[t]);
  std::vector<std::vector<SegRaw> > chainsRaw(T);
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    chainsRaw[t].reserve(nSeg);
    for (int sj = 0; sj < nSeg; ++sj)
      chainsRaw[t].push_back(extract_seg_raw(segs[sj], nStates, w, p));
  }
  // an optional continuation series of the leaves (nLeaves x T)
  std::vector<std::vector<u64> > leafSer;
  if (!Rf_isNull(pointSer)) {
    IntegerMatrix M = as<IntegerMatrix>(pointSer);
    if (M.nrow() == nLeaves && M.ncol() > 0) {
      leafSer.assign(nLeaves, std::vector<u64>(M.ncol(), 0));
      for (int L = 0; L < nLeaves; ++L)
        for (int j = 0; j < M.ncol(); ++j) leafSer[L][j] = red(M(L, j), p);
    }
  }

  // parallel per-condition build: each thread owns its row block; a vanishing
  // denominator in any condition marks failure (no early return inside OpenMP).
  std::vector<std::vector<std::vector<u64> > > perRows(T), perSer(T);
  std::vector<char> okFlag(T, 1);
  int inner = chain_inner_threads(cores, T);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t) {
    int B = chain_block_size(chainsRaw[t], nLeaves, nStates, nz, ntOf[t], Mtot,
                             !leafSer.empty(), chain_block_bytes());
    if (!build_chain_blocked(chainsRaw[t], nLeaves, nStates, nz, dualCol, leafPt,
                             ntOf[t], Mtot, p, perRows[t], perSer[t],
                             leafSer.empty() ? nullptr : &leafSer, B, inner))
      okFlag[t] = 0;
  }

  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);

  std::vector<std::vector<u64> > rows;
  for (int t = 0; t < T; ++t)
    for (size_t i = 0; i < perRows[t].size(); ++i)
      rows.push_back(std::move(perRows[t][i]));

  std::vector<int> pivots = rref_mod(rows, p);
  rows.resize(pivots.size());

  int N = Mtot + 1;
  std::vector<std::vector<u64> > ser;
  for (int t = 0; t < T; ++t)
    for (size_t i = 0; i < perSer[t].size(); ++i) ser.push_back(std::move(perSer[t][i]));
  int rankS = rref_series_mod(ser, nz, N, p, std::vector<char>(nz, 1));
  bool atOne;
  kernel_at_one_or_keep(ser, rankS, nz, N, p, rows, pivots, atOne);
  return chain_result(rows, pivots, nz, rankS, ser, N, atOne);
}

// Rank of each segment's own rows at orders 0..Nt, at gap order 0, with a unit dual
// column per state at the segment start and, where the start time depends on the
// coordinates, a column -(k+1) y_{k+1}. `stateVals` (nSeg x nStates) replaces the start
// states. Also returns the start states and the field at them.
List symSegmentRanks(List segs, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt,
                     IntegerMatrix stateVals = IntegerMatrix(0, 0), int threads = 1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), nSeg = segs.size();
  int wA = nz + nStates + 2, nzA = wA - 1, w0 = nz + 1;
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<u64> leaf(nLeaves);
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p; if (pv < 0) pv += p;
    leaf[L] = (u64)pv;
  }
  bool givenStates = stateVals.nrow() == nSeg && stateVals.ncol() == nStates;
  IntegerMatrix ranks(nSeg, Nt + 1), start(nSeg, nStates), field(nSeg, nStates);
  LogicalVector moves(nSeg);
  std::vector<u64> q(nStates, 0);
  for (int sj = 0; sj < nSeg; ++sj) {
    SegRaw seg = extract_seg_raw(segs[sj], nStates, 1, p);
    // start state at gap order 0
    if (sj == 0) {
      if (seg.hasIcSeed) {
        for (int i = 0; i < nStates; ++i) q[i] = seg.icSeed[i][0];
      } else if (seg.hasIcTape) {
        std::vector<std::vector<u64> > icVal(nLeaves + seg.icOp.size(), std::vector<u64>(1, 0));
        for (int L = 0; L < nLeaves; ++L) icVal[L][0] = leaf[L];
        if (!eval_ic_order0(icVal, seg.icOp, seg.icA, seg.icB, seg.icCval, nLeaves, 1, p))
          return List::create(_["ok"] = false);
        for (int i = 0; i < nStates; ++i) q[i] = icVal[seg.icOut[i]][0];
      } else {
        return List::create(_["ok"] = false);
      }
    } else if (seg.hasEv) {
      std::vector<std::vector<u64> > evVal(nLeaves + seg.evOp.size(), std::vector<u64>(1, 0));
      for (int L = 0; L < nLeaves; ++L) evVal[L][0] = leaf[L];
      if (!eval_ic_order0(evVal, seg.evOp, seg.evA, seg.evB, seg.evCval, nLeaves, 1, p))
        return List::create(_["ok"] = false);
      for (int e = 0; e < (int)seg.evVarIdx.size(); ++e) {
        int st = seg.evVarIdx[e], meth = seg.evMethod[e];
        u64 v = evVal[seg.evOut[e]][0] % p;
        if (meth == 0) q[st] = v;
        else if (meth == 1) q[st] = addmod(q[st], v, p);
        else q[st] = mulmod(q[st], v, p);
      }
    }
    // a segment whose start time depends on the coordinates
    if (seg.hasTm) {
      std::vector<std::vector<u64> > tmVal(nLeaves + seg.tmOp.size(), std::vector<u64>(w0, 0));
      for (int L = 0; L < nLeaves; ++L) {
        tmVal[L][0] = leaf[L];
        if (dualCol[L] >= 0) tmVal[L][1 + dualCol[L]] = 1;
      }
      if (!eval_ic_order0(tmVal, seg.tmOp, seg.tmA, seg.tmB, seg.tmCval, nLeaves, w0, p))
        return List::create(_["ok"] = false);
      for (int c = 1; c < w0; ++c) if (tmVal[seg.tmOut[0]][c]) moves[sj] = true;
    }
    // rows in (coordinates, states), the seeded lanes split into column blocks; the
    // last column (time) is filled from the values below
    int nInstr = (int)seg.op.size(), instrBase = nLeaves + nStates;
    int nLane = nz + nStates;
    int nb = std::max(1, std::min(threads, nLane / 16));
    int B = (nLane + nb - 1) / nb;
    double perLane = (double)(instrBase + nInstr) * (Nt + 2) * 8.0;
    B = std::min(B, std::max(1, (int)(chain_block_bytes() / perLane) - 1));
    nb = (nLane + B - 1) / B;            // no empty trailing block
    int nThrS = std::max(1, std::min(threads, nb));
    std::vector<u64> qs(nStates);
    for (int i = 0; i < nStates; ++i) {
      qs[i] = givenStates ? red(stateVals(sj, i), p) : q[i];
      start(sj, i) = (int)qs[i];
    }
    int nOutS = (int)seg.gOut.size();
    std::vector<std::vector<std::vector<u64> > > rB(nb);
    std::vector<u64> fVal(nStates, 0), yNext((size_t)nOutS * (Nt + 1), 0);
    std::vector<char> okB(nb, 1);
    #pragma omp parallel for num_threads(nThrS) schedule(dynamic) if (nThrS > 1)
    for (int b = 0; b < nb; ++b) {
      int s0 = b * B, s1 = std::min(nLane, s0 + B), wb = s1 - s0 + 1;
      if (wb <= 1) continue;
      PolyBasis pb(1, 0, wb);
      std::vector<std::vector<u64> > val(instrBase + nInstr,
                                         std::vector<u64>((size_t)(Nt + 2) * wb, 0));
      for (int L = 0; L < nLeaves; ++L) {
        val[L][0] = leaf[L];
        if (dualCol[L] >= s0 && dualCol[L] < s1) val[L][1 + dualCol[L] - s0] = 1;
      }
      for (int i = 0; i < nStates; ++i) {
        val[nLeaves + i][0] = qs[i];
        if (nz + i >= s0 && nz + i < s1) val[nLeaves + i][1 + nz + i - s0] = 1;
      }
      if (!build_obs_rows_poly(val, seg.op, seg.a, seg.b, seg.cval, instrBase,
                               seg.stateSlots, seg.fOut, seg.gOut, pb, Nt + 1, Nt, p, rB[b])) {
        okB[b] = 0; continue;
      }
      if (b == 0) {
        for (int i = 0; i < nStates && i < (int)seg.fOut.size(); ++i)
          fVal[i] = val[seg.fOut[i]][0] % p;
        for (int gi = 0; gi < nOutS; ++gi)
          for (int k = 0; k <= Nt; ++k)
            yNext[(size_t)gi * (Nt + 1) + k] = val[seg.gOut[gi]][(size_t)(k + 1) * wb] % p;
      }
    }
    for (int b = 0; b < nb; ++b)
      if (!okB[b] || rB[b].size() != rB[0].size()) return List::create(_["ok"] = false);
    std::vector<std::vector<u64> > rows(rB[0].size(), std::vector<u64>(nzA, 0));
    for (int b = 0; b < nb; ++b)
      for (size_t r = 0; r < rB[b].size(); ++r)
        std::copy(rB[b][r].begin(), rB[b][r].end(), rows[r].begin() + b * B);
    for (int i = 0; i < nStates && i < (int)seg.fOut.size(); ++i)
      field(sj, i) = (int)fVal[i];
    if (moves[sj])
      for (int gi = 0; gi < nOutS; ++gi)
        for (int k = 0; k <= Nt; ++k) {
          u64 y1 = yNext[(size_t)gi * (Nt + 1) + k];
          rows[(size_t)gi * (Nt + 1) + k][nzA - 1] = submod(0, mulmod((u64)(k + 1) % p, y1, p), p);
        }
    // rows are ordered by output, then order; rank up to each order, incrementally
    int nOut = (int)seg.gOut.size();
    std::vector<std::vector<u64> > basis;
    std::vector<int> pivCol;
    for (int k = 0; k <= Nt; ++k) {
      for (int gi = 0; gi < nOut; ++gi) {
        std::vector<u64> r = rows[(size_t)gi * (Nt + 1) + k];
        for (size_t b = 0; b < basis.size(); ++b) {
          u64 f = r[pivCol[b]] % p;
          if (!f) continue;
          u64 nf = p - f;
          for (int c = 0; c < nzA; ++c)
            if (basis[b][c]) r[c] = addmod(r[c] % p, mulmod(nf, basis[b][c], p), p);
        }
        int pc = -1;
        for (int c = 0; c < nzA; ++c) if (r[c] % p) { pc = c; break; }
        if (pc < 0) continue;
        u64 inv = invmod(r[pc] % p, p);
        for (int c = 0; c < nzA; ++c) r[c] = mulmod(r[c] % p, inv, p);
        basis.push_back(r);
        pivCol.push_back(pc);
      }
      ranks(sj, k) = (int)basis.size();
    }
    if (givenStates) for (int i = 0; i < nStates; ++i) q[i] = red(stateVals(sj, i), p);
  }
  return List::create(_["ok"] = true, _["ranks"] = ranks, _["start"] = start,
                      _["field"] = field, _["moves"] = moves);
}

// Batched twin of per-condition symObsNullChain calls on the joint/equilibrate path:
// one OpenMP task per (chain, seed, prime) triple. evalChain[e] selects the chain,
// `seeds` (nB x nLeaves) holds the leaf residues with states written in, `primes`
// one prime per eval. Segments are extracted once per (chain, distinct prime).
// Returns nB results shaped like symObsNullChain; a vanishing denominator yields
// ok = FALSE. `NtEval` (one per eval) caps an eval's Lie order below Nt.
List symObsNullChainSeedBatch(List chains, IntegerVector evalChain,
                              IntegerMatrix seeds, NumericVector primes,
                              int nLeaves, int nStates, IntegerVector zSlots,
                              int Nt, int Mtot, int cores = 1,
                              IntegerVector NtEval = IntegerVector::create(),
                              SEXP seedSer = R_NilValue) {
  int nz = zSlots.size(), w = nz + 1;
  int T = chains.size();
  int nB = evalChain.size();
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;

  // primes and their distinct set (extraction is per distinct prime)
  std::vector<u64> pr(nB);
  for (int e = 0; e < nB; ++e) pr[e] = (u64)primes[e];
  std::vector<u64> distinct;
  std::vector<int> primeIdx(nB);
  for (int e = 0; e < nB; ++e) {
    int idx = -1;
    for (size_t k = 0; k < distinct.size(); ++k) if (distinct[k] == pr[e]) { idx = (int)k; break; }
    if (idx < 0) { idx = (int)distinct.size(); distinct.push_back(pr[e]); }
    primeIdx[e] = idx;
  }
  int nPr = (int)distinct.size();
  int N = Mtot + 1;

  // serial pre-pass per distinct prime: the parallel build must touch no R object
  std::vector<std::vector<std::vector<SegRaw> > >
      segsRaw(nPr, std::vector<std::vector<SegRaw> >(T));
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    for (int pi = 0; pi < nPr; ++pi) {
      segsRaw[pi][t].reserve(nSeg);
      for (int sj = 0; sj < nSeg; ++sj)
        segsRaw[pi][t].push_back(extract_seg_raw(segs[sj], nStates, w, distinct[pi]));
    }
  }

  std::vector<int> ec(nB), ntOf(nB, Nt);
  for (int e = 0; e < nB; ++e) ec[e] = evalChain[e];
  // per eval an optional continuation series of the leaves (nLeaves x T), reduced here
  std::vector<std::vector<std::vector<u64> > > serE(nB);
  std::vector<char> hasSer(nB, 0);
  if (!Rf_isNull(seedSer)) {
    List sl(seedSer);
    for (int e = 0; e < nB && e < sl.size(); ++e) {
      if (Rf_isNull(sl[e])) continue;
      IntegerMatrix M = as<IntegerMatrix>(sl[e]);
      if (M.nrow() != nLeaves || M.ncol() == 0) continue;
      hasSer[e] = 1;
      serE[e].assign(nLeaves, std::vector<u64>(M.ncol(), 0));
      for (int L = 0; L < nLeaves; ++L)
        for (int j = 0; j < M.ncol(); ++j) serE[e][L][j] = red(M(L, j), pr[e]);
    }
  }
  if (NtEval.size() == nB)
    for (int e = 0; e < nB; ++e) ntOf[e] = std::min(Nt, (int)NtEval[e]);
  std::vector<int> sd((size_t)nB * nLeaves);
  for (int e = 0; e < nB; ++e)
    for (int L = 0; L < nLeaves; ++L) sd[(size_t)e * nLeaves + L] = seeds(e, L);

  std::vector<char> okFlag(nB, 1);
  std::vector<std::vector<std::vector<u64> > > redRows(nB);
  std::vector<std::vector<int> > redPiv(nB);
  std::vector<std::vector<std::vector<u64> > > redSer(nB);
  std::vector<int> rankS(nB, 0);
  std::vector<char> atOneE(nB, 0);
  int inner = chain_inner_threads(cores, nB);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int e = 0; e < nB; ++e) {
    u64 p = pr[e];
    std::vector<int> leafPt(nLeaves);
    for (int L = 0; L < nLeaves; ++L) leafPt[L] = sd[(size_t)e * nLeaves + L];
    std::vector<std::vector<u64> > rows;
    const std::vector<SegRaw>& segs = segsRaw[primeIdx[e]][ec[e]];
    int B = chain_block_size(segs, nLeaves, nStates, nz, ntOf[e], Mtot, hasSer[e],
                             chain_block_bytes());
    if (!build_chain_blocked(segs, nLeaves, nStates, nz, dualCol, leafPt, ntOf[e], Mtot, p,
                             rows, redSer[e], hasSer[e] ? &serE[e] : nullptr, B, inner)) {
      okFlag[e] = 0; continue; }
    std::vector<int> pivots = rref_mod(rows, p);
    rows.resize(pivots.size());
    rankS[e] = rref_series_mod(redSer[e], nz, N, p, std::vector<char>(nz, 1));
    bool atOne;
    kernel_at_one_or_keep(redSer[e], rankS[e], nz, N, p, rows, pivots, atOne);
    atOneE[e] = atOne;
    redRows[e] = rows;
    redPiv[e] = pivots;
  }

  List out(nB);
  for (int e = 0; e < nB; ++e) {
    if (!okFlag[e]) { out[e] = List::create(_["ok"] = false); continue; }
    out[e] = chain_result(redRows[e], redPiv[e], nz, rankS[e], redSer[e], N, atOneE[e]);
  }
  return out;
}

// Batched twin of per-point symObsNullChain calls on the plain gap path: every
// (point, chain) jet in one OpenMP loop, then each point's stacked rows reduced as
// symObsNullChain does, again in parallel. `points` (nB x nLeaves) holds one leaf
// point per evaluation and `primes` its prime; NtChain caps a chain's Lie order as in
// symObsNullChain. Segments are extracted once per (chain, distinct prime).
List symObsNullChainPointBatch(List chains, int nLeaves, int nStates, IntegerVector zSlots,
                               IntegerMatrix points, NumericVector primes, int Nt, int Mtot,
                               int cores = 1, IntegerVector NtChain = IntegerVector::create()) {
  int nz = zSlots.size(), w = nz + 1;
  int T = chains.size();
  int nB = points.nrow();
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<int> ntOf(T, Nt);
  if (NtChain.size() == T)
    for (int t = 0; t < T; ++t) ntOf[t] = std::min(Nt, (int)NtChain[t]);

  std::vector<u64> pr(nB);
  for (int e = 0; e < nB; ++e) pr[e] = (u64)primes[e];
  std::vector<u64> distinct;
  std::vector<int> primeIdx(nB);
  for (int e = 0; e < nB; ++e) {
    int idx = -1;
    for (size_t k = 0; k < distinct.size(); ++k) if (distinct[k] == pr[e]) { idx = (int)k; break; }
    if (idx < 0) { idx = (int)distinct.size(); distinct.push_back(pr[e]); }
    primeIdx[e] = idx;
  }
  int nPr = (int)distinct.size();
  int N = Mtot + 1;

  // serial pre-pass: the parallel part must touch no R object
  std::vector<std::vector<std::vector<SegRaw> > >
      segsRaw(nPr, std::vector<std::vector<SegRaw> >(T));
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    for (int pi = 0; pi < nPr; ++pi) {
      segsRaw[pi][t].reserve(nSeg);
      for (int sj = 0; sj < nSeg; ++sj)
        segsRaw[pi][t].push_back(extract_seg_raw(segs[sj], nStates, w, distinct[pi]));
    }
  }
  std::vector<int> pts((size_t)nB * nLeaves);
  for (int e = 0; e < nB; ++e)
    for (int L = 0; L < nLeaves; ++L) pts[(size_t)e * nLeaves + L] = points(e, L);

  // every (point, chain) jet
  std::vector<std::vector<std::vector<u64> > > rowsET((size_t)nB * T), serET((size_t)nB * T);
  std::vector<char> okET((size_t)nB * T, 1);
  long nTask = (long)nB * T;
  int inner = chain_inner_threads(cores, (int)std::min<long>(nTask, 1 << 20));
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nTask > 1)
  for (long k = 0; k < nTask; ++k) {
    int e = (int)(k / T), t = (int)(k % T);
    std::vector<int> leafPt(pts.begin() + (size_t)e * nLeaves,
                            pts.begin() + (size_t)(e + 1) * nLeaves);
    const std::vector<SegRaw>& segs = segsRaw[primeIdx[e]][t];
    int B = chain_block_size(segs, nLeaves, nStates, nz, ntOf[t], Mtot, false,
                             chain_block_bytes());
    if (!build_chain_blocked(segs, nLeaves, nStates, nz, dualCol, leafPt, ntOf[t], Mtot,
                             pr[e], rowsET[k], serET[k], nullptr, B, inner))
      okET[k] = 0;
  }

  // per point: stack, reduce, series rank
  std::vector<char> okE(nB, 1), atOneE(nB, 0);
  std::vector<std::vector<std::vector<u64> > > redRows(nB), redSer(nB);
  std::vector<std::vector<int> > redPiv(nB);
  std::vector<int> rankS(nB, 0);
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int e = 0; e < nB; ++e) {
    std::vector<std::vector<u64> > rows, ser;
    for (int t = 0; t < T; ++t) {
      size_t k = (size_t)e * T + t;
      if (!okET[k]) { okE[e] = 0; break; }
      for (size_t i = 0; i < rowsET[k].size(); ++i) rows.push_back(std::move(rowsET[k][i]));
      for (size_t i = 0; i < serET[k].size(); ++i) ser.push_back(std::move(serET[k][i]));
    }
    if (!okE[e]) continue;
    u64 p = pr[e];
    std::vector<int> pivots = rref_mod(rows, p);
    rows.resize(pivots.size());
    rankS[e] = rref_series_mod(ser, nz, N, p, std::vector<char>(nz, 1));
    bool atOne;
    kernel_at_one_or_keep(ser, rankS[e], nz, N, p, rows, pivots, atOne);
    atOneE[e] = atOne;
    redRows[e] = std::move(rows);
    redPiv[e] = std::move(pivots);
    redSer[e] = std::move(ser);
  }

  List out(nB);
  for (int e = 0; e < nB; ++e) {
    if (!okE[e]) { out[e] = List::create(_["ok"] = false); continue; }
    out[e] = chain_result(redRows[e], redPiv[e], nz, rankS[e], redSer[e], N, atOneE[e]);
  }
  return out;
}

// Rank over GF(p)((eps)) of series rows S (rows x nz*N, entry c at columns
// c*N .. c*N+N-1), restricted to the 0-based columns `cols` (all when empty).
// Returns the rank and the echelon rows; with support = TRUE also the columns a
// kernel vector can move, those whose removal keeps the rank.
List symSeriesRank(IntegerMatrix S, int nz, int N, double pIn, IntegerVector cols,
                   bool support = false, int atOneBelow = -1, int cores = 1,
                   IntegerVector rowPrec = IntegerVector()) {
  u64 p = (u64)pIn;
  int nr = S.nrow();
  std::vector<char> mask(nz, cols.size() ? 0 : 1);
  for (int i = 0; i < cols.size(); ++i) mask[cols[i]] = 1;
  std::vector<std::vector<u64> > A(nr, std::vector<u64>((size_t)nz * N));
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nz * N; ++j) A[i][j] = red(S(i, j), p);
  std::vector<int> P(nr, N);
  for (int i = 0; i < rowPrec.size() && i < nr; ++i) P[i] = std::min(N, (int)rowPrec[i]);
  int rank = rref_series_mod(A, nz, N, p, mask, cores, false, &P);
  List out = List::create(_["rank"] = rank, _["S"] = series_matrix(A, rank, nz * N));
  // atOneBelow: the rank of the stacked coefficient rows; below it, try the kernel at
  // eps = 1 and return its reduced rows
  if (atOneBelow > rank && !cols.size()) {
    std::vector<std::vector<u64> > K, R;
    std::vector<int> piv;
    if (series_kernel_at_one(A, rank, nz, N, p, K)) {
      annihilator_rows(K, nz, p, R, piv);
      IntegerMatrix Rm((int)piv.size(), nz);
      for (size_t i = 0; i < piv.size(); ++i)
        for (int c = 0; c < nz; ++c) Rm(i, c) = (int)R[i][c];
      out["R"] = Rm;
      out["pivots"] = IntegerVector(piv.begin(), piv.end());
    }
  }
  if (support) {
    std::vector<char> keep(nz, 0);
    #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic)
    for (int c = 0; c < nz; ++c) {
      if (!mask[c]) continue;
      std::vector<std::vector<u64> > B(A.begin(), A.begin() + rank);
      std::vector<char> m2 = mask; m2[c] = 0;
      keep[c] = rref_series_mod(B, nz, N, p, m2) == rank;
    }
    std::vector<int> supp;
    for (int c = 0; c < nz; ++c) if (keep[c]) supp.push_back(c);
    out["support"] = IntegerVector(supp.begin(), supp.end());
  }
  return out;
}

// Series rows S (rows x nz*N) reduced on the columns `first` (0-based); the rows without
// a pivot there, on the other columns, are Q (nOther*N wide, zero rows dropped) with
// their precisions prec. Returns r1 (the pivots in `first`), Q and prec.
List symSeriesProject(IntegerMatrix S, int nz, int N, double pIn, IntegerVector first,
                      int cores = 1) {
  u64 p = (u64)pIn;
  int nr = S.nrow();
  std::vector<char> mask(nz, 0);
  for (int i = 0; i < first.size(); ++i) mask[first[i]] = 1;
  std::vector<std::vector<u64> > A(nr, std::vector<u64>((size_t)nz * N));
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nz * N; ++j) A[i][j] = red(S(i, j), p);
  std::vector<int> P(nr, N), CP;
  int r1 = rref_series_mod(A, nz, N, p, mask, cores, true, &P, &CP);
  std::vector<int> other;
  for (int c = 0; c < nz; ++c) if (!mask[c]) other.push_back(c);
  int no = (int)other.size();
  std::vector<int> keep;
  for (int i = r1; i < nr; ++i) {
    bool nzRow = false;
    for (int k = 0; k < no && !nzRow; ++k)
      for (int d = 0; d < CP[i] && !nzRow; ++d) nzRow = A[i][(size_t)other[k] * N + d] != 0;
    if (nzRow) keep.push_back(i);
  }
  IntegerMatrix Q((int)keep.size(), no * N);
  IntegerVector qp((int)keep.size());
  for (size_t i = 0; i < keep.size(); ++i) {
    qp[i] = CP[keep[i]];
    for (int k = 0; k < no; ++k)
      for (int d = 0; d < CP[keep[i]]; ++d)
        Q((int)i, k * N + d) = (int)A[keep[i]][(size_t)other[k] * N + d];
  }
  return List::create(_["r1"] = r1, _["Q"] = Q, _["prec"] = qp);
}

// Solve A x = b over GF(p); returns the solution (free variables zero) or
// R_NilValue when the system is inconsistent.
SEXP symSolveMod(IntegerMatrix A, IntegerVector b, double pIn) {
  u64 p = (u64)pIn;
  int nr = A.nrow(), nc = A.ncol();
  std::vector<std::vector<u64> > aug(nr, std::vector<u64>(nc + 1));
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nc; ++j) aug[i][j] = red(A(i, j), p);
    aug[i][nc] = red(b[i], p);
  }
  std::vector<int> pivots = rref_mod(aug, p);
  for (size_t i = 0; i < pivots.size(); ++i)
    if (pivots[i] == nc) return R_NilValue;
  IntegerVector x(nc);
  for (size_t i = 0; i < pivots.size(); ++i)
    if (pivots[i] < nc) x[pivots[i]] = (int)aug[i][nc];
  return x;
}

// Compiled twin of R's .sym_rref_modp (joint/equilibrate path, per sample point).
// Entries are doubles in [0, p); returns the pivot rows, 0-based pivot columns and
// rank.
List symRrefMod(NumericMatrix M, double pIn) {
  u64 p = (u64)pIn;
  int nr = M.nrow(), nc = M.ncol();
  std::vector<std::vector<u64> > A(nr, std::vector<u64>(nc));
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nc; ++j) {
      long long v = (long long)M(i, j) % (long long)p;
      if (v < 0) v += p;
      A[i][j] = (u64)v;
    }
  std::vector<int> pivots = rref_mod(A, p);
  int rank = (int)pivots.size();
  NumericMatrix R(rank, nc);
  for (int i = 0; i < rank; ++i)
    for (int j = 0; j < nc; ++j) R(i, j) = (double)A[i][j];
  IntegerVector piv(rank);
  for (int i = 0; i < rank; ++i) piv[i] = pivots[i];   // 0-based, ascending
  return List::create(_["R"] = R, _["piv"] = piv, _["rank"] = rank);
}

// Gauge fill, incremental. F is the RREF (rows by ascending pivot, 0-based in piv) of
// the kernel vectors that vanish on the coordinates fixed so far. Fixing column k
// drops the row with pivot k, or else eliminates k with the last row that moves it;
// the result is again an RREF. symGaugeScores returns the nonzero count after fixing
// each candidate, symGaugeFix applies one.
static void gauge_fix_rows(std::vector<std::vector<u64> >& A, std::vector<int>& piv,
                           int k, u64 p) {
  int d = (int)A.size(), js = -1;
  for (int i = 0; i < d; ++i) if (A[i][k]) js = i;
  if (js < 0) return;
  if (piv[js] != k) {
    u64 inv = invmod(A[js][k], p);
    for (int i = 0; i < js; ++i) {
      if (!A[i][k]) continue;
      u64 c = mulmod(A[i][k], inv, p);
      for (size_t j = 0; j < A[i].size(); ++j)
        if (A[js][j]) A[i][j] = submod(A[i][j], mulmod(c, A[js][j], p), p);
    }
  }
  A.erase(A.begin() + js);
  piv.erase(piv.begin() + js);
}

static std::vector<std::vector<u64> > gauge_rows(NumericMatrix F, u64 p) {
  int d = F.nrow(), n = F.ncol();
  std::vector<std::vector<u64> > A(d, std::vector<u64>(n));
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < n; ++j) {
      long long v = (long long)F(i, j) % (long long)p;
      if (v < 0) v += p;
      A[i][j] = (u64)v;
    }
  return A;
}

NumericVector symGaugeScores(NumericMatrix F, IntegerVector piv, double pIn,
                             IntegerVector cand) {
  u64 p = (u64)pIn;
  std::vector<std::vector<u64> > A = gauge_rows(F, p);
  int d = (int)A.size(), n = F.ncol();
  std::vector<double> rowNnz(d, 0);
  double total = 0;
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < n; ++j) if (A[i][j]) rowNnz[i] += 1;
    total += rowNnz[i];
  }
  NumericVector out(cand.size());
  for (int c = 0; c < cand.size(); ++c) {
    int k = cand[c], js = -1;
    for (int i = 0; i < d; ++i) if (A[i][k]) js = i;
    if (js < 0) { out[c] = total; continue; }
    double f = total - rowNnz[js];
    if (piv[js] != k) {
      u64 inv = invmod(A[js][k], p);
      for (int i = 0; i < js; ++i) {
        if (!A[i][k]) continue;
        u64 cf = mulmod(A[i][k], inv, p);
        double nz = 0;
        for (int j = 0; j < n; ++j) {
          u64 v = A[js][j] ? submod(A[i][j], mulmod(cf, A[js][j], p), p) : A[i][j];
          if (v) nz += 1;
        }
        f += nz - rowNnz[i];
      }
    }
    out[c] = f;
  }
  return out;
}

List symGaugeFix(NumericMatrix F, IntegerVector piv, double pIn, int k) {
  u64 p = (u64)pIn;
  std::vector<std::vector<u64> > A = gauge_rows(F, p);
  std::vector<int> pv(piv.begin(), piv.end());
  gauge_fix_rows(A, pv, k, p);
  int d = (int)A.size(), n = F.ncol();
  NumericMatrix R(d, n);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < n; ++j) R(i, j) = (double)A[i][j];
  return List::create(_["R"] = R, _["piv"] = IntegerVector(pv.begin(), pv.end()));
}

// Fit num/den to samples of one nullspace entry over GF(p): sampleU holds the
// variables per sample, mons the monomial exponents (row 0 constant), rvals the
// entry values. num - r*den = 0 is homogeneous, so the fit is the kernel of the
// sample matrix (denominators without constant term included). status is
// "inconsistent" (degree too low), "ambiguous" (kernel dimension > 1) or "ok" with
// the kernel vector (free coefficient one) and free column, to agree across primes.
List symFitRational(IntegerMatrix sampleU, IntegerMatrix mons,
                    IntegerVector rvals, double pIn) {
  u64 p = (u64)pIn;
  int nS = sampleU.nrow(), nrel = sampleU.ncol(), nMon = mons.nrow();
  int ncols = 2 * nMon;
  std::vector<std::vector<u64> > A(nS, std::vector<u64>(ncols, 0));
  for (int s = 0; s < nS; ++s) {
    long long rv = (long long)rvals[s] % (long long)p; if (rv < 0) rv += p;
    u64 r = (u64)rv;
    std::vector<u64> uu(nrel);
    for (int k = 0; k < nrel; ++k) {
      long long uv = (long long)sampleU(s, k) % (long long)p; if (uv < 0) uv += p;
      uu[k] = (u64)uv;
    }
    for (int j = 0; j < nMon; ++j) {
      u64 mon = 1;
      for (int k = 0; k < nrel; ++k) mon = mulmod(mon, powmod(uu[k], (u64)mons(j, k), p), p);
      A[s][j] = mon;
      A[s][nMon + j] = submod(0, mulmod(r, mon, p), p);
    }
  }
  std::vector<int> pivots = rref_mod(A, p);
  int rank = (int)pivots.size();
  int nullity = ncols - rank;
  if (nullity == 0) return List::create(_["status"] = "inconsistent");
  if (nullity > 1) return List::create(_["status"] = "ambiguous");
  std::vector<char> isPiv(ncols, 0);
  for (int c : pivots) isPiv[c] = 1;
  int freeCol = 0;
  for (int c = 0; c < ncols; ++c) if (!isPiv[c]) { freeCol = c; break; }
  IntegerVector coeffs(ncols);
  coeffs[freeCol] = 1;
  for (int ri = 0; ri < rank; ++ri)
    coeffs[pivots[ri]] = (int)submod(0, A[ri][freeCol], p);
  return List::create(_["status"] = "ok", _["coeffs"] = coeffs,
                      _["free_col"] = freeCol);
}

// Per-row CRT over the given primes followed by rational reconstruction.
// Returns num/den as decimal strings; den is "0" when reconstruction fails.
List symRatRecon(IntegerMatrix residues, IntegerVector primes) {
#ifndef SYMIDENT_HAVE_INT128
  stop("symRatRecon needs __int128");
  return List();
#else
  int k = residues.nrow(), nprime = residues.ncol();
  // r + M*t stays below 2*product, so the product must fit under 2^127 (four
  // primes near 2^31); guarded by summed bit length
  int prodBits = 0;
  for (int j = 0; j < nprime; ++j) {
    u64 pj = (u64)primes[j];
    prodBits += pj ? (64 - __builtin_clzll(pj)) : 0;
  }
  if (prodBits >= 127)
    stop("symRatRecon: prime product exceeds the u128 CRT capacity (2^127); "
         "reduce the number or size of primes in .symPrimes");
  CharacterVector num(k), den(k);
  for (int row = 0; row < k; ++row) {
    u128 r = 0, M = 1;
    bool first = true;
    for (int j = 0; j < nprime; ++j) {
      u64 pj = (u64)primes[j];
      long long rv = (long long)residues(row, j) % (long long)pj; if (rv < 0) rv += pj;
      u64 res = (u64)rv;
      if (first) { r = res; M = pj; first = false; continue; }
      u64 Mmod = (u64)(M % pj);
      u64 diff = submod(res, (u64)(r % pj), pj);
      u64 t = mulmod(diff, invmod(Mmod, pj), pj);
      r = r + M * (u128)t;
      M = M * (u128)pj;
    }
    i128 n, d;
    if (rational_reconstruct(r, M, n, d)) {
      num[row] = i128_to_string(n);
      den[row] = i128_to_string(d);
    } else {
      num[row] = "0";
      den[row] = "0";
    }
  }
  return List::create(_["num"] = num, _["den"] = den);
#endif
}


// Berlekamp-Massey: minimal connection polynomial C (C[0]=1) of a GF(p) sequence.
// The sequence satisfies s[n] = -sum_{i>=1} C[i] s[n-i]; deg C is the LFSR length.
static std::vector<u64> berlekamp_massey(const std::vector<u64>& s, u64 p) {
  std::vector<u64> C(1, 1), B(1, 1);
  int L = 0, m = 1;
  u64 b = 1;
  for (int n = 0; n < (int)s.size(); ++n) {
    u64 d = s[n] % p;
    for (int i = 1; i <= L; ++i) d = addmod(d, mulmod(C[i], s[n - i], p), p);
    if (d == 0) { ++m; continue; }
    u64 coef = mulmod(d, invmod(b, p), p);
    if ((int)C.size() < (int)B.size() + m) C.resize(B.size() + m, 0);
    if (2 * L <= n) {
      std::vector<u64> T = C;
      for (int i = 0; i < (int)B.size(); ++i)
        C[i + m] = submod(C[i + m], mulmod(coef, B[i], p), p);
      L = n + 1 - L; B = T; b = d; m = 1;
    } else {
      for (int i = 0; i < (int)B.size(); ++i)
        C[i + m] = submod(C[i + m], mulmod(coef, B[i], p), p);
      ++m;
    }
  }
  C.resize(L + 1, 0);
  return C;
}

// Sparse multivariate polynomial interpolation (Ben-Or-Tiwari) from a geometric
// evaluation sequence seq[k] = f(b_1^k, ..., b_n^k) mod p. monoTab lists candidate
// exponent vectors (rows) and monoRes their monomial values prod b_j^{e_j} mod p;
// the recovered term monomials are the candidates whose value is a root of the
// Berlekamp-Massey locator, and their coefficients come from a transposed
// Vandermonde solve. status is "ok", "needmore" (sequence too short for the
// recovered order) or "noroots" (locator roots are not legal monomial values).
List symSparsePoly(IntegerVector seq, IntegerMatrix monoTab, IntegerVector monoRes,
                   double pIn) {
  u64 p = (u64) pIn;
  int len = seq.size();
  int nvar = monoTab.ncol();
  std::vector<u64> s(len);
  for (int i = 0; i < len; ++i) s[i] = red(seq[i], p);

  std::vector<u64> C = berlekamp_massey(s, p);
  int L = (int)C.size() - 1;
  if (L == 0)
    return List::create(_["status"] = "ok", _["nterms"] = 0,
                        _["exps"] = IntegerMatrix(0, nvar),
                        _["coeffs"] = IntegerVector(0));
  if (2 * L > len) return List::create(_["status"] = "needmore");

  // roots of the characteristic polynomial chi(x) = sum_i C[i] x^{L-i}
  int ncand = monoRes.size();
  std::vector<int> rootIdx;
  for (int c = 0; c < ncand; ++c) {
    u64 m = red(monoRes[c], p);
    u64 r = 0;
    for (int i = 0; i <= L; ++i) r = addmod(mulmod(r, m, p), C[i] % p, p);
    if (r == 0) rootIdx.push_back(c);
  }
  if ((int)rootIdx.size() != L) return List::create(_["status"] = "noroots");

  // transposed Vandermonde solve: sum_j coeff_j * nodes_j^k = s[k], k = 0..L-1
  int t = L;
  std::vector<u64> nodes(t);
  for (int j = 0; j < t; ++j)
    nodes[j] = red(monoRes[rootIdx[j]], p);
  std::vector<std::vector<u64> > A(t, std::vector<u64>(t + 1, 0));
  for (int k = 0; k < t; ++k) {
    for (int j = 0; j < t; ++j) A[k][j] = powmod(nodes[j], (u64)k, p);
    A[k][t] = s[k];
  }
  std::vector<int> piv = rref_mod(A, p);
  if ((int)piv.size() != t) return List::create(_["status"] = "noroots");

  IntegerMatrix exps(t, nvar);
  IntegerVector coeffs(t);
  for (int j = 0; j < t; ++j) {
    for (int v = 0; v < nvar; ++v) exps(j, v) = monoTab(rootIdx[j], v);
    coeffs[j] = (int)A[j][t];
  }
  return List::create(_["status"] = "ok", _["nterms"] = t,
                      _["exps"] = exps, _["coeffs"] = coeffs);
}


// Modular values of the monomials prod_j bases_j^{expts(i,j)} mod p, one per row
// of `expts`. Negative exponents use the modular inverse, so Laurent monomials are
// supported. Builds the candidate-monomial residues for sparse interpolation.
IntegerVector symMonoResidues(IntegerMatrix expts, IntegerVector bases, double pIn) {
  u64 p = (u64) pIn;
  int nr = expts.nrow(), nc = expts.ncol();
  IntegerVector out(nr);
  for (int i = 0; i < nr; ++i) {
    u64 v = 1;
    for (int j = 0; j < nc; ++j) {
      int e = expts(i, j);
      u64 b = red(bases[j], p);
      if (e >= 0) v = mulmod(v, powmod(b, (u64)e, p), p);
      else v = mulmod(v, invmod(powmod(b, (u64)(-e), p), p), p);
    }
    out[i] = (int)v;
  }
  return out;
}


// Berlekamp-Massey order (LFSR length) of a GF(p) sequence: the number of terms
// the sequence requires, used to grow the sparse sampling until it stabilises.
int symBMorder(IntegerVector seq, double pIn) {
  u64 p = (u64) pIn;
  int len = seq.size();
  std::vector<u64> s(len);
  for (int i = 0; i < len; ++i) s[i] = red(seq[i], p);
  return (int)berlekamp_massey(s, p).size() - 1;
}


// Univariate Cauchy (rational) interpolation A(t)/B(t) = r(t) over GF(p), with
// deg A = dN, deg B = dD, from samples (tnodes, rvals); normalised so B(0) = 1.
// Returns the values A(1) and B(1) (= N(point)/D(s) and D(point)/D(s) when the
// caller samples r along the ray s + t*(point - s)). status is "ok",
// "ambiguous" (the (dN, dD) guess does not give a one-dimensional fit) or
// "badshift" (B(0) = 0 at this shift). Used by the general sparse-rational path.
List symCauchyEval(IntegerVector tnodes, IntegerVector rvals, int dN, int dD,
                   double pIn) {
  u64 p = (u64) pIn;
  int ns = tnodes.size(), nc = dN + dD + 2;
  std::vector<std::vector<u64> > M(ns, std::vector<u64>(nc, 0));
  for (int s = 0; s < ns; ++s) {
    u64 t = red(tnodes[s], p);
    u64 r = red(rvals[s], p);
    u64 tp = 1;
    for (int d = 0; d <= dN; ++d) { M[s][d] = tp; tp = mulmod(tp, t, p); }
    tp = 1;
    for (int e = 0; e <= dD; ++e) {
      M[s][dN + 1 + e] = (p - mulmod(r, tp, p)) % p;
      tp = mulmod(tp, t, p);
    }
  }
  std::vector<int> piv = rref_mod(M, p);
  std::vector<bool> isPiv(nc, false);
  for (size_t i = 0; i < piv.size(); ++i) isPiv[piv[i]] = true;
  int fcol = -1;
  for (int c = 0; c < nc; ++c) if (!isPiv[c]) { if (fcol >= 0) { fcol = -2; break; } fcol = c; }
  if (fcol < 0) return List::create(_["status"] = "ambiguous");
  std::vector<u64> v(nc, 0);
  v[fcol] = 1;
  for (size_t ri = 0; ri < piv.size(); ++ri) v[piv[ri]] = (p - M[ri][fcol]) % p;
  u64 b0 = v[dN + 1] % p;
  if (b0 == 0) return List::create(_["status"] = "badshift");
  u64 ib0 = invmod(b0, p), A1 = 0, B1 = 0;
  for (int d = 0; d <= dN; ++d) A1 = addmod(A1, v[d], p);
  for (int e = 0; e <= dD; ++e) B1 = addmod(B1, v[dN + 1 + e], p);
  return List::create(_["status"] = "ok",
                      _["N"] = (double) mulmod(A1, ib0, p),
                      _["D"] = (double) mulmod(B1, ib0, p));
}

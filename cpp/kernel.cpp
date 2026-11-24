// Exact GF(p) kernel for the analytic observability path of detect().
// Builds the observability-identifiability matrix by a Taylor-mode construction
// over a finite field (Lie derivatives as truncated power series in time,
// parameter gradients as forward-mode dual numbers), reduces it, and supports
// the rational reconstruction used by the Python side. Primes are < 2^31 so
// residue products stay below 2^62 in uint64; the CRT product of four primes
// stays below 2^124 in unsigned __int128 (symRatRecon only; compilers without
// __int128 leave it out and the Python side reconstructs with arbitrary precision).

#include "kernel.h"

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <utility>
#include <exception>
#include <stdexcept>
#include <atomic>
#include <mutex>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <cstdio>

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

const int OP_CONST = 0, OP_ADD = 1, OP_MUL = 2, OP_INV = 3, OP_SQRT = 4, OP_ROOT = 5,
          OP_EXPN = 6, OP_LOG1P = 7, OP_SINN = 8, OP_COSN = 9, OP_SPOW = 10;

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

// The first exception thrown inside an OpenMP loop, rethrown after the loop: an
// exception that leaves a parallel region terminates the process.
struct OmpGuard {
  std::exception_ptr ex;
  void capture() {
    #pragma omp critical(omp_guard)
    { if (!ex) ex = std::current_exception(); }
  }
  void rethrow() const { if (ex) std::rethrow_exception(ex); }
};

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

// Reduce a (possibly negative) integer to the residue in [0, p).
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

// Parse a signed decimal integer to the residue modulo p (handles arbitrary length
// without overflow); anything but digits after the sign is an error.
u64 parse_mod(const std::string& s, u64 p) {
  size_t i = 0;
  bool neg = false;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; ++i; }
  if (i == s.size()) throw std::invalid_argument("not an integer: '" + s + "'");
  u64 r = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') throw std::invalid_argument("not an integer: '" + s + "'");
    r = (r * 10 + (u64)(s[i] - '0')) % p;
  }
  return neg ? (p - r) % p : r;
}

// Residue of the rational constant num/den (decimal strings) modulo p; a denominator
// divisible by p is an error, as the constant has no residue there.
inline u64 reduce_rational(const std::string& num, const std::string& den, u64 p) {
  u64 d = parse_mod(den, p);
  if (d == 0) throw std::domain_error("the constant " + num + "/" + den + " has no residue modulo "
                                      + std::to_string(p));
  return mulmod(parse_mod(num, p), invmod(d, p), p);
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
  int serAxis, Tser;                           // a series axis with a separate order, or -1
  std::vector<std::vector<int> > idx;          // nMono multi-indices, degree-ordered
  std::vector<std::vector<int> > sumSlot;      // sumSlot[a][b] = slot(idx[a]+idx[b]) or -1
  std::vector<std::vector<std::pair<int,int> > > decomp;  // decomp[s] = {(a,b): a+b == s}
  std::map<std::vector<int>, int> slotOf;

  // the gap axes within the total degree Mtot, the series axis within Tser
  bool within(const std::vector<int>& mu) const {
    int td = 0;
    for (int i = 0; i < K; ++i) if (i != serAxis) td += mu[i];
    return td <= Mtot && (serAxis < 0 || mu[serAxis] <= Tser);
  }

  // slot of idx[a] with `axis` incremented by k (promoting a time-order into a
  // gap axis), or -1 if it overflows the degree budget
  int shiftSlot(int a, int axis, int k) const {
    std::vector<int> mu = idx[a];
    mu[axis] += k;
    if (!within(mu)) return -1;
    std::map<std::vector<int>, int>::const_iterator it = slotOf.find(mu);
    return it == slotOf.end() ? -1 : it->second;
  }

  PolyBasis(int K_, int Mtot_, int w_, int serAxis_ = -1, int Tser_ = 0)
      : K(K_), Mtot(Mtot_), w(w_), serAxis(serAxis_), Tser(serAxis_ < 0 ? 0 : Tser_) {
    std::vector<int> mu(K, 0);
    for (int d = 0; d <= Mtot + Tser; ++d) enumerate(d, 0, d, mu);
    nMono = (int)idx.size();
    for (int s = 0; s < nMono; ++s) slotOf[idx[s]] = s;
    sumSlot.assign(nMono, std::vector<int>(nMono, -1));
    decomp.assign(nMono, std::vector<std::pair<int,int> >());
    std::vector<int> sum(K);
    for (int a = 0; a < nMono; ++a)
      for (int b = 0; b < nMono; ++b) {
        for (int i = 0; i < K; ++i) sum[i] = idx[a][i] + idx[b][i];
        if (!within(sum)) continue;
        int s = slotOf[sum];
        sumSlot[a][b] = s;
        decomp[s].push_back(std::make_pair(a, b));
      }
  }
  // emit the K-compositions of `remaining` (total degree d) from axis pos within the budgets
  void enumerate(int remaining, int pos, int d, std::vector<int>& mu) {
    if (pos == K - 1) { mu[pos] = remaining; if (within(mu)) idx.push_back(mu); return; }
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

// B = f(A) for a series A without constant value and f = exp, log(1 + .), sin or cos (the
// opcode): A is nilpotent, so the Taylor sum ends where A^k vanishes, past the total degree
// plus one.
inline bool poly_nilpotent(u64* B, const u64* A, int f, const PolyBasis& pb, u64 p) {
  size_t blk = (size_t)pb.nMono * pb.w;
  if (A[0] % p != 0) return false;
  std::fill(B, B + blk, (u64)0);
  if (f != OP_LOG1P && f != OP_SINN) B[0] = 1;
  // pw = A^k / k! (exp, sin, cos) or A^k (log)
  std::vector<u64> pw(A, A + blk), next(blk, 0);
  for (int k = 1; k <= pb.Mtot + pb.Tser + 2; ++k) {
    bool any = false;
    for (size_t c = 0; c < blk && !any; ++c) any = pw[c] != 0;
    if (!any) return true;
    u64 cf = 0;
    if (f == OP_EXPN) cf = 1;
    else if (f == OP_LOG1P) cf = k % 2 ? invmod((u64)k % p, p) : submod(0, invmod((u64)k % p, p), p);
    else if ((f == OP_SINN) == (k % 2 == 1)) cf = (k / 2) % 2 ? p - 1 : 1;
    if (cf)
      for (size_t c = 0; c < blk; ++c) B[c] = addmod(B[c], mulmod(pw[c], cf, p), p);
    std::fill(next.begin(), next.end(), (u64)0);
    poly_mul_acc(next.data(), pw.data(), A, pb, p);
    u64 ik = f == OP_LOG1P ? 1 : invmod((u64)(k + 1) % p, p);
    for (size_t c = 0; c < blk; ++c) pw[c] = mulmod(next[c], ik, p);
  }
  for (size_t c = 0; c < blk; ++c) if (pw[c]) return false;
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

// ---- Reduced echelon basis, rows added a batch at a time ----
// Basis rows live on the non-pivot columns `cols` (ascending), 1 at the pivot of the row, 0 at the other pivots.
// A batch is reduced by the basis in parallel, echelonized, and clears the basis at the new pivots; the reduced
// form of a row space is unique, so the result does not depend on the batching.
struct RrefBasis {
  int nz;
  u64 p;
  std::vector<int> cols, piv;                  // free columns; the pivot column of each row
  std::vector<std::vector<u64> > rows;         // per basis row, the entries at `cols`
  RrefBasis(int nz_, u64 p_) : nz(nz_), p(p_) {
    cols.resize(nz);
    for (int c = 0; c < nz; ++c) cols[c] = c;
  }
  int rank() const { return (int)piv.size(); }

  // the rows of X (nz residues each, reduced below p) added; X is consumed. With `from`, the indices of the rows
  // of X that are independent of the basis and of the rows of X before them, ascending
  void add(std::vector<std::vector<u64> >& X, int threads, std::vector<int>* from = nullptr) {
    if (X.empty() || cols.empty()) return;
    int m = (int)X.size(), nf = (int)cols.size(), r = rank();
    const Barrett br(p);
    const u64 LIM = (((u64)1 << 63) / p) * p;
    // reduction by the basis: y = x at the free columns minus the sum of x at the pivots times the rows
    std::vector<std::vector<u64> > Y(m);
    OmpGuard guard_;
    int nThr = std::max(1, std::min(threads, m));
    #pragma omp parallel for num_threads(nThr) schedule(dynamic, 16) if (nThr > 1)
    for (int i = 0; i < m; ++i) {
      try {
        std::vector<u64> acc(nf, 0);
        const std::vector<u64>& x = X[i];
        for (int j = 0; j < r; ++j) {
          u64 f = x[piv[j]] % p;
          if (!f) continue;
          const u64* bj = rows[j].data();
          for (int n = 0; n < nf; ++n) {
            u64 t = acc[n] + f * bj[n];
            acc[n] = t >= LIM ? t - LIM : t;
          }
        }
        std::vector<u64>& y = Y[i];
        y.resize(nf);
        bool any = false;
        for (int n = 0; n < nf; ++n) {
          y[n] = submod(x[cols[n]] % p, br.red(acc[n]), p);
          any |= y[n] != 0;
        }
        if (!any) y.clear();
        std::vector<u64>().swap(X[i]);
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
    // echelon of the reduced rows among themselves, a chunk at a time: the rows of a chunk are reduced by
    // the new rows of the earlier chunks in parallel, then by each other in order
    std::vector<std::vector<u64> > E;
    std::vector<int> epiv;                       // position in `cols` of each new pivot
    auto reduce_by = [&](std::vector<u64>& y, size_t k0, size_t k1) {
      for (size_t k = k0; k < k1; ++k) {
        u64 f = y[epiv[k]];
        if (!f) continue;
        u64 nfac = p - f;
        const u64* e = E[k].data();
        for (int n = 0; n < nf; ++n)
          if (e[n]) y[n] = br.red(y[n] + nfac * e[n]);
      }
    };
    const int CH = 64;
    for (int i0 = 0; i0 < m; i0 += CH) {
      int i1 = std::min(m, i0 + CH);
      size_t q0 = E.size();
      int nThr3 = std::max(1, std::min(threads, i1 - i0));
      #pragma omp parallel for num_threads(nThr3) schedule(static, 1) if (nThr3 > 1 && q0 > 0)
      for (int i = i0; i < i1; ++i)
        if (!Y[i].empty()) reduce_by(Y[i], 0, q0);
      for (int i = i0; i < i1; ++i) {
        std::vector<u64>& y = Y[i];
        if (y.empty()) continue;
        reduce_by(y, q0, E.size());
        int pc = -1;
        for (int n = 0; n < nf; ++n) if (y[n]) { pc = n; break; }
        if (pc < 0) continue;
        u64 inv = invmod(y[pc], p);
        for (int n = pc; n < nf; ++n) if (y[n]) y[n] = br.red(y[n] * inv);
        E.push_back(std::move(y));
        epiv.push_back(pc);
        if (from) from->push_back(i);
      }
    }
    if (E.empty()) return;
    int q = (int)E.size();
    // the new rows reduced at each other's pivots
    int nThr4 = std::max(1, std::min(threads, q));
    for (int k = 0; k < q; ++k) {
      const u64* ek = E[k].data();
      #pragma omp parallel for num_threads(nThr4) schedule(static, 16) if (nThr4 > 1 && q > 64)
      for (int l = 0; l < q; ++l) {
        if (l == k) continue;
        u64 f = E[l][epiv[k]];
        if (!f) continue;
        u64 nfac = p - f;
        u64* el = E[l].data();
        for (int n = 0; n < nf; ++n)
          if (ek[n]) el[n] = br.red(el[n] + nfac * ek[n]);
      }
    }
    // the old rows cleared at the new pivots
    int nThr2 = std::max(1, std::min(threads, r));
    #pragma omp parallel for num_threads(nThr2) schedule(dynamic, 16) if (nThr2 > 1)
    for (int j = 0; j < r; ++j) {
      std::vector<u64>& bj = rows[j];
      for (int k = 0; k < q; ++k) {
        u64 f = bj[epiv[k]];
        if (!f) continue;
        u64 nfac = p - f;
        const u64* e = E[k].data();
        for (int n = 0; n < nf; ++n)
          if (e[n]) bj[n] = br.red(bj[n] + nfac * e[n]);
      }
    }
    // the new pivots leave the free columns
    std::vector<char> isNew(nf, 0);
    for (int k = 0; k < q; ++k) isNew[epiv[k]] = 1;
    std::vector<int> keep;
    for (int n = 0; n < nf; ++n) if (!isNew[n]) keep.push_back(n);
    auto compact = [&](std::vector<u64>& v) {
      std::vector<u64> w(keep.size());
      for (size_t n = 0; n < keep.size(); ++n) w[n] = v[keep[n]];
      v.swap(w);
    };
    for (auto& bj : rows) compact(bj);
    for (int k = 0; k < q; ++k) {
      piv.push_back(cols[epiv[k]]);
      compact(E[k]);
      rows.push_back(std::move(E[k]));
    }
    std::vector<int> nc(keep.size());
    for (size_t n = 0; n < keep.size(); ++n) nc[n] = cols[keep[n]];
    cols.swap(nc);
  }

  // the basis as full rows ordered by pivot column, as rref_mod returns them
  void full(std::vector<std::vector<u64> >& R, std::vector<int>& pivots) const {
    int r = rank();
    std::vector<int> ord(r);
    for (int j = 0; j < r; ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](int x, int y) { return piv[x] < piv[y]; });
    R.assign(r, std::vector<u64>(nz, 0));
    pivots.assign(r, 0);
    for (int i = 0; i < r; ++i) {
      int j = ord[i];
      pivots[i] = piv[j];
      R[i][piv[j]] = 1;
      for (size_t n = 0; n < cols.size(); ++n) R[i][cols[n]] = rows[j][n];
    }
  }
};

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
    OmpGuard guard_;
    #pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(dynamic, 8)
    for (int i = r + 1; i < nr; ++i) {
      try {
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
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
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

// The value at eps = 1 of the series c (n known coefficients) by the smallest Pade form
// N/D, D(0) = 1, with n at least twice the number of unknowns plus one and every coefficient
// consistent. False if none exists or D(1) = 0.
static bool pade_at_one(const std::vector<u64>& c, int n, u64 p, u64& val) {
  for (int tot = 0; 2 * (tot + 1) + 1 <= n; ++tot) {
    for (int dd = 0; dd <= tot; ++dd) {
      int dn = tot - dd;
      // D from c[dn+1..dn+dd]: sum_{j=0..dd} D_j c[k-j] = 0 for k = dn+1..dn+dd
      std::vector<std::vector<u64> > A(dd, std::vector<u64>(dd + 1, 0));
      for (int i = 0; i < dd; ++i) {
        int k = dn + 1 + i;
        for (int j = 1; j <= dd; ++j) A[i][j - 1] = (k - j >= 0 && k - j < n) ? c[k - j] : 0;
        A[i][dd] = submod(0, k < n ? c[k] : 0, p);
      }
      std::vector<u64> D(dd + 1, 0);
      D[0] = 1;
      if (dd) {
        std::vector<std::vector<u64> > M = A;
        std::vector<int> piv = rref_mod(M, p);
        if ((int)piv.size() != dd || piv.back() == dd) continue;
        for (int i = 0; i < dd; ++i) D[piv[i] + 1] = M[i][dd];
      }
      bool okc = true;
      for (int k = dn + 1; k < n && okc; ++k) {
        u64 acc = 0;
        for (int j = 0; j <= dd && j <= k; ++j) acc = addmod(acc, mulmod(D[j], c[k - j], p), p);
        if (acc) okc = false;
      }
      if (!okc) continue;
      u64 num = 0, den = 0;
      for (int k = 0; k <= dn; ++k) {
        u64 acc = 0;
        for (int j = 0; j <= dd && j <= k; ++j) acc = addmod(acc, mulmod(D[j], c[k - j], p), p);
        num = addmod(num, acc, p);
      }
      for (int j = 0; j <= dd; ++j) den = addmod(den, D[j], p);
      if (!den) return false;
      val = mulmod(num, invmod(den, p), p);
      return true;
    }
  }
  return false;
}

// Kernel of the series echelon rows U (rank r, entries mod eps^N) at eps = 1: each vector
// by back substitution in Laurent series from eps^0, accepted only if every entry is a
// rational function determined by the known coefficients (pade_at_one).
bool series_kernel_at_one(const std::vector<std::vector<u64> >& U, int r, int nz, int N,
                          u64 p, std::vector<std::vector<u64> >& K) {
  std::vector<int> pc(r, -1), pv(r, N);
  std::vector<char> isPiv(nz, 0);
  for (int t = 0; t < r; ++t) {
    // the pivot of row t: the entry of minimal valuation among columns no earlier
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
      if (deg < 0) continue;
      if (known >= 2 * (deg + 1) + 1) {
        for (size_t i = 0; i < e.c.size(); ++i)
          if (e.lo + (int)i < e.prec) k1[c] = addmod(k1[c], e.c[i], p);
        continue;
      }
      std::vector<u64> cs(known, 0);
      for (size_t i = 0; i < e.c.size(); ++i) {
        int k = e.lo + (int)i - m;
        if (k >= 0 && k < known) cs[k] = e.c[i];
      }
      if (!pade_at_one(cs, known, p, k1[c])) return false;
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

// Line mode: per group of nMono rows and gap monomial g, one series row holds the
// coefficient of g + j e_ser in column c at c*N + j. The rows from `from` on are
// compacted to their gap monomials.
void rows_to_series_axis(std::vector<std::vector<u64> >& rows, size_t from,
                         const PolyBasis& pb, int nz, int N,
                         std::vector<std::vector<u64> >& srows, bool keepZero = false) {
  int nM = pb.nMono, sa = pb.serAxis;
  std::vector<int> gaps;
  for (int mo = 0; mo < nM; ++mo) if (pb.idx[mo][sa] == 0) gaps.push_back(mo);
  size_t keep = from;
  for (size_t g = from; g + nM <= rows.size(); g += nM) {
    for (int gm : gaps) {
      std::vector<u64> s((size_t)nz * N, 0);
      bool any = false;
      std::vector<int> mu = pb.idx[gm];
      for (int j = 0; j < N; ++j) {
        mu[sa] = j;
        std::map<std::vector<int>, int>::const_iterator it = pb.slotOf.find(mu);
        if (it == pb.slotOf.end()) continue;
        const std::vector<u64>& row = rows[g + it->second];
        for (int c = 0; c < nz; ++c)
          if (row[c]) { s[(size_t)c * N + j] = row[c]; any = true; }
      }
      if (any || keepZero) srows.push_back(s);
    }
    for (int gm : gaps) rows[keep++] = rows[g + gm];
  }
  rows.resize(keep);
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
      case OP_EXPN: case OP_LOG1P: case OP_SINN: case OP_COSN: {
        // a dual number without value: exp(a) = 1 + a, log(1 + a) = sin(a) = a, cos(a) = 1
        const u64* pa = icVal[icA[i]].data();
        if (pa[0] % p != 0) return false;
        bool cosine = icOp[i] == OP_COSN;
        for (int c = 1; c < w; ++c) o[c] = cosine ? 0 : pa[c] % p;
        o[0] = icOp[i] == OP_EXPN || cosine ? 1 : 0;
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
  // values constant in time: zero past order 0, so a product with one takes a single term
  std::vector<char> tconst(instrBase + nInstr, 0);
  {
    std::vector<char> isState(instrBase, 0);
    for (int s : stateSlots) if (s >= 0 && s < instrBase) isState[s] = 1;
    for (int v = 0; v < instrBase; ++v) tconst[v] = !isState[v];
    for (int i = 0; i < nInstr; ++i) {
      int o = op[i];
      tconst[instrBase + i] = o == OP_CONST || ((o == OP_ADD || o == OP_MUL) && tconst[a[i]] && tconst[b[i]]) ||
                              (o == OP_INV && tconst[a[i]]);
    }
  }
  for (int k = 0; k <= Nt && !fail; ++k) {
    for (int i = 0; i < nInstr; ++i) {
      if (k > 0 && tconst[instrBase + i]) continue;
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
          int j0 = tconst[b[i]] ? k : 0, j1 = tconst[a[i]] ? 0 : k;
          for (int ii = j0; ii <= j1; ++ii)
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
        default:
          fail = true;                      // a power of a state at 0: the polyseries path
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

// ---- Laurent series in time of polyseries blocks ----
// An output such as a ratio of two states that both start at 0 is analytic for t > 0, but the
// tape of the output divides by a series without constant value. Such instructions are evaluated after the
// state recursion, one instruction at a time, as Laurent series: block j is the coefficient of
// t^(lo + j), every coefficient below t^prec is known, and blocks past c are zero. A reciprocal
// splits the operand a = t^v0 (U + M) with U a series whose constant block has a unit value
// (v0 the first order with a nonzero value at the constant monomial) and M the terms below t^v0,
// which lie in the nilpotent ideal of the dual lanes and the monomials of positive degree:
// 1/a = t^-v0 U^-1 sum_e (-M U^-1)^e, a finite sum.
const int LB_EXACT = 1 << 28;
struct LBlk { int lo, prec; std::vector<u64> c; };

inline int lb_len(const LBlk& a, int blk) { return (int)(a.c.size() / blk); }

// drop leading zero blocks below the precision, so that lo is the valuation
inline void lb_norm(LBlk& a, int blk) {
  int n = lb_len(a, blk), j = 0;
  while (j < n) {
    bool nz = false;
    for (int c = 0; c < blk && !nz; ++c) nz = a.c[(size_t)j * blk + c] != 0;
    if (nz) break;
    ++j;
  }
  if (j == n) { a.c.clear(); a.lo = a.prec; return; }
  if (j > 0) { a.c.erase(a.c.begin(), a.c.begin() + (size_t)j * blk); a.lo += j; }
  int keep = std::max(0, a.prec - a.lo);
  if (lb_len(a, blk) > keep) a.c.resize((size_t)keep * blk);
}

inline LBlk lb_add(const LBlk& a, const LBlk& b, int blk, u64 p, bool sub = false) {
  LBlk o;
  o.prec = std::min(a.prec, b.prec);
  o.lo = std::min(a.c.empty() ? b.lo : a.lo, b.c.empty() ? a.lo : b.lo);
  int hi = std::max(a.lo + lb_len(a, blk), b.lo + lb_len(b, blk));
  hi = std::min(hi, o.prec);
  o.c.assign((size_t)std::max(0, hi - o.lo) * blk, 0);
  int n = std::max(0, hi - o.lo);
  for (int j = 0; j < lb_len(a, blk); ++j) {
    int e = a.lo + j - o.lo;
    if (e < 0 || e >= n) continue;
    for (int c = 0; c < blk; ++c)
      o.c[(size_t)e * blk + c] = addmod(o.c[(size_t)e * blk + c], a.c[(size_t)j * blk + c], p);
  }
  for (int j = 0; j < lb_len(b, blk); ++j) {
    int e = b.lo + j - o.lo;
    if (e < 0 || e >= n) continue;
    for (int c = 0; c < blk; ++c) {
      u64 v = b.c[(size_t)j * blk + c];
      o.c[(size_t)e * blk + c] = sub ? submod(o.c[(size_t)e * blk + c], v, p)
                                     : addmod(o.c[(size_t)e * blk + c], v, p);
    }
  }
  lb_norm(o, blk);
  return o;
}

inline LBlk lb_mul(const LBlk& a, const LBlk& b, const PolyBasis& pb, u64 p) {
  int blk = pb.nMono * pb.w;
  LBlk o;
  o.lo = a.lo + b.lo;
  o.prec = std::min(a.prec + b.lo, b.prec + a.lo);
  int la = lb_len(a, blk), lbn = lb_len(b, blk);
  int n = (la && lbn) ? std::min(o.prec - o.lo, la + lbn - 1) : 0;
  o.c.assign((size_t)std::max(0, n) * blk, 0);
  for (int i = 0; i < la; ++i)
    for (int j = 0; j < lbn && i + j < n; ++j)
      poly_mul_acc(o.c.data() + (size_t)(i + j) * blk, a.c.data() + (size_t)i * blk,
                   b.c.data() + (size_t)j * blk, pb, p);
  lb_norm(o, blk);
  return o;
}

// 1/a, false where no order below the precision has a unit value (more orders are needed) or
// the constant block cannot be inverted
inline bool lb_inv(LBlk& out, const LBlk& a, const PolyBasis& pb, u64 p) {
  int blk = pb.nMono * pb.w, la = lb_len(a, blk), j0 = -1;
  for (int j = 0; j < la; ++j)
    if (a.c[(size_t)j * blk] % p) { j0 = j; break; }
  if (j0 < 0) return false;
  int v0 = a.lo + j0;
  int n = a.prec - v0;                         // relative precision of U and of U^-1
  if (n <= 0) return false;
  LBlk Ui;
  Ui.lo = 0; Ui.prec = n;
  Ui.c.assign((size_t)n * blk, 0);
  const u64* U = a.c.data() + (size_t)j0 * blk;
  int lu = la - j0;
  if (!poly_inv(Ui.c.data(), U, pb, p)) return false;
  std::vector<u64> s(blk);
  for (int j = 1; j < n; ++j) {
    std::fill(s.begin(), s.end(), (u64)0);
    for (int i = 1; i <= j && i < lu; ++i)
      poly_mul_acc(s.data(), U + (size_t)i * blk, Ui.c.data() + (size_t)(j - i) * blk, pb, p);
    for (int c = 0; c < blk; ++c) s[c] = submod(0, s[c], p);
    poly_mul_acc(Ui.c.data() + (size_t)j * blk, Ui.c.data(), s.data(), pb, p);
  }
  lb_norm(Ui, blk);
  LBlk S = Ui;
  if (j0 > 0) {
    LBlk M;
    M.lo = a.lo - v0; M.prec = LB_EXACT;
    M.c.assign(a.c.begin(), a.c.begin() + (size_t)j0 * blk);
    // M in the dual lanes only squares to zero; with monomials of positive degree the ideal
    // vanishes past the total degree plus the dual lane
    bool dualOnly = true;
    for (int j = 0; j < j0 && dualOnly; ++j)
      for (int mo = 0; mo < pb.nMono; ++mo)
        if (M.c[(size_t)j * blk + (size_t)mo * pb.w] % p) { dualOnly = false; break; }
    int E = dualOnly ? 2 : pb.Mtot + pb.Tser + 2;
    LBlk T = lb_mul(M, Ui, pb, p);
    for (auto& x : T.c) x = submod(0, x, p);
    LBlk P = Ui;
    for (int e = 1; e < E; ++e) {
      P = lb_mul(P, T, pb, p);
      S = lb_add(S, P, blk, p);
    }
  }
  S.lo -= v0;
  S.prec -= v0;
  out = S;
  return true;
}

// Polyseries version of build_obs_rows: every value is a width-(nMono*w) block (a
// truncated power series in the gap lengths, each monomial a dual). Runs to time
// order Nrun and emits z-gradient rows for orders 0..Ntemit and all gap monomials.
// `val` slots are pre-sized to (Nrun+1)*blk with states seeded at order 0.
// A reciprocal of a series without constant value outside the right-hand sides (an output
// that is 0/0 at the start) is evaluated after the recursion as a Laurent series; where the
// truncation needs more orders, the recursion is run again further. Rows of negative order
// (poles of the dual lanes, e.g. along a state of the start) go to `preRows` as
// (output, -order, monomial, lanes...); without it, a pole fails. `deferredOut` is set where an
// instruction is taken as a Laurent series.
bool build_obs_rows_poly(std::vector<std::vector<u64> >& val,
                         const std::vector<int>& op, const std::vector<int>& a,
                         const std::vector<int>& b, const std::vector<u64>& cval,
                         int instrBase, const std::vector<int>& stateSlots,
                         const std::vector<int>& fOut, const std::vector<int>& gOut,
                         const PolyBasis& pb, int Nrun, int Ntemit, u64 p,
                         std::vector<std::vector<u64> >& rows,
                         const u64* dtau = nullptr,
                         std::vector<std::vector<u64> >* preRows = nullptr, int depth = 0,
                         int need = -1, int spowAxis0 = -1, char* deferredOut = nullptr) {
  if (need < 0) need = Nrun + 1;
  int nInstr = (int)op.size();
  int m = (int)stateSlots.size();
  int w = pb.w, nM = pb.nMono, blk = nM * w, nz = w - 1;
  std::vector<std::vector<u64> > invA0(nInstr);
  bool fail = false;
  // instructions the right-hand sides read; the others may be deferred to Laurent series
  std::vector<char> fcone(nInstr, 0), deferred(nInstr, 0);
  {
    int nSlot = instrBase + nInstr;
    std::vector<char> need(nSlot, 0);
    for (int i = 0; i < m; ++i) if (fOut[i] >= 0 && fOut[i] < nSlot) need[fOut[i]] = 1;
    for (int i = nInstr - 1; i >= 0; --i) {
      if (!need[instrBase + i]) continue;
      fcone[i] = 1;
      if (op[i] == OP_ADD || op[i] == OP_MUL) { need[a[i]] = 1; need[b[i]] = 1; }
      else if (op[i] == OP_INV) need[a[i]] = 1;
    }
  }
  bool anyDeferred = false;
  auto isDef = [&](int slot) { return slot >= instrBase && deferred[slot - instrBase]; };
  // P = x^a of a state x that starts at 0 (OP_SPOW). With x = c t^k (1 + ...), P = S Q where
  // S = (c t^k)^a is a formal variable on axis `axis` of the basis and Q = (x / (c t^k))^a a
  // series; a coefficient of t^n S^m stands for t^(n - k m + k m a). The time index n is the
  // exponent at a = 1, so S counts k orders and P at order n reads x up to order n: Q follows
  // from A = x / (c t^k) by n Q_n = A_0^-1 sum_{j=1..n} (a j + j - n) A_j Q_(n-j).
  struct SPowAx {
    int axis, kv;
    std::vector<u64> c, cinv, ex, A0inv;
    std::vector<std::vector<u64> > A, Q;
  };
  std::vector<SPowAx> sp;
  std::vector<int> spOf(nInstr, -1);
  for (int i = 0; i < nInstr; ++i)
    if (op[i] == OP_SPOW) {
      if (spowAxis0 < 0) return false;
      SPowAx x;
      x.axis = spowAxis0 + (int)sp.size();
      x.kv = -1;
      if (x.axis >= pb.K) return false;
      spOf[i] = (int)sp.size();
      sp.push_back(x);
    }
  // dual products: o (cleared) = X * s for a block X and a dual s; inverse of a dual
  auto scaleBlock = [&](u64* o, const u64* X, const u64* sc) {
    std::fill(o, o + blk, (u64)0);
    for (int mo = 0; mo < nM; ++mo) dual_mul_acc(o + (size_t)mo * w, X + (size_t)mo * w, sc, w, p);
  };
  auto dualInv = [&](std::vector<u64>& o, const u64* x) {
    o.assign(w, 0);
    u64 vi = invmod(x[0] % p, p), vi2 = mulmod(vi, vi, p);
    o[0] = vi;
    for (int c = 1; c < w; ++c) o[c] = submod(0, mulmod(x[c], vi2, p), p);
  };
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
  // values constant in time: the leaves (slots below instrBase that hold no state) and the
  // instructions formed from them alone. Their coefficients past time order 0 are zero, so a
  // product with one takes a single term and the instruction is not formed again.
  std::vector<char> tconst(nVal, 0);
  {
    std::vector<char> isState(instrBase, 0);
    for (int i = 0; i < m; ++i)
      if (stateSlots[i] >= 0 && stateSlots[i] < instrBase) isState[stateSlots[i]] = 1;
    for (int v = 0; v < instrBase && v < nVal; ++v) tconst[v] = !isState[v];
    for (int i = 0; i < nInstr && instrBase + i < nVal; ++i) {
      int o = op[i];
      tconst[instrBase + i] = o == OP_CONST ||
          ((o == OP_ADD || o == OP_MUL) && tconst[a[i]] && tconst[b[i]]) ||
          (o == OP_INV && tconst[a[i]]);
    }
  }
  // out += sum_{ii = j0..j1} A_ii * B_{k-ii} over the monomials, A and B tape values
  auto convAdd = [&](u64* out, int av, int bv, int j0, int j1, int k) {
    if (tconst[av]) {
      if (j0 > 0) return;
      j1 = 0;
    }
    if (tconst[bv]) {
      if (k < j0 || k > j1) return;
      j0 = j1 = k;
    }
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
      if (k == 0 && op[i] != OP_CONST &&
          (isDef(a[i]) || (op[i] != OP_INV && isDef(b[i])) ||
           (op[i] == OP_INV && val[a[i]][0] % p == 0))) {
        if (fcone[i]) { fail = true; break; }
        deferred[i] = 1;
        anyDeferred = true;
        if (deferredOut) *deferredOut = 1;
      }
      if (deferred[i] || (k > 0 && tconst[instrBase + i])) continue;
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
        case OP_SPOW: {
          SPowAx& x = sp[spOf[i]];
          const u64* ev = val[b[i]].data();
          if (k == 0) x.ex.assign(ev, ev + w);
          else
            for (int c = 0; c < blk; ++c)
              if (ev[(size_t)k * blk + c]) { fail = true; break; }   // the exponent is constant
          if (fail) break;
          const u64* xb = val[a[i]].data() + (size_t)k * blk;
          if (x.kv < 0) {
            if (xb[0] % p == 0) {
              // below the first order with a value, a dual or another axis is not generic
              for (int c = 0; c < blk; ++c) if (xb[c]) { fail = true; break; }
              break;
            }
            if (k == 0) { fail = true; break; }  // a base that does not start at 0
            x.kv = k;
            x.c.assign(xb, xb + w);
            dualInv(x.cinv, xb);
          }
          int n = k - x.kv;
          std::vector<u64> An(blk), Qn(blk, 0), t1(blk), t2(blk);
          scaleBlock(An.data(), xb, x.cinv.data());
          x.A.push_back(An);
          if (n == 0) {
            std::vector<u64> N0 = An;
            N0[0] = submod(N0[0], 1, p);
            x.A0inv.assign(blk, 0);
            if (!poly_inv(x.A0inv.data(), An.data(), pb, p) ||
                !poly_nilpotent(t1.data(), N0.data(), OP_LOG1P, pb, p)) { fail = true; break; }
            scaleBlock(t2.data(), t1.data(), x.ex.data());
            if (!poly_nilpotent(Qn.data(), t2.data(), OP_EXPN, pb, p)) { fail = true; break; }
          } else {
            std::vector<u64> acc(blk, 0), sc(w);
            for (int j = 1; j <= n; ++j) {
              std::fill(t1.begin(), t1.end(), (u64)0);
              poly_mul_acc(t1.data(), x.A[j].data(), x.Q[n - j].data(), pb, p);
              sc[0] = addmod(mulmod(x.ex[0], (u64)j % p, p), red((long long)j - n, p), p);
              for (int c = 1; c < w; ++c) sc[c] = mulmod(x.ex[c], (u64)j % p, p);
              scaleBlock(t2.data(), t1.data(), sc.data());
              for (int c = 0; c < blk; ++c) acc[c] = addmod(acc[c], t2[c], p);
            }
            std::fill(t1.begin(), t1.end(), (u64)0);
            poly_mul_acc(t1.data(), x.A0inv.data(), acc.data(), pb, p);
            u64 in = invmod((u64)n % p, p);
            for (int c = 0; c < blk; ++c) Qn[c] = mulmod(t1[c], in, p);
          }
          x.Q.push_back(Qn);
          for (int mo = 0; mo < nM; ++mo) {
            int dst = pb.shiftSlot(mo, x.axis, 1);
            if (dst < 0) continue;
            for (int c = 0; c < w; ++c)
              o[(size_t)dst * w + c] = addmod(o[(size_t)dst * w + c], Qn[(size_t)mo * w + c], p);
          }
          break;
        }
        default:
          fail = true;
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
    // on a monomial with powers S, d/dt t^n S^m = (n + sum k m (a - 1)) t^(n-1) S^m
    if (!sp.empty())
      for (int mo = 1; mo < nM && !fail; ++mo) {
        std::vector<u64> lam(w, 0), li;
        lam[0] = (u64)(k + 1) % p;
        bool any = false;
        for (const SPowAx& x : sp) {
          int mu = pb.idx[mo][x.axis];
          if (!mu || x.kv < 0) continue;
          any = true;
          u64 km = (u64)((long long)x.kv * mu % (long long)p);
          lam[0] = addmod(lam[0], mulmod(km, submod(x.ex[0], 1, p), p), p);
          for (int c = 1; c < w; ++c) lam[c] = addmod(lam[c], mulmod(km, x.ex[c], p), p);
        }
        if (!any) continue;
        if (lam[0] == 0) { fail = true; break; }
        dualInv(li, lam.data());
        for (int i = 0; i < m; ++i) {
          const u64* fk = val[fOut[i]].data() + (size_t)k * blk + (size_t)mo * w;
          u64* st = val[stateSlots[i]].data() + (size_t)(k + 1) * blk + (size_t)mo * w;
          std::fill(st, st + w, (u64)0);
          dual_mul_acc(st, fk, li.data(), w, p);
        }
      }
  }
  if (fail) return false;
  std::vector<std::vector<u64> > negRows;
  if (anyDeferred) {
    std::map<int, LBlk> L;
    auto get = [&](int slot) -> const LBlk& {
      auto it = L.find(slot);
      if (it != L.end()) return it->second;
      LBlk x;
      x.lo = 0; x.prec = Nrun + 1;
      x.c.assign(val[slot].begin(), val[slot].begin() + (size_t)(Nrun + 1) * blk);
      lb_norm(x, blk);
      return L[slot] = x;
    };
    bool more = false;
    for (int i = 0; i < nInstr && !more; ++i) {
      if (!deferred[i]) continue;
      LBlk r;
      if (op[i] == OP_ADD) r = lb_add(get(a[i]), get(b[i]), blk, p);
      else if (op[i] == OP_MUL) r = lb_mul(get(a[i]), get(b[i]), pb, p);
      else if (!lb_inv(r, get(a[i]), pb, p)) { more = true; break; }
      L[instrBase + i] = r;
    }
    // every output known below the order `need` (Nrun + 1 of the first call), else the
    // recursion runs further
    int deficit = 0;
    if (more) deficit = Nrun + 1;
    else
      for (int gi = 0; gi < (int)gOut.size(); ++gi)
        if (isDef(gOut[gi])) deficit = std::max(deficit, need - L[gOut[gi]].prec);
    if (deficit > 0) {
      if (depth >= 6) return false;
      int Nrun2 = Nrun + deficit;
      std::vector<std::vector<u64> > val2(val.size());
      for (size_t s = 0; s < val.size(); ++s) {
        val2[s].assign((size_t)(Nrun2 + 1) * blk, 0);
        if ((int)s < instrBase) std::copy(val[s].begin(), val[s].begin() + blk, val2[s].begin());
      }
      if (!build_obs_rows_poly(val2, op, a, b, cval, instrBase, stateSlots, fOut, gOut, pb,
                               Nrun2, Ntemit, p, rows, dtau, preRows, depth + 1, need, spowAxis0))
        return false;
      for (size_t s = 0; s < val.size(); ++s)
        std::copy(val2[s].begin(), val2[s].begin() + (size_t)(Nrun + 1) * blk, val[s].begin());
      return true;
    }
    for (int gi = 0; gi < (int)gOut.size(); ++gi) {
      if (!isDef(gOut[gi])) continue;
      const LBlk& y = L[gOut[gi]];
      u64* g = val[gOut[gi]].data();
      std::fill(g, g + (size_t)(Nrun + 1) * blk, (u64)0);
      for (int j = 0; j < lb_len(y, blk); ++j) {
        int t = y.lo + j;
        const u64* src = y.c.data() + (size_t)j * blk;
        if (t > Nrun) break;
        if (t >= 0) { std::copy(src, src + blk, g + (size_t)t * blk); continue; }
        for (int mo = 0; mo < nM; ++mo) {
          if (src[(size_t)mo * w] % p) return false;   // a pole of the value: no series
          // keyed by output, order and monomial, so that column blocks can be joined
          std::vector<u64> row(nz + 3);
          row[0] = (u64)gi; row[1] = (u64)(-t); row[2] = (u64)mo;
          bool nzr = false;
          for (int c = 0; c < nz; ++c) { row[3 + c] = src[(size_t)mo * w + 1 + c]; nzr |= row[3 + c] != 0; }
          if (!nzr) continue;
          if (Ntemit < 0) continue;
          if (!preRows) return false;
          negRows.push_back(row);
        }
      }
    }
  }
  if (preRows)
    for (auto& r : negRows) preRows->push_back(std::move(r));
  if (!sp.empty()) {
    // the coefficient d of t^n S^m in the expansion of an output stands for d c^(m a)
    // t^(n - k m + k m a), with the gradient c^(m a) (grad d + m d (a grad c / c + log c grad a))
    // plus m k d log t grad a. Exponents with distinct (n, m) are independent functions for
    // generic a, so the rows are the gradients with log c grad a dropped against the rows
    // grad a of the log t terms, which the coefficients with m > 0 give. Rows run by output
    // and monomial, then order, and each power adds one output of rows grad a.
    int nS = (int)sp.size();
    std::vector<int> firstK(nS, -1);
    std::vector<std::vector<u64> > pat;
    for (int gi = 0; gi < (int)gOut.size(); ++gi) {
      const u64* g = val[gOut[gi]].data();
      for (int mo = 0; mo < nM; ++mo)
        for (int k = 0; k <= Ntemit; ++k) {
          const u64* d = g + (size_t)k * blk + (size_t)mo * w;
          std::vector<u64> row(nz);
          for (int c = 0; c < nz; ++c) row[c] = d[1 + c];
          u64 d0 = d[0] % p;
          if (d0 && mo > 0) {
            std::vector<u64> pv(nS, 0);
            bool anyS = false;
            for (int j = 0; j < nS; ++j) {
              const SPowAx& x = sp[j];
              int mu = pb.idx[mo][x.axis];
              if (!mu || x.kv < 0) continue;
              anyS = true;
              pv[j] = (u64)((long long)mu * x.kv % (long long)p);
              if (firstK[j] < 0 || k < firstK[j]) firstK[j] = k;
              u64 f = mulmod(mulmod((u64)mu % p, d0, p), mulmod(x.ex[0], x.cinv[0], p), p);
              for (int c = 0; c < nz; ++c)
                row[c] = addmod(row[c], mulmod(f, x.c[1 + c], p), p);
            }
            if (anyS) pat.push_back(pv);
          }
          rows.push_back(row);
        }
    }
    // the rows grad a stand for the log t terms only where those span every power that occurs
    std::vector<int> occ;
    for (int j = 0; j < nS; ++j) if (firstK[j] >= 0) occ.push_back(j);
    if (!occ.empty()) {
      std::vector<std::vector<u64> > P2;
      for (auto& pv : pat) {
        std::vector<u64> r;
        for (int j : occ) r.push_back(pv[j]);
        P2.push_back(r);
      }
      if ((int)rref_mod(P2, p).size() < (int)occ.size()) return false;
    }
    for (int j = 0; j < nS; ++j)
      for (int k = 0; k <= Ntemit; ++k) {
        std::vector<u64> row(nz, 0);
        if (firstK[j] >= 0 && k >= firstK[j])
          for (int c = 0; c < nz; ++c) row[c] = sp[j].ex[1 + c];
        rows.push_back(row);
      }
    return true;
  }
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
// A condition of a shared tape sets leaves: leaf ovSlot[i] takes the value and duals of leaf
// ovSrc[i], or the constant ovNum[i]/ovDen[i] where ovSrc[i] < 0.
struct CondRaw {
  std::vector<int> op, a, b, stateSlots, fOut, gOut, icLeaf, icOp, icA, icB, icOut;
  std::vector<std::string> cnum, cden, icCnum, icCden, icNum, icDen;
  bool hasIcTape, hasIcSeed;
  std::vector<std::vector<int> > icSeedRaw;
  std::vector<int> ovSlot, ovSrc;
  std::vector<std::string> ovNum, ovDen;
};

void extract_leaf_set(const List& tp, CondRaw& cd) {
  if (!tp.containsElementNamed("leaf_slot")) return;
  cd.ovSlot = to_ivec(tp["leaf_slot"]); cd.ovSrc = to_ivec(tp["leaf_src"]);
  cd.ovNum = to_svec(tp["leaf_num"]); cd.ovDen = to_svec(tp["leaf_den"]);
}

CondRaw extract_cond_raw(List tp, int nStates, int w) {
  CondRaw cd;
  extract_leaf_set(tp, cd);
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

// The tapes of a call. A tape with an entry `base` holds the arrays of the earlier tape `base` and
// only the leaves it sets, so that the arrays of a shared tape are read once.
struct TapeSet {
  std::vector<CondRaw> td;
  std::vector<int> base;
  TapeSet(List& tapes, int nStates, int w) {
    int T = tapes.size();
    td.resize(T);
    base.resize(T);
    for (int t = 0; t < T; ++t) {
      List tp = tapes[t];
      base[t] = t;
      if (tp.containsElementNamed("base")) {
        base[t] = (int)tp["base"];
        if (base[t] < 0 || base[t] >= t || base[base[t]] != base[t])
          throw std::invalid_argument("a tape names a base that is no earlier full tape");
        extract_leaf_set(tp, td[t]);
      } else {
        td[t] = extract_cond_raw(tp, nStates, w);
      }
    }
  }
  int size() const { return (int)td.size(); }
  // tape t, assembled in `tmp` where it holds the arrays of a base
  const CondRaw& get(int t, CondRaw& tmp) const {
    if (base[t] == t) return td[t];
    tmp = td[base[t]];
    tmp.ovSlot = td[t].ovSlot; tmp.ovSrc = td[t].ovSrc;
    tmp.ovNum = td[t].ovNum; tmp.ovDen = td[t].ovDen;
    return tmp;
  }
};

// The leaves of a condition: the shared leaf values with the leaves the condition sets replaced.
struct LeafView {
  const std::vector<std::vector<u64> >& base;
  std::vector<int> at;
  std::vector<std::vector<u64> > own;
  explicit LeafView(const std::vector<std::vector<u64> >& b) : base(b) {}
  const std::vector<u64>& operator[](int L) const {
    return (!at.empty() && at[L] >= 0) ? own[at[L]] : base[L];
  }
};

LeafView cond_leaves(const CondRaw& cd, const std::vector<std::vector<u64> >& leafVal, int w, u64 p) {
  LeafView v(leafVal);
  if (cd.ovSlot.empty()) return v;
  v.at.assign(leafVal.size(), -1);
  for (size_t i = 0; i < cd.ovSlot.size(); ++i) {
    v.at[cd.ovSlot[i]] = (int)v.own.size();
    if (cd.ovSrc[i] >= 0) {
      v.own.push_back(leafVal[cd.ovSrc[i]]);
    } else {
      std::vector<u64> c(w, 0);
      c[0] = reduce_rational(cd.ovNum[i], cd.ovDen[i], p);
      v.own.push_back(c);
    }
  }
  return v;
}

// The constants of a condition's tape modulo p and the start of each state as a dual of width w.
// False on a vanishing denominator of a start value.
bool cond_start(const CondRaw& cd, const std::vector<std::vector<u64> >& leafVal,
                int nLeaves, int nStates, int w, u64 p, std::vector<u64>& cval,
                std::vector<std::vector<u64> >& st0) {
  int nInstr = (int)cd.op.size();
  cval.assign(nInstr, 0);
  for (int i = 0; i < nInstr; ++i)
    if (cd.op[i] == OP_CONST)
      cval[i] = reduce_rational(cd.cnum[i], cd.cden[i], p);

  const LeafView lv = cond_leaves(cd, leafVal, w, p);
  std::vector<std::vector<u64> > icVal;
  if (cd.hasIcTape && !cd.hasIcSeed) {
    int nIc = (int)cd.icOp.size();
    std::vector<u64> icCval(nIc, 0);
    for (int i = 0; i < nIc; ++i)
      if (cd.icOp[i] == OP_CONST)
        icCval[i] = reduce_rational(cd.icCnum[i], cd.icCden[i], p);
    // the leaves the initial-value tape reads
    icVal.assign(nLeaves + nIc, std::vector<u64>());
    auto need = [&](int s) { if (s >= 0 && s < nLeaves && icVal[s].empty()) icVal[s] = lv[s]; };
    for (int i = 0; i < nIc; ++i) {
      icVal[nLeaves + i].assign(w, 0);
      if (cd.icOp[i] == OP_CONST) continue;
      need(cd.icA[i]);
      if (cd.icOp[i] == OP_ADD || cd.icOp[i] == OP_MUL) need(cd.icB[i]);
    }
    if (!eval_ic_order0(icVal, cd.icOp, cd.icA, cd.icB, icCval, nLeaves, w, p))
      return false;
  }

  std::vector<u64> icConst;
  if (!cd.hasIcTape && !cd.hasIcSeed) {
    icConst.assign(nStates, 0);
    for (int i = 0; i < nStates; ++i)
      icConst[i] = reduce_rational(cd.icNum[i], cd.icDen[i], p);
  }

  st0.assign(cd.stateSlots.size(), std::vector<u64>(w, 0));
  for (int i = 0; i < (int)cd.stateSlots.size(); ++i) {
    if (cd.hasIcSeed) {
      for (int c = 0; c < w; ++c) st0[i][c] = red(cd.icSeedRaw[i][c], p);
      continue;
    }
    int src = cd.hasIcTape ? cd.icOut[i] : (cd.icLeaf[i] >= 0 ? cd.icLeaf[i] : -1);
    if (src >= 0) {
      const std::vector<u64>& s = src < nLeaves ? lv[src] : icVal[src];
      for (int c = 0; c < w; ++c) st0[i][c] = s[c];
    } else {
      st0[i][0] = icConst[i];
    }
  }
  return true;
}

// Build one condition's observability rows at a single point/prime. leafVal holds
// each leaf's value and dual already reduced mod p; the tape and IC constants are
// reduced from their strings here, so the same CondRaw serves any prime. Returns
// false on a vanishing denominator (tape reciprocal or reciprocal IC).
bool build_one_condition(const CondRaw& cd,
                         const std::vector<std::vector<u64> >& leafVal,
                         int nLeaves, int nStates, int nz, int w, int Nt, u64 p,
                         std::vector<std::vector<u64> >& outRows, bool laurent = false) {
  int nInstr = (int)cd.op.size();
  int instrBase = nLeaves + nStates;
  int S = instrBase + nInstr;
  std::vector<u64> cval;
  std::vector<std::vector<u64> > st0;
  if (!cond_start(cd, leafVal, nLeaves, nStates, w, p, cval, st0)) return false;

  std::vector<std::vector<u64> > val(S, std::vector<u64>((size_t)(Nt + 1) * w, 0));
  {
    const LeafView lv = cond_leaves(cd, leafVal, w, p);
    for (int L = 0; L < nLeaves; ++L)
      for (int c = 0; c < w; ++c) val[L][c] = lv[L][c];
  }
  for (int i = 0; i < (int)cd.stateSlots.size(); ++i)
    for (int c = 0; c < w; ++c) val[cd.stateSlots[i]][c] = st0[i][c];

  if (laurent) {
    // one axis per power of a state at 0, within the total degree Nt (a power counts at
    // least one order)
    int nS = 0;
    for (int i = 0; i < nInstr; ++i) if (cd.op[i] == OP_SPOW) ++nS;
    PolyBasis pb(nS > 0 ? nS : 1, nS > 0 ? Nt : 0, w);
    if (pb.nMono > 1) {
      // the seeds at order 0 sit at monomial 0 of a wider block
      size_t blk = (size_t)pb.nMono * w;
      for (int sl = 0; sl < S; ++sl) {
        std::vector<u64> wide((size_t)(Nt + 1) * blk, 0);
        if (sl < instrBase) std::copy(val[sl].begin(), val[sl].begin() + w, wide.begin());
        val[sl].swap(wide);
      }
    }
    return build_obs_rows_poly(val, cd.op, cd.a, cd.b, cval, instrBase, cd.stateSlots,
                               cd.fOut, cd.gOut, pb, Nt, Nt, p, outRows, nullptr, nullptr, 0,
                               -1, nS > 0 ? 0 : -1);
  }
  if (build_obs_rows(val, cd.op, cd.a, cd.b, cval, instrBase, cd.stateSlots,
                     cd.fOut, cd.gOut, nz, w, Nt, p, outRows))
    return true;
  // a reciprocal without constant value: the polyseries path takes it as a Laurent series
  return build_one_condition(cd, leafVal, nLeaves, nStates, nz, w, Nt, p, outRows, true);
}

// SYMIDENT_KERNEL_MODE: `forward` or `reverse` fixes the mode of the rows, `check` builds a call in
// both and requires equal rows; otherwise the mode that sweeps fewer residues.
enum { KM_AUTO, KM_FORWARD, KM_REVERSE, KM_CHECK };
int kernel_mode() {
  const char* e = std::getenv("SYMIDENT_KERNEL_MODE");
  std::string m = e ? e : "";
  if (m == "forward") return KM_FORWARD;
  if (m == "reverse") return KM_REVERSE;
  if (m == "check") return KM_CHECK;
  return KM_AUTO;
}

// rows of the two modes, equal modulo p, or an error naming the call
void require_same_rows(const std::vector<std::vector<u64> >& fw,
                       const std::vector<std::vector<u64> >& rv, u64 p, const char* what) {
  bool same = fw.size() == rv.size();
  for (size_t i = 0; same && i < fw.size(); ++i) {
    same = fw[i].size() == rv[i].size();
    for (size_t c = 0; same && c < fw[i].size(); ++c) same = fw[i][c] % p == rv[i][c] % p;
  }
  if (!same)
    throw std::runtime_error(std::string("kernel: the reverse mode differs from the forward mode (") +
                             what + ")");
}

// dst += c * src over n residues
inline void axpy_mod(u64* dst, const u64* src, u64 c, int n, const Barrett& br) {
  for (int j = 0; j < n; ++j) dst[j] = br.red(dst[j] + src[j] * c);
}

// ---- Reverse mode of build_one_condition ----
// A pass of the values alone, then adjoint sweeps over blocks of rows. Row r is output r % nOut at
// order Nt - r / nOut, so the sweep down from order j covers a prefix of the rows.
struct RevPass {
  int nOut = 0, Nt = 0, S = 0, nMarked = 0;
  std::vector<std::vector<u64> > v;                     // Taylor values per slot
  std::vector<std::vector<u64> > nsq;                   // per reciprocal: -(o*o) as a series
  std::vector<std::vector<std::pair<int, u64> > > d0;  // per slot: the duals at order 0
  std::vector<char> tconst;                             // values constant in time
  std::vector<int> mark;                                // slot*(Nt+1)+k: adjoint index or -1
};

enum { REV_OK, REV_FAIL, REV_FORWARD };

// The orders ii of A in sum_ii A_ii B_(k-ii) where neither factor vanishes for being constant in time
inline void conv_range(bool ca, bool cb, int k, int& j0, int& j1) {
  j0 = cb ? k : 0;
  j1 = ca ? 0 : k;
}

// The values and the adjoint cells of the reverse mode. REV_FAIL on a vanishing denominator of a
// start value; REV_FORWARD for an operation other than CONST/ADD/MUL/INV or a reciprocal without
// value at order 0, which the forward mode takes as a Laurent series.
int rev_values(const CondRaw& cd, const std::vector<std::vector<u64> >& leafVal,
               int nLeaves, int nStates, int w, int Nt, u64 p, RevPass& rp) {
  int nInstr = (int)cd.op.size(), instrBase = nLeaves + nStates;
  int S = instrBase + nInstr, N1 = Nt + 1;
  for (int o : cd.op)
    if (o != OP_CONST && o != OP_ADD && o != OP_MUL && o != OP_INV) return REV_FORWARD;
  std::vector<u64> cval;
  std::vector<std::vector<u64> > st0;
  if (!cond_start(cd, leafVal, nLeaves, nStates, w, p, cval, st0)) return REV_FAIL;
  rp.nOut = (int)cd.gOut.size(); rp.Nt = Nt; rp.S = S;
  rp.v.assign(S, std::vector<u64>(N1, 0));
  rp.d0.assign(S, {});
  std::vector<char> isState(S, 0);
  for (int s : cd.stateSlots) isState[s] = 1;
  rp.tconst.assign(S, 0);
  for (int L = 0; L < instrBase; ++L) rp.tconst[L] = !isState[L];
  const LeafView lv = cond_leaves(cd, leafVal, w, p);
  for (int L = 0; L < nLeaves; ++L) {
    const std::vector<u64>& x = lv[L];
    rp.v[L][0] = x[0] % p;
    for (int c = 1; c < w; ++c)
      if (x[c] % p) rp.d0[L].push_back({c - 1, x[c] % p});
  }
  for (size_t i = 0; i < cd.stateSlots.size(); ++i) {
    int s = cd.stateSlots[i];
    rp.v[s][0] = st0[i][0] % p;
    rp.d0[s].clear();
    for (int c = 1; c < w; ++c)
      if (st0[i][c] % p) rp.d0[s].push_back({c - 1, st0[i][c] % p});
  }
  for (int i = 0; i < nInstr; ++i) {
    int o = cd.op[i], a = cd.a[i], b = cd.b[i];
    rp.tconst[instrBase + i] = o == OP_CONST ||
        ((o == OP_ADD || o == OP_MUL) && rp.tconst[a] && rp.tconst[b]) ||
        (o == OP_INV && rp.tconst[a]);
  }
  const Barrett br(p);
  for (int k = 0; k <= Nt; ++k) {
    for (int i = 0; i < nInstr; ++i) {
      int s = instrBase + i;
      if (k > 0 && rp.tconst[s]) continue;
      int a = cd.a[i], b = cd.b[i];
      u64& o = rp.v[s][k];
      switch (cd.op[i]) {
        case OP_CONST: o = cval[i]; break;
        case OP_ADD: o = addmod(rp.v[a][k], rp.v[b][k], p); break;
        case OP_MUL: {
          int j0, j1;
          conv_range(rp.tconst[a], rp.tconst[b], k, j0, j1);
          u64 acc = 0;
          for (int ii = j0; ii <= j1; ++ii) acc = br.red(acc + rp.v[a][ii] * rp.v[b][k - ii]);
          o = acc;
          break;
        }
        case OP_INV:
          if (k == 0) {
            if (rp.v[a][0] == 0) return REV_FORWARD;
            o = invmod(rp.v[a][0], p);
          } else {
            u64 acc = 0;
            for (int j = 1; j <= k; ++j) acc = br.red(acc + rp.v[a][j] * rp.v[s][k - j]);
            o = mulmod(submod(0, acc, p), rp.v[s][0], p);
          }
          break;
      }
    }
    if (k == Nt) break;
    u64 ik = invmod((u64)(k + 1), p);
    for (size_t i = 0; i < cd.stateSlots.size(); ++i)
      rp.v[cd.stateSlots[i]][k + 1] = mulmod(rp.v[cd.fOut[i]][k], ik, p);
  }
  rp.nsq.assign(nInstr, {});
  for (int i = 0; i < nInstr; ++i) {
    if (cd.op[i] != OP_INV) continue;
    const std::vector<u64>& o = rp.v[instrBase + i];
    rp.nsq[i].assign(N1, 0);
    int top = rp.tconst[instrBase + i] ? 1 : N1;
    for (int k = 0; k < top; ++k) {
      u64 s = 0;
      for (int j = 0; j <= k; ++j) s = br.red(s + o[j] * o[k - j]);
      rp.nsq[i][k] = submod(0, s, p);
    }
  }
  // the cells an adjoint reaches: a leaf or a start with a dual at order 0, a state past order 0,
  // an instruction that is not a constant, past order 0 only where it varies in time
  auto live = [&](int s, int k) {
    if (s < instrBase) return isState[s] ? (k > 0 || !rp.d0[s].empty()) : (k == 0 && !rp.d0[s].empty());
    if (cd.op[s - instrBase] == OP_CONST) return false;
    return k == 0 || !rp.tconst[s];
  };
  rp.mark.assign((size_t)S * N1, -1);
  rp.nMarked = 0;
  auto mark = [&](int s, int k) {
    int& m = rp.mark[(size_t)s * N1 + k];
    if (m < 0 && live(s, k)) m = rp.nMarked++;
  };
  for (int g : cd.gOut)
    for (int k = 0; k <= Nt; ++k) mark(g, k);
  for (int k = Nt; k >= 0; --k) {
    if (k < Nt)
      for (size_t i = 0; i < cd.stateSlots.size(); ++i)
        if (rp.mark[(size_t)cd.stateSlots[i] * N1 + k + 1] >= 0) mark(cd.fOut[i], k);
    for (int i = nInstr - 1; i >= 0; --i) {
      if (rp.mark[(size_t)(instrBase + i) * N1 + k] < 0) continue;
      int a = cd.a[i], b = cd.b[i];
      switch (cd.op[i]) {
        case OP_ADD: mark(a, k); mark(b, k); break;
        case OP_MUL: {
          int j0, j1;
          conv_range(rp.tconst[a], rp.tconst[b], k, j0, j1);
          for (int ii = j0; ii <= j1; ++ii) {
            if (rp.v[b][k - ii]) mark(a, ii);
            if (rp.v[a][ii]) mark(b, k - ii);
          }
          break;
        }
        case OP_INV:
          for (int ii = 0; ii <= (rp.tconst[a] ? 0 : k); ++ii)
            if (rp.nsq[i][k - ii]) mark(a, ii);
          break;
      }
    }
  }
  return REV_OK;
}

// The adjoint sweep over rows [r0, r1) of the order-descending numbering; `out` receives the
// entries of each row over the nz columns.
void rev_sweep(const CondRaw& cd, const RevPass& rp, int nLeaves, int nStates, int nz,
               u64 p, int r0, int r1, std::vector<std::vector<u64> >& out) {
  int B = r1 - r0, Nt = rp.Nt, N1 = Nt + 1, nOut = rp.nOut;
  int nInstr = (int)cd.op.size(), instrBase = nLeaves + nStates;
  const Barrett br(p);
  std::vector<u64> pool((size_t)rp.nMarked * B, 0);
  auto at = [&](int s, int k) -> u64* {
    int m = rp.mark[(size_t)s * N1 + k];
    return m < 0 ? nullptr : pool.data() + (size_t)m * B;
  };
  for (int k = Nt; k >= 0; --k) {
    int act = std::min(B, nOut * (Nt - k + 1) - r0);
    if (act <= 0) continue;
    for (int r = std::max(r0, nOut * (Nt - k)); r < std::min(r1, nOut * (Nt - k + 1)); ++r)
      if (u64* g = at(cd.gOut[r % nOut], k)) g[r - r0] = addmod(g[r - r0], 1, p);
    if (k < Nt) {
      u64 ik = invmod((u64)(k + 1), p);
      for (size_t i = 0; i < cd.stateSlots.size(); ++i) {
        const u64* src = at(cd.stateSlots[i], k + 1);
        u64* dst = at(cd.fOut[i], k);
        if (src && dst) axpy_mod(dst, src, ik, act, br);
      }
    }
    for (int i = nInstr - 1; i >= 0; --i) {
      const u64* o = at(instrBase + i, k);
      if (!o) continue;
      int a = cd.a[i], b = cd.b[i];
      switch (cd.op[i]) {
        case OP_ADD:
          if (u64* d = at(a, k)) axpy_mod(d, o, 1, act, br);
          if (u64* d = at(b, k)) axpy_mod(d, o, 1, act, br);
          break;
        case OP_MUL: {
          int j0, j1;
          conv_range(rp.tconst[a], rp.tconst[b], k, j0, j1);
          for (int ii = j0; ii <= j1; ++ii) {
            u64 vb = rp.v[b][k - ii], va = rp.v[a][ii];
            if (vb) if (u64* d = at(a, ii)) axpy_mod(d, o, vb, act, br);
            if (va) if (u64* d = at(b, k - ii)) axpy_mod(d, o, va, act, br);
          }
          break;
        }
        case OP_INV:
          for (int ii = 0; ii <= (rp.tconst[a] ? 0 : k); ++ii) {
            u64 c = rp.nsq[i][k - ii];
            if (c) if (u64* d = at(a, ii)) axpy_mod(d, o, c, act, br);
          }
          break;
      }
    }
  }
  out.assign(B, std::vector<u64>(nz, 0));
  for (int s = 0; s < rp.S; ++s) {
    if (rp.d0[s].empty()) continue;
    const u64* ad = at(s, 0);
    if (!ad) continue;
    for (int rr = 0; rr < B; ++rr) {
      if (!ad[rr]) continue;
      for (const auto& e : rp.d0[s])
        out[rr][e.first] = br.red(out[rr][e.first] + ad[rr] * e.second);
    }
  }
}

double chain_block_bytes();

// The rows of build_one_condition in reverse mode, in the order of the forward mode; REV_FAIL and
// REV_FORWARD as rev_values.
int build_one_condition_reverse(const CondRaw& cd,
                                const std::vector<std::vector<u64> >& leafVal,
                                int nLeaves, int nStates, int nz, int Nt, u64 p,
                                std::vector<std::vector<u64> >& outRows, int threads) {
  RevPass rp;
  int st = rev_values(cd, leafVal, nLeaves, nStates, nz + 1, Nt, p, rp);
  if (st != REV_OK) return st;
  int nRows = rp.nOut * (Nt + 1);
  if (nRows == 0) return REV_OK;
  // rows per block: split over the threads, and bounded so that one block's adjoints stay within
  // the budget of chain_block_bytes()
  double perRow = std::max(1.0, (double)rp.nMarked * 8.0);
  int Bmem = std::max(1, (int)(chain_block_bytes() / perRow));
  int Bt = threads > 1 ? (nRows + threads - 1) / threads : nRows;
  int B = std::max(1, std::min(Bt, Bmem));
  int nb = (nRows + B - 1) / B;
  std::vector<std::vector<std::vector<u64> > > rB(nb);
  int nThr = std::max(1, std::min(threads, nb));
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
  for (int b = 0; b < nb; ++b) {
    try {
      rev_sweep(cd, rp, nLeaves, nStates, nz, p, b * B, std::min(nRows, (b + 1) * B), rB[b]);
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  size_t r0 = outRows.size();
  outRows.resize(r0 + nRows);
  for (int b = 0; b < nb; ++b)
    for (size_t i = 0; i < rB[b].size(); ++i) {
      int r = b * B + (int)i, gi = r % rp.nOut, k = Nt - r / rp.nOut;
      outRows[r0 + (size_t)gi * (Nt + 1) + k] = std::move(rB[b][i]);
    }
  return REV_OK;
}

// The reverse mode sweeps about half the rows per order, the forward mode every column.
bool reverse_pays(size_t nOut, int nz, int Nt) {
  return (double)nOut * (Nt + 2) < (double)(nz + 1);
}

// build_one_condition over column blocks on `threads` threads: each block holds the
// value and a separate range of dual lanes, the rows are assembled over all columns. A steady-state
// seed is laid out over all columns, so a seeded condition is built unblocked.
bool build_one_condition_forward(const CondRaw& cd,
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
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
  for (int b = 0; b < nb; ++b) {
    try {
      int s0 = b * B, s1 = std::min(nz, s0 + B), nzB = s1 - s0;
      if (nzB <= 0) continue;
      std::vector<std::vector<u64> > lv(nLeaves, std::vector<u64>(nzB + 1, 0));
      for (int L = 0; L < nLeaves; ++L) {
        lv[L][0] = leafVal[L][0];
        if (dualCol[L] >= s0 && dualCol[L] < s1) lv[L][1 + dualCol[L] - s0] = 1;
      }
      ok[b] = build_one_condition(cd, lv, nLeaves, nStates, nzB, nzB + 1, Nt, p, rB[b]);
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
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

// The rows of build_one_condition in the mode of kernel_mode(), on `threads` threads.
bool build_one_condition_blocked(const CondRaw& cd,
                                 const std::vector<std::vector<u64> >& leafVal,
                                 int nLeaves, int nStates, int nz, int Nt, u64 p,
                                 std::vector<std::vector<u64> >& outRows, int threads) {
  int mode = kernel_mode();
  if (mode == KM_FORWARD || (mode == KM_AUTO && !reverse_pays(cd.gOut.size(), nz, Nt)))
    return build_one_condition_forward(cd, leafVal, nLeaves, nStates, nz, Nt, p, outRows, threads);
  std::vector<std::vector<u64> > rv;
  int st = build_one_condition_reverse(cd, leafVal, nLeaves, nStates, nz, Nt, p, rv, threads);
  if (st == REV_FORWARD)
    return build_one_condition_forward(cd, leafVal, nLeaves, nStates, nz, Nt, p, outRows, threads);
  if (mode == KM_CHECK) {
    std::vector<std::vector<u64> > fw;
    bool okF = build_one_condition_forward(cd, leafVal, nLeaves, nStates, nz, Nt, p, fw, threads);
    if (okF != (st == REV_OK))
      throw std::runtime_error("kernel: the reverse mode fails where the forward mode does not");
    require_same_rows(fw, rv, p, "one condition");
  }
  if (st == REV_FAIL) return false;
  for (auto& r : rv) outRows.push_back(std::move(r));
  return true;
}

// One chain segment in thread-safe plain C++: the regime-substituted tape, the
// reduced constants, and the optional first-segment steady-state seed, IC tape
// and state-dose event map. Constants are reduced against the single call prime.
struct SegRaw {
  std::vector<int> op, a, b, stateSlots, fOut, gOut;
  std::vector<u64> cval;
  bool hasIcSeed, hasIcTape, hasEv, evStates = false, evReadsStates = false;
  std::vector<std::vector<u64> > icSeed;
  std::vector<int> icOp, icA, icB, icOut;
  std::vector<u64> icCval;
  std::vector<int> evVarIdx, evMethod, evOp, evA, evB, evOut;
  std::vector<u64> evCval;
  bool hasTm;
  std::vector<int> tmOp, tmA, tmB, tmOut;
  std::vector<u64> tmCval;
};

// A tape whose slots below `base` are leaves and states, rewritten to read leaf slot[i] from leaf
// src[i], or from the constant cv[i] where src[i] < 0: the constants become instructions ahead of the
// others. `outs` are further slot arrays of the tape, remapped alike.
void rebind_leaves(std::vector<int>& op, std::vector<int>& a, std::vector<int>& b,
                   std::vector<u64>& cval, int base, const std::vector<int>& slot,
                   const std::vector<int>& src, const std::vector<u64>& cv,
                   std::initializer_list<std::vector<int>*> outs) {
  std::vector<int> to(base, -1);
  std::vector<u64> consts;
  for (size_t i = 0; i < slot.size(); ++i) {
    if (slot[i] < 0 || slot[i] >= base) continue;
    if (src[i] >= 0) {
      to[slot[i]] = src[i];
    } else {
      to[slot[i]] = base + (int)consts.size();
      consts.push_back(cv[i]);
    }
  }
  int nC = (int)consts.size();
  auto m = [&](int x) { return x < 0 ? x : x >= base ? x + nC : to[x] >= 0 ? to[x] : x; };
  for (size_t i = 0; i < op.size(); ++i) {
    if (op[i] == OP_CONST) continue;
    a[i] = m(a[i]);
    if (op[i] == OP_ADD || op[i] == OP_MUL || op[i] == OP_SPOW) b[i] = m(b[i]);
  }
  for (std::vector<int>* v : outs)
    for (int& x : *v) x = m(x);
  op.insert(op.begin(), nC, OP_CONST);
  a.insert(a.begin(), nC, 0);
  b.insert(b.begin(), nC, 0);
  cval.insert(cval.begin(), consts.begin(), consts.end());
}

SegRaw extract_seg_raw(List seg, int nLeaves, int nStates, int w, u64 p) {
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
  // event tapes flagged ev_state_slots read the states at slots [nLeaves, nLeaves + nStates); others read leaves only
  s.evStates = seg.containsElementNamed("ev_state_slots") && (int)seg["ev_state_slots"] != 0;
  if (s.hasEv) {
    s.evVarIdx = to_ivec(seg["ev_var_idx"]); s.evMethod = to_ivec(seg["ev_method"]);
    s.evOp = to_ivec(seg["ev_op"]); s.evA = to_ivec(seg["ev_a"]); s.evB = to_ivec(seg["ev_b"]);
    s.evOut = to_ivec(seg["ev_out"]);
    if (s.evStates) {
      std::vector<int> leafOnly;
      if (seg.containsElementNamed("ev_leaf_only")) leafOnly = to_ivec(seg["ev_leaf_only"]);
      s.evReadsStates = leafOnly.size() != s.evVarIdx.size() ||
                        std::find(leafOnly.begin(), leafOnly.end(), 0) != leafOnly.end();
    }
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
  if (seg.containsElementNamed("leaf_slot")) {
    // the leaves a condition of a shared tape sets
    std::vector<int> slot = to_ivec(seg["leaf_slot"]), src = to_ivec(seg["leaf_src"]);
    std::vector<std::string> num = to_svec(seg["leaf_num"]), den = to_svec(seg["leaf_den"]);
    std::vector<u64> cv(slot.size(), 0);
    for (size_t i = 0; i < slot.size(); ++i)
      if (src[i] < 0) cv[i] = reduce_rational(num[i], den[i], p);
    rebind_leaves(s.op, s.a, s.b, s.cval, nLeaves + nStates, slot, src, cv, {&s.fOut, &s.gOut});
    if (s.hasIcTape) rebind_leaves(s.icOp, s.icA, s.icB, s.icCval, nLeaves, slot, src, cv, {&s.icOut});
    if (s.hasEv)
      rebind_leaves(s.evOp, s.evA, s.evB, s.evCval, nLeaves + (s.evStates ? nStates : 0), slot, src, cv,
                    {&s.evOut});
    if (s.hasTm) rebind_leaves(s.tmOp, s.tmA, s.tmB, s.tmCval, nLeaves, slot, src, cv, {&s.tmOut});
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
      case OP_EXPN: case OP_LOG1P: case OP_SINN: case OP_COSN:
        if (!poly_nilpotent(o, V[a[i]].data(), op[i], pb, p)) return false;
        break;
      default: if (!poly_inv(o, V[a[i]].data(), pb, p)) return false;
    }
  }
  return true;
}

// What the reverse mode reads of a value pass of a chain: per segment the Taylor values of every
// slot, the orders run and whether a boundary row follows; per boundary the start of the next
// segment before the events; whether an instruction was taken as a Laurent series.
struct ChainTrace {
  std::vector<std::vector<std::vector<u64> > > val;
  std::vector<int> nrun;
  std::vector<char> kink;
  std::vector<std::vector<std::vector<u64> > > preEvents;
  char deferred = 0;
};

// Build one condition's chain (all segments) at a single point/prime, propagating
// the state across each gap as a formal power series and applying state-dose
// events to the propagated state. leafPt holds the leaf residues; the z-leaves seed a dual
// unit. Returns false on a vanishing denominator. Reads only flat vectors, so threads may call it.
bool build_one_chain(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                     int nz, int w, const std::vector<int>& dualCol,
                     const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                     std::vector<std::vector<u64> >& rows,
                     std::vector<std::vector<u64> >& srows,
                     const std::vector<std::vector<u64> >* ser = nullptr,
                     const std::vector<int>* lane = nullptr,
                     const std::vector<char>* movesIn = nullptr, int Tser = -1,
                     bool keepZeroSer = false, ChainTrace* tr = nullptr) {
  int instrBase = nLeaves + nStates;
  int nSeg = (int)segs.size();
  int K = nSeg - 1; if (K < 1) K = 1;
  // a continuation series in the leaves (coefficients of eps^1..eps^T) on one more axis,
  // read on the line with coefficient 1; in line mode that axis has a separate order Tser
  int serAxis = -1;
  if (ser) { serAxis = K; K += 1; }
  bool lineMode = ser && Tser >= 0;
  PolyBasis pb(K, Mtot, w, lineMode ? serAxis : -1, Tser);
  int nM = pb.nMono, blk = nM * w;
  int Ns = lineMode ? Tser + 1 : Mtot + 1;
  if (lineMode && !segs.empty() && segs[0].hasIcSeed) return false;

  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(blk, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)leafPt[L] % (long long)p; if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }
  if (ser) {
    std::vector<int> mu(K, 0);
    for (int L = 0; L < nLeaves; ++L)
      for (int j = 1; j <= (int)(*ser)[L].size() && j <= (lineMode ? Tser : Mtot); ++j) {
        mu[serAxis] = j;
        leafVal[L][(size_t)pb.slotOf.at(mu) * w] = (*ser)[L][j - 1] % p;
      }
  }
  // in line mode an event time must not move along the line: the gaps stay formal
  if (lineMode)
    for (int sj = 0; sj < nSeg; ++sj) {
      const SegRaw& sg = segs[sj];
      if (!sg.hasTm) continue;
      std::vector<std::vector<u64> > tmPoly(nLeaves + sg.tmOp.size(), std::vector<u64>(blk, 0));
      for (int L = 0; L < nLeaves; ++L) tmPoly[L] = leafVal[L];
      if (!eval_tape_poly(tmPoly, sg.tmOp, sg.tmA, sg.tmB, sg.tmCval, nLeaves, pb, p)) return false;
      const std::vector<u64>& tv = tmPoly[sg.tmOut[0]];
      for (int mo = 0; mo < nM; ++mo)
        if (pb.idx[mo][serAxis] > 0 && tv[(size_t)mo * w]) return false;
    }

  // each segment's left boundary time with duals; the gaps between them set the
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
                             moves[sj] ? tau[sj].data() + 1 : nullptr, nullptr, 0, -1, -1,
                             tr ? &tr->deferred : nullptr))
      return false;
    bool keepZero = lane != nullptr || keepZeroSer;
    if (lineMode) rows_to_series_axis(rows, from, pb, nz, Ns, srows, keepZero);
    else rows_to_line(rows, from, pb, lineC, nz, Mtot + 1, p, srows, keepZero);
    // where the output is not analytic across a moving boundary, the boundary time is
    // itself read off the output: the gradient of the boundary time joins the rows
    bool kink = false;
    if (sj > 0 && moves[sj]) {
      kink = leftJet.size() != seg.gOut.size();
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
        std::vector<u64> srow((size_t)nz * Ns, 0);
        for (int c = 0; c < nz; ++c) srow[(size_t)c * Ns] = row[c];
        rows.push_back(row);
        srows.push_back(srow);
      }
    }
    if (tr) tr->kink.push_back(kink);
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
      if (tr) tr->preEvents.push_back(carry);
      const SegRaw& nxt = segs[sj + 1];
      if (nxt.hasEv) {
        // event values over the leaves and the states just before the events
        int evBase = nLeaves + (nxt.evStates ? nStates : 0);
        std::vector<std::vector<u64> > evVal(evBase + nxt.evOp.size(), std::vector<u64>(blk, 0));
        for (int L = 0; L < nLeaves; ++L) evVal[L] = leafVal[L];
        if (nxt.evStates)
          for (int i = 0; i < nStates; ++i) evVal[nLeaves + i] = carry[i];
        if (!eval_tape_poly(evVal, nxt.evOp, nxt.evA, nxt.evB, nxt.evCval, evBase, pb, p))
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
      }
    }
    if (tr) { tr->nrun.push_back(Nrun); tr->val.push_back(std::move(val)); }
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

// ---- Reverse mode of build_one_chain ----
// One value pass, then adjoint sweeps from the last segment to the first through the events and
// the gap shifts. Rows run by descending segment, then order, so that each sweep covers a prefix.
int build_one_chain_reverse(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                            int nz, const std::vector<int>& dualCol,
                            const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                            std::vector<std::vector<u64> >& rows,
                            std::vector<std::vector<u64> >& srows,
                            const std::vector<std::vector<u64> >* ser, int threads, int Tser,
                            bool keepZeroSer) {
  int w = nz + 1, nSeg = (int)segs.size(), instrBase = nLeaves + nStates;
  // the sweep takes an event value as a function of the leaves alone
  for (const SegRaw& sg : segs) {
    if (sg.evReadsStates) return REV_FORWARD;
    for (int o : sg.op)
      if (o != OP_CONST && o != OP_ADD && o != OP_MUL && o != OP_INV) return REV_FORWARD;
  }
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, w, dualCol, leafPt, p, moves)) return REV_FORWARD;
  ChainTrace tr;
  {
    std::vector<int> none(nLeaves, -1);
    std::vector<std::vector<u64> > r1, s1;
    if (!build_one_chain(segs, nLeaves, nStates, 0, 1, none, leafPt, Nt, Mtot, p, r1, s1, ser,
                         nullptr, &moves, Tser, false, &tr) || tr.deferred)
      return REV_FORWARD;
  }
  int K = nSeg - 1; if (K < 1) K = 1;
  int serAxis = -1;
  if (ser) { serAxis = K; K += 1; }
  bool lineMode = ser && Tser >= 0;
  PolyBasis pb(K, Mtot, 1, lineMode ? serAxis : -1, Tser);
  int nM = pb.nMono, blkW = nM * w;
  int Ns = lineMode ? Tser + 1 : Mtot + 1;

  // the sources of duals, in forward mode over all columns: the leaves, the boundary times, the
  // start of the first segment and the event values
  std::vector<std::vector<u64> > leafW(nLeaves, std::vector<u64>(blkW, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)leafPt[L] % (long long)p; if (pv < 0) pv += p;
    leafW[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafW[L][1 + dualCol[L]] = 1;
  }
  if (ser) {
    std::vector<int> mu(K, 0);
    for (int L = 0; L < nLeaves; ++L)
      for (int j = 1; j <= (int)(*ser)[L].size() && j <= (lineMode ? Tser : Mtot); ++j) {
        mu[serAxis] = j;
        leafW[L][(size_t)pb.slotOf.at(mu) * w] = (*ser)[L][j - 1] % p;
      }
  }
  std::unique_ptr<PolyBasis> pbW;
  if (ser) pbW.reset(new PolyBasis(K, Mtot, w, lineMode ? serAxis : -1, Tser));
  // a side tape on the leaves: polyseries with a series, else order 0 at monomial 0; the slots
  // from nLeaves to `base` are states the values do not read
  auto side = [&](const std::vector<int>& op, const std::vector<int>& a, const std::vector<int>& b,
                  const std::vector<u64>& cval, std::vector<std::vector<u64> >& V, int base) {
    if (ser) {
      V.assign(base + op.size(), std::vector<u64>(blkW, 0));
      for (int L = 0; L < nLeaves; ++L) V[L] = leafW[L];
      return eval_tape_poly(V, op, a, b, cval, base, *pbW, p);
    }
    std::vector<std::vector<u64> > S(base + op.size(), std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L) std::copy(leafW[L].begin(), leafW[L].begin() + w, S[L].begin());
    if (!eval_ic_order0(S, op, a, b, cval, base, w, p)) return false;
    V.assign(S.size(), std::vector<u64>(blkW, 0));
    for (size_t i = 0; i < S.size(); ++i) std::copy(S[i].begin(), S[i].end(), V[i].begin());
    for (int L = 0; L < nLeaves; ++L) V[L] = leafW[L];
    return true;
  };
  std::vector<std::vector<u64> > tau(nSeg, std::vector<u64>(w, 0));
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    if (!sg.hasTm) continue;
    std::vector<std::vector<u64> > tmVal(nLeaves + sg.tmOp.size(), std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L) std::copy(leafW[L].begin(), leafW[L].begin() + w, tmVal[L].begin());
    if (!eval_ic_order0(tmVal, sg.tmOp, sg.tmA, sg.tmB, sg.tmCval, nLeaves, w, p)) return REV_FORWARD;
    tau[sj] = tmVal[sg.tmOut[0]];
  }
  std::vector<u64> lineC(K, 1);
  for (int i = 0; i + 1 < nSeg; ++i)
    if (segs[i].hasTm && segs[i + 1].hasTm) lineC[i] = submod(tau[i + 1][0], tau[i][0], p);
  std::vector<std::vector<u64> > start(nStates, std::vector<u64>(blkW, 0));
  const SegRaw& s0 = segs[0];
  if (s0.hasIcSeed) {
    for (int i = 0; i < nStates; ++i)
      for (int c = 0; c < w; ++c) start[i][c] = s0.icSeed[i][c];
  } else if (s0.hasIcTape) {
    std::vector<std::vector<u64> > icV;
    if (!side(s0.icOp, s0.icA, s0.icB, s0.icCval, icV, nLeaves)) return REV_FORWARD;
    for (int i = 0; i < nStates; ++i) start[i] = icV[s0.icOut[i]];
  }
  std::vector<std::vector<std::vector<u64> > > evv(std::max(0, nSeg - 1));
  for (int b = 0; b + 1 < nSeg; ++b) {
    const SegRaw& nx = segs[b + 1];
    if (!nx.hasEv) continue;
    std::vector<std::vector<u64> > eV;
    if (!side(nx.evOp, nx.evA, nx.evB, nx.evCval, eV, nLeaves + (nx.evStates ? nStates : 0)))
      return REV_FORWARD;
    for (size_t e = 0; e < nx.evVarIdx.size(); ++e) {
      std::vector<u64> vv = eV[nx.evOut[e]];
      for (u64& x : vv) x %= p;
      evv[b].push_back(vv);
    }
  }

  // per segment: -(o*o) of each reciprocal, the values constant in time and the (slot, order)
  // cells an adjoint reaches
  struct RevSeg {
    std::vector<std::vector<u64> > nsq;
    std::vector<char> tconst;
    std::vector<int> mark;
    int nMarked;
  };
  std::vector<RevSeg> rsg(nSeg);
  size_t maxMarked = 1;
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    const std::vector<std::vector<u64> >& V = tr.val[sj];
    int N1 = tr.nrun[sj] + 1, nInstr = (int)sg.op.size(), S = (int)V.size();
    RevSeg& rs = rsg[sj];
    std::vector<char> isState(S, 0);
    for (int s : sg.stateSlots) isState[s] = 1;
    rs.tconst.assign(S, 0);
    for (int v = 0; v < instrBase; ++v) rs.tconst[v] = !isState[v];
    for (int i = 0; i < nInstr; ++i) {
      int o = sg.op[i];
      rs.tconst[instrBase + i] = o == OP_CONST ||
          ((o == OP_ADD || o == OP_MUL) && rs.tconst[sg.a[i]] && rs.tconst[sg.b[i]]) ||
          (o == OP_INV && rs.tconst[sg.a[i]]);
    }
    rs.nsq.assign(nInstr, {});
    for (int i = 0; i < nInstr; ++i) {
      if (sg.op[i] != OP_INV) continue;
      const std::vector<u64>& o = V[instrBase + i];
      std::vector<u64>& q = rs.nsq[i];
      q.assign((size_t)N1 * nM, 0);
      for (int k = 0; k < N1; ++k)
        for (int ii = 0; ii <= k; ++ii)
          for (int am = 0; am < nM; ++am) {
            u64 x = o[(size_t)ii * nM + am];
            if (!x) continue;
            for (int bm = 0; bm < nM; ++bm) {
              int sl = pb.sumSlot[am][bm];
              if (sl < 0) continue;
              u64& t = q[(size_t)k * nM + sl];
              t = addmod(t, mulmod(x, o[(size_t)(k - ii) * nM + bm], p), p);
            }
          }
      for (u64& t : q) t = submod(0, t, p);
    }
    auto nzv = [&](const std::vector<u64>& v, int k) {
      for (int m = 0; m < nM; ++m) if (v[(size_t)k * nM + m]) return true;
      return false;
    };
    auto live = [&](int s, int k) {
      if (s < instrBase && !isState[s]) return k == 0 && s < nLeaves && dualCol[s] >= 0;
      if (s < instrBase) return true;
      if (sg.op[s - instrBase] == OP_CONST) return false;
      return k == 0 || !rs.tconst[s];
    };
    rs.mark.assign((size_t)S * N1, -1);
    rs.nMarked = 0;
    auto mark = [&](int s, int k) {
      int& m = rs.mark[(size_t)s * N1 + k];
      if (m < 0 && live(s, k)) m = rs.nMarked++;
    };
    for (int g : sg.gOut)
      for (int k = 0; k <= Nt && k < N1; ++k) mark(g, k);
    if (sj + 1 < nSeg)
      for (int s : sg.stateSlots)
        for (int k = 0; k <= Mtot && k < N1; ++k) mark(s, k);
    for (int s : sg.stateSlots) mark(s, 0);
    for (int k = N1 - 1; k >= 0; --k) {
      if (k + 1 < N1)
        for (int i = 0; i < (int)sg.stateSlots.size(); ++i)
          if (rs.mark[(size_t)sg.stateSlots[i] * N1 + k + 1] >= 0) mark(sg.fOut[i], k);
      for (int i = nInstr - 1; i >= 0; --i) {
        if (rs.mark[(size_t)(instrBase + i) * N1 + k] < 0) continue;
        int a = sg.a[i], b = sg.b[i];
        switch (sg.op[i]) {
          case OP_ADD: mark(a, k); mark(b, k); break;
          case OP_MUL: {
            int j0, j1;
            conv_range(rs.tconst[a], rs.tconst[b], k, j0, j1);
            for (int ii = j0; ii <= j1; ++ii) {
              if (nzv(V[b], k - ii)) mark(a, ii);
              if (nzv(V[a], ii)) mark(b, k - ii);
            }
            break;
          }
          case OP_INV:
            for (int ii = 0; ii <= (rs.tconst[a] ? 0 : k); ++ii) {
              bool any = false;
              for (int m = 0; m < nM && !any; ++m) any = rs.nsq[i][(size_t)(k - ii) * nM + m] != 0;
              if (any) mark(a, ii);
            }
            break;
        }
      }
    }
    maxMarked = std::max(maxMarked, (size_t)rs.nMarked);
  }
  // shift of time order k at monomial a into the gap axis of boundary b
  std::vector<std::vector<std::vector<int> > > shiftTab(std::max(0, nSeg - 1),
      std::vector<std::vector<int> >(Mtot + 1, std::vector<int>(nM, -1)));
  for (int b = 0; b + 1 < nSeg; ++b)
    for (int k = 0; k <= Mtot; ++k)
      for (int am = 0; am < nM; ++am) shiftTab[b][k][am] = pb.shiftSlot(am, b, k);

  std::vector<int> nOut(nSeg), offset(nSeg + 1, 0);
  for (int sj = nSeg - 1; sj >= 0; --sj) {
    nOut[sj] = (int)segs[sj].gOut.size();
    offset[sj] = offset[sj + 1] + nOut[sj] * (Nt + 1) * nM;
  }
  // the rows of segment sj start after those of the later segments
  std::vector<int> first(nSeg);
  for (int sj = 0; sj < nSeg; ++sj) first[sj] = offset[sj + 1];
  int Rtot = offset[0];

  auto sweep = [&](int r0, int r1, std::vector<std::vector<u64> >& out) {
    int B = r1 - r0;
    size_t cell = (size_t)nM * B;
    const Barrett br(p);
    out.assign(B, std::vector<u64>(nz, 0));
    std::vector<u64> coef(B);
    // out += coef * the duals of `dual` (w residues: value, then duals)
    auto addDual = [&](const u64* cf, const u64* dual) {
      for (int c = 1; c < w; ++c) {
        u64 d = dual[c] % p;
        if (!d) continue;
        for (int rr = 0; rr < B; ++rr)
          if (cf[rr]) out[rr][c - 1] = br.red(out[rr][c - 1] + cf[rr] * d);
      }
    };
    std::vector<std::vector<u64> > adjC;
    for (int sj = nSeg - 1; sj >= 0; --sj) {
      const SegRaw& sg = segs[sj];
      const RevSeg& rs = rsg[sj];
      const std::vector<std::vector<u64> >& V = tr.val[sj];
      int Nrun = tr.nrun[sj], N1 = Nrun + 1, nO = nOut[sj], nInstr = (int)sg.op.size();
      std::vector<u64> pool((size_t)rs.nMarked * cell, 0);
      auto at = [&](int s, int k) -> u64* {
        int m = rs.mark[(size_t)s * N1 + k];
        return m < 0 ? nullptr : pool.data() + (size_t)m * cell;
      };
      if (sj + 1 < nSeg) {
        bool mv = moves[sj] || moves[sj + 1];
        std::vector<u64> dd(w, 0);
        if (mv) for (int c = 1; c < w; ++c) dd[c] = submod(tau[sj + 1][c], tau[sj][c], p);
        std::fill(coef.begin(), coef.end(), (u64)0);
        for (int i = 0; i < nStates; ++i) {
          int s = sg.stateSlots[i];
          const std::vector<u64>& xv = V[s];
          for (int k = 0; k <= Mtot; ++k) {
            u64* d = at(s, k);
            for (int am = 0; am < nM; ++am) {
              int dst = shiftTab[sj][k][am];
              if (dst < 0) continue;
              const u64* a = adjC[i].data() + (size_t)dst * B;
              if (d) axpy_mod(d + (size_t)am * B, a, 1, B, br);
              if (!mv) continue;
              u64 fk = mulmod((u64)(k + 1) % p, xv[(size_t)(k + 1) * nM + am], p);
              if (fk) for (int rr = 0; rr < B; ++rr) coef[rr] = br.red(coef[rr] + a[rr] * fk);
            }
          }
        }
        if (mv) addDual(coef.data(), dd.data());
      }
      for (int k = Nrun; k >= 0; --k) {
        int act = std::min(B, first[sj] + std::max(0, Nt - k + 1) * nO * nM - r0);
        if (act <= 0) continue;
        if (k <= Nt)
          for (int gi = 0; gi < nO; ++gi)
            for (int mo = 0; mo < nM; ++mo) {
              int r = first[sj] + ((Nt - k) * nO + gi) * nM + mo;
              if (r < r0 || r >= r1) continue;
              if (u64* g = at(sg.gOut[gi], k)) {
                u64& e = g[(size_t)mo * B + (r - r0)];
                e = addmod(e, 1, p);
              }
            }
        if (k < Nrun) {
          u64 ik = invmod((u64)(k + 1), p);
          for (int i = 0; i < (int)sg.stateSlots.size(); ++i) {
            const u64* src = at(sg.stateSlots[i], k + 1);
            u64* dst = at(sg.fOut[i], k);
            if (src && dst)
              for (int m = 0; m < nM; ++m)
                axpy_mod(dst + (size_t)m * B, src + (size_t)m * B, ik, act, br);
          }
        }
        for (int i = nInstr - 1; i >= 0; --i) {
          const u64* o = at(instrBase + i, k);
          if (!o) continue;
          int a = sg.a[i], b = sg.b[i];
          switch (sg.op[i]) {
            case OP_ADD:
              for (int x : {a, b})
                if (u64* d = at(x, k))
                  for (int m = 0; m < nM; ++m)
                    axpy_mod(d + (size_t)m * B, o + (size_t)m * B, 1, act, br);
              break;
            case OP_MUL: {
              int j0, j1;
              conv_range(rs.tconst[a], rs.tconst[b], k, j0, j1);
              for (int ii = j0; ii <= j1; ++ii) {
                const u64* A = V[a].data() + (size_t)ii * nM;
                const u64* Bv = V[b].data() + (size_t)(k - ii) * nM;
                u64* da = at(a, ii);
                u64* db = at(b, k - ii);
                if (!da && !db) continue;
                for (int am = 0; am < nM; ++am)
                  for (int bm = 0; bm < nM; ++bm) {
                    int sl = pb.sumSlot[am][bm];
                    if (sl < 0) continue;
                    const u64* os = o + (size_t)sl * B;
                    if (da && Bv[bm]) axpy_mod(da + (size_t)am * B, os, Bv[bm], act, br);
                    if (db && A[am]) axpy_mod(db + (size_t)bm * B, os, A[am], act, br);
                  }
              }
              break;
            }
            case OP_INV:
              for (int ii = 0; ii <= (rs.tconst[a] ? 0 : k); ++ii) {
                u64* da = at(a, ii);
                if (!da) continue;
                const u64* ns = rs.nsq[i].data() + (size_t)(k - ii) * nM;
                for (int am = 0; am < nM; ++am)
                  for (int bm = 0; bm < nM; ++bm) {
                    int sl = pb.sumSlot[am][bm];
                    if (sl < 0 || !ns[bm]) continue;
                    axpy_mod(da + (size_t)am * B, o + (size_t)sl * B, ns[bm], act, br);
                  }
              }
              break;
          }
        }
      }
      for (int L = 0; L < nLeaves; ++L) {
        if (dualCol[L] < 0) continue;
        const u64* d = at(L, 0);
        if (!d) continue;
        for (int rr = 0; rr < B; ++rr)
          if (d[rr]) out[rr][dualCol[L]] = addmod(out[rr][dualCol[L]], d[rr], p);
      }
      std::vector<std::vector<u64> > adjS(nStates, std::vector<u64>(cell, 0));
      for (int i = 0; i < nStates; ++i)
        if (const u64* d = at(sg.stateSlots[i], 0)) std::copy(d, d + cell, adjS[i].begin());
      if (sj == 0) {
        for (int i = 0; i < nStates; ++i)
          for (int m = 0; m < nM; ++m)
            addDual(adjS[i].data() + (size_t)m * B, start[i].data() + (size_t)m * w);
        break;
      }
      // the events into this segment, reversed; a scaling reads the start before it
      int bnd = sj - 1;
      if (sg.hasEv) {
        int E = (int)sg.evVarIdx.size();
        std::vector<std::vector<u64> > cur = tr.preEvents[bnd];
        std::vector<std::vector<u64> > pre(E);
        for (int e = 0; e < E; ++e) {
          int s = sg.evVarIdx[e], meth = sg.evMethod[e];
          const std::vector<u64>& vv = evv[bnd][e];
          if (meth == 0) {
            for (int m = 0; m < nM; ++m) cur[s][m] = vv[(size_t)m * w];
          } else if (meth == 1) {
            for (int m = 0; m < nM; ++m) cur[s][m] = addmod(cur[s][m], vv[(size_t)m * w], p);
          } else {
            pre[e] = cur[s];
            std::vector<u64> sc(nM, 0);
            for (int am = 0; am < nM; ++am)
              for (int bm = 0; bm < nM; ++bm) {
                int sl = pb.sumSlot[am][bm];
                if (sl >= 0) sc[sl] = addmod(sc[sl], mulmod(pre[e][am], vv[(size_t)bm * w], p), p);
              }
            cur[s] = sc;
          }
        }
        for (int e = E - 1; e >= 0; --e) {
          int s = sg.evVarIdx[e], meth = sg.evMethod[e];
          const std::vector<u64>& vv = evv[bnd][e];
          std::vector<u64>& aS = adjS[s];
          if (meth != 2) {
            for (int m = 0; m < nM; ++m) addDual(aS.data() + (size_t)m * B, vv.data() + (size_t)m * w);
            if (meth == 0) std::fill(aS.begin(), aS.end(), (u64)0);
            continue;
          }
          std::vector<u64> nA(cell, 0);
          for (int am = 0; am < nM; ++am)
            for (int bm = 0; bm < nM; ++bm) {
              int sl = pb.sumSlot[am][bm];
              if (sl < 0) continue;
              const u64* as = aS.data() + (size_t)sl * B;
              u64 vb = vv[(size_t)bm * w];
              if (vb) axpy_mod(nA.data() + (size_t)am * B, as, vb, B, br);
              u64 pa = pre[e][am];
              if (!pa) continue;
              for (int rr = 0; rr < B; ++rr) coef[rr] = mulmod(as[rr], pa, p);
              addDual(coef.data(), vv.data() + (size_t)bm * w);
            }
          aS.swap(nA);
        }
      }
      adjC.swap(adjS);
    }
  };

  if (Rtot == 0) return REV_OK;
  // rows per block: split over the threads, and bounded so that one block's adjoints stay within
  // the budget of chain_block_bytes()
  double perRow = std::max(1.0, (double)maxMarked * nM * 8.0);
  int Bmem = std::max(1, (int)(chain_block_bytes() / perRow));
  int Bt = threads > 1 ? (Rtot + threads - 1) / threads : Rtot;
  int B = std::max(1, std::min(Bt, Bmem));
  int nb = (Rtot + B - 1) / B;
  std::vector<std::vector<std::vector<u64> > > rB(nb);
  int nThr = std::max(1, std::min(threads, nb));
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
  for (int b = 0; b < nb; ++b) {
    try {
      sweep(b * B, std::min(Rtot, (b + 1) * B), rB[b]);
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();

  // the rows in the order of the forward mode, with the shift of a moving segment start
  std::vector<std::vector<std::vector<u64> > > segRows(nSeg);
  for (int sj = 0; sj < nSeg; ++sj) segRows[sj].resize((size_t)nOut[sj] * (Nt + 1) * nM);
  for (int b = 0; b < nb; ++b)
    for (size_t i = 0; i < rB[b].size(); ++i) {
      int r = b * B + (int)i, sj = 0;
      while (r < first[sj]) ++sj;
      int loc = r - first[sj], nO = nOut[sj];
      int k = Nt - loc / (nO * nM), gi = (loc % (nO * nM)) / nM, mo = loc % nM;
      segRows[sj][((size_t)gi * (Nt + 1) + k) * nM + mo] = std::move(rB[b][i]);
    }
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    if (moves[sj])
      for (int gi = 0; gi < nOut[sj]; ++gi) {
        const std::vector<u64>& g = tr.val[sj][sg.gOut[gi]];
        for (int k = 0; k <= Nt; ++k)
          for (int mo = 0; mo < nM; ++mo) {
            u64 fk = mulmod((u64)(k + 1) % p, g[(size_t)(k + 1) * nM + mo], p);
            if (!fk) continue;
            std::vector<u64>& row = segRows[sj][((size_t)gi * (Nt + 1) + k) * nM + mo];
            for (int c = 0; c < nz; ++c)
              row[c] = submod(row[c], mulmod(fk, tau[sj][1 + c], p), p);
          }
      }
    size_t from = rows.size();
    for (std::vector<u64>& r : segRows[sj]) rows.push_back(std::move(r));
    if (lineMode) rows_to_series_axis(rows, from, pb, nz, Ns, srows, keepZeroSer);
    else rows_to_line(rows, from, pb, lineC, nz, Mtot + 1, p, srows, keepZeroSer);
    if (tr.kink[sj]) {
      std::vector<u64> row(tau[sj].begin() + 1, tau[sj].end());
      std::vector<u64> srow((size_t)nz * Ns, 0);
      for (int c = 0; c < nz; ++c) srow[(size_t)c * Ns] = row[c];
      rows.push_back(row);
      srows.push_back(srow);
    }
  }
  return REV_OK;
}

// Reverse mode where the rows of a chain sweep fewer residues than the columns hold, as in
// reverse_pays
bool reverse_pays_chain(const std::vector<SegRaw>& segs, int nz, int Nt, int Mtot, bool hasSer,
                        int Tser) {
  bool line = hasSer && Tser >= 0;
  int K = std::max(1, (int)segs.size() - 1) + (hasSer && !line ? 1 : 0);
  double nMono = 1;
  for (int i = 1; i <= Mtot; ++i) nMono = nMono * (K + i) / i;
  if (line) nMono *= Tser + 1;
  double rowsTot = 0;
  for (const SegRaw& sg : segs) rowsTot += (double)sg.gOut.size() * (Nt + 2) * nMono;
  return rowsTot < (double)(nz + 1);
}

// Columns per block so that one block's Taylor values stay within `budget` bytes.
int chain_block_size(const std::vector<SegRaw>& segs, int nLeaves, int nStates, int nz,
                     int Nt, int Mtot, bool hasSer, double budget, int Tser = -1) {
  size_t Sn = 0;
  for (const SegRaw& sg : segs) Sn = std::max(Sn, (size_t)(nLeaves + nStates) + sg.op.size());
  bool line = hasSer && Tser >= 0;
  int K = std::max(1, (int)segs.size() - 1) + (hasSer && !line ? 1 : 0);
  double nMono = 1;                                    // binomial(K + Mtot, Mtot)
  for (int i = 1; i <= Mtot; ++i) nMono = nMono * (K + i) / i;
  if (line) nMono *= Tser + 1;
  double perLane = (double)Sn * (std::max(Nt, Mtot) + 2) * nMono * 8.0;
  int B = (int)(budget / perLane) - 1;
  return std::max(2, std::min(nz, B));
}

// the default of chain_block_bytes(), set by symSetBlockBytes
std::atomic<long long> g_block_bytes(256LL * 1048576LL);

// Taylor values per column block, bytes; SYMIDENT_BLOCKMB overrides (MB)
double chain_block_bytes() {
  const char* e = std::getenv("SYMIDENT_BLOCKMB");
  return (e && *e) ? std::atof(e) * 1048576.0 : (double)g_block_bytes.load();
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

// build_one_chain over column blocks: each block holds the value and a separate range of dual
// lanes, the rows are assembled over all columns. Rows and series rows keep the order
// of the unblocked call, zero series rows dropped.
int lean_chain(const std::vector<SegRaw>& segs, int nLeaves, int nStates, int nz,
               const std::vector<int>& dualCol, const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
               std::vector<std::vector<u64> >& rows, std::vector<std::vector<u64> >& srows,
               int threads, bool keepZeroSer, const std::string& key);

bool build_chain_forward(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                         int nz, const std::vector<int>& dualCol,
                         const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                         std::vector<std::vector<u64> >& rows,
                         std::vector<std::vector<u64> >& srows,
                         const std::vector<std::vector<u64> >* ser, int B, int threads,
                         int Tser, bool keepZeroSer, const std::string& key = std::string()) {
  if (ser == nullptr) {
    int st = lean_chain(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, rows, srows, threads,
                        keepZeroSer, key);
    if (st != REV_FORWARD) return st == REV_OK;
  }
  if (B >= nz)
    return build_one_chain(segs, nLeaves, nStates, nz, nz + 1, dualCol, leafPt, Nt, Mtot,
                           p, rows, srows, ser, nullptr, nullptr, Tser, keepZeroSer);
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, nz + 1, dualCol, leafPt, p, moves)) return false;
  int nb = (nz + B - 1) / B, N = (ser && Tser >= 0) ? Tser + 1 : Mtot + 1;
  std::vector<std::vector<std::vector<u64> > > rB(nb), sB(nb);
  std::vector<char> ok(nb, 1);
  OmpGuard guard_;
  #pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(dynamic) \
          if (threads > 1)
  for (int b = 0; b < nb; ++b) {
    try {
      int s0 = b * B, s1 = std::min(nz, s0 + B), nzB = s1 - s0;
      std::vector<int> dc(nLeaves, -1), lane(nzB);
      for (int L = 0; L < nLeaves; ++L)
        if (dualCol[L] >= s0 && dualCol[L] < s1) dc[L] = dualCol[L] - s0;
      for (int j = 0; j < nzB; ++j) lane[j] = s0 + j;
      ok[b] = build_one_chain(segs, nLeaves, nStates, nzB, nzB + 1, dc, leafPt, Nt, Mtot, p,
                              rB[b], sB[b], ser, &lane, &moves, Tser);
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
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
    if (any || keepZeroSer) srows.push_back(sr);
  }
  return true;
}

// The rows of build_one_chain in the mode of kernel_mode(); B columns per block of the forward mode.
bool build_chain_blocked(const std::vector<SegRaw>& segs, int nLeaves, int nStates,
                         int nz, const std::vector<int>& dualCol,
                         const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
                         std::vector<std::vector<u64> >& rows,
                         std::vector<std::vector<u64> >& srows,
                         const std::vector<std::vector<u64> >* ser, int B, int threads,
                         int Tser = -1, bool keepZeroSer = false,
                         const std::string& key = std::string()) {
  int mode = kernel_mode();
  if (mode == KM_FORWARD ||
      (mode == KM_AUTO && !reverse_pays_chain(segs, nz, Nt, Mtot, ser != nullptr, Tser)))
    return build_chain_forward(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, rows,
                               srows, ser, B, threads, Tser, keepZeroSer, key);
  std::vector<std::vector<u64> > rv, sv;
  int st = build_one_chain_reverse(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, rv,
                                   sv, ser, threads, Tser, keepZeroSer);
  if (st == REV_FORWARD)
    return build_chain_forward(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, rows,
                               srows, ser, B, threads, Tser, keepZeroSer, key);
  if (mode == KM_CHECK) {
    std::vector<std::vector<u64> > fr, fs;
    bool okF = build_chain_forward(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, fr,
                                   fs, ser, B, threads, Tser, keepZeroSer);
    if (okF != (st == REV_OK))
      throw std::runtime_error("kernel: the reverse mode fails where the forward mode does not (chain)");
    require_same_rows(fr, rv, p, "chain rows");
    require_same_rows(fs, sv, p, "chain series rows");
  }
  if (st == REV_FAIL) return false;
  for (auto& r : rv) rows.push_back(std::move(r));
  for (auto& r : sv) srows.push_back(std::move(r));
  return true;
}

// ---- Jets of one segment continued to higher orders ----
// Taylor coefficients of a segment over a block of dual lanes, kept between calls so that a higher order
// continues from the last one. Values a convolution reads keep every order, the others the current order only.
struct LeanJets {
  int s0 = 0, wb = 1, K = -1;              // lanes [s0, s0 + wb - 1), last order formed
  int nM = 1;                              // monomials in the gap lengths per value
  std::shared_ptr<PolyBasis> pb;           // the monomials, with wb lanes
  std::vector<std::vector<u64> > H;        // per slot: the current order, or order 0
  std::vector<std::vector<std::vector<u64> > > C;  // per slot that keeps every order: chunks of orders
  std::vector<std::vector<u64> > invA0;    // per reciprocal: 1/a at order 0
  std::vector<std::vector<int> > rlo, rhi; // per kept slot, order and monomial: the nonzero lanes
  std::vector<std::vector<u64> > rows;     // per order, output and monomial: the lanes
  std::vector<u64> gVal;                   // per order, output and monomial: the value
  std::vector<u64> fVal;                   // the field at order 0, monomial 0
  int keepEarly = -1;                      // the state orders kept for the next segment
  std::vector<std::vector<std::vector<u64> > > early;  // per state and order up to keepEarly
  size_t bytes() const {
    size_t b = 0;
    for (const auto& h : H) b += h.capacity() * 8;
    for (const auto& c : C)
      for (const auto& x : c) b += x.capacity() * 8;
    for (const auto& r : rows) b += r.capacity() * 8;
    return b;
  }
};

struct LeanTape {
  std::vector<char> hist, tconst;
  int instrBase = 0;
  bool ok = true;
};

// The kept orders of a segment tape; ok = false for an operation other than CONST/ADD/MUL/INV.
LeanTape lean_tape(const SegRaw& seg, int nLeaves, int nStates) {
  LeanTape lt;
  int ib = nLeaves + nStates, nI = (int)seg.op.size(), S = ib + nI;
  lt.instrBase = ib;
  lt.hist.assign(S, 0);
  lt.tconst.assign(S, 0);
  std::vector<char> isState(S, 0);
  for (int s : seg.stateSlots) isState[s] = 1;
  for (int v = 0; v < ib; ++v) lt.tconst[v] = !isState[v];
  for (int i = 0; i < nI; ++i) {
    int o = seg.op[i], a = seg.a[i], b = seg.b[i];
    if (o != OP_CONST && o != OP_ADD && o != OP_MUL && o != OP_INV) { lt.ok = false; return lt; }
    lt.tconst[ib + i] = o == OP_CONST || ((o == OP_ADD || o == OP_MUL) && lt.tconst[a] && lt.tconst[b]) ||
                        (o == OP_INV && lt.tconst[a]);
    if (o == OP_MUL && !lt.tconst[a] && !lt.tconst[b]) lt.hist[a] = lt.hist[b] = 1;
    if (o == OP_INV && !lt.tconst[a]) lt.hist[a] = lt.hist[ib + i] = 1;
  }
  return lt;
}

// A block of lanes [s0, s1) at order 0: the leaves at their values with a unit lane per coordinate in
// the block, state i at st[i] (per monomial the value, then the lanes over all nLane columns).
LeanJets lean_block(const LeanTape& lt, const SegRaw& seg, int nLeaves, const std::vector<u64>& leaf,
                    const std::vector<int>& dualCol, const std::vector<std::vector<u64> >& st, int s0, int s1,
                    int nLane, int K = 1, int Mtot = 0) {
  LeanJets J;
  int S = (int)lt.hist.size();
  J.s0 = s0; J.wb = s1 - s0 + 1;
  J.pb = std::make_shared<PolyBasis>(K, Mtot, J.wb);
  J.nM = J.pb->nMono;
  size_t blk = (size_t)J.nM * J.wb;
  J.H.assign(S, std::vector<u64>());
  J.rlo.assign(S, std::vector<int>());
  J.rhi.assign(S, std::vector<int>());
  J.invA0.assign(seg.op.size(), std::vector<u64>());
  for (int v = 0; v < S; ++v) J.H[v].assign(blk, 0);
  for (int L = 0; L < nLeaves; ++L) {
    J.H[L][0] = leaf[L];
    if (dualCol[L] >= s0 && dualCol[L] < s1) J.H[L][1 + dualCol[L] - s0] = 1;
  }
  for (size_t i = 0; i < seg.stateSlots.size(); ++i) {
    std::vector<u64>& h = J.H[seg.stateSlots[i]];
    for (int mo = 0; mo < J.nM; ++mo) {
      const u64* src = st[i].data() + (size_t)mo * (1 + nLane);
      u64* d = h.data() + (size_t)mo * J.wb;
      d[0] = src[0];
      for (int c = s0; c < s1; ++c) d[1 + c - s0] = src[1 + c];
    }
  }
  return J;
}

// The orders K + 1 .. Nrun of the jets `J`; false where a reciprocal has no unit value at order 0.
bool lean_extend(LeanJets& J, const LeanTape& lt, const SegRaw& seg, int Nrun, u64 p) {
  int ib = lt.instrBase, nI = (int)seg.op.size(), wb = J.wb, m = (int)seg.stateSlots.size();
  int nOut = (int)seg.gOut.size(), nM = J.nM;
  size_t blk = (size_t)nM * wb;
  const PolyBasis& pb = *J.pb;
  const Barrett br(p);
  const u64 LIM = (((u64)1 << 63) / p) * p;
  std::vector<u64> zero(blk, 0), acc(blk), tmp(blk);
  // orders in chunks, so that a higher order adds chunks and copies none
  const int CH = 32;
  if (J.C.empty()) J.C.resize(J.H.size());
  for (size_t s = 0; s < J.H.size(); ++s)
    if (lt.hist[s]) {
      if (J.C[s].empty()) J.C[s].push_back(J.H[s]);
      while ((int)J.C[s].size() * CH < Nrun + 2) J.C[s].push_back(std::vector<u64>());
      for (auto& c : J.C[s])
        if (c.size() != (size_t)CH * blk) c.resize((size_t)CH * blk, 0);
      J.rlo[s].resize((size_t)(Nrun + 2) * nM, -1);
      J.rhi[s].resize((size_t)(Nrun + 2) * nM, 0);
    }
  if (J.keepEarly >= 0 && J.early.empty()) {
    J.early.assign(m, std::vector<std::vector<u64> >());
    for (int i = 0; i < m; ++i) J.early[i].push_back(J.H[seg.stateSlots[i]]);
  }
  auto hp = [&](int s, int k) -> u64* { return J.C[s][k / CH].data() + (size_t)(k % CH) * blk; };
  auto rd = [&](int s, int k) -> const u64* {
    if (lt.tconst[s]) return k == 0 ? J.H[s].data() : zero.data();
    return lt.hist[s] ? hp(s, k) : J.H[s].data();
  };
  auto wr = [&](int s, int k) -> u64* {
    return lt.hist[s] ? hp(s, k) : J.H[s].data();
  };
  auto dense = [&](const u64* x, int& lo, int& hi) {
    lo = 1; while (lo < wb && x[lo] == 0) ++lo;
    hi = wb; while (hi > lo && x[hi - 1] == 0) --hi;
  };
  auto range = [&](int s, int k, int mo, int& lo, int& hi) {
    size_t key = (size_t)k * nM + mo;
    if (J.rlo[s][key] < 0) {
      int l, h;
      dense(hp(s, k) + (size_t)mo * wb, l, h);
      J.rlo[s][key] = l; J.rhi[s][key] = h;
    }
    lo = J.rlo[s][key]; hi = J.rhi[s][key];
  };
  // acc[sl] += x * y over the duals, unreduced below LIM
  auto dmul1 = [&](u64* ac, const u64* x, const u64* y, int lx, int hx, int ly, int hy) {
    u64 x0 = x[0], y0 = y[0];
    if (x0 && y0) { u64 t = ac[0] + x0 * y0; ac[0] = t >= LIM ? t - LIM : t; }
    if (x0) for (int c = ly; c < hy; ++c) { u64 t = ac[c] + x0 * y[c]; ac[c] = t >= LIM ? t - LIM : t; }
    if (y0) for (int c = lx; c < hx; ++c) { u64 t = ac[c] + y0 * x[c]; ac[c] = t >= LIM ? t - LIM : t; }
  };
  // acc += X * Y over the monomials; `xr`, `yr` give the lanes of a monomial
  auto pmul = [&](const u64* X, const u64* Y, auto xr, auto yr) {
    for (int am = 0; am < nM; ++am) {
      const u64* x = X + (size_t)am * wb;
      int lx, hx;
      xr(am, lx, hx);
      if (x[0] == 0 && lx == hx) continue;
      for (int bm = 0; bm < nM; ++bm) {
        int sl = pb.sumSlot[am][bm];
        if (sl < 0) continue;
        const u64* y = Y + (size_t)bm * wb;
        int ly, hy;
        yr(bm, ly, hy);
        if (y[0] == 0 && ly == hy) continue;
        dmul1(acc.data() + (size_t)sl * wb, x, y, lx, hx, ly, hy);
      }
    }
  };
  auto dr = [&](const u64* X) { return [X, &dense, wb](int mo, int& lo, int& hi) { dense(X + (size_t)mo * wb, lo, hi); }; };
  for (int k = J.K + 1; k <= Nrun; ++k) {
    if (k > 0) {
      // the states at order k, from the field at order k - 1
      u64 ik = invmod((u64)k, p);
      std::vector<std::vector<u64> > nx(m, std::vector<u64>(blk));
      for (int i = 0; i < m; ++i) {
        const u64* f = rd(seg.fOut[i], k - 1);
        for (size_t c = 0; c < blk; ++c) nx[i][c] = mulmod(f[c], ik, p);
      }
      for (int i = 0; i < m; ++i) {
        std::copy(nx[i].begin(), nx[i].end(), wr(seg.stateSlots[i], k));
        if (k <= J.keepEarly) J.early[i].push_back(nx[i]);
      }
    }
    for (int i = 0; i < nI; ++i) {
      int s = ib + i, a = seg.a[i], b = seg.b[i];
      if (k > 0 && lt.tconst[s]) continue;
      u64* o = wr(s, k);
      std::fill(acc.begin(), acc.end(), (u64)0);
      switch (seg.op[i]) {
        case OP_CONST:
          std::fill(o, o + blk, (u64)0);
          o[0] = seg.cval[i];
          continue;
        case OP_ADD: {
          const u64* x = rd(a, k); const u64* y = rd(b, k);
          for (size_t c = 0; c < blk; ++c) o[c] = addmod(x[c], y[c], p);
          continue;
        }
        case OP_MUL:
          if (lt.tconst[a] || lt.tconst[b]) {
            const u64* x = rd(a, lt.tconst[a] ? 0 : k); const u64* y = rd(b, lt.tconst[b] ? 0 : k);
            pmul(x, y, dr(x), dr(y));
          } else {
            for (int ii = 0; ii <= k; ++ii)
              pmul(hp(a, ii), hp(b, k - ii),
                   [&](int mo, int& lo, int& hi) { range(a, ii, mo, lo, hi); },
                   [&](int mo, int& lo, int& hi) { range(b, k - ii, mo, lo, hi); });
          }
          break;
        case OP_INV: {
          if (k == 0) {
            J.invA0[i].assign(blk, 0);
            if (!poly_inv(J.invA0[i].data(), rd(a, 0), pb, p)) return false;
            std::copy(J.invA0[i].begin(), J.invA0[i].end(), o);
            continue;
          }
          for (int j = 1; j <= k; ++j)
            pmul(hp(a, j), hp(s, k - j),
                 [&](int mo, int& lo, int& hi) { range(a, j, mo, lo, hi); },
                 [&](int mo, int& lo, int& hi) { range(s, k - j, mo, lo, hi); });
          for (size_t c = 0; c < blk; ++c) tmp[c] = submod(0, br.red(acc[c]), p);
          std::fill(acc.begin(), acc.end(), (u64)0);
          pmul(J.invA0[i].data(), tmp.data(), dr(J.invA0[i].data()), dr(tmp.data()));
          break;
        }
      }
      for (size_t c = 0; c < blk; ++c) o[c] = br.red(acc[c]);
    }
    if (k == 0) {
      J.fVal.assign(m, 0);
      for (int i = 0; i < m; ++i) J.fVal[i] = rd(seg.fOut[i], 0)[0] % p;
    }
    for (int gi = 0; gi < nOut; ++gi) {
      const u64* g = rd(seg.gOut[gi], k);
      for (int mo = 0; mo < nM; ++mo) {
        const u64* gm = g + (size_t)mo * wb;
        J.rows.push_back(std::vector<u64>(gm + 1, gm + wb));
        J.gVal.push_back(gm[0] % p);
      }
    }
    J.K = k;
  }
  return true;
}

// Returns the freed heap of every thread to the system: the jets of many threads leave the arenas
// fragmented, which counts against a limit on the address space. Called after the calls at high
// orders only, since the walk over the arenas holds every thread.
void release_heap() {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}

// Taylor values of a slot over many orders are mapped and unmapped on their own rather than taken from
// the arenas of the threads, where the freed ones stay.
struct MapLargeValues {
  MapLargeValues() {
#if defined(__GLIBC__)
    mallopt(M_MMAP_THRESHOLD, 256 * 1024);
#endif
  }
} g_map_large_values;

// Sessions of continued jets: one per key, the oldest dropped past a memory budget.
struct SegSession {
  std::mutex mu;
  int nz = 0, nLane = 0, B = 0, nb = 0, nOut = 0;
  std::vector<LeanJets> blocks;
  std::vector<std::vector<LeanJets> > segBlocks;   // a chain: the blocks of each segment
  std::vector<std::vector<u64> > basis;
  std::vector<int> pivCol, ranks;
  size_t bytes = 0;
  long long used = 0;
};
std::mutex g_sess_mu;
std::map<std::string, std::shared_ptr<SegSession> > g_sess;
long long g_sess_clock = 0;

double session_budget() {
  const char* e = std::getenv("SYMIDENT_JETMB");
  return (e && *e) ? std::atof(e) * 1048576.0 : 8.0 * chain_block_bytes();
}

std::shared_ptr<SegSession> session_get(const std::string& key, bool& fresh) {
  std::lock_guard<std::mutex> lk(g_sess_mu);
  auto it = g_sess.find(key);
  fresh = it == g_sess.end();
  if (fresh) it = g_sess.emplace(key, std::make_shared<SegSession>()).first;
  it->second->used = ++g_sess_clock;
  return it->second;
}

// The bytes of the jets of `lt` over nLane lanes in nb blocks to order Nrun, with the rows.
double lean_bytes(const LeanTape& lt, int nLane, int nb, int Nrun, int nOut, int nM = 1, int early = 0,
                  int nStates = 0) {
  double h = 0, S = (double)lt.hist.size();
  for (char x : lt.hist) h += x;
  double orders = 32.0 * ((Nrun + 2 + 31) / 32);
  return 8.0 * nM * (nLane + nb) * (h * orders + S + (double)nOut * (Nrun + 1) + (double)nStates * early);
}

// jets being formed, in bytes, which the budget holds besides the sessions kept
double g_sess_inflight = 0;

// Reserves `need` bytes of the budget, dropping the oldest sessions not in use; false where the
// bytes do not fit.
bool session_reserve(double need, const SegSession* self) {
  std::lock_guard<std::mutex> lk(g_sess_mu);
  double budget = session_budget();
  for (;;) {
    double tot = g_sess_inflight;
    for (auto& kv : g_sess) if (kv.second.get() != self) tot += (double)kv.second->bytes;
    if (tot + need <= budget) {
      g_sess_inflight += need;
      return true;
    }
    auto old = g_sess.end();
    for (auto it = g_sess.begin(); it != g_sess.end(); ++it) {
      if (it->second.get() == self || !it->second->mu.try_lock()) continue;
      it->second->mu.unlock();
      if (old == g_sess.end() || it->second->used < old->second->used) old = it;
    }
    if (old == g_sess.end()) return false;
    g_sess.erase(old);
  }
}

void session_release(double need) {
  std::lock_guard<std::mutex> lk(g_sess_mu);
  g_sess_inflight -= need;
}

void session_trim() {
  std::lock_guard<std::mutex> lk(g_sess_mu);
  double budget = session_budget();
  for (;;) {
    double tot = 0;
    for (auto& kv : g_sess) tot += (double)kv.second->bytes;
    if (tot <= budget || g_sess.empty()) return;
    auto old = g_sess.begin();
    for (auto it = g_sess.begin(); it != g_sess.end(); ++it)
      if (it->second->used < old->second->used) old = it;
    g_sess.erase(old);
  }
}

// The ranks of one segment from the states `q` at orders 0..Nt (symSegmentRanks with fromStates),
// continued from the jets kept under `key`; false where the session does not apply.
bool session_ranks(const std::string& key, const SegRaw& seg, int nLeaves, int nStates, int nz,
                   const std::vector<int>& dualCol, const std::vector<u64>& leaf,
                   const std::vector<u64>& qs, bool moves, int Nt, int threads,
                   const std::vector<int>& dropStates, u64 p, std::vector<int>& ranksOut,
                   std::vector<u64>& fValOut) {
  LeanTape lt = lean_tape(seg, nLeaves, nStates);
  if (!lt.ok) return false;
  bool fresh;
  std::shared_ptr<SegSession> ss = session_get(key, fresh);
  std::lock_guard<std::mutex> lk(ss->mu);
  int nzA = nz + nStates + 1, nOut = (int)seg.gOut.size();
  int Nrun = Nt + 1;
  if (ss->blocks.empty()) {
    ss->nz = nz; ss->nOut = nOut;
    ss->nLane = nz + nStates;
    ss->nb = std::max(1, std::min(threads, ss->nLane / 16));
    ss->B = (ss->nLane + ss->nb - 1) / ss->nb;
    ss->nb = (ss->nLane + ss->B - 1) / ss->B;
  }
  if (ss->nz != nz || ss->nOut != nOut) return false;
  // jets past the budget are formed by the blocked kernel instead
  double need = ss->blocks.empty() || ss->blocks[0].K < Nrun ?
      std::max(0.0, lean_bytes(lt, ss->nLane, ss->nb, Nrun, nOut) - (double)ss->bytes) : 0.0;
  if (!session_reserve(need, ss.get())) {
    std::lock_guard<std::mutex> lk2(g_sess_mu);
    g_sess.erase(key);
    return false;
  }
  struct Release { double n; ~Release() { session_release(n); } } rel{need};
  if (ss->blocks.empty()) {
    // each state a separate unit lane after the coordinates
    std::vector<std::vector<u64> > st(nStates, std::vector<u64>(1 + ss->nLane, 0));
    for (int i = 0; i < nStates; ++i) {
      st[i][0] = qs[i];
      st[i][1 + nz + i] = 1;
    }
    for (int b = 0; b < ss->nb; ++b)
      ss->blocks.push_back(lean_block(lt, seg, nLeaves, leaf, dualCol, st, b * ss->B,
                                      std::min(ss->nLane, (b + 1) * ss->B), ss->nLane));
  }
  if (ss->blocks[0].K < Nrun) {
    std::vector<char> okB(ss->nb, 1);
    int nThr = std::max(1, std::min(threads, ss->nb));
    OmpGuard guard_;
    #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
    for (int b = 0; b < ss->nb; ++b) {
      try {
        okB[b] = lean_extend(ss->blocks[b], lt, seg, Nrun, p);
      } catch (const std::bad_alloc&) {
        okB[b] = 0;                      // the blocked kernel forms them instead
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
    for (char o : okB)
      if (!o) {
        std::lock_guard<std::mutex> lk2(g_sess_mu);
        g_sess.erase(key);
        return false;
      }
  }
  // rows of the new orders, ranked after those kept
  const LeanJets& J0 = ss->blocks[0];
  for (int k = (int)ss->ranks.size(); k <= Nt; ++k) {
    for (int gi = 0; gi < nOut; ++gi) {
      std::vector<u64> r(nzA, 0);
      for (int b = 0; b < ss->nb; ++b) {
        const std::vector<u64>& lane = ss->blocks[b].rows[(size_t)k * nOut + gi];
        std::copy(lane.begin(), lane.end(), r.begin() + ss->blocks[b].s0);
      }
      if (moves) r[nzA - 1] = submod(0, mulmod((u64)(k + 1) % p, J0.gVal[(size_t)(k + 1) * nOut + gi], p), p);
      for (int d : dropStates) r[nz + d] = 0;
      for (size_t b = 0; b < ss->basis.size(); ++b) {
        u64 f = r[ss->pivCol[b]] % p;
        if (!f) continue;
        u64 nf = p - f;
        for (int c = 0; c < nzA; ++c)
          if (ss->basis[b][c]) r[c] = addmod(r[c] % p, mulmod(nf, ss->basis[b][c], p), p);
      }
      int pc = -1;
      for (int c = 0; c < nzA; ++c) if (r[c] % p) { pc = c; break; }
      if (pc < 0) continue;
      u64 inv = invmod(r[pc] % p, p);
      for (int c = 0; c < nzA; ++c) r[c] = mulmod(r[c] % p, inv, p);
      ss->basis.push_back(std::move(r));
      ss->pivCol.push_back(pc);
    }
    ss->ranks.push_back((int)ss->basis.size());
  }
  ranksOut.assign(ss->ranks.begin(), ss->ranks.begin() + Nt + 1);
  fValOut = J0.fVal;
  size_t by = 0;
  for (const auto& J : ss->blocks) by += J.bytes();
  for (const auto& r : ss->basis) by += r.capacity() * 8;
  ss->bytes = by;
  session_trim();
  return true;
}


// build_one_chain in the lean jets without a series, a moving boundary or an event after the start, else
// REV_FORWARD: a segment starts at the orders up to Mtot of the segment before, shifted into the boundary gap.
// The jets are kept under `key`, where one is given, and continued by a call at a higher order.
int lean_chain(const std::vector<SegRaw>& segs, int nLeaves, int nStates, int nz,
               const std::vector<int>& dualCol, const std::vector<int>& leafPt, int Nt, int Mtot, u64 p,
               std::vector<std::vector<u64> >& rows, std::vector<std::vector<u64> >& srows,
               int threads, bool keepZeroSer, const std::string& key) {
  int nSeg = (int)segs.size(), w = nz + 1;
  if (nSeg == 0) return REV_FORWARD;
  std::vector<LeanTape> lts;
  for (int sj = 0; sj < nSeg; ++sj) {
    lts.push_back(lean_tape(segs[sj], nLeaves, nStates));
    if (!lts.back().ok || (sj > 0 && segs[sj].hasEv)) return REV_FORWARD;
  }
  const SegRaw& s0 = segs[0];
  if (!s0.hasIcSeed && !s0.hasIcTape) return REV_FORWARD;
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, w, dualCol, leafPt, p, moves)) return REV_FAIL;
  for (char m : moves) if (m) return REV_FORWARD;
  std::vector<u64> leaf(nLeaves);
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)leafPt[L] % (long long)p; if (pv < 0) pv += p;
    leaf[L] = (u64)pv;
  }
  // the gaps between the start times, which must not vanish
  std::vector<u64> tau(nSeg, 0);
  for (int sj = 0; sj < nSeg; ++sj) {
    const SegRaw& sg = segs[sj];
    if (!sg.hasTm) continue;
    std::vector<std::vector<u64> > tmVal(nLeaves + sg.tmOp.size(), std::vector<u64>(1, 0));
    for (int L = 0; L < nLeaves; ++L) tmVal[L][0] = leaf[L];
    if (!eval_ic_order0(tmVal, sg.tmOp, sg.tmA, sg.tmB, sg.tmCval, nLeaves, 1, p)) return REV_FAIL;
    tau[sj] = tmVal[sg.tmOut[0]][0] % p;
  }
  int K = std::max(1, nSeg - 1);
  std::vector<u64> lineC(K, 1);
  for (int i = 0; i + 1 < nSeg; ++i) {
    if (!segs[i].hasTm || !segs[i + 1].hasTm) continue;
    lineC[i] = submod(tau[i + 1], tau[i], p);
    if (!lineC[i]) return REV_FAIL;
  }
  // the start of the chain, a dual over the coordinates
  std::vector<std::vector<u64> > st(nStates, std::vector<u64>(w, 0));
  if (s0.hasIcSeed) {
    for (int i = 0; i < nStates; ++i)
      for (int c = 0; c < w; ++c) st[i][c] = s0.icSeed[i][c];
  } else {
    std::vector<std::vector<u64> > icVal(nLeaves + s0.icOp.size(), std::vector<u64>(w, 0));
    for (int L = 0; L < nLeaves; ++L) {
      icVal[L][0] = leaf[L];
      if (dualCol[L] >= 0) icVal[L][1 + dualCol[L]] = 1;
    }
    if (!eval_ic_order0(icVal, s0.icOp, s0.icA, s0.icB, s0.icCval, nLeaves, w, p)) return REV_FAIL;
    for (int i = 0; i < nStates; ++i) st[i] = icVal[s0.icOut[i]];
  }
  std::shared_ptr<SegSession> ss;
  if (!key.empty()) {
    bool fresh;
    ss = session_get(key, fresh);
  } else {
    ss = std::make_shared<SegSession>();
  }
  std::lock_guard<std::mutex> lk(ss->mu);
  int Nrun = std::max(Nt, Mtot);
  PolyBasis pbS(K, Mtot, 1);
  int nM = pbS.nMono;
  if (ss->segBlocks.empty()) {
    ss->nz = nz;
    ss->nb = std::max(1, std::min(threads, nz / 16));
    ss->B = std::max(1, (nz + ss->nb - 1) / ss->nb);
    ss->nb = std::max(1, (nz + ss->B - 1) / ss->B);
  }
  if (ss->nz != nz) return REV_FORWARD;
  // jets past the budget are formed by the blocked kernel instead
  double need = -(double)ss->bytes;
  for (int sj = 0; sj < nSeg; ++sj)
    need += lean_bytes(lts[sj], nz, ss->nb, Nrun, (int)segs[sj].gOut.size(), nM, Mtot + 1, nStates);
  need = std::max(0.0, need);
  if (!session_reserve(need, ss.get())) {
    if (!key.empty()) {
      std::lock_guard<std::mutex> lk2(g_sess_mu);
      g_sess.erase(key);
    }
    return REV_FORWARD;
  }
  struct Release { double n; ~Release() { session_release(n); } } rel{need};
  if (ss->segBlocks.empty()) {
    ss->segBlocks.resize(nSeg);
    // each segment from the start the segment before gives it, the orders up to Mtot first
    std::vector<std::vector<u64> > cur(nStates, std::vector<u64>((size_t)nM * w, 0));
    for (int i = 0; i < nStates; ++i) std::copy(st[i].begin(), st[i].end(), cur[i].begin());
    for (int sj = 0; sj < nSeg; ++sj) {
      for (int b = 0; b < ss->nb; ++b) {
        ss->segBlocks[sj].push_back(lean_block(lts[sj], segs[sj], nLeaves, leaf, dualCol, cur, b * ss->B,
                                               std::min(nz, (b + 1) * ss->B), nz, K, Mtot));
        ss->segBlocks[sj].back().keepEarly = sj + 1 < nSeg ? Mtot : -1;
      }
      if (sj + 1 == nSeg) break;
      std::vector<char> okE(ss->nb, 1);
      for (int b = 0; b < ss->nb; ++b) okE[b] = lean_extend(ss->segBlocks[sj][b], lts[sj], segs[sj], Mtot, p);
      for (char o : okE)
        if (!o) {
          ss->segBlocks.clear();
          if (!key.empty()) {
            std::lock_guard<std::mutex> lk2(g_sess_mu);
            g_sess.erase(key);
          }
          return REV_FORWARD;
        }
      for (auto& c : cur) std::fill(c.begin(), c.end(), (u64)0);
      for (int i = 0; i < nStates; ++i)
        for (int k = 0; k <= Mtot; ++k)
          for (int am = 0; am < nM; ++am) {
            int dst = pbS.shiftSlot(am, sj, k);
            if (dst < 0) continue;
            u64* d = cur[i].data() + (size_t)dst * w;
            for (const LeanJets& J : ss->segBlocks[sj]) {
              const u64* x = J.early[i][k].data() + (size_t)am * J.wb;
              if (J.s0 == 0) d[0] = addmod(d[0], x[0], p);
              for (int c = 1; c < J.wb; ++c) d[J.s0 + c] = addmod(d[J.s0 + c], x[c], p);
            }
          }
    }
  }
  if ((int)ss->segBlocks.size() != nSeg) return REV_FORWARD;
  int nb = ss->nb, nJobs = nSeg * nb;
  std::vector<char> okB(nJobs, 1);
  int nThr = std::max(1, std::min(threads, nJobs));
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1)
  for (int jb = 0; jb < nJobs; ++jb) {
    try {
      int sj = jb / nb, b = jb % nb;
      LeanJets& J = ss->segBlocks[sj][b];
      if (J.K < Nrun) okB[jb] = lean_extend(J, lts[sj], segs[sj], Nrun, p);
    } catch (const std::bad_alloc&) {
      okB[jb] = 0;                       // the blocked kernel forms them instead
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  for (char o : okB)
    if (!o) {
      if (!key.empty()) {
        std::lock_guard<std::mutex> lk2(g_sess_mu);
        g_sess.erase(key);
      }
      return REV_FORWARD;
    }
  for (int sj = 0; sj < nSeg; ++sj) {
    int nOut = (int)segs[sj].gOut.size();
    size_t from = rows.size();
    for (int gi = 0; gi < nOut; ++gi)
      for (int k = 0; k <= Nt; ++k)
        for (int mo = 0; mo < nM; ++mo) {
          std::vector<u64> r(nz, 0);
          for (const LeanJets& J : ss->segBlocks[sj]) {
            const std::vector<u64>& lane = J.rows[((size_t)k * nOut + gi) * nM + mo];
            std::copy(lane.begin(), lane.end(), r.begin() + J.s0);
          }
          rows.push_back(std::move(r));
        }
    rows_to_line(rows, from, pbS, lineC, nz, Mtot + 1, p, srows, keepZeroSer);
  }
  if (!key.empty()) {
    size_t by = 0;
    for (const auto& sb : ss->segBlocks)
      for (const auto& J : sb) by += J.bytes();
    ss->bytes = by;
    session_trim();
  }
  return REV_OK;
}

}  // namespace

// Multi-condition observability over a shared coordinate space. Each element of
// `tapes` is one condition on the same slot layout: leaves [0, nLeaves), states
// [nLeaves, nLeaves + nStates), then the instructions of the condition. Initial values are a
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

  // serial pre-pass into flat vectors: the parallel region reads no Value or List, whose named
  // access can insert and whose conversion copies
  TapeSet ts(tapes, nStates, w);

  // the conditions a chunk at a time, built in parallel; the rows of a chunk join the basis and are
  // dropped. A vanishing denominator in any condition marks failure (no early return inside OpenMP).
  int nThr = cores > 0 ? cores : 1;
  int chunk = std::max(64, 4 * nThr);
  RrefBasis basis(nz, p);
  for (int t0 = 0; t0 < T; t0 += chunk) {
    int t1 = std::min(T, t0 + chunk), nc = t1 - t0;
    std::vector<std::vector<std::vector<u64> > > perRows(nc);
    std::vector<char> okFlag(nc, 1);
    int inner = chain_inner_threads(cores, nc);
    OmpGuard guard_;
    #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1 && nc > 1)
    for (int t = t0; t < t1; ++t) {
      try {
        CondRaw tmp;
        if (!build_one_condition_blocked(ts.get(t, tmp), leafVal, nLeaves, nStates, nz, Nt, p,
                                         perRows[t - t0], inner))
          okFlag[t - t0] = 0;
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
    for (int t = 0; t < nc; ++t)
      if (!okFlag[t]) return List::create(_["ok"] = false);
    std::vector<std::vector<u64> > rows;
    for (int t = 0; t < nc; ++t)
      for (auto& r : perRows[t]) rows.push_back(std::move(r));
    basis.add(rows, nThr);
  }

  std::vector<std::vector<u64> > R0;
  std::vector<int> pivots;
  basis.full(R0, pivots);
  int rank = (int)pivots.size();
  IntegerMatrix R(rank, nz);
  for (int i = 0; i < rank; ++i)
    for (int c = 0; c < nz; ++c) R(i, c) = (int)R0[i][c];
  IntegerVector piv(rank);
  for (int i = 0; i < rank; ++i) piv[i] = pivots[i];

  return List::create(_["ok"] = true, _["R"] = R, _["pivots"] = piv,
                      _["rank"] = rank, _["dim"] = nz);
}

// Per column of V (nz x D residues) whether every row of every condition annihilates it: the
// directions in the kernel of the stacked rows, checked condition by condition without stacking.
List symObsAnnihilates(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                       double pIn, int Nt, IntegerMatrix V, int cores = 1) {
  if (V.ncol() < zSlots.size()) {
    IntegerMatrix pts(1, nLeaves);
    for (int L = 0; L < nLeaves; ++L) pts(0, L) = point[L];
    NumericVector pr(1);
    pr[0] = pIn;
    SEXP lanes = symObsAnnihilatesBatch(tapes, nLeaves, nStates, zSlots, pts, pr, Nt, V, cores);
    if (!Rf_isNull(lanes)) return List(List(lanes)[0]);
  }
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1, T = tapes.size(), D = V.ncol();
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p;
    if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }
  std::vector<std::vector<u64> > Vc(D, std::vector<u64>(nz, 0));
  for (int d = 0; d < D; ++d)
    for (int c = 0; c < nz; ++c) Vc[d][c] = red(V(c, d), p);
  TapeSet ts(tapes, nStates, w);
  int nThr = cores > 0 ? cores : 1;
  std::vector<char> okFlag(T, 1);
  std::vector<std::vector<char> > bad(T, std::vector<char>(D, 0));
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1 && T > 1)
  for (int t = 0; t < T; ++t) {
    try {
      CondRaw tmp;
      std::vector<std::vector<u64> > rows;
      if (!build_one_condition_blocked(ts.get(t, tmp), leafVal, nLeaves, nStates, nz, Nt, p, rows, 1)) {
        okFlag[t] = 0;
        continue;
      }
      const Barrett br(p);
      for (const auto& r : rows)
        for (int d = 0; d < D; ++d) {
          if (bad[t][d]) continue;
          u64 acc = 0;
          for (int c = 0; c < nz; ++c)
            if (r[c] && Vc[d][c]) acc = br.red(acc + (r[c] % p) * Vc[d][c]);
          if (acc) bad[t][d] = 1;
        }
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);
  LogicalVector zero(D);
  for (int d = 0; d < D; ++d) {
    zero[d] = 1;
    for (int t = 0; t < T; ++t) if (bad[t][d]) { zero[d] = 0; break; }
  }
  return List::create(_["ok"] = true, _["zero"] = zero);
}

// Per point (row of `points`) and column of V (nz x D residues) whether every row of every condition annihilates
// the column, from jets seeded along V where cheaper than reverse-mode rows, else from the rows. ok is FALSE at a
// vanishing denominator; NULL where a tape holds a steady-state seed (duals laid out over the coordinates).
SEXP symObsAnnihilatesBatch(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerMatrix points,
                            NumericVector primes, int Nt, IntegerMatrix V, int cores) {
  int nz = zSlots.size(), T = tapes.size(), D = V.ncol(), nB = points.nrow();
  TapeSet ts(tapes, nStates, nz + 1);
  for (int t = 0; t < T; ++t)
    if (ts.td[ts.base[t]].hasIcSeed) return R_NilValue;
  std::vector<int> colOf(nLeaves, -1);
  for (int c = 0; c < nz; ++c) colOf[zSlots[c]] = c;
  // lanes per pass, bounded so that the Taylor values of one condition stay within chain_block_bytes()
  size_t slots = 0;
  for (int t = 0; t < T; ++t) slots = std::max(slots, ts.td[ts.base[t]].op.size());
  double perLane = (double)(nLeaves + nStates + slots) * (Nt + 1) * 8.0;
  int B = std::max(1, std::min(D, (int)(chain_block_bytes() / perLane) - 1));
  int nBlk = D > 0 ? (D + B - 1) / B : 0;
  std::vector<char> lanes(T);
  bool anyRows = false;
  for (int t = 0; t < T; ++t) {
    const CondRaw& cd = ts.td[ts.base[t]];
    lanes[t] = !(reverse_pays(cd.gOut.size(), nz, Nt) && (double)D + 1 >= (double)cd.gOut.size() * (Nt + 2));
    anyRows = anyRows || !lanes[t];
  }
  // the leaves with one dual per coordinate, per point, for the conditions read from their rows
  std::vector<std::vector<std::vector<u64> > > oneHot(anyRows ? nB : 0);
  for (int bi = 0; bi < (int)oneHot.size(); ++bi) {
    u64 p = (u64)primes[bi];
    oneHot[bi].assign(nLeaves, std::vector<u64>(nz + 1, 0));
    for (int L = 0; L < nLeaves; ++L) {
      long long pv = (long long)points(bi, L) % (long long)p;
      if (pv < 0) pv += p;
      oneHot[bi][L][0] = (u64)pv;
      if (colOf[L] >= 0) oneHot[bi][L][1 + colOf[L]] = 1;
    }
  }
  std::vector<std::atomic<char> > bad((size_t)nB * D);
  for (auto& x : bad) x.store(0);
  std::vector<std::atomic<char> > okPt(nB);
  for (auto& x : okPt) x.store(1);
  long long nTask = (long long)nB * T * nBlk;
  int nThr = cores > 0 ? cores : 1;
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1 && nTask > 1)
  for (long long k = 0; k < nTask; ++k) {
    try {
      int bi = (int)(k / ((long long)T * nBlk)), rem = (int)(k % ((long long)T * nBlk));
      int t = rem / nBlk, b = rem % nBlk;
      int d0 = b * B, d1 = std::min(D, d0 + B), nd = d1 - d0;
      if (!okPt[bi].load() || (!lanes[t] && b > 0)) continue;
      bool open = false;
      for (int d = d0; d < d1 && !open; ++d) open = !bad[(size_t)bi * D + d].load();
      if (!open) continue;
      u64 p = (u64)primes[bi];
      if (!lanes[t]) {
        CondRaw tmp;
        std::vector<std::vector<u64> > rows;
        if (!build_one_condition_blocked(ts.get(t, tmp), oneHot[bi], nLeaves, nStates, nz, Nt, p, rows, 1)) {
          okPt[bi].store(0);
          continue;
        }
        const Barrett br(p);
        for (int d = 0; d < D; ++d) {
          if (bad[(size_t)bi * D + d].load()) continue;
          for (const auto& r : rows) {
            u64 acc = 0;
            for (int c = 0; c < nz; ++c)
              if (r[c] && V(c, d)) acc = br.red(acc + (r[c] % p) * red(V(c, d), p));
            if (acc) { bad[(size_t)bi * D + d].store(1); break; }
          }
        }
        continue;
      }
      std::vector<std::vector<u64> > lv(nLeaves, std::vector<u64>(nd + 1, 0));
      for (int L = 0; L < nLeaves; ++L) {
        long long pv = (long long)points(bi, L) % (long long)p;
        if (pv < 0) pv += p;
        lv[L][0] = (u64)pv;
        if (colOf[L] >= 0)
          for (int d = d0; d < d1; ++d) lv[L][1 + d - d0] = red(V(colOf[L], d), p);
      }
      CondRaw tmp;
      std::vector<std::vector<u64> > rows;
      if (!build_one_condition(ts.get(t, tmp), lv, nLeaves, nStates, nd, nd + 1, Nt, p, rows)) {
        okPt[bi].store(0);
        continue;
      }
      for (const auto& r : rows)
        for (int j = 0; j < nd; ++j)
          if (r[j] % p) bad[(size_t)bi * D + d0 + j].store(1);
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  List out(nB);
  for (int bi = 0; bi < nB; ++bi) {
    if (!okPt[bi].load()) { out[bi] = List::create(_["ok"] = false); continue; }
    LogicalVector zero(D);
    for (int d = 0; d < D; ++d) zero[d] = !bad[(size_t)bi * D + d].load();
    out[bi] = List::create(_["ok"] = true, _["zero"] = zero);
  }
  return out;
}

// Conditions in `order` (all tapes if empty) join a reduced basis with the rows to Lie order Nt at the point; one is
// chosen where the rows raise the rank or `keep` marks it, until the rank reaches `target`. Chunks give the choice of
// one at a time; `gain` is the rank each chosen condition adds, `taken` the conditions read.
List symObsCover(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                 double pIn, int Nt, int target, int cores, IntegerVector order, LogicalVector keep) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1, T = tapes.size();
  std::vector<int> ord;
  if (order.size() == 0) {
    for (int t = 0; t < T; ++t) ord.push_back(t);
  } else {
    for (int i = 0; i < order.size(); ++i) {
      if (order[i] < 0 || order[i] >= T) throw std::invalid_argument("order names no tape");
      ord.push_back(order[i]);
    }
  }
  int n = (int)ord.size();
  std::vector<char> forced(T, 0);
  for (int t = 0; t < keep.size() && t < T; ++t) forced[t] = keep[t] ? 1 : 0;
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
  for (int L = 0; L < nLeaves; ++L) {
    long long pv = (long long)point[L] % (long long)p;
    if (pv < 0) pv += p;
    leafVal[L][0] = (u64)pv;
    if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
  }
  TapeSet ts(tapes, nStates, w);
  int nThr = cores > 0 ? cores : 1, chunk = std::max(8, 2 * nThr);
  RrefBasis basis(nz, p);
  std::vector<int> chosen, gain;
  int taken = 0;
  for (int i0 = 0; i0 < n && basis.rank() < target; i0 += chunk) {
    int i1 = std::min(n, i0 + chunk), nc = i1 - i0;
    std::vector<std::vector<std::vector<u64> > > perRows(nc);
    std::vector<char> okFlag(nc, 1);
    OmpGuard guard_;
    #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1 && nc > 1)
    for (int i = i0; i < i1; ++i) {
      try {
        CondRaw tmp;
        if (!build_one_condition_blocked(ts.get(ord[i], tmp), leafVal, nLeaves, nStates, nz, Nt, p,
                                         perRows[i - i0], 1))
          okFlag[i - i0] = 0;
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
    for (int i = 0; i < nc; ++i)
      if (!okFlag[i]) return List::create(_["ok"] = false);
    std::vector<int> owner;
    std::vector<std::vector<u64> > rows;
    for (int i = 0; i < nc; ++i)
      for (auto& r : perRows[i]) {
        rows.push_back(std::move(r));
        owner.push_back(i);
      }
    std::vector<int> from;
    basis.add(rows, nThr, &from);
    std::vector<int> g(nc, 0);
    for (int k : from) ++g[owner[k]];
    for (int i = 0; i < nc; ++i)
      if (g[i] > 0 || forced[ord[i0 + i]]) {
        chosen.push_back(ord[i0 + i]);
        gain.push_back(g[i]);
      }
    taken = i1;
  }
  IntegerVector out(chosen.begin(), chosen.end()), gv(gain.begin(), gain.end());
  return List::create(_["ok"] = true, _["conditions"] = out, _["gain"] = gv, _["rank"] = basis.rank(),
                      _["taken"] = taken);
}

// Rank of the stacked single-segment rows of `tapes` up to each Lie order 0..Nt: the rows of an
// order join the reduced basis together, the orders ascending. The rows up to order k do not depend
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
  TapeSet ts(tapes, nStates, w);
  std::vector<std::vector<std::vector<u64> > > perRows(T);
  std::vector<char> okFlag(T, 1);
  int inner = chain_inner_threads(cores, T);
  int nThr = cores > 0 ? cores : 1;
  OmpGuard guard_;
  #pragma omp parallel for num_threads(nThr) schedule(dynamic) if (nThr > 1 && T > 1)
  for (int t = 0; t < T; ++t) {
    try {
      CondRaw tmp;
      if (!build_one_condition_blocked(ts.get(t, tmp), leafVal, nLeaves, nStates, nz, Nt, p,
                                       perRows[t], inner))
        okFlag[t] = 0;
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);

  // rows of a condition are ordered by output, then order
  RrefBasis basis(nz, p);
  IntegerVector ranks(Nt + 1);
  for (int k = 0; k <= Nt; ++k) {
    std::vector<std::vector<u64> > batch;
    for (int t = 0; t < T; ++t) {
      int nOut = (int)(perRows[t].size() / (size_t)(Nt + 1));
      for (int gi = 0; gi < nOut; ++gi) batch.push_back(std::move(perRows[t][(size_t)gi * (Nt + 1) + k]));
    }
    basis.add(batch, nThr);
    ranks[k] = basis.rank();
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
  for (int sj = 0; sj < nSeg; ++sj) segs.push_back(extract_seg_raw(chain[sj], nLeaves, nStates, w, p));
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
  TapeSet ts(tapes, nStates, 2);
  std::vector<std::vector<std::vector<u64> > > perRows(T);
  std::vector<char> okFlag(T, 1);
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t) {
    try {
      CondRaw tmp;
      if (!build_one_condition(ts.get(t, tmp), leafVal, nLeaves, nStates, 1, 2, Nt, p, perRows[t]))
        okFlag[t] = 0;
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
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

  // flat copy of the points for the parallel region; constants are reduced per prime inside
  // the build, so one extraction serves every pair
  std::vector<int> pts((size_t)nB * nLeaves);
  for (int bi = 0; bi < nB; ++bi)
    for (int L = 0; L < nLeaves; ++L) pts[(size_t)bi * nLeaves + L] = points(bi, L);
  std::vector<u64> pr(nB);
  for (int bi = 0; bi < nB; ++bi) pr[bi] = (u64)primes[bi];

  TapeSet ts(tapes, nStates, w);

  std::vector<char> okFlag(nB, 1);
  // fewer pairs than cores: the rest of the cores build the column blocks of each condition
  int inner = chain_inner_threads(cores, nB);
  std::vector<std::vector<std::vector<u64> > > redRows(nB);
  std::vector<std::vector<int> > redPiv(nB);
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int bi = 0; bi < nB; ++bi) {
    try {
      u64 p = pr[bi];
      std::vector<std::vector<u64> > leafVal(nLeaves, std::vector<u64>(w, 0));
      for (int L = 0; L < nLeaves; ++L) {
        long long pv = (long long)pts[(size_t)bi * nLeaves + L] % (long long)p; if (pv < 0) pv += p;
        leafVal[L][0] = (u64)pv;
        if (dualCol[L] >= 0) leafVal[L][1 + dualCol[L]] = 1;
      }
      std::vector<std::vector<u64> > rows;
      bool ok = true;
      CondRaw tmp;
      for (int t = 0; t < T && ok; ++t)
        if (!build_one_condition_blocked(ts.get(t, tmp), leafVal, nLeaves, nStates, nz, Nt, p, rows, inner))
          ok = false;
      if (!ok) { okFlag[bi] = 0; continue; }
      std::vector<int> pivots = rref_mod(rows, p);
      int rank = (int)pivots.size();
      redRows[bi].assign(rank, std::vector<u64>(nz));
      for (int i = 0; i < rank; ++i)
        for (int c = 0; c < nz; ++c) redRows[bi][i][c] = rows[i][c];
      redPiv[bi] = pivots;
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();

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

// Series echelon rows as an integer matrix (rank x nz*N).
static IntegerMatrix series_matrix(const std::vector<std::vector<u64> >& S, int rank, int width) {
  IntegerMatrix M(rank, width);
  for (int i = 0; i < rank; ++i)
    for (int j = 0; j < width; ++j) M(i, j) = (int)S[i][j];
  return M;
}

// Where the series rank falls below the rank of the stacked coefficient rows, some
// kernel direction moves with the gap lengths. If the series kernel is polynomial in
// eps it is replaced by the value at eps = 1, the actual gaps: R, pivots and rank then
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
// the state crosses a gap by promoting the local-time Taylor coefficients into that
// gap's axis. Each segment's output jet, to total gap-degree Mtot, contributes
// z-gradient rows per gap monomial; a direction is non-identifiable iff every
// monomial annihilates it, so the stacked rows are reduced once over GF(p).
List symObsNullChain(List chains, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int Mtot, int cores = 1,
                     IntegerVector NtChain = IntegerVector::create(),
                     SEXP pointSer = R_NilValue, int serOrder = -1) {
  u64 p = (u64)pIn;
  int nz = zSlots.size(), w = nz + 1;
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  int T = chains.size();

  // serial pre-pass into flat vectors: the parallel build reads no Value or List
  std::vector<int> leafPt(nLeaves);
  for (int L = 0; L < nLeaves; ++L) leafPt[L] = point[L];
  // a chain whose rank saturated at a lower Lie order is built only to that order:
  // the rows at Nt add nothing, and the jet cost grows with the order
  std::vector<int> ntOf(T, Nt);
  if (NtChain.size() == T)
    for (int t = 0; t < T; ++t) ntOf[t] = std::min(Nt, (int)NtChain[t]);
  std::vector<std::vector<SegRaw> > chainsRaw(T);
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    chainsRaw[t].reserve(nSeg);
    for (int sj = 0; sj < nSeg; ++sj)
      chainsRaw[t].push_back(extract_seg_raw(segs[sj], nLeaves, nStates, w, p));
  }
  // an optional continuation series of the leaves (nLeaves x T)
  std::vector<std::vector<u64> > leafSer;
  bool lineMode = serOrder >= 0;
  if (!Rf_isNull(pointSer)) {
    IntegerMatrix M = as<IntegerMatrix>(pointSer);
    if (M.nrow() == nLeaves && M.ncol() > 0) {
      leafSer.assign(nLeaves, std::vector<u64>(M.ncol(), 0));
      for (int L = 0; L < nLeaves; ++L)
        for (int j = 0; j < M.ncol(); ++j) leafSer[L][j] = red(M(L, j), p);
    } else if (lineMode) {
      throw std::invalid_argument("point_ser must have one row per leaf");
    }
  }
  // line mode without a series: the leaves stay at the point, the axis still has an order
  if (lineMode && leafSer.empty()) leafSer.assign(nLeaves, std::vector<u64>(1, 0));

  // parallel per-condition build: each thread owns one row block; a vanishing
  // denominator in any condition marks failure (no early return inside OpenMP).
  std::vector<std::vector<std::vector<u64> > > perRows(T), perSer(T);
  std::vector<char> okFlag(T, 1);
  int inner = chain_inner_threads(cores, T);
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && T > 1)
  for (int t = 0; t < T; ++t) {
    try {
      int B = chain_block_size(chainsRaw[t], nLeaves, nStates, nz, ntOf[t], Mtot,
                               !leafSer.empty(), chain_block_bytes(), serOrder);
      if (!build_chain_blocked(chainsRaw[t], nLeaves, nStates, nz, dualCol, leafPt,
                               ntOf[t], Mtot, p, perRows[t], perSer[t],
                               leafSer.empty() ? nullptr : &leafSer, B, inner, serOrder))
        okFlag[t] = 0;
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();

  for (int t = 0; t < T; ++t)
    if (!okFlag[t]) return List::create(_["ok"] = false);

  std::vector<std::vector<u64> > rows;
  for (int t = 0; t < T; ++t)
    for (size_t i = 0; i < perRows[t].size(); ++i)
      rows.push_back(std::move(perRows[t][i]));

  std::vector<int> pivots = rref_mod(rows, p);
  rows.resize(pivots.size());

  int N = lineMode ? serOrder + 1 : Mtot + 1;
  std::vector<std::vector<u64> > ser;
  for (int t = 0; t < T; ++t)
    for (size_t i = 0; i < perSer[t].size(); ++i) ser.push_back(std::move(perSer[t][i]));
  int rankS = rref_series_mod(ser, nz, N, p, std::vector<char>(nz, 1));
  bool atOne = false;
  // along a line the result is the series in t, not the value at t = 1
  if (!lineMode) kernel_at_one_or_keep(ser, rankS, nz, N, p, rows, pivots, atOne);
  return chain_result(rows, pivots, nz, rankS, ser, N, atOne);
}

// The unreduced rows of one chain to Lie order Nt and gap order Mtot, as symObsNullChain
// builds them for that chain alone, with the Lie order of each row and series row. Zero
// series rows are kept, so that every series row has an order. The coefficients at an
// order do not depend on the order the jets are taken to, so where no segment boundary
// moves (`plain`) the rows of a lower order are those with a lower tag, in the same
// sequence; symChainReduce then gives the kernel of that order. With `profile` (gap order
// 0) the rank per Lie order, as symObsChainRankProfile, is returned as well.
List chain_rows(List chain, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                double pIn, int Nt, int Mtot, int threads, bool profile, std::string session);

List symObsChainRows(List chain, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int Mtot, int threads = 1,
                     bool profile = false, std::string session = "") {
  List out = chain_rows(chain, nLeaves, nStates, zSlots, point, pIn, Nt, Mtot, threads, profile, session);
  if (Nt >= 200) release_heap();
  return out;
}

List chain_rows(List chain, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                double pIn, int Nt, int Mtot, int threads, bool profile, std::string session) {
  u64 p = (u64)pIn;
  int nz = zSlots.size();
  std::vector<int> dualCol(nLeaves, -1);
  for (int c = 0; c < nz; ++c) dualCol[zSlots[c]] = c;
  std::vector<int> leafPt(nLeaves);
  for (int L = 0; L < nLeaves; ++L) leafPt[L] = point[L];
  int nSeg = chain.size();
  std::vector<SegRaw> segs;
  segs.reserve(nSeg);
  for (int sj = 0; sj < nSeg; ++sj) segs.push_back(extract_seg_raw(chain[sj], nLeaves, nStates, nz + 1, p));
  std::vector<char> moves;
  if (!chain_moves(segs, nLeaves, nz + 1, dualCol, leafPt, p, moves))
    return List::create(_["ok"] = false);
  bool plain = true;
  for (int sj = 0; sj < nSeg; ++sj) {
    if (moves[sj]) plain = false;
    for (int o : segs[sj].op) if (o == OP_SPOW) plain = false;
  }
  int B = chain_block_size(segs, nLeaves, nStates, nz, Nt, Mtot, false, chain_block_bytes());
  std::vector<std::vector<u64> > rows, srows;
  if (!build_chain_blocked(segs, nLeaves, nStates, nz, dualCol, leafPt, Nt, Mtot, p, rows,
                           srows, nullptr, B, threads, -1, true, session))
    return List::create(_["ok"] = false);
  int N = Mtot + 1;
  // rows run by segment, output, order and gap monomial; series rows by segment, output, order
  int nM = 1;
  {
    int K = std::max(1, nSeg - 1);
    for (int i = 1; i <= Mtot; ++i) nM = nM * (K + i) / i;
  }
  std::vector<int> tag, stag;
  for (int sj = 0; sj < nSeg; ++sj)
    for (size_t gi = 0; gi < segs[sj].gOut.size(); ++gi)
      for (int k = 0; k <= Nt; ++k) {
        stag.push_back(k);
        for (int mo = 0; mo < nM; ++mo) tag.push_back(k);
      }
  if (tag.size() != rows.size() || stag.size() != srows.size()) {
    plain = false;
    tag.assign(rows.size(), Nt);
    stag.assign(srows.size(), Nt);
  }
  IntegerMatrix Rm((int)rows.size(), nz), Sm((int)srows.size(), nz * N);
  for (size_t i = 0; i < rows.size(); ++i)
    for (int c = 0; c < nz; ++c) Rm((int)i, c) = (int)rows[i][c];
  for (size_t i = 0; i < srows.size(); ++i)
    for (int c = 0; c < nz * N; ++c) Sm((int)i, c) = (int)srows[i][c];
  IntegerVector tv(tag.begin(), tag.end()), sv(stag.begin(), stag.end());
  List out = List::create(_["ok"] = true, _["rows"] = Rm, _["tag"] = tv, _["srows"] = Sm,
                          _["stag"] = sv, _["N"] = N, _["Nt"] = Nt, _["plain"] = plain);
  if (profile && plain && Mtot == 0) {
    std::vector<std::vector<int> > byOrder(Nt + 1);
    for (size_t i = 0; i < rows.size(); ++i) byOrder[tag[i]].push_back((int)i);
    std::vector<std::vector<u64> > basis;
    std::vector<int> pivCol;
    IntegerVector ranks(Nt + 1);
    for (int k = 0; k <= Nt; ++k) {
      for (size_t q = 0; q < byOrder[k].size(); ++q) {
        std::vector<u64> r = rows[byOrder[k][q]];
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
    out["ranks"] = ranks;
  }
  return out;
}

// Drops the jets kept for continued calls; returns how many were kept.
int symClearJets() {
  std::lock_guard<std::mutex> lk(g_sess_mu);
  int n = (int)g_sess.size();
  g_sess.clear();
  release_heap();
  return n;
}

// Sets the bytes of Taylor values per column block when SYMIDENT_BLOCKMB is not set; returns the
// bytes in effect.
double symSetBlockBytes(double bytes) {
  if (bytes >= 1048576.0) g_block_bytes.store((long long)bytes);
  return chain_block_bytes();
}

// The kernel of one chain from the unreduced rows and series rows (symObsChainRows, the
// rows of a Lie order selected by tag), reduced as symObsNullChain reduces them: the same
// rows in the same sequence give the same result. Zero series rows are left out.
List symChainReduce(IntegerMatrix rowsIn, IntegerMatrix srowsIn, int nz, int N, double pIn) {
  u64 p = (u64)pIn;
  std::vector<std::vector<u64> > rows(rowsIn.nrow(), std::vector<u64>(nz, 0)), ser;
  for (int i = 0; i < rowsIn.nrow(); ++i)
    for (int c = 0; c < nz; ++c) rows[i][c] = red(rowsIn(i, c), p);
  for (int i = 0; i < srowsIn.nrow(); ++i) {
    std::vector<u64> s((size_t)nz * N, 0);
    bool any = false;
    for (int c = 0; c < nz * N; ++c) {
      s[c] = red(srowsIn(i, c), p);
      any |= s[c] != 0;
    }
    if (any) ser.push_back(std::move(s));
  }
  std::vector<int> pivots = rref_mod(rows, p);
  rows.resize(pivots.size());
  int rankS = rref_series_mod(ser, nz, N, p, std::vector<char>(nz, 1));
  bool atOne = false;
  kernel_at_one_or_keep(ser, rankS, nz, N, p, rows, pivots, atOne);
  return chain_result(rows, pivots, nz, rankS, ser, N, atOne);
}

// Rank of each segment's own rows at orders 0..Nt, at gap order 0, with a unit dual
// column per state at the segment start and, where the start time depends on the
// coordinates, a column -(k+1) y_{k+1}. `stateVals` (nSeg x nStates) replaces the start
// states. Also returns the start states and the field at them. The columns of
// `dropStates` (the jets of known inputs) are left out of the ranks. With `fromStates`,
// the initial and event values are not evaluated: the starts are the rows of `stateVals`.
List segment_ranks(List segs, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                   double pIn, int Nt, IntegerMatrix stateVals, int threads, IntegerVector dropStates,
                   bool fromStates, std::string session);

List symSegmentRanks(List segs, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt,
                     IntegerMatrix stateVals = IntegerMatrix(0, 0), int threads = 1,
                     IntegerVector dropStates = IntegerVector(0), bool fromStates = false,
                     std::string session = "") {
  List out = segment_ranks(segs, nLeaves, nStates, zSlots, point, pIn, Nt, stateVals, threads, dropStates,
                           fromStates, session);
  if (Nt >= 200) release_heap();
  return out;
}

List segment_ranks(List segs, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                   double pIn, int Nt, IntegerMatrix stateVals, int threads, IntegerVector dropStates,
                   bool fromStates, std::string session) {
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
  if (!session.empty() && nSeg == 1 && fromStates && givenStates) {
    // one segment from given states: the jets kept under `session` are continued
    SegRaw seg = extract_seg_raw(segs[0], nLeaves, nStates, 1, p);
    bool mv = false;
    if (seg.hasTm) {
      std::vector<std::vector<u64> > tmVal(nLeaves + seg.tmOp.size(), std::vector<u64>(w0, 0));
      for (int L = 0; L < nLeaves; ++L) {
        tmVal[L][0] = leaf[L];
        if (dualCol[L] >= 0) tmVal[L][1 + dualCol[L]] = 1;
      }
      if (eval_ic_order0(tmVal, seg.tmOp, seg.tmA, seg.tmB, seg.tmCval, nLeaves, w0, p))
        for (int c = 1; c < w0; ++c) if (tmVal[seg.tmOut[0]][c]) mv = true;
    }
    std::vector<u64> qs(nStates);
    for (int i = 0; i < nStates; ++i) qs[i] = red(stateVals(0, i), p);
    std::vector<int> dr(dropStates.begin(), dropStates.end()), rk;
    std::vector<u64> fv;
    if (session_ranks(session, seg, nLeaves, nStates, nz, dualCol, leaf, qs, mv, Nt, threads, dr, p,
                      rk, fv)) {
      for (int k = 0; k <= Nt; ++k) ranks(0, k) = rk[k];
      for (int i = 0; i < nStates; ++i) {
        start(0, i) = (int)qs[i];
        field(0, i) = i < (int)fv.size() ? (int)fv[i] : 0;
      }
      moves[0] = mv;
      return List::create(_["ok"] = true, _["ranks"] = ranks, _["start"] = start,
                          _["field"] = field, _["moves"] = moves);
    }
  }
  std::vector<u64> q(nStates, 0);
  for (int sj = 0; sj < nSeg; ++sj) {
    SegRaw seg = extract_seg_raw(segs[sj], nLeaves, nStates, 1, p);
    // start state at gap order 0
    if (fromStates && givenStates) {
      // the starts are those of stateVals alone, so each segment of a chain is ranked alone
    } else if (sj == 0) {
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
      int evBase = nLeaves + (seg.evStates ? nStates : 0);
      std::vector<std::vector<u64> > evVal(evBase + seg.evOp.size(), std::vector<u64>(1, 0));
      for (int L = 0; L < nLeaves; ++L) evVal[L][0] = leaf[L];
      if (seg.evStates)
        for (int i = 0; i < nStates; ++i) evVal[nLeaves + i][0] = q[i];
      if (!eval_ic_order0(evVal, seg.evOp, seg.evA, seg.evB, seg.evCval, evBase, 1, p))
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
    std::vector<std::vector<std::vector<u64> > > rB(nb), nB(nb);
    std::vector<u64> fVal(nStates, 0), yNext((size_t)nOutS * (Nt + 1), 0);
    std::vector<char> okB(nb, 1);
    OmpGuard guard_;
    #pragma omp parallel for num_threads(nThrS) schedule(dynamic) if (nThrS > 1)
    for (int b = 0; b < nb; ++b) {
      try {
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
                                 seg.stateSlots, seg.fOut, seg.gOut, pb, Nt + 1, Nt, p, rB[b],
                                 nullptr, &nB[b])) {
          okB[b] = 0; continue;
        }
        if (b == 0) {
          for (int i = 0; i < nStates && i < (int)seg.fOut.size(); ++i)
            fVal[i] = val[seg.fOut[i]][0] % p;
          for (int gi = 0; gi < nOutS; ++gi)
            for (int k = 0; k <= Nt; ++k)
              yNext[(size_t)gi * (Nt + 1) + k] = val[seg.gOut[gi]][(size_t)(k + 1) * wb] % p;
        }
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
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
    // rows of negative order (an output 0/0 at the start, poles along the states) join order 0
    std::map<std::vector<u64>, std::vector<u64> > neg;
    for (int b = 0; b < nb; ++b)
      for (const auto& r : nB[b]) {
        std::vector<u64> key(r.begin(), r.begin() + 3);
        std::vector<u64>& d = neg[key];
        if (d.empty()) d.assign(nzA, 0);
        std::copy(r.begin() + 3, r.end(), d.begin() + b * B);
      }
    std::vector<std::vector<u64> > negRows;
    for (auto& kv : neg) negRows.push_back(kv.second);
    for (auto& r : rows)
      for (int i = 0; i < dropStates.size(); ++i) r[nz + dropStates[i]] = 0;
    for (auto& r : negRows)
      for (int i = 0; i < dropStates.size(); ++i) r[nz + dropStates[i]] = 0;
    // rows are ordered by output, then order; rank up to each order, incrementally
    int nOut = (int)seg.gOut.size();
    std::vector<std::vector<u64> > basis;
    std::vector<int> pivCol;
    for (int k = 0; k <= Nt; ++k) {
      for (int gi = 0; gi < nOut + (k == 0 ? (int)negRows.size() : 0); ++gi) {
        std::vector<u64> r = gi < nOut ? rows[(size_t)gi * (Nt + 1) + k] : negRows[gi - nOut];
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

  // serial pre-pass per distinct prime into flat vectors: the parallel build reads no Value
  // or List
  std::vector<std::vector<std::vector<SegRaw> > >
      segsRaw(nPr, std::vector<std::vector<SegRaw> >(T));
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    for (int pi = 0; pi < nPr; ++pi) {
      segsRaw[pi][t].reserve(nSeg);
      for (int sj = 0; sj < nSeg; ++sj)
        segsRaw[pi][t].push_back(extract_seg_raw(segs[sj], nLeaves, nStates, w, distinct[pi]));
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
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int e = 0; e < nB; ++e) {
    try {
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
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();

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
// point per evaluation and `primes` the matching prime; NtChain caps a chain's Lie order as in
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

  // serial pre-pass into flat vectors: the parallel part reads no Value or List
  std::vector<std::vector<std::vector<SegRaw> > >
      segsRaw(nPr, std::vector<std::vector<SegRaw> >(T));
  for (int t = 0; t < T; ++t) {
    List segs = chains[t];
    int nSeg = segs.size();
    for (int pi = 0; pi < nPr; ++pi) {
      segsRaw[pi][t].reserve(nSeg);
      for (int sj = 0; sj < nSeg; ++sj)
        segsRaw[pi][t].push_back(extract_seg_raw(segs[sj], nLeaves, nStates, w, distinct[pi]));
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
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nTask > 1)
  for (long k = 0; k < nTask; ++k) {
    try {
      int e = (int)(k / T), t = (int)(k % T);
      std::vector<int> leafPt(pts.begin() + (size_t)e * nLeaves,
                              pts.begin() + (size_t)(e + 1) * nLeaves);
      const std::vector<SegRaw>& segs = segsRaw[primeIdx[e]][t];
      int B = chain_block_size(segs, nLeaves, nStates, nz, ntOf[t], Mtot, false,
                               chain_block_bytes());
      if (!build_chain_blocked(segs, nLeaves, nStates, nz, dualCol, leafPt, ntOf[t], Mtot,
                               pr[e], rowsET[k], serET[k], nullptr, B, inner))
        okET[k] = 0;
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();

  // per point: stack, reduce, series rank
  std::vector<char> okE(nB, 1), atOneE(nB, 0);
  std::vector<std::vector<std::vector<u64> > > redRows(nB), redSer(nB);
  std::vector<std::vector<int> > redPiv(nB);
  std::vector<int> rankS(nB, 0);
  OmpGuard guard2_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic) \
          if (cores > 1 && nB > 1)
  for (int e = 0; e < nB; ++e) {
    try {
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
    } catch (...) {
      guard2_.capture();
    }
  }
  guard2_.rethrow();

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
  // eps = 1 and return the reduced kernel rows
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
    OmpGuard guard_;
    #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic)
    for (int c = 0; c < nz; ++c) {
      try {
        if (!mask[c]) continue;
        std::vector<std::vector<u64> > B(A.begin(), A.begin() + rank);
        std::vector<char> m2 = mask; m2[c] = 0;
        keep[c] = rref_series_mod(B, nz, N, p, m2) == rank;
      } catch (...) {
        guard_.capture();
      }
    }
    guard_.rethrow();
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

// A B over GF(p) for residues below p (int64 entries), rows of A in parallel.
NumericMatrix symMatMulMod(NumericMatrix A, NumericMatrix B, double pIn, int cores = 1) {
  u64 p = (u64)pIn;
  int n = A.nrow(), m = A.ncol(), k = B.ncol();
  if (B.nrow() != m) throw std::invalid_argument("matmul: inner dimensions differ");
  const Barrett br(p);
  const u64 LIM = (((u64)1 << 63) / p) * p;
  std::vector<u64> Bt((size_t)k * m);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < m; ++i) {
      long long v = (long long)B(i, j) % (long long)p;
      Bt[(size_t)j * m + i] = (u64)(v < 0 ? v + (long long)p : v);
    }
  NumericMatrix C(n, k);
  OmpGuard guard_;
  #pragma omp parallel for num_threads(cores > 0 ? cores : 1) schedule(dynamic, 8)
  for (int r = 0; r < n; ++r) {
    try {
      std::vector<u64> a(m);
      std::vector<int> nzc;
      for (int i = 0; i < m; ++i) {
        long long v = (long long)A(r, i) % (long long)p;
        a[i] = (u64)(v < 0 ? v + (long long)p : v);
        if (a[i]) nzc.push_back(i);
      }
      for (int j = 0; j < k; ++j) {
        const u64* b = &Bt[(size_t)j * m];
        u64 acc = 0;
        for (int i : nzc) {
          u64 t = acc + a[i] * b[i];
          acc = t >= LIM ? t - LIM : t;
        }
        C(r, j) = (double)br.red(acc);
      }
    } catch (...) {
      guard_.capture();
    }
  }
  guard_.rethrow();
  return C;
}

// The rows of M (residues below p), in order, that are independent of the rows before them, and the rank.
List symIndependentRows(NumericMatrix M, double pIn, int cores = 1) {
  u64 p = (u64)pIn;
  int nr = M.nrow(), nc = M.ncol();
  std::vector<std::vector<u64> > X(nr, std::vector<u64>(nc));
  for (int i = 0; i < nr; ++i)
    for (int c = 0; c < nc; ++c) {
      long long v = (long long)M(i, c) % (long long)p;
      X[i][c] = (u64)(v < 0 ? v + (long long)p : v);
    }
  RrefBasis basis(nc, p);
  std::vector<int> from;
  basis.add(X, cores > 0 ? cores : 1, &from);
  IntegerVector out(from.begin(), from.end());
  return List::create(_["rows"] = out, _["rank"] = basis.rank());
}

// Compiled twin of gfp.rref_modp (joint/equilibrate path, per sample point).
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
// the sequence requires, used to grow the sparse sampling until it stabilizes.
int symBMorder(IntegerVector seq, double pIn) {
  u64 p = (u64) pIn;
  int len = seq.size();
  std::vector<u64> s(len);
  for (int i = 0; i < len; ++i) s[i] = red(seq[i], p);
  return (int)berlekamp_massey(s, p).size() - 1;
}


// Univariate Cauchy (rational) interpolation A(t)/B(t) = r(t) over GF(p), with
// deg A = dN, deg B = dD, from samples (tnodes, rvals); normalized so B(0) = 1.
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

// The reduction of the polynomial system of a fiber over GF(p) before msolve. A
// polynomial is a map from a monomial, (variable, exponent) pairs by ascending variable,
// to a nonzero residue. (1) An equation a*v + b in one variable v fixes v = -b/a, which
// is substituted in every equation and factor of the saturation that holds v; a nonzero
// constant equation or a vanishing factor leaves no solution. (2) A variable from m0 on,
// in no factor, that occurs in one equation only and there only as the monomial v, is
// fixed by that equation for any values of the others: the equation goes. The result
// does not depend on the order of the steps.
namespace {

typedef std::vector<std::pair<int, int> > Mono;
typedef std::map<Mono, u64> SPoly;

std::vector<SPoly> fiber_decode(const IntegerVector& start, const IntegerVector& coef,
                                const IntegerVector& tstart, const IntegerVector& var,
                                const IntegerVector& ex) {
  std::vector<SPoly> out(start.size() > 0 ? start.size() - 1 : 0);
  for (size_t j = 0; j < out.size(); ++j)
    for (int t = start[(int)j]; t < start[(int)j + 1]; ++t) {
      Mono m;
      for (int k = tstart[t]; k < tstart[t + 1]; ++k) m.push_back({var[k], ex[k]});
      if (coef[t]) out[j][m] = (u64)coef[t];
    }
  return out;
}

List fiber_encode(const std::vector<SPoly>& ps, const std::vector<char>& alive) {
  std::vector<int> start{0}, coef, tstart{0}, var, ex;
  for (size_t j = 0; j < ps.size(); ++j) {
    if (!alive[j]) continue;
    for (const auto& kv : ps[j]) {
      coef.push_back((int)kv.second);
      for (const auto& ve : kv.first) { var.push_back(ve.first); ex.push_back(ve.second); }
      tstart.push_back((int)var.size());
    }
    start.push_back((int)coef.size());
  }
  return List::create(_["start"] = IntegerVector(start.begin(), start.end()),
                      _["coef"] = IntegerVector(coef.begin(), coef.end()),
                      _["tstart"] = IntegerVector(tstart.begin(), tstart.end()),
                      _["var"] = IntegerVector(var.begin(), var.end()),
                      _["exp"] = IntegerVector(ex.begin(), ex.end()));
}

// v = x in P; true if P changed
bool fiber_subst(SPoly& P, int v, u64 x, u64 p) {
  SPoly out;
  bool hit = false;
  for (const auto& kv : P) {
    Mono m;
    u64 c = kv.second;
    for (const auto& ve : kv.first) {
      if (ve.first == v) { c = mulmod(c, powmod(x, (u64)ve.second, p), p); hit = true; }
      else m.push_back(ve);
    }
    u64& slot = out[m];
    slot = addmod(slot, c, p);
  }
  if (!hit) return false;
  for (auto it = out.begin(); it != out.end();) it = it->second ? std::next(it) : out.erase(it);
  P.swap(out);
  return true;
}

// the variable of an equation a*v + b with a != 0, else -1
int fiber_linear_var(const SPoly& P) {
  int v = -1;
  for (const auto& kv : P) {
    if (kv.first.empty()) continue;
    if (kv.first.size() != 1 || kv.first[0].second != 1) return -1;
    if (v >= 0 && kv.first[0].first != v) return -1;
    v = kv.first[0].first;
  }
  return v;
}

bool fiber_constant(const SPoly& P) { return P.size() == 1 && P.begin()->first.empty(); }

}  // namespace

List symFiberReduce(IntegerVector ps, IntegerVector pc, IntegerVector pt, IntegerVector pv,
                    IntegerVector pe, IntegerVector ss, IntegerVector sc, IntegerVector st,
                    IntegerVector sv, IntegerVector se, int nv, int m0, double pIn) {
  u64 p = (u64)pIn;
  std::vector<SPoly> P = fiber_decode(ps, pc, pt, pv, pe), S = fiber_decode(ss, sc, st, sv, se);
  std::vector<char> palive(P.size(), 1), salive(S.size(), 1);
  std::vector<std::vector<int> > pocc((size_t)nv), socc((size_t)nv);
  for (size_t j = 0; j < P.size(); ++j) {
    if (P[j].empty()) palive[j] = 0;
    for (const auto& kv : P[j]) for (const auto& ve : kv.first) pocc[(size_t)ve.first].push_back((int)j);
  }
  for (size_t j = 0; j < S.size(); ++j)
    for (const auto& kv : S[j]) for (const auto& ve : kv.first) socc[(size_t)ve.first].push_back((int)j);
  std::vector<int> fixedVar, fixedVal;
  std::vector<char> isFixed((size_t)nv, 0);
  std::vector<int> queue;
  for (size_t j = 0; j < P.size(); ++j) if (palive[j]) queue.push_back((int)j);
  auto none = [] { return List::create(_["ok"] = false); };
  size_t qi = 0;
  while (qi < queue.size()) {
    int j = queue[qi++];
    if (!palive[(size_t)j]) continue;
    int v = fiber_linear_var(P[(size_t)j]);
    if (v < 0 || isFixed[(size_t)v]) continue;
    const SPoly& L = P[(size_t)j];
    Mono one{{v, 1}};
    u64 a = L.at(one), b = L.count(Mono()) ? L.at(Mono()) : 0;
    u64 x = mulmod(submod(0, b, p), invmod(a, p), p);
    isFixed[(size_t)v] = 1;
    fixedVar.push_back(v);
    fixedVal.push_back((int)x);
    for (int k : pocc[(size_t)v]) {
      if (!palive[(size_t)k] || !fiber_subst(P[(size_t)k], v, x, p)) continue;
      if (P[(size_t)k].empty()) { palive[(size_t)k] = 0; continue; }
      if (fiber_constant(P[(size_t)k])) return none();
      queue.push_back(k);
    }
    for (int k : socc[(size_t)v]) {
      if (!salive[(size_t)k] || !fiber_subst(S[(size_t)k], v, x, p)) continue;
      if (S[(size_t)k].empty()) return none();
      if (fiber_constant(S[(size_t)k])) salive[(size_t)k] = 0;
    }
  }
  if (!fixedVar.empty()) {
    // as after a substitution: a constant equation has no solution, a constant factor vanishes nowhere
    for (size_t j = 0; j < P.size(); ++j) if (palive[j] && fiber_constant(P[j])) return none();
    for (size_t j = 0; j < S.size(); ++j) {
      if (!salive[j]) continue;
      if (S[j].empty()) return none();
      if (fiber_constant(S[j])) salive[j] = 0;
    }
  }
  // the jets that one equation fixes
  std::vector<char> inSat((size_t)nv, 0);
  for (size_t j = 0; j < S.size(); ++j)
    if (salive[j]) for (const auto& kv : S[j]) for (const auto& ve : kv.first) inSat[(size_t)ve.first] = 1;
  std::vector<int> count((size_t)nv, 0);
  for (size_t j = 0; j < P.size(); ++j) {
    if (!palive[j]) continue;
    std::vector<int> vs;
    for (const auto& kv : P[j]) for (const auto& ve : kv.first) vs.push_back(ve.first);
    std::sort(vs.begin(), vs.end());
    vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
    for (int v : vs) ++count[(size_t)v];
  }
  std::vector<int> gone;
  bool more = true;
  while (more) {
    more = false;
    for (size_t j = 0; j < P.size() && !more; ++j) {
      if (!palive[j]) continue;
      for (const auto& kv : P[j]) {
        if (kv.first.size() != 1 || kv.first[0].second != 1) continue;
        int v = kv.first[0].first;
        if (v < m0 || inSat[(size_t)v] || count[(size_t)v] != 1) continue;
        bool alone = true;
        for (const auto& kw : P[j])
          if (&kw != &kv) for (const auto& ve : kw.first) if (ve.first == v) alone = false;
        if (!alone) continue;
        gone.push_back(v);
        palive[j] = 0;
        std::vector<int> vs;
        for (const auto& kw : P[j]) for (const auto& ve : kw.first) vs.push_back(ve.first);
        std::sort(vs.begin(), vs.end());
        vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
        for (int w : vs) --count[(size_t)w];
        more = true;
        break;
      }
    }
  }
  return List::create(_["ok"] = true, _["polys"] = fiber_encode(P, palive),
                      _["sat"] = fiber_encode(S, salive),
                      _["fixed_var"] = IntegerVector(fixedVar.begin(), fixedVar.end()),
                      _["fixed_val"] = IntegerVector(fixedVal.begin(), fixedVal.end()),
                      _["gone"] = IntegerVector(gone.begin(), gone.end()));
}

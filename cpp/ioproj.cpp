// Input-output projections over GF(p): polynomials evaluated at many points, the kernel of a dense matrix and the
// Taylor series of monomials in series (detection/ioproj.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

typedef uint64_t u64;
typedef int64_t i64;

// Reduction of x < 2^62 modulo p < 2^31 through a floating-point quotient, off by at most one.
struct Mod {
  u64 p;
  double ip;
  explicit Mod(u64 p_) : p(p_), ip(1.0 / (double)p_) {}
  inline u64 red(u64 x) const {
    u64 q = (u64)((double)x * ip);
    i64 r = (i64)(x - q * p);
    if (r < 0) r += (i64)p;
    if (r >= (i64)p) r -= (i64)p;
    return (u64)r;
  }
  inline u64 mul(u64 a, u64 b) const { return red(a * b); }
  inline u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + p - b; }
  u64 pow(u64 a, u64 e) const {
    u64 r = 1;
    while (e) {
      if (e & 1) r = mul(r, a);
      a = mul(a, a);
      e >>= 1;
    }
    return r;
  }
  u64 inv(u64 a) const { return pow(a, p - 2); }
};

u64 residue(i64 v, u64 p) {
  i64 r = v % (i64)p;
  return (u64)(r < 0 ? r + (i64)p : r);
}

void check_prime(double p) {
  if (!(p > 2 && p < 2147483648.0)) throw std::runtime_error("the prime must lie below 2^31");
}

// Values of K polynomials at M points: polynomial k has the terms off[k]..off[k+1]-1, term t the exponents exps[t, :]
// and the coefficient coef[t] modulo p.
py::array_t<i64> poly_eval(py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps,
                           py::array_t<i64, py::array::c_style | py::array::forcecast> coef,
                           py::array_t<i64, py::array::c_style | py::array::forcecast> off,
                           py::array_t<i64, py::array::c_style | py::array::forcecast> pts, double pIn, int threads) {
  check_prime(pIn);
  Mod md((u64)pIn);
  if (exps.ndim() != 2 || pts.ndim() != 2 || exps.shape(1) != pts.shape(1))
    throw std::runtime_error("exponents and points need the same number of variables");
  const int T = (int)exps.shape(0), V = (int)exps.shape(1), M = (int)pts.shape(0), K = (int)off.shape(0) - 1;
  const int32_t* E = exps.data();
  const i64* O = off.data();
  std::vector<u64> C(T);
  for (int t = 0; t < T; ++t) C[t] = residue(coef.data()[t], md.p);
  std::vector<int> top(V, 0);
  for (int t = 0; t < T; ++t)
    for (int v = 0; v < V; ++v) {
      if (E[t * V + v] < 0) throw std::runtime_error("negative exponent");
      if (E[t * V + v] > top[v]) top[v] = E[t * V + v];
    }
  std::vector<u64> X((size_t)M * V);
  for (size_t i = 0; i < X.size(); ++i) X[i] = residue(pts.data()[i], md.p);
  py::array_t<i64> out({M, K});
  i64* R = out.mutable_data();
  {
    py::gil_scoped_release rel;
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads > 0 ? threads : 1)
    for (int i = 0; i < M; ++i) {
      std::vector<std::vector<u64> > pw(V);
      for (int v = 0; v < V; ++v) {
        pw[v].resize(top[v] + 1);
        pw[v][0] = 1;
        for (int e = 1; e <= top[v]; ++e) pw[v][e] = md.mul(pw[v][e - 1], X[(size_t)i * V + v]);
      }
      for (int k = 0; k < K; ++k) {
        u64 s = 0;
        for (i64 t = O[k]; t < O[k + 1]; ++t) {
          u64 x = C[t];
          const int32_t* e = E + t * V;
          for (int v = 0; v < V && x; ++v)
            if (e[v]) x = md.mul(x, pw[v][e[v]]);
          s += x;
          if (s >= md.p) s -= md.p;
        }
        R[(size_t)i * K + k] = (i64)s;
      }
    }
  }
  return out;
}

// The kernel of A over GF(p): {"rank", "pivots", "free", "basis"}, each basis vector 1 on one free column and 0 on the
// other free columns.
py::dict kernel(py::array_t<i64, py::array::c_style | py::array::forcecast> A, double pIn, int threads) {
  check_prime(pIn);
  Mod md((u64)pIn);
  if (A.ndim() != 2) throw std::runtime_error("a matrix is needed");
  const int R = (int)A.shape(0), C = (int)A.shape(1);
  std::vector<u64> W((size_t)R * C);
  for (size_t i = 0; i < W.size(); ++i) W[i] = residue(A.data()[i], md.p);
  std::vector<int> piv;
  {
    py::gil_scoped_release rel;
    // row echelon form in blocks of three pivots: the rows below see the pending pivots through their multipliers
    // and take the three updates at once, f1 a1 + f2 a2 + f3 a3 < 2^64 for entries below 2^31
    const int Bk = 3;
    std::vector<std::vector<u64> > f(Bk, std::vector<u64>(R, 0));
    std::vector<int> pc;
    int r0 = 0, np = 0;
    auto flush = [&]() {
      if (!np) return;
      const int lo = r0 + np, c0 = pc[0];
      const u64* P0 = &W[(size_t)r0 * C];
      const u64* P1 = np > 1 ? &W[(size_t)(r0 + 1) * C] : nullptr;
      const u64* P2 = np > 2 ? &W[(size_t)(r0 + 2) * C] : nullptr;
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1) if (R - lo > 64)
      for (int i = lo; i < R; ++i) {
        const u64 f0 = f[0][i], f1 = np > 1 ? f[1][i] : 0, f2 = np > 2 ? f[2][i] : 0;
        if (!f0 && !f1 && !f2) continue;
        u64* row = &W[(size_t)i * C];
        for (int j = c0; j < C; ++j) {
          u64 acc = f0 * P0[j];
          if (P1) acc += f1 * P1[j];
          if (P2) acc += f2 * P2[j];
          row[j] = md.sub(row[j], md.red(acc));
        }
      }
      for (int k = 0; k < np; ++k) std::fill(f[k].begin(), f[k].end(), 0);
      r0 += np;
      np = 0;
      pc.clear();
    };
    auto colval = [&](int i, int c) {
      u64 v = W[(size_t)i * C + c];
      for (int k = 0; k < np; ++k)
        if (f[k][i]) v = md.sub(v, md.mul(f[k][i], W[(size_t)(r0 + k) * C + c]));
      return v;
    };
    for (int c = 0; c < C && r0 + np < R; ++c) {
      int pr = -1;
      for (int i = r0 + np; i < R; ++i)
        if (colval(i, c)) {
          pr = i;
          break;
        }
      if (pr < 0) continue;
      const int at = r0 + np;
      if (pr != at) {
        for (int j = 0; j < C; ++j) std::swap(W[(size_t)pr * C + j], W[(size_t)at * C + j]);
        for (int k = 0; k < np; ++k) std::swap(f[k][pr], f[k][at]);
      }
      // the new pivot row takes the pending updates and is normalized
      u64* prow = &W[(size_t)at * C];
      for (int k = 0; k < np; ++k) {
        const u64 g = f[k][at];
        if (!g) continue;
        const u64* P = &W[(size_t)(r0 + k) * C];
        for (int j = pc[k]; j < C; ++j)
          if (P[j]) prow[j] = md.sub(prow[j], md.mul(g, P[j]));
        f[k][at] = 0;
      }
      const u64 iv = md.inv(prow[c]);
      for (int j = c; j < C; ++j) prow[j] = md.mul(prow[j], iv);
      for (int i = at + 1; i < R; ++i) f[np][i] = colval(i, c);
      pc.push_back(c);
      piv.push_back(c);
      ++np;
      if (np == Bk) flush();
    }
    flush();
  }
  const int rank = (int)piv.size();
  std::vector<char> isPiv(C, 0);
  for (int c : piv) isPiv[c] = 1;
  std::vector<int> fr;
  for (int c = 0; c < C; ++c)
    if (!isPiv[c]) fr.push_back(c);
  const int k = (int)fr.size();
  py::array_t<i64> basis({k, C});
  i64* B = basis.mutable_data();
  {
    py::gil_scoped_release rel;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1)
    for (int q = 0; q < k; ++q) {
      std::vector<u64> v(C, 0);
      v[fr[q]] = 1;
      for (int a = rank - 1; a >= 0; --a) {
        const u64* row = &W[(size_t)a * C];
        u64 s = 0;
        for (int j = piv[a] + 1; j < C; ++j)
          if (row[j] && v[j]) {
            s += md.mul(row[j], v[j]);
            if (s >= md.p) s -= md.p;
          }
        v[piv[a]] = s ? md.p - s : 0;
      }
      for (int j = 0; j < C; ++j) B[(size_t)q * C + j] = (i64)v[j];
    }
  }
  py::dict out;
  out["rank"] = rank;
  out["pivots"] = py::cast(piv);
  out["free"] = py::cast(fr);
  out["basis"] = basis;
  return out;
}

// Truncated Taylor series of monomials: S holds nb series of K coefficients, mons the exponent vectors; row t of the
// result is the series of prod_v S_v^{mons[t, v]} modulo t^K.
py::array_t<i64> series_monomials(py::array_t<i64, py::array::c_style | py::array::forcecast> S,
                                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> mons, double pIn,
                                  int threads) {
  check_prime(pIn);
  Mod md((u64)pIn);
  const int nb = (int)S.shape(0), K = (int)S.shape(1), T = (int)mons.shape(0);
  if (mons.shape(1) != nb) throw std::runtime_error("monomials and series differ in the number of variables");
  const int32_t* E = mons.data();
  std::vector<u64> base((size_t)nb * K);
  for (size_t i = 0; i < base.size(); ++i) base[i] = residue(S.data()[i], md.p);
  std::vector<int> top(nb, 0);
  for (int t = 0; t < T; ++t)
    for (int v = 0; v < nb; ++v) top[v] = std::max(top[v], (int)E[t * nb + v]);
  auto mul = [&](const u64* a, const u64* b, u64* c) {
    for (int n = 0; n < K; ++n) {
      u64 s = 0;
      for (int i = 0; i <= n; ++i) s = md.red(s + a[i] * b[n - i]);
      c[n] = s;
    }
  };
  py::array_t<i64> out({T, K});
  i64* R = out.mutable_data();
  {
    py::gil_scoped_release rel;
    // the powers of each series
    std::vector<std::vector<std::vector<u64> > > pw(nb);
    for (int v = 0; v < nb; ++v) {
      pw[v].assign(top[v] + 1, std::vector<u64>(K, 0));
      pw[v][0][0] = 1;
      for (int e = 1; e <= top[v]; ++e) mul(pw[v][e - 1].data(), &base[(size_t)v * K], pw[v][e].data());
    }
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads > 0 ? threads : 1)
    for (int t = 0; t < T; ++t) {
      std::vector<u64> acc(K, 0), tmp(K);
      acc[0] = 1;
      for (int v = 0; v < nb; ++v) {
        int e = E[t * nb + v];
        if (!e) continue;
        mul(acc.data(), pw[v][e].data(), tmp.data());
        acc.swap(tmp);
      }
      for (int n = 0; n < K; ++n) R[(size_t)t * K + n] = (i64)acc[n];
    }
  }
  return out;
}

// The values of the monomials mons (T x nb exponents) at the rows of B (M x nb values) modulo p, M x T.
py::array_t<i64> mono_matrix(py::array_t<i64, py::array::c_style | py::array::forcecast> B,
                             py::array_t<int32_t, py::array::c_style | py::array::forcecast> mons, double pIn,
                             int threads) {
  check_prime(pIn);
  Mod md((u64)pIn);
  const int M = (int)B.shape(0), nb = (int)B.shape(1), T = (int)mons.shape(0);
  if (mons.shape(1) != nb) throw std::runtime_error("monomials and values differ in the number of variables");
  const int32_t* E = mons.data();
  std::vector<int> top(nb, 0);
  for (int t = 0; t < T; ++t)
    for (int v = 0; v < nb; ++v) top[v] = std::max(top[v], (int)E[t * nb + v]);
  std::vector<u64> X((size_t)M * nb);
  for (size_t i = 0; i < X.size(); ++i) X[i] = residue(B.data()[i], md.p);
  py::array_t<i64> out({M, T});
  i64* R = out.mutable_data();
  {
    py::gil_scoped_release rel;
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
    for (int i = 0; i < M; ++i) {
      std::vector<std::vector<u64> > pw(nb);
      for (int v = 0; v < nb; ++v) {
        pw[v].resize(top[v] + 1);
        pw[v][0] = 1;
        for (int e = 1; e <= top[v]; ++e) pw[v][e] = md.mul(pw[v][e - 1], X[(size_t)i * nb + v]);
      }
      for (int t = 0; t < T; ++t) {
        u64 x = 1;
        for (int v = 0; v < nb; ++v)
          if (E[t * nb + v]) x = md.mul(x, pw[v][E[t * nb + v]]);
        R[(size_t)i * T + t] = (i64)x;
      }
    }
  }
  return out;
}

}  // namespace

void register_ioproj(py::module_& m) {
  m.def("io_poly_eval", &poly_eval, py::arg("exps"), py::arg("coef"), py::arg("off"), py::arg("points"),
        py::arg("p"), py::arg("threads") = 1);
  m.def("io_kernel", &kernel, py::arg("A"), py::arg("p"), py::arg("threads") = 1);
  m.def("io_mono_matrix", &mono_matrix, py::arg("B"), py::arg("mons"), py::arg("p"), py::arg("threads") = 1);
  m.def("io_series_monomials", &series_monomials, py::arg("S"), py::arg("mons"), py::arg("p"),
        py::arg("threads") = 1);
}

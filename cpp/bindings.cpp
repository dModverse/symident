// Python bindings of the kernel: arguments are converted to plain C++ values while the GIL
// is held, the kernel runs without it, the result is converted back.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "kernel.h"

namespace py = pybind11;

namespace {

Value to_value(py::handle h);

template <typename T>
Vec<T> vec_from_array(const py::array& a) {
  py::array_t<T, py::array::forcecast> b(a);
  Vec<T> o((int)b.size());
  auto r = b.template unchecked<1>();
  for (py::ssize_t i = 0; i < r.shape(0); ++i) o[(int)i] = r(i);
  return o;
}

template <typename T>
Mat<T> mat_from_array(const py::array& a) {
  py::array_t<T, py::array::forcecast> b(a);
  auto r = b.template unchecked<2>();
  Mat<T> o((int)r.shape(0), (int)r.shape(1));
  for (py::ssize_t j = 0; j < r.shape(1); ++j)
    for (py::ssize_t i = 0; i < r.shape(0); ++i) o((int)i, (int)j) = r(i, j);
  return o;
}

Value from_array(const py::array& a) {
  char k = a.dtype().kind();
  if (k == 'U' || k == 'S' || k == 'O') {
    if (a.ndim() != 1) throw std::runtime_error("string arrays must be one-dimensional");
    CharacterVector o((int)a.size());
    py::list l = a.attr("tolist")();
    for (int i = 0; i < o.size(); ++i) o[i] = py::str(l[(size_t)i]);
    return o;
  }
  bool real = k == 'f';
  if (a.ndim() == 2)
    return real ? Value(mat_from_array<double>(a)) : Value(mat_from_array<int>(a));
  if (a.ndim() == 0) {
    if (real) return Value(a.attr("item")().cast<double>());
    return Value(a.attr("item")().cast<int>());
  }
  return real ? Value(vec_from_array<double>(a)) : Value(vec_from_array<int>(a));
}

Value from_sequence(py::sequence s) {
  size_t n = s.size();
  if (n == 0) return Value();
  bool allStr = true, allInt = true, allNum = true;
  for (auto x : s) {
    bool isBool = py::isinstance<py::bool_>(x);
    bool isInt = py::isinstance<py::int_>(x) && !isBool;
    bool isFloat = py::isinstance<py::float_>(x);
    allStr = allStr && py::isinstance<py::str>(x);
    allInt = allInt && (isInt || isBool);
    allNum = allNum && (isInt || isFloat || isBool);
  }
  if (allStr) {
    CharacterVector o((int)n);
    for (size_t i = 0; i < n; ++i) o[(int)i] = s[i].cast<std::string>();
    return o;
  }
  if (allInt) {
    IntegerVector o((int)n);
    for (size_t i = 0; i < n; ++i) o[(int)i] = s[i].cast<int>();
    return o;
  }
  if (allNum) {
    NumericVector o((int)n);
    for (size_t i = 0; i < n; ++i) o[(int)i] = s[i].cast<double>();
    return o;
  }
  List o((int)n);
  for (size_t i = 0; i < n; ++i) o[(int)i] = to_value(s[i]);
  return o;
}

Value to_value(py::handle h) {
  if (h.is_none()) return Value();
  if (py::isinstance<py::bool_>(h)) return Value(h.cast<bool>());
  if (py::isinstance<py::int_>(h)) return Value(h.cast<int>());
  if (py::isinstance<py::float_>(h)) return Value(h.cast<double>());
  if (py::isinstance<py::str>(h)) return Value(h.cast<std::string>());
  if (py::isinstance<py::array>(h)) return from_array(py::reinterpret_borrow<py::array>(h));
  if (py::isinstance<py::dict>(h)) {
    List o;
    for (auto kv : py::reinterpret_borrow<py::dict>(h))
      o.add(NamedArg{kv.first.cast<std::string>(), to_value(kv.second)});
    return o;
  }
  if (py::isinstance<py::sequence>(h)) return from_sequence(py::reinterpret_borrow<py::sequence>(h));
  if (py::hasattr(h, "dtype")) return from_array(py::array::ensure(h));
  throw std::runtime_error("unsupported argument type");
}

template <typename T>
py::array_t<T> vec_to_py(const Vec<T>& x) {
  py::array_t<T> o(x.size());
  auto w = o.template mutable_unchecked<1>();
  for (int i = 0; i < x.size(); ++i) w(i) = x[i];
  return o;
}

template <typename T>
py::array_t<T> mat_to_py(const Mat<T>& x) {
  py::array_t<T> o({(py::ssize_t)x.nr, (py::ssize_t)x.nc});
  auto w = o.template mutable_unchecked<2>();
  for (int j = 0; j < x.nc; ++j)
    for (int i = 0; i < x.nr; ++i) w(i, j) = x(i, j);
  return o;
}

py::object to_py(const Value& v) {
  struct Visitor {
    py::object operator()(const Nil&) const { return py::none(); }
    py::object operator()(bool x) const { return py::bool_(x); }
    py::object operator()(int x) const { return py::int_(x); }
    py::object operator()(double x) const { return py::float_(x); }
    py::object operator()(const std::string& x) const { return py::str(x); }
    py::object operator()(const IntegerVector& x) const { return vec_to_py(x); }
    py::object operator()(const NumericVector& x) const { return vec_to_py(x); }
    py::object operator()(const CharacterVector& x) const {
      py::list o;
      for (int i = 0; i < x.size(); ++i) o.append(py::str(x[i]));
      return o;
    }
    py::object operator()(const IntegerMatrix& x) const { return mat_to_py(x); }
    py::object operator()(const NumericMatrix& x) const { return mat_to_py(x); }
    py::object operator()(const std::shared_ptr<List>& l) const {
      if (!l->names.empty() && l->names.size() == l->items.size()) {
        py::dict o;
        for (size_t i = 0; i < l->items.size(); ++i) o[py::str(l->names[i])] = to_py(l->items[i]);
        return o;
      }
      py::list o;
      for (const Value& x : l->items) o.append(to_py(x));
      return o;
    }
  };
  return std::visit(Visitor(), v.d);
}

template <typename T>
T cv(py::handle h) { return (T)to_value(h); }

template <typename F>
py::object run(F f) {
  Value out;
  {
    py::gil_scoped_release nogil;
    out = f();
  }
  return to_py(out);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "GF(p) observability kernel";
#if defined(__SIZEOF_INT128__) && !defined(SYMIDENT_NO_INT128)
  m.attr("has_int128") = true;
#else
  m.attr("has_int128") = false;
#endif
  using py::arg;

  m.def("sym_obs_null_multi", [](py::object tapes, int nLeaves, int nStates, py::object zSlots,
                                 py::object point, double p, int Nt, int cores) {
    List t = cv<List>(tapes); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    return run([&] { return Value(symObsNullMulti(t, nLeaves, nStates, z, pt, p, Nt, cores)); });
  }, arg("tapes"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("cores") = 1);

  m.def("sym_obs_rank_profile", [](py::object tapes, int nLeaves, int nStates, py::object zSlots,
                                   py::object point, double p, int Nt, int cores) {
    List t = cv<List>(tapes); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    return run([&] { return Value(symObsRankProfile(t, nLeaves, nStates, z, pt, p, Nt, cores)); });
  }, arg("tapes"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("cores") = 1);

  m.def("sym_obs_chain_rank_profile", [](py::object chain, int nLeaves, int nStates,
                                         py::object zSlots, py::object point, double p, int Nt,
                                         int threads) {
    List c = cv<List>(chain); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    return run([&] { return Value(symObsChainRankProfile(c, nLeaves, nStates, z, pt, p, Nt, threads)); });
  }, arg("chain"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("threads") = 1);

  m.def("sym_obs_directional", [](py::object tapes, int nLeaves, int nStates, py::object zSlots,
                                  py::object point, double p, int Nt, py::object dir, int cores) {
    List t = cv<List>(tapes); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    NumericVector d = cv<NumericVector>(dir);
    return run([&] { return Value(symObsDirectional(t, nLeaves, nStates, z, pt, p, Nt, d, cores)); });
  }, arg("tapes"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("dir"), arg("cores") = 1);

  m.def("sym_obs_null_batch", [](py::object tapes, int nLeaves, int nStates, py::object zSlots,
                                 py::object points, py::object primes, int Nt, int cores) {
    List t = cv<List>(tapes); IntegerVector z = cv<IntegerVector>(zSlots);
    IntegerMatrix pts = cv<IntegerMatrix>(points); NumericVector pr = cv<NumericVector>(primes);
    return run([&] { return Value(symObsNullBatch(t, nLeaves, nStates, z, pts, pr, Nt, cores)); });
  }, arg("tapes"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("points"), arg("primes"),
     arg("Nt"), arg("cores") = 1);

  m.def("sym_obs_null_chain", [](py::object chains, int nLeaves, int nStates, py::object zSlots,
                                 py::object point, double p, int Nt, int Mtot, int cores,
                                 py::object NtChain, py::object pointSer) {
    List c = cv<List>(chains); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    IntegerVector nc = cv<IntegerVector>(NtChain); Value ps = to_value(pointSer);
    return run([&] { return Value(symObsNullChain(c, nLeaves, nStates, z, pt, p, Nt, Mtot, cores, nc, ps)); });
  }, arg("chains"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("Mtot"), arg("cores") = 1, arg("nt_chain") = py::none(),
     arg("point_ser") = py::none());

  m.def("sym_segment_ranks", [](py::object segs, int nLeaves, int nStates, py::object zSlots,
                                py::object point, double p, int Nt, py::object stateVals,
                                int threads) {
    List s = cv<List>(segs); IntegerVector z = cv<IntegerVector>(zSlots), pt = cv<IntegerVector>(point);
    IntegerMatrix sv = stateVals.is_none() ? IntegerMatrix(0, 0) : cv<IntegerMatrix>(stateVals);
    return run([&] { return Value(symSegmentRanks(s, nLeaves, nStates, z, pt, p, Nt, sv, threads)); });
  }, arg("segs"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("point"), arg("p_in"),
     arg("Nt"), arg("state_vals") = py::none(), arg("threads") = 1);

  m.def("sym_obs_null_chain_seed_batch", [](py::object chains, py::object evalChain, py::object seeds,
                                            py::object primes, int nLeaves, int nStates,
                                            py::object zSlots, int Nt, int Mtot, int cores,
                                            py::object NtEval, py::object seedSer) {
    List c = cv<List>(chains); IntegerVector ec = cv<IntegerVector>(evalChain);
    IntegerMatrix sd = cv<IntegerMatrix>(seeds); NumericVector pr = cv<NumericVector>(primes);
    IntegerVector z = cv<IntegerVector>(zSlots), ne = cv<IntegerVector>(NtEval);
    Value ss = to_value(seedSer);
    return run([&] { return Value(symObsNullChainSeedBatch(c, ec, sd, pr, nLeaves, nStates, z, Nt, Mtot, cores, ne, ss)); });
  }, arg("chains"), arg("eval_chain"), arg("seeds"), arg("primes"), arg("n_leaves"),
     arg("n_states"), arg("z_slots"), arg("Nt"), arg("Mtot"), arg("cores") = 1,
     arg("nt_eval") = py::none(), arg("seed_ser") = py::none());

  m.def("sym_obs_null_chain_point_batch", [](py::object chains, int nLeaves, int nStates,
                                             py::object zSlots, py::object points, py::object primes,
                                             int Nt, int Mtot, int cores, py::object NtChain) {
    List c = cv<List>(chains); IntegerVector z = cv<IntegerVector>(zSlots);
    IntegerMatrix pts = cv<IntegerMatrix>(points); NumericVector pr = cv<NumericVector>(primes);
    IntegerVector nc = cv<IntegerVector>(NtChain);
    return run([&] { return Value(symObsNullChainPointBatch(c, nLeaves, nStates, z, pts, pr, Nt, Mtot, cores, nc)); });
  }, arg("chains"), arg("n_leaves"), arg("n_states"), arg("z_slots"), arg("points"), arg("primes"),
     arg("Nt"), arg("Mtot"), arg("cores") = 1, arg("nt_chain") = py::none());

  m.def("sym_series_rank", [](py::object S, int nz, int N, double p, py::object cols, bool support,
                              int atOneBelow, int cores, py::object rowPrec) {
    IntegerMatrix s = cv<IntegerMatrix>(S); IntegerVector c = cv<IntegerVector>(cols);
    IntegerVector rp = cv<IntegerVector>(rowPrec);
    return run([&] { return Value(symSeriesRank(s, nz, N, p, c, support, atOneBelow, cores, rp)); });
  }, arg("S"), arg("nz"), arg("N"), arg("p_in"), arg("cols"), arg("support") = false,
     arg("at_one_below") = -1, arg("cores") = 1, arg("row_prec") = py::none());

  m.def("sym_series_project", [](py::object S, int nz, int N, double p, py::object first, int cores) {
    IntegerMatrix s = cv<IntegerMatrix>(S); IntegerVector f = cv<IntegerVector>(first);
    return run([&] { return Value(symSeriesProject(s, nz, N, p, f, cores)); });
  }, arg("S"), arg("nz"), arg("N"), arg("p_in"), arg("first"), arg("cores") = 1);

  m.def("sym_solve_mod", [](py::object A, py::object b, double p) {
    IntegerMatrix a = cv<IntegerMatrix>(A); IntegerVector bb = cv<IntegerVector>(b);
    return run([&] { return symSolveMod(a, bb, p); });
  }, arg("A"), arg("b"), arg("p_in"));

  m.def("sym_rref_mod", [](py::object M, double p) {
    NumericMatrix mm = cv<NumericMatrix>(M);
    return run([&] { return Value(symRrefMod(mm, p)); });
  }, arg("M"), arg("p_in"));

  m.def("sym_gauge_scores", [](py::object F, py::object piv, double p, py::object cand) {
    NumericMatrix f = cv<NumericMatrix>(F); IntegerVector pv = cv<IntegerVector>(piv), c = cv<IntegerVector>(cand);
    return run([&] { return Value(symGaugeScores(f, pv, p, c)); });
  }, arg("F"), arg("piv"), arg("p_in"), arg("cand"));

  m.def("sym_gauge_fix", [](py::object F, py::object piv, double p, int k) {
    NumericMatrix f = cv<NumericMatrix>(F); IntegerVector pv = cv<IntegerVector>(piv);
    return run([&] { return Value(symGaugeFix(f, pv, p, k)); });
  }, arg("F"), arg("piv"), arg("p_in"), arg("k"));

  m.def("sym_fit_rational", [](py::object sampleU, py::object mons, py::object rvals, double p) {
    IntegerMatrix su = cv<IntegerMatrix>(sampleU), mo = cv<IntegerMatrix>(mons);
    IntegerVector rv = cv<IntegerVector>(rvals);
    return run([&] { return Value(symFitRational(su, mo, rv, p)); });
  }, arg("sample_u"), arg("mons"), arg("rvals"), arg("p_in"));

  m.def("sym_rat_recon", [](py::object residues, py::object primes) {
    IntegerMatrix r = cv<IntegerMatrix>(residues); IntegerVector pr = cv<IntegerVector>(primes);
    return run([&] { return Value(symRatRecon(r, pr)); });
  }, arg("residues"), arg("primes"));

  m.def("sym_sparse_poly", [](py::object seq, py::object monoTab, py::object monoRes, double p) {
    IntegerVector s = cv<IntegerVector>(seq), mr = cv<IntegerVector>(monoRes);
    IntegerMatrix mt = cv<IntegerMatrix>(monoTab);
    return run([&] { return Value(symSparsePoly(s, mt, mr, p)); });
  }, arg("seq"), arg("mono_tab"), arg("mono_res"), arg("p_in"));

  m.def("sym_mono_residues", [](py::object expts, py::object bases, double p) {
    IntegerMatrix e = cv<IntegerMatrix>(expts); IntegerVector b = cv<IntegerVector>(bases);
    return run([&] { return Value(symMonoResidues(e, b, p)); });
  }, arg("expts"), arg("bases"), arg("p_in"));

  m.def("sym_b_morder", [](py::object seq, double p) {
    IntegerVector s = cv<IntegerVector>(seq);
    return run([&] { return Value(symBMorder(s, p)); });
  }, arg("seq"), arg("p_in"));

  m.def("sym_cauchy_eval", [](py::object tnodes, py::object rvals, int dN, int dD, double p) {
    IntegerVector t = cv<IntegerVector>(tnodes), r = cv<IntegerVector>(rvals);
    return run([&] { return Value(symCauchyEval(t, r, dN, dD, p)); });
  }, arg("tnodes"), arg("rvals"), arg("d_n"), arg("d_d"), arg("p_in"));
}

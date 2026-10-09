// Plain C++ value types with the subset of the Rcpp interface the kernel uses: vectors,
// column-major matrices and named lists. They hold no Python object, so the kernel runs
// without the GIL; bindings.cpp converts at the boundary.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Rcpp {

template <typename T>
struct Vec {
  std::vector<T> v;
  Vec() {}
  explicit Vec(int n) : v((size_t)(n > 0 ? n : 0), T()) {}
  template <typename It>
  Vec(It first, It last) : v(first, last) {}
  static Vec create() { return Vec(); }
  int size() const { return (int)v.size(); }
  T& operator[](int i) { return v[(size_t)i]; }
  const T& operator[](int i) const { return v[(size_t)i]; }
  typename std::vector<T>::iterator begin() { return v.begin(); }
  typename std::vector<T>::iterator end() { return v.end(); }
  typename std::vector<T>::const_iterator begin() const { return v.begin(); }
  typename std::vector<T>::const_iterator end() const { return v.end(); }
};

template <typename T>
struct Mat {
  int nr = 0, nc = 0;
  std::vector<T> v;                       // column-major
  Mat() {}
  Mat(int r, int c) : nr(r), nc(c), v((size_t)r * (size_t)c, T()) {}
  int nrow() const { return nr; }
  int ncol() const { return nc; }
  T& operator()(int i, int j) { return v[(size_t)j * nr + i]; }
  const T& operator()(int i, int j) const { return v[(size_t)j * nr + i]; }
};

typedef Vec<int> IntegerVector;
typedef Vec<double> NumericVector;
typedef Vec<int> LogicalVector;
typedef Vec<std::string> CharacterVector;
typedef Mat<int> IntegerMatrix;
typedef Mat<double> NumericMatrix;

struct List;
struct Nil {};

struct Value {
  typedef std::variant<Nil, bool, int, double, std::string, IntegerVector, NumericVector,
                       CharacterVector, IntegerMatrix, NumericMatrix, std::shared_ptr<List> >
      Data;
  Data d;
  Value() : d(Nil()) {}
  Value(bool x) : d(x) {}
  Value(int x) : d(x) {}
  Value(double x) : d(x) {}
  Value(const char* x) : d(std::string(x)) {}
  Value(const std::string& x) : d(x) {}
  Value(const IntegerVector& x) : d(x) {}
  Value(const NumericVector& x) : d(x) {}
  Value(const CharacterVector& x) : d(x) {}
  Value(const IntegerMatrix& x) : d(x) {}
  Value(const NumericMatrix& x) : d(x) {}
  Value(const List& x);
  bool isNull() const { return std::holds_alternative<Nil>(d); }
  operator IntegerVector() const;
  operator NumericVector() const;
  operator CharacterVector() const;
  operator IntegerMatrix() const;
  operator NumericMatrix() const;
  operator List() const;
  explicit operator int() const;
  explicit operator double() const;
};

typedef Value SEXP;
static const Value R_NilValue;
inline bool Rf_isNull(const Value& x) { return x.isNull(); }

struct NamedArg {
  std::string name;
  Value value;
};

struct Namer {
  std::string name;
  template <typename T>
  NamedArg operator=(const T& x) const { return NamedArg{name, Value(x)}; }
};

struct NameMaker {
  Namer operator[](const char* s) const { return Namer{s}; }
};
static const NameMaker _;

struct List {
  std::vector<Value> items;
  std::vector<std::string> names;          // empty when unnamed
  List() {}
  explicit List(int n) : items((size_t)(n > 0 ? n : 0)) {}
  int size() const { return (int)items.size(); }
  Value& operator[](int i) { return items[(size_t)i]; }
  const Value& operator[](int i) const { return items[(size_t)i]; }
  int find(const std::string& s) const {
    for (size_t i = 0; i < names.size(); ++i) if (names[i] == s) return (int)i;
    return -1;
  }
  bool containsElementNamed(const char* s) const { return find(s) >= 0; }
  Value& operator[](const std::string& s) {
    int i = find(s);
    if (i >= 0) return items[(size_t)i];
    if (names.size() < items.size()) names.resize(items.size());
    names.push_back(s);
    items.push_back(Value());
    return items.back();
  }
  Value& operator[](const char* s) { return (*this)[std::string(s)]; }
  const Value& operator[](const char* s) const {
    int i = find(s);
    if (i < 0) throw std::runtime_error(std::string("missing list element ") + s);
    return items[(size_t)i];
  }
  template <typename... A>
  static List create(const A&... a) {
    List out;
    (out.add(a), ...);
    return out;
  }
  void add(const NamedArg& a) {
    if (names.size() < items.size()) names.resize(items.size());
    names.push_back(a.name);
    items.push_back(a.value);
  }
};

inline Value::Value(const List& x) : d(std::make_shared<List>(x)) {}

inline Value::operator List() const {
  if (auto p = std::get_if<std::shared_ptr<List> >(&d)) return **p;
  if (isNull()) return List();
  throw std::runtime_error("not a list");
}

template <typename T, typename S>
Vec<T> vec_cast(const Vec<S>& x) {
  Vec<T> o(x.size());
  for (int i = 0; i < x.size(); ++i) o[i] = (T)x[i];
  return o;
}

inline Value::operator IntegerVector() const {
  if (auto p = std::get_if<IntegerVector>(&d)) return *p;
  if (auto p = std::get_if<NumericVector>(&d)) return vec_cast<int>(*p);
  if (auto p = std::get_if<int>(&d)) { IntegerVector o(1); o[0] = *p; return o; }
  if (auto p = std::get_if<double>(&d)) { IntegerVector o(1); o[0] = (int)*p; return o; }
  if (auto p = std::get_if<bool>(&d)) { IntegerVector o(1); o[0] = *p; return o; }
  if (isNull()) return IntegerVector();
  throw std::runtime_error("not an integer vector");
}

inline Value::operator NumericVector() const {
  if (auto p = std::get_if<NumericVector>(&d)) return *p;
  if (auto p = std::get_if<IntegerVector>(&d)) return vec_cast<double>(*p);
  if (auto p = std::get_if<int>(&d)) { NumericVector o(1); o[0] = *p; return o; }
  if (auto p = std::get_if<double>(&d)) { NumericVector o(1); o[0] = *p; return o; }
  if (isNull()) return NumericVector();
  throw std::runtime_error("not a numeric vector");
}

inline Value::operator CharacterVector() const {
  if (auto p = std::get_if<CharacterVector>(&d)) return *p;
  if (auto p = std::get_if<std::string>(&d)) { CharacterVector o(1); o[0] = *p; return o; }
  if (isNull()) return CharacterVector();
  throw std::runtime_error("not a character vector");
}

inline Value::operator IntegerMatrix() const {
  if (auto p = std::get_if<IntegerMatrix>(&d)) return *p;
  if (auto p = std::get_if<NumericMatrix>(&d)) {
    IntegerMatrix o(p->nr, p->nc);
    for (size_t i = 0; i < o.v.size(); ++i) o.v[i] = (int)p->v[i];
    return o;
  }
  throw std::runtime_error("not an integer matrix");
}

inline Value::operator NumericMatrix() const {
  if (auto p = std::get_if<NumericMatrix>(&d)) return *p;
  if (auto p = std::get_if<IntegerMatrix>(&d)) {
    NumericMatrix o(p->nr, p->nc);
    for (size_t i = 0; i < o.v.size(); ++i) o.v[i] = p->v[i];
    return o;
  }
  throw std::runtime_error("not a numeric matrix");
}

inline Value::operator int() const {
  if (auto p = std::get_if<int>(&d)) return *p;
  if (auto p = std::get_if<double>(&d)) return (int)*p;
  if (auto p = std::get_if<bool>(&d)) return *p;
  if (auto p = std::get_if<IntegerVector>(&d)) if (p->size() == 1) return (*p)[0];
  if (auto p = std::get_if<NumericVector>(&d)) if (p->size() == 1) return (int)(*p)[0];
  throw std::runtime_error("not an integer scalar");
}

inline Value::operator double() const {
  if (auto p = std::get_if<double>(&d)) return *p;
  return (double)(int)*this;
}

template <typename T>
T as(const Value& x) { return (T)x; }

template <>
inline std::string as<std::string>(const Value& x) {
  if (auto p = std::get_if<std::string>(&x.d)) return *p;
  throw std::runtime_error("not a string");
}

inline std::string as_string(const std::string& s) { return s; }

[[noreturn]] inline void stop(const std::string& msg) { throw std::runtime_error(msg); }

}  // namespace Rcpp

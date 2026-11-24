// Entry points of the GF(p) observability kernel (kernel.cpp).

#pragma once

#include "rvalue.h"

using namespace Rcpp;

List symObsNullMulti(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int cores);
List symObsAnnihilates(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                       double pIn, int Nt, IntegerMatrix V, int cores);
SEXP symObsAnnihilatesBatch(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerMatrix points,
                            NumericVector primes, int Nt, IntegerMatrix V, int cores);
List symObsCover(List tapes, int nLeaves, int nStates, IntegerVector zSlots, IntegerVector point,
                 double pIn, int Nt, int target, int cores, IntegerVector order, LogicalVector keep);
List symObsRankProfile(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                       IntegerVector point, double pIn, int Nt, int cores);
List symObsChainRankProfile(List chain, int nLeaves, int nStates, IntegerVector zSlots,
                            IntegerVector point, double pIn, int Nt, int threads);
List symObsDirectional(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                       IntegerVector point, double pIn, int Nt, NumericVector dir, int cores);
List symObsNullBatch(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerMatrix points, NumericVector primes, int Nt, int cores);
List symObsNullChain(List chains, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int Mtot, int cores,
                     IntegerVector NtChain, SEXP pointSer, int serOrder);
List symObsChainRows(List chain, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int Mtot, int threads,
                     bool profile, std::string session);
List symChainReduce(IntegerMatrix rowsIn, IntegerMatrix srowsIn, int nz, int N, double pIn);
double symSetBlockBytes(double bytes);
int symClearJets();
List symSegmentRanks(List segs, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, IntegerMatrix stateVals,
                     int threads, IntegerVector dropStates, bool fromStates, std::string session);
List symObsNullChainSeedBatch(List chains, IntegerVector evalChain, IntegerMatrix seeds,
                              NumericVector primes, int nLeaves, int nStates,
                              IntegerVector zSlots, int Nt, int Mtot, int cores,
                              IntegerVector NtEval, SEXP seedSer);
List symObsNullChainPointBatch(List chains, int nLeaves, int nStates, IntegerVector zSlots,
                               IntegerMatrix points, NumericVector primes, int Nt, int Mtot,
                               int cores, IntegerVector NtChain);
List symSeriesRank(IntegerMatrix S, int nz, int N, double pIn, IntegerVector cols,
                   bool support, int atOneBelow, int cores, IntegerVector rowPrec);
List symSeriesProject(IntegerMatrix S, int nz, int N, double pIn, IntegerVector first,
                      int cores);
SEXP symSolveMod(IntegerMatrix A, IntegerVector b, double pIn);
List symRrefMod(NumericMatrix M, double pIn);
List symIndependentRows(NumericMatrix M, double pIn, int cores);
NumericMatrix symMatMulMod(NumericMatrix A, NumericMatrix B, double pIn, int cores);
NumericVector symGaugeScores(NumericMatrix F, IntegerVector piv, double pIn,
                             IntegerVector cand);
List symGaugeFix(NumericMatrix F, IntegerVector piv, double pIn, int k);
List symFitRational(IntegerMatrix sampleU, IntegerMatrix mons, IntegerVector rvals,
                    double pIn);
List symRatRecon(IntegerMatrix residues, IntegerVector primes);
List symSparsePoly(IntegerVector seq, IntegerMatrix monoTab, IntegerVector monoRes,
                   double pIn);
IntegerVector symMonoResidues(IntegerMatrix expts, IntegerVector bases, double pIn);
int symBMorder(IntegerVector seq, double pIn);
List symCauchyEval(IntegerVector tnodes, IntegerVector rvals, int dN, int dD, double pIn);
List symFiberReduce(IntegerVector ps, IntegerVector pc, IntegerVector pt, IntegerVector pv,
                    IntegerVector pe, IntegerVector ss, IntegerVector sc, IntegerVector st,
                    IntegerVector sv, IntegerVector se, int nv, int m0, double pIn);

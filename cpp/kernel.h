// Entry points of the GF(p) observability kernel (kernel.cpp).

#pragma once

#include "rvalue.h"

using namespace Rcpp;

List symObsNullMulti(List tapes, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, int cores);
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
                     IntegerVector NtChain, SEXP pointSer);
List symSegmentRanks(List segs, int nLeaves, int nStates, IntegerVector zSlots,
                     IntegerVector point, double pIn, int Nt, IntegerMatrix stateVals,
                     int threads);
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

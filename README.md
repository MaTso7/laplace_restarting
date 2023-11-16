# Restarted Arnoldi method for Laplace transforms and complete Bernstein functions
This repository contains an implementation of the restarted Arnoldi method for Laplace transforms and complete Bernstein functions based on [[1](https://doi.org/10.1137/22M1499674)], [[2](https://doi.org/10.25926/BUW/0-106)].
In particular, the function `laplace_restarting` computes an approximation to F(A)*b, where F is a Laplace transform or a complete Bernstein function. 

## Usage
In addition to A, b and a target accuracy, the user needs to provide the function handle f which is
- the inverse Laplace transform of F (if F is a Laplace transform) or
- the density (wrt Lebesgue measure) of the Lévy measure in the Lévy–Khintchine representation of F (if F is a Bernstein function).

For detailed information on how to use `laplace_restarting`, see its function header and the examples below. Note that MATLAB R2019b or later is required.

## Examples
The folder `experiments` contains several exemplary scripts:
- The scripts in `experiments/comparisons` contains the examples presented in [1],[2]. For Hermitian A, they also compare the performance of `laplace_restarting` to the two-pass Lanczos method.
- The script in `experiments/error_bounds` demonstrates the computation of a posteriori error bounds as in [2], [3].

The examples need access to the files in `solutions`, which need to be downloaded using `git annex get` or manually via
https://uni-wuppertal.sciebo.de/s/3lltZM3WdNrYCLT .

## How to cite
Please use the original paper [1] to cite this package.

## References
[1] A. Frommer, K. Kahl, M. Schweitzer, and M. Tsolakis: *Krylov subspace restarting for matrix Laplace transforms*, SIAM J. Matrix Anal. Appl., 44 (2023), pp. 693–717, doi: [10.1137/22M1499674](https://doi.org/10.1137/22M1499674)

[2] M. Tsolakis: *Efficient Computation of the Action of Matrix Rational Functions and Laplace transforms* (Doctoral Thesis), Bergische Universität Wuppertal, Germany, 2023, doi: [10.25926/BUW/0-106](https://doi.org/10.25926/BUW/0-106)

[3] A. Frommer, and M. Schweitzer: *Error bounds and estimates for Krylov subspace approximations of Stieltjes matrix functions*, BIT, 56 (2016), pp. 865–892, doi: [10.1007/s10543-015-0596-3](https://doi.org/10.1007/s10543-015-0596-3)

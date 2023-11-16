%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 6:
% A: 3D-Laplacian
% F(A)b: A^{-3/2}b (a Laplace transform)
%
% Compute the a posteriori error bounds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Problem setup and parameter choice %%%
f = @(t) 2*sqrt(t/pi);
F = @(z) z.^-(3/2);
tol = 1e-7;
k = 20;
max_cycles = 200;
N_vec = 20:10:100;

%%% Construct discretized Laplacian %%%
i = 4;
N = N_vec(i);
e=ones(N,1);
A1 = spdiags([-e 2*e -e], [-1 0 1], N, N);
I1 = eye(N);
A = kron(A1,kron(I1,I1)) + kron(I1, kron(A1,I1)) + kron(I1, kron(I1,A1));

%%% precomputed solution and rhs %%%
load('../../solutions/laplace3d_sol.mat');
b = b_c{i};
xtrue = xtrue_c{i};


%%% Run Laplace restarting %%%
clear options
options.F = F;
options.restart_length = k;
options.xtrue = xtrue;
options.max_cycles = max_cycles;
options.int_prec = 1e-3*tol;
options.compute_bounds = true;

use_exact_lmin = true; % switch for different versions of experiment

if use_exact_lmin
    lmin = 12*sin(pi/(2*(N+1)))^2;
    options.lmin = lmin;
    nvp = namedargs2cell(options);
    [x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
else
    % can play around with safety factor.
    % if safety factor is too large, estimate might fail to be an upper bound.
    % If safety factor is too small, error might be severly overestimated.
    options.safety_factor = 1e-2;
    nvp = namedargs2cell(options);
    [x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
end

semilogy(out_L.err)
hold all
semilogy(out_L.lbound/norm(xtrue),'--')
semilogy(out_L.ubound/norm(xtrue),'--')
hold off
legend('error','lower bound','upper bound')
grid on
xlabel('cycle')
ylabel('relative error norm')

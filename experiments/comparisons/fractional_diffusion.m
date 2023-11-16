%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2 (fractional diffusion):
% A: Graph Laplacians
% F(A)b: expm(-sqrtm(A))b
%
% Compare Laplace restarting to two-pass
% Lanczos
%
% Vector b is explicitly chosen such that it does
% not contain a component from null(A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Problem setup %%%
f = @(t) (exp(-1./(4*t)))./(2* sqrt(pi) * t.^(3/2));
F = @(z)  exp(-sqrt(z));

%%% Load precomputed solutions %%%
load('../../solutions/fractional_diffusion/usroads-48_sol.mat'); name = "usroads-48";
% load('../../solutions/fractional_diffusion/loc-Gowalla_sol.mat'); name = "loc-Gowalla";
% load('../../solutions/fractional_diffusion/dblp-2010_sol.mat'); name = "dblp-2010";
% load('../../solutions/fractional_diffusion/com-Amazon_sol.mat'); name = "com-Amazon";

N = size(A,1);
fprintf("%s, N = %d: \n\n", name, N);

%%% Parameter choice %%%
max_cycles = 1000;
tol = 1e-7;
k = 50;
m = 4000; %for two-pass Lanczos


%%% Run Laplace restarting %%%
clear options
options.F = F;
options.restart_length = k;
options.xtrue = xtrue;
options.max_cycles = max_cycles;
options.int_prec = 1e-3*tol;
nvp = namedargs2cell(options);
tic
[x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
time_laplace=toc;
err_norm_laplace = norm(x_laplace-xtrue)/norm(xtrue);
num_matvecs_laplace = k*length(out_L.err);
fprintf('Laplace finished.\n\n');

%%% Run two-pass Lanczos %%%
tic
[x_2pl, err2pl, mv2pl] = twopass_lanczos(A, b, m, @(X) expm(-sqrtm(full(X))), xtrue, tol, k);
time_2pl = toc;
num_matvecs_2pl = mv2pl;
err_norm_2pl = err2pl(end);
fprintf("2PL finished.\n");

%%% Print results %%%
fprintf("Number of matrix vector products\n")
fprintf("--------------------------------\n")
fprintf("2PL:     %d\n", mv2pl)
fprintf("Laplace: %d\n\n", num_matvecs_laplace)

fprintf("Wall clock time\n")
fprintf("--------------------------------\n")
fprintf("2PL:     %4.2f\n", time_2pl)
fprintf("Laplace: %4.2f\n\n", time_laplace)

fprintf("Accuracy\n")
fprintf("--------------------------------\n")
fprintf("2PL:     %e\n", err_norm_2pl)
fprintf("Laplace: %e\n\n", err_norm_laplace)

%%% Plot convergence curves %%%

figure()
colororder([0 0.4470 0.7410; 0.1719 0.625 0.1719; 0 0 0]);
semilogy((1:out_L.cycles)*k, out_L.err,'-')
hold all
semilogy((1:length(err2pl))*2*k, err2pl,'-.')
semilogy([1, max([out_L.cycles, 2*length(err2pl)])*k],[tol tol],'k--')
hold off
grid on
xlabel('matvecs')
ylabel('relative error norm')
title(sprintf('Convergence for %s (N = %d)', name, N))
legend('Laplace restarting','Two-pass Lanczos','target accuracy','Location','NorthEast')

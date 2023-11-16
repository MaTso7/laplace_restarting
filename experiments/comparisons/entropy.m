%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 5:
% A: Graph Laplacians
% F(A)b: log(A)c (with c=Ab)
% 
% Compare Laplace restarting to two-pass 
% Lanczos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Load precomputed solutions %%%
load('../../solutions/entropy/usroads-48_sol.mat'); name = "usroads-48";
% load('../../solutions/entropy/loc-Gowalla_sol.mat'); name = "loc-Gowalla";
% load('../../solutions/entropy/dblp-2010_sol.mat'); name = "dblp-2010";
% load('../../solutions/entropy/com-Amazon_sol.mat'); name = "com-Amazon";


%%% Problem setup and parameter choice %%%
tol = 1e-7;
k = 50;
max_cycles = 10;
f = @(t) -1./(t+1e-10);
F = @(z) log(z);
N = size(A,1);
fprintf("%s, N = %d: \n\n", name, N);


%%% Run Laplace restarting %%%
clear options
options.F = F;
options.restart_length = k;
options.xtrue = xtrue;
options.max_cycles = max_cycles;
options.int_prec = 1e-3*tol;
nvp = namedargs2cell(options);
tic
[x_laplace, out_L] = laplace_restarting(A, A*b, f, tol, nvp{:});
time_laplace=toc;
err_norm_laplace = norm(x_laplace-xtrue)/norm(xtrue);
num_matvecs_laplace = k*length(out_L.err);
fprintf('Laplace finished.\n\n');

%%% Run two-pass Lanczos %%%
tic
m = 2000;
[x_2pl, err2pl, mv2pl] = twopass_lanczos(A, A*b, m, @(X) logm(X), xtrue, tol, k);
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
semilogy((1:out_L.cycles)*k, out_L.err,'-o')
hold all
semilogy((1:length(err2pl))*2*k, err2pl,'-.^')
semilogy([1, max([out_L.cycles, 2*length(err2pl)])*k],[tol tol],'k--')
hold off
grid on
xlabel('matvecs')
ylabel('relative error norm')
title(sprintf('Convergence for %s (N = %d)', name, N))
legend('Laplace restarting','Two-pass Lanczos','target accuracy','Location','NorthEast')

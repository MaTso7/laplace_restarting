%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4a:
% A: 3D-Laplacian
% F(A)b: A^{1/2}b
% 
% Compare Laplace restarting to two-pass 
% Lanczos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Load precomputed solutions %%%
load('../../solutions/sqrt/laplace3d_sol.mat');

%%% Problem setup and parameter choice %%%
f = @(t) t.^(-3/2)/(2*sqrt(pi));
F = @(z) z.^(1/2);
tol = 1e-7;
k = 50;
max_cycles = 100;

time_laplace = zeros(1,length(N_vec));
time_2pl = zeros(1,length(N_vec));
num_matvecs_laplace = zeros(1,length(N_vec));
num_matvecs_2pl = zeros(1,length(N_vec));
err_norm_laplace = zeros(1,length(N_vec));
err_norm_2pl = zeros(1,length(N_vec));


for i = 1:length(N_vec)

    %%% Construct discretized Laplacian %%%
    N = N_vec(i);
    fprintf("N = %d: ", N_vec(i));
    e=ones(N,1);
    A1 = spdiags([-e 2*e -e], [-1 0 1], N, N);
    I1 = eye(N);
    A = kron(A1,kron(I1,I1)) + kron(I1, kron(A1,I1)) + kron(I1, kron(I1,A1));

    %%% precomputed solution and rhs %%%
    b = ones(N*N*N,1); b = b/norm(b);
    xtrue = xtrue_c{i};


    %%% Run Laplace restarting %%%
    clear options
    options.FunctionClass = "Bernstein";
    options.F = F;
    options.restart_length = k;
    options.xtrue = xtrue;
    options.max_cycles = max_cycles;
    options.int_prec = 1e-3*tol;
    nvp = namedargs2cell(options);
    tic
    [x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
    time_laplace(i) = toc;
    num_matvecs_laplace(i) = k*length(out_L.err);
    err_norm_laplace(i) = norm(x_laplace-xtrue)/norm(xtrue);
    fprintf("Laplace finished. ");
    
    %%% Run two-pass Lanczos %%%
    tic
    m = 1000;
    [x_2pl, err2pl, mv2pl] = twopass_lanczos(A, b, m, @(X) sqrtm(X), xtrue, tol, k);
    time_2pl(i) = toc;
    num_matvecs_2pl(i) = mv2pl;
    err_norm_2pl(i) = err2pl(end);
    fprintf("2PL finished.\n");
end

% Plot some informative stuff
set(groot, "defaultaxescolororder", [0 0.4470 0.7410; 0.1719 0.625 0.1719; 0 0 0])

figure();
plot(N_vec, num_matvecs_laplace,'-o');
hold all
plot(N_vec, num_matvecs_2pl,'-.^');
hold off
grid on
xlabel('N')
ylabel('matvecs')
title('Number of matrix-vector products')
legend('Laplace restarting','Two-pass Lanczos','Location','NorthWest')

figure();
plot(N_vec.^3, time_laplace,'-o');
hold all
plot(N_vec.^3, time_2pl,'-.^');
hold off
grid on
xlabel('N^3')
ylabel('time [s]')
title('Wall-clock time')
legend('Laplace restarting','Two-pass Lanczos','Location','NorthWest')

figure();
semilogy(N_vec, err_norm_laplace, 'o');
hold all
semilogy(N_vec, err_norm_2pl,'^');
semilogy([N_vec(1)-8,N_vec(end)+8], tol*[1,1],'k--')
hold off
grid on
xlabel('N')
ylabel('relative error norm')
title('Accuracy')
legend('Laplace restarting','Two-pass Lanczos','Location','NorthWest')

figure()
semilogy((1:out_L.cycles)*k, out_L.err,'-o')
hold all
semilogy((1:length(err2pl))*2*k, err2pl,'-.^')
semilogy([1, max([out_L.cycles, 2*length(err2pl)])*k],[tol tol],'k--')
hold off
grid on
xlabel('matvecs')
ylabel('relative error norm')
title(sprintf('Convergence for N = %d', N))
legend('Laplace restarting','Two-pass Lanczos','target accuracy','Location','NorthEast')

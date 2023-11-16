%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3a:
% A: 2D-Laplacian
% F(A)b: Gamma(A)b
% 
% Observe convergence and reasonable time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Problem setup and parameter choice %%%
f = @(t) exp(-exp(-t));
F = @(z) gamma(z);
tol = 1e-7;
k = 50;
max_cycles = 300;
N_vec = 20:10:120;

time_laplace = zeros(1,length(N_vec));
time_2pl = zeros(1,length(N_vec));
num_matvecs_laplace = zeros(1,length(N_vec));
num_matvecs_2pl = zeros(1,length(N_vec));
err_norm_laplace = zeros(1,length(N_vec));
err_norm_2pl = zeros(1,length(N_vec));

%%% Load precomputed solutions %%%
load('../../solutions/gamma/laplace2d_sol.mat');

for i = 1:length(N_vec)

    %%% Construct discretized Laplacian %%%
    N = N_vec(i);
    fprintf("N = %d: ", N_vec(i));
    e=ones(N,1);
    h = 1;
    A1 = spdiags([-e/h^2 2*e/h^2 -e/h^2], [-1 0 1], N, N);
    I1 = eye(N);
    A = kron(A1,I1) + kron(I1,A1);

    %%% precomputed solution and rhs %%%
    b = ones(N*N,1); b = b/norm(b);
    xtrue = xtrue_c{i};

    %%% Run Laplace restarting %%%
    clear options
    options.restart_length = k;
    options.max_cycles = max_cycles;
    options.int_prec = 1e-3*tol;
    nvp = namedargs2cell(options);
    tic
    [x_laplace1, out_L1] = laplace_restarting(A, b, f, tol, nvp{:});
    cycle1 = out_L1.cycles;
    [x_laplace2, out_L2] = laplace_restarting(-A, b, @(t) f(-t), tol, nvp{:});
    cycle2 = out_L2.cycles;
    time_laplace(i) = toc;
    num_matvecs_laplace(i) = k*max([cycle1 cycle2]);
    err_norm_laplace(i) = norm(x_laplace1+x_laplace2-xtrue)/norm(xtrue);
    fprintf("Laplace finished.");
    
    %%% Run 2PL %%%
    tic
    m = 1000;
    [x_2pl, err2pl, mv2pl] = twopass_lanczos(A, b, m, @(X) gammam(X), xtrue, tol, k);
    time_2pl(i) = toc;
    num_matvecs_2pl(i) = mv2pl;
    err_norm_2pl(i) = err2pl(end);
    fprintf("2PL finished.\n");
end

% Plot some informative stuff

figure();
plot(N_vec, num_matvecs_laplace,'-o');
hold on
plot(N_vec, num_matvecs_2pl, '-.d');
grid on
xlim([N_vec(1)-8,N_vec(end)+8]);
ylim([0, 1.1*max(num_matvecs_laplace)]);
xlabel('N')
ylabel('matvecs')
title('Number of matrix-vector products')
legend('Laplace restarting','Two-pass Lanczos', 'Location','NorthWest')

figure();
plot(N_vec.^2, time_laplace,'-o');
hold on
plot(N_vec.^2, time_2pl, '-.d');
xlim([N_vec(1)^2-8,N_vec(end)^2+8]);
ylim([0 1.1*max(time_laplace)]);
grid on
xlabel('N^2')
ylabel('time [s]')
title('Wall-clock time')
legend('Laplace restarting','Two-pass Lanczos', 'Location','NorthWest')

figure();
semilogy(N_vec, err_norm_laplace, 'o');
hold on
semilogy(N_vec, err_norm_2pl, 'd');
hold on
semilogy([N_vec(1)-8,N_vec(end)+8], tol*[1,1],'k--')
hold off
grid on
xlabel('N')
ylabel('relative error norm')
xlim([N_vec(1)-8, N_vec(end)+8]);
ylim([1e-7*tol, 1e1*tol])
title('Accuracy')
legend('Laplace restarting','Two-pass Lanczos', 'Location','SouthEast')

%%
function M = gammam(A)
    [X, D] = eig(A,'vector');
    M = X*diag(gamma(D))*X';
end
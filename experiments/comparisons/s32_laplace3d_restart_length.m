%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1c:
% A: 3D-Laplacian
% F(A)b: A^{-3/2}b
% 
% Compare Laplace restarting to two-pass 
% Lanczos
% Check the behaviour for different restart
% lengths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Problem setup and parameter choice %%%
f = @(t) 2*sqrt(t/pi);
F = @(z) z.^-(3/2);
tol = 1e-7;
k_vec = 10:10:100;
max_cycles = 500;
N = 100;

time_laplace = zeros(1,length(k_vec));
time_2pl = zeros(1,length(k_vec));
num_matvecs_laplace = zeros(1,length(k_vec));
num_matvecs_2pl = zeros(1,length(k_vec));
err_norm_laplace = zeros(1,length(k_vec));
err_norm_2pl = zeros(1,length(k_vec));

e=ones(N,1);
A1 = spdiags([-e 2*e -e], [-1 0 1], N, N);
I1 = eye(N);
A = kron(A1,kron(I1,I1)) + kron(I1, kron(A1,I1)) + kron(I1, kron(I1,A1));

%%% Load precomputed solutions %%%
load('../../solutions/laplace3d_sol.mat');

%%% precomputed solution and rhs %%%
b = b_c{end};
xtrue = xtrue_c{end};

for i = 1:length(k_vec)
    k = k_vec(i);
    fprintf("k = %d: ", k_vec(i));


    %%% Run Laplace restarting %%%
    clear options
    options.F = F;
    options.restart_length = k;
    options.xtrue = xtrue;
    options.max_cycles = max_cycles;
    if k==10
        options.int_prec = 1e-5*tol;
    else
        options.int_prec = 1e-3*tol;
    end
    options.adaptive_quad = false;
    nvp = namedargs2cell(options);
    tic
    [x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
    time_laplace(i) = toc;
    num_matvecs_laplace(i) = k*length(out_L.err);
    err_norm_laplace(i) = norm(x_laplace-xtrue)/norm(xtrue);
    fprintf("Laplace finished. ");

    %%% Run two-pass Lanczos %%%
    tic
    m = 1200;
    [x_2pl, err2pl, mv2pl] = twopass_lanczos(A, b, m, @(X) inv(X*sqrtm(full(X))), xtrue, tol, k);
    time_2pl(i) = toc;
    num_matvecs_2pl(i) = mv2pl;
    err_norm_2pl(i) = err2pl(end);
    fprintf("2PL finished.\n");
end

% Plot some informative stuff
set(groot, "defaultaxescolororder", [0 0.4470 0.7410; 0.1719 0.625 0.1719; 0 0 0])

figure()
plot(k_vec, num_matvecs_laplace,'-o')
hold all
plot(k_vec, num_matvecs_2pl,'-.^')
hold off
grid on
xlabel('restart length')
ylabel('matvecs')
title('Number of matrix-vector products')
legend('Laplace restarting','Two-pass Lanczos','Location','NorthWest')

figure()
plot(k_vec, time_laplace,'-o')
hold all
plot(k_vec, time_2pl,'-.^')
hold off
grid on
xlabel('restart length')
ylabel('time [s]')
title('Wall-clock time')
legend('Laplace restarting','Two-pass Lanczos','Location','NorthWest')

figure()
semilogy(k_vec, err_norm_laplace, 'o')
hold all
semilogy(k_vec, err_norm_2pl,'^');
semilogy([k_vec(1)-8,k_vec(end)+8], tol*[1,1],'k--')
hold off
grid on
xlabel('restart length')
ylabel('relative error norm')
title('Accuracy')
legend('Laplace restarting','Two-pass Lanczos','Location','SouthEast')

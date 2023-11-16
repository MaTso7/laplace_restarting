%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 3b:
% A: 2D convection diffusion operator
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
k = 20;
max_cycles = 200;
N_vec = 20:10:120;

time_laplace = zeros(1,length(N_vec));
num_matvecs_laplace = zeros(1,length(N_vec));
err_norm_laplace = zeros(1,length(N_vec));

%%% Load precomputed solutions %%%
load('../../solutions/gamma/convdiff2d_sol.mat');

for i = 1:length(N_vec)

    %%% 2d convection-diffusion, upwind, notation: trottenberg 7.1.11
    N = N_vec(i);
    fprintf("N = %d: ", N_vec(i));
    h = 100/(N+1);
    
    e=ones(N,1);
    A1 = spdiags([-e/h^2 2*e/h^2 -e/h^2], [-1 0 1], N, N);
    I1 = eye(N);
    
    ab = [1 -1];
    a=ab(1); b=ab(2);
    epsilon = 1e-3;
    A2a = spdiags([(-a-abs(a))*e/(2*h) abs(a)*e/h (a-abs(a))*e/(2*h)],[-1 0 1],N,N);
    A2b = spdiags([(-b-abs(b))*e/(2*h) abs(b)*e/h (b-abs(b))*e/(2*h)],[-1 0 1],N,N);
    
    A = kron(epsilon*A1+A2b,I1) + kron(I1,epsilon*A1+A2a);
    

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
    fprintf("Laplace finished.\n");
end

%%
% Plot some informative stuff

figure();
plot(N_vec, num_matvecs_laplace,'-o');
grid on
xlim([N_vec(1)-8,N_vec(end)+8]);
ylim([0, 1.1*max(num_matvecs_laplace)]);
xlabel('N')
ylabel('matvecs')
title('Number of matrix-vector products')
% legend('Laplace restarting', 'Location','NorthWest')

figure();
plot(N_vec.^2, time_laplace,'-o');
xlim([N_vec(1)^2-8,N_vec(end)^2+8]);
ylim([0 1.1*max(time_laplace)]);
grid on
xlabel('n=N^2')
ylabel('time [s]')
title('Wall-clock time')
% legend('Laplace restarting', 'Location','NorthWest')

figure();
semilogy(N_vec, err_norm_laplace, 'o');
hold on
semilogy([N_vec(1)-8,N_vec(end)+8], tol*[1,1],'k--')
hold off
grid on
xlabel('N')
ylabel('relative error norm')
xlim([N_vec(1)-8, N_vec(end)+8]);
ylim([min([1e-3*tol err_norm_laplace]), max([1e1*tol err_norm_laplace])])
title('Accuracy')
% legend('Laplace restarting', 'Location','SouthEast')

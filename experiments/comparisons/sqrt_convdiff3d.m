%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 4b:
% A: 3D-convection-diffusion
% F(A)b: A^{1/2}b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../algorithms')
addpath('../../quadrature')

%%% Load precomputed solutions %%%
load('../../solutions/sqrt/convdiff3d_sol.mat');

%%% Problem setup and parameter choice %%%
f = @(t) t.^(-3/2)/(2*sqrt(pi));
F = @(z) z.^(1/2);
tol = 1e-7;
k = 20;
max_cycles = 100;

time_laplace = zeros(1,length(N_vec));
num_matvecs_laplace = zeros(1,length(N_vec));
err_norm_laplace = zeros(1,length(N_vec));


for i = 1:length(N_vec)

    % 3d convection-diffusion, 1st order upwind, see "Multigrid" by
    % Trottenberg et al., eq. (7.1.11)
    N = N_vec(i);
    fprintf("N = %d: ", N_vec(i));
    e=ones(N,1);
    A1 = spdiags([-e 2*e -e], [-1 0 1], N, N);
    I1 = eye(N);
    
    abc = [1 -1 1];
    h = 1/(N+1);
    a=abc(1); b=abc(2); c=abc(3);
    epsilon = 1e-3;
    A2a = spdiags([(-a-abs(a))*e/(2*h) abs(a)*e/h (a-abs(a))*e/(2*h)],[-1 0 1],N,N);
    A2b = spdiags([(-b-abs(b))*e/(2*h) abs(b)*e/h (b-abs(b))*e/(2*h)],[-1 0 1],N,N);
    A2c = spdiags([(-c-abs(c))*e/(2*h) abs(c)*e/h (c-abs(c))*e/(2*h)],[-1 0 1],N,N);
    
    A = kron(epsilon/(h^2)*A1+A2c, kron(I1,I1)) + kron(I1,kron(epsilon/(h^2)*A1+A2b,I1)) + kron(I1,kron(I1,epsilon/(h^2)*A1+A2a));
    
    %%% precomputed solution and rhs %%%
    b = ones(N*N*N,1); b = b/norm(b);
    xtrue = xtrue_c{i};


    %%% Run Laplace restarting %%%
    clear options
    options.FunctionClass = "Bernstein";
    options.restart_length = k;
    options.xtrue = xtrue;
    options.max_cycles = max_cycles;
    options.int_prec = 1e-3*tol;
    nvp = namedargs2cell(options);
    tic
    [x_laplace, out_L] = laplace_restarting(A, b, f, tol, nvp{:});
    time_laplace(i) = toc;
    num_matvecs_laplace(i) = k*out_L.cycles;
    err_norm_laplace(i) = norm(x_laplace-xtrue)/norm(xtrue);
    fprintf("Laplace finished.\n");
end

% Plot some informative stuff

figure();
plot(N_vec, num_matvecs_laplace,'-o');
grid on
xlabel('N')
ylabel('matvecs')
title('Number of matrix-vector products')

figure();
plot(N_vec.^3, time_laplace,'-o');
grid on
xlabel('N^3')
ylabel('time [s]')
title('Wall-clock time')

figure();
semilogy(N_vec, err_norm_laplace, 'o');
hold all
semilogy([N_vec(1)-8,N_vec(end)+8], tol*[1,1],'k--')
hold off
grid on
xlabel('N')
ylabel('relative error norm')
title('Accuracy')

figure()
semilogy((1:out_L.cycles)*k, out_L.err,'-o')
hold all
semilogy([1 out_L.cycles]*k, [tol tol],'k-.')
hold off
grid on
xlabel('matvecs')
ylabel('relative error norm')
title(sprintf('Convergence for N = %d', N))

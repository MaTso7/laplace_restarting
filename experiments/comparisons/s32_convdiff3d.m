%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 1b:
% A: 3D convection diffusion operator
% F(A)b: A^{-3/2}b
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

time_laplace = zeros(1,length(N_vec));
num_matvecs_laplace = zeros(1,length(N_vec));
err_norm_laplace = zeros(1,length(N_vec));

%%% Load precomputed solutions %%%
load('../../solutions/convdiff3d_sol.mat');

for i = 1:length(N_vec)
    N = N_vec(i);
    fprintf("N = %d: ", N_vec(i));
          
    % 3d convection-diffusion, 1st order upwind, see "Multigrid" by
    % Trottenberg et al., eq. (7.1.11)
    e=ones(N,1);
    A1 = spdiags([-e 2*e -e],[-1 0 1],N,N);
    I1 = eye(N);

    abc = [1 -1 1];
    h = 1/(N+1);
    a=abc(1); b=abc(2); c=abc(3);
    epsilon = 1e-3;
    A2a = spdiags([(-a-abs(a))*e/(2*h) abs(a)*e/h (a-abs(a))*e/(2*h)],[-1 0 1],N,N);
    A2b = spdiags([(-b-abs(b))*e/(2*h) abs(b)*e/h (b-abs(b))*e/(2*h)],[-1 0 1],N,N);
    A2c = spdiags([(-c-abs(c))*e/(2*h) abs(c)*e/h (c-abs(c))*e/(2*h)],[-1 0 1],N,N);
    
    A = kron(epsilon/(h^2)*A1+A2c, kron(I1,I1)) + kron(I1,kron(epsilon/(h^2)*A1+A2b,I1)) + kron(I1,kron(I1,epsilon/(h^2)*A1+A2a));
    b = ones(N*N*N,1); b = b/norm(b);
    xtrue = xtrue_c{i};


    %%% Run Laplace restarting %%%
    clear options
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
    fprintf("Laplace finished.\n");
end

figure();
plot(N_vec, num_matvecs_laplace,'-o')
grid on
xlabel('N')
ylabel('matvecs')
title('Number of matrix-vector products')

figure();
plot(N_vec.^3, time_laplace,'-o')
grid on
xlabel('N^3')
ylabel('time [s]')
title('Wall-clock time')

figure();
semilogy(N_vec, err_norm_laplace, 'o')
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
semilogy(tol*ones(1+length(out_L.err)*k),'k-.')
hold off
grid on
xlabel('matvecs')
ylabel('relative error norm')
title(sprintf('Convergence for N = %d', N))

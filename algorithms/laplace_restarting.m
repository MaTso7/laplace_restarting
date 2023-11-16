function [x, out] = laplace_restarting(A, b, f, tol, options)
% Approximate F(A)b by restarted Arnoldi with
% F a Laplace transform or a complete Bernstein function
% 
% This method is part of the package laplace_restarting as described in
%
%   A. Frommer, K. Kahl, M. Schweitzer, and M. Tsolakis:
%   "Krylov subspace restarting for matrix Laplace transforms", 
%   SIAM J. Matrix Anal. Appl., 44 (2023), pp. 693–717, 
%   doi: 10.1137/22M1499674 
% 
% and in
% 
%   M. Tsolakis:
%   "Efficient Computation of the Action of Matrix Rational Functions and
%   Laplace transforms" (Doctoral Thesis),
%   Bergische Universität Wuppertal, Germany, 2023,
%   doi: 10.25926/BUW/0-106
% 
arguments %(Input)
    A (:,:)             % matrix
    b (:,1)             % right-hand side
    f function_handle   % function to be transformed; f ~= F (see FunctionClass); needs to accept array-valued arguments
    tol double          % relative target accuracy; 2-norm of updates if exact solution not given
    options.restart_length {mustBeInteger} = 20     % restart length
    options.FunctionClass string = "Laplace"        % "Laplace": F=L{f}, "Bernstein": F(s) = int_0^inf (1-exp(-ts))*f(t) dt
    options.F function_handle = function_handle.empty() % (scalar) function F for F(A)b; needs to accept array-valued arguments
    options.max_cycles {mustBeInteger} = 20         % maximum number of restart cycles
    options.int_prec double = max(1e-3*tol,1e-15)   % target precision for quadrature(s) (relative and absolute)
    options.int_maxnodes {mustBeInteger} = 1000     % try to use at most int_maxnodes quadrature nodes (at least 150); will not use more than twice this amount
    options.tol_spl double  = []                    % target precision for interpolating splines
    options.compute_bounds logical = false          % whether to compute upper and lower bounds for the error
    options.lmin double = inf                       % smallest eigenvalue of A. If not available, smallest Ritz value is used as proxy
    options.xtrue (:,1) = []                        % exact solution to compute the error
    options.safety_factor double = 1e-3             % factor to multiply smallest Ritz value with in order to hopefully bound lmin from below
    options.verbose logical = false                 % whether to print information to the command line while running
    options.expmv logical = false                   % whether to use expmv (https://github.com/higham/expmv); only for non-Hermitian A; needs to be in MATLAB search path
    options.adaptive_quad logical = false           % experimental: determine a new quadrature rule in each cycle
    options.spline_cond logical = false             % experimental: ensure that the spline is well-conditioned by removing interpolation nodes
end
% arguments (Output) %requires MATLAB R2022b
%     x (:,1)     % final approximation to F(A)b
%     out struct  % additional information; see below
%         % out.cycles int    % number of computed restart cycles
%         % out.N_quad (1,:)  % number of quadrature nodes used
%         % out.refine (1,:)  % number of spline refinement steps; -1: used t_i+t_j
%         % out.lbound (1,:)  % lower a posteriori bound for absolute 2-norm error (if compute_bounds == true)
%         % out.ubound (1,:)  % upper a posteriori bound for absolute 2-norm error (if compute_bounds == true)
%         % out.err    (1,:)  % relative 2-norm error (if xtrue was provided)
% end

%% preparations
if options.restart_length > length(b)
    k = length(b);
    if options.verbose
        fprintf("Warning: Restart length larger than matrix dimension; won't use restarts.\n")
    end
else
    k = options.restart_length;
end
if options.int_maxnodes < 150
    options.int_maxnodes = 150;
    if options.verbose
        fprintf("Warning: Maximum number of quadrature nodes too low; will use 150.\n")
    end
end
if isempty(options.tol_spl)
    options.tol_spl = options.int_prec;
end
adaptive_quadrature = options.adaptive_quad;
spline_cond         = options.spline_cond;
safety_factor_bound = options.safety_factor;

if options.verbose && (adaptive_quadrature || spline_cond)
    fprintf("Warning: experimental option enabled.\n")
end

out = struct( "cycles", 0, ...
              "N_quad", 0, ...
              "refine", 0);
pred = zeros(1,options.max_cycles);

beta = norm(b);

hermitian = ishermitian(A);
if hermitian
    krylov = @(v) lanczos(A,v,k);
else
    krylov = @(v) arnoldi(A,v,k);
end

%% first cycle
if options.verbose
    fprintf("Restart cycle 1: ")
end
cycle = 1;
[V_i,H_i] = krylov(b);
if options.verbose
    fprintf("...finished Arnoldi. ")
end

%%%quadrature rule
%get smallest ew
if hermitian
    [X,D] = eig(H_i(1:k,:),'vector');
else
    D = eig(H_i(1:k,:));
end
[~,lmin_i] = min(real(D));
lmin = D(lmin_i);
if options.compute_bounds && isinf(options.lmin)
    lmin_bound = safety_factor_bound*lmin;
else
    lmin_bound = options.lmin;
end

%determine quadrature rule
if options.FunctionClass == "Laplace"
    [t,w] = quadrature(@(t) exp(log(f(t))-t*lmin), options.int_prec, options.int_maxnodes);
elseif options.FunctionClass == "Bernstein"
    [t,w] = quadrature(@(t) (1-exp(-t*lmin)).*f(t), options.int_prec, options.int_maxnodes);
end
[t, ind]=sort(transpose(t));
w = w(ind);

%remove values for t that are too large
if lmin < 0
    max_t = max(t(abs(f(t))>0));
    t = t(t<=max_t);
    w = w(1:length(t));
end

Nn = length(t);
out.N_quad(cycle) = Nn;
if options.verbose
    fprintf("Chose %d quadrature nodes.\n", Nn)
end
if ~hermitian
    dt = t(2:end)-t(1:end-1);
end


%compute the matrix exponentials and the first Arnoldi approximation
wexptHe1 = zeros(k,Nn); % wexptHe1(:,j) = w_j * exp(-t(j)*H) * e_1
if hermitian
    %matrix exponentials
    Xinv_e1 = X\[1; zeros(k-1,1)];
    for j=1:Nn
        wexptHe1(:,j) = w(j)*(exp(-t(j)*D).*Xinv_e1);
    end
    wexptHe1 = X*wexptHe1;
    
    %Arnoldi approximation
    if ~isempty(options.F)
        x = V_i*([beta*(X*(options.F(D).*Xinv_e1)); 0]);
    else
        if options.FunctionClass == "Laplace"
            x = V_i*[wexptHe1*f(t)*beta; 0];
        elseif options.FunctionClass == "Bernstein"
            x = V_i*[ ([w; zeros(k-1,Nn)] - wexptHe1) *f(t)*beta; 0];
        end
    end
else
    %matrix exponentials
    wexptHe1(:,1) = expm(-t(1)*H_i(1:k,:))*[1; zeros(k-1,1)];

    %use expm or expmv
    if options.expmv
        try
            M = select_taylor_degree(-H_i(1:k,:), [1;zeros(k-1,1)], [], [], [], true);
            Mmin = min(M(M~=0));
            for j=2:Nn
                if abs(dt(j-1))*Mmin > 1
                    wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
                else
                    wexptHe1(:,j) = expmv(-dt(j-1), H_i(1:k,:), wexptHe1(:,j-1), M);
                end
            end
        catch
            ME=MException('expmv:failed','Error using expmv. Make sure that the path is set correctly.\n');
            throw(ME)
        end
    else
        for j=2:Nn
            wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
        end
    end
    wexptHe1 = w.*wexptHe1;

    %Arnoldi approximation
    if options.FunctionClass == "Laplace"
        x = V_i*[wexptHe1*f(t)*beta; 0];
    elseif options.FunctionClass == "Bernstein"
        x = V_i*[ ([w; zeros(k-1,Nn)] - wexptHe1) *f(t)*beta; 0];
    end
end

wg_i = transpose(wexptHe1(k,:));

%check if done
out.cycles = 1;
if ~isempty(options.xtrue)
    out.err = norm(x-options.xtrue)/norm(options.xtrue);
    if out.err < tol
        if options.verbose
            fprintf("Reached target accuracy.\n")
        end
        return
    end
end
if options.max_cycles==1 || k == length(b)
    return
end

%% first restart (no spline needed)
cycle = 2;
if options.verbose
    fprintf("Restart cycle 2: ")
end
if options.FunctionClass == "Laplace"
    func_im1 = @(x) f(x);
elseif options.FunctionClass == "Bernstein"
    func_im1 = @(x) -f(x);
end
H_im1 = H_i;
wg_im1 = wg_i;
t_im1 = transpose(t);
[V_i,H_i] = krylov(V_i(:,end));
if options.verbose
    fprintf("...finished Arnoldi.\n")
end


if adaptive_quadrature
    %get smallest ew
    if hermitian
        [X,D] = eig(H_i(1:k,:),'vector');
    else
        D = eig(H_i(1:k,:));
    end
    [~,lmin_i] = min(real(D));
    lmin = D(lmin_i);
    if options.compute_bounds && isinf(options.lmin)
        lmin_bound = min([lmin_bound, safety_factor_bound*lmin]);
    end

    %determine new quadrature rule
    func_i = @(t) -H_im1(k+1,k)*( func_im1(t+t_im1)*wg_im1 );
    [t,w] = quadrature(@(t) exp(log(func_i(t))-t*lmin), options.int_prec, options.int_maxnodes);
    [t, ind]=sort(transpose(t));
    w = w(ind);
    
    %remove values for t that are too large
    if lmin < 0
        max_t = max(t(abs(f(t))>0));
        t = t(t<=max_t);
        w = w(1:length(t));
    end
    
    Nn = length(t);
    out.N_quad(cycle) = Nn;
    if options.verbose
        fprintf("   Chose %d quadrature nodes.\n", Nn)
    end
    if ~hermitian
        dt = t(2:end)-t(1:end-1);
    end
end

F_im1 = func_im1(t+t_im1);
f_i = -H_im1(k+1,k)*( F_im1*wg_im1 );

%compute the matrix exponentials
wexptHe1 = zeros(k,Nn);
if hermitian
    if ~adaptive_quadrature
        [X,D] = eig(H_i(1:k,:),'vector');
        if options.compute_bounds && isinf(options.lmin)
            [~,lmin_i] = min(real(D));
            lmin_bound = min([lmin_bound, safety_factor_bound*D(lmin_i);]);
        end
    end
    Xinv_e1 = X\[1; zeros(k-1,1)];
    for j=1:Nn
        wexptHe1(:,j) = w(j)*(exp(-t(j)*D).*Xinv_e1);
    end
    wexptHe1 = X*wexptHe1;
else
    wexptHe1(:,1) = expm(-t(1)*H_i(1:k,:))*[1; zeros(k-1,1)];

    if options.expmv
        try
            M = select_taylor_degree(-H_i(1:k,:), [1;zeros(k-1,1)], [], [], [], true);
            Mmin = min(M(M~=0));
            for j=2:Nn
                if abs(dt(j-1))*Mmin > 1
                    wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
                else
                    wexptHe1(:,j) = expmv(-dt(j-1), H_i(1:k,:), wexptHe1(:,j-1), M);
                end
            end
        catch
            ME=MException('expmv:failed','Error using expmv. Make sure that the path is set correctly.\n');
            throw(ME)
        end
    else
        for j=2:Nn
            wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
        end
    end
    wexptHe1 = w.*wexptHe1;
end

wg_i = transpose(wexptHe1(k,:));

xcorr = wexptHe1*f_i;
xcorr = V_i*[beta*xcorr; 0];
x = x+xcorr;

if options.compute_bounds == true
    out.lbound = norm(xcorr);

    e_k = zeros(k,1); e_k(k) = 1;
    eta = H_i(k+1,k);
    delta = (H_i(1:k,:)-lmin_bound*eye(k))\(eta^2*e_k);
    T_hat = [H_i(1:k,:) eta*e_k; eta*e_k' delta(k)];
    % compute eigenvalue decompositions for the Jacobi matrices
    [WW, DD] = eig(full(T_hat),'vector');
    wexptHe1 = zeros(k+1,Nn);
    WWinv_e1 = WW\[1; zeros(k,1)];
    for j=1:Nn
        wexptHe1(:,j) = w(j)*(exp(-t(j)*DD).*WWinv_e1);
    end
    xcorr_radau = WW*(wexptHe1*f_i);
    out.ubound = beta*norm(xcorr_radau);
end

%check if done
out.cycles = 2;
if ~isempty(options.xtrue)
    out.err(cycle) = norm(x-options.xtrue)/norm(options.xtrue);
    if out.err(cycle) < tol %|| abs(out.err(cycle)-out.err(cycle-1)) < tol
        out.refine=pred(1:cycle);
        if options.verbose
            fprintf("Reached target accuracy.\n")
        end
        return
    end
elseif norm(xcorr)/norm(x) < tol
    out.refine=pred(1:cycle);
    if options.verbose
            fprintf("Reached target accuracy.\n")
    end
    return
end
if options.max_cycles==2
    if options.verbose
            fprintf("Reached max_cycles but not target accuracy.\n")
    end
    return
end


%% further restarts
for cycle=3:options.max_cycles
    if options.verbose
        fprintf("Restart cycle %d: ", cycle)
    end
    H_im2 = H_im1;
    H_im1 = H_i;
    [V_i,H_i] = krylov(V_i(:,end));
    if options.verbose
        fprintf("...finished Arnoldi.\n")
    end

    wg_im2 = wg_im1;
    wg_im1 = wg_i;
    t_im2 = t_im1;
    t_im1 = transpose(t);
    Nn_im1 = Nn;
    
    f_im1 = f_i;
    func_im2 = func_im1;

    
    if adaptive_quadrature
        %get smallest ew
        if hermitian
            [X,D] = eig(H_i(1:k,:),'vector');
        else
            D = eig(H_i(1:k,:));
        end
        [~,lmin_i] = min(real(D));
        lmin = D(lmin_i);
        if options.compute_bounds && isinf(options.lmin)
            lmin_bound = min([lmin_bound, safety_factor_bound*lmin]);
        end

        %determine new quadrature rule
        func_i = @(t) -H_im1(k+1,k)*( func_im1(t+t_im1)*wg_im1 );
        [t,w] = quadrature(@(t) exp(log(func_i(t))-t*lmin), options.int_prec, options.int_maxnodes);
        [t, ind]=sort(transpose(t));
        w = w(ind);

        %remove values for t that are too large
        if lmin < 0
            max_t = max(t(abs(f(t))>0));
            t = t(t<=max_t);
            w = w(1:length(t));
        end
        
        Nn = length(t);
        out.N_quad(cycle) = Nn;
        if options.verbose
            fprintf("   Chose %d quadrature nodes.\n", Nn)
        end
        if ~hermitian
            dt = t(2:end)-t(1:end-1);
        end
    end
    
    %compute the matrix exponentials
    wexptHe1 = zeros(k,Nn);
    if hermitian
        if ~adaptive_quadrature
            [X,D] = eig(H_i(1:k,:),'vector');
            if options.compute_bounds && isinf(options.lmin)
                [~,lmin_i] = min(real(D));
                lmin_bound = min([lmin_bound, safety_factor_bound*D(lmin_i);]);
            end
        end
        Xinv_e1 = X\[1; zeros(k-1,1)];
        for j=1:Nn
            wexptHe1(:,j) = w(j)*(exp(-t(j)*D).*Xinv_e1);
        end
        wexptHe1 = X*wexptHe1;
    else
        wexptHe1(:,1) = expm(-t(1)*H_i(1:k,:))*[1; zeros(k-1,1)];

        if options.expmv
            try
                M = select_taylor_degree(-H_i(1:k,:), [1;zeros(k-1,1)], [], [], [], true);
                Mmin = min(M(M~=0));
                for j=2:Nn
                    if abs(dt(j-1))*Mmin > 1
                        wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
                    else
                        wexptHe1(:,j) = expmv(-dt(j-1), H_i(1:k,:), wexptHe1(:,j-1), M);
                    end
                end
            catch
                ME=MException('expmv:failed','Error using expmv. Make sure that the path is set correctly.\n');
                throw(ME)
            end
        else
            for j=2:Nn
                wexptHe1(:,j) = expm(-dt(j-1)*H_i(1:k,:))*wexptHe1(:,j-1);
            end
        end
        wexptHe1 = w.*wexptHe1;
    end

    wg_i = transpose(wexptHe1(k,:));

    t_fine = t_im1;
    f_im1_fine = f_im1;
    
    %construct first spline this cycle
    f_app_fine = spline(t_fine, f_im1_fine);
    func_im1_fine = @(t) ppval(f_app_fine,t);

    %construct the Arnoldi update
    F_im1_fine = func_im1_fine(t+t_im1);
    f_i_fine = -H_im1(k+1,k)*( F_im1_fine*wg_im1 );
    xcorr_fine = wexptHe1*f_i_fine;

    %refine spline if necessary
    current_pred = -1;
    while( current_pred ~=0 )
        %construct the Arnoldi update with a "coarse" spline
        f_app_coarse = spline(t_fine(1:2:end),f_im1_fine(1:2:end));
        func_im1_coarse = @(t) ppval(f_app_coarse,t);

        F_im1_coarse = func_im1_coarse(t+t_im1);
        f_i_coarse = -H_im1(k+1,k)*( F_im1_coarse*wg_im1 );
        xcorr_coarse = wexptHe1*f_i_coarse;

        %now check the difference to the "fine" update
        diff = beta*norm(xcorr_coarse - xcorr_fine);
        current_pred = max(ceil( log(diff/norm(x)/options.tol_spl) / log(16) ), 0);

        if( current_pred > 0 ) %refine spline
            
            %determine additional nodes
            if( 2^(current_pred+pred(cycle)) >= Nn_im1) %refining more expensive than direct evaluation
                t_add = t+t_im1;
                t_add = t_add(:);
                t_add = [0 t_add'];
                current_pred = 0;
                pred(cycle) = -1;
            else
                t_add = zeros(1,(2^current_pred)*length(t_fine));
                j0 = 1;
                for j1=1:length(t_fine)-1
                    for j2=1:2^current_pred-1
                        t_add(j0) = t_fine(j1) + j2/2^current_pred*(t_fine(j1+1)-t_fine(j1));
                        j0 = j0+1;
                    end
                end
            end
            
            %remove zeros
            if spline_cond
                %remove nodes too close to each other
                t_add = uniquetol(t_add,1e-7,'DataScale',1);
            else
                t_add = t_add(t_add~=0);
            end
            
            %compute the values at the new nodes
            F_im2 = func_im2(t_add'+t_im2);
            f_add = -H_im2(k+1,k)*( F_im2*wg_im2 );

            %construct union
            [t_fine, ind] = sort([t_add t_fine]);
            f_im1_fine = [f_add; f_im1_fine]; %#ok
            f_im1_fine = f_im1_fine(ind);

            if spline_cond
                %remove nodes too close to each other
                [t_fine, IA] = uniquetol(t_fine,1e-7);
                f_im1_fine = f_im1_fine(IA);
            else
                [t_fine, IA] = unique(t_fine);
                f_im1_fine = f_im1_fine(IA);
            end
                
            
            %construct refined spline
            f_app_fine = spline(t_fine, f_im1_fine);
            func_im1_fine = @(t) ppval(f_app_fine,t);

            %evaluate refined spline and then anew the quadrature rule(s)
            F_im1_fine = func_im1_fine(t+t_im1);
            f_i_fine = -H_im1(k+1,k)*( F_im1_fine*wg_im1 );
            xcorr_fine = wexptHe1*f_i_fine;
        end
        pred(cycle) = pred(cycle) + current_pred;
    end
    func_im1 = func_im1_fine;
    f_i = f_i_fine;

    
    xcorr = V_i*[beta*xcorr_fine; 0];
    x = x+xcorr;

    if options.compute_bounds == true
        out.lbound = [out.lbound, norm(xcorr)]; %#ok
        e_k = zeros(k,1); e_k(k) = 1;
        eta = H_i(k+1,k);
        delta = (H_i(1:k,:)-lmin_bound*eye(k))\(eta^2*e_k);
        T_hat = [H_i(1:k,:) eta*e_k; eta*e_k' delta(k)];
        % compute eigenvalue decompositions for the Jacobi matrices
        [WW, DD] = eig(full(T_hat),'vector');
        wexptHe1 = zeros(k+1,Nn);
        WWinv_e1 = WW\[1; zeros(k,1)];
        for j=1:Nn
            wexptHe1(:,j) = w(j)*(exp(-t(j)*DD).*WWinv_e1);
        end
        xcorr_radau = WW*(wexptHe1*f_i);
        out.ubound = [out.ubound, beta*norm(xcorr_radau)]; %#ok
    end
    
    %check if done
    if ~isempty(options.xtrue)
        out.err(cycle) = norm(x-options.xtrue)/norm(options.xtrue); %#ok
        if out.err(cycle) < tol %|| abs(err(cycle)-err(cycle-1)) < tol
            out.refine=pred(1:cycle);
            out.cycles=cycle;
            if options.verbose
                fprintf("Reached target accuracy.\n")
            end
            return
        end
    elseif norm(xcorr)/norm(x) < tol
        out.refine=pred(1:cycle);
        out.cycles=cycle;
        if options.verbose
            fprintf("Reached target accuracy.\n")
        end
        return
    end
end
if options.verbose
    fprintf("Reached max_cycles but not target accuracy.\n")
end
out.cycles = options.max_cycles;
end

%
function [V, H] = arnoldi(A,b,k)

V = b; V(:,1) = V(:,1)/norm(V(:,1));
H = zeros(k+1,k);

for j = 1:k
    w = A*V(:,j);
    H(1:j,j) = 0;

    for i = 1:j
        h = V(:,i)'*w;
        w = w - V(:,i)*h;
        H(i,j) = H(i,j) + h;
    end

    H(j+1,j) = norm(w);
    V(:,j+1) = w/H(j+1,j);
end

end

function [V, H] = lanczos(A,b,k)

V = zeros(length(b),k+1);
v0 = 0*b;
v1 = b/norm(b);
V(:,1) = v1;

H = zeros(k+1,k);

for j = 1:k
    v = A*v1;
    if j>1
        H(j-1,j) = H(j,j-1);
        v = v - H(j,j-1)*v0;
    end
    h = v1'*v;
    v = v - h*v1;
    H(j,j) = h;

    H(j+1,j) = norm(v);
    v0 = v1;
    v1 = v*(1/H(j+1,j));
    V(:,j+1) = v1;
end

end

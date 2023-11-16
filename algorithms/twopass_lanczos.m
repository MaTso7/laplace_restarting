function [x, err, matvec] = twopass_lanczos(A, b, m, f, ex, tol, d)

N = size(A,1);
V = zeros(N,m+1);

nb = norm(b);
v0 = 0*b;
v1 = b/nb;
V(:,1) = v1;

H = zeros(m+1,m);
ii=1;
for j = 1:m
    if isnumeric(A), v = A*V(:,j); else, v = A(V(:,j)); end

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

    if mod(j,d) == 0
        y = nb * (f(H(1:j,1:j)) * eye(j,1));
        % Actually, forming x here is not necessary for checking the
        % stopping criterion. But this is the only way we can check
        % the "actual error norm", so that things are comparable to
        % our other experiments. It should not influence the run time
        % too much, though...
        x = V*[y; zeros(m+1-length(y),1)];
        err(ii) = norm(x-ex)/norm(ex);
        if err(ii) < tol
            m = j;
            break
        else
            ii=ii+1;
        end
    end
end


V = zeros(N,m+1);
v0 = 0*b;
v1 = b/norm(b);
V(:,1) = v1;
x = zeros(N,1);

for j = 1:m
    if isnumeric(A), v = A*V(:,j); else, v = A(V(:,j)); end

    if j>1
        H(j-1,j) = H(j,j-1);
        v = v - H(j,j-1)*v0;
    end
    
    h = v1'*v;
    v = v - h*v1;
    
    x = x + y(j)*v1;

    v0 = v1;
    v1 = v*(1/H(j+1,j));
    V(:,j+1) = v1;
end

matvec = 2*m;
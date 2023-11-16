function [t, w] = quadrature(f, tol, maxnodes)
% Computes nodes t and weights w of an adaptive Gauss-Kronrod quadrature
% for the integral of f in [0,inf) as described in
%
%   L. F. Shampine:
%   "Vectorized adaptive quadrature in MATLAB", 
%   J. Comput. Appl. Math., 211 (2008), pp. 131-140, 
%   doi: 10.1016/j.cam.2006.11.021
% 
% and similarly as in the MATLAB function integral()
%
% Part of the package laplace_restarting

phi = @(z) (z./(1-z)).^2; %use transformation t = phi(z)
phi_deriv = @(z) 2*z./(1-z).^3;

borders = 0:0.1:1; %initial splitting of the integration interval
borders = [borders(1:end-1)' borders(2:end)']; %new integration intervals
total_error = 0;
app_value = 0;


t = zeros(1, maxnodes);
w = t;
nodes = 0;

while ~isempty(borders) && nodes+size(borders,1)*15 <= maxnodes
    %compute Gauss-Kronrod 15 in each interval (given by borders) and 
    %compare with GK 7
    gk15 = zeros(size(borders,1),1);
    app_error = gk15;
    for i=1:size(borders,1)
        [z, w15, w7] = gk_nodes(borders(i,1),borders(i,2));
        integrand = f(phi(z)).*phi_deriv(z);
        gk15(i) = w15*integrand;
        gk7 = w7*integrand;
        app_error(i) = gk15(i)-gk7;
    end
    
    %check each interval for sufficiently small error
    finished = false(size(borders,1),1);
    bad_errors = 0;
    current_tol = tol*max(1,abs(app_value+sum(gk15)));
    for i=1:size(borders,1)
        if abs(app_error(i)) <= current_tol*(borders(i,2)-borders(i,1))
            finished(i) = true;
            total_error = total_error + app_error(i);
            [t_new, w_new] = gk_nodes(borders(i,1),borders(i,2));
            t(nodes+1:nodes+length(t_new)) = transpose(phi(t_new));
            w(nodes+1:nodes+length(t_new)) = w_new.*transpose(phi_deriv(t_new));
            nodes = nodes + length(t_new);
        else
            bad_errors = bad_errors + abs(app_error(i));
        end
    end
    
    %remove intervals with accurate approximations
    borders = borders(~finished,:);
    app_value = app_value + sum(gk15(finished));

    %allow for (only) small cancellations in the error by computing
    %total_error and bad_errors separately
    if abs(total_error)+bad_errors <= current_tol
        %include remaining intervals
        for i=1:size(borders,1)
            [t_new, w_new] = gk_nodes(borders(i,1),borders(i,2));
            t(nodes+1:nodes+length(t_new)) = transpose(phi(t_new));
            w(nodes+1:nodes+length(t_new)) = w_new.*transpose(phi_deriv(t_new));
            nodes = nodes + length(t_new);
        end
        t = t(1:nodes);
        w = w(1:nodes);
        return
    end
    
    %intervals with an error too large are split in half
    middlepoints = diff(borders,1,2)/2;
    borders = [borders + middlepoints*[1 0]; borders + middlepoints*[0 -1]];
end

if ~isempty(borders)
    fprintf("Warning: could not reach target precision for quadrature; try using more quadrature nodes.\n")

    for i=1:size(borders,1)
        [t_new, w_new] = gk_nodes(borders(i,1),borders(i,2));
        t(nodes+1:nodes+length(t_new)) = transpose(phi(t_new));
        w(nodes+1:nodes+length(t_new)) = w_new.*transpose(phi_deriv(t_new));
        nodes = nodes + length(t_new);
    end
    t = t(1:nodes);
    w = w(1:nodes);
end

end

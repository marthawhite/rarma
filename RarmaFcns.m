classdef RarmaFcns
%% Class implementing many of the helper functions required for
%% regularized ARMA
% Assumes 
%  A = [A^(1) ... A^(p)];
%  B = [B^(1); ...; B^(q)];
%  Phi = [phi_1 ... phi_t];
% Always start from the back first, and goes backwards numsamples
% Xminusone = [X1 ... X_t-1], samples as columns
    
methods(Static)

%%%%%%%%%%%%%%%%%%%%%%%%% RARMA BASIC MODEL COMPUTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = computeARMA(A, Xminusone, B, Epsilon, ardim, madim, xdim, numsamples)
    result = RarmaFcns.computeAR(A, Xminusone, ardim, xdim, numsamples) + ...
             RarmaFcns.computeMA(B, Epsilon, madim, xdim, numsamples);
end

function result = computeAR(A, Xminusone, ardim, xdim, numsamples)
% returns result = 
% [sum_i A^(i) X_{ardim+1-i}, ..., sum_i A^(i) X_{t-i}] = A X_hist
% if numsamples is less than full matrix, then returns the last
% numsamples of the array
    
% % Treat numsamples = 1 specially; for others, inefficiently
% % computes the entire AR and then returns a subset
%     if numsamples == 1
%         result = A * Xminusone(:,end-ardim+1);
%         return;
%     end
    
    Xhist = RarmaFcns.generate_history(Xminusone, ardim);
    result = A*Xhist;
    result = [zeros(xdim, ardim) result];
    if numsamples < size(result,2), result = result(:, end-numsamples+1:end); end
end

function result = computeMA(B, Epsilon, madim, xdim, numsamples)
    if isempty(Epsilon)
        % result = sum_j Z^(j)_{:,t-j}
        % Does not use Z^(j)_{:, T-madim+j+1:T} 
        result = zeros(xdim, numsamples);
        T = size(B,2);
%         % JF: not working. This is a bug I never noticed, because at a
%         time, we decided that the prediction (in
%         convex_multi_view_model.m, line 300) should not use the "noise"
%         for future prediction.
%         for j = 1:madim
%             idx = RarmaFcns.blockInd(j,xdim);
%             Tidx = T:-1:max(T-numsamples+1,j);
%             result(:,Tidx) = result(:,Tidx) + B(idx, Tidx-j+1);           
%         end
        for j = 1:madim
            idx = RarmaFcns.blockInd(j,xdim);
            Tidx = numsamples:-1:max(1,numsamples-T+j);
            result(:,Tidx) = result(:,Tidx) + B(idx, Tidx-numsamples+T-j+1);
        end
    else
        result = zeros(xdim, numsamples);
        T = size(Epsilon,2);
        % T-madim+1 is the maximum # of samples that can be generated
        % E.g., q=4, T=9, numsamples=14
        % 00000000XXXXXX  (0: padding zero, X: generated point)
        % =====---------
        %      q-1 T
        %   numsamples
        endpoint = max(1, numsamples-T+madim);
        Tidx = numsamples:-1:endpoint;
        for j = 1:madim
            idx = RarmaFcns.blockInd(j,xdim);
            result(:,Tidx) = result(:,Tidx) + B(idx,:) * Epsilon(:,Tidx-numsamples+T-j+1);
        end
    end
end

function Xiterated = iterateModel(Xstart, A, B, Epsilon, ardim, madim, xdim, numsamples)
    Xiterated = [];
    Xminusone = Xstart(:, end-ardim+1:end);
    if isempty(Epsilon)
      	for i = 1:numsamples
            xt = RarmaFcns.computeAR(A, Xminusone, ardim, xdim, 1);
            Xiterated = [Xiterated xt];
            Xminusone = [Xminusone(:, 2:end) xt];
       end        
    else  
        if madim <= ardim
            Epsilon = Epsilon(:,ardim-madim+1:end);
        else
            Epsilon = [zeros(size(Epsilon,1), madim-ardim) Epsilon];
        end
      	for i = 1:numsamples
            xt = RarmaFcns.computeARMA(A, Xminusone, B, Epsilon(:,i:i+madim-1), ardim, madim, xdim, 1);
            Xiterated = [Xiterated xt];
            Xminusone = [Xminusone(:, 2:end) xt];
       end
    end
end

function ind = blockInd(j, xdim)
    indStart = xdim*(j-1) + 1;
    ind = indStart:(indStart+xdim-1);
end

%%%%%%%%%%%%%%%%%%%%%%%%% Prediction Functions %%%%%%%%%%%%%%%%%%%%%%%%%%
% Model contains 
% model.Aall, model.B, model.z and model.zparam for the exponential scalar variable
% Phistart = [Phi_1, ..., Phi_startnum]

function Xiterated = iterate_predict(Xstart, Epsilonstart, model, horizon, opts)
%% ITERATE_PREDICT iteratively applies the ARMA model
% If Epsilonstart is empty, only uses the autoregressive part
% Otherwise, uses Epsilonstart for as many steps as possible, until
% pass the end of it and then only using autoregressive part
% Epsilonstart can either be Zstart or the epsilon; this is
% determined by the size of the matrix
    
    if isempty(Epsilonstart) 
        Xiterated = iterate_predict_ar(Xstart, model, horizon, opts);
        return;
    end
    
    % Use models A and B to predict the next point from Xstart
    r = opts.ardim;
    xdim = size(Xstart,1);
    if size(Xstart, 2) < r
        Xminusone = [zeros(size(Xstart,1), r-size(Xstart,2)) Xstart];
    else  
        Xminusone = Xstart(:, (end-r+1):end);
    end  
    Zq = Epsilonstart(:, (end-opts.madim+1):end);
    if size(Zq, 1) ~= opts.madim*xdim
        if isempty(model.B)
            Zq = zeros(opts.madim*xdim, size(Xstart,2));
        else    
            Zq = model.B*Epsilonstart(:, (end-opts.madim+1):end);
        end
    end
    
    xt = RarmaFcns.computeARMA(model.A, Xminusone, Zq, [], opts.ardim, opts.madim, xdim, 1);
    Xiterated = xt;
    Xminusone = [Xminusone xt];
    ldim = size(Zq,1);
    for i = 2:horizon
        Zq = [Zq(:,2:end) zeros(ldim,1)];
        xt = RarmaFcns.computeARMA(model.A, Xminusone, Zq, [], opts.ardim, opts.madim, xdim, 1); 
        Xminusone = [Xminusone(:, 2:end) xt];
        Xiterated = [Xiterated xt];
    end
end

function Xiterated = iterate_predict_ar(Xstart, model, horizon, opts)
%% ITERATE_PREDICT_AR
% Use models A and B to predict the next point from Xstart
% If Phistart is empty, then compute it before proceeding
% The predicted phi will use a generative Laplace model
    
    xdim = size(Xstart, 1);
    Xminusone = Xstart(:, (end-opts.ardim+1):end);
    xt = RarmaFcns.computeAR(model.A, Xminusone, opts.ardim, xdim, 1);
    Xiterated = xt;
    Xminusone = [Xminusone(:, 2:end) xt];
    for i = 2:horizon
        xt = RarmaFcns.computeAR(model.A, Xminusone, opts.ardim, xdim, 1);
        Xminusone = [Xminusone(:, 2:end) xt];
        Xiterated = [Xiterated xt];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%% RARMA LOSS FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,g] = euclidean_rarma(X, A, B, Epsilon, var)
% EUCLIDEAN_RARMA implements the euclidean loss
% See genericloss_rarma below for a description of the parameters

    lossType = 'euclidean';

    [f,g] = RarmaFcns.genericloss_rarma(X, A, B, Epsilon, var, lossType);
end

function [f,g] = robust_rarma(X, A, B, Epsilon, var)
% ROBUST_RARMA implements the l1 loss
% See genericloss_rarma below for a description of the parameters

    sigma = 1;
    lossType = 'robust';
    
    [f,g] = RarmaFcns.genericloss_rarma(X, A, B, Epsilon, var, lossType, sigma);
end

function [f,g] = genericloss_rarma(X, A, B, Epsilon, var, lossType, param)
% If Epsilon empty, then B = Z
% Ignores the first ardim parts of X in the loss
% EITHER:
% L( sum_{i=1}^p A^(i) x_t-i + sum_{j=1}^q Z^(j)_{:, t-j}, x_t)  
% OR
% L( sum_{i=1}^p A^(i) x_t-i + sum_{j=1}^q B^(j) Epsilon_{:, t-j}, x_t)  
    
% Xhist = [ X_{:,p} X_{:,p+1} ... X_{:,t-1};
%           X_{:,p-1} X_{:,p} ... X_{:,t-2}; ...
%           X_{:,1} X_{:,2} ... X_{:,t-p}; ...
%         ]
%
% A = [A^(1) A^(2) ...  A^(p)];  n x n*p
% B = [B^(1); B^(2); ... ; B^(q)];  n*q x latent_dim
% Epsilon = [Epsilon_{:,1} Epsilon_{:, 2} ... Epsilon_{:, t}];  latent_dim x t
% Z = [Z^(1); Z^(2); ... ; Z^(q)] = [B^(1) Epsilon; B^(2) Epsilon; ... ; B^(q) Epsilon];  n*q x t

% When nargin == 5, var says for which variable to return gradient
% var = 1 means gradient in A
% var = 2 means gradient in Z
% var = 3 means gradient in B
% var = 4 means gradient in Epsilon
    
    xdim = size(X, 1);
    ardim = size(A, 2)/xdim;
    madim = size(B, 1)/xdim;
    r = max(ardim, madim-1); % this is because phi_t and x_t always in the same spot
    numsamples = size(X,2);
    
    switch lossType
        case 'euclidean'
            lossfcnar = @euclidean_loss_ar;
            lossfcnma = @euclidean_loss_ma;
        case 'robust'
            lossfcnar = @robust_loss_ar;
            lossfcnma = @robust_loss_ma;
            sigma = param;
        otherwise
            error('Unknown RARMA loss type!');
    end
    
    % Now A is the variable; A = [A^(1) ...  A^(p)]
    % where A^(i) is n x n
    if var == 1
        maPart = RarmaFcns.computeMA(B, Epsilon, madim, xdim, numsamples);
        X_modified = X - maPart;
        X_modified(:,1:r) = 0; % don't care about the first r points
        
        % F = sum_i A^(i) X_{t-i}, so [Xhat_p+1, ..., Xhat_t]
        F = RarmaFcns.computeAR(A, X(:, 1:end-1), ardim, xdim, numsamples);    
        F(:,1:r) = 0;
        
        diff = F - X_modified;
        [f,g] = lossfcnar();

        % Now Z is the variable
    elseif var == 2
        arPart = RarmaFcns.computeAR(A, X(:, 1:end-1), ardim, xdim, numsamples);
        X_modified = X - arPart;
        X_modified(:,1:r) = 0;

        F = RarmaFcns.computeMA(B, [], madim, xdim, numsamples);
        F(:,1:r) = 0;
        
        diff = F - X_modified;
        [f,g] = lossfcnma();

        % Now B is the variable
    elseif var == 3
        arPart = RarmaFcns.computeAR(A, X(:, 1:end-1), ardim, xdim, numsamples);
        X_modified = X - arPart;
        X_modified(:,1:r) = 0;
        
        F = RarmaFcns.computeMA(B, Epsilon, madim, xdim, numsamples);
        F(:,1:r) = 0;
        diff = F - X_modified;
        
        [f,g] = lossfcnma();
        g = g*Epsilon';
        
        % Now Epsilon is the variable
    elseif var == 4
        arPart = RarmaFcns.computeAR(A, X(:, 1:end-1), ardim, xdim, numsamples);
        X_modified = X - arPart;
        X_modified(:,1:r) = 0;
        
        F = RarmaFcns.computeMA(B, Epsilon, madim, xdim, numsamples);
        F(:,1:r) = 0;
        diff = F - X_modified;
        
        [f,g] = lossfcnma();
        g = B'*g;   
        
    else
        error('modified_euclidean_loss -> var must be 1, 2, 3 or 4.');
    end
    
    f = f / numsamples;
    g = g ./ numsamples;

    function [f,g] = euclidean_loss_ar
        f = (0.5) * sum(sum(diff.^2));
        Xhist = [zeros(xdim*ardim, ardim) RarmaFcns.generate_history(X(:,1:end-1),ardim)];
        g = diff * Xhist';
    end

    function [f,g] = euclidean_loss_ma
        f = (0.5) * sum(sum(diff.^2));
        T = size(diff, 2);
        g = zeros(xdim*madim, T);
        for j = 1:madim
            idx = RarmaFcns.blockInd(j,xdim);
            g(idx,1:T-j+1) = diff(:,j:T);
        end
    end

    function [f,g] = robust_loss_ar
        idx = abs(diff) < sigma;
        Z = abs(diff) - (sigma/2);
        Z(idx) = diff(idx).^2 * (0.5/sigma);      
        f = sum(sum(Z));
        Xhist = [zeros(xdim*ardim, ardim) RarmaFcns.generate_history(X(:,1:end-1),ardim)];
        g = sign(diff);
        g(idx) = diff(idx)/sigma;
        g = g * Xhist';
    end

    function [f,g] = robust_loss_ma
        idx = abs(diff) < sigma;
        Z = abs(diff) - (sigma/2);
        Z(idx) = diff(idx).^2 * (0.5/sigma);
        f = sum(sum(Z));
        gg = sign(diff);
        gg(idx) = diff(idx)/sigma;
        T = size(gg, 2);
        g = zeros(xdim*madim, T);
        for j = 1:madim
            idx = RarmaFcns.blockInd(j,xdim);
            g(idx,1:T-j+1) = gg(:,j:T);
        end
    end
    
end

function [val, G] = trace_norm(A) 
%% TRACE_NORM
% Computes the trace norm on mxn A
% 
%	||A||_tr = sum_i^{min(m,n)} sigma_i

    [U, S, V] = svd(A, 'econ');
    val = sum(S(:));  % diag allows S to be rectangular
    G = U*V'; % subgradient

end

function [f, g] = frob_norm_sq(A) 
%% FROB_NORM_SQ Computes the Frobenius norm on mxn A
    f = sum(sum(A.^2));
    g = 2.*A;
end

function X_hist = generate_history(Xminusone, ardim)
% create a matrix where each column is vectorized last ardim samples
% X_hist = [[X_ardim; X_ardim-1;...;X_1 ] ... [X_t-1; X_t-2;...;X_t-ardim ]]   
% X_hist is n*ardim x t-ardim
% X is n x t-1
  
  tminusone = size(Xminusone, 2);
  X_hist = [];
  for i = 1:ardim
    X_hist = [X_hist; Xminusone(:,(ardim-i+1):(tminusone-i+1))];
  end
  
  %for i = ardim:tminusone
  %  x_iplusone = Xminusone(:,i:-1:(i-ardim+1));
  %  X_hist = [X_hist x_iplusone(:)];
  %end
end

% END OF METHODS
end    
% END OF CLASSDEF
end

function [model, obj] = rarma(X, opts)
%% Using global convex approach to solve regularized ARMA (RARMA)
% Solves the optimization
% min_{A, B, Epsilon} sum_{t=1}^T L(sum_{i=1}^p A^(i) X_{t-i} + sum_{j=0}^q B^(j) Epsilon_{t-i})
% 	     	      		  		      	      + alpha || B ||^2_{F} + alpha || Epsilon ||^2_{F}
% Requires that loss L returns a gradient.
%
% Could also be extended to partitioned structure on B; for
% simplicity and speed, the non-partitioned optimization is used.
%
% Note that returns Phi = [Phi_1 ... Phi_t], rather than full Phi matrix
%
% See DEFAULTS below for all the optional parameters.
%
% Authors: Junfeng Wen (University of Alberta)
%          Martha White (Indiana University) 
%          Last Update: Nov 2015

if nargin < 1
  error('rarma requires at least data matrix X = [X1, ..., XT]');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFAULT PARAMETERS STARTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEFAULTS.optimizer = @(lossfcn, xinit)(RarmaSolvers.fmin_LBFGS(lossfcn, xinit, struct('maxiter', 1000)));
%options = optimoptions(@fminunc,'GradObj','on')
%DEFAULTS.optimizer = @(lossfcn, xinit)(fminunc(lossfcn, xinit, options));
DEFAULTS.ardim = 5;
DEFAULTS.init_stepsize = 10;
DEFAULTS.Loss = @RarmaFcns.euclidean_rarma; 
DEFAULTS.maxiter = 1000;
DEFAULTS.madim = 5;
DEFAULTS.recover = 1;  % Recover B and Epsilon from learned Z
DEFAULTS.reg_ar = @RarmaFcns.frob_norm_sq;
DEFAULTS.reg_wgt_ar = 1e-2;
DEFAULTS.reg_wgt_ma = 1e-1;
DEFAULTS.TOL = 1e-6;
DEFAULTS.verbose = 0 ; % 0: nothing
                       % 1: output at start and end of optimization
                       % 2: optimization feedback along the way

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFAULT PARAMETERS ENDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 2
  opts = DEFAULTS;
else
  opts = RarmaUtilities.getOptions(opts, DEFAULTS);
end

% Validity check
if opts.madim < 0 || opts.ardim < 0
  error('Incorrect parameters');
end

% Solve for RARMA, AR or MA dependning on madim and ardim choices
ar_solver = @solve_rarma;
if opts.madim < 1
  ar_solver = @solve_ar;
elseif opts.ardim < 1
  ar_solver = @solve_ma;
end

if opts.verbose > 0
  printf('\n\nconvex_multi_view_models -> Starting optimization with ardim = %u, madim = %u\n', ...
          opts.ardim, opts.madim); 
end

%%%%%%%%%%%%%%%%%%%%%%%%% START OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize variables for learning
[xdim, numsamples] = size(X);
sizeA = [xdim, xdim*opts.ardim];
sizeZ = [xdim*opts.madim, numsamples];
if opts.ardim > 0
  Ainit = initAParams();
end
if opts.madim > 0
  Zinit = zeros(sizeZ);
end

% START the outer optimization; save any learned variables in model along the way
model = [];
[model, obj] = ar_solver();

%%%%%%%%%%%%%%%%%%%%%%  END OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%% BEGIN AUXILIARY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [model,obj] = solve_ar()
%% SOLVE_AR
% Solve only for A, since q = 0
  [A, obj, iter, msg] = opts.optimizer(@(Avec)(objA(Avec, X, [])), Ainit(:));
  model.A = reshape(A, sizeA);
  model.predict = @(Xstart, horizon, opts)(RarmaFcns.iterate_predict_ar(Xstart, model, horizon, opts));  
end

function [model,obj] = solve_ma()
%% SOLVE_MA
% Solve only for B and Epsilon, since p = 0
% Note: the code is currently not well-designed to do long-horizon prediction
% with only a moving average models, since future moving average
% components are not imputed; should work, however, for 1-step prediction
  [Z, obj, iter, msg] = opts.optimizer(@(Zvec)(objZ(Zvec, zeros(sizeA))), Zinit(:));
  model.Z = reshape(Z, sizeZ);
  model.A = zeros(sizeA);
  if opts.recover == 1
      [model.B, model.Epsilon] = recoverModels(Z);
  end
  % Cannot really 'predict' without autoregressive component
  model.predict = @(Xstart, horizon, opts)...
      (RarmaFcns.iterate_predict(Xstart, [], model, horizon, opts));
end


function [model,obj] = solve_rarma()
%% SOLVE_RARMA
% Solve for A first (with Z = 0), then iterate between Z and A
    [A, obj, iter, msg] = opts.optimizer(@(Avec)(objA(Avec, X, Zinit)), Ainit(:));
    A = reshape(A, sizeA);
    [Z, prev_obj] = iterateZ(Zinit, A, opts.init_stepsize);
  
  for i = 1:opts.maxiter
    % Do A first since it returns the incorrect obj
    A = iterateA(A, Z, opts.init_stepsize/i); % adaptive stepsize
    [Z, obj] = iterateZ(Z, A, opts.init_stepsize/i);
    
    if abs(prev_obj-obj) < opts.TOL % doing minimization
      break;
    end
    prev_obj = obj;
  end
  
  % Store learned models
  model.A = A;
  model.Z = Z;  
  % Recover B and Epsilon from Z, if desired
  % Note that the recoverd B and Epsilon could be rescaled
  if opts.recover == 1
      [model.B, model.Epsilon] = recoverModels(Z);
  end
  % Only use AR to predict, consider MA as noise
%   model.predict = @(Xstart, Epsilonstart, horizon, opts)(RarmaFcns.iterate_predict(Xstart, Epsilonstart, model, horizon, opts));
  model.predict = @(Xstart, horizon, opts)(RarmaFcns.iterate_predict_ar(Xstart, model, horizon, opts));
end

function [A, f] = iterateA(A, Z, init_stepsize)
  [f,g] = objA(A, X, Z);
  stepsize = RarmaSolvers.line_search(A, f, g, @(A)(objA(A, X, Z)),init_stepsize);
  A = A - stepsize*(g);
end

function [Z, f] = iterateZ(Z, A, init_stepsize)
  [f,g] = objZ(Z, A);
  stepsize = RarmaSolvers.line_search(Z, f, g, @(Z)(objZ(Z, A)), init_stepsize); 
  Z = Z - stepsize*g;
end

function [B, Epsilon] = recoverModels(Z)
      [Usvd,Sigma,V] = svd(Z, 'econ');
      sqrtSigma = sqrt(Sigma);
      B = Usvd * sqrtSigma;
      Epsilon = sqrtSigma * V';
end


% Note: reshaping a matrix to be of the same size does nothing
% so objA and objZ can either be called with the vector or matrix

function [f,g] = objA(Ain, X, Z)
%% OBJA Ain can either be a vector or matrix 
% the gradient is returned to be of the same size
  Amat = reshape(Ain,sizeA);
  if nargout > 1
    [f,g] = opts.Loss(X, Amat, Z, [], 1);  
    [f2,g2] = opts.reg_ar(Amat);
    g = g + opts.reg_wgt_ar*g2;
    g = reshape(g, size(Ain));
  else
    f = opts.Loss(X, Amat, Z, [], 1);
    f2 = opts.reg_ar(Amat);
  end
  f = f + opts.reg_wgt_ar*f2;
end

% Need to linearize for LBFGS or fminunc
function [f,g] = objZ(Zin, A)
%% OBJZ Zin can either be a vector or matrix 
% the gradient is returned to be of the same size    
  Zmat = reshape(Zin,sizeZ);
  if nargout < 2
    f = opts.Loss(X, A, Zmat, [], 2);
    f2 = RarmaFcns.trace_norm(Zmat);
  else
    [f, g] = opts.Loss(X, A, Zmat, [], 2);
    [f2, g2] = RarmaFcns.trace_norm(Zmat);
    g = g + opts.reg_wgt_ma * g2;
    g = reshape(g, size(Zin));    
  end
  f = f + opts.reg_wgt_ma * f2;
end

function Aparams = initAParams()
% INITAPARAMS
% Could initialize in many ways; for speed, we choose
% a regression between X = AXhist
  Xhist = RarmaFcns.generate_history(X(:, 1:end-1), opts.ardim);
  Aparams = (Xhist' \ X(:,opts.ardim+1:end)')';
end


end


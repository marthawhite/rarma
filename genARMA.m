function [Xstartall,Xtrainall,Xtestall] = genARMA(opts)

%% Setup parameters
DEFAULTS.xdim = 6; % dimension of observation
DEFAULTS.ldim = 3; % dimension of latent innovation
DEFAULTS.ardim = 2; % p, lag of AR
DEFAULTS.madim = 2; % q, lag of MA
DEFAULTS.tau = 0.5; % noise level of MA component
DEFAULTS.num_reps = 1; % number of time series generated
DEFAULTS.num_trainsamples = 100; % number of training points
DEFAULTS.num_testsamples = 100; % number of test points

if nargin < 1
    opts = DEFAULTS;
else
    opts = RarmaUtilities.getOptions(opts, DEFAULTS);
end

xdim = opts.xdim;
ldim = opts.ldim;
ardim = opts.ardim;
madim = opts.madim;
tau = opts.tau;
num_reps = opts.num_reps;
num_trainsamples = opts.num_trainsamples;
num_testsamples = opts.num_testsamples;
numstart = max(ardim, madim) + 1;

%% Constructing stable model
partSize = floor(xdim/ardim);
Asize = partSize;
im = sqrt(-1);
A = [];
period = floor(num_trainsamples/3);
scale = 0.999; % will shrink to scale^T
for i = 1:ardim
    idx = (i-1)*partSize+1;
    if i == ardim
        Asize = xdim-(i-1)*partSize;
    end
    smallA = 0;
    lower = 1/Asize; upper = 2/Asize;
    while sum(sum(abs(smallA)))<lower*Asize^2 || sum(sum(abs(smallA)))>upper*Asize^2 % less variance in results, 0.5~0.6
        if mod(Asize,2)
            D = scale; V = randn(Asize,1);
        else
            D = []; V = [];
        end
        nPairs = floor(Asize/2);
        for ii = 1:nPairs % conjugate eigenvalues and vecs
            d = exp(2*pi*im/period).*scale;
            D = [D, d, conj(d)];
            v = randn(Asize,2);
            V = [V, v(:,1)+im*v(:,2), v(:,1)-im*v(:,2)];
        end
        smallA = real((V*diag(D))/V); % ignore rounding error
    end
    Atmp = zeros(xdim);
    Atmp(idx:(idx+Asize-1),idx:(idx+Asize-1)) = smallA;
    A = [A, Atmp];
end
% % Check the spectrum
% bigA = A;
% if(ardim ~= 1)
%     bigA = [bigA; eye((ardim-1)*xdim), zeros((ardim-1)*xdim,xdim)];
% end
% d = abs(eig(bigA));
% maxEig = max(d); % should be <=1
B = randn(madim*xdim, ldim);
B = B./repmat(sqrt(sum(B.^2,1)),madim*xdim,1); % normalization

%% Generate observations
Xstartall = cell(num_reps,1);
Xtrainall = cell(num_reps,1);
Xtestall = cell(num_reps,1);
sigma = ones(ldim,1)*tau;
for rep = 1:num_reps
    EpsilonBase = randn(ldim, numstart + num_trainsamples + num_testsamples);
    Epsilon = repmat(sigma, 1, size(EpsilonBase,2)).*EpsilonBase; % Link Laplace variables across time
    Xstart = randn(xdim, numstart);
    Xstart = normalizeMatrix(Xstart); % normalize to have unit columns
    Xiterated = RarmaFcns.iterateModel(Xstart, A, B, Epsilon, ardim, madim, xdim, num_trainsamples + num_testsamples);
    Xstartall{rep} = Xstart;
    Xtrainall{rep} = Xiterated(:, 1:num_trainsamples);
    Xtestall{rep} = Xiterated(:, num_trainsamples+1:end);
end

% END OF FUNCTION
end

function X = normalizeMatrix(X) 
% NORMALIZEMATRIX makes entries in X have variance 1
% Each column of X is a training example.  So divide it by its radius.
  
  num_fea = size(X, 1);
  X = double(X) - repmat(mean(X, 1), num_fea, 1);
  radii = sum(X.^2,1);
  X = X./repmat(sqrt(radii),num_fea,1);
end

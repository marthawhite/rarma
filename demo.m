clear
clc

rng(10);
num_reps = 10;

%% Generate data
opts = [];
opts.num_reps = num_reps;
[Xstartall,Xtrainall,Xtestall] = genARMA(opts);

%% Learn RARMA
Models = cell(num_reps,1);
isStable = zeros(num_reps,1);
% In practice, the following paramters should be 
% cross-validated for EACH dataset before applied
opts = [];
opts.ardim = 2;
opts.madim = 2;
opts.reg_wgt_ar = 0.07; % stronger regularization on A -> more stable
opts.reg_wgt_ma = 0.01;
for ii = 1:num_reps
    Models{ii} = rarma(Xtrainall{ii},opts);
    if opts.ardim > 0
        isStable(ii) = RarmaUtilities.checkStable(Models{ii}.A);
    end
end

%% Prediction and Evaluation (only when ardim > 0)
Xpredictall = cell(num_reps,1);
Err = zeros(num_reps,1);
for ii = 1:num_reps
    Xpredictall{ii} = Models{ii}.predict(Xtrainall{ii},...
        size(Xtestall{ii},2), opts);
    Err(ii) = sum(sum((Xpredictall{ii}-Xtestall{ii}).^2))/size(Xtestall{ii},2);
end

Err

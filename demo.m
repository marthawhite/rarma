clear
clc

rng(100);
num_reps = 2;
Models = cell(num_reps,1);
Xpredictall = cell(num_reps,1);
Err = zeros(num_reps,1);

%% Generate data
opts = [];
opts.num_reps = num_reps;
[Xstartall,Xtrainall,Xtestall] = genARMA(opts);

%% Learn RARMA
% In practice, the following paramters should be 
% cross-validated before applied
opts = [];
opts.ardim = 2;
opts.madim = 2;
opts.reg_wgt_ar = 1e-2;
opts.reg_wgt_ma = 1e-1;
for ii = 1:num_reps
    Models{ii} = rarma(Xtrainall{ii},opts);
    [isStable, eigs] = RarmaUtilities.checkStable(Models{ii}.A)
end

%% Prediction and Evaluation
for ii = 1:num_reps
    Xpredictall{ii} = Models{ii}.predict(Xtrainall{ii},...
        size(Xtestall{ii},2), opts);
    Err(ii) = sum(sum((Xpredictall{ii}-Xtestall{ii}).^2));
end

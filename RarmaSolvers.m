classdef RarmaSolvers
% Class implementing many of the optimizers required for
% regularized ARMA

    properties(Constant)
        SUCCESS = 0;
        ERROR_MAXITER = 10;
        ERROR_BACKTRACK = 20;
        ERROR_BACKTRACK_MAXITER = 21;
    end
    
    methods(Static)
        
        function stepsize = line_search(x0, f0, dir0, fun, stepsize)
        % LINE_SEARCH a very basic line search
        % x0 can be a matrix
            
            backtrack_backoff = 0.5;
            backtrack_maxiter = 100; % make this large to ensure descend

            for iter = 1:backtrack_maxiter

                x = x0 - stepsize*dir0;
                f = fun(x);

                % no imaginary function values
                if ~imag(f) && f < f0 
                    break; 
                end

                stepsize = stepsize*backtrack_backoff;
            end
        end
        
        function [x,f,iter,flag] = fmin_LBFGS(fun,x0,opts)
        % limited memory BFGS
        % minimizes unconstrained fun
            
            if ~isa(fun,'function_handle'), error('fmin_LBFGS -> improper function handle'); end

            DEFAULTS.curvTol = 1e-6;  % for curvature condition
            DEFAULTS.funTol = 1e-6;
            DEFAULTS.m = 50;          % number gradients in bundle
            DEFAULTS.maxiter = 1000;   
            DEFAULTS.verbose = 0;

            % Options needed for backtrack
            DEFAULTS.backtrack_maxiter = 50;   
            DEFAULTS.backtrack_funTol = 1e-4;   
            DEFAULTS.backtrack_timeout = -1;   
            DEFAULTS.backtrack_init_stepsize = 1;   
            DEFAULTS.backtrack_backoff = 0.5;    
            DEFAULTS.backtrack_acceptfrac = 0.1;

            if nargin < 3
                opts = DEFAULTS;
            else
                opts = RarmaUtilities.getOptions(opts, DEFAULTS);
            end

            x = x0;
            flag = RarmaSolvers.SUCCESS;

            t = length(x0);
            H0 = speye(t);
            Rho = zeros(1,opts.m);
            Y = zeros(t,opts.m);
            S = zeros(t,opts.m);
            inds = [];
            slope = Inf;

            % damped limited memory BFGS method
            [f,g] = fun(x);
            for iter = 1:opts.maxiter
                % compute search direction
                dir = RarmaSolvers.invhessmult(-g,Y,S,Rho,H0,inds,opts.m);
                slope = dir'*g;
                if -slope < opts.funTol, break; end

                [xnew,flag] = RarmaSolvers.backtrack(fun,x,f,dir,slope,opts);
                if flag ~= RarmaSolvers.SUCCESS, break; end
                [fnew,gnew] = fun(xnew);
                
                % update memory for estimating inverse Hessian
                s = xnew - x;
                y = gnew - g;
                curvature = y'*s;
                if curvature > opts.curvTol
                    rho = 1/curvature;
                    if length(inds) < opts.m
                        i = length(inds)+1;
                        inds = [inds i];
                    else
                        i = inds(1);
                        inds = [inds(2:end) inds(1)];
                    end
                    Rho(i) = rho;
                    Y(:,i) = y;
                    S(:,i) = s;
                end

                x = xnew;
                f = fnew;
                g = gnew;
            end  

            if iter >= opts.maxiter
                flag = RarmaSolvers.ERROR_MAXITER;
            end
            
        end

        function R = invhessmult(V,Y,S,Rho,H0,inds,m)
        % implicit multiplication of V by limited memory inverse Hessian approximation

            [t,n] = size(V);
            Alpha = zeros(length(inds),n);
            gamma = 1;

            Q = V;
            for j = length(inds):-1:1
                i = inds(j);
                Alpha(i,:) = Rho(i)*S(:,i)'*Q;
                Q = Q - Y(:,i)*Alpha(i,:);
            end
            if length(inds) == m
                gamma = S(:,inds(1))'*Y(:,inds(1)) / (Y(:,inds(1))'*Y(:,inds(1)));
            end
            R = gamma*H0*Q;
            for j = 1:length(inds)
                i = inds(j);
                Beta = Rho(i)*Y(:,i)'*R;
                R = R + S(:,i)*(Alpha(i,:)-Beta);
            end
        end
        

        function [x,flag] = backtrack(fun,x0,f0,dir,slope,opts)
        % backtrack line search, using more info then above linesearch 

            flag = RarmaSolvers.SUCCESS;

            if any(imag(x0)), x=x0;f=f0;g=-dir;flag=RarmaSolvers.ERROR_BACKTRACK;return, end;
            if any(imag(f0)), x=x0;f=f0;g=-dir;flag=RarmaSolvers.ERROR_BACKTRACK;return, end;
            if any(imag(dir)), x=x0;f=f0;g=-dir;flag=RarmaSolvers.ERROR_BACKTRACK;return, end;

            % backtrack
            alpha = opts.backtrack_init_stepsize;
            for iter = 1:opts.backtrack_maxiter

                x = x0 + alpha*dir;

                f = fun(x);
                if imag(f) 
                    alpha = alpha*opts.backtrack_backoff;
                    continue
                end
                
                if f < f0 || f < f0 + opts.backtrack_acceptfrac*alpha*slope 
                    break; 
                end
                alpha = alpha*opts.backtrack_backoff;
            end

            if iter == opts.backtrack_maxiter
                x = x0;
                flag = RarmaSolvers.ERROR_BACKTRACK_MAXITER;
            end
        end
    end
end

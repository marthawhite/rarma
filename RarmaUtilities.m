classdef RarmaUtilities
%% Class implementing additional utlities

    methods(Static)

        function opts = getOptions(opts, DEFAULTS) 
        % Must provide opts and DEFAULTS; if opts empty, then DEFAULTS
        % overrides the opts struct. Otherwise, all the fields in DEFAULTS
        % are added to opts, as long as they are not already set in opts.
        %
        % Note: copies fields from DEFAULTS deep into sub-struct in opts.
        %
        % Author: Martha White, University of Alberta, 2012
 
            if isempty(opts)
                opts = DEFAULTS;
            else
        		opts = copyFields(opts, DEFAULTS);
            end
            
            function opts = copyFields(opts, DEFAULTS)
                field_names = fieldnames(DEFAULTS)';

                % If a field is a struct, then recursively call copyFields
                for i=1:length(field_names)
                    f = field_names{i};
                    if ~isfield(opts,f)
                        opts = setfield(opts,f,getfield(DEFAULTS,f));
                    elseif isstruct(getfield(DEFAULTS, f))
                        opts = setfield(opts, f, copyFields(getfield(opts,f), getfield(DEFAULTS,f)));          
                    end
                end
            end
        end
        
        function [isStable, eigs] = checkStable(A, threshold)
        % This function check whether a model is stable based on the
        % eigenvalues of A
            if nargin < 2
                threshold = 1.0;
            end
            [xdim, tmp] = size(A);
            ardim = round(tmp/xdim);
            bigA = A;
            if ardim < 1
                error('ardim < 1');
            elseif ardim ~= 1
                bigA = [bigA; eye((ardim-1)*xdim),...
                    zeros((ardim-1)*xdim,xdim)];
            end
            eigs = abs(eig(bigA));
            maxEig = max(eigs); % should be <=1
            isStable = (maxEig<=threshold);
        end
        
    end
end

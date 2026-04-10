function [z_vec, self] = invert(self, f, L, opts)

    %
    % invert
    %
    % Builds a reconstruction of source dipoles from a given lead field with
    % the CSM method.
    %
    % Inputs:
    %
    % - self
    %
    %   An instance of CSMInverter with the method-specific parameters.
    %
    % - f
    %
    %   Some vector.
    %
    % - L
    %
    %   The lead field that is being inverted.
    %
    % - procFile
    %
    %   A struct with source space indices.
    %
    % - source_direction_mode
    %
    %   The way the orientations of the sources should be interpreted.
    %
    % - opts.use_gpu = false
    %
    %   A logical flag for choosing whether a GPU will be used in
    %   computations, if available.
    %
    % Outputs:
    %
    % - reconstruction
    %
    %   The reconstrution of the dipoles.
    %
    % - self
    %
    %   An instance of possibly modified IASInverter with the method-specific parameters.
    %

    arguments

        self (1,1) inverse.ConditionallyGaussianInverter

        f (:,1) {mustBeA(f,["double","gpuArray"])}

        L (:,:) {mustBeA(L,["double","gpuArray"])}

        opts.use_gpu (1,1) logical = false

        opts.normalize_data (1,1) double = 1

    end

    % Initialize waitbar with a cleanup object that automatically closes the
    % waitbar, if there is an interruption with Ctrl + C or when this function
    % exits.

    if self.number_of_frames <= 1 && strcmp(self.computation_mode,"Waitbar")
        h = waitbar(0,'Cond Gaussian Reconstruction.');
        cleanup_fn = @(wb) close(wb);
        cleanup_obj = onCleanup(@() cleanup_fn(h));
    end

    % Get needed parameters from self and others.
    date_str = '';
    update_freq = floor(self.n_map_iterations/10);
    theta0 = self.theta0;
    beta = self.beta;
    d_sqrt = self.d_sqrt;
    if max(size(d_sqrt)) == 1
        d_sqrt = repmat(d_sqrt,size(L,2),1);
    else
        d_sqrt = d_sqrt(:);
    end
    S_mat = self.noise_cov;
    
    % Set GPU arrays
    if opts.use_gpu && gpuDeviceCount > 0
        S_mat = gpuArray(S_mat);
        L = gpuArray(L);
        f = gpuArray(f);
        d_sqrt = gpuArray(d_sqrt);
    end

    % Then start inverting.
    for i = 1:self.n_map_iterations
        if self.number_of_frames <=1 && strcmp(self.computation_mode,"Waitbar")
            tic;
            time_val = toc;
            date_str = display_waitbar(h,i,self.n_map_iterations,update_freq,date_str,time_val);
        end

        H = L .* d_sqrt';     
        z_vec = d_sqrt.*( H' * (( H * H' + S_mat )\f));
        
        if opts.use_gpu && gpuDeviceCount > 0
            z_vec = gather(z_vec);
        end

        if strcmp(self.MAP_algorithm,"IAS")
            if strcmp(self.hyperprior,"Inverse gamma")
                d_sqrt = sqrt((theta0+0.5*z_vec.^2)./(beta-2.5));
            elseif strcmp(self.hyperprior,"Gamma")
                d_sqrt = sqrt(0.5*theta0.*(beta-2.5 + sqrt((2./theta0).*z_vec.^2 + (beta-2.5).^2)));
            end
        elseif strcmp(self.MAP_algorithm,"EM")
            if strcmp(self.hyperprior,"Inverse gamma")
                d_sqrt = sqrt((theta0+z_vec.^2)./(beta+1.5));
            elseif strcmp(self.hyperprior,"Gamma")
                d_sqrt = sqrt( sqrt(2*theta0).*abs(z_vec).*besselk(beta-1.5,sqrt(2/theta0)*abs(z_vec))./(besselk(beta-2.5,sqrt(2/theta0)*abs(z_vec))+1e-12) );
            end
        end
    end % MAP iterations

    if strcmp(self.computation_mode,"Waitbar")
        close(h)
    end

end % function

%%

function date_str = display_waitbar(h,index,max_iter,update_frequency,date_str,time_val)
if index > 1
    if mod(index,update_frequency) == update_frequency - 1
        date_str = char(datetime(datevec(now+(max_iter/(index-1) - 1)*time_val/86400)));
    end

    if mod(index,update_frequency) == 0
        waitbar(index/max_iter,h,['Step ' int2str(index) ' of ' int2str(max_iter) '. Ready: ' date_str '.' ]);
    end
end
end % function

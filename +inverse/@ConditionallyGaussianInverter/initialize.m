function self = initialize(self,L,f_data)
    %
    % initialization function
    %
    % Initialize recursively updated variables before the computation of
    % the first time step.
    %
    % Inputs:
    %
    % - self
    %
    %   An instance of ConditionallyGaussianInverter with the method-specific parameters.
    %
    % Outputs:
    %
    % - self
    %
    %   Recursive variables initialized
    %

    arguments

        self (1,1) inverse.ConditionallyGaussianInverter

        L (:,:) {mustBeA(L,["double","gpuArray"])}

        f_data (:,:) {mustBeA(f_data,["double","gpuArray"])}

    end

    if not(isprop(self,'d_sqrt'))
        self.addprop('d_sqrt');
    end
    if not(isprop(self,'noise_cov'))
        self.addprop('noise_cov');
    end
    
    noise_var = mean(f_data.^2,'all')/self.SNR;
    self.noise_cov = noise_var*eye(size(f_data,1));

    if strcmp(self.hyperprior_mode,"Sensitivity weights")
        self.beta = 3.5;
        if strcmp(self.hyperprior,"Inverse gamma")
            self.theta0 = trace(self.noise_cov/self.beta)*(self.SNR - 1)./repelem(sum(reshape(sum(L.^2),3,[])),3);
        elseif strcmp(self.hyperprior,"Gamma")
            self.theta0 = (self.beta-1)*trace(self.noise_cov)*(self.SNR - 1)./repelem(sum(reshape(sum(L.^2),3,[])),3);
        end
        %self.theta0 = min(self.theta0(:));
        self.theta0 = self.theta0(:);
    end
    
    if strcmp(self.MAP_algorithm,"IAS")
        if strcmp(self.hyperprior,"Inverse gamma")
            self.d_sqrt = sqrt(self.theta0./(self.beta-2.5));
        elseif strcmp(self.hyperprior,"Gamma")
            self.d_sqrt = sqrt(self.theta0.*(self.beta-2.5));
        end
    elseif strcmp(self.MAP_algorithm,"EM")
        if strcmp(self.hyperprior,"Inverse gamma")
            self.d_sqrt = sqrt(self.theta0./(self.beta+1.5));
        elseif strcmp(self.hyperprior,"Gamma")
            self.d_sqrt = sqrt(self.theta0*(self.beta-1.5));
        end
    end


end
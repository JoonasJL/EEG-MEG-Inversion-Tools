%% Copyright © 2025- Joonas Lahtinen
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
        %   An instance of HALpRInverter with the method-specific parameters.
        %
        % - L
        % The lead field matrix
        %
        % - f_data 
        % The measurement vector that is in the matrix format 
        % <# of challels> x <# of time steps>
        % 
        %
        % Outputs:
        %
        % - self
        %
        %   Recursive variables initialized
        %
    
        arguments
    
            self (1,1) inverse.OrientationalHALpRInverter
    
            L (:,:) {mustBeA(L,["double","gpuArray"])}
    
            f_data (:,:) {mustBeA(f_data,["double","gpuArray"])}
    
        end

        noise_var = mean(f_data.^2,'all')/self.SNR;
        if isempty(self.noise_cov)
            if size(f_data,2) > 1
                self.noise_cov = cov(f_data');
            else
                self.noise_cov = noise_var*eye(size(f_data,1));
            end
        end
        

    if strcmp(self.hyperprior_mode,"Sensitivity weights")
        if isempty(self.beta)
            self.beta = 3.5;
        end
        self.theta0 = sqrt(0.5*(self.beta-1)*(self.beta-2)*trace(self.noise_cov)*(self.SNR - 1)./repelem(sum(reshape(sum(L.^2),3,[])),3));    
    end
    self.theta0 = self.theta0(:);
    
end
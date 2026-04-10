%% Copyright © 2025- Joonas Lahtinen and Alexandra Koulouri 
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
        %   An instance of GroupLassoInverter with the method-specific parameters.
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
    
            self (1,1) inverse.GroupLaplaceInverter
    
            L (:,:) {mustBeA(L,["double","gpuArray"])}
    
            f_data (:,:) {mustBeA(f_data,["double","gpuArray"])}
    
        end

        noise_var = mean(f_data.^2,'all')/self.SNR;
        self.noise_cov = noise_var*eye(size(f_data,1));

        if strcmp(self.hyperprior_mode,"Sensitivity weights")
            self.lambda = sqrt(4*repelem(sum(reshape(sum(L.^2),3,[])),3)/(trace(self.noise_cov)*(self.SNR - 1)));
        
        end
        self.lambda = self.lambda(:);
        
end
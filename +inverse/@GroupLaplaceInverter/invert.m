%% Copyright © 2025- Joonas Lahtinen and Alexandra Koulouri
function [z_vec, self] = invert(self, f_data, L, opts)
%
    % invert
    %
    % Builds a reconstruction of source dipoles from a given lead field with
    % the GroupLasso method.
    %
    % Inputs:
    %
    % - self
    %
    %   An instance of GroupLassoInverter with the method-specific parameters.
    %
    % - f_data
    %  A matrix containing the time serial measurement data in the format
    %  <channels>x<time steps>
    %
    % - L
    %
    %   The lead field that is being inverted.
    %
    % - procFile
    %
    %   A struct with source space indices.
    %
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

    arguments

        self (1,1) inverse.GroupLaplaceInverter

        f_data (:,:) {mustBeA(f_data,["double","gpuArray"])}

        L (:,:) {mustBeA(L,["double","gpuArray"])}

        opts.use_gpu (1,1) logical = false

        opts.normalize_data (1,1) double = 1

    end

n_L1_iterations = self.n_L1_iterations;
lambda = self.lambda;

C = self.noise_cov;
if opts.use_gpu == 1 && gpuDeviceCount > 0
    L = gpuArray(L);
    C = gpuArray(C);
end

if opts.use_gpu == 1 && gpuDeviceCount > 0
    f = gpuArray(f_data);
else
    f = f_data;
end

    % inversion starts here
%-----------------------------------------------------------------------------------

%__ Initialization __
n = size(L,2);

z_vec = plugins.LASSOoptimization.LG_optimization(L,C,f,lambda(1:3:end),ones(n,1),n_L1_iterations,"Normal");
    if sum(isnan(z_vec))>0
        z_vec(isnan(z_vec))=mean(abs(z_vec(not(isnan(z_vec)))));
    end

if opts.use_gpu == 1 && gpuDeviceCount > 0
z_vec = gather(z_vec);
end

end
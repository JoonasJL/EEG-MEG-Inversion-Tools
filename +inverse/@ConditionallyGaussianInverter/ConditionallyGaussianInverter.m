classdef ConditionallyGaussianInverter < inverse.CommonInverseParameters & dynamicprops

    %
    % IASInverter
    %
    % A class which defines the properties needed by the IAS inversion method,
    % and the method itself.
    %

    properties
        %
        % Defines the used hyperprior model; either gamma or inverse gamma
        %distribution
        %
        hyperprior (1,1) string { mustBeMember(hyperprior, ["Inverse gamma", "Gamma"]) } = "Inverse gamma"      
        
        %
        % MAP iteration algorithm. Options:
        %- Iterative Alternating Sequential (IAS)
        %- Expectation-Maximization (EM)
        %
        MAP_algorithm (1,1) string { mustBeMember(MAP_algorithm, ["IAS", "EM"]) } = "IAS"


        %
        % Number of MAP iterations
        %
        n_map_iterations (1,1) double {mustBeInteger, mustBePositive} = 25

        %
        % Hyperprior balancing options:
        %- "Constant" (non-balanced)
        %- "Balanced" (eLORETA-type variance-balancing)
        %
        hyperprior_mode (1,1) string { mustBeMember(hyperprior_mode, ...
            ["Sensitivity weights", "Resolution matrix-based"] ) } = "Sensitivity weights";

        theta0 (:,:) double = 1e-6

        beta (:,:) double = 3.5

        SNR (1,1) double = 0.05

    end % properties

    properties (SetAccess = protected)
        %
        % DOI for the corresponding article
        %
        DOI (1,1) string = "https://doi.org/10.1137/080723995"
    
    end

    methods

        function self = ConditionallyGaussianInverter(args)

            %
            % IASInverter
            %
            % The constructor for this class.
            %

            arguments

                args.hyperprior_mode = "Sensitivity weights"  %Will replace the 'ias_hyperprior' field

                args.hyperprior = "Inverse gamma"

                args.MAP_algorithm = "IAS"

                args.n_map_iterations = 25      %Will replace the 'ias_n_map_iterations' field

                args.data_normalization_method = "Maximum entry"

                args.high_cut_frequency = 9

                args.low_cut_frequency = 7

                args.number_of_frames = 1

                args.sampling_frequency = 1024

                args.signal_to_noise_ratio = 30

                args.time_start = 0

                args.time_window = 0

                args.time_step = 1

                args.theta0 = 1e-8

                args.beta = 3.5

                args.SNR = 0.05

            end

            % Initialize superclass fields.

            self = self@inverse.CommonInverseParameters( ...
                "low_cut_frequency" ,args.low_cut_frequency, ...
                "high_cut_frequency", args.high_cut_frequency, ...
                "data_normalization_method", args.data_normalization_method, ...
                "number_of_frames", args.number_of_frames, ...
                "sampling_frequency", args.sampling_frequency, ...
                "time_start", args.time_start, ...
                "time_window", args.time_window, ...
                "time_step", args.time_step, ...
                "signal_to_noise_ratio", args.signal_to_noise_ratio ...
            );

            % Initialize own fields.

            self.n_map_iterations = args.n_map_iterations;
            
            self.hyperprior = args.hyperprior;

            self.MAP_algorithm = args.MAP_algorithm;

            self.theta0 = args.theta0;

            self.beta = args.beta;

            self.SNR = args.SNR;

            %Print the statement
            self.InitialStatement
        end

        % Declare the initialize and inverse method defined in the files invert and initialize in this same
        % folder.

        self = initialize(self,L,f_data)

        [reconstruction, self] = invert(self, f, L, opts)

        function self = terminateComputation(self)
            
            d_sqrt = findprop(self,'d_sqrt');
            noise_cov = findprop(self,'noise_cov');
            delete(noise_cov);
            delete(d_sqrt);
        end

    end % methods

    methods (Static)
        function InitialStatement 
            txt = strcat('This class object is for computing inversion with the Iterative Alter-\n'...
                , 'nating Sequential (IAS) hyperparameter updating method for a conditio-\n' ...
                , 'nally Gaussian model with inverse-gamma or gamma distributed hyperparam-\n' ...
                , 'eters.\n'...
                , 'If You find this method useful for Your thesis, or research or refer to\n'...
                , 'it in any text format, please consider citing the following articles:\n\n' ...
                , '⦁ Daniela Calvetti and Erkki Somersalo. "Gaussian hypermodels and recov-\n'...
                , 'ery of blocky objects", In: Inverse Problems, 23 (2007), pp. 733–754.\n\n'...
                , '⦁ Daniela Calvetti, Harri Hakula, Sampsa Pursiainen, and Erkki Somer-\n'...
                , 'salo. "Conditionally gaussian hypermodels for cerebral source localiza-\n'...
                , 'tion". In: SIAM Journal on Imaging Sciences, 2(3), pp. 879-31.\n'...
                , 'DOI:https://doi.org/10.1137/080723995 \n');
            
             fprintf(txt)
        end
    end %static methods

end % classdef
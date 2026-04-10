%% Copyright © 2025- Joonas Lahtinen and Alexandra Koulouri
classdef GroupLaplaceInverter < inverse.CommonInverseParameters & handle

    %
    % GroupLassoInverter
    %
    % A class which defines the properties needed by the GroupLasso inversion method,
    % and the method itself.
    %

    properties

        hyperprior_mode (1,1) string { mustBeMember( ...
            hyperprior_mode, ...
            [ "Sensitivity weights", "Manually selected" ] ...
        ) } = "Sensitivity weights";

        lambda (:,:) double {mustBePositive} = 3;

        n_L1_iterations (1,1) int16 {mustBePositive,mustBeInteger} = 5;

        SNR (1,1) double = 1000;

        %
        % Parameter for prior variance selection
        %

        initial_prior_steering_db (1,1) {mustBeA(initial_prior_steering_db,["double","gpuArray"])} = 0

        %
        % tag string for waitbar especially
        %

        tag (1,1) string = ""

        %
        % Checking flag if user has changed the value of noise_cov parameter
        %
        noise_covSetted (1,1) {mustBeNumericOrLogical} = false

    end % properties

    properties (SetObservable)

        %
        % Measurement noise covariace (optional)
        % User can set their own noise covariance matrix through this
        % property.
        %
        
        noise_cov (:,:) {mustBeA(noise_cov,["double","gpuArray"])} = []

        %
        % Checking flag if the inversion computing is running. This
        % prevents the value changes made by computation to be identified
        % as user-made.
        %
        computing_parameters (1,1) {mustBeNumericOrLogical} = false

    end

    methods

        function self = GroupLaplaceInverter(args)

            %
            % GroupLassoInverter
            %
            % The constructor for this class.
            %

            arguments

                args.lambda = 3

                args.hyperprior_mode = "Sensitivity weights"

                args.n_L1_iterations = 5

                args.SNR = 1000

                args.data_normalization_method = "Maximum entry"

                args.high_cut_frequency = 9

                args.low_cut_frequency = 7

                args.number_of_frames = 1

                args.sampling_frequency = 1024

                args.signal_to_noise_ratio = 30

                args.time_start = 0

                args.time_window = 1

                args.time_step = 1

                args.tag = ""

                args.initial_prior_steering_db = 0

                args.noise_cov = []

                args.noise_covSetted = false

                args.computing_parameters = false

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
                "signal_to_noise_ratio", args.signal_to_noise_ratio...
            );

            % Initialize own fields.

            self.lambda = args.lambda;

            self.hyperprior_mode = args.hyperprior_mode;

            self.SNR = args.SNR;

            self.n_L1_iterations = args.n_L1_iterations;

            self.initial_prior_steering_db = args.initial_prior_steering_db;
            
            self.tag = args.tag;

            self.noise_cov = args.noise_cov;

            self.noise_covSetted = args.noise_covSetted;

            self.computing_parameters = args.computing_parameters;

            % Initialize listeners 
            addlistener(self,'noise_cov','PostSet',@(src,evnt)self.setEventsFlags(src,evnt,self));

        end
    
        % Declare the initialize and inverse method defined in the files invert and initialize in this same
        % folder.
        self = initialize(self,L,f_data)

        [reconstruction, self] = invert(self,f_data,L)

        function self = terminateComputation(self)
            %If the user has not given their own inversion parameters, we
            %reset the automatically computed parameters because the user 
            %could change the data or model between separate runs.
            SNR_variable = findprop(self,'SNR_variable');
            delete(SNR_variable);
            if not(self.noise_covSetted)
                self.noise_cov = [];
            end
        end

    end % methods

    methods (Static)
        %The function that is woken by listener when the value of theta or
        %noise_cov is changed to something else than empty by hand. 
        % The function set the respective *Setted property value true when 
        % value is changed.
        function setEventsFlags(src,evnt,self) %two first inputs must be there and have these dedicated roles. The third 'self' is an extra variable.
         if not(self.computing_parameters)
             if isempty(self.noise_cov)
                 self.noise_covSetted = false;
             else
                 self.noise_covSetted = true;
             end
         end
        end % function
    end % static methods

end % classdef
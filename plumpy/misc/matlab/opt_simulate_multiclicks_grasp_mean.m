function [TP, FP] = opt_simulate_multiclicks_grasp(params)
% function simulates multiclicks and returns the scores:
% true positives (TP) anf false positives (FP)
% it runs for specified brain function, session and runs
%
% parameters that are optimized:
% channels              : list of integers: channels to include. Can be 
%                         1 - N ch
%                         e.g. [1, 2]
% lowFreq               : list of integers or integer: the lowest frequency 
%                         in the bin of interest (= length of channels for 
%                         list or same value is applied to all channels)
%                         e.g. [65 65] or 65
% highFreq              : list of integers or integer: the highest 
%                         frequency in the bin of interest (= length of 
%                         channels or same value is applied to all channels)
%                         e.g. [95 95] or 95
% featureWeights        : list of floats: weights over channels. If you 
%                         want to get an average signal, do 1/length 
%                         (channels) for all channel weights (= length of
%                         channels)
%                         e.g. [1/2 1/2]
% timeSmoothing         : list of floats: weights for the samples to 
%                         smooth over.Example says 6 samples, which means
%                         equal weighthing over time. If equal weights,
%                         make sure that each weight = 1/length(samples)
%                         (= length of samples over time)
%                         e.g. [1/6 1/6 1/6 1/6 1/6 1/6]
% linearWeights         : list of floats: weights per feature. If equal 
%                         weights, divide 1 / num features (= length of 
%                         channels)
%                         e.g. [1 1]
% threshold             : float within 0-1 to threshiold the z-scored 
%                         signal. Above the threshold click can be detected
%                         e.g. 0.35
% activePeriod          : float (in seconds) for the time during which 
%                         you want to check for consecutive crossed 
%                         thresholds 
%                         e.g. 0.8
% activeRate            : float within 0-1 of time in the activePeriod 
%                         needed to cross the threshold for a click
%                         e.g. 1
% refractoryPeriod      : float in seconds of time after the detected 
%                         click before another click could be detected
%                         e.g. 3


addpath(genpath('/home/julia/Documents/MATLAB/Plumtree/Plumtree'))

%% Load data
header.subjName         = 'CC2';
header.task             = 'MultiClicks';
header.brainFunction    = 'Grasp';
header.app              = 'PT'; % 'PT' = palmtree 'PRES' = presentation (central)
header.session          = [17, 18]; % empty = all 

file_paths              = pt_selectDatafiles(header);
data                    = pt_loadData2StructFromFile(header,file_paths);

%% Click to choose
sequenceDuration        = 1; % 2 and 5 in later sessions

%% Your parameters to optimize:
% these are referred in the simulation script using the names provided here
% 
if isfield(params, 'channels')
    channels                = double(params.channels);
else
    channels                = [64, 65, 67, 69];
end
if isfield(params, 'lowFreq')
    lowFreq                 = params.lowFreq;
else
    lowFreq                 = 65;
end
if isfield(params, 'highFreq')
    highFreq                = params.highFreq;
else
    highFreq                = 95;
end
if isfield(params, 'featureWeights')
    featureWeights          = params.featureWeights;
else
    featureWeights          = ones(1,length(channels))*(1/length(channels));
end
if isfield(params, 'timeSmoothing')
    %timeSmoothing           = params.timeSmoothing; 
    timeSmoothing           = ones(1,params.timeSmoothing)* ...
                              (1/params.timeSmoothing);
else
    timeSmoothing           = ones(1,6)*(1/6);
end
if isfield(params, 'linearClassWeights')
    linearClassWeights      = params.linearWeights;
else
    linearClassWeights      = 1;
end
if isfield(params, 'threshold')
    threshold               = params.threshold; 
else
    threshold               = .45; 
end
if isfield(params, 'activePeriod')
    activePeriod            = params.activePeriod;    
else
    activePeriod            = 1; 
end
if isfield(params, 'activeRate')
    activeRate              = params.activeRate;  
else
    activeRate              = .8; 
end
if isfield(params, 'refractoryPeriod')
    refractoryPeriod        = params.refractoryPeriod;
else
    refractoryPeriod        = 3;
end

%% Report parameters
%for ifield = 1:length(fieldnames(params))
%    fprintf(['--------- Parameter ', params.(ifield) '=', )
%end
struct2table(params)
disp(['Channels:        ' num2str(params.channels)])
disp(['Feature weights: ' num2str(params.featureWeights)])

%% Initialize output
TP                          = zeros(length(data), 1);
FP                          = zeros(length(data), 1);

%% Run simulation
for run = 1:length(data)

    % Select data and events
    timedata  = data(run).srcData;
    events = data(run).events.srcData;


    % PalmTree Filters:

    % [ARF] = Auto-Regressive filter (in NexusSignal module), converts time to power
    % [FSF] = Feature selector filter, selects power features
    % [TSF] = Time-Smoothing filter, smooths the signal
    % [AF]  = Adaptation filter, z-scores the signal
    % [LCF] = Linear Classifer filter, combines features into control signal
    % [TF]  = Threshold Classifier filter, thresholds control signal
    % [CTF] = Click translater filter, produces clicks from binary signal
    % [KSF] = key sequence (escape) filter, produces clicks from binary signal


    %% Auto-Regressive Filter [ARF] - Converts data in time domain to power domain:
    fprm                          = [];          %simulate with original parameters (feedback runs)
    
    inputOutput = [channels;                           % input
                   1:length(channels);                  % output
                   ones(1, length(channels))*lowFreq;   % lowF
                   ones(1, length(channels))*highFreq;  % highF
                   ones(1, length(channels))*5;];       % evaluationsPerBin
    fprm.ARF.inputOutput          = inputOutput; %[chIn; chOut; lowF; highF; evaluationsPerBin];
    fprm.ARF.modelOrder           = 25;
    fprm.ARF.samplingFrequency    = data(run).fs.srcData;
    fprm.ARF.transformNumSubsets  = 1;           %how many subset to divide each package
    fprm.ARF.powerUnits           = 1;           %1=sqrt(mem) - spectral amplitude, 2=sqrt(2*mem) values - spectral power
    fprm.ARF.toplot               = 0;           %1=plot data_out, 0=don't plot

    % Apply the filter
    [powerdata, Fs] = pt_runARFilterOnline(data(run), fprm.ARF); %it takes the whole structure with data, not data_in;
    % powerdata format: [ch1_bin1...ch1_binN...chN_bin1...chN_binN]

    fprm.samplingFrequencySimulated = Fs;

    %% Feature selection filter [FSF] - Selects which power features to use:
    fprm.FSF          = [];                   %simulate with the original parameters
    fprm.FSF.weights  = [1:length(channels) ; ones(1, length(channels)); featureWeights];      %[InputCHs ; OutputCHs ; Weights]
    fprm.FSF.toplot   = 0;                    %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = powerdata;

    % Apply the fitler
    fsel_data             = pt_runFeatureSelectorFilter(data_in, fprm.FSF, data(run).origParams);


    %% Time Smoothing filter [TSF] - Smooths the selected features:
    fprm.TSF         = [];                    %simulate with the original parameters
    fprm.TSF.weights = repmat(timeSmoothing, [size(fsel_data,2),1]);  %[NCh x samples weights]
    fprm.TSF.toplot  = 0;                                   %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = fsel_data;

    % Apply the fitler
    sm_data               = pt_runTimeSmoothingFilter(data_in, fprm.TSF, data(run).origParams);


    %% Adaptation Filter [AF] - zscores the data using either fixed values or 30s calibration window:
    fprm.AF                   = [];               %simulate with the original parameters
    %fprm.samplingFrequency    = 5;                %sampling frequency of power signal
    %     or
    fprm.AF.method            = ones(1, size(sm_data,2))*2;            %[ch1...chN], 1=initial parameters, 2=first samples, 3=latest samples
    fprm.AF.initalMeans       = [zeros(1, size(sm_data,2))];           %[ch1...chN]
    fprm.AF.initalStds        = [ones(1, size(sm_data,2))];            %[ch1...chN]
    fprm.AF.bufferLength      = 30;               %in seconds
    fprm.AF.minimalLength     = 30;               %in seconds
    fprm.AF.bufferDiscard     = 1.2;              %in seconds
    fprm.AF.excludeThr        = ones(1, size(sm_data,2))*100;              %in standard normal distribution values (~ -3 to 3)
    fprm.AF.samplingFrequency = Fs;                %sampling frequency of power signal
    fprm.AF.toplot            = 0;                %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = sm_data;

    % Apply the fitler
    z_data                = pt_runAdaptationFilter(data_in, fprm.AF, data(run).origParams);



    %% Linear Classifier Filter [LCF] - combines features into control signal:


    fprm.LCF              = [];                   %simulate with the original parameters
    %     or
    %fprm.LCF.weights      = [1:length(channels); ones(1,length(channels)); linearClassWeights];  %[InputCHs ; OutputCHs ; Weights]
    fprm.LCF.weights      = [1:size(z_data,2); ones(1,size(z_data,2)); linearClassWeights];
                            % The position changed where the averaging happened. 
                            % So originally you had let’s say 5 channels, which 
                            % would only be combined here, so you need 5 weights 
                            % in the linear classifier.
                            % Now, the signals are combined in an earlier 
                            % filter, so you are only left with 1 ch.
                            % Which should be the second dimension in z_data
    fprm.LCF.toplot       = 0;                    %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = z_data; %figure; plot(z_data);

    % Apply the fitler
    control_signal        = pt_runLinearClassifierFilter(data_in, fprm.LCF, data(run).origParams);

    data(run).controlSignal = control_signal;
    data(run).simulationParams = fprm;


    %% Get the onsets

    [data] = pt_defineTrials(data, 1);

    trials = data(run).trialInfo;
    trials(:,1:2) = round( trials(:,1:2) * (Fs / fprm.ARF.samplingFrequency));  % downsample original trial onsets to the Fs of the signal

    % correct the last trial if necessary
    if trials(end,2) > length(control_signal)
        trials(end,2) = length(control_signal);
    end


    %% plot control signal around the multiclick trials

    if fprm.LCF.toplot
        eventCodes = unique(trials(:,3));
        for ec = 1:length(eventCodes) % all eventcodes


            if eventCodes(ec) == 0

                ec_trls_rest = find(trials(:,3) ==  eventCodes(ec));
                ec_trls_rest(ec_trls_rest == length(trials)) = []; % if last, ignore

                tr_length = trials(ec_trls_rest,2) - trials(ec_trls_rest,1);
                tr_length = round(tr_length ./ Fs);

                tr_lengths = unique(tr_length);

                for t = 1:length(tr_lengths)

                    figure;

                    ec_trls = ec_trls_rest(tr_length==tr_lengths(t));
                    trls = zeros(length(ec_trls),(tr_lengths(t)*Fs)+(2*Fs));
                    for ect = 1:length(ec_trls)
                        trls(ect, :) = control_signal( trials( ec_trls(ect),1) - Fs : trials( ec_trls(ect),1) + (tr_lengths(t)*Fs)  + Fs -1 );
                    end

                    plot(trls', 'Color', [0 0 1 0.5]); hold on;

                    plot(mean(trls)' , 'Color', 'b', 'LineWidth', 3);
                    xline(Fs, '--'); xline((tr_lengths(t)*Fs)+Fs, '--');
                    yline(0, '--');

                    title(['ITI length ', num2str(tr_lengths(t)), ' sec'])

                end

            else

                figure;

                ec_trls = find(trials(:,3) ==  eventCodes(ec));
                tr_length = min(trials(ec_trls,2) - trials(ec_trls,1));

                trls = zeros(length(ec_trls),tr_length+(2*Fs));
                for ect = 1:length(ec_trls)

                    trls(ect, :) = control_signal( trials( ec_trls(ect),1) - Fs : trials( ec_trls(ect),1) + tr_length  + Fs -1 );
                end

                plot(trls', 'Color', [0 0 1 0.5]); hold on;

                plot(mean(trls)' , 'Color', 'b', 'LineWidth', 3);
                xline(Fs, '--'); xline(tr_length+Fs, '--');
                yline(0, '--');

                title(['Bar width ', num2str(tr_length/Fs), ' sec'])


            end


        end
    end

    %% Threshold Classifier Filter [TCF] - thresholds the control signal:

    %fprm.TCF              = [];               %simulate with the original parameters
    %     or
    fprm.TCF.weights      = [1; threshold; 1];      %[InputCHs ; threshold; direction]
    fprm.TCF.toplot       = 0;                %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = control_signal;

    % Apply the fitler
    binary_data           = pt_runThresholdClassifierFilter(data_in, fprm.TCF, data(run).origParams);

    %% Click Translater Filter [CTF] - convert the binary signal into click events:

    %fprm.CTF                    = [];         %simulate with the original parameters
    %fprm.samplingFrequency      = 5;          %sampling frequency of power signal
    %     or
    fprm.CTF.activePeriod       = activePeriod;          %in seconds
    fprm.CTF.activeRate         = activeRate;        %ratio 0-1
    fprm.CTF.refractoryPeriod   = refractoryPeriod;          %in seconds
    fprm.CTF.samplingFrequency  = Fs;         %sampling frequency
    fprm.CTF.toplot             = 0;          %1=plot data_out, 0=don't plot

    % Set data to be filtered
    data_in               = binary_data; %figure; plot(data_in);

    % Apply the fitler
    clicks                = pt_runClickTranslaterFilter(data_in, fprm.CTF, data(run).origParams);

    data(run).clicks = clicks;

    %% Score the MC data

    fprm.scoring.mercy_window = [1 1];
    fprm.scoring.samplingFrequency = Fs;
    fprm.scoring.toplot        = 0;

    [stats] = pt_scoreMultiClicks_simulated(data(run), fprm.scoring);

    %% Collect output
    all_tps = [stats.hitRate];          % percentage
    all_fps = [stats.FPatRest_total];   % counts
    TP(run) = all_tps([stats.sequenceDuration] == sequenceDuration); 
    FP(run) = all_fps([stats.sequenceDuration] == sequenceDuration);

end %loop runs

%%
TP = mean(TP);
FP = mean(FP);




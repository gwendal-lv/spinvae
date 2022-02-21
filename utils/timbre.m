function rc = timbre(audio_root_path)
    disp(strcat('Script starts: ', datestr(now, 'yy/mm/dd-HH:MM:SS')));

    disp('audio_root_path: ')
    disp(audio_root_path)

    % FIXME replace audio_root_path
    sub_folders = dir(audio_root_path);
    sub_folders = sub_folders( [sub_folders.isdir] );
    sub_folders = sub_folders(3:end);  % remove . and ..
    sub_folders_names = {sub_folders.folder};  % TODO

    % Process folders one by one
    for folder_index = 1 : length(sub_folders)  % FIXME TEMP: process only 1 folder

        soundsDirectory = strcat(sub_folders(folder_index).folder, '/', sub_folders(folder_index).name);

        % Parts of: https://github.com/VincentPerreault0/timbretoolbox/blob/master/doc/Full_Config_Example.m
        singleFileName = '';
        csvDirectory = soundsDirectory;
        matDirectory = '';  % soundsDirectory;
        pltDirectory = '';

        sndConfig = struct();


        evalConfig = struct();

        % OK if not audio signal descriptor?
        evalConfig.AudioSignal.NoDescr = struct();

        %
        evalConfig.TEE = struct();
        evalConfig.TEE.CutoffFreq = 5;

        evalConfig.TEE.Att = struct();          % Specified to be evaluated/plotted
        evalConfig.TEE.Att.Method = 3;          % params shared with Dec, Rel, LAT, AttSlope, DecSlope
        evalConfig.TEE.Att.NoiseThresh = 0.15;  %                   (LAT = Log-Attack Time)
        evalConfig.TEE.Att.DecrThresh = 0.4;
        evalConfig.TEE.TempCent = struct();     % Specified to be evaluated/plotted
        evalConfig.TEE.TempCent.Threshold = 0.15;
        evalConfig.TEE.EffDur = struct();       % Specified to be evaluated/plotted
        evalConfig.TEE.EffDur.Threshold = 0.4;
        evalConfig.TEE.FreqMod = struct();      % Specified to be evaluated/plotted
        evalConfig.TEE.FreqMod.Method = 'fft';  % shared with TEE.AmpMod; require Dec and Rel

        evalConfig.TEE.RMSEnv = struct();       % Specified to be evaluated/plotted
        evalConfig.TEE.RMSEnv.HopSize_sec = 0.0029;
        evalConfig.TEE.RMSEnv.WinSize_sec = 0.0232;

        evalConfig.STFT = struct();             % Specified to be evaluated/plotted
        evalConfig.STFT.DistrType = 'pow';
        evalConfig.STFT.HopSize_sec = 0.0058;
        evalConfig.STFT.WinSize_sec = 0.0232;
        evalConfig.STFT.WinType = 'hamming';
        evalConfig.STFT.FFTSize = 1024;
        % If no descriptors are specified in the evalConfig.STFT structure, all descriptors will be evaluated

        evalConfig.ERB = struct();              % Specified to be evaluated/plotted
        evalConfig.ERB.HopSize_sec = 0.0058;
        evalConfig.ERB.Method = 'fft';
        evalConfig.ERB.Exponent = 1/4;
        % If no descriptors are specified in the evalConfig.ERB structure, all descriptors will be evaluated

        evalConfig.Harmonic = struct();             % Specified to be evaluated/plotted
        evalConfig.Harmonic.Threshold = 0.3;
        evalConfig.Harmonic.NHarms = 20;
        evalConfig.Harmonic.HopSize_sec = 0.025;
        evalConfig.Harmonic.WinSize_sec = 0.1;
        evalConfig.Harmonic.WinType = 'blackman';
        evalConfig.Harmonic.FFTSize = 32768;
        % If no descriptors are specified in the evalConfig.Harmonic structure, all descriptors will be evaluated

        csvConfig = struct();
        csvConfig.Directory = csvDirectory;
        csvConfig.TimeRes = 10;
        % Default grouping: 'sound' (1 CSV file / audio file)
        csvConfig.Grouping = 'sound';               % group by descriptor: replace with 'descr'
        % default: {'stats', 'ts'}
        csvConfig.ValueTypes = {'stats'};     % only statistics: replace with 'stats'
        %%%                                 % only time series: replace with 'ts'
        matConfig = struct();
        matConfig.Directory = matDirectory;

        plotConfig = struct();
        plotConfig.Directory = pltDirectory;
        plotConfig.TimeRes = 0;

        if ~isdir(soundsDirectory)
            error('soundsDirectory must be a valid directory.');
        end
        if ~isempty(singleFileName)
            filelist.name = singleFileName;
        else
            filelist = dir(soundsDirectory);
        end
        acceptedFormats = {'wav', 'ogg', 'flac', 'au', 'aiff', 'aif', 'aifc', 'mp3', 'm4a', 'mp4'};
        for i = 1:length(filelist)
            [~, fileName, fileExt] = fileparts(filelist(i).name);
            if ~isempty(fileName) && fileName(1) ~= '.' && (any(strcmp(fileExt(2:end), acceptedFormats)) || (length(filelist) == 1 && strcmp(fileExt(2:end), 'raw')))
                sound = SoundFile([soundsDirectory '/' fileName fileExt], sndConfig);
                sound.Eval(evalConfig);
                if ~isempty(csvDirectory)
                    sound.ExportCSV(csvConfig);
                end
                if ~isempty(matDirectory)
                    sound.Save(matConfig);
                end
                if ~isempty(pltDirectory)
                    sound.Plot(plotConfig);
                    close all;
                    clc
                end
                clear 'sound';
            end
        end


    end % subfolder-by-subfolder processing


    fprintf('Processed %d audio folders.\n', length(sub_folders))
    disp(strcat('Script ends: ', datestr(now, 'yy/mm/dd-HH:MM:SS')));
    disp('timbre.m   EXIT_SUCCESS');
    rc = 0;
end

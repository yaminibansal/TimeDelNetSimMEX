%%
rmpath('F:\Users\Arjun\Desktop\Acads\SRE\TimeDelNetSimMEX\x64\Debug_Lib');
if strcmpi(Build, 'Debug')
	rmpath('../../x64/Release_Lib');
	addpath('../../x64/Debug_Lib');
elseif strcmpi(Build, 'Release')
	rmpath('../../x64/Debug_Lib');
	addpath('../../x64/Release_Lib');
end
%%
rng(25);
N = 1000;
E = 0.2;
RecurrentNetParams.NExc = round(N*E);
RecurrentNetParams.NInh = round(N - N*E);

RecurrentNetParams.NSynExctoExc = 50;
RecurrentNetParams.NSynExctoInh = 100;
RecurrentNetParams.NSynInhtoExc = 200;

RecurrentNetParams.MeanExctoExc = 3;
RecurrentNetParams.MeanExctoInh = 5;
RecurrentNetParams.MeanInhtoExc = -15;

RecurrentNetParams.Var          = 2;
RecurrentNetParams.DelayRange   = 2;

[A, Ninh, Weights, Delays] = RecurrentNetwork(RecurrentNetParams);

a = 0.02*ones(N,1);
b = 0.2*ones(N,1);
c = -65*ones(N,1);
d = 8*ones(N,1);

a(Ninh) = 0.1;
b(Ninh) = 0.2;
c(Ninh) = -65;
d(Ninh) = 2;
Delays = Delays + 10;
[NEndVect, NStartVect] = find(A);

%% Input setup
% Setting up input settings
OutputOptions = {'VCF'};

% Clearing InputList
clear InputList;

% Getting Midway state
% InputList = ConvertStatetoInitialCond(StateVars1, 1);
InputList.a = single(a);
InputList.b = single(b);
InputList.c = single(c);
InputList.d = single(d);

InputList.NStart = int32(NStartVect);
InputList.NEnd   = int32(NEndVect);
InputList.Weight = single(Weights);
InputList.Delay  = single(Delays);

InputList.onemsbyTstep          = int32(4);
InputList.NoOfms                = int32(800);
InputList.DelayRange            = int32(20);
InputList.StorageStepSize       = int32(0);
InputList.OutputControl         = strjoin(OutputOptions);
InputList.StatusDisplayInterval = int32(600);

tic;
[OutputVar2, StateVars2, FinalState2] = TimeDelNetSimMEX_Lib(InputList);
toc;
clear functions;

%% Grid plot
relinds = 1:InputList.onemsbyTstep*InputList.NoOfms;
plotMat = StateVars2.V(:,relinds) ~= 30;		% Black for spike
plotMat(1:1) = 0;
[r,c] = size(plotMat);                           %# Get the matrix size
imagesc(StateVars2.Time(relinds), (1:r), plotMat);            %# Plot the image
colormap(gray);                              %# Use a gray colormap
axis equal                                   %# Make axes grid sizes equal
set(gca, ...
        'GridLineStyle','-','XGrid','off','YGrid','off');

%% Random plots
relInds = 1:InputList.onemsbyTstep*InputList.NoOfms;
figure; plot(StateVars2.Time(relInds), StateVars2.I(100, relInds));
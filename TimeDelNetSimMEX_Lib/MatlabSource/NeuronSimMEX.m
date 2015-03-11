if strcmpi(Build, 'Debug')
	rmpath('../../x64/Release_Lib');
	addpath('../../x64/Debug_Lib');
elseif strcmpi(Build, 'Release')
	rmpath('../../x64/Debug_Lib');
	addpath('../../x64/Release_Lib');
end

%% 
rng(25);

N = 10000;
Ninh = ~logical(floor(rand(N,1)*15));
[A, InhSyn, nonInhSyn] = CompleteRandomNet(N, Ninh);
[NEndVect, NStartVect] = find(A);
M = length(NStartVect);
Delays = floor(rand(M,1)*20) + 1;
Weights = sparse(NEndVect, NStartVect, -1000, N,N);

% % Complete Random (Useless Except for performance testing)
Weights(nonInhSyn) = rand(nnz(nonInhSyn), 1)*5 + 5;
Weights(InhSyn) = rand(nnz(InhSyn), 1)*5 - 8;
Weights(Weights == -1000) = 0;
[~,~, Weights] = find(Weights);

clear InhSyn nonInhSyn ;
clear A;

a = 0.02*ones(N,1);
b = 0.2*ones(N,1);
c = -65*ones(N,1);
d = 8*ones(N,1);

a(Ninh) = 0.1;
b(Ninh) = 0.2;
c(Ninh) = -65;
d(Ninh) = 2;

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
InputList.NoOfms                = int32(8000);
InputList.DelayRange            = int32(20);
InputList.StorageStepSize       = int32(500);
InputList.OutputControl         = strjoin(OutputOptions);
InputList.StatusDisplayInterval = int32(600);
tic;
[OutputVar2, StateVars2, FinalState2] = TimeDelNetSimMEX_Lib(InputList);
toc;
clear functions;
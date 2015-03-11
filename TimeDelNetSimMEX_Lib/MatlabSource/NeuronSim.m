deltaTsPerms = 1;
delayT = 1; % delayT = quantization level among delays / deltaT
tau = 15E-3;
tau_s = tau/4;
I0 = 1E-12; % Proportionality constant for synaptic induce current
Tlimit = 1; % Time limit for simulation

% % LIF Neuron Parameters
% C = 300E-12;
% gL = 30E-9;
% VT = 20E-3;
% EL = -70E-3;

%% Getting Synaptic Connectivity

rng(80);
N = 10000;

A = zeros(N);

for i = 1:N
	A(i+1:end, i) = ~logical(floor(rand(N-i,1)*(N-i)/4));
end
Ninh = ~logical(floor(rand(N,1)*5));
% A = tril(A, -1);
a = 0.02;
b = 0.2;
c = -65;
d = 8;

a_inh = 0.1;
b_inh = 0.2;
c_inh = -65;
d_inh = 2;

InhSyn = repmat(Ninh', N, 1) & A(:, :) ;

[X,Y] = find(A);
M = length(X);

Delays = uint8(floor(rand(M,1)*20) + 1);
Weights = zeros(N);
Weights(~InhSyn & A) = rand(length(find(~InhSyn & A)), 1)*2 + 2;
Weights(InhSyn) = rand(length(find(InhSyn)), 1)*3 - 5;
[~,~, Weights] = find(Weights);

%% 
rng(25);

N = 10000;
Ninh = ~logical(floor(rand(N,1)*5));
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

Weights = single(Weights);
Delays = single(Delays);
X = int32(X); % PostSynaptic Neuron
Y = int32(Y); % PreSynaptic Neuron

FID2 = fopen('D:\\Users\\acer\\Desktop\\Acads\\SRE\\Neuron Simulation Data\\TimeDelNetSim\\TimeDelNetSim\\NetworkGraph.bin', 'w');
fwrite(FID2, N, 'int');
fwrite(FID2, M, 'int');
fwrite(FID2, [a,b,c,d]', 'single');
for i = 1:M
	fwrite(FID2, Y(i), 'int');
	fwrite(FID2, X(i), 'int');
	fwrite(FID2, Weights(i), 'single');
	fwrite(FID2, Delays(i), 'single');
end
fclose(FID2);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NETWORK DESIGN PORTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

InputDesign;
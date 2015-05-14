#include <vector>
#include <iostream>
#include <tbb\parallel_for.h>
#include <tbb\blocked_range.h>
#include <tbb\atomic.h>
#include <fstream>
#include <chrono>
#include <cmath>
#include "..\Headers\MexMem.hpp"
#include "..\Headers\Network.hpp"
#include "..\Headers\NeuronSim.hpp"
#include "..\Headers\FiltRandomTBB.hpp"

using namespace std;

#define VECTOR_IMPLEMENTATION

void CountingSort(int N, MexVector<Synapse> &Network, MexVector<int> &indirection)
{
	MexVector<int> CumulativeCountStart(N,0);
	MexVector<int> IOutsertIndex(N);
	size_t M = Network.size();
	if (indirection.size() != M) indirection.resize(M);

	for (int i = 0; i < M; ++i){
		CumulativeCountStart[Network[i].NEnd - 1]++;
	}
	IOutsertIndex[0] = 0;
	for (int i = 1; i < N; ++i){
		CumulativeCountStart[i] += CumulativeCountStart[i - 1];
		IOutsertIndex[i] = CumulativeCountStart[i - 1];
	}
	int indirIter = 0;
	for (int i = 0; i < M; ++i){
		int NeuronEnd = Network[i].NEnd;
		indirection[IOutsertIndex[NeuronEnd-1]] = i;
		IOutsertIndex[NeuronEnd - 1]++;
	}
}

void CurrentUpdate::operator () (const tbb::blocked_range<int*> &BlockedRange) const{
	int *begin = BlockedRange.begin();
	int *end = BlockedRange.end();
	for (int * iter = begin; iter < end; ++iter){
		int CurrentSynapse = *iter;
		float AddedCurrent = (Network[CurrentSynapse].Weight)*(1 << 17);
		Iin1[Network[CurrentSynapse].NEnd - 1].fetch_and_add((long long)AddedCurrent);
		Iin2[Network[CurrentSynapse].NEnd - 1].fetch_and_add((long long)AddedCurrent);
		LastSpikedTimeSyn[CurrentSynapse] = time;
	}
		
}

void NeuronSimulate::operator() (tbb::blocked_range<int> &Range) const{
	int RangeBeg = Range.begin();
	int RangeEnd = Range.end();
	for (int j = RangeBeg; j < RangeEnd; ++j){
		if (Vnow[j] == 30.0f){
			//Implementing Izhikevich resetting
			Vnow[j] = Neurons[j].c;
			Unow[j] += Neurons[j].d;
		}
		else{
			//Implementing Izhikevich differential equation
			float Vnew, Unew;
			Vnew = Vnow[j] + (Vnow[j] * (0.04f*Vnow[j] + 5.0f) + 140.0f - Unow[j] + I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17) + Iext[j] + StdDev*Irand[j]) / onemsbyTstep;
			Unew = Unow[j] + (Neurons[j].a*(Neurons[j].b*Vnow[j] - Unow[j])) / onemsbyTstep;
			Vnow[j] = (Vnew > -100)? Vnew: -100;
			Unow[j] = Unew;

			//Implementing Network Computation in case a Neuron has spiked in the current interval
			if (Vnow[j] >= 30.0f){
				Vnow[j] = 30.0f;

				LastSpikedTimeNeuron[j] = time;
				if (PreSynNeuronSectionBeg[j] >= 0){
					for (size_t k = PreSynNeuronSectionBeg[j]; k < PreSynNeuronSectionEnd[j]; ++k){
						NAdditionalSpikesNow[(CurrentQueueIndex + Network[k].DelayinTsteps) % QueueSize].fetch_and_increment();
					}
				}
				//Implementing Causal Learning Rule
				if (PostSynNeuronSectionBeg[j] >= 0){
					for (size_t k = PostSynNeuronSectionBeg[j]; k < PostSynNeuronSectionEnd[j]; ++k){
						if (Network[AuxArray[k]].Plastic == 1){
							//STDP rule
							if (time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart] <= Neurons[j].tmax*onemsbyTstep*1000){
								Network[AuxArray[k]].Weight += ltp;
							}
							else{
								Network[AuxArray[k]].Weight -= ltd;
							}
						}
					}
				}

				//Implementing Metaplasticity by changing tmax
				Neurons[j].tmax += 0.001f;
				
					
			}
		}
	}
}
void CurrentAttenuate::operator() (tbb::blocked_range<int> &Range) const {
	tbb::atomic<long long> *Begin1 = &Iin1[Range.begin()];
	tbb::atomic<long long> *End1 = &Iin1[Range.end()-1] + 1;
	tbb::atomic<long long> *Begin2 = &Iin2[Range.begin()];
	tbb::atomic<long long> *End2 = &Iin2[Range.end() - 1] + 1;

	for (tbb::atomic<long long> *i = Begin1, *j = Begin2; i < End1; ++i, ++j){
		(*i) = (long long)(float(i->load()) * attenFactor1);
		(*j) = (long long)(float(j->load()) * attenFactor2);
	}
}
void SpikeRecord::operator()(tbb::blocked_range<int> &Range) const{
	int RangeBeg = Range.begin();
	int RangeEnd = Range.end();
	for (int j = RangeBeg; j < RangeEnd; ++j){
		if (Vnow[j] == 30.0){
			size_t CurrNeuronSectionBeg = PreSynNeuronSectionBeg[j];
			size_t CurrNeuronSectionEnd = PreSynNeuronSectionEnd[j];
			if (CurrNeuronSectionBeg >= 0)
				for (size_t k = CurrNeuronSectionBeg; k < CurrNeuronSectionEnd; ++k){
					int ThisQueue = (CurrentQueueIndex + Network[k].DelayinTsteps) % QueueSize;
					int ThisLoadingInd = CurrentSpikeLoadingInd[ThisQueue].fetch_and_increment();
					SpikeQueue[ThisQueue][ThisLoadingInd] = k;
				}
		}
	}
}
void InputArgs::IExtFunc(int time, MexMatrix<float> &InpCurr, MexVector<float> &Iext)
{
	//Iext function added by Yamini
	int N = Iext.size();
	int Ninp = InpCurr.nrows();
	for (int i = 0; i < Ninp; ++i){
		Iext[i] = InpCurr(i, time);
	}
	//((int)(time / 0.1))
/*	int N = Iext.size();
	if (time - 0.1 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 9;
	}
	else if (time - 0.8 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 3;
	}
	else{
		for (int i = 0; i < 100*N/2000; ++i)
			Iext[i] = 3;
	}*/
}

void StateVarsOutStruct::initialize(const InternalVars &IntVars) {

	int onemsbyTstep = IntVars.onemsbyTstep;
	int NoOfms = IntVars.NoOfms;
	int StorageStepSize = IntVars.StorageStepSize;
	int Tbeg = IntVars.Time;
	int nSteps = onemsbyTstep * NoOfms;
	int OutputControl = IntVars.OutputControl;
	int beta = IntVars.beta;
	int N = IntVars.N;
	int M = IntVars.M;
	int DelayRange = IntVars.DelayRange;

	int TimeDimLen;  // beta is the time offset from Tbeg which 
	// corresponds to the first valid storage location
	if (StorageStepSize){
		TimeDimLen = (nSteps - beta) / (StorageStepSize*onemsbyTstep) + 1;	//No. of times (StorageStepSize * onemsbyTstep)|time happens
	}
	else{
		TimeDimLen = nSteps;
	}
	if (OutputControl & OutOps::WEIGHT_REQ)
		if (!(IntVars.InterestingSyns.size()))
			this->WeightOut = MexMatrix<float>(TimeDimLen, M);

	if (OutputControl & OutOps::V_REQ)
		this->VOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::U_REQ)
		this->UOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::I_IN_1_REQ)
		this->Iin1Out = MexMatrix<float>(TimeDimLen, N);
	
	if (OutputControl & OutOps::I_IN_2_REQ)
		this->Iin2Out = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::I_RAND_REQ)
		this->IrandOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::GEN_STATE_REQ)
		this->GenStateOut = MexMatrix<uint32_t>(TimeDimLen, 4);

	this->TimeOut = MexVector<int>(TimeDimLen);

	if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
		this->LSTNeuronOut = MexMatrix<int>(TimeDimLen, N);

	if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
		this->LSTSynOut = MexMatrix<int>(TimeDimLen, M);

	if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
		this->SpikeQueueOut = MexVector<MexVector<MexVector<int> > >(TimeDimLen,
			MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>()));

	if (OutputControl & OutOps::CURRENT_QINDS_REQ)
		this->CurrentQIndexOut = MexVector<int>(TimeDimLen);

	if (OutputControl & OutOps::TMAX_REQ)
		this->tmaxOut = MexMatrix<float>(TimeDimLen, N);
}
void OutputVarsStruct::initialize(const InternalVars &IntVars){
	int TimeDimLen;
	int N = IntVars.N;
	int onemsbyTstep = IntVars.onemsbyTstep;
	int NoOfms = IntVars.NoOfms;
	int StorageStepSize = IntVars.StorageStepSize;
	int Tbeg = IntVars.Time;
	int nSteps = onemsbyTstep * NoOfms;
	int OutputControl = IntVars.OutputControl;
	int beta = IntVars.beta;

	if (IntVars.StorageStepSize){
		TimeDimLen = (nSteps - beta) / (StorageStepSize*onemsbyTstep) + 1;	//No. of times (StorageStepSize * onemsbyTstep)|time happens
	}
	else{
		TimeDimLen = nSteps;
	}

	if (OutputControl & OutOps::WEIGHT_REQ)
		if (IntVars.InterestingSyns.size())
			this->WeightOut = MexMatrix<float>(TimeDimLen, IntVars.InterestingSyns.size());
	if (OutputControl & OutOps::I_IN_REQ)
		this->Iin = MexMatrix<float>(TimeDimLen, N);
	if (OutputControl & OutOps::I_TOT_REQ)
		this->Itot = MexMatrix<float>(TimeDimLen, N);
	if (OutputControl & OutOps::TMAX_REQ)
		this->tmaxOut = MexMatrix<float>(TimeDimLen, N);
}
void FinalStateStruct::initialize(const InternalVars &IntVars){
	int OutputControl	= IntVars.OutputControl;
	int DelayRange		= IntVars.DelayRange;
	int onemsbyTstep	= IntVars.onemsbyTstep;
	int N				= IntVars.N;
	int M				= IntVars.M;

	if (OutputControl & OutOps::FINAL_STATE_REQ){
		this->V = MexVector<float>(N);
		this->U = MexVector<float>(N);
		this->Iin1 = MexVector<float>(N);
		this->Iin2 = MexVector<float>(N);
		this->Irand = MexVector<float>(N);
		this->GenState = MexVector<uint32_t>(4);
		this->Weight = MexVector<float>(M);
		this->LSTNeuron = MexVector<int>(N);
		this->LSTSyn = MexVector<int>(M);
		this->SpikeQueue = MexVector<MexVector<int> >(DelayRange*onemsbyTstep, MexVector<int>());
		this->tmax = MexVector<float>(N);
	}
	this->CurrentQIndex = -1;
	this->Time = -1;
}
void InitialStateStruct::initialize(const InternalVars &IntVars){
	int OutputControl = IntVars.OutputControl;
	int DelayRange = IntVars.DelayRange;
	int onemsbyTstep = IntVars.onemsbyTstep;
	int N = IntVars.N;
	int M = IntVars.M;

	if (OutputControl & OutOps::INITIAL_STATE_REQ){
		this->V = MexVector<float>(N);
		this->U = MexVector<float>(N);
		this->Iin1 = MexVector<float>(N);
		this->Iin2 = MexVector<float>(N);
		this->GenState = MexVector<uint32_t>(4);
		this->Weight = MexVector<float>(M);
		this->LSTNeuron = MexVector<int>(N);
		this->LSTSyn = MexVector<int>(M);
		this->SpikeQueue = MexVector<MexVector<int> >(DelayRange*onemsbyTstep, MexVector<int>());
		this->tmax = MexVector<float>(N);
	}
	this->CurrentQIndex = -1;
	this->Time = -1;
}
void InternalVars::DoSparseOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){

	int CurrentInsertPos = (i - beta) / (onemsbyTstep * StorageStepSize);
	int QueueSize = onemsbyTstep * DelayRange;
	// Storing U,V,Iin and Time
	if (OutputControl & OutOps::V_REQ)
		StateOut.VOut[CurrentInsertPos] = V;
	if (OutputControl & OutOps::U_REQ)
		StateOut.UOut[CurrentInsertPos] = U;
	if (OutputControl & OutOps::I_IN_1_REQ)
		for (int j = 0; j < N; ++j)
			StateOut.Iin1Out(CurrentInsertPos, j) = (float)Iin1[j] / (1 << 17);
	if (OutputControl & OutOps::I_IN_2_REQ)
		for (int j = 0; j < N; ++j)
			StateOut.Iin2Out(CurrentInsertPos, j) = (float)Iin2[j] / (1 << 17);
	// Storing Random Current related state vars
	if (OutputControl & OutOps::I_RAND_REQ)
		StateOut.IrandOut[CurrentInsertPos] = Irand;
	if (OutputControl & OutOps::GEN_STATE_REQ){
		BandLimGaussVect::StateStruct TempStateStruct;
		Irand.readstate(TempStateStruct);
		TempStateStruct.Generator1.getstate().ConvertStatetoVect(StateOut.GenStateOut[CurrentInsertPos]);
	}
	StateOut.TimeOut[CurrentInsertPos] = Time;

	// Storing Weights
	if (OutputControl & OutOps::WEIGHT_REQ && InterestingSyns.size()){
		size_t tempSize = InterestingSyns.size();
		for (int j = 0; j < tempSize; ++j)
			OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
	}
	else if (OutputControl & OutOps::WEIGHT_REQ){
		for (int j = 0; j < M; ++j)
			StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
	}

	// Storing tmax
	if (OutputControl & OutOps::TMAX_REQ){
		size_t tempSize = Neurons.size();
		for (int j = 0; j < tempSize; ++j){
			OutVars.tmaxOut(CurrentInsertPos, j) = Neurons[j].tmax;
			StateOut.tmaxOut(CurrentInsertPos, j) = Neurons[j].tmax;
		}
	}



	// Storing Spike Queue related state informations
	if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
		for (int j = 0; j < QueueSize; ++j)
			StateOut.SpikeQueueOut[CurrentInsertPos][j] = SpikeQueue[j];
	if (OutputControl & OutOps::CURRENT_QINDS_REQ)
		StateOut.CurrentQIndexOut[CurrentInsertPos] = CurrentQIndex;

	// Storing last Spiked timings
	if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
		StateOut.LSTNeuronOut[CurrentInsertPos] = LSTNeuron;
	if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
		StateOut.LSTSynOut[CurrentInsertPos] = LSTSyn;

	// Storing Iin
	if (OutputControl & OutOps::I_IN_REQ){
		for (int j = 0; j < N; ++j)
			OutVars.Iin(CurrentInsertPos, j) = (float)(Iin2[j] - Iin1[j]) / (1 << 17);
	}

	// Storing Itot
	if (OutputControl & OutOps::I_TOT_REQ){
		for (int j = 0; j < N; ++j)
			OutVars.Itot(CurrentInsertPos, j) = Iext[j] + StdDev*Irand[j] + I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17);
	}
}
void InternalVars::DoFullOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){
	if (!StorageStepSize){
		int CurrentInsertPos = i - 1;
		int QueueSize = onemsbyTstep * DelayRange;
		// Storing U,V,Iout and Time
		if (OutputControl & OutOps::V_REQ)
			StateOut.VOut[CurrentInsertPos] = V;
		if (OutputControl & OutOps::U_REQ)
			StateOut.UOut[CurrentInsertPos] = U;
		if (OutputControl & OutOps::I_IN_1_REQ)
			for (int j = 0; j < N; ++j)
				StateOut.Iin1Out(CurrentInsertPos, j) = (float)Iin1[j] / (1 << 17);
		if (OutputControl & OutOps::I_IN_2_REQ)
			for (int j = 0; j < N; ++j)
				StateOut.Iin2Out(CurrentInsertPos, j) = (float)Iin2[j] / (1 << 17);
		// Storing Random Curent Related State vars
		if (OutputControl & OutOps::I_RAND_REQ)
			StateOut.IrandOut[CurrentInsertPos] = Irand;
		if (OutputControl & OutOps::GEN_STATE_REQ){
			BandLimGaussVect::StateStruct TempStateStruct;
			Irand.readstate(TempStateStruct);
			TempStateStruct.Generator1.getstate().ConvertStatetoVect(StateOut.GenStateOut[CurrentInsertPos]);
		}
		StateOut.TimeOut[CurrentInsertPos] = Time;

		// Storing Weights
		if (OutputControl & OutOps::WEIGHT_REQ && InterestingSyns.size()){
			size_t tempSize = InterestingSyns.size();
			for (int j = 0; j < tempSize; ++j)
				OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
		}
		else if (OutputControl & OutOps::WEIGHT_REQ){
			for (int j = 0; j < M; ++j)
				StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
		}

		// Storing tmax
		if (OutputControl & OutOps::TMAX_REQ){
			size_t tempSize = Neurons.size();
			for (int j = 0; j < tempSize; ++j){
				OutVars.tmaxOut(CurrentInsertPos, j) = Neurons[j].tmax;
				StateOut.tmaxOut(CurrentInsertPos, j) = Neurons[j].tmax;
			}
		}

		// Storing Spike Queue related state informations
		if (OutputControl & OutOps::SPIKE_QUEUE_REQ)
			for (int j = 0; j < QueueSize; ++j)
				StateOut.SpikeQueueOut[CurrentInsertPos][j] = SpikeQueue[j];
		if (OutputControl & OutOps::CURRENT_QINDS_REQ)
			StateOut.CurrentQIndexOut[CurrentInsertPos] = CurrentQIndex;

		// Storing last Spiked timings
		if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
			StateOut.LSTNeuronOut[CurrentInsertPos] = LSTNeuron;
		if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
			StateOut.LSTSynOut[CurrentInsertPos] = LSTSyn;

		// Storing Iin
		if (OutputControl & OutOps::I_IN_REQ){
			for (int j = 0; j < N; ++j)
				OutVars.Iin(CurrentInsertPos, j) = (float)(Iin2[j] - Iin1[j]) / (1 << 17);
		}

		// Storing Itot
		if (OutputControl & OutOps::I_TOT_REQ){
			for (int j = 0; j < N; ++j)
				OutVars.Itot(CurrentInsertPos, j) = Iext[j] + StdDev*Irand[j] + I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17);
		}
	}
}
void InternalVars::DoSingleStateOutput(SingleStateStruct &FinalStateOut){
	int QueueSize = onemsbyTstep * DelayRange;
	for (int j = 0; j < N; ++j){
		FinalStateOut.Iin1[j] = (float)Iin1[j] / (1 << 17);
		FinalStateOut.Iin2[j] = (float)Iin2[j] / (1 << 17);
	}
	// storing Random curret related state vars
	FinalStateOut.Irand = Irand;
	BandLimGaussVect::StateStruct TempStateStruct;
	Irand.readstate(TempStateStruct);
	TempStateStruct.Generator1.getstate().ConvertStatetoVect(FinalStateOut.GenState);

	FinalStateOut.V = V;
	FinalStateOut.U = U;
	for (int j = 0; j < M; ++j){
		FinalStateOut.Weight[j] = Network[j].Weight;
	}
	for (int j = 0; j < N; ++j){
		FinalStateOut.tmax[j] = Neurons[j].tmax;
	}
	for (int j = 0; j < QueueSize; ++j){
		FinalStateOut.SpikeQueue[j] = SpikeQueue[j];
	}
	FinalStateOut.CurrentQIndex = CurrentQIndex;
	FinalStateOut.LSTNeuron = LSTNeuron;
	FinalStateOut.LSTSyn = LSTSyn;
	FinalStateOut.Time = Time;
}

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVarsStruct &PureOutputs,
	StateVarsOutStruct &StateVarsOutput,
	FinalStateStruct &FinalStateOutput,
	InitialStateStruct &InitialStateOutput
)
{
	// Aliasing Input Arguments Into Appropriate
	// "In Function" state and input variables

	// Initialization and aliasing of All the input / State / parameter variables.
	InternalVars IntVars(InputArguments);

	// Aliasing of Data members in IntVar
	MexVector<Synapse>			&Network				= IntVars.Network;
	MexVector<Neuron>			&Neurons				= IntVars.Neurons;
	MexVector<float>			&Vnow					= IntVars.V;
	MexVector<float>			&Unow					= IntVars.U;
	MexMatrix<float>			&InpCurr				= IntVars.InpCurr;
	MexVector<int>				&InterestingSyns		= IntVars.InterestingSyns;
	atomicLongVect				&Iin1					= IntVars.Iin1;
	atomicLongVect				&Iin2					= IntVars.Iin2;
	BandLimGaussVect			&Irand					= IntVars.Irand;
	MexVector<float>			&Iext					= IntVars.Iext;
	MexVector<MexVector<int> >	&SpikeQueue				= IntVars.SpikeQueue;
	MexVector<int>				&LastSpikedTimeNeuron	= IntVars.LSTNeuron;
	MexVector<int>				&LastSpikedTimeSyn		= IntVars.LSTSyn;
	

	int &NoOfms				= IntVars.NoOfms;
	int &onemsbyTstep		= IntVars.onemsbyTstep;
	int Tbeg				= IntVars.Time;				//Tbeg is just an initial constant, 
	int &time				= IntVars.Time;				//time is the actual changing state variable
	int &DelayRange			= IntVars.DelayRange;
	int &CurrentQueueIndex	= IntVars.CurrentQIndex;
	int &StorageStepSize	= IntVars.StorageStepSize;
	int &OutputControl		= IntVars.OutputControl;
	int &i					= IntVars.i;

	float &ltp				= IntVars.ltp;
	float &ltd				= IntVars.ltd;

	const float &I0			= IntVars.I0;	// Value of the current factor to be multd with weights (constant)
	// calculate value of alpha for filtering
	// alpha = 0 => no filtering
	// alpha = 1 => complete filtering
	const float &alpha		= IntVars.alpha;
	const float &StdDev		= IntVars.StdDev;
	const float &CurrentDecayFactor1	= IntVars.CurrentDecayFactor1;	//Current Decay Factor in the current model (possibly input in future)
	const float &CurrentDecayFactor2	= IntVars.CurrentDecayFactor2;
	const int &StatusDisplayInterval = IntVars.StatusDisplayInterval;

	// other data members. probably derived from inputs or something
	// I think should be a constant. (note that it is possible that 
	// I club some of these with the inputs in future revisions like
	// CurrentDecayFactor

	size_t QueueSize = SpikeQueue.size();
	int nSteps = NoOfms*onemsbyTstep;
	size_t N = InputArguments.Neurons.size(), M = InputArguments.Network.size();			

	
	// VARIOuS ARRAYS USED apart from those in the argument list and Output List.
	// Id like to call them intermediate arrays, required for simulation but are
	// not state, input or output vectors.
	// they are typically of the form of some processed version of an input vector
	// thus they dont change with time and are prima facie not used to generate output

	MexVector<int> AuxArray(M);					    // Auxillary Array that is an indirection between Network
												    // and an array sorted lexicographically by (NEnd, NStart)
	MexVector<size_t> PreSynNeuronSectionBeg(N, -1);	// PreSynNeuronSectionBeg[j] Maintains the list of the 
														// index of the first synapse in Network with NStart = j+1
	MexVector<size_t> PreSynNeuronSectionEnd(N, -1);	// PostSynNeuronSectionEnd[j] Maintains the list of the 
														// indices one greater than index of the last synapse in 
														// Network with NStart = j+1

	MexVector<size_t> PostSynNeuronSectionBeg(N, -1);	// PostSynNeuronSectionBeg[j] Maintains the list of the 
														// index of the first synapse in AuxArray with NEnd = j+1
	MexVector<size_t> PostSynNeuronSectionEnd(N, -1);	// PostSynNeuronSectionEnd[j] Maintains the list of the 
														// indices one greater than index of the last synapse in 
														// AuxArray with NEnd = j+1
	// NAdditionalSpikesNow - A vector of atomic integers that stores the number 
	//              of spikes generated corresponding to each of the sub-vectors 
	//              above. Used to reallocate memory before parallelization of 
	//              write op
	// 
	// CurrentSpikeLoadingInd - A vector of Atomic integers such that the j'th 
	//              element represents the index into SpikeQueue[j] into which 
	//              the spike is to be added by the current loop instance. 
	//              used in parallelizing spike storage

	atomicIntVect NAdditionalSpikesNow(onemsbyTstep * DelayRange);
	atomicIntVect CurrentSpikeLoadingInd(onemsbyTstep * DelayRange);
	
	//----------------------------------------------------------------------------------------------//
	//--------------------------------- Initializing output Arrays ---------------------------------//
	//----------------------------------------------------------------------------------------------//

	StateVarsOutput.initialize(IntVars);
	PureOutputs.initialize(IntVars);
	FinalStateOutput.initialize(IntVars);
	InitialStateOutput.initialize(IntVars);
	
	//---------------------------- Initializing the Intermediate Arrays ----------------------------//
	CountingSort(N, Network, AuxArray);	// Perform counting sort by (NEnd, NStart)
	                                    // to get AuxArray

	// Sectioning the Network and AuxArray Arrays as according to 
	// definition of respective variables above
	PreSynNeuronSectionBeg[Network[0].NStart - 1] = 0;
	PostSynNeuronSectionBeg[Network[AuxArray[0]].NEnd - 1] = 0;
	PreSynNeuronSectionEnd[Network[M - 1].NStart - 1] = M;
	PostSynNeuronSectionEnd[Network[AuxArray[M - 1]].NEnd - 1] = M;

	for (i = 1; i<M; ++i){
		if (Network[i - 1].NStart != Network[i].NStart){
			PreSynNeuronSectionBeg[Network[i].NStart - 1] = i;
			PreSynNeuronSectionEnd[Network[i - 1].NStart - 1] = i;
		}
		if (Network[AuxArray[i - 1]].NEnd != Network[AuxArray[i]].NEnd){
			PostSynNeuronSectionBeg[Network[AuxArray[i]].NEnd - 1] = i;
			PostSynNeuronSectionEnd[Network[AuxArray[i-1]].NEnd - 1] = i;
		}
	}

	// The Structure of iteration below is given below
	/*
		->Iin[t-1] ----------
		|                    \
		|                     \
		|                      >Itemp ---------- * CurrentDecayFactor ---- Iin[t]
		|                     /             \                                |
		|                    /               \                               |
		->SpikeQueue[t-1] ---                 > V,U[t] ----- SpikeQueue[t]   |
		|       |                            /    |                |         |
		|       |               V,U[t-1] ----     |                |         |
		|       |                  |              |                |         |
		|===<===|====<=========<===|==Loop Iter<==|=======<========|===<=====|

		The vector corresponding to the spikes processed in the current 
		iteration is cleared after the calculation of Itemp
	*/
	// Giving Initial State if Asked For
	if (OutputControl & OutOps::INITIAL_STATE_REQ){
		IntVars.DoSingleStateOutput(InitialStateOutput);
	}
	size_t maxSpikeno = 0;
	tbb::affinity_partitioner apCurrentUpdate;
	tbb::affinity_partitioner apNeuronSim;
	int TotalStorageStepSize = (StorageStepSize*onemsbyTstep); // used everywhere
	int epilepsyctr = 0;

	// ------------------------------------------------------------------------------ //
	// ------------------------------ Simulation Loop ------------------------------- //
	// ------------------------------------------------------------------------------ //
	for (i = 1; i<=nSteps; ++i){
		
		time = time + 1;
		InputArgs::IExtFunc(time, InpCurr, Iext);
		Irand.generate();

		// This iteration applies time update equation for internal current
		// in this case, it is just an exponential attenuation
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 3000),
			CurrentAttenuate(Iin1, Iin2, CurrentDecayFactor1, CurrentDecayFactor2));

		size_t QueueSubEnd = SpikeQueue[CurrentQueueIndex].size();
		maxSpikeno += QueueSubEnd;
		// Epilepsy Check
		if (QueueSubEnd > (2*M) / (5)){
			epilepsyctr++;
			if (epilepsyctr > 100){
			#ifdef MEX_LIB
				mexErrMsgTxt("Epileptic shit");
			#elif defined MEX_EXE
				printf("Epilepsy Nyuh!!");
			#endif
				return;
			}
		}

		// This iter calculates Itemp as in above diagram
		if (SpikeQueue[CurrentQueueIndex].size() != 0)
			tbb::parallel_for(tbb::blocked_range<int*>((int*)&SpikeQueue[CurrentQueueIndex][0],
				(int*)&SpikeQueue[CurrentQueueIndex][QueueSubEnd - 1] + 1, 10000), 
				CurrentUpdate(SpikeQueue[CurrentQueueIndex], Network, Iin1, Iin2, LastSpikedTimeSyn, I0, time), apCurrentUpdate);
		SpikeQueue[CurrentQueueIndex].clear();

		// Calculation of V,U[t] from V,U[t-1], Iin = Itemp
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 100), NeuronSimulate(
			Vnow, Unow, Iin1, Iin2, Irand, Iext, Neurons, Network,
			CurrentQueueIndex, QueueSize, onemsbyTstep, time, StdDev, I0, ltp, ltd, PreSynNeuronSectionBeg,
			PreSynNeuronSectionEnd, PostSynNeuronSectionBeg,
			PostSynNeuronSectionEnd, AuxArray, NAdditionalSpikesNow, LastSpikedTimeNeuron), apNeuronSim);

		/////// This is code to extend vectors before they are written to.
		for (int k = 0; k < QueueSize; ++k){
			CurrentSpikeLoadingInd[k] = SpikeQueue[k].size();
			SpikeQueue[k].push_size(NAdditionalSpikesNow[k].load());
		}
		// This is code for storing spikes
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 1000),
			SpikeRecord(
				Vnow,
				Network,
				CurrentQueueIndex, QueueSize,
				PreSynNeuronSectionBeg,
				PreSynNeuronSectionEnd,
				CurrentSpikeLoadingInd,
				SpikeQueue
			));

		for (int k = 0; k < QueueSize; ++k) NAdditionalSpikesNow[k] = 0;
		CurrentQueueIndex = (CurrentQueueIndex + 1) % QueueSize;

		IntVars.DoOutput(StateVarsOutput, PureOutputs);

		// Status Display Section
		if (!(i % StatusDisplayInterval)){
		#ifdef MEX_LIB
			mexPrintf("Completed  %d steps with Total no. of Spikes = %d\n", i, maxSpikeno);
			mexEvalString("drawnow");
		#elif defined MEX_EXE
			printf("Completed  %d steps with Total no. of Spikes = %d\n", i, maxSpikeno);
		#endif
			maxSpikeno = 0;
		}
	}
	if ((OutputControl & OutOps::FINAL_STATE_REQ)){
		IntVars.DoSingleStateOutput(FinalStateOutput);
	}
}


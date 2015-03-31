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
		float AddedCurrent = (I0*Network[CurrentSynapse].Weight)*(1 << 17);
		Iin[Network[CurrentSynapse].NEnd - 1].fetch_and_add((long long)AddedCurrent);
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
			Vnew = Vnow[j] + (Vnow[j] * (0.04f*Vnow[j] + 5.0f) + 140.0f - Unow[j] + (float)Iin[j] / (1 << 17) + Iext[j]) / onemsbyTstep;
			Unew = Unow[j] + (Neurons[j].a*(Neurons[j].b*Vnow[j] - Unow[j])) / onemsbyTstep;
			Vnow[j] = (Vnew > -100)? Vnew: -100;
			Unow[j] = Unew;

			//Implementing Network Computation in case a Neuron has spiked in the current interval
			if (Vnow[j] >= 30.0f){
				Vnow[j] = 30.0f;

				LastSpikedTimeNeuron[j] = time;
				if (PreSynNeuronSectionBeg[j] >= 0)
					for (size_t k = PreSynNeuronSectionBeg[j]; k < PreSynNeuronSectionEnd[j]; ++k)
						NAdditionalSpikesNow[(CurrentQueueIndex + Network[k].DelayinTsteps) % QueueSize].fetch_and_increment();
				//Space to implement any causal Learning Rule
			}
		}
	}
}
void CurrentAttenuate::operator() (tbb::blocked_range<int> &Range) const {
	tbb::atomic<long long> *Begin = &Iin[Range.begin()];
	tbb::atomic<long long> *End = &Iin[Range.end()-1] + 1;
	for (tbb::atomic<long long> * i = Begin; i < End; ++i)
		(*i) = (long long)(float(i->load()) * attenFactor);
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
void InputArgs::IExtFunc(float time, MexVector<float> &Iin)
{
	//((int)(time / 0.1))
	if (time - 0.1 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 200; ++i)
			Iin[i] = 9;
	}
	else if (time - 0.8 <= 0.015){	//((int)(time / 0.1))*
		for (int i = 0; i < 200; ++i)
			Iin[i] = 4;
	}
	else{
		for (int i = 0; i < 200; ++i)
			Iin[i] = 4;
	}
}

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVars &PureOutputs,
	StateVarsOut &StateVarsOutput,
	FinalState &FinalStateOutput
)
{
	// Aliasing Input Arguments Into Appropriate
	// "In Function" state and input variables
	MexVector<Synapse> &Network = InputArguments.Network;
	MexVector<Neuron> &Neurons = InputArguments.Neurons;
	MexVector<float> &Vnow = InputArguments.V;
	MexVector<float> &Unow = InputArguments.U;
	MexVector<int> &InterestingSyns = InputArguments.InterestingSyns;

	// MexVector<float> &Iin = InputArguments.Iin0;	 // This statement is not included because the internal 
	//                                               // state variable for Iin is stored in special atomic vector.
	//                                               // which requires special initialization and not just a reference
	MexVector<MexVector<int> > &SpikeQueue = InputArguments.SpikeQueue;
	MexVector<int> &LastSpikedTimeNeuron = InputArguments.LSTNeuron;
	MexVector<int> &LastSpikedTimeSyn = InputArguments.LSTSyn;

	int NoOfms = InputArguments.NoOfms;
	int onemsbyTstep = InputArguments.onemsbyTstep;
	int Tbeg = InputArguments.Time;
	int DelayRange = InputArguments.DelayRange;
	int CurrentQueueIndex = InputArguments.CurrentQIndex;
	int StorageStepSize = InputArguments.StorageStepSize;
	int OutputControl = InputArguments.OutputControl;
	const int StatusDisplayInterval = InputArguments.StatusDisplayInterval;

	int nSteps = NoOfms*onemsbyTstep;
	size_t N = InputArguments.Neurons.size(), M = InputArguments.Network.size();
	int i;	//Generic Loop Variable (GLV)
	int time = Tbeg;            // The variable that gives the time instant in terms of 
	                            // number of steps
	const int I0 = 1;           // Value of the current factor to be multd with weights
	
	// VARIOuS ARRAYS USED apart from those in the argument list
	atomicLongVect Iin(N);                      // The vector that holds the currents due to internal 
	                                            // synaptic firings
	MexVector<float> Iext(N, 0.0f);             // External Input Current. calculated as a function 
	                                            // of time through function IExtFunc given in the
	                                            // argument list
	//vector<float> LastSpikedTimeSyn(M, -1);       // Last Spiked Timing for the synapses
	//vector<float> LastSpikedTimeNeuron(N, -1);    // Last Spiked Timing for the neurons
	MexVector<int> PrevFiredNeurons;                // Maintains a 'list' of previously fired neuron indices 
	                                                // in sequence of their index
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

	float CurrentDecayFactor;					//Current Decay Factor in the current model
	CurrentDecayFactor = powf(7.0f / 10, 1.0f / onemsbyTstep);
	//----------------------------------------------------------------------------------------------//
	//--------------------------------- Initializing output Arrays ---------------------------------//
	//----------------------------------------------------------------------------------------------//

	int TimeDimLen, beta;  // beta is the time offset from Tbeg which 
	                       // corresponds to the first valid storage location
	if (StorageStepSize){
		beta = StorageStepSize * onemsbyTstep - (Tbeg % (StorageStepSize * onemsbyTstep));
		TimeDimLen = (nSteps - beta) / (StorageStepSize*onemsbyTstep) + 1;	//No. of times (StorageStepSize * onemsbyTstep)|time happens
	}
	else{
		TimeDimLen = nSteps;
	}
	if (OutputControl & OutOps::WEIGHTOUT_REQ)
		if (InterestingSyns.size())
			PureOutputs.WeightOut = MexMatrix<float>(TimeDimLen, InterestingSyns.size());
		else
			StateVarsOutput.WeightOut = MexMatrix<float>(TimeDimLen, M);

	if (OutputControl & OutOps::VOUT_REQ)
		StateVarsOutput.VOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::UOUT_REQ)
		StateVarsOutput.UOut = MexMatrix<float>(TimeDimLen, N);

	if (OutputControl & OutOps::IOUT_REQ)
		StateVarsOutput.IOut = MexMatrix<float>(TimeDimLen, N);

	StateVarsOutput.TimeOut = MexVector<int>(TimeDimLen);

	if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
		StateVarsOutput.LSTNeuronOut = MexMatrix<int>(TimeDimLen, N);

	if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
		StateVarsOutput.LSTSynOut = MexMatrix<int>(TimeDimLen, M);

	if (OutputControl & OutOps::SPIKE_QUEUE_OUT_REQ)
		StateVarsOutput.SpikeQueueOut = MexVector<MexVector<MexVector<int> > >(TimeDimLen,
			MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>()));

	if (OutputControl & OutOps::CURRENT_QINDS_REQ)
		StateVarsOutput.CurrentQIndexOut = MexVector<int>(TimeDimLen);

	//----------------------- Initializing Finaloutput Arrays--------------------------------------//
	if (OutputControl & OutOps::FINAL_STATE_REQ){
		FinalStateOutput.V = MexVector<float>(N);
		FinalStateOutput.U = MexVector<float>(N);
		FinalStateOutput.I = MexVector<float>(N);
		FinalStateOutput.Weight = MexVector<float>(M);
		FinalStateOutput.LSTNeuron = MexVector<int>(N);
		FinalStateOutput.LSTSyn = MexVector<int>(M);
		FinalStateOutput.SpikeQueue = MexVector<MexVector<int> >(DelayRange*onemsbyTstep, MexVector<int>());
		FinalStateOutput.CurrentQIndex = -1;
		FinalStateOutput.Time = -1;
	}

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

	// CurrentQueueIndex (Argument) - The index in SpikeQueue which corresponds to the 
	//              vector of spikes which are to arrive (i.e.  be processed) 
	//              in the current time instant
	// 
	// QueueSize - No of subvectors (i.e. delay slots) in SpikeQueue

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

	// Setting Initial Conditions of V and U
	if (Unow.istrulyempty()){
		Unow.resize(N);
		for (i = 0; i<N; ++i)
			Unow[i] = Neurons[i].b*(Neurons[i].b - 5.0f - sqrt((5.0f - Neurons[i].b)*(5.0f - Neurons[i].b) - 22.4f)) / 0.08f;
	}
	else if (Unow.size() != N){
		// GIVE ERROR MESSAGE HERE
		return;
	}

	if (Vnow.istrulyempty()){
		Vnow.resize(N);
		for (i = 0; i<N; ++i){
			Vnow[i] = (Neurons[i].b - 5.0f - sqrt((5.0f - Neurons[i].b)*(5.0f - Neurons[i].b) - 22.4f)) / 0.08f;
		}
	}
	else if (Vnow.size() != N){
		// GIVE ERROR MESSAGE HERE
		return;
	}

	// Setting Initial Conditions for INPUT CURRENT
	if (InputArguments.I.size() == N){
		InputArgs::IExtFunc((time-1)*0.001f / onemsbyTstep, Iext);
		for (i = 0; i < N; ++i){
			Iin[i] = (long long int)((InputArguments.I[i]-Iext[i]) * (1 << 17));
		}
	}
	else if (InputArguments.I.size()){
		// GIVE ERROR MESSAGE HERE
		return;
	}
	//else{
	//	Iin is already initialized to zero by tbb::zero_allocator<long long>
	//}

	// Setting Initial Conditions of SpikeQueue
	if (SpikeQueue.istrulyempty()){
		SpikeQueue = MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>());
	}
	else if (SpikeQueue.size() != onemsbyTstep * DelayRange){
		// GIVE ERROR MESSAGE HERE
		return;
	}
	size_t QueueSize = SpikeQueue.size();

	// Setting Initial Conditions for LastSpikedTimes
	if (LastSpikedTimeNeuron.istrulyempty()){
		LastSpikedTimeNeuron = MexVector<int>(N,-1);
	}
	else if (LastSpikedTimeNeuron.size() != N){
		//GIVE ERROR MESSAGE HERE
		return;
	}
	if (LastSpikedTimeSyn.istrulyempty()){
		LastSpikedTimeSyn = MexVector<int>(M, -1);
	}
	else if (LastSpikedTimeSyn.size() != M){
		//GIVE ERROR MESSAGE HERE
		return;
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
	size_t maxSpikeno = 0;
	tbb::affinity_partitioner apCurrentUpdate;
	tbb::affinity_partitioner apNeuronSim;
	int TotalStorageStepSize = (StorageStepSize*onemsbyTstep); // used everywhere
	int epilepsyctr = 0;
	for (i = 1; i<=nSteps; ++i){
		
		InputArgs::IExtFunc(time*0.001f/onemsbyTstep, Iext);
		time = time + 1;

		// This iteration applies time update equation for internal current
		// in this case, it is just an exponential attenuation
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 3000),
			CurrentAttenuate(Iin, CurrentDecayFactor));

		size_t QueueSubEnd = SpikeQueue[CurrentQueueIndex].size();
		//maxSpikeno = max((unsigned int)QueueSubEnd, maxSpikeno);
		maxSpikeno += QueueSubEnd;
		if (QueueSubEnd > (2*M) / (5))
		{
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
				CurrentUpdate(SpikeQueue[CurrentQueueIndex], Network, Iin, LastSpikedTimeSyn, I0, time), apCurrentUpdate);
		SpikeQueue[CurrentQueueIndex].clear();

		// Calculation of V,U[t] from V,U[t-1], Iin = Itemp
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 100), NeuronSimulate(Vnow, Unow, Iin, Iext, Neurons, Network,
			CurrentQueueIndex, QueueSize, onemsbyTstep, CurrentDecayFactor, time, PreSynNeuronSectionBeg, 
			PreSynNeuronSectionEnd, NAdditionalSpikesNow, LastSpikedTimeNeuron), apNeuronSim);

		/////// This is code to extend vectors before they are written to.
		for (int k = 0; k < QueueSize; ++k){
			CurrentSpikeLoadingInd[k] = SpikeQueue[k].size();
			SpikeQueue[k].push_size(NAdditionalSpikesNow[k].load());
		}

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

		//Storing outputs if necessary
		int CurrentInsertPos = (StorageStepSize) ? (i - beta) / TotalStorageStepSize : i - 1;
		bool ShouldStoreNow = (StorageStepSize) ? !(time % TotalStorageStepSize) : true;
		if (ShouldStoreNow){
			// Storing U,V,Iout and Time
			if (OutputControl & OutOps::VOUT_REQ)
				StateVarsOutput.VOut[CurrentInsertPos] = Vnow;
			if (OutputControl & OutOps::UOUT_REQ)
				StateVarsOutput.UOut[CurrentInsertPos] = Unow;
			if (OutputControl & OutOps::IOUT_REQ)
				for (int j = 0; j < N; ++j)
					StateVarsOutput.IOut(CurrentInsertPos, j) = Iext[j] + (float)Iin[j] / (1 << 17);
			StateVarsOutput.TimeOut[CurrentInsertPos] = time;

			// Storing Weights
			if (OutputControl & OutOps::WEIGHTOUT_REQ && InterestingSyns.size()){
				size_t tempSize = InterestingSyns.size();
				for (int j = 0; j < tempSize; ++j)
					PureOutputs.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
			}
			else if (OutputControl & OutOps::WEIGHTOUT_REQ){
				for (int j = 0; j < M; ++j)
					StateVarsOutput.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
			}

			// Storing Spike Queue related state informations
			if (OutputControl & OutOps::SPIKE_QUEUE_OUT_REQ)
				for (int j = 0; j < QueueSize; ++j)
					StateVarsOutput.SpikeQueueOut[CurrentInsertPos][j] = SpikeQueue[j];
			if (OutputControl & OutOps::CURRENT_QINDS_REQ)
				StateVarsOutput.CurrentQIndexOut[CurrentInsertPos] = CurrentQueueIndex;

			// Storing last Spiked timings
			if (OutputControl & OutOps::LASTSPIKED_NEU_REQ)
				StateVarsOutput.LSTNeuronOut[CurrentInsertPos] = LastSpikedTimeNeuron;
			if (OutputControl & OutOps::LASTSPIKED_SYN_REQ)
				StateVarsOutput.LSTSynOut[CurrentInsertPos] = LastSpikedTimeSyn;
			
		}

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
		for (int j = 0; j < N; ++j){
			FinalStateOutput.I[j] = Iext[j] + (float)Iin[j] / (1 << 17);
		}
		FinalStateOutput.V = Vnow;
		FinalStateOutput.U = Unow;
		for (int j = 0; j < M; ++j){
			FinalStateOutput.Weight[j] = Network[j].Weight;
		}
		for (int j = 0; j < QueueSize; ++j){
			FinalStateOutput.SpikeQueue[j] = SpikeQueue[j];
		}
		FinalStateOutput.CurrentQIndex = CurrentQueueIndex;
		FinalStateOutput.LSTNeuron = LastSpikedTimeNeuron;
		FinalStateOutput.LSTSyn = LastSpikedTimeSyn;
		FinalStateOutput.Time = time; 
	}
}


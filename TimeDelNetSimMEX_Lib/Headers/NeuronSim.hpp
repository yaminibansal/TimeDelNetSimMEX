#ifndef NEURONSIM_HPP
#define NEURONSIM_HPP
#include "Network.hpp"
#include "MexMem.hpp"
#include <mex.h>
#include <matrix.h>
#include <xutility>
#include <vector>
#include <tbb\atomic.h>
#include <tbb\parallel_for.h>

#define DEFAULT_STORAGE_STEP 500
#define DEFAULT_STATUS_DISPLAY_STEP 400
using namespace std;

struct OutOps{
	enum {
		VOUT_REQ = 0x0001,
		IOUT_REQ = 0x0002,
		TIME_REQ = 0x0004,
		UOUT_REQ = 0x0008,
		WEIGHTOUT_REQ = 0x0010,
		CURRENT_QINDS_REQ = 0x0020,
		SPIKE_QUEUE_OUT_REQ = 0x0040,
		LASTSPIKED_NEU_REQ = 0x0080,
		LASTSPIKED_SYN_REQ = 0x0100,
		INITIAL_STATE_REQ = 0x0200,
		FINAL_STATE_REQ = 0x8000,
	};
};

typedef vector<tbb::atomic<long long>, tbb::zero_allocator<long long> > atomicLongVect;
typedef vector<tbb::atomic<int>, tbb::zero_allocator<int> > atomicIntVect;

struct CurrentUpdate
{
	MexVector<int> & SpikeList;
	MexVector<Synapse> &Network;
	atomicLongVect &Iin;
	MexVector<int> &LastSpikedTimeSyn;
	int time;
	float I0;
	CurrentUpdate(MexVector<int> &SpikeList_,
		MexVector<Synapse> &Network_,
		atomicLongVect &Iin_,
		MexVector<int> &LastSpikedTimeSyn_, float I0_, int time_) :
		SpikeList(SpikeList_),
		Network(Network_),
		Iin(Iin_),
		LastSpikedTimeSyn(LastSpikedTimeSyn_),
		I0(I0_),
		time(time_){};
	void operator () (const tbb::blocked_range<int*> &BlockedRange) const;
};
struct NeuronSimulate{
	MexVector<float> &Vnow;
	MexVector<float> &Unow;
	atomicLongVect &Iin;
	MexVector<float> &Iext;
	MexVector<Neuron> &Neurons;
	MexVector<Synapse> &Network;
	int CurrentQueueIndex, QueueSize, onemsbyTstep, time;
	float CurrentDecayFactor;
	MexVector<size_t> &PreSynNeuronSectionBeg;
	MexVector<size_t> &PreSynNeuronSectionEnd;
	atomicIntVect &NAdditionalSpikesNow;
	MexVector<int> &LastSpikedTimeNeuron;

	NeuronSimulate(
		MexVector<float> &Vnow_,
		MexVector<float> &Unow_,
		atomicLongVect &Iin_,
		MexVector<float> &Iext_,
		MexVector<Neuron> &Neurons_,
		MexVector<Synapse> &Network_,
		int CurrentQueueIndex_, int QueueSize_, int onemsbyTstep_,
		float CurrentDecayFactor_, int time_,
		MexVector<size_t> &PreSynNeuronSectionBeg_,
		MexVector<size_t> &PreSynNeuronSectionEnd_,
		atomicIntVect &NAdditionalSpikesNow_,
		MexVector<int> &LastSpikedTimeNeuron_
		) :
		Vnow(Vnow_),
		Unow(Unow_),
		Iin(Iin_),
		Iext(Iext_),
		Neurons(Neurons_),
		Network(Network_),
		CurrentQueueIndex(CurrentQueueIndex_), QueueSize(QueueSize_), onemsbyTstep(onemsbyTstep_),
		CurrentDecayFactor(CurrentDecayFactor_), time(time_),
		PreSynNeuronSectionBeg(PreSynNeuronSectionBeg_),
		PreSynNeuronSectionEnd(PreSynNeuronSectionEnd_),
		NAdditionalSpikesNow(NAdditionalSpikesNow_),
		LastSpikedTimeNeuron(LastSpikedTimeNeuron_)
	{};
	void operator() (tbb::blocked_range<int> &Range) const;
};

struct SpikeRecord{
	MexVector<float> &Vnow;
	MexVector<Synapse> &Network;
	int CurrentQueueIndex, QueueSize;
	MexVector<size_t> &PreSynNeuronSectionBeg;
	MexVector<size_t> &PreSynNeuronSectionEnd;
	atomicIntVect &CurrentSpikeLoadingInd;
	MexVector<MexVector<int> > &SpikeQueue;

	SpikeRecord(
		MexVector<float> &Vnow_,
		MexVector<Synapse> &Network_,
		int CurrentQueueIndex_, int QueueSize_,
		MexVector<size_t> &PreSynNeuronSectionBeg_,
		MexVector<size_t> &PreSynNeuronSectionEnd_,
		atomicIntVect &CurrentSpikeLoadingInd_,
		MexVector<MexVector<int> > &SpikeQueue_
		) :
		Vnow(Vnow_),
		Network(Network_),
		CurrentQueueIndex(CurrentQueueIndex_), QueueSize(QueueSize_), 
		PreSynNeuronSectionBeg(PreSynNeuronSectionBeg_),
		PreSynNeuronSectionEnd(PreSynNeuronSectionEnd_),
		CurrentSpikeLoadingInd(CurrentSpikeLoadingInd_),
		SpikeQueue(SpikeQueue_){}

	void operator()(tbb::blocked_range<int> &Range) const;
};

struct CurrentAttenuate{
	atomicLongVect &Iin;
	float attenFactor;

	CurrentAttenuate(
		atomicLongVect &Iin_,
		float attenFactor_) :
		Iin(Iin_),
		attenFactor(attenFactor_) {}

	void operator() (tbb::blocked_range<int> &Range) const; 
};

// Incomplete declarations
struct InputArgs;
struct StateVarsOutStruct;
struct SingleStateStruct;
struct FinalStateStruct;
struct InitialStateStruct;
struct OutputVarsStruct;


struct InputArgs{
	static void IExtFunc(float, MexVector<float> &);
	MexVector<Synapse> Network;
	MexVector<Neuron> Neurons;
	MexVector<int> InterestingSyns;
	MexVector<float> V;
	MexVector<float> U;
	MexVector<float> I;
	MexVector<MexVector<int> > SpikeQueue;
	MexVector<int> LSTNeuron;
	MexVector<int> LSTSyn;
	int onemsbyTstep;
	int NoOfms;
	int DelayRange;
	int Time;
	int CurrentQIndex;
	int OutputControl;
	int StorageStepSize;
	int StatusDisplayInterval;

	InputArgs() :
		Network(),
		Neurons(),
		InterestingSyns(),
		V(),
		U(),
		I(),
		SpikeQueue(),
		LSTNeuron(),
		LSTSyn() {}
};

struct InternalVars{
	int N;
	int M;
	int i;	//This is the most important loop index that is definitely a state variable
			// and plays a crucial role in deciding the index into which the output must be performed
	int Time;	// must be initialized befor beta
	int beta;	// This is another parameter that plays a rucial role when storing sparsely.
				// It is the first value of i for which the sparse storage must be done.
				// goes from 1 to StorageStepSize * onemsbyTstep
	int onemsbyTstep;
	int NoOfms;
	int DelayRange;
	int CurrentQIndex;
	int OutputControl;
	int StorageStepSize;
	const int StatusDisplayInterval;

	MexVector<Synapse> &Network;
	MexVector<Neuron> &Neurons;
	MexVector<int> &InterestingSyns;
	MexVector<float> &V;
	MexVector<float> &U;
	atomicLongVect Iin;
	MexVector<float> Iext;
	MexVector<MexVector<int> > &SpikeQueue;
	MexVector<int> &LSTNeuron;
	MexVector<int> &LSTSyn;

	InternalVars(InputArgs &IArgs) :
		N(IArgs.Neurons.size()),
		M(IArgs.Network.size()),
		i(0),
		Time(IArgs.Time),
		// beta defined conditionally below
		CurrentQIndex(IArgs.CurrentQIndex),
		OutputControl(IArgs.OutputControl),
		StorageStepSize(IArgs.StorageStepSize),
		StatusDisplayInterval(IArgs.StatusDisplayInterval),
		Network(IArgs.Network),
		Neurons(IArgs.Neurons),
		InterestingSyns(IArgs.InterestingSyns),
		V(IArgs.V),
		U(IArgs.U),
		Iin(N),	// Iin is defined separately as an atomic vect.
		Iext(N, 0.0f),
		SpikeQueue(IArgs.SpikeQueue),
		LSTNeuron(IArgs.LSTNeuron),
		LSTSyn(IArgs.LSTSyn),
		onemsbyTstep(IArgs.onemsbyTstep),
		NoOfms(IArgs.NoOfms),
		DelayRange(IArgs.DelayRange) {
		// Setting value of beta
		if (StorageStepSize)
			beta = (onemsbyTstep * NoOfms) - StorageStepSize % (onemsbyTstep * NoOfms);
		else
			beta = 0;

		// Setting Initial Conditions of V and U
		if (U.istrulyempty()){
			U.resize(N);
			for (int i = 0; i<N; ++i)
				U[i] = Neurons[i].b*(Neurons[i].b - 5.0f - sqrt((5.0f - Neurons[i].b)*(5.0f - Neurons[i].b) - 22.4f)) / 0.08f;
		}
		else if (U.size() != N){
			// GIVE ERROR MESSAGE HERE
			return;
		}

		if (V.istrulyempty()){
			V.resize(N);
			for (int i = 0; i<N; ++i){
				V[i] = (Neurons[i].b - 5.0f - sqrt((5.0f - Neurons[i].b)*(5.0f - Neurons[i].b) - 22.4f)) / 0.08f;
			}
		}
		else if (V.size() != N){
			// GIVE ERROR MESSAGE HEREx
			return;
		}

		// Setting Initial Conditions for INTERNAL CURRENT
		if (IArgs.I.size() == N){
			InputArgs::IExtFunc((IArgs.Time - 1)*0.001f / onemsbyTstep, Iext);
			for (int i = 0; i < N; ++i){
				Iin[i] = (long long int)((IArgs.I[i] - Iext[i]) * (1 << 17));
			}
		}
		else if (IArgs.I.size()){
			// GIVE ERROR MESSAGE HERE
			return;
		}
		//else{
		//	I is already initialized to zero by tbb::zero_allocator<long long>
		//}

		// Setting Initial Conditions of SpikeQueue
		if (SpikeQueue.istrulyempty()){
			SpikeQueue = MexVector<MexVector<int> >(onemsbyTstep * DelayRange, MexVector<int>());
		}
		else if (SpikeQueue.size() != onemsbyTstep * DelayRange){
			// GIVE ERROR MESSAGE HERE
			return;
		}

		// Setting Initial Conditions for LSTs
		if (LSTNeuron.istrulyempty()){
			LSTNeuron = MexVector<int>(N, -1);
		}
		else if (LSTNeuron.size() != N){
			//GIVE ERROR MESSAGE HERE
			return;
		}
		if (LSTSyn.istrulyempty()){
			LSTSyn = MexVector<int>(M, -1);
		}
		else if (LSTSyn.size() != M){
			//GIVE ERROR MESSAGE HERE
			return;
		}
	}
	void DoStateOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars){
		DoFullOutput(StateOut, OutVars);
		if (StorageStepSize && !(Time % (StorageStepSize*onemsbyTstep))){
			DoSparseOutput(StateOut, OutVars);
		}
	}
	void DoSingleStateOutput(SingleStateStruct &FinalStateOut);
private:
	void DoSparseOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars);
	void DoFullOutput(StateVarsOutStruct &StateOut, OutputVarsStruct &OutVars);
};

struct OutputVarsStruct{
	MexMatrix<float> WeightOut;

	OutputVarsStruct() :
		WeightOut() {}

	void initialize(const InternalVars &);
};

struct StateVarsOutStruct{
	MexMatrix<float> WeightOut;
	MexMatrix<float> VOut;
	MexMatrix<float> UOut;
	MexMatrix<float> IOut;
	MexVector<int> TimeOut;
	MexVector<MexVector<MexVector<int> > > SpikeQueueOut;
	MexVector<int> CurrentQIndexOut;
	MexMatrix<int> LSTNeuronOut;
	MexMatrix<int> LSTSynOut;

	StateVarsOutStruct() :
		WeightOut(),
		VOut(),
		UOut(),
		IOut(),
		TimeOut(),
		SpikeQueueOut(),
		CurrentQIndexOut(),
		LSTNeuronOut(),
		LSTSynOut() {}

	void initialize(const InternalVars &);
};
struct SingleStateStruct{
	MexVector<float> Weight;
	MexVector<float> V;
	MexVector<float> U;
	MexVector<float> I;
	MexVector<MexVector<int > > SpikeQueue;
	MexVector<int> LSTNeuron;
	MexVector<int> LSTSyn;
	int Time;
	int CurrentQIndex;

	SingleStateStruct() :
		Weight(),
		V(),
		U(),
		I(),
		SpikeQueue(),
		LSTNeuron(),
		LSTSyn() {}
	virtual void initialize(const InternalVars &) {}
};
struct FinalStateStruct : public SingleStateStruct{
	void initialize(const InternalVars &);
};

struct InitialStateStruct : public SingleStateStruct{
	void initialize(const InternalVars &);
};

void CountingSort(int N, MexVector<Synapse> &Network, MexVector<int> &indirection);

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVarsStruct &PureOutputs,
	StateVarsOutStruct &StateVarsOutput,
	FinalStateStruct &FinalStateOutput,
	InitialStateStruct &InitalStateOutput);

#endif
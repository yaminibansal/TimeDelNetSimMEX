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
		FINAL_STATE_REQ = 0x8000
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
	MexVector<int> &PreSynNeuronSectionBeg;
	MexVector<int> &PreSynNeuronSectionEnd;
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
		MexVector<int> &PreSynNeuronSectionBeg_,
		MexVector<int> &PreSynNeuronSectionEnd_,
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
	MexVector<int> &PreSynNeuronSectionBeg;
	MexVector<int> &PreSynNeuronSectionEnd;
	atomicIntVect &CurrentSpikeLoadingInd;
	MexVector<MexVector<int> > &SpikeQueue;

	SpikeRecord(
		MexVector<float> &Vnow_,
		MexVector<Synapse> &Network_,
		int CurrentQueueIndex_, int QueueSize_,
		MexVector<int> &PreSynNeuronSectionBeg_,
		MexVector<int> &PreSynNeuronSectionEnd_,
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

struct OutputVars{
	MexMatrix<float> WeightOut;

	OutputVars() :
		WeightOut() {}
};

struct StateVarsOut{
	MexMatrix<float> WeightOut;
	MexMatrix<float> VOut;
	MexMatrix<float> UOut;
	MexMatrix<float> IOut;
	MexVector<int> TimeOut;
	MexVector<MexVector<MexVector<int> > > SpikeQueueOut;
	MexVector<int> CurrentQIndexOut;
	MexMatrix<int> LSTNeuronOut;
	MexMatrix<int> LSTSynOut;

	StateVarsOut() :
		WeightOut(),
		VOut(),
		UOut(),
		IOut(),
		TimeOut(),
		SpikeQueueOut(),
		CurrentQIndexOut(),
		LSTNeuronOut(),
		LSTSynOut() {}
};

struct FinalState{
	MexVector<float> Weight;
	MexVector<float> V;
	MexVector<float> U;
	MexVector<float> I;
	MexVector<MexVector<int > > SpikeQueue;
	MexVector<int> LSTNeuron;
	MexVector<int> LSTSyn;
	int Time;
	int CurrentQIndex;

	FinalState() :
		Weight(),
		V(),
		U(),
		I(),
		SpikeQueue(),
		LSTNeuron(),
		LSTSyn() {}
};

void CountingSort(int N, MexVector<Synapse> &Network, MexVector<int> &indirection);

void SimulateParallel(
	InputArgs &&InputArguments,
	OutputVars &PureOutputs,
	StateVarsOut &StateVarsOutput,
	FinalState &FinalStateOutput);

#endif
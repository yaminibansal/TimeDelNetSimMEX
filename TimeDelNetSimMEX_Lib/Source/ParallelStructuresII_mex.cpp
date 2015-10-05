#include <vector>
#include <random>
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
		if (LastSpikedTimeSyn[CurrentSynapse] != time){
			float AddedCurrent = (Network[CurrentSynapse].Weight)*(1 << 17);
			LastSpikedTimeSyn[CurrentSynapse] = time;
			Iin1[Network[CurrentSynapse].NEnd - 1].fetch_and_add((long long)AddedCurrent);
			//Iin2[Network[CurrentSynapse].NEnd - 1].fetch_and_add((long long)AddedCurrent);
		}
	}
}

void NeuronSimulate::operator() (tbb::blocked_range<int> &Range) const{
	int RangeBeg = Range.begin();
	int RangeEnd = Range.end();
	int N = RangeEnd;
	int Tref = 0 * onemsbyTstep;
	
	//For all neurons, generate stochastic spike dependent on Iext[j]
	for (int j = RangeBeg; j < RangeEnd; ++j){
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 1);
		float rand_n;
		rand_n = dis(gen);
		if (rand_n < (float)(Iext[j] * 0.001 / onemsbyTstep)){
			Vnow[j] = rand_n; //Vnow is useless in this implementation
			LastSpikedTimeNeuron[j] = time;
			if (j >= Ninp) LastSpikedTimeNeuron[N - 1] = time; //Inhibitory neuron spike if output neuron spike
		}

		float eta, check1, check2;

		//For output neurons, update the membrane potential depending on Inhibition
		if (j >= Ninp && j!=N-1){
			//Iin2[j] = (1<<17)*(Neurons[j].tmax-Neurons[j].d;
			if (LastSpikedTimeNeuron[Network[AuxArray[PostSynNeuronSectionEnd[j] - 1]].NStart - 1] != -1){
				Vnow[j] = ((float)Iin1[j]/(1<<17) + (Neurons[j].tmax - Neurons[j].d) + Network[AuxArray[PostSynNeuronSectionEnd[j] - 1]].Weight*exp(-(float)(time - LastSpikedTimeNeuron[Network[AuxArray[PostSynNeuronSectionEnd[j] - 1]].NStart - 1]) / (Neurons[j].b * 1000 * onemsbyTstep)));
			}
			else{
				Vnow[j] = ((float)Iin1[j]/(1<<17) + (Neurons[j].tmax - Neurons[j].d));
			}
			/*if (PostSynNeuronSectionBeg[j] >= 0){
				for (size_t k = PostSynNeuronSectionBeg[j]; k < PostSynNeuronSectionEnd[j]; ++k){
					if (Network[AuxArray[k]].NStart != N){
						if (LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] != -1 && (time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1]) <= (int)(1000 * onemsbyTstep*Neurons[j].a + 0.5f) && time != LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1]){
							Vnow[j] += Network[AuxArray[k]].Weight;
						}
					}
					else
					{
						if (LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] != -1){
							Vnow[j] += Network[AuxArray[k]].Weight*exp(-(float)(time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1]) / (Neurons[j].b * 1000 * onemsbyTstep));
						}
					}
				}
			}*/
			Iext[j] = exp(Vnow[j]);
			//tmax (Neuron threshold update): Threshold decreases uniformly at all times
			//Efficiency
			
			if (SpikeTimes[j].size() > 0){
				eta = 1 / (float)(Neurons[j].c+SpikeTimes[j].size());
				//eta = 1;
			}
			else
			{
				eta = 0;
			}

			//Neurons[j].tmax += - eta*0.01;
		}

		//If spike:
		if (LastSpikedTimeNeuron[j] == time){
			//Store in SpikeBuffer and SpikeTimes
			SpikeTimes[j].push_back(time);
			
			//NEW //This part is not required for this implementation as we aren't using the spike buffer
			if (PreSynNeuronSectionBeg[j] >= 0 && j!=N-1){
				for (size_t k = PreSynNeuronSectionBeg[j]; k < PreSynNeuronSectionEnd[j]; ++k){
					for (int epsp_time = 1; epsp_time <= DelayRange*onemsbyTstep; epsp_time++){
						NAdditionalSpikesNow[(CurrentQueueIndex + epsp_time) % QueueSize].fetch_and_increment();
					}
				}
			}//NEW*/

			// Causal STDP
			if (PostSynNeuronSectionBeg[j] >= 0 && j >= Ninp){
				int delT, sigma;
				for (size_t k = PostSynNeuronSectionBeg[j]; k < PostSynNeuronSectionEnd[j]; ++k){
					if (Network[AuxArray[k]].Plastic == 1){
						delT = time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1];
						//STDP rule
						sigma = (int)(Neurons[j].a*onemsbyTstep * 1000 + 0.5f);
						if (delT <= sigma && LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] != -1){
							Network[AuxArray[k]].Weight += eta*(ltp*exp(-Network[AuxArray[k]].Weight) - 1);
						}
						else{
							//if (Network[AuxArray[k]].Weight - ltd >= 0){
								Network[AuxArray[k]].Weight -= eta*ltd;
							//}
						}
						if (Network[AuxArray[k]].Weight < 0){
							Network[AuxArray[k]].Weight = 0;
						}
						if (Network[AuxArray[k]].Weight > log(ltp)){
							Network[AuxArray[k]].Weight = log(ltp);
						}
					}
				}
				//tmax (Neuron threshold update): If there is a spike, it increases
				float eta0;
				if (SpikeTimes[N-1].size() > 0){
					eta0 = 1 / (float)(Neurons[j].c+SpikeTimes[N-1].size());
				}
				else
				{
					eta0 = 0;
				}
				if (j != N - 1){
					for (int lat = Ninp; lat < N - 1; lat++){
						if (j == lat){
							Neurons[lat].tmax += eta0*(exp(-Neurons[lat].tmax) - 1);
						}
						else {
							Neurons[lat].tmax -= eta0;
						}

						//if (Neurons[lat].tmax < 0) Neurons[lat].tmax = 0;
						//if (Neurons[lat].tmax > log(ltp)) Neurons[lat].tmax = log(ltp);
					}
					//Neurons[j].tmax += eta*ltp*exp(-Neurons[j].tmax);
				}
			}
		}		
		/* //If input neuron, check time has become arrival time from Iext 
		if (j < Ninp){
			if (Iext[j] == 1){
				Vnow[j] = 4*Neurons[j].c;
			}
			else{
				Vnow[j] = Neurons[j].d;
			}
		}
		//Else, continue membrane potential calculation
		else{
			if (Vnow[j] == 4 * Neurons[j].c || time - LastSpikedTimeNeuron[j] <= Tref){
				Vnow[j] = Neurons[j].d;
			}
			else{
				//Implementing LIF differential equation
				float Vnew, Unew, k1, k2;
				
				k1 = (I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17) + Iext[j] + StdDev*Irand[j]) / Neurons[j].b - Neurons[j].a*(Vnow[j] - Neurons[j].d) / Neurons[j].b;
				k2 = (I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17) + Iext[j] + StdDev*Irand[j]) / Neurons[j].b - Neurons[j].a*(Vnow[j] + k1*0.001f / onemsbyTstep - Neurons[j].d) / Neurons[j].b;
				Vnew = Vnow[j] + 0.001f / onemsbyTstep*(k1 + k2) / 2;
				Vnow[j] = (Vnew > -100) ? Vnew : -100;

				if (Vnow[j] >= Neurons[j].c){
					Vnow[j] = 4 * Neurons[j].c;
				}
			}
		}

		//If spike....
		if (Vnow[j] == 4 * Neurons[j].c){
			SpikeTimes[j].push_back(time);

			int delT, delTmin, tmaxtmp;

			LastSpikedTimeNeuron[j] = time;
			if (PreSynNeuronSectionBeg[j] >= 0){
				for (size_t k = PreSynNeuronSectionBeg[j]; k < PreSynNeuronSectionEnd[j]; ++k){
					NAdditionalSpikesNow[(CurrentQueueIndex + Network[k].DelayinTsteps) % QueueSize].fetch_and_increment();

					//Implementing anti-causal learning rule
					if (Network[k].Plastic == 1 && LastSpikedTimeNeuron[Network[k].NEnd - 1] != -1 && Network[k].STDPcount){
						Network[k].STDPcount = false;
						delT = time - LastSpikedTimeNeuron[Network[k].NEnd - 1];
						//tmaxtmp = (int)(Neurons[j].tmax*onemsbyTstep * 1000 + 0.5f)*ltp/(ltd*0.9);
						tmaxtmp = (int)(0.0168*onemsbyTstep * 1000 + 0.5f);
						if (delT <= tmaxtmp && Network[k].Weight - ltd >= 0 ){
							Network[k].Weight -= ltd;
						}
					}


				}
			}

			//Implementing Causal Learning Rule
			if (PostSynNeuronSectionBeg[j] >= 0){
				delTmin = 0.3 * 1000 * onemsbyTstep;
				for (size_t k = PostSynNeuronSectionBeg[j]; k < PostSynNeuronSectionEnd[j]; ++k){
					//Setting STDPcount for all incoming synapses to 1
					Network[AuxArray[k]].STDPcount = true;
					if (Network[AuxArray[k]].Plastic == 1 && LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] != -1){
						delT = time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1];
						//STDP rule
						//tmaxtmp = (int)(Neurons[j].tmax*onemsbyTstep * 1000 + 0.5f);
						tmaxtmp = (int)(0.01*onemsbyTstep * 1000 + 0.5f);
						if (delT <= tmaxtmp){
							Network[AuxArray[k]].Weight += ltp;
						}
						/*else{
							//if (Network[AuxArray[k]].Weight - ltd >= 0){
							if (Network[AuxArray[k]].Weight - ltd >= 0 && delT < 0.2 * 1000 * onemsbyTstep){
								Network[AuxArray[k]].Weight -= ltd;
							}
						}*__/	
					}
					if (delTmin > delT){
						delTmin = delT;
					}
					/*if (Network[AuxArray[k]].Plastic == 1 && LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] == -1) {
						if (Network[AuxArray[k]].Weight - ltd >= 0){
							Network[AuxArray[k]].Weight -= ltd;
						}
					}*__/	
				}
				//Implementing Metaplasticity by changing tmax (Assuming all delT in one pattern are the same)
				/*float N_s = SpikeTimes[j].size();
				float N_t = 5;
				float temp1 = (1 - exp(-(N_s - 1) / N_t)) / (1 - exp(-1 / N_t));
				float temp2 = (1 - exp(-(N_s) / N_t)) / (1 - exp(-1 / N_t));
				if (delT < 0.2 * 1000 * onemsbyTstep){
					if (N_s == 1){
						Neurons[j].tmax = delTmin*0.001 / onemsbyTstep;
					}
					else{
						Neurons[j].tmax = (exp(-1 / N_t)*Neurons[j].tmax*temp1 + delTmin*0.001 / onemsbyTstep) / (temp2);
					}
				}*/

				/*if (SpikeTimes[j].size() == 1){
				Neurons[j].tmax = ((SpikeTimes[j].size() - 1)*Neurons[j].tmax + delT*0.001 / onemsbyTstep) / (SpikeTimes[j].size());
				}
				else{
				//Neurons[j].tmax = ((SpikeTimes[j].size() - 1)*Neurons[j].tmax + 2 * delTmin*0.001 / onemsbyTstep) / (3 * SpikeTimes[j].size());
				Neurons[j].tmax = (Neurons[j].tmax + 2 * delT*0.001 / onemsbyTstep) / 3;
				}*__/
				//}
				//if (Neurons[j].tmax > 1.05*delT*0.001 / onemsbyTstep){
				//	Neurons[j].tmax = 1.05*delT*0.001 / onemsbyTstep;
				//}


			}
			

			//Implementing artificial lateral inhibition
			if (j >= Ninp) {
				for (int nlat = Ninp; nlat < N; ++nlat){
					if (nlat != j){
						Vnow[nlat] = Neurons[j].d;
					}
				}
				break;
			}

		}
		/*if (Vnow[j] == 4*Neurons[j].c || time - LastSpikedTimeNeuron[j] <= Tref){ 
			Vnow[j] = Neurons[j].d;
		}
		else{
			//Implementing LIF differential equation
			float Vnew, Unew, k1, k2;
			int delT, delTmin, tmaxtmp;
			k1 = (I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17) + Iext[j] + StdDev*Irand[j]) / Neurons[j].b - Neurons[j].a*(Vnow[j] - Neurons[j].d) / Neurons[j].b;
			k2 = (I0*(float)(Iin2[j] - Iin1[j]) / (1 << 17) + Iext[j] + StdDev*Irand[j]) / Neurons[j].b - Neurons[j].a*(Vnow[j] + k1*0.001f / onemsbyTstep - Neurons[j].d) / Neurons[j].b;
			Vnew = Vnow[j] + 0.001f/onemsbyTstep*(k1+k2)/2;
			Vnow[j] = (Vnew > -100)? Vnew:-100;

			//Implementing Poisson Spike Train
			/*float firingrate[2];
			firingrate[0] = Iext[0];
			firingrate[1] = Iext[1];
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0, 1);
			float rand_n;
			rand_n = dis(gen);
			if (rand_n < firingrate[j] * 0.001 / onemsbyTstep)
				Vnow[j] = Neurons[j].c;//////


			//Implementing Network Computation in case a Neuron has spiked in the current interval
			if (Vnow[j] >= Neurons[j].c){
				Vnow[j] = 4 * Neurons[j].c;
				SpikeTimes[j].push_back(time);

				LastSpikedTimeNeuron[j] = time;
				if (PreSynNeuronSectionBeg[j] >= 0){
					for (size_t k = PreSynNeuronSectionBeg[j]; k < PreSynNeuronSectionEnd[j]; ++k){
						NAdditionalSpikesNow[(CurrentQueueIndex + Network[k].DelayinTsteps) % QueueSize].fetch_and_increment();
					}
				}
				//Implementing Causal Learning Rule
				if (PostSynNeuronSectionBeg[j] >= 0){
					delTmin = 0.3*1000*onemsbyTstep;
					for (size_t k = PostSynNeuronSectionBeg[j]; k < PostSynNeuronSectionEnd[j]; ++k){
						if (Network[AuxArray[k]].Plastic == 1 && LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] != -1){
							delT = time - LastSpikedTimeNeuron[Network[AuxArray[k]].NStart-1];
							//STDP rule
							tmaxtmp = (int) (Neurons[j].tmax*onemsbyTstep * 1000 +0.5f);
							if (delT <= tmaxtmp){
								Network[AuxArray[k]].Weight += ltp;
							}
							else{
								if (Network[AuxArray[k]].Weight - ltd >= 0){
								//if (Network[AuxArray[k]].Weight - ltd >= 0 && delT < 0.2 * 1000 * onemsbyTstep){
									Network[AuxArray[k]].Weight -= ltd;
								}
							}
							if (delTmin > delT){
								delTmin = delT;
							}
						}
						if (Network[AuxArray[k]].Plastic == 1 && LastSpikedTimeNeuron[Network[AuxArray[k]].NStart - 1] == -1) {
							if (Network[AuxArray[k]].Weight - ltd >= 0){
								Network[AuxArray[k]].Weight -= ltd;
							}
						}
					}
				}
				//Implementing Metaplasticity by changing tmax (Assuming all delT in one pattern are the same)
				float check = delTmin*0.001 / onemsbyTstep;
				float N_s = SpikeTimes[j].size();
				float N_t = 40;
				float temp1 = (1 - exp(-(N_s - 1) / N_t)) / (1 - exp(-1 / N_t));
				float temp2 = (1 - exp(-(N_s) / N_t)) / (1 - exp(-1 / N_t));
				/*if (delTmin*0.001 / onemsbyTstep < 0.2) {
					if (N_s == 1){
						Neurons[j].tmax = Neurons[j].tmax;
					}
					else{
						Neurons[j].tmax = (exp(-1/N_t)*Neurons[j].tmax*temp1 + delTmin*0.001 / onemsbyTstep) / (temp2);
					}
				}//////
				if (delTmin*0.001 / onemsbyTstep < 0.2) {
					if (SpikeTimes[j].size() == 1){
						Neurons[j].tmax = ((SpikeTimes[j].size() - 1)*Neurons[j].tmax +  delTmin*0.001 / onemsbyTstep) / (SpikeTimes[j].size());
					}
					else{
						//Neurons[j].tmax = ((SpikeTimes[j].size() - 1)*Neurons[j].tmax + 2 * delTmin*0.001 / onemsbyTstep) / (3 * SpikeTimes[j].size());
						Neurons[j].tmax = (Neurons[j].tmax + 2 * delTmin*0.001 / onemsbyTstep) / 3;
					}
				}
				//if (Neurons[j].tmax > 1.05*delT*0.001 / onemsbyTstep){
				//	Neurons[j].tmax = 1.05*delT*0.001 / onemsbyTstep;
				//}

				//Implementing artificial lateral inhibition
				if (j >= Ninp) {
					for (int nlat = Ninp; nlat < N; ++nlat){
						if (nlat != j){
							Vnow[nlat] = Neurons[j].d;
						}
					}
					break;
				} 

			}
		}*/
		
	}
}
void CurrentAttenuate::operator() (tbb::blocked_range<int> &Range) const {
	tbb::atomic<long long> *Begin1 = &Iin1[Range.begin()];
	tbb::atomic<long long> *End1 = &Iin1[Range.end()-1] + 1;
	tbb::atomic<long long> *Begin2 = &Iin2[Range.begin()];
	tbb::atomic<long long> *End2 = &Iin2[Range.end() - 1] + 1;

	for (tbb::atomic<long long> *i = Begin1, *j = Begin2; i < End1; ++i, ++j){
		(*i) = 0; // (long long)(float(i->load()) * attenFactor1);
		//(*j) = 0; // (long long)(float(j->load()) * attenFactor2);
	}
}
void SpikeRecord::operator()(tbb::blocked_range<int> &Range) const{
	int RangeBeg = Range.begin();
	int RangeEnd = Range.end();
	for (int j = RangeBeg; j < RangeEnd; ++j){
		if (LastSpikedTimeNeuron[j]==time && j!=N){ 
			size_t CurrNeuronSectionBeg = PreSynNeuronSectionBeg[j];
			size_t CurrNeuronSectionEnd = PreSynNeuronSectionEnd[j];
			if (CurrNeuronSectionBeg >= 0)
				for (size_t k = CurrNeuronSectionBeg; k < CurrNeuronSectionEnd; ++k){
					for (int epsp_time = 1; epsp_time <= DelayRange*onemsbyTstep; epsp_time++){
						int ThisQueue = (CurrentQueueIndex + epsp_time) % QueueSize;
						int ThisLoadingInd = CurrentSpikeLoadingInd[ThisQueue].fetch_and_increment();
						SpikeQueue[ThisQueue][ThisLoadingInd] = k;
					}
				}
		}
	}
}
void InputArgs::IExtFunc(int time, MexMatrix<float> &InpCurr, MexVector<float> &Iext, MexVector<int> &IextPtr)
{
	// patTime = Time for 1 datapoint
	// patONTime = patTime - margin time (10 ms cuz order of membrane potential time const)
	int patTime = 50, patONTime = 40;
	int currPat = time / patTime;
	int Ninp = InpCurr.ncols();
	for (int i = 0; i < Ninp; ++i){
		//if (InpCurr(IextPtr[i], i) == time){
		if (time % patTime < patONTime){
			Iext[i] = InpCurr(currPat, i);
		}
		else{
			Iext[i] = 0;
		}
	}
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
		//if (!(IntVars.InterestingSyns.size()))
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
	if (OutputControl & OutOps::SPIKETIMES_REQ)
		this->SpikeTimesOut = MexVector<MexVector<int> >(N, MexVector<int>());
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
		this->SpikeTimes = MexVector<MexVector<int> >(N, MexVector<int>());
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
		this->SpikeTimes = MexVector<MexVector<int> >(N, MexVector<int>());
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
		size_t tempSize2 = Network.size();
		for (int j = 0; j < tempSize; ++j){
			OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
		}
		for (int j = 0; j < tempSize2; ++j){
			StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
		}
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

	// Storing Spike Times of all Neurons
		if (OutputControl & OutOps::SPIKETIMES_REQ){
			OutVars.SpikeTimesOut=SpikeTimes;
		}

		
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
			size_t tempSize2 = Network.size();
			for (int j = 0; j < tempSize; ++j){
				OutVars.WeightOut(CurrentInsertPos, j) = Network[InterestingSyns[j]].Weight;
			}
			for (int j = 0; j < tempSize2; ++j){
				StateOut.WeightOut(CurrentInsertPos, j) = Network[j].Weight;
			}
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
			// Storing Spike Times of all Neurons
		if (OutputControl & OutOps::SPIKETIMES_REQ){
			for (int j = 0; j < N; ++j){
				OutVars.SpikeTimesOut[j] = SpikeTimes[j];
			}
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
	for (int j = 0; j < N; ++j){
		FinalStateOut.SpikeTimes[j] = SpikeTimes[j];
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
	MexVector<int>				&IextPtr				= IntVars.IextPtr;
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

	MexVector<MexVector<int> > &SpikeTimes   = IntVars.SpikeTimes;

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
	int nSteps = NoOfms*onemsbyTstep, Ninp = InputArguments.InpCurr.ncols();
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
		InputArgs::IExtFunc(time, InpCurr, Iext, IextPtr);
		Irand.generate();

		// This iteration applies time update equation for internal current
		// in this case, it is just an exponential attenuation
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 3000),
			CurrentAttenuate(Iin1, Iin2, CurrentDecayFactor1, CurrentDecayFactor2));

		size_t QueueSubEnd = SpikeQueue[CurrentQueueIndex].size();
		maxSpikeno += QueueSubEnd;
	/*	// Epilepsy Check
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
		} */

		// This iter calculates Itemp as in above diagram
		if (SpikeQueue[CurrentQueueIndex].size() != 0)
			tbb::parallel_for(tbb::blocked_range<int*>((int*)&SpikeQueue[CurrentQueueIndex][0],
				(int*)&SpikeQueue[CurrentQueueIndex][QueueSubEnd - 1] + 1, 10000), 
				CurrentUpdate(SpikeQueue[CurrentQueueIndex], Network, Iin1, Iin2, LastSpikedTimeSyn, I0, time), apCurrentUpdate);
		SpikeQueue[CurrentQueueIndex].clear();

		// Calculation of V,U[t] from V,U[t-1], Iin = Itemp
		tbb::parallel_for(tbb::blocked_range<int>(0, N, 10), NeuronSimulate(
			Vnow, Unow, Iin1, Iin2, Irand, Iext, Neurons, Network,
			CurrentQueueIndex, QueueSize, onemsbyTstep, time, DelayRange, StdDev, I0, ltp, ltd, Ninp, SpikeTimes, PreSynNeuronSectionBeg,
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
				Neurons,
				CurrentQueueIndex, QueueSize,
				PreSynNeuronSectionBeg,
				PreSynNeuronSectionEnd,
				CurrentSpikeLoadingInd,
				SpikeQueue,
				LastSpikedTimeNeuron,
				time, DelayRange, onemsbyTstep, N
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


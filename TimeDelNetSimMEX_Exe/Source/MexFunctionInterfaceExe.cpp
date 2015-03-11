#include <mex.h>
#include <matrix.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <chrono>
#include <type_traits>
#include "..\..\TimeDelNetSimMEX_Lib\Headers\NeuronSim.hpp"
#include "..\..\TimeDelNetSimMEX_Lib\Headers\MexMem.hpp"

using namespace std;

int getOutputControl(char* OutputControlSequence){
	char * SequenceWord;
	char * NextNonDelim = NULL;
	char * Delims = " -,";
	int OutputControl = 0x00000000;
	SequenceWord = strtok_s(OutputControlSequence, Delims, &NextNonDelim);
	bool AddorRemove; // TRUE for ADD
	while (SequenceWord != NULL) {
		AddorRemove = true;
		if (SequenceWord[0] == '/') {
			AddorRemove = false;
			SequenceWord++;
		}
		if (AddorRemove && !_strcmpi(SequenceWord, "VCF"))
			OutputControl |= OutOps::VOUT_REQ | OutOps::UOUT_REQ | OutOps::IOUT_REQ
		                   | OutOps::FINAL_STATE_REQ;
		if (AddorRemove && !_strcmpi(SequenceWord, "VCWF"))
			OutputControl |= OutOps::VOUT_REQ | OutOps::UOUT_REQ | OutOps::IOUT_REQ 
			               | OutOps::WEIGHTOUT_REQ
			               | OutOps::FINAL_STATE_REQ;
		if (AddorRemove && !_strcmpi(SequenceWord, "FSF"))
			OutputControl |= OutOps::VOUT_REQ | OutOps::UOUT_REQ | OutOps::IOUT_REQ
			               | OutOps::WEIGHTOUT_REQ
						   | OutOps::CURRENT_QINDS_REQ
						   | OutOps::SPIKE_QUEUE_OUT_REQ
						   | OutOps::LASTSPIKED_NEU_REQ
						   | OutOps::LASTSPIKED_SYN_REQ
						   | OutOps::FINAL_STATE_REQ;
		if (!_strcmpi(SequenceWord, "Vout"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::VOUT_REQ : 
					 OutputControl & ~(OutOps::VOUT_REQ);
		if (!_strcmpi(SequenceWord, "Uout"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::UOUT_REQ : 
					 OutputControl & ~(OutOps::UOUT_REQ);
		if (!_strcmpi(SequenceWord, "Iout"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::IOUT_REQ : 
					 OutputControl & ~(OutOps::IOUT_REQ);
		if (!_strcmpi(SequenceWord, "Wout"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::WEIGHTOUT_REQ : 
					 OutputControl & ~(OutOps::WEIGHTOUT_REQ);
		if (!_strcmpi(SequenceWord, "CQInds"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::CURRENT_QINDS_REQ : 
					 OutputControl & ~(OutOps::CURRENT_QINDS_REQ);
		if (!_strcmpi(SequenceWord, "SQOut"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::SPIKE_QUEUE_OUT_REQ : 
					 OutputControl & ~(OutOps::SPIKE_QUEUE_OUT_REQ);
		if (!_strcmpi(SequenceWord, "LSTNOut"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::LASTSPIKED_NEU_REQ : 
					 OutputControl & ~(OutOps::LASTSPIKED_NEU_REQ);
		if (!_strcmpi(SequenceWord, "LSTSOut"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::LASTSPIKED_SYN_REQ : 
					 OutputControl & ~(OutOps::LASTSPIKED_SYN_REQ);
		if (!_strcmpi(SequenceWord, "Final"))
			OutputControl = AddorRemove ? 
			         OutputControl | OutOps::FINAL_STATE_REQ : 
					 OutputControl & ~(OutOps::FINAL_STATE_REQ);
		SequenceWord = strtok_s(NULL, Delims, &NextNonDelim);
	}
	return OutputControl;
}

void takeInputFromFiles(char* FileNetwork, char * FileSimControl, InputArgs &InputArgList){

	InputArgList.DelayRange = 20;
	InputArgList.CurrentQIndex = 0;
	InputArgList.Time = 0;
	InputArgList.StorageStepSize = DEFAULT_STORAGE_STEP;
	InputArgList.OutputControl = 0;
	InputArgList.StatusDisplayInterval = DEFAULT_STATUS_DISPLAY_STEP;

	//Setting up file stream
	fstream fin;
	fin.open(FileSimControl, ios_base::in | ios_base::binary);
	fin.read((char*)&InputArgList.onemsbyTstep, 4);
	fin.close();

	fin.open(FileNetwork, ios_base::in | ios_base::binary);
	//Declarations
	int N, M;	//N, M as defined above
	int i;		//Generic Loop Variable (GLV)
	char tempinput; // generic char to convert from char to int

	// Reading Neural Network Properties from File FileNetwork
	fin.read((char *)&N, 4);
	fin.read((char *)&M, 4);

	InputArgList.Network.resize(M);
	InputArgList.Neurons.resize(N);

	for (i = 0; i<N; ++i){
		fin.read((char*)&(InputArgList.Neurons[i].a), 4);
		fin.read((char*)&(InputArgList.Neurons[i].b), 4);
		fin.read((char*)&(InputArgList.Neurons[i].c), 4);
		fin.read((char*)&(InputArgList.Neurons[i].d), 4);
	}

	for (i = 0; i<M; ++i){
		fin.read((char*)&(InputArgList.Network[i].NStart), 4);
		fin.read((char*)&(InputArgList.Network[i].NEnd), 4);
		fin.read((char*)&(InputArgList.Network[i].Weight), 4);
		fin.read((char*)&(tempinput), 1);
		InputArgList.Network[i].DelayinTsteps = (int)(tempinput) * InputArgList.onemsbyTstep;
	}

	fin.close();

	int HasInitialCond = false, HasSpikeList = false;

	fin.open(FileSimControl, ios_base::in | ios_base::binary);
	fin.read((char*)&InputArgList.onemsbyTstep, 4);
	fin.read((char*)&InputArgList.NoOfms, 4);
	fin.read((char*)&InputArgList.StorageStepSize, 4);
	fin.read((char*)&HasInitialCond, 4);
	
	if (InputArgList.StorageStepSize)
		InputArgList.OutputControl = getOutputControl("FSF");
	else
		InputArgList.OutputControl = getOutputControl("VCF");

	if (HasInitialCond){
		// Vectors for initial conditions. Note that these vectors will be destroyed by simulation
		InputArgList.V.resize(N);
		InputArgList.U.resize(N);
		InputArgList.I.resize(N);

		fin.read((char*)&InputArgList.Time, sizeof(int));
		fin.read((char*)&InputArgList.V[0], N*sizeof(float));
		fin.read((char*)&InputArgList.U[0], N*sizeof(float));
		fin.read((char*)&InputArgList.I[0], N*sizeof(float));
		fin.read((char*)&HasSpikeList, sizeof(int));

		if (HasSpikeList){
			InputArgList.SpikeQueue = MexVector<MexVector<int> >(InputArgList.DelayRange * InputArgList.onemsbyTstep,
			                                                     MexVector<int>(0));
			InputArgList.LSTNeuron = MexVector<int>(N);
			InputArgList.LSTSyn = MexVector<int>(M);

			fin.read((char*)&InputArgList.CurrentQIndex, sizeof(int));
			int CurrSubQSize;

			for (int i = 0; i < 20 * InputArgList.onemsbyTstep; ++i){
				fin.read((char*)&CurrSubQSize, sizeof(int));
				InputArgList.SpikeQueue[i] = MexVector<int>(CurrSubQSize);
				fin.read((char*)&InputArgList.SpikeQueue[i][0], sizeof(int)*CurrSubQSize);
			}
			fin.read((char*)&InputArgList.LSTNeuron[0], N*sizeof(int));
			fin.read((char*)&InputArgList.LSTSyn[0], M*sizeof(int));
		}
	}
}

void putOutputToFile(char* FileOut, InputArgs &InputArgList, StateVarsOut &StateVarsList, FinalState &FinalStateList){
	
	fstream fout;
	fout.open(FileOut, ios_base::out | ios_base::binary);

	int N = InputArgList.Neurons.size();
	int M = InputArgList.Network.size();
	int i; // GLV
	char tempoutput; // temp char to convert from int to char;

	fout.write((char *)&N, 4);
	fout.write((char *)&M, 4);
	fout.write((char *)&InputArgList.onemsbyTstep, 4);
	fout.write((char *)&InputArgList.NoOfms, 4);
	fout.write((char *)&InputArgList.StorageStepSize, 4);

	// Saving the network.
	for (i = 0; i<N; ++i){
		fout.write((char*)&(InputArgList.Neurons[i].a), 4);
		fout.write((char*)&(InputArgList.Neurons[i].b), 4);
		fout.write((char*)&(InputArgList.Neurons[i].c), 4);
		fout.write((char*)&(InputArgList.Neurons[i].d), 4);
	}

	for (i = 0; i<M; ++i){
		fout.write((char*)&(InputArgList.Network[i].NStart), 4);
		fout.write((char*)&(InputArgList.Network[i].NEnd), 4);
		fout.write((char*)&(InputArgList.Network[i].Weight), 4);
		tempoutput = (char)(InputArgList.Network[i].DelayinTsteps);
		fout.write((char*)&(tempoutput), 1);
	}

	int timeDimensionLen, NoofSpikeQs;
	timeDimensionLen = StateVarsList.VOut.nrows();
	NoofSpikeQs = (InputArgList.StorageStepSize) ? StateVarsList.SpikeQueueOut.size() : 1;

	for (int j = 0; j < timeDimensionLen; ++j)
		fout.write((char*)&(StateVarsList.VOut(j, 0)), N*sizeof(int));
	for (int j = 0; j < timeDimensionLen; ++j)
		fout.write((char*)&(StateVarsList.UOut(j, 0)), N*sizeof(int));
	for (int j = 0; j < timeDimensionLen; ++j)
		fout.write((char*)&(StateVarsList.IOut(j, 0)), N*sizeof(int));

	fout.write((char*)&StateVarsList.TimeOut[0], sizeof(int)*timeDimensionLen);

	//Write SpikeQueues and  Last SpikedTimes Neuron and Synapses into file
	if (InputArgList.StorageStepSize){
		for (i = 0; i < NoofSpikeQs; ++i){
			fout.write((char*)&StateVarsList.CurrentQIndexOut[i], sizeof(int));
			for (int j = 0; j < InputArgList.onemsbyTstep * InputArgList.DelayRange; ++j){
				int CurrSubQSize = StateVarsList.SpikeQueueOut[i][j].size();
				fout.write((char*)&CurrSubQSize, sizeof(int));
				fout.write((char*)&StateVarsList.SpikeQueueOut[i][j][0], sizeof(int)*CurrSubQSize);
			}
		}
		for (i = 0; i < NoofSpikeQs; ++i){
			fout.write((char*)&StateVarsList.LSTNeuronOut(i, 0), sizeof(int)*N);
		}
		for (i = 0; i < NoofSpikeQs; ++i){
			fout.write((char*)&StateVarsList.LSTSynOut(i, 0), sizeof(int)*M);
		}
	}
	else{
		fout.write((char*)&FinalStateList.CurrentQIndex, sizeof(int));
		for (int j = 0; j < InputArgList.onemsbyTstep * InputArgList.DelayRange; ++j){
			int CurrSubQSize = FinalStateList.SpikeQueue[j].size();
			fout.write((char*)&CurrSubQSize, sizeof(int));
			fout.write((char*)&FinalStateList.SpikeQueue[j][0], sizeof(int)*CurrSubQSize);
		}
		fout.write((char*)&FinalStateList.LSTNeuron[0], sizeof(int)*N);
		fout.write((char*)&FinalStateList.LSTSyn[0], sizeof(int)*M);
	}
	
	fout.close();
}
template<typename T> mxArray * assignmxArray(T &ScalarOut, mxClassID ClassID){

	mxArray * ReturnPointer;
	if (is_arithmetic<T>::value){
		ReturnPointer = mxCreateNumericMatrix_730(1, 1, ClassID, mxREAL);
		*reinterpret_cast<T *>(mxGetData(ReturnPointer)) = ScalarOut;
	}
	else{
		ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	}

	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexMatrix<T> &VectorOut, mxClassID ClassID){

	mxArray * ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	if (VectorOut.ncols() && VectorOut.nrows()){
		mxSetM(ReturnPointer, VectorOut.ncols());
		mxSetN(ReturnPointer, VectorOut.nrows());
		mxSetData(ReturnPointer, VectorOut.releaseArray());
	}

	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexVector<T> &VectorOut, mxClassID ClassID){

	mxArray * ReturnPointer = mxCreateNumericMatrix_730(0, 0, ClassID, mxREAL);
	if (VectorOut.size()){
		mxSetM(ReturnPointer, VectorOut.size());
		mxSetN(ReturnPointer, 1);
		mxSetData(ReturnPointer, VectorOut.releaseArray());
	}
	return ReturnPointer;
}

template<typename T> mxArray * assignmxArray(MexVector<MexVector<T> > &VectorOut, mxClassID ClassID){
	
	mxArray * ReturnPointer;
	if (VectorOut.size()){
		ReturnPointer = mxCreateCellMatrix(VectorOut.size(), 1);
		
		int VectVectSize = VectorOut.size();
		for (int i = 0; i < VectVectSize; ++i){
			mxSetCell(ReturnPointer, i, assignmxArray(VectorOut[i], ClassID));
		}
	}
	else{
		ReturnPointer = mxCreateCellMatrix_730(0, 0);
	}
	return ReturnPointer;
}

mxArray * putOutputToMatlabStruct(OutputVars &Output){
	const char *FieldNames[] = { "WeightOut" };

	int NFields = 1;
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);
	
	// Assigning Weightout
	mxSetField(ReturnPointer, 0, "WeightOut", assignmxArray(Output.WeightOut, mxSINGLE_CLASS));

	return ReturnPointer;
}

mxArray * putStateToMatlabStruct(StateVarsOut &Output){
	const char *FieldNames[] = {
		"V",
		"I",
		"Time",
		"U",
		"Weight",
		"CurrentQIndex",
		"SpikeQueue",
		"LSTNeuron",
		"LSTSyn"
	};
	int NFields = 9;
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);

	// Assigning V, U, I, Time
	mxSetField(ReturnPointer, 0, "V"             , assignmxArray(Output.VOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "U"             , assignmxArray(Output.UOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "I"             , assignmxArray(Output.IOut, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "Time"          , assignmxArray(Output.TimeOut, mxINT32_CLASS));

	// Assigning Weight
	mxSetField(ReturnPointer, 0, "Weight"        , assignmxArray(Output.WeightOut, mxSINGLE_CLASS));

	// Assigning Spike Queue Related Shiz
	mxSetField(ReturnPointer, 0, "CurrentQIndex" , assignmxArray(Output.CurrentQIndexOut, mxINT32_CLASS));
	// Assigning SpikeQueue

	mxSetField(ReturnPointer, 0, "SpikeQueue"    , assignmxArray(Output.SpikeQueueOut, mxINT32_CLASS));

	// Assigning Last Spiked Time related information
	mxSetField(ReturnPointer, 0, "LSTNeuron"     , assignmxArray(Output.LSTNeuronOut, mxINT32_CLASS));
	mxSetField(ReturnPointer, 0, "LSTSyn"        , assignmxArray(Output.LSTSynOut, mxINT32_CLASS));

	return ReturnPointer;
}

mxArray* putFinalStatetoMatlabStruct(FinalState &FinalStateList){
	const char *FieldNames[] = {
		"V",
		"I",
		"Time",
		"U",
		"Weight",
		"CurrentQIndex",
		"SpikeQueue",
		"LSTNeuron",
		"LSTSyn"
	};
	int NFields = 9;
	mwSize StructArraySize[2] = { 1, 1 };

	mxArray * ReturnPointer = mxCreateStructArray_730(2, StructArraySize, NFields, FieldNames);

	// Assigning vout, Uout, Iout, TimeOut
	mxSetField(ReturnPointer, 0, "V"                 , assignmxArray(FinalStateList.V, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "U"                 , assignmxArray(FinalStateList.U, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "I"                 , assignmxArray(FinalStateList.I, mxSINGLE_CLASS));
	if (FinalStateList.Time >= 0)
		mxSetField(ReturnPointer, 0, "Time"          , assignmxArray(FinalStateList.Time, mxSINGLE_CLASS));
	else
		mxSetField(ReturnPointer, 0, "Time"          , mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));

	// Assigning WeightOut
	mxSetField(ReturnPointer, 0, "Weight"            , assignmxArray(FinalStateList.Weight, mxSINGLE_CLASS));

	// Assigning Spike Queue Related Shiz
	if (FinalStateList.CurrentQIndex >= 0)
		mxSetField(ReturnPointer, 0, "CurrentQIndex" , assignmxArray(FinalStateList.CurrentQIndex, mxSINGLE_CLASS));
	else
		mxSetField(ReturnPointer, 0, "CurrentQIndex" , mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
	mxSetField(ReturnPointer, 0, "SpikeQueue"        , assignmxArray(FinalStateList.SpikeQueue, mxSINGLE_CLASS));

	// Assigning Last Spiked Time related information
	mxSetField(ReturnPointer, 0, "LSTNeuron"         , assignmxArray(FinalStateList.LSTNeuron, mxSINGLE_CLASS));
	mxSetField(ReturnPointer, 0, "LSTSyn"            , assignmxArray(FinalStateList.LSTSyn, mxSINGLE_CLASS));

	return ReturnPointer;
}

int main(){
	// NOTE THAT THERE IS NO DATA VALIDATION AS THIS IS EXPECTED TO HAVE 
	// BEEN DONE IN THE MATLAB SIDE OF THE INTERFACE TO THIS MEX FUNCTION

	InputArgs InputArgList;
	char *FileNetwork = "Data\\NetworkGraph.bin";
	char *FileSimControl = "Data\\SimOptions.bin";
	takeInputFromFiles(FileNetwork, FileSimControl, InputArgList);

	// Declaring Output Vectors
	OutputVars PureOutput;
	StateVarsOut StateVarsOutput;
	FinalState FinalStateOutput;
	// Declaring Final State output vectors
	
	// Running Simulation Function.
	chrono::system_clock::time_point TStart = chrono::system_clock::now();
	SimulateParallel(
		move(InputArgList),
		PureOutput,
		StateVarsOutput,
		FinalStateOutput);
	chrono::system_clock::time_point TEnd = chrono::system_clock::now();
	printf("The Time taken = %d milliseconds", chrono::duration_cast<chrono::milliseconds>(TEnd - TStart).count());
	//mexEvalString("drawnow");

	putOutputToFile("Data/Spikeout.bin", InputArgList, StateVarsOutput, FinalStateOutput);
	system("pause");
	return 0;
}
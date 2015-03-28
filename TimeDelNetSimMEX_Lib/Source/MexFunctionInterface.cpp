#include <mex.h>
#include <matrix.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include <type_traits>
#include "..\Headers\NeuronSim.hpp"
#include "..\Headers\MexMem.hpp"

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

void takeInputFromMatlabStruct(mxArray* MatlabInputStruct, InputArgs &InputArgList){

	size_t N = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "a"));
	size_t M = mxGetNumberOfElements(mxGetField(MatlabInputStruct, 0, "NStart"));

	InputArgList.onemsbyTstep = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "onemsbyTstep")));
	InputArgList.NoOfms = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NoOfms")));
	InputArgList.DelayRange = *reinterpret_cast<int *>(mxGetData(mxGetField(MatlabInputStruct, 0, "DelayRange")));
	InputArgList.CurrentQIndex = 0;
	InputArgList.Time = 0;
	InputArgList.StorageStepSize = DEFAULT_STORAGE_STEP;
	InputArgList.OutputControl = 0;
	InputArgList.StatusDisplayInterval = DEFAULT_STATUS_DISPLAY_STEP;

	float*      genFloatPtr[4];     // Generic float Pointers used around the place to access data
	int*        genIntPtr[2];       // Generic int Pointers used around the place to access data
	short *     genCharPtr;         // Generic short Pointer used around the place to access data (delays specifically)
	mxArray *   genmxArrayPtr;      // Generic mxArray Pointer used around the place to access data

	// Initializing neuron specification structure array Neurons
	genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "a")));	// a[N]
	genFloatPtr[1] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "b")));	// b[N]
	genFloatPtr[2] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "c")));	// c[N]
	genFloatPtr[3] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "d")));	// d[N]

	InputArgList.Neurons = MexVector<Neuron>(N);

	for (int i = 0; i < N; ++i){
		InputArgList.Neurons[i].a = genFloatPtr[0][i];
		InputArgList.Neurons[i].b = genFloatPtr[1][i];
		InputArgList.Neurons[i].c = genFloatPtr[2][i];
		InputArgList.Neurons[i].d = genFloatPtr[3][i];
	}

	// Initializing network (Synapse) specification structure array Network
	genIntPtr[0]   = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NStart")));	  // NStart[M]
	genIntPtr[1]   = reinterpret_cast<int   *>(mxGetData(mxGetField(MatlabInputStruct, 0, "NEnd")));      // NEnd[M]
	genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Weight")));    // Weight[M]
	genFloatPtr[1] = reinterpret_cast<float *>(mxGetData(mxGetField(MatlabInputStruct, 0, "Delay")));     // Delay[M]

	InputArgList.Network = MexVector<Synapse>(M);

	for (int i = 0; i < M; ++i){
		InputArgList.Network[i].NStart = genIntPtr[0][i];
		InputArgList.Network[i].NEnd = genIntPtr[1][i];
		InputArgList.Network[i].Weight = genFloatPtr[0][i];
		InputArgList.Network[i].DelayinTsteps = (int(genFloatPtr[1][i] * InputArgList.onemsbyTstep + 0.5) > 0) ?
			int(genFloatPtr[1][i] * InputArgList.onemsbyTstep + 0.5) : 1;
	}

	// Initializing Time
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "Time");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.Time = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing StorageStepSize
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "StorageStepSize");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.StorageStepSize = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing StatusDisplayInterval
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "StatusDisplayInterval");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.StatusDisplayInterval = *reinterpret_cast<int *>(mxGetData(genmxArrayPtr));

	// Initializing InterestingSyns
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "InterestingSyns");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		size_t NumElems = mxGetNumberOfElements(genmxArrayPtr);
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.InterestingSyns = MexVector<int>(NumElems);
		InputArgList.InterestingSyns.copyArray(0, genIntPtr[0], NumElems);
	}

	// Initializing V0, U0 and Iin0
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "V");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.V = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.V.copyArray(0, genFloatPtr[0], N);
	}
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "U");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.U = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.U.copyArray(0, genFloatPtr[0], N);
	}
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "I");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		InputArgList.I = MexVector<float>(N);
		genFloatPtr[0] = reinterpret_cast<float *>(mxGetData(genmxArrayPtr));
		InputArgList.I.copyArray(0, genFloatPtr[0], N);
	}

	// Initializing CurrentQueueIndex
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "CurrentQIndex");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr))
		InputArgList.CurrentQIndex = *reinterpret_cast<int *>(genmxArrayPtr);

	// Initializing InitSpikeQueue
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "SpikeQueue");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		mxArray **SpikeQueueArr = reinterpret_cast<mxArray **>(mxGetData(genmxArrayPtr));
		int SpikeQueueSize = InputArgList.onemsbyTstep * InputArgList.DelayRange;
		InputArgList.SpikeQueue = MexVector<MexVector<int> >(SpikeQueueSize);
		for (int i = 0; i < SpikeQueueSize; ++i){
			size_t NumOfSpikes = mxGetNumberOfElements(SpikeQueueArr[i]);
			InputArgList.SpikeQueue[i] = MexVector<int>(NumOfSpikes);
			int * CurrQueueArr = reinterpret_cast<int *>(mxGetData(SpikeQueueArr[i]));
			InputArgList.SpikeQueue[i].copyArray(0, CurrQueueArr, NumOfSpikes);
		}
	}

	// Initializing InitLastSpikedTimeNeuron
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "LSTNeuron");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.LSTNeuron = MexVector<int>(N);
		InputArgList.LSTNeuron.copyArray(0, genIntPtr[0], N);
	}

	// Initializing InitLastSpikedTimeSyn
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "LSTSyn");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		genIntPtr[0] = reinterpret_cast<int *>(mxGetData(genmxArrayPtr));
		InputArgList.LSTSyn = MexVector<int>(M);
		InputArgList.LSTSyn.copyArray(0, genIntPtr[0], M);
	}

	// Initializing OutputControl
	genmxArrayPtr = mxGetField(MatlabInputStruct, 0, "OutputControl");
	if (genmxArrayPtr != NULL && !mxIsEmpty(genmxArrayPtr)){
		char * OutputControlSequence = mxArrayToString(genmxArrayPtr);
		InputArgList.OutputControl = getOutputControl(OutputControlSequence);
		mxFree(OutputControlSequence);
	}
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
		
		size_t VectVectSize = VectorOut.size();
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]){
	// NOTE THAT THERE IS NO DATA VALIDATION AS THIS IS EXPECTED TO HAVE 
	// BEEN DONE IN THE MATLAB SIDE OF THE INTERFACE TO THIS MEX FUNCTION

	InputArgs InputArgList;
	takeInputFromMatlabStruct(prhs[0], InputArgList);

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
	mexPrintf("The Time taken = %d milliseconds", chrono::duration_cast<chrono::milliseconds>(TEnd - TStart).count());
	mexEvalString("drawnow");

	mwSize StructArraySize[2] = { 1, 1 };
	int NFields = 10;
	
	plhs[0] = putOutputToMatlabStruct(PureOutput);
	plhs[1] = putStateToMatlabStruct(StateVarsOutput);
	plhs[2] = putFinalStatetoMatlabStruct(FinalStateOutput);
	NFields = 15;
}
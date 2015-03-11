#ifndef MEXMEM_HPP
#define MEXMEM_HPP

struct ExOps{
	enum{
		EXCEPTION_MEM_FULL = 0xFF,
		EXCEPTION_EXTMEM_MOD = 0x7F,
		EXCEPTION_CONST_MOD = 0x3F
	};
};


template<class T>
class MexVector{
	int NumOfElems, Capacity;
	bool isCurrentMemExternal;
	T* Array;

public:
	inline MexVector() : NumOfElems(0), Capacity(0), isCurrentMemExternal(false), Array(NULL){};
	inline explicit MexVector(int Size){
		if (Size > 0){
			Array = reinterpret_cast<T*>(mxCalloc(Size, sizeof(T)));
			if (Array == NULL)     // Full Memory exception
				throw ExOps::EXCEPTION_MEM_FULL;
			for (int i = 0; i < Size; ++i)
				new (Array + i) T;	// Constructing Default Objects
		}
		else{
			Array = NULL;
		}
		NumOfElems = Size;
		Capacity = Size;
		isCurrentMemExternal = false;
	}
	inline MexVector(const MexVector<T> &M){
		if (M.NumOfElems > 0){
			Array = reinterpret_cast<T*>(mxCalloc(M.NumOfElems, sizeof(T)));
			if (Array != NULL)
				for (int i = 0; i < M.NumOfElems; ++i)
					new (Array + i) T(M.Array[i]);
			else{	// Checking for memory full shit
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array = NULL;
		}
		NumOfElems = M.NumOfElems;
		Capacity = M.NumOfElems;
		isCurrentMemExternal = false;
	}
	inline MexVector(MexVector<T> &&M){
		isCurrentMemExternal = M.isCurrentMemExternal;
		NumOfElems = M.NumOfElems;
		Capacity = M.Capacity;
		Array = M.Array;
		if (!(M.Array == NULL)){
			M.isCurrentMemExternal = true;
		}
	}
	inline explicit MexVector(int Size, const T &Elem){
		if (Size > 0){
			Array = reinterpret_cast<T*>(mxCalloc(Size, sizeof(T)));
			if (Array == NULL){
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array = NULL;
		}
		NumOfElems = Size;
		Capacity = Size;
		isCurrentMemExternal = false;
		for (int i = 0; i < NumOfElems; ++i)
			new (Array + i) T(Elem);
	}
	inline explicit MexVector(int Size, T* Array_, bool SelfManage = 1) :
		Array(Size ? Array_ : NULL), NumOfElems(Size), Capacity(Size), isCurrentMemExternal(Size ? !SelfManage : false){}

	inline ~MexVector(){
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
	}
	inline void operator = (const MexVector<T> &M){
		assign(M);
	}
	inline void operator = (const MexVector<T> &&M){
		assign(move(M));
	}
	inline void operator = (const MexVector<T> &M) const{
		this->assign(M);
	}
	inline T& operator[] (int Index) const{
		return Array[Index];
	}

	// If Ever this operation is called, no funcs except will work (Vector will point to empty shit) unless 
	// the assign function is explicitly called to self manage another array.
	inline T* releaseArray(){
		if (isCurrentMemExternal)
			return NULL;
		else{
			isCurrentMemExternal = false;
			T* temp = Array;
			Array = NULL;
			NumOfElems = 0;
			Capacity = 0;
			return temp;
		}
	}
	inline void assign(const MexVector<T> &M){
		if (M.NumOfElems > this->Capacity && !isCurrentMemExternal){
			if (Array != NULL)
				mxFree(Array);
			Array = reinterpret_cast<T*>(mxCalloc(M.NumOfElems, sizeof(T)));
			if (Array == NULL)
				throw ExOps::EXCEPTION_MEM_FULL;
			for (int i = 0; i < M.NumOfElems; ++i)
				Array[i] = M.Array[i];
			NumOfElems = M.NumOfElems;
			Capacity = M.NumOfElems;
		}
		else if (M.NumOfElems <= this->Capacity && !isCurrentMemExternal){
			for (int i = 0; i < M.NumOfElems; ++i)
				Array[i] = M.Array[i];
			NumOfElems = M.NumOfElems;
		}
		else if (M.NumOfElems == this->NumOfElems){
			for (int i = 0; i < M.NumOfElems; ++i)
				Array[i] = M.Array[i];
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;	// Attempted resizing or reallocation of Array holding External Memory
		}
	}
	inline void assign(MexVector<T> &&M){
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
		isCurrentMemExternal = M.isCurrentMemExternal;
		NumOfElems = M.NumOfElems;
		Capacity = M.Capacity;
		Array = M.Array;
		if (Array != NULL){
			M.isCurrentMemExternal = true;
		}
	}
	inline void assign(const MexVector<T> &M) const{
		if (M.NumOfElems == this->NumOfElems){
			for (int i = 0; i < M.NumOfElems; ++i)
				Array[i] = M.Array[i];
		}
		else{
			throw ExOps::EXCEPTION_CONST_MOD;	// Attempted resizing or reallocation or reassignment of const Array
		}
	}
	inline void assign(int Size, T* Array_, bool SelfManage = 1){
		NumOfElems = Size;
		Capacity = Size;
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
		if (Size > 0){
			isCurrentMemExternal = !SelfManage;
			Array = Array_;
		}
		else{
			isCurrentMemExternal = false;
			Array = NULL;
		}
	}
	inline void copyArray(int Position, T* ArrBegin, int NumElems) const{
		if (Position + NumElems > NumOfElems){
			throw ExOps::EXCEPTION_CONST_MOD;
		}
		else{
			for (int i = 0; i<NumElems; ++i)
				Array[i + Position] = ArrBegin[i];
		}
	}
	inline void reserve(int Cap){
		if (!isCurrentMemExternal && Cap > Capacity){
			T* Temp;
			if (Array != NULL)
				Temp = reinterpret_cast<T*>(mxRealloc(Array, Cap*sizeof(T)));
			else
				Temp = reinterpret_cast<T*>(mxCalloc(Cap, sizeof(T)));
			if (Temp != NULL){
				Array = Temp;
				Capacity = Cap;
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL;
		}
		else if (isCurrentMemExternal)
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted reallocation of external memory
	}
	inline void resize(int NewSize){
		if (NewSize > Capacity && !isCurrentMemExternal){
			T* Temp;
			if (Array != NULL)
				Temp = reinterpret_cast<T*>(mxRealloc(Array, NewSize*sizeof(T)));
			else
				Temp = reinterpret_cast<T*>(mxCalloc(NewSize, sizeof(T)));
			if (Temp != NULL){
				Array = Temp;
				Capacity = NewSize;
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL;
		}
		else if (isCurrentMemExternal){
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted resizing of External memory
		}
		NumOfElems = NewSize;
	}
	inline void sharewith(MexVector<T> &M) const{
		if (!M.isCurrentMemExternal && M.Array != NULL)
			mxFree(M.Array);
		if (Capacity > 0){
			M.NumOfElems = NumOfElems;
			M.Capacity = Capacity;
			M.Array = Array;
			M.isCurrentMemExternal = true;
		}
		else{
			M.NumOfElems = 0;
			M.Capacity = 0;
			M.Array = NULL;
			M.isCurrentMemExternal = false;
		}
	}
	inline void trim(){
		if (!isCurrentMemExternal){
			if (NumOfElems > 0){
				T* Temp = reinterpret_cast<T*>(mxRealloc(Array, NumOfElems*sizeof(T)));
				if (Temp != NULL)
					Array = Temp;
				else
					throw ExOps::EXCEPTION_MEM_FULL;
			}
			else{
				Array = NULL;
			}
			Capacity = NumOfElems;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;
		}
	}
	inline void clear(){
		if (!isCurrentMemExternal)
			NumOfElems = 0;
		else
			throw ExOps::EXCEPTION_EXTMEM_MOD; //Attempt to resize External memory
	}
	inline int size() const{
		return NumOfElems;
	}
	inline int capacity() const{
		return Capacity;
	}
	inline bool ismemext() const{
		return isCurrentMemExternal;
	}
	inline bool isempty() const{
		return NumOfElems == 0;
	}
	inline bool istrulyempty() const{
		return Capacity == 0;
	}
};


template<class T>
class MexMatrix{
	int NRows, NCols;
	int Capacity;
	MexVector<T> RowReturnVector;
	T* Array;
	bool isCurrentMemExternal;

public:
	inline MexMatrix() : NRows(0), NCols(0), Capacity(0), isCurrentMemExternal(false), Array(NULL), RowReturnVector(){};
	inline explicit MexMatrix(int NRows_, int NCols_) : RowReturnVector() {
		if (NRows_*NCols_ > 0){
			Array = reinterpret_cast<T*>(mxCalloc(NRows_*NCols_, sizeof(T)));
			if (Array == NULL)     // Full Memory exception
				throw ExOps::EXCEPTION_MEM_FULL;
			for (int i = 0; i < NRows_*NCols_; ++i)
				new (Array + i) T;
		}
		else{
			Array = NULL;
		}
		NRows = NRows_;
		NCols = NCols_;
		Capacity = NRows_*NCols_;
		isCurrentMemExternal = false;
	}
	inline MexMatrix(const MexMatrix<T> &M) : RowReturnVector(){
		int MNumElems = M.NRows * M.NCols;
		if (MNumElems > 0){
			Array = reinterpret_cast<T*>(mxCalloc(MNumElems, sizeof(T)));
			if (Array != NULL)
				for (int i = 0; i < MNumElems; ++i){
				Array[i] = M.Array[i];
				}
			else{	// Checking for memory full shit
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array = NULL;
		}
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = MNumElems;
		isCurrentMemExternal = false;
	}
	inline MexMatrix(MexMatrix<T> &&M) : RowReturnVector(){
		isCurrentMemExternal = M.isCurrentMemExternal;
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = M.Capacity;
		Array = M.Array;
		if (!(M.Array == NULL)){
			M.isCurrentMemExternal = true;
		}
	}
	inline explicit MexMatrix(int NRows_, int NCols_, const T &Elem) : RowReturnVector(){
		int NumElems = NRows_*NCols_;
		if (NumElems > 0){
			Array = reinterpret_cast<T*>(mxCalloc(NumElems, sizeof(T)));
			if (Array == NULL){
				throw ExOps::EXCEPTION_MEM_FULL;
			}
		}
		else{
			Array = NULL;
		}

		NRows = NRows_;
		NCols = NCols_;
		Capacity = NumElems;
		isCurrentMemExternal = false;
		for (int i = 0; i < NumElems; ++i){
			Array[i] = Elem;
		}
	}
	inline MexMatrix(int NRows_, int NCols_, T* Array_, bool SelfManage = 1) : 
		RowReturnVector(),
		Array((NRows_*NCols_) ? Array_ : NULL),
		NRows(NRows_), NCols(NCols_),
		Capacity(NRows_*NCols_),
		isCurrentMemExternal((NRows_*NCols_) ? ~SelfManage : false){}

	inline ~MexMatrix(){
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
	}
	inline void operator = (const MexMatrix<T> &M){
		assign(M);
	}
	inline void operator = (const MexMatrix<T> &&M){
		assign(move(M));
	}
	inline void operator = (const MexMatrix<T> &M) const{
		assign(M);
	}
	inline const MexVector<T>& operator[] (int Index) {
		RowReturnVector.assign(NCols, Array + Index*NCols, false);
		return  RowReturnVector;
	}
	inline T& operator()(int RowIndex, int ColIndex){
		return *(Array + RowIndex*NCols + ColIndex);
	}
	// If Ever this operation is called, no funcs except will work (Vector will point to NULL) unless 
	// the assign function is explicitly called to self manage another array.
	inline T* releaseArray(){
		if (isCurrentMemExternal)
			return NULL;
		else{
			isCurrentMemExternal = false;
			T* temp = Array;
			Array = NULL;
			NRows = 0;
			NCols = 0;
			Capacity = 0;
			return temp;
		}
	}
	inline void assign(const MexMatrix<T> &M){

		int MNumElems = M.NRows * M.NCols;

		if (MNumElems > this->Capacity && !isCurrentMemExternal){
			if (Array != NULL)
				mxFree(Array);
			Array = reinterpret_cast<T*>(mxCalloc(MNumElems, sizeof(T)));
			if (Array == NULL)
				throw ExOps::EXCEPTION_MEM_FULL;
			for (int i = 0; i < MNumElems; ++i)
				Array[i] = M.Array[i];
			NRows = M.NRows;
			NCols = M.NCols;
			Capacity = MNumElems;
		}
		else if (MNumElems <= this->Capacity && !isCurrentMemExternal){
			for (int i = 0; i < MNumElems; ++i)
				Array[i] = M.Array[i];
			NRows = M.NRows;
			NCols = M.NCols;
		}
		else if (MNumElems == this->NRows * this->NCols){
			for (int i = 0; i < MNumElems; ++i)
				Array[i] = M.Array[i];
			NRows = M.NRows;
			NCols = M.NCols;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD;	// Attempted resizing or reallocation of Array holding External Memory
		}
	}
	inline void assign(MexMatrix<T> &&M){
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
		isCurrentMemExternal = M.isCurrentMemExternal;
		NRows = M.NRows;
		NCols = M.NCols;
		Capacity = M.Capacity;
		Array = M.Array;
		if (Array != NULL){
			M.isCurrentMemExternal = true;
		}
	}
	inline void assign(const MexMatrix<T> &M) const{
		int MNumElems = M.NRows * M.NCols;
		if (M.NRows == NRows && M.NCols == NCols){
			for (int i = 0; i < MNumElems; ++i)
				Array[i] = M.Array[i];
		}
		else{
			throw ExOps::EXCEPTION_CONST_MOD;
		}
	}
	inline void assign(int NRows_, int NCols_, T* Array_, bool SelfManage = 1){
		NRows = NRows_;
		NCols = NCols_;
		Capacity = NRows_*NCols_;
		if (!isCurrentMemExternal && Array != NULL){
			mxFree(Array);
		}
		if (Capacity > 0){
			isCurrentMemExternal = !SelfManage;
			Array = Array_;
		}
		else{
			isCurrentMemExternal = false;
			Array = NULL;
		}
	}
	inline void copyArray(int RowPos, int ColPos, T* ArrBegin, int NumElems) const{
		int Position = RowPos*NCols + ColPos;
		if (Position + NumElems > NRows*NCols){
			throw ExOps::EXCEPTION_CONST_MOD;
		}
		else{
			for (int i = 0; i<NumElems; ++i)
				Array[i + Position] = ArrBegin[i];
		}
	}
	inline void reserve(int Cap){
		if (!isCurrentMemExternal && Cap > Capacity){
			T* temp;
			if (Array != NULL){
				mxFree(Array);
			}
			temp = reinterpret_cast<T*>(mxCalloc(Cap, sizeof(T)));
			if (temp != NULL){
				Array = temp;
				Capacity = Cap;
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL; // Full memory
		}
		else if (isCurrentMemExternal)
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted reallocation of external memory
	}
	inline void resize(int NewNRows, int NewNCols){
		int NewSize = NewNRows * NewNCols;
		if (NewSize > Capacity && !isCurrentMemExternal){
			T* Temp;
			if (Array != NULL){
				mxFree(Array);
			}
			Temp = reinterpret_cast<T*>(mxCalloc(NewSize, sizeof(T)));
			if (Temp != NULL){
				Array = Temp;
				Capacity = NewSize;
			}
			else
				throw ExOps::EXCEPTION_MEM_FULL;
		}
		else if (isCurrentMemExternal){
			throw ExOps::EXCEPTION_EXTMEM_MOD;	//Attempted resizing of External memory
		}
		NRows = NewNRows;
		NCols = NewNCols;
	}
	inline void trim(){
		if (!isCurrentMemExternal){
			if (NRows > 0){
				T* Temp = reinterpret_cast<T*>(mxRealloc(Array, NRows*NCols*sizeof(T)));
				if (Temp != NULL)
					Array = Temp;
				else
					throw ExOps::EXCEPTION_MEM_FULL;
			}
			else{
				Array = NULL;
			}
			Capacity = NRows*NCols;
		}
		else{
			throw ExOps::EXCEPTION_EXTMEM_MOD; // trying to reallocate external memory
		}
	}
	inline void sharewith(MexMatrix<T> &M) const{
		if (!M.isCurrentMemExternal && M.Array != NULL)
			mxFree(M.Array);
		if (Capacity > 0){
			M.NRows = NRows;
			M.NCols = NCols;
			M.Capacity = Capacity;
			M.Array = Array;
			M.isCurrentMemExternal = true;
		}
		else{
			M.NRows = 0;
			M.NCols = 0;
			M.Capacity = 0;
			M.Array = NULL;
			M.isCurrentMemExternal = false;
		}
	}
	inline void clear(){
		if (!isCurrentMemExternal)
			NRows = 0;
		else
			throw ExOps::EXCEPTION_EXTMEM_MOD; //Attempt to resize External memory
	}
	inline int nrows() const{
		return NRows;
	}
	inline int ncols() const{
		return NCols;
	}
	inline int capacity() const{
		return Capacity;
	}
	inline bool ismemext() const{
		return isCurrentMemExternal;
	}
	inline bool isempty() const{
		return (NCols*NRows == 0);
	}
	inline bool istrulyempty() const{
		return Capacity == 0;
	}
};
#endif
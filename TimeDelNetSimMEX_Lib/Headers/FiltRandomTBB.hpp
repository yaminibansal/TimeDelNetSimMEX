#include <random>
#include "MexMem.hpp"

using namespace std;
class FiltRandomTBB : public mt19937{
public:
	void getState(MexVector<unsigned int> &V){
		V.resize(624);
		for (int i = 0; i < 624; ++i){
			V[i] = this->_At(i);
		}
	}
	void setState(MexVector<unsigned int> &V){
		for (int i = 0; i < 624; ++i)
			this->_Ax[i] = V[i] & _WMSK;
		this->_Idx = 624;
	}
	void setState(unsigned int Seed){
		this->seed(Seed);
	}
};

void Coolshit(){
	FiltRandomTBB YEAHHNN;
	YEAHHNN();
}
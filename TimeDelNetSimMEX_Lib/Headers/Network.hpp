#ifndef NETWORK_H
#define NETWORK_H
struct Synapse{
	int		NStart;
	int		NEnd;
	float	Weight;
	int	DelayinTsteps;
	int Plastic;
};

struct Neuron{
	float a;
	float b;
	float c;
	float d;
	float tmax;
};
#endif
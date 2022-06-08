// Constants used to navigate in 1d arrays

#include "scalar.H"

using namespace Foam;

// We only use singed int since label is signed too and we might need to compare.
typedef int len_t;
	
const len_t flatParticleLength = 12;

// Value used for padding
const scalar padVal = -333.333;

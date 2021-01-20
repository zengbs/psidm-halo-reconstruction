#include<cuda_runtime.h>
#include <cstdarg>

#define CUDA_CHECK_ERROR( Call )   CUDA_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )

inline bool CUDA_Check_Error( cudaError Return, const char *File, const int Line, const char *Func )
{
   if ( Return != cudaSuccess )
   {
      printf("CUDA ERROR : %s at %s : Line: %d ; Function: %s !!\n", cudaGetErrorString( Return ), File, Line, Func);
      return false;
   }
   else
      return true;
}

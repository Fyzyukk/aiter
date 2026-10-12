#include "../fused_register/launch.hip"
extern "C" int hipLaunchKernel(const void*,dim3,dim3,void**,size_t,void*){return 0;}
int main(){static_assert(sizeof(opus745_fused_kargs)==120);return 0;}

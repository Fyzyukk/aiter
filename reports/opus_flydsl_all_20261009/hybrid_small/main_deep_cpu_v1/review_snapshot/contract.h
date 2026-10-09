#pragma once
#include <cstdint>
inline bool main_shape_valid(int m,int n,int k,int bn,int fixed) {
 constexpr int64_t maxbytes=2147483647;
 return m>0 && n>0 && k>=128 && k<=16384 && m%16==0 && n%128==0 && n%bn==0 && k%128==0 &&
        (!fixed||k==fixed) && int64_t(m)*k<=maxbytes && int64_t(n)*k<=maxbytes && int64_t(m)*n*2<=maxbytes;
}
inline bool main_pointers_valid(const void* a,const void* b,const void* sa,const void* sb,const void* c) {
 return a&&b&&sa&&sb&&c && uintptr_t(a)%16==0 && uintptr_t(b)%16==0 && uintptr_t(sa)%16==0 && uintptr_t(c)%16==0;
}

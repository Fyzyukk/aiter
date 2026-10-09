#include <iostream>
#include <cstdint>
#include "hybrid_small/contract.h"
#include "small_split/contract.h"
#include "main_variants/contract.h"
#include "large9030/contract.h"
int main(){int index,m,n,k;while(std::cin>>index>>m>>n>>k){
const void* a=reinterpret_cast<const void*>(uintptr_t(0x100000000));
const void* b=reinterpret_cast<const void*>(uintptr_t(0x200000000));
const void* sa=reinterpret_cast<const void*>(uintptr_t(0x300000000));
const void* sb=reinterpret_cast<const void*>(uintptr_t(0x400000001));
void* c=reinterpret_cast<void*>(uintptr_t(0x500000000));
void* ws=reinterpret_cast<void*>(uintptr_t(0x600000000));
std::cout<<index<<" hybrid 9043 "<<(opus_private_small_shape_valid(9043,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9044 "<<(opus_private_small_shape_valid(9044,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9045 "<<(opus_private_small_shape_valid(9045,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9046 "<<(opus_private_small_shape_valid(9046,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9047 "<<(opus_private_small_shape_valid(9047,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9049 "<<(opus_private_small_shape_valid(9049,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9055 "<<(opus_private_small_shape_valid(9055,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" hybrid 9056 "<<(opus_private_small_shape_valid(9056,m,n,k)&&opus_private_small_pointer_valid(a,b,sa,sb,c))<<" 0\n";
{PrivateConfig cfg{};bool ok=private_config(110,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 110 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(111,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 111 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(120,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 120 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(121,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 121 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(130,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 130 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(140,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 140 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(210,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 210 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(211,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 211 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(220,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 220 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(221,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 221 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(230,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 230 "<<ok<<" "<<bytes<<"\n";}
{PrivateConfig cfg{};bool ok=private_config(231,true,m,n,k,cfg);uint64_t bytes=ok?private_workspace_bytes(cfg,m,n):0;ok=ok&&private_pointers_valid(cfg,a,b,sa,sb,c,bytes?ws:nullptr,bytes,m,n,k);std::cout<<index<<" small 231 "<<ok<<" "<<bytes<<"\n";}
std::cout<<index<<" main 92000 "<<(main_shape_valid(m,n,k,128,0)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92001 "<<(main_shape_valid(m,n,k,128,0)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92002 "<<(main_shape_valid(m,n,k,128,0)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92003 "<<(main_shape_valid(m,n,k,128,384)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92004 "<<(main_shape_valid(m,n,k,128,768)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92005 "<<(main_shape_valid(m,n,k,128,1536)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92006 "<<(main_shape_valid(m,n,k,128,1536)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92007 "<<(main_shape_valid(m,n,k,128,3072)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92008 "<<(main_shape_valid(m,n,k,128,7168)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92009 "<<(main_shape_valid(m,n,k,128,7168)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92010 "<<(main_shape_valid(m,n,k,128,3072)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92011 "<<(main_shape_valid(m,n,k,64,7168)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92012 "<<(main_shape_valid(m,n,k,64,7168)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92100 "<<(main_shape_valid(m,n,k,256,384) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92101 "<<(main_shape_valid(m,n,k,256,1536) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92102 "<<(main_shape_valid(m,n,k,256,3072) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92103 "<<(main_shape_valid(m,n,k,256,7168) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92104 "<<(main_shape_valid(m,n,k,256,16384) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92110 "<<(main_shape_valid(m,n,k,256,384) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92111 "<<(main_shape_valid(m,n,k,256,1536) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92112 "<<(main_shape_valid(m,n,k,256,3072) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92113 "<<(main_shape_valid(m,n,k,256,7168) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92114 "<<(main_shape_valid(m,n,k,256,16384) && m%256==0&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" main 92120 "<<(main_shape_valid(m,n,k,256,7168)&&main_pointers_valid(a,b,sa,sb,c))<<" 0\n";
std::cout<<index<<" large 0 "<<(private9030_shape_valid(m,n,k,true)&&private9030_pointer_valid(a,b,sa,sb,c))<<" 0\n";
}}

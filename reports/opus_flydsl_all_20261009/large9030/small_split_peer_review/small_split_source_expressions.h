#pragma once
#include "candidate/traits.cuh"
#include "contract.h"
struct KernelArgs { int k; };
struct RegisterPartition {int global_begin,global_loops,tile_begin,loops;};
template<class T> RegisterPartition register_partition(int k,int split,int wk) { KernelArgs args{k};
    const int total_tiles = (T::FIXED_K ? T::FIXED_K : args.k) / T::W_K;
    const int global_per = total_tiles / T::GLOBAL_SPLIT_K;
    const int global_extra = total_tiles % T::GLOBAL_SPLIT_K;
    const int global_loops = global_per + (split < global_extra);
    const int global_begin = split * global_per + (split < global_extra ? split : global_extra);
    const int tiles_per_wave = global_loops / T::WAVE_K;
    const int extra_tiles = global_loops % T::WAVE_K;
    const int loops = tiles_per_wave + (wk < extra_tiles);
    const int tile_begin = global_begin + wk * tiles_per_wave + (wk < extra_tiles ? wk : extra_tiles);
return {global_begin,global_loops,tile_begin,loops};
}
struct FinePartition {int tile_begin,loops;};
template<class T> FinePartition fine_partition(int k,int split) {KernelArgs args{k};
    const int total_loops = (T::FIXED_K ? T::FIXED_K : args.k) / T::B_K;
    const int per_split = total_loops / T::SPLIT_K, extra = total_loops % T::SPLIT_K;
    const int loops = per_split + (split < extra);
    const int tile_begin = split * per_split + (split < extra ? split : extra);
return {tile_begin,loops};
}
struct ReviewedShape {int m,n,k,parent,baseline_split,baseline_bm,baseline_bn; bool fine;};
const ReviewedShape reviewed_shapes[] = {
{16,768,7168,9041,1,16,16,false},
{112,768,7168,9052,1,16,32,false},
{16,2048,7168,9041,1,16,16,false},
{32,2048,7168,9041,1,16,16,false},
{32,768,7168,9041,1,16,16,false},
{48,7168,16384,9063,4,48,128,true},
{1408,768,7168,9062,2,96,128,true},
{928,768,7168,9063,4,96,128,true},
{832,768,7168,9063,4,96,128,true},
{32,7168,16384,9053,1,32,32,false},
{960,768,7168,9063,4,96,128,true},
{16,6144,7168,9052,1,16,32,false},
{896,768,7168,9063,4,96,128,true},
{32,7168,3072,9042,1,32,32,false},
{64,7168,1024,9054,1,32,64,false},
{8,768,7168,9041,1,16,16,false},
{4,6144,7168,9041,1,16,16,false},
{8,2048,7168,9041,1,16,16,false},
{1536,768,7168,9062,2,96,128,true},
{64,7168,16384,9063,4,80,128,true},
{8,7168,16384,9053,1,32,32,false},
{16,7168,16384,9053,1,32,32,false},
{8,6144,7168,9041,1,16,16,false},
{864,768,7168,9063,4,96,128,true},
{1,6144,7168,9041,1,16,16,false},
{48,7168,768,9054,1,32,64,false},
{16,7168,3072,9052,1,16,32,false},
{2,7168,16384,9041,1,16,16,false},
{4,7168,16384,9041,1,16,16,false},
{2,6144,7168,9041,1,16,16,false},
{16,7168,7168,9052,1,16,32,false},
{1,7168,16384,9041,1,16,16,false},
{800,768,7168,9063,4,80,128,true},
{48,768,7168,9041,1,16,16,false},
{80,6144,7168,9063,4,80,128,true},
{768,768,7168,9063,4,80,128,true},
{8,7168,7168,9041,1,16,16,false},
{32,7168,1024,9053,1,32,32,false},
{736,768,7168,9063,4,80,128,true},
{4,7168,7168,9041,1,16,16,false},
{288,2048,7168,9063,4,80,128,true},
{64,768,7168,9041,1,16,16,false},
{1792,768,7168,9062,2,96,128,true},
{1728,768,7168,9062,2,96,128,true},
{1600,768,7168,9062,2,96,128,true},
{128,7168,16384,9063,4,128,128,true},
{4,65536,1536,9054,1,32,64,false},
{4,768,7168,9041,1,16,16,false},
{704,768,7168,9063,4,80,128,true},
{1,7168,7168,9041,1,16,16,false},
{4,2048,7168,9041,1,16,16,false},
{2,7168,3072,9052,1,16,32,false},
{4,7168,3072,9052,1,16,32,false},
{320,2048,7168,9063,4,80,128,true},
{64,7168,768,9054,1,32,64,false},
{1,7168,1024,9051,1,16,32,false},
{2,65536,1536,9051,1,16,32,false},
{8,7168,384,9040,1,16,32,false},
{352,2048,7168,9063,4,96,128,true},
{4,7168,1024,9051,1,16,32,false},
{2,7168,1024,9051,1,16,32,false},
{8,16384,1536,9051,1,16,32,false},
{1,2048,7168,9041,1,16,16,false},
{80,768,7168,9041,1,16,16,false},
{1216,2048,7168,9060,1,80,128,true},
{2,16384,1536,9051,1,16,32,false},
{2,7168,7168,9041,1,16,16,false},
{352,7168,3072,9061,1,96,128,true},
{576,2048,7168,9062,2,96,128,true},
{1,65536,1536,9052,1,16,32,false},
{544,2048,7168,9062,2,96,128,true},
{144,16384,1536,9060,1,80,128,true},
{384,2048,7168,9063,4,96,128,true},
{1344,2048,7168,9061,1,96,128,true},
{1408,2048,7168,9061,1,96,128,true},
{8,7168,768,9051,1,16,32,false},
{1152,2048,7168,9060,1,80,128,true},
{1472,2048,7168,9061,1,96,128,true},
{1280,2048,7168,9060,1,80,128,true},
{8,7168,3072,9052,1,16,32,false},
{80,7168,16384,9063,4,80,128,true},
{832,7168,16384,9061,1,96,128,true},
{1536,2048,7168,9061,1,96,128,true},
{16,7168,768,9040,1,16,32,false},
{2,7168,384,9040,1,16,32,false},
{1,7168,3072,9052,1,16,32,false},
{160,16384,1536,9060,1,80,128,true},
{112,7168,16384,9063,4,112,128,true},
{2,7168,768,9040,1,16,32,false},
{864,7168,16384,9061,1,96,128,true}
};

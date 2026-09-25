// CPU memory substitutes preserve production indexing. Coordinates use int
// tokens so address aliases cannot be hidden by repeated 8-bit test values.
namespace cpu_check {
struct CoordinateGlobal {
    std::vector<int> reads;
    explicit CoordinateGlobal(int elements) : reads(elements, 0) {}
};
struct CoordinateLDS {
    std::vector<int> values, reads, writes;
    CoordinateLDS* ptr = this;
    int lane = 0;
    explicit CoordinateLDS(int bytes) : values(bytes, -1), reads(bytes, 0), writes(bytes, 0) {}
    void reset_slot(int offset, int size) {
        std::fill_n(values.begin() + offset, size, -1);
        std::fill_n(reads.begin() + offset, size, 0);
        std::fill_n(writes.begin() + offset, size, 0);
    }
};
template<int Vec, class LayoutG, class LayoutS, class Offset>
void async_load(CoordinateGlobal& global, CoordinateLDS* shared,
                const LayoutG& source_layout, const LayoutS& dest_layout, Offset scalar_offset) {
    const auto source = opus::layout_to_offsets<Vec>(source_layout);
    const auto dest = opus::layout_to_offsets<Vec>(dest_layout);
    assert(source.size() == dest.size());
    for (int i = 0; i < source.size(); ++i) for (int j = 0; j < Vec; ++j) {
        const int src = source[i] + static_cast<int>(scalar_offset) + j;
        const int dst = dest[i] + shared->lane * Vec + j;
        assert(src >= 0 && src < static_cast<int>(global.reads.size()));
        assert(dst >= 0 && dst < static_cast<int>(shared->values.size()));
        assert(++shared->writes[dst] == 1);
        assert(++global.reads[src] == 1);
        shared->values[dst] = src;
    }
}
template<int Vec> auto load(CoordinateLDS& shared, int offset) {
    assert(offset >= 0 && offset + Vec <= static_cast<int>(shared.values.size()));
    std::array<int, Vec> result;
    for (int j = 0; j < Vec; ++j) {
        assert(shared.writes[offset + j] == 1);
        ++shared.reads[offset + j];
        result[j] = shared.values[offset + j];
    }
    return result;
}

struct Memory {
    std::vector<unsigned char> values;
    std::vector<int> reads, writes;
    explicit Memory(int size) : values(size, 0xa5), reads(size, 0), writes(size, 0) {}
};
template<int Vec> auto load(Memory& memory, int index) {
    assert(index >= 0 && index + Vec <= static_cast<int>(memory.values.size()));
    opus::vector_t<unsigned char, Vec> result;
    for (int j = 0; j < Vec; ++j) {
        ++memory.reads[index + j];
        result[j] = memory.values[index + j];
    }
    return result;
}
template<int Vec, class Value> void store(Memory& memory, const Value& value, int index) {
    assert(index >= 0 && index + Vec <= static_cast<int>(memory.values.size()));
    for (int j = 0; j < Vec; ++j) {
        assert(++memory.writes[index + j] == 1);
        memory.values[index + j] = value[j];
    }
}
}

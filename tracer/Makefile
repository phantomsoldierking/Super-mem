LLVM_CONFIG ?= llvm-config
CXX := clang++
CC := clang
BUILD := build

all: $(BUILD)/libMemTracePass.so $(BUILD)/memtrace_rt.o

$(BUILD)/libMemTracePass.so: MemTracePass.cpp
	@mkdir -p $(BUILD)
	$(CXX) -shared -fPIC -O2 $(shell $(LLVM_CONFIG) --cxxflags) $< -o $@ $(shell $(LLVM_CONFIG) --ldflags)

$(BUILD)/memtrace_rt.o: memtrace_rt.c
	@mkdir -p $(BUILD)
	$(CC) -O3 -fPIC -c $< -o $@

test: all
	$(CC) -O2 -fpass-plugin=$(BUILD)/libMemTracePass.so test/stream.c $(BUILD)/memtrace_rt.o -o $(BUILD)/stream
	MT_OUT=$(BUILD)/stream.trace MT_WARM=1000 MT_INSTS=2000000 $(BUILD)/stream
	@cat $(BUILD)/stream.trace.json
	@head -3 $(BUILD)/stream.trace

clean:
	rm -rf $(BUILD)

.PHONY: all test clean

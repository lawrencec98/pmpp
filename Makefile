NVCC := nvcc

CU_FILES := $(wildcard *.cu)

TARGETS := $(CU_FILES:.cu=)

all: $(TARGETS)

%: %.cu
	$(NVCC) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
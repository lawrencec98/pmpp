NVCC := nvcc

SRC := src
DEPENDS := depends
OPENCV := $(DEPENDS)/opencv4

# OpenCV flags
OPENCV_INC := -I$(OPENCV)/include/opencv4
OPENCV_LIB := -L$(OPENCV)/lib
OPENCV_LINK := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

RPATH := -Xlinker -rpath -Xlinker $(OPENCV)/lib




CU_FILES := $(wildcard $(SRC)/*.cu)

TARGETS := $(notdir $(CU_FILES:.cu=.out))

all: $(TARGETS)

%.out: $(SRC)/%.cu
	$(NVCC) $< -o $@ $(OPENCV_INC) $(OPENCV_LIB) $(OPENCV_LINK) $(RPATH)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
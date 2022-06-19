KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_DEVICES = Cuda,OpenMP
EXE_NAME = "LBM"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"



CXX = mpicxx
EXE = ${EXE_NAME}.exe
KOKKOS_ARCH = "Ampere80"
KOKKOS_CUDA_OPTIONS = "enable_lambda"

CXXFLAGS = -O3 -g
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.dat *.exe

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)

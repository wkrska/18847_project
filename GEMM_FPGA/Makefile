VERSION = v1

GEMMF = build/gemm_$(VERSION)_fpgaG_$(MATDIM)_$(TILE_A)_$(TILE_B)_$(DWIDTH)_$(SEED)_$(MHZ)
GEMMP = build/gemm_$(VERSION)_vTune_$(MATDIM)_$(TILE_A)_$(TILE_B)_$(DWIDTH)_$(SEED)_$(MHZ)
GEMMH = build/gemm_$(VERSION)_hw_$(MATDIM)_$(TILE_A)_$(TILE_B)_$(DWIDTH)_$(SEED)_$(MHZ)

BOARD_PACKAGE = $(OFS_BSP_BASE)
TARGET = ofs_n6001
# TARGET = ofs_n6001_usm
# TARGET = ofs_n6001_iopipes

MATDIM = -DMATDIM=512
TILE_A = -DTILE_A=32
TILE_B = -DTILE_B=32
DWIDTH = -DDWIDTH=8

REPETITIONS = 1

CONFIG = $(MATDIM) $(TILE_A) $(TILE_B) $(DWIDTH)

MHZ = 650MHz

SEED = 0

EARLY = -fsycl-link=early

SIMULATE = -DFPGA_EMULATOR=1 -fsycl -fintelfpga -Wall -Wextra -g -O0 -w #-fp-model=precise -no-fma #-qactypes 

SYNTHESIS = -Xsv -fsycl -fintelfpga -DFPGA_HARDWARE=1 -Xshardware -Xsclock=$(MHZ) -Xsseed=$(SEED) -Xsboard-package=$(BOARD_PACKAGE) -Xstarget=$(TARGET) -DIS_BSP -O3 #-fp-model=precise -no-fma #-qactypes 

RTLSIM = -Xsv -fsycl -fintelfpga -DFPGA_SIMULATOR=1 -Xssimulation #-fp-model=precise -no-fma #-qactypes 

GEMMSRC = matmul_demo.cpp
# GEMMSRC = main.cpp

INCLUDE = -I ./include/
# INCLUDE = -I $(HOME)/Documents/oneAPI-samples/DirectProgramming/C++SYCL_FPGA/include/

sim:
	icpx $(CONFIG) $(SIMULATE) $(INCLUDE) $(GEMMSRC) -o build/gemm_sim
	build/gemm_sim $(REPETITIONS) > test.txt

rtl:
	icpx $(CONFIG) $(RTLSIM) $(INCLUDE) $(GEMMSRC) -o build/gemm_rtl

fpga:
	icpx $(CONFIG) $(SYNTHESIS) $(INCLUDE) $(GEMMSRC) -o $(GEMMF)

link:
	icpx $(CONFIG) $(SYNTHESIS) $(EARLY) $(INCLUDE) $(GEMMSRC) -o $(GEMMH)

profile: 
	icpx $(CONFIG) -Xsprofile $(SYNTHESIS) $(INCLUDE) $(GEMMSRC) -o $(GEMMP)
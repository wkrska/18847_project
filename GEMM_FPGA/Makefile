TIMERUN = -DTIMERUN=0

VERSION = v8

AGGF = build/agg_$(VERSION)_fpgaG$(PP)$(RR)$(CC)$(LL)$(TIMERUN)_$(SEED)_$(MHZ)
AGGP = build/agg_$(VERSION)_vTune$(PP)$(RR)$(CC)$(LL)$(TIMERUN)_$(SEED)_$(MHZ)
AGGH = build/agg_$(VERSION)_hw$(PP)$(RR)$(CC)$(LL)$(TIMERUN)_$(SEED)_$(MHZ)

PP = -D_P_=1
RR = -D_R_=8192
CC = -D_C_=8
LL = -D_L_=1
# If higher parallelism seems to not work in emulation, try 'export DPCPP_CPU_NUM_CUS=...'

# Defines what aggregation operation is being performed
#	0 DYNAMIC
#	1 SUM/Avg/Cnt ONLY
#	2 Min ONLY
#	3 Max ONLY
AGG_OP = -DAGG_OP=0

WHICH = -DWHICH=0

CONFIG = $(PP) $(RR) $(CC) $(LL) $(AGG_OP)


BOARD_PACKAGE = $(OFS_BSP_BASE)
TARGET = ofs_n6001
# TARGET = ofs_n6001_usm
# TARGET = ofs_n6001_iopipes

MHZ = 650MHz

SEED = 0

EARLY = -fsycl-link=early

SIMULATE = -DNO_MY_ASSERT=0 -DFPGA_EMULATOR=1 -fsycl -fintelfpga -Wall -Wextra -g -O0 -w #-qactypes 

SYNTHESIS = -DNO_MY_ASSERT=1 -Xsv -fsycl -fintelfpga -DFPGA_HARDWARE=1 -Xshardware -Xsclock=$(MHZ) -Xsseed=$(SEED) -Xsboard-package=$(BOARD_PACKAGE) -Xstarget=$(TARGET) -DIS_BSP -O3 # -qactypes 

RTLSIM = -DNO_MY_ASSERT=1 -Xsv -fsycl -fintelfpga -DFPGA_SIMULATOR=1 -Xssimulation #-Xstarget=$(TARGET) #-DIS_BSP -O3 # -qactypes 

AGGSRC = main.cpp

# INCLUDE = -I ../include/
INCLUDE = -I $(HOME)/Documents/oneAPI-samples/DirectProgramming/C++SYCL_FPGA/include/

sim:
	icpx $(TIMERUN) $(CONFIG) $(SIMULATE) $(INCLUDE) $(AGGSRC) -o build/agg_sim
	build/agg_sim > test.txt

rtl:
	icpx $(TIMERUN) $(CONFIG) $(RTLSIM) $(INCLUDE) $(AGGSRC) -o build/agg_rtl

fpga:
	icpx $(TIMERUN) $(CONFIG) $(SYNTHESIS) $(INCLUDE) $(AGGSRC) -o $(AGGF)

link:
	icpx $(TIMERUN) $(CONFIG) $(SYNTHESIS) $(EARLY) $(INCLUDE) $(AGGSRC) -o $(AGGH)

profile: 
	icpx $(TIMERUN) $(CONFIG) -Xsprofile $(SYNTHESIS) $(INCLUDE) $(AGGSRC) -o $(AGGP)

############# misc ##################

clang:
	clang-format -i *.cpp *.h

clean:
	rm -f *#*#* *~ test.txt simplerun

reset:
	/ofs/ofs-2023.1-1/ofs_init.sh

resetold:
	/ofs/ofs-2023.1-1/ofs_init_old.sh

query:
	/ofs/ofs-2023.1-1/ofs_n6001_eval.sh


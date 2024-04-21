AGGF = build/TF2_fpgaG$(MODEL)_$(MHZ)

MODEL =-DRESNET50
# MODEL =-DRESNET50_PRUNED
# MODEL =

CONFIG = $(MODEL)

TARGET = $(OFS_BSP_BASE):ofs_n6001
# TARGET = intel_s10sx_pac:pac_s10
# TARGET = de10_agilex:B2E2_8GBx4tmux 

-board-package=<board_package_path> -board=<board_name>

MHZ = 650MHz

SIMULATE = $(CONFIG) -march=emulator -Wall -g -O0

SYNTHESIS = $(CONFIG) -v -O3

TF2_SRC = Runtime_Engine/cnn/device/src/cnn.cl

sim:
	aoc $(CONFIG) $(SIMULATE) $(TF2_SRC) -o build/TF2_sim
	build/agg_sim > test.txt

fpga:
	aoc $(CONFIG) $(SYNTHESIS) $(TF2_SRC) -o $(AGGF)

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


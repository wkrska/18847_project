#!/bin/bash

# Define the command
command="icpx -Xsv -fsycl -fintelfpga -DFPGA_HARDWARE=1 -Xshardware -Xsclock=650MHz -Xsseed=0 -Xsboard-package=/temp_ofs_install/ofs/n6001/2023.3/oneapi-asp/n6001 -Xstarget=ofs_n6001 -DIS_BSP -O3 -I ./include/"
src="matmul_demo.cpp"

# Define the list of arguments (as an array)

# 0
# n_args=5
# args_matdim=( "2" "2" "2" "2" "2" )
# args_tile_a=( "1" "1" "1" "1" "1" )
# args_tile_b=( "1" "1" "1" "1" "1" )
# args_dwidth=( "32" "16" "12" "8" "4")

# 1
# n_args=5
# args_matdim=( "16" "16" "16" "16" "16" )
# args_tile_a=( "16" "16" "16" "16" "16" )
# args_tile_b=( "16" "16" "16" "16" "16" )
# args_dwidth=( "32" "16" "12" "8" "4")

# 2
n_args=5
args_matdim=( "64" "64" "64" "64" "64" )
args_tile_a=( "16" "32" "32" "32" "32" )
args_tile_b=( "16" "32" "32" "32" "32" )
args_dwidth=( "32" "16" "12" "8" "4")

# 3
# n_args=5
# args_matdim=( "256" "256" "512" "256" "256" )
# args_tile_a=( "32" "32" "32" "32" "32" )
# args_tile_b=( "32" "32" "32" "32" "32" )
# args_dwidth=( "16" "12" "12" "8" "4")


# Loop through each argument
# Loop through each argument
for i in $(seq 0 $n_args); do
    # Run the command with the current argument
    echo "gemm_v1_fpgaG_-DMATDIM="${args_matdim[i]}"-DTILE_A="${args_tile_a[i]}"-DTILE_B="${args_tile_b[i]}"-DDWIDTH="${args_dwidth[i]}"_0_650MHz"
    $command "-DMATDIM="${args_matdim[i]} "-DTILE_A="${args_tile_a[i]} "-DTILE_B="${args_tile_b[i]} "-DDWIDTH="${args_dwidth[i]} $src -o build/gemm_v1_fpgaG_-DMATDIM="${args_matdim[i]}"-DTILE_A="${args_tile_a[i]}"-DTILE_B="${args_tile_b[i]}"-DDWIDTH="${args_dwidth[i]}"_0_650MHz
done

echo "Finished running $command with all arguments."
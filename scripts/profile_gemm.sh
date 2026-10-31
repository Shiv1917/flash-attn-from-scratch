#!/bin/sh
# Nsight Compute profile of each GEMM kernel -> profiles/<kernel>.ncu-rep + a metrics CSV.
# Requires access to GPU performance counters (on the DoC gpu32 box ncu fails with
# ERR_NVGPUCTRPERM: counters are root-only there). Run where you have counter access, e.g.
#   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0   (or run ncu with sudo)
# Usage: scripts/profile_gemm.sh [N=2048]
set -e
N=${1:-2048}
mkdir -p profiles
for k in 01_naive 01b_naive_coalesced 02_tiled_smem 03_register_tiled; do
  ncu --set full -f -k regex:gemm -c 1 -o profiles/$k ./build/$k $N $N $N
  ncu --import profiles/$k.ncu-rep --csv --page raw --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active > profiles/$k.metrics.csv
done

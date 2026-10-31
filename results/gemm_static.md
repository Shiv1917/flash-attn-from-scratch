| kernel | threads/block | regs/thread | static smem (B) | spill stores (B) | blocks/SM | theoretical occupancy |
|---|---|---|---|---|---|---|
| 01_naive | 1024 | 40 | 0 | 0 | 1 | 67% |
| 01b_naive_coalesced | 1024 | 40 | 0 | 0 | 1 | 67% |
| 02_tiled_smem | 1024 | 38 | 8192 | 0 | 1 | 67% |
| 03_register_tiled | 256 | 118 | 8192 | 0 | 2 | 33% |

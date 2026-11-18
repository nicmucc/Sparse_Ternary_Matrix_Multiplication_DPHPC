# Whole-tile ownership: V7 versus V6

B=32. Identical weights, block width and prepared format within every comparison.
Ratios below are geometric means over regular iid matrix shapes and sparsities; the first table excludes one-thread runs.

| Scheduling situation | Workload points | V7/V6 speedup |
|---|---|---|
| Fewer tiles than requested threads | 256 | 0.386× |
| Tiles divide evenly across threads | 160 | 0.997× |
| Enough tiles, uneven division | 144 | 0.920× |

Static V6 partitions a contiguous flattened (tile, group) iteration space. When the tile count divides evenly by the thread count, each static range can already contain complete tiles. V7 then has similar ownership while expressing it with a simpler loop.
When there are 16 tiles and 10 threads, V7 assigns two tiles to some workers and one to others. V6 can split tiles at group boundaries, giving finer load balance.
When there are fewer tiles than threads, V7 has no group-level parallelism to fill the remaining workers.
These statements describe scheduling. Ratios alone do not establish whether cache misses or instruction overhead caused a measured difference.

## Best observed V7 thread count by input size

Based on regular-matrix geometric means at fixed block width. For M=32, V7's parallel-region condition is false and it runs serially at every requested thread setting.

| M | Tiles | Best requested threads | V7 speedup vs its own 1T | V7/V6 at that setting |
|---|---|---|---|---|
| 32 | 1 | serial for all settings | 1.00× | 1.01× |
| 64 | 2 | 10 | 1.86× | 0.28× |
| 256 | 8 | 10 | 6.88× | 0.84× |
| 512 | 16 | 14 | 6.86× | 0.99× |
| 1024 | 32 | 14 | 7.00× | 1.03× |

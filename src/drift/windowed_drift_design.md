- 2 variants
- 1 totally in memory
- 1 that spills to disk

- in memory variant
    - holds all windows in a single contiguous chunk of memory
        - each individual window will have a window_size chunk of the entire allocation
        - base offset determines the current window
        - this chunk is used a ring buffer
        - when the entire buffer is saturated, then the offset wraps around
        - each window that is saturated, ie totally filled, gets cleaned up only after the entire buffer has been saturated
    - global histogram bins are stored for cheap computation
        - when a window gets "cleaned up" then the global histogram is cleaned to remove the state of the window that is getting cleaned up
        - drift computations are done using the global set
    - pushing data can be done with one of 4 methods
        - single example push with drift reporting
        - single example push without drift reporting
        - batch push with drift reporting
        - batch push without drift reporting
        - the drift reporting methods can take any number of drift metrics to report on
    - current window is stored as an offset
        - The given bin to update is at (window_idx * window_size) + bin_idx

- The spill to disk variant
    - There is a tempfile that stores up to total buffer size * size_of(f64/f32)
    - Only the global and current window is in memory
    - When a window is saturated, it is written to disk at the offset window_size * window index
    - If the buffer is entirely saturated, the window to be evicted is pulled into memory
        - The global window state gets resolved
        - The current window buffer is cleared
    - The file buffer is then a sort of ring buffer for the lifecycle of the stream
    - A full seek is only paid once every cycle around the buffer

- Questions
    - What is the total buffer size where storing the buffer on disk is requierd?
    - Should this be dynamic or strictly specified by users?


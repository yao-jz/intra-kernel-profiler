## Trace instrumentation & capacity sizing (practical notes)

### Event count and CAP sizing

Per-warp event count is roughly:

- `2 * (#begin/end pairs)` + `1 * (#marks)`

Ring-buffer behavior: once `cnt > per_warp_cap`, only the **last** `per_warp_cap` events are retained and the earliest events are overwritten.

### Common pitfall: unmatched B/E

Macros record only on **lane0** by default, so you must ensure:

- lane0 does not take a different control-flow that breaks begin/end pairing (especially avoid lane0 early-exit)
- every begin eventually has an end
- call `IKP_TRACE_CTX_FLUSH` before the kernel finishes

`unmatched_begin/unmatched_end` in the summary JSON is the fastest diagnostic signal.

### Make traces smaller (faster iteration)

Two common methods:

- **Block filtering**: `sess.set_block_filter({0, 7});`
- **Sampling**: use `IKP_TRACE_REC_IF` to record once every \(2^k\) iterations in tight loops

See `examples/trace/block_filter.cu` and `examples/trace/sampled_loop.cu`.


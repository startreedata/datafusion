// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmark for Arrow IPC serialization performance.
//!
//! This benchmark measures the overhead of serializing RecordBatches to Arrow IPC format,
//! comparing two approaches:
//!
//! 1. **serialize_to_sink**: Writes to a sink that discards data (measures pure CPU cost)
//! 2. **serialize_to_ipc**: Writes to a Vec<u8> (includes memory allocation overhead)
//!
//! The benchmark helps understand:
//! - Pure serialization CPU cost vs. memory allocation overhead
//! - How serialization performance scales with data size and binary column sizes
//! - Cost of IPC format encoding (metadata + data alignment)
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench serde -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench serde -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run only the sink benchmark
//! cargo bench --bench serde -p datafusion-physical-plan -- serialize_to_sink
//!
//! # Run only the memory benchmark
//! cargo bench --bench serde -p datafusion-physical-plan -- serialize_to_memory
//!
//! # Change measurement time (per benchmark, default is 5 seconds)
//! cargo bench --bench serde -p datafusion-physical-plan -- --measurement-time 10
//!
//! # Run specific configuration
//! cargo bench --bench serde -p datafusion-physical-plan -- "1M_rows_binary_10B"
//! ```
//!
//! ## Baseline Management
//!
//! ```bash
//! # Save current results as a named baseline
//! cargo bench --bench serde -p datafusion-physical-plan -- --save-baseline my-baseline
//!
//! # Compare against a specific baseline
//! cargo bench --bench serde -p datafusion-physical-plan -- --baseline my-baseline
//!
//! # Delete all benchmark history and start fresh
//! rm -rf target/criterion
//! ```

// Include shared benchmark utilities
#[path = "bench_utils.rs"]
mod bench_utils;

use std::hint::black_box;
use std::sync::Arc;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};

use bench_utils::{
    FunctionalBatchGenerator, create_schema, serialize_batches_to_sink
};

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Benchmarks serialization to a sink that drops all data.
///
/// This measures the pure CPU cost of Arrow IPC serialization without
/// including memory allocation or I/O overhead. Useful for understanding
/// the baseline serialization cost.
fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Test different binary column sizes to understand serialization overhead
    let binary_sizes = vec![10, 1024, 2048];

    for binary_size in binary_sizes {
        let label = format!("1M_rows_binary_{binary_size}B");

        // Generate test data
        let schema = create_schema();
        let mut generator = FunctionalBatchGenerator::new(
            Arc::clone(&schema),
            rows_per_batch,
            num_batches,
            binary_size,
        );
        let batches = generator.generate_batches();

        // Calculate expected output size for throughput metric
        let expected_size = estimate_serialized_size(&batches, binary_size, total_rows);

        // Set throughput metric for bytes/second calculations
        group.throughput(Throughput::Bytes(expected_size as u64));

        // Log configuration
        println!(
            "Config (sink): {} rows, binary_size={} bytes, estimated output={:.2} MB",
            total_rows,
            binary_size,
            expected_size as f64 / (1024.0 * 1024.0)
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let bytes_written = serialize_batches_to_sink(batches, &schema);
                    // black_box prevents compiler from optimizing away unused results
                    black_box(bytes_written)
                })
            },
        );
    }

    group.finish();
}

/// Estimates the serialized size of batches for throughput calculations.
///
/// This is an approximation based on the data types and sizes. For accurate
/// measurements, we could do one actual serialization, but this is good enough
/// for throughput reporting.
fn estimate_serialized_size(batches: &[arrow::array::RecordBatch], binary_size: usize, total_rows: usize) -> usize {
    // Rough estimate of IPC overhead:
    // - File header/footer: ~1KB
    // - Per-batch metadata: ~200 bytes per batch
    // - Per-column data: actual column size + alignment padding

    let num_batches = batches.len();
    let overhead = 1024 + (num_batches * 200);

    // Data size estimation per row:
    // - Int32: 4 bytes
    // - Int64: 8 bytes
    // - Float32: 4 bytes
    // - Float64: 8 bytes
    // - StringView: ~16 bytes (view) + actual string data (~6 bytes for "str_XX")
    // - BinaryView: ~16 bytes (view) + binary_size bytes
    let per_row_size = 4 + 8 + 4 + 8 + 16 + 6 + 16 + binary_size;

    overhead + (total_rows * per_row_size)
}

criterion_group!(benches, bench_serialize);
criterion_main!(benches);


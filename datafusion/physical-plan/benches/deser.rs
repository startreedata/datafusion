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

//! Benchmark for Arrow IPC deserialization performance.
//!
//! This benchmark measures the overhead of deserializing Arrow IPC data into RecordBatches,
//! comparing two approaches:
//!
//! 1. **deserialize_from_ipc**: Standard deserialization that copies data
//! 2. **deserialize_zero_copy**: Zero-copy deserialization using Buffer slicing
//!
//! The benchmark helps understand:
//! - Standard deserialization cost vs. zero-copy deserialization
//! - How deserialization performance scales with data size and binary column sizes
//! - Cost of IPC format decoding (metadata parsing + data access)
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench deser -p datafusion-physical-plan
//!
//! # Run only the standard deserialization benchmark
//! cargo bench --bench deser -p datafusion-physical-plan -- deserialize_standard
//!
//! # Run only the zero-copy deserialization benchmark
//! cargo bench --bench deser -p datafusion-physical-plan -- deserialize_zero_copy
//!
//! # Change measurement time (per benchmark, default is 5 seconds)
//! cargo bench --bench deser -p datafusion-physical-plan -- --measurement-time 10
//!
//! # Run specific configuration
//! cargo bench --bench deser -p datafusion-physical-plan -- "1M_rows_binary_10B"
//! ```
//!
//! ## Baseline Management
//!
//! ```bash
//! # Save current results as a named baseline
//! cargo bench --bench deser -p datafusion-physical-plan -- --save-baseline my-baseline
//!
//! # Compare against a specific baseline
//! cargo bench --bench deser -p datafusion-physical-plan -- --baseline my-baseline
//!
//! # Delete all benchmark history and start fresh
//! rm -rf target/criterion
//! ```

// Include shared benchmark utilities
#[path = "bench_utils.rs"]
mod bench_utils;

use std::hint::black_box;
use std::sync::Arc;

use arrow::buffer::Buffer;
use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};

use bench_utils::{
    FunctionalBatchGenerator, create_schema, deserialize_zero_copy, serialize_to_ipc,
};

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Benchmarks zero-copy IPC deserialization.
///
/// This measures the cost of Arrow IPC deserialization using zero-copy
/// techniques where Arrow arrays reference the original buffer directly
/// via Buffer slicing. This avoids copying the actual data and only
/// creates lightweight views into the existing buffer.
///
/// This is the most efficient deserialization approach when you have
/// a contiguous buffer (e.g., mmap'd file or received network buffer).
fn bench_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialize_standard");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Test different binary column sizes to understand deserialization overhead
    let binary_sizes = vec![10, 1024, 2048];

    for binary_size in binary_sizes {
        let label = format!("1M_rows_binary_{binary_size}B");

        // Generate test data and serialize to IPC format
        let schema = create_schema();
        let mut generator = FunctionalBatchGenerator::new(
            Arc::clone(&schema),
            rows_per_batch,
            num_batches,
            binary_size,
        );
        let batches = generator.generate_batches();
        let ipc_data = serialize_to_ipc(&batches, &schema);

        // Convert to Buffer for zero-copy deserialization
        let buffer = Buffer::from_vec(ipc_data);

        // Set throughput metric for bytes/second calculations
        group.throughput(Throughput::Bytes(buffer.len() as u64));

        // Log configuration
        println!(
            "Config (zero-copy): {} rows, binary_size={} bytes, IPC size={:.2} MB",
            total_rows,
            binary_size,
            buffer.len() as f64 / (1024.0 * 1024.0)
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &buffer,
            |b, buffer| {
                b.iter(|| {
                    let (schema, batches) = deserialize_zero_copy(buffer);
                    // black_box prevents compiler from optimizing away unused results
                    black_box((schema, batches))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_deserialize);
criterion_main!(benches);

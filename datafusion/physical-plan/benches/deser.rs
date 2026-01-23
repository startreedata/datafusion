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

//! Benchmark for DataFusion FilterExec with Arrow IPC serialization.
//!
//! This benchmark measures the end-to-end latency of:
//! 1. Deserializing Arrow IPC data into RecordBatches
//! 2. Executing a FilterExec operator (predicate: colInt > 2500)
//! 3. Serializing the output back to Arrow IPC format
//!
//! The benchmark helps understand the overhead of IPC deserialization
//! and serialization relative to actual query execution, and how filter
//! performance scales with data size.
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench filter_bench -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run only the deser_only benchmark
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- deser_only
//!
//! # Change measurement time (per benchmark, default is 5 seconds)
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --measurement-time 10
//!
//! # Run specific configuration
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- "1M_rows_binary_10B"
//! ```
//!
//! ## Baseline Management
//!
//! Criterion stores benchmark results in `target/criterion/` and automatically compares
//! new runs against previous results. Each benchmark has three states:
//! - **base/**: The baseline for comparison (saved with --save-baseline)
//! - **new/**: The most recent benchmark run
//! - **change/**: Statistics about the change from base to new
//!
//! ```bash
//! # Save current results as a named baseline (e.g., "main" or "before-optimization")
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --save-baseline my-baseline
//!
//! # Compare against a specific baseline
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --baseline my-baseline
//!
//! # List all saved baselines (stored in target/criterion/<benchmark-name>/<test-name>/)
//! ls target/criterion/filter_bench/deser_only/1M_rows_binary_10B/
//!
//! # Delete all benchmark history and start fresh
//! rm -rf target/criterion
//!
//! # Run without saving results (useful for quick checks)
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --profile-time 1
//! ```
//!
//! **Typical workflow for tracking performance:**
//! 1. Before making changes: `cargo bench --bench filter_bench -- --save-baseline before`
//! 2. Make your code changes
//! 3. Compare: `cargo bench --bench filter_bench -- --baseline before`
//! 4. Criterion will show % change from the "before" baseline

// Include shared benchmark utilitiesi
#[path = "bench_utils.rs"]
mod bench_utils;

use std::hint::black_box;
use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::{ExecutionPlan, collect};
use tokio::runtime::Runtime;

use bench_utils::{
    FunctionalBatchGenerator, create_schema, deserialize_from_ipc, serialize_to_ipc
};

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Main benchmark function for filter execution.
///
/// This benchmark measures four scenarios for each binary column size:
///
/// 1. **deser_only**: Just IPC deserialization, no execution
///    - Establishes baseline deserialization cost
///    - Useful for understanding I/O vs compute ratio
fn bench_deser(c: &mut Criterion) {
    // Create a Tokio runtime for async execution
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("filter_bench");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Set measurement time (default is 5 seconds)
    // Uncomment and adjust the duration as needed:
    // group.measurement_time(std::time::Duration::from_secs(10));

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Test different binary column sizes to understand serialization overhead
    let binary_sizes = vec![2048];

    for binary_size in binary_sizes {
        let label = format!("1M_rows_binary_{binary_size}B");

        // Generate test data and serialize to IPC format
        let schema = create_schema();

        let (batches, ipc_data, ipc_size) = create_input(rows_per_batch, num_batches, total_rows, binary_size, &schema);

        // Set throughput metric for bytes/second calculations
        group.throughput(Throughput::Bytes(ipc_size as u64));

        // Benchmark 1: IPC Deserialization only
        // Measures the cost of parsing Arrow IPC format into RecordBatches
        group.bench_with_input(
            BenchmarkId::new("deser_only", &label),
            &ipc_data,
            |b, ipc_data| {
                b.iter(|| {
                    let (schema, batches) = deserialize_from_ipc(ipc_data);
                    // black_box prevents compiler from optimizing away unused results
                    black_box((schema, batches))
                })
            },
        );
    }

    group.finish();
}

fn create_input(rows_per_batch: usize, num_batches: usize, total_rows: usize, binary_size: usize, schema: &SchemaRef) -> (Vec<RecordBatch>, Vec<u8>, usize) {
    let mut generator = FunctionalBatchGenerator::new(
        Arc::clone(&schema),
        rows_per_batch,
        num_batches,
        binary_size,
    );
    let batches = generator.generate_batches();
    let ipc_data = serialize_to_ipc(&batches, &schema);
    let ipc_size = ipc_data.len();

    // Log configuration for visibility in benchmark output
    println!(
        "Config: {} rows, binary_size={} bytes, IPC size={:.2} MB",
        total_rows,
        binary_size,
        ipc_size as f64 / (1024.0 * 1024.0)
    );
    (batches, ipc_data, ipc_size)
}

criterion_group!(benches, bench_deser);
criterion_main!(benches);

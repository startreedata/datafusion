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

//! Benchmark for DataFusion HashJoinExec (inner equi-join).
//!
//! This benchmark measures the performance of hash-based inner joins with
//! varying match rates, key repetition patterns, and data sizes.
//!
//! ## Test configurations:
//!
//! - **Binary sizes**: 10B, 4096B (affects left side row size)
//! - **Match rate**: 0.5, 1.0 (fraction of left keys that match right keys)
//! - **Repeated right keys**: 1, 10, 100 (join fan-out factor)
//!
//! ## Data setup:
//!
//! - **Left side (probe)**: 1M rows (100 batches × 10K rows)
//!   - Schema: colInt, colLong, colFloat, colDouble, colString, colBinary
//!   - colInt values: 0-4999 (using `i % 5000` pattern)
//!
//! - **Right side (build)**: Variable size based on parameters
//!   - Schema: colInt (single column)
//!   - colInt values: 0 to (matchRate × 5000 - 1)
//!   - Each key repeated `repeatedRightKeys` times
//!   - This smaller side is used to build the hash table
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench hash_join_bench -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench hash_join_bench -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run specific configuration
//! cargo bench --bench hash_join_bench -p datafusion-physical-plan -- "binary_10B"
//! cargo bench --bench hash_join_bench -p datafusion-physical-plan -- "match_1.0"
//! ```

// Include shared benchmark utilities
#[path = "bench_utils.rs"]
mod bench_utils;

use std::hint::black_box;
use std::sync::Arc;

use arrow::buffer::Buffer;
use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::{ExecutionPlan, collect};

use bench_utils::{
    BatchSourceExec, FunctionalBatchGenerator, JoinBuildSideGenerator, create_schema,
    deserialize_zero_copy, serialize_results_to_ipc, serialize_to_ipc,
};

// ============================================================================
// Hash Join Plan Creation
// ============================================================================

/// Creates a HashJoinExec that performs an inner equi-join on colInt.
///
/// The join is: `probe.colInt = build.colInt`
///
/// In DataFusion's HashJoinExec:
/// - First argument (left) is the BUILD side (gets hashed into hash table)
/// - Second argument (right) is the PROBE side (scans and probes hash table)
///
/// # Arguments
/// * `probe` - Probe side execution plan (larger, 1M rows)
/// * `build` - Build side execution plan (smaller, variable rows)
///
/// # Returns
/// A HashJoinExec wrapped in Arc<dyn ExecutionPlan>
fn create_hash_join_plan(
    probe: Arc<dyn ExecutionPlan>,
    build: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let probe_schema = probe.schema();
    let build_schema = build.schema();

    // Build join condition: build.colInt = probe.colInt
    // Note: In HashJoinExec, the "on" condition is (left_col, right_col) = (build_col, probe_col)
    let build_col = Arc::new(Column::new_with_schema("colInt", &build_schema).unwrap())
        as Arc<dyn PhysicalExpr>;
    let probe_col = Arc::new(Column::new_with_schema("colInt", &probe_schema).unwrap())
        as Arc<dyn PhysicalExpr>;

    let on = vec![(build_col, probe_col)];

    Arc::new(
        HashJoinExec::try_new(
            build,                        // Left = build side (gets hashed)
            probe,                        // Right = probe side (probes hash table)
            on,
            None,                         // No additional filter
            &JoinType::Inner,             // Inner join
            None,                         // No projection
            PartitionMode::CollectLeft,   // Build hash table from left (build) side
            NullEquality::NullEqualsNothing, // NULLs don't match
            false,                        // Not null-aware
        )
        .unwrap(),
    )
}

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Benchmark configurations for different scenarios.
#[derive(Debug, Clone)]
struct BenchConfig {
    /// Binary column size for left side (affects row size)
    binary_size: usize,
    /// Match rate: fraction of left keys that match right keys
    match_rate: f64,
    /// Number of times each right key is repeated (fan-out factor)
    repeated_right_keys: usize,
}

impl BenchConfig {
    fn label(&self) -> String {
        format!(
            "binary_{}B/match_{}/repeat_{}",
            self.binary_size, self.match_rate, self.repeated_right_keys
        )
    }
}

/// Main benchmark function for hash join execution.
///
/// This benchmark measures inner equi-join performance across:
/// - Different left side row sizes (binary column 10B vs 4096B)
/// - Different match rates (0.5 vs 1.0)
/// - Different fan-out factors (repeated right keys: 1, 10, 100)
///
/// For each configuration, we measure:
/// - **join_only**: Pure HashJoinExec performance using pre-generated batches
/// - **full_pipeline**: Complete deser + join + output serialization
fn bench_hash_join(c: &mut Criterion) {
    // Create a single-threaded Tokio runtime for async execution.
    // We use current_thread to ensure all async work runs on the benchmark thread,
    // making results comparable to single-threaded Java benchmarks.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("hash_join_bench");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Probe side configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_probe_rows = rows_per_batch * num_batches;

    // Generate all benchmark configurations
    let configs: Vec<BenchConfig> = vec![
        // Binary size 10B configurations
        BenchConfig { binary_size: 10, match_rate: 0.5, repeated_right_keys: 1 },
        BenchConfig { binary_size: 10, match_rate: 0.5, repeated_right_keys: 10 },
        BenchConfig { binary_size: 10, match_rate: 0.5, repeated_right_keys: 100 },
        BenchConfig { binary_size: 10, match_rate: 1.0, repeated_right_keys: 1 },
        BenchConfig { binary_size: 10, match_rate: 1.0, repeated_right_keys: 10 },
        BenchConfig { binary_size: 10, match_rate: 1.0, repeated_right_keys: 100 },
        // Binary size 4096B configurations
        BenchConfig { binary_size: 4096, match_rate: 0.5, repeated_right_keys: 1 },
        BenchConfig { binary_size: 4096, match_rate: 0.5, repeated_right_keys: 10 },
        BenchConfig { binary_size: 4096, match_rate: 0.5, repeated_right_keys: 100 },
        BenchConfig { binary_size: 4096, match_rate: 1.0, repeated_right_keys: 1 },
        BenchConfig { binary_size: 4096, match_rate: 1.0, repeated_right_keys: 10 },
        BenchConfig { binary_size: 4096, match_rate: 1.0, repeated_right_keys: 100 },
    ];

    for config in &configs {
        let label = config.label();

        // Generate probe side data (left side - larger, 1M rows)
        let probe_schema = create_schema();
        let mut probe_generator = FunctionalBatchGenerator::new(
            Arc::clone(&probe_schema),
            rows_per_batch,
            num_batches,
            config.binary_size,
        );
        let probe_batches = probe_generator.generate_batches();

        // Generate build side data (smaller, variable rows - used for hash table)
        let build_generator =
            JoinBuildSideGenerator::new(config.match_rate, config.repeated_right_keys);
        let build_schema = build_generator.schema();
        let build_batches = build_generator.generate_batches();
        let total_build_rows = build_generator.total_rows();

        // Serialize batches to IPC format for full pipeline benchmark
        let probe_ipc_data = serialize_to_ipc(&probe_batches, &probe_schema);
        let build_ipc_data = serialize_to_ipc(&build_batches, &build_schema);
        let total_ipc_size = probe_ipc_data.len() + build_ipc_data.len();

        // Calculate approximate data sizes
        let probe_data_size: usize =
            probe_batches.iter().map(|b| b.get_array_memory_size()).sum();
        let build_data_size: usize =
            build_batches.iter().map(|b| b.get_array_memory_size()).sum();

        // Log configuration for visibility in benchmark output
        println!(
            "Config: {}, probe={} rows ({:.2} MB), build={} rows ({:.2} MB), IPC={:.2} MB",
            label,
            total_probe_rows,
            probe_data_size as f64 / (1024.0 * 1024.0),
            total_build_rows,
            build_data_size as f64 / (1024.0 * 1024.0),
            total_ipc_size as f64 / (1024.0 * 1024.0)
        );

        // Set throughput metric for bytes/second calculations (based on input size)
        group.throughput(Throughput::Bytes(total_ipc_size as u64));

        // Benchmark 1: Join execution only
        // Uses pre-generated batches directly, isolating HashJoinExec performance
        group.bench_with_input(
            BenchmarkId::new("join_only", &label),
            &(&probe_batches, &build_batches),
            |b, (probe_batches, build_batches)| {
                b.iter_batched(
                    // Setup: clone batches (NOT timed) - needed because execution consumes them
                    || ((*probe_batches).clone(), (*build_batches).clone()),
                    // Benchmark: execute join (TIMED)
                    |(probe_batches, build_batches)| {
                        rt.block_on(async {
                            let probe_source = Arc::new(BatchSourceExec::new(
                                Arc::clone(&probe_schema),
                                probe_batches,
                            )) as Arc<dyn ExecutionPlan>;
                            let build_source = Arc::new(BatchSourceExec::new(
                                Arc::clone(&build_schema),
                                build_batches,
                            )) as Arc<dyn ExecutionPlan>;
                            let plan = create_hash_join_plan(probe_source, build_source);
                            let task_ctx = Arc::new(TaskContext::default());
                            let results = collect(plan, task_ctx).await.unwrap();
                            black_box(results)
                        })
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Convert to Buffer for zero-copy deserialization
        let probe_buffer = Buffer::from_vec(probe_ipc_data.clone());
        let build_buffer = Buffer::from_vec(build_ipc_data.clone());

        // Benchmark 2: Full pipeline (deser + join + output serialization)
        // Measures complete round-trip: IPC in -> join -> IPC out
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", &label),
            &(&probe_buffer, &build_buffer),
            |b, (probe_buffer, build_buffer)| {
                b.iter(|| {
                    rt.block_on(async {
                        let (probe_schema, probe_batches) = deserialize_zero_copy(probe_buffer);
                        let (build_schema, build_batches) = deserialize_zero_copy(build_buffer);

                        let probe_source = Arc::new(BatchSourceExec::new(
                            Arc::clone(&probe_schema),
                            probe_batches,
                        )) as Arc<dyn ExecutionPlan>;
                        let build_source = Arc::new(BatchSourceExec::new(
                            Arc::clone(&build_schema),
                            build_batches,
                        )) as Arc<dyn ExecutionPlan>;

                        let plan = create_hash_join_plan(probe_source, build_source);
                        let task_ctx = Arc::new(TaskContext::default());
                        let results = collect(plan, task_ctx).await.unwrap();

                        // Serialize results back to IPC format
                        let output_ipc = serialize_results_to_ipc(&results);
                        black_box(output_ipc)
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_hash_join);
criterion_main!(benches);

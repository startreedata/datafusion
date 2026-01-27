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

//! Benchmark for DataFusion AggregateExec (COUNT(*) GROUP BY).
//!
//! This benchmark measures the performance of hash-based aggregation with
//! varying group cardinalities and column types.
//!
//! ## Test configurations:
//!
//! - **Column types**: Int32, Binary (10B), Binary (1024B)
//! - **Distinct counts**: 4096, 16384, 65536
//! - **Data size**: 1M rows total (100 batches × 10K rows)
//!
//! The benchmark helps understand:
//! - How grouping performance scales with cardinality
//! - Impact of key type (fixed-width int vs variable-length binary)
//! - Impact of key size (10B vs 1024B binary)
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench count_group_by_bench -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench count_group_by_bench -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run specific configuration (e.g., only int column benchmarks)
//! cargo bench --bench count_group_by_bench -p datafusion-physical-plan -- "int_col"
//!
//! # Run specific cardinality
//! cargo bench --bench count_group_by_bench -p datafusion-physical-plan -- "distinct_4096"
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
use datafusion_execution::TaskContext;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion_physical_plan::{ExecutionPlan, collect};

use bench_utils::{
    BatchSourceExec, SingleColumnBatchGenerator, SingleColumnType, deserialize_zero_copy,
    serialize_results_to_ipc, serialize_to_ipc,
};

// ============================================================================
// Aggregate Plan Creation
// ============================================================================

/// Creates an AggregateExec that performs COUNT(*) GROUP BY groupCol.
///
/// This simulates a common aggregation pattern where we count the number
/// of rows for each distinct value in the grouping column.
///
/// # Arguments
/// * `input` - The input execution plan (typically BatchSourceExec)
///
/// # Returns
/// An AggregateExec wrapped in Arc<dyn ExecutionPlan>
fn create_count_groupby_plan(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let schema = input.schema();

    // Build GROUP BY expression: GROUP BY groupCol
    let group_col =
        Arc::new(Column::new_with_schema("groupCol", &schema).unwrap()) as Arc<dyn PhysicalExpr>;
    let group_expr = vec![(Arc::clone(&group_col), "groupCol".to_string())];
    let group_by = PhysicalGroupBy::new_single(group_expr);

    // Build aggregate expression: COUNT(groupCol)
    // We use groupCol as the argument since it's non-null; effectively COUNT(*)
    let aggr_expr = vec![Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![group_col])
            .schema(Arc::clone(&schema))
            .alias("count")
            .build()
            .unwrap(),
    )];

    // Create the aggregate execution plan
    // Using Single mode (not partial/final) for simplicity in benchmarks
    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            aggr_expr,
            vec![None], // No filter expressions
            input,
            schema,
        )
        .unwrap(),
    )
}

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Benchmark configurations for different column types.
#[derive(Debug, Clone)]
struct BenchConfig {
    /// Human-readable name for the configuration
    name: &'static str,
    /// Column type to generate
    column_type: SingleColumnType,
}

/// Main benchmark function for COUNT(*) GROUP BY execution.
///
/// This benchmark measures COUNT(*) GROUP BY performance across:
/// - Different column types (Int32, Binary 10B, Binary 1024B)
/// - Different cardinalities (4096, 16384, 65536 distinct values)
///
/// For each configuration, we measure:
/// - **agg_only**: Pure aggregation execution using pre-generated batches
///   This isolates the AggregateExec performance from serialization overhead
/// - **full_pipeline**: Complete deser + aggregation + output serialization
///   Real-world end-to-end latency including IPC serde
fn bench_count_group_by(c: &mut Criterion) {
    // Create a single-threaded Tokio runtime for async execution.
    // We use current_thread to ensure all async work runs on the benchmark thread,
    // making results comparable to single-threaded Java benchmarks.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("count_group_by_bench");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Column type configurations
    let configs = vec![
        BenchConfig {
            name: "int_col",
            column_type: SingleColumnType::Int,
        },
        BenchConfig {
            name: "binary_10B",
            column_type: SingleColumnType::Binary { binary_size: 10 },
        },
        BenchConfig {
            name: "binary_1024B",
            column_type: SingleColumnType::Binary { binary_size: 1024 },
        },
    ];

    // Cardinality configurations (number of distinct values)
    let distinct_counts = vec![4096, 16384, 65536];

    for config in &configs {
        for &distinct_count in &distinct_counts {
            let label = format!("{}/distinct_{}", config.name, distinct_count);

            // Generate test data
            let mut generator = SingleColumnBatchGenerator::new(
                config.column_type,
                rows_per_batch,
                num_batches,
                distinct_count,
            );
            let schema = generator.schema();
            let batches = generator.generate_batches();

            // Serialize batches to IPC format for full pipeline benchmark
            let ipc_data = serialize_to_ipc(&batches, &schema);
            let ipc_size = ipc_data.len();

            // Calculate approximate data size for throughput metric
            let data_size: usize = batches
                .iter()
                .map(|b| b.get_array_memory_size())
                .sum();

            // Log configuration for visibility in benchmark output
            println!(
                "Config: {} rows, {}, distinct={}, data size={:.2} MB, IPC size={:.2} MB",
                total_rows,
                config.name,
                distinct_count,
                data_size as f64 / (1024.0 * 1024.0),
                ipc_size as f64 / (1024.0 * 1024.0)
            );

            // Set throughput metric for bytes/second calculations
            group.throughput(Throughput::Bytes(ipc_size as u64));

            // Benchmark 1: Aggregation execution only
            // Uses pre-generated batches directly, isolating AggregateExec performance
            group.bench_with_input(
                BenchmarkId::new("agg_only", &label),
                &batches,
                |b, batches| {
                    b.iter_batched(
                        // Setup: clone batches (NOT timed) - needed because execution consumes them
                        || batches.clone(),
                        // Benchmark: execute aggregation (TIMED)
                        |batches| {
                            rt.block_on(async {
                                let source = Arc::new(BatchSourceExec::new(
                                    Arc::clone(&schema),
                                    batches,
                                )) as Arc<dyn ExecutionPlan>;
                                let plan = create_count_groupby_plan(source);
                                let task_ctx = Arc::new(TaskContext::default());
                                let results = collect(plan, task_ctx).await.unwrap();
                                black_box(results)
                            })
                        },
                        BatchSize::SmallInput,
                    )
                },
            );

            let data_buffer = Buffer::from_vec(ipc_data);

            // Benchmark 2: Full pipeline (deser + aggregation + output serialization)
            // Measures complete round-trip: IPC in -> aggregate -> IPC out
            // Relevant for scenarios where results are sent over network or stored
            group.bench_with_input(
                BenchmarkId::new("full_pipeline", &label),
                &data_buffer,
                |b, data_buffer| {
                    b.iter(|| {
                        rt.block_on(async {
                            let (schema, batches) = deserialize_zero_copy(data_buffer);
                            let source = Arc::new(BatchSourceExec::new(
                                Arc::clone(&schema),
                                batches,
                            )) as Arc<dyn ExecutionPlan>;
                            let plan = create_count_groupby_plan(source);
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
    }

    group.finish();
}

criterion_group!(benches, bench_count_group_by);
criterion_main!(benches);

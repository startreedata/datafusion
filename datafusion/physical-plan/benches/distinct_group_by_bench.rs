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

//! Benchmark for DataFusion AggregateExec (COUNT(DISTINCT bytes) GROUP BY colInt).
//!
//! This benchmark measures the performance of hash-based aggregation with
//! distinct counting over binary columns, varying binary sizes and group cardinalities.
//!
//! ## Test configurations:
//!
//! - **Binary sizes**: 10B, 1000B
//! - **Distinct keys (group cardinalities)**: 4096, 16384, 65536
//! - **Data size**: 1M rows total (100 batches × 10K rows)
//!
//! The benchmark helps understand:
//! - How COUNT(DISTINCT) performance scales with group cardinality
//! - Impact of binary value size on distinct counting
//! - Performance characteristics of two-column aggregation (GROUP BY colInt, COUNT(DISTINCT bytes))
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench distinct_group_by_bench -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench distinct_group_by_bench -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run specific configuration
//! cargo bench --bench distinct_group_by_bench -p datafusion-physical-plan -- "INT_BYTES_10"
//!
//! # Run specific cardinality
//! cargo bench --bench distinct_group_by_bench -p datafusion-physical-plan -- "distinct_4096"
//! ```

// Include shared benchmark utilities
#[path = "bench_utils.rs"]
mod bench_utils;

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Int32Array, RecordBatch};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
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

use bench_utils::{BatchSourceExec, deserialize_zero_copy, serialize_results_to_ipc, serialize_to_ipc};

// ============================================================================
// Two-Column Data Generation (Int + Binary)
// ============================================================================

/// Generates two-column record batches for COUNT(DISTINCT bytes) GROUP BY colInt benchmarks.
///
/// This generator creates data following the Java IntBytesGenerator logic:
/// - colInt: i % numGroups
/// - bytes: ByteBuffer of bytesLength with int value (i % numGroups) at position 0,
///   followed by zeros to fill remaining bytes
///
/// This ensures that:
/// - There are exactly `numGroups` distinct values for colInt (0 to numGroups-1)
/// - Each group (colInt value) has exactly 1 distinct bytes value
/// - The bytes values vary in size (10 or 1000 bytes) to test size impact
pub struct TwoColumnBatchGenerator {
    /// Schema for generated batches (colInt: Int32, bytes: Binary)
    schema: SchemaRef,
    /// Number of rows in each batch
    rows_per_batch: usize,
    /// Total number of batches to generate
    num_batches: usize,
    /// Number of groups (modulo value for colInt)
    num_groups: usize,
    /// Size of binary values in bytes
    bytes_length: usize,
}

impl TwoColumnBatchGenerator {
    /// Creates a new two-column batch generator.
    ///
    /// # Arguments
    /// * `rows_per_batch` - Number of rows per batch
    /// * `num_batches` - Total number of batches to generate
    /// * `num_groups` - Number of distinct groups (colInt values)
    /// * `bytes_length` - Length of binary values (10 or 1000)
    ///
    /// # Returns
    /// A new generator configured for the specified parameters.
    pub fn new(
        rows_per_batch: usize,
        num_batches: usize,
        num_groups: usize,
        bytes_length: usize,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("colInt", DataType::Int32, false),
            Field::new("bytes", DataType::Binary, false),
        ]));

        Self {
            schema,
            rows_per_batch,
            num_batches,
            num_groups,
            bytes_length,
        }
    }

    /// Returns the schema of the generated batches.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Generates a single record batch for the given batch index.
    ///
    /// Follows Java's IntBytesGenerator logic:
    /// - colInt[i] = (batch_start + i) % numGroups
    /// - bytes[i] = ByteBuffer with int value ((batch_start + i) % numGroups) + zero padding
    fn generate_batch(&self, batch_index: usize) -> RecordBatch {
        let start_row = batch_index * self.rows_per_batch;
        let num_rows = self.rows_per_batch;

        // Generate colInt values: i % numGroups
        let col_int_values: Vec<i32> = (0..num_rows)
            .map(|i| ((start_row + i) % self.num_groups) as i32)
            .collect();
        let col_int_array = Arc::new(Int32Array::from(col_int_values)) as ArrayRef;

        // Generate bytes values: ByteBuffer with int at start + zero padding
        // Follows Java's: ByteBuffer.allocate(bytesLength).putInt(i % numGroups)
        let bytes_values: Vec<Vec<u8>> = (0..num_rows)
            .map(|i| {
                let group_id = ((start_row + i) % self.num_groups) as i32;
                let mut buffer = vec![0u8; self.bytes_length];
                // Write the int value at the start (little-endian, matching Java's ByteBuffer)
                buffer[0..4].copy_from_slice(&group_id.to_le_bytes());
                buffer
            })
            .collect();
        let bytes_refs: Vec<&[u8]> = bytes_values.iter().map(|v| v.as_slice()).collect();
        let bytes_array = Arc::new(BinaryArray::from(bytes_refs)) as ArrayRef;

        RecordBatch::try_new(Arc::clone(&self.schema), vec![col_int_array, bytes_array])
            .expect("Failed to create record batch")
    }

    /// Generates all batches.
    ///
    /// Returns a vector of `num_batches` record batches, each containing
    /// `rows_per_batch` rows.
    pub fn generate_batches(&self) -> Vec<RecordBatch> {
        (0..self.num_batches)
            .map(|i| self.generate_batch(i))
            .collect()
    }
}

// ============================================================================
// Aggregate Plan Creation
// ============================================================================

/// Creates an AggregateExec that performs COUNT(DISTINCT bytes) GROUP BY colInt.
///
/// This simulates a common aggregation pattern where we count the number of
/// distinct binary values for each integer group value.
///
/// # Arguments
/// * `input` - The input execution plan (typically BatchSourceExec)
///
/// # Returns
/// An AggregateExec wrapped in Arc<dyn ExecutionPlan>
fn create_distinct_count_groupby_plan(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let schema = input.schema();

    // Build GROUP BY expression: GROUP BY colInt
    let group_col =
        Arc::new(Column::new_with_schema("colInt", &schema).unwrap()) as Arc<dyn PhysicalExpr>;
    let group_expr = vec![(Arc::clone(&group_col), "colInt".to_string())];
    let group_by = PhysicalGroupBy::new_single(group_expr);

    // Build aggregate expression: COUNT(DISTINCT bytes)
    let bytes_col =
        Arc::new(Column::new_with_schema("bytes", &schema).unwrap()) as Arc<dyn PhysicalExpr>;
    let aggr_expr = vec![Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![bytes_col])
            .schema(Arc::clone(&schema))
            .alias("count_distinct")
            .distinct()  // Enable DISTINCT counting
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

/// Benchmark configurations for different binary sizes.
#[derive(Debug, Clone, Copy)]
struct BenchConfig {
    /// Human-readable name for the configuration
    name: &'static str,
    /// Size of binary values in bytes
    bytes_length: usize,
}

/// Main benchmark function for COUNT(DISTINCT bytes) GROUP BY colInt execution.
///
/// This benchmark measures COUNT(DISTINCT) GROUP BY performance across:
/// - Different binary sizes (10B, 1000B)
/// - Different cardinalities (4096, 16384, 65536 distinct groups)
///
/// For each configuration, we measure:
/// - **agg_only**: Pure aggregation execution using pre-generated batches
///   This isolates the AggregateExec performance from serialization overhead
/// - **full_pipeline**: Complete deser + aggregation + output serialization
///   Real-world end-to-end latency including IPC serde
fn bench_distinct_group_by(c: &mut Criterion) {
    // Create a single-threaded Tokio runtime for async execution.
    // We use current_thread to ensure all async work runs on the benchmark thread,
    // making results comparable to single-threaded Java benchmarks.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("distinct_group_by_bench");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Binary size configurations
    let configs = vec![
        BenchConfig {
            name: "INT_BYTES_10",
            bytes_length: 10,
        },
        BenchConfig {
            name: "INT_BYTES_1000",
            bytes_length: 1000,
        },
    ];

    // Distinct keys configurations (number of groups)
    let distinct_keys = vec![4096, 16384, 65536];

    for config in &configs {
        for &num_groups in &distinct_keys {
            let label = format!("{}/distinct_{}", config.name, num_groups);

            // Generate test data
            let generator = TwoColumnBatchGenerator::new(
                rows_per_batch,
                num_batches,
                num_groups,
                config.bytes_length,
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
                "Config: {} rows, {}, distinct_keys={}, data size={:.2} MB, IPC size={:.2} MB",
                total_rows,
                config.name,
                num_groups,
                data_size as f64 / (1024.0 * 1024.0),
                ipc_size as f64 / (1024.0 * 1024.0)
            );

            // Set throughput metric for bytes/second calculations
            group.throughput(Throughput::Bytes(ipc_size as u64));

            // Validation (NOT timed - run once before benchmarking)
            // Verify the number of groups matches expected distinct_keys
            {
                let validation_batches = batches.clone();
                let validation_result = rt.block_on(async {
                    let source = Arc::new(BatchSourceExec::new(
                        Arc::clone(&schema),
                        validation_batches,
                    )) as Arc<dyn ExecutionPlan>;
                    let plan = create_distinct_count_groupby_plan(source);
                    let task_ctx = Arc::new(TaskContext::default());
                    collect(plan, task_ctx).await.unwrap()
                });
                let total_result_rows: usize = validation_result.iter()
                    .map(|batch| batch.num_rows())
                    .sum();
                assert_eq!(
                    total_result_rows,
                    num_groups,
                    "Expected {} distinct groups, got {}",
                    num_groups,
                    total_result_rows
                );
            }

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
                                let plan = create_distinct_count_groupby_plan(source);
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
                            let plan = create_distinct_count_groupby_plan(source);
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

criterion_group!(benches, bench_distinct_group_by);
criterion_main!(benches);


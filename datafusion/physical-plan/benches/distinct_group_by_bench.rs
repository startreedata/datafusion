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

use std::fs::File;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{RecordBatch, ArrayRef, BinaryArray, BinaryViewArray};
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::FileReader;
use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use datafusion_execution::TaskContext;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion_physical_plan::{ExecutionPlan, collect};

use bench_utils::{BatchSourceExec, serialize_results_to_ipc};

// ============================================================================
// Aggregate Plan Creation
// ============================================================================

/// Creates an AggregateExec that performs COUNT(DISTINCT bytes) GROUP BY colInt.
///
/// This benchmark doesn't generate data. Instead, we have to run the equivalent JMH benchmark in
/// Apache Pinot and then copy the generated Arrow IPC files into the `benches/` folder, keeping
/// the name conventions used in the JMH benchmark.
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
        Arc::new(Column::new_with_schema("colint", &schema).unwrap()) as Arc<dyn PhysicalExpr>;
    let group_expr = vec![(Arc::clone(&group_col), "colint".to_string())];
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
    /// Size of binary values in bytes (not used, but kept for compatibility)
    #[allow(dead_code)]
    bytes_length: usize,
}

/// Enum to select which binary array type to use
#[derive(Debug, Clone, Copy)]
enum BinaryType {
    Binary,
    BinaryView,
}

fn get_arrow_file_path(
    folder: &str,
    config_name: &str,
    num_groups: usize,
    distinct_values_per_group: usize,
) -> PathBuf {
    let file_name = format!(
        "group_distinct_by_{}_groups_{}_distinctPerGroup_{}.arrow",
        config_name,
        num_groups,
        distinct_values_per_group
    );
    PathBuf::from(folder).join(file_name)
}

fn load_batches_from_arrow_file(path: &PathBuf, binary_type: BinaryType) -> (SchemaRef, Vec<RecordBatch>) {
    let file = File::open(path).unwrap_or_else(|_| panic!("Arrow file not found: {}", path.display()));
    let mut reader = FileReader::try_new(file, None).expect("Failed to open Arrow IPC file");
    let orig_schema = reader.schema();
    let orig_batches = reader.collect::<arrow::error::Result<Vec<_>>>().expect("Failed to read batches from Arrow file");

    match binary_type {
        BinaryType::Binary => (orig_schema, orig_batches),
        BinaryType::BinaryView => {
            // Find the index of the "bytes" column
            let bytes_idx = orig_schema.fields().iter().position(|f| f.name() == "bytes").expect("No 'bytes' column");
            // Create new schema with bytes as BinaryView
            let mut new_fields: Vec<Arc<arrow::datatypes::Field>> = orig_schema.fields().to_vec();
            new_fields[bytes_idx] = Arc::new(arrow::datatypes::Field::new("bytes", arrow::datatypes::DataType::BinaryView, false));
            let new_schema = Arc::new(arrow::datatypes::Schema::new(
                new_fields.iter().map(|f| f.as_ref().clone()).collect::<Vec<arrow::datatypes::Field>>()
            ));
            // Convert each batch
            let new_batches = orig_batches.into_iter().map(|batch| {
                let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
                let binary_array = batch.column(bytes_idx).as_any().downcast_ref::<BinaryArray>().expect("'bytes' column is not BinaryArray");
                let binaryview_vec: Vec<&[u8]> = (0..batch.num_rows()).map(|i| binary_array.value(i)).collect();
                let binaryview_array = BinaryViewArray::from(binaryview_vec);
                columns[bytes_idx] = Arc::new(binaryview_array);
                RecordBatch::try_new(Arc::clone(&new_schema), columns).expect("Failed to create BinaryView batch")
            }).collect();
            (new_schema, new_batches)
        }
    }
}

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

    // Distinct values per group (JMH param)
    let distinct_values_per_group_list = vec![1, 4, 16, 64, 256, 1024];

    let binary_types = vec![BinaryType::Binary, BinaryType::BinaryView];

    let arrow_folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benches");

    for config in &configs {
        for &binary_type in &binary_types {
            let binary_type_label = match binary_type {
                BinaryType::Binary => "Binary",
                BinaryType::BinaryView => "BinaryView",
            };
            for &distinct_values_per_group in &distinct_values_per_group_list {
                let num_groups = 512;
                let num_batches = 1;
                let rows_per_batch = num_groups * distinct_values_per_group;
                let total_rows = rows_per_batch * num_batches;

                let label = format!("{}/{}/dvg_{}", config.name, binary_type_label, distinct_values_per_group);

                // Load test data from Arrow file
                let arrow_file_path = get_arrow_file_path(
                    arrow_folder.to_str().unwrap(),
                    config.name, // Use config.name for the file name
                    num_groups,
                    distinct_values_per_group,
                );
                println!("Reading Arrow file: {}", arrow_file_path.display());
                let (schema, batches) = load_batches_from_arrow_file(&arrow_file_path, binary_type);

                // Calculate approximate data size for throughput metric
                let data_size: usize = batches
                    .iter()
                    .map(|b| b.get_array_memory_size())
                    .sum();

                // Log configuration for visibility in benchmark output
                println!(
                    "Config: {} rows, {}, {}, dvg={}, data size={:.2} MB, Arrow file: {}",
                    total_rows,
                    config.name,
                    binary_type_label,
                    distinct_values_per_group,
                    data_size as f64 / (1024.0 * 1024.0),
                    arrow_file_path.display()
                );

                // Set throughput metric for bytes/second calculations
                group.throughput(Throughput::Bytes(data_size as u64));

                // Validation (NOT timed - run once before benchmarking)
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

                // Benchmark 2: Full pipeline (deser + aggregation + output serialization)
                // Measures complete round-trip: IPC in -> aggregate -> IPC out
                // Relevant for scenarios where results are sent over network or stored
                group.bench_with_input(
                    BenchmarkId::new("full_pipeline", &label),
                    &batches,
                    |b, batches| {
                        b.iter(|| {
                            rt.block_on(async {
                                let source = Arc::new(BatchSourceExec::new(
                                    Arc::clone(&schema),
                                    batches.clone(),
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
    }
    group.finish();
}

criterion_group!(benches, bench_distinct_group_by);
criterion_main!(benches);


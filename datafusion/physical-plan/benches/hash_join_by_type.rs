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

//! Benchmark for DataFusion HashJoinExec comparing different column types.
//!
//! This benchmark measures the performance of hash-based inner joins across
//! different column types (Int32, Utf8, Utf8View, Dictionary-encoded strings).
//!
//! ## Test configurations:
//!
//! - **Column types**: Int32, Utf8, Utf8View, Dictionary(Int16, Utf8), Dictionary(Int16, Utf8View)
//! - **Match rate**: 0.5, 1.0 (fraction of left keys that match right keys)
//! - **Repeated right keys**: 1, 10 (join fan-out factor)
//!
//! ## Data setup:
//!
//! - **Left side (probe)**: 10K rows (1 batch × 10K rows)
//!   - Schema: colInt, colLong, colFloat, colDouble, colString, colBinary
//!   - Binary column: 10 bytes (small rows)
//!   - colInt values: 0-4999 (using `i % 5000` pattern)
//!   - colString values: "str_0000" to "str_4999" (9 bytes each, 5000 distinct values)
//!
//! - **Right side (build)**: Variable size based on parameters
//!   - Schema: Single column (colInt or colString depending on join type)
//!   - Values: 0 to (matchRate × 5000 - 1) for Int, "str_0000" to "str_{matchRate × 5000 - 1}" for String
//!   - Each key repeated `repeatedRightKeys` times
//!   - This smaller side is used to build the hash table
//!
//! ## Column type details:
//!
//! - **Int**: Standard Int32 join on colInt
//! - **String**: Utf8 join on colString (9-byte strings)
//! - **StringView**: Utf8View join on colString (optimized for strings ≤12 bytes)
//! - **DictionaryString**: Dictionary(Int16, Utf8) join on colString
//! - **DictionaryStringView**: Dictionary(Int16, Utf8View) join on colString
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all configurations
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan -- --sample-size 10
//!
//! # Run specific column type
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan -- "joinInt"
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan -- "joinStr"
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan -- "joinStrView"
//!
//! # Run specific configuration
//! cargo bench --bench hash_join_by_type -p datafusion-physical-plan -- "match_1.0"
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
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::{ExecutionPlan, collect};

use bench_utils::{
    BatchSourceExec, FunctionalBatchGenerator, JoinBuildSideGenerator, JoinColumnType,
    StringColumnType, create_schema_with_string_type, deserialize_zero_copy,
    serialize_results_to_ipc, serialize_to_ipc,
};

// ============================================================================
// Hash Join Plan Creation
// ============================================================================

/// Creates a HashJoinExec that performs an inner equi-join.
///
/// In DataFusion's HashJoinExec:
/// - First argument (left) is the BUILD side (gets hashed into hash table)
/// - Second argument (right) is the PROBE side (scans and probes hash table)
///
/// # Arguments
/// * `probe` - Probe side execution plan (larger, 10K rows)
/// * `build` - Build side execution plan (smaller, variable rows)
/// * `join_col` - Name of the column to join on ("colInt" or "colString")
///
/// # Returns
/// A HashJoinExec wrapped in Arc<dyn ExecutionPlan>
fn create_hash_join_plan(
    probe: Arc<dyn ExecutionPlan>,
    build: Arc<dyn ExecutionPlan>,
    join_col: &str,
) -> Arc<dyn ExecutionPlan> {
    let probe_schema = probe.schema();
    let build_schema = build.schema();

    // Build join condition: build.{join_col} = probe.{join_col}
    // Note: In HashJoinExec, the "on" condition is (left_col, right_col) = (build_col, probe_col)
    let build_col = Arc::new(Column::new_with_schema(join_col, &build_schema).unwrap())
        as Arc<dyn PhysicalExpr>;
    let probe_col = Arc::new(Column::new_with_schema(join_col, &probe_schema).unwrap())
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
    /// Match rate: fraction of left keys that match right keys
    match_rate: f64,
    /// Number of times each right key is repeated (fan-out factor)
    repeated_right_keys: usize,
    /// Type of column to join on
    join_column_type: JoinColumnType,
}

impl BenchConfig {
    fn label(&self) -> String {
        format!(
            "match_{}/repeat_{}",
            self.match_rate, self.repeated_right_keys
        )
    }

    fn benchmark_name(&self) -> &'static str {
        match self.join_column_type {
            JoinColumnType::Int => "joinInt",
            JoinColumnType::String => "joinStr",
            JoinColumnType::StringView => "joinStrView",
            JoinColumnType::DictionaryString => "joinDictStr",
            JoinColumnType::DictionaryStringView => "joinDictStrView",
        }
    }

    fn join_column_name(&self) -> &'static str {
        match self.join_column_type {
            JoinColumnType::Int => "colInt",
            JoinColumnType::String
            | JoinColumnType::StringView
            | JoinColumnType::DictionaryString
            | JoinColumnType::DictionaryStringView => "colString",
        }
    }

    fn string_column_type(&self) -> StringColumnType {
        match self.join_column_type {
            JoinColumnType::Int => StringColumnType::Utf8, // Not used for Int joins
            JoinColumnType::String => StringColumnType::Utf8,
            JoinColumnType::StringView => StringColumnType::Utf8View,
            JoinColumnType::DictionaryString => StringColumnType::DictionaryUtf8,
            JoinColumnType::DictionaryStringView => StringColumnType::DictionaryUtf8View,
        }
    }
}

/// Main benchmark function for hash join execution across different column types.
///
/// This benchmark measures inner equi-join performance across:
/// - Different column types (Int32, Utf8, Utf8View, Dictionary variants)
/// - Different match rates (0.5 vs 1.0)
/// - Different fan-out factors (repeated right keys: 1, 10)
///
/// For each configuration, we measure the complete pipeline:
/// deser + join + output serialization
fn bench_hash_join_by_type(c: &mut Criterion) {
    // Create a single-threaded Tokio runtime for async execution.
    // We use current_thread to ensure all async work runs on the benchmark thread,
    // making results comparable to single-threaded Java benchmarks.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("hash_join_by_type");

    // Use flat sampling to collect exactly the requested samples without time constraints
    group.sampling_mode(SamplingMode::Flat);

    // Probe side configuration: 10K rows total (10K rows × 1 batch)
    let rows_per_batch = 10_000;
    let num_batches = 1;
    let total_probe_rows = rows_per_batch * num_batches;
    let binary_size = 10; // Small binary column (10 bytes)

    // Generate all benchmark configurations
    let mut configs: Vec<BenchConfig> = Vec::new();

    // For each base configuration (match_rate × repeated_keys)
    for match_rate in &[0.5, 1.0] {
        for repeated_right_keys in &[1, 10] {
            // Create a config for each column type
            for column_type in &[
                JoinColumnType::Int,
                JoinColumnType::String,
                JoinColumnType::StringView,
                JoinColumnType::DictionaryString,
                JoinColumnType::DictionaryStringView,
            ] {
                configs.push(BenchConfig {
                    match_rate: *match_rate,
                    repeated_right_keys: *repeated_right_keys,
                    join_column_type: *column_type,
                });
            }
        }
    }

    for config in &configs {
        let label = config.label();
        let benchmark_name = config.benchmark_name();
        let join_column_name = config.join_column_name();

        // Generate probe side data
        let probe_schema = create_schema_with_string_type(config.string_column_type());
        let mut probe_generator = FunctionalBatchGenerator::new_with_string_type(
            Arc::clone(&probe_schema),
            rows_per_batch,
            num_batches,
            binary_size,
            config.string_column_type(),
        );
        let probe_batches = probe_generator.generate_batches();

        // Generate build side data
        let build_generator = JoinBuildSideGenerator::new_with_column_type(
            config.match_rate,
            config.repeated_right_keys,
            config.join_column_type,
        );
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
            "Config: {} ({}), probe={} rows ({:.2} MB), build={} rows ({:.2} MB), IPC={:.2} MB",
            benchmark_name,
            label,
            total_probe_rows,
            probe_data_size as f64 / (1024.0 * 1024.0),
            total_build_rows,
            build_data_size as f64 / (1024.0 * 1024.0),
            total_ipc_size as f64 / (1024.0 * 1024.0)
        );

        // Set throughput metric for bytes/second calculations (based on input size)
        group.throughput(Throughput::Bytes(total_ipc_size as u64));

        // Convert to Buffer for zero-copy deserialization
        let probe_buffer = Buffer::from_vec(probe_ipc_data.clone());
        let build_buffer = Buffer::from_vec(build_ipc_data.clone());

        // Benchmark: Full pipeline (deser + join + output serialization)
        // Measures complete round-trip: IPC in -> join -> IPC out
        group.bench_with_input(
            BenchmarkId::new(benchmark_name, &label),
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

                        let plan = create_hash_join_plan(probe_source, build_source, join_column_name);
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

criterion_group!(benches, bench_hash_join_by_type);
criterion_main!(benches);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Calculates the expected number of rows from a hash join.
    ///
    /// For an inner join:
    /// - Probe side: 10K rows with 5000 distinct keys (each key appears 2 times)
    /// - Build side: (match_rate * 5000) distinct keys, each repeated `repeated_keys` times
    ///
    /// Expected output rows:
    /// - For each matching key, output = probe_occurrences * build_occurrences
    /// - Total = matching_keys * 2 * repeated_keys
    fn expected_row_count(match_rate: f64, repeated_keys: usize) -> usize {
        let total_probe_rows = 10_000;
        let probe_distinct_keys = 5_000;
        let probe_key_occurrences = total_probe_rows / probe_distinct_keys; // = 2

        let build_distinct_keys = (match_rate * 5000.0) as usize;

        // Each matching key produces: probe_occurrences * build_occurrences rows
        build_distinct_keys * probe_key_occurrences * repeated_keys
    }

    /// Test helper to execute a join and verify row count
    async fn test_join_row_count(
        match_rate: f64,
        repeated_keys: usize,
        column_type: JoinColumnType,
    ) {
        let rows_per_batch = 10_000;
        let num_batches = 1;
        let binary_size = 10;

        let string_column_type = match column_type {
            JoinColumnType::Int => StringColumnType::Utf8,
            JoinColumnType::String => StringColumnType::Utf8,
            JoinColumnType::StringView => StringColumnType::Utf8View,
            JoinColumnType::DictionaryString => StringColumnType::DictionaryUtf8,
            JoinColumnType::DictionaryStringView => StringColumnType::DictionaryUtf8View,
        };

        let join_column_name = match column_type {
            JoinColumnType::Int => "colInt",
            _ => "colString",
        };

        // Generate probe side data
        let probe_schema = create_schema_with_string_type(string_column_type);
        let mut probe_generator = FunctionalBatchGenerator::new_with_string_type(
            Arc::clone(&probe_schema),
            rows_per_batch,
            num_batches,
            binary_size,
            string_column_type,
        );
        let probe_batches = probe_generator.generate_batches();

        // Generate build side data
        let build_generator = JoinBuildSideGenerator::new_with_column_type(
            match_rate,
            repeated_keys,
            column_type,
        );
        let build_schema = build_generator.schema();
        let build_batches = build_generator.generate_batches();

        // Execute join
        let probe_source = Arc::new(BatchSourceExec::new(
            Arc::clone(&probe_schema),
            probe_batches,
        )) as Arc<dyn ExecutionPlan>;
        let build_source = Arc::new(BatchSourceExec::new(
            Arc::clone(&build_schema),
            build_batches,
        )) as Arc<dyn ExecutionPlan>;

        let plan = create_hash_join_plan(probe_source, build_source, join_column_name);
        let task_ctx = Arc::new(TaskContext::default());
        let results = collect(plan, task_ctx).await.unwrap();

        // Calculate actual row count
        let actual_row_count: usize = results.iter().map(|batch| batch.num_rows()).sum();
        let expected = expected_row_count(match_rate, repeated_keys);

        assert_eq!(
            actual_row_count, expected,
            "Row count mismatch for match_rate={}, repeated_keys={}, column_type={:?}. Expected {}, got {}",
            match_rate, repeated_keys, column_type, expected, actual_row_count
        );
    }

    #[tokio::test]
    async fn test_join_row_counts_int() {
        // Test Int column type with different configurations
        test_join_row_count(0.5, 1, JoinColumnType::Int).await;
        test_join_row_count(0.5, 10, JoinColumnType::Int).await;
        test_join_row_count(1.0, 1, JoinColumnType::Int).await;
        test_join_row_count(1.0, 10, JoinColumnType::Int).await;
    }

    #[tokio::test]
    async fn test_join_row_counts_string() {
        // Test String column type with different configurations
        test_join_row_count(0.5, 1, JoinColumnType::String).await;
        test_join_row_count(0.5, 10, JoinColumnType::String).await;
        test_join_row_count(1.0, 1, JoinColumnType::String).await;
        test_join_row_count(1.0, 10, JoinColumnType::String).await;
    }

    #[tokio::test]
    async fn test_join_row_counts_string_view() {
        // Test StringView column type with different configurations
        test_join_row_count(0.5, 1, JoinColumnType::StringView).await;
        test_join_row_count(0.5, 10, JoinColumnType::StringView).await;
        test_join_row_count(1.0, 1, JoinColumnType::StringView).await;
        test_join_row_count(1.0, 10, JoinColumnType::StringView).await;
    }

    #[tokio::test]
    async fn test_join_row_counts_dictionary_string() {
        // Test DictionaryString column type with different configurations
        test_join_row_count(0.5, 1, JoinColumnType::DictionaryString).await;
        test_join_row_count(0.5, 10, JoinColumnType::DictionaryString).await;
        test_join_row_count(1.0, 1, JoinColumnType::DictionaryString).await;
        test_join_row_count(1.0, 10, JoinColumnType::DictionaryString).await;
    }

    #[tokio::test]
    async fn test_join_row_counts_dictionary_string_view() {
        // Test DictionaryStringView column type with different configurations
        test_join_row_count(0.5, 1, JoinColumnType::DictionaryStringView).await;
        test_join_row_count(0.5, 10, JoinColumnType::DictionaryStringView).await;
        test_join_row_count(1.0, 1, JoinColumnType::DictionaryStringView).await;
        test_join_row_count(1.0, 10, JoinColumnType::DictionaryStringView).await;
    }

    #[test]
    fn test_expected_row_count_calculation() {
        // Verify the expected row count formula

        // match_rate=0.5, repeated_keys=1
        // - Matching keys: 2500
        // - Probe occurrences per key: 2
        // - Build occurrences per key: 1
        // - Total: 2500 * 2 * 1 = 5000
        assert_eq!(expected_row_count(0.5, 1), 5_000);

        // match_rate=0.5, repeated_keys=10
        // - Total: 2500 * 2 * 10 = 50000
        assert_eq!(expected_row_count(0.5, 10), 50_000);

        // match_rate=1.0, repeated_keys=1
        // - Matching keys: 5000
        // - Total: 5000 * 2 * 1 = 10000
        assert_eq!(expected_row_count(1.0, 1), 10_000);

        // match_rate=1.0, repeated_keys=10
        // - Total: 5000 * 2 * 10 = 100000
        assert_eq!(expected_row_count(1.0, 10), 100_000);
    }
}


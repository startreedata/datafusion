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

//! Benchmark for DataFusion FilterExec with Java UDF via JNI and Arrow FFI.
//!
//! This benchmark measures the overhead of calling a Java UDF for filtering
//! using JNI and zero-copy Arrow C Data Interface, compared to native Rust filtering.
//!
//! ## Prerequisites
//!
//! Before running this benchmark, you must compile the Java code:
//!
//! ```bash
//! cd datafusion/physical-plan/benches/jvm
//! mvn compile
//! cd ../../../../
//! ```
//!
//! ## Running the benchmark
//!
//! ```bash
//! # Run all JNI filter benchmarks
//! cargo bench --bench filter_jni_benchmark -p datafusion-physical-plan
//!
//! # Run with fewer samples for quick testing
//! cargo bench --bench filter_jni_benchmark -p datafusion-physical-plan -- --sample-size 10
//!
//! # Compare with native filter benchmark
//! cargo bench --bench filter_bench -p datafusion-physical-plan -- --save-baseline native
//! cargo bench --bench filter_jni_benchmark -p datafusion-physical-plan -- --baseline native
//! ```
//!
//! ## What it measures
//!
//! - **JNI call overhead**: Cost of calling Java methods from Rust
//! - **FFI conversion overhead**: Cost of converting between Rust and C Data Interface
//! - **Java Arrow operations**: Cost of import/filter/export in Java
//! - **Total overhead**: End-to-end comparison with native Rust filtering
//!
//! ## Architecture
//!
//! ```text
//! Rust (DataFusion)           JNI Boundary              Java (Arrow)
//! ─────────────────────────────────────────────────────────────────
//! RecordBatch
//!     ↓
//! arrow::ffi::to_ffi()
//!     ↓
//! FFI_ArrowSchema*   ────────→  long schemaPtr
//! FFI_ArrowArray*    ────────→  long arrayPtr
//!                                ↓
//!                           Data.importRecordBatch()
//!                                ↓
//!                           Apply filter: colInt > 2500
//!                                ↓
//!                           Data.exportRecordBatch()
//!                                ↓
//! FFI_ArrowSchema*   ←────────  long[] [schemaPtr, arrayPtr]
//! FFI_ArrowArray*    ←────────
//!     ↓
//! arrow::ffi::from_ffi()
//!     ↓
//! RecordBatch (filtered)
//! ```

// Note: JVM library loading is handled at runtime by the jni crate (with invocation feature)
// via java-locator. No manual #[link] directive is needed. Ensure JAVA_HOME is set before running.

// Include shared benchmark utilities
#[path = "bench_utils.rs"]
mod bench_utils;

use std::any::Any;



use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int32Array, StructArray};
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, SamplingMode,
    Throughput,
};
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::TaskContext;
use datafusion_expr::{col, ColumnarValue, ScalarFunctionImplementation, Volatility};
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::{collect, ExecutionPlan};
use jni::objects::{JLongArray, JValue};
use jni::sys::jlong;
use jni::JavaVM;

use bench_utils::{
    create_schema, deserialize_zero_copy, serialize_batches_to_sink, serialize_to_ipc,
    BatchSourceExec, FunctionalBatchGenerator,
};

// ============================================================================
// JVM Initialization
// ============================================================================

/// Global JVM instance initialized once before benchmarks
static mut JVM: Option<JavaVM> = None;

/// Initialize JVM with classpath pointing to compiled Java classes.
///
/// This function should be called once before running any benchmarks.
/// The classpath is hardcoded to point to the Maven JAR file.
///
/// The JVM library is dynamically loaded by the jni crate via java-locator,
/// which uses JAVA_HOME environment variable to locate the JVM installation.
fn init_jvm() {
    use jni::InitArgsBuilder;

    unsafe {
        let jvm_ptr = std::ptr::addr_of_mut!(JVM);
        if (*jvm_ptr).is_some() {
            return; // Already initialized
        }

        // Set up JVM arguments with classpath pointing to the compiled JAR
        // Notice you need to run `mvn package` in the jvm directory to compile and package the Java code first
        let classpath = "benches/jvm/target/datafusion-jni-benchmark-1.0-SNAPSHOT.jar";

        // Check if the JAR file exists
        let classpath_path = std::path::Path::new(classpath);
        if !classpath_path.exists() {
            let absolute_path = std::env::current_dir()
                .map(|p| p.join(classpath))
                .unwrap_or_else(|_| classpath_path.to_path_buf());

            panic!(
                "JAR file not found at: {}\n\
                Absolute path: {}\n\
                Please compile the Java code first by running:\n\
                  cd datafusion/physical-plan/benches/jvm\n\
                  mvn package\n\
                  cd ../../../../",
                classpath,
                absolute_path.display()
            );
        }

        let classpath_option = format!("-Djava.class.path={}", classpath);

        let jvm_args = InitArgsBuilder::new()
            .option(&classpath_option)
            .option("--add-opens=java.base/java.nio=ALL-UNNAMED")
            .build()
            .expect("Failed to build JVM arguments");

        let jvm = JavaVM::new(jvm_args)
            .expect("Failed to create JVM. Ensure JAVA_HOME is set and Java is installed.");

        *jvm_ptr = Some(jvm);
    }
}

/// Get JVM instance (panics if not initialized)
fn get_jvm() -> &'static JavaVM {
    unsafe {
        let jvm_ptr = std::ptr::addr_of!(JVM);
        (*jvm_ptr)
            .as_ref()
            .expect("JVM not initialized. Call init_jvm() first.")
    }
}

// ============================================================================
// Java UDF Wrapper - Scalar UDF Implementation
// ============================================================================

/// Calls Java evaluatePredicate method with Arrow FFI pointers for a single column.
///
/// This function:
/// 1. Wraps Int32Array in a RecordBatch and converts to FFI pointers
/// 2. Calls Java Udf.evaluatePredicate(schemaPtr, arrayPtr) via JNI
/// 3. Receives result pointers from Java (boolean array in a RecordBatch)
/// 4. Converts result back to BooleanArray using arrow::ffi::from_ffi()
///
/// # Memory Management
/// - Input FFI pointers are released after Java imports the data
/// - Output FFI pointers are managed by Arrow's Drop implementation
/// - Java is responsible for releasing exported data after Rust imports it
fn call_java_predicate(int_array: &Int32Array) -> Result<BooleanArray> {
    use arrow::datatypes::{Schema, Field};
    use arrow::record_batch::RecordBatch;

    // Create a schema for the input (single Int32 column)
    let input_schema = Schema::new(vec![Field::new("colint", DataType::Int32, true)]);

    // Create a RecordBatch with the Int32Array
    let record_batch = RecordBatch::try_new(
        Arc::new(input_schema),
        vec![Arc::new(int_array.clone()) as Arc<dyn Array>],
    )?;

    // Convert RecordBatch to FFI pointers via StructArray
    let struct_array: StructArray = record_batch.into();
    let (ffi_array, ffi_schema) = to_ffi(&struct_array.to_data())?;

    // Get raw pointers for JNI call
    let schema_ptr = &ffi_schema as *const FFI_ArrowSchema as jlong;
    let array_ptr = &ffi_array as *const FFI_ArrowArray as jlong;

    // Call Java UDF via JNI
    let jvm = get_jvm();
    let mut env = jvm.attach_current_thread()
        .map_err(|e| datafusion_common::DataFusionError::Execution(
            format!("Failed to attach JVM thread: {}", e)
        ))?;

    // Call static method: Udf.evaluatePredicate(long, long) -> long[]
    let result_ptrs = env.call_static_method(
        "org/apache/datafusion/benchmark/Udf",
        "evaluatePredicate",
        "(JJ)[J",
        &[JValue::Long(schema_ptr), JValue::Long(array_ptr)],
    ).map_err(|e| datafusion_common::DataFusionError::Execution(
        format!("Java UDF call failed: {}", e)
    ))?;

    // Extract the long[] result containing [schemaPtr, arrayPtr]
    let result_array = result_ptrs.l()
        .map_err(|e| datafusion_common::DataFusionError::Execution(
            format!("Failed to extract result array: {}", e)
        ))?;

    let result_array = JLongArray::from(result_array);

    // Get the two pointers from the result array
    let mut ptrs = [0i64; 2];
    env.get_long_array_region(&result_array, 0, &mut ptrs)
        .map_err(|e| datafusion_common::DataFusionError::Execution(
            format!("Failed to read result pointers: {}", e)
        ))?;

    let result_schema_ptr = ptrs[0] as *mut FFI_ArrowSchema;
    let result_array_ptr = ptrs[1] as *mut FFI_ArrowArray;

    // Safety: We trust that Java has allocated valid FFI structures
    // The from_ffi call will take ownership and handle cleanup via release callbacks
    let result_array_data = unsafe {
        let result_ffi_schema = FFI_ArrowSchema::from_raw(result_schema_ptr);
        let result_ffi_array = FFI_ArrowArray::from_raw(result_array_ptr);
        from_ffi(result_ffi_array, &result_ffi_schema)?
    };

    // Java returns a VectorSchemaRoot (struct type) with one boolean column
    // We need to extract the child array directly from the ArrayData
    if !matches!(result_array_data.data_type(), DataType::Struct(_)) {
        return Err(datafusion_common::DataFusionError::Execution(
            format!("Expected Struct type from Java, got {:?}", result_array_data.data_type())
        ));
    }

    // Get the first child array data (the boolean column) directly
    // We don't use StructArray::from() because it tries to slice child arrays
    // based on the struct's offset/length, which can cause issues
    let child_data = result_array_data.child_data().get(0)
        .ok_or_else(|| datafusion_common::DataFusionError::Execution(
            "Expected at least one child in result struct".to_string()
        ))?;

    // Construct the BooleanArray directly from the child ArrayData
    let boolean_array = BooleanArray::from(child_data.clone());

    Ok(boolean_array)
}

/// Create a scalar UDF that wraps the Java predicate function
fn create_java_predicate_udf() -> ScalarFunctionImplementation {
    Arc::new(move |args: &[ColumnarValue]| -> Result<ColumnarValue> {
        // Extract the Int32Array from the input
        let int_array = match &args[0] {
            ColumnarValue::Array(arr) => arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Expected Int32Array")
                .clone(),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(val))) => {
                // Single value - create array with one element
                Int32Array::from(vec![*val])
            }
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => {
                // Null value - create array with one null
                Int32Array::from(vec![None as Option<i32>])
            }
            _ => {
                return Err(datafusion_common::DataFusionError::Execution(
                    "Expected Int32 input".to_string(),
                ))
            }
        };

        // Call Java predicate
        let result_array = call_java_predicate(&int_array)?;

        Ok(ColumnarValue::Array(Arc::new(result_array)))
    })
}

// ============================================================================
// Filter Plan Creation with Java UDF
// ============================================================================

/// Creates a FilterExec plan that uses a Java UDF for the predicate evaluation.
///
/// This function creates a standard DataFusion FilterExec that applies the
/// Java UDF predicate (colInt > 2500) via JNI.
fn create_java_filter_plan(input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    use datafusion_expr::create_udf;
    use datafusion_physical_expr::create_physical_expr;
    use datafusion_expr::Expr;
    use datafusion_common::DFSchema;
    use datafusion_expr::execution_props::ExecutionProps;

    let schema = input.schema();

    // Create the Java predicate UDF
    let java_udf_impl = create_java_predicate_udf();

    // Create UDF with signature
    let java_udf = create_udf(
        "java_gt_2500",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Immutable,
        java_udf_impl,
    );

    // Create the expression: java_gt_2500(colint)
    let col_expr = col("colint");
    let udf_expr = Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction::new_udf(
        Arc::new(java_udf),
        vec![col_expr],
    ));

    // Convert logical expression to physical expression
    let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
    let execution_props = ExecutionProps::new();
    let physical_expr = create_physical_expr(
        &udf_expr,
        &df_schema,
        &execution_props,
    )?;

    // Create FilterExec with the Java UDF predicate
    Ok(Arc::new(FilterExec::try_new(physical_expr, input)?))
}

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Main benchmark function for JNI filter execution.
///
/// This benchmark measures two scenarios:
///
/// 1. **filter_only**: Filter execution only (using Java UDF)
///    - Isolates the JNI/FFI overhead and Java filter performance
///    - Uses pre-generated batches directly (no deserialization)
///
/// 2. **full_pipeline**: Complete deser + Java filter + output serialization
///    - Real-world end-to-end latency including JNI overhead
///    - Relevant for understanding total cost of Java UDF integration
fn bench_filter(c: &mut Criterion) {
    // Initialize JVM once before all benchmarks
    init_jvm();

    // Create a single-threaded Tokio runtime for async execution
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("filter_jni_benchmark");
    group.sampling_mode(SamplingMode::Flat);

    // Configuration: 1M rows total (10K rows × 100 batches)
    let rows_per_batch = 10_000;
    let num_batches = 100;
    let total_rows = rows_per_batch * num_batches;

    // Test different binary column sizes to understand serialization overhead
    let binary_sizes = vec![10, 1024, 2048];

    for binary_size in binary_sizes {
        let label = format!("1M_rows_binary_{binary_size}B");

        // Generate test data and serialize to IPC format
        let schema = create_schema();
        let mut generator =
            FunctionalBatchGenerator::new(Arc::clone(&schema), rows_per_batch, num_batches, binary_size);
        let batches = generator.generate_batches();
        let ipc_data = serialize_to_ipc(&batches, &schema);
        let ipc_size = ipc_data.len();
        let ipc_buffer = Buffer::from_vec(ipc_data);

        // Log configuration
        println!(
            "Config: {} rows, binary_size={} bytes, IPC size={:.2} MB",
            total_rows,
            binary_size,
            ipc_size as f64 / (1024.0 * 1024.0)
        );

        group.throughput(Throughput::Bytes(ipc_size as u64));

        // Benchmark 2: Full pipeline (deser + Java filter + output serialization)
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", &label),
            &ipc_buffer,
            |b, ipc_buffer| {
                b.iter(|| {
                    rt.block_on(async {
                        let (schema, batches) = deserialize_zero_copy(ipc_buffer);
                        let source = Arc::new(BatchSourceExec::new(Arc::clone(&schema), batches))
                            as Arc<dyn ExecutionPlan>;
                        let plan = create_java_filter_plan(source).unwrap();
                        let task_ctx = Arc::new(TaskContext::default());
                        let results = collect(plan, task_ctx).await.unwrap();
                        black_box(serialize_batches_to_sink(&results, &schema))
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the Java filter returns the correct number of rows.
    ///
    /// The filter predicate is `colInt > 2500`, and colInt values follow the pattern `i % 5000`.
    /// This means:
    /// - Values range from 0 to 4999
    /// - Values > 2500 are: 2501, 2502, ..., 4999 (2499 values)
    /// - Expected selectivity: 2499/5000 = 49.98%
    ///
    /// For 1M total rows (100 batches × 10K rows), we expect:
    /// - Filtered rows: 1,000,000 × 0.4998 = 499,800 rows
    #[test]
    fn test_java_filter_row_count() {
        // Initialize JVM
        init_jvm();

        // Create Tokio runtime
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        // Configuration matching the benchmark
        let rows_per_batch = 10_000;
        let num_batches = 100;
        let total_rows = rows_per_batch * num_batches;
        let binary_size = 10;

        // Generate test data
        let schema = create_schema();
        let mut generator =
            FunctionalBatchGenerator::new(Arc::clone(&schema), rows_per_batch, num_batches, binary_size);
        let batches = generator.generate_batches();

        // Create execution plan with Java filter
        let source = Arc::new(BatchSourceExec::new(Arc::clone(&schema), batches))
            as Arc<dyn ExecutionPlan>;
        let plan = create_java_filter_plan(source).unwrap();

        // Execute the plan
        let task_ctx = Arc::new(TaskContext::default());
        let results = rt.block_on(async {
            collect(plan, task_ctx).await.unwrap()
        });

        // Count total rows in results
        let filtered_row_count: usize = results.iter().map(|batch| batch.num_rows()).sum();

        // Calculate expected count
        // colInt values: i % 5000, so values are 0..4999
        // Filter: colInt > 2500, so we keep 2501..4999 = 2499 values per 5000
        // Expected: (total_rows / 5000) * 2499
        let expected_count = (total_rows / 500) * 2499;

        assert_eq!(
            filtered_row_count, expected_count,
            "Java filter returned {} rows, expected {} rows ({}% selectivity)",
            filtered_row_count,
            expected_count,
            (expected_count as f64 / total_rows as f64) * 100.0
        );

        println!(
            "✓ Java filter correctness test passed: {} rows filtered from {} total rows ({:.2}% selectivity)",
            filtered_row_count,
            total_rows,
            (filtered_row_count as f64 / total_rows as f64) * 100.0
        );
    }

    /// Test that the Java filter produces the same results as the expected filter logic
    /// by verifying that all returned values actually satisfy the predicate.
    #[test]
    fn test_java_filter_correctness() {
        // Initialize JVM
        init_jvm();

        // Create Tokio runtime
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        // Use smaller dataset for detailed validation
        let rows_per_batch = 1_000;
        let num_batches = 10;
        let binary_size = 10;

        // Generate test data
        let schema = create_schema();
        let mut generator =
            FunctionalBatchGenerator::new(Arc::clone(&schema), rows_per_batch, num_batches, binary_size);
        let batches = generator.generate_batches();

        // Create execution plan with Java filter
        let source = Arc::new(BatchSourceExec::new(Arc::clone(&schema), batches))
            as Arc<dyn ExecutionPlan>;
        let plan = create_java_filter_plan(source).unwrap();

        // Execute the plan
        let task_ctx = Arc::new(TaskContext::default());
        let results = rt.block_on(async {
            collect(plan, task_ctx).await.unwrap()
        });

        // Verify all returned rows satisfy the predicate: colInt > 2500
        for (batch_idx, batch) in results.iter().enumerate() {
            let colint_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Expected Int32Array for colint");

            for row_idx in 0..colint_array.len() {
                let value = colint_array.value(row_idx);
                assert!(
                    value > 2500,
                    "Batch {}, row {}: expected value > 2500, got {}",
                    batch_idx,
                    row_idx,
                    value
                );
            }
        }

        let total_filtered_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
        println!(
            "✓ Java filter correctness test passed: all {} filtered rows satisfy colInt > 2500",
            total_filtered_rows
        );
    }
}


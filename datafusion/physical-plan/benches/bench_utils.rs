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

//! Shared utilities for deserialization + execution benchmarks.
//!
//! This module provides common functionality for benchmarks that measure
//! Arrow IPC deserialization combined with DataFusion execution plans.

use std::io::Write;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::convert::fb_to_schema;
use arrow::ipc::reader::{FileDecoder, read_footer_length};
use arrow::ipc::writer::FileWriter;
use arrow::ipc::root_as_footer;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::memory::MemoryStream;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ============================================================================
// Schema Definition
// ============================================================================

/// Creates the benchmark schema with the following columns:
/// - colInt: Int32 - integer values with modulo pattern
/// - colLong: Int64 - long values with modulo pattern  
/// - colFloat: Float32 - floating point values
/// - colDouble: Float64 - double precision values
/// - colString: Utf8 - string values with limited cardinality
/// - colBinary: Binary - random binary data of configurable size
pub fn create_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("colInt", DataType::Int32, false),
        Field::new("colLong", DataType::Int64, false),
        Field::new("colFloat", DataType::Float32, false),
        Field::new("colDouble", DataType::Float64, false),
        Field::new("colString", DataType::Utf8, false),
        Field::new("colBinary", DataType::Binary, false),
    ]))
}

// ============================================================================
// Data Generation
// ============================================================================

/// Generates record batches with deterministic data for benchmarking.
///
/// Each cell value is computed as a function of the row index, ensuring
/// reproducible benchmark results. The data patterns are designed to be
/// realistic for query processing benchmarks:
///
/// - `colInt`: `i % 5000` - creates 5000 distinct values
/// - `colLong`: `i % 5000` - same pattern as colInt
/// - `colFloat`: `i / 2.0` - monotonically increasing
/// - `colDouble`: `i / 3.0` - monotonically increasing
/// - `colString`: `"str_" + (i % 100)` - 100 distinct string values
/// - `colBinary`: random bytes of configurable size (seeded for reproducibility)
pub struct FunctionalBatchGenerator {
    /// Schema for generated batches
    schema: SchemaRef,
    /// Number of rows in each batch
    rows_per_batch: usize,
    /// Total number of batches to generate
    num_batches: usize,
    /// Size in bytes for the binary column
    binary_size: usize,
    /// Random number generator for binary data (seeded for reproducibility)
    rng: StdRng,
}

impl FunctionalBatchGenerator {
    /// Creates a new batch generator.
    ///
    /// # Arguments
    /// * `schema` - Arrow schema for the generated batches
    /// * `rows_per_batch` - Number of rows per batch
    /// * `num_batches` - Total number of batches to generate
    /// * `binary_size` - Size in bytes for the binary column values
    pub fn new(
        schema: SchemaRef,
        rows_per_batch: usize,
        num_batches: usize,
        binary_size: usize,
    ) -> Self {
        // Use a fixed seed for reproducible benchmarks
        let rng = StdRng::seed_from_u64(42);
        Self {
            schema,
            rows_per_batch,
            num_batches,
            binary_size,
            rng,
        }
    }

    /// Generates a single record batch for the given batch index.
    ///
    /// Row indices are calculated as: `batch_index * rows_per_batch + local_row_index`
    fn generate_batch(&mut self, batch_index: usize) -> RecordBatch {
        let start_row = batch_index * self.rows_per_batch;
        let num_rows = self.rows_per_batch;

        // Clone field names to avoid borrowing self while calling generate_column
        let field_names: Vec<String> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let columns: Vec<ArrayRef> = field_names
            .iter()
            .map(|name| self.generate_column(name, start_row, num_rows))
            .collect();

        RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .expect("Failed to create record batch")
    }

    /// Generates a single column array based on field name.
    ///
    /// Values are deterministic functions of the global row index `i`:
    /// - colInt: `i % 5000` (5000 distinct values)
    /// - colLong: `i % 5000` (5000 distinct values)
    /// - colFloat: `i / 2.0` (monotonically increasing)
    /// - colDouble: `i / 3.0` (monotonically increasing)
    /// - colString: `"str_{i % 100}"` (100 distinct values)
    /// - colBinary: random bytes of `binary_size` length
    fn generate_column(&mut self, field_name: &str, start_row: usize, num_rows: usize) -> ArrayRef {
        match field_name {
            "colInt" => {
                // Integer values with modulo 5000 pattern for reasonable cardinality
                let values: Vec<i32> = (0..num_rows)
                    .map(|i| ((start_row + i) % 5000) as i32)
                    .collect();
                Arc::new(Int32Array::from(values))
            }
            "colLong" => {
                // Long values with same modulo pattern as colInt
                let values: Vec<i64> = (0..num_rows)
                    .map(|i| ((start_row + i) % 5000) as i64)
                    .collect();
                Arc::new(Int64Array::from(values))
            }
            "colFloat" => {
                // Monotonically increasing float values
                let values: Vec<f32> = (0..num_rows)
                    .map(|i| ((start_row + i) as f32) / 2.0)
                    .collect();
                Arc::new(Float32Array::from(values))
            }
            "colDouble" => {
                // Monotonically increasing double values
                let values: Vec<f64> = (0..num_rows)
                    .map(|i| ((start_row + i) as f64) / 3.0)
                    .collect();
                Arc::new(Float64Array::from(values))
            }
            "colString" => {
                // String values with 100 distinct values (low cardinality)
                let values: Vec<String> = (0..num_rows)
                    .map(|i| format!("str_{}", (start_row + i) % 100))
                    .collect();
                Arc::new(StringArray::from(values))
            }
            "colBinary" => {
                // Random binary data of configurable size
                let values: Vec<Vec<u8>> = (0..num_rows)
                    .map(|_| {
                        let mut buf = vec![0u8; self.binary_size];
                        self.rng.fill(&mut buf[..]);
                        buf
                    })
                    .collect();
                let values: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
                Arc::new(BinaryArray::from(values))
            }
            _ => panic!("Unknown column: {}", field_name),
        }
    }

    /// Generates all batches.
    ///
    /// Returns a vector of `num_batches` record batches, each containing
    /// `rows_per_batch` rows.
    pub fn generate_batches(&mut self) -> Vec<RecordBatch> {
        (0..self.num_batches)
            .map(|i| self.generate_batch(i))
            .collect()
    }
}

// ============================================================================
// Arrow IPC Serialization / Deserialization
// ============================================================================

/// Serializes record batches to Arrow IPC file format (in-memory).
///
/// This simulates receiving Arrow data over a network or reading from storage.
/// Uses the IPC file format (with footer) to enable zero-copy deserialization.
///
/// # Arguments
/// * `batches` - Record batches to serialize
/// * `schema` - Arrow schema (must match batches)
///
/// # Returns
/// Serialized IPC data as a byte vector
pub fn serialize_to_ipc(batches: &[RecordBatch], schema: &SchemaRef) -> Vec<u8> {
    let mut buffer = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buffer, schema).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
    }
    buffer
}

/// Deserializes record batches from Arrow IPC file format using zero-copy.
///
/// This is the operation being benchmarked - converting serialized Arrow IPC
/// data back into in-memory record batches that can be processed by DataFusion.
///
/// Zero-copy means the Arrow arrays refer directly to the provided buffer,
/// avoiding memory copying during deserialization.
///
/// # Arguments
/// * `buffer` - Serialized IPC data
///
/// # Returns
/// Tuple of (schema, batches) extracted from the IPC data
pub fn deserialize_zero_copy(buffer: &Buffer) -> (SchemaRef, Vec<RecordBatch>) {
    // Read the footer to get schema and batch locations
    let trailer_start = buffer.len() - 10;
    let footer_len = read_footer_length(buffer[trailer_start..].try_into().unwrap()).unwrap();
    let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start]).unwrap();

    let schema = Arc::new(fb_to_schema(footer.schema().unwrap()));
    let mut decoder = FileDecoder::new(Arc::clone(&schema), footer.version());

    // Read dictionaries if present
    for block in footer.dictionaries().iter().flatten() {
        let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
        let data = buffer.slice_with_length(block.offset() as _, block_len);
        decoder.read_dictionary(block, &data).unwrap();
    }

    // Read all record batches
    let mut batches = Vec::new();
    if let Some(batch_blocks) = footer.recordBatches() {
        for block in batch_blocks {
            let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
            let data = buffer.slice_with_length(block.offset() as _, block_len);
            if let Some(batch) = decoder.read_record_batch(&block, &data).unwrap() {
                batches.push(batch);
            }
        }
    }

    (schema, batches)
}

/// Serializes execution results (record batches) back to Arrow IPC format.
///
/// This measures the overhead of serializing query results, which is relevant
/// for scenarios where results need to be sent over a network or stored.
///
/// # Arguments
/// * `batches` - Result record batches from query execution
///
/// # Returns
/// Serialized IPC data as a byte vector
pub fn serialize_results_to_ipc(batches: &[RecordBatch]) -> Vec<u8> {
    if batches.is_empty() {
        return Vec::new();
    }
    let schema = batches[0].schema();
    serialize_to_ipc(batches, &schema)
}

/// A writer that discards all data written to it.
///
/// This is useful for benchmarking serialization overhead without
/// including actual I/O or memory allocation costs.
struct SinkWriter {
    bytes_written: usize,
}

impl SinkWriter {
    fn new() -> Self {
        Self { bytes_written: 0 }
    }

    /// Returns the total number of bytes that would have been written.
    fn bytes_written(&self) -> usize {
        self.bytes_written
    }
}

impl Write for SinkWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes_written += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Serializes a record batch to a sink that drops all data.
///
/// This function measures pure serialization overhead by writing to a sink
/// that discards data instead of allocating memory or performing I/O.
/// Useful for benchmarking the CPU cost of serialization alone.
///
/// # Arguments
/// * `batch` - The record batch to serialize
///
/// # Returns
/// The number of bytes that would have been written
///
/// # Example
/// ```ignore
/// let batch = create_test_batch();
/// let bytes_written = serialize_to_sink(&batch);
/// println!("Serialization would write {} bytes", bytes_written);
/// ```
pub fn serialize_to_sink(batch: &RecordBatch) -> usize {
    let schema = batch.schema();
    let mut sink = SinkWriter::new();
    {
        let mut writer = FileWriter::try_new(&mut sink, &schema).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
    sink.bytes_written()
}

/// Serializes multiple record batches to a sink that drops all data.
///
/// This function measures pure serialization overhead by writing to a sink
/// that discards data instead of allocating memory or performing I/O.
/// Useful for benchmarking the CPU cost of serialization alone.
///
/// # Arguments
/// * `batches` - The record batches to serialize
/// * `schema` - The schema for the batches
///
/// # Returns
/// The number of bytes that would have been written
///
/// # Example
/// ```ignore
/// let batches = create_test_batches();
/// let schema = batches[0].schema();
/// let bytes_written = serialize_batches_to_sink(&batches, &schema);
/// println!("Serialization would write {} bytes", bytes_written);
/// ```
pub fn serialize_batches_to_sink(batches: &[RecordBatch], schema: &SchemaRef) -> usize {
    let mut sink = SinkWriter::new();
    {
        let mut writer = FileWriter::try_new(&mut sink, schema).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
    }
    sink.bytes_written()
}

// ============================================================================
// Execution Plan Source
// ============================================================================

/// A simple execution plan that serves pre-loaded record batches.
///
/// This is used as the leaf node in benchmark execution plans. It wraps
/// already-deserialized batches and makes them available to downstream
/// operators (Filter, Sort, etc.) via the standard ExecutionPlan interface.
///
/// Unlike file-based sources, this has no I/O overhead - it simply streams
/// the in-memory batches, allowing us to isolate operator performance.
#[derive(Debug)]
pub struct BatchSourceExec {
    /// Schema of the batches
    schema: SchemaRef,
    /// Pre-loaded record batches to serve
    batches: Vec<RecordBatch>,
    /// Cached plan properties (partitioning, ordering, etc.)
    cache: PlanProperties,
}

impl BatchSourceExec {
    /// Creates a new batch source execution plan.
    ///
    /// # Arguments
    /// * `schema` - Schema for the batches
    /// * `batches` - Pre-loaded record batches to serve
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            schema,
            batches,
            cache,
        }
    }
}

impl DisplayAs for BatchSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BatchSourceExec: batches={}", self.batches.len())
    }
}

impl ExecutionPlan for BatchSourceExec {
    fn name(&self) -> &'static str {
        "BatchSourceExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        Ok(Box::pin(MemoryStream::try_new(
            self.batches.clone(),
            Arc::clone(&self.schema),
            None,
        )?))
    }
}

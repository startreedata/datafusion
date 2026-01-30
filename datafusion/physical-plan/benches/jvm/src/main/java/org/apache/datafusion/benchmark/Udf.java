package org.apache.datafusion.benchmark;


import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.CDataDictionaryProvider;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;


/**
 * Java UDF for filtering Arrow RecordBatches via JNI with zero-copy FFI.
 *
 * This class receives Arrow data from Rust via the C Data Interface,
 * applies a filter predicate (colInt > 2500), and returns a boolean array
 * indicating which rows pass the filter.
 */
public class Udf {

  // Static allocator initialized once for all operations
  private static final BufferAllocator allocator = new RootAllocator();

  /**
   * Evaluates the predicate: value > 2500
   *
   * Returns a boolean array indicating which rows pass the filter. For each filter,
   * there is a corresponding boolean value: true if the row passes, false otherwise.
   *
   * @param schemaPtr Pointer to FFI_ArrowSchema (C Data Interface)
   * @param arrayPtr Pointer to FFI_ArrowArray (C Data Interface)
   * @return Array of [newSchemaPtr, newArrayPtr] for the boolean result array
   */
  public static long[] evaluatePredicate(long schemaPtr, long arrayPtr) {
    try {
      // Import Array from FFI pointers
      ArrowSchema arrowSchema = ArrowSchema.wrap(schemaPtr);
      ArrowArray arrowArray = ArrowArray.wrap(arrayPtr);

      VectorSchemaRoot root = Data.importVectorSchemaRoot(
        allocator,
        arrowArray,
        arrowSchema,
        new CDataDictionaryProvider()
      );

      // Get the integer column (assuming single column input)
      IntVector intVector = (IntVector) root.getVector(0);
      if (intVector == null) {
        throw new RuntimeException("Expected integer vector as input");
      }

      int rowCount = root.getRowCount();

      // Create result schema root with single boolean column
      Field field = new Field("result", FieldType.nullable(new ArrowType.Bool()), null);
      Schema resultSchema = new Schema(java.util.Collections.singletonList(field));

      VectorSchemaRoot resultRoot = VectorSchemaRoot.create(resultSchema, allocator);
      resultRoot.setRowCount(rowCount);

      // Get the BitVector from the result root
      BitVector resultVector = (BitVector) resultRoot.getVector(0);
      resultVector.allocateNew(rowCount);

      // Evaluate predicate for each row
      for (int i = 0; i < rowCount; i++) {
        boolean passes = !intVector.isNull(i) && intVector.get(i) > 2500;
        if (passes) {
          resultVector.set(i, 1);
        }
        // Note: BitVector bits are initialized to 0 by allocateNew(), so no need to explicitly set false values
      }
      resultVector.setValueCount(rowCount);

      // Export result to FFI pointers
      ArrowArray resultArray = ArrowArray.allocateNew(allocator);
      ArrowSchema resultArrowSchema = ArrowSchema.allocateNew(allocator);

      Data.exportVectorSchemaRoot(
        allocator,
        resultRoot,
        new CDataDictionaryProvider(),
        resultArray,
        resultArrowSchema
      );

      // Clean up input root (can be closed after export since we've consumed it)
      root.close();

      // NOTE: resultRoot is NOT closed here because the exported FFI pointers
      // reference its memory. Arrow's C Data Interface handles cleanup via
      // release callbacks when Rust calls from_ffi() to import the data.
      // The release callback will eventually free the resultRoot memory.

      return new long[] { resultArrowSchema.memoryAddress(), resultArray.memoryAddress() };
    } catch (Exception e) {
      throw new RuntimeException("Error in Java UDF evaluation", e);
    }
  }

  /**
   * Legacy method for backward compatibility - filters entire batches
   * For use with standard DataFusion filter, use evaluatePredicate instead
   */
  public static long[] filterBatch(long schemaPtr, long arrayPtr) {
    return evaluatePredicate(schemaPtr, arrayPtr);
  }
}

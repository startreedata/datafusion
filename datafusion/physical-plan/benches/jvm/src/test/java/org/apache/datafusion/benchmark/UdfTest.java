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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for the Udf.evaluatePredicate method to verify correct behavior
 * when returning boolean arrays via Arrow FFI.
 */
public class UdfTest {

    private BufferAllocator allocator;

    @BeforeEach
    public void setup() {
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @AfterEach
    public void tearDown() {
        allocator.close();
    }

    @Test
    public void testEvaluatePredicate_BasicFunctionality() {
        // Create input with 10 rows
        int rowCount = 10;

        // Create input schema and data
        Field intField = new Field("colint", FieldType.nullable(new ArrowType.Int(32, true)), null);
        Schema inputSchema = new Schema(Collections.singletonList(intField));

        VectorSchemaRoot inputRoot = VectorSchemaRoot.create(inputSchema, allocator);
        inputRoot.setRowCount(rowCount);

        IntVector intVector = (IntVector) inputRoot.getVector(0);

        // Fill with test data: [1000, 2000, 2500, 2501, 3000, 3500, 4000, null, 1500, 2600]
        int[] testValues = {1000, 2000, 2500, 2501, 3000, 3500, 4000, -1, 1500, 2600};
        boolean[] expectedResults = {false, false, false, true, true, true, true, false, false, true};

        for (int i = 0; i < rowCount; i++) {
            if (testValues[i] == -1) {
                intVector.setNull(i);
            } else {
                intVector.set(i, testValues[i]);
            }
        }
        intVector.setValueCount(rowCount);

        // Export input to FFI
        ArrowArray inputArray = ArrowArray.allocateNew(allocator);
        ArrowSchema inputArrowSchema = ArrowSchema.allocateNew(allocator);

        Data.exportVectorSchemaRoot(
            allocator,
            inputRoot,
            new CDataDictionaryProvider(),
            inputArray,
            inputArrowSchema
        );

        long inputSchemaPtr = inputArrowSchema.memoryAddress();
        long inputArrayPtr = inputArray.memoryAddress();

        // Call the UDF
        long[] resultPtrs = Udf.evaluatePredicate(inputSchemaPtr, inputArrayPtr);

        assertNotNull(resultPtrs, "Result pointers should not be null");
        assertEquals(2, resultPtrs.length, "Should return array with 2 pointers [schemaPtr, arrayPtr]");

        // Import result from FFI
        ArrowSchema resultArrowSchema = ArrowSchema.wrap(resultPtrs[0]);
        ArrowArray resultArray = ArrowArray.wrap(resultPtrs[1]);

        VectorSchemaRoot resultRoot = Data.importVectorSchemaRoot(
            allocator,
            resultArray,
            resultArrowSchema,
            new CDataDictionaryProvider()
        );

        // Clean up input resources after UDF call
        inputArray.close();
        inputArrowSchema.close();
        inputRoot.close();

        // Verify result structure
        assertEquals(1, resultRoot.getFieldVectors().size(), "Result should have 1 column");
        assertEquals(rowCount, resultRoot.getRowCount(), "Result row count should match input");

        // Verify result is a BitVector (boolean)
        assertTrue(resultRoot.getVector(0) instanceof BitVector,
            "Result column should be BitVector, got: " + resultRoot.getVector(0).getClass().getName());

        BitVector resultVector = (BitVector) resultRoot.getVector(0);
        assertEquals(rowCount, resultVector.getValueCount(),
            "Result vector value count should be " + rowCount + ", got: " + resultVector.getValueCount());

        // Verify predicate results
        for (int i = 0; i < rowCount; i++) {
            boolean actual = resultVector.isSet(i) != 0;
            assertEquals(expectedResults[i], actual,
                String.format("Row %d: value=%s, expected=%s, got=%s",
                    i,
                    testValues[i] == -1 ? "null" : testValues[i],
                    expectedResults[i],
                    actual));
        }

        // Cleanup
        resultRoot.close();
        resultArrowSchema.close();
        resultArray.close();
    }

    @Test
    public void testEvaluatePredicate_LargeDataset() {
        // Test with 10,000 rows (same as benchmark)
        int rowCount = 10000;

        Field intField = new Field("colint", FieldType.nullable(new ArrowType.Int(32, true)), null);
        Schema inputSchema = new Schema(Collections.singletonList(intField));

        VectorSchemaRoot inputRoot = VectorSchemaRoot.create(inputSchema, allocator);
        inputRoot.setRowCount(rowCount);

        IntVector intVector = (IntVector) inputRoot.getVector(0);

        // Fill with values 0 to 9999
        int expectedPassCount = 0;
        for (int i = 0; i < rowCount; i++) {
            intVector.set(i, i);
            if (i > 2500) {
                expectedPassCount++;
            }
        }
        intVector.setValueCount(rowCount);

        // Export input to FFI
        ArrowArray inputArray = ArrowArray.allocateNew(allocator);
        ArrowSchema inputArrowSchema = ArrowSchema.allocateNew(allocator);

        Data.exportVectorSchemaRoot(
            allocator,
            inputRoot,
            new CDataDictionaryProvider(),
            inputArray,
            inputArrowSchema
        );

        long inputSchemaPtr = inputArrowSchema.memoryAddress();
        long inputArrayPtr = inputArray.memoryAddress();

        // Call the UDF
        long[] resultPtrs = Udf.evaluatePredicate(inputSchemaPtr, inputArrayPtr);

        // Import result
        ArrowSchema resultArrowSchema = ArrowSchema.wrap(resultPtrs[0]);
        ArrowArray resultArray = ArrowArray.wrap(resultPtrs[1]);

        VectorSchemaRoot resultRoot = Data.importVectorSchemaRoot(
            allocator,
            resultArray,
            resultArrowSchema,
            new CDataDictionaryProvider()
        );

        // Clean up input resources after UDF call
        inputArray.close();
        inputArrowSchema.close();
        inputRoot.close();

        // Verify structure
        assertEquals(rowCount, resultRoot.getRowCount(),
            "Result row count should be " + rowCount + ", got: " + resultRoot.getRowCount());

        BitVector resultVector = (BitVector) resultRoot.getVector(0);
        assertEquals(rowCount, resultVector.getValueCount(),
            "Result vector value count should be " + rowCount + ", got: " + resultVector.getValueCount());

        // Count how many pass the predicate
        int actualPassCount = 0;
        for (int i = 0; i < rowCount; i++) {
            if (resultVector.isSet(i) != 0) {
                actualPassCount++;
            }
        }

        assertEquals(expectedPassCount, actualPassCount,
            String.format("Expected %d rows to pass predicate (> 2500), but got %d",
                expectedPassCount, actualPassCount));

        // Verify specific values
        assertFalse(resultVector.isSet(0) != 0, "Value 0 should not pass (0 <= 2500)");
        assertFalse(resultVector.isSet(2500) != 0, "Value 2500 should not pass (2500 <= 2500)");
        assertTrue(resultVector.isSet(2501) != 0, "Value 2501 should pass (2501 > 2500)");
        assertTrue(resultVector.isSet(9999) != 0, "Value 9999 should pass (9999 > 2500)");

        // Cleanup
        resultRoot.close();
        resultArrowSchema.close();
        resultArray.close();
    }

    @Test
    public void testEvaluatePredicate_AllNull() {
        int rowCount = 100;

        Field intField = new Field("colint", FieldType.nullable(new ArrowType.Int(32, true)), null);
        Schema inputSchema = new Schema(Collections.singletonList(intField));

        VectorSchemaRoot inputRoot = VectorSchemaRoot.create(inputSchema, allocator);
        inputRoot.setRowCount(rowCount);

        IntVector intVector = (IntVector) inputRoot.getVector(0);

        // All null values
        for (int i = 0; i < rowCount; i++) {
            intVector.setNull(i);
        }
        intVector.setValueCount(rowCount);

        // Export and call UDF
        ArrowArray inputArray = ArrowArray.allocateNew(allocator);
        ArrowSchema inputArrowSchema = ArrowSchema.allocateNew(allocator);

        Data.exportVectorSchemaRoot(
            allocator,
            inputRoot,
            new CDataDictionaryProvider(),
            inputArray,
            inputArrowSchema
        );

        long[] resultPtrs = Udf.evaluatePredicate(inputArrowSchema.memoryAddress(), inputArray.memoryAddress());

        // Import result
        ArrowSchema resultArrowSchema = ArrowSchema.wrap(resultPtrs[0]);
        ArrowArray resultArray = ArrowArray.wrap(resultPtrs[1]);

        VectorSchemaRoot resultRoot = Data.importVectorSchemaRoot(
            allocator,
            resultArray,
            resultArrowSchema,
            new CDataDictionaryProvider()
        );

        // Clean up input resources after UDF call
        inputArray.close();
        inputArrowSchema.close();
        inputRoot.close();

        // All nulls should result in false (not passing predicate)
        BitVector resultVector = (BitVector) resultRoot.getVector(0);
        for (int i = 0; i < rowCount; i++) {
            assertFalse(resultVector.isSet(i) != 0,
                "Null values should not pass predicate");
        }

        // Cleanup
        resultRoot.close();
        resultArrowSchema.close();
        resultArray.close();
    }
}

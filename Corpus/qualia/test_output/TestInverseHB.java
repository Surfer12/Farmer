import java.util.*;

public class TestInverseHB {
    public static void main(String[] args) {
        System.out.println("Testing InverseHierarchicalBayesianModel...");

        try {
            // Create model
            var model = new InverseHierarchicalBayesianModel();
            System.out.println("✅ Model created successfully");

            // Create simple test data
            var claim = new ClaimData("test", true, 0.5, 0.3, 0.7);
            var observation = new InverseHierarchicalBayesianModel.Observation(claim, 0.8, true);
            var observations = List.of(observation);

            System.out.println("✅ Test observation created");

            // Try parameter recovery
            var result = model.recoverParameters(observations);
            if (result != null) {
                System.out.println("✅ Parameter recovery completed");
                System.out.println("Recovered parameters: S=" + result.recoveredParameters.S() +
                                 ", N=" + result.recoveredParameters.N());
                System.out.println("Confidence: " + result.confidence);
            } else {
                System.out.println("⚠️  Parameter recovery returned null");
            }

            // Try structure learning
            var structureResult = model.learnStructure(observations);
            if (structureResult != null) {
                System.out.println("✅ Structure learning completed");
                System.out.println("Confidence: " + structureResult.structureConfidence);
            } else {
                System.out.println("⚠️  Structure learning returned null");
            }

            System.out.println("✅ Inverse HB Model tests completed!");

        } catch (Exception e) {
            System.out.println("❌ Inverse HB Model test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

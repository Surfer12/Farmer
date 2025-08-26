import java.util.*;

public class TestModelParameters {
    public static void main(String[] args) {
        System.out.println("Testing ModelParameters...");

        try {
            // Test valid parameters
            var params = new ModelParameters(0.7, 0.6, 0.5, 1.2);
            System.out.println("✅ Valid parameters: S=" + params.S() +
                             ", N=" + params.N() + ", alpha=" + params.alpha() +
                             ", beta=" + params.beta());

            // Test invalid parameters
            try {
                new ModelParameters(-0.1, 0.5, 0.5, 1.0); // Invalid S
                System.out.println("❌ Should have thrown exception for invalid S");
            } catch (Exception e) {
                System.out.println("✅ Correctly rejected invalid S: " + e.getMessage());
            }

            System.out.println("✅ ModelParameters tests passed!");

        } catch (Exception e) {
            System.out.println("❌ ModelParameters test failed: " + e.getMessage());
        }
    }
}

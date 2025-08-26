public class TestModelPriors {
    public static void main(String[] args) {
        System.out.println("Testing ModelPriors...");

        try {
            // Test default priors
            var priors = new ModelPriors(1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0);
            System.out.println("✅ Created priors with lambda1=" + priors.lambda1() +
                             ", lambda2=" + priors.lambda2());

            // Test static defaults
            var defaults = ModelPriors.defaults;
            System.out.println("✅ Static defaults work: lambda1=" + defaults.lambda1());

            System.out.println("✅ ModelPriors tests passed!");

        } catch (Exception e) {
            System.out.println("❌ ModelPriors test failed: " + e.getMessage());
        }
    }
}

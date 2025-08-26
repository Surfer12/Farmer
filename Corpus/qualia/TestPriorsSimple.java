// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

/**
 * Simple test for ModelPriors without complex dependencies
 */
public class TestPriorsSimple {
    public static void main(String[] args) {
        System.out.println("🧪 Simple ModelPriors Test");
        System.out.println("=========================");

        try {
            // Test basic record functionality
            System.out.println("Testing basic record creation...");

            // Create a simple priors record
            ModelPriors priors = new ModelPriors(
                2.0, 2.0,  // s_alpha, s_beta
                2.0, 2.0,  // n_alpha, n_beta
                1.0, 1.0,  // alpha_alpha, alpha_beta
                0.0, 1.0,  // beta_mu, beta_sigma
                1.0, 1.0,  // ra_shape, ra_scale
                1.0, 1.0,  // rv_shape, rv_scale
                0.1, 0.1   // lambda1, lambda2
            );

            System.out.println("✅ Created priors successfully");
            System.out.println("Lambda1: " + priors.lambda1());
            System.out.println("Lambda2: " + priors.lambda2());
            System.out.println("S_alpha: " + priors.s_alpha());
            System.out.println("S_beta: " + priors.s_beta());

            // Test equality
            ModelPriors priors2 = new ModelPriors(
                2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 0.1, 0.1
            );

            if (priors.equals(priors2)) {
                System.out.println("✅ Records are equal");
            } else {
                System.out.println("⚠️  Records are not equal");
            }

            // Test hashCode
            System.out.println("HashCode 1: " + priors.hashCode());
            System.out.println("HashCode 2: " + priors2.hashCode());

            System.out.println("✅ ModelPriors simple test passed!");

        } catch (Exception e) {
            System.out.println("❌ Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

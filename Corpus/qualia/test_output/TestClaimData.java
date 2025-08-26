public class TestClaimData {
    public static void main(String[] args) {
        System.out.println("Testing ClaimData...");

        try {
            // Test valid claim data
            var claim = new ClaimData("test_123", true, 0.3, 0.2, 0.8);
            System.out.println("✅ Valid claim: ID=" + claim.id() +
                             ", Verified=" + claim.isVerifiedTrue() +
                             ", Authenticity=" + claim.riskAuthenticity() +
                             ", Virality=" + claim.riskVirality() +
                             ", P(H|E)=" + claim.probabilityHgivenE());

            // Test invalid parameters
            try {
                new ClaimData("test", true, -0.1, 0.5, 0.5); // Invalid authenticity
                System.out.println("❌ Should have thrown exception for invalid authenticity");
            } catch (Exception e) {
                System.out.println("✅ Correctly rejected invalid authenticity: " + e.getMessage());
            }

            System.out.println("✅ ClaimData tests passed!");

        } catch (Exception e) {
            System.out.println("❌ ClaimData test failed: " + e.getMessage());
        }
    }
}

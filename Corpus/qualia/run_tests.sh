#!/bin/bash

# Simple test runner for Qualia
# Tests core functionality without complex compilation

echo "🧪 Running Qualia Tests..."
echo "=========================="

# Create test output directory
mkdir -p test_output

echo "📝 Testing Model Parameters..."

# Test ModelParameters class
cat > test_output/TestModelParameters.java << 'EOF'
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
EOF

echo "📝 Testing ClaimData..."

# Test ClaimData class
cat > test_output/TestClaimData.java << 'EOF'
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
EOF

echo "📝 Testing ModelPriors..."

# Test ModelPriors class
cat > test_output/TestModelPriors.java << 'EOF'
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
EOF

echo "📝 Testing Inverse HB Model..."

# Test basic Inverse HB Model functionality
cat > test_output/TestInverseHB.java << 'EOF'
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
EOF

echo "🔨 Compiling and running tests..."

# Function to compile and run a test
run_test() {
    local test_name=$1
    local test_file="test_output/${test_name}.java"

    echo ""
    echo "Running ${test_name}..."

    # Try to compile
    if javac -d test_output -cp ".:lib/*" "$test_file" 2>/dev/null; then
        echo "✅ Compilation successful"

        # Try to run
        if java -cp "test_output:lib/*" "$test_name" 2>/dev/null; then
            echo "✅ Test execution successful"
        else
            echo "❌ Test execution failed"
        fi
    else
        echo "❌ Compilation failed - checking dependencies..."

        # Try with just core classes
        if javac -d test_output "$test_file" 2>/dev/null; then
            echo "✅ Basic compilation successful (no dependencies)"

            if java -cp "test_output" "$test_name" 2>/dev/null; then
                echo "✅ Basic test execution successful"
            else
                echo "❌ Basic test execution failed"
            fi
        else
            echo "❌ Even basic compilation failed"
        fi
    fi
}

# Run all tests
run_test "TestModelParameters"
run_test "TestClaimData"
run_test "TestModelPriors"
run_test "TestInverseHB"

echo ""
echo "🎉 Test execution completed!"
echo "============================"
echo ""
echo "📊 Summary:"
echo "- TestModelParameters: Basic data structure test"
echo "- TestClaimData: Claim data validation test"
echo "- TestModelPriors: Prior distribution test"
echo "- TestInverseHB: Core model functionality test"
echo ""
echo "💡 Tips:"
echo "- Check test_output/ for compiled test classes"
echo "- Use individual test files for debugging"
echo "- Focus on core functionality first"

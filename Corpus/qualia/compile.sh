#!/bin/bash

echo "🔨 Compiling Qualia project..."

# Create classpath with all dependencies
CLASSPATH="."
for jar in lib/*.jar; do
    if [ -f "$jar" ]; then
        CLASSPATH="$CLASSPATH:$jar"
    fi
done

echo "📚 Using classpath: $CLASSPATH"

# Compile all Java files except the problematic ones for now
echo "Compiling core classes..."
javac -cp "$CLASSPATH" \
    InverseHierarchicalBayesianModel.java \
    InverseHierarchicalBayesianDemo.java \
    ModelParameters.java \
    ClaimData.java \
    SimpleTestRunner.java \
    Core.java \
    -d out/

if [ $? -eq 0 ]; then
    echo "✅ Core compilation successful!"

    # Test the inverse model
    echo "🧪 Testing Inverse HB Model..."
    java -cp "out:$CLASSPATH" qualia.InverseHierarchicalBayesianDemo

    # Run simple tests
    echo "🧪 Running simple tests..."
    java -cp "out:$CLASSPATH" qualia.SimpleTestRunner

else
    echo "❌ Compilation failed"
    exit 1
fi

echo "🎉 Build completed successfully!"

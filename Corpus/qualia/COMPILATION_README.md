# Qualia Compilation Guide

## Quick Start

### Option 1: Simple Compilation (Recommended)
```bash
./compile.sh
```

### Option 2: With Maven (if installed)
```bash
mvn clean compile
mvn exec:java -Dexec.mainClass="qualia.InverseHierarchicalBayesianDemo"
```

### Option 3: With Gradle (if installed)
```bash
gradle build
gradle runJavaDemo
```

## Dependencies

The following dependencies are automatically downloaded to the `lib/` directory:

- **Jackson** (JSON processing)
- **JFreeChart** (Visualization)
- **JUnit 5** (Testing)
- **OkHttp** (HTTP client)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Download dependencies manually
   ./fix_compilation_errors.sh
   ```

2. **JavaFX Issues**
   - JavaFX visualization classes are disabled by default
   - Use console-based versions instead

3. **Java Version Issues**
   - Ensure Java 11 or higher is installed
   - Check with: `java -version`

### Running Specific Components

```bash
# Run Inverse HB Model demo
java -cp "out:lib/*" qualia.InverseHierarchicalBayesianDemo

# Run simple tests
java -cp "out:lib/*" qualia.SimpleTestRunner

# Run core functionality
java -cp "out:lib/*" qualia.Core
```

## Project Structure

```
qualia/
├── Core.java                          # Main application entry
├── InverseHierarchicalBayesianModel.java  # Inverse HB implementation
├── InverseHierarchicalBayesianDemo.java   # HB demo application
├── ModelParameters.java               # Parameter definitions
├── ClaimData.java                     # Data structures
├── SimpleTestRunner.java              # Simple test runner
├── lib/                               # Dependencies
├── out/                               # Compiled classes
├── compile.sh                         # Compilation script
├── fix_compilation_errors.sh          # Error fixing script
└── COMPILATION_README.md              # This file
```

## Build Process

1. **Fix compilation errors**: `./fix_compilation_errors.sh`
2. **Download dependencies**: Script creates `lib/` directory
3. **Compile project**: `./compile.sh`
4. **Run tests**: `java -cp "out:lib/*" qualia.SimpleTestRunner`
5. **Run demos**: `java -cp "out:lib/*" qualia.InverseHierarchicalBayesianDemo`

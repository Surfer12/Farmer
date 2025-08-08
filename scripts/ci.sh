#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Compile all Java sources under Corpus/qualia
mkdir -p out-qualia
javac --release 21 -d out-qualia $(find Corpus/qualia -name '*.java')

# Run tests (exit non-zero on failure)
java -cp out-qualia qualia.PsiMcdaTest
java -cp out-qualia qualia.HmcSmokeTest
java -cp out-qualia qualia.HmcPersistenceTest

echo "CI: All tests passed."



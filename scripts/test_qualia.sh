#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# compile only what's needed for the Psi tests and sinks
javac -cp out-qualia -d out-qualia \
  Corpus/qualia/QualiaException.java \
  Corpus/qualia/ErrorReporter.java \
  Corpus/qualia/MetricsRegistry.java \
  Corpus/qualia/AuditRecord.java \
  Corpus/qualia/ExtendedAuditRecord.java \
  Corpus/qualia/AuditOptions.java \
  Corpus/qualia/AuditSink.java \
  Corpus/qualia/AuditTrail.java \
  Corpus/qualia/ConsoleAuditSink.java \
  Corpus/qualia/FileAuditSink.java \
  Corpus/qualia/HttpAuditSink.java \
  Corpus/qualia/PsiMcda.java \
  Corpus/qualia/PsiMcdaTest.java
java -cp out-qualia qualia.PsiMcdaTest

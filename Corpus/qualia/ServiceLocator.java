// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
package qualia;

import java.util.Objects;

/**
 * Minimal service locator / DI container. For simplicity, only wires a few bindings.
 */
public final class ServiceLocator {
    private final ModelFactory modelFactory;
    private final AuditSink auditSink;

    private ServiceLocator(ModelFactory modelFactory, AuditSink auditSink) {
        this.modelFactory = Objects.requireNonNull(modelFactory);
        this.auditSink = Objects.requireNonNull(auditSink);
    }

    public static Builder builder() { return new Builder(); }

    public PsiModel psiModel(ModelPriors priors, int parallelThreshold) {
        return modelFactory.createPsiModel(priors, parallelThreshold);
    }

    public AuditSink auditSink() { return auditSink; }

    public static final class Builder {
        private ModelFactory modelFactory = ModelFactory.defaultFactory();
        private AuditSink auditSink = AuditSink.noop();

        public Builder modelFactory(ModelFactory f) { this.modelFactory = Objects.requireNonNull(f); return this; }
        public Builder auditSink(AuditSink s) { this.auditSink = Objects.requireNonNull(s); return this; }

        public ServiceLocator build() { return new ServiceLocator(modelFactory, auditSink); }
    }
}



# SPDX-License-Identifier: LicenseRef-Internal-Use-Only
#
# Dev Container Dockerfile for Farmer Project (Internal Use Only)
#
# Includes: Java 21, Python 3.11+, Swift, LaTeX, Node.js, pixi, jq, and build tools
#
# © 2025 Ryan David Oates. Confidential — Internal Use Only

FROM ubuntu:22.04

# Non-interactive for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install core tools and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl wget git make build-essential \
    unzip zip jq \
    python3 python3-pip python3-venv \
    openjdk-21-jdk \
    texlive-full \
    nodejs npm \
    lsb-release \
    ca-certificates \
    gnupg2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Swift (latest stable) — robust download with arch detection and retries
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    case "$ARCH" in \
      amd64)  DIST_DIR="ubuntu2204"; SWIFT_TARBALL="swift-5.9.2-RELEASE-ubuntu22.04.tar.gz" ;; \
      arm64)  DIST_DIR="ubuntu2204-aarch64"; SWIFT_TARBALL="swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz" ;; \
      *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac; \
    SWIFT_URL="https://swift.org/builds/swift-5.9.2-release/${DIST_DIR}/swift-5.9.2-RELEASE/${SWIFT_TARBALL}"; \
    curl -fL --retry 5 --retry-delay 3 --retry-connrefused -o /tmp/swift.tar.gz "$SWIFT_URL"; \
    tar -tzf /tmp/swift.tar.gz >/dev/null; \
    tar -xzf /tmp/swift.tar.gz -C /usr/local --strip-components=1; \
    rm -f /tmp/swift.tar.gz
ENV PATH="/usr/local/bin:$PATH"

# Install pixi (Python env manager)
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Python deps will be installed in postCreate to speed up base image builds

# Set JAVA_HOME for Java 21 (arch-agnostic)
RUN set -eux; \
    JAVAC_PATH="$(command -v javac)"; \
    JAVA_HOME_REAL="$(dirname "$(dirname "$(readlink -f "$JAVAC_PATH")") )"; \
    ln -sfn "$JAVA_HOME_REAL" /usr/lib/jvm/java-21-openjdk
ENV JAVA_HOME="/usr/lib/jvm/java-21-openjdk"

# Node.js (ensure latest LTS)
RUN npm install -g npm@latest

# Set up a non-root user for dev containers
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /workspace
ENV PATH="/home/devuser/.local/bin:$PATH"

# Internal-use only: Do not distribute this image outside the collaboration.

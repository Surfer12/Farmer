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

# Install Swift (latest stable)
RUN wget -qO- https://swift.org/builds/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz | tar xz -C /usr/local --strip-components=1
ENV PATH="/usr/local/bin:$PATH"

# Install pixi (Python env manager)
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Upgrade pip and install Python requirements if present
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    if [ -f /tmp/requirements.txt ]; then pip3 install -r /tmp/requirements.txt; fi

# Set JAVA_HOME for Java 21
ENV JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"

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

# Internal-use only: Do not distribute this image outside the collaboration.

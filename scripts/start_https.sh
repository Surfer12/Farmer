#!/bin/bash
# Start Farmer Assistants with HTTPS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FARMER_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting Farmer Assistants with HTTPS..."
echo "Root directory: $FARMER_ROOT"

# Check if certificates exist
if [ ! -f "$FARMER_ROOT/certs/cert.pem" ] || [ ! -f "$FARMER_ROOT/certs/key.pem" ]; then
    echo "SSL certificates not found. Creating self-signed certificates..."
    python3 "$SCRIPT_DIR/setup_https.py" --mode dev
fi

# Load configuration if it exists (but don't override command line env vars)
if [ -f "$FARMER_ROOT/config/https.env" ]; then
    echo "Loading HTTPS configuration..."
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^#.* ]] && [[ "$line" =~ ^[A-Z_]+=.* ]]; then
            var_name=$(echo "$line" | cut -d'=' -f1)
            var_value=$(echo "$line" | cut -d'=' -f2-)
            # Only set if not already set
            if [ -z "${!var_name}" ]; then
                export "$var_name"="$var_value"
            fi
        fi
    done < "$FARMER_ROOT/config/https.env"
fi

# Set default values if not already set
export USE_HTTPS=${USE_HTTPS:-1}
export HR_PORT=${HR_PORT:-8443}
export IT_PORT=${IT_PORT:-8444}
export REQUIRE_AUTH=${REQUIRE_AUTH:-1}
export AUTH_TOKEN=${AUTH_TOKEN:-dev-local-token}
export LOG_DIR=${LOG_DIR:-data/logs}

echo "Configuration:"
echo "  HTTPS: $USE_HTTPS"
echo "  HR Port: $HR_PORT"
echo "  IT Port: $IT_PORT"
echo "  Auth Required: $REQUIRE_AUTH"

cd "$FARMER_ROOT"
python3 assistants/launcher/launch_all.py

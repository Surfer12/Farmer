#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

# Build-Test-Iterate Loop for Invisible Equipment Optimization
# Automates the fabrication-testing-analysis cycle for equipment development

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_DATA_DIR="${PROJECT_ROOT}/data/blind_tests"
ANALYSIS_DIR="${PROJECT_ROOT}/data/analysis"

# Default parameters
RIDER_ID=""
TEST_NAME=""
CONFIG_FILE=""
SESSIONS_PER_CONFIG=5
SESSION_DURATION=30
ANALYSIS_ONLY=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Build-Test-Iterate Loop for Invisible Equipment Optimization

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    setup           Initialize new test series
    generate-plan   Generate randomized session plan
    run-session     Execute single test session
    analyze         Analyze completed test data
    iterate         Complete build-test-iterate cycle
    status          Show current test status

OPTIONS:
    -r, --rider-id ID       Test rider identifier
    -t, --test-name NAME    Test series name
    -c, --config FILE       Configuration file path
    -s, --sessions N        Sessions per configuration (default: 5)
    -d, --duration N        Session duration in minutes (default: 30)
    -a, --analysis-only     Skip data collection, analyze only
    -v, --verbose           Verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Initialize new fin stiffness test
    $0 -t "fin_stiffness_v2" -r "rider_001" setup

    # Generate session plan
    $0 -t "fin_stiffness_v2" -r "rider_001" generate-plan

    # Run single session
    $0 -t "fin_stiffness_v2" run-session SESSION_ID

    # Analyze completed test
    $0 -t "fin_stiffness_v2" analyze

    # Complete iteration cycle
    $0 -t "fin_stiffness_v2" -r "rider_001" iterate

CONFIGURATION FILE FORMAT:
    JSON file defining test configurations:
    {
        "configurations": [
            {
                "config_id": "baseline",
                "description": "Current production fin",
                "physical_specs": {"stiffness": 100, "material": "fiberglass"}
            }
        ]
    }

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--rider-id)
                RIDER_ID="$2"
                shift 2
                ;;
            -t|--test-name)
                TEST_NAME="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -s|--sessions)
                SESSIONS_PER_CONFIG="$2"
                shift 2
                ;;
            -d|--duration)
                SESSION_DURATION="$2"
                shift 2
                ;;
            -a|--analysis-only)
                ANALYSIS_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                COMMAND="$1"
                shift
                break
                ;;
        esac
    done

    # Store remaining arguments
    ARGS=("$@")
}

# Validate required parameters
validate_params() {
    if [[ -z "$COMMAND" ]]; then
        error "Command required"
        usage
        exit 1
    fi

    case "$COMMAND" in
        setup|generate-plan|iterate)
            if [[ -z "$TEST_NAME" ]]; then
                error "Test name required for $COMMAND"
                exit 1
            fi
            if [[ -z "$RIDER_ID" ]]; then
                error "Rider ID required for $COMMAND"
                exit 1
            fi
            ;;
        run-session)
            if [[ -z "$TEST_NAME" ]]; then
                error "Test name required for $COMMAND"
                exit 1
            fi
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                error "Session ID required for run-session"
                exit 1
            fi
            ;;
        analyze|status)
            if [[ -z "$TEST_NAME" ]]; then
                error "Test name required for $COMMAND"
                exit 1
            fi
            ;;
    esac
}

# Setup new test series
setup_test() {
    log "Setting up new test series: $TEST_NAME"
    
    # Create test directory structure
    TEST_DIR="${TEST_DATA_DIR}/${TEST_NAME}"
    mkdir -p "$TEST_DIR"/{configs,sessions,data,analysis}
    
    # Initialize configuration if provided
    if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
        log "Loading configuration from $CONFIG_FILE"
        cp "$CONFIG_FILE" "$TEST_DIR/configs/test_config.json"
        
        # Initialize test using Python script
        python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
            --test-name "$TEST_NAME" \
            --config-file "$CONFIG_FILE" \
            --action setup
    else
        # Create template configuration
        cat > "$TEST_DIR/configs/test_config.json" << 'EOF'
{
    "configurations": [
        {
            "config_id": "baseline",
            "description": "Current production configuration",
            "physical_specs": {
                "parameter_1": "value_1",
                "parameter_2": "value_2"
            }
        }
    ]
}
EOF
        warn "Created template configuration file: $TEST_DIR/configs/test_config.json"
        warn "Please edit this file before proceeding"
    fi
    
    success "Test series $TEST_NAME initialized"
}

# Generate session plan
generate_session_plan() {
    log "Generating session plan for $RIDER_ID"
    
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --rider-id "$RIDER_ID" \
        --sessions-per-config "$SESSIONS_PER_CONFIG" \
        --session-duration "$SESSION_DURATION" \
        --action generate-plan
    
    success "Session plan generated"
    
    # Display first few sessions
    PLAN_FILE="${TEST_DATA_DIR}/${TEST_NAME}/session_plan_${RIDER_ID}.json"
    if [[ -f "$PLAN_FILE" ]]; then
        log "First few sessions:"
        python3 -c "
import json
with open('$PLAN_FILE') as f:
    plan = json.load(f)
for i, session in enumerate(plan[:5]):
    print(f\"  {session['order']}: {session['test_code']} (duration: {session['duration_minutes']}min)\")
if len(plan) > 5:
    print(f\"  ... and {len(plan)-5} more sessions\")
"
    fi
}

# Run single test session
run_session() {
    SESSION_ID="${ARGS[0]}"
    log "Running test session: $SESSION_ID"
    
    # Get session information
    SESSION_INFO=$(python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --session-id "$SESSION_ID" \
        --action get-session-info)
    
    if [[ $? -ne 0 ]]; then
        error "Failed to get session information"
        exit 1
    fi
    
    # Extract test code for rider
    TEST_CODE=$(echo "$SESSION_INFO" | python3 -c "
import json, sys
info = json.load(sys.stdin)
print(info['test_code'])
")
    
    log "Session $SESSION_ID uses test code: $TEST_CODE"
    
    # Pre-session checklist
    echo
    echo "PRE-SESSION CHECKLIST:"
    echo "1. [ ] Equipment with test code '$TEST_CODE' installed"
    echo "2. [ ] IMU system installed and recording"
    echo "3. [ ] GPS system active (if available)"
    echo "4. [ ] Environmental conditions recorded"
    echo "5. [ ] Rider briefed on blind protocol"
    echo
    
    read -p "All items checked? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Session aborted by user"
        exit 0
    fi
    
    # Start session timer
    START_TIME=$(date +%s)
    log "Session started at $(date)"
    
    # Wait for session completion
    echo "Session in progress..."
    echo "Press ENTER when session is complete"
    read -r
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION_SEC=$((END_TIME - START_TIME))
    DURATION_MIN=$(echo "scale=1; $DURATION_SEC / 60" | bc -l)
    
    log "Session completed. Duration: ${DURATION_MIN} minutes"
    
    # Collect data file paths
    echo
    echo "DATA COLLECTION:"
    read -p "IMU data file path (or ENTER if none): " IMU_FILE
    read -p "GPS data file path (or ENTER if none): " GPS_FILE
    read -p "Session notes: " NOTES
    
    # Complete session in database
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --session-id "$SESSION_ID" \
        --duration "$DURATION_MIN" \
        --imu-file "$IMU_FILE" \
        --gps-file "$GPS_FILE" \
        --notes "$NOTES" \
        --action complete-session
    
    # Collect subjective ratings
    echo
    echo "SUBJECTIVE RATINGS:"
    echo "How often did you notice the equipment? (0=never, 10=constantly)"
    read -p "Invisibility score (0-10): " INVISIBILITY
    
    echo "How automatic did turns feel? (0=required constant attention, 10=completely automatic)"
    read -p "Effortlessness score (0-10): " EFFORTLESSNESS
    
    read -p "Number of times attention went to equipment: " DISRUPTIONS
    
    echo "How confident are you in these ratings? (0=not confident, 10=very confident)"
    read -p "Confidence (0-10): " CONFIDENCE
    
    read -p "Additional notes: " ADDITIONAL_NOTES
    
    # Record subjective ratings
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --session-id "$SESSION_ID" \
        --invisibility "$INVISIBILITY" \
        --effortlessness "$EFFORTLESSNESS" \
        --disruptions "$DISRUPTIONS" \
        --confidence "$CONFIDENCE" \
        --additional-notes "$ADDITIONAL_NOTES" \
        --action record-ratings
    
    success "Session $SESSION_ID completed and recorded"
}

# Analyze test data
analyze_test() {
    log "Analyzing test data for $TEST_NAME"
    
    # Check if all sessions are complete
    INCOMPLETE=$(python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --action check-completeness)
    
    if [[ $? -ne 0 ]]; then
        warn "Some sessions may be incomplete"
    fi
    
    # Run IMU analysis if data files exist
    if [[ "$ANALYSIS_ONLY" == "false" ]]; then
        log "Running IMU flow analysis..."
        
        # Find IMU data files
        DATA_DIR="${TEST_DATA_DIR}/${TEST_NAME}/data"
        if [[ -d "$DATA_DIR" ]]; then
            find "$DATA_DIR" -name "*.csv" -type f | while read -r imu_file; do
                log "Analyzing: $(basename "$imu_file")"
                python3 "${PROJECT_ROOT}/scripts/python/imu_flow_analysis.py" \
                    --input "$imu_file" \
                    --output "${ANALYSIS_DIR}/$(basename "$imu_file" .csv)_analysis.json"
            done
        fi
    fi
    
    # Generate test summary
    log "Generating test summary..."
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --action analyze
    
    # Unblind results
    log "Unblinding results..."
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --action unblind
    
    # Generate recommendation
    RECOMMENDATION=$(python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --action recommend)
    
    echo
    success "Analysis complete"
    echo "$RECOMMENDATION"
    
    # Show results location
    RESULTS_FILE="${TEST_DATA_DIR}/${TEST_NAME}/analysis/unblinded_results.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        log "Detailed results: $RESULTS_FILE"
    fi
}

# Show test status
show_status() {
    log "Test status for $TEST_NAME"
    
    python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
        --test-name "$TEST_NAME" \
        --action status
}

# Complete build-test-iterate cycle
iterate_cycle() {
    log "Starting complete build-test-iterate cycle for $TEST_NAME"
    
    # Check if test exists
    TEST_DIR="${TEST_DATA_DIR}/${TEST_NAME}"
    if [[ ! -d "$TEST_DIR" ]]; then
        log "Test not found, setting up..."
        setup_test
    fi
    
    # Generate plan if not exists
    PLAN_FILE="${TEST_DIR}/session_plan_${RIDER_ID}.json"
    if [[ ! -f "$PLAN_FILE" ]]; then
        log "Session plan not found, generating..."
        generate_session_plan
    fi
    
    # Show remaining sessions
    show_status
    
    echo
    echo "ITERATION CYCLE MENU:"
    echo "1. Run next session"
    echo "2. Analyze current data"
    echo "3. Show status"
    echo "4. Exit"
    echo
    
    while true; do
        read -p "Select option (1-4): " -n 1 -r
        echo
        
        case $REPLY in
            1)
                # Find next session
                NEXT_SESSION=$(python3 "${PROJECT_ROOT}/scripts/python/blind_test_protocol.py" \
                    --test-name "$TEST_NAME" \
                    --action next-session)
                
                if [[ -n "$NEXT_SESSION" ]]; then
                    run_session "$NEXT_SESSION"
                else
                    log "No more sessions to run"
                fi
                ;;
            2)
                analyze_test
                ;;
            3)
                show_status
                ;;
            4)
                log "Exiting iteration cycle"
                break
                ;;
            *)
                error "Invalid option"
                ;;
        esac
        
        echo
        echo "Continue iteration? (y/N)"
        read -p "> " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            break
        fi
    done
}

# Main execution
main() {
    parse_args "$@"
    validate_params
    
    # Ensure required directories exist
    mkdir -p "$TEST_DATA_DIR" "$ANALYSIS_DIR"
    
    # Check Python dependencies
    if ! python3 -c "import numpy, pandas, scipy" 2>/dev/null; then
        error "Required Python packages not found. Please install: numpy pandas scipy"
        exit 1
    fi
    
    # Execute command
    case "$COMMAND" in
        setup)
            setup_test
            ;;
        generate-plan)
            generate_session_plan
            ;;
        run-session)
            run_session
            ;;
        analyze)
            analyze_test
            ;;
        status)
            show_status
            ;;
        iterate)
            iterate_cycle
            ;;
        *)
            error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
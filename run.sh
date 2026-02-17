#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
REQUIREMENTS="requirements.txt"
HOST="127.0.0.1"
PORT=8000

# ---------- helpers ----------

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# ---------- locate Python ----------

find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "$cmd"
            return
        fi
    done
    error "Python not found. Please install Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+."
}

check_python_version() {
    local py="$1"
    local version
    version=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local major minor
    major=${version%%.*}
    minor=${version#*.}
    if [ "$major" -lt "$PYTHON_MIN_MAJOR" ] || { [ "$major" -eq "$PYTHON_MIN_MAJOR" ] && [ "$minor" -lt "$PYTHON_MIN_MINOR" ]; }; then
        error "Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ is required (found $version)."
    fi
    info "Found Python $version ($py)"
}

# ---------- virtual environment ----------

setup_venv() {
    local py="$1"
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment in $VENV_DIR ..."
        "$py" -m venv "$VENV_DIR"
    else
        info "Virtual environment already exists at $VENV_DIR"
    fi
    # Activate
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    info "Virtual environment activated"
}

# ---------- install dependencies ----------

install_deps() {
    if [ ! -f "$REQUIREMENTS" ]; then
        error "$REQUIREMENTS not found."
    fi
    info "Installing dependencies from $REQUIREMENTS ..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "$REQUIREMENTS" --extra-index-url https://download.pytorch.org/whl/cpu
    info "Dependencies installed"
}

# ---------- main ----------

main() {
    info "Starting setup ..."

    local py
    py=$(find_python)
    check_python_version "$py"
    setup_venv "$py"
    install_deps

    info "Launching server on http://$HOST:$PORT ..."
    python main.py
}

# Only run main when executed directly (not when sourced for testing)
if [ -z "${SKIP_MAIN:-}" ]; then
    main "$@"
fi

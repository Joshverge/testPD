#!/usr/bin/env bash
# Run the sprc00 PD smoke test outside Docker. Activates venv, sets
# TT_METAL_HOME / PYTHONPATH, disables the device profiler, runs smoke_test.py.
#
# Usage:
#   ./scripts/run_smoke_test.sh
#   ./scripts/run_smoke_test.sh --num-devices 4
#   ./scripts/run_smoke_test.sh --skip-paged-attn

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv if present.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "[smoke] activated $PROJECT_ROOT/.venv"
else
    echo "[smoke] no .venv at $PROJECT_ROOT/.venv (using current python)"
fi

# Locate tt-metal.
if [ -z "${TT_METAL_HOME:-}" ]; then
    for candidate in \
        "$PROJECT_ROOT/../tt-metal" \
        "$PROJECT_ROOT/deps/tt-metal" \
        "$HOME/tt-metal" \
        "/opt/tenstorrent/tt-metal"; do
        if [ -d "$candidate" ]; then
            export TT_METAL_HOME="$(cd "$candidate" && pwd)"
            echo "[smoke] auto-detected TT_METAL_HOME=$TT_METAL_HOME"
            break
        fi
    done
fi

if [ -z "${TT_METAL_HOME:-}" ]; then
    echo "[smoke] WARNING: TT_METAL_HOME not set; Generator import will fail."
    echo "        Set it manually: export TT_METAL_HOME=/path/to/tt-metal"
else
    case ":${PYTHONPATH:-}:" in
        *":$TT_METAL_HOME:"*) : ;;
        *) export PYTHONPATH="$TT_METAL_HOME:${PYTHONPATH:-}" ;;
    esac
fi

# Profiler off for smoke (issue #30393); re-enable for op-level profiling.
unset TT_METAL_DEVICE_PROFILER
unset TT_METAL_SLOW_DISPATCH_MODE

# Unbuffered output makes paged-attn hangs easier to diagnose.
export PYTHONUNBUFFERED=1

echo "[smoke] python: $(command -v python)"
echo "[smoke] TT_METAL_HOME=${TT_METAL_HOME:-<unset>}"
python "$SCRIPT_DIR/smoke_test.py" "$@"

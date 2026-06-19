#!/bin/bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

# msd.sh - ModelScope Downloader
# ModelScope model downloader (similar to hfd)
# Usage: ./msd.sh <REPO_ID> [options]

set -e

# ===== Color output =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC}   $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[Error]${NC} $1"; exit 1; }

# ===== Help info =====
usage() {
    cat <<EOF
Usage:
  msd <REPO_ID> [--include pattern1 pattern2 ...] [--exclude pattern1 pattern2 ...]
      [--ms_token token] [--local-dir path] [--revision rev] [--dataset]

Description:
  Download models or datasets from ModelScope, similar to hfd.

Arguments:
  REPO_ID         ModelScope repo ID (required)
                  Format: 'org_name/repo_name', e.g. 'unsloth/Qwen3.5-35B-A3B-GGUF'
                  Note: HuggingFace model IDs use the same format on ModelScope

Options:
  --include       (optional) Only download matching files, supports multiple patterns and wildcards
                  e.g.: --include "*.gguf" "*.json"
  --exclude       (optional) Exclude matching files, supports multiple patterns and wildcards
                  e.g.: --exclude "*.safetensors" "*.md"
  --ms_token      (optional) ModelScope access token for private models
  --local-dir     (optional) Download to specified directory, default: ./<repo_name>
  --revision      (optional) Specify model version, default: master
  --dataset       (optional) Download a dataset instead of a model

Example:
  ./msd.sh Qwen/Qwen2.5-7B-Instruct
  ./msd.sh unsloth/Qwen3.5-35B-A3B-GGUF --include "*.gguf"
  ./msd.sh unsloth/Qwen3.5-35B-A3B-GGUF --include "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
  ./msd.sh bigscience/bloom-560m --exclude "*.safetensors"
  ./msd.sh meta-llama/Llama-2-7b --ms_token mytoken
  ./msd.sh lavita/medical-qa --dataset
  ./msd.sh Qwen/Qwen2.5-7B --revision v1.0.0 --local-dir ./mymodel
EOF
    exit 0
}

# ===== Check dependencies =====
check_deps() {
    PYTHON_BIN="${EDGESTUDIO_PYTHON:-$(command -v python3 || command -v python || true)}"
    if [[ -z "$PYTHON_BIN" ]]; then
        print_error "Python not found. Please install Python 3.11+"
    fi
    if ! "$PYTHON_BIN" -c "import modelscope" &>/dev/null; then
        print_error "modelscope not installed. Reinstall EdgeStudio deps: python -m pip install edge-studio"
    fi
    MODELSCOPE_CLI="$(dirname "$PYTHON_BIN")/modelscope"
    if [[ ! -x "$MODELSCOPE_CLI" ]]; then
        MODELSCOPE_CLI="$(command -v modelscope || true)"
    fi
    if [[ -z "$MODELSCOPE_CLI" ]]; then
        print_error "modelscope CLI not found. Reinstall EdgeStudio deps: python -m pip install edge-studio"
    fi
}

# ===== Parse arguments =====
if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
fi

REPO_ID="$1"
shift

INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
MS_TOKEN=""
LOCAL_DIR=""
REVISION="master"
IS_DATASET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                INCLUDE_PATTERNS+=("$1")
                shift
            done
            ;;
        --exclude)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXCLUDE_PATTERNS+=("$1")
                shift
            done
            ;;
        --ms_token)
            MS_TOKEN="$2"; shift 2 ;;
        --local-dir)
            LOCAL_DIR="$2"; shift 2 ;;
        --revision)
            REVISION="$2"; shift 2 ;;
        --dataset)
            IS_DATASET=true; shift ;;
        *)
            print_error "Unknown argument: $1\nUse --help for usage info"
            ;;
    esac
done

# ===== Set default download directory =====
if [[ -z "$LOCAL_DIR" ]]; then
    REPO_NAME="${REPO_ID##*/}"
    LOCAL_DIR="./${REPO_NAME}"
fi

# ===== Print download info =====
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}     msd - ModelScope Downloader        ${NC}"
echo -e "${CYAN}========================================${NC}"
print_info "Repo ID   : $REPO_ID"
print_info "Directory : $LOCAL_DIR"
print_info "Revision  : $REVISION"
[[ "${#INCLUDE_PATTERNS[@]}" -gt 0 ]] && print_info "Include   : ${INCLUDE_PATTERNS[*]}"
[[ "${#EXCLUDE_PATTERNS[@]}" -gt 0 ]] && print_info "Exclude   : ${EXCLUDE_PATTERNS[*]}"
[[ "$IS_DATASET" == true ]]           && print_info "Type      : dataset"
echo ""

# ===== Check dependencies =====
check_deps

# ===== Build download command =====
if [[ "$IS_DATASET" == true ]]; then
    CMD="\"$MODELSCOPE_CLI\" download --dataset \"$REPO_ID\""
else
    CMD="\"$MODELSCOPE_CLI\" download --model \"$REPO_ID\""
fi

CMD="$CMD --local_dir \"$LOCAL_DIR\" --revision \"$REVISION\""

[[ -n "$MS_TOKEN" ]] && CMD="$CMD --token \"$MS_TOKEN\""

# modelscope CLI uses: --include pat1 pat2 pat3 (single flag, multiple values)
# NOT: --include pat1 --include pat2 (each --include overwrites the previous)
if ((${#INCLUDE_PATTERNS[@]})); then
    CMD="$CMD --include"
    for p in "${INCLUDE_PATTERNS[@]}"; do
        CMD="$CMD \"$p\""
    done
fi

if ((${#EXCLUDE_PATTERNS[@]})); then
    CMD="$CMD --exclude"
    for p in "${EXCLUDE_PATTERNS[@]}"; do
        CMD="$CMD \"$p\""
    done
fi

# ===== Execute download =====
print_info "Starting download..."
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Force unbuffered Python output so progress shows in PTY terminal
export PYTHONUNBUFFERED=1
eval $CMD

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "Download complete! Files saved to: $LOCAL_DIR"
    echo ""
    echo -e "${GREEN}File listing:${NC}"
    ls -lh "$LOCAL_DIR" 2>/dev/null || true
else
    print_error "Download failed, exit code: $EXIT_CODE"
fi

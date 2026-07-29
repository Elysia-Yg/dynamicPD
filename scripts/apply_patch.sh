#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SRC_DIR="${PATCH_SRC_DIR:-${SCRIPT_DIR}/patches}"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VLLM_TARGET_DIR="${VLLM_TARGET_DIR:-${VLLM_DIR:-${WORKSPACE_DIR}/vllm}}"
VLLM_ASCEND_TARGET_DIR="${VLLM_ASCEND_TARGET_DIR:-${VLLM_ASCEND_DIR:-${WORKSPACE_DIR}/vllm-ascend}}"

VLLM_PATCH="${PATCH_SRC_DIR}/vllm-0.18.0-dynamicpd.patch"
VLLM_ASCEND_PATCH="${PATCH_SRC_DIR}/vllm-ascend-0.18.0-dynamicpd.patch"

LEGACY_VLLM_PATCHES=(
  "${PATCH_SRC_DIR}/FinishReason.patch"
  "${PATCH_SRC_DIR}/RequestStatus.patch"
  "${PATCH_SRC_DIR}/Get_device_indices.patch"
)

log() {
  echo "[dynamicPD][patch] $*"
}

ensure_dir() {
  local dir="$1"
  local label="$2"
  if [[ ! -d "${dir}" ]]; then
    echo "[ERROR] ${label} does not exist: ${dir}" >&2
    exit 1
  fi
}

apply_one_patch() {
  local target_dir="$1"
  local patch_file="$2"
  local label="$3"

  if [[ ! -f "${patch_file}" ]]; then
    echo "[ERROR] Patch file does not exist: ${patch_file}" >&2
    exit 1
  fi

  log "Checking ${label}: ${patch_file}"
  if git -C "${target_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
    git -C "${target_dir}" apply "${patch_file}"
    log "Applied ${label}"
    return
  fi

  if git -C "${target_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    log "Skipped ${label}; patch is already applied"
    return
  fi

  echo "[ERROR] ${label} cannot be applied cleanly to ${target_dir}" >&2
  echo "        Patch: ${patch_file}" >&2
  echo "        Inspect with: git -C ${target_dir} apply --check ${patch_file}" >&2
  exit 1
}

apply_vllm_patches() {
  ensure_dir "${VLLM_TARGET_DIR}" "VLLM_TARGET_DIR"

  if [[ -f "${VLLM_PATCH}" ]]; then
    apply_one_patch "${VLLM_TARGET_DIR}" "${VLLM_PATCH}" "vLLM 0.18.0 dynamicPD patch"
    return
  fi

  log "Aggregated vLLM patch not found; falling back to legacy patches"
  for patch_file in "${LEGACY_VLLM_PATCHES[@]}"; do
    apply_one_patch "${VLLM_TARGET_DIR}" "${patch_file}" "legacy vLLM patch $(basename "${patch_file}")"
  done
}

apply_vllm_ascend_patches() {
  ensure_dir "${VLLM_ASCEND_TARGET_DIR}" "VLLM_ASCEND_TARGET_DIR"

  if [[ -f "${VLLM_ASCEND_PATCH}" ]]; then
    apply_one_patch "${VLLM_ASCEND_TARGET_DIR}" "${VLLM_ASCEND_PATCH}" "vLLM-Ascend 0.18.0 dynamicPD patch"
  else
    log "No vLLM-Ascend patch found; skipping"
  fi
}

main() {
  log "Patch source: ${PATCH_SRC_DIR}"
  log "vLLM target: ${VLLM_TARGET_DIR}"
  log "vLLM-Ascend target: ${VLLM_ASCEND_TARGET_DIR}"

  apply_vllm_patches
  apply_vllm_ascend_patches

  log "All requested patches are applied"
}

main "$@"

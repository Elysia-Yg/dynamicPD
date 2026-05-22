#!/bin/bash

set -e

PATCH_SRC_DIR="$(pwd)"
TARGET_DIR="/vllm-workspace/vllm"

echo "========== vLLM Patch Apply Script =========="
echo

# 检查目标目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Target directory does not exist:"
    echo "        $TARGET_DIR"
    exit 1
fi

# 检查 patch 文件是否存在
if [ ! -f "$PATCH_SRC_DIR/FinishReason.patch" ]; then
    echo "[ERROR] FinishReason.patch not found"
    exit 1
fi

if [ ! -f "$PATCH_SRC_DIR/RequestStatus.patch" ]; then
    echo "[ERROR] RequestStatus.patch not found"
    exit 1
fi

if [ ! -f "$PATCH_SRC_DIR/Get_device_indices.patch" ]; then
    echo "[ERROR] Get_device_indices.patch not found"
    exit 1
fi

echo "[INFO] Copying patch files..."

cp "$PATCH_SRC_DIR/FinishReason.patch" "$TARGET_DIR/"
cp "$PATCH_SRC_DIR/RequestStatus.patch" "$TARGET_DIR/"
cp "$PATCH_SRC_DIR/Get_device_indices.patch" "$TARGET_DIR/"

echo "[INFO] Patch files copied."
echo

cd "$TARGET_DIR"

echo "[INFO] Applying FinishReason.patch ..."
git apply --check FinishReason.patch
echo "[INFO] FinishReason.patch check passed."
git apply FinishReason.patch

echo "[INFO] FinishReason.patch applied successfully."
echo

echo "[INFO] Applying RequestStatus.patch ..."
git apply --check RequestStatus.patch
git apply RequestStatus.patch

echo "[INFO] RequestStatus.patch applied successfully."
echo

echo "========== ALL PATCHES APPLIED SUCCESSFULLY =========="

echo "[INFO] Applying Get_device_indices.patch ..."
git apply --check Get_device_indices.patch
git apply Get_device_indices.patch

echo "[INFO] Get_device_indices.patch applied successfully."
echo

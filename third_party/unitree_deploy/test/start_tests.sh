#!/bin/bash

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 拼接各个测试脚本的具体路径
ARM_TEST="$SCRIPT_DIR/test/arm/g1/test_g1_wbc_arm.py"
CAMERA_TEST="$SCRIPT_DIR/test/camera/test_image_client_camera.py"
DEX1_TEST="$SCRIPT_DIR/test/endeffector/test_dex1.py"

echo "========================================"
echo "准备依次（串行）执行测试服务..."
echo "========================================"

echo "[1/3] 开始执行 test_g1_wbc_arm.py ..."
python "$ARM_TEST"
if [ $? -ne 0 ]; then
    echo "❌ test_g1_wbc_arm.py 执行失败或被中断，退出后续测试。"
    exit 1
fi

echo "[2/3] 开始执行 test_image_client_camera.py ..."
python "$CAMERA_TEST"
if [ $? -ne 0 ]; then
    echo "❌ test_image_client_camera.py 执行失败或被中断，退出后续测试。"
    exit 1
fi

echo "[3/3] 开始执行 test_dex1.py ..."
python "$DEX1_TEST"
if [ $? -ne 0 ]; then
    echo "❌ test_dex1.py 执行失败或被中断，退出后续测试。"
    exit 1
fi

echo "========================================"
echo "✅ 所有测试均已依次成功执行完毕！"
echo "========================================"

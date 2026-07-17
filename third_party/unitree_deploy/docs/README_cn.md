# Unitree Deploy

本文档提供了为 Unitree G1 和 Z1 平台设置部署环境的说明，包括依赖安装、图像服务启动和夹爪控制。

# 0. 📖 简介

此代码库用于 Unitree 机器人模型的部署。

---

# 1. 🛠️ 环境设置

```bash
conda create -n unitree_deploy python=3.10 && conda activate unitree_deploy

conda install pinocchio -c conda-forge
pip install -e .

# 可选：安装 lerobot 依赖
pip install -e ".[lerobot]"

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e . && cd ..
```

---
# 2. 🚀 启动

**提示：确保所有设备处于同一局域网内**

## 2.1 🤖 运行 G1 和 Dex_1 夹爪

### 2.1.1 📷 图像捕获服务设置（G1 pc2）

[按照以下步骤启动 image_server](https://github.com/unitreerobotics/xr_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-service)
1. 连接到 G1：
  ```bash
  ssh unitree@192.168.123.164  # 密码：123
  ```

2. 激活环境并启动图像服务：
  ```bash
  conda activate tv
  cd ~/image_server
  python image_server.py
  ```

---

### 2.1.2 🤏 Dex_1 夹爪服务设置（开发 PC2）

参考 [Dex_1 夹爪安装指南](https://github.com/unitreerobotics/dex1_1_service?tab=readme-ov-file#1--installation) 获取详细设置说明。

1. 进入服务目录：
  ```bash
  cd ~/dex1_1_service/build
  ```

2. 启动夹爪服务，**ifconfig 检查其自身的 dds 网络接口**：
  ```bash
  sudo ./dex1_1_gripper_server --network eth0 -l -r
  ```

3. 验证与夹爪服务的通信：
  ```bash
  ./test_dex1_1_gripper_server --network eth0 -l -r
  ```

---

### 2.1.3 ✅ 测试

执行以下测试以确保功能正常：

- **Dex1 夹爪测试**：
  ```bash
  python test/endeffector/test_dex1.py
  ```

- **G1 机械臂测试**：
  ```bash
  python test/arm/g1/test_g1_arm.py
  ```

- **图像客户端相机测试**：
  ```bash
  python test/camera/test_image_client_camera.py
  ```

- **G1 数据集回放**：
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., unitree_z1_dual_dex1_realsense, unitree_z1_realsense, unitree_g1_dex1, 

  python test/test_replay.py --repo-id unitreerobotics/G1_CameraPackaging_NewDataset --robot_type unitree_g1_dex1
  ```
---

## 2.2 🦿 运行 Z1

### 2.2.1 🦿 Z1 设置
克隆并构建所需的代码库：

1. 下载 [z1_controller](https://github.com/unitreerobotics/z1_controller.git) 和 [z1_sdk](https://github.com/unitreerobotics/z1_sdk.git)。

2. 构建代码库：
  ```bash
  mkdir build && cd build
  cmake .. && make -j
  ```

3. 复制 `unitree_arm_interface` 库：[根据您的路径修改]
  ```bash
  cp z1_sdk/lib/unitree_arm_interface.cpython-310-x86_64-linux-gnu.so ./unitree_deploy/robot_devices/arm
  ```

4. 启动 Z1 控制器 [根据您的路径修改]：
  ```bash
  cd z1_controller/build && ./z1_ctrl
  ```

---

### 2.2.2 ✅ 测试

运行以下测试：

- **Realsense 相机测试**：
  ```bash
  python test/camera/test_realsense_camera.py # 根据您的 Realsense 修改对应的序列号
  ```

- **Z1 机械臂测试**：
  ```bash
  python test/arm/z1/test_z1_arm.py
  ```

- **Z1 环境测试**：
  ```bash
  python test/arm/z1/test_z1_env.py
  ```

- **Z1 数据集回放**：
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., unitree_z1_dual_dex1_realsense, unitree_z1_realsense, unitree_g1_dex1, 

  python test/test_replay.py --repo-id unitreerobotics/Z1_StackBox_Dataset --robot_type unitree_z1_realsense
  ```
---

## 2.3 🦿 运行 Z1_Dual

### 2.3.1 🦿 Z1 设置和 Dex1 设置
克隆并构建所需的代码库：

1. 按照上述 Z1 步骤下载并编译代码，并下载夹爪程序以本地启动。

2. [根据文档修改多机控制](https://support.unitree.com/home/zh/Z1_developer/sdk_operation)

3. [下载修改后的 z1_sdk_1 并编译](https://github.com/unitreerobotics/z1_sdk/tree/z1_dual)，复制 `unitree_arm_interface` 库：[根据您的路径修改]
  ```bash
  cp z1_sdk/lib/unitree_arm_interface.cpython-310-x86_64-linux-gnu.so ./unitree_deploy/robot_devices/arm
  ```

4. 启动 Z1 控制器 [根据您的路径修改]：
  ```bash
  cd z1_controller/builb && ./z1_ctrl
  cd z1_controller_1/builb && ./z1_ctrl
  ```
5. 启动夹爪服务，**ifconfig 检查其自身的 dds 网络接口**：
  ```
  sudo ./dex1_1_gripper_server --network eth0 -l -r
  ```
---

### 2.3.2 ✅ 测试

运行以下测试：

- **Z1_Dual 机械臂测试**：
  ```bash
  python test/arm/z1/test_z1_arm_dual.py
  ```

- **Z1_Dual 数据集回放**：
  ```bash
  # --repo-id     Your unique repo ID on Hugging Face Hub 
  # --robot_type     The type of the robot e.g., unitree_z1_dual_dex1_realsense, unitree_z1_realsense, unitree_g1_dex1,

  python test/test_replay.py --repo-id unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2 --robot_type unitree_z1_dual_dex1_realsense
  ```
---


# 3.🧠 推理与部署
1. [根据您的配置修改相应参数](./unitree_deploy/robot/robot_configs.py)
2. 返回 [决策模式下的推理与部署](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/README.md) 中的 **客户端设置步骤 2**。

# 4.🏗️ 代码结构

[如果您想添加自己的机器人设备，可以根据此文档进行构建](./docs/GettingStarted.md)

# 5. 🤔 故障排除

如需帮助，请联系项目维护人员或参考相应的 GitHub 仓库文档。📖

# 6. 🙏 致谢

此代码基于以下开源代码库构建。请访问相关 URL 查看相应的 LICENSES（如果您觉得这些项目有价值，请为它们点亮星星）：

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python

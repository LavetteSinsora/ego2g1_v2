import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from unitree_deploy.robot_devices.cameras.configs import ImageClientCameraConfig
from unitree_deploy.robot_devices.cameras.utils import make_cameras_from_configs
from unitree_deploy.utils.rich_logger import log_success


# ============================From configs============================
def run_camera():
    def image_client_default_factory():
        return {
            "imageclient": ImageClientCameraConfig(),
        }

    # ===========================================

    cameras = make_cameras_from_configs(image_client_default_factory())
    print(cameras)
    for name in cameras:
        cameras[name].connect()
        log_success(f"Connecting {name} cameras.")

    for _ in tqdm(range(20), desc="Camera warming up"):
        for name in cameras:
            cameras[name].async_read()
        time.sleep(1 / 30)

    while True:
        images = {}
        for name in cameras:
            output = cameras[name].async_read()
            if isinstance(output, dict):
                for k, v in output.items():
                    if v is not None:
                        images[k] = torch.from_numpy(v)
                    else:
                        print("Drop Frame.")
            else:
                if output is not None:
                    images[name] = torch.from_numpy(output)
                else:
                    print("Drop Frame.")

        image_list = [np.stack([img.numpy()] * 3, axis=-1) if img.ndim == 2 else img.numpy() for img in images.values()]

        stacked_image = np.hstack(image_list)
        cv2.imshow("Stacked Image", stacked_image)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    run_camera()

import numpy as np

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, idx, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 5, 6, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Input image width is larger than 1600px, auto-resizing to 1600. "
                        "Use '--resolution 1' to disable."
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image[:3, ...]
    loaded_mask = resized_image[3:4, ...] if resized_image.shape[0] == 4 else None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=idx,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    return [loadCam(args, idx, cam_info, resolution_scale) for idx, cam_info in enumerate(cam_infos)]


def camera_to_JSON(idx, camera):
    fovy = getattr(camera, "FoVy", getattr(camera, "FovY"))
    fovx = getattr(camera, "FoVx", getattr(camera, "FovX"))
    width = getattr(camera, "width")
    height = getattr(camera, "height")

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_rot = [row.tolist() for row in rot]
    return {
        "id": idx,
        "img_name": camera.image_name,
        "width": width,
        "height": height,
        "position": pos.tolist(),
        "rotation": serializable_rot,
        "fy": fov2focal(fovy, height),
        "fx": fov2focal(fovx, width),
    }

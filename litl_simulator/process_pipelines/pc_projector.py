"""Point cloud projector module."""
import numpy as np

from ..bases import BaseUtility


class PointCloudProjector(BaseUtility):
    """Computes projection of a point cloud onto an image."""

    def __init__(self, pc, lidar_matrix, camera_matrix, image_width, image_height, hitmask=None, **kwargs):
        """Point cloud projector init method."""
        super().__init__(**kwargs)
        if pc.shape[-1] > 8:  # XYZ COS RGBA
            raise ValueError("Wrong PC shape...")
        self.pc = pc  # cartesian coordinates of shape nfeats x npts
        self.lidar_matrix = lidar_matrix
        self.image_width = image_width
        self.image_height = image_height
        self.camera_matrix = camera_matrix
        if hitmask is None:
            # Mask when A channel is 0
            self.hitmask = pc[..., -1] != 0
        else:
            self.hitmask = hitmask
        self.inside_mask_flatten = None  # will be set once uv projection computed
        self._logger.debug(f"Initialized with image width={image_width}, height={image_height}.")

    def compute_uv_projection(self):
        """Computes the uv coordinates of the point cloud in the image."""
        self._logger.debug("Computing UV projection.")
        # PC was already inverted y axis -> revert back to match transformation matrices
        npts = np.prod(self.pc.shape[:-1])
        pc_flat = self.pc.reshape(npts, self.pc.shape[-1])  # preserve last axis
        hitmask_flat = self.hitmask.flatten()  # makes a copy
        pc = pc_flat[hitmask_flat, :3].T  # (XYZ x npts)
        pc[1, :] *= -1
        # for some reason frame of reference is rotated by 90 degrees
        # (possibly because of change in angles when processing)
        pc[[0, 1], :] = pc[[1, 0], :]
        pc[0, :] *= -1
        # promote to 4 coordinates where last axis is 1 in order to multiply with transform matrix
        local_pts = np.r_[pc, [np.ones(pc.shape[1])]]  # (XYZ1 x npts)
        world_pts = self.lidar_matrix.dot(local_pts)
        cam_pts = self.camera_matrix.dot(world_pts)[:3, :]  # pts in camera reference frame
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        # apply transform
        pc = np.array([cam_pts[0], -cam_pts[2], cam_pts[1]])
        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        matrixK = np.identity(3)
        # focal = w / (2 tan((fov * pi)/360))  and fov = 90 deg here => tan(pi/4) = 1
        matrixK[0, 0] = matrixK[1, 1] = self.image_width / 2
        matrixK[0, 2] = self.image_width / 2
        matrixK[1, 2] = self.image_height / 2

        # project pts in 2D camera planew
        pc2D = np.dot(matrixK, pc)  # 3 x npts
        # normalize x and y by z
        pc2D[0, :] = np.divide(pc2D[0, :], pc2D[2, :])
        pc2D[1, :] = np.divide(pc2D[1, :], pc2D[2, :])
        # discard pts outside camera FOV and behind it
        inside_mask = (
                (pc2D[0, :] > 0) & (pc2D[0, :] < self.image_width) &
                (pc2D[1, :] > 0) & (pc2D[1, :] < self.image_height) &
                (pc2D[2, :] > 0)
                )  # size = npts, array of booleans
        # inside mask here only applies to pts where hitmask is True
        # translate to full pc array
        final_mask = np.copy(hitmask_flat)
        final_mask[hitmask_flat] = inside_mask
        self.inside_mask = final_mask.reshape(self.hitmask.shape)
        # return UV coordinates
        uvcoords = pc2D[:2, inside_mask].astype(int).T  # npts x 2
        # reverse u coordinates
        w_1 = self.image_width - 1
        uvcoords[:, 0] = w_1 - uvcoords[:, 0]
        # not sure why but u and v are inverted
        return uvcoords[:, [1, 0]]  # invert to get actual image coordinates

    def compute_image_projection(
            self, image=None, radius=5, colors=None):
        """Computes point cloud projection on original image.

        This method changes the initial image with a new one containing the pt cloud projection.

        Args:
            image: np.ndarray
                The image to project the points on.
            radius: int
                The radius in pixels of the projected points.
            colors: np.ndarray
                The color arrays. All axis except for the last one must match
                the ones from the pt cloud (except its last as well).
        """
        if image is None:
            raise ValueError("Need to give image.")
        if colors is None:
            raise ValueError("Need to give colors.")
        # assume colors have same shape (except last axis) than PC
        uvcoords = self.compute_uv_projection()
        # only retain colors ones inside the mask
        colors = colors[self.inside_mask]
        if colors.dtype != np.uint8:
            # convert to uint 8 bits
            colors = np.round(255 * colors).astype(np.uint8)
        # colors = self.pc[hitmask][inside_mask, -4:]
        # this is the slowest part of the code
        minus = np.clip(uvcoords[:, 0] - radius, 0, None)
        maxus = np.clip(uvcoords[:, 0] + radius + 1, None, self.image_height - 1)
        minvs = np.clip(uvcoords[:, 1] - radius, 0, None)
        maxvs = np.clip(uvcoords[:, 1] + radius + 1, None, self.image_width - 1)
        for minu, maxu, minv, maxv, color in zip(minus, maxus, minvs, maxvs, colors):
            image[minu:maxu, minv:maxv] = color
        return image

import numpy as np
import scipy.interpolate
import warnings


class SphericalInterpolator:
    def __init__(self, start: np.ndarray, end: np.ndarray, sin_omega_eps=1e-7):
        """
        Class to perform spherical interpolation between two non-normalized (non-quaternion...) vectors.
        The interpolation ensures a constant angular speed (usual SLERP).
        Vector length will be linearly interpolated.

        Extrapolation is allowed.

        Scipy.interpolate-like interpolator, which accepts only a start and an end points.

        Interpolation call will return a matrix:
        """
        assert len(start.shape) == 1 and len(end.shape) == 1, "Please provide 1D start and end vectors."
        self.start, self.end = start, end
        self.start_norm, self.end_norm = np.linalg.norm(start, 2), np.linalg.norm(end, 2)
        self.unit_start, self.unit_end = start / self.start_norm, end / self.end_norm
        self.omega = np.arccos(np.dot(self.unit_start, self.unit_end))  # TODO check numerical stability? <-1, >+1 ?
        self.sin_omega = np.sin(self.omega)
        if self.sin_omega < sin_omega_eps:
            warnings.warn("Start and End vectors are almost collinear (sin(Omega) = {}). "
                          "Interpolation can be numerically unstable.".format(self.sin_omega))
        self.length_interpolate = scipy.interpolate.interp1d(
            [0.0, 1.0], [self.start_norm, self.end_norm], kind='linear', axis=0,
            bounds_error=False, fill_value="extrapolate"
        )

    def __call__(self, *args, **kwargs):
        assert len(args) == 1, "only interpolation 'time' steps can be provided at input"
        t = np.asarray(args[0])
        assert len(kwargs) == 0, "kwargs not supported"

        assert len(t.shape) == 1, "Input array must be 1D"
        tau1 = np.sin((1.0 - t) * self.omega)
        tau2 = np.sin(t * self.omega)
        interp_vectors = (np.outer(tau1, self.unit_start) + np.outer(tau2, self.unit_end)) / self.sin_omega
        # Length is simply applied in a for loop instead of using a (sparse) diagonal matrix
        norm_interp = self.length_interpolate(t)
        for i, norm in enumerate(norm_interp):
            interp_vectors[i, :] = interp_vectors[i, :] * norm
        return interp_vectors


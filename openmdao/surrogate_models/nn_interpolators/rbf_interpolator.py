"""Define the RBFInterpolator class."""

import numpy as np

from openmdao.surrogate_models.nn_interpolators.nn_base import NNBase
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class RBFInterpolator(NNBase):
    """
    Compactly Supported Radial Basis Function.

    Parameters
    ----------
    training_points : ndarray
        Ndarray of shape (num_points x independent dims) containing training input locations.
    training_values : ndarray
        Ndarray of shape (num_points x dependent dims) containing training output values.
    num_leaves : int
        How many leaves the tree should have.
    num_neighbors : int
        The number of neighbors to use for interpolation.
    rbf_family : int
        Specifies the order of the radial basis function to be used.
        <-2> uses an 11th order, <-1> uses a 9th order, and any value from <0> to <4> uses an
        order equal to <floor((dimensions-1)/2) + (3*comp) +1>.

    Attributes
    ----------
    rbf_family : int
        Specifies the order of the radial basis function to be used.
        <-2> uses an 11th order, <-1> uses a 9th order, and any value from <0> to <4> uses an order
        equal to <floor((dimensions-1)/2) + (3*comp) +1>.
    N : int
        The number of neighbors used for interpolation.
    weights : ndarray
        Weights for each interpolation point.
    """

    def __init__(
        self, training_points, training_values, num_leaves=2, num_neighbors=5, rbf_family=2
    ):
        """
        Initialize all attributes.
        """
        super().__init__(training_points, training_values, num_leaves)

        if self._ntpts < num_neighbors:
            self._raise(
                "RBFInterpolator only given {0} training points, "
                "but requested num_neighbors={1}.".format(self._ntpts, num_neighbors),
                exc_type=ValueError,
            )

        # rbf_family is an arbitrary value that picks a function to use
        self.rbf_family = rbf_family

        # For weights, first find the training points radial neighbors
        tdist, tloc = self._KData.query(self._tp, num_neighbors)
        Tt = tdist[:, :-1] / tdist[:, -1:]
        # Next determine weight matrix
        Rt = self._find_R(self._ntpts, Tt, tloc)
        weights = (spsolve(csc_matrix(Rt), self._tv))[..., np.newaxis]

        self.N = num_neighbors
        self.weights = weights

    def _find_R(self, npp, T, neighbor_idx):
        """
        Evaluate RBF polynomial.

        Parameters
        ----------
        npp : int
            Number of prediction points
        T : int
            Radial distance
        neighbor_idx : ndarray
            Int array of neighbor indices.

        Returns
        -------
        ndarray
            Evaluation of RBF polynomial.
        """
        R = np.zeros((npp, self._ntpts), dtype="float")

        # Choose type of CRBF R matrix
        if self.rbf_family == -1:
            # Comp #1 - a
            Cf = np.power((1.0 - T), 5)
            cb_poly = [5.0, 72.0, 48.0, 40.0, 8.0]
            # Cb = (8. + (40. * T) + (48. * T * T) +
            #       (72. * T * T * T) + (5. * T * T * T * T))
        elif self.rbf_family == -2:
            # Comp #2
            Cf = np.power((1.0 - T), 6)
            cb_poly = [5.0, 30.0, 72.0, 82.0, 36.0, 6.0]
            # Cb = (6. + (36. * T) + (82. * T * T) + (72. * T * T * T) +
            #       (30. * T * T * T * T) + (5. * T * T * T * T * T))
        elif self.rbf_family == -3:
            # Comp #3
            # Cf = np.ones_like(T)
            # Cb = np.sqrt((T * T) + 1.)

            # Re-arranged to fit polyval scheme below.
            Cf = np.sqrt(np.square(T) + 1.0)
            cb_poly = [1.0]
        else:
            # The above options did not specify a dimensional requirement
            # in the paper found but the rest are said to only be guaranteed
            # as positive definite iff the dimensional requirements are.
            # Because of this, the user can select 0 through 4 to adjust to a
            # level of order trying to be attained.
            dims = self._indep_dims + 1
            if dims <= 2:
                if self.rbf_family == 0:
                    # This starts the dk comps, here d=1, k=0
                    Cf = 1.0 - T
                    cb_poly = [1.0]
                    # Cb = T ** 0.
                elif self.rbf_family == 1:
                    Cf = np.power(1.0 - T, 3.0) / 12.0
                    cb_poly = [3.0, 1.0]
                    # Cb = 1. + (3. * T)
                elif self.rbf_family == 2:
                    Cf = np.power(1.0 - T, 5.0) / 840.0
                    cb_poly = [24.0, 15.0, 3.0]
                    # Cb = 3. + (15. * T) + (24. * T * T)
                elif self.rbf_family == 3:
                    Cf = np.power(1.0 - T, 7.0) / 151200.0
                    cb_poly = [315.0, 285.0, 105.0, 15.0]
                    # Cb = 15. + (105. * T) + (285. * T * T) + (315. * T * T * T)
                elif self.rbf_family == 4:
                    Cf = np.power(1.0 - T, 9.0) / 51891840.0
                    cb_poly = [5760.0, 6795.0, 3555.0, 945.0, 105.0]
                    # Cb = (105. + (945. * T) + (3555. * T * T) + (6795. * T * T * T) +
                    #       (5760. * T * T * T * T))
            elif dims <= 4:
                if self.rbf_family == 0:
                    Cf = 1.0 - T
                    # Cb = (1. - T)
                    cb_poly = [-1.0, 1.0]
                elif self.rbf_family == 1:
                    Cf = np.power(1.0 - T, 4) / 20.0
                    # Cb = 1. + (4. * T)
                    cb_poly = [4.0, 1.0]
                elif self.rbf_family == 2:
                    Cf = np.power(1.0 - T, 6.0) / 1680.0
                    # Cb = 3. + (18. * T) + (35. * T * T)
                    cb_poly = [35.0, 18.0, 3.0]
                elif self.rbf_family == 3:
                    Cf = np.power(1.0 - T, 8.0) / 332640.0
                    cb_poly = [480.0, 375.0, 120.0, 15.0]
                    # Cb = 15. + (120. * T) + (375. * T * T) + (480. * T * T * T)
                elif self.rbf_family == 4:
                    Cf = np.power(1.0 - T, 10.0) / 121080960.0
                    cb_poly = [9009.0, 9450.0, 4410.0, 1050.0, 105.0]
                    # Cb = (105. + (1050. * T) + (4410. * T * T) + (9450. * T * T * T) +
                    #       (9009. * T * T * T * T))
            elif dims <= 6:
                if self.rbf_family == 0:
                    Cf = np.power(1.0 - T, 2.0)
                    cb_poly = [-1.0, 1.0]
                    # Cb = (1. - T)
                elif self.rbf_family == 1:
                    Cf = np.power(1.0 - T, 5.0) / 30.0
                    cb_poly = [5.0, 1.0]
                    # Cb = 1. + (5. * T)
                elif self.rbf_family == 2:
                    Cf = np.power(1.0 - T, 7.0) / 3024.0
                    cb_poly = [48.0, 21.0, 3.0]
                    # Cb = 3. + (21. * T) + (48. * T * T)
                elif self.rbf_family == 3:
                    Cf = np.power(1.0 - T, 9.0) / 665280.0
                    cb_poly = [693.0, 477.0, 135.0, 15.0]
                    # Cb = 15. + (135. * T) + (477. * T * T) + (693. * T * T * T)
                elif self.rbf_family == 4:
                    Cf = np.power(1.0 - T, 11.0) / 259459200.0
                    cb_poly = [13440.0, 12705.0, 5355.0, 1155.0, 105.0]
                    # Cb = (105. + (1155. * T) + (5355. * T * T) + (12705. * T * T * T) +
                    #       (13440. * T * T * T * T))
            else:
                # Although not listed, this is ideally for 8 dim or less
                if self.rbf_family == 0:
                    Cf = np.power(1.0 - T, 2.0)
                    cb_poly = [1.0, -2.0, 1.0]
                    # Cb = (1. - T) * (1. - T)
                elif self.rbf_family == 1:
                    Cf = np.power(1.0 - T, 6.0) / 42.0
                    cb_poly = [6.0, 1.0]
                    # Cb = 1. + (6. * T)
                elif self.rbf_family == 2:
                    Cf = np.power(1.0 - T, 8.0) / 5040.0
                    cb_poly = [63.0, 24.0, 3.0]
                    # Cb = 3. + (24. * T) + (63. * T * T)
                elif self.rbf_family == 3:
                    Cf = np.power(1.0 - T, 10.0) / 1235520.0
                    cb_poly = [960.0, 591.0, 150.0, 15.0]
                    # Cb = 15. + (150. * T) + (591. * T * T) + (960. * T * T * T)
                elif self.rbf_family == 4:
                    Cf = np.power(1.0 - T, 12.0) / 518918400.0
                    cb_poly = [19305.0, 16620.0, 6390.0, 1260.0, 105.0]
                    # Cb = (105. + (1260. * T) + (6390. * T * T) + (16620. * T * T * T) +
                    #       (19305. * T * T * T * T))

        Cb = np.polyval(cb_poly, T)

        for i in range(npp):
            R[i, neighbor_idx[i, :-1]] = Cf[i, :] * Cb[i, :]

        return R

    def _find_dR(self, prediction_points, neighbor_idx, neighbor_dists):
        """
        Find dR.

        Parameters
        ----------
        prediction_points : ndarray
            Array of prediction locations.
        neighbor_idx : ndarray
            Nearest neighbor indices for prediction points.
        neighbor_dists : ndarray
            Distances from prediction points to neighbors.

        Returns
        -------
        ndarray
            Gradient value at prediction points.
        """
        T = neighbor_dists[:, :-1] / neighbor_dists[:, -1:]
        # Solve for the gradient analytically
        # The first quantity needed is dRp/dt
        if self.rbf_family == -1:
            frnt = np.power((1.0 - T), 4)
            dRp_poly = [-45.0, -556.0, -120.0, -144.0, 0.0]
            # dRp = frnt * ((-5. * (8. + (40. * T) + (48. * T * T) +
            #                       (72. * T * T * T) + (5. * T * T * T * T))) +
            #               ((1. - T) * (40. + (96. * T) +
            #                            (216. * T * T) + (20. * T * T * T))))
        elif self.rbf_family == -2:
            frnt = np.power((1.0 - T), 5.0)
            dRp_poly = [-55.0, -275.0, -528.0, -440.0, -88.0, 0.0]
            # dRp = frnt * ((-6. * (6. + (36. * T) +
            #                       (82. * T * T) + (72. * T * T * T) +
            #                       (30. * T * T * T * T) + (5. * T * T * T * T * T))) +
            #               ((1. - T)) * (36. + (164. * T) +
            #                             (216. * T * T) + (120. * T * T * T) +
            #                             (25. * T * T * T * T)))
        elif self.rbf_family == -3:
            frnt = T / np.sqrt((T * T) * 1.0)
            dRp_poly = [1.0]
        else:
            dims = self._indep_dims + 1
            # Start dim dependent comps, review first occurrence for more info
            if dims <= 2:
                if self.rbf_family == 0:
                    # This starts the dk comps(Wendland Functs), here d=1, k=0
                    frnt = 1.0
                    dRp_poly = [-1.0]
                    # dRp = -1.
                elif self.rbf_family == 1:
                    frnt = 1.0
                    dRp_poly = [1.0, -2.0, 1.0, 0.0]
                    # dRp = -T * (1. - T) * (1. - T)
                elif self.rbf_family == 2:
                    frnt = np.power(1.0 - T, 4.0) / -20.0
                    dRp_poly = [4.0, 1.0, 0.0]
                    # dRp = frnt * (T + (4. * T * T))
                elif self.rbf_family == 3:
                    frnt = np.power(1.0 - T, 6.0) / -1680.0
                    dRp_poly = [35.0, 18.0, 3.0, 0.0]
                    # dRp = frnt * ((3. * T) + (18. * T * T) + (35. * T * T * T))
                elif self.rbf_family == 4:
                    frnt = np.power(1.0 - T, 8.0) / -22176.0
                    dRp_poly = [32.0, 25.0, 8.0, 1.0, 0.0]
                    # dRp = frnt * (T + (8. * T * T) + (25. * T * T * T) + (32. * T * T * T * T))
            elif dims <= 4:
                if self.rbf_family == 0:
                    frnt = 1.0
                    dRp_poly = [2.0, -2.0]
                    # dRp = -2. * (1 - T)
                elif self.rbf_family == 1:
                    frnt = 1.0
                    dRp_poly = [1.0, -3.0, 3.0, -1.0, 0.0]
                    # dRp = -T * (1. - T) * (1. - T) * (1. - T)
                elif self.rbf_family == 2:
                    frnt = np.power(1.0 - T, 5.0) / -30.0
                    dRp_poly = [5.0, 1.0, 0.0]
                    # dRp = frnt * (T + (5. * T * T))
                elif self.rbf_family == 3:
                    frnt = np.power(1.0 - T, 7.0) / -1008.0
                    dRp_poly = [16.0, 7.0, 1.0, 0.0]
                    # dRp = frnt * (T + (7. * T * T) + (16. * T * T * T))
                elif self.rbf_family == 4:
                    frnt = np.power(1.0 - T, 9.0) / -221760.0
                    dRp_poly = [231.0, 159.0, 45.0, 5.0, 0.0]
                    # dRp = frnt * ((5. * T) + (45. * T * T) + (159. * T * T * T)
                    #                        + (231. * T * T * T * T))
            elif dims <= 6:
                if self.rbf_family == 0:
                    frnt = 1.0
                    dRp_poly = [-3.0, 6.0, -3.0]
                    # dRp = -3. * (1. - T) * (1. - T)
                elif self.rbf_family == 1:
                    frnt = 1.0
                    dRp_poly = [-1.0, 4.0, -6.0, 4.0, -1.0, 0.0]
                    # dRp = -T * ((1. - T) ** 4)
                elif self.rbf_family == 2:
                    frnt = np.power(1.0 - T, 6.0) / -42.0
                    dRp_poly = [6.0, 1.0, 0.0]
                    # dRp = frnt * (T + (6. * T * T))
                elif self.rbf_family == 3:
                    frnt = np.power(1.0 - T, 8) / -1680.0
                    dRp_poly = [21.0, 8.0, 1.0, 0.0]
                    # dRp = frnt * (T + (8. * T * T) + (21. * T * T * T))
                elif self.rbf_family == 4:
                    frnt = np.power(1.0 - T, 10.0) / -411840.0
                    dRp_poly = [320.0, 197.0, 50.0, 5.0, 0.0]
                    # dRp = frnt * ((5. * T) + (50. * T * T) + (197. * T * T * T)
                    #                        + (320. * T * T * T * T))
            else:
                # Although not listed, this is ideally for 8 dim or less
                if self.rbf_family == 0:
                    frnt = 1.0
                    dRp_poly = [4.0, -12.0, 12.0, -4.0]
                    # dRp = -4. * (1. - T) * (1. - T) * (1. - T)
                elif self.rbf_family == 1:
                    frnt = 1.0
                    dRp_poly = [1.0, -5.0, 10.0, -10.0, 5.0, -1.0, 0.0]
                    # dRp = -T * ((1. - T) ** 5)
                elif self.rbf_family == 2:
                    frnt = np.power(1.0 - T, 7.0) / -56.0
                    dRp_poly = [7.0, 1.0, 0.0]
                    # dRp = frnt * (T + (7. * T * T))
                elif self.rbf_family == 3:
                    frnt = np.power(1.0 - T, 9.0) / -7920.0
                    dRp_poly = [80.0, 27.0, 3.0, 0.0]
                    # dRp = frnt * ((3. * T) + (27. * T * T) + (80. * T * T * T))
                elif self.rbf_family == 4:
                    frnt = np.power(1.0 - T, 11.0) / -720720.0
                    dRp_poly = [429.0, 239.0, 55.0, 5.0, 0.0]
                    # dRp = frnt * ((5. * T) + (55. * T * T) + (239. * T * T * T)
                    #                        + (429. * T * T * T * T))

        dRp = frnt * np.polyval(dRp_poly, T)

        # dt/dx becomes unstable at the training points, so perturb T slightly.
        zero_idx = np.where(T == 0.0)
        T[zero_idx] += 1.0e-11

        # Now need dt/dx
        xpi = np.subtract(prediction_points, self._tp[neighbor_idx[:, :-1], :])
        xpm = prediction_points - self._tp[neighbor_idx[:, -1:], :]
        dtx = (xpi - (np.square(T) * xpm)) / (np.square(neighbor_dists[:, -1:, :]) * T)

        # The gradient then is the summation across neighs of w*df/dt*dt/dx
        grad = np.einsum("ijk,ijk,ijl...->ilk...", dRp, dtx, self.weights[neighbor_idx[:, :-1]])

        return grad.reshape((prediction_points.shape[0], self._dep_dims, self._indep_dims))

    def __call__(self, prediction_points):
        """
        Interpolate at the requested points.

        Parameters
        ----------
        prediction_points : ndarray
            Points at which interpolation is done.

        Returns
        -------
        predz
            Need a type for predz.
        """
        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self._tpm) / self._tpr
        nppts = normalized_pts.shape[0]
        # Setup prediction points and find their radial neighbors
        ndist, nloc = self._KData.query(normalized_pts, self.N)
        # Check if complex step is being run
        if np.any(np.abs(normalized_pts[0, :].imag)) > 0:
            dimdiff = np.subtract(
                normalized_pts.reshape((nppts, 1, self._indep_dims)), self._tp[nloc, :]
            )
            # KD Tree ignores imaginary part, muse redo ndist if complex
            ndist = np.sqrt(np.sum((dimdiff * dimdiff), axis=2))

        # Take farthest distance of each point
        Tp = ndist[:, :-1] / ndist[:, -1:]

        Rp = self._find_R(nppts, Tp, nloc)
        predz = ((np.dot(Rp, self.weights[..., 0]) * self._tvr) + self._tvm).reshape(
            nppts, self._dep_dims
        )

        self._pt_cache = (normalized_pts, ndist, nloc)

        return predz

    def gradient(self, prediction_points):
        """
        Find the gradient at each location of a set of supplied predicted points.

        Parameters
        ----------
        prediction_points : ndarray
            Points at which interpolation is done.

        Returns
        -------
        ndarray
            Gradient value at prediction points.
        """
        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self._tpm) / self._tpr
        # Setup prediction points and find their radial neighbors
        if self._pt_cache is not None and np.allclose(self._pt_cache[0], normalized_pts):
            pdist, ploc = self._pt_cache[1:]
        else:
            pdist, ploc = self._KData.query(normalized_pts, self.N)

        # Find Gradient
        grad = self._find_dR(normalized_pts[:, np.newaxis, :], ploc, pdist[:, :, np.newaxis]) * (
            self._tvr[..., np.newaxis] / self._tpr
        )

        return grad

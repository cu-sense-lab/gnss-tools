from typing import Tuple, Optional
import numpy as np
import numba as nb


@nb.jit
def compute_least_squares_position_and_clock(
    corrected_pseudoranges: np.ndarray,
    tx_positions: np.ndarray,
    initial_position: Optional[np.ndarray] = None,
    initial_clock_bias: float = 0.0,
    num_iter: int = 100,
    convergence_threshold: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Given the corrected pseudorange measurements (i.e. pseudorange measurements corrected for
    all effects other than transmitter-receiver range and receiver clock bias), computes the position
    and clock bias of the receiver using an iterative least squares method.  (Newton's method)

    Input:
        `corrected_pseudoranges` -- the array of shape (M,) containing transmitter range + rx clock bias
            measurements
        `tx_positions` -- array of shape (M, 3) containing corrected transmitter positions at the
            time of transmission corresponding to the measurments in `corrected_pseudoranges`
        `initial_position` -- optional (default=None) initial guess for receiver position to speed up convergence.
            If None, algorithm just uses (0, 0, 0) as initial guess.
        `initial_clock_bias` -- optional (default=0.0), initial guess for receiver clock bias
        `num_iter` -- optional (default=100), specifies the number of iterations to perform before
            giving up.
        `convergence_threshold` -- optional (default=1e-6), specifies the threshold for convergence
            of the position and clock component estimates (in meters)

    Output:
        tuple (`x_hat`, `b_hat`, `debug_dict`) where `x_hat` is estimated x-y-z position vector of shape (3,)
        and `b_hat` is the receiver clock bias in meters (i.e. multiplied by speed of light).  `debug_dict`
        is a dictionary containing helpful information, such as the residuals after each iteration.
    """
    M: int = len(corrected_pseudoranges)
    assert M == len(tx_positions)

    # initial estimates for receiver state
    if initial_position is not None:
        x_hat = initial_position.copy()
    else:
        x_hat = np.zeros(3)
    b_hat = initial_clock_bias
    
    dx = np.ones((1, 4))  # position/clock error
    count = 0

    # Preallocate geometry matrix
    G = np.zeros((M, 4))
    G[:, 3] = 1.0

    for i in range(num_iter):

        # Compute geometry matrix
        tx_rx_pos_delta = tx_positions - x_hat
        # ranges = np.linalg.norm(tx_rx_pos_delta, axis=1)
        ranges = np.sqrt(np.sum(tx_rx_pos_delta * tx_rx_pos_delta, axis=1))
        G[:, 0:3] = -tx_rx_pos_delta / ranges[:, None]
        G[:, 3] = 1.0

        # Compute residuals
        dy = corrected_pseudoranges - ranges - b_hat

        try:
            Q, R = np.linalg.qr(G)  # (M, 4), (4, 4)
            R_inv = np.linalg.inv(R)
            # dx = R_inv @ Q.T @ dy
            dx = np.sum(R_inv[:, :, None] * Q.T[None, :, :], axis=1) @ dy
            # dx = np.sum(np.sum(R_inv[None, :, :] * Q[:, None, :], axis=2) * dy[None, :], axis=1)
        except Exception:
            return (x_hat * np.nan, np.nan)

        # alternative to QR
        # dx = np.linalg.lstsq(np.dot(G.T, G), np.dot(G.T, dy), rcond=None)[0]

        # update position
        x_hat += dx[0:3]
        b_hat += dx[3]

        if np.all(np.abs(dx) < convergence_threshold):
            break

        count += 1

    return (x_hat, b_hat)
# any standard kalman functions that i could need
"""
    .. code::

        while True:
            z, R = read_sensor()
            x, P = predict(x, P, F, Q)
            x, P = update(x, P, z, R, H)
"""
from __future__ import absolute_import, division
from copy import deepcopy
from math import log, exp, sqrt
import sys
import warnings
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
from copy import deepcopy
from math import log, exp, sqrt
import numpy as np
from numpy import dot, zeros, eye, isscalar
import numpy.linalg as linalg

def predict(x, P, F, Q, u=None, B=None, alpha_sq=1.0):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.

    Parameters
    ----------
    x : np.array
        State estimate vector.
    P : np.array
        Covariance estimate matrix.
    F : np.array
        State transition matrix.
    Q : np.array
        Process noise covariance matrix.
    u : np.array, optional
        Control vector. If not None, it is multiplied by B to create the control input into the system.
    B : np.array, optional
        Control transition matrix.
    alpha_sq : float, optional
        Fudge factor used to inflate the process noise covariance. Default is 1.0.

    Returns
    -------
    x : np.array
        Predicted state estimate vector.
    P : np.array
        Predicted covariance estimate matrix.
    """

    if isscalar(Q):
        Q = eye(len(x)) * Q

    # x = Fx + Bu
      x = dot(F, x) + dot(B, u)

    # P = FPF' + Q
    P = alpha_sq * dot(dot(F, P), F.T) + Q

    return x, P


def update(x, P, z, R, H):
    """
    Add a new measurement (z) to the Kalman filter.

    Parameters
    ----------
    x : np.array
        Predicted state estimate vector.
    P : np.array
        Predicted covariance estimate matrix.
    z : np.array
        Measurement vector.
    R : np.array or scalar
        Measurement noise covariance matrix.
    H : np.array
        Measurement function matrix.

    Returns
    -------
    x : np.array
        Updated state estimate vector.
    P : np.array
        Updated covariance estimate matrix.
    """

    if z is None:
        return x, P

    if isscalar(R):
        R = eye(len(z)) * R

    # y = z - Hx
    y = z - dot(H, x)

    # S = HPH' + R
    PHT = dot(P, H.T)
    S = dot(H, PHT) + R
    SI = linalg.inv(S)

    # K = PH'inv(S)
    K = dot(PHT, SI)

    # x = x + Ky
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    I_KH = eye(len(x)) - dot(K, H)
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

    return x, P


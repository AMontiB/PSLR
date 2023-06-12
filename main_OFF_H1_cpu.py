# import tensorflow_addons as tfa
import tensorflow as tf
import time
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']= ' '
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(1, 'CameraFingerprint_python/CameraFingerprint/')
import src.Functions as Fu
import src.Filter as Ft
import cv2
import numpy as np
import glob
from scipy.io import savemat
from scipy.interpolate import interp1d



def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def radial_cordinates(M, N):
    center = [M // 2, N // 2]
    xi, yi = np.meshgrid(np.arange(M), np.arange(N))
    print('xi', xi.dtype)
    xt = xi - center[0]
    yt = yi - center[1]
    r = np.sqrt(xt ** 2 + yt ** 2)
    theta = np.arctan2(xt, yt)
    R = math.sqrt(center[0] ** 2 + center[1] ** 2)
    r = r / R
    return r, theta, R, xi, yi, center, xt, yt


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


@tf.function
def new_coordinates(r, k1, k2, R):
    s = (r) * (1 + k1 * (r ** 2) + k2 * (r ** 4))
    s2 = s * R
    return s2


def polar2cart(s2, theta, M, N):
    v = tf.cast(s2 * np.cos(theta), tf.int64)
    u = tf.cast(s2 * np.sin(theta), tf.int64)
    if (tf.reduce_max(v + tf.abs(tf.reduce_max(tf.cast(N, tf.int64)) // 2)) * tf.reduce_max(
            u + tf.abs(tf.reduce_max(tf.cast(M, tf.int64)))) >=
            tf.reduce_max(v + tf.abs(tf.reduce_min(tf.cast(v, tf.int64)))) * tf.reduce_max(
                u + tf.abs(tf.reduce_min(tf.cast(u, tf.int64))))):
        v = tf.round(v + N // 2)
        u = tf.round(u + M // 2)
    else:
        v = tf.round(v + tf.abs(tf.reduce_min(tf.cast(v, tf.int64))))
        u = tf.round(u + tf.abs(tf.reduce_min(tf.cast(u, tf.int64))))
    return u, v


def linear_patters(mat):
    row_l = [np.mean(col[col != 0]) for col in mat]
    col_l = [np.mean(col[col != 0]) for col in mat.T]
    return row_l, col_l

def bilinear_interpolation(img):
  '''
  This function fill the 0 values interpolating them
  '''
  x = np.arange(len(img))
  img_new=np.zeros(img.shape)
  aux_x=[]
  aux_y=[]
  for i in range(len(img[0])-1):
      ix = np.where(img[:,i] != 0)
      if (len(ix[0])>1):
          f = interp1d(x[ix],img[ix,i], fill_value='extrapolate')
          img_new[x[ix[0][0]:ix[0][-1]],i] = f(x[ix[0][0]:ix[0][-1]])
          aux_y.append(x[0:ix[0][0]])
          aux_y.append(x[ix[0][-1]:len(img[0])])
          aux_x.append(i*np.ones(len(x[0:ix[0][0]])+len(x[ix[0][-1]:len(img[0])])))
  x=np.arange(len(img[0]))
  for i in range(0,(len(img)-1)):
      ix = np.where(img[i,:] != 0)
      if (len(ix[0])>1):
          f = interp1d(x[ix],img[i,ix], fill_value='extrapolate')
          img_new[i,x[ix[0][0]:ix[0][-1]]] = f(x[ix[0][0]:ix[0][-1]])
          aux_x.append(x[0:ix[0][0]])
          aux_x.append(x[ix[0][-1]:len(img[0])])
          aux_y.append(i*np.ones(len(x[0:ix[0][0]])+len(x[ix[0][-1]:len(img[0])])))
  aux_x=np.concatenate(np.array(aux_x)).astype(int)
  aux_y=np.concatenate(np.array(aux_y)).astype(int)
  img_new[aux_y,aux_x]=0
  return img_new


def crosscorr_Fingeprint_GPU(batchW, TA, norm2, sizebatch_K):
    meanW_batch = (tf.repeat(tf.repeat((tf.expand_dims(tf.expand_dims(tf.reduce_mean(batchW,
                                                                                     axis=[1, 2]), axis=2), axis=3)),
                                       repeats=[sizebatch_K[1]], axis=1), repeats=[sizebatch_K[2]], axis=2))
    batchW = batchW - meanW_batch
    normalizator = tf.math.sqrt(tf.reduce_sum(tf.math.pow(batchW, 2)) * norm2)
    FA = tf.signal.fft2d(tf.cast(tf.squeeze(batchW, axis=3), tf.complex64))
    AC = tf.multiply(FA, tf.repeat(tf.cast(TA, dtype=tf.complex64), axis=0, repeats=len(batchW.numpy())))
    return tf.math.real(tf.signal.ifft2d(AC)) / normalizator


def parallel_PCE(CXC, idx, ranges, squaresize=11):
    out = np.zeros(idx)
    for i in range(0, idx):
        shift_range = ranges[i]
        Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])
        C = CXC[i]
        Cinrange = C[-1 - shift_range[0]:, -1 - shift_range[1]:]
        [max_cc, imax] = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
        [ypeak, xpeak] = np.unravel_index(imax, Cinrange.shape)[0], np.unravel_index(imax, Cinrange.shape)[1]
        Out['peakheight'] = Cinrange[ypeak, xpeak]
        del Cinrange
        Out['PeakLocation'] = [shift_range[0] - ypeak, shift_range[1] - xpeak]
        C_without_peak = _RemoveNeighborhood(C,
                                             np.array(C.shape) - Out['PeakLocation'],
                                             squaresize)
        # signed PCE, peak-to-correlation energy
        PCE_energy = np.mean(C_without_peak * C_without_peak)
        out[i] = (Out['peakheight'] ** 2) / PCE_energy * np.sign(Out['peakheight'])
    return out


def _RemoveNeighborhood(X, x, ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    [M, N] = X.shape
    radius = (ssize - 1) / 2
    X = np.roll(X, [np.int(radius - x[0]), np.int(radius - x[1])], axis=[0, 1])
    Y = X[ssize:, :ssize];
    Y = Y.flatten()
    Y = np.concatenate([Y, X.flatten()[int(M * ssize):]], axis=0)
    return Y


def Energy_computer(noise_rc1):
    # crop
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc1[0:int(M / 2), :], zero)
    indices = tf.where(where)
    idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
    #
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc1, zero)
    indices = tf.where(where)
    #
    idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

    noise_rc1 = tf.image.crop_to_bounding_box(tf.squeeze(noise_rc1, axis=-1), idx_ul1.numpy(), idx_ul2.numpy(),
                                              idx_dr1.numpy() - idx_ul1.numpy(),
                                              idx_dr2.numpy() - idx_ul2.numpy())

    # linear patterns computer
    size_noise_rc = np.shape(noise_rc1.numpy())
    rK = np.mean(tf.squeeze(noise_rc1, axis=-1).numpy(), axis=1)
    ck = np.mean(tf.squeeze(noise_rc1, axis=-1).numpy(), axis=0)
    norm_r = np.linalg.norm(np.mean(tf.squeeze(noise_rc1, axis=-1).numpy(), axis=1)) ** 2  # col/row
    norm_c = np.linalg.norm(np.mean(tf.squeeze(noise_rc1, axis=-1).numpy(), axis=0)) ** 2  # row/col
    variance = np.var(tf.squeeze(noise_rc1, axis=-1).numpy())
    rKn = rK - rK * (1 / norm_r) * np.sqrt((size_noise_rc[1] * variance) / size_noise_rc[0])
    cKn = ck - ck * (1 / norm_c) * np.sqrt((size_noise_rc[0] * variance) / size_noise_rc[1])
    noise_LP2 = np.repeat(np.expand_dims(rKn, axis=1), size_noise_rc[1], axis=1) + np.repeat(
        np.expand_dims(cKn, axis=0), size_noise_rc[0], axis=0)
    noise_LP3 = tf.squeeze(noise_rc1, axis=-1).numpy() - noise_LP2
    rK = np.mean(noise_LP3, axis=1)
    ck = np.mean(noise_LP3, axis=0)
    return (np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2))


def optimizer(W, x, N, M, Ri, batch_thetai, batch_M, batch_N, ri, variance, xi, yi):
    alpha = 1
    gamma = 2
    rho = 0.5
    sigma = 0.5
    itero = 0
    # initialize simplex
    a = [[x, x ** 2], [x, -x ** 2], [0.8 * x, 0]]
    flag_stop = 1
    i = 0
    a1 = a[0][:]
    a2 = a[1][:]
    a3 = a[2][:]
    while flag_stop == 1:
        E = []
        s = new_coordinates(tf.stack([ri]),
                            tf.constant(a[0][0], dtype=tf.float64, shape=[1, N, M]),
                            tf.constant(a[0][1], dtype=tf.float64,
                                        shape=[1, N, M]),
                            tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
        u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
        v = np.round(v + np.abs(np.amax((N)) // 2))
        u = np.round(u + np.abs(np.amax((M)) // 2))
        u = u[0].astype(np.int32)
        v = v[0].astype(np.int32)
        dist = np.zeros([np.max(v + 1), np.max(u + 1)])
        yii = yi.numpy().astype(np.int32)
        xii = xi.numpy().astype(np.int32)

        dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
            v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]

        W1 = bilinear_interpolation(dist)
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W1[0:int(M / 2), :], zero)
        indices = tf.where(where)
        idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
        #
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W1, zero)
        indices = tf.where(where)
        #
        idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

        W1 = tf.image.crop_to_bounding_box(np.expand_dims(W1, axis=-1), idx_ul1.numpy(),
                                                 idx_ul2.numpy(),
                                                 idx_dr1.numpy() - idx_ul1.numpy(),
                                                 idx_dr2.numpy() - idx_ul2.numpy())
        rK = np.mean(tf.squeeze(W1, axis=-1).numpy(), axis=1)
        ck = np.mean(tf.squeeze(W1, axis=-1).numpy(), axis=0)
        E.append(-(np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))
        #
        s = new_coordinates(tf.stack([ri]),
                            tf.constant(a[1][0], dtype=tf.float64, shape=[1, N, M]),
                            tf.constant(a[1][1], dtype=tf.float64,
                                        shape=[1, N, M]),
                            tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
        u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
        #
        v = np.round(v + np.abs(np.amax((N)) // 2))
        u = np.round(u + np.abs(np.amax((M)) // 2))
        u = u[0].astype(np.int32)
        v = v[0].astype(np.int32)
        dist = np.zeros([np.max(v + 1), np.max(u + 1)])
        yii = yi.numpy().astype(np.int32)
        xii = xi.numpy().astype(np.int32)

        dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
            v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]

        W2 = bilinear_interpolation(dist)
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W2[0:int(M / 2), :], zero)
        indices = tf.where(where)
        idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
        #
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W2, zero)
        indices = tf.where(where)
        #
        idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

        W2 = tf.image.crop_to_bounding_box(np.expand_dims(W2, axis=-1), idx_ul1.numpy(),
                                           idx_ul2.numpy(),
                                           idx_dr1.numpy() - idx_ul1.numpy(),
                                           idx_dr2.numpy() - idx_ul2.numpy())
        rK = np.mean(tf.squeeze(W2, axis=-1).numpy(), axis=1)
        ck = np.mean(tf.squeeze(W2, axis=-1).numpy(), axis=0)
        E.append(-(np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))
        #
        s = new_coordinates(tf.stack([ri]),
                            tf.constant(a[2][0], dtype=tf.float64, shape=[1, N, M]),
                            tf.constant(a[2][1], dtype=tf.float64,
                                        shape=[1, N, M]),
                            tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
        u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
        v = np.round(v + np.abs(np.amax((N)) // 2))
        u = np.round(u + np.abs(np.amax((M)) // 2))
        u = u[0].astype(np.int32)
        v = v[0].astype(np.int32)
        dist = np.zeros([np.max(v + 1), np.max(u + 1)])
        yii = yi.numpy().astype(np.int32)
        xii = xi.numpy().astype(np.int32)

        dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
            v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  
        W3 = bilinear_interpolation(dist)
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W3[0:int(M / 2), :], zero)
        indices = tf.where(where)
        idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
        #
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(W3, zero)
        indices = tf.where(where)
        #
        idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

        W3 = tf.image.crop_to_bounding_box(np.expand_dims(W3, axis=-1), idx_ul1.numpy(),
                                           idx_ul2.numpy(),
                                           idx_dr1.numpy() - idx_ul1.numpy(),
                                           idx_dr2.numpy() - idx_ul2.numpy())
        # linear patterns computer
        size_noise_rc = np.shape(W3.numpy())
        rK = np.mean(tf.squeeze(W3, axis=-1).numpy(), axis=1)
        ck = np.mean(tf.squeeze(W3, axis=-1).numpy(), axis=0)
        E.append(-(np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))
        #
        E_sorted = sorted(E)
        a1 = a[E.index(E_sorted[0])]
        a2 = a[E.index(E_sorted[1])]
        a3 = a[E.index(E_sorted[2])]
        # centroid
        x0 = [(a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2]
        # reflection
        xr = [x0[0] + alpha * (x0[0] - a3[0]), x0[1] + alpha * (x0[1] - a3[1])]
        s = new_coordinates(tf.stack([ri]),
                            tf.constant(xr[0], dtype=tf.float64, shape=[1, N, M]),
                            tf.constant(xr[1], dtype=tf.float64,
                                        shape=[1, N, M]),
                            tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
        u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
        v = np.round(v + np.abs(np.amax((N)) // 2))
        u = np.round(u + np.abs(np.amax((M)) // 2))
        u = u[0].astype(np.int32)
        v = v[0].astype(np.int32)
        dist = np.zeros([np.max(v + 1), np.max(u + 1)])
        yii = yi.numpy().astype(np.int32)
        xii = xi.numpy().astype(np.int32)

        dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
            v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]
        Wr = bilinear_interpolation(dist)
        # crop
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(Wr[0:int(M / 2), :], zero)
        indices = tf.where(where)
        idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
        #
        zero = tf.constant(0, dtype=tf.double)
        where = tf.not_equal(Wr, zero)
        indices = tf.where(where)
        #
        idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
        idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

        Wr = tf.image.crop_to_bounding_box(np.expand_dims(Wr, axis=-1), idx_ul1.numpy(),
                                           idx_ul2.numpy(),
                                           idx_dr1.numpy() - idx_ul1.numpy(),
                                           idx_dr2.numpy() - idx_ul2.numpy())
        # linear patterns computer
        size_noise_rc = np.shape(Wr.numpy())
        rK = np.mean(tf.squeeze(Wr, axis=-1).numpy(), axis=1)
        ck = np.mean(tf.squeeze(Wr, axis=-1).numpy(), axis=0)
        Er = -((np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))

        if E_sorted[0] <= Er < E_sorted[2]:
            a3 = xr
            E_sorted[2] = Er
        elif Er < E_sorted[0]:

            xe = [x0[0] + gamma * (xr[0] - x0[0]), x0[1] + gamma * (xr[1] - x0[1])]
            s = new_coordinates(tf.stack([ri]),
                                tf.constant(xe[0], dtype=tf.float64, shape=[1, N, M]),
                                tf.constant(xe[1], dtype=tf.float64,
                                            shape=[1, N, M]),
                                tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
            u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
            v = np.round(v + np.abs(np.amax((N)) // 2))
            u = np.round(u + np.abs(np.amax((M)) // 2))
            u = u[0].astype(np.int32)
            v = v[0].astype(np.int32)
            dist = np.zeros([np.max(v + 1), np.max(u + 1)])
            yii = yi.numpy().astype(np.int32)
            xii = xi.numpy().astype(np.int32)

            dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
                v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                    np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]

            We = bilinear_interpolation(dist)
            # crop
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(We[0:int(M / 2), :], zero)
            indices = tf.where(where)
            idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
            #
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(We, zero)
            indices = tf.where(where)
            #
            idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

            We = tf.image.crop_to_bounding_box(np.expand_dims(We, axis=-1), idx_ul1.numpy(),
                                               idx_ul2.numpy(),
                                               idx_dr1.numpy() - idx_ul1.numpy(),
                                               idx_dr2.numpy() - idx_ul2.numpy())
            # linear patterns computer
            size_noise_rc = np.shape(We.numpy())
            rK = np.mean(tf.squeeze(We, axis=-1).numpy(), axis=1)
            ck = np.mean(tf.squeeze(We, axis=-1).numpy(), axis=0)
            Ee = -((np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))
            if Ee < Er:
                a3 = xe
                E_sorted[2] = Ee
        elif Er >= E_sorted[1]:
            xc = [x0[0] + rho * (a3[0] - x0[0]), x0[1] + gamma * (a3[1] - x0[1])]
            s = new_coordinates(tf.stack([ri]),
                                tf.constant(xc[0], dtype=tf.float64, shape=[1, N, M]),
                                tf.constant(xc[1], dtype=tf.float64,
                                            shape=[1, N, M]),
                                tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))
            u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
            v = np.round(v + np.abs(np.amax((N)) // 2))
            u = np.round(u + np.abs(np.amax((M)) // 2))
            u = u[0].astype(np.int32)
            v = v[0].astype(np.int32)
            dist = np.zeros([np.max(v + 1), np.max(u + 1)])
            yii = yi.numpy().astype(np.int32)
            xii = xi.numpy().astype(np.int32)

            dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
                v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                    np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]

            Wc = bilinear_interpolation(dist)
            # crop
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(Wc[0:int(M / 2), :], zero)
            indices = tf.where(where)
            idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
            #
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(Wc, zero)
            indices = tf.where(where)
            #
            idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

            Wc = tf.image.crop_to_bounding_box(np.expand_dims(Wc, axis=-1), idx_ul1.numpy(),
                                               idx_ul2.numpy(),
                                               idx_dr1.numpy() - idx_ul1.numpy(),
                                               idx_dr2.numpy() - idx_ul2.numpy())

            size_noise_rc = np.shape(Wc.numpy())
            rK = np.mean(tf.squeeze(Wc, axis=-1).numpy(), axis=1)
            ck = np.mean(tf.squeeze(Wc, axis=-1).numpy(), axis=0)
            Ec = -((np.sum(np.array(rK) ** 2) + np.sum(np.array(ck) ** 2)))
            if Ec < E_sorted[2]:
                a3 = xc
                E_sorted[2] = Ec
            else:
                a1 = [a1[0] + sigma * (a1[0] - a1[0]), a1[0] + sigma * (a1[1] - a1[1])]
                a2 = [a1[0] + sigma * (a2[0] - a1[0]), a1[0] + sigma * (a2[1] - a1[1])]
                a3 = [a1[0] + sigma * (a3[0] - a1[0]), a1[0] + sigma * (a3[1] - a1[1])]
        # termination criteria
        a = [a1, a2, a3]
        print('a: ', a)
        if np.std(E_sorted) < 0.005 or (np.abs(a) > 0.33).any():
            a_off = a
            flag_stop = 0
        if i == 0:
            min_std = np.std(E_sorted)
            a_off = a
        else:
            if np.std(E_sorted) >= min_std:
                itero += 1
                if itero == 10:
                    flag_stop = 0
            else:
                itero = 0
                min_std = np.std(E_sorted)
                a_off = a
        i += 1
    return a_off


def crop_center_image(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def crop_center_finger(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def alpha_validation(W, K, alpha, batch_thetai, batch_M, batch_N, ri, N, M, Ri, xi, yi):
    s = new_coordinates(tf.stack([ri]), tf.constant(alpha + 0.1,
                                                    dtype=tf.float64, shape=[1, N, M]),
                        tf.constant(0, dtype=tf.float64, shape=[1, N, M]),
                        tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))

    u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
    v = np.round(v + np.abs(np.amax((N)) // 2))
    u = np.round(u + np.abs(np.amax((M)) // 2))
    u = u[0].astype(np.int32)
    v = v[0].astype(np.int32)
    dist = np.zeros([np.max(v + 1), np.max(u + 1)])
    yii = yi.numpy().astype(np.int32)
    xii = xi.numpy().astype(np.int32)

    dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
        np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
        v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  # yi, xi]#[v, u]  # [yi, xi]

    noise_rc = bilinear_interpolation(dist)
    # crop
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc[0:int(M / 2), :], zero)
    indices = tf.where(where)
    idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
    #
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc, zero)
    indices = tf.where(where)
    #
    idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

    noise_rc = tf.image.crop_to_bounding_box(np.expand_dims(noise_rc, axis=-1), idx_ul1.numpy(),
                                       idx_ul2.numpy(),
                                       idx_dr1.numpy() - idx_ul1.numpy(),
                                       idx_dr2.numpy() - idx_ul2.numpy())
    #
    Fingeprint_2 = tf.expand_dims(K, axis=-1)
    cropped_Fingeprint = tf.image.crop_to_bounding_box(Fingeprint_2, idx_ul1.numpy(),
                                                       idx_ul2.numpy(),
                                                       idx_dr1.numpy() - idx_ul1.numpy(),
                                                       idx_dr2.numpy() - idx_ul2.numpy())
    cropped_Fingeprint = tf.squeeze(cropped_Fingeprint, axis=2)
    array2c = cropped_Fingeprint.numpy()
    array2c = array2c - array2c.mean()
    tilted_array2c = np.fliplr(array2c)
    tilted_array2c = np.flipud(tilted_array2c)
    norm2c = np.sum(np.power(array2c, 2))
    TAc = np.fft.fft2(tilted_array2c)
    TA_tfc = tf.expand_dims(tf.convert_to_tensor(TAc, dtype=tf.complex64), axis=0)
    XC = (crosscorr_Fingeprint_GPU(tf.cast(tf.expand_dims(noise_rc, axis=0), dtype=tf.float32),
                                   TA_tfc,
                                   norm2c, np.shape(TA_tfc)))
    ranges = [[0, 0]]
    PCE_H0_1 = parallel_PCE(XC.numpy(), len(XC), ranges)

    s = new_coordinates(tf.stack([ri]), tf.constant(alpha - 0.1,
                                                    dtype=tf.float64, shape=[1, N, M]),
                        tf.constant(0, dtype=tf.float64, shape=[1, N, M]),
                        tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))

    u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
    v = np.round(v + np.abs(np.amax((N)) // 2))
    u = np.round(u + np.abs(np.amax((M)) // 2))
    u = u[0].astype(np.int32)
    v = v[0].astype(np.int32)
    dist = np.zeros([np.max(v + 1), np.max(u + 1)])
    yii = yi.numpy().astype(np.int32)
    xii = xi.numpy().astype(np.int32)

    dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
        np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = W[
        v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
            np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  

    noise_rc = bilinear_interpolation(dist)
    # crop
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc[0:int(M / 2), :], zero)
    indices = tf.where(where)
    idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
    #
    zero = tf.constant(0, dtype=tf.double)
    where = tf.not_equal(noise_rc, zero)
    indices = tf.where(where)
    #
    idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
    idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

    noise_rc = tf.image.crop_to_bounding_box(np.expand_dims(noise_rc, axis=-1), idx_ul1.numpy(),
                                             idx_ul2.numpy(),
                                             idx_dr1.numpy() - idx_ul1.numpy(),
                                             idx_dr2.numpy() - idx_ul2.numpy())
    #
    Fingeprint_2 = tf.expand_dims(K, axis=-1)
    cropped_Fingeprint = tf.image.crop_to_bounding_box(Fingeprint_2, idx_ul1.numpy(),
                                                       idx_ul2.numpy(),
                                                       idx_dr1.numpy() - idx_ul1.numpy(),
                                                       idx_dr2.numpy() - idx_ul2.numpy())
    cropped_Fingeprint = tf.squeeze(cropped_Fingeprint, axis=2)
    array2c = cropped_Fingeprint.numpy()
    array2c = array2c - array2c.mean()
    tilted_array2c = np.fliplr(array2c)
    tilted_array2c = np.flipud(tilted_array2c)
    norm2c = np.sum(np.power(array2c, 2))
    TAc = np.fft.fft2(tilted_array2c)
    TA_tfc = tf.expand_dims(tf.convert_to_tensor(TAc, dtype=tf.complex64), axis=0)
    XC = (crosscorr_Fingeprint_GPU(tf.cast(tf.expand_dims(noise_rc, axis=0), dtype=tf.float32),
                                   TA_tfc,
                                   norm2c, np.shape(TA_tfc)))
    ranges = [[0, 0]]
    PCE_H0_2 = parallel_PCE(XC.numpy(), len(XC), ranges)

    if (PCE_max - np.mean([PCE_H0_1, PCE_H0_2])) < 10.58:
        print('STOP')
        return 0
    else:
        print('OK')
        return 1

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
with tf.device('/cpu:0'):
    PCE_array_v = [[], [], []]
    PCE_grid_search = []
    time_grid_search = []
    time_array = [[], [], [], []]
    validity_array = []

    Fingeprint_list = ['../../official_Code/PRNU_FILES/Fingerprint_CanonEOS1200d.dat',
                       '../../official_Code/PRNU_FILES/Fingerprint_CanonEOS1200d.dat',
                       '../../official_Code/PRNU_FILES/Fingerprint_CanonEOS1200d.dat',
                       '../../official_Code/PRNU_FILES/Fingerprint_CanonEOS1200d.dat',
                       '../../official_Code/PRNU_FILES/Fingerprint_CanonEOS1200d.dat']

    images_set = ['../../official_Code/DATASET_RADIAL_CORRECTION/large_scale_test_H1/*.JPG',
                  '../../official_Code/DATASET_RADIAL_CORRECTION/diff_rc/*.JPG',
                  '../../official_Code/DATASET_RADIAL_CORRECTION/gimp/*.JPG',
                  '../../official_Code/DATASET_RADIAL_CORRECTION/photoshop/*.JPG',
                  '../../official_Code/DATASET_RADIAL_CORRECTION/pt_lens/*.JPG']
    
    outfile_name = 'OUTPUT_H1_CPU.mat'

    for idx_fing in range(len(Fingeprint_list)):
        Fingerprint_original = np.genfromtxt(Fingeprint_list[idx_fing])
        fingersize_or = Fingerprint_original.shape
        diagonal_or = np.sqrt(fingersize_or[1] ** 2 + fingersize_or[0] ** 2)
        Fingerprint_tf_or = tf.cast(Fingerprint_original, tf.double)
        array2_or = Fingerprint_original.astype(np.double)
        array2_or = array2_or - array2_or.mean()
        tilted_array2_or = np.fliplr(array2_or)
        tilted_array2_or = np.flipud(tilted_array2_or)
        norm2_or = np.sum(np.power(array2_or, 2))
        TA_or = np.fft.fft2(tilted_array2_or)
        TA_tf_or = tf.expand_dims(tf.convert_to_tensor(TA_or, dtype=tf.complex64), axis=0)

        images = glob.glob(images_set[idx_fing])
        idx_img = 0
        for im_name in images:
            time_grid_search.append(0)
            PCE_grid_search.append(0)
            start = time.time()
            print(im_name)
            im_arr = cv2.imread(im_name)
            print('-> size img: ', np.shape(im_arr))
            if np.shape(im_arr)[0] > np.shape(im_arr)[1]:
                print('rotate')
                im_arr = np.rot90(im_arr)
            if np.shape(im_arr)[0] > np.shape(im_arr)[1]:
                print('rotate')
                im_arr = np.rot90(im_arr)
                print('-> size img: ', np.shape(im_arr))
            if np.shape(im_arr)[0] != fingersize_or[0] or np.shape(im_arr)[1] != fingersize_or[1]:
                print('crop')
                if np.shape(im_arr)[0] > fingersize_or[0] and np.shape(im_arr)[1] > fingersize_or[1]:
                    print('crop image')
                    im_arr = crop_center_image(im_arr, fingersize_or[1], fingersize_or[0])
                    Fingerprint = Fingerprint_original
                    fingersize = fingersize_or
                    diagonal = diagonal_or
                    Fingerprint_tf = Fingerprint_tf_or
                    array2 = array2_or
                    tilted_array2 = tilted_array2_or
                    norm2 = norm2_or
                    TA = TA_or
                    TA_tf = TA_tf_or
                elif fingersize_or[0] > np.shape(im_arr)[0] and fingersize_or[1] > np.shape(im_arr)[1]:
                    print('crop fingerprint')
                    Fingerprint = crop_center_finger(Fingerprint_original, np.shape(im_arr)[1], np.shape(im_arr)[0])
                    fingersize = Fingerprint.shape
                    diagonal = np.sqrt(fingersize[1] ** 2 + fingersize[0] ** 2)
                    Fingerprint_tf = tf.cast(Fingerprint, tf.double)
                    array2 = Fingerprint.astype(np.double)
                    array2 = array2 - array2.mean()
                    tilted_array2 = np.fliplr(array2)
                    tilted_array2 = np.flipud(tilted_array2)
                    norm2 = np.sum(np.power(array2, 2))
                    TA = np.fft.fft2(tilted_array2)
                    TA_tf = tf.expand_dims(tf.convert_to_tensor(TA, dtype=tf.complex64), axis=0)
                else:
                    if np.shape(im_arr)[0] >= fingersize_or[0] and fingersize_or[1] >= np.shape(im_arr)[1]:
                        print('crop image and crop fingerprint')
                        im_arr = crop_center_image(im_arr, np.shape(im_arr)[1], fingersize_or[0])
                        Fingerprint = crop_center_finger(Fingerprint_original, np.shape(im_arr)[1], fingersize_or[0])
                        fingersize = Fingerprint.shape  
                        diagonal = np.sqrt(fingersize[1] ** 2 + fingersize[0] ** 2)
                        Fingerprint_tf = tf.cast(Fingerprint, tf.double)
                        array2 = Fingerprint.astype(np.double)
                        array2 = array2 - array2.mean()
                        tilted_array2 = np.fliplr(array2)
                        tilted_array2 = np.flipud(tilted_array2)
                        norm2 = np.sum(np.power(array2, 2))
                        TA = np.fft.fft2(tilted_array2)
                        TA_tf = tf.expand_dims(tf.convert_to_tensor(TA, dtype=tf.complex64), axis=0)
                    elif np.shape(im_arr)[0] <= fingersize_or[0] and fingersize_or[1] <= np.shape(im_arr)[1]:
                        print('crop image and crop fingerprint')
                        im_arr = crop_center_image(im_arr, fingersize_or[1], np.shape(im_arr)[0])
                        Fingerprint = crop_center_finger(Fingerprint_original, fingersize_or[1], np.shape(im_arr)[0])
                        fingersize = Fingerprint.shape
                        diagonal = np.sqrt(fingersize[1] ** 2 + fingersize[0] ** 2)
                        Fingerprint_tf = tf.cast(Fingerprint, tf.double)
                        array2 = Fingerprint.astype(np.double)
                        array2 = array2 - array2.mean()
                        tilted_array2 = np.fliplr(array2)
                        tilted_array2 = np.flipud(tilted_array2)
                        norm2 = np.sum(np.power(array2, 2))
                        TA = np.fft.fft2(tilted_array2)
                        TA_tf = tf.expand_dims(tf.convert_to_tensor(TA, dtype=tf.complex64), axis=0)
            else:
                Fingerprint = Fingerprint_original
                fingersize = fingersize_or
                diagonal = diagonal_or
                Fingerprint_tf = Fingerprint_tf_or
                array2 = array2_or
                tilted_array2 = tilted_array2_or
                norm2 = norm2_or
                TA = TA_or
                TA_tf = TA_tf_or

            noise_LP = Ft.NoiseExtractFromImage(im_arr, sigma=2., noZM=True)
            noise_LP = Fu.WienerInDFT(noise_LP, np.std(noise_LP))
            M, N = [im_arr.shape[1], im_arr.shape[0]]
            noise_LP = noise_LP - np.mean(noise_LP)
            variance = np.var(noise_LP)
            noise_LP = noise_LP / np.std(noise_LP)
            rK = np.mean(noise_LP, axis=1)
            ck = np.mean(noise_LP, axis=0)
            norm_r = np.linalg.norm(rK)  # col/row
            norm_c = np.linalg.norm(ck)  # row/col

            rKn = rK - rK * (1 / norm_r) * np.sqrt((N * variance) / M)
            cKn = ck - ck * (1 / norm_c) * np.sqrt((M * variance) / N)
            noise_LP2 = np.repeat(np.expand_dims(rKn, axis=1), M, axis=1) + np.repeat(np.expand_dims(cKn, axis=0), N,
                                                                                      axis=0)
            noise_LP = noise_LP - noise_LP2

            noise = Ft.NoiseExtractFromImage(im_arr, sigma=2.)
            noise = Fu.WienerInDFT(noise, np.std(noise))
            noise_tf = tf.cast(noise, tf.float32)
            M, N = [im_arr.shape[1], im_arr.shape[0]]
            variance = np.var(noise_LP)

            # RADIAL CORRECTION PARAM
            r, theta, R, xi, yi, center, xt, yt = radial_cordinates(M, N)
            ri = tf.convert_to_tensor(r, dtype=tf.float64)
            thetai = tf.convert_to_tensor(theta, dtype=tf.float64)
            Ri = tf.convert_to_tensor(R, dtype=tf.float64)
            xi = tf.convert_to_tensor(xi, dtype=tf.float64)
            yi = tf.convert_to_tensor(yi, dtype=tf.float64)
            batch_thetai = tf.stack([thetai])
            batch_M = tf.convert_to_tensor([M * tf.ones([N, M])], dtype=tf.int64)
            batch_N = tf.convert_to_tensor([N * tf.ones([N, M])], dtype=tf.int64)

            XC = (crosscorr_Fingeprint_GPU((tf.expand_dims(tf.expand_dims(noise_tf, axis=0), axis=3)), TA_tf,
                                       norm2, np.shape(TA_tf.numpy())))
            ranges = [[0, 0]]
            PCE_max = parallel_PCE(XC.numpy(), len(XC), ranges)
            PCE_array_v[0].append(PCE_max[0])
            time_array[0].append(time.time() - start)
            print('PCE before radial correction inversion: ', PCE_max)

            A = 0.24
            k_max = 6
            alphas = [-0.24, 0, 0.24]
            PCE_array = []
            a_bkp = alphas
            tau1 = 15
            tau_2 = 75
            tau3 = 75
            flag_out = 0
            alpha_best = 0
            for k in range(k_max):
                if k == 0:
                    print('compute PCE first 3 samples')
                    for a in alphas:
                        s = new_coordinates(tf.stack([ri]), tf.constant(a, dtype=tf.float64, shape=[1, N, M]),
                                            tf.constant(0, dtype=tf.float64, shape=[1, N, M]),
                                            tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))

                        u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
                        v = np.round(v + np.abs(np.amax((N)) // 2))
                        u = np.round(u + np.abs(np.amax((M)) // 2))
                        u = u[0].astype(np.int32)
                        v = v[0].astype(np.int32)
                        dist = np.zeros([np.max(v + 1), np.max(u + 1)])
                        yii = yi.numpy().astype(np.int32)
                        xii = xi.numpy().astype(np.int32)
                        dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
                                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = noise[
                                v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                                    np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]]  
                        noise_rc = bilinear_interpolation(dist)
                        # crop
                        zero = tf.constant(0, dtype=tf.double)
                        where = tf.not_equal(noise_rc[0:int(M / 2), :], zero)
                        indices = tf.where(where)
                        idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
                        idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
                        #
                        zero = tf.constant(0, dtype=tf.double)
                        where = tf.not_equal(noise_rc, zero)
                        indices = tf.where(where)
                        #
                        idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
                        idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
                        noise_rc = tf.image.crop_to_bounding_box(np.expand_dims(noise_rc, axis=-1), idx_ul1.numpy(),
                                                                 idx_ul2.numpy(),
                                                                 idx_dr1.numpy() - idx_ul1.numpy(),
                                                                 idx_dr2.numpy() - idx_ul2.numpy())
                        #
                        Fingeprint_2 = tf.expand_dims(Fingerprint_tf, axis=-1)
                        cropped_Fingeprint = tf.image.crop_to_bounding_box(Fingeprint_2, idx_ul1.numpy(),
                                                                           idx_ul2.numpy(),
                                                                           idx_dr1.numpy() - idx_ul1.numpy(),
                                                                           idx_dr2.numpy() - idx_ul2.numpy())
                        cropped_Fingeprint = tf.squeeze(cropped_Fingeprint, axis=2)
                        array2c = cropped_Fingeprint.numpy()  
                        array2c = array2c - array2c.mean()
                        tilted_array2c = np.fliplr(array2c)
                        tilted_array2c = np.flipud(tilted_array2c)
                        norm2c = np.sum(np.power(array2c, 2))
                        TAc = np.fft.fft2(tilted_array2c)
                        TA_tfc = tf.expand_dims(tf.convert_to_tensor(TAc, dtype=tf.complex64), axis=0)
                        XC = (crosscorr_Fingeprint_GPU(tf.cast(tf.expand_dims(noise_rc, axis=0), dtype=tf.float32),
                                                       TA_tfc,
                                                       norm2c, np.shape(TA_tfc)))
                        ranges = [[0, 0]]
                        PCE_radial = parallel_PCE(XC.numpy(), len(XC), ranges)
                        PCE_array.append(PCE_radial[0])
                        PCE_grid_search.append(PCE_radial[0])
                        time_grid_search.append(time.time()-start)
                        print('----------------------')
                else:
                    for i in range(len(alphas) - 1):
                        if not round((alphas[i] + alphas[i + 1]) / 2, 2) in a_bkp:
                            a_bkp.append(round((alphas[i] + alphas[i + 1]) / 2, 2))
                            # compute PCE
                            s = new_coordinates(tf.stack([ri]), tf.constant(round((alphas[i] + alphas[i + 1]) / 2, 2),
                                                                            dtype=tf.float64, shape=[1, N, M]),
                                                tf.constant(0, dtype=tf.float64, shape=[1, N, M]),
                                                tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))

                            u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
                            v = np.round(v + np.abs(np.amax((N)) // 2))
                            u = np.round(u + np.abs(np.amax((M)) // 2))
                            u = u[0].astype(np.int32)
                            v = v[0].astype(np.int32)
                            dist = np.zeros([np.max(v + 1), np.max(u + 1)])
                            yii = yi.numpy().astype(np.int32)
                            xii = xi.numpy().astype(np.int32)
                            dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
                                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = noise[
                                v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                                    np.logical_and(v < N, v > 0) * np.logical_and(u < M,
                                                                                  u > 0)]] 
                            noise_rc = bilinear_interpolation(dist)
                            zero = tf.constant(0, dtype=tf.double)
                            where = tf.not_equal(noise_rc[0:int(M / 2), :], zero)
                            indices = tf.where(where)
                            idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
                            idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
                            #
                            zero = tf.constant(0, dtype=tf.double)
                            where = tf.not_equal(noise_rc, zero)
                            indices = tf.where(where)
                            #
                            idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
                            idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])


                            noise_rc = tf.image.crop_to_bounding_box(np.expand_dims(noise_rc, axis=-1), idx_ul1.numpy(),
                                                                     idx_ul2.numpy(),
                                                                     idx_dr1.numpy() - idx_ul1.numpy(),
                                                                     idx_dr2.numpy() - idx_ul2.numpy())

                            #
                            Fingeprint_2 = tf.expand_dims(Fingerprint_tf, axis=-1)
                            cropped_Fingeprint = tf.image.crop_to_bounding_box(Fingeprint_2, idx_ul1.numpy(),
                                                                               idx_ul2.numpy(),
                                                                               idx_dr1.numpy() - idx_ul1.numpy(),
                                                                               idx_dr2.numpy() - idx_ul2.numpy())
                            cropped_Fingeprint = tf.squeeze(cropped_Fingeprint, axis=2)
                            array2c = cropped_Fingeprint.numpy()
                            array2c = array2c - array2c.mean()
                            tilted_array2c = np.fliplr(array2c)
                            tilted_array2c = np.flipud(tilted_array2c)
                            norm2c = np.sum(np.power(array2c, 2))
                            TAc = np.fft.fft2(tilted_array2c)
                            TA_tfc = tf.expand_dims(tf.convert_to_tensor(TAc, dtype=tf.complex64), axis=0)
                            XC = (crosscorr_Fingeprint_GPU(tf.cast(tf.expand_dims(noise_rc, axis=0), dtype=tf.float32),
                                                           TA_tfc,
                                                           norm2c, np.shape(TA_tfc)))
                            ranges = [[0, 0]]
                            PCE_radial = parallel_PCE(XC.numpy(), len(XC), ranges)
                            if PCE_radial > PCE_max:
                                PCE_max = PCE_radial[0]
                                print('PCE: ', PCE_max)
                                print('alpha: ', round((alphas[i] + alphas[i + 1]) / 2, 2))
                                alpha_best = round((alphas[i] + alphas[i + 1]) / 2, 2)
                            PCE_array.append(PCE_radial[0])
                            PCE_grid_search.append(PCE_radial[0])
                            time_grid_search.append(time.time() - start)
                    alphas = sorted(a_bkp)
                    Z = [x for _, x in sorted(zip(a_bkp, PCE_array))]
                    PCE_array = Z
                a_bkp = alphas
            print('-------END GRID SEARCH a1-------')
            print('alphas: ', alphas)
            print('--------------')
            print('PCE: ', PCE_array)
            print('alpha_best: ', alpha_best)
            PCE_array_v[1].append(PCE_max)
            time_array[1].append(time.time() - start)

            #validation
            print('VALIDATION')
            validity_array.append(alpha_validation(noise, Fingerprint_tf, alpha_best, batch_thetai, batch_M, batch_N, ri, N, M, Ri, xi, yi))
            time_array[2].append(time.time() - start)

            # Nelder-Mead
            a = optimizer(noise_LP, alpha_best, N, M, Ri, batch_thetai, batch_M, batch_N, ri, variance, xi, yi)
            #final PCE
            s = new_coordinates(tf.stack([ri]), tf.constant(a[0][0], dtype=tf.float64, shape=[1, N, M]),
                                tf.constant(a[0][1], dtype=tf.float64, shape=[1, N, M]),
                                tf.constant(Ri, dtype=tf.float64, shape=[1, N, M]))

            u, v = polar2cart(s, batch_thetai, batch_M, batch_N)
            v = np.round(v + np.abs(np.amax((N)) // 2))
            u = np.round(u + np.abs(np.amax((M)) // 2))
            u = u[0].astype(np.int32)
            v = v[0].astype(np.int32)
            dist = np.zeros([np.max(v + 1), np.max(u + 1)])
            yii = yi.numpy().astype(np.int32)
            xii = xi.numpy().astype(np.int32)
            dist[yii[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)].astype(np.int32), xii[
                np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)]] = noise[
                v[np.logical_and(v < N, v > 0) * np.logical_and(u < M, u > 0)], u[
                    np.logical_and(v < N, v > 0) * np.logical_and(u < M,
                                                                  u > 0)]]  
            noise_rc = bilinear_interpolation(dist)
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(noise_rc[0:int(M / 2), :], zero)
            indices = tf.where(where)
            idx_ul1 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_ul2 = (indices[np.argmin(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])
            #
            zero = tf.constant(0, dtype=tf.double)
            where = tf.not_equal(noise_rc, zero)
            indices = tf.where(where)
            #
            idx_dr1 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 0])
            idx_dr2 = (indices[np.argmax(np.sqrt(indices[:, 0] ** 2 + indices[:, 1] ** 2)), 1])

            noise_rc = tf.image.crop_to_bounding_box(np.expand_dims(noise_rc, axis=-1), idx_ul1.numpy(),
                                                     idx_ul2.numpy(),
                                                     idx_dr1.numpy() - idx_ul1.numpy(),
                                                     idx_dr2.numpy() - idx_ul2.numpy())

            #
            Fingeprint_2 = tf.expand_dims(Fingerprint_tf, axis=-1)
            cropped_Fingeprint = tf.image.crop_to_bounding_box(Fingeprint_2, idx_ul1.numpy(), idx_ul2.numpy(),
                                                               idx_dr1.numpy() - idx_ul1.numpy(),
                                                               idx_dr2.numpy() - idx_ul2.numpy())
            cropped_Fingeprint = tf.squeeze(cropped_Fingeprint, axis=2)
            array2c = cropped_Fingeprint.numpy()
            array2c = array2c - array2c.mean()
            tilted_array2c = np.fliplr(array2c)
            tilted_array2c = np.flipud(tilted_array2c)
            norm2c = np.sum(np.power(array2c, 2))
            TAc = np.fft.fft2(tilted_array2c)
            TA_tfc = tf.expand_dims(tf.convert_to_tensor(TAc, dtype=tf.complex64), axis=0)
            XC = (crosscorr_Fingeprint_GPU(tf.cast(tf.expand_dims(noise_rc, axis=0), dtype=tf.float32),
                                           TA_tfc,
                                           norm2c, np.shape(TA_tfc)))
            ranges = [[0, 0]]
            PCE_radial = parallel_PCE(XC.numpy(), len(XC), ranges)

            print('PCE post a2 estimation: ', PCE_radial)
            PCE_array_v[2].append(PCE_radial[0])
            time_array[3].append(time.time() - start)
            mdir = {'PCE': np.asarray(PCE_array_v), 'flag': np.asarray(validity_array), 'time': np.asarray(time_array),
                    'PCE_grid': PCE_grid_search, 'time_grid': time_grid_search}
            savemat(outfile_name, mdir)
            idx_img += 1

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext
from scipy.io import FortranFile
from tqdm import tqdm
import click


@click.command()
@click.option('--mesh', type=str, default='mesh.dat')
@click.option('--res', type=str, default='cont.res')
@click.option('--center', type=float, nargs=4, default=(0,0,0,0))
@click.option('--variable', type=click.Choice(['sqrtk', 't', 'w', 'ua']), prompt=True)
@click.option('--arrow-skip', type=int, default=10)
@click.option('--attack-angle', type=float, default=4.5)
@click.option('--rad', type=float, default=9260)
def main(mesh, res, center, variable, arrow_skip, attack_angle, rad):

    attack_angle /= 180 / np.pi

    with FortranFile(mesh, 'r') as f:
        npts, _, imax, jmax, kmax, _ = f.read_ints()
        coords = f.read_reals(dtype='f4').reshape((npts, 3))

    with FortranFile(res, 'r') as f:
        data = f.read_reals(dtype='f4')
        time = data[0]
        data = data[1:].reshape((npts, 11))
        u = data[:,:3]
        tk = data[:,4]
        vtef = data[:,6]
        pt = data[:,7]


    # Physical units
    u *= 20
    indexes = np.linspace(1,npts,npts)
    tk = 20**2 * tk + 1e-5 * indexes / npts

    print('ua_max (m/s) = ', max(np.linalg.norm(u, axis=1)))
    print('w_max (m/s) = ', max(np.abs(u[:,2])))


    # Airport
    xL, yB = 1500.0, 250.0
    coords[:,0] -= center[0]
    coords[:,1] -= center[1]
    tt = (90 - center[3]) / 180 * np.pi

    xf = np.zeros((3,4), dtype='f4')
    xf[0,:] = np.array([-.5, .5, -.5, .5]) * xL
    xf[1,:] = np.array([-.5, -.5, .5, .5]) * yB
    xf[2,:] = center[2]

    cf = np.array([
        [np.cos(tt), -np.sin(tt), 0],
        [np.sin(tt), np.cos(tt), 0],
        [0, 0, 1],
    ], dtype='f4').dot(xf)


    # Terrain coordinates
    ij = np.array(list(product(range(0, imax), range(0, jmax))))
    idx_in = kmax * ij[:,0] + imax * kmax * ij[:,1]
    idx_out = ij[:,0] + imax * ij[:,1]
    xy = np.zeros((imax*jmax, 3), dtype='f4')
    xy[idx_out,:] = coords[idx_in,:]


    # Circular cone
    zb = center[2] - .5 * xL * np.tan(attack_angle)

    conal_idx_in, conal_idx_out = [], []
    for ip, kpt in tqdm(zip(idx_out, idx_in)):
        z = np.linalg.norm(xy[ip,:2]) * np.tan(attack_angle) + zb
        for kp in range(kpt, kpt + kmax - 1):
            zp = coords[kp,2]
            zp1 = coords[kp+1,2]
            if zp < z < zp1:
                break
        else:
            continue
        conal_idx_in.append(kp)
        conal_idx_out.append(ip)

    ucon = np.empty((imax*jmax,2), dtype='f4')
    ucon[:] = np.NAN
    ucon[conal_idx_out,:] = u[conal_idx_in,:2]

    scalar = np.empty((imax*jmax,), dtype='f4')
    scalar[:] = np.NAN
    if variable == 'sqrtk':
        scalar[conal_idx_out] = np.sqrt(tk[conal_idx_in])


    X = xy[:,0].reshape((jmax, imax))
    Y = xy[:,1].reshape((jmax, imax))
    C = scalar.reshape((jmax, imax))
    UX = ucon[:,0].reshape((jmax,imax))
    UY = ucon[:,1].reshape((jmax,imax))
    UA = np.linalg.norm(ucon, axis=1).reshape((jmax,imax))
    K = arrow_skip

    plt.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C), C), shading='gouraud')
    plt.colorbar()
    plt.quiver(X[::K,::K], Y[::K,::K], UX[::K,::K], UY[::K,::K], UA[::K,::K], pivot='middle')

    plt.fill(cf[0,(0,1,3,2)], cf[1,(0,1,3,2)], color='#ffffff')
    max_rad = max(np.linalg.norm(xy[conal_idx_out,:2], axis=1))
    for r in np.arange(rad, max_rad, rad):
        angs = np.linspace(0, 2 * np.pi, 100, dtype='f4')
        xs = r * np.sin(angs)
        ys = r * np.cos(angs)
        plt.plot(xs, ys, color='#ffffff', linewidth=1.5)

    plt.plot([0, -max_rad * np.cos(tt)], [0, -max_rad * np.sin(tt)], color='#ffffff', linewidth=1.5)

    plt.axes().set_aspect(1)
    plt.xlim((min(xy[conal_idx_out,0]), max(xy[conal_idx_out,0])))
    plt.ylim((min(xy[conal_idx_out,1]), max(xy[conal_idx_out,1])))
    plt.axis('off')
    plt.show()

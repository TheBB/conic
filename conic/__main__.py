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
@click.option('--out', type=str, default=None)
@click.option('--variable', type=click.Choice(['sqrtk', 't', 'w', 'ua']), prompt=True)
@click.option('--arrow-skip', type=int, default=10)
def main(mesh, res, center, variable, out, arrow_skip):

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
    tt = center[3] / 180 * np.pi

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
    aa = 4.5 / 180 * np.pi
    zb = center[2] - .5 * xL * np.tan(aa)

    xy2 = np.zeros(xy.shape, dtype='f4')
    xy2[idx_out,:] = coords[idx_in+2,:]

    conal_idx_in, conal_idx_out = [], []
    for ip, kpt in tqdm(zip(idx_out, idx_in)):
        z = np.linalg.norm(xy[ip,:2]) * np.tan(aa) + zb
        for kp in range(kpt, kpt + kmax - 1):
            zp = coords[kp,2]
            zp1 = coords[kp+1,2]
            if zp < z < zp1:
                break
        else:
            continue
        conal_idx_in.append(kp)
        conal_idx_out.append(ip)

    stk = np.zeros((imax*jmax,), dtype='f4')
    stk[conal_idx_out] = np.sqrt(tk[conal_idx_in])
    ucon = np.zeros((imax*jmax,2), dtype='f4')
    ucon[conal_idx_out,:] = u[conal_idx_in,:2]


    # Height circles
    r1, r2, x0, y0 = 5000.0, 10000.0, 0.0, 0.0
    angs = np.linspace(0, 2 * np.pi, 100, dtype='f4')
    xc = np.zeros((100, 3), dtype='f4')
    xc[:,0] = np.sin(angs)
    xc[:,1] = np.cos(angs)
    xc[:,2] = np.tan(angs)
    xc2 = xc * r2
    xc *= r1
    xc[:,2] += zb
    xc2[:,2] += zb


    # Plotting
    if out is None:
        X = xy[:,0].reshape((jmax, imax))
        Y = xy[:,1].reshape((jmax, imax))
        C = xy[:,2].reshape((jmax, imax))
        UX = ucon[:,0].reshape((jmax,imax))
        UY = ucon[:,1].reshape((jmax,imax))
        UA = np.linalg.norm(ucon, axis=1).reshape((jmax,imax))
        K = arrow_skip
        plt.axes().set_axis_bgcolor('white')
        plt.pcolormesh(X, Y, C, alpha=0.15, shading='gouraud', cmap=plt.get_cmap('terrain'))
        plt.quiver(X[::K,::K], Y[::K,::K], UX[::K,::K], UY[::K,::K], UA[::K,::K])
        plt.fill(cf[0,(0,1,3,2)], cf[1,(0,1,3,2)], color='#ffffff')
        plt.plot(xc[:,0], xc[:,1], color='#ffffff', linewidth=1.5)
        plt.plot(xc2[:,0], xc2[:,1], color='#ffffff', linewidth=1.5)
        plt.axes().set_aspect(1)
        plt.xlim((min(X.flat), max(X.flat)))
        plt.ylim((min(Y.flat), max(Y.flat)))
        plt.axis('off')
        plt.colorbar()
        plt.show()

        return


    with FortranFile(out + '.pos1', 'w') as f:
        f.write_record(coords)
    with FortranFile(out + '.pos2', 'w') as f:
        f.write_record(xy)
    with FortranFile(out + '.ter', 'w') as f:
        f.write_record(xy[:,2])
    with FortranFile(out + '.pos3', 'w') as f:
        f.write_record(cf.T)
    with FortranFile(out + '.ter3', 'w') as f:
        f.write_record(cf[2,:])
    with FortranFile(out + '.pos4', 'w') as f:
        f.write_record(xc)
    with FortranFile(out + '.pos5', 'w') as f:
        f.write_record(xc2)
    with FortranFile(out + '.niv', 'w') as f:
        f.write_record(xy2)
    with FortranFile(out + '.scl', 'w') as f:
        if variable == 'sqrtk':
            f.write_record(stk)
        elif variable == 't':
            f.write_record(pt)
        elif variable == 'w':
            f.write_record(u[:,2])
        elif variable == 'ua':
            f.write_record(np.linalg.norm(ucon, axis=1))
    with FortranFile(out + '.vec', 'w') as f:
        f.write_record(ucon)

    with open(out + '.dx', 'w') as f:
        f.write('# OpenDX header file\n\n')

        f.write('# Nodes\n')
        f.write('object 101 class array type float rank 1 shape 3 items {} lsb binary\n'.format(imax * jmax))
        f.write('data file {}.pos2, 4\n'.format(out))
        f.write('object 103 class array type float rank 1 shape 3 items {} lsb binary\n'.format(imax * jmax))
        f.write('data file {}.niv, 4\n'.format(out))
        f.write('attribute "dep" string "positions"\n\n')

        f.write('# Connectivity, structured mesh\n')
        f.write('object 102 class gridconnections counts {} {}\n'.format(jmax, imax))
        f.write('attribute "element type" string "quads"\n')
        f.write('attribute "ref" string "positions"\n\n')

        f.write('object 104 class array type float rank 0 items {} lsb binary\n'.format(imax * jmax))
        f.write('data file {}.ter, 4\n'.format(out))
        f.write('attribute "dep" string "positions"\n\n')

        f.write('# Scalar field\n')
        f.write('object 4 class array type float rank 0 items {} lsb binary\n'.format(imax * jmax))
        f.write('data file {}.scl, 4\n'.format(out))
        f.write('attribute "dep" string "positions"\n\n')

        f.write('# Velocity vector field\n')
        f.write('object 3 class array type float rank 1 shape 2 items {} lsb binary\n'.format(imax * jmax))
        f.write('data file {}.vec, 4\n'.format(out))
        f.write('attribute "dep" string "positions"\n\n')

        f.write('# Airport\n')
        f.write('object 105 class array type float rank 1 shape 3 items 4 lsb binary\n')
        f.write('data file {}.pos3, 4\n'.format(out))
        f.write('object 106 class gridconnections counts 2 2\n')
        f.write('attribute "element type" string "quads"\n')
        f.write('attribute "ref" string "positions"\n\n')

        f.write('object 107 class array type float rank 0 items 4 lsb binary\n')
        f.write('data file {}.ter3, 4\n'.format(out))
        f.write('attribute "dep" string "positions"\n\n')

        f.write('object 108 class array type float rank 1 shape 3 items 100 lsb binary\n')
        f.write('data file {}.pos4, 4\n'.format(out))
        f.write('object 109 class array type float rank 1 shape 3 items 100 lsb binary\n')
        f.write('data file {}.pos5, 4\n\n'.format(out))

        f.write('object "terrain" class field\n')
        f.write('component "positions" value 101\n')
        f.write('component "connections" value 102\n')
        f.write('component "data" value 104\n\n')

        f.write('object "scalar" class field\n')
        f.write('component "positions" value 101\n')
        f.write('component "connections" value 102\n')
        f.write('component "data" value 4\n\n')

        f.write('object "vector" class field\n')
        f.write('component "positions" value 101\n')
        f.write('component "connections" value 102\n')
        f.write('component "data" value 3\n\n')

        f.write('object "vector" class field\n')
        f.write('component "positions" value 105\n')
        f.write('component "connections" value 106\n')
        f.write('component "data" value 107\n\n')

        f.write('object "circles" class field\n')
        f.write('component "positions" value 108\n')
        f.write('object "circle2" class field\n')
        f.write('component "positions" value 109\n\n')

        f.write('END\n')

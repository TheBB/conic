import click
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import splitext
from scipy.io import FortranFile
from tqdm import tqdm
import vtk


def make_vtk(mesh, res, center):

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
    tk = 20**2 * tk

    print('ua_max (m/s) = ', max(np.linalg.norm(u, axis=1)))
    print('w_max (m/s) = ', max(np.abs(u[:,2])))

    # Airport
    coords[:,0] -= center[0]
    coords[:,1] -= center[1]

    # Grid
    coords = coords.reshape((jmax, imax, kmax, 3))
    points = vtk.vtkPoints()
    for k, j, i in product(range(kmax), range(jmax), range(imax)):
        points.InsertNextPoint(*map(float, coords[j, i, k]))
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(imax, jmax, kmax)
    grid.SetPoints(points)

    # Fields
    for (name, data) in [('u', u), ('sqrtk', np.sqrt(tk)), ('t', pt), ('w', u[:,2]),
                         ('ua', np.linalg.norm(u, axis=1))]:
        ncomps = 1 if len(data.shape) == 1 else data.shape[-1]
        data = data.reshape((jmax, imax, kmax, ncomps))
        array = vtk.vtkDoubleArray()
        array.SetNumberOfComponents(ncomps)
        array.SetNumberOfTuples(grid.GetNumberOfPoints())
        for n, (k, j, i) in enumerate(product(range(kmax), range(jmax), range(imax))):
            array.SetTuple(n, data[j, i, k, :])
        array.SetName(name)
        grid.GetPointData().AddArray(array)

    return grid


def make_plane(grid, angle, h_res, v_res):
    xmin, xmax, ymin, ymax, zmin, zmax = grid.GetBounds()
    rmin, rmax = np.Inf, np.Inf
    if angle < np.pi / 2 or angle > 3/2 * np.pi:
        rmin = min(rmin, - xmin / np.cos(angle))
        rmax = min(rmax, xmax / np.cos(angle))
    if np.pi / 2 < angle < 3/2 * np.pi:
        rmin = min(rmin, xmax / np.cos(np.pi - angle))
        rmax = min(rmax, - xmin / np.cos(np.pi - angle))
    if angle < np.pi:
        rmin = min(rmin, - ymin / np.cos(np.pi / 2 - angle))
        rmax = min(rmax, ymax / np.cos(np.pi / 2 - angle))
    if np.pi < angle:
        rmin = min(rmin, ymax / np.cos(3/2 * np.pi - angle))
        rmax = min(rmax, - ymin / np.cos(3/2 * np.pi - angle))

    rmin -= 1000
    rmax -= 1000

    nrpts = int(np.ceil((rmin + rmax) / h_res))
    rpts = np.linspace(-rmin, rmax, nrpts)
    nzpts = int(np.ceil((zmax - zmin) / v_res))
    zpts = np.linspace(zmin, zmax, nzpts)
    points = vtk.vtkPoints()
    for r, z in product(rpts, zpts):
        points.InsertNextPoint(r * np.cos(angle), r * np.sin(angle), z)

    plane = vtk.vtkStructuredGrid()
    plane.SetDimensions(nzpts, nrpts, 1)
    plane.SetPoints(points)

    return plane, rpts, zpts


def make_cone(grid, altitude, angle, h_res):
    xmin, xmax, ymin, ymax, _, zmax = grid.GetBounds()
    rmax = (zmax - altitude) / np.tan(angle)

    xmin = max(xmin, -rmax) + 1000
    xmax = min(xmax, rmax) - 1000
    ymin = max(ymin, -rmax) + 1000
    ymax = min(ymax, rmax) - 1000

    nxpts = int(np.ceil((xmax - xmin) / h_res))
    xpts = np.linspace(xmin, xmax, nxpts)
    nypts = int(np.ceil((ymax - ymin) / h_res))
    ypts = np.linspace(ymin, ymax, nypts)
    points = vtk.vtkPoints()
    for x, y in product(xpts, ypts):
        z = np.sqrt(x**2 + y**2) * np.tan(angle) + altitude
        points.InsertNextPoint(x, y, z)

    cone = vtk.vtkStructuredGrid()
    cone.SetDimensions(nypts, nxpts, 1)
    cone.SetPoints(points)

    return cone, xpts, ypts


def interpolate(original, new):
    filt = vtk.vtkProbeFilter()
    filt.SetSourceData(original)
    filt.SetInputData(new)
    filt.Update()
    return filt.GetStructuredGridOutput()


def extract(grid, name, shape):
    array = grid.GetPointData().GetArray(name)
    ncomps = array.GetNumberOfComponents()
    ret = np.zeros((np.prod(shape), ncomps))
    for n in range(len(ret)):
        ret[n,:] = array.GetTuple(n)
    new_shape = list(shape) + [ncomps]
    return ret.reshape(new_shape)


def planar_plot(rpts, zpts, field, vr, vz, u, altitude, length, angle, radius, aspect, K, out):
    plt.contourf(rpts, zpts, field[:,:,0].T)
    plt.plot([-length/2, length/2], [altitude]*2, color='w', linewidth=5)
    plt.colorbar(fraction=0.05, aspect=1.5/aspect/0.05)

    h = altitude - (rpts[0] + length/2) * np.sin(angle)
    plt.plot([rpts[0], -length/2], [h, altitude], color='w', linewidth=2)

    # mask = vz == 0.0
    # vr = np.ma.masked_where(mask, vr)[::K,::K]
    # vz = np.ma.masked_where(mask, vz)[::K,::K]
    # plt.quiver(rpts[::K], zpts[::K], vr.T, vz.T, u[::K,::K].T, pivot='middle')

    plt.xlim((rpts[0], rpts[-1]))
    plt.ylim((zpts[0], zpts[-1]))

    plt.axes().set_aspect(aspect)
    plt.axis('off')
    if out:
        plt.savefig(out + '_planar.png', bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.close('all')


def conal_plot(xpts, ypts, field, velocity, runway, angle, radius, K, out):
    plt.contourf(xpts, ypts, field[:,:,0].T)
    plt.colorbar()

    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    xf = np.zeros((2,4), dtype='f4')
    xf[0,:] = np.array([-.5, .5, .5, -.5]) * runway[0]
    xf[1,:] = np.array([-.5, -.5, .5, .5]) * runway[1]
    xf = rot.dot(xf)

    plt.fill(xf[0,:], xf[1,:], color='#ffffff')

    root = (xf[0,:] + xf[1,:]) / 2
    rmax = 2 * max(-xpts[0], xpts[-1], -ypts[0], ypts[-1])
    plt.plot([root[0], root[0] - rmax*np.cos(angle)],
             [root[1], root[1] - rmax*np.sin(angle)], color='w', linewidth=2)

    for r in np.arange(radius, rmax, radius):
        angs = np.linspace(-np.pi/2, np.pi/2, 100)
        pts = np.hstack((
            np.array([runway[0] / 2 + r * np.cos(angs), r * np.sin(angs)]),
            np.array([-runway[0] / 2 - r * np.cos(angs), -r * np.sin(angs)]),
            np.array([[runway[0] / 2 + r * np.cos(angs[0])], [r * np.sin(angs[0])]]),
        ))
        pts = rot.dot(pts)
        plt.plot(pts[0,:], pts[1,:], color='#ffffff', linewidth=2)

    plt.xlim((xpts[0], xpts[-1]))
    plt.ylim((ypts[0], ypts[-1]))

    mask = velocity == 0.0
    v = np.ma.masked_where(mask, velocity)[::K,::K,:]
    u = np.linalg.norm(v, axis=2)
    plt.quiver(xpts[::K], ypts[::K], v[...,0].T, v[...,1].T, u.T, pivot='middle')

    plt.axes().set_aspect(1)
    plt.axis('off')
    if out:
        plt.savefig(out + '_conal.png', bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.close('all')


@click.command()
@click.option('--mesh', type=str, default='mesh.dat')
@click.option('--res', type=str, default='cont.res')
@click.option('--center', type=float, nargs=4, default=(0,0,0,0))
@click.option('--variable', type=click.Choice(['sqrtk', 't', 'w', 'ua']), prompt=True)
@click.option('--arrow-skip-conal', type=int, default=15)
@click.option('--arrow-skip-planar', type=int, default=30)
@click.option('--attack-angle', type=float, default=4.5)
@click.option('--rad', type=float, default=7408)
@click.option('--h-res', type=float, default=100.0)
@click.option('--v-res', type=float, default=10.0)
@click.option('--runway', type=float, nargs=2, default=(1500, 250))
@click.option('--aspect', type=float, default=5)
@click.option('--show/--no-show', default=False)
def main(mesh, res, center, variable, arrow_skip_conal,
         arrow_skip_planar, attack_angle, rad,
         h_res, v_res, runway, aspect, show):

    attack_angle /= 180 / np.pi
    approach_angle = ((90 - center[3]) % 360) / 180 * np.pi

    original_grid = make_vtk(mesh, res, center)

    if show:
        out = None
    else:
        out, _ = splitext(res)

    plane, rpts, zpts = make_plane(original_grid, approach_angle, h_res, v_res)
    plane = interpolate(original_grid, plane)
    field = extract(plane, variable, (len(rpts), len(zpts)))
    velocity = extract(plane, 'u', (len(rpts), len(zpts)))
    norm = np.array([np.cos(approach_angle), np.sin(approach_angle), 0])
    vr = np.tensordot(velocity, norm, axes=1)
    vz = velocity[...,2]
    u = extract(plane, 'ua', (len(rpts), len(zpts)))
    planar_plot(rpts, zpts, field, vr, vz, u, center[2], runway[0],
                attack_angle, rad, aspect, arrow_skip_planar, out)

    cone, xpts, ypts = make_cone(original_grid, center[2], attack_angle, h_res)
    cone = interpolate(original_grid, cone)
    field = extract(cone, variable, (len(xpts), len(ypts)))
    velocity = extract(cone, 'u', (len(xpts), len(ypts)))
    conal_plot(xpts, ypts, field, velocity, runway, approach_angle, rad, arrow_skip_conal, out)

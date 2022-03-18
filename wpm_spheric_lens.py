import numpy as np
from numpy.fft import fftfreq, fft2, ifft2
from tqdm import tqdm
import matplotlib.pyplot as plt


def wpm(n, wavelength, mfd, resolution):
    num_z, num_y, num_x = n.shape
    dx, dy, dz = 1 / np.array(resolution)
    sx, sy, sz = dx * num_x, dy * num_y, dz * num_z
    xs, ys, zs = np.linspace(-sx / 2., sx / 2., num_x), np.linspace(-sy / 2, sy / 2, num_y), np.linspace(0, sz, num_z)
    print('area: {sx:.1f},{sy:.1f},{sz:.1f}'.format(sx=sx, sy=sy, sz=sz))
    print('grid: {num_x:d},{num_y:d},{num_z:d}'.format(num_x=num_x, num_y=num_y, num_z=num_z))
    print('dxyz: {dx:.2f},{dy:.2f},{dz:.2f}'.format(dx=dx, dy=dy, dz=dz))
    print('lambda={lam:.2f}'.format(lam=wavelength))
    print('lambda/dxyz: {dx:.2f},{dy:.2f},{dz:.2f}'.format(dx=wavelength / dx, dy=wavelength / dy, dz=wavelength / dz))

    E_xyz = np.zeros((num_z, num_y, num_x), dtype=np.complex64)
    E_kxky = np.zeros((num_y, num_x), dtype=np.complex64)

    def gaussian(x, y, sig_x, sig_y, A0):
        return A0 * np.exp(-(x / sig_x) ** 2 - (y / sig_y) ** 2)

    E_xyz[0, :, :] = gaussian(x[None, :], y[:, None], mfd, mfd, 1)

    extent = [y[0], y[-1], x[0], x[-1]]
    plt.imshow(np.abs(E_xyz)[0].T, extent=extent)
    plt.colorbar()
    plt.show()

    k_x, k_y = fftfreq(num_x, d=dx) * 2 * np.pi, fftfreq(num_y, d=dy) * 2 * np.pi
    k_xy2 = 0j + (k_x ** 2)[None, :] + (k_y ** 2)[:, None]

    for z_i in tqdm(range(num_z - 1)):
        E_kxky[...] = fft2(E_xyz[z_i, ...])
        for n_ in np.unique(n[z_i, ...]):
            indices = n[z_i, ...] == n_
            E_xyz[z_i + 1][indices] = \
                ifft2(np.exp(1j * np.sqrt((2 * np.pi / wavelength * n_) ** 2 - k_xy2) * dz) * E_kxky)[indices]

    extent = [y[0], y[-1], x[0], x[-1]]
    plt.imshow(np.abs(E_xyz)[-1].T, extent=extent)
    plt.colorbar()
    plt.show()

    extent = [z[0], z[-1], x[0], x[-1]]
    plt.contour(n[:, len(y) // 2, :].T, extent=extent)
    plt.imshow(np.abs(E_xyz)[:, len(y) // 2, :].T, extent=extent)
    plt.colorbar()
    plt.show()

    extent = [z[0], z[-1], y[0], y[-1]]
    plt.contour(n[:, :, len(x) // 2].T, extent=extent)
    plt.imshow(np.abs(E_xyz)[:, :, len(x) // 2].T, extent=extent)
    plt.colorbar()
    plt.show()


xy_dist = 25

x = np.linspace(-xy_dist, xy_dist, 300)
y = np.linspace(-xy_dist, xy_dist, 300)
z = np.linspace(0, 70, 1000)

zz = z[:, None, None]
yy = y[None, :, None]
xx = x[None, None, :]

n = np.where((abs((xx / 2) ** 2 + yy ** 2) + zz ** 2 <= 15 ** 2), 1.53, 1)

wpm(n, 1.55, 5.2, [len(xyz) / (xyz[-1] - xyz[0]) for xyz in [x, y, z]])

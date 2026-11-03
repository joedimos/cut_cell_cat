"""Build MITgcm, run a closed linear basin, and compare native MDS snapshots."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutcell.research.mitgcm import BarotropicGrid, ghost_kernel, half_plane_fractions, reflected_ghost

CASE = Path(__file__).resolve().parent / 'mitgcm_case'
PIN = 'fef697a599dd2ad9918d2876d17b68764d32f871'
DT = 15.
NSTEPS = 5


def run_logged(command, directory, log):
    with (directory / log).open('w') as stream:
        completed = subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{command[0]} failed. See {directory / log}')


def run_basin(executable, directory, *, rest):
    directory.mkdir()
    for name in ('data', 'data.pkg', 'eedata'):
        shutil.copyfile(CASE / name, directory / name)
    bathy = np.zeros((7, 8))
    bathy[1:-1, 1:-1] = -16
    bathy[1:-1, 1] = -4
    bathy[1, 2:-1] = -8
    bathy[3, 4] = 0
    eta = np.where(bathy < 0, 0.04, 0.) if rest else np.zeros_like(bathy)
    if not rest:
        eta[2, 2], eta[4, 5] = 0.07, -0.02
    for name, array in (('bathy', bathy), ('eta', eta)):
        array.astype('>f8').tofile(directory / f'{name}.bin')
    run_logged([str(executable)], directory, 'run.log')
    if 'PROGRAM MAIN: Execution ended Normally' not in (directory / 'run.log').read_text():
        raise RuntimeError(f'MITgcm did not finish normally. See {directory / "run.log"}')


def compare_native(directory, rdmds, *, rest):
    def read(name, *args):
        return rdmds(str(directory / name), *args)

    grid = BarotropicGrid.from_mds(read)
    hfac = read('hFacC')
    if not np.any((hfac > 0) & (hfac < 1)):
        raise AssertionError('native basin must exercise partial bottom cells')

    def state(step):
        return (read('Eta', step)[grid.wet],
                grid.pack_velocity(read('U', step), read('V', step)))

    eta, velocity = state(0)
    initial_eta = eta.copy()
    if not rest and np.ptp(eta) <= 0:
        raise AssertionError('initial free-surface pulse was not loaded')
    eta_error, velocity_error, native_continuity_error, native_volume_drift = 0., 0., 0., 0.
    for step in range(1, NSTEPS + 1):
        expected_eta, expected_velocity, _, _ = grid.step(eta, velocity, DT)
        observed_eta, observed_velocity = state(step)
        eta_error = max(eta_error, np.max(np.abs(observed_eta - expected_eta)))
        velocity_error = max(velocity_error, np.max(np.abs(observed_velocity - expected_velocity)))
        native_residual = (grid.area * (observed_eta - eta)
                           - DT * (grid.boundary @ (grid.cross_section * observed_velocity)))
        native_continuity_error = max(native_continuity_error, np.max(np.abs(native_residual / grid.area)))
        native_volume_drift = max(native_volume_drift, abs(grid.area @ (observed_eta - initial_eta)))
        if rest:
            np.testing.assert_allclose(observed_eta, initial_eta, atol=2e-12, rtol=0)
            np.testing.assert_allclose(observed_velocity, 0., atol=2e-12, rtol=0)
        eta, velocity = observed_eta, observed_velocity
    if not rest and np.max(np.abs(eta - initial_eta)) < 0.001:
        raise AssertionError('native pulse failed to propagate')
    assert eta_error < 2e-11, eta_error
    assert velocity_error < 2e-11, velocity_error
    assert native_continuity_error < 2e-11, native_continuity_error
    assert native_volume_drift < 1e-6, native_volume_drift
    return {'steps': NSTEPS, 'wet_cells': int(grid.area.size), 'open_faces': int(grid.distance.size),
            'native_partial_bottom_cells': int(np.count_nonzero((hfac > 0) & (hfac < 1))),
            'max_surface_difference_m': float(eta_error),
            'max_velocity_difference_m_s': float(velocity_error),
            'max_native_continuity_defect_m': float(native_continuity_error),
            'max_native_volume_drift_m3': float(native_volume_drift)}


def oblique_case():
    # Native masks have a dry exterior ring. Inside, an independent diagonal
    # wall produces cells with tiny storage and partial shared face openings.
    x = np.arange(9, dtype=float)
    y = np.arange(8, dtype=float)
    area, west, south = half_plane_fractions(x, y, (1., 1.), 8.99)
    wet = np.zeros(area.shape)
    wet[1:-1, 1:-1] = 1.
    hfac_w, hfac_s = np.zeros_like(wet), np.zeros_like(wet)
    hfac_w[:, 1:] = wet[:, :-1] * wet[:, 1:]
    hfac_s[1:, :] = wet[:-1, :] * wet[1:, :]
    metrics = np.ones_like(wet)
    grid = BarotropicGrid(rac=metrics, dxg=metrics, dyg=metrics, dxc=metrics, dyc=metrics,
                         drf=[2.], hfac_c=wet[None], hfac_w=hfac_w[None], hfac_s=hfac_s[None],
                         area_fraction=area, west_aperture=west, south_aperture=south)
    eta = np.linspace(-0.02, 0.05, grid.area.size)
    velocity = np.zeros(grid.distance.size)
    before = grid.energy(eta, velocity)
    next_eta, next_velocity, flux, continuity = grid.step(eta, velocity, DT)
    after = grid.energy(next_eta, next_velocity)
    energy_defect = after - before + grid.energy(next_eta - eta, next_velocity - velocity)
    a, q, bc = grid.quotient(np.indices(area.shape)[1][grid.wet] // 2)
    np.testing.assert_array_equal(a @ grid.boundary, bc @ q)
    np.testing.assert_allclose(a @ (grid.area * (next_eta - eta)), DT * (bc @ (q @ flux)), atol=2e-12, rtol=0)
    volume = grid.area * (grid.depth + eta)
    # The linear pressure step has no donor-CFL bound. With its transport
    # frozen, two tracer substeps satisfy that separate condition here.
    next_volume, tracer = volume, np.full(grid.area.size, 0.27)
    for _ in range(2):
        next_volume, tracer = grid.tracer_step(next_volume, tracer, flux, DT / 2)
    np.testing.assert_allclose(next_volume, grid.area * (grid.depth + next_eta), atol=2e-12, rtol=0)
    np.testing.assert_allclose(tracer, 0.27, atol=2e-15, rtol=0)
    stiffness = (grid.boundary * (grid.cross_section / grid.distance)) @ grid.boundary.T
    coarse_area = a @ grid.area
    restriction = (a * grid.area) / coarse_area[:, None]
    fine_generator = -stiffness / grid.area[:, None]
    coarse_generator = -(a @ stiffness @ a.T) / coarse_area[:, None]
    closure_defect = np.max(np.abs(restriction @ fine_generator - coarse_generator @ restriction))
    assert closure_defect > 1e-3
    rest_eta, rest_velocity, _, _ = grid.step(np.full(grid.area.size, 0.04), velocity, DT)
    np.testing.assert_allclose(rest_eta, 0.04, atol=2e-12, rtol=0)
    np.testing.assert_allclose(rest_velocity, 0., atol=2e-12, rtol=0)
    ghost = ghost_kernel(0.25, 0.5, [True, True, True, False], [True, False, True, False])
    state = reflected_ghost(ghost, (1., 1.), [0.04, 5., 0.04, -2.], [-2., 5., -1., -2.], np.zeros((4, 2)))
    np.testing.assert_allclose(state['surface'], 0.04, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(state['discharge'], 0.)
    assert abs(energy_defect) < 2e-12 and after < before
    return {'min_active_area_fraction': float(np.min(grid.area)),
            'max_cell_volume_defect_m3': float(np.max(np.abs(continuity))),
            'backward_euler_energy_identity_defect': float(energy_defect),
            'chain_map_exact': True, 'pressure_operator_coarse_closure_defect_per_m': float(closure_defect),
            'tracer_substeps': 2, 'max_constant_tracer_difference': float(np.max(np.abs(tracer - 0.27)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mitgcm', type=Path, help=f'MITgcm checkout, tested at {PIN}')
    parser.add_argument('workdir', type=Path, help='a new working directory for compilation and MDS output')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    upstream, workdir = args.mitgcm.resolve(), args.workdir.resolve()
    build = workdir / 'build'
    workdir.mkdir(parents=True)
    build.mkdir()
    optfile = upstream / 'tools' / 'build_options' / 'linux_amd64_gfortran'
    run_logged([str(upstream / 'tools' / 'genmake2'), '-rootdir', str(upstream),
                '-mods', str(CASE), '-of', str(optfile), '-ieee'], build, 'genmake.log')
    run_logged(['make', 'depend'], build, 'depend.log')
    run_logged(['make', '-j', str(args.jobs)], build, 'make.log')
    sys.path.insert(0, str(upstream / 'utils' / 'python' / 'MITgcmutils'))
    from MITgcmutils import rdmds

    report = {'mitgcm_commit': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'],
                                                       text=True).strip()}
    for name in ('pulse', 'rest'):
        directory = workdir / name
        run_basin(build / 'mitgcmuv', directory, rest=name == 'rest')
        report[name] = compare_native(directory, rdmds, rest=name == 'rest')
    report['oblique_cut_extension'] = oblique_case()
    result = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (workdir / 'report.json').write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()

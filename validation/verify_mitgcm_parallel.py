"""Verify stock three-layer cut geometry and decomposition-independent pickups."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from compare_mitgcm import PIN, run_logged
from cutcell.research.mitgcm import half_plane_fractions
from verify_native_cutcell import audit_faces, channel, compile_case, storage, write_binary

STEPS = 6
SPLIT = 3
STOCK_FIELDS = ('Eta', 'U', 'V', 'W', 'T', 'S')
SWE_FIELDS = ('CCDepth', 'CCQx', 'CCQy', 'CCTracer')
FLUX_FIELDS = ('CCVolW', 'CCVolS', 'CCTrW', 'CCTrS')


def run(executable, directory, ranks):
    command = [str(executable)]
    if ranks > 1:
        command = ['mpiexec', '-n', str(ranks), *command]
    run_logged(command, directory, 'run.log')
    log = (directory / ('STDOUT.0000' if ranks > 1 else 'run.log')).read_text()
    if 'PROGRAM MAIN: Execution ended Normally' not in log:
        raise RuntimeError(f'MITgcm did not finish normally. See {directory / "run.log"}')


def setup(directory, mode, spacing, bed, eta, geometry, *, first=0, steps=STEPS, passive=None, tracer=None):
    directory.mkdir()
    ny, nx = bed.shape
    dx, dy = spacing
    stock = mode == 3
    native = mode != 2
    dt = 15. if native else 0.2
    free_surface = (' nonlinFreeSurf=4, select_rStar=2, staggerTimeStep=.TRUE.,\n'
                    ' tempAdvScheme=1, saltAdvScheme=1,\n'
                    ' diffKhT=0., diffKhS=0., diffKrT=0., diffKrS=0.,\n'
                    ' tempForcing=.FALSE., saltForcing=.FALSE.,\n' if stock else '')
    data = f''' &PARM01
 eosType='LINEAR', tAlpha={0.0002 if stock else 0.}, sBeta=0.,
 gravity=9.81, gBaro=9.81, f0=0., beta=0.,
 momAdvection=.FALSE., momViscosity=.FALSE., momForcing=.FALSE.,
 tempStepping={'.TRUE.' if stock else '.FALSE.'}, saltStepping={'.TRUE.' if stock else '.FALSE.'},
 implicitFreeSurface=.TRUE., exactConserv=.TRUE.,
 implicDiv2Dflow=1., implicSurfPress=1., hFacMin=0.,
 readBinaryPrec=64, writeBinaryPrec=64, useSingleCpuIO=.TRUE.,
{free_surface} &
 &PARM02
 cg2dMaxIters=200, cg2dTargetResidual=1.E-14,
 &
 &PARM03
 nIter0={first}, nTimeSteps={steps}, deltaT={dt},
 dumpFreq={dt if native else 0.}, pChkptFreq={SPLIT * dt if native else 0.},
 doAB_onGtGs=.FALSE.,
 &
 &PARM04
 usingCartesianGrid=.TRUE., delX={nx}*{dx}, delY={ny}*{dy},
 delR={'4.,5.,7.' if stock else '16.'},
 &
 &PARM05
 bathyFile='bathy.bin', pSurfInitFile='eta.bin',
 {"hydrogThetaFile='temperature.bin', hydrogSaltFile='salt.bin'," if stock else ''}
 &
'''
    (directory / 'data').write_text(data)
    (directory / 'data.pkg').write_text(' &PACKAGES\n &\n')
    (directory / 'eedata').write_text(' &EEPARMS\n &\n')
    options = '' if stock else ' ccWriteFluxes=.TRUE., ccPickupEvery=3,'
    (directory / 'data.cutcell').write_text(f' &CUTCELL_PARM\n ccMode={mode},{options}\n /\n')
    write_binary(directory / 'bathy.bin', bed)
    write_binary(directory / 'eta.bin', eta)
    write_binary(directory / 'cut_geometry.bin', *geometry)
    if stock and first == 0:
        write_binary(directory / 'temperature.bin', passive[0])
        write_binary(directory / 'salt.bin', passive[1])
    if not stock and first == 0:
        write_binary(directory / 'cut_initial.bin', *tracer)


def compare(rdmds, reference, other, fields, iterations, *, atol):
    error = {field: 0. for field in fields}
    for step in iterations:
        for field in fields:
            left = rdmds(str(reference / field), step)
            right = rdmds(str(other / field), step)
            context = f'{field}, iteration {step}, {other.name}'
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise AssertionError(f'nonfinite state in {context}')
            if atol == 0.:
                np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64), err_msg=context)
            else:
                np.testing.assert_allclose(left, right, atol=atol, rtol=0, err_msg=context)
            error[field] = max(error[field], float(np.max(np.abs(left - right))))
    return error


def stock_inputs():
    dx, dy = 100., 125.
    nx, ny = 12, 10
    x, y = np.arange(nx + 1) * dx, np.arange(ny + 1) * dy
    area, west, south = half_plane_fractions(x, y, (1., 0.75), 865., full_faces=True)
    area[[0, -1], :] = 0.
    area[:, [0, -1]] = 0.
    area[5, 8] = 0.
    wet = area > 0.
    west[:, 1:-1] *= wet[:, :-1] & wet[:, 1:]
    south[1:-1, :] *= wet[:-1, :] & wet[1:, :]
    west[:, [0, -1]] = 0.
    south[[0, -1], :] = 0.
    j, i = np.indices(area.shape)
    depth = np.where(wet, np.where(i < 5, 7., np.where(j < 5, 12.5, 16.)), 0.)
    eta = np.where(wet, 0.07 * np.exp(-((i - 6)**2 + (j - 5)**2) / 6), 0.)
    k = np.arange(3)[:, None, None]
    temperature = 12. - 2 * k + 0.12 * i + 0.04 * j
    salinity = 34. + 0.5 * k + 0.015 * j + 0.006 * i
    return (dx, dy), -depth, eta, (area, west, south), (temperature, salinity)


def copy_stock_pickup(source, target):
    for ext in ('data', 'meta'):
        filename = f'pickup.{SPLIT:010d}.{ext}'
        shutil.copyfile(source / filename, target / filename)


def verify_stock(serial, mpi, directory, rdmds):
    directory.mkdir()
    spacing, bed, eta, geometry, passive = stock_inputs()
    baseline, parallel, resumed = (directory / name for name in ('serial', 'mpi', 'mpi_to_serial'))
    for executable, path, ranks in ((serial, baseline, 1), (mpi, parallel, 4)):
        setup(path, 3, spacing, bed, eta, geometry, passive=passive)
        run(executable, path, ranks)
    setup(resumed, 3, spacing, bed, eta, geometry, first=SPLIT, steps=STEPS-SPLIT)
    copy_stock_pickup(parallel, resumed)
    run(serial, resumed, 1)
    rank_error = compare(rdmds, baseline, parallel, STOCK_FIELDS, range(STEPS + 1), atol=3e-10)
    restart_error = compare(rdmds, parallel, resumed, STOCK_FIELDS, range(SPLIT, STEPS + 1), atol=3e-10)

    def read(name, *args):
        return rdmds(str(parallel / name), *args)

    layer = read('hFacC') * read('DRF').reshape(3, 1, 1)
    np.testing.assert_array_equal(layer.sum(axis=0), -bed)
    native_area = read('RAC')
    wet = bed < 0.
    np.testing.assert_allclose(native_area[wet], (spacing[0] * spacing[1] * geometry[0])[wet], rtol=1e-15)
    volume = native_area[None] * layer
    depth = layer.sum(axis=0)

    def budgets(step):
        stretch = 1. + np.divide(read('Eta', step), depth, where=wet, out=np.zeros_like(depth))
        volumes = volume * stretch
        return np.array([np.sum(volumes), np.sum(volumes * read('T', step)), np.sum(volumes * read('S', step))])

    budget = budgets(0)
    drift = np.max(np.abs(np.stack([budgets(step) for step in range(1, STEPS + 1)]) - budget) / budget, axis=0)
    np.testing.assert_array_less(drift, 3e-12)
    # Native tracer transport and baroclinic shear must both be exercised.
    tracer_change = float(np.max(np.abs(read('T', STEPS) - read('T', 0))))
    shear = float(np.max(np.abs(np.diff(read('U', STEPS), axis=0) * (read('hFacW')[1:] > 0.))))
    if min(tracer_change, shear) < 1e-6:
        raise AssertionError('the stock case did not exercise native tracer and layered flow')
    return {'levels': 3, 'steps': STEPS, 'mpi_ranks': 4, 'restart_iteration': SPLIT,
            'cut_cells': int(np.count_nonzero((geometry[0] > 0.) & (geometry[0] < 1.))),
            'partial_bottom_cells': int(np.count_nonzero((layer > 0.) & (layer < read('DRF').reshape(3, 1, 1)))),
            'relative_volume_temperature_salinity_budget_drift': drift.tolist(), 'max_temperature_change': tracer_change,
            'max_wet_vertical_u_difference_m_s': shear, 'serial_vs_mpi': rank_error, 'mpi_to_serial_pickup': restart_error}


def verify_linear(serial, mpi, directory, rdmds):
    directory.mkdir()
    _, geometry, (x, y) = channel(32)
    area, west, south = geometry
    area[[0, -1], :] = 0.
    area[:, [0, -1]] = 0.
    wet = area > 0.
    west[:, 1:-1] *= wet[:, :-1] & wet[:, 1:]
    south[1:-1, :] *= wet[:-1, :] & wet[1:, :]
    west[:, [0, -1]] = 0.
    south[[0, -1], :] = 0.
    bathy = np.where(wet, -16., 0.)
    eta = np.where(wet, 0.03 * np.exp(-((x + y - 0.5)**2)), 0.)
    zero = np.zeros_like(x)
    tracer = (zero, zero, zero, zero, np.where(wet, 0.4 + 0.1 * np.cos(x), 0.))
    baseline, parallel, resumed = (directory / name for name in ('serial', 'mpi', 'mpi_to_serial'))
    for executable, path, ranks in ((serial, baseline, 1), (mpi, parallel, 4)):
        setup(path, 1, (100., 100.), bathy, eta, geometry, tracer=tracer)
        run(executable, path, ranks)
    setup(resumed, 1, (100., 100.), bathy, eta, geometry, first=SPLIT, steps=STEPS-SPLIT)
    copy_stock_pickup(parallel, resumed)
    filename = f'cut_pickup.{SPLIT:010d}.bin'
    shutil.copyfile(parallel / filename, resumed / filename)
    run(serial, resumed, 1)
    fields = ('Eta', 'U', 'V', 'CCDepth', 'CCTracer')
    rank_error = compare(rdmds, baseline, parallel, fields, range(STEPS + 1), atol=3e-10)
    restart_error = compare(rdmds, parallel, resumed, fields, range(SPLIT, STEPS + 1), atol=3e-10)
    for step in range(1, STEPS + 1):
        previous, current = [storage(1e4 * area, rdmds(str(parallel / 'CCDepth'), i),
                                     rdmds(str(parallel / 'CCTracer'), i)) for i in (step - 1, step)]
        audit_faces(lambda name, i: rdmds(str(parallel / name), i), step, geometry, previous, current)
    return {'mpi_ranks': 4, 'restart_iteration': SPLIT, 'serial_vs_mpi': rank_error,
            'mpi_to_serial_pickup': restart_error}


def verify_swe(serial, mpi, directory, rdmds):
    directory.mkdir()
    edges, geometry, (x, y) = channel(32)
    dx = edges[1] - edges[0]
    active = geometry[0] > 0.
    zero = np.zeros_like(x)
    background = zero.copy()
    background[1:-1, 1:-1] = -16.
    depth = np.where(active, np.where(x+y < 0., 1., np.where(x+y == 0., 0.5, 0.)), 0.)
    tracer = (zero, depth, zero, zero, np.where(depth > 0., 0.3 + 0.1 * np.sin(x), 0.))
    baseline, parallel, resumed = (directory / name for name in ('serial', 'mpi', 'mpi_to_serial'))
    for executable, path, ranks in ((serial, baseline, 1), (mpi, parallel, 4)):
        setup(path, 2, (dx, dx), background, zero, geometry, tracer=tracer)
        run(executable, path, ranks)
    setup(resumed, 2, (dx, dx), background, zero, geometry, first=SPLIT, steps=STEPS-SPLIT)
    filename = f'cut_pickup.{SPLIT:010d}.bin'
    shutil.copyfile(parallel / filename, resumed / filename)
    run(serial, resumed, 1)
    compare(rdmds, baseline, parallel, SWE_FIELDS, range(STEPS + 1), atol=0.)
    compare(rdmds, baseline, parallel, FLUX_FIELDS, range(1, STEPS + 1), atol=0.)
    compare(rdmds, parallel, resumed, SWE_FIELDS, range(SPLIT, STEPS + 1), atol=0.)
    compare(rdmds, parallel, resumed, FLUX_FIELDS, range(SPLIT + 1, STEPS + 1), atol=0.)

    def read(name, step):
        return rdmds(str(parallel / name), step)

    previous = storage(dx**2 * geometry[0], read('CCDepth', 0), read('CCTracer', 0))
    max_residual = np.zeros(2)
    seam_volume = 0.
    for step in range(1, STEPS + 1):
        current = storage(dx**2 * geometry[0], read('CCDepth', step), read('CCTracer', step))
        _, residual = audit_faces(read, step, geometry, previous, current)
        max_residual = np.maximum(max_residual, np.max(np.abs(residual), axis=(1, 2)))
        seam_volume += np.abs(read('CCVolW', step)[:, 16]).sum() + np.abs(read('CCVolS', step)[16, :]).sum()
        previous = current
    if seam_volume < 0.1:
        raise AssertionError('wetting/drying case did not exercise MPI seam transfers')

    # A validly shaped changed fraction must be rejected before stepping.
    bad = directory / 'changed_geometry'
    altered = tuple(field.copy() for field in geometry)
    j, i = np.argwhere(altered[0] == 1.)[0]
    altered[0][j, i] = 0.95
    setup(bad, 2, (dx, dx), background, zero, altered, first=SPLIT, steps=1)
    shutil.copyfile(parallel / filename, bad / filename)
    bad_run = subprocess.run([str(serial)], cwd=bad, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    diagnostic = bad_run.stdout + (bad / 'STDERR.0000').read_text()
    if not bad_run.returncode or 'cut-cell pickup geometry changed' not in diagnostic:
        raise AssertionError('incompatible cut-cell pickup was accepted')
    return {'mpi_ranks': 4, 'restart_iteration': SPLIT, 'serial_mpi_bitwise': True, 'mpi_to_serial_pickup_bitwise': True,
            'sum_absolute_mpi_seam_transfers_m3': float(seam_volume), 'local_water_tracer_residual': max_residual.tolist(),
            'rejected_changed_geometry': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mitgcm', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--jobs', default=4, type=int)
    args = parser.parse_args()
    upstream, work = args.mitgcm.resolve(), args.workdir.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '-uno'], text=True)
    if revision != PIN or changed or args.jobs < 1:
        parser.error(f'expected a clean checkout at {PIN} and a positive build job count')
    sys.path.insert(0, str(upstream / 'utils/python/MITgcmutils'))
    from MITgcmutils import rdmds

    work.mkdir(parents=True)
    report = {'mitgcm_commit': revision}
    for name, verifier, nx, ny, levels in (('stock', verify_stock, 12, 10, 3), ('wetting_drying', verify_swe, 32, 32, 1)):
        serial = compile_case(upstream, work / f'{name}_serial', nx, ny, args.jobs, levels=levels)
        mpi = compile_case(upstream, work / f'{name}_mpi', nx, ny, args.jobs, levels=levels, ranks=(2, 2))
        report[name] = verifier(serial, mpi, work / name, rdmds)
        if levels == 1:
            report['linear'] = verify_linear(serial, mpi, work / 'linear', rdmds)
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (work / 'report.json').write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()

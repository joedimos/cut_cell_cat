"""Build and verify the experiment-local native MITgcm cut-cell option."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from compare_mitgcm import CASE, DT, NSTEPS, PIN, build_model, run_logged
from cutcell.research.mitgcm import BarotropicGrid, half_plane_fractions

OVERLAY = Path(__file__).resolve().parent / 'mitgcm_cutcell'
GRAVITY = 9.81


def prepare_overlay(upstream, directory, nx, ny, *, levels=1, ranks=(1, 1)):
    shutil.copytree(OVERLAY, directory)
    px, py = ranks
    if nx % px or ny % py or min(nx // px, ny // py) < 2:
        raise ValueError('the MPI process grid must divide the domain with two cells per tile axis')
    size = (CASE / 'SIZE.h').read_text().replace('sNx=8, sNy=7', f'sNx={nx // px}, sNy={ny // py}')
    size = size.replace('nPx=1, nPy=1, Nr=1', f'nPx={px}, nPy={py}, Nr={levels}')
    (directory / 'SIZE.h').write_text(size)
    shutil.copyfile(CASE / 'packages.conf', directory / 'packages.conf')
    if levels > 1:
        cpp = (upstream / 'model/inc/CPP_OPTIONS.h').read_text()
        if cpp.count('#undef NONLIN_FRSURF') != 1:
            raise ValueError('unexpected MITgcm nonlinear free-surface option')
        (directory / 'CPP_OPTIONS.h').write_text(cpp.replace('#undef NONLIN_FRSURF', '#define NONLIN_FRSURF'))
    hooks = {
        'initialise_fixed.F': ('      CALL INI_MASKS_ETC( myThid )',
                               '      CALL INI_MASKS_ETC( myThid )\n      CALL CC_SETUP( myThid )'),
        'main_do_loop.F': ('        CALL FORWARD_STEP( iloop, myTime, myIter, myThid )',
                           '        CALL CC_FORWARD_STEP( iloop, myTime, myIter, myThid )'),
        'initialise_varia.F': ('      CALL INI_FIELDS( myThid )', '      CALL CC_INI_FIELDS( myThid )'),
    }
    for filename, (original, replacement) in hooks.items():
        source = (upstream / 'model' / 'src' / filename).read_text()
        if source.count(original) != 1:
            raise ValueError(f'unexpected MITgcm hook in {filename}')
        (directory / filename).write_text(source.replace(original, replacement))


def compile_case(upstream, directory, nx, ny, jobs, *, levels=1, ranks=(1, 1)):
    directory.mkdir()
    mods = directory / 'code'
    prepare_overlay(upstream, mods, nx, ny, levels=levels, ranks=ranks)
    return build_model(upstream, directory / 'build', mods, jobs, mpi=ranks != (1, 1))


def write_binary(path, *arrays):
    with path.open('wb') as stream:
        for array in arrays:
            np.asarray(array, dtype='>f8').tofile(stream)


def run_case(executable, directory, data, bathy, eta, *, mode=None, geometry=None, initial=None, clock=None):
    directory.mkdir()
    (directory / 'data').write_text(data)
    for name in ('data.pkg', 'eedata'):
        shutil.copyfile(CASE / name, directory / name)
    write_binary(directory / 'bathy.bin', bathy)
    write_binary(directory / 'eta.bin', eta)
    if mode is not None:
        (directory / 'data.cutcell').write_text(f' &CUTCELL_PARM\n ccMode={mode}, ccWriteFluxes=.TRUE.,\n /\n')
        write_binary(directory / 'cut_geometry.bin', *geometry)
        write_binary(directory / 'cut_initial.bin', *initial)
    run_logged([str(executable)], directory, 'run.log')
    log = (directory / 'run.log').read_text()
    if 'PROGRAM MAIN: Execution ended Normally' not in log:
        raise RuntimeError(f'MITgcm did not finish normally. See {directory / "run.log"}')
    if clock is not None:
        observed = re.findall(r'CUTCELL clock\s+(\d+)\s+(\S+)', log)
        np.testing.assert_array_equal([int(i) for i, _ in observed], np.arange(1, len(clock) + 1))
        np.testing.assert_allclose([float(t) for _, t in observed], clock, atol=2e-14, rtol=0)


def storage(area, depth, tracer):
    volume = area * depth
    return np.stack((volume, volume * tracer))


def audit_faces(read, step, geometry, previous, current):
    """Check the two local balances against signed native face transfers."""
    _, west_aperture, south_aperture = geometry
    west = np.stack((read('CCVolW', step), read('CCTrW', step)))
    south = np.stack((read('CCVolS', step), read('CCTrS', step)))
    np.testing.assert_array_equal(west[:, west_aperture[:, :-1] == 0.], 0.)
    np.testing.assert_array_equal(south[:, south_aperture[:-1, :] == 0.], 0.)
    np.testing.assert_array_equal(west[:, :, 0], 0.)
    np.testing.assert_array_equal(south[:, 0, :], 0.)
    east, north = np.zeros_like(west), np.zeros_like(south)
    east[:, :, :-1] = west[:, :, 1:]
    north[:, :-1, :] = south[:, 1:, :]
    faces = np.stack((west, east, south, north))
    residual = current - previous - (west - east + south - north)
    scale = np.maximum(1., np.abs(previous) + np.abs(current) + np.abs(faces).sum(axis=0))
    np.testing.assert_array_less(np.abs(residual), 512 * np.finfo(float).eps * scale)
    return faces, residual


def cell_trace(step, j, i, fraction, old, new, faces, residual):
    return {'iteration': step, 'fortran_cell_ij': [int(i) + 1, int(j) + 1],
            'area_fraction': float(fraction[j, i]), 'old_volume_m3': float(old[0, j, i]),
            'new_volume_m3': float(new[0, j, i]), 'old_tracer_content': float(old[1, j, i]),
            'new_tracer_content': float(new[1, j, i]),
            'volume_transfers_west_east_south_north_m3': faces[:, 0, j, i].tolist(),
            'tracer_transfers_west_east_south_north': faces[:, 1, j, i].tolist(),
            'volume_residual_m3': float(residual[0, j, i]), 'tracer_residual': float(residual[1, j, i])}


def verify_linear(executable, directory, rdmds):
    x = np.r_[0., np.cumsum([90., 80., 100., 95., 110., 85., 120., 105.])]
    y = np.r_[0., np.cumsum([120., 100., 105., 85., 90., 110., 95.])]
    area, west, south = half_plane_fractions(x, y, (1., 0.82), 560., full_faces=True)
    # Match MITgcm's fixed wet mask and allow a partial-bottom island basin.
    basin = np.zeros(area.shape, dtype=bool)
    basin[1:-1, 1:-1] = True
    basin[3, 4] = False
    area *= basin
    wet = area > 0
    west[:, 1:-1] *= wet[:, :-1] & wet[:, 1:]
    south[1:-1, :] *= wet[:-1, :] & wet[1:, :]
    west[:, [0, -1]] = 0
    south[[0, -1], :] = 0
    bathy = np.where(wet, -16., 0.)
    bathy[1, wet[1]] = -8.
    bathy[wet[:, 2], 2] = -4.
    eta = np.zeros_like(bathy)
    eta[2, 5], eta[4, 5] = 0.04, -0.02
    tracer = np.where(wet, 0.2 + 0.05 * np.indices(area.shape)[1], 0.)
    control, cut = directory / 'control', directory / 'cut'
    data = (CASE / 'data').read_text()
    run_case(executable, control, data, bathy, eta)
    run_case(executable, cut, data, bathy, eta, mode=1, geometry=(area, west, south),
             initial=(eta, eta * 0., eta * 0., eta * 0., tracer), clock=DT * np.arange(1, NSTEPS + 1))

    def control_field(name, *args):
        return rdmds(str(control / name), *args)

    def cut_field(name, *args):
        return rdmds(str(cut / name), *args)

    grid = BarotropicGrid.from_mds(control_field, area_fraction=area,
                                   west_aperture=west[:, :-1], south_aperture=south[:-1, :])
    control_grid = BarotropicGrid.from_mds(control_field)
    if not np.any((area > 0) & (area < 1)) or not np.any((west > 0) & (west < 1)):
        raise AssertionError('linear basin must exercise horizontal cut cells and faces')
    np.testing.assert_array_equal(grid.wet, wet)
    np.testing.assert_allclose(cut_field('RAC')[grid.wet], grid.area, rtol=2e-15, atol=0)
    np.testing.assert_allclose(cut_field('DYG')[grid.west], (control_field('DYG') * west[:, :-1])[grid.west], rtol=2e-15, atol=0)
    np.testing.assert_allclose(cut_field('DXG')[grid.south], (control_field('DXG') * south[:-1, :])[grid.south], rtol=2e-15, atol=0)
    for metric in ('DXC', 'DYC', 'hFacC', 'hFacW', 'hFacS'):
        np.testing.assert_array_equal(cut_field(metric), control_field(metric))

    def state(step, reader, geometry):
        return reader('Eta', step)[geometry.wet], geometry.pack_velocity(reader('U', step), reader('V', step))

    np.testing.assert_array_equal(cut_field('Eta', 0), eta)
    old_eta, velocity = state(0, cut_field, grid)
    control_eta, control_velocity = state(0, control_field, control_grid)
    volume = grid.area * (grid.depth + old_eta)
    concentration = tracer[grid.wet]
    physical_area = control_field('RAC') * area
    previous = storage(physical_area, cut_field('CCDepth', 0), cut_field('CCTracer', 0))
    initial_volume = volume.sum()
    initial_content = volume @ concentration
    eta_error, velocity_error, tracer_error, transport_error = 0., 0., 0., 0.
    volume_drift, content_drift = 0., 0.
    local_budget_error = np.zeros(2)
    for step in range(1, NSTEPS + 1):
        expected_eta, expected_velocity, _, _ = grid.step(old_eta, velocity, DT)
        expected_control_eta, expected_control_velocity, _, _ = control_grid.step(control_eta, control_velocity, DT)
        control_eta, control_velocity = state(step, control_field, control_grid)
        np.testing.assert_allclose(control_eta, expected_control_eta, rtol=0, atol=2e-11)
        np.testing.assert_allclose(control_velocity, expected_control_velocity, rtol=0, atol=2e-11)
        old_eta, velocity = state(step, cut_field, grid)
        transport = grid.cross_section * velocity
        native_depth, native_c = cut_field('CCDepth', step), cut_field('CCTracer', step)
        current = storage(physical_area, native_depth, native_c)
        faces, residual = audit_faces(cut_field, step, (area, west, south), previous, current)
        local_budget_error = np.maximum(local_budget_error, np.max(np.abs(residual), axis=(1, 2)))
        flux, tracer_flux = (grid.pack_velocity(faces[0, k], faces[2, k]) for k in range(2))
        donor = np.where(transport >= 0., grid.tail, grid.head)
        np.testing.assert_allclose(flux, DT * transport, atol=2e-12, rtol=2e-15)
        np.testing.assert_allclose(tracer_flux, flux * concentration[donor], atol=2e-12, rtol=2e-15)
        if step == 1:
            cut_cells = (area > 0.) & (area < 1.)
            change = np.where(cut_cells, np.abs(current[0] - previous[0]), 0.)
            if not np.any(change > 0.):
                raise AssertionError('first linear step had no volume change in a cut cell')
            j, i = np.unravel_index(np.argmax(change), area.shape)
            trace = cell_trace(step, j, i, area, previous, current, faces, residual)
        previous = current
        volume, concentration = grid.tracer_step(volume, concentration, transport, DT)
        native_volume = current[0, grid.wet]
        native_tracer = native_c[grid.wet]
        eta_error = max(eta_error, np.max(np.abs(old_eta - expected_eta)))
        velocity_error = max(velocity_error, np.max(np.abs(velocity - expected_velocity)))
        tracer_error = max(tracer_error, np.max(np.abs(concentration - native_tracer)))
        transport_error = max(transport_error, np.max(np.abs(native_volume / grid.area - grid.depth - old_eta)))
        volume_drift = max(volume_drift, abs(native_volume.sum() - initial_volume))
        content_drift = max(content_drift, abs(native_volume @ native_tracer - initial_content))
        np.testing.assert_allclose(volume / grid.area, native_volume / grid.area, rtol=0, atol=2e-11)
    np.testing.assert_array_less([eta_error, velocity_error, tracer_error, transport_error], 2e-11)
    np.testing.assert_array_less([volume_drift, content_drift], 1e-6)
    effect = np.max(np.abs(old_eta - control_eta))
    if effect < 1e-4:
        raise AssertionError('cut geometry did not affect native dynamics')
    return {'steps': NSTEPS, 'wet_cells': int(wet.sum()), 'cut_cells': int(np.count_nonzero((area > 0) & (area < 1))),
            'max_native_surface_error_m': float(eta_error), 'max_native_velocity_error_m_s': float(velocity_error),
            'max_tracer_error': float(tracer_error), 'max_transport_depth_error_m': float(transport_error),
            'max_volume_drift_m3': float(volume_drift), 'max_tracer_content_drift': float(content_drift),
            'final_native_surface_difference_from_uncut_m': float(effect),
            'max_local_volume_budget_residual_m3': float(local_budget_error[0]),
            'max_local_tracer_budget_residual': float(local_budget_error[1]), 'cut_cell_trace': trace}


STRIP_EDGE = 4. / 3.
DAM_TIME = 1.2
SWE_DT = 0.2


def channel(size):
    edges = np.linspace(-8., 8., size + 1)
    lower = half_plane_fractions(edges, edges, (-1., 1.), -STRIP_EDGE, full_faces=True)
    upper = half_plane_fractions(edges, edges, (-1., 1.), STRIP_EDGE, full_faces=True)
    geometry = tuple(np.clip(a - b, 0., 1.) for a, b in zip(lower, upper))
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, geometry, np.meshgrid(centers, centers)


def ritter_cell_averages(edges, fraction):
    """Integrate analytic Ritter depth and discharge over each cut polygon."""
    root2 = np.sqrt(2.)
    wave = np.sqrt(GRAVITY)
    start, end = -wave * DAM_TIME, 2 * wave * DAM_TIME
    lower, upper = -STRIP_EDGE, STRIP_EDGE
    result = np.zeros((2, *fraction.shape))

    def primitive(x, y):
        s = (x + y) / root2
        fan = np.clip(s, start, end)
        width, distance = end - start, end - fan
        depth = (width**3 - distance**3) / (27 * GRAVITY * DAM_TIME**2)
        discharge = 2 * wave * depth - (width**4 - distance**4) / (54 * GRAVITY * DAM_TIME**3)
        return np.stack((np.minimum(s - start, 0.) + depth, discharge))

    for j, i in zip(*np.nonzero(fraction)):
        x0, x1 = edges[i:i + 2]
        y0, y1 = edges[j:j + 2]
        knots = [y0 - lower, y0 - upper, y1 - lower, y1 - upper]
        for front in (start * root2, end * root2):
            knots.extend((front - y0, front - y1, (front - lower) / 2, (front - upper) / 2))
        intervals = np.array([x0, *sorted(x for x in knots if x0 < x < x1), x1])
        half = np.diff(intervals) / 2
        mid = (intervals[:-1] + intervals[1:]) / 2
        # Degree four in x on each interval, including the discharge primitive.
        nodes = mid[:, None] + half[:, None] * np.array([-1., 0., 1.]) * np.sqrt(3. / 5.)
        yl, yr = np.maximum(y0, nodes + lower), np.minimum(y1, nodes + upper)
        integrand = root2 * np.where(yr > yl, primitive(nodes, yr) - primitive(nodes, yl), 0.)
        integral = (integrand @ np.array([5., 8., 5.]) / 9.) @ half
        result[:, j, i] = integral / ((x1 - x0) * (y1 - y0) * fraction[j, i])
    return result


def verify_shallow(executable, directory, rdmds, size, case):
    edges, geometry, (x, y) = channel(size)
    dx = edges[1] - edges[0]
    fraction = geometry[0]
    area = dx**2 * fraction
    active = area > 0
    along = (x + y) / np.sqrt(2.)
    bed = np.zeros_like(x) if case == 'dam' else (x + y) / 16
    if case == 'dam':
        # The reflection (x,y) -> (-y,-x) leaves the strip invariant.
        depth = np.where(along < 0., 1., np.where(along == 0., 0.5, 0.))
    else:
        surface = np.zeros_like(x)
        if case == 'runup':
            surface = np.where(along < 0., 0.06 * np.exp(-((along + 0.75) / 0.5)**2), 0.)
        depth = np.maximum(surface - bed, 0.)
    depth = np.where(active, depth, 0.)
    zero = np.zeros_like(x)
    tracer = np.where(depth > 0., 0.3 + 0.1 * np.sin(along), 0.)
    bathy = zero.copy()
    bathy[1:-1, 1:-1] = -16.
    duration = 4.8 if case == 'runup' else DAM_TIME
    steps = round(duration / SWE_DT)
    data = (CASE / 'data').read_text()
    settings = {'delX': f'{size}*{dx}', 'delY': f'{size}*{dx}', 'nTimeSteps': steps,
                'deltaT': SWE_DT, 'dumpFreq': 0.}
    for key, value in settings.items():
        data, count = re.subn(rf'(?m)^ +{key}=.*$', f' {key}={value},', data)
        if count != 1:
            raise ValueError(f'expected exactly one {key} in the MITgcm fixture')
    run_case(executable, directory, data, bathy, zero, mode=2, geometry=geometry,
             initial=(bed, depth, zero, zero, tracer), clock=SWE_DT * np.arange(1, steps + 1))

    def read(field, step):
        return rdmds(str(directory / field), step)

    def require_finite(*fields):
        if any(not np.all(np.isfinite(field)) for field in fields):
            raise AssertionError('nonfinite native shallow-water output')

    np.testing.assert_allclose(read('CCDepth', 0), depth, atol=4e-16, rtol=0)
    initial_volume = np.sum(area * depth)
    initial_content = np.sum(area * depth * tracer)
    volume_drift, tracer_drift, rest_drift, momentum_drift = 0., 0., 0., 0.
    local_budget_error = np.zeros(2)
    previous = storage(area, read('CCDepth', 0), read('CCTracer', 0))
    trace = None
    threshold = 1e-3
    new_wet, dried = 0, 0
    cut_wet, cut_dried = 0, 0
    cut_cells = (fraction > 0.) & (fraction < 1.)
    was_wet = depth > threshold
    for step in range(1, steps + 1):
        h, c = read('CCDepth', step), read('CCTracer', step)
        qx, qy = read('CCQx', step), read('CCQy', step)
        require_finite(h, c, qx, qy)
        if np.any(h < 0.) or np.any(h[~active] != 0.):
            raise AssertionError('negative water depth or transport into solid cells')
        if np.any((c[h > 0.] < 0.2 - 2e-14) | (c[h > 0.] > 0.4 + 2e-14)):
            raise AssertionError('tracer donor bound violated')
        current = storage(area, h, c)
        faces, residual = audit_faces(read, step, geometry, previous, current)
        local_budget_error = np.maximum(local_budget_error, np.max(np.abs(residual), axis=(1, 2)))
        if case == 'runup' and trace is None:
            newly_wet = cut_cells & (previous[0] == 0.) & (h > threshold)
            if np.any(newly_wet):
                j, i = np.unravel_index(np.argmax(np.where(newly_wet, current[0], 0.)), area.shape)
                trace = cell_trace(step, j, i, fraction, previous, current, faces, residual)
        previous = current
        volume_drift = max(volume_drift, abs(np.sum(area * h) - initial_volume))
        tracer_drift = max(tracer_drift, abs(np.sum(area * h * c) - initial_content))
        rest_drift = max(rest_drift, np.max(np.abs(h - depth)))
        momentum_drift = max(momentum_drift, np.max(np.hypot(qx, qy)))
        is_wet = h > threshold
        wetting = is_wet & ~was_wet
        drying = ~is_wet & was_wet
        new_wet += int(np.count_nonzero(wetting))
        dried += int(np.count_nonzero(drying))
        cut_wet += int(np.count_nonzero(wetting & cut_cells))
        cut_dried += int(np.count_nonzero(drying & cut_cells))
        was_wet = is_wet
    np.testing.assert_array_less([volume_drift, tracer_drift], 2e-11)
    result = {'dx_m': float(dx), 'duration_s': duration, 'wet_depth_threshold_m': threshold, 'wetting_events': new_wet,
              'drying_events': dried, 'cut_cell_wetting_events': cut_wet, 'cut_cell_drying_events': cut_dried,
              'minimum_cut_area_fraction': float(fraction[active].min()),
              'max_volume_drift_m3': float(volume_drift), 'max_tracer_content_drift': float(tracer_drift),
              'max_local_volume_budget_residual_m3': float(local_budget_error[0]),
              'max_local_tracer_budget_residual': float(local_budget_error[1])}
    if case == 'rest':
        np.testing.assert_array_less([rest_drift, momentum_drift], 2e-12)
        if new_wet or dried:
            raise AssertionError('a steady shoreline changed wet status')
        result.update(max_depth_drift_m=float(rest_drift), max_discharge_m2_s=float(momentum_drift))
    else:
        if new_wet == 0:
            raise AssertionError(f'{case} did not wet any initially dry cells')
        if case == 'runup':
            if trace is None or cut_dried == 0:
                raise AssertionError('run-up must wet and recede across cells cut by the wall')
            result['wetting_cell_trace'] = trace
        if case == 'dam':
            exact_h, exact_q = ritter_cell_averages(edges, fraction)
            np.testing.assert_allclose(np.sum(area * exact_h), initial_volume, atol=4e-13, rtol=0)
            channel_width = np.sqrt(2.) * STRIP_EDGE
            exact_momentum = 0.5 * GRAVITY * channel_width * DAM_TIME
            np.testing.assert_allclose(np.sum(area * exact_q), exact_momentum, atol=4e-13, rtol=0)
            along_discharge = (qx + qy) / np.sqrt(2.)
            across_discharge = (qy - qx) / np.sqrt(2.)
            for key, error in (('depth_l1_m', h - exact_h), ('streamwise_discharge_l1_m2_s', along_discharge - exact_q),
                               ('cross_stream_discharge_l1_m2_s', across_discharge)):
                result[f'area_weighted_{key}'] = float(np.sum(area * np.abs(error)) / np.sum(area))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mitgcm', type=Path, help=f'clean MITgcm checkout at {PIN}')
    parser.add_argument('workdir', type=Path, help='new directory for builds, MDS output and report')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    upstream, workdir = args.mitgcm.resolve(), args.workdir.resolve()
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    tracked_changes = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '-uno'], text=True)
    if actual != PIN or tracked_changes:
        parser.error(f'expected a clean MITgcm checkout at {PIN}')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    sys.path.insert(0, str(upstream / 'utils' / 'python' / 'MITgcmutils'))
    from MITgcmutils import rdmds

    workdir.mkdir(parents=True)
    linear = workdir / 'linear'
    report = {'mitgcm_commit': actual, 'linear': verify_linear(compile_case(upstream, linear, 8, 7, args.jobs), linear, rdmds),
              'shoreline_rest': [], 'dam_break': []}
    for size in (32, 64, 128):
        directory = workdir / f'grid_{size}'
        executable = compile_case(upstream, directory, size, size, args.jobs)
        report['shoreline_rest'].append(verify_shallow(executable, directory / 'rest', rdmds, size, 'rest'))
        report['dam_break'].append(verify_shallow(executable, directory / 'dam', rdmds, size, 'dam'))
        if size == 64:
            report['shoreline_runup'] = verify_shallow(executable, directory / 'runup', rdmds, size, 'runup')
    errors = np.array([case['area_weighted_depth_l1_m'] for case in report['dam_break']])
    rates = np.log2(errors[:-1] / errors[1:])
    if not np.all(rates > 0.35) or errors[-1] >= 0.02:
        raise AssertionError(f'Ritter depth error did not decrease at the required rate: {errors}')
    report['dam_break_observed_orders'] = rates.tolist()
    discharge = np.array([case['area_weighted_streamwise_discharge_l1_m2_s'] for case in report['dam_break']])
    if not np.all(np.diff(discharge) < 0.):
        raise AssertionError(f'Ritter streamwise discharge error did not decrease: {discharge}')
    across = np.array([case['area_weighted_cross_stream_discharge_l1_m2_s'] for case in report['dam_break']])
    if not np.all(np.diff(across) < 0.) or across[0] >= 1e-3:
        raise AssertionError(f'cross-channel discharge exceeded the convergence check: {across}')
    result = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (workdir / 'report.json').write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()

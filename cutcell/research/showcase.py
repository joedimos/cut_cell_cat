"""Reproducible synthetic audits of the Korn/WeatherNext mathematical interfaces."""
import argparse
import json
from importlib.resources import files
import numpy as np
from . import (ColumnPressureSplit, dispersion, consistent_tracer_step,
               ConservativeFluxEnsemble, field_score, pooled_crps,
               SecondOrderWeatherKernel, FunctionalGeneratorKernel,
               thin_fluid_calibration, flux_corrected_reconstruction,
               tracer_variance_diagnostics, energy_dissipation_diagnostics,
               mixing_diagnostics)
from .. import CutCellGrid
from ..categorical_numerics import Coarsening
from ..category import FiniteSet, FiniteMap, FiniteKernel, product
from ..io import atomic_json


def showcase():
    references = json.loads(files('cutcell.research').joinpath('references.json').read_text(encoding='utf-8'))

    # Korn pressure/acoustic kernels.
    grid = ColumnPressureSplit(np.linspace(0, 2, 9), np.linspace(-1, 0, 17))
    x, z = grid.x.centers[:, None], grid.z.centers[None, :]
    source = np.cos(np.pi*x/2)*np.cos(np.pi*(z+1)/2)
    split = grid.split(source)
    acoustic = grid.acoustic_stage(np.zeros((9, 16)), np.zeros((8, 17)), source, .1, 100.)

    # Conservative stochastic scalar-field example.
    scalar_grid = CutCellGrid.uniform(8)
    faces = scalar_grid.faces
    modes = np.stack([np.sin(np.pi*faces), np.sin(2*np.pi*faces)], axis=1)*.01
    modes[[0, -1]] = 0
    ensemble = ConservativeFluxEnsemble(scalar_grid, modes)
    width = scalar_grid.volumes
    cell_cosine = np.diff(np.sin(np.pi*faces))/(np.pi*width)
    initial = .5+.1*cell_cosine
    sampled = ensemble.sample(initial, .001, np.random.default_rng(1973).normal(size=(64, 2)))
    coarsening = Coarsening(scalar_grid, (0, 2, 4, 6, 8))
    coarse_members = sampled['members'] @ coarsening.restrict.T
    analytic = .5+.1*cell_cosine*np.exp(-.1*np.pi**2*sampled['actual_dt'])
    scores = field_score(sampled['members'], analytic, scalar_grid.volumes)
    pooled = pooled_crps(sampled['members'], analytic,
                         tuple((i, min(i+1, 7)) for i in range(8)),
                         reducer='mean', weights=scalar_grid.volumes)

    # Korn pseudo-density transport and information-gain diagnostics.
    transport = consistent_tracer_step(CutCellGrid.uniform(2), [1, 1], [1, 3], [0, .1, 0], 1.)
    thin = thin_fluid_calibration(1025., 9.81, 4000., 1.)
    tracer = np.array((1., 3., 2.))
    internal_flux = np.array((.2, -.1))
    reconstruction = flux_corrected_reconstruction(tracer, internal_flux, (.25, .75))
    variance = tracer_variance_diagnostics(
        (1., 1., 1.), (1., 1.01, .99), tracer, internal_flux,
        (.05, .05), (1., 1.), reconstruction, rho0=1025.)
    energy = energy_dissipation_diagnostics(
        (1., 1., 1.), (1., 1.01, .99), (.1, .1, .1), (1., 2., .5),
        (.1, .1, .1), (1., .25, 4.), (.02, .02, .02), (.01, -.02, .01))
    mixing = mixing_diagnostics(
        epsilon_b=.2, epsilon=energy['physical_dissipation'], stratification_n2=.01,
        compressibility_error=energy['compressibility_dissipation'],
        advection_error=.001, closure_residual=.0001)

    # WeatherNext second-order trajectory structure and shared functional noise.
    weather_state = FiniteSet(('low', 'high'))
    pair_state, _, _ = product(weather_state, weather_state)
    weather_transition = FiniteKernel(pair_state, weather_state,
                                      ((.9, .1), (.4, .6), (.7, .3), (.2, .8)))
    weather_model = SecondOrderWeatherKernel(weather_state, weather_transition)
    trajectory_probability = weather_model.trajectory_probability(('low', 'high'), ('high', 'low'))
    noise = FiniteSet(('z0', 'z1'))
    field_target = FiniteSet(((0, 0), (1, 1), (0, 1), (1, 0)))
    generator_domain, _, _ = product(weather_state, noise)
    generator = FunctionalGeneratorKernel(
        weather_state, noise, field_target,
        FiniteMap(generator_domain, field_target, ((0, 0), (1, 1), (0, 1), (1, 0))),
        (.25, .75))
    generator_kernel = generator.kernel()

    # Strong lumpability remains an independent coarse stochastic condition.
    fine, coarse = FiniteSet((0, 1, 2)), FiniteSet(('a', 'b'))
    partition = FiniteMap(fine, coarse, ('a', 'a', 'b'))
    kernel = FiniteKernel(fine, fine, [[.2, .3, .5], [.4, .1, .5], [.1, .2, .7]])
    coarse_kernel = FiniteKernel(coarse, coarse, [[.5, .5], [.3, .7]])

    residuals = {
        'column_solve': float(np.max(np.abs(split['column_residual']))),
        'pressure_split_identity': float(np.max(np.abs(split['split_identity_residual']))),
        'pseudo_mass': abs(transport['pseudo_mass_residual']),
        'tracer_content': abs(transport['tracer_content_residual']),
        'ensemble_mass': float(np.max(np.abs(sampled['mass_residuals']))),
        'coarse_ensemble_mass': float(np.max(np.abs(coarse_members @ coarsening.coarse_volumes-initial @ scalar_grid.volumes))),
        'strong_lumpability_example': float(np.max(np.abs(kernel.lumpability_defect(partition, coarse_kernel)))),
        'weather_generator_stochasticity': float(np.max(np.abs(generator_kernel.matrix.sum(axis=1)-1.))),
    }
    if any(not np.isfinite(v) or v > 1e-10 for v in residuals.values()):
        raise ArithmeticError('research showcase residual gate failed')

    return {
        'schema_version': 2,
        'sources': references['sources'],
        'scope': 'source-traceable finite abstractions; no trained WeatherNext model or complete AC/DC ocean circulation model',
        'residuals': residuals,
        'residual_absolute_tolerance': 1e-10,
        'acoustic_substeps': acoustic['substeps'],
        'dispersion_sweep': [dict(alpha_over_rho0=a, **dispersion(1., 3., 2., .5, a)) for a in (100., 200., 400.)],
        'korn_thin_fluid_calibration': thin,
        'korn_tracer_variance_diagnostics': variance,
        'korn_energy_dissipation_diagnostics': energy,
        'korn_mixing_diagnostics': mixing,
        'weathernext_second_order_trajectory_probability': trajectory_probability,
        'weathernext_functional_generator_kernel': generator_kernel.matrix.tolist(),
        'synthetic_scores_not_weather_skill': scores,
        'weathernext_style_pooled_crps_not_operational_skill': pooled['weighted_score'],
        'physical_tracer_content_change_not_conserved': transport['physical_content_change'],
        'forecast_members': len(sampled['members']),
        'actual_forecast_dt': sampled['actual_dt'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    try:
        result = showcase()
        if args.output:
            atomic_json(args.output, result)
            print(f'Research mathematical audit passed; report={args.output}')
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()

"""filter.py Support for particle filtering.  Separated from benettin.py

"""

from __future__ import annotations  # Enables, eg, (self: Particle
import sys
import argparse
import pickle
import copy

import numpy
import numpy.linalg
import numpy.random

import hmmds.synthetic.bounds.lorenz


class Particle:
    """A point in the Lorenz system with a box (parallelogram) and a weight.

    Args:
        x: 3-vector position
        box: 3x3 derivative matrix
        weight: Scalar

    The corners are given by x +/- box/2.

    """

    # pylint: disable=invalid-name
    def __init__(self: Particle, x, box, weight):
        self.x = x
        self.box = box
        self.weight = weight

    def step(self: Particle, time, atol):
        """Map the box forward by the interval "time"

        Args:
            time: The amount of time
            atol: Integration absolute error tolerance
        """
        self.x, self.box = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            time, self.x, self.box, atol=atol)
        assert self.box.shape == (3, 3)

    def divide(self: Particle, n_divide, U, S, VT, stretch=1.0):
        """Divide self into n_divide new particles along S_0 direction

        Args:
            n_divide: number of new particles
            U, S, VT: Singular value decomposition of self.box
            stretch: Stretch daughter boxes this much to avoid gaps
        """
        assert n_divide > 0
        S[0] /= (n_divide / stretch)
        new_box = numpy.dot(U * S, VT)
        x_step = U[:, 0] * S[0]
        new_weight = self.weight / n_divide

        back_up = int(n_divide / 2)
        result = [
            Particle(
                self.x + i * x_step,  #
                new_box,  #
                new_weight,  #
            ) for i in range(-back_up, n_divide - back_up)
        ]
        return result

    def resample(self: Particle, rescale: float, weight=1.0, rng=None):
        """Return a new box sampled from self

        Args:
            rescale: New box = old box * rescale
            weight: Weight of new box
            rng: numpy.random.Generator
        """
        # For now simply return copy of self with rescaled box.  In
        # the future I may want to draw new x from a uniform
        # distribution in self.box
        return Particle(self.x, self.box * rescale, weight)


class Filter:
    """Variant of particle filter using Lorenz equations for discrete
    observations

    Args:
        epsilon_min: Edge length of small box
        epsilon_max: Failure of linear approximation gives maximum length
        n_min: Minimum number of particles
        bins: Quatization boundaries for observations
        time_step: Integrate Lorenz this interval between samples
        sub_steps: Number of time steps between observations
        atol: Absolute error tolerance for integrator
    

    The four values: stretch, s_augment, and margin militate
    against particle exhaustion.

    stretch: Multiply the largest singular value of box by this value
        before division in Particle.divide.

    s_augment: Add this value to each singular value in
        Filter.forecast_x.  Augmentation prevents collapse of smallest
        edge of boxes and spreads particles in the contracting
        direction.

    margin: In update, don't drop particles that are within a fraction
        of the box size producing the actual observed y value.

    """

    def __init__(self: Filter, epsilon_min, epsilon_max, bins, time_step,
                 sub_steps, atol, stretch, rng):
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.bins = bins
        self.time_step = time_step
        self.sub_steps = sub_steps
        self.atol = atol
        self.particles = []
        self.stretch = stretch
        self.s_augment = (epsilon_min * 1.0e-1) / sub_steps
        self.rng = rng

    def initialize(self: Filter, initial_x: numpy.ndarray, delta: float):
        """Populate self.particles

        Args:
            initial_x: Integrate this 3-vector n_times for initial distribution
            delta: Length of edges of initial particles

        
        """
        weight = 1.0
        self.particles = [Particle(initial_x, numpy.eye(3) * delta, weight)]
        self.normalize()
        assert len(self.particles) > 0

    def forecast_x(self: Filter, time: float):
        """Map each particle forward by time.  If the largest singular
        value S[0] > epsilon_max, subdivide the particle.

        Args:
            time: Map via integrating Lorenz for this time step.

        """
        new_particles = []
        for particle in self.particles:
            particle.step(time, self.atol)
            U, S, VT = numpy.linalg.svd(particle.box)
            # Augment S to spread cloud and prevent particle
            # exhaustion.
            S += self.s_augment
            if S[0] < self.epsilon_max:
                particle.box = numpy.dot(U * S, VT)
                new_particles.append(particle)
            else:
                new_particles.extend(
                    particle.divide(int(S[0] / (self.epsilon_min)), U, S, VT,
                                    self.stretch))
        self.particles = new_particles

    def resample(self: Filter, n: int, rescale=1.0):
        """Draw n new particles from distribution implied by self.particles

        Args:
            n: Number of new particles
            rescale: New boxes = old boxes * rescale

        """
        cdf = numpy.cumsum(
            numpy.asarray([particle.weight for particle in self.particles]))
        cdf /= cdf[-1]
        new_particles = []
        for index in numpy.searchsorted(cdf, self.rng.uniform(size=n)):
            new_particles.append(self.particles[index].resample(rescale))
        self.particles = new_particles

    def update(self: Filter, y: int):
        """Delete particles that don't match y within some margin.

        Args:
            y: A scalar integer observation
        """
        # FixMe: I hope changes here will fix particle exhaustion
        new_particles = []
        margin = 0.0

        def zero():
            upper = self.bins[0]
            for particle in self.particles:
                # box_0 is the total length in the 0 direction
                box_0 = numpy.abs(particle.box[0, :]).sum()
                if particle.x[0] < upper + margin * box_0:
                    new_particles.append(particle)

        def top():
            lower = self.bins[-1]
            for particle in self.particles:
                box_0 = numpy.abs(particle.box[0, :]).sum()
                if particle.x[0] > lower - margin * box_0:
                    new_particles.append(particle)

        if y == 0:
            zero()
        elif y == len(self.bins):
            top()
        else:
            lower = self.bins[y - 1]
            upper = self.bins[y]
            for particle in self.particles:
                box_0 = numpy.abs(particle.box[0, :]).sum()
                if lower - margin * box_0 < particle.x[
                        0] < upper + margin * box_0:
                    new_particles.append(particle)
        if len(self.particles) > 0 and len(new_particles) == 0:
            print(f'In update {len(self.particles)=} {len(new_particles)=}')
        self.particles = new_particles

    def normalize(self: Filter):
        """Scale weights so that total is 1
        """
        total_weight = 0.0
        for particle in self.particles:
            total_weight += particle.weight
        for particle in self.particles:
            particle.weight /= total_weight

    def p_y(self: Filter):
        """Calculate probability mass function for possible y values.

        """
        result = numpy.zeros(len(self.bins) + 1)
        for particle in self.particles:
            y = numpy.digitize(particle.x, self.bins)
            result[y] += particle.weight
        return result

    def forward(self: Filter,
                y_ts: numpy.ndarray,
                t_start,
                t_stop,
                gamma,
                clouds=None):
        """Estimate and assign gamma[t] = p(y[t] | y[0:t]) for t from t_start to t_stop.

        Args:
            y_ts: A time series of observations
            t_start:
            t_stop:
            gamma:
            clouds: Optional dict for saving particles

        """
        for t in range(t_start, t_stop):
            y = y_ts[t]
            print(f'y[{t}]={y} {len(self.particles)=}')
            assert len(self.particles) < 1e6

            self.normalize()
            gamma[t] = self.p_y()[y]
            if clouds is not None:
                clouds[(t, 'forecast')] = copy.deepcopy(self.particles)
            self.update(y)
            if len(self.particles) == 0:
                return
            if clouds is not None:
                clouds[(t, 'update')] = copy.deepcopy(self.particles)
            for _ in range(self.sub_steps):
                self.forecast_x(self.time_step)  # Calls divide
                length = len(self.particles)
                if length > 4000:
                    self.resample(1000)
                    print(
                        f'resampled from {length} particles to {len(self.particles)=}'
                    )
        return


# Local Variables:
# mode: python
# End:

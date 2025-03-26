"""filter.py Support for particle filtering.  Separated from benettin.py

"""

from __future__ import annotations  # Enables, eg, (self: Particle

import copy

import numpy
import numpy.linalg
import numpy.random

import hmmds.synthetic.bounds.lorenz


def angles(v0, v1):
    """Resolve v1 into a part parallel to v0 and a part perpendicular
    """
    length_v1 = numpy.linalg.norm(v1)
    parallel = v0 * (v0 @ v1) / (v0 @ v0)
    perpendicular = v1 - parallel
    cosine = numpy.linalg.norm(parallel) / length_v1
    sine = numpy.linalg.norm(perpendicular) / length_v1
    return parallel, perpendicular, sine, cosine


class Particle:
    """A point in the Lorenz system with a box (parallelogram) and a weight.

    Args:
        x: 3-vector position
        box: 3x3 rows are edges.  box[0] = box[0,:] is first edge
        weight: Scalar

    The corners are given by x +/- box/2.

    """

    # pylint: disable=invalid-name
    def __init__(self: Particle, x, box, weight, argmax=-1, ratio_=-1):
        assert x.shape == (3,)
        assert box.shape == (3, 3)
        self.x = x
        self.box = box
        self.weight = weight
        self.argmax = argmax
        self.ratio_ = ratio_

    def step(self: Particle, time, atol):
        """Map the box forward by the interval "time"

        Args:
            time: The amount of time
            atol: Integration absolute error tolerance
        """
        self.x, self.box = hmmds.synthetic.bounds.lorenz.integrate_tangent(
            time, self.x, self.box, atol=atol)
        assert self.box.shape == (3, 3)

    def ratio(self: Particle):
        """Calculate a ratio of quadratic to linear velocity
        approximations.  Return the index of the edge with the largest
        ratio and the value of the ratio for that edge.

        """
        s = 10.0
        r = 28.0
        b = 8.0 / 3

        max_q_squared = -1.0
        argmax = -1
        for i, edge in enumerate(self.box):
            # q_squared is the square of the quadratic term in the
            # Taylor series for the velocity of edge[i].  q[i,j] =
            # (1/2) * edge[i].transpose \frac{\partial^2 f[j]}{\partial x^2} edge[i]
            q_squared = (edge[0] * edge[2])**2 + (edge[0] * edge[1])**2
            if q_squared > max_q_squared:
                max_q_squared = q_squared
                argmax = i
        dF = numpy.array([  # The derivative of Lorenz f wrt x
            [-s, s, 0],  #
            [r - self.x[2], -1, -self.x[0]],  #
            [self.x[1], self.x[0], -b]
        ])
        l = dF @ self.box[argmax]
        ratio = numpy.sqrt(q_squared / (l @ l))
        self.argmax = argmax
        self.ratio_ = ratio
        return argmax, ratio

    def divide(self: Particle, n_divide: int, edge_index):
        """Divide self into n_divide new particles along edge_index direction

        Args:
            n_divide: number of new particles
            edge_index: Specifies edge along which to divide
        """
        assert n_divide > 1
        x_step = self.box[edge_index] / n_divide
        new_box = numpy.empty((3, 3))
        for i, edge in enumerate(self.box):
            if i == edge_index:
                new_box[i] = x_step
            else:
                new_box[i] = angles(x_step, edge)[1]

        back_up = int(n_divide / 2)
        new_weight = self.weight / n_divide
        result = [
            Particle(
                self.x + i * x_step,  #
                new_box.copy(),  #
                new_weight,  #
                self.argmax,
                self.ratio_
            ) for i in range(-back_up, n_divide - back_up)
        ]
        return result

    def resample(self: Particle, rng, weight=1.0):
        """Return a new box sampled from self

        Args:
            rng: numpy.random.Generator
            weight: Weight of new box
        """
        new_x = self.x.copy()
        for edge in self.box:
            new_x += rng.uniform(-.5, .5) * edge
        return Particle(new_x, self.box, weight)


class Filter:
    """Variant of particle filter using Lorenz equations for discrete
    observations

    Args:
        args: Command line arguments
        bins: Boundaries for observation
        rng: Random number generator for resampling

    Use the following attributes of args:
        r_threshold: Subdivide a box if the ratio quadratic/linear > r_threshold
        r_extra: Subdivide more finely than required by r_threshold
        time_step: Integrate Lorenz this interval between samples
        atol: Absolute error tolerance for integrator
        s_augment: Small growth of box in all directions at each step

    """

    def __init__(self: Filter, args, bins, rng):
        self.r_threshold = args.r_threshold
        self.r_extra = args.r_extra
        self.edge_max = args.edge_max
        self.time_step = args.time_step
        self.resample_pair = args.resample
        self.atol = args.atol
        self.s_augment = args.s_augment
        self.margin = args.margin
        self.bins = bins
        self.rng = rng
        self.particles: list[Particle] = []

    def initialize(self: Filter, initial_x: numpy.ndarray, delta: float):
        """Populate self.particles

        Args:
            initial_x: Cheat by simply using single box for now
            delta: Length of edges of initial particles

        
        """
        weight = 1.0
        self.particles = [Particle(initial_x, numpy.eye(3) * delta, weight)]
        self.normalize()
        assert len(self.particles) > 0

    def forecast_x(self: Filter, time: float):
        """Map each particle forward by time.  If the quadratic term
        is too large, subdivide the particle.

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
            particle.box = numpy.dot(U * S, VT)
            argmax, ratio = particle.ratio()
            edge_lengths = numpy.linalg.norm(particle.box, axis=1)
            max_edge = edge_lengths.max()
            if ratio > self.r_threshold:
                n_new = int(ratio * self.r_extra / self.r_threshold)
                new_particles.extend(particle.divide(n_new, argmax))
            elif max_edge > self.edge_max:
                argmax = numpy.argmax(edge_lengths)
                n_new = int(max_edge * self.r_extra / self.edge_max)
                new_particles.extend(particle.divide(n_new, argmax))
            else:
                new_particles.append(particle)
        self.particles = new_particles

    def resample(self: Filter, n: int):
        """Draw n new particles from distribution implied by self.particles

        Args:
            n: Number of new particles

        """
        cdf = numpy.cumsum(
            numpy.asarray([particle.weight for particle in self.particles]))
        cdf /= cdf[-1]
        new_particles = []
        for index in numpy.searchsorted(cdf, self.rng.uniform(size=n)):
            new_particles.append(self.particles[index].resample(self.rng))
        self.particles = new_particles

    def update(self: Filter, y: int):
        """Delete particles that don't match y.

        Args:
            y: A scalar integer observation
        """
        new_particles = []

        def zero():
            """Use if y==0.  Keep a particle if any part of the box is
            below the bottom bin boundary.

            """
            upper = self.bins[0]
            for particle in self.particles:
                # box_0 is the total length of the box in the 0
                # direction
                box_0 = numpy.abs(particle.box[:, 0]).sum()
                if particle.x[0] - self.margin * box_0 < upper:
                    new_particles.append(particle)

        def top():
            """Use if y==top bin.  Keep a particle if any part of the
            box is above the top bin boundary.

            """
            lower = self.bins[-1]
            for particle in self.particles:
                box_0 = numpy.abs(particle.box[:, 0]).sum()
                if particle.x[0] + self.margin * box_0 > lower:
                    new_particles.append(particle)

        if y == 0:
            zero()
        elif y == len(self.bins):
            top()
        else:
            lower = self.bins[y - 1]
            upper = self.bins[y]
            for particle in self.particles:
                box_0 = numpy.abs(particle.box[:, 0]).sum()
                if lower - self.margin * box_0 < particle.x[
                        0] < upper + self.margin * box_0:
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
            y = numpy.digitize(particle.x[0], self.bins)
            result[y] += particle.weight
        assert 0.9999 < result.sum() < 1.00001
        return result

    def forward(self: Filter,
                y_ts: numpy.ndarray,
                t_range: tuple,
                gamma: numpy.ndarray,
                clouds=None):
        """Estimate and assign gamma[t] = p(y[t] | y[0:t]) for t from t_start to t_stop.

        Args:
            y_ts: A time series of observations
            t_range: (t_start, t_stop)
            gamma:
            clouds: Optional dict for saving particles

        """
        for t in range(*t_range):
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
            self.forecast_x(self.time_step)  # Calls divide
            length = len(self.particles)
            if length > self.resample_pair[0]:
                self.resample(self.resample_pair[1])
                print(
                    f'resampled from {length} particles to {len(self.particles)=}'
                )
        return


# Local Variables:
# mode: python
# End:

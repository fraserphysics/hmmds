"""em.py Illustrate EM algorithm difference between Log likelihood and Q

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import os.path
import pickle
import argparse

import numpy
import scipy.optimize

import hmm.simple


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser("Illustrate convergence of EM")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def logistic(x):
    '''The standard logistic function.
    Maps (-\infty,infty) to (0,1)
    '''
    return 1 / (1 + numpy.exp(-x))


def logit(p):
    '''Inverse logistic function
    '''
    return numpy.log(p / (1 - p))


def logistic_space(low, high, n):
    '''Calculate array of n values uniformly spaced between logit(low)
    and logit(high).  Return that array and an array of corresponding
    points between low and high.

    '''
    assert 0 < low < high < 1
    logit_values = numpy.linspace(logit(low), logit(high), n)
    values = logistic(logit_values)
    return values, logit_values


def level_set(f, hmm, center, value, n_angles=100, initial_r=0.1):
    '''Calculate a sequence of pairs for plotting a level set around a
    center

    Args:
        f: function that maps numpy pairs to real
        center: numpy pair
        value:
        n_angles: number of angles

    '''
    # Map to logit values and work there.  Map to probabilities on return

    direction = numpy.array([1.0, 0.0])
    r = initial_r

    l_center = logit(center)

    def f_r(_r):
        '''
        Args:
            _r: 
        '''
        vector = logistic(_r * direction + l_center)
        return q_u(vector[0], hmm) + q_v(vector[1], hmm) - value

    result = []
    for theta in numpy.linspace(0, 2 * numpy.pi, n_angles, endpoint=True):
        direction = numpy.array([numpy.cos(theta), numpy.sin(theta)])
        r = scipy.optimize.brentq(f_r, 0, 10)
        logit_result = r * direction + l_center
        result.append(logistic(logit_result))
    return numpy.array(result)


class Observation(hmm.simple.Observation):

    def __init__(self: Observation, py_state: numpy.ndarray, rng, untrainable):
        hmm.simple.Observation.__init__(self, py_state, rng)
        self.untrainable = set([x for x, y in untrainable])

    def reestimate(self: Observation, weight: numpy.ndarray):
        """
        Estimate new _py_state

        Args:
            weight: weight[t,s] = Prob(state[t]=s) given data and old model
        """

        old_py_state = self._py_state.copy()
        hmm.simple.Observation.reestimate(self, weight)
        for state in self.untrainable:
            self._py_state[state, :] = old_py_state[state, :]
        self._py_state.normalize()
        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)


class HMM(hmm.simple.HMM):
    """Has untrainable transitions

    Args:
        p_state_initial : Initial distribution of states
        p_state_time_average : Time average distribution of states
        p_state2state : Probability of state given state:
            p_state2state[a, b] = Prob(s[1]=b|s[0]=a)
        y_mod : Instance of class for probabilities of observations
        rng : Numpy generator with state

    KWArgs:
        untrainable_indices: List of ordered node pairs
        untrainable_values: Fixed transition probabilities

    Other variant features:

        * Untrainable state transition probabilities
    
        * likelihood method calculates probability of y[t]|y[:t] for
          all t

    """

    def __init__(self: HMM,
                 p_state_initial: numpy.ndarray,
                 p_state_time_average: numpy.ndarray,
                 p_state2state: numpy.ndarray,
                 y_mod: Observation,
                 rng: numpy.random.Generator,
                 untrainable_indices=None,
                 untrainable_values=None):
        """Option of holding some elements of p_state2state constant
        in reestimation.

        """
        hmm.simple.HMM.__init__(self,
                                p_state_initial,
                                p_state_time_average,
                                p_state2state,
                                y_mod,
                                rng=rng)
        self.untrainable_indices = untrainable_indices
        self.untrainable_values = untrainable_values

    def reestimate(self: HMM):
        """Variant can hold some self.p_state2state values constant.

        """

        hmm.simple.HMM.reestimate(self)
        if self.untrainable_indices is None or len(
                self.untrainable_indices) == 0:
            return
        self.p_state2state[self.untrainable_indices] = self.untrainable_values
        self.p_state2state.normalize()
        return

    def likelihood(self: HMM, y: numpy.ndarray) -> numpy.ndarray:
        """Calculate p(y[t]|y[:t]) for t < len(y)

        Args:
            y: An array of ints appropriate for self.y_mod.observe([y])

        Returns Prob y[t]|y[:t] for all t

        """
        self.y_mod.observe(y)
        state_likelihood = self.y_mod.calculate()
        length = len(state_likelihood)
        result = numpy.empty(length)
        last = numpy.copy(self.p_state_initial)  # p(s)
        for t in range(length):
            last *= state_likelihood[t]  # p(y,s) = p(y|s) p(s)
            last_sum = last.sum()  # p(y[t]|y[:t]) = sum_s p(y,s)
            result[t] = last_sum
            assert last_sum > 0.0
            last /= last_sum  #  p(s|y) = p(y,s)/p(y)
            self.p_state2state.step_forward(last)  # p(s[t+1])
        return result

    def uv(self: HMM):
        return numpy.array(
            [self.p_state2state[1, 0], self.y_mod._py_state[1, 0]])


def make_model(u, v, rng):
    """
    Args:
        u: Probabilty of state[1] -> state[0]
        v: Probability y=0 | state = 1
    """
    p_state_initial = numpy.array([.5, .5])
    p_state_time_average = numpy.array([.5, .5])
    p_state2state = numpy.array([[.9, .1], [u, 1 - u]])
    p_y_state = numpy.array([[.9, .1], [v, 1 - v]])
    untrainable_indices = ((0, 0), (0, 1))
    untrainable_values = (.9, .1)
    y_mod = Observation(p_y_state, rng, untrainable_indices)
    return HMM(p_state_initial, p_state_time_average, p_state2state, y_mod, rng,
               untrainable_indices, untrainable_values)


LOW = 0.05


def survey_like(n_u, n_v, y_values, rng):
    """Calculate LogLikelihood per time sample at a grid of u,v values
    """
    u_values, u_logistic = logistic_space(LOW, .3, n_u)
    v_values, v_logistic = logistic_space(LOW, .3, n_v)
    like_array = numpy.empty((n_u, n_v))
    for i_u, u in enumerate(u_values):
        for i_v, v in enumerate(v_values):
            hmm_uv = make_model(u, v, rng)
            like_array[i_u, i_v] = numpy.log(
                hmm_uv.likelihood(y_values)).sum() / len(y_values)
    return {'u_s': u_logistic, 'v_s': v_logistic, 'like_array': like_array}


def q_u(u, hmm):
    return numpy.log(u) * hmm.p_state2state[1, 0] + numpy.log(
        1 - u) * hmm.p_state2state[1, 1]


def q_v(v, hmm):
    p_y_given_s1 = hmm.y_mod._py_state[1]
    return numpy.log(v) * p_y_given_s1[0] + numpy.log(1 - v) * p_y_given_s1[1]


def q_sum(uv, hmm):
    u, v = uv
    return q_u(u, hmm) + q_v(v, hmm)


def survey_q(hmm, n_u, n_v):
    """Calculate Q_u and Q_v for hmm at a grid of u,v values

    Args:
        hmm: After training step so that hmm.p_state2state and
            hmm.y_mod._py_state reflect sums of weights

    """
    u_values, u_logistic = logistic_space(LOW, .3, n_u)
    v_values, v_logistic = logistic_space(LOW, .3, n_v)
    q_array = numpy.empty((n_u, n_v))
    for i_u, u in enumerate(u_values):
        for i_v, v in enumerate(v_values):
            q_array[i_u, i_v] = q_u(u, hmm) + q_v(v, hmm)
    return {'u_s': u_logistic, 'v_s': v_logistic, 'q_array': q_array}


def view_survey(ax, survey_dict, trajectory, key='like_array'):

    X, Y = numpy.meshgrid(survey_dict['u_s'], survey_dict['v_s'])
    CS = ax.contour(X, Y, survey_dict[key].T, levels=200)
    ax.plot(logit(trajectory[:, 0]), logit(trajectory[:, 1]), color='blue')
    ax.plot(logit(trajectory[:, 0]),
            logit(trajectory[:, 1]),
            marker='.',
            color='black',
            linestyle='',
            markersize=8)
    tick_values = numpy.array([.05, .1, .15, .2, .25])
    logit_values = logit(tick_values)
    tick_labels = [f'{x}' for x in tick_values]
    ax.set_xticks(logit_values, tick_labels)
    ax.set_yticks(logit_values, tick_labels)


def do_surveys_plot(trained_hmm, y_values, trajectory):
    '''Do surveys of log likelihood and q
    '''
    import matplotlib.pyplot as plt
    fig, (upper, lower) = plt.subplots(nrows=2)

    #l_dict = survey_u_v(60,25, y_values, rng)
    l_dict = survey_like(20, 10, y_values, trained_hmm.rng)
    q_dict = survey_q(trained_hmm, 100, 100)

    view_survey(upper, l_dict, trajectory, 'like_array')
    view_survey(lower, q_dict, trajectory, 'q_array')
    plt.show()


def plot_trajectory(trajectory, hmm, y_values):
    import matplotlib.pyplot as plt

    fig, (xy_axes, q_axes, like_axes) = plt.subplots(nrows=3)
    q_axes.set_ylabel('Q')
    like_axes.set_ylabel('Log Likelihood')
    xy_axes.plot(trajectory[:, 0], trajectory[:, 1], color='blue')
    xy_axes.plot(trajectory[:, 0],
                 trajectory[:, 1],
                 marker='.',
                 color='black',
                 linestyle='',
                 markersize=8)
    like_trajectory = []
    q_trajectory = []
    for (u, v) in trajectory:
        q_trajectory.append(q_sum((u, v), hmm))
        hmm_uv = make_model(u, v, hmm.rng)
        like_trajectory.append(
            numpy.log(hmm_uv.likelihood(y_values)).sum() / len(y_values))
    like_axes.plot(like_trajectory)
    q_axes.plot(q_trajectory)
    for uv in trajectory[0:-1]:
        new_hmm = make_model(uv[0], uv[1], hmm.rng)
        new_hmm.train(y_values, 1, display=False)
        # Get a level set centered at new uv that goes through old uv
        value = q_sum(uv, new_hmm)
        loop = level_set(q_sum, new_hmm, new_hmm.uv(), value)
        xy_axes.plot(loop[:, 0], loop[:, 1], color='b')
    plt.show()


def main(argv=None):
    """
    """

    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)
    rng = numpy.random.default_rng(7)
    true_u = .1
    true_v = .2
    true_hmm = make_model(true_u, true_v, rng)
    n = 10000
    states, y_values = true_hmm.simulate(n)
    initial_u = .001
    initial_v = .01
    hmm = make_model(initial_u, initial_v, rng)
    n_train = 10
    trajectory = numpy.empty((n_train, 2))
    for iteration in range(n_train):
        trajectory[iteration, :] = hmm.uv()
        hmm.train(y_values, 1, display=False)
    print(f'{hmm.__str__()}')
    #do_surveys_plot(hmm, y_values, trajectory)
    plot_trajectory(trajectory, hmm, y_values)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

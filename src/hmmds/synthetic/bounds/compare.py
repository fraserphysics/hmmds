import pickle

import numpy

main = pickle.load(open('main_test', 'rb'))
trial = pickle.load(open('trial_test', 'rb'))

for t in range(20):
    for s in 'forecast update'.split():
        main_particles = main['clouds'][t,s]
        trial_particles = trial['clouds'][t,s]
        assert len(main_particles) == len(trial_particles), f'{t=} {s=}'
        for particle_index in range(len(main_particles)):
            trial_particle = trial_particles[particle_index]
            main_particle = main_particles[particle_index]
            debug_message = f'''{t=} {s=} {particle_index=}
trial.x {trial_particle.x}
main.x  {main_particle.x}
diff    {trial_particle.x - main_particle.x}

trial box
{trial_particle.box}

main box
{main_particle.box}

difference
{main_particle.box-trial_particle.box}
            '''
            assert numpy.allclose(trial_particle.x, main_particle.x), debug_message
            assert numpy.allclose(trial_particle.box, main_particle.box), debug_message

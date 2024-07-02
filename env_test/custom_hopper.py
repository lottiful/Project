from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv

import re


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

    def modify_xml_for_inclination(self):
        """Modify the xml file to modify the inclination angle of the environment"""
        with open('env_test/assets/hopper.xml', 'r') as file:
            xml_content = file.read()

            # Compute the quaternion that corresponds to the angle
            alpha = np.deg2rad(self.inclination_angle)
            w = np.cos(alpha / 2)
            x = 0
            y = np.sin(alpha / 2)
            z = 0

            xml_content = re.sub(r'quat="[^"]*"', f'quat="{w} {x} {y} {z}"', xml_content)

        with open('env_test/assets/hopper.xml', 'w') as file:
            file.write(xml_content)

    def modify_inclination(self, inclination_angle):
        """Build the model with the new angle"""
        self.inclination_angle = inclination_angle
        self.modify_xml_for_inclination()
        self.build_model()

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""

        new_masses = [np.random.uniform(m - 0.5, m + 0.5) for m in self.original_masses[1:]]

        return [self.original_masses[0]] + new_masses


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        height_before = self.sim.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        #Compute the reward
        if self.inclination_angle == 0:
            reward = (posafter - posbefore) / self.dt
        else:
            reward = np.sqrt(np.square(posafter - posbefore) + np.square(height_before - height)) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()

        #Set conditions for Done
        if self.inclination_angle == 0:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        else:
            alpha = np.deg2rad(self.inclination_angle)
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7 - np.tan(alpha)*posafter) and (abs(ang) < .2))
        
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)


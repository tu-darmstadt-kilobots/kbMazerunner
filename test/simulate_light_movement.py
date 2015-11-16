#!/usr/bin/env python2

"""
    Only the light moves without any collisions.
    Allows communication with the learner to send samples.
    The light is moved based on a policy provided by the learner.

    NOTE: For now needs to be started before the learner to work correctly.
"""

import pygame
from pygame.locals import *
from pygame import draw, gfxdraw

import random

from zmq import Context, PAIR
import pickle
import importlib

from numpy import *
import numpy as np
import math


class LightMovementSimulator:
    WIDTH, HEIGHT = 1200, 600

    ZMQ_PORT = 2357

    def __init__(self):
        # pygame
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                HWSURFACE | DOUBLEBUF, 32)
        self.clock = pygame.time.Clock()

        # zqm
        context = Context()
        self.socket = context.socket(PAIR)
        self.socket.connect('tcp://localhost:{}'.format(self.ZMQ_PORT))

    def run(self):
        while True:
            msg = pickle.loads(self.socket.recv())

            if msg['message'] == 'sentPolicyModules':
                for fileName, source in msg['modules']:
                    with open(fileName, 'w') as f:
                        f.write(source)
                policyModule = importlib.import_module(msg['policyModule'])
            elif msg['message'] == 'getSamples':
                policyDict = msg['policyDict']
                self.policy = policyModule.fromSerializableDict(policyDict)

                S, A, R, S_ = self._generateSamples()

                msg = {'message': 'sentSamples',
                       'samples': (S, A, R, S_)}
                self.socket.send(pickle.dumps(msg, protocol=2))
            else:
                print('got unexpected message')

    def _generateSamples(self):
        numEpisodes = 40
        numStepsPerEpisode = 40

        numSamples = numEpisodes * numStepsPerEpisode

        goal = np.matrix([1.0, 0.5]) # center of the screen (screen is 2 x 1)
        thresh = 0.1 # m

        stepsPerSec = 60

        # s: light pos (x, y)
        # a: light movement (dx, dy)
        # r: dist(goal, s) <= thres ? 1 : 0
        S = asmatrix(empty((numSamples, 2)))
        A = asmatrix(empty((numSamples, 2)))
        R = asmatrix(empty((numSamples, 1)))
        S_ = asmatrix(empty((numSamples, 2)))

        for ep in range(numEpisodes):
            # start position
            x = random.random() * 2.0 # in [0, 2]
            y = random.random() * 1.0 # in [0, 1]

            s = asmatrix([x, y])

            for step in range(numStepsPerEpisode):
                """ user interaction and drawing """
                # handle keys
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_PLUS:
                            stepsPerSec += 5
                        elif event.key == K_MINUS:
                            stepsPerSec = np.max([1, stepsPerSec - 5])

                # draw labyrinth
                self.screen.fill((0, 0, 0, 0))

                # draw goal
                gx = int(self.HEIGHT * goal[0, 0])
                gy = int(self.HEIGHT * (1.0 - goal[0, 1]))
                gfxdraw.aacircle(self.screen, gx, gy, 5, (0, 255, 0, 255))

                # draw light
                lx = int(self.HEIGHT * s[0, 0])
                ly = int(self.HEIGHT * (1.0 - s[0, 1]))
                gfxdraw.aacircle(self.screen, lx, ly, 5, (255, 255, 0, 255))

                pygame.display.set_caption(('ep: {} - step: {} - ' +
                    'stepsPerSec: {} - goalDist: {:.2f} cm')
                        .format(ep + 1, step + 1, stepsPerSec,
                            linalg.norm(goal - s) * 100))

                pygame.display.flip()
                self.clock.tick(stepsPerSec)

                """ simulation """
                # choose action
                a = self.policy.evaluate(s)

                # take action
                s_ = s + a

                # keep position in bounds
                s_[0, 0] = np.max([0.0, np.min([2.0, s_[0, 0]])])
                s_[0, 1] = np.max([0.0, np.min([1.0, s_[0, 1]])])

                # get reward
                #r = 0
                #if linalg.norm(goal - s) <= thresh:
                #    r = 1

                r = -linalg.norm(goal - s)

                # record sample
                sampleIdx = ep * numStepsPerEpisode + step

                S[sampleIdx, :] = s
                A[sampleIdx, :] = a
                R[sampleIdx, :] = r
                S_[sampleIdx, :] = s_

                # update state
                s = s_

        return S, A, R, S_

if __name__ == '__main__':
    sim = LightMovementSimulator()
    sim.run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:01:14 2021

@author: kunal
"""

import numpy as np

class replayBuffer:
    def __init__(self, maxSize, inputShape, numActions):
        self.mem_size = maxSize
        self.memCounter = 0
        self.stateMem = np.zeros((self.mem_size, *inputShape))
        self.newStateMem = np.zeros((self.mem_size, *inputShape))
        self.actionMem = np.zeros((self.mem_size, numActions))
        self.rewardMem = np.zeros(self.mem_size)
        self.terminalMem = np.zeros(self.mem_size, dtype=np.bool)
    def storeTransition(self, state, action, reward, newState, done):
        index = self.memCounter % self.mem_size
        
        self.stateMem[index] = state
        self.actionMem[index] = action
        self.newStateMem[index] = newState
        self.rewardMem[index] = reward
        self.terminalMem[index] = done
        
        self.memCounter += 1
    def sampleBuff(self, batch_size):
        maxMem = min(self.memCounter, self.mem_size)
        
        batch = np.random.choice(maxMem, batch_size, replace=False)
        state = self.stateMem[batch]
        newState = self.newStateMem[batch]
        action = self.actionMem[batch]
        done = self.terminalMem[batch]
        reward = self.rewardMem[batch]
        
        return state, action, reward, newState, done

#%%
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, layer1Dims=512, layer2Dims=512, name='critic', checkpointDir='/Users/kunal/OneDrive - Dartmouth College/Research/Mutli-Agent-Reinforcement-Learning/Deep RL Approach to Stock Portfolio/checkpoints'):
        super(CriticNetwork, self).__init__()
        self.layer1Dims = layer1Dims
        self.layer2Dims = layer2Dims
        
        self.modelName = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = os.path.join(self.checkpointDir, self.modelName+'_ddpg.h5')
        
        self.layer1 = Dense(self.layer1Dims, activation='relu')
        self.layer2 = Dense(self.layer2Dims, activation='relu')
        self.q = Dense(1, activation=None)
    def call(self, state, action):
        actionValue = self.layer1(tf.concat([state, action], axis=1))
        actionValue = self.layer2(actionValue)
        q = self.q(actionValue)
        return q

#%%
class ActorNetwork(keras.Model):
    def __init__(self, layer1Dims=512, layer2Dims=512, numAction=3, name='actor', checkpointDir='/Users/kunal/OneDrive - Dartmouth College/Research/Mutli-Agent-Reinforcement-Learning/Deep RL Approach to Stock Portfolio/checkpoints'):
        super(ActorNetwork, self).__init__()
        self.layer1Dims = layer1Dims
        self.layer2Dims = layer2Dims
        self.numActions = numAction
        
        self.modelName = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = os.path.join(self.checkpointDir, self.modelName+'_ddpg.h5')
        
        self.layer1 = Dense(self.layer1Dims, activation='relu')
        self.layer2 = Dense(self.layer2Dims, activation='relu')
        self.mu = Dense(self.numActions, activation='tanh')
    def call(self, state):
        prob = self.layer1(state)
        prob = self.layer2(prob)

        mu = self.mu(prob)
        return mu

#%%
from tensorflow.keras.optimizers import Adam
class DDPGAgent:
    def __init__(self, inputDims, alpha=0.001, beta=0.002, env=None, gamma=0.99, numActions=3, maxSize=1000000, 
                 tau = 0.005, layer1=400, layer2=300, batchSize=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.numActions = numActions
        self.memory = replayBuffer(maxSize, inputDims, self.numActions)
        self.batchSize = batchSize
        self.noise = noise
        self.maxAction = max(env.actionSpace)
        self.minAction = min(env.actionSpace)
        
        self.actor = ActorNetwork(layer1, layer2, self.numActions)
        self.targetActor = ActorNetwork(layer1, layer2, self.numActions, name='targetActor')
        
        self.critic = CriticNetwork(layer1, layer2)
        self.targetCritic = CriticNetwork(layer1, layer2, name='targetCritic')
        
        self.actor.compile(Adam(learning_rate=alpha))
        self.critic.compile(Adam(learning_rate=beta))
        self.targetActor.compile(Adam(learning_rate=alpha))
        self.targetCritic.compile(Adam(learning_rate=beta))
        
        self.updateNetworkParams(tau =1)
    def updateNetworkParams(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.targetActor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.targetActor.set_weights(weights)
        
        weights = []
        targets = self.targetCritic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.targetCritic.set_weights(weights)
    def remember(self, state, action, reward, newState, done):
        self.memory.storeTransition(state, action, reward, newState, done)
    def saveModels(self):
        print('..... saving models .....')
        self.actor.save_weights(self.actor.checkpointFile)
        self.targetActor.save_weights(self.targetActor.checkpointFile)
        self.critic.save_weights(self.critic.checkpointFile)
        self.targetCritic.save_weights(self.targetCritic.checkpointFile)
    def loadModels(self):
        print('..... loading models .....')
        self.actor.load_weights(self.actor.checkpointFile)
        self.targetActor.load_weights(self.targetActor.checkpointFile)
        self.critic.load_weights(self.critic.checkpointFile)
        self.targetCritic.load_weights(self.targetCritic.checkpointFile)
    def chooseAction(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.numActions])
        actions = tf.clip_by_value(actions, self.minAction, self.maxAction)

        chosenAction = tf.math.argmax(actions[0])
        return chosenAction
    def learn(self):
        if self.memory.memCounter < self.batchSize:
            return
        state, action, reward, newState, done = \
            self.memory.sampleBuff(self.batchSize)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        newStates = tf.convert_to_tensor(newState, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            targetActions = self.targetActor(newStates)
            criticValue = tf.squeeze(self.targetCritic(newStates, targetActions), 1)
            criticValue = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*criticValue*(1-done)
            criticLoss = keras.losses.MSE(target, criticValue)
        criticNetworkGradient = tape.gradient(criticLoss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(criticNetworkGradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            newPolicyActions = self.actor(states)
            actorLoss = -self.critic(states, newPolicyActions)
            actorLoss = tf.math.reduce_mean(actorLoss)
        
        actorNetworkGradient = tape.gradient(actorLoss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actorNetworkGradient, self.actor.trainable_variables))
        
        self.updateNetworkParams()
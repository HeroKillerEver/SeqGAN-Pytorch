import copy
import numpy as np
import torch
import torch.nn.functional as F


class Rollout(object):
	"""Rollout policy"""
	def __init__(self, generator, discriminator, update_rate):
		super(Rollout, self).__init__()
		self.generator_theta = generator
		self.generator_beta = copy.deepcopy(generator)	
		self.discriminator = discriminator
		self.update_rate = update_rate


	def reward(self, x, rollout_num):

		batch_size, sequence_len = x.size()
		rewards = []
		for i in range(rollout_num):
			for l in range(1, sequence_len):
				data = x[:, 0:l]
				samples = self.generator_beta.sample(batch_size, sequence_len, data) # (batch_size, sequence_len)
				reward = F.sigmoid(self.discriminator(samples)) # (batch_size, 1)
				reward = reward.data.cpu().numpy()
				if i == 0:
					rewards.append(reward)
				else:
					rewards[l-1] += reward

			reward = F.sigmoid(self.discriminator(x))
			reward = reward.data.cpu().numpy()
			if i == 0:
				rewards.append(reward)
			else:
				rewards[sequence_len-1] += reward

		rewards = (np.array(rewards).squeeze().T) / (1. * rollout_num) # (batch_size, sequence_len)

		return rewards


	def update_params(self):
		dic = {}
		for name, param in self.generator_theta.named_parameters():
			dic[name] = param.data
		for name, param in self.generator_beta.named_parameters():
			if name.startswith('emb'):
				param.data = dic[name]
			else:
				param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]



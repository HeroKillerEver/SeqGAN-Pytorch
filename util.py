import torch 
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	if volatile:
		return Variable(x, volatile=True)
	else:
		return Variable(x)



def to_cuda(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return x




def generate_samples(model, batch_size, sequence_len, generate_num, output_file):
	samples = []
	for _ in range(int(generate_num / batch_size)):
		sample = model.sample(batch_size, sequence_len)
		sample = sample.data.cpu().numpy()
		samples.append(sample)

	outputs = np.vstack(samples)
	np.savetxt(output_file, outputs, fmt='%d', delimiter=',')




class GenData(Dataset):
	"""docstring for GenData"""
	def __init__(self, file):
		super(GenData, self).__init__()
		self.data = np.genfromtxt(file, dtype='int', delimiter=',') # (generate_num, sequence_len)
		N, seq_len  = self.data.shape
		zeros = np.zeros(shape=(N, 1), dtype='int')
		inputs = np.concatenate((zeros, self.data), axis=1) # (generate_num, sequence_len+1)
		targets = np.concatenate((self.data, zeros), axis=1) # (generate_num, sequence_len+1)

		self.inputs = torch.from_numpy(inputs)
		self.targets = torch.from_numpy(targets)


	def __getitem__(self, idx):
		return self.inputs[idx], self.targets[idx]

	def __len__(self):
		return self.data.shape[0]




class DisData(Dataset):
	"""docstring for DisData"""
	def __init__(self, real_file, fake_file):
		super(DisData, self).__init__()
		real_data = np.genfromtxt(real_file, dtype='int', delimiter=',')
		fake_data = np.genfromtxt(fake_file, dtype='int', delimiter=',')
		N_real, N_fake = real_data.shape[0], fake_data.shape[0]
		ones = np.ones(shape=(N_real, 1), dtype='int')
		zeros = np.zeros(shape=(N_fake, 1), dtype='int')
		data = np.concatenate((real_data, fake_data), axis=0)
		labels = np.concatenate((ones, zeros), axis=0)
		self.data = torch.from_numpy(data)
		self.labels = torch.from_numpy(labels.astype('float32'))


	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

	def __len__(self):
		return self.data.size(0) 
		
		

	
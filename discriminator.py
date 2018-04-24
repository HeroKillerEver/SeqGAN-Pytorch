import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import util






class Discriminator(nn.Module):
	"""A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Linear
    """
	def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, dropout):
		super(Discriminator, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.convs = nn.ModuleList([nn.Conv2d(1, num_filter, kernel_size=[filter_size, embedding_size]) \
									 for filter_size, num_filter in zip(filter_sizes, num_filters)])
		self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=(sequence_length-filter_size+1, 1), stride=1) \
									 for filter_size, num_filter in zip(filter_sizes, num_filters)])
		

		# for filter_size, num_filter in zip(filter_sizes, num_filters):
		# 	conv = nn.Conv2d(1, num_filter, kernel_size=[filter_size, embedding_size])
		# 	pool = nn.MaxPool2d(kernel_size=(sequence_length-filter_size+1, 1), stride=1)
		# 	self.convs.append(conv)
		# 	self.pools.append(pool)
		self.relu = nn.ReLU(inplace=True)
		self.highway = nn.Linear(np.sum(num_filters), np.sum(num_filters))
		self.dropout = nn.Dropout(p=dropout)
		self.linear  = nn.Linear(np.sum(num_filters), 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		#### x: [N, sequence_length]
		embedding = self.embedding(x)  # (N, sequence_length, embedding_size)
		embedding = embedding.unsqueeze(1) # (N, 1, sequence_length, embedding_size)
		outs = []

		for conv, pool in zip(self.convs, self.pools):
			out = conv(embedding) 
			out = self.relu(out)
			out = pool(out)
			outs.append(out)

		f = torch.cat(outs, dim=1)
		f = torch.squeeze(f)
		highway = (self.highway(f))
		g = self.relu(highway)
		t = self.sigmoid(highway)
		highway_out = t * g + (1 - t) * f
		highway_out_drop = self.dropout(highway_out)
		logits = self.linear(highway_out)

		return logits  # (N, 1)








			
		





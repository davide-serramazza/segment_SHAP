from torch import nn
from torch.nn import functional as F
import torch
from torch.utils import data

"""
	code from dcam publication
	
	@inproceedings{boniol2022dcam,
	               title={dcam: Dimension-wise class activation map for explaining multivariate data series classification},
	author={Boniol, Paul and Meftah, Mohammed and Remy, Emmanuel and Palpanas, Themis},
	booktitle={Proceedings of the 2022 International Conference on Management of Data},
	pages={1175--1189},
	year={2022}
	}
"""

class TSDataset(data.Dataset):
	def __init__(self,x_train,labels, device="cpu"):

		self.samples = torch.tensor(x_train).type(torch.float32).to(device)
		self.labels = torch.tensor(labels).to(torch.long).to(device)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self,idx):
		return self.samples[idx],self.labels[idx]


######################################################################################################

class Conv1dSamePadding(nn.Conv1d):
	def forward(self, input):
		return conv1d_same_padding(input, self.weight, self.bias, self.stride,
								   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
	kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
	l_out = l_in = input.size(2)
	padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
	if padding % 2 != 0:
		input = F.pad(input, [0, 1])

	return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
					padding=padding // 2,
					dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size,
				 stride):
		super().__init__()

		self.layers = nn.Sequential(
			Conv1dSamePadding(in_channels=in_channels,
							  out_channels=out_channels,
							  kernel_size=kernel_size,
							  stride=stride),
			nn.BatchNorm1d(num_features=out_channels),
			nn.ReLU(),
		)

	def forward(self, x):

		return self.layers(x)


class ResNetBaseline(nn.Module):

	def __init__(self, in_channels, mid_channels = 64, num_pred_classes = 1 ):
		super().__init__()

		self.layers = nn.Sequential(*[
			ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
			ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
			ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

		])
		self.final = nn.Linear(mid_channels * 2, num_pred_classes)


	def forward(self, x):
		x = self.layers(x)
		return self.final(x.mean(dim=-1))


class ResNetBlock(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()

		channels = [in_channels, out_channels, out_channels, out_channels]
		kernel_sizes = [8, 5, 3]

		self.layers = nn.Sequential(*[
			ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
					  kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
		])

		self.match_channels = False
		if in_channels != out_channels:
			self.match_channels = True
			self.residual = nn.Sequential(*[
				Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
								  kernel_size=1, stride=1),
				nn.BatchNorm1d(num_features=out_channels)
			])

	def forward(self, x):

		if self.match_channels:
			return self.layers(x) + self.residual(x)
		return self.layers(x)

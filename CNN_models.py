from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
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
	def __init__(self,x_train,labels):
		self.samples = x_train
		self.labels = labels

	def __len__(self):
		return len(self.samples)

	def __getitem__(self,idx):
		return self.samples[idx],self.labels[idx]


class ModelCNN():
	def __init__(self,
				 model,
				 n_epochs_stop,
				 save_path=None,
				 device="cpu",
				 criterion=nn.CrossEntropyLoss(),
				 learning_rate=0.00001):

		self.model = model
		self.n_epochs_stop = n_epochs_stop
		self.save_path = save_path
		self.criterion = criterion
		self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
		self.device = device


	def __test(self,dataloader):
		mean_loss = []
		tot_correct = []
		total_sample = []

		with torch.no_grad():
			for i,batch_data in enumerate(dataloader):
				self.model.eval()
				ts, label = batch_data
				img = Variable(ts.float()).to(self.device)
				v_label = Variable(label.float()).to(self.device)
				# ===================forward=====================
				output = self.model(img.float()).to(self.device)

				loss = self.criterion(output.float(), v_label.long())

				# ================eval on test===================
				total = label.size(0)
				_, predicted = torch.max(output.data, 1)
				correct = (predicted.to(self.device) == label.to(self.device)).sum().item()

				mean_loss.append(loss.item())
				tot_correct.append(correct)
				total_sample.append(total)

		return mean_loss,tot_correct,total_sample

	def train(self,num_epochs,train_loader,test_loader):

		#inner function to print statistics
		def print_stats():
			print('Epoch [{}/{}], Loss Train: {:.4f},Loss Test: {:.4f}, Accuracy Train: {:.2f}%, Accuracy Test: {:.2f}%'
				  .format(epoch + 1,
						  num_epochs,
						  np.mean(mean_loss_train),
						  np.mean(mean_loss_test),
						  (np.sum(mean_accuracy_train)/np.sum(total_sample_train)) * 100,
						  current_val_accuracy * 100))


		epochs_no_improve = 0
		min_val_loss = np.Inf
		max_val_accuracy = 0.0
		loss_train_history = []
		loss_test_history = []
		accuracy_test_history = []

		for epoch in range(num_epochs):
			mean_loss_train = []
			mean_accuracy_train = []
			total_sample_train = []


			for i,batch_data_train in enumerate(train_loader):
				self.model.train()

				ts_train, label_train = batch_data_train
				img_train = Variable(ts_train.float()).to(self.device)
				v_label_train = Variable(label_train.float()).to(self.device)

				# ===================forward=====================
				self.optimizer.zero_grad()
				output_train = self.model(img_train.float()).to(self.device)

				# ===================backward====================
				loss_train = self.criterion(output_train.float(), v_label_train.long())
				loss_train.backward()
				self.optimizer.step()

				# ================eval on train==================
				total_train = label_train.size(0)
				_, predicted_train = torch.max(output_train.data, 1)
				correct_train = (predicted_train.to(self.device) == label_train.to(self.device)).sum().item()
				mean_loss_train.append(loss_train.item())
				mean_accuracy_train.append(correct_train)
				total_sample_train.append(total_train)

			# ==================eval on test=====================
			mean_loss_test,tot_correct_test,total_sample_test = self.__test(test_loader)
			current_val_accuracy = np.sum(tot_correct_test)/np.sum(total_sample_test)

			# ====================verbose========================
			if epoch % 10 == 0:
				print_stats()

			#TODO log more compact?
			# ======================log==========================
			loss_test_history.append(np.mean(mean_loss_test))
			loss_train_history.append(np.mean(mean_loss_train))
			accuracy_test_history.append(current_val_accuracy)
			self.loss_test_history = loss_test_history
			self.loss_train_history = loss_train_history
			self.accuracy_test_history = accuracy_test_history

			# ================early stopping=====================
			if epoch == 3:
				min_val_loss = np.sum(mean_loss_test)
				max_val_accuracy = current_val_accuracy

			if current_val_accuracy>max_val_accuracy:
				#np.sum(mean_loss_test) < min_val_loss:
				if self.save_path!=None:
					torch.save(self.model, self.save_path)
				epochs_no_improve = 0
				min_val_loss = np.sum(mean_loss_test)
				max_val_accuracy = current_val_accuracy
			else:
				epochs_no_improve += 1
				if epochs_no_improve == self.n_epochs_stop:
					print("TRAIN EARLY STOPPED; best accuracy is {:.2f} best loss is {:.4f}"
						  .format( max_val_accuracy,np.average(min_val_loss) ))
					if self.save_path!=None:
						self.model = torch.load(self.save_path)
					break
		return max_val_accuracy

	def predict(self,test_loader):
		predictions = []
		with torch.no_grad():
			for i,batch_data in enumerate(test_loader):
				X, y = batch_data
				X= Variable(X.float()).to(self.device)
				output = self.model(X.float()).to(self.device)
				predictions.append(torch.max(output.data,1).indices)
		return  torch.concat(predictions).cpu().numpy()



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

	def __init__(self, in_channels, mid_channels = 64,
				 num_pred_classes = 1):
		super().__init__()


		self.input_args = {
			'in_channels': in_channels,
			'num_pred_classes': num_pred_classes
		}

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

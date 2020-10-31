import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision import transforms

import dataset_loader as dl
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod
from constraint_losses import winston_loss, cifar100_loss, pascalpart_loss
from dataset_loader import SemiSupDataset
from utils import Progbar, find_folder, show_batch

SIMPLE_CNN          = "Simple_CNN"
RESNET_TL           = "Resnet_tl"
RESNET_FT           = "Resnet_ft"


class RexLearnClassifier(ABC, nn.Module):
	"""
		RexLearnClassifier is an abstract class representing a classifier for learning with constraints.
		init() methods is required to be implemented by extending classes

		Attributes
		----------
		trained: bool
			flag used to determine whether the model has already been trained or not
		name: str
			name of the model extending the abstract class (among MODELS)
	"""

	@abstractmethod
	def __init__(self, dataset: str, name: str = "net", seed: int = 0, main_classes=False, to_normalize=False, reject=False):
		"""
		Parameters
		----------
		dataset: str
			name of the dataset to work on. Needs to be amongst the one implemented (animals, cifar100), because
			constraints are dataset-specific
		name: str
			name of the network: used for saving and loading purposes
		seed: int
			seed used for results reproducibility
		main_classes: bool
			whether to work only on the main_classes or not
		"""

		assert dataset in dl.DATASETS, "dataset "+dataset+" not among those available. Available datasets " + str(dl.DATASETS)
		assert hasattr(self, 'class_name'), "class_name attribute must be set before calling RexLearnClassifier init()"

		super(RexLearnClassifier, self).__init__()
		self.name       = name
		self.dataset    = dataset
		self.transforms = self.get_transforms()
		self.register_buffer("trained", torch.tensor(False))
		self.seed       = seed
		self.set_seed(seed)
		self.main_classes = main_classes
		self.to_normalize = to_normalize
		self.reject       = reject
		self.eval_main_classes = False
		self.threshold = None

	def forward(self, x, logits=True, classes=None, get_max=False):
		assert not (classes is None and get_max), "get max can be used only when the classes on which to take the " \
		                                          "max value are specified"
		if self.to_normalize:
			transform = transforms.Compose(self.transforms.transforms[-1:])
			x = torch.stack([transform(img) for img in x])
		if self.eval_main_classes:
			classes     = dl.MAIN_CLASSES[self.dataset]
		# print("x", x[0, :1, :1, :5])
		predictions = self.model(x)
		if classes:
			predictions = predictions[:, classes]
		if logits:
			return predictions
		else:
			activations = torch.sigmoid(predictions)
			if get_max:
				return activations.max(dim=1)[1]
			else:
				# print("activations", activations[0, :5].detach().numpy())
				return activations

	@staticmethod
	def set_seed(seed):
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def get_device(self):
		return next(self.model.parameters()).device

	def get_transforms(self):
		from .transforms import TRANSFORMS
		transform_name = self.class_name + self.dataset
		return TRANSFORMS[transform_name]

	def set_eval_main_classes(self, eval_main_classes=True):
		self.eval_main_classes = eval_main_classes

	def set_to_normalize(self, normalize=True):
		self.to_normalize = normalize

	def compute_loss(self, output, targets, sup_mask, constraint_weight, train=False):
		n_sup = torch.sum(sup_mask != 0, dtype=torch.float32)/sup_mask.shape[1]
		c_sup = output.shape[0]
		sup_loss = F.binary_cross_entropy(output, targets, weight=sup_mask, reduction="sum")
		norm_sup_loss = sup_loss/n_sup if n_sup != 0 else torch.tensor(0, device=output.device)
		constr_loss = self.constraint_loss(output, train=train) / c_sup
		tot_loss = norm_sup_loss + constraint_weight * constr_loss
		return tot_loss, norm_sup_loss, constr_loss

	def constraint_loss(self, output, train=False, sum=True):
		if self.main_classes or self.eval_main_classes:
			return torch.tensor([0 for _ in range(output.shape[0])] if not sum else 0)
		if self.dataset == dl.ANIMAL_DATASET:
			return winston_loss(output, sum=sum)
		elif self.dataset == dl.CIFAR_100:
			return cifar100_loss(output, train=train, sum=sum)
		elif self.dataset == dl.PASCAL_PART:
			return pascalpart_loss(output, sum=sum)
		else:
			raise NotImplementedError("Constraint Loss related to dataset " + self.dataset + "not available")

	def fit(self, train_set: dl.SemiSupDataset, val_set: dl.SemiSupDataset, batch_size: int = 16,
	        constraint_weight: float = .0, epochs: int = 10, device: str = "cpu", output: bool = True,
	        main_classes: bool = False):
		assert hasattr(self, 'optim'), "Model optimizer must be set before calling RexLearnClassifier fit()"

		if self.trained:
			print("Model already trained. Training continue")
		if main_classes:
			self.set_eval_main_classes()

		# Setting device
		device = torch.device(device)
		self.to(device), self.model.to(device)

		# Define constraint_weight growth during training epochs (reaches maximum at epochs/2)
		constraint_weight_delta = constraint_weight / (epochs * 0.5)
		constraint_weight_max = constraint_weight
		constraint_weight = 0.

		# Check constraint consistency with dataset
		assert self.constraint_loss(train_set.targets).item()/len(train_set) < 0.1, "Constraint not consistent with train set"

		# Training epochs
		best_val_f1, best_epoch = 0.0, 0
		train_accs, train_f1s, train_cons_loss, val_accs, val_f1s, val_cons_loss = [], [], [], [], [], []
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
		                                           num_workers=0)  # if load_in_memory else 3) TODO: crash on Windows
		pbar = tqdm(range(0, epochs), ncols=100)
		for epoch in pbar:
			pbar.set_description("Epoch %d/%d, Constraint_weight %.3f" % (epoch+1, epochs, constraint_weight))
			print("\r")
			prog_bar = Progbar(target=len(train_set), verbose=1, width=10)

			batch_acc = [0., 0.]
			train_set_output, train_set_labels, train_set_masks = [], [], []
			data_processed = 0

			for data in train_loader:
				# Load batch (dataset, labels, supervision mask) on the correct device
				batch_data, batch_labels, batch_mask = data[0].to(device), data[1].to(device), data[2].to(device)
				real_batch_size = batch_data.size(0)
				data_processed  += real_batch_size
				# self.writer.add_graph(self.model, batch_data)
				self.optim.zero_grad()

				# FOR DEBUGGING PURPOSE ONLY
				# show_batch(batch_data, batch_labels, train_set.classes)

				# Network outputs on the current batch of dataset (after sigmoid activation)
				outputs = self.forward(batch_data, logits=False)

				# Compute losses and update gradients
				tot_loss, sup_loss, constraint_loss = self.compute_loss(outputs, batch_labels, batch_mask,
				                                                        constraint_weight, train=True)
				if tot_loss != 0.0:
					tot_loss.backward()
					self.optim.step()

				# Compute accuracy on batch dataset
				with torch.no_grad():
					predictions = (outputs > 0.5).to(torch.float)
					if output:
						batch_acc_tmp = self.accuracy(predictions, batch_labels, batch_mask)
						if batch_acc_tmp is not None:
							batch_acc = batch_acc_tmp
						prog_bar.update(data_processed, [("T_l", tot_loss.item()), ("S_l", sup_loss.item()),
						                                 ("C_l", constraint_loss.item()), ("Acc0", batch_acc[0]),
						                                 ("Acc1", batch_acc[1])])
				# Data moved to cpu again
				outputs, batch_labels, batch_mask = outputs.detach().to("cpu"), batch_labels.detach().to("cpu"), batch_mask.detach().to("cpu")
				train_set_output.append(outputs), train_set_labels.append(batch_labels), train_set_masks.append(batch_mask)

			train_set_output    = torch.cat(train_set_output, dim=0).detach().to("cpu")
			train_set_labels    = torch.cat(train_set_labels, dim=0).detach().to("cpu")
			train_set_masks     = torch.cat(train_set_masks, dim=0).detach().to("cpu")
			
			# Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
			train_acc, train_f1, train_constraint_loss = self.evaluate(train_set, batch_size, device, train_set_output,
			                                                           train_set_labels, train_set_masks)
			val_acc, val_f1, val_constraint_loss = self.evaluate(val_set, batch_size, device)

			# Save best model
			if val_f1 >= best_val_f1 and epoch >= epochs / 2:
				best_val_f1 = val_f1
				best_epoch = epoch + 1
				self.save()

			print("Train_acc={:.4f}, {:.4f}, train_f1={:.4f}, train_constr_loss={:.4f}".
			      format(train_acc[0], train_acc[1], train_f1, train_constraint_loss))
			print("Val_acc={:.4f}, {:.4f}, val_f1={:.4f}, val_constr_loss={:.4f}, best_epoch={}".
			      format(val_acc[0], val_acc[1], val_f1, val_constraint_loss, best_epoch))
			if "cuda" in device.type:
				print(f"{device} memory used:", torch.cuda.memory_allocated(device))

			# Increment constraint_weight
			constraint_weight += constraint_weight_delta
			constraint_weight = min(constraint_weight, constraint_weight_max)

			# Append epoch results
			train_accs.append(train_acc[1]), train_f1s.append(train_f1), train_cons_loss.append(train_constraint_loss)
			val_accs.append(val_acc[1]), val_f1s.append(val_f1), val_cons_loss.append(val_constraint_loss)

		# Best model is loaded and saved with buffer .trained set to true
		self.load(device, set_trained=True)

		return train_accs, train_f1s, train_cons_loss, val_accs, val_f1s, val_cons_loss, best_epoch

	def evaluate(self, dataset: dl.SemiSupDataset, batch_size, device, outputs=None, labels=None,
	             masks=None, classes=None, get_max=False):
		assert not(classes is None and get_max), "get max can be used only when the classes on which to take the " \
		                                         "max value are specified"
		if self.main_classes:
			classes = dl.MAIN_CLASSES[self.dataset]
		self.eval()
		self.to(device), self.model.to(device)
		with torch.no_grad():
			if outputs is None or labels is None or masks is None:
				outputs, labels, masks = self.predict(dataset, device, batch_size)
			if get_max:
				max_predictions = torch.argmax(outputs[:, classes], dim=1)
				predictions = torch.zeros_like(outputs)
				for i, max_p in enumerate(max_predictions):
					predictions[i, max_p] = 1
			else:
				predictions = (outputs > 0.5).to(torch.float)[:, classes]
			acc         = self.accuracy(predictions, labels, masks, classes)
			f1          = self.f1(predictions, labels, masks, classes)
			constraint_loss = self.constraint_loss(outputs) / outputs.shape[0]
		self.train()
		return acc, f1, constraint_loss.item()

	def predict(self, dataset: dl.SemiSupDataset, device, batch_size=64, classes=None):
		# print("Predicting...")
		outputs, labels, masks = [], [], []
		loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
		for i, data in enumerate(loader):
			batch_data, batch_labels, batch_mask = data[0].to(device), data[1].to(device), data[2].to(device)
			batch_outputs = self.forward(batch_data, logits=False, classes=classes)
			batch_outputs, batch_labels, batch_mask = batch_outputs.to("cpu"), batch_labels.to("cpu"), batch_mask.to("cpu")
			outputs.append(batch_outputs), labels.append(batch_labels), masks.append(batch_mask)
		outputs = torch.cat(outputs, dim=0)
		labels = torch.cat(labels, dim=0)
		masks = torch.cat(masks, dim=0)
		return outputs, labels, masks

	def calc_threshold(self, dataset: SemiSupDataset, reject_rate: float = 0.1):
		with torch.no_grad():
			outputs, labels, _ = self.predict(dataset, self.get_device())
			# print("Output", outputs)
			output_single_label = outputs[:, :len(dl.MAIN_CLASSES[self.dataset])].max(dim=1)[1]
			label_single_label = labels.max(dim=1)[1]
			# print("Output", output_single_label, "Labels", label_single_label)
			acc = output_single_label.eq(label_single_label).sum().cpu().numpy() / len(dataset)
			cons_losses = self.constraint_loss(outputs, sum=False).sort()[0]
			reject_number = int(reject_rate * len(dataset))
			self.threshold = cons_losses[len(dataset)-reject_number-1]
			self.threshold = self.threshold.to(self.get_device())
			print("Threshold", self.threshold.item(), "Acc on val", acc)

	def single_label_evaluate(self, dataset: dl.SingleLabelDataset, device, batch_size=64, reject=False, adv=False):
		self.eval(), torch.no_grad()  # ALWAYS REMEMBER TO SET EVAL WHEN EVALUATING A RESNET
		self.to(device), self.model.to(device)
		classes = dl.MAIN_CLASSES[self.dataset]
		outputs, labels, cons_losses, rejections = [], [], [], []
		with torch.no_grad():
			loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
			for i, data in enumerate(loader):
				batch_data, batch_labels = data[0].to(device), data[1].to(device)
				batch_output = self.forward(batch_data, logits=False)
				cons_loss = self.constraint_loss(batch_output, sum=False)
				batch_output = batch_output[:, classes].max(dim=1)[1]
				if reject:
					assert self.threshold is not None, "Threshold not calculated. self.calc_threshold need to be called beforehand " \
					                                   "any forward operation when reject is set\n"
					self.threshold = self.threshold.to(device=cons_loss.device)
					batch_rejections = cons_loss > self.threshold
					batch_output[batch_rejections == 1] = -1
				else:
					batch_rejections = torch.zeros_like(batch_output)
				outputs.append(batch_output), labels.append(batch_labels)
				cons_losses.append(cons_loss), rejections.append(batch_rejections)

			outputs, labels, cons_losses = torch.cat(outputs, dim=0), torch.cat(labels, dim=0), torch.cat(cons_losses, dim=0)
			# print("Output", outputs.cpu().numpy(), "Labels", labels.cpu().numpy(), "Cons_losses", cons_losses.cpu().numpy())
			rejections = torch.cat(rejections, dim=0)
			if adv:
				outputs[rejections == 1] = labels[rejections == 1]
			acc_single_label = labels.eq(outputs).sum().item() / outputs.shape[0]
			rejection_rate = rejections.sum().item()/len(dataset)
		return acc_single_label, rejection_rate

	def save(self, set_trained=False):
		cur_dir = os.path.abspath(os.curdir)
		os.chdir(os.path.dirname(self.name))
		if set_trained:
			self._set_trained()
		torch.save(self.state_dict(), os.path.basename(self.name))
		os.chdir(cur_dir)

	def load(self, device, set_trained=False):
		cur_dir = os.path.abspath(os.curdir)
		os.chdir(os.path.dirname(self.name))
		try:
			incompatible_keys = self.load_state_dict(torch.load(os.path.basename(self.name),
			                                                    map_location=torch.device(device)), strict=False)
		except FileNotFoundError:
			raise ClassifierNotTrainedError()
		else:
			if len(incompatible_keys.missing_keys) > 1 or len(incompatible_keys.unexpected_keys) > 0:
				raise IncompatibleClassifierError(incompatible_keys.missing_keys, incompatible_keys.unexpected_keys)
		finally:
			os.chdir(cur_dir)

		# Sometimes torch.load() does not bring the model on the correct device
		self.to(device), self.model.to(device)
		assert self.get_device() == torch.device(device), f"Some problem occurred when bringing model to {device}, " \
		                                                  f"it is on {self.get_device()}"

		if set_trained or (len(incompatible_keys.missing_keys) > 0 and incompatible_keys.missing_keys[0] == 'trained'):
			self.save(set_trained=True)
		if not self.trained:
			raise ClassifierNotTrainedError()

	def _set_trained(self):
		self.trained = torch.tensor(True)

	@staticmethod
	def accuracy(output, targets, mask, classes=None):
		semi_sup_idx = (torch.sum(mask, dim=1) > 0)
		n_sup = torch.sum(semi_sup_idx.to(torch.float))
		if n_sup == 0:
			return None

		if classes is not None:
			output = output[:, classes]
			targets = targets[:, classes]
			mask = mask[:, classes]

		output = output[semi_sup_idx, :]
		targets = targets[semi_sup_idx, :]
		mask = mask[semi_sup_idx, :]

		mask_float = (mask > 0.0).to(torch.float)
		mask_bool = mask.to(torch.bool)
		output_masked = (output * mask_float)
		targets_masked = (targets * mask_float)

		acc_soft = (torch.mean((output == targets).view(-1).to(torch.float)[mask_bool.view(-1)]) * 100.0).item()
		acc_rigid = (torch.mean((torch.sum((output_masked == targets_masked).to(torch.float),
		                                   dim=1) == targets.shape[1])
		                        .to(torch.float)) * 100.0).item()

		return np.array([acc_soft, acc_rigid])

	@staticmethod
	def f1(output, targets, mask, classes=None):
		semi_sup_idx = (torch.sum(mask, dim=1) > 0)
		n_sup = torch.sum(semi_sup_idx.to(torch.float))
		if n_sup == 0:
			return None

		output = output[semi_sup_idx, :]
		targets = targets[semi_sup_idx, :]
		mask = mask[semi_sup_idx, :]

		if classes is not None:
			output = output[:, classes]
			targets = targets[:, classes]
			mask = mask[:, classes]

		f1 = 0.0
		c = targets.shape[1]

		for i in range(0, c):
			output_per_class = output[:, i].cpu().numpy()
			targets_per_class = targets[:, i].cpu().numpy()
			mask_per_class = (mask[:, i] > 0.0).to(torch.bool).cpu().numpy()
			output_per_class = output_per_class[mask_per_class]
			targets_per_class = targets_per_class[mask_per_class]
			f1_per_class = f1_score(output_per_class, targets_per_class, average="binary")
			f1 += f1_per_class

		f1 = f1 / c
		return f1


class SimpleCNNRexLearnClassifier(RexLearnClassifier):
	"""
	SimpleCNNRexLearnClassifier is a class implementing the abstract class RexLearnClassifier.
	It is a simple CNN with 3 Convolutional layer and 1 FC layer on top.
	"""
	def __init__(self, dataset: str, name: str = SIMPLE_CNN, gray_scale: bool = False, l_r: float = 0.001,
	             seed: int = 0, extract_features=False, main_classes=False):
		"""
		Parameters
		----------
		l_r: float
			learning rate used by the Adam optimizer
		gray_scale: bool
			parameters used to determine whether to use 3 channels RGB images or gray scale images, through transforms
		extract_features: bool
			added for compatibility with other models but need to be False here
		"""

		assert not extract_features, "Cannot extract features when employing the SimpleCNN model"

		self.class_name = SIMPLE_CNN + dl.GRAY if gray_scale else SIMPLE_CNN
		self.l_r        = l_r
		super(SimpleCNNRexLearnClassifier, self).__init__(dataset, name + dl.GRAY if gray_scale else name, seed,
		                                                  main_classes=main_classes)
		modules = [
			nn.Conv2d((1 if gray_scale else 3), 64, kernel_size=5, stride=1, padding=2),  # 32x32
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),  # 15x15
			nn.Conv2d(64, 128, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),  # 7x7
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),  # 3x3
			Flatten(),
			nn.Linear(256 * 3 * 3, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
		]
		# number of classes depends on the dataset at hand
		# if working only on the main_classes last layer is reduced
		if not main_classes:
			modules.append(nn.Linear(4096, dl.NUM_CLASSES[dataset]))
		else:
			modules.append(nn.Linear(4096, dl.MAIN_CLASSES[dataset]))
		self.model = nn.Sequential(*modules)

		self.optim = torch.optim.Adam(self.model.parameters(), self.l_r)


class ResnetRexLearnClassifier(RexLearnClassifier):
	"""
	ResnetRexLearnClassifier is a class implementing the abstract class RexLearnClassifier.
	It is based on the famous Resnet50 network that is used here for transfer-learning.
	"""

	def __init__(self, dataset: str, l_r: float = 0.001, fine_tuning: bool = True, main_classes=False, **kwargs):
		"""
		Parameters
		----------
		l_r: float
			learning rate used by the Adam optimizer
		fine_tuning: bool
			If True the last layers of the Resnet backbone are also trained together with the last fc layer.
			If False the inner backbone is used as feature extractor and only the last fully connected layer is trained.
		"""

		self.class_name = RESNET_FT if fine_tuning else RESNET_TL
		self.l_r = l_r
		self.fine_tuning = fine_tuning

		RexLearnClassifier.__init__(self, dataset, main_classes=main_classes, **kwargs)

		self.model = torchvision.models.resnet50(pretrained=True)
		num_features = self.model.fc.in_features
		for param in self.model.parameters():
			param.requires_grad = False
		if not main_classes:
			self.model.fc = nn.Linear(num_features, dl.NUM_CLASSES[dataset])
		else:
			self.model.fc = nn.Linear(num_features, len(dl.MAIN_CLASSES[dataset]))
		if fine_tuning:
			# In fine tuning also the last convolutional block is trained (although with lower l_r)
			for param in self.model.layer4.parameters():
				param.requires_grad = True
			self.optim = torch.optim.Adam([
				{'params': self.model.layer4.parameters(), 'lr': l_r * 1e-1},
				{'params': self.model.fc.parameters(), 'lr': l_r}
			])
		else:
			self.optim = torch.optim.Adam(self.model.fc.parameters(), l_r)


class ExtractFeaturesRexLearnClassifier(RexLearnClassifier):
	"""
	ExtractFeaturesRexLearnClassifier is a class implementing the abstract class RexLearnClassifier.
	It is based on the famous Resnet50 network but here only the last layers are present: dataset features need to be
	pre-processed with Resnet50 backbone.
	"""

	def __init__(self, dataset: str, l_r: float = 0.001, fine_tuning: bool = True, main_classes=False, **kwargs):
		"""
		Parameters
		----------
		l_r: float
			learning rate used by the Adam optimizer
		fine_tuning: bool
			If True the last convolutional block of the Resnet backbone is also trained together with the fc layer.
			If False the inner backbone is used as feature extractor and only the last fully connected layer is trained.
		"""

		self.class_name = RESNET_FT if fine_tuning else RESNET_TL
		self.l_r = l_r
		self.fine_tuning = fine_tuning

		RexLearnClassifier.__init__(self, dataset, main_classes=main_classes, **kwargs)

		resnet = torchvision.models.resnet50(pretrained=True)
		num_features = resnet.fc.in_features
		if self.fine_tuning:
			modules = [
				resnet.layer4,
				resnet.avgpool,
				Flatten(),
			]
			if not main_classes:
				modules.append(nn.Linear(num_features, dl.NUM_CLASSES[dataset]))
			else:
				modules.append(nn.Linear(num_features, len(dl.MAIN_CLASSES[dataset])))
			self.model = torch.nn.Sequential(*modules)
			self.optim = torch.optim.Adam([
				{'params': list(self.model.children())[0].parameters(), 'lr': self.l_r * 1e-1},
				{'params': list(self.model.children())[3].parameters(), 'lr': self.l_r}
			])

		else:
			num_features = resnet.fc.in_features
			if not main_classes:
				self.model = torch.nn.Sequential(
					torch.nn.Linear(num_features, dl.NUM_CLASSES[dataset]),
				)
			else:
				self.model = torch.nn.Sequential(
					torch.nn.Linear(num_features, len(dl.MAIN_CLASSES[dataset])),
				)
			self.optim = torch.optim.Adam(self.model.parameters(), self.l_r)

	def convert(self, model_name):
		net = ResnetRexLearnClassifier(dataset=self.dataset, name=model_name, l_r=self.l_r, fine_tuning=self.fine_tuning,
		                               seed=self.seed, main_classes=self.main_classes, reject=self.reject)
		if self.fine_tuning:
			net.model.layer4   = copy.deepcopy(list(self.model.children())[0])
			net.model.fc       = copy.deepcopy(list(self.model.children())[3])
		else:
			net.model.fc       = copy.deepcopy(list(self.model.children())[0])
		# self.model = copy.deepcopy(net)
		return net


class ResnetTLRexLearnClassifier(ResnetRexLearnClassifier, ExtractFeaturesRexLearnClassifier):
	"""
	ResnetTLRexLearnClassifier is a decorator class that always trains only the last fully connected layer of a Resnet50.
	It is also used to distinguish ResnetRexLearnClassifier from ExtractFeaturesRexLearnClassifier. According to whether
	extract_feature is set the first or the second is implemented
	"""
	def __init__(self, extract_features: bool = False, **kwargs):
		if extract_features:
			ExtractFeaturesRexLearnClassifier.__init__(self, fine_tuning=False, **kwargs)
		else:
			ResnetRexLearnClassifier.__init__(self, fine_tuning=False, **kwargs)


class ResnetFTRexLearnClassifier(ResnetRexLearnClassifier, ExtractFeaturesRexLearnClassifier):
	"""
	ResnetFTRexLearnClassifier is a decorator class that always trains the fully connected layer and the last
	convolutional block of a Resnet50.
	It is also used to distinguish ResnetRexLearnClassifier from ExtractFeaturesRexLearnClassifier. According to whether
	extract_feature is set the first or the second is implemented
	"""

	def __init__(self, extract_features: bool = False, **kwargs):
		if extract_features:
			ExtractFeaturesRexLearnClassifier.__init__(self, fine_tuning=True, **kwargs)
		else:
			ResnetRexLearnClassifier.__init__(self, fine_tuning=True, **kwargs)


class Flatten(torch.nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)


class ClassifierNotTrainedError(Exception):
	def __init__(self):
		self.message = "Classifier not trained"

	def __str__(self):
		return self.message


class IncompatibleClassifierError(Exception):
	def __init__(self, missing_keys, unexpected_keys):
		self.message = "Unable to load the selected classifier.\n"
		for key in missing_keys:
			self.message += "Missing key: " + str(key) + ".\n"
		for key in unexpected_keys:
			self.message += "Unexpected key: " + str(key) + ".\n"

	def __str__(self):
		return self.message

MODELS = {
	SIMPLE_CNN: SimpleCNNRexLearnClassifier,
	RESNET_TL: ResnetTLRexLearnClassifier,
	RESNET_FT: ResnetFTRexLearnClassifier
}

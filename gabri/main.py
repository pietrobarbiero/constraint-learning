import copy
import os

import torch

import rexlearnclassifier as rx
import dataset_loader as dl
from utils import find_folder, file_exists, get_model_path_name


def main(model: str, dataset: str, train_dir: str, val_dir: str, test_dir: str = None, mode: str = "train",
         constr_weight: float = 0.1, l_r: float = 0.0001, sup_data_frac: float = 0.1, partially_labeled: float = 0.1,
         batch_size: int = 64, epochs: int = 20, device: str = "cpu", extract_features=False, main_classes: bool = False,
         load_in_memory: bool = True, output: bool = True, seed: int = 0, model_folder=None, model_name: str = None):
	assert model in rx.MODELS, "Unavailable model {}. Models available: {}".format(model, rx.MODELS)
	assert dataset in dl.DATASETS, "Unavailable datasets {}. Datasets available: {}".format(dataset, dl.DATASETS)

	print("Model parameters")
	loc_args = copy.deepcopy(locals())
	important_args = ["model", "dataset", "constr_weight", "l_r", "sup_data_frac", "partially_labeled", "batch_size",
	                  "epochs", "extract_features", "seed"]
	for arg in important_args:
		print("{}: {}".format(arg, loc_args[arg]))

	# Models are saved in the model folder (if not given a folder "models" is searched)
	if model_folder is None:
		model_folder = "models"
	model_folder = find_folder(model_folder)

	# Name of the model depends on the parameters if not given
	if model_name is not None:
		model_name = find_folder(os.path.join(model_folder, model_name))
	else:
		model_name = get_model_path_name(loc_args, important_args, model_folder, main_classes)
	print("Model name:", model_name)

	print("Creating model...")
	classifier = rx.MODELS[model](dataset=dataset, name=model_name, l_r=l_r, seed=seed, extract_features=extract_features, main_classes=main_classes)
	fine_tuning = model == rx.RESNET_FT

	print("Creating datasets...")
	if mode == "train" or mode == "val":
		if not extract_features:
			train_set = dl.SemiSupDataset(train_dir, dataset, classifier.transforms, load_in_memory, sup_data_frac,
			                              partially_labeled, seed=seed, main_classes=main_classes)
			val_set = dl.SemiSupDataset(val_dir, dataset, classifier.transforms, load_in_memory, 1.0, partially_labeled,
			                            seed=seed, main_classes=main_classes)
		else:
			train_set = dl.ExtractFeaturesSemiSupDataset(train_dir, dataset, classifier.transforms, load_in_memory, sup_data_frac, partially_labeled,
			                                             fine_tuning=fine_tuning, device=device, seed=seed, main_classes=main_classes)
			val_set = dl.ExtractFeaturesSemiSupDataset(val_dir, dataset, classifier.transforms, load_in_memory, 1.0, partially_labeled,
			                                           fine_tuning=fine_tuning, device=device, seed=seed, main_classes=main_classes)

	metrics = {}
	if mode == "train":
		try:
			classifier.load(device)
			print("Model already trained skipped training")
		except rx.ClassifierNotTrainedError as e:
			print(e, "Training model...")
			train_accs, train_f1s, train_cons_losses, val_accs, val_f1s, val_cons_losses, best_epoch = classifier.fit(train_set, val_set, batch_size,
			                                                                                                          constr_weight, epochs, device,
			                                                                                                          output, main_classes)
			metrics.update({"train_accs": train_accs, "train_f1s": train_f1s, "train_cons_losses": train_cons_losses})
			metrics.update({"val_accs": val_accs, "val_f1s": val_f1s, "val_cons_losses": val_cons_losses})
			metrics.update({"best_epoch": best_epoch})
	else:
		classifier.load(device)

	if mode != "test":
		print("Running evaluation on training, validation, test datasets...")
		if not main_classes:
			train_acc, train_f1, train_cons_loss = classifier.evaluate(train_set, batch_size, device)
			print("train_acc={:.4f} {:.4f}, train_f1={:.4f}, train_constr_loss={:.4f}".format(train_acc[0], train_acc[1], train_f1, train_cons_loss))
			metrics.update({"train_acc": train_acc[1], "train_f1": train_f1, "train_cons_loss": train_cons_loss})
		main_train_acc, main_train_f1, _ = classifier.evaluate(train_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=False)
		print("main_train_acc={:.4f} {:.4f}, main_train_f1={:.4f}".format(main_train_acc[0], main_train_acc[1], main_train_f1))
		metrics.update({"main_train_acc": main_train_acc[1], "main_train_f1": main_train_f1})
		if dataset is not dl.PASCAL_PART:
			main_train_acc_2, main_train_f1_2, _ = classifier.evaluate(train_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=True)
			print("main_train_acc_2={:.4f} {:.4f}, main_train_f1_2={:.4f}".format(main_train_acc_2[0], main_train_acc_2[1], main_train_f1))
			metrics.update({"main_train_acc_2": main_train_acc_2[1], "main_train_f1_2": main_train_f1_2})

		if not main_classes:
			val_acc, val_f1, val_cons_loss = classifier.evaluate(val_set, batch_size, device)
			print("val_acc={:.4f} {:.4f}, val_f1={:.4f}, val_constr_loss={:.4f}".format(val_acc[0], val_acc[1], val_f1, val_cons_loss))
			metrics.update({"val_acc": val_acc[1], "val_f1": val_f1, "val_cons_loss": val_cons_loss})
		main_val_acc, main_val_f1, _ = classifier.evaluate(val_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=False)
		print("main_val_acc_={:.4f} {:.4f}, main_val_f1_={:.4f}".format(main_val_acc[0], main_val_acc[1], main_val_f1))
		metrics.update({"main_val_acc": main_val_acc[1], "main_val_f1": main_val_f1})
		if dataset is not dl.PASCAL_PART:
			main_val_acc_2, main_val_f1_2, _ = classifier.evaluate(val_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=True)
			print("main_val_acc_2={:.4f} {:.4f}, main_val_f1_2={:.4f}".format(main_val_acc_2[0], main_val_acc_2[1], main_val_f1_2))
			metrics.update({"main_val_acc_2": main_val_acc_2[1], "main_val_f1_2": main_val_f1_2})

	if test_dir is not None:
		print("Testing on the test set...")
		if not extract_features:
			test_set = dl.SemiSupDataset(test_dir, dataset, classifier.transforms, load_in_memory, seed=seed, main_classes=main_classes)
		else:
			test_set = dl.ExtractFeaturesSemiSupDataset(test_dir, dataset, classifier.transforms, seed=seed,
			                                            fine_tuning=fine_tuning, device=device, main_classes=main_classes)

		if not main_classes:
			test_acc, test_f1, test_constraint_loss = classifier.evaluate(test_set, batch_size, device)
			print("test_acc={:.4f}, {:.4f}, test_f1={:.4f}, test_constr_loss={:.4f}".format(test_acc[0], test_acc[1], test_f1, test_constraint_loss))
			metrics.update({"test_acc": test_acc[1], "test_f1": test_f1, "test_cons_loss": test_constraint_loss})
		main_test_acc, main_test_f1, _ = classifier.evaluate(test_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=False)
		print("main_test_acc={:.4f}, {:.4f}, main_test_f1={:.4f}".format(main_test_acc[0], main_test_acc[1], main_test_f1))
		metrics.update({"main_test_acc": main_test_acc[1], "main_test_f1": main_test_f1})
		if dataset is not dl.PASCAL_PART:
			main_test_acc_2, main_test_f1_2, _ = classifier.evaluate(test_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=True)
			print("main_test_acc_2={:.4f} {:.4f}, main_test_f1_2={:.4f}".format(main_test_acc_2[0], main_test_acc_2[1], main_test_f1_2))
			metrics.update({"main_test_acc_2": main_test_acc_2[1], "main_test_f1_2": main_test_f1_2})

		if extract_features:
			loc_args["extract_features"] = False
			model_name = get_model_path_name(loc_args, important_args, model_folder, main_classes)
			if file_exists(model_name):
				net = rx.MODELS[model](dataset=dataset, name=model_name, l_r=l_r, seed=seed, extract_features=False, main_classes=main_classes)
				net.load(device)
			else:
				print("Converting model...")
				net = classifier.convert(model_name)

			# Checking model conversion correctness and saving converted model
			# test_set = dl.SemiSupDataset(test_dir, dataset, classifier.transforms, load_in_memory=False, seed=seed, main_classes=main_classes)
			# main_test_acc_conv, main_test_f1_conv, _ = net.evaluate(test_set, batch_size, device, classes=dl.MAIN_CLASSES[dataset], get_max=False)
			# print("main_test_acc={:.4f}, {:.4f}, main_test_f1={:.4f}".format(main_test_acc_conv[0], main_test_acc_conv[1], main_test_f1_conv))
			# assert main_test_acc[1] == main_test_acc_conv[1] and main_test_f1 == main_test_f1_conv, \
			# 	"Error while converting model, should have accuracy {:.2f}, it has {:.2f}; f1 {:.2f} it has {:.2f}".format(
			# 		main_test_acc[1], main_test_acc_conv[1], main_test_f1, main_test_f1_conv)
			net.save(set_trained=True)

	print("\nDone!\n")

	return metrics


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='')
	# parser.add_argument('--model', default=rx.RESNET_TL, type=str, choices=rx.MODELS,
	#                     help='name of the model to use (default: ' + rx.RESNET_TL + ')')
	# parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'],
	#                     help='mode in {"train", "eval"}')
	# parser.add_argument('--train_dir', type=str, default='../data/cifar_splits/train',
	#                     help='train dataset folder (default: ../data/cifar_splits/train)')
	# parser.add_argument('--val_dir', type=str, default='../data/cifar_splits/val',
	#                     help='validation dataset folder ../data/cifar_splits/val')
	# parser.add_argument('--test_dir', type=str, default='../data/cifar_splits/test',
	#                     help='test dataset folder (default: ../data/cifar_splits/test')
	# parser.add_argument('--dataset', type=str, default=dl.CIFAR_100, choices=dl.DATASETS,
	#                     help='dataset to use (default ' + dl.CIFAR_100 + ')')
	# parser.add_argument('--batch_size', type=int, default=64,
	#                     help='input batch size for training (default: 64)')
	# parser.add_argument('--epochs', type=int, default=300,
	#                     help='number of epochs to train (default: 300)')
	# parser.add_argument('--lr', type=float, default=0.0001, dest="l_r",
	#                     help='learning rate (default: 0.0001)')
	# parser.add_argument('--constr_weight', type=float, default=0.0,
	#                     help='weight of the constraint penalty loss (default: 0.0)')
	# parser.add_argument('--sup_data_frac', type=float, default=0.3,
	#                     help='fraction of training points that will be supervised (semi-sup. setting) (default: 0.33)')
	# parser.add_argument('--partially_labeled', type=float, default=1.0,
	#                     help='fraction of labels that will be given (semi-sup. setting) (default: 1.0)')
	# parser.add_argument('--device', default='cpu', type=str,
	#                     help='device to be used (default: cpu)')
	# parser.add_argument('--output', default='1', type=int,
	#                     help='output training epoch  (default: 1 (true))')
	# parser.add_argument('--load_in_memory', default='0', type=int,
	#                     help='whether to load in memory the dataset when training (default: 0 (false))')
	#
	# args: dict = vars(parser.parse_args())
	# args['output'] = not args['output'] == 0
	# args['load_in_memory'] = not args['load_in_memory'] == 0

	# main(**args)

	base_path = '..'
	model = rx.RESNET_FT
	dataset = dl.CIFAR_100
	lr = 1e-4
	sup_data_frac = 0.3
	partially_labeled = 1.0
	batch_size = 64
	epochs = 100
	device = "cuda:1" if torch.cuda.is_available() else "cpu"
	extract_features = True
	load_in_memory = True
	main_classes = True
	constr_weight = 0.0  # 3.0, 10.0]
	train_dir = os.path.join("..", "data", "cifar_splits", "train")
	val_dir = os.path.join("..", "data", "cifar_splits", "val")
	test_dir = os.path.join("..", "data", "cifar_splits", "test")
	mode = "train"
	seed = 2

	main(model=model, dataset=dataset, l_r=lr, sup_data_frac=sup_data_frac, partially_labeled=partially_labeled,
	     batch_size=batch_size, epochs=epochs, device=device, extract_features=extract_features, main_classes=main_classes,
	     constr_weight=constr_weight, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
	     load_in_memory=load_in_memory, mode=mode, seed=seed)

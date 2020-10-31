from torchvision import transforms
import rexlearnclassifier as rx
import dataset_loader as dl

TRANSFORMS = {
	# ANIMAL DATASET TRANSFORMS
	rx.SIMPLE_CNN + dl.ANIMAL_DATASET: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4755, 0.4670, 0.4081], std=[0.2129, 0.2057, 0.2031]),
	]),

	rx.SIMPLE_CNN + dl.ANIMAL_DATASET_GRAY: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4609], std=[0.2045]),
	]),

	rx.RESNET_FT + dl.ANIMAL_DATASET: transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

	rx.RESNET_TL + dl.ANIMAL_DATASET: transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

	# CIFAR100 TRANSFORMS
	rx.SIMPLE_CNN + dl.CIFAR_100: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
	]),

	rx.SIMPLE_CNN + dl.CIFAR_100_GRAY: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4856], std=[0.1940]),
	]),

	rx.RESNET_FT + dl.CIFAR_100: transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

	rx.RESNET_TL + dl.CIFAR_100: transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

	# PASCAL_PART TRANSFORMS
	rx.SIMPLE_CNN + dl.PASCAL_PART: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
	]),

	rx.SIMPLE_CNN + dl.PASCAL_PART_GRAY: transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4856], std=[0.1940]),
	]),

	rx.RESNET_FT + dl.PASCAL_PART: transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

	rx.RESNET_TL + dl.PASCAL_PART: transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]),

}



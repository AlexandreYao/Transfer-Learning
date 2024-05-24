from torchvision.datasets import ImageFolder
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import time
import os


class FilteredImageFolder(ImageFolder):
    """
    A custom dataset class that extends torchvision.datasets.ImageFolder to include only a specified subset of classes.
    Args:
        root (str): Root directory path.
        classes_to_include (list): List of class names to include in the dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    Attributes:
        classes_to_include (list): List of class names to include in the dataset.
        samples (list): List of (sample path, class_index) tuples filtered to include only the specified classes.
        targets (list): List of class_index values for each sample in the filtered dataset.
        class_to_idx (dict): Dictionary mapping class names to class indices for the filtered dataset.
        idx_to_class (dict): Dictionary mapping class indices to class names for the filtered dataset.
        classes (list): List of class names included in the filtered dataset.
    Example:
        >>> data_dir = 'path/to/your/data'
        >>> classes_to_include = ['class1', 'class2', 'class3']
        >>> data_transforms = transforms.Compose([
        >>>     transforms.Resize((128, 128)),
        >>>     transforms.ToTensor(),
        >>> ])
        >>> dataset = FilteredImageFolder(root=data_dir, classes_to_include=classes_to_include, transform=data_transforms)
    """

    def __init__(self, root, classes_to_include, transform=None):
        self.classes_to_include = classes_to_include
        super().__init__(root, transform=transform)
        # Filter the samples to include only the desired classes
        filtered_samples = []
        for path, target in self.samples:
            class_name = self.classes[target]
            if class_name in self.classes_to_include:
                filtered_samples.append((path, target))
        self.samples = filtered_samples
        self.targets = [s[1] for s in self.samples]
        # Recompute class_to_idx and idx_to_class based on filtered classes
        filtered_class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes_to_include)
        }
        self.class_to_idx = filtered_class_to_idx
        self.idx_to_class = {
            idx: class_name for class_name, idx in filtered_class_to_idx.items()
        }
        # Update targets to new indices
        self.targets = [
            self.class_to_idx[self.classes[target]] for target in self.targets
        ]
        self.samples = [
            (path, self.class_to_idx[self.classes[target]])
            for path, target in self.samples
        ]
        self.classes = self.classes_to_include


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    device,
    scheduler,
    num_epochs=25,
    use_validation=False
):
    """
    Train a deep learning model with given data loaders and other training components.

    Args:
        model (torch.nn.Module): The model to train.
        dataloaders (dict): A dictionary containing 'train' and 'val' DataLoader objects.
        dataset_sizes (dict): A dictionary containing the sizes of the training and validation datasets.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        device (torch.device): The device to train the model on (e.g., 'cuda' or 'cpu').
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        num_epochs (int, optional): The number of epochs to train the model. Default is 25.

    Returns:
        model (torch.nn.Module): The trained model with the best validation accuracy.

    Example:
        >>> model = models.resnet18(pretrained=True)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        >>> scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        >>> model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, scheduler, num_epochs=25)

    The function follows these steps:
    1. Saves the initial model parameters.
    2. Iterates over the specified number of epochs.
    3. For each epoch, switches between training and validation phases.
    4. In the training phase, updates the model parameters using backpropagation.
    5. Tracks the loss and accuracy for both training and validation phases.
    6. Saves the model parameters when the validation accuracy improves.
    7. Prints the loss and accuracy for each phase at the end of each epoch.
    8. Loads the model parameters that achieved the best validation accuracy.

    Raises:
        AssertionError: If the dataset sizes are not specified for both 'train' and 'val' phases.
        AssertionError: If the dataloaders are not specified for both 'train' and 'val' phases.
    """
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        nb_batches = len(dataloaders["train"])
        nb_batches_to_display = nb_batches // 10
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("=" * 50)
            nb_samples_used = 0
            # Each epoch has a training and validation phase
            for phase in ["train", "val"] if use_validation else ["train"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for batch_index, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            nb_samples_used += inputs.size(0)
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if (batch_index + 1) % nb_batches_to_display == 0 and phase == "train":
                        print(
                            f"\tBatch [{batch_index+1}/{nb_batches}], loss: {(running_loss / nb_samples_used):.4f}"
                        )
                if phase == "train":
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            print()
        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")
        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def count_trainable_parameters(m):
    """
    Compte le nombre de paramètres entraînables dans un modèle PyTorch.
    Args:
        m (torch.nn.Module): Le modèle PyTorch dont les paramètres entraînables doivent être comptés.
    Returns:
        int: Le nombre total de paramètres entraînables dans le modèle.
    Example:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> num_trainable_params = count_trainable_parameters(model)
        >>> print(num_trainable_params)
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_elements_per_class(image_folder):
    """
    Compte le nombre d'éléments par classe dans un dataset ImageFolder.
    Args:
        image_folder (torchvision.datasets.ImageFolder): Le dataset ImageFolder contenant les images et les classes.
    Returns:
        dict: Un dictionnaire où les clés sont les noms des classes et les valeurs sont le nombre d'éléments par classe.
    Example:
        >>> data_dir = 'path/to/your/data'
        >>> dataset = ImageFolder(root=data_dir)
        >>> class_counts = count_elements_per_class(dataset)
        >>> for class_name, count in class_counts.items():
        >>>     print(f"Classe: {class_name}, Nombre d'éléments: {count}")
    """
    class_counts = Counter()
    for _, target in image_folder.samples:
        class_name = image_folder.classes[target]
        class_counts[class_name] += 1
    return dict(class_counts)


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def show_images(dataset, nb_rows=4, nb_cols=3, class_map=None):
    """
    Display a grid of images from a dataset.
    Args:
        dataset (Dataset): The dataset from which to display images. The dataset should return (image, label) tuples.
        nb_rows (int, optional): The number of rows in the image grid. Must be greater than or equal to 1. Default is 4.
        nb_cols (int, optional): The number of columns in the image grid. Must be greater than or equal to 1. Default is 3.
        class_map (dict, optional): A dictionary mapping class labels to class names. If None, labels are displayed as is. Default is None.
    Raises:
        AssertionError: If `nb_rows` or `nb_cols` is less than 1.
        AssertionError: If the dataset has fewer samples than `nb_rows * nb_cols`.
    Example:
        >>> from torchvision.datasets import CIFAR10
        >>> import torchvision.transforms as transforms
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        >>> class_map = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        >>> show_images(dataset, nb_rows=4, nb_cols=3, class_map=class_map)
    """
    nb_samples = len(dataset)
    num_samples_to_display = nb_rows * nb_cols
    assert (
        nb_rows >= 1 and nb_cols >= 1
    ), "Le nombre de colonnes et le nombre de lignes doivent être supérieurs ou égaux à 1"
    assert (
        nb_samples >= num_samples_to_display
    ), f"Trop peu de données à afficher. Il y a {nb_samples} données, vous souhaitez en afficher {num_samples_to_display} !"
    _, axes = plt.subplots(nb_rows, nb_cols, figsize=(15, 8))
    for i, ith_dataset in enumerate(
        tqdm(np.random.randint(low=0, high=nb_samples, size=num_samples_to_display))
    ):
        image, label = dataset[ith_dataset]
        ii = i // nb_cols
        jj = i % nb_cols
        axes[ii, jj].imshow(image.numpy().transpose((1, 2, 0)))
        axes[ii, jj].axis("off")
        axes[ii, jj].set_title(class_map[label] if class_map is not None else label)
    plt.tight_layout()
    plt.show()
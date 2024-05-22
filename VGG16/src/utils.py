from torchvision.datasets import ImageFolder
from collections import Counter


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
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetEmbeddings:
    def __init__(self, pkl_file, split_type='train', validation_split=0.1, test_split=0.2, random_state=42):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = np.array(data['embeddings'])
        labels = np.array(data['labels'])

        # Filtra i sample con label "no_annotations"
        mask = labels != "no_annotations"
        self.embeddings = embeddings[mask]
        self.labels = labels[mask]

        # Converti le labels da stringhe a interi
        label_mapping = {"tumorali": 1, "non_tumorali": 0}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        
        # Perform train/test/validation split
        self.train_embeddings, self.test_embeddings, self.train_labels, self.test_labels = train_test_split(
            self.embeddings, self.labels, test_size=test_split, random_state=random_state)
        
        if validation_split > 0:
            self.train_embeddings, self.val_embeddings, self.train_labels, self.val_labels = train_test_split(
                self.train_embeddings, self.train_labels, test_size=validation_split, random_state=random_state)
        else:
            self.val_embeddings, self.val_labels = None, None

        if split_type == 'train':
            self.embeddings, self.labels = self.train_embeddings, self.train_labels
        elif split_type == 'test':
            self.embeddings, self.labels = self.test_embeddings, self.test_labels
        elif split_type == 'validation':
            if self.val_embeddings is not None:
                self.embeddings, self.labels = self.val_embeddings, self.val_labels
            else:
                raise ValueError("Validation split is not defined. Set validation_split > 0.")
        else:
            raise ValueError("split_type should be 'train', 'test', or 'validation'")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
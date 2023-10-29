import os
import numpy as np
import pickle as pkl


def load_data(base_path, domain):
    """
    Load embeddings and labels for a given domain.

    Args:
    - base_path (str): base directory for data
    - domain (str): specific domain or "general"

    Returns:
    - data: loaded data embeddings
    - labels: concatenated labels from train and test
    """
    with open(f'{base_path}/{domain}/sentemb/sentemb_unlabeled.pkl', 'rb') as f:
        data = pkl.load(f)
    
    with open(f'{base_path}/{domain}/label_domain/label_domain_train_sentemb_unlabeled.pkl', 'rb') as f:
        labels_train = pkl.load(f)
        
    with open(f'{base_path}/{domain}/label_domain/label_domain_test_sentemb_unlabeled.pkl', 'rb') as f:
        labels_test = pkl.load(f)

    labels = np.hstack((labels_train, labels_test))
    return data.transpose(), labels


def sort_array(array_to_sort, array_ref):
    """
    
    Function for sorting two arrays such that both arrays have the same labels.
    It sorts one array based on another reference array's structure.
# It returns indeces_sorted which consists of indices.
    """
    y, y_ref = array_to_sort[0].astype(int), array_ref[0].astype(int)
    
    # Find indices where labels are 0 or 1
    indices_by_label = {label: np.where(y == label)[0].tolist() for label in [0, 1]}
    
    # Build a sorted list of indices based on y_ref
    idx_sorted = [indices_by_label[label].pop(0) for label in y_ref]
    
    return np.array(idx_sorted)

def ensure_directory_exists(filename):
    """
    Create directory for the given filename if it does not exist.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(data, labels, split_indices, filename):
    """
    Save data and labels to pickle based on split indices.
    """
    data_to_save = np.vstack((data[split_indices[1]:split_indices[2]], data[split_indices[2]:]))
    labels_to_save = np.hstack((labels[:, split_indices[1]:split_indices[2]], labels[:, split_indices[2]:]))
    
    ensure_directory_exists(filename)  # Make sure directory exists
    with open(filename, 'wb') as f:
        pkl.dump(data_to_save, f)

    label_filename = filename.replace("data", "labels")
    ensure_directory_exists(label_filename)  # Make sure directory exists
    with open(label_filename, 'wb') as f:
        pkl.dump(labels_to_save, f)
 
 

def main():
    # Load data
    data_general, labels_general = load_data('data/sentence_embeddings/general/unsorted', 'general')
    data_spec, labels_spec = load_data('data/sentence_embeddings/specific', 'sentemb_unlabeled4_5')

    labels_spec2 = np.repeat(labels_spec, repeats=3, axis=1)
    ind = sort_array(labels_general, labels_spec2)

    data_left = np.delete(data_general, ind, axis=0)
    labels_left = np.delete(labels_general, ind, axis=1)

    data_general = data_general[ind]
    labels_general = labels_general[:, ind]

    # Data splitting and saving
    split_indices = [0, 4200, 4800]
    base_path = "data/sentence_embeddings/general/sorted"

    save_data(data_general, labels_general, split_indices, f"{base_path}/val_test/vt_data.p")
    save_data(np.vstack((data_general[:4200], data_left)), np.hstack((labels_general[:, :4200], labels_left)), split_indices, f"{base_path}/train/train_data4_5.p")


    save_data(data_general, labels_general, split_indices, f"{base_path}/val_test/vt_data.p")
    save_data(np.vstack((data_general[:4200], data_left)), np.hstack((labels_general[:, :4200], labels_left)), split_indices, f"{base_path}/train/train_data4_3.p")

    if __name__ == "__main__":
        main()



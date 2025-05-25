import scipy.io


def load_mat_file_into_numpy(file_path_data, file_path_label):
    mat = scipy.io.loadmat(file_path_data)
    data = mat['F'].transpose(0, 2, 1)

    mat = scipy.io.loadmat(file_path_label)
    label = mat['ans']
    return data, label


if __name__ == '__main__':
    data_path = '../data/Wmel_Feature.mat'
    label_path = '../data/Label.mat'
    ori_features, label = load_mat_file_into_numpy(data_path, label_path)




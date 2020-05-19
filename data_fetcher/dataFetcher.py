"""
    Implement different data format for different data set

    Author: Haotian Xue
"""

from utils.data_set_class import DataFetcher
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import cuda


class SenSemEvalDataSet(Dataset):

    """
    Sentence level SemEval2020 task6 data set
    """

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=None, is_gpu=cuda.is_available()):
        self.is_gpu = is_gpu
        self.data_fetcher = SenSemEvalHelper(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.x = self.data_fetcher.data_x
        self.y = self.data_fetcher.data_y
        self.num_data = self.y.shape[0]
        self.word_embedding = self.data_fetcher.word_embedding
        self.max_sen_len = max_sen_len

    def __getitem__(self, index):
        x_i, y_i = self.x[index], self.y[index]
        if self.max_sen_len is not None:
            x_i = self.pad(x_i, self.max_sen_len)
        if self.is_gpu:
            return torch.LongTensor(x_i).cuda(), torch.LongTensor(y_i).cuda()
        return x_i, y_i

    def __len__(self):
        return self.data_fetcher.data_y.shape[0]

    def pad(self, sen, max_sen_len):
        """
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        """
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


class SenSemEvalHelper(DataFetcher):

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150):
        super(SenSemEvalHelper, self).__init__(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.data_x, self.data_y, self.max_sen_len = self.load_data()

    def get_x_id(self, x_tokens):
        """
        Given tokens of the sentence, return the ids of each token
        :param x_tokens:
        :return: [token_id]
        """
        x = []
        for x_token in x_tokens:
            if x_token in self.word2id:
                x.append(self.word2id[x_token])
            else:
                x.append(self.word2id[self.OOV])
        return x

    def load_data(self):
        """
        Extract raw input into matrix form
        :return: data_x :: ndarray (if padding, [ndarray] otherwise);  data_y: ndarray
        """
        data_x = []
        data_y = []
        max_len = 0
        with open(self.data_path, 'r') as file:
            for i, line in enumerate(file):
                tokens = line.strip().split()[1:]  # remove first ""
                if len(tokens) == 0:
                    continue
                x_tokens = tokens[:-1]
                if x_tokens[0].isdigit():
                    x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
                if self.padding and len(x_tokens) > max_len:
                    max_len = len(x_tokens)
                y_token = int(tokens[-1][1])
                data_x.append(torch.tensor(self.get_x_id(x_tokens), dtype=torch.int64))
                data_y.append(y_token)
        if self.padding:
            data_x = pad_sequence(data_x, batch_first=True).numpy()
        data_y = np.array(data_y, dtype=np.int64)
        return data_x, data_y, max_len

    def pad(self, sen, max_sen_len):
        """
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        """
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


# TODO: error check data set: try to seperate each data set class into different python files
class ErrorCheckSemEvalDataSet(Dataset):

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=None, is_gpu=cuda.is_available()):
        self.is_gpu = is_gpu
        self.fetcher = ErrorCheckSemEvalHelper(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.x = self.fetcher.data_x
        self.y = self.fetcher.data_y
        self.tokens = self.fetcher.data_error_check
        self.num_data = self.y.shape[0]
        self.word_embedding = self.fetcher.word_embedding
        self.max_sen_len = max_sen_len

    def __getitem__(self, index):
        x_i, y_i = self.x[index], self.y[index]
        if self.max_sen_len is not None:
            x_i = self.pad(x_i, self.max_sen_len)
        if self.is_gpu:
            return torch.LongTensor(x_i).cuda(), torch.LongTensor(y_i).cuda()
        return x_i, y_i

    def __len__(self):
        return self.num_data

    def pad(self, sen, max_sen_len):
        """
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        """
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


# TODO: error-checking data set
class ErrorCheckSemEvalHelper(DataFetcher):

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150):
        super(ErrorCheckSemEvalHelper, self).__init__(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.data_x, self.data_y, self.max_sen_len, self.data_error_check = self.load_data()
        print("-----error checking data set built-----")

    def get_x_id(self, x_tokens):
        """
        Given tokens of the sentence, return the ids of each token
        :param x_tokens:
        :return: [token_id]
        """
        x = []
        for x_token in x_tokens:
            if x_token in self.word2id:
                x.append(self.word2id[x_token])
            else:
                x.append(self.word2id[self.OOV])
        return x

    def load_data(self):
        """
        Extract raw input into matrix form
        :return: data_x :: ndarray (if padding, [ndarray] otherwise);  data_y: ndarray
        """
        data_x = []
        data_y = []
        max_len = 0
        data_tokens = []
        with open(self.data_path, 'r') as file:
            for i, line in enumerate(file):
                tokens = line.strip().split()[1:]  # remove first ""
                if len(tokens) == 0:
                    continue
                x_tokens = tokens[:-1]
                if x_tokens[0].isdigit():
                    x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
                data_tokens.append(line)
                if self.padding and len(x_tokens) > max_len:
                    max_len = len(x_tokens)
                y_token = int(tokens[-1][1])
                data_x.append(torch.tensor(self.get_x_id(x_tokens), dtype=torch.int64))
                data_y.append(y_token)
        if self.padding:
            data_x = pad_sequence(data_x, batch_first=True).numpy()
        data_y = np.array(data_y, dtype=np.int64)
        return data_x, data_y, max_len, data_tokens

    def pad(self, sen, max_sen_len):
        """
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        """
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


# ToDo: char level data set
class CharSemEvalDataSet(Dataset):

    """
        Character level SemEval2020 task6 data set
    """

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=None, is_gpu=cuda.is_available()):
        self.is_gpu = is_gpu
        self.data_fetcher = CharSemEvalHelper(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.x = self.data_fetcher.data_x
        self.y = self.data_fetcher.data_y
        self.num_data = self.y.shape[0]
        self.max_sen_len = max_sen_len

    def __getitem__(self, index):
        x_i, y_i = self.x[index], self.y[index]
        if self.max_sen_len is not None:
            x_i = self.pad(x_i, self.max_sen_len)
        if self.is_gpu:
            return torch.LongTensor(x_i).cuda(), torch.LongTensor(y_i).cuda()
        return x_i, y_i

    def __len__(self):
        return self.data_fetcher.data_y.shape[0]

    def pad(self, sen, max_sen_len):
        """
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        """
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


class CharSemEvalHelper(DataFetcher):

    def __init__(self, data_path, w2v_path, emb_dim, padding=False, max_sen_len=150):
        super(CharSemEvalHelper, self).__init__(data_path, w2v_path, emb_dim, padding, max_sen_len)
        self.alphabet = \
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.OOV = 'OOV'
        self.NULL = 'NULL'
        self.word2id = self.make_dict()
        self.data_x, self.data_y, self.max_sen_len = self.load_data()

    def make_dict(self):
        """
        Get a dict form of alphabet
        :return: {'a': 0, 'b': 1, ....}  len: 97
        """
        size = len(self.alphabet)
        res = {}
        res[self.NULL] = 0
        for i, c in enumerate(self.alphabet):
            res[c] = i + 1
        res[self.OOV] = size + 1
        return res

    def get_x_id(self, x_tokens):
        """
        Given tokens of the sentence, return the ids of each token (character in this case)
        :param x_tokens:
        :return: [token_id]
        """
        x = []
        for x_token in x_tokens:
            if x_token in self.word2id:
                x.append(self.word2id[x_token])
            else:
                x.append(self.word2id[self.OOV])
        return x

    def load_data(self):
        """
        Extract raw input into matrix form
        :return: data_x :: ndarray (if padding, [ndarray] otherwise);  data_y: ndarray
        """
        data_x = []
        data_y = []
        max_len = 0
        with open(self.data_path, 'r') as file:
            for i, line in enumerate(file):
                tokens = line.strip().split()[1:]  # remove first ""
                if len(tokens) == 0:
                    continue
                y_token = int(tokens[-1][1])
                data_y.append(y_token)
                x_tokens = tokens[:-1]
                if x_tokens[0].isdigit():
                    x_tokens = x_tokens[2:]  # ['5', '.', 'Science', ...] => ['Science', ...]
                x_tokens = " ".join(x_tokens)
                x_tokens = [c for c in x_tokens]
                if self.padding and len(x_tokens) > max_len:
                    max_len = len(x_tokens)
                data_x.append(torch.tensor(self.get_x_id(x_tokens), dtype=torch.int64))
        if self.padding:
            data_x = pad_sequence(data_x, batch_first=True).numpy()
        data_y = np.array(data_y, dtype=np.int64)
        return data_x, data_y, max_len


if __name__ == "__main__":
    ds = ErrorCheckSemEvalDataSet("../data/test.txt", "../data/word_embedding/glove.6B.50d.txt", 50, padding=True, max_sen_len=150)
    ds_loader = DataLoader(ds, 1, shuffle=False)
    print(ds.num_data)
    for i, d in enumerate(ds_loader):
        x, y = d
        print(x.size())
        print(x)
        print(y)
        break
    '''
    ds = CharSemEvalDataSet("../data/train.txt", None, 50, True)  # train: 842, test: 656
    train_size = int(0.8 * ds.num_data)
    test_size = ds.num_data - train_size
    train, test = torch.utils.data.random_split(ds, [train_size, test_size])
    ds_loader = DataLoader(train, 4, False)
    print(ds.data_fetcher.max_sen_len)
    for i, d in enumerate(ds_loader):
        x, y = d
        print(x.size())
        print(x)
        print(y.size())
        print(y)
        break
    '''
    """
    ds = SenSemEvalDataSet("../data/train.txt", "../data/word_embedding/glove.6B.50d.txt", 50, True)
    ds_loader = DataLoader(ds, 4, False)
    print(ds.num_data)
    for i, d in enumerate(ds_loader):
        x, y = d
        print(x.size())
        print(x)
        # print(y.size())
        print(y)
        break
    """

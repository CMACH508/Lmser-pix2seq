import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from hyper_params import hp
import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
from encoder import myencoder
from decoder import DecoderRNN
from utils.inference_sketch_processing import make_graph, draw_three, make_graph_


################################# load and prepare data
class SketchesDataset:
    def __init__(self, path: str, category: list, mode="train"):
        self.sketches = None
        self.sketches_categroy_count: list = []
        self.sketches_normed = None
        """上面两个sketches 是完全拷贝的"""
        self.max_sketches_len = 0
        self.path = path
        self.category = category
        self.mode = mode

        tmp_sketches = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            self.sketches_categroy_count.append(len(dataset[self.mode]))
            print(f"dataset: {c} added.")
        data_sketches = np.concatenate(tmp_sketches)
        print(f"length of train set: {len(data_sketches)}")

        data_sketches = self.purify(data_sketches)  # data clean.  # remove too long and too stort sketches.
        self.sketches = data_sketches.copy()
        self.sketches_normed = self.normalize(data_sketches)
        self.Nmax = self.max_size(data_sketches)  # max size of a sketch.
        print(f"max length of sketch is: {self.Nmax}")

    def max_size(self, sketches):
        """返回所有sketch中 转折最多的一个sketch"""
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches):
        """
        移除太短或过长的stroke
        移除单个stroke 太长的."""
        data = []
        for sketch in sketches:
            if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
        return data

    @staticmethod
    def calculate_normalizing_scale_factor(sketches):
        """计算所有sketches中的标准差"""
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor.
        将所有的sketches 标准化, 即除以标准差. 使得方差等于1"""
        data = []
        #scale_factor = self.calculate_normalizing_scale_factor(sketches)
        scale_factor = 44.18034
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data

    def get_sample(self, sketch_index: int):
        """
        :return:
        返回 batch, lengths. batch为sketches的连接, lengths是每一个sketch的长度列表
        """
        batch_idx = [sketch_index]
        batch_sketches = [self.sketches_normed[idx] for idx in batch_idx]  # 从标准化后的抽取
        batch_sketches_graphs = [self.sketches[idx] for idx in batch_idx]  # 图卷积使用, 图卷积不能使用归一化后的
        sketches = []
        lengths = []
        graphs = []  # (batch_size * graphs_num_constant, x, y) # 注意按照 graphs num 切分
        adjs = []
        index = 0
        for _sketch in batch_sketches:
            len_seq = len(_sketch[:, 0])  # sketch 笔画数量
            new_sketch = np.zeros((self.Nmax, 5))  # new a _sketch, all length of sketch in size is Nmax.
            new_sketch[:len_seq, :2] = _sketch[:, :2]  # 1. 将x y拷贝进新的sketch

            # set p into one-hot.
            new_sketch[:len_seq - 1, 2] = 1 - _sketch[:-1, 2]
            new_sketch[:len_seq, 3] = _sketch[:, 2]

            # len to Nmax set as 0,0,0,0,1
            new_sketch[(len_seq - 1):, 4] = 1
            new_sketch[len_seq - 1, 2:4] = 0  # x, y, 0, 0, 1
            lengths.append(len(_sketch[:, 0]))  # lengths is _sketch length, not new_sketch length.
            sketches.append(new_sketch)
            index += 1

        for _each_sketch in batch_sketches_graphs:
            _graph_tensor, _adj_matrix = make_graph(_each_sketch, graph_num=hp.graph_number,
                                                    graph_picture_size=hp.graph_picture_size, mask_prob=hp.mask_prob)
            graphs.append(_graph_tensor)
            adjs.append(_adj_matrix)

        if hp.use_cuda:
            batch = torch.from_numpy(np.stack(sketches, 1)).cuda().float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).cuda().float()  # (batch_size, len, 5)
            adjs = torch.from_numpy(np.stack(adjs, 0)).cuda().float()

        else:
            batch = torch.from_numpy(np.stack(sketches, 1)).float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).float()
            adjs = torch.from_numpy(np.stack(adjs, 0)).float()

        return batch, lengths, graphs, adjs


def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor,
                            sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                            rho_xy: torch.Tensor, greedy=False):
    """
    根据网络输出, 进行采样
    1. 获取 x, y的均值及标准差
    2. 计算相关系数
    """
    mu_x = mu_x.item()
    mu_y = mu_y.item()
    sigma_x = sigma_x.item()
    sigma_y = sigma_y.item()
    rho_xy = rho_xy.item()
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]

    sigma_x *= np.sqrt(hp.temperature)  # 乘以热度开根号
    sigma_y *= np.sqrt(hp.temperature)

    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, sketch_index, name='_output_', path="./visualize/"):
    """分离strokes, 并画图"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)  # 指出所有满足条件的坐标, +1 是因为split类似于[m:n]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    os.makedirs(f"{path}/{name}", exist_ok=True)
    name = f"{path}" + str(sketch_index) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


"""
encoder and decoder modules
"""


class Model:
    def __init__(self):
        if hp.use_cuda:
            self.encoder: nn.Module = myencoder(hps = hp).cuda()
            self.decoder: nn.Module = DecoderRNN().cuda()
        else:
            self.encoder: nn.Module = myencoder(hps =hp)
            self.decoder: nn.Module = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

        self.pi: torch.Tensor = torch.Tensor()
        self.z: torch.Tensor = torch.Tensor()
        self.mu_x: torch.Tensor = torch.Tensor()
        self.mu_y: torch.Tensor = torch.Tensor()
        self.sigma_x: torch.Tensor = torch.Tensor()
        self.sigma_y: torch.Tensor = torch.Tensor()
        self.rho_xy: torch.Tensor = torch.Tensor()
        self.q: torch.Tensor = torch.Tensor()

    def validate(self, sketch_dataset, save_middle_path="visualize"):
        self.encoder.eval()
        self.decoder.eval()
        # some print and save:
        with torch.no_grad():
            self.conditional_generation(sketch_dataset, save_middle_path)

    def conditional_generation(self, sketch_dataset, save_middle_path="visualize"):
        count = 0
        category_flag = 0
        category_name = sketch_dataset.category[category_flag].split(".")[0]
        category_count = sketch_dataset.sketches_categroy_count[category_flag]

        result_z_list = []
        ret_z_list = []

        for sketch_index, sketch in enumerate(sketch_dataset.sketches_normed):
            print(sketch_index)
            batch, lengths, graphs, adjs = sketch_dataset.get_sample(sketch_index)
            # encode:
            self.z, mu, sigma, _, _, _ = self.encoder(graphs)
            # result_z_list.append(self.z.cpu().numpy())
            result_z_list.append(mu.cpu().numpy())

            count += 1

            # if sketch_index % 100 != 0 or True:
            #     continue
            print(f"drawing {category_name} {count}")
            if hp.use_cuda:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
            else:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            s = sos
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            for i in range(hp.Nmax):  # Nmax = 151
                _input = torch.cat([s, self.z.unsqueeze(0)], 2)  # start of stroke concatenate with z
                # decode:

                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(_input, self.z, hidden_cell)

                hidden_cell = (hidden, cell)
                # sample from parameters:
                s, dx, dy, pen_down, eos = self.sample_next_state()
                # ------
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    # print(i)
                    break
            ret_seq_x = np.asarray(seq_x)
            ret_seq_y = np.asarray(seq_y)
            ret_seq_z = np.asarray(seq_z)
            ret_sequence = np.stack([ret_seq_x, ret_seq_y, ret_seq_z]).T
            _graph_tensor, _adj_matrix = make_graph_(ret_sequence, graph_num=hp.graph_number,
                                                    graph_picture_size=hp.graph_picture_size, mask_prob=0.0)
            _graph_tensor = torch.tensor(_graph_tensor).to(torch.float)
            _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.cuda().unsqueeze(0))
            ret_z_list.append(ret_mu.cpu().numpy())


            # visualize result:
            x_sample = np.cumsum(seq_x, 0)  # 累加, 梯形求和
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence = np.stack([x_sample, y_sample, z_sample]).T
            # # visualize result:
            _sketch = np.stack([seq_x, seq_y, seq_z]).T
            try:
                sketch_cv = draw_three(_sketch, img_size=256)
            except Exception as e:
                print('draw error')
                _sketch = np.zeros((256, 256, 1))

            print('draw:')
            os.makedirs(f"{save_middle_path}/sketch/{category_name}", exist_ok=True)
            cv2.imwrite(f"{save_middle_path}/sketch/{category_name}/{sketch_index}.jpg", sketch_cv)
            # cv2.imwrite(f"{save_middle_path}/sketch/{category_name}/{sketch_index}.jpg", _sketch)

            #make_image(sequence, count - 1, name=f"_{category_name}", path=f"./{save_middle_path}/sketch/")
            if count == category_count:
                os.makedirs(f"{save_middle_path}/npz", exist_ok=True)
                np.savez(f"./{save_middle_path}/npz/{category_name}.npz", z=np.array(result_z_list))
                result_z_list = []
                os.makedirs(f"{save_middle_path}/retnpz", exist_ok=True)
                np.savez(f"./{save_middle_path}/retnpz/{category_name}.npz", z=np.array(ret_z_list))
                ret_z_list = []
                count = 0
                print(f"{category_name} finished")
                category_flag += 1
                if category_flag < len(hp.category):
                    category_name = sketch_dataset.category[category_flag].split(".")[0]
                    category_count = sketch_dataset.sketches_categroy_count[category_flag]


                                                                      
                      

    def conditional_generate_by_z(self, z, index=-1, plt_show=False):  #
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if hp.use_cuda:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
            else:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            s = sos
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            for i in range(151):  # Nmax = 151
                _input = torch.cat([s, z.unsqueeze(0)], 2)  # start of stroke concatenate with z
                # decode:
                self.pi, \
                self.mu_x, self.mu_y, \
                self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = self.decoder(_input, z, hidden_cell)
                hidden_cell = (hidden, cell)
                # sample from parameters:
                s, dx, dy, pen_down, eos = self.sample_next_state()
                # ------
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    seq_x.append(0)
                    seq_y.append(0)
                    seq_z.append(True)
                    break
            # visualize result:
            x_sample = np.cumsum(seq_x, 0)  # 累加, 梯形求和
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
                                      
            sequence = np.stack([x_sample, y_sample, z_sample]).T
            if plt_show:
                make_image(sequence, index, name=f"_z_generated", path="./visualize/generate_z/")
            _sketch = np.stack([seq_x, seq_y, seq_z]).T
            return _sketch

    def sample_next_state(self):
        def adjust_temp(pi_pdf):
            """
            SoftMax
            """
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)  # 抽一个数字
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)  # 抽一个数字
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)  # get samples.
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        if hp.use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)


if __name__ == "__main__":
    import random
    import glob
    import cv2

    hp.mask_prob = 0.1 #0.0 0.1 0.3 0.5
    sketch_dataset = SketchesDataset(hp.data_location, hp.category, "test")
    hp.Nmax = sketch_dataset.Nmax
    hp.Nmax = 177
    hp.temperature = 0.01
    model = Model()
    model.load(f"./model_save/encoderRNN_epoch_150000.pth",
               f"./model_save/decoderRNN_epoch_150000.pth")

    print(hp.mask_prob, hp.Nmax)
    model.validate(sketch_dataset, save_middle_path=f"results/{hp.mask_prob}")
    exit(0)


import torch


class HParams:
    def __init__(self):
        self.data_location = './dataset/'#location of  of origin data
        #self.category = ["airplane.npz"]
        self.category = ["airplane.npz", "angel.npz", "alarm clock.npz", "apple.npz",
                         "butterfly.npz", "belt.npz", "bus.npz",
                         "cake.npz", "cat.npz", "clock.npz", "eye.npz", "fish.npz",
                         "pig.npz", "sheep.npz", "spider.npz", "The Great Wall of China.npz",
                         "umbrella.npz"]
        self.model_save = "model_save"
        
        self.dec_hidden_size = 512 #Recommended settings are 1024 and above
        self.Nz = 128  # encoder output size
        self.M = 20 
        self.dropout = 0.0
        self.batch_size = 100


        # Unused
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.l1weight = 0.0


        self.lr = 0.001
        self.lr_decay = 0.99999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.2
        self.T = 2 # Lmser loops
        self.max_seq_length = 200  
        self.min_seq_length = 0  


        #Unused (only for SketchHealer). 
        self.Nmax = 0  # max stroke number of a sketch
        self.graph_number = 1 + 20  # the number of graph for each sketch,first for global
        self.graph_picture_size = 128  # size of graph 128
        self.out_f_num = 512  # 1000 -> 512
        self.res_number = 2


        self.mask_prob = 0.1 # 0.1 for train. When inference is performed, it is modified in inference.py (Line 395).
        self.use_cuda = torch.cuda.is_available()
        self.l2weight = 0.5



hp = HParams()


import torch
from torch import nn
#x.detach() 会返回一个与 x 数值相同，但不参与反向传播梯度计算的张量
def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha) 

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        #定义一个 基于 GRU 的策略头/actor 模块
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)# 初始化时缩小权重幅度一半，避免输入太大

        self.gru = nn.GRUCell(192, 192)  # 输入是192维，隐状态也是192维，记忆单元
        self.fc = nn.Linear(192, dim_action, bias=False)    # 最终映射到动作空间
        self.fc.weight.data.mul_(0.01)  # 输出层权重初始化得很小，避免初始动作过大
        self.act = nn.LeakyReLU(0.05)   # 激活函数，负数部分留一点梯度

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x) #1. 图像输入提取特征
        x = self.act(img_feat + self.v_proj(v))  # 2. 拼接额外观测 v 的投影，并激活
        hx = self.gru(x, hx)    # 3. 传入 GRUCell，更新隐状态
        act = self.fc(self.act(hx))  # 4. 映射到动作空间，并激活
        return act, None, hx


if __name__ == '__main__':
    Model()

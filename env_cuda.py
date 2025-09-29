import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda

# 梯度衰减层，类似于model.py x*alpha + x.detach()*(1-alpha) 的写法
class GDecay(torch.autograd.Function):  #想要自己定义某个操作的前向和反向传播，就需要继承 torch.autograd.Function
    @staticmethod   #要实现autograd.Function，必须实现 forward 和 backward 静态方法
    def forward(ctx, x, alpha):
        ctx.alpha = alpha   #衰减因子
        return x

    @staticmethod
    def backward(ctx, grad_output): #grad_output 是链式法则里从后面传回来的梯度
        # 返回的是相对于 forward 函数输入的梯度，这里是 x 的梯度和 alpha 的梯度
        return grad_output * ctx.alpha, None   

g_decay = GDecay.apply  #起别名 .apply -> 执行 forward，同时在计算图里注册 backward


class RunFunction(torch.autograd.Function):
    #调用 quadsim_cuda.run_forward（应该是 C++/CUDA 写的四旋翼动力学仿真函数），返回动作、位置、速度、加速度。
    @staticmethod   #要实现autograd.Function，必须实现 forward 和 backward 静态方法
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        # 上下文对象 ctx 用于在 forward 和 backward 之间传递信息
        # 会把一组 Tensor 保存在 ctx 中，backward 时可以通过 ctx.saved_tensors 取回
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay,  v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next
    #调用 quadsim_cuda.run_backward，把梯度传回去
    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        # 在 torch.autograd.Function 里，backward 的返回值顺序必须严格和 forward 的输入顺序一一对应
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class Env:
    # fov_x_half_tan: 相机水平视场角的一半的正切值，影响成像投影
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay    #传给仿真函数的梯度衰减系数
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device) #控制尺度/方向/大小 权重
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)   #控制平移/位置 偏置
        # 体素障碍物参数。6 维向量 → 3 个尺度 + 3 个位
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device) 
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device) 
        # 地面体素，形状更扁平
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        # 圆柱体障碍参数
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        # 水平方向的圆柱体
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        # 方形门（无人机竞速里常见的 gate）参数
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        # 风场参数（权重）
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        # 重力常数
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        # 额外的屋顶约束参数
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        # 时间/空间的细分，用于轨迹采样或者渲染。这里生成 10 个点，范围 [0, 1/15]。
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        #预定义了 8 个初始位置点，分布在左右两侧
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        #目标点，同样预定义了 8 个
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        #光流占位张量
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        #保存环境控制参数
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        B = self.batch_size
        device = self.device
        # 随机化相机角度
        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        # 辅助张量，生成全0、全1的张量，后面用来拼接成旋转矩阵
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        # 构造旋转矩阵，绕y轴旋转
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b#每个 batch 生成 30 个“球”障碍物（每个用 4 个参数表示
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b#30 个体素方块障碍物（每个用 6 个参数
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b#30 个圆柱体障碍物（3 参数
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b
        
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan#相机视场角 fov 会在一定范围内随机
        self.n_drones_per_group = random.choice([4, 8]) #每个小队的无人机数量随机 4 或 8
        self.drone_radius = random.uniform(0.1, 0.15)   #无人机半径在 [0.1, 0.15] 之间随机
        if self.single:     #如果是单机模式，就把每个小队的无人机数量设为 1
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)#给每组无人机采样一个随机速度系数
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp #无人机最大速度，受 speed_mtp 控制
        scale = (self.max_speed - 0.5).clamp_min(1) #速度越快，scale 越大

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01   # 给推力估计加 1% 左右的噪声（模拟实际传感器误差）
        
        roof = torch.rand((B,)) < 0.5   #一半概率 roof=True（有天花板/特殊布局）
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        #否则把一部分球和方块障碍物的位置替换成圆柱体的位置，然后整体平移
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        # 这些约束确保障碍物不会“卡出边界”，也不会重叠太紧，0.3/scale：留出安全间隔（速度越快，间隔越大）
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        # 如果是roof场景，就把某个voxel的高度拉高，形成天花板
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200


        #如果 ground_voxels=True，环境会额外加：
            # 两个大球体障碍物，半埋在地面中。
            # 一组地面方块体素。
            # 并且整个场景的障碍物在 y 方向会随速度拉伸。
        # 这样环境会有“地形起伏 + 平面方块 + 随速度调整”的复杂效果
        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6  #球的半径，范围 [8, 14]
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4   #地面与球交截的半径，范围 [2, 6]
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()#球心高度
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r   #前两个球的半径
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1  #球心高度，确保球体下缘在地面以下 1 米

            # planner shape in (0.1-2.0) times (0.1-2.0)
            #随机生成 10 个方块体素障碍物
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1#把 z 坐标设成 高度 - 1，即这些方块刚好落在地面上
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)#再拼接到原来的 voxels 里
        # y轴随速度缩放
        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            #调用 find_nearest_pt，找到离 gate 最近的障碍点
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)#按最后一维拆开
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50#如果最近点距离小于 0.5，说明 gate 位置和障碍物冲突 → 把 gate_x 设置成 -50，直接“踢出地图
            # 这里构造了 4 个长方体体素，作为 gate 的“边框”
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        # 缩放x轴坐标
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        # 在 ground_voxels 模式下 对前两个“地面球体”的 x 坐标 做边界约束，保证球不会超出环境边界或嵌入过深
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device) #俯仰控制延迟，均值 12，标准差 1.2
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device) #偏航控制延迟，均值 6，标准差 0.6
        # 每组无人机生成一个随机缩放因子
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        # 把之前的速度缩放（前面对障碍物的）和位置缩放组合
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1#初始位置 = 预定义位置 * 缩放 + 少量噪声
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1#目标位置 = 预定义目标位置 * 缩放 + 少量噪声
        # 旋转整个场景，让训练更鲁棒，避免过拟合特定方向
        if self.random_rotation:
            #每组无人机生成一个 yaw 偏置 [-0.75, 0.75] 弧度
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            # 构造 3x3 旋转矩阵 R，只绕 z 轴旋转
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            # 把旋转矩阵应用到：1.无人机起始位置 2.p目标位置 p_target 3.所有障碍物坐标 (voxels, balls, cyl)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # scaffold
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            # 每个网格点生成一个细长柱子 z=0.02,flatten 把网格展开为 (N, 3) 的列表
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            # x_bias → 整体 x 偏移
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),#y方向噪声
                torch.rand_like(self.max_speed) * 0.01#z方向扰动
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)#把纵向 scaffold 拼接到现有 cyl 障碍物
            _x, _z = torch.meshgrid(x, z)   #横向scaffold，生成x-z网络
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            #对横向 scaffold 同样添加缩放 + 随机偏移
            scaf_h = scaf_h * scale + torch.stack([ 
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            #拼接到 cyl_h（横向柱子列表）
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2   #无人机自身速度，随机扰动 ±0.2 m/s
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w    #风速，乘以每 batch 风速系数 self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1 
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2  #角速度向量

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1   #每个无人机的安全边距 [0.1, 0.3] m，用于碰撞检测

        # drag coef 阻力系数
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3#xy 平面阻力系数，随机 [0.3, 0.45]，第 0 列置 0
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)    #z 方向阻力系数，初始化为 1

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]    #无人机机头方向
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std   
        thrust = torch.norm(a_thr, 2, -1, True) #推力大小
        self_up_vec = a_thr / thrust    #无人机的 机体向上方向 (z 轴)，由推力方向决定
        forward_vec = self_forward_vec * yaw_inertia + v_pred   #给旧前向向量增加惯性，再加上预测速度 v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)  # 用 alpha 插值
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]#这一步是强制约束：保证前向向量与 up 向量正交
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)# 右手系，用叉积得到左向量(y轴)，保证三轴正交
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        # 创建空的画布，每个与元素是像素灰值
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None
    #计算无人机到障碍的向量
    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p
    #调用 CUDA 仿真一步，更新状态
    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        #ou噪声
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run( #这里是调用run = RunFunction.apply ，不是递归
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt) #指数衰减得到一个 alpha，相当于滤波系数
        self.R_old = self.R.clone() #保存旧的姿态，用来做光流计算
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
    #PyTorch 原生实现仿真（可微分）
    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)   #一阶惯性环节
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0 #垂直方向额外阻力
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()   #由推力大小估计的螺旋桨转速
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07  #与电机转速成正比的水平方向附加阻力
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)  #经典二次阻力模型：Fd ∝ v * |v|
        a_next = self.act + self.dg - z_drag - drag #总加速度 = 控制输入 + 扰动 - 下洗 drag - 空气阻力。

        self.p_old = self.p
        # 梯度衰减，防止梯度爆炸
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt#梯形积分
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)


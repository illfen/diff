from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
args = parser.parse_args()  #从命令行读取参数，如果没传就用默认值
writer = SummaryWriter()
print(args)

device = torch.device('cuda')

env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(7+3, 6)
model = model.to(device)
# 从已有的模型权重恢复训练
if args.resume: # args.resume 往往是一个 checkpoint 文件路径
    state_dict = torch.load(args.resume, map_location=device)# 从磁盘读取一个 PyTorch 保存的对象（一般是 checkpoint）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)# 把 checkpoint 的参数加载到模型里，False：允许不完全匹配，会返回两类信息：
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
# Adam 的改进版，带 权重衰减（Weight Decay），更稳定，常用于 Transformer 类模型
optim = AdamW(model.parameters(), args.lr)
# 用余弦退火（Cosine Annealing）的方式调整学习率。会把 lr 从 args.lr 逐步降到 args.lr * 0.01
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)    
#表示 控制时间步长（控制频率 = 15 Hz）
ctl_dt = 1 / 15


scaler_q = defaultdict(list) #创建一个key->list的字典容器。defaultdict(list)：当访问一个不存在的 key 时，自动创建一个空列表    
def smooth_dict(ori_dict):  #把 ori_dict 里的值存到 scaler_q 里，按 key 分类存储
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()
#训练前期更密集保存（防止崩掉/发散），后期稳定后减少存档频率
def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80)#生成一个带进度条的迭代器。ncols=80 控制进度条宽度
# depths = []
# states = []
B = args.batch_size
for i in pbar:  #外层循环：训练迭代次数（episode） args.num_iters，每次表示一条完整的轨迹 episode
    # 1、训练循环框架
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    h = None
    # 2、环境交互与状态构建
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)  #动作缓冲区，模拟控制延迟
    target_v_raw = env.p_target - env.p #初始目标速度 = 目标位置 - 当前无人机位置
    if args.yaw_drift:
        # torch.randn(B)：生成 B 个服从标准正态分布（均值0，方差1）的随机数
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)


    for t in range(args.timesteps): #内层循环：环境步长 args.timesteps，每次迭代表示无人机在仿真中的一步。
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)    # 控制周期带噪声
        depth, flow = env.render(ctl_dt)    # 渲染深度图、光流
        p_history.append(env.p) # 无人机位置
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())# 无人机到最近障碍物的向量

        if is_save_iter(i):
            vid.append(depth[4])

        if args.yaw_drift:  #是否加入偏航角漂移误差
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)#@：矩阵乘法（批量形式）
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)    #执行动作，并推进环境一步

        R = env.R
        fwd = env.R[:, :, 0].clone()    # 机体系前向向量 [B, 3]
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0   # 把每个前向向量的 z 分量置零，让前向方向 完全位于水平面上（x-y 平面）
        up[:, 2] = 1    # 把每个“上方向”的 z 分量设为1，让“up”朝世界坐标系的 +Z 方向
        fwd = F.normalize(fwd, 2, -1)
        # 这三个方向向量作为列堆叠成旋转矩阵
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)    # 正交基

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            # target_v[:, None]增加一个维度，变成[B, 1, 3]，R是[B, 3, 3]，矩阵乘法后变成[B, 1, 3]
            # 再用 torch.squeeze 去掉第1个维度，变回[B, 3]
            torch.squeeze(target_v[:, None] @ R, 1),    # 目标速度在机体系坐标下
            env.R[:, 2],                                # 无人机朝向的 z 轴（俯仰姿态）
            env.margin[:, None]]                        # 安全边界
        local_v = torch.squeeze(env.v[:, None] @ R, 1)  # 当前速度在机体系下
        if not args.no_odom:
            state.insert(0, local_v)    # 把 local_v 插入到列表开头（索引 0）
        state = torch.cat(state, -1)    # 拼接成最终状态向量

        # normalize,    depth 预处理：将深度图转为类似稀疏代价图的输入特征  clamp_(0.3, 24)：裁剪深度范围
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)  # [B, 1, H, W],降采样，保留局部极大值。
        act, values, h = model(x, state, h) # 前向传播
        # 把动作从机体系转到世界坐标系 在最后一维切开成多个 tensor
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        #把 act 映射到机体系坐标系，并加入推力估计误差、重力补偿
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)
    # 3、损失函数设计与优化
    # 地面约束（惩罚z坐标小于0
    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    # 速度跟踪
    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    # 计算滑动平均速度 (窗口=30)，减少抖动影响
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    # 当前速度和目标速度之间的差异（L2范数）
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    # 速度跟踪损失，鼓励实际速度和目标速度接近
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds = torch.stack(v_preds)
    # 预测速度和真实速度之间的MSE，作为辅助监督
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())
    # --- 速度方向约束 (偏差惩罚) ---
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    # 归一化目标速度向量 (只保留方向)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    # 投影分量：实际速度在目标速度方向上的大小
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    # 惩罚实际速度偏离目标方向的部分 (正交分量)，鼓励速度方向一致
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3
    # --- 平滑控制约束 ---
    # jerk = 动作一阶差分 (控制输入变化率)，乘15是归一化时间尺度
    jerk_history = act_buffer.diff(1, 0).mul(15)
    # snap = 控制输入二阶差分 (动作曲率)，反映更高阶平滑性
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    # sum(-1)在张量最后一个维度求和
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()
    # --- 避障相关 ---
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1) #与障碍物的欧式距离
    distance = distance - env.margin    #减去安全边距，得到实际净距离
    #估计逼近障碍物的速度，放缩到 >=1，避免梯度消失
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
        
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean() #模拟碰撞惩罚

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity + loss_ground_affinity

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    #梯度清零 → 反向传播 → 参数更新 → 学习率调度器更新
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()


    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        #保存日志指标
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})
        log_dict = {}
        if is_save_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            # writer.add_video('demo', vid, i + 1, 15)
            # TensorBoard 可视化
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
        if (i + 1) % 25 == 0:   #滑动日志记录器
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()

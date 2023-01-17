import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from config import opt
from model import GenerateorNet, DiscriminatorNet
import matplotlib.pyplot as plt
import numpy as np


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    opt.create_dir()

    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    # 数据预处理
    transform = tv.transforms.Compose([
        # [3, 96, 96]
        tv.transforms.Resize(opt.img_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据集
    dataset = tv.datasets.ImageFolder(root=opt.data_path, transform=transform)
    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    # 初始化神经网络
    netG, netD = GenerateorNet(opt), DiscriminatorNet(opt)

    # 如果存在模型路径则直接加载
    if opt.netg_path.exists():
        print('using exist GenerateorNet model')
        netG.load_state_dict(torch.load(f=opt.netg_path, map_location=device))
    if opt.netd_path.exists():
        print('using exist DiscriminatorNet model')
        netD.load_state_dict(torch.load(f=opt.netd_path, map_location=device))
    netG.to(device)
    netD.to(device)

    # 带动量的梯度下降算法
    optimize_g = torch.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.betas, 0.999))
    optimize_d = torch.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.betas, 0.999))
    # 计算交叉熵损失
    criterions = nn.BCELoss().to(device)
    # 定义标签
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    # 生成噪声
    noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)
    # 用作后期的噪声样本
    noises_tmp = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

    error_d_list = []
    error_d_loss = []
    error_g_list = []
    error_g_loss = []

    for epoch in range(opt.max_epoch):
        # 进度条
        with tqdm(total=len(dataLoader.dataset), desc='epoch{:3}'.format(epoch), leave=True, unit='img', unit_scale=True) as tq_bar:
            for batch_idx, (img, _) in (enumerate(dataLoader)):
                real_img = img.to(device)
                # 训练判别器
                if batch_idx % opt.d_every == 0:
                    optimize_d.zero_grad()
                    # 真实的图片
                    output = netD(real_img)
                    error_d_real = criterions(output, true_labels)
                    error_d_real.backward()
                    error_d_loss.append(error_d_real.item())
                    # 生成的图片
                    noises = noises.detach()
                    fake_image = netG(noises).detach()
                    output = netD(fake_image)
                    error_d_fake = criterions(output, fake_labels)
                    error_d_fake.backward()
                    error_d_loss.append(error_d_fake.item())
                    optimize_d.step()
                    
                # 训练生成器
                if batch_idx % opt.g_every == 0:
                    optimize_g.zero_grad()
                    noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1))
                    fake_image = netG(noises)
                    output = netD(fake_image)
                    error_g = criterions(output, true_labels)
                    error_g.backward()
                    error_g_loss.append(error_g.item())
                    optimize_g.step()
                    
                tq_bar.update(opt.batch_size)
            # 调试模式下保存各代训练模型及噪声图像
            if opt.debug and (epoch + 1) % opt.save_every == 0:
                torch.save(netD.state_dict(), opt.model_path.joinpath('model_D').joinpath('netd_{}.pth'.format(epoch)))
                torch.save(netG.state_dict(), opt.model_path.joinpath('model_G').joinpath('netg_{}.pth'.format(epoch)))
                fix_fake_image = netG(noises_tmp)
                tv.utils.save_image(fix_fake_image.data[:opt.gen_num], opt.noise_path.joinpath(str(epoch)).with_suffix('.png'), normalize=True)

        error_d_list.append(np.average(error_d_loss))
        error_g_list.append(np.average(error_g_loss))
        error_d_loss.clear()
        error_g_loss.clear()
    # 生成loss趋势图
    x_d = torch.arange(len(error_d_list))
    plt.title('Discriminate Loss')
    plt.plot(x_d, error_d_list, 'r', label='real')
    plt.legend()
    plt.savefig(opt.result_path.joinpath('loss_d').with_suffix('.png'))
    plt.close()
    x_g = torch.arange(len(error_g_list))
    plt.title('Generator Loss')
    plt.plot(x_g, error_g_list, 'g')
    plt.savefig(opt.result_path.joinpath('loss_g').with_suffix('.png'))
    plt.close()



    # 保存最终模型
    torch.save(netD.state_dict(), opt.netd_path)
    torch.save(netG.state_dict(), opt.netg_path)
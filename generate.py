import torch
import torchvision as tv
from config import opt
from model import GenerateorNet, DiscriminatorNet

@torch.no_grad()
def generate(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    if not opt.model_path.exists():
        raise FileExistsError('model dir not exist!')

    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    # 加载参数
    netG, netD = GenerateorNet(opt).eval(), DiscriminatorNet(opt).eval()
    netD.load_state_dict(torch.load(opt.netd_path, map_location=device), False)
    netG.load_state_dict(torch.load(opt.netg_path, map_location=device), False)
    netD.to(device)
    netG.to(device)

    # 生成图片
    noise = torch.randn(opt.gen_maxnum, opt.noise_dim, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)
    fake_image = netG(noise)

    # 选取高分图片
    score = netD(fake_image).detach()
    indexs = score.topk(opt.gen_num)[1]
    result = [fake_image.data[idx] for idx in indexs]

    # 保存图片
    tv.utils.save_image(torch.stack(result), opt.result_path.joinpath(opt.gen_img), normalize=True, range=(-1, 1))
    tv.utils.save_image(result[0], opt.result_path.joinpath('best_img.png'), normalize=True, range=(-1, 1))
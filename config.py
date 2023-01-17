from pathlib import Path


class Config:
    """定义配置类"""
    # 保存路径
    path = Path().cwd()
    data_path = path.joinpath('data')   # 数据集路径
    model_path = path.joinpath('model') # 模型保存路径
    noise_path = path.joinpath('noise') # 噪声样本保存路径
    result_path = path.joinpath('result')   # 生成器生成图片保存路径
    netd_path = model_path.joinpath('discriminator_model')    # 判别器参数保存路径
    netg_path = model_path.joinpath('generateor_model')    # 生成器参数保存路径

    # 训练参数
    img_size = 96
    batch_size = 256
    max_epoch = 200
    lr_g = 2e-4
    lr_d = 2e-4
    betas = 0.5
    gpu = True
    noise_dim = 100
    ngf = 64    # 生成器的卷积核个数
    ndf = 64    # 判别器的卷积核个数
    d_every = 1  # 每d_every个batch训练一次判别器
    g_every = 5  # 每g_every个batch训练一次生成器
    save_every = 10  # 每save_every次保存一次模型
    debug = True    # 是否保存每save_every次的模型及噪声图片

    # 生成模型参数
    gen_img = 'result.png'
    gen_num = 64
    gen_maxnum = 512
    gen_mean = 0    # 噪声均值
    gen_std = 1     # 噪声方差

    # 创建目录
    def create_dir(self):
        if not self.data_path.exists():
            raise FileExistsError('data dirctory not exist!')
        if not self.model_path.exists():
            self.model_path.mkdir()
        if not self.result_path.exists():
            self.result_path.mkdir()

        if self.debug:
            if not self.noise_path.exists():
                self.noise_path.mkdir()
            if not self.model_path.joinpath('model_D').exists():
                self.model_path.joinpath('model_D').mkdir()
            if not self.model_path.joinpath('model_G').exists():
                self.model_path.joinpath('model_G').mkdir()

opt = Config()

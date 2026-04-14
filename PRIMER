#环境初始化：管理-数据-变量-执行
from    ml_collections  import  ConfigDict  #type:ignore
import  numpy   as  np
import  random#type:ignore
import  torch#type:ignore

#定义函数
def default_samples_filter_fn(data,prec_threshold_ratio=0.02,skip_probability=0.08):
    #计算值
    prec_count=np.count_nonezero(data>-1)
    total_points=data.size
    prec_ratio=prec_count/total_points
    #数据筛选
    if  prec_ratio<prec_threshold_ratio:
        return  random.random()>=skip_probability
    return  True
#开启配置
def get_config():
    #准备好大的场地
    config=ConfigDict()

#执行总部run,一个分部，下属5个干将(name,experiment,wandb_dit,wandb_mode,unique_name)
    config.run=run=ConfigDict()
    run.name='PRIMER'
    run.experiment='finetuning  with gauges'
    run.wandb_dit=''
    run.wandb_mode='online'
    run.unique_name=None

#选数据
    config.data=data=ConfigDict()
    data.name='precipitation'
    data.root_dit='./gauges.npy'
    #在数据这个抽屉里进行数据安检
    data.expected_img_size=(1,250,250)
    #对输出层数进行强调
    data.channels=1
    # 数据预处理
    data.normalization='standard'
    #加载数据类型，进行数据统计
    data.property_path='./process_gauges_before/property_results.npy'
    #对6类数据进行读取，保存（max_val,min_val,mean,std,clip_min,clip_max)
    data.max_val=np.load(data.property_path,allow_pickle=True),item()['max_val']# noqa
    data.min_val=np.load(data.property_path,allow_pickle=True),item()['min_val']# noqa
    data.mean=np.load(data.property_path,allow_pickle=True),item()['mean']# noqa
    data.std=np.load(data.property_path,allow_pickle=True),item()['std']# noqa
    data.clip_min=np.load(data.property_path,allow_pickle=True),item()['clip_min']# noqa
    data.clip_max=np.load(data.property_path,allow_pickle=True),item()['clip_max']# noqa
    #预留一个自定义过滤器
    data.samples_filter_fn=None
    #限制一次循环读取的样本数量，小样本，提高单次运行速度
    data.fixed_length=100
    #数据搬运速度
    data.num_workers=5

#创建训练中心
    #新建一个抽屉放train
    config.train=train=ConfigDict()
    #设置游戏存档的起点
    train.load_checkpoint=False
    #存档读取的位置设定
    train.checkpoint_num=None
    #记录已经训练过的次数
    train.retrain_poach=None
    #自动混合精度
    train.amp=True
    #一次喂给模型的代码数设定
    train.batch_size=1
    #规定画图频率
    train.plot_graph_steps=5
    #用指数平均移动技术（ema）进行数据更新频率(update)，新旧记忆占比（decay）设置
    train.ema_update_every=10
    train.ema_decay=0.995
    #全局时间计步器，从这里继续开始（global_step）
    train.whether_global_step_zero=True
    #多源数据的抽样占比（proportion)
    train.sources_proportion=[0.2,0.4,0.4]

#模型结构设定
    config.model=model=ConfigDict()
    #模型在看降水图时的提取特征角度数为192
    model.nf=192
    #时间记忆维度（time_emb_dim）：模型把时间以长度为192*2的数轴进行记忆，方便排序
    model.time_emb_dim=model.nf*2
    #大脑厚度：模型会以‘提取特征-加工-传递’（这一部分叫卷积块num_conv_blocks）的结构重复运行10遍
    model.num_conv_blocks=10
    #寻找邻居（knn_neighbours）：当模型在读取数据时，会去参考距离它最近的3个数据站的数据-这里用了稀疏卷积
    model.knn_neighbours=3
    #深度稀疏化（depthwise_sparse):通过True这个开关，让模型从有降雨数据的地方开始读取
    model.depthwise_sparse=True
    #观察视野：通过设置卷积核（kernel_size)的大小，让模型每次观察时，眼镜扫过的范围是7*7的小格子
    model.kernel_size=7
    #底层引擎（backend）:请调用torch sparse这个专门处理稀疏数据的加速库
    model.backend='torchsparse'# noqa
    #uno_系列
    #设定输入图像的分辨率（res）：128*128
    model.uno_rest=(128,128)
    #设定基础通道值为200（base_channels）
    model.uno_base_channels=200
    #设定每一层的通道倍率（mults)由低到高分别为1倍，2倍，4倍，8倍# noqa
    model.uno_mults=(1,2,4,8)# noqa
    #通过每层设定2个残差块处理信息传递过程中的梯度消失问题（blocks_per_level）
    model.uno_blocks_per_level=(2,2,2,2)
    #注意力机制的引入时机(attn_resolutions)：当分辨率的宽度/4时，开始引入注意力机制（u型模型由左侧的局部到右侧的全局观察）
    model.uno_attn_resolutions=int(model.uno_rest[0]/4)
    #丢弃层（dropout_resolutions)的引入时机；与注意力机制引入的分辨率相同
    model.dropout_resolutions=int(model.uno_rest[0]/4)
    #丢弃概率的设定：10%
    model.dropout=0.1
    #对卷积层类型进行选择（conv_type）:这里为了整体的稳定性选了标准卷积层（conv）
    model.conv_type='conv'
    #网络最后输出端（z_dim）的通道深度为3
    model.z_dim=3
    #开启稳定输出模式，让模型丢弃不确定成果，输出最确定稳定的结果（sigma_small，要小方差）
    model.sigma_small=True

#接入扩散模型（diffusion）
    config.diffusion=diffusion=ConfigDict()
    #设定扩散的步调数1000
    diffusion.steps=1000
    #设定加噪音的节奏（nosie_schedule）:为余弦函数（cosine）
    diffusion.nosie_schedule='cosine'
    #模型抽样练习(schedule_sampler)：采取均匀分布（uniform）的方式在1000步中抽出试题让模型还原
    diffusion.schedule_sampler='uniform'
    #模型绘制技术评分：用损失函数（loss_type）给模型画出的图进行打分，L1（绝对值误差），MSE(平方差误差)
    diffusion.loss_type='MSE'
    #对图像进行预处理：用高斯滤波器（gaussian_filter_std）对原图像进行0.5级别的预处理
    diffusion.gaussian_filter_std=0.5
    #设定模型预测的目标（model_mean_type）：软化后的噪音（mollified_epsilon）
    diffusion.model_mean_type='mollified_epsilon'
    #软化噪音的具体实现方法：DCT
    diffusion.mollifier='dct'

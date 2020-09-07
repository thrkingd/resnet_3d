# resnet_3d
1.本工程是将resnet_3d 由pytorch实现转换为paddle实现

2.图像增广：训练时首先将图像短边缩放到120，然后随机裁剪112大小图片，再进行随机翻转和颜色扰动；测试时将图像短边缩放到120后进行中心裁剪112的图片。 注：还尝试了多尺度裁剪、10 augment（论文中的方法）等增广方法，效果均不如随机裁剪，具体参数配置见/configs/c3d_ucf101.yaml 

3.时域采样：训练时从视频中随机抽取连续的（step=1）16帧组成一个样本；验证时从视频中均匀抽取1个（seg_num=1）16帧的样本，将其预测输出作为视频预测分数；测试时将视频划分为n个不重叠的16帧视频段，将n段样本的输出平均后作为整段视频的预测输出 注：还尝试了间隔一帧采样（step=2），效果不如连续采样（大约低了0.2个百分点） 验证时取seg_num=1，其准确率≈clip的准确率，训练收敛时为91%。 

4.预训练参数：采用所要求的/pretrain/r3d50km_200ep.pdparams（由pytorch参数转换得来,转换代码见torch_para2paddle.py）文件 

5.训练细节：采用了两段式训练，（1）第一阶段没有使用颜色扰动进行训练：batchsize=256，learning_rate=0.01,共训练30epoch，15个epoch后学习率降低0.1； （2）第二阶段加入颜色扰动在第一阶段的训练结果上进行微调：batchsize=256，learning_rate=0.001,共训练20epoch,10个epoch后学习率降低0.1 5.大batchsize的实现：采用了两次反向运算一次梯度更新然后清零的方法在显存受限时训练更大的batchsize； 注：尝试过batchsize=64,128，发现性能均不如256,batchsize=64时 valid acc≈89.5%，batchsize=128时 valid acc≈90.4% 

6.结果说明：最终收敛后验证准确率会在0.91左右浮动，第一阶段的训练保存的中间结果为/work/checkpoints/resnet_3d_model_936.pdparams文件,其测试准确率为93.65%；第二阶段的模型也就是最终的结果模型保存为/checkpoints/resnet_3d_model_9376.pdparams文件,其测试准确率为93.76%，可以看出颜色扰动的增广方法能带来0.1个涨点。最终结果超过了论文公布的结果0.8个百分点。 

7.训练和测试过程：第一阶段参考/log/logger_0830.log文件，第二阶段参考/log/logger_0904.log文件

8.训练时需要将avi视频转换为jpg图像，见avi2jpg.py
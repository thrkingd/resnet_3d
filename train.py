import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from collections import OrderedDict
import datetime

from model.resnet_3d import ResNet_3d
from ucf101_reader import Ucf101Reader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='c3d',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/c3d_ucf101.yaml',
      #  default='/home/aistudio/work/configs/c3d_ucf101.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default='./pretrain/r3d50_km_200ep',
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=ast.literal_eval,
        default=False,
        help=''
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./checkpoints',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    curdir=os.getcwd()
    os.chdir(os.path.join(curdir,'work'))
    print(datetime.datetime.now())
    logger.info(datetime.datetime.now())
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        valid_config = merge_configs(config, 'valid', vars(args))
        print_configs(train_config, 'Train')

        #根据自己定义的网络，声明train_model
        train_model=ResNet_3d()
        train_model.train()
     #设置优化器和学习率
        batch_size = train_config.TRAIN.batch_size
        lr = train_config.TRAIN.learning_rate
        lr_decay = train_config.TRAIN.learning_rate_decay
        step = int(train_config.TRAIN.total_videos/batch_size+1)
        epochs =  train_config.TRAIN.epoch
        bd = [step*epochs/2]
        lr = [lr,lr*lr_decay]
        lr = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        opt = fluid.optimizer.Momentum(lr, 0.9, parameter_list=train_model.parameters(),
        regularization=fluid.regularizer.L2Decay(config.TRAIN.l2_weight_decay))
        
        #加载预训练参数
        if args.resume==True:
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/resnet_3d_model')
            train_model.load_dict(model)
            print('Resueme from '+ args.save_dir + '/resnet_3d_model')
        elif args.pretrain:
            pretrain_weights = fluid.io.load_program_state(args.pretrain)
            inner_state_dict = train_model.state_dict()
            print('Pretrain with '+ args.pretrain)
            for name, para in inner_state_dict.items():
                if((name in pretrain_weights) and (not('fc' in para.name))):
                    para.set_value(pretrain_weights[name])
                else:
                    print('del '+ para.name)
        #用2D参数初始化
        # elif args.pretrain:
        #     state_dict = fluid.load_program_state(args.pretrain)
        #     dict_keys = list(state_dict.keys())
        #     inner_state_dict = train_model.state_dict()
        #     for name in dict_keys:
        #         if "fc" in name:
        #             del state_dict[name]
        #             print('Delete {} from pretrained parameters. Do not load it'.format(name))
        #         if 'weights' in name:
        #             tmp = state_dict[name]
        #             tmp_shape = tmp.shape
        #             if (tmp_shape[-1] > 1) and (len(tmp_shape)) == 4:
        #                 tmp = tmp.reshape([tmp_shape[0], tmp_shape[1], 1, tmp_shape[2], tmp_shape[3]])
        #                 tmp = tmp.repeat(tmp_shape[3], axis=2)
        #                 state_dict[name] = tmp / tmp_shape[3]
        #             if (tmp_shape[-1] == 1):
        #                 state_dict[name] = tmp.reshape([tmp_shape[0], tmp_shape[1], 1, tmp_shape[2], tmp_shape[3]])

        #     dict_keys = state_dict.keys()
        #     for name, para in inner_state_dict.items():
        #         key_name =  para.name
        #         if key_name in dict_keys:
        #             para.set_value(state_dict[key_name])
        else:
            pass;
    


        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = Ucf101Reader(args.model_name.upper(), 'train', train_config).create_reader()
        valid_reader = Ucf101Reader(args.model_name.upper(), 'valid', valid_config).create_reader()
       
        
        for i in range(epochs):
            train_model.train()
            lr=opt.current_step_lr()
            logger.info('Epoch{} lr={}'.format(i,lr))
            print('Epoch{} lr={}'.format(i,lr))
         
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                batchsize = dy_x_data.shape[0]
                
                img = fluid.dygraph.to_variable(dy_x_data[0:batchsize//2,:,:,:,:])
                label = fluid.dygraph.to_variable(y_data[0:batchsize//2,:])
                label.stop_gradient = True
                out = train_model(img)
                acc1 = fluid.layers.accuracy(out,label)
                loss1 = fluid.layers.cross_entropy(out, label)
                avg_loss1 = fluid.layers.mean(loss1)/2
                avg_loss1.backward()

                img = fluid.dygraph.to_variable(dy_x_data[batchsize//2:,:,:,:,:])
                label = fluid.dygraph.to_variable(y_data[batchsize//2:,:])
                label.stop_gradient = True
                out = train_model(img)
                acc2 = fluid.layers.accuracy(out,label)
                loss2 = fluid.layers.cross_entropy(out, label)
                avg_loss2 = fluid.layers.mean(loss2)/2
                avg_loss2.backward()

                opt.minimize(avg_loss1+avg_loss2)
                train_model.clear_gradients()

                acc = (acc1+acc2)/2
                avg_loss = avg_loss1+avg_loss2
                
                
                if batch_id % 10 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/resnet_3d_model')
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/resnet_3d_model_epoch{}'.format(i))
            # if((i%3)==0 and i!=0):
            if(i%1==0):
                acc_list = []
                avg_loss_list = []
                train_model.eval()
                for batch_id, data in enumerate(valid_reader()):
                    dy_x_data = np.array([x[0] for x in data]).astype('float32')
                    y_data = np.array([[x[1]] for x in data]).astype('int64')

                    img = fluid.dygraph.to_variable(dy_x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    label.stop_gradient = True
                    out = train_model(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    acc_list.append(acc.numpy()[0])
                    avg_loss_list.append(avg_loss.numpy())
                    if batch_id %20 == 0:
                        logger.info("valid Loss at step {}: {}, acc: {}".format(batch_id, avg_loss.numpy(), acc.numpy()))
                        print("valid Loss at  step {}: {}, acc: {}".format(batch_id, avg_loss.numpy(), acc.numpy()))
                print("验证集准确率为:{}".format(np.mean(acc_list)))
                logger.info("验证集准确率为:{}".format(np.mean(acc_list)))
                print("验证集loss为:{}".format(np.mean(avg_loss_list)))
                logger.info("验证集loss为:{}".format(np.mean(avg_loss_list)))

        #
        # logger.info("Final loss: {}".format(avg_loss.numpy()))
        # print("Final loss: {}".format(avg_loss.numpy()))

def log_lr_and_step():
    try:
        # In optimizers, if learning_rate is set as constant, lr_var
        # name is 'learning_rate_0', and iteration counter is not 
        # recorded. If learning_rate is set as decayed values from 
        # learning_rate_scheduler, lr_var name is 'learning_rate', 
        # and iteration counter is recorded with name '@LR_DECAY_COUNTER@', 
        # better impliment is required here
        lr_var = fluid.global_scope().find_var("learning_rate")
        if not lr_var:
            lr_var = fluid.global_scope().find_var("learning_rate_0")
        lr = np.array(lr_var.get_tensor())

        lr_count = '[-]'
        lr_count_var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@")
        if lr_count_var:
            lr_count = np.array(lr_count_var.get_tensor())
        logger.info("------- learning rate {}, learning rate counter {} -----"
                    .format(np.array(lr), np.array(lr_count)))
        print("------- learning rate {}, learning rate counter {} -----"
                    .format(np.array(lr), np.array(lr_count)))
    except:
        logger.warn("Unable to get learning_rate and LR_DECAY_COUNTER.")  
        print("Unable to get learning_rate and LR_DECAY_COUNTER.")             


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
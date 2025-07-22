import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
from medpy.metric import dc,hd95

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
            
# def inference(args, model, best_performance):
#     db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)

#     #————————————————————win上关闭多进程
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     #对于每一个测试样本呢
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         #这个dice返回的是当前batch下，不同分类下的dice值的列表，列表长度为args.num_classes
#         metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#     metric_list = metric_list / len(db_test)
#     performance = np.mean(metric_list, axis=0)
#     logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
#     return performance


def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)

    #————————————————————win上关闭多进程
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = np.zeros((8,2),dtype=float)
    #对于每一个测试样本呢
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        #这个dice返回的是当前batch下，不同分类下的dice值的列表，列表长度为args.num_classes
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i,dtype=float)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info(
            'Testing performance in val model: '
            'mean_dice : %f, '
            #'hd95 : %f, '
            'iou : %f, '
            'best_dice : %f' 
            % (performance[0], performance[1], best_performance)
        )
    return performance[0],performance[1]

def inference_ACDC(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd
        


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    #小样本200个训练
    
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
    #                        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # # 获取数据集长度
    # total_data = len(db_train)
    # print(f"The length of original train set is: {total_data}")

    # # 随机选择200个样本
    # random_indices = random.sample(range(total_data), 200)

    # # 更新数据集，仅使用随机选择的200个样本
    # db_train.sample_list = [db_train.sample_list[i] for i in random_indices]

    # print(f"The length of randomly sampled train set is: {len(db_train.sample_list)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    #此处也关闭
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    #                      worker_init_fn=worker_init_fn)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    #相比
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            
            #p1的shape torch.Size([4, 9, 224, 224]) P是一个list包含四个p1一样的tensor
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(P)
                # 输出的分支索引列表
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                # ss：根据监督策略生成的组合方式列表，决定后续如何计算损失  分别是，融合监督，分别监督，只最终监督
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                # iout 的形状为 bchw[4, 9, 224, 224] 只不过每个样本有c个类别的预测值 标签的形状为 bhw [4, 224, 224] 
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        
        performance,iou = inference(args, model, best_performance)
        writer.add_scalar('val/Dice', performance, epoch_num)
        writer.add_scalar('val/Dice', iou, epoch_num)
        
        save_interval = 50

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_ACDC(args, net, snapshot_path):
    train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(train_dataset)))
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
    valloader=DataLoader(db_val, batch_size=1, shuffle=False)
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    iterator = tqdm(range(0, args.max_epochs), ncols=70)
    iter_num = 0

    Loss = []
    Test_Accuracy = []

    Best_dcs = 0.8

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    max_iterations = args.max_epochs * len(Train_loader)
    optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
    #optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    def val():
        logging.info("Validation ===>")
        dc_sum=0
        metric_list = 0.0
        net.eval()
        for i, val_sampled_batch in enumerate(valloader):
            val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
            val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
            val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
            p1, p2, p3, p4 = net(val_image_batch)
            val_outputs = p1 + p2 + p3 + p4
            val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

            dc_sum+=dc(val_outputs.cpu().data.numpy(),val_label_batch[:].cpu().data.numpy())
        performance = dc_sum / len(valloader)
        logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

        print("val avg_dsc: %f" % (performance))
        return performance 


    for epoch in iterator:
        net.train()
        train_loss = 0
        for i_batch, sampled_batch in enumerate(Train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            p1, p2, p3, p4 = net(image_batch) # forward

            outputs = p1 + p2 + p3 + p4 # additive output aggregation

            loss_ce1 = ce_loss(p1, label_batch[:].long())
            loss_ce2 = ce_loss(p2, label_batch[:].long())
            loss_ce3 = ce_loss(p3, label_batch[:].long())
            loss_ce4 = ce_loss(p4, label_batch[:].long())
            loss_dice1 = dice_loss(p1, label_batch, softmax=True)
            loss_dice2 = dice_loss(p2, label_batch, softmax=True)
            loss_dice3 = dice_loss(p3, label_batch, softmax=True)
            loss_dice4 = dice_loss(p4, label_batch, softmax=True)


            loss_p1 = 0.3 * loss_ce1 + 0.7 * loss_dice1
            loss_p2 = 0.3 * loss_ce2 + 0.7 * loss_dice2
            loss_p3 = 0.3 * loss_ce3 + 0.7 * loss_dice3
            loss_p4 = 0.3 * loss_ce4 + 0.7 * loss_dice4

            alpha, beta, gamma, zeta = 1., 1., 1., 1.
            loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4 # current setting is for additive aggregation.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if iter_num%20 == 0:
                logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            train_loss += loss.item()
        Loss.append(train_loss/len(train_dataset))
        logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))


        avg_dcs = val()

        if avg_dcs > Best_dcs:
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))

            Best_dcs = avg_dcs
    
            avg_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
            print("test avg_dsc: %f" % (avg_dcs))
            Test_Accuracy.append(avg_dcs)


        if epoch >= args.max_epochs - 1:
            save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            iterator.close()
            break


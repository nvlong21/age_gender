import os
import re
import cv2
import time
import copy
import math
import glob
import datetime
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from shuffle import ShuffleNetV2
from dataload import FaceDataset

from collections import OrderedDict

def image_transformer():
  """
  :return:  A transformer to convert a PIL image to a tensor image
            ready to feed into a neural network
  """
  return {
      'train': transforms.Compose([
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }
def load_data(batch_size, num_worker = 0):
    """
    initiate dataloader processes
    :return: 
    """
    transforms = image_transformer()
    print("[AgePredModel] load_data: start loading...")
    image_datasets = {x: FaceDataset(transforms[x])
                                        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                                     batch_size=batch_size,
                                                                     shuffle=True,
                                                                     num_workers=num_worker, drop_last = True)
                                            for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[AgePredModel] load_data: Done! Get {} for train and {} for test!"
                .format(dataset_sizes['train'],
                                dataset_sizes['val']))
    print("[AgePredModel] load_data: loading finished !")
    return dataloaders, dataset_sizes
def train_model(model, num_epochs = 300):
    print("[AgePredModel] train_model: Start training...")
    age_cls_unit = 101

    # 1.0.0.0 define Vars
    best_gen_acc = 0.
    best_age_acc = 0.
    best_age_mae = 99.
    not_reduce_rounds = 0
    lr = 0.01
    age_divide  = 10

    use_gpu = torch.cuda.is_available() 

    # 2.0.0.0 init optimizer
    
    age_cls_criterion = nn.BCELoss()
    gender_criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model = model.cuda()
        age_cls_criterion= age_cls_criterion.cuda()
        gender_criterion = gender_criterion.cuda()


    all_params = sum([np.prod(p.size()) for p in model.parameters()])
    trainable_params = sum([np.prod(p.size()) for p in
                                                    filter(lambda p: p.requires_grad, model.parameters())])
    print("[AgePredModel] Model has {}k out of {}k trainable params "
                .format(trainable_params // 1000, all_params // 1000))

    # use when having multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    reduce_gen_loss     = 0.01
    reduce_age_mae      = 0.1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                            lr=lr,
                                                            )
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [5, 60, 90, 140], gamma=0.1)
    # 4.0.0.0 start each epoch
    layer_to_freeze = 0
    dataloaders, dataset_sizes = load_data(64, 8)
    for epoch in range(num_epochs):
        print('\nStart Epoch {}/{} ...'.format(epoch + 1, num_epochs))
        print('-' * 16)
        exp_lr_scheduler.step()

        for phase in ['train', 'val']:
            # 4.1.1.0 shift train/eval model
            model.train(phase == 'train')
            torch.cuda.empty_cache()

            epoch_age_tp = 0.
            epoch_age_mae = 0.
            epoch_gender_tp = 0.
            processed_data = 0

            # 4.1.2.0 iterate over each batch.
            epoch_start_time = time.time()
            for idx, data in enumerate(dataloaders[phase]):
                # 4.1.2.1 get the inputs and labels
                inputs, gender_true, age_rgs_true, age_cls_true = data
                processed_data += inputs.size(0)

                # 4.1.2.2 wrap inputs&oputpus into Variable
                #         NOTE: set voloatile = True when
                #         doing evaluation helps reduce
                #         gpu mem usage.
                volatile = phase == 'val'
                if use_gpu:
                    inputs = Variable(inputs.cuda(), volatile=volatile)
                    gender_true = Variable(gender_true.cuda(), volatile=volatile)
                    # age_rgs_true  = Variable(age_rgs_true.cuda(), volatile=volatile)
                    age_cls_true = Variable(age_cls_true.cuda(), volatile=volatile)
                else:
                    inputs = Variable(inputs, volatile=volatile)
                    gender_true = Variable(gender_true, volatile=volatile)
                    # age_rgs_true  = Variable(age_rgs_true, volatile=volatile)
                    age_cls_true = Variable(age_cls_true, volatile=volatile)

                # 4.1.2.3 zero gradients
                optimizer.zero_grad()

                # 4.1.2.4 forward and get outputs
                gender_out, age_out = model(inputs)
                _, gender_pred = torch.max(gender_out, 1)
                _, max_cls_pred_age = torch.max(age_out, 1)
                gender_true = gender_true.view(-1)
                age_cls_true = age_cls_true.view(-1, age_cls_unit)

                # 4.1.2.5 get the loss
                # target = target.squeeze(1)
                gender_loss = gender_criterion(gender_out, gender_true)
                age_cls_loss = age_cls_criterion(age_out, age_cls_true)
                # print(gender_loss, age_cls_loss)
                # age_rgs_loss  = self.age_rgs_criterion(age_out, age_rgs_true)

                # *Note: reduce some age loss and gender loss
                #         enforce the model to focuse on reducing
                #         age classification loss
                gender_loss *= reduce_gen_loss

                loss = age_cls_loss
                loss = gender_loss + age_cls_loss

                gender_loss_perc = 100 * (gender_loss / loss).cpu().data.numpy()
                age_cls_loss_perc = 100 * (age_cls_loss / loss).cpu().data.numpy()
                # age_rgs_loss_perc = 100 * (age_rgs_loss / loss).cpu().data.numpy()[0]

                age_rgs_loss_perc = 0
                # age_cls_loss_perc = 0
                # gender_loss_perc = 0

                # convert cls result to rgs result by weigted sum
                weigh = np.linspace(1, age_cls_unit, age_cls_unit)
                age_cls_raw = age_out.cpu().data.numpy()
                age_cls_raw = np.sum(age_cls_raw * weigh, axis=1)
                age_rgs_true = age_rgs_true.view(-1)
                age_rgs_true = age_rgs_true.cpu().numpy() * age_divide
                age_rgs_loss = np.mean(np.abs(age_cls_raw - age_rgs_true))

                # 4.1.2.6 backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 4.1.2.7 statistics
                gender_pred = gender_pred.cpu().data.numpy()
                gender_true = gender_true.cpu().data.numpy()
                batch_gender_tp = np.sum(gender_pred == gender_true)

                max_cls_pred_age = max_cls_pred_age.cpu().data.numpy()
                age_cls_true = age_rgs_true
                batch_age_tp = np.sum(np.abs(age_cls_true - max_cls_pred_age) <= 2)  # if true, MAE < 5

                epoch_age_mae += age_rgs_loss * inputs.size(0)
                epoch_age_tp += batch_age_tp
                epoch_gender_tp += batch_gender_tp

                # 4.1.2.8 print info for each bach done
                if idx%100 ==0:
                    print("|| LOSS = {:.2f} || DISTR% {:.0f} : {:.0f} : {:.0f} "
                            "|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}% "
                            "|| LR {} ||  BEST {:.2f} / {:.2f}% / {:.2f}% ||"
                            .format(loss.cpu().data.numpy(),
                                            age_rgs_loss_perc,
                                            age_cls_loss_perc,
                                            gender_loss_perc,
                                            age_rgs_loss,
                                            100 * batch_age_tp / inputs.size(0),
                                            100 * batch_gender_tp / inputs.size(0),
                                            lr,
                                            best_age_mae,
                                            100 * best_age_acc,
                                            100 * best_gen_acc),
                            end='\r')

                # 4.1.2.9 unlink cuda variables and free up mem
                del inputs, gender_true, age_rgs_true, age_cls_true
                del age_rgs_loss, loss  # , gen_loss, age_cls_loss
                del gender_loss_perc, age_cls_loss_perc, age_rgs_loss_perc

            # 4.1.3.0 epoch done
            epoch_gender_acc = epoch_gender_tp / dataset_sizes[phase]
            epoch_age_acc = epoch_age_tp / dataset_sizes[phase]
            epoch_age_mae /= dataset_sizes[phase]

            # 4.1.4.0 print info after each epoch done
            print('\n--{} {}/{} Done! '
                        '|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}%  '
                        '|| COST {:.0f}s'
                        .format(phase.upper(),
                                        epoch,
                                        num_epochs,
                                        epoch_age_mae,
                                        100 * epoch_age_acc,
                                        100 * epoch_gender_acc,
                                        time.time() - epoch_start_time))

            # 4.1.5.0, save model weights
            if phase == 'val' and epoch_age_mae < best_age_mae:
                best_gen_acc = epoch_gender_acc
                best_age_acc = epoch_age_acc
                best_age_mae = epoch_age_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch,
                                        'state_dic': best_model_wts,
                                        "best_gen_acc": best_gen_acc,
                                        "best_age_acc": best_age_acc,
                                        "best_age_mae": best_age_mae,
                                        "lr_rate": lr,
                                        "optimizer": optimizer.state_dict()
                                        }, "checkpoint_best.pt")
                not_reduce_rounds = 0
                print("--New BEST FOUND!! || "
                            " AMAE/AACC/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}%"
                            .format(best_age_mae,
                                            100 * best_age_acc,
                                            100 * best_gen_acc))
            elif phase == 'val':
                not_reduce_rounds += 1
                torch.save({'epoch': epoch,
                                        'state_dic': model.state_dict(),
                                        "best_gen_acc": best_gen_acc,
                                        "best_age_acc": best_age_acc,
                                        "best_age_mae": best_age_mae,
                                        "lr_rate": lr,
                                        "optimizer": optimizer.state_dict()
                                        }, "checkpoint_last.pt")


            # 4.1.7.0 reduce learning rate if nessessary
            # if phase == "val" and lr > 0.0001:
            #                 # and not_reduce_rounds >= self.max_no_reduce 
                            
            #     lr = max(0.0001, lr / 10)
            #     print("[reduce_lr_rate] Reduce Learning Rate From {} --> {}"
            #                 .format(lr * 10, lr))
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     not_reduce_rounds = 0

    return model


def getAgeGender(img, transformed=False,
                 return_all_faces=True,
                 return_info=False):
    """
    evaluation/test funtion
    :param img: str or numpy array represent the image
    :param transformed: if the image is transformed into standarlized pytorch image.
                    applicable when using this in train loop
    :param return_all_faces: if set, return prediction results of all faces detected.
                    set to False if it's known that all images comtain only 1 face
    :param return_info: if set, return a list of rects (x, y, w, h) represents loc of faces
    :return: a list of [gender_pred, age_pred]
    """
    # load model params
    if not self.weight_loaded:
        path = self.checkpoint_best if self.load_best else self.checkpoint_last
        checkpoint = torch.load(path, map_location='gpu' if self.use_gpu else 'cpu')
        self.soft_load_statedic(checkpoint['state_dic'])
        # self.model.load_state_dict(checkpoint['state_dic'])
        self.model.train(False)
        self.weight_loaded = True

    # load images if not provided
    if type(img) == str:
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    # get faces and rects
    aligned = self.aligner.getAligns(img, return_info=return_info)
    if return_info:
        aligned, rects, scores = aligned
    if not len(aligned):  # no face detected
        scores = [1]
        rects = [(0, 0, img.shape[0], img.shape[1])]
        faces = [img]
    else:
        faces = aligned
    if not return_all_faces:
        faces = faces[0]
    faces = [transforms.ToPILImage()(fc) for fc in faces]
    if not transformed:
        faces = [self.transformer['val'](fc) for fc in faces]

    # get predictions of each face
    preds = self.model.evaluate(faces)

    if return_info:
        return preds, rects, scores
    return preds


if __name__ == "__main__":
    net = ShuffleNetV2()
    train_model(net)






import argparse
import os
import warnings

import cv2
import numpy as np


class Evaluate(object):
    def __init__(self, dir_GT):
        self.dir_GT = dir_GT
        self.image_list = os.listdir(dir_GT)
        self.image_list.sort()
        self.num = len(self.image_list)

    def ROC(self, dir_sal):
        TPR_all = np.zeros((self.num, 256))
        FPR_all = np.zeros((self.num, 256))

        for i, GT_name in enumerate(self.image_list):
            img_GT = self.__read_GT(self.dir_GT, GT_name)
            sal_name = os.path.splitext(GT_name)[0] + ".jpg"
            img_sal = self.__read_image(dir_sal, sal_name)

            tpr, fpr = self.__cal_ROC(img_GT, img_sal)
            TPR_all[i, :] = tpr
            FPR_all[i, :] = fpr

        TPR = np.mean(TPR_all, axis=0)
        FPR = np.mean(FPR_all, axis=0)
        AUC = self.__cal_AUC(TPR, FPR)

        label = dir_sal.split('/')[-2]
        print("AUC: %.3f" % (AUC))
        with open(txt_path, 'a') as f:
            f.write("AUC: %.3f\n" % (AUC))

    def CC(self, dir_sal):
        CC_all = np.zeros((self.num))

        for i, GT_name in enumerate(self.image_list):
            img_GT = self.__read_GT(self.dir_GT, GT_name)
            sal_name = os.path.splitext(GT_name)[0] + ".jpg"
            img_sal = self.__read_image(dir_sal, sal_name)

            cc = self.__cal_CC(img_GT, img_sal)
            CC_all[i] = cc

        CC = np.mean(CC_all)

        label = dir_sal.split('/')[-2]
        print("CC: %.3f" % (CC))
        with open(txt_path, 'a') as f:
            f.write("CC: %.3f\n\n" % (CC))

    def __read_GT(self, dir_GT, GT_name):
        img = cv2.imread(os.path.join(dir_GT, GT_name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 128
        return img

    def __read_image(self, dir_sal, sal_name):
        img = cv2.imread(os.path.join(dir_sal, sal_name), cv2.IMREAD_GRAYSCALE)
        return img

    def __cal_ROC(self, img_GT, img_sal):
        target = img_sal[img_GT]
        nontarget = img_sal[(1 - img_GT).astype(np.bool_)]

        tp = np.zeros((256))
        fp = np.zeros((256))
        fn = np.zeros((256))
        tn = np.zeros((256))
        for i in range(256):
            tp[i] = np.sum(target >= i)
            fp[i] = np.sum(nontarget >= i)
            fn[i] = np.sum(target < i)
            tn[i] = np.sum(nontarget < i)
        tp = np.flipud(tp)
        fp = np.flipud(fp)
        fn = np.flipud(fn)
        tn = np.flipud(tn)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return tpr, fpr

    def __cal_AUC(self, TPR, FPR):
        AUC = 0
        for i in range(255):
            AUC += (TPR[i] + TPR[i + 1]) * (FPR[i + 1] - FPR[i]) / 2
        return AUC

    def __cal_CC(self, img_GT, img_sal):
        map1 = img_GT.astype(np.float32)
        map1 = map1 - np.mean(map1)
        map2 = img_sal.astype(np.float32) / 255
        map2 = map2 - np.mean(map2)

        cov = np.sum(map1 * map2)
        d1 = np.sum(map1 * map1)
        d2 = np.sum(map2 * map2)
        cc = cov / (np.sqrt(d1) * np.sqrt(d2) + 1e-3)
        return cc


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--sm_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    args = parser.parse_args()
    txt_path = "result.txt"

    eva = Evaluate(args.gt_dir)
    eva.ROC(args.sm_dir)
    eva.CC(args.sm_dir)

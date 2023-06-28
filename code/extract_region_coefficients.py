import argparse
from loguru import logger
import pathlib
import random
import time
import numpy as np
import pandas as pd
import pingouin as pg
import cv2
import torch
import torchmetrics
import datasets

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

from zennit.attribution import Gradient, SmoothGrad
from zennit.core import Stabilizer
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
from zennit.image import imgify, imsave
from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
from zennit.types import BatchNorm, MaxPool
from zennit.torchvision import VGGCanonizer, ResNetCanonizer

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients
from zennit.composites import GuidedBackprop, ExcitationBackprop, DeconvNet, EpsilonPlus, EpsilonPlusFlat, EpsilonAlpha2Beta1, EpsilonAlpha2Beta1Flat, EpsilonGammaBox
from zennit.image import imgify, imsave
from captum.attr import Occlusion
from captum.attr import DeepLift, GuidedGradCam
from captum.attr import LayerDeepLift, LayerGradCam, LayerAttribution, LayerGradientShap
from pytorch_grad_cam.utils.image import show_cam_on_image # TODO: later remove dependency

from skimage.measure import label
import warnings
warnings.filterwarnings("ignore")


def normalize_relevance(attribution):
    # absolute sum over the channels and min-max [0, 1]
    relevance = attribution.abs().sum(1).detach().cpu()
    #relevance = ((relevance - relevance.min()) / (relevance.max() - relevance.min())).cpu().numpy().squeeze()
    return relevance.cpu().numpy().squeeze() # relevance

# https://zennit.readthedocs.io/en/latest/reference/zennit.torchvision.html#zennit.torchvision.ResNetBasicBlockCanonizer
#from zennit.torchvision import VGGCanonizer, ResNetCanonizer
canonizer = ResNetCanonizer()

def get_relevance(model, input, pred_label_idx, method, device, num_classes=12):

    if method=='Gradient':
        # Gradient
        attributor = Gradient(model)
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='SmoothGrad':
        # SmoothGrad
        attributor = SmoothGrad(model, noise_level=0.1, n_iter=20)
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='IntegratedGradients':
        # IntegratedGradients
        attributor = IntegratedGradients(model, n_iter=20)
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='GuidedBackprop':
        # GuidedBackprop
        attributor = Gradient(model=model, composite=GuidedBackprop())
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='ExcitationBackprop':
        # ExcitationBackprop
        attributor = Gradient(model=model, composite=ExcitationBackprop())
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='DeconvNet':
        # DeconvNet
        attributor = Gradient(model=model, composite=DeconvNet())
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='LRP-EpsilonPlus':
        # LRP-EpsilonPlus
        attributor = Gradient(model=model, composite=EpsilonPlus(canonizers=[canonizer]))
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='LRP-EpsilonPlusFlat':
        # LRP-EpsilonPlusFlat
        attributor = Gradient(model=model, composite=EpsilonPlusFlat(canonizers=[canonizer]))
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='LRP-EpsilonAlpha2Beta1':
        # LRP-EpsilonAlpha2Beta1
        attributor = Gradient(model=model, composite=EpsilonAlpha2Beta1(canonizers=[canonizer]))
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='LRP-EpsilonAlpha2Beta1Flat':
        # LRP-EpsilonAlpha2Beta1Flat
        attributor = Gradient(model=model, composite=EpsilonAlpha2Beta1Flat(canonizers=[canonizer]))
        target = torch.eye(num_classes).to(device)[[pred_label_idx]]
        output, attribution = attributor(input, target)
    elif method=='DeepLIFT':
        # DeepLIFT
        attributor = DeepLift(model)
        attribution = attributor.attribute(input, target=int(pred_label_idx))
    elif method=='GuidedGradCam':
        # GuidedGradCam
        attributor = GuidedGradCam(model, model.layer4)
        attribution = attributor.attribute(input, target=int(pred_label_idx))
    elif method=='LayerDeepLIFT':
        # GuidedGradCam
        attributor = LayerDeepLift(model, model.layer4)
        attribution = attributor.attribute(input, target=int(pred_label_idx))
        attribution = LayerAttribution.interpolate(attribution, (224, 224), interpolate_mode='bicubic')
    elif method=='LayerGradCam':
        # LayerGradCam
        attributor = LayerGradCam(model, model.layer4)
        attribution = attributor.attribute(input, target=int(pred_label_idx))
        attribution = LayerAttribution.interpolate(attribution, (224, 224), interpolate_mode='bicubic')
    elif method=='Occlusion':
        # Occlusion
        attributor = Occlusion(model)
        attribution = attributor.attribute(input,
                                           strides = (3, 8, 8),
                                           target=int(pred_label_idx),
                                           sliding_window_shapes=(3,15, 15),
                                           baselines=0)
    else:
        raise ValueError('XAI saliency map not implemented!')

    return normalize_relevance(attribution)


def evaluate_xai_maps(dataset_folder, fold, xai_method, device):

    project_root = './'
    mean_bgr = {'fold-1':[112.482, 123.050, 147.127],
                'fold-2':[112.475, 123.011, 147.073],
                'fold-3':[112.359, 122.850, 147.066],
                'fold-4':[112.912, 123.480, 147.665],
                'fold-5':[112.554, 123.063, 147.243]}
    num_classes = 12
    categories = ['22q11DS', 'Angelman', 'BWS', 'CdLS', 'Down', 'KS', 'NS', 'PWS', 'RSTS1', 'Unaffected', 'WHS', 'WS']

    # Test dataset
    image_size = 224 # we use only VGG-Face2 pretrained ResNet50
    test_dataset = datasets.NIHFacesDataset(root_dir=dataset_folder,
                                            metadata_file='./metadata/partitions.csv',
                                            fold=fold,
                                            split='val',
                                            mean_bgr=mean_bgr[fold],
                                            image_size=image_size)
    num_samples = len(test_dataset)

    # Load the models
    #model_path = list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-25-*.pt'))[-1]
    val_acc = np.array([float(s.as_posix().split('-test_accuracy-')[-1].replace('.pt','')) for s in list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-*.pt'))])
    model_path = list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-*.pt'))[np.argmax(val_acc)].as_posix()
    logger.info('model_path: %s'%model_path)
    model = torch.load(model_path)
    model.to(device)
    eval_mode = model.eval()

    results = pd.DataFrame(columns=['backbone', 'fold', 'image_name', 'method', 'eye_xai', 'nose_xai', 'mouth_xai'] + ['label','predicted']+['p_%s'%i for i in categories])
    n = 0
    for index in range(0, num_samples):

        #if index%300==0:
        #    print(index, num_samples)
        # load images and labels
        image, gt_label, segmap, filename, landmark = test_dataset.__getitem__(index)
        input = image.unsqueeze(0).to(device)
        input.requires_grad = True

        #print(index, filename) # --> there is only one problematic sample, exclude it: BWSSlide165
        if filename!='BWSSlide165':

            output = model(input)
            prob_output = torch.nn.functional.softmax(output, dim=1)
            prob_output = prob_output.detach().cpu().numpy().squeeze()

            pred_label_idx = np.argmax(prob_output)
            predicted_label = categories[pred_label_idx]

            # get 3-channel RGB map for eye, nose and mouth regions
            def get_largest_connected_components(segmentation):
                labels = label(segmentation)
                assert( np.unique(label(segmap)).size>1)
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                return largestCC

            segmap = cv2.resize(segmap, (224, 224))
            right_eye = [list(np.argwhere(get_largest_connected_components(segmap==i)==True).min(axis=0)) + list(np.argwhere(get_largest_connected_components(segmap==i)==True).max(axis=0)) for i in [2,4]]
            left_eye  = [list(np.argwhere(get_largest_connected_components(segmap==i)==True).min(axis=0)) + list(np.argwhere(get_largest_connected_components(segmap==i)==True).max(axis=0)) for i in [3,5]]
            right_eye, left_eye = np.array(right_eye), np.array(left_eye)

            reye_x1, reye_y1, reye_x2, reye_y2 = right_eye[:,1].min(), right_eye[:,0].min(), right_eye[:,3].max(), right_eye[:,2].max()
            leye_x1, leye_y1, leye_x2, leye_y2 = left_eye[:,1].min(),  left_eye[:,0].min(),  left_eye[:,3].max(),  left_eye[:,2].max()

            reye_w,  reye_h = reye_x2-reye_x1, reye_y2-reye_y1
            reye_x1, reye_y1, reye_x2, reye_y2 = reye_x1-int(0.2*reye_w), reye_y1-int(0.2*reye_h), reye_x2+int(0.1*reye_w), reye_y2+int(0.2*reye_h)
            leye_w,  leye_h = leye_x2-leye_x1, leye_y2-leye_y1
            leye_x1, leye_y1, leye_x2, leye_y2 = leye_x1-int(0.1*leye_w), leye_y1-int(0.2*leye_h), leye_x2+int(0.2*leye_w), leye_y2+int(0.2*leye_h) 

            nose_x1, nose_y1, nose_x2, nose_y2  = list(np.argwhere(get_largest_connected_components(segmap==6)==True).min(axis=0))[::-1] + list(np.argwhere(get_largest_connected_components(segmap==6)==True).max(axis=0))[::-1]  
            nose_y1 = min(reye_y1, leye_y1) + int(0.3 * min(reye_h, leye_h))

            mouth = np.array([list(np.argwhere(get_largest_connected_components(segmap==i)==True).min(axis=0)) + list(np.argwhere(get_largest_connected_components(segmap==i)==True).max(axis=0)) for i in [7,9]])
            mouth_x1, mouth_y1, mouth_x2, mouth_y2  = mouth[:,1].min(), mouth[:,0].min(), mouth[:,3].max(), mouth[:,2].max()
            mouth_y2 += int(0.5 * (mouth_y1 - nose_y2))
            mouth_y1, nose_y2 = 2*[int(0.5 * (mouth_y1 + nose_y2))]
            mouth_x1, mouth_x2 = mouth_x1-int(0.2*(mouth_x2-mouth_x1)), mouth_x2+int(0.2*(mouth_x2-mouth_x1))
            mouth_y2 = min(mouth_y2, 224)

            region_map = np.zeros(shape=(224,224,3), dtype=np.uint8)
            region_map[reye_y1:reye_y2, reye_x1:reye_x2, 0] = 1
            region_map[leye_y1:leye_y2, leye_x1:leye_x2, 0] = 1
            region_map[nose_y1:nose_y2, nose_x1:nose_x2, 1] = 1
            region_map[mouth_y1:mouth_y2, mouth_x1:mouth_x2, 2] = 1
            for i in range(0, 224):
                for j in range(0, 224):
                    region_map[i,j,:] = [1,0,0]  if (region_map[i,j,:]==[1,1,0]).all() else region_map[i,j,:]


            # get saliency map
            relevance_map = get_relevance(model, input, pred_label_idx, xai_method, device)
            relevance_map = cv2.resize(relevance_map, (224, 224))
            relevance_map = region_map[:,:,0] * relevance_map + \
                            region_map[:,:,1] * relevance_map + \
                            region_map[:,:,2] * relevance_map

            # saliency region coefficients
            eye_xai   = np.sum(relevance_map * region_map[:,:,0]) / np.sum(region_map[:,:,0]!=0)
            nose_xai  = np.sum(relevance_map * region_map[:,:,1]) / np.sum(region_map[:,:,1]!=0)
            mouth_xai = np.sum(relevance_map * region_map[:,:,2]) / np.sum(region_map[:,:,2]!=0)

            results.loc[n] = ['VGGFace2_ResNet50'] + [fold] + [filename] + [xai_method] + [eye_xai, nose_xai, mouth_xai] + [test_dataset.categories[int(gt_label)]] + [test_dataset.categories[pred_label_idx]] + ['%2.3f'%i for i in list(prob_output)]
            n += 1

    return results





def main(args):

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms = True

    # set GPU device ID
    device = torch.device("cuda:%d"%args.device if torch.cuda.is_available() and args.device>=0 else "cpu")

    # 'Gradient', 'SmoothGrad', 'IntegratedGradients', 'GuidedBackprop', 'ExcitationBackprop', 'DeconvNet',
    # 'LRP-EpsilonPlus', 'LRP-EpsilonPlusFlat', 'LRP-EpsilonAlpha2Beta1', 'LRP-EpsilonAlpha2Beta1Flat',
    # 'DeepLIFT', 'GuidedGradCam', 'LayerDeepLIFT', 'Occlusion'
    xai_methods = ['LayerGradCam'] #

    pathlib.Path('./results', 'results-rq1').mkdir(parents=True, exist_ok=True)

    for xai_method in xai_methods:
        print(xai_method)
        time.sleep(1)

        rq1 = pd.DataFrame(columns=['backbone', 'fold', 'image_name', 'method', 'eye_xai', 'nose_xai',
                                    'mouth_xai', 'predicted', 'label', 'p_22q11DS', 'p_Angelman', 'p_BWS',
                                    'p_CdLS', 'p_Down', 'p_KS', 'p_NS', 'p_PWS', 'p_RSTS1', 'p_Unaffected',
                                    'p_WHS', 'p_WS'] )
        for fold in ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']:
            results = evaluate_xai_maps(args.dataset_folder, fold, xai_method, device)
            rq1 = pd.concat([rq1, results])
        rq1.to_csv(pathlib.Path('./results', 'results-rq1', '%s.csv'%xai_method).as_posix(), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',   default='VGGFace2_ResNet50',   type=str, help='experiment name')
    parser.add_argument('--seed',           default=42,      type=int, help='random seed')
    parser.add_argument('--device',         default=0,       type=int, help='device: cpu (-1), cuda: 0, 1')
    parser.add_argument('--project_root',   default='./',    type=str, help='project root')
    parser.add_argument('--dataset_folder', type=str, help='Root data directory containing images subfolder with all NIH-Faces.')
    parser.add_argument('--num_classes',    default=12,      type=int,    help='number of classes')

    args = parser.parse_args()
    main(args)



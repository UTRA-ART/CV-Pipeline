import argparse
import os, sys
import shutil
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.function import AverageMeter
from tqdm import tqdm

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def detect(cfg, opt):
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger, opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    # Run inference
    t0 = time.time()

    # vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    iterable = tqdm(enumerate(dataset), total=len(dataset))
    bundle = np.zeros((len(iterable), 3, 3), dtype=np.float64)
    for i, (path, img, img_det, vid_cap, shapes) in iterable:
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = model(img)
        t2 = time_synchronized()
        inf_out, _ = det_out
        inf_time.update(t2 - t1, img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        t4 = time_synchronized()

        nms_time.update(t4 - t3, img.size(0))

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        # _, da_seg_mask = torch.max(da_seg_mask, 1)

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        ### TESTING
        import matplotlib.pyplot as plt

        contours, hierarchy = cv2.findContours(ll_seg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = tuple(
            reversed(sorted(((cv2.contourArea(contour), contour) for contour in contours), key=lambda x: x[0])))

        first = areas[0] if len(areas) >= 1 else None
        second = areas[1] if len(areas) >= 2 else None
        third = areas[2] if len(areas) >= 3 else None

        plt.figure(path)
        plt.xlim([0, 1280])
        plt.ylim([720, 0])

        def plot_line(area_contour):
            if area_contour is None:
                return
            _, contour = area_contour
            x, y = contour[:, :, 0].flatten(), contour[:, :, 1].flatten()
            polyn_coeff = np.polyfit(x, y, 2)
            polyn = np.poly1d(polyn_coeff)
            xp = np.linspace(0, 1200, 100)
            _ = plt.plot(xp, polyn(xp), '-')
            plt.scatter(x, y, 10, 'k')

            return polyn_coeff

        first = plot_line(first)
        second = plot_line(second)
        third = plot_line(third)

        bundle[i, 0, :] = first if first is not None else np.full(3, np.nan)
        bundle[i, 1, :] = second if second is not None else np.full(3, np.nan)
        bundle[i, 2, :] = third if third is not None else np.full(3, np.nan)

        plt.show()

        ###

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

    return bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=r'C:\Users\theda\PycharmProjects\YOLOP\weights\End-to-end.pth',
                        help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='../inference/images',
                        help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        bundle = detect(cfg, opt)

    print(bundle)

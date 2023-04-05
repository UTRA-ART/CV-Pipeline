import argparse

import torch
from models.models import *
from utils.torch_utils import select_device


from utils.google_utils import attempt_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="weights path")
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[448, 448], help="image size"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    # image size(1,3,320,192) iDetection
    img = torch.randn(1, 3, 448, 448, requires_grad=True)
    # Load PyTorch model
    opt.weights = "C:\\Users\\ammar\\Documents\\CodingProjects\\ART\\CV-Pipeline\\src\\pothole_detection\\YOLOv4\\PyTorch_YOLOv4-train\\runs\\train\\exp130\\weights\\best.pt"
    device = select_device("1")
    cfg = "cfg/yolov4-tiny.cfg"

    # attempt_download(WEIGHTS)
    model = Darknet(cfg, 448).cuda()

    model.load_state_dict(torch.load(opt.weights, map_location=device)["model"])
    model.eval()
    # model.model[-1].export = True  # set Detect() layer export=True

    print(f"{img.shape=}")
    img = img.to(device)
    y = model(img)  # dry run

    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        f = opt.weights.replace(".pt", ".onnx")  # filename
        print(f"{f=}")
        model.fuse()  # only for ONNX
        torch.onnx.export(
            model,
            img,
            f,
            verbose=False,
            opset_version=12,
            input_names=["images"],
            output_names=["classes", "boxes"] if y is None else ["output"],
        )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print a human readable model
        print(onnx.helper.printable_graph(onnx_model.graph))
        print("ONNX export success, saved as %s" % f)
    except Exception as e:
        print("ONNX export failure: %s" % e)

    # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' %
    #           torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export

    # CoreML export
    # try:
    #     import coremltools as ct

    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(
    #         name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print("\nExport complete. Visualize with https://github.com/lutzroeder/netron.")

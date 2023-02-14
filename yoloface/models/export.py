import argparse
import sys
import time
import warnings

sys.path.append("./")  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./yolor-csp-c.pt", help="weights path")
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[640, 640], help="image size"
    )  # height, width
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="dynamic batch onnx for tensorrt and onnx-runtime",
    )
    parser.add_argument("--grid", action="store_true", help="export Detect() layer grid")
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument(
        "--max-wh",
        type=int,
        default=None,
        help="None for tensorrt nms, int value for onnx-runtime nms",
    )
    parser.add_argument("--topk-all", type=int, default=100, help="topk objects for every images")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="iou threshold for NMS")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="conf threshold for NMS")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--simplify", action="store_true", help="simplify onnx model")
    parser.add_argument("--fp16", action="store_true", help="CoreML FP16 half-precision export")
    parser.add_argument("--int8", action="store_true", help="CoreML INT8 quantization")
    parser.add_argument(
        "--trt", action="store_true", help="True for tensorrt, false for onnx-runtime"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="True for using onnx_graphsurgeon to sort and remove unused",
    )
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(
        device
    )  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run

    import onnx

    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    f = opt.weights.replace(".pt", ".onnx")  # filename
    model.eval()
    output_names = ["output"]

    dynamic_axes = None
    if opt.dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},  # size(1,3,640,640)
            "output": {0: "batch", 2: "y", 3: "x"},
        }
    if opt.dynamic_batch:
        opt.batch_size = "batch"
        dynamic_axes = {
            "images": {
                0: "batch",
            },
        }
        if opt.end2end and opt.trt:
            output_axes = {
                "num_dets": {0: "batch"},
                "det_boxes": {0: "batch"},
                "det_scores": {0: "batch"},
                "det_classes": {0: "batch"},
                "det_lmks": {0: "batch"},
            }
        else:
            output_axes = {
                "output": {0: "batch"},
            }
        dynamic_axes.update(output_axes)

    if opt.grid:
        if opt.end2end:
            print(
                "\nStarting export end2end onnx model for %s..." % "TensorRT"
                if opt.trt
                else "onnxruntime"
            )
            model = End2End(
                model=model,
                max_obj=opt.topk_all,
                iou_thres=opt.iou_thres,
                score_thres=opt.conf_thres,
                max_wh=opt.max_wh,
                trt=opt.trt,
                device=device,
            )
            if opt.end2end and opt.trt:
                output_names = ["num_dets", "det_boxes", "det_scores", "det_classes", "det_lmks"]
                shapes = [
                    opt.batch_size,
                    1,
                    opt.batch_size,
                    opt.topk_all,
                    4,
                    opt.batch_size,
                    opt.topk_all,
                    opt.batch_size,
                    opt.topk_all,
                    opt.batch_size,
                    opt.topk_all,
                    10,
                ]
            else:
                output_names = ["output"]
        else:
            model.model[-1].concat = True

    torch.onnx.export(
        model,
        img,
        f,
        verbose=False,
        opset_version=12,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if opt.end2end and opt.trt:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    # # Metadata
    # d = {'stride': int(max(model.stride))}
    # for k, v in d.items():
    #     meta = onnx_model.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(onnx_model, f)

    if opt.simplify:
        try:
            import onnxsim

            print("\nStarting to simplify ONNX...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")

    if opt.cleanup:
        try:
            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(onnx_model)
            graph = graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model, f)
    print("ONNX export success, saved as %s" % f)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )

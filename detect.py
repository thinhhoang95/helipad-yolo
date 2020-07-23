import argparse

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2 as cv2

def bprj(x, Ric, xb1, yb1, xb2, yb2, mfoc, R):
    a1, b1, c1 = Ric[0,0], Ric[0,1], Ric[0,2]
    a2, b2, c2 = Ric[1,0], Ric[1,1], Ric[1,2]
    a3, b3, c3 = Ric[2,0], Ric[2,1], Ric[2,2]
    s = 0.00000575
    xb1 = (xb1-320) * s
    yb1 = (yb1-240) * s
    xb2 = (xb2-320) * s
    yb2 = (yb2-240) * s
    return np.array([
        ((a1 - a3*xb1/mfoc)*x[0] + (b1 - b3*xb1/mfoc)*x[1] + (c1 -c3*xb1/mfoc)*x[4]),
        ((a2 - a3*yb1/mfoc)*x[0] + (b2 - b3*yb1/mfoc)*x[1] + (c2 -c3*yb1/mfoc)*x[4]),
        ((a1 - a3*xb2/mfoc)*x[2] + (b1 - b3*xb2/mfoc)*x[3] + (c1 -c3*xb2/mfoc)*x[4]),
        ((a2 - a3*yb2/mfoc)*x[2] + (b2 - b3*yb2/mfoc)*x[3] + (c2 -c3*yb2/mfoc)*x[4]),
        10*((x[2]-x[0])**2 + (x[3]-x[1])**2 - R**2),
    ])
    
def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def detect(save_img=False):
    
    # Load the camera matrix and distortion from file
    cam_mat = np.load('cam_mat.pca.npy')
    dist = np.load('dist.pca.npy')
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # Declaration of common variables -------------------------
    eulang_file = np.genfromtxt('data/samples/eulang.txt', delimiter=',')
    eulang_cursor = 0
    foc = 3.04e-3
    Rbc = Rotation.from_euler('ZYX', np.array([-90,0,0]), degrees=True).as_dcm()
    pose_sol_a = np.array([0.1,0.1,0.4,0.4,-1.5]) # initial solution for optimization
    Ritip = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # rotates around X axis for 180 degrees
    Ripi = Rotation.from_euler('ZYX',np.array([150,0,0]), degrees=True).as_dcm()
    # ----------------------------------------------------------
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # Initialize model
    model = Darknet(opt.cfg, imgsz)
    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0, im1= path, '', im0s, im0s.copy()

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                imwrite_row = 0
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im1, label=label, color=colors[int(cls)])
                        # Further processing of detection results
                        # Get the timestamp
                        img_timestamp = float(path.split('_',1)[1][:-4])
                        while (eulang_file[eulang_cursor, 0] < img_timestamp):
                            eulang_cursor = eulang_cursor + 1
                        # Get the Euler angles of this image
                        ypr = eulang_file[eulang_cursor, 1:4]
                        Rib = Rotation.from_euler('ZYX', ypr, degrees=False).as_dcm().T
                        Ric = Rbc @ Rib
                        print('Ric from IMU: ', Rotation.from_dcm(Ric).as_euler('ZYX', degrees=True))
                        # Perform optimization
                        res = least_squares(bprj, pose_sol_a, args=(Ric, float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), foc, 0.362))
                        # res_2 = least_squares(bprj, pose_sol_b, args=(Ric, float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), foc, 0.362))
                        # pose_sol = res_1.x
                        # Write this information on the image
                        xi = np.array([(res.x[0] + res.x[2]) / 2.0, (res.x[1] + res.x[3]) / 2.0, res.x[4]])
                        imwrite_row = imwrite_row + 1
                        cv2.putText(im1, 'AI: ' + str(xi), (5, imwrite_row * 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        cv2.aruco.drawAxis(im1, cam_mat, dist, Ric, -Ric @ xi.T, 0.05) # Helipad by AI information
                        imwrite_row = imwrite_row + 1
                        cv2.putText(im1, 'Sol 1: ' + str(res.x), (5, imwrite_row * 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        #imwrite_row = imwrite_row + 1
                        #cv2.putText(im1, 'Sol 2: ' + str(res_2.x), (5, imwrite_row * 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        imwrite_row = imwrite_row + 1
                        cv2.putText(im1, 'YPR: ' + str(ypr/np.pi*180), (5, imwrite_row * 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        imwrite_row = imwrite_row + 1
                        cv2.putText(im1, 'Box: %d %d %d %d' % (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])), (5, imwrite_row * 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        # >>> Infer data from ARUCO tag >>>
                        
                        #Load the dictionary that was used to generate the markers.
                        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

                        # Initialize the detector parameters using default values
                        parameters =  cv2.aruco.DetectorParameters_create()

                        # Detect the markers in the image
                        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(im0, dictionary, parameters=parameters)
                        rvecs, tvecs, *other = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.18, cam_mat, dist)
                        if rvecs is None:
                            print('Invalid ARUCO tag detected in the image. Skipping this image.')
                        else:
                            for rvec, tvec in zip(rvecs, tvecs):
                                rvec = rvec[0]
                                tvec = tvec[0]
                                Ritc = Rotation.from_rotvec(rvec).as_dcm()
                                Riti = Ripi @ Ritip
                                Ric2 = Ritc @ Riti.T
                                print('Ric from ArucoTag: ', Rotation.from_dcm(Ric2).as_euler('ZYX', degrees=True))
                                heli_pos = Riti @ np.array([0.245,0,0]).T - Ric.T @ tvec.T # position with respect to the helipad
                                print('ARUCO detected at ', heli_pos)
                                cv2.aruco.drawAxis(im1, cam_mat, dist, Ric2, - Ric2 @ heli_pos, 0.05) # Position of helipad by ARUCO information
                                cv2.aruco.drawAxis(im1, cam_mat, dist, Ritc, tvec, 0.05) # ARUCO tag
                                imwrite_row = imwrite_row + 1
                                cv2.putText(im1, 'ARUCO: ' + str(heli_pos), (5, imwrite_row*10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                        # <<< Infer data from ARUCO tag <<<

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im1)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples/', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()

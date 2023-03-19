import cv2
import numpy as np
import sys
from copy import deepcopy


with open("failed.txt", mode='w') as f :   
    f.write('')


def scale_to_width(img, width):
    h, w, _ = img.shape
    height = round(h * (width / w))
    dst = cv2.resize(img, dsize=(width, height))

    scale = w / width

    return dst, scale

def napprox(cnt, n=0.1):
        if n > 1:
            print("Failed max n")
            return "Failed"
        epsilon = n*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        try:
            areas = approx.reshape(4,2)
            return areas
        except ValueError:
            return napprox(cnt, n+0.05)


def crop(file):
    print("Processing {}".format(file))
    # 画像を読み込む。
    img_src = cv2.imread(file)

    img_resized, scale = scale_to_width(deepcopy(img_src), 500)

    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # 2値化する
    img_bin = cv2.inRange(img_hsv, (66, 12, 88), (255, 255, 255))
    img_bin_ng = cv2.bitwise_not(img_bin)

    tmp= cv2.findContours(img_bin_ng, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = tmp[0]

    # 四角形(not 矩形)を近似する

    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 10000:
            areas = napprox(cnt)

    if len(areas) == 0:
        return file + " AreaNotFound"

    sort_indexes = np.argsort(areas,axis=0).T

    def pick_pt_index(is_top, is_left):
        if is_top:
            l1 = sort_indexes[1][:2]
        else:
            l1 = sort_indexes[1][2:]
        if is_left:
            l2 = sort_indexes[0][:2]
        else:
            l2 = sort_indexes[0][2:]

        return list(set(l1) & set(l2))[0]

    pt1=areas[pick_pt_index(True, True)] #左上
    pt2=areas[pick_pt_index(True, False)] #右上
    pt3=areas[pick_pt_index(False, True)] #左下
    pt4=areas[pick_pt_index(False, False)] #右下
    pts = np.float32(np.array([pt1,pt2,pt3,pt4])) * scale

    o_width = np.linalg.norm(pts[1] - pts[0])
    o_width=int(np.floor(o_width))
    o_height = np.linalg.norm(pts[2] - pts[0])
    o_height=int(np.floor(o_height))

    dst_cor=np.float32([[0,0],[o_width,0],[0, o_height],[o_width, o_height]])

    M = cv2.getPerspectiveTransform(pts,dst_cor)

    img_res = cv2.warpPerspective(deepcopy(img_src),M,(o_width,o_height))

    cv2.imwrite("./crop/" + file, img_res)

if __name__ == "__main__":
    files = sys.argv[1:]
    res = ""
    for f in files:
        res = crop(f)
        if res is not None:
            print("Error: " + res)
            with open("failed.txt", mode='a') as fn:   
                fn.write(f + "\n")
        else:
            res = ""

#    p = Pool(cpu_count()) # プロセス数を指定する
#    try:
#        result = p.map(crop, files)  # リストで引数を指定する
#        print("Failed: {}".format(list(filter(None, result))))
#    except Exception as e:
#        print("main", e)

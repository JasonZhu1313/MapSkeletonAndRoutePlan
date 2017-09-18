import cv2
import yaml
from skimage import morphology
import numpy as np


META_PATH = "/home/jason/turtlebot_custom_maps/test0.yaml"


def skeleton():
    image_path, resolution, origin, negate, occupied_thresh, free_thresh = readconfig()
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, bin_image = cv2.threshold(gray, 220, 1, cv2.THRESH_BINARY)
    cv2.imshow("oringin",gray)
    dilate = dilate_image(bin_image)
    skeleton = thin2_image(dilate)
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    harris_result, point_list = harris_corner_point(skeleton_rgb)
    robot1_route = []
    robot2_route = []
    if len(point_list) is not 0:
        #binary division of the points
        for i in range(len(point_list)):
            if i < len(point_list) / 2:
                robot1_route.append(point_list[i])
            else:
                robot2_route.append(point_list[i])
    print ("Robot1 route: {}".format(robot1_route))
    print ("Robot2 route: {}".format(robot2_route))

    cv2.imshow("harris_skeleton", harris_result)
    cv2.waitKey()


    # show_image(gray,skeleton=skeleton)


#
# def show_image(grey,skeleton):
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#     ax1.imshow(grey, cmap=plt.cm.gray)
#     ax1.axis('off')
#     ax1.set_title('original', fontsize=20)
#     ax2.imshow(skeleton, cmap=plt.cm.gray)
#     ax2.axis('off')
#     ax2.set_title('skeleton', fontsize=20)
#     fig.tight_layout()
#     plt.show()
#     cv2.waitKey()

# erode method
def erode_image(grey):
    kernel = np.uint8(np.zeros((3, 3)))
    for x in range(3):
        kernel[x, 1] = 1;
        kernel[1, x] = 1;
    eroded = cv2.erode(grey, kernel)
    return eroded


# p9 p2 p3
# p8 p1 p4
# p7 p6 p5
# A fast parallel algorithm for thinning digital patterns
def thin1_image(image, max_iterations=-1):
    height = image.shape[0]
    width = image.shape[1]
    # record the iterate time
    count = 0;
    while (1):
        count = count + 1
        if (max_iterations != -1 and count > max_iterations):
            break
        # the first subiternation
        for i in range(height):
            p = []
            for j in range(width):
                # print image[i,j]
                p1 = image[i, j]

                if p1 == 1:
                    p4 = 0 if j == width - 1 else image[i, j + 1]
                    p8 = 0 if j == 0 else image[i, j - 1]
                    p2 = 0 if i == 0 else image[i - 1, j]
                    p3 = 0 if (i == 0 or j == width - 1) else image[i - 1, j + 1]
                    p9 = 0 if (i == 0 or j == 0) else image[i - 1, j - 1]
                    p6 = 0 if (i == height - 1) else image[i + 1, j]
                    p5 = 0 if (i == height - 1 or j == width - 1) else image[i + 1, j + 1]
                    p7 = 0 if (i == height - 1 or j == 0) else image[i + 1, j - 1]
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2
                        and (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6):
                        ap = 0
                        if (p2 == 0 and p3 == 1):
                            ap = ap + 1
                        if (p3 == 0 and p4 == 1):
                            ap = ap + 1
                        if (p4 == 0 and p5 == 1):
                            ap = ap + 1
                        if (p5 == 0 and p6 == 1):
                            ap = ap + 1
                        if (p6 == 0 and p7 == 1):
                            ap = ap + 1
                        if (p7 == 0 and p8 == 1):
                            ap = ap + 1
                        if (p8 == 0 and p9 == 1):
                            ap = ap + 1
                        if (p9 == 0 and p2 == 1):
                            ap = ap + 1
                        print ap
                        if (ap == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                            print i, j
                            p.append((i, j))
                            image[i, j] = 0
        if len(p) == 0:
            break
        else:
            p = []
        for i in range(height):
            for j in range(width):
                p1 = image[i, j]
                if p1 == 1:
                    p4 = 0 if j == width - 1 else image[i, j + 1]
                    p8 = 0 if j == 0 else image[i, j - 1]
                    p2 = 0 if i == 0 else image[i - 1, j]
                    p3 = 0 if (i == 0 or j == width - 1) else image[i - 1, j + 1]
                    p9 = 0 if (i == 0 or j == 0) else image[i - 1, j - 1]
                    p6 = 0 if (i == height - 1) else image[i + 1, j]
                    p5 = 0 if (i == height - 1 or j == width - 1) else image[i + 1, j + 1]
                    p7 = 0 if (i == height - 1 or j == 0) else image[i + 1, j - 1]
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2
                        and (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6):
                        ap = 0
                        if (p2 == 0 and p3 == 1):
                            ap = ap + 1
                        if (p3 == 0 and p4 == 1):
                            ap = ap + 1
                        if (p4 == 0 and p5 == 1):
                            ap = ap + 1
                        if (p5 == 0 and p6 == 1):
                            ap = ap + 1
                        if (p6 == 0 and p7 == 1):
                            ap = ap + 1
                        if (p7 == 0 and p8 == 1):
                            ap = ap + 1
                        if (p8 == 0 and p9 == 1):
                            ap = ap + 1
                        if (p9 == 0 and p2 == 1):
                            ap = ap + 1
                        if (ap == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                            p.append((i, j))
                            image[i, j] = 0
        if len(p) == 0:
            break
        else:
            p = []
    return image


# open operation
def open_operation(grey):
    erode = erode_image(grey)
    result = dilate_image(erode)
    return result


# close operation
def close_operation(grey):
    dilate = dilate_image(grey)
    result = erode_image(dilate)
    return result


# dilate the image
def dilate_image(grey):
    kernel = np.uint8(np.zeros((3, 3)))
    for x in range(3):
        kernel[x, 1] = 1;
        kernel[1, x] = 1;
    dilate = cv2.dilate(grey, kernel)
    return dilate


def thin2_image(grey):
    skeleton = morphology.skeletonize(grey)
    height = skeleton.shape[0]
    width = skeleton.shape[1]
    print ("height {}".format(skeleton.shape[0]))
    print ("width {}".format(skeleton.shape[1]))
    result = np.uint8(np.zeros((height, width)))
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j]:
                result[i, j] = 255
    return result


def harris_corner_point(image):
    image_temp = image
    grey = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
    grimage_tempey = np.float32(grey)
    dst = cv2.cornerHarris(grey, 2, 15, 0.18)
    dst = cv2.dilate(dst, None)
    count = 0
    keypoint = (0, 0)
    point_list = []
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i, j] > 0.1 * dst.max():

                # delete the points to make the route more concise
                euclidean = map(lambda x, y: (x - y) ** 2, keypoint, (i, j))
                distance = euclidean[0] + euclidean[1]
                if distance > 800:
                    print distance
                    keypoint = (i, j)
                    point_list.append((i, j))
                    image_temp[i, j] = [0, 0, 255]
                    count = count + 1
                    print ("point x :{}, point y:{}".format(i, j))
                else:
                    continue

    print count
    return image_temp, point_list


def readconfig():
    with open(META_PATH, 'r') as f:
        attr = yaml.load(f)
        image_path = attr["image"]
        resolution = attr["resolution"]
        origin = attr["origin"]
        negate = attr["negate"]
        occupied_thresh = attr["occupied_thresh"]
        free_thresh = attr["free_thresh"]
        return image_path, resolution, origin, negate, occupied_thresh, free_thresh


if __name__ == "__main__":
    skeleton()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('error')


def conv(mat1, mat2):
    len = mat1.shape[0]
    sum = 0
    for i in range(len):
        for j in range(len):
            sum += mat1[i][j] * mat2[i][j]
    return sum


def filtering(img, pad_size=1):
    f_xy = np.array([[0, 2, 0], [2, -1, 2], [0, 2, 0]])
    f_45135 = np.array([[2, 0, 2], [0, 0, 0], [2, 0, 2]])

    background = np.zeros(shape=(np.add(img.shape, [pad_size * 2, pad_size * 2])))
    background[pad_size:-pad_size, pad_size:-pad_size] = img

    output = np.zeros(shape=img.shape)
    for y in range(pad_size, len(background) - pad_size):
        for x in range(pad_size, len(background[y]) - pad_size):
            target = background[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
            output[y - 1, x - 1] = np.round(
                np.sqrt(np.power(conv(target, f_45135), 2) + np.power(conv(target, f_xy), 2)))
    return output


def normalize(img):
    output = img
    max_value = np.max(img)
    min_value = np.min(img)
    differ = np.abs(max_value - min_value)
    for i in range(len(img)):
        for j in range(len(img[i])):
            output[i][j] = ((output[i][j] - min_value) / differ * 255)
    return output.astype(np.uint8)


def edge_detection(img, pad_size=1):
    G_x = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]]) / 2 * 2 ** 0.5
    G_y = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]]) / 2 * 2 ** 0.5
    G_45 = np.array([[0, 1, -np.sqrt(2)], [-1, 0, 1], [np.sqrt(2), -1, 0]]) / 2 * 2 ** 0.5
    G_135 = np.array([[-np.sqrt(2), 1, 0], [1, 0, -1], [0, -1, np.sqrt(2)]]) / 2 * 2 ** 0.5

    background = np.array(img)
    G_output = np.array(img)
    Theta_output = np.zeros(shape=img.shape)
    Output = np.array(img)

    for y in range(pad_size, len(background) - pad_size):
        for x in range(pad_size, len(background[y]) - pad_size):
            target = background[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
            G_output[y - 1, x - 1] = np.round(
                np.sqrt(
                    np.power(conv(target, G_x), 2) + np.power(conv(target, G_y), 2) + np.power(conv(target, G_45), 2)
                    + np.power(conv(target, G_135), 2)))
            try:
                Theta_output[y - 1, x - 1] = (np.arctan(
                    (conv(target, G_y)) / (conv(target, G_x))) * 180 / np.pi) % 180
                # print(Theta_output[y - 1, x - 1])
            except RuntimeWarning as exc:
                Theta_output[y - 1, x - 1] = (np.arctan(
                    (conv(target, G_y) + 0.001) / (conv(target, G_x) + 0.001)) * 180 / np.pi) % 180
                # print(Theta_output[y - 1, x - 1])
    return G_output.astype(np.uint8), Theta_output


def edge_suppression(G, Theta, pad_size=1):
    output = np.array(G)
    for y in range(pad_size, len(G) - pad_size):
        for x in range(pad_size, len(G[y]) - pad_size):
            degree = Theta[y - 1][x - 1]
            if degree <= 22.5 or degree >= 157.5:
                if G[y - 1][x - 1] < G[y - 1][x - 2] or G[y - 1][x - 1] < G[y - 1][x]:
                    output[y - 1][x - 1] = 0
            elif 22.5 < degree <= 67.5:
                if G[y - 1][x - 1] < G[y - 2][x] or G[y - 1][x - 1] < G[y][x - 2]:
                    output[y - 1][x - 1] = 0
            elif 67.5 < degree <= 112.5:
                if G[y - 1][x - 1] < G[y][x - 1] or G[y - 1][x - 1] < G[y - 2][x - 1]:
                    output[y - 1][x - 1] = 0
            elif 112.5 < degree <= 157.5:
                if G[y - 1][x - 1] < G[y][x] or G[y - 1][x - 1] < G[y - 2][x - 2]:
                    output[y - 1][x - 1] = 0
    return output


def otsu_threshold(img):
    width, height = img.shape
    # wid = np.array([i * np.int(width / 3) for i in range(1, 3)])
    # hei = np.array([i * np.int(height / 3) for i in range(1, 3)])
    # size=[]
    # for i in range(3):
    #     for j in range(3):
    #         size.append([])
    retval1, threshold1 = cv2.threshold(img[:np.int(width / 2), :np.int(height / 2)], 0, 255, cv2.THRESH_OTSU)
    retval2, threshold2 = cv2.threshold(img[np.int(width / 2):, :np.int(height / 2)], 0, 255, cv2.THRESH_OTSU)
    retval3, threshold3 = cv2.threshold(img[:np.int(width / 2), np.int(height / 2):], 0, 255, cv2.THRESH_OTSU)
    retval4, threshold4 = cv2.threshold(img[np.int(width / 2):, np.int(height / 2):], 0, 255, cv2.THRESH_OTSU)

    return (retval1 + retval2 + retval3 + retval4) / (4 * 2)


def db_threshold(img, high, low, pad_size=1):
    output = np.array(img)
    for y in range(pad_size, len(img) - pad_size):
        for x in range(pad_size, len(img[0]) - pad_size):
            if img[y][x] < low:
                output[y][x] = 0
            elif low <= img[y][x] <= high:
                clear = True
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if img[y + i][x + j] >= high:
                            clear = False
                if clear:
                    output[y][x] = 0
            else:
                output[y][x] = 255
    return output


def cv_model(img):
    filtered = filtering(test_img)
    normalized = normalize(filtered)
    detected_G, detected_Theta = edge_detection(normalized)
    thinned = edge_suppression(detected_G, detected_Theta)
    # todo: if want the paper version, comment below 1 line or inversing
    _, otsued = cv2.threshold(thinned, 0, 255, cv2.THRESH_OTSU)
    # todo: if want the paper version, uncomment below and return final or inversing
    # threshold_high = otsu_threshold(thinned)
    # threshold_low = threshold_high / 2
    # final = db_threshold(thinned, threshold_high, threshold_low)
    return otsued


if __name__ == '__main__':
    test_img = cv2.imread("demo2.jpg", cv2.IMREAD_GRAYSCALE)
    print(test_img.shape)
    # cv2.imshow("123", test_img)
    # cv2.waitKey(0)
    filtered = filtering(test_img)

    # todo: cv2.imshow uint8:0-255 double:0-1

    normalized = normalize(filtered)
    # cv2.imshow("234", normalized)
    # cv2.waitKey(0)

    detected_G, detected_Theta = edge_detection(normalized)
    # cv2.imshow("345", detected_G)
    # cv2.waitKey(0)
    # sobelxy = cv2.Sobel(normalized, cv2.CV_8U, 1, 1, ksize=3)
    # cv2.imshow("sobel", sobelxy)
    # cv2.waitKey(0)

    thinned = edge_suppression(detected_G, detected_Theta)
    # cv2.imshow("456", thinned)
    # cv2.waitKey(0)
    threshold_high = otsu_threshold(thinned)
    threshold_low = threshold_high / 2
    final = db_threshold(thinned, threshold_high, threshold_low)
    cv2.imshow("567", final)
    cv2.waitKey(0)
    # _, finall = cv2.threshold(thinned, threshold_high, 255, cv2.THRESH_BINARY)
    # cv2.imshow("678", finall)
    # cv2.waitKey(0)
    _, otsued = cv2.threshold(thinned, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("789", otsued)
    cv2.waitKey(0)
    print(final.shape)

import cv2
import numpy as np
import time


def calculateMaxAndMinDistance(dst):
    z = np.array([])
    y = dst.reshape(-1).reshape(-1, 2)
    z = np.append(z, (np.linalg.norm(np.array(y[0]) - np.array(y[1]))))
    z = np.append(z, (np.linalg.norm(np.array(y[0]) - np.array(y[3]))))
    z = np.append(z, (np.linalg.norm(np.array(y[1]) - np.array(y[2]))))
    z = np.append(z, (np.linalg.norm(np.array(y[2]) - np.array(y[3]))))
    return z.max(), z.min()


def findLogo(frame, kp1, des1,h,w):
    stra = ""
    masksum = 0
    M_topLeft_det = 0
    area = 0
    perimeter = 0
    start_time = time.time()
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # img2 = frame.copy()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(len(kp2), "Features frame ")
    print("Time:", time.time() - start_time)
    if len(kp2) != 0 and des2 is not None:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        print(len(matches),"matches found")

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        print(len(good),"good matches found")
        print("Time:", time.time() - start_time)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # h, w = img1.shape
            # h = h
            # w = w
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M is not None:
                for i in matchesMask:
                    masksum = masksum + i
                # print(masksum)
                # print("Masksum", masksum)
                # print (M)
                # Check if the homography is good.
                # If the determinant of the top left 2x2 matrix is >0, then it is good
                M_topLeft = M[0:2:1, 0:2:1]
                M_topLeft_det = np.linalg.det(M_topLeft)
                dst = cv2.perspectiveTransform(pts, M)
                # Calculate relative gap between the longest and shortest sides
                (longest_side, shortest_side) = calculateMaxAndMinDistance(dst)
                relative_gap = (longest_side - shortest_side) / longest_side
                # Find the area of the box of the 4 points
                area = cv2.contourArea(dst.reshape((-1, 1, 2)).astype(np.int32))

                if masksum > 15 and M_topLeft_det > 0 and area > 100 and relative_gap < 0.8:
                    print("Drawing")
                    for i in dst:
                        for j in i:
                            stra = stra + str(j[0]) + " " + str(j[1])
                        stra = stra + " "
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    frame = cv2.putText(frame, str(masksum) + " " +
                                        str(M_topLeft_det) + " " +
                                        str(area) + " " + str(relative_gap),
                                        (0, 300), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                # else:
                #     print("Not Drawing")
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
    print("Time:", time.time() - start_time)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # frame = cv2.putText(frame, str(len(good)), (0, 300), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # frame = cv2.putText(frame, stra, (0, 400), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    return frame

import numpy as np
import cv2
import random
import math
from scipy.optimize import least_squares
from tqdm import tqdm

def least_sq_H(pts1, pts2):

    A = np.zeros((2*len(pts1), 9))
    idx = 0

    for pt1, pt2 in zip(pts1, pts2):
        ele12 = pt2[2] * pt1
        ele13 = pt2[1] * pt1
        ele23 = -1 * pt2[0] * pt1
        A[idx][3:6] = -1 * ele12
        A[idx][6:9] = ele13
        A[idx+1][:3] = ele12
        A[idx+1][6:9] = ele23
        idx+=2
    
    _, _, V = np.linalg.svd(A)
    H = np.transpose(V)[:,-1]
    H = H.reshape(3,3)
    H /= H[2,2]

    return H

## IMPLIMENTATION USING INHOMOGENOUS EQUATIONS
def in_least_sq_H(pts1, pts2):

    A = np.zeros((2*len(pts1), 8))
    B = np.zeros((2*len(pts1),))
    idx = 0

    for pt1, pt2 in zip(pts1, pts2):
        ele12 = pt2[2] * pt1
        ele13 = pt2[1] * pt1
        ele23 = -1 * pt2[0] * pt1

        A[idx][3:6] = -1 * ele12
        A[idx][6:8] = ele13[:2]
        A[idx+1][:3] = ele12
        A[idx+1][6:8] = ele23[:2]

        B[idx] = -1 * ele13[2]
        B[idx + 1] = -1 * ele23[2]

        idx+=2
    
    A_p = np.linalg.pinv(A)
    H = np.matmul(A_p,B)
    H = np.append(H,1)
    H = H.reshape(3,3)

    return H

def ransac(pts1, pts2):

    sample_size = 6
    p = 0.99
    epsilon = 0.5
    delta = 3
    M = int(len(pts1) * (1-epsilon))

    trails = math.ceil((math.log(1 - p)) / (math.log(1 - ((1-epsilon)**sample_size))))

    best_inliers_1 = []
    best_inliers_2 = []
    best_outliers_1 = []
    best_outliers_2 = []
    best_H = np.zeros((3,3))

    for i in range(trails):
        inliers_1 = []
        inliers_2 = []

        outliers_1 = []
        outliers_2 = []

        rng_pts_idx = random.sample(range(len(pts1)), sample_size)
        rnd_pts1 = pts1[rng_pts_idx]
        rnd_pts2 = pts2[rng_pts_idx]

        est_H = least_sq_H(rnd_pts1, rnd_pts2)

        for pt1, pt2 in zip(pts1, pts2):
            test_p2 = np.matmul(est_H, pt1)
            test_p2 /= test_p2[2]

            diff = pt2 - test_p2
            diff = np.sqrt(diff[0]**2 + diff[1]**2)

            if diff <= delta:
                inliers_1.append(pt1)
                inliers_2.append(pt2)
            else:
                outliers_1.append(pt1)
                outliers_2.append(pt2)
        

        if len(inliers_1) > M:
            M = len(inliers_1)
            best_H = est_H

            best_inliers_1 = inliers_1
            best_inliers_2 = inliers_2

            best_outliers_1 = outliers_1
            best_outliers_2 = outliers_2
    
    return best_H, np.array(best_inliers_1), np.array(best_inliers_2), np.array(best_outliers_1), np.array(best_outliers_2), est_H

def sift(image1, image2, output="out.jpg"):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift_obj = cv2.SIFT_create()
    matching_criteria = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    interest_points1, desc_1 = sift_obj.detectAndCompute(img1, None)
    interest_points2, desc_2 = sift_obj.detectAndCompute(img2, None)

    corres_features = matching_criteria.match(desc_1, desc_2)
    corres_features = sorted(corres_features, key = lambda x:x.distance)
    
    pts1 = []
    pts2 = []

    for feature in corres_features:
        pts1.append([interest_points1[feature.queryIdx].pt[0], interest_points1[feature.queryIdx].pt[1], 1])
        pts2.append([interest_points2[feature.trainIdx].pt[0], interest_points2[feature.trainIdx].pt[1], 1])

    img1 = cv2.drawMatches(image1, interest_points1, image2, interest_points2, corres_features[:100], None)
    cv2.imwrite(output, img1)

    return (np.array(pts1), np.array(pts2))

def visualizer(img1, img2, best_inliers_1, best_inliers_2, best_outliers_1, best_outliers_2, output="line.jpg"):
    best_inliers_1 = np.delete(best_inliers_1, 2, 1)
    best_inliers_2 = np.delete(best_inliers_2, 2, 1)

    best_inliers_2[:, 0] += img1.shape[1] # offsetting interest points in second image for plotting purpose
    combo_image = np.concatenate((img1, img2), axis=1)

    best_inliers_1 = best_inliers_1.astype(int)
    best_inliers_2 = best_inliers_2.astype(int)

    best_outliers_1 = np.delete(best_outliers_1, 2, 1)
    best_outliers_2 = np.delete(best_outliers_2, 2, 1)

    best_outliers_2[:, 0] += img1.shape[1] # offsetting interest points in second image for plotting purpose

    best_outliers_1 = best_outliers_1.astype(int)
    best_outliers_2 = best_outliers_2.astype(int)

    for i in range(len(best_inliers_1)):
        cv2.circle(combo_image, best_inliers_1[i], radius=5, color=(255, 0, 0), thickness=-1)
        cv2.circle(combo_image, best_inliers_2[i], radius=5, color=(255, 0, 0), thickness = -1)
        cv2.line(combo_image, best_inliers_1[i], best_inliers_2[i], (255 ,0, 0), 1)

    for i in range(40):#len(best_outliers_1)):
        cv2.circle(combo_image, best_outliers_1[i], radius=5, color=(0 ,0, 255), thickness=-1)
        cv2.circle(combo_image, best_outliers_2[i], radius=5, color=(0 ,0, 255), thickness = -1)
        cv2.line(combo_image, best_outliers_1[i], best_outliers_2[i], (0 ,0, 255), 1)

    cv2.imwrite(output, combo_image)

def cost(H, x, y):

    H = H.reshape(3,3)
    f_p = np.matmul(H, x.T)
    f_p = (f_p.T / f_p.T[:,-1].reshape(-1,1)).astype(int)

    cost = np.square(y - f_p)
    cost = np.sum(cost, axis=1)
    cost = np.sqrt(cost)

    return cost

def LM_estimation(H, x, y):

    H = H.flatten()
    H = least_squares(cost, H, method='lm', args=(x, y))
    H = H.x.reshape(3,3)

    return H

def image_warp(img1, img2, H, offset_x, offset_y):
    final = img1

    for i in tqdm(range(img1.shape[1])):
        for j in range(img1.shape[0]):
            x, y, w = np.dot(np.linalg.inv(H), np.array([i + offset_x, j + offset_y, 1]))
            x_int = int(np.floor(x / w))
            y_int = int(np.floor(y / w))
            if y_int > 0 and y_int < img2.shape[0] and x_int > 0 and x_int < img2.shape[1]:
                final[j,i] = img2[y_int, x_int]

    return final

def reposition(p, q, r, s, H):
    x_max = 0
    y_max = 0
    x_min = 0
    y_min = 0

    for i in [p, q, r , s]:
        coord = np.dot(H, np.array([i[0], i[1], 1]))
        if coord[0] > x_max:
             x_max = coord[0]
        if coord[1] > y_max:
             y_max = coord[1]
        if coord[0] < x_min:
             x_min = coord[0]
        if coord[1] < y_min:
             y_min = coord[1]

    return int(x_max), int(x_min), int(y_max), int(y_min)  

def get_bounds(all_img, all_H):
    
    final_max_x = 0
    final_max_y = 0
    final_min_x = np.inf
    final_min_y = np.inf

    for img, H in zip(all_img, all_H):
        x_max, x_min, y_max, y_min = reposition((0,0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0]), H)

        if x_max > final_max_x:
            final_max_x = x_max
        if y_max > final_max_y:
            final_max_y = y_max
        if x_min < final_min_x:
            final_min_x = x_min
        if y_min < final_min_y:
            final_min_y = y_min
    
    return int(final_max_x), int(final_min_x), int(final_max_y), int(final_min_y)

def resize_all(all_images, all_H):
    for i in range(len(all_images)):
        x_max, x_min, y_max, y_min = reposition((0,0), (all_images[i].shape[1], 0), (all_images[i].shape[1], all_images[i].shape[0]), (0, all_images[i].shape[0]), all_H[i])

        w_out = x_max - x_min
        h_out = y_max - y_min
        aspect_ratio = w_out/h_out
        h_out_final =500 # user defined 
        w_out_final = int(aspect_ratio * h_out_final)
        H_resize = np.array([
            [w_out_final/ w_out, 0, 0],
            [0, h_out_final/h_out, 0],
            [0, 0, 1]
            ])
        all_H[i] = np.matmul(H_resize, all_H[i])
    
    return all_H

def common_frame(all_H):

    all_H[0] = np.matmul(all_H[1], all_H[0])
    all_H[0] /= all_H[0][2,2]

    temp = np.linalg.inv(np.matmul(all_H[3], all_H[2]))
    temp /= temp[2,2]
    all_H.append(temp)

    all_H[3] = np.linalg.inv(all_H[2])
    all_H[3] /= all_H[4][2,2]

    all_H[2] = np.eye(3)

    return all_H

def main():

## OPENING FOUNTAIN IMAGES
    fntn1 = cv2.imread("1.jpg")
    fntn2 = cv2.imread("2.jpg")
    fntn3 = cv2.imread("3.jpg")
    fntn4 = cv2.imread("4.jpg")
    fntn5 = cv2.imread("5.jpg")
    fnt_images = [fntn1, fntn2, fntn3, fntn4, fntn5]

## OPENING MSEE IMAGES
    msee1 = cv2.imread("msee1.jpg")
    msee2 = cv2.imread("msee2.jpg")
    msee3 = cv2.imread("msee3.jpg")
    msee4 = cv2.imread("msee4.jpg")
    msee5 = cv2.imread("msee5.jpg")

    msee_images = [msee1, msee2, msee3, msee4, msee5]

## GETTING MATCHES FOR ADJACENT IMAGES 
    adj_matches = []
    adj_matches_msee = []

    for i in range(1,len(fnt_images)):
        adj_matches.append(sift(fnt_images[i-1], fnt_images[i],str(i)+"ipf.jpg"))
    for i in range(1,len(msee_images)):
        adj_matches_msee.append(sift(msee_images[i-1], msee_images[i],str(i)+"ipm.jpg"))

## GETTING ALL INLIERS AND OUTLIERS USING RANSAC AND ESTIMATING HOMOGRAPHY
    all_H = []
    all_est_H = []

    all_H_msee = []
    all_est_H_msee = []

    for i in range(len(adj_matches)):
        best_H, best_inliers_1, best_inliers_2, best_outliers_1, best_outliers_2, estH = ransac(adj_matches[i][0], adj_matches[i][1])
        visualizer(fnt_images[i], fnt_images[i+1], best_inliers_1, best_inliers_2, best_outliers_1, best_outliers_2, str(i)+str(i+1)+"fountain.jpg")
        final_H = LM_estimation(best_H, best_inliers_1, best_inliers_2)
        final_H /= final_H[2,2]

        all_H.append(final_H)
        all_est_H.append(estH)

    for i in range(len(adj_matches_msee)):
        best_H, best_inliers_1, best_inliers_2, best_outliers_1, best_outliers_2, estH = ransac(adj_matches_msee[i][0], adj_matches_msee[i][1])
        visualizer(msee_images[i], msee_images[i+1], best_inliers_1, best_inliers_2, best_outliers_1, best_outliers_2, str(i)+str(i+1)+"msee.jpg")
        final_H = LM_estimation(best_H, best_inliers_1, best_inliers_2)
        final_H /= final_H[2,2]

        all_H_msee.append(final_H)
        all_est_H_msee.append(estH)

## CONVERTING ALL IMAGES TO COMMON FRAME
    all_H = common_frame(all_H)
    all_H_msee = common_frame(all_H_msee)

    all_est_H = common_frame(all_est_H)
    all_est_H_msee = common_frame(all_est_H_msee)

## GETTING BOUNDS OF FOUNTAIN IMAGE AND CREATING PANORAMA
    x_max, x_min, y_max, y_min = get_bounds(fnt_images, all_H)
    blank_image = np.zeros((y_max-y_min, x_max-x_min, 3), np.uint8)

    for img, H in zip(fnt_images, all_H):
        blank_image = image_warp(blank_image, img, H, x_min, y_min)
    
    cv2.imwrite("outputfountain.jpg", blank_image)

## GETTING BOUNDS OF FOUNTAIN IMAGE AND CREATING PANORAMA
    x_max, x_min, y_max, y_min = get_bounds(msee_images, all_H_msee)
    blank_image = np.zeros((y_max-y_min, x_max-x_min, 3), np.uint8)

    for img, H in zip(msee_images, all_H_msee):
        blank_image = image_warp(blank_image, img, H, x_min, y_min)
    
    cv2.imwrite("outputmsee.jpg", blank_image)

## GETTING BOUNDS OF FOUNTAIN IMAGE AND CREATING PANORAMA USING RANSAC estimate
    x_max, x_min, y_max, y_min = get_bounds(fnt_images, all_est_H)
    blank_image = np.zeros((y_max-y_min, x_max-x_min, 3), np.uint8)

    for img, H in zip(fnt_images, all_est_H):
        blank_image = image_warp(blank_image, img, H, x_min, y_min)
    
    cv2.imwrite("outputfountainRSAC.jpg", blank_image)

## GETTING BOUNDS OF FOUNTAIN IMAGE AND CREATING PANORAMA USING RANSAC estimate
    x_max, x_min, y_max, y_min = get_bounds(msee_images, all_est_H_msee)
    blank_image = np.zeros((y_max-y_min, x_max-x_min, 3), np.uint8)

    for img, H in zip(msee_images, all_est_H_msee):
        blank_image = image_warp(blank_image, img, H, x_min, y_min)
    
    cv2.imwrite("outputmseeRSAC.jpg", blank_image)
    

if __name__ == "__main__":
    main()
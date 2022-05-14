import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy import interpolate
import matplotlib.pyplot as plt
import time

img_pth = '/Users/jasonyuan/Desktop/Test9.png'

# Add later to catch RankWarning
# import numpy as np
# import warnings
# x = [1]
# y = [2]
#
# with warnings.catch_warnings():
#     warnings.filterwarnings('error')
#     try:
#         coefficients = np.polyfit(x, y, 2)
#     except np.RankWarning:
#         print "not enought data"

def sort_by_cluster(labels,data):
    clusters = {}
    for n,pt in enumerate(data):
        if labels[n] not in clusters:
            clusters[labels[n]] = [pt]
        else:
            clusters[labels[n]].append(pt)

    return clusters

def lane_fitting(points):
    ''' Fitting lanes to a function with a variation on the sliding windows '''

    fit_points = []
    sorted_points_x = sorted(points,key=lambda x:x[1])
    sorted_points_y = sorted(points,key=lambda x:x[0])

    x_width = abs(sorted_points_x[-1][1] - sorted_points_x[0][1])
    y_width = abs(sorted_points_y[-1][0] - sorted_points_y[0][0])

    if (x_width < 15) or (y_width < 15):    # Hard-coded parameter, update maybe
        return [[],[]]

    # print(sorted_points)
    pts_added = 0
    total_pts = len(points)
    num_windows = 20

    slice = int(total_pts//num_windows)

    #TODO: Instead of just using arbitrary slices, use local cluster like centers
    # to choose the points to be included in the average
    for n in range(num_windows):
        start_idx = n*slice
        end_idx = min((n+1)*slice,total_pts)

        group = np.array(points[start_idx:end_idx])
        x_avg = np.mean(group,axis=0)[1]
        y_avg = np.mean(group,axis=0)[0]

        sigma_x = np.sqrt(np.sum(np.power(group[:,1]-x_avg,2))/group.shape[0])
        sigma_y = np.sqrt(np.sum(np.power(group[:,0]-y_avg,2))/group.shape[0])

        # print(sigma_x, sigma_y)

        if (sigma_x < 5) and (sigma_y < 5):
            fit_points.append([y_avg,x_avg])

    if len(fit_points) == 0:
        return [[],[]]

    fit_points = np.array(fit_points)
    # print(fit_points)

    x = fit_points[:,1]
    y = fit_points[:,0]
    tck,u = interpolate.splprep([x,y],k=3,s=32)
    # print(tck)
    out = interpolate.splev(u,tck)
    return out

if __name__ == "__main__":
    # start = time.perf_counter()

    input = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
    input_norm = input/255

    rows = np.where(input_norm==1)[0].reshape(-1,1)
    cols = np.where(input_norm==1)[1].reshape(-1,1)
    coords = np.concatenate((rows,cols),axis=1)     # (y,x) points

    clustering = DBSCAN(eps=15, min_samples=30).fit(coords)
    labels = clustering.labels_

    for i,pt in enumerate(coords):
        if labels[i] == 0:
            color = 'g'
        elif labels[i] == 1:
            color = 'r'
        elif labels[i] == 2:
            color = 'y'
        elif labels[i] == 3:
            color = 'b'
        elif labels[i] == 4:
            color = 'm'
        elif labels[i] == 5:
            color = 'c'
        elif labels[i] == -1:
            color = 'k'

        plt.scatter(pt[1],pt[0],c=color)

    clusters = sort_by_cluster(labels,coords)

    for label,pts in clusters.items():
        if label == -1:
            continue
        else:
            # coefficients = lane_fitting(pts,15)
            # poly = np.poly1d(coefficients)
            # min_x = pts[0][1]
            # max_x = pts[len(pts)-1][1]
            #
            # xrange = np.linspace(min_x,max_x,endpoint=True)
            # plt.plot(xrange,poly(xrange),'-',c='k')
            out = lane_fitting(pts)
            # print(out[0])
            # print(out[1])
            plt.plot(out[0],out[1],c='k')
            plt.gca().invert_yaxis()

    plt.show()

    # end = time.perf_counter()
    # print(end-start)

import cv2

def predict_people(img,model):
    results = model.predict(img, verbose = False)
    count = 0
    indexes = []
    for ind,i in enumerate(results[0].boxes.cls):
        if i == 0:
            indexes.append(ind)
    results[0].boxes.xyxy
    l = []
    for ind,i in enumerate(results[0].boxes.xyxy.numpy()):
        if ind in indexes:
            l.append(i)
    return l

def draw(bb,img):
    image = img.copy()
    return cv2.rectangle(image, (bb[0],bb[1]), (bb[2],bb[3]), (0,0,255), 2)


def crop_roi(bb, img):
    # drawn_img = draw(bb,img)
    x1,y1,x2,y2 = bb
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def find_centroids(boxes):
    centroids = []
    for l in boxes:
        centroid_x = (l[0] + l[2])//2
        centroid_y = (l[1] + l[3])//2
        centroids.append((centroid_x, centroid_y))
    return centroids
        

def check(point, roi):
    point_x = point[0]
    point_y = point[1]

    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]
    
    if((point_x > x1 and point_x < x2) and (point_y > y1 and point_y < y2) ):
        return True
    return False
        

def plot_people(l,img):
    image = img.copy()
    for i in l:
        start_point = (int(i[0]), int(i[1])) 
        end_point = (int(i[2]), int(i[3])) 
        color = (255, 0, 0) 
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image
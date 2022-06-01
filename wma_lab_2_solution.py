import numpy as np
import cv2

def count_magnetic_balls(file_name):
    kulki = cv2.imread(file_name)
    
    kulki_small = cv2.pyrDown(kulki)
    cv2.imshow("Resized input image", kulki_small)
    cv2.waitKey()
    kulki_gray = cv2.cvtColor(kulki_small, cv2.COLOR_BGR2GRAY)
    kulki_thresh = cv2.adaptiveThreshold(kulki_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,-35)
    cv2.imshow("Thresholded imgage", kulki_thresh)
    cv2.waitKey()
    
    kulki_opened = cv2.morphologyEx(kulki_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imshow("Open", kulki_opened)
    cv2.waitKey()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kulki_closed = cv2.morphologyEx(kulki_opened, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed", kulki_closed)
    cv2.waitKey()
    
    conts, hier = cv2.findContours(kulki_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    kulki_draw = kulki_small.copy()
    kulki_draw = cv2.drawContours(kulki_draw, conts, -1, (255, 0, 255), 2)
    cv2.imshow("All contours", kulki_draw)
    cv2.waitKey()
    
    good_conts = []
    for cont in conts:
      area = cv2.contourArea(cont)
      if area > 50:
        good_conts.append(cont)
    
    kulki_draw_good = kulki_small.copy()
    kulki_draw_good = cv2.drawContours(kulki_draw_good, good_conts, -1, (255, 0, 0), 2)
    cv2.imshow("Good contours", kulki_draw_good)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return len(good_conts)


if __name__=='__main__':
    print(count_magnetic_balls('./data/wm_lab_2/1.jpg'))

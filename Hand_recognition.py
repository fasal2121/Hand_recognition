import cv2
import numpy as np
import math

IMAGE_MIN = 100
IMAGE_MAX = 350
OBJ_THRESHOLD = 100
TOP_LEFT = (350, 50) 
BOTTOM_RIGHT = (600, 300)

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("trackbars")
cv2.createTrackbar("Lower-H", "trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower-S", "trackbars", 42, 255, nothing)
cv2.createTrackbar("Lower-V", "trackbars", 92, 255, nothing)
cv2.createTrackbar("Upper-H", "trackbars", 22, 179, nothing)
cv2.createTrackbar("Upper-S", "trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "trackbars", 255, 255, nothing)

while(1):
        
    try: 
        # Find finger (skin) color using trackbars
        low_h = cv2.getTrackbarPos("Lower-H", "trackbars")
        low_s = cv2.getTrackbarPos("Lower-S", "trackbars")
        low_v = cv2.getTrackbarPos("Lower-V", "trackbars")
        up_h = cv2.getTrackbarPos("Upper-H", "trackbars")
        up_s = cv2.getTrackbarPos("Upper-S", "trackbars")
        up_v = cv2.getTrackbarPos("Upper-V", "trackbars")

         
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
    # Define the Region Of Interest (ROI) window 
         
        cv2.rectangle(frame, (TOP_LEFT[0], TOP_LEFT[1]), (BOTTOM_RIGHT[0], BOTTOM_RIGHT[1]), (255,0,0), 0)   # région d'intéret

    #define region of interest
        roi = frame[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # les pixels de la zone d'interet sont convertis en hsv
        
    # Create a range for the colors (skin color)
        # lower_skin = np.array([0, 48, 80], dtype = "uint8") # la nuit
        # lower_skin = np.array([0,48,75], dtype = "uint8") 
        # lower_skin = np.array([0,42,92],np.uint8) # night value
        lower_skin = np.array([low_h, low_s, low_v])
        # upper_skin = np.array([20, 150, 255], dtype = "uint8")
        # upper_skin = np.array([15, 200, 255], dtype = "uint8")
        # upper_skin = np.array([22, 255, 255], dtype = "uint8") # pour la nuit 
        upper_skin = np.array([up_h, up_s, up_v])

    # skin colour image
        mask = cv2.inRange(hsv, lower_skin, upper_skin) #masque

    # blured = cv2.blur(mask, (2, 2))
        blured = cv2.GaussianBlur(mask, (5,5), 0)

        ret, thresh = cv2.threshold(blured, OBJ_THRESHOLD , 255, cv2.THRESH_BINARY)
        
        
        
    #find contours
        contours,hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #on trouve les contours qui se trouvent dans la region d'interet
    
    #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x)) #trouve le contour maximal qui est la main
        # cv2.drawContours(roi, cnt, -1, (255,0,0), 3)

    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt, True) #degré d'écart par rapport à la forme exacte, True car contour fermé
    # epsilon = 0.001*cv2.arcLength(cnt, True) #degré d'écart par rapport à la forme exacte, True car contour fermé
        approx= cv2.approxPolyDP(cnt, epsilon, True) #simplifie le contour, True car on veut un contour fermé
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt) # fait l'enveloppe qui entoure la main 
        
     #define area of hull and area of hand
        areaHull = cv2.contourArea(hull) #aire de l'enveloppe
        areaHand = cv2.contourArea(cnt) #aire de la main
      
    #find the percentage of area not covered by hand in convex hull
        areaUncovered=((areaHull-areaHand)/areaHand)*100
    
    #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull) #defaut entre les contours et l'enveloppe
        #c'est les zones qui ne sont pas couvertes par l'enveloppe
        # print(defects)

        
    # l = number of defects
        l=0
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start_pts  = tuple(approx[s][0])
            end_pts = tuple(approx[e][0])
            far_pts = tuple(approx[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end_pts[0] - start_pts[0])**2 + (end_pts[1] - start_pts[1])**2)
            b = math.sqrt((far_pts[0] - start_pts[0])**2 + (far_pts[1] - start_pts[1])**2)
            c = math.sqrt((end_pts[0] - far_pts[0])**2 + (end_pts[1] - far_pts[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull
            # and d>30:
            if angle <= 90 and d>30:
                l+= 1
                cv2.circle(roi, far_pts, 3, [255,0,0], -1)

            #draw lines around hand
            cv2.line(roi,start_pts, end_pts, [0,255,0], 2)
            
        l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            # print(areaUncovered)
            if areaHand<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if areaUncovered<12:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif areaUncovered<20:
                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif areaUncovered<35:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame,'L',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            # else:   
            #     cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif l==2:
            # print(areaUncovered)
            # cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            if areaUncovered<65:
                cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame,'L',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
            #   print(areaUncovered)
              if areaUncovered<32:
                    cv2.putText(frame,'OK',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else :
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            #   else:
                    # cv2.putText(frame,'L',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #show the windows
        cv2.imshow('mask',thresh)
        cv2.imshow('frame',frame)
    except :
        pass
        
    k = cv2.waitKey(25) & 0xFF
    if  k  == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    





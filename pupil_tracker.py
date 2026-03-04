import cv2
import numpy as np

cap = cv2.VideoCapture(0) #Turning on the Camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #Sets view width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #Sets view height
cap.set(cv2.CAP_PROP_FPS, 60) #Attempts 60 frames per second, but I find that this isn't accurate and is limited by hardware (Need to see if this is true)

print("Detecting Pupils...")

while True:
    ret, frame = cap.read() #Takes picture of a saved frame for detecting pupils
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Grayscales the frame
    blurred = cv2.medianBlur(gray, 5) #Makes the frame fuzzy. Noise, which is typically Guassian, gets hidden (I am trial-and-erroring the intensity of this blur)
    
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV) #Inverts shade (dark turns light, light turns dark)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Traces the edges of the detected shape (pupil)

    for cnt in contours: #The result is arbitrary because I have not implemented camera calibration
        area = cv2.contourArea(cnt) #Calculates area of the shape
        perimeter = cv2.arcLength(cnt, True) #Calculate the perimeter of the shape
        
        if perimeter == 0:
            continue #I added this so it ignores any shape that is too small. I am trial-and-erroring how big or small I should make the threshold for a valid shape.
            
        circularity = (4 * np.pi * area) / (perimeter**2) #If the answer to this is close to 1, the shape is close to a perfect circle (This isn't perfectly accurate right now)

        if 400 < area < 15000 and circularity > 0.75: #Currently trial-and-erroring this as well! Ignore the specific values
            (x, y), radius = cv2.minEnclosingCircle(cnt) #Finds the radius and center of the shape
            center = (int(x), int(y)) #Turns the values to a whole number
            radius = int(radius) 

            cv2.circle(frame, center, radius, (0, 255, 0), 2) #Visual green ring around the shape. 
            
            diameter = radius * 2 #Displays the arbitary size on screen (This is currently overlapped right now, will have to change this)
            cv2.putText(frame, f"Diam: {diameter}px", (center[0] + 15, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Circ: {round(circularity, 2)}", (center[0] + 15, center[1] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("OMEN High-Speed Pupil Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #Waits for the highest possible refresh rate
        break

cap.release()
cv2.destroyAllWindows()
import  cv2

#creating a video capture
Video_Capture=cv2.VideoCapture(0)

XML_Path = 'E:\Shavin Perera\AI car recognition\haarcascade_car.xml'
#Load the car model
Car_Cascade=cv2.CascadeClassifier(XML_Path)

while True:
    #read frames from cam
    ret,frame=Video_Capture.read()

    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the cars
    cars = Car_Cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(25,25))

    #number of cars
    Number_of_Cars=len(cars)

    #draw boundry around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the count on the frame
    cv2.putText(frame, f'Cars: {Number_of_Cars}', (10, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Car Counting', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
Video_Capture.release()
cv2.destroyAllWindows()
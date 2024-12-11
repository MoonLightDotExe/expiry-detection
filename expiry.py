from ultralytics import YOLO
import cv2
from google.cloud import vision
import io
import os
import time
import re
import logging
import pymongo
from datetime import datetime

# Set up Google Cloud Vision client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # Update with your service account file path
client = vision.ImageAnnotatorClient()

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load the YOLO model
model = YOLO("best.pt")  # Update the path to your YOLO model

# Initialize webcam
cap = cv2.VideoCapture(2)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Timer for Google Vision API usage
last_api_call_time = time.time()
api_call_interval = 5  # seconds

# Regular expressions for date formats
date_patterns = [
    r'\b\d{1,2}/\d{1,2}/\d{2}\b',  # Matches dates in the format DD/MM/YY (allow single digit days and months)
    r'\b\d{2}/[A-Za-z]{3}/\d{2}\b'  # Matches dates in the format DD/MMM/YY (existing format for month abbreviation)
]

def extract_dates(text):
    """Extract dates from text using regex patterns."""
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates

# Function to filter dates that occur after the current date
def filter_dates_after_current(dates):
    """Filters and returns dates that occur after the current date."""
    # Get the current date
    current_date = datetime.now()

    filtered_dates = []
    for date_str in dates:
        try:
            # Check the format of the date
            if len(date_str.split("/")[1]) == 3:  # If month is in abbreviation (e.g., OCT, FEB)
                date_obj = datetime.strptime(date_str, "%d/%b/%y")
            else:  # If month and day are single digits (e.g., 10/1/25)
                date_obj = datetime.strptime(date_str, "%d/%m/%y")
            
            # Compare the date with the current date
            if date_obj > current_date:
                filtered_dates.append(date_str)
        except ValueError:
            # If the date parsing fails, skip that date
            continue
    
    return filtered_dates

# MongoDB connection setup
client_mongo = pymongo.MongoClient("")  # Adjust the MongoDB URI if necessary
db = client_mongo["date_db"]  # Create or use the 'date_db' database
collection = db["dates"]  # Create or use the 'dates' collection

# Variable to store the last detected text
last_detected_text = ""

# Create a named window
cv2.namedWindow("YOLO Detection with Google Vision OCR", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Detection with Google Vision OCR", cv2.WND_PROP_TOPMOST, 1)
cv2.moveWindow("YOLO Detection with Google Vision OCR", 100, 100)  # Move the window to a specific position

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO detection on the frame
    results = model(frame)

    for result in results:
        detections = result.boxes.xyxy
        confidence = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(detections.numpy(), confidence.numpy(), classes.numpy()):
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = float(conf)
            cls = int(cls)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract ROI for OCR
            roi = frame[y1:y2, x1:x2]

            # Check if it's time to call the Google Vision API
            current_time = time.time()
            if current_time - last_api_call_time >= api_call_interval:
                # Convert ROI to bytes for Google Vision API
                _, encoded_image = cv2.imencode('.jpg', roi)
                content = encoded_image.tobytes()

                # Perform OCR on the ROI using Google Vision API
                image = vision.Image(content=content)
                response = client.text_detection(image=image)
                texts = response.text_annotations

                if texts:
                    detected_text = texts[0].description.strip()
                    if detected_text != last_detected_text:
                        print(detected_text)

                        # Extract dates from the detected text
                        extracted_dates = extract_dates(detected_text)
                        print(f"Extracted Dates: {extracted_dates}")

                        # Filter the dates that are after the current date
                        filtered_dates = filter_dates_after_current(extracted_dates)
                        print(f"Filtered Dates (after current date): {filtered_dates}")

                        # Insert filtered dates into MongoDB
                        if filtered_dates:
                            collection.insert_many([{"date": date} for date in filtered_dates])
                            print("Dates inserted into MongoDB successfully.")

                        # Optionally display OCR result on the frame
                        for date in filtered_dates:
                            cv2.putText(frame, date, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Update the last detected text
                        last_detected_text = detected_text

                # Update the last API call time
                last_api_call_time = current_time

    # Display the frame with detections and OCR text
    cv2.imshow("YOLO Detection with Google Vision OCR", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
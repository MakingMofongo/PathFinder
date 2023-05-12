import cv2
import yolo #loads yolo model

# Initialize the tracker
def create_tracker():
    tracker = cv2.TrackerKCF_create()
    return tracker

# Get the bounding box of the selected object
def get_bounding_box(frame, objects, locations):
    print("Select the object you want to track:")
    for i, obj in enumerate(objects):
        print(f"{i + 1}. {obj}")

    choice = int(input("Enter the number of the object: ")) - 1
    xmin, ymin, xmax, ymax = [int(v) for v in locations[choice]]
    return (xmin, ymin, xmax - xmin, ymax - ymin)

# Main function
def main():
    
    # Initialize the tracker
    tracker = create_tracker()
    # Load the video
    video = cv2.VideoCapture(0)

    # Get the first frame
    ret, frame = video.read()

    # Detect objects in the first frame
    result= yolo.inference(frame, Loop=False)
    objects = result[0]
    locations = result[1]

    # Get the bounding box of the selected object
    bbox = get_bounding_box(frame, objects,locations)

    tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ret, frame = video.read()

        # Update the tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw the bounding box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

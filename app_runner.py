# Open the video.
import cv2
from goog_mods import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing
#from my_mods import my_visualizer#*#, RepetitionCounter
from my_modules import my_visualizer, RepetitionCounter #*#
from mediapipe.python.solutions import pose as mp_pose, drawing_utils as mp_drawing
import os
import sys
import numpy as np

video_cap = cv2.VideoCapture(0)

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
pose_samples_folder = 'CSVs_out_all'

# Set class of pose to be classified.
#class_name= 'squats_down' #'squats_down' 'dips_down'

# Set list of exercises and reps required for a round
exercise_dict = {"pushups_down":3, "squats_down":2}

# Initialize tracker.
# pose_tracker = mp_pose.Pose(upper_body_only=False)
pose_tracker = mp_pose.Pose()

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Check that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    #class_name=class_name, #*#
    exercise_dict = exercise_dict, #*#
    enter_threshold=6,
    exit_threshold=4)


# Run classification on a webstream.

if not video_cap.isOpened():
    print("Cannot open camera")
    exit()

# Continuously capture images from the camera and run inference
while video_cap.isOpened():
    success, input_frame = video_cap.read()
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )
    #input_frame=np.uint8(input_frame)
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    result = pose_tracker.process(image=input_frame)
    pose_landmarks = result.pose_landmarks

    # Draw pose prediction.
    if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=input_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

    if pose_landmarks is not None:
        # Get landmarks.
        frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                 for lmk in pose_landmarks.landmark], dtype=np.float32)
        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks)

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(pose_classification)

        # Count repetitions.
        repetitions_count = repetition_counter(pose_classification_filtered)
        class_name = repetition_counter.class_name
    else:
        # No pose => no classification on current frame.
        pose_classification = None

        # Still add empty classification to the filter to maintaing correct
        # smoothing for future frames.
        pose_classification_filtered = pose_classification_filter(dict())
        pose_classification_filtered = None

        # Don't update the counter presuming that person is 'frozen'. Just
        # take the latest repetitions count.
        repetitions_count = repetition_counter.n_repeats
        class_name = repetition_counter.class_name

    num_rounds = repetition_counter.n_rounds
    input_frame  = my_visualizer(input_frame, repetitions_count, class_name, num_rounds)
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow('Video', input_frame)

video_cap.release()
cv2.destroyAllWindows()

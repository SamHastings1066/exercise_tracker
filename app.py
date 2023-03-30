import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
from mediapipe.python.solutions import pose as mp_pose, drawing_utils as mp_drawing
from my_pose_module import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing, RepetitionCounter
from my_modules import my_visualizer

# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
pose_samples_folder = 'CSVs_out_all'

# Set class of pose to be classified.
class_name= 'squats_down' #'squats_down' 'dips_down'

# Set list of exercises and reps required for a round
exercise_dict = {"squats_down" : 2, "dips_down" : 2}

# Initialize pose tracker.
pose_tracker = mp_pose.Pose(upper_body_only=False)

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,
    exercise_dict = exercise_dict,
    enter_threshold=6,
    exit_threshold=4)

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = pose_tracker.process(image=img)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = img.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                    for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)

            # Count repetitions.
            repetitions_count, current_exercise = repetition_counter(pose_classification_filtered) #needs to output current_exercise and num_rounds
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
            current_exercise = repetition_counter.class_name

        #current_exercise = "squats_down"
        output_frame  = my_visualizer(output_frame, repetitions_count, current_exercise)

        return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

webrtc_streamer(key="web_cam_classifier",
                video_processor_factory = VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                #async_processing=True,
                )

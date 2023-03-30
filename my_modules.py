import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

def my_visualizer(frame,
                  repetitions_count,
                  current_exercise,
                  num_rounds,
                  counter_font_size=0.1,
                  counter_font_color='red',
                  counter_location_x=0.5,
                  counter_location_y=0.05):
    # Output frame with counter.
    output_img = Image.fromarray(frame) # converts np array to PIL image object

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    exercise_names = {"squats_down" : "Squats: ",
                      "dips_down": "Dips: ",
                      "pushups_down" : "Push-ups: ",
                      "chinups_up": "Chin-ups: "}
    output_text = """
    {}
    Rounds: {}
    """.format(exercise_names[current_exercise] + str(repetitions_count), num_rounds)
    # Draw the count.
    output_img_draw = ImageDraw.Draw(output_img)
    font_size = int(output_height * counter_font_size)
    #font_request = requests.get(counter_font_path, allow_redirects=True)
    counter_font = ImageFont.truetype(r'/Users/samhastings/Documents/coding/apps/pose_detector/MediaPipe/open-sans/OpenSans-Regular.ttf', size=font_size)
    output_img_draw.text((output_width * counter_location_x,
                          output_height * counter_location_y),
                          output_text,
                          #exercise_names[current_exercise] + str(repetitions_count),
                          font=counter_font,
                          fill=counter_font_color)
    return np.asarray(output_img)


class RepetitionCounter(object):
  """Counts number of repetitions of given target pose class."""

  def __init__(self, exercise_dict, enter_threshold=6, exit_threshold=4):

    # If pose counter passes given threshold, then we enter the pose.
    self._enter_threshold = enter_threshold
    self._exit_threshold = exit_threshold

    # Either we are in given pose or not.
    self._pose_entered = False

    # Number of times we exited the pose.
    self._n_repeats = 0

    # Keeps track of how many exercises we have done we are doing
    self._n_exercises = 0

    # Keep track of how many rounds have been completed
    self._n_rounds = 0

    # List of exercises and corresponding reps required for one round
    self._exercises = list(exercise_dict.keys())
    self._reps = list(exercise_dict.values())

    self._class_name = self._exercises[self._n_exercises]

  @property
  def n_repeats(self):
    return self._n_repeats

  @property
  def n_exercises(self):
    return self._n_exercises

  @property
  def n_rounds(self):
    return self._n_rounds

  @property
  def class_name(self):
    return self._class_name

  def __call__(self, pose_classification):
    """Counts number of repetitions happend until given frame.

    We use two thresholds. First you need to go above the higher one to enter
    the pose, and then you need to go below the lower one to exit it. Difference
    between the thresholds makes it stable to prediction jittering (which will
    cause wrong counts in case of having only one threshold).

    Args:
      pose_classification: Pose classification dictionary on current frame.
        Sample:
          {
            'pushups_down': 8.3,
            'pushups_up': 1.7,
          }

    Returns:
      Integer counter of repetitions.
    """
    # Get pose confidence.
    pose_confidence = 0.0
    if self._class_name in pose_classification:
      pose_confidence = pose_classification[self._class_name]

    # On the very first frame or if we were out of the pose, just check if we
    # entered it on this frame and update the state.
    if not self._pose_entered:
      self._pose_entered = pose_confidence > self._enter_threshold
      return self._n_repeats

    # If we were in the pose and are exiting it, then increase the counter and
    # update the state.
    if pose_confidence < self._exit_threshold:
      self._n_repeats += 1
      self._pose_entered = False

    # When num repeats reaches the rep limit for this exercise, reset n_repeats
    # increment n_exercises by 1, update class name to reflect new exercise
    if self._n_repeats == self._reps[self._n_exercises % len(self._exercises)]:
      self._n_repeats = 0
      self._n_exercises += 1
      self._class_name = self._exercises[self._n_exercises % len(self._exercises)]
      if self._n_exercises % len(self._exercises) == 0:
        self._n_rounds += 1

    return self._n_repeats #, self._class_name


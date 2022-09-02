# Lane Follower with Reinforcement Learning

A lane-following vehicle is trained with reinforcement learning to
achieve minimal self-driving capabilities along a track. Currently it
only recognizes straight roads and left and right turns.



### How to run

1. Build and source the catkin package

  ```
  catkin_make
  source devel/setup.bash
  ```

2. Run the environment and agent

  ```
  roslaunch driving_track run.launch
  python src/qlearning.py # or
  python src/ddpg.py
  ```

# ME467 Robotics Project 3  
By Caleb Hottes

## Description

This project finds the foreward and inverse kinematics of the NEXCO MiniBoT 6 degree of freedom robot arm and the kinematic jacobian matrix. 
Additionally some functions and a script were written to make an animation using the foreward and inverse kinematics of the robot moving. 

## Installation

To install the required dependencies, run:

> pip install mujoco spatialmath-python numpy nbconvert scipy

## Report Export

To export the Jupyter notebook report to HTML, run:

    jupyter nbconvert --to html Answers.ipynb

## Project Structure

```
ME467_ROBOTICS_PROJECT3/
├── Instructions/  
│   ├── Instructions.pdf  
│   └── MiniBOT_Robot.pdf  
├── kinematics/  
│   ├── __init__.py  
│   ├── DHKinematics.py  
│   ├── minibotinversekinematics.py  
│   └── trajectory.py  
├── mujoco_files/  
|   |── <various CAD files for the model>
|   |── reference.jpg
|   └── robot_model.xml  
├── resources/  
│   ├── Actual_robot.png  
│   ├── Animation.gif  
│   ├── Animation.mp4  
│   ├── articulatedReference.png  
│   ├── coordDiagram.png  
│   ├── Coords.SLDPRT  
│   └── inverseKinematicsDiagram.png  
├── Answers.ipynb  
├── AnswersFatForExport.ipynb  
├── geometric_inverse_tester.py  
├── testbed.py  
├── Hottes_Caleb_Final_Report.html
├── Hottes_Caleb_Final_Report.pdf  
├── LICENSE  
└── README.md  
```

### Instructions/
- **Instructions.pdf**: Project assignment and guidelines.  
- **MiniBOT_Robot.pdf**: Robot specifications and diagrams.

### kinematics/
- **__init__.py**: Python package initializer.  
- **DHKinematics.py**: Implements functions general to a DH robot.  
- **minibotinversekinematics.py**: Inverse kinematics routines for the minibot.  
- **trajectory.py**: Trajectory generation functions.

### mujoco_files/
- **reference.jpg**: A diagram of the robot
- **robot_model.xml**: The mujoco model of the robot

### resources/
- **Actual_robot.png**: Photo of the physical robot.  
- **Animation.gif** / **Animation.mp4**: Movement animation.  
- **articulatedReference.png**: Articulated model reference diagram.  
- **coordDiagram.png**: Coordinate-frame diagram.  
- **Coords.SLDPRT**: SolidWorks part file for the robot.  
- **inverseKinematicsDiagram.png**: Inverse kinematics setup diagram.

### Notebooks & Scripts
- **Answers.ipynb**: Main Jupyter notebook with solutions and analysis.  
- **AnswersFatForExport.ipynb**: Expanded notebook for detailed HTML export.  
- **geometric_inverse_tester.py**: Tests geometric inverse kinematics implementations.  
- **testbed.py**: A file to just try things out in

### Reports
- **Hottes_Caleb_Final_Report.html**: Final report (html).  
- **Hottes_Caleb_Final_Report.pdf**: Final report (PDF).

### Other
- **LICENSE**: Project license.  
- **README.md**: This file.

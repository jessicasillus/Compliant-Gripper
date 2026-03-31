# Design, Simulation, and Manufacturing of a Tendon Driven Compliant Robotic Gripper With 3D Printed Force Sensing

This repository contains all design, firmware, simulation, and data acquisition files for a tendon-driven, three-finger compliant robotic gripper with integrated capacitive force sensing, developed as part of a master's thesis at The Ohio State University. The gripper uses TPU fin-ray finger structures, pseudo-rigid body model (PRBM) kinematics, and is designed for integration with a UR5e robotic arm for space-relevant manipulation tasks.

---

## Repository Structure

```
Compliant-Gripper/
├── Gripper Assembly/
├── PCB Design/
├── ESP32 Programs/
│   └── MPR121_Editor/
├── DAQ Tools and Post Processing Tools/
├── Mujoco Simulation Scripts/
└── Python Simulation Scripts/
```

---

## CAD Files

**Folder:** [`Gripper Assembly/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/Gripper%20Assembly)

This folder contains all SolidWorks part and assembly files (`.SLDPRT` / `.SLDASM`) for the compliant gripper system. The parts cover the full gripper assembly including the tendon routing pulley, actuation gears (left and right), the enclosure (top box, bottom box, and cover), a tendon component, the TPU fin-ray finger geometry, and an adapter plate for mounting to the UR5e flange. The two `.SLDASM` files define the full gripper assembly and the ST3215 servo motor sub-assembly.

| File | Description |
|---|---|
| [`45degreefinger.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/45degreefinger.SLDPRT) | TPU compliant fin-ray finger part at 45° mounting angle |
| [`Assem2_ref_002.SLDASM`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/Assem2_ref_002.SLDASM) | Top-level gripper assembly file |
| [`Bottom_Box_2_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/Bottom_Box_2_002.SLDPRT) | Lower enclosure body (variant 2) |
| [`Bottom_Box_3_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/Bottom_Box_3_002.SLDPRT) | Lower enclosure body (variant 3) |
| [`LeftGear_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/LeftGear_002.SLDPRT) | Left actuation gear |
| [`Pulley_New_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/Pulley_New_002.SLDPRT) | Tendon routing pulley |
| [`RightGear_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/RightGear_002.SLDPRT) | Right actuation gear |
| [`STS3215_03a v1_002.SLDASM`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/STS3215_03a%20v1_002.SLDASM) | ST3215 servo motor sub-assembly |
| [`Top_Box_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/Top_Box_002.SLDPRT) | Upper enclosure body |
| [`box_cover_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/box_cover_002.SLDPRT) | Enclosure cover plate |
| [`tendon_002.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/tendon_002.SLDPRT) | Tendon routing component |
| [`ur5eadapterplate.SLDPRT`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Gripper%20Assembly/ur5eadapterplate.SLDPRT) | Mounting adapter plate for the UR5e tool flange |

> SolidWorks 2022 or later is recommended to open these files.

---

## PCB Files

### KiCAD Design Files

**Folder:** [`PCB Design/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/PCB%20Design)

This folder contains KiCAD project files for two versions of the capacitive force sensing PCB (`PCB_V2` and `PCB_V3`). Each version includes the full project file set: the PCB layout (`.kicad_pcb`), schematic (`.kicad_sch`), project settings (`.kicad_pro`), and local project rules (`.kicad_prl`). A custom symbol library (`New_Library.kicad_sym`) and footprint library (`Library.pretty/`) are also included for any non-standard components used in the design.

| File | Description |
|---|---|
| [`Library.pretty/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/PCB%20Design/Library.pretty) | Custom KiCAD footprint library |
| [`New_Library.kicad_sym`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/New_Library.kicad_sym) | Custom KiCAD symbol library |
| [`PCB_V2.kicad_pcb`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V2.kicad_pcb) | PCB layout — Version 2 |
| [`PCB_V2.kicad_sch`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V2.kicad_sch) | Schematic — Version 2 |
| [`PCB_V2.kicad_pro`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V2.kicad_pro) | Project file — Version 2 |
| [`PCB_V2.kicad_prl`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V2.kicad_prl) | Local project rules — Version 2 |
| [`PCB_V3.kicad_pcb`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V3.kicad_pcb) | PCB layout — Version 3 |
| [`PCB_V3.kicad_sch`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V3.kicad_sch) | Schematic — Version 3 |
| [`PCB_V3.kicad_pro`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V3.kicad_pro) | Project file — Version 3 |
| [`PCB_V3.kicad_prl`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/PCB%20Design/PCB_V3.kicad_prl) | Local project rules — Version 3 |

> KiCAD 7.0 or later is required to open these files.

### ESP32 Firmware

**Folder:** [`ESP32 Programs/MPR121_Editor/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/ESP32%20Programs/MPR121_Editor)

This folder contains the Arduino firmware for the ESP32 microcontroller used to interface with the MPR121 capacitive touch/force sensing IC.

| File | Description |
|---|---|
| [`MPR121_Editor.ino`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/ESP32%20Programs/MPR121_Editor/MPR121_Editor.ino) | Arduino sketch for configuring and reading the MPR121 capacitive sensor over I2C from the ESP32 |

> Upload using the Arduino IDE with the ESP32 board package installed. The `Adafruit_MPR121` library is required.

---

## DAQ Tools and Post Processors

**Folder:** [`DAQ Tools and Post Processing Tools/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/DAQ%20Tools%20and%20Post%20Processing%20Tools)

This folder contains Python tools for live data acquisition from the gripper sensors and for post-processing and calibrating the collected data.

| File | Description |
|---|---|
| [`DAQ_Tool.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/DAQ%20Tools%20and%20Post%20Processing%20Tools/DAQ_Tool.py) | Live data acquisition script. Reads sensor output from the ESP32 over serial, timestamps and logs capacitance and force data during gripper operation. |
| [`Post_Processor_Calibrator_Tool.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/DAQ%20Tools%20and%20Post%20Processing%20Tools/Post_Processor_Calibrator_Tool.py) | Post-processing and calibration script. Applies calibration curves to raw sensor data, computes force estimates, and exports processed results for analysis. |

**Dependencies:** `pyserial`, `numpy`, `matplotlib`

---

## Simulation

### MuJoCo Simulation Scripts

**Folder:** [`Mujoco Simulation Scripts/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/Mujoco%20Simulation%20Scripts)

This folder contains the MuJoCo environment definition files and the Python control scripts for simulating the compliant gripper both in isolation and integrated with the UR5e robotic arm. The simulation supports pick-and-place task execution with a full 6-DOF inverse kinematics solver and an orientation-aware state machine.

| File | Description |
|---|---|
| [`gripper_3finger.xml`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Mujoco%20Simulation%20Scripts/gripper_3finger.xml) | MuJoCo XML model of the three-finger compliant gripper in isolation, including TPU finger geometry, tendon actuators, and contact properties. |
| [`ur5e_gripper.xml`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Mujoco%20Simulation%20Scripts/ur5e_gripper.xml) | MuJoCo XML model of the full UR5e arm with the compliant gripper mounted at the tool flange. Includes the scene, pedestal, and grasping target objects. |
| [`Gripper_Mujoco_Sim.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Mujoco%20Simulation%20Scripts/Gripper_Mujoco_Sim.py) | Standalone simulation script for the isolated three-finger gripper. Drives tendon actuators and visualizes finger deflection under load. |
| [`ur5e_gripper_sim.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Mujoco%20Simulation%20Scripts/ur5e_gripper_sim.py) | Full UR5e + gripper simulation script. Implements the pick-and-place state machine, 6-DOF IK with orientation error terms, waypoint sequencing, and optional GUI controls via tkinter. |

**Dependencies:** `mujoco`, `numpy`, `scipy`, `tkinter`

### Python Simulation Scripts

**Folder:** [`Python Simulation Scripts/`](https://github.com/jessicasillus/Compliant-Gripper/tree/main/Python%20Simulation%20Scripts)

This folder contains Python scripts for kinematic modeling and visualization of finger deflection and grasping geometry, independent of the MuJoCo physics engine.

| File | Description |
|---|---|
| [`Finger_Sim_Objects.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Python%20Simulation%20Scripts/Finger_Sim_Objects.py) | Defines the finger geometry and PRBM kinematic model as Python objects. Computes joint angles and tip positions as a function of tendon displacement using the pseudo-rigid body method. |
| [`grip_objects.py`](https://github.com/jessicasillus/Compliant-Gripper/blob/main/Python%20Simulation%20Scripts/grip_objects.py) | Defines grasping target objects and contact geometry for use in kinematic grasp analysis and visualization. |

**Dependencies:** `numpy`, `matplotlib`

---

## License

This project was developed as part of a master's thesis at The Ohio State University. Feel free to use the files and adapt on this research! :) Email with any questions.

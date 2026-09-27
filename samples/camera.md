# Camera Scene Description (`samples/camera.md`)

## Overview
Fixed overhead CCTV camera monitoring a four-way urban signalized intersection (`1280x720`, 25–30 fps, static viewpoint).

## Road Layout & Lanes
1. **Main Horizontal Arterial (`y: 260..640`)**:
   - **Eastbound (left-to-right)**: Lower lanes (`y: 430..640`), vehicles enter from the left (`x=0`), cross the west stop-line (`x=220`), traverse the intersection box (`x=220..1000`), and exit right (`x=1280`).
   - **Westbound (right-to-left)**: Upper-middle lanes (`y: 280..430`), vehicles enter from the right (`x=1280`), stop behind the east stop-line (`x=1020`), and proceed west (`x=0`).
2. **Cross Street / North-South Approach (`x: 350..950, y: 0..320`)**:
   - **Southbound (top-to-bottom)**: Vehicles enter from the top-right approach (`x: 500..880, y: 0..260`), wait at the north stop-line (`y=250`), and either proceed straight south or make a left turn onto the eastbound lanes.
   - **Northbound (bottom-to-top)**: Vehicles exit upwards along the left side of the yellow double-solid divider (`x: 700..950`).

## Crosswalks & Stop Lines
- **North Crosswalk**: `[[360, 210], [880, 270], [850, 320], [330, 260]]`
- **West Crosswalk**: `[[110, 310], [260, 330], [160, 580], [20, 560]]`
- **East Crosswalk**: `[[920, 360], [1060, 380], [930, 680], [790, 650]]`
- **Traffic Signals**: Overhead traffic lights are mounted on the signal poles at the north-west (`[420, 75]`), north-east (`[770, 155]`), and south-east (`[750, 455]`) corners of the intersection.

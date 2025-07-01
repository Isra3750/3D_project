# **Object measurement detection script**

This project is a proof of concept on the measurement of a box dimension from a point cloud file.

## Tech stack and modules

- Python
- Open3D
- numpy
- matplotlib
- scipy

## Features

- Display point cloud visualization of segmented box
- Allows for x1, y1, x2, y2 coordinate input for selective object detection
- Calculate lid coverage and hollow lid detection (reflected LIDAR from box top)
- Measurement of length, width, and height of box

## Usage

- Works with various PCD files
- "A" samples require input coordinate adjustment
- "E" samples works automatically (via input)
- Custom PCD requires changing the file reading paths and input coordinate for box location

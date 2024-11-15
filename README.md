# BikeMeasure - Bike Pipes Length Estimation Tool

## Overview

This script was developed as part of the **Introduction to Machine Vision** course at the **Warsaw University of
Technology**. It analyzes an image of a bike frame to estimates the lengths of pipes given the length of the upper pipe
(main horizontal tube).

The script leverages **histogram analysis** to detect and isolate the bike frame for further geometric analysis and pipe
length calculations.

## Features

1. **Image Processing**: Converts the image to the HSV color space and processes the histogram for feature detection.
2. **Color Segmentation**: Masks colors based on the HSV histogram peaks to isolate the bike frame.
3. **Line Detection**: Uses the Hough Transform to detect lines representing the bike frame's tubes.
4. **Intersection Calculation**: Computes intersections of detected lines to locate pipe joints.
5. **Pipe Length Calculation**: Estimates the length of the upper pipe and calculates other pipe lengths using geometric
   properties.
6. **Visualization**: Displays detected lines and intersections with annotations showing pipe lengths.

## Setup
```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements
```

```bash
python3 BikeMeasure.py
```

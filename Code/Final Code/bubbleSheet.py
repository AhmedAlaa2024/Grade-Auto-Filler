import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

from extractExcel import generateBubbleSheetExcel
from readConfigFiles import readConfigBubbleSheet
from bubbleSheetAutoCorrector import bubbleSheetAutoCorrector

# =============================================================================================
# Read the configuration file
# =============================================================================================
config = readConfigBubbleSheet()

# =============================================================================================
# Get the data from the bubble sheet image
# =============================================================================================
data = bubbleSheetAutoCorrector(config)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generateBubbleSheetExcel(config["result"], config["sheetName"], data)

print("Happy Ending")

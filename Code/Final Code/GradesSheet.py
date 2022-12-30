from CellsExtractionPhase import main
from DetectionPhase import detectionPhase
from ExtractExcel import generateGradeSheetExcel

# =============================================================================================
# Extract cells from the table
# =============================================================================================
SampleNumber = 15
cellImages = main("../Samples/Samples/{}.jpg".format(SampleNumber))

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================

# True if we want to extract names with grades
getNames = False
# True if we want to use already-made OCR, False if we want to use features + classifier
OCR = True
# Method that we want to use to detect symbols
method = "HOG"

columnTitles, data = detectionPhase(cellImages, getNames, OCR, method)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generateGradeSheetExcel("Result_{}".format(SampleNumber), "FirstSheet", columnTitles, data, getNames)

print("Happy Ending")

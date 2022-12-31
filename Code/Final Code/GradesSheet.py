from CellsExtractionPhase import main
from DetectionPhase import detectionPhase
from ExtractExcel import generateGradeSheetExcel

# =============================================================================================
# Extract cells from the table
# =============================================================================================
SampleNumber = 5
cellImages = main("../Samples/Samples/{}.jpg".format(SampleNumber))

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================

# True if we want to extract names with grades
getNames = False
# Hybrid for [id -> OCR, digits -> KNN], OCR, or KNN
digits = "KNN"
# Method that we want to use to detect symbols
method = "HOG"

columnTitles, data = detectionPhase(cellImages, getNames, digits, method)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generateGradeSheetExcel("Result_{}".format(SampleNumber), "FirstSheet", columnTitles, data, getNames)

print("Happy Ending")

from CellsExtractionPhase import main
from DetectionPhase import detectionPhase
from ExtractExcel import generate_excel

# =============================================================================================
# Extract cells from the table
# =============================================================================================
cellImages = main("../Samples/Samples/15.jpg")

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================

# True if we want to extract names with grades
getNames = False
# True if we want to use already-made OCR, False if we want to use features + classifier
OCR = True

columnTitles, data = detectionPhase(cellImages, getNames, OCR)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generate_excel("Yarab", "FirstSheet", columnTitles, data, getNames)

print("Happy Ending")

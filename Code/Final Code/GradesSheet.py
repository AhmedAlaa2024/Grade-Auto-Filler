from cellsExtractionPhase import extractCells
from detectionPhase import detectionPhase
from extractExcel import generateGradeSheetExcel
from readConfigFiles import readConfigGradeSheet

# =============================================================================================
# Read the configuration file
# =============================================================================================
config = readConfigGradeSheet()

# =============================================================================================
# Extract cells from the table
# =============================================================================================
cellImages = extractCells(config["samplePath"])

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================

# True if we want to extract names with grades
getNames = config["getNames"]
# Hybrid for [id -> OCR, digits -> KNN], OCR, or KNN
digitsDetectionMethod = config["digitsDetectionMethod"]
# Method that we want to use to detect symbols
symbolsDetectionMethod = config["symbolsDetectionMethod"]

columnTitles, data = detectionPhase(cellImages, getNames, 
																		digitsDetectionMethod, symbolsDetectionMethod)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generateGradeSheetExcel(config["fileName"], config["sheetName"], columnTitles, data, getNames)

print("Happy Ending")

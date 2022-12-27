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
getNames = False

columnTitles, data = detectionPhase(cellImages, getNames)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generate_excel("Yarab", "FirstSheet", columnTitles, data, getNames)

print("Happy Ending")

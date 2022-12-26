from CellsExtractionPhase import main
from DetectionPhase import detectionPhase
from ExtractExcel import generate_excel
import cv2

# =============================================================================================
# Extract cells from the table
# =============================================================================================
cellImages = main('../Samples/Samples/5.jpg')

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================
data = detectionPhase(cellImages)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generate_excel("Yarab", "FirstSheet", [
               "Code", "Student Name", "English Name", "1", "2", "3"], data)

print("Happy Ending")

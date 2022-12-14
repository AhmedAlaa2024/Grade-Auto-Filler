from CellsExtractionPhase import main
from DetectionPhase import detectionPhase
from ExtractExcel import generate_excel
import cv2

# =============================================================================================
# Extract cells from the table
# =============================================================================================
# img1 = cv2.imread("../Samples/Detection phase samples/right1.png")
# img2 = cv2.imread("../Samples/Detection phase samples/question mark3.png")
# img3 = cv2.imread("../Samples/Detection phase samples/digits/9.png")

# img4 = cv2.imread("../Samples/Detection phase samples/english name2.png")
# img5 = cv2.imread("../Samples/Detection phase samples/arabic name3.png")
# img6 = cv2.imread("../Samples/Detection phase samples/code1.png")

# cellImages = [img1, img2, img3, img4, img5, img6]
cellImages = main('../Samples/Samples/15.jpg')

# =============================================================================================
# Extract the data from the cell images
# =============================================================================================
data = detectionPhase(cellImages)

# =============================================================================================
# Prepare the excel sheet
# =============================================================================================
generate_excel("Yarab", "FirstSheet", [
               "Code", "Student Name", "English Name", "1", "2", "3"], data)

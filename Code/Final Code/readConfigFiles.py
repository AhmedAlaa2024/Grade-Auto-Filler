
def readConfigGradeSheet():
	"""
		Function used to extract the parameters needed for the program to start working

		Returns:
			Configuration needed for the program to work
	"""
	file = open("configGrade.conf", "r")
	
	samplePath = "../Samples/Samples/1.jpg"
	getNames = False
	digitsDetectionMethod = "Hybrid"
	symbolsDetectionMethod = "HOG"
	fileName = "Result"
	sheetName = "Sheet 1"

	for line in file:
		lineData = line.split("=")
		param = lineData[0]
		value = lineData[1].split("\n")[0]

		if param == "SAMPLE_PATH":
			samplePath = value
		elif param == "GET_NAMES":
			if value.lower() == "false":
				getNames = False
			else:
				getNames = True
		elif param == "DIGITS_DETECTION_METHOD":
			digitsDetectionMethod = value
		elif param == "SYMBOLS_DETECTION_METHOD":
			symbolsDetectionMethod = value
		elif param == "FILE_NAME":
			fileName = value
		elif param == "SHEET_NAME":
			sheetName = value

	file.close()
	return {
		"samplePath": samplePath,
		"getNames": getNames,
		"digitsDetectionMethod": digitsDetectionMethod,
		"symbolsDetectionMethod": symbolsDetectionMethod,
		"fileName": fileName,
		"sheetName": sheetName
	}


def readConfigBubbleSheet():
	"""
		Function used to extract the parameters needed for the program to start working

		Returns:
			Configuration needed for the program to work
	"""
	file = open("configBubble.conf", "r")

	studentAnswerPaperPath = "./StudentAnswerPapers/1.jpg"
	numStudents = 75
	numChoices = 5
	idExist = True
	IdLen = 5
	modelAnsFile = "./ModelAnswers/model_1.ans"
	idList = "./list_1.txt"
	result = "Result"
	sheetName = "Sheet 1"
	saveImages = True
	saveImagesDir = "./MarkedPapers"
	numCol = 3

	for line in file:
		lineData = line.split("=")
		param = lineData[0]
		value = lineData[1].split("\n")[0]

		if param == "SAMPLE_PATH":
			studentAnswerPaperPath = value
		elif param == "ID_EXIST":
			if value.lower() == "false":
				idExist = False
			else:
				idExist = True
		elif param == "SAVE_IMAGES":
			if value.lower() == "false":
				saveImages = False
			else:
				saveImages = True
		elif param == "NUM_STD":
			numStudents = int(value)
		elif param == "NUM_CHOICES":
			numChoices = int(value)
		elif param == "NUM_COL":
			numChoices = int(value)
		elif param == "STD_ID_LEN":
			IdLen = int(value)
		elif param == "FILE_NAME":
			result = value
		elif param == "SHEET_NAME":
			sheetName = value
		elif param == "MODEL_ANSWERS_FILE":
			modelAnsFile = value
		elif param == "STD_ID_LIST":
			idList = value
		elif param == "SAVE_IMAGEs_DIR":
			saveImagesDir = value

	file.close()
	return {
		"studentAnswerPaperPath": studentAnswerPaperPath,
		"numStudents": numStudents,
		"numChoices": numChoices,
		"idExist": idExist,
		"saveImages": saveImages,
		"IdLen": IdLen,
		"result": result,
		"sheetName": sheetName,
		"modelAnsFile": modelAnsFile,
		"idList": idList,
		"numCol": numCol,
		"saveImagesDir": saveImagesDir
	}
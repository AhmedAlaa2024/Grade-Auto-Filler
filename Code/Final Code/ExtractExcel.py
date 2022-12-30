import xlsxwriter


def generateGradeSheetExcel(workbookName, worksheetName, headersList, data, names=False):
	"""
		Function used to create the excel sheet from the data extracted from the table
	
		Arguments:
			workbookName: File name
			worksheetName: Sheet name
			headersList: List of names of the first row [header]
			data: List of data used to fill the excel sheet
			names: Boolean to determine if we are going to show english and arabic names or not
	"""
	# Creating workbook
	workbook = xlsxwriter.Workbook(workbookName + ".xlsx")

	# Red background that will be used with question marks
	redBG = workbook.add_format()
	redBG.set_bg_color("red")

	# Align all inputs
	alignCenter = workbook.add_format()
	alignCenter.set_align("center")
	alignLeft = workbook.add_format()
	alignLeft.set_align("left")
	alignRight = workbook.add_format()
	alignRight.set_align("right")

	# Creating worksheet
	worksheet = workbook.add_worksheet(worksheetName)

	# Adding headers
	for index, header in enumerate(headersList):
		worksheet.write(0, index, str(header).capitalize(), alignCenter)

	# Widths array
	# code, student name, english name
	widths = [len("Code"), len("Student Name"), len("English Name")]

	# Adding data
	for index1, entry in enumerate(data):
		for index2, header in enumerate(headersList):
			if entry[header] == -1:
				worksheet.write(index1+1, index2, "", redBG)
			elif entry[header] == -2:
				worksheet.write(index1+1, index2, "")
			elif header == "Code" or header == "English Name":
				worksheet.write(index1+1, index2, entry[header], alignLeft)

				# get the max width for [code, english name] columns
				if header == "Code":
						widths[0] = max(widths[0], len(entry[header]))
				else:
						widths[2] = max(widths[2], len(entry[header]))

			elif header == "Student Name":
				worksheet.write(index1+1, index2, entry[header], alignRight)

				# get the max width for [student name] column
				widths[1] = max(widths[1], len(entry[header]))
			else:
				worksheet.write(index1+1, index2, entry[header], alignCenter)

	if names:
		for i in range(3):
			worksheet.set_column(i, i, widths[i]+2)

	# Close workbook
	workbook.close()


def generateBubbleSheetExcel(workbookName, worksheetName, data):
	"""
		Function used to create the excel sheet from the data extracted from the bubble sheet
	
		Arguments:
			workbookName: File name
			worksheetName: Sheet name
			data: List of data used to fill the excel sheet
						example: {
											"setName": False,
											"id": 151111,
											"name": "Beshoy Morad Atya",
											"answers": [True, False, True, ....]
											}
	"""
	# Creating workbook
	workbook = xlsxwriter.Workbook(workbookName + ".xlsx")

	# Align the inputs
	alignCenter = workbook.add_format()
	alignCenter.set_align("center")

	# Creating worksheet
	worksheet = workbook.add_worksheet(worksheetName)

	# Add headers
	if data["setName"]:
		headersList = ["Name"]
	else:
		headersList = ["Code"]

	for i in range(len(data["answers"])):
		headersList.append(f"Q{i+1}")

	# Adding headers
	for index, header in enumerate(headersList):
		worksheet.write(0, index, str(header).capitalize(), alignCenter)

	# Adding the id or name
	if data["setName"]:
		worksheet.write(1, 0, data["name"], alignCenter)
		worksheet.set_column(0, 0, len(data["name"])+2)
	else:
		worksheet.write(1, 0, data["id"], alignCenter)


	# Adding data
	for index, entry in enumerate(data["answers"]):
		write = 0
		if entry:
			write = 1
		worksheet.write(1, index+1, write, alignCenter)

	# Close workbook
	workbook.close()


generateBubbleSheetExcel("Yarab", "FirstSheet", {
	"setName": False,
	"id": 151111,
	"name": "Beshoy Morad Atya",
	"answers": [True, False, True]
	})
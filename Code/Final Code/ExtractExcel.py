import xlsxwriter


def generate_excel(workbook_name, worksheet_name, headers_list, data):
    """
                Function used to create the excel sheet from the data extracted from the table

                workbook_name: File name
                worksheet_name: Sheet name
                headers_list: List of names of the first row [header]
                data: List of data used to fill the excel sheet
    """
    # Creating workbook
    workbook = xlsxwriter.Workbook(workbook_name + ".xlsx")

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
    worksheet = workbook.add_worksheet(worksheet_name)

    # Adding headers
    for index, header in enumerate(headers_list):
        worksheet.write(0, index, str(header).capitalize(), alignCenter)

    # Widths array
    # code, student name, english name
    widths = [len("Code"), len("Student Name"), len("English Name")]

    # Adding data
    for index1, entry in enumerate(data):
        for index2, header in enumerate(headers_list):
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

    for i in range(3):
        worksheet.set_column(i, i, widths[i]+2)

    # Close workbook
    workbook.close()

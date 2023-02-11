[Needed Libraries]:
===================
1- OpenCV
2- NumPy
3- Skimage
4- Imutils
5- PyTesseract
6- Xlsxwriter

[Module 1]:
===========
1- Set the path of the table image in the config file.
2- Set any other needed features in the config file 
	[get names, wanted method to detect symbols and numbers, etc..]
3- Use the command "python gradesSheet.py"

[Module 2]:
===========
1- Set the path of the directory of the samples in the config file
2- If there is no id in the image, set the ids in the id list file
3- Set all the other important parameters in the config file
	[number of students, number of questions, number of choices,
	 id length if exist, number of coulmns and rows, etc...]
4- Use the command "python bubbleSheet.py"
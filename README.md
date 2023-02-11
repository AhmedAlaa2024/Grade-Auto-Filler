# Grades Auto Filler

## 📝 Table of Contents

- [About](#about)
- [Technology](#technolgies)
- [Getting Started](#started)
- [Grade Auto Filler Model](#grade_filler)
  - [Overview](graded_sheet_overview)
  - [Flow Diagram](graded_sheet_flow)
  - [Results](graded_sheet_results)
- [Bubble Sheet Correction Model](#bubble_sheet)
  - [Overview](bubble_sheet_overview)
  - [Flow Diagram](bubble_sheet_results)
  - [Results]()
- [Contributors](#contributors)
- [License](#license)


## 📙 About <a name = "about"></a>
- The main idea of (Grade Auto Filler) is to give an image of a table with some data to the program and get an output of excel sheet containing the data that was in that image after mapping the symbols to the wanted grades.

- The main idea of (Bubble sheet correction) islocalize the filled circles referenced to the number of columns and number of rowsand compare them to a given model answer and provide sheet of students’ grades.

## 💻 Technology <a name = "technolgies"></a>

- Python 
- Jupyter Notebook
- OpenCV

## 🏁 Getting Started <a name = "started"></a>

<ul>
<li>Clone the repository

<br>

```
git clone https://github.com/AhmedAlaa2024/Grade-Auto-Filler.git
```

</li>

<li>Install Packages

<br>

```
pip install -r requirements.txt
```

</li>

<li>To run the Grade Auto Filler

<br>

```
Set the path of the table image in the config file [configGrade.conf]

Set any other needed features in the config file [get names, wanted method to detect symbols and numbers, etc..]

Use the command "python gradesSheet.py"
```

</li>

<li>To run the Bubble Sheet Correction

<br>

```
Set the path of the directory of the samples in the config file [configBubble.conf]

If there is no id in the image, set the ids in the id list file

Set all the other important parameters in the config file [number of students, number of questions, number of choices, id length if exist, number of coulmns and rows, etc...]

Use the command "python bubbleSheet.py"
```

</li>
</ul>

<br>

***

<br>

<h2 align=center > Grade Auto Filler Model <a name = "grade_filler"></a></h2>

### 📷 OverView<a name = "graded_sheet_overview"></a>

- It allows you to turn an image into a digital form (excel sheet)
- It handles Skewing and orientation
- Printed Student ID is detected using OCR or Feature and execration
- Written Symbols like ✓ & ? are detect using HOG feature extractor and predicted using SVM or with normal image processing techniques
- Handwritten numeric values are detected using OCR and Feature and execration

***

### 🖍 Flow Diagram <a name = "graded_sheet_flow"></a>

<img src="Documents/Grade Autofiller.jpg" draggable="false">

***

### 📚 Results <a name = "graded_sheet_results"></a>

<h4 align=center>(1)</a></h4>

<table>
  <tr>
    <td width=40% valign="center"><img src="Final Outputs/Grade samples and outputs/1.jpg"/></td>
    <td width=40% valign="center"><img src="Final Outputs/Grade samples and outputs/Result_1.jpg"/></td>
  </tr>
</table>


<h4 align=center>(2)</a></h4>

<table>
  <tr>
    <td width=40% valign="center"><img src="Final Outputs/Grade samples and outputs/5.jpg"/></td>
    <td width=40% valign="center"><img src="Final Outputs/Grade samples and outputs/Result_2.jpg"/></td>
  </tr>
</table>

<br>

***

<br>

<h2 align=center > Bubble Sheet Correction Model <a name = "bubble_sheet"></a></h2>

### 📷 OverView <a name = "bubble_sheet_overview"></a>

- It handles Skewing and orientation
- It handles different ink colors
- It allows different formats for the sheet ( but bubbles must be vertically aligned in all formats )
- Differnet number of questions and choices

***

### 🖍 Flow Diagram <a name = "graded_sheet_flow"></a>

<img src="Documents/Bubble Sheet.jpg" draggable="false">

<br>

***

### 📚 Results <a name = "bubble_sheet_results"></a>

<h4 align=center>(1)</a></h4>

<table>
  <tr>
    <td width=40% valign="center"><img src="Final Outputs/BubbleSheet Samples & Outputs/StudentAnswers/Patch_1/02141.jpg"/></td>
    <td width=40% valign="center"><img src="Final Outputs/BubbleSheet Samples & Outputs/Outputs/MarkedPapers/Patch_1_02141.jpg"/></td>
  </tr>
</table>


<h4 align=center>(2)</a></h4>

<table>
  <tr>
    <td width=40% valign="center"><img src="Final Outputs/BubbleSheet Samples & Outputs/StudentAnswers/Patch_2/1.jpg"/></td>
    <td width=40% valign="center"><img src="Final Outputs/BubbleSheet Samples & Outputs/Outputs/MarkedPapers/Patch_2_02141.jpg"/></td>
  </tr>
</table>


***
 
## Contributors <a name = "contributors"></a>

<table>
  <tr>
		<td align="center">
    <a href="https://github.com/AhmedAlaa2024" target="_black">
    <img src="https://avatars.githubusercontent.com/u/62505107?v=4" width="150px;" alt="Ahmed Alaa"/>
    <br />
    <sub><b>Ahmed Alaa</b></sub></a>
    </td>
		<td align="center">
    <a href="https://github.com/BeshoyMorad" target="_black">
    <img src="https://avatars.githubusercontent.com/u/82404564?v=4" width="150px;" alt="Beshoy Morad"/>
    <br />
    <sub><b>Beshoy Morad</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/waleedhesham446" target="_black">
    <img src="https://avatars.githubusercontent.com/u/72695729?v=4" width="150px;" alt="Waleed Hesham"/>
    <br />
    <sub><b>Waleed Hesham</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/ZeyadTarekk" target="_black">
    <img src="https://avatars.githubusercontent.com/u/76125650?v=4" width="150px;" alt="Zeyad Tarek"/>
    <br />
    <sub><b>Zeyad Tarek</b></sub></a>
    </td>
  </tr>
 </table>


## License <a name="license"></a>
This software is licensed under MIT License, See [License](https://github.com/AhmedAlaa2024/Grade-Auto-Filler/blob/master/LICENSE) for more information.
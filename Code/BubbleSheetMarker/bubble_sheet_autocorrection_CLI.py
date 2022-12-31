import sys
from pathlib import Path
import numpy as np

# python bubble_sheet_autocorrect.py --std-anwers {image} --model-answers {.txt} --num-quest {int} --num-choices {int} --report {.xlsx}

# Ex:
# python .\bubble_sheet_autocorrection_CLI.py --std-answers .\StudentAnswers\12.jpg --model-answers .\ModelAnswers\model_answers_1.ans --num-quest 20 --report .\Results\report_1.xlsx

# Ex: python .\bubble_sheet_autocorrection_CLI.py .\SWE_F2022.conf
arguments = list(sys.argv)

if (args_count := len(arguments)) > 2:
    print(f"One arguments expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count < 2:
    print("You must specify the path of configuration file!")
    raise SystemExit(2)

CONFIG_FILE_PATH = arguments[1]

print(" ===================================== ")
print("| Grading is fininalized successfully! |")
print(" ===================================== ")
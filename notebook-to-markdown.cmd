RMDIR README_files /S /Q
jupyter nbconvert --to markdown README.ipynb
python clean-markdown.py

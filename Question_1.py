#Syntax and Style

#1.a:

import os
import glob 
import pandas as pd

directory_path = "/Users/caryswilliams/Documents/Masters Degree Folder/Coursework Pack NLP/texts"

txt_files = glob.glob(os.path.join(directory_path, "**", "*.txt"), recursive=True)

def read_novels(txt_files):
    novels = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            content = file.read()
            filename = os.path.basename(txt_file).split('-')
            novels.append({'Text': content, 'Title': filename[0], 'Author': filename[1], 'Year of Publication': filename[2]})
    novels_dataframe = pd.DataFrame(novels)
    #novels_dataframe.sort_values(by='Year of Publication', ignore_index=False)
    
    return novels_dataframe.sort_values(by='Year of Publication', ignore_index=True)

novels_dataframe = read_novels(txt_files)
print(novels_dataframe)



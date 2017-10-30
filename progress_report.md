_Entry #1: Oct 1, 2017_

**Updates**  
- Created local and remote project repositories
- Created README, LICENSE and progress_report md files
- Updated project_plan.md and included it in the project repo

**Next steps**
- Find out about spoken text files and create initial CSV / dataframe
- Experiment with code for cleaning feedback files of identifying names

<br>
<br>
_Entry #2: Oct 8, 2017_

**Updates**  
- started cleanup_Ben notebook with initial code.  
- merged answer.csv and Ss_info_csv into one larger dataframe (data1_df), linked by the anon_id index
- changed all null values to NaN  
- found basic info about data1_df  
- removed unnecessary apostrophes from 'gender column'  
- created small sample csv with 100 rows to share in class project repo

**Next steps**
- Determining which files are appropriate speaking files and creating a speaking_df  
- Figure out how to change all floats back to ints in dataframe
- Remove more unnecessary apostrophes from other columns  
- continuing to explore data

<br>
<br>
_Entry #2: Oct 29, 2017_

**Updates**  
- *Creating of new 'Project_Code2'*
  This new document has been created as a number of significant changes have been made to the original code. Based on discussions with other members of the ELI Data Mining Group, the following points were determined:  

  - For the sake of efficiency, it is better not to merge the different data frames into one big one
  - A 'sanitization' step of the data was completed which duplicated some of the steps of my initial code. These duplications include removing unwanted apostrophes, changing all 'null' and 'ull' to NaN, and removing empty or unreal students (who were most likely teachers). As such, the dataset is now ready for more in-depth cleaning and analysis, i.e. the purpose of this notebook.  

- New code attempts and descriptions of three goals:
  #1 create find_stuff function to find info from multiple dataframes
  #2 find a link between the different dataframes to identify which 'text' from answer_df have class_id == 3 and file_id == 6
  #3 tokenize 'text' column in answer_df  

- Of the above goals, the first two were unsuccessful although at least there is some clarity of what is needed
- Tokenized column created successfully
- Creation of 2 new dataframes: course_df and user_df which (hopefully) contain information useful for finding the necessary info  

- Completed LICENSE doc which outlines the proposed future CC LICENSE
- Updated project_plan.md to reflect ongoing changes to overall project and to set realistic goals  

**Next steps**
- Consult with colleagues about goals 1 and 2
- Create dataframe with class_id = 3, file_type_id = 6, corresponding answer_id and 'text'
- Gather stats on bigrams and trigrams and sort by L1 and proficiency level
- Apply lexical diversity code to different columns (toks, bigrams, trigrams)

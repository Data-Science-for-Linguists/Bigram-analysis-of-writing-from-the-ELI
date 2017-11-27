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
_Entry #3: Oct 29, 2017_

**Updates**  
- *Creating of new 'Project_Code2'*
  This new document has been created as a number of significant changes have been made to the original code. Based on discussions with other members of the ELI Data Mining Group, the following points were determined:  

  - For the sake of efficiency, it is better not to merge the different data frames into one big one
  - A 'sanitization' step of the data was completed which duplicated some of the steps of my initial code. These duplications include removing unwanted apostrophes, changing all 'null' and 'ull' to NaN, and removing empty or unreal students (who were most likely teachers). As such, the dataset is now ready for more in-depth cleaning and analysis, i.e. the purpose of this notebook.  

- New code attempts and descriptions of three goals:
  #1 create find_stuff function to find info from multiple dataframes
  #2 find a link between the different dataframes to identify which 'text' from answer_df have class_id == 3 and file_id == 6
  #3 tokenize 'text' column in answer_df  

- Of the above goals, the first is slightly successful, allowing me to now plug in a course type and all the relevant classes are then provided. The next step is to create similar functions mapping class id to other stats  
- goal 2 was unsuccessful although at least there is some clarity of what is needed  
- Tokenized column created successfully  
- Bigram column created (I believe) successfully although more testing is needed and also consideration of whether to remove punctuation  
- Creation of 2 new dataframes: course_df and user_df which (hopefully) contain information useful for finding the necessary info  
- Completed LICENSE doc which outlines the proposed future CC LICENSE
- Updated project_plan.md to reflect ongoing changes to overall project and to set realistic goals  

**Next steps**
- Consult with colleagues about goals 1 and 2  
- Create dataframe with class_id = 3, file_type_id = 6, corresponding answer_id and 'text'  
- Analyze stats on bigrams and sort by L1 and proficiency level  
- Apply lexical diversity code to different columns (toks, bigrams)  
<br>
<br>
_Entry #4: Nov 6, 2017_

**Updates**  
- *Change of plan:*  
  After a lot of searching, it turns out the speaking transcripts I wanted had not been included in the answer_csv but were still on a hard drive and not yet part of the dataset. Rather than spending more time on data acquisition, the plan is to continue with the ideas as before, but on the written texts which I have already processed. This will allow for more time to be spent on data analysis and machine learning.

- *Continuing on from 'Project_Code2'*  
  - entire corpus created by joining all texts from 'texts' columns
  - unigram and bigram frequency dictionaries created from the new corpus
  - MI function created that produces MI when two words from the corpus are entered

**Next steps**  
Create another DF called bigrams_df with bigrams, MI scores, occurrences per million score, and perhaps more to bge added later. To do so:  
1) Create function for calculating MI
2) Create function for calculating occurrences per million for unigrams and bigrams  
3) Apply the MI formula for pairs of words in the bigram list and create a column in the new DF  
4) Apply the occurrences per million for bigrams and create a column in the new DF  
5) Create a column showing percentage of time the bigrams are used by the three proficiency levels  

<br>
<br>

_Entry #5: Nov 27, 2017_

**Updates**  
- *Based on Progress report 2 feedback*  
  - Create data sharing plan
  - Separate own licensing form ELI licensing
  - Include sample data in repo
  - Rename LICENSE.md to LICENSE_notes.md
  - Make a user-facing LICENSE.md file as a binding document

- *Based on Visitor log feedback*  
  - Fix sample data (no empty users)
  - Update README and Project Plan to focus on bigrams

- *Based on Progress report 3 instructions*
  - give project repo a descriptive name
- Update your README.md file:
- Update project_plan.md

- *Continued analysis*  
  - Add starter code  
  - Add class abbreviations and version number to each answer
  - Complete 'Next Steps' from previous update:
  Create another DF called bigrams_df with bigrams, MI scores, occurrences per million score, and perhaps more to be added later. To do so:  
  1) Create function for calculating MI
  2) Create function for calculating occurrences per million for unigrams and bigrams  
  3) Apply the MI formula for pairs of words in the bigram list and create a column in the new DF  
  4) Apply the occurrences per million for bigrams and create a column in the new DF  
  5) Create a column showing percentage of time the bigrams are used by the three proficiency levels
- Narrow down answer_df to new df with only writing class type and version type 1

**Next steps for final report and presentation**  
- *Machine learning:*
  - Predict level based on bigram frequency (types and tokens)
  - Predict level based on MI of bigrams used
- Create visualizations (heat maps for predictions and bar graphs for observed stats)
- Simple statistical analysis of significance of correlation and variance
- Create powerpoint presentation

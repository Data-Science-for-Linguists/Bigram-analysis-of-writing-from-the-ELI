***Project plan***  

Last revised 29/10/17

- Project title: *An analysis of spoken lexical bundles in the Pitt ELI corpus*

- Project summary:
  At the University of Pittsburgh, a data set was created from the output of learners at the English Language Institute (ELI) from 2005-2012. This output was in response to speaking, writing, reading, and grammar tasks, with the resulting corpus consisting of written and transcribed speaking texts from three different proficiency levels (low-intermediate, intermediate, and advanced). The learners in this corpus primarily have three different L1s: Arabic, Chinese, and Korean.

  To date, there has already been analysis of the corpus, e.g. in relation to lexical development (Juffs 2015), yet numerous avenues for research remain. The proposed project seeks to complement the current work of Juffs (2017) which analyzes lexical bundles in academic writing in the corpus. In this related project, lexical bundles of the spoken transcripts would be collected and analyzed, with a view to discovering differences and similarities to those found in the learners' academic writing.
<br>
<br>

**Data**  
The data under analysis have been extracted from the original MYSQL database and uploaded to a private GitHub repository. Within this repository, there are 19 csv files and one README file. Six of the csv files provide the different identifying codes used. The remaining csv files divide the data into such useful sub-groupings as responses, the prompts used, the feedback from the teachers, the student surveys, and the participant information.

The data in the dataset has already been cleaned up to a certain extent although there remains work to be done in this regard. To date, the following steps have already been taken (according to the README file):
  - identifying information (student name, birth date, email) was removed from the indices
  - each student (user_id) was mapped to an anonymous identifier (anon_id)
  - records that do not have any link out have been removed
  - research-related entries (researchers, research consent, etc.) have been stored away
  - some 'sanitization' of the original CSV files (see project code 2 for details)

_*Data cleaning*_  
There is no need for data sourcing, but continued cleaning efforts would assist both this project and future projects using this dataset. To aid in this cleaning, for this project I am doing the following:
- removing the identifying names of students from teachers' feedback. To do so, I will first compile a list of all the students' first names from the original data, then use a replace method to substitute these names in the feedback texts with a generic substitute like Jane/John Doe. Next, to deal with remaining names, like abbreviations or adopted 'English names' of students, I will extract a list of all capitalized words from the feedback, eliminate any words which also appear in a common word list (list to be determined), and manually check the remaining items.
- removing unnecessary apostrophes, standardizing null responses, and other normalizing of existing data
- removing blank entries and non-student entries
- other cleaning which arises when investigating the data

_*Size*_  
The exact size is currently unknown at present, but finding out this information (using the text tags) is one stage of the investigation.

_*Access*_  
At present, the dataset is private and the property of the University of Pittsburgh. As a linguistics student involved with analyzing the data, I have access to this private repository. Since the dataset is not yet ready to be shared publicly in its entirety, for this project I would be presenting a sample, with permission from the dataset's authors.
<br>
<br>
<br>
**Analysis**

_*Goal*_  
The end goal of the project is to gain a clearer understanding of the nature of spoken lexical bundles of learners with different levels of English language proficiency. In addition, by compiling the necessary dataframes to achieve this goal, I will be creating useful data for future projects and analyses which utilize this same dataset. Ultimately, any linguistic discoveries made will tell us something about properties of learners' lexis, and in the future this information can be compared to similar findings in other general English corpora.

_*Linguistic analysis*_  
The primary linguistic analysis will be in the gathering and sorting of bigrams, trigrams, four-grams, and five-grams from the spoken transcripts. Subsequent analysis will be necessary for deciding which smaller bundles are actually just components of larger bundles, and which bundles are distinct. Time permitting I will attempt to apply new code that has been created publicly for measuring lexical diversity.

_*Hypotheses*_  
This particular research focuses more on data collection and description rather than testing a particular hypothesis. However, on a broad level, the research hypothesis is that the bundles in speaking and writing will be significantly different, as will the bundles used at different levels of English language proficiency.

_*Predictive analysis (machine learning and classification)*_  
It is still undetermined to what extent there will machine learning and classification. However, if feasible, the user will be able to input a lexical bundle and the machine learning will then predict the proficiency level of the learner based on their ELI level. To do so, the program would need to be supplied with typical attributes of expected bundles at each level, based upon the collected data, e.g. typical words in the bundles, typical collocations, frequency bands of words/bundles in general corpora, etc.
<br>
<br>
<br>
**Presentation**  
As mentioned in the *Size* section above, only a sample of the entire private dataset will be shared publicly. To supplement the typical powerpoint presentation and screenshots, an interactive element will be added to the presentation. This interactive part will be a chance for the audience to predict whether lexical bundles displayed are from writing or speaking texts, followed by a discussion of the differences. Similarly, the audience will then predict which level the displayed bundles are from, with a discussion of how they reflect level and potential implications for teaching.
<br>
<br>
<br>
**References**  
Juffs, A. (Forthcoming). Lexical Development In The Writing Of Intensive   English Program Students. In R.M. DeKeyser & G. Preito Botana (Eds.), *Reconciling methodological demands with pedagogic applicability.* Amsterdam: John Benjamins.

Juffs, A. 2017. The longitudinal development of lexical bundles in the written output of Arabic-speaking ESL learners. Unpublished manuscript.

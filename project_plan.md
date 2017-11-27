***Project plan***  

Last revised 27/11/17

- *Change of plan 6/11/17:*  
  After a lot of searching, it turns out the speaking transcripts I wanted had not been included in the answer_csv but were still on a hard drive and not yet part of the dataset. Rather than spending more time on data acquisition, the plan is to continue with the ideas as before, but on the written texts which I have already processed. This will allow for more time to be spent on data analysis and machine learning. All references to 'spoken texts' now refers to 'written texts'.  

- *Change of plan 20/11/17:*  
  Based on peer feedback and the results of data analysis, it seemed appropriate to focus in on bigrams and the metric of Mutual Information, rather than looking at other types of formulaic sequences. Given the wealth of data, further explorations can be considered for future research projects, e.g. in relation to ConcGrams.


- Project title: *Analysis of bigrams from learners' written work at the Pitt English Language Institute (ELI)*

- Project summary:
  At the University of Pittsburgh, a data set was created from the output of learners at the English Language Institute (ELI) from 2005-2012. This output was in response to speaking, writing, reading, and grammar tasks, with the resulting corpus consisting of written and transcribed speaking texts from three different proficiency levels (low-intermediate, intermediate, and advanced). The learners in this corpus primarily have three different L1s: Arabic, Chinese, and Korean.

  To date, there has already been analysis of the corpus, e.g. in relation to lexical development (Juffs 2015), yet numerous avenues for research remain. The proposed project seeks to complement the current work of Juffs (2017) which analyzes lexical bundles in academic writing in the corpus. In this related project, lexical bundles, specifically written bigrams, are collected and analyzed, with a view to investigating the relationship between the metric of Mutual Information (MI), learner proficiency, and bigram frequency.

  MI is the strength of the association between words, i.e. the two-way likelihood of them co-occurring (Simpson-Vlach & Ellis, 2010). MI is of particular importance when considering how meaningful formulaic sequences are, as they are the statistical measure which corresponds most closely to native speaker judgements of the salience of formulaic sequences (Paquot & Granger, 2012).  
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
- removing unnecessary apostrophes, standardizing null responses, and other normalizing of existing data
- removing blank entries and non-student entries
- other cleaning which arises when investigating the data

_*Size*_  
There are 24544 written texts produced by learners

_*Access*_  
At present, the dataset is private and the property of the University of Pittsburgh. As a linguistics student involved with analyzing the data, I have access to this private repository. Since the dataset is not yet ready to be shared publicly in its entirety, for this project I would be presenting a sample, with permission from the dataset's authors.
<br>
<br>
<br>
**Analysis**

_*Goal*_  
The end goal of the project is to gain a clearer understanding of the relationship between characteristics of bigrams, especially Mutual Information, different levels of English language proficiency. In addition, by compiling the necessary dataframes to achieve this goal, I will be creating useful data for future projects and analyses which utilize this same dataset. Ultimately, any linguistic discoveries made will tell us something about properties of learners' lexis, and in the future this information can be compared to similar findings in other general English corpora.

_*Linguistic analysis*_  
The primary linguistic analysis will be in the gathering and sorting of bigrams from the written transcripts. Analysis will be made in terms of MI, frequency (types and tokens), and occurrences per million

_*Hypotheses*_  
This particular research focuses more on data collection and description rather than testing a particular hypothesis. However, on a broad level, the research hypothesis is that bigrams with higher MI will be more predictive of proficiency than just the frequency of all bigrams, as MI is more likely to indicate meaningful collocations.

_*Predictive analysis (machine learning and classification)*_  
If feasible, the user will be able to input a text and the machine learning will then predict the proficiency level of the learner based on the quantity of high, mid, and low MI bigrams.
<br>
<br>
<br>
**Presentation**  
As mentioned in the *Size* section above, only a sample of the entire private dataset will be shared publicly. To supplement the typical powerpoint presentation and screenshots, an interactive element will be added to the presentation. This interactive part will be a chance for the audience to predict the level of MI of bigrams, followed by a discussion of the implications. We will also look at how the BYU-COCA interface uses MI in their sorting.
<br>
<br>
<br>
**References**  
Juffs, A. (Forthcoming). Lexical Development In The Writing Of Intensive   English Program Students. In R.M. DeKeyser & G. Preito Botana (Eds.), *Reconciling methodological demands with pedagogic applicability.* Amsterdam: John Benjamins.

Juffs, A. 2017. The longitudinal development of lexical bundles in the written output of Arabic-speaking ESL learners. Unpublished manuscript.

#!/usr/bin/env python
# coding: utf-8

# In[9]:


pip install spacy


# In[13]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[1]:


import spacy
import random


# In[2]:


#for processing text load the english language model-"en_core_web_sm" from spacy
nlp = spacy.load('en_core_web_sm')


# In[3]:


#defines the function to generate multiple choice questions which takes the  paragraph(str) and the number of questions(int) to generate 
def get_mca_questions(context:str,no_of_qstns:int):
    the_doc=nlp(context)
    
    # defines a function to generate questions with multiple correct answers
    def generate_mcq_wmanswrs(qstn,correct_options,other_options,no_of_options=4):
        options=correct_options+other_options
        random.shuffle(options)
        mcq={'qstn':qstn,
             'options':options,
             'correct_options':correct_options

            }
        return mcq
    
    def generate_qstn():
        sentence=random.choice(list(the_doc.sents))
        blankword=random.choice([token for token in sentence if not token.is_punct])
        the_question=sentence.text.replace(blankword.text,'___________')
        correct_options=[blankword.text]

        other_options=[token.text for token in the_doc if token.is_alpha and token.text != correct_options[0]]
        correct_options_count = random.randint(1, 2)  # Generate 1 or 2 correct options
        correct_options.extend(random.sample(other_options, correct_options_count))

        other_options_count = min(4 - correct_options_count, len(other_options))
        other_options = random.sample(other_options, other_options_count)
        mcq = generate_mcq_wmanswrs(the_question, correct_options, other_options)
        return mcq
    
    thequestions = [generate_qstn() for _ in range(no_of_qstns)]

    mca_questions = []
    for i, question in enumerate(thequestions, start=1):
        question_str = f"Q{i}: {question['qstn']}\n"
        options_str = ""
        for j, option in enumerate(question['options']):
            options_str += f"{j+1}. {option}\n"

        correct_options_formatted = " & ".join([f"({chr(97+question['options'].index(ans))})" for ans in question['correct_options']])
        correct_options_str = f"Correct Options: {correct_options_formatted}"

        mca_question = f"{question_str}{options_str}{correct_options_str}\n"
        mca_questions.append(mca_question)

    return mca_questions


# In[4]:


context = input("Enter the paragraph: ")
no_of_qstns = int(input("Enter the number of questions: "))
mca_questions = get_mca_questions(context, no_of_qstns)
for question in mca_questions:
    print(question)


# # Functions seperately

# In[20]:


# defines a function to generate questions with multiple correct answers
def generate_mcq_wmanswrs(qstn,correct_options,other_options,no_of_options=4):
    options=correct_options+other_options
    random.shuffle(options)
    mcq={'qstn':qstn,
         'options':options,
         'correct_options':correct_options
         
        }
    return mcq
    


# In[21]:


def generate_qstn():
    sentence=random.choice(list(doc.sents))
    blankword=random.choice([token for token in sentence if not token.is_punct])
    the_question=sentence.text.replace(blankword.text,'___________')
    correct_options=[blankword.text]
    
    other_options=[token.text for token in the_doc if token.is_alpha and token.text != correct_options[0]]
    correct_options_count = random.randint(1, 2)  # Generate 1 or 2 correct options
    correct_options.extend(random.sample(other_options, correct_options_count))

    other_options_count = min(4 - correct_options_count, len(other_options))
    other_options = random.sample(other_options, other_options_count)
    mcq = generate_mcq_wmanswrs(question_text, correct_answers, other_options)
    return mcq


# In[24]:


thequestions = [generate_qstn() for _ in range(no_of_qstns)]

mca_questions = []
for i, question in enumerate(thequestions, start=1):
        question_str = f"Q{i}: {question['qstn']}\n"
        options_str = ""
        for j, option in enumerate(question['options']):
            options_str += f"{j+1}. {option}\n"

        correct_options_formatted = " & ".join([f"({chr(97+question['options'].index(ans))})" for ans in question['correct_options']])
        correct_options_str = f"Correct Options: {correct_options_formatted}"

        mca_question = f"{question_str}{options_str}{correct_options_str}\n"
        mca_questions.append(mca_question)

        return mca_questions


# In[ ]:





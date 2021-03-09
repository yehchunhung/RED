# Reddit Empathetic Dialogue DataSet (RED Dataset)
Most people suffer from emotional distress due to going through a significant life change, financial crisis, being a caregiver or due to various physical and mental health conditions. Inability to regulate emotion in such episodes can potentially lead to self-destructive behavior such as substance abuse, self-harm or suicide. However, due to public and personal “stigma” associated with mental health, most people do not reach out for help. Even therapeutic consultations are limited and are not available 24/7 to support people when they are going through a traumatic episode. Therefore, it is important to assess the ability of AI driven chatbots to help people to deal with emotional distress and help them regulate emotion. One of the major limitations in developing such a chatbot is the unavailability of a curated dialogue dataset containing emotional support. With this project, we aim to curate and analyse such a dataset having the potential to train and evaluate mental care giving chatbot that can support people in emotional distress.

## Table of Contents
- [Dependencies](#dependencies)
- [Files Description](#files-description)
- [Dataset](#dataset)
- [Steps to Reproduce Result](#steps-to-reproduce-our-result)
- [References](#references)
- [License](#license)

## Dependencies
The codes are implemented in Python 3. You will need the following dependencies installed:

* [requests]
    ```bash
    $ pip install requests
    ```
* [NLTK]
    ```bash
    $ pip install nltk
    ```

* [contractions]
    ```bash
    $ pip install contractions
    ```
    
* [swifter]
    ```bash
    $ pip install swifter
    ```
    
* [language-tool-python]
    ```bash
    $ pip install language-tool-python
    ```

* [tqdm]
    ```bash
    $ pip install tqdm
    ```
    
* [emoji]
    ```bash
    $ pip install emoji
    ```
    
* [joblib]
    ```bash
    $ pip install joblib
    ```
    
* [profanity-check]
    ```bash
    $ pip install profanity-check
    ```

* [vaderSentiment]
    ```bash
    $ pip install vaderSentiment
    ```
    
* [tensorflow]
    ```bash
    $ pip install tensorflow
    ```

## Files Description
- `reddit-scrape-pushshift.ipynb`: The notebook is mainly used for scraping Reddit textual data using Pushshift APIs.
- `preprocess.ipynb`: Preprocess raw scraped conversation data and convert them to table-like data frames.
- `EDA.ipynb`: The notebook presents various analyses and graphical representations to attain insights and find patterns.
- `utils4text.py`: This file contains the supporting functions applied in `EDA.ipynb`.
- `EmoBERT.ipynb`: The notebook for making emotion prediction on the messages in dialogues. Before running it, make sure to load the checkpoints [HERE](https://drive.google.com/drive/folders/1bsMW6AA_vytzwLDNA5OXhp6IZK_RC0GF?usp=sharing).

## Dataset
* The dataset can be found in two folders, **raw** and **dataset**, in [Google Drive](https://drive.google.com/drive/folders/1d74Po6N-es2-2UOsoWSWjCcAACpF_dCG?usp=sharing). 
* Categories 
  - **raw**: Raw data scraped by Pushshift can be found [HERE](https://drive.google.com/drive/folders/1_tSoGY2TP7ytGpg8i_EZc777X36Cc5xy?usp=sharing).
  - **dataset**: Refined data after preprocessing can be found [HERE](https://drive.google.com/drive/folders/1WyaenOEfs9KI7bHYST7pKwGQZP5TMC3R?usp=sharing). Inside the folder, there are three sub-folders containing pure conversation data, conversation data annotated with sentiments, and the data annotated with emotions. 


## Steps towards Results
1. For scraping dialogues from Reddit, run the notebook `reddit-scrape-pushshift.ipynb`. Note that it would take several hours to finish scraping on the subreddits like r/depression, r/offmychest and r/suicidewatch.
2. Run `preprocess.ipynb` to transform the scraped data in the JSON format into data frames.
3. If you want to explore the dialogues, check `EDA.ipynb` for more details.
4. Run `EmoBERT.ipynb` to get the emotion prediction of the utterances.



[requests]: <https://pypi.org/project/requests/>
[NLTK]: <https://pypi.org/project/nltk/>
[contractions]: <https://pypi.org/project/contractions/>
[swifter]: <https://pypi.org/project/swifter/>
[language-tool-python]: <https://pypi.org/project/language-tool-python/>
[tqdm]: <https://pypi.org/project/tqdm/>
[emoji]: <https://pypi.org/project/emoji/>
[joblib]: <https://pypi.org/project/joblib/>
[profanity-check]: <https://pypi.org/project/profanity-check/>
[vaderSentiment]: <https://pypi.org/project/vaderSentiment/>
[tensorflow]: <https://pypi.org/project/tensorflow/>

## References
1. [The Pushshift Reddit Dataset](https://arxiv.org/abs/2001.08435)
2. [EmpatheticIntents](https://github.com/anuradha1992/EmpatheticIntents.git)

## License
Licensed under [MIT License](LICENSE)

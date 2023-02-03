- `cd DeepPavlov` - for the correct config path. Or add DeepPavlov to the beginning

1. **Context Question Answering**
    + Question answer model for SQuAD dataset
    + Stanford Question Answering Dataset ( SQuAD ) (EN)
    + SDSJ task B ( RU )
    + Answering a question in an SQuAD dataset is the task of finding the answer to a question in a given context (e.g. a Wikipedia paragraph), where the answer to each question is a context segment: < br >
    - python -m deeppavlov install DeepPavlov / deeppavlov /configs/squad/ squad_ru_rubert_infer.json
        - python -m deeppavlov download DeepPavlov / deeppavlov /configs/squad/ squad_ru_rubert_infer.json
        - python -m deeppavlov interact DeepPavlov / deeppavlov /configs/squad/ squad_ru_rubert_infer.json - works . Cool thing. In some places it is very cool! Now it has begun to fly out .... And this is due to the fact that another neuron was launched.
        - Performance:
        - 0.03   200 symbols
        - 0.07   900 symbols
        - 0.1   2400 symbols
        - 0.25   9500 symbols
        - Works very cool. Can give an answer if there is none, but the confidence indicator (the last digit in the answer) decreases.
        - It also usually does not give any answer if it thinks that there is no answer.
        - It seems that it doesn’t work very well for English words (or does he perceive them purely as just symbols ??)
        - In principle, it finds even if only 1 word is given, but it reacts more to the Russian word ( Docker installation ) than to English (that is, it will give out everything related to the installation and not the docker)
        - If 1 word is supplied, even English, then it finds its occurrences, but if it is not a related context that is checked at all, then it can still find some occurrences in it (although they are not close)
        - Responds to similar meanings Q : "problems with docker " A: 'library is no longer supported'
        - In some places it works strangely (perhaps because of the English language). Finds it well in places.
        - Plus, you can use it in the "all texts at once" mode (considering that the execution speed decreases less and less with increasing text volume).
        <br> _ _
    - python -m deeppavlov install deeppavlov/configs/squad/squad_bert_multilingual_freezed_emb.json
        - python -m deeppavlov download deeppavlov/configs/squad/squad_bert_multilingual_freezed_emb.json
        - python -m deeppavlov interact deeppavlov/configs/squad/squad_bert_multilingual_freezed_emb.json
        - works poorly, almost always returns nothing
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/squad/ squad_bert_infer.json
        - python -m deeppavlov download deeppavlov /configs/squad/ squad_bert_infer.json
        - python -m deeppavlov interact deeppavlov /configs/squad/ squad_bert_infer.json
        - does not work if there is Russian text.
        - slow (several times)
        <br> _ _
    - python -m deeppavlov install deeppavlov/configs/squad/multi_squad_ru_retr_noans_rubert_infer.json - With " no " field answer "
        - python -m deeppavlov download deeppavlov/configs/squad/multi_squad_ru_retr_noans_rubert_infer.json
        - python -m deeppavlov interact deeppavlov/configs/squad/multi_squad_ru_retr_noans_rubert_infer.json
        - works strangely, I found a few pieces very well (exactly the file I needed), but in some places I didn’t find anything at all (although for past neurons this was the easiest place) `< br >`
    - python -m deeppavlov install deeppavlov /configs/squad/ multi_squad_noans.json - With " no " field answer "
        - python -m deeppavlov download deeppavlov /configs/squad/ multi_squad_noans.json
        - python -m deeppavlov interact deeppavlov /configs/squad/ multi_squad_noans.json
        - 1.2-1.3 times faster.
        - Works well with English words.
        - Does not work with Russian.
        - Failed to force return "no response"
        <br> _ _
    - python - m deeppavlov install deeppavlov / configs / squad / multi _ squad _ ru _ retr _ noans . json                    - With "no answer" field
        - python -m deeppavlov download deeppavlov /configs/squad/ multi_squad_ru_retr_noans.json
        - python -m deeppavlov interact deeppavlov /configs/squad/ multi_squad_ru_retr_noans.json
        - Didn't find anything at all.
<br> <br> _ _ _

2. ** Frequently Asked Questions **
   Responsible for classifying incoming questions.
<br> <br> _ _ _

3. **Classification**
    + RuSentiment dataset contains sentiment classification of social media posts for Russian language within 5 classes 'positive', 'negative', 'neutral', 'speech', 'skip'. <br> _ _
    - python -m deeppavlov install deeppavlov /configs/classifiers/ rusentiment_convers_bert.json
        - python -m deeppavlov download deeppavlov /configs/classifiers/ rusentiment_convers_bert.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers / rusentiment_convers_bert.json - works but much Not recognizes . Good for well built applications with explicit connections.
        - 0.01   200 symbols
        - 0.01   900 symbols
        - 0.02   2400 symbols
        - 0.03 9500 symbols
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/classifiers/ yahoo_convers_vs_info_bert.json
        - python -m deeppavlov download deeppavlov /configs/classifiers/ yahoo_convers_vs_info_bert.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers/ yahoo_convers_vs_info_bert.json - works , 1 - colloquial rech . 0 - informational
        - 0.01 200 symbols
        - 0.01 900 symbols
        - 0.02 2400 symbols
        - 0.04 9500 symbols
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/classifiers/intents_dstc2_bert.json - english model 28 classes ( requests )
        - python -m deeppavlov download deeppavlov /configs/classifiers/intents_dstc2_bert.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers/intents_dstc2_bert.json - works but strange . Russian does not support. Doesn't work well in English either.
        - 0.01   200 symbols
        - 0.01   900 symbols
        - 0.02   2400 symbols
        - 0.04 9500 symbols
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/classifiers/ relation_prediction_rus.json
        - python -m deeppavlov download deeppavlov /configs/classifiers/ relation_prediction_rus.json - 6.59 GB weighs
        - python -m deeppavlov interact deeppavlov /configs/classifiers/ relation_prediction_rus.json - Started , but after input text Not could work - apparently memory Not enough , computer stuck , had to restart .
        - earned on the script version
        - 0.001   200 symbols
        - 0.001   900 symbols
        - 0.002   2400 symbols
        - 0.004   9500 symbols
        - produces results like this: [' P 178', ' P 495', ' P 57', ' P 19', ' P 27'] and a large table with normalized values
        - 246 classes
        - I can't find what it is. Most likely this is a predictor to which branch of the wiki the text belongs.
        <br> _ _
    - python - m deeppavlov install deeppavlov / configs / classifiers / query_pr . _ _ json                      - most likely an English model. classifying questions or requests into categories. You can train (you only need csv files like (QUESTION, CATEGORY_NUMBER))
        - python -m deeppavlov download deeppavlov /configs/classifiers/ query_pr.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers/ query_pr.json - Like How works on _ Russian . But it works weird. It is not clear what kind of classes - I can’t find it yet (most likely the one below).
        - 0.001 200 symbols
        - 0.001 900 symbols
        - 0.002 2400 symbols
        - 0.005 9500 symbols
        + Possible classes :
        + Complex questions with numerical values:
        + “What position did Angela Merkel hold on November 10, 1994?”
        + Complex question where the answer is number or date:
        + “When did Jean-Paul Sartre move to Le Havre?”
        + Questions with counting of answer entities:
        + “How many sponsors are for Juventus FC?”
        + Questions with ordering of answer entities by ascending or descending of some parameter:
        + “Which country has the highest individual tax rate?”
        + Simple questions:
        + “What is crew member Yuri Gagarin's Vostok?”
<br> <br> _ _ _

4. ** Paraphrase **
    + Returns 1 if two texts are the same in meaning, just one text is paraphrased relative to the other.
    + Application - you can see the similarity of requests\letters\messages and any other texts. <br> _ _
    - python -m deeppavlov install deeppavlov /configs/classifiers/paraphraser_rubert.json
        - python -m deeppavlov download deeppavlov /configs/classifiers/paraphraser_rubert.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers/ paraphraser_rubert.json - works pretty well Not bad but _ necessary stronger check
        - 0.01   200 symbols
        - 0.01   900 symbols
        - 0.02   2400 symbols
        - 0.03   9500 symbols
        - It may not work with English, or it works a little differently with it
        - in some places it says that there is a paraphrase when it is not there (for large texts)
        - seems to be highly responsive if there are similar symbols ( docker and Document )
        <br> _ _
    - python -m deeppavlov install deeppavlov/configs/classifiers/entity_ranking_bert_rus_no_mention.json
        - python -m deeppavlov download deeppavlov/configs/classifiers/entity_ranking_bert_rus_no_mention.json
        - python -m deeppavlov interact deeppavlov/configs/classifiers/entity_ranking_bert_rus_no_mention.json
        - also a paraphrase, but outputs [ x , y ], where x + y = 1, and x is the confidence that it is not a paraphrase, and y is that it is a paraphrase. Works good. Apparently it works on extracting entities.
        - 0.01   200 symbols
        - 0.01   900 symbols
        - 0.02   2400 symbols
        - 0.04   9500 symbols
        <br> _ _
    - python - m deeppavlov install deeppavlov / configs / classifiers / rel _ ranking _ bert _ rus . json                  is a model for ranking candidate relationship paths for a question.
        - python -m deeppavlov download deeppavlov /configs/classifiers/ rel_ranking_bert_rus.json
        - python -m deeppavlov interact deeppavlov /configs/classifiers/ rel_ranking_bert_rus.json
        - works, same as above [ x , y ], where x + y = 1, and x is the confidence that it is not a paraphrase, and y is that it is a paraphrase. Works good. Apparently it works on extracting entities.
        - 0.01   200 symbols
        - 0.01   900 symbols
        - 0.02   2400 symbols
        - 0.04   9500 symbols
        - Works pretty well, but there is a problem with the fact that it does not know the identity of the Russian and English concepts (Docker and docker , At the same time, it recognizes English words separately.
<br> <br> _ _ _

5. **Named Entity Recognition (NER)**
    + Multilingual arrangement tags :
        + PERSON People including fictional
        + NORP Nationalities or religious or political groups
        + FAC Buildings, airports, highways, bridges, etc.
        + ORG Companies, agencies, institutions, etc.
        + GPE Countries, cities, states
        + LOC Non-GPE locations, mountain ranges, bodies of water
        + PRODUCT Vehicles, weapons, foods, etc. (Not services)
        + EVENT Named hurricanes, battles, wars, sports events, etc.
        + WORK OF ART Titles of books, songs, etc.
        + LAW Named documents made into laws
        + LANGUAGE Any named language
        + DATE Absolute or relative dates or periods
        + TIME Times smaller than a day
        + PERCENT Percentage (including “%”)
        + MONEY Monetary values, including units
        + QUANTITY Measurements, as of weight or distance
        + ORDINAL “ first”, “second”
        + CARDINAL Numerals that do not fall under another type < br >
    - python -m deeppavlov install deeppavlov /configs/ ner / ner_ontonotes_bert_mult.json
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_ontonotes_bert_mult.json
        - python -m deeppavlov interact DeepPavlov / deeppavlov /configs/ ner / ner_ontonotes_bert_mult.json
        - takes off. Installed via manual download. Reinstalled. Works. Defines very well both Russian and English.
        - Swears if more than 512 tokens (fixed by breaking text on lines)
        - 0.07 200 symbols
        - 0.1 900 symbols
        - RAM 6 GB
        - 1-6 GB GPU
        - Not everything is found for Russian-language names (MAOU. Gymnasium. They studied at Lomonosov Moscow State University. - does not find)
        <br> _ _
    + Russian tagging:
        + PER
        + LOC
        + ORG
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/ ner / ner_rus.json
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_rus.json
        - python -m deeppavlov interact deeppavlov /configs/ ner / ner_rus.json
        - takes off. Reinstalled, still crashes. Swears that vocabularies are not loaded
        <br> _ _
    - python - m deeppavlov install deeppavlov / configs / ner / ner _ rus _ bert _ torch . json                - puts anew transformers and torch =1.6 stopped at the installation of the torch. Changed in the end torch .
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_rus_bert_torch.json
        - python -m deeppavlov interact deeppavlov /configs/ ner / ner_rus_bert_torch.json
        - gave an error that something could not be found, I'll try to install all the same. After the rearrangement, it loads a bunch of everything (mb when downloading due to the fact that something was not installed, it did not load)
        - works but determines the average. + defines only 3 (name, organization, place)
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/ ner / ner_kb_rus.json
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_kb_rus.json - hung from the very beginning on chala on loading Nothing Not displays in general . Apparently it's just a bug. Or something else. Downloaded in the end.
        - python -m deeppavlov interact deeppavlov /configs/ ner / ner_kb_rus.json
        - Swears that the vocabulary is not loaded. At the same time, according to the above commands, everything is installed and loaded. It is possible that this error is with those who do not have pre-trained models.
        <br> _ _
    - python - m deeppavlov install deeppavlov / configs / ner / ner _ few _ shot _ ru . json                    - A thing that should be quickly trained on small volumes
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_few_shot_ru.json
        - python -m deeppavlov interact deeppavlov /configs/ ner / ner_few_shot_ru.json
        - ' NoneType ' object has no attribute 'predict' ( rather Total her train need )
        <br> _ _
    - python -m deeppavlov install deeppavlov /configs/ ner / ner_bert_ent_and_type_rus.json - Flag entities . O-TAG - for other tokens, E-TAG - for entities, T-TAG corresponds to tokens of entity types
        - python -m deeppavlov download deeppavlov /configs/ ner / ner_bert_ent_and_type_rus.json
        - python -m deeppavlov interact DeepPavlov / deeppavlov /configs/ ner / ner_bert_ent_and_type_rus.json
        - Breaks the sentence into words, punctuation marks, etc. Returns an Array with broken entities and a separate array [a, b, c], where a b c is float 32 totaling 1 most likely. 1 of the 3 above classes.
        - 0.07   200 symbols
        - 0.1   900 symbols
        - RAM 6 GB
        - 1-6 GB GPU
        <br> _ _
    + With text splitting on lines:
        + 0.45   200 symbols
        + 0.88   900 symbols
        + 2.21   2400 symbols
        + 8.00   9500 symbols
    - There is an idea with splitting tags into levels: level 1 ( ner _ ontonotes _ bert _ mult ) is stronger, level 2 ( ner _ bert _ ent _ and _ type _ rus ) is more extended.
<br> <br> _ _ _

6. ** entity linking **
    + Entity linking is the task of matching words from text (such as names of people, locations, and organizations) with entities from the target knowledge base (in our case, Wikidata).
    + Entity Component Linking takes the following steps:
        - a substring found using NER (English) or NER (Russian) is fed into the TfidfVectorizer and the resulting sparse vector is converted to a dense one.
        - The Faiss library is used to find k nearest neighbors for a tf - idf vector in a matrix where rows correspond to tf - idf vectors of words in entity headers.
        - entities are ranked by the number of relationships in Wikidata (the number of outgoing node edges in the knowledge graph).
        - BERT (English) or BERT (Russian) is used to rank entities by the description of the entity and the sentence in which it is mentioned.
        <br> _ _
    - python -m deeppavlov install entity_linking_rus
        - python -m deeppavlov download entity_linking_rus - a bunch Total downloading ...
        - python -m deeppavlov interact entity_linking_rus
        - the first time I launched it and the process hung and died. second time too MB not enough RAM or video memory ( About 16 GB of RAM required . - specified in TF - IDF Ranker )
        - With the script version, it also freezes (the whole computer).
<br> <br> _ _ _

7. **Spelling correction**
    - python -m deeppavlov install deeppavlov/configs/spelling_correction/levenshtein_corrector_ru.json
        - python -m deeppavlov download deeppavlov/configs/spelling_correction/levenshtein_corrector_ru.json
        - python -m deeppavlov interact deeppavlov/configs/spelling_correction/levenshtein_corrector_ru.json
        - Works average. Looks at each word individually. But good for basic stuff.
        - English ignores. But at the same time, it corrects punctuation (when it is written together) and capital letters.
        - slow
        - 0.1 200 symbols
        - 0.3 900 symbols
        - 1.0 2400 symbols
        - 8.5 9500 symbols
<br> <br> _ _ _

8. **Morphological Tagging**
    - python -m deeppavlov install deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
        - python -m deeppavlov download deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
        - python -m deeppavlov interact deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
        - Error . input sequence after bert tokenization shouldn't exceed 512 tokens. Error .
        - 0.02 200 symbols
        - 0.04 900 symbols
        - 2 GB RAM
        - 1-6 GB GPU
        - Works good .
        - English words recognizes as foreign.
<br> <br> _ _ _

9. **Syntactic parsing**
    - python -m deeppavlov install deeppavlov /configs/syntax/syntax_ru_syntagrus_bert.json
        - python -m deeppavlov download deeppavlov /configs/syntax/syntax_ru_syntagrus_bert.json
        - python -m deeppavlov interact deeppavlov /configs/syntax/syntax_ru_syntagrus_bert.json
        - It seems to work with English too
        - Punctuation Same analyzed
        - Error . input sequence after bert tokenization shouldn't exceed 512 tokens. Error.
        - 0.09   200 symbols
        - 0.15   900 symbols
<br> <br> _ _ _

10. ** DSL **
   DSL implementation . The DSL makes it easy to create user-defined skills for conversational systems.
   In the case where the DSL skill has matched the statement and found an answer, it returns the answer with a confidence value.
   If there is no match, the DSL skill returns the argument on _ invalid _ command ("Sorry, I misunderstood you" by default) as a proposition and sets the confidence interval to the null _ confidence attribute (default 0).
<br> <br> _ _ _

eleven. **Knowledge Base Question Answering (KBQA)**
The knowledge base:
    is a comprehensive repository of information about a given domain or multiple domains.
   reflects the ways in which we model knowledge about a given subject or subjects in terms of concepts, entities, properties, and relationships.
   allows us to use this structured knowledge where it is needed, for example, answering factoid questions
    The question answerer :
   validates questions against a preconfigured list of question templates, disambiguates entities using entity binding, and answers questions posed in natural language.
   can be used with Wikidata (English, Russian) and (in future versions) custom knowledge graphs.
<br> <br> _ _ _

12. ** Telegram integration **
   Any model specified in the DeepPavlov config can be launched as a Telegram bot . You can do this with the command line interface or with Python.
python -m deeppavlov telegram < config_path > [-t < telegram_token > ] [ -d ]
<br> <br> _ _ _

#### Options for the classifier:
1. Based on Entities (build a database as it is now for confl and then compare the number of certain entities in the request with the entities in the database for each of the pages)
    - the relationship of some tags to others?
2. Based on paraphrase:
    - all text
    - paragraphs
    - offers
    - Here, in general, you need to see how to use a paraphrase also for confl
    - Most likely it will be slow if all texts are analyzed
    - Or use the extraction of tags and paraphrase on them
    - Paraphrasing for text tags (to combine similar words or even the same words just in different forms)
    - ? use to find synonyms for tags (one to the other)? Can help implement a dictionary system
3. Teaching a BERT neuron for classification (you can see the query classification of questions and other classifiers - based on some additional training)
4. An ensemble of neurons (several neurons connected by 1 neuron that is learning. Some can also learn in advance or together)
5. Based on QA - but I don't know how it will work with large texts, and it will be slow.

+ Decisions when there is a teacher:
    + Suggested solution for frequently changing data (both classes and data itself) and when there is little data:
        - Analysis of all texts on the Entity ( NER , Entity , Text (+ paraphrase? to combine similar ones)).
        - Compilation of the Dictionary based on Essences. Normalized, compressed (it is necessary to see which is faster) dictionary.
            - Will take into account the number of a certain tag in a certain text.
            - Vector representation? for additional tag relationship?
        - Teaching ML classification models based on vocabulary, supervised method. You can use DL or distance in general - the main thing is that it learns quickly and automatically.
        - Analysis of the request text for Entities ( NER , Entity , Text (+ paraphrase? to combine similar ones)).
        - Compilation of an instance according to the Dictionary for a query based on Entities (+paraphrase? replacing entities if they are not found in the Dictionary).
        - Classification of an instance ( ML , DL , distance - everything is possible, the main thing is that it will be faster both in training and in application).
        - After the request is successfully closed (service completed \ request processed successfully), the request with all tagged tags is added to the common database.
        - When a given number of new requests accumulate (or if some requests are deleted or become invalid for some reason), or after a period of time has elapsed:
            - The dictionary is compiled anew, taking into account the Essences of new requests.
            - The model is retrained on the new Dictionary (it would be nice to see the possibilities of reinforcement learning in this topic).
        - The cycle repeats over and over again.

    + Proposed solution for a permanent system (classes are constant, there is enough data for a qualitative model):
        - A dictionary is compiled from all the words of all texts.
            - Vector representation taking into account relationships.
            - Form (compressed, discharged) for the most accurate calculation.
        - Dictionary is used to train DL Models for classification. With a teacher.
        - A copy is made according to the Dictionary for the request text.
        - Calculation of the DL Model for instance classification.
        - Collection of statistics of successful / unsuccessful classifications.
        - In case of unsatisfactory statistics, retraining of the Model taking into account new data.
            - It is assumed that new classes will not be added or will be added very rarely (with training data available for them).

+ Solutions when there is no teacher:
    + Clustering - it's hard to come up with a direct application yet.
        - Separate clustering of texts in the database and further classification of the new text according to the resulting classes (to narrow the search circle). Or taking into account (that is, initially the texts are classified with the teacher) and both.
    + Minimum teacher:
        - The minimum number of texts with known classes. Rather, it will be used to pull out tags and search for a match (distance or paraphrase) on them.
        - Specially manually listed tags for the class. Or a set of unrelated texts, but describing the topic of the class (they may not be requests, but simply separate texts). Also in the future search by tags.


- use meta-information (number of sentences, number of unique words per total number of words, number of paragraphs, total number of words, etc. for each page)
- TF - IDF for all tags (tag uniqueness) - that is, there can be only 1 tag, but it can only be present in one document and for it it can be very meaningful. In fact, this is automatically taken into account in Entity Search, since AND is used - that is, it is necessary that each query tag be present in the document. And if only 1 document has a certain tag, then when you enter it during the Search, only this document will be displayed.
- graph analysis (tag relation, different centralities)



### Installation:

1.  Install conda.<br> <br>

2.  Install CUDA if a graphics card will be used. <br> <br>

3.  Clone repo <br> <br>

4.  `GPU` Option via full installation of the whole environment (for GPU):
    - `conda env create --file environment.yml`
    - `conda activate deeppavlov`
<br> <br>

5.  `CPU` - manual installation (you can also manually install for the GPU):
    + CPU:
        - `conda create -n deeppavlov python=3.7 tensorflow=1.15`
        - `conda activate deeppavlov`

    + GPU:
        - `conda create -n deeppavlov python=3.7 tensorflow-gpu=1.15`
        - `conda activate deeppavlov`
 <br> <br>

6.  Install dependencies :
    - ` sudo apt-get install libxml2-dev libxslt-dev python-dev`
    - `pip install lxml ` - html parser
    - `pip install requests`
    - `pip install fast-autocomplete[levenshtein]` - fast search using_metrics distances ( Levenshtein ).
    - pip_install rich ` - for the progress bar in the console
    - `cd DeepPavlov `
    - `pip install -e .`
    - `cd .. `
    - `cd networkx `
    - `pip install -e .`
    - `cd..`
    - To visualize the graph of the analyzed database:
        - pip_install plotly`_         - the main tool for generating html with a graph image.
        - pip_install pyvis`_         - optional (just another display option, but less informative, but with the presence of physics and settings during the show)
        - pip_install python - igraph`_ - used to apply various models for calculating the coordinates of graph nodes.
        - `pip install numpy ==1.19`
<br> <br>

7. Download and install neurons:
   - ` cd DeepPavlov` _

   + MorphoTager (morphological tagging):
     - Install dependencies:
       * `python -m deeppavlov install deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json`

     - Downloading neurons and additional parts, if necessary:
       * `python -m deeppavlov download deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json`

     - Interactive mode through console :
       * `python -m deeppavlov interact deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json`
       <br> <br>

    + NER (Entity Recognition in Sentences - Names, Organizations, Places, etc.):
        - Installation dependencies :
            * `python -m deeppavlov install deeppavlov /configs/ ner/ner_ontonotes_bert_mult.json `

        - Downloading neurons and additional parts, if necessary:
            * `python -m deeppavlov download deeppavlov /configs/ ner/ner_ontonotes_bert_mult.json `

        - Interactive mode through console :
            * `python -m deeppavlov interact deeppavlov /configs/ ner/ner_ontonotes_bert_mult.json `
    <br> <br>

    + Entity (Tagging entities. O-TAG - for other tokens, E-TAG - for entities, T-TAG corresponds to tokens of entity types):
        - Installation dependencies :
            * `python -m deeppavlov install deeppavlov /configs/ ner/ner_bert_ent_and_type_rus.json `

        - Downloading neurons and additional parts, if necessary:
            * `python -m deeppavlov download deeppavlov /configs/ ner/ner_bert_ent_and_type_rus.json `

        - Interactive mode through console :
            * `python -m deeppavlov interact deeppavlov /configs/ ner/ner_bert_ent_and_type_rus.json `
    <br> <br>

    + QA (Finding an answer to a question in context):
        - Install dependencies:
            * ` python - m deeppavlov install deeppavlov/configs/squad/squad_ru_rubert_infer.json` _

        - Downloading neurons and additional parts, if necessary:
            * `python -m deeppavlov download deeppavlov /configs/squad/ squad_ru_rubert_infer.json `

        - Interactive mode through console :
            * `python -m deeppavlov interact deeppavlov /configs/squad/ squad_ru_rubert_infer.json `
    <br> <br>

    + All other models can be installed if needed using the commands in the ` TEST.md ` or from the DeepPavlov website :
        - [ DeepPavlov ]( http://docs.deeppavlov.ai/en/master/features/overview.html ) Features:
            * `http://docs.deeppavlov.ai/en/master/features/overview.html`
<br> <br>

8.  Edit the file ` config.json` : _
    - `cp config.json .sample config.json`
    - In the copied `config.json` to set the appropriate parameters.
<br> <br>

9. Launch (options):

    1. Use the file `TEST.md` to search for available commands for interactive mode for a certain neuron through the console, and fill in if necessary (and tests).

    2. Run through an activated conda environment :
        * `conda activate deeppavlov`
        * `python COMMAND`
        - The launch can be long, as neurons are loaded.
        - Completion and exit:
            * `q` send to console while interactive or ` CTRL + C ` without waiting for completion.

        - Loading page url from specified in `config.json`_url confluence :
            - `python scripts/ConfluencePageLoader.py`
        - Morphological analysis of loaded pages:
            - `python scripts/MorphoTager.py`
        - Analysis of all previously loaded pages ( NER , Entity ):
            - `python scripts/NER.py`
        - Interactive search mode:
            - `python scripts/FastFinder.py` - no QA model (+ AND mode for searching).
            - `python scripts/FastFinder.py - qa ` - with QA model.(+ OR mode for search)
        - Visualization of the graph of the current database:
            - ` python scripts/GraphVisualizer.py`

        + If you need to create a new database, then you need to edit `config.json` with a new path `path_save` .

<br> <br>

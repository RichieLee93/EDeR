# EDeR
EDeR: A Dataset for Exploring Event Dependency Relations Between Events (submitted to SIGIR2023 Resource Paper Track)

EDeR is a human-annotated dataset that extracts event dependency information from events and provides refined semantic role-labelled event representations based on this information. We also provide the code of related baseline models for further research.

# Dataset statistics

<table>

  <tr>
    <td></td>
    <td colspan="2"><b>argument</td>
    <td colspan="2"><b>non-argument</td>
    <td><b>overall</td>
  </tr>
  <tr>
    <td></td>
    <td><b>required</td>
    <td><b>optional</td>
    <td><b>condition</td>
    <td><b>independent</td>
    <td></td>
  </tr>
  <tr>
    <td><b>train</td>
    <td>4096</td>
    <td>2837</td>
    <td>335</td>
    <td>1861</td>
    <td>9129</td>
  </tr>
  <tr>
    <td><b>dev</td>
    <td>635</td>
    <td>421</td>
    <td>41</td>
    <td>355</td>
    <td>1452</td>
  </tr>
  <tr>
    <td><b>test</td>
    <td>594</td>
    <td>368</td>
    <td>70</td>
    <td>239</td>
    <td>1271</td>
  </tr>
  <tr>
    <td><b>overall</td>
    <td>5325</td>
    <td>3626</td>
    <td>446</td>
    <td>2455</td>
    <td>11852</td>
  </tr>
</table>

# Data format
```data/train.json```, ```data/dev.json``` and ```data/test.json``` are the training, development and test sets, respectively. After loading each file, you will get a list of dictionaries. The format of the data is shown as the following example:
```
{'Event 1': "We {V: know} you teach the truth about God 's way",
 'Event 2': "you {V: teach} the truth about God 's way",
 'refined Event 1': NAN,
 'label': 'required argument',
 'Event 1 SRL': '{'ARG0': ['We'], 'V': ['know'], 'ARG1': ['you', 'teach', 'the', 'truth', 'about', 'God', "'s", 'way']}',
 'Event 2 SRL': '{'ARG0': ['you'], 'V': ['teach'], 'ARG1': ['the', 'truth', 'about', 'God', "'s", 'way']}',
 'sentence': '['We', 'know', 'you', 'teach', 'the', 'truth', 'about', 'God', "'s", 'way', '.']',
 'Event-Event span': "We {V: know} you teach the truth about God 's way[SEP]you {V: teach} the truth about God 's way",
 'Event-Event-SRL': "We {V: know} you teach the truth about God 's way[SEP]you {V: teach} the truth about God 's way[SRL]ARG1",
 'Event-Event-SRL-DEP': "We {V: know} you teach the truth about God 's way[SEP]you {V: teach} the truth about God 's way[SRL]ARG1[DEP]parataxis",
 'Marked-predicate sentence': "We [V1] know [\V1] you [V2] teach [\V2] the truth about God 's way ."}
```
``` Event 1 and Event 2 ``` are the containing and contained event pair.
``` refined Event 1 ```  is the refined Event 1 if label is condition or independent. Otherwise, it is NAN. 
```Event 1 SRL``` and ```Event 2 SRL``` are semantic role labels of the two events, respectively. 
```sentence``` is the tokenized sentence that contains the two events. The four types of inputs are also included, details can be found in the paper.

# Baseline models
## Requirements
Python 3.7+

transformers==4.16.2

scikit-learn==1.0.1

pytorch-lightning==1.5.10

pandas==1.3.5

pycorenlp==0.3.0

[Stanford CoreNLP tookit](https://stanfordnlp.github.io/CoreNLP/download.html)
 
## Train and test baseline models

You can find the command lines to train and test baseline models on a small sample data in `run_sample.sh`.

Here are some important parameters:

* `--m`: name of the selected model, e.g., roberta.
* `--i`: input type, e.g., Event-Event-SRL-DEP.
* `--t`: task type, binary or three.



 # Citing us

If you feel the dataset helpful, please cite:

```  
We will upload the paper to arXiv soon.
```

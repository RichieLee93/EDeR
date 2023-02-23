##!/bin/bash

python event_relation_classification_mix.py --m 'roberta' --i 'Marked-predicate sentence' --t 'binary'

python event_relation_classification_gpt2.py --i 'Marked-predicate sentence' --t 'three'

python event_relation_classification_distilbert.py --i 'Event-Event-SRL-DEP' --t 'three'

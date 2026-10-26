# Real Amazon Polarity results

All three architectures completed the same 20,000-review experiment. MLP (TF-IDF) had the highest positive-class F1 in this run (0.8829).

| model | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| MLP (TF-IDF) | 0.8820 | 0.8760 | 0.8900 | 0.8829 |
| TextCNN | 0.8540 | 0.8371 | 0.8790 | 0.8576 |
| BiLSTM | 0.8610 | 0.8588 | 0.8640 | 0.8614 |

## Common experimental protocol

Source: [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity/tree/9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289), revision `9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`. Label 0 is negative; label 1 is positive. The subset has 20,000 distinct cleaned reviews, balanced across labels. Empty, duplicate, and conflicting cleaned texts are excluded before splitting. All models share source review IDs, cleaning, labels, and the seed-42 stratified 80/10/10 split (16,000/2,000/2,000 for the default 20,000-row run).

TF-IDF unigrams and bigrams (up to 20,000 features) are fitted on training reviews only for the MLP. The shared training-only tokenizer produces 100-token sequences and learned 100-dimensional embeddings for TextCNN and BiLSTM. Long reviews are truncated, which can remove sentiment cues near the end. BiLSTM masks padded tokens.

Both candidates for every model used batch size 32, Adam with weight decay, at most 8 epochs, and early stopping on validation loss with patience 2. The identical search grid changes dropout and learning rate. The best checkpoint and candidate are selected only by validation loss. Each selected model is then evaluated once on the held-out test set; the decision threshold is fixed at 0.5. Precision, recall, and F1 refer to the positive class.

## Validation selection and overfitting

| model | candidate | epochs | best_epoch | validation_loss | train_accuracy | validation_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| MLP (TF-IDF) | regularized | 3 | 1 | 0.2627 | 0.8342 | 0.8910 |
| TextCNN | standard | 4 | 2 | 0.2969 | 0.9306 | 0.8750 |
| BiLSTM | standard | 3 | 1 | 0.2802 | 0.8385 | 0.8810 |

Early stopping and best-checkpoint restoration prevent reporting a later, worse validation epoch. They do not eliminate overfitting: the training/validation accuracy gap at the selected epoch remains a useful diagnostic. Histories for both candidates are retained under `models/`; test scores were not used to change the grid or choose epochs.

## Architectural interpretation

The MLP sees sparse unigram and bigram sentiment cues throughout the review but has limited explicit order information. TextCNN pools local patterns from the first 100 tokens. BiLSTM models sequential context in both directions over those same tokens. Learned embeddings start from scratch, and the dataset size, sequence truncation, optimizer settings, and training dynamics can all affect the ranking. This single-seed run is evidence about these configurations; it does not establish that one architecture is universally superior.

## Failure analysis

Every test prediction and misclassified review is exported with source row ID, raw review, true label, predicted label, and positive probability. Negation, mixed-word-polarity, and possible-sarcasm flags are nonexclusive heuristic review aids, not ground-truth annotations. In particular, the sarcasm error rate cannot be inferred from these cue counts.

### MLP (TF-IDF)

Misclassified 236 of 2000 test reviews.

| category | reviews | errors | error_rate |
| --- | --- | --- | --- |
| negation_cue | 1352 | 168 | 0.1243 |
| mixed_sentiment_cue | 220 | 20 | 0.0909 |
| sarcasm_cue | 18 | 4 | 0.2222 |

- Source row 2005780: true=1, predicted=0, P(positive)=0.0110. Review excerpt: Repetitive action, so-so graphics, linear plot. i would not recommend this game at all. i think the creators of The game put no work into this game. The environment is not interactive and gameplay is not very flexible. The plot is too linear, with only one way to solve a certain puzzle. Because of this, many of the puzzles are almost unsolvable, with only a single solution. I dropped the game at the third world solely for this reason. Battle sequences occur superfluously and with too many of the
- Source row 3232440: true=1, predicted=0, P(positive)=0.0211. Review excerpt: So for so good. Just recently I bought this phone. I received it 2 weeks from the date I order. Amazon did not ship until 9 days, though it says "usually ships in 24 hrs" . But it did not bother me though.The phone looks sleek and sexy. Everyone around me think it is coolest phone ever. I am getting compliments from all over.I did not have any problems, just opened the box, charged the phone, put in the SIM and with in a minute started talking.The phone menu is little confusing, but once you get
- Source row 2270055: true=1, predicted=0, P(positive)=0.0262. Review excerpt: Pretty complete for a 'Nutshell' book. This is the second book I've bought on ASP. Being new to the subject, I thought I'd be better off with a more comprehensive offering.Was I wrong. With my shallow VB knowledge I had to do a lot of cross-reading which is the worst thing happening to someone trying to develop an application.This book however lays it on gently. I do not agree that beginners should steer clear of this title, perhaps they should allow themselves more time to assimilate the materi

Full records: `evaluation/mlp/predictions.csv`, `misclassified.csv`, `failure_examples.csv`, `classification_report.json`, and `confusion_matrix.csv`.

### TextCNN

Misclassified 292 of 2000 test reviews.

| category | reviews | errors | error_rate |
| --- | --- | --- | --- |
| negation_cue | 1352 | 224 | 0.1657 |
| mixed_sentiment_cue | 220 | 33 | 0.1500 |
| sarcasm_cue | 18 | 2 | 0.1111 |

- Source row 2005780: true=1, predicted=0, P(positive)=0.0045. Review excerpt: Repetitive action, so-so graphics, linear plot. i would not recommend this game at all. i think the creators of The game put no work into this game. The environment is not interactive and gameplay is not very flexible. The plot is too linear, with only one way to solve a certain puzzle. Because of this, many of the puzzles are almost unsolvable, with only a single solution. I dropped the game at the third world solely for this reason. Battle sequences occur superfluously and with too many of the
- Source row 1934203: true=1, predicted=0, P(positive)=0.0050. Review excerpt: Much better than a feather duster. Whereas a standard feather duster (or even a wet rag) can kick up lots of dust into the air, this thing just sucks it up gone completely. I've run it over delicate electronics, picture frames, keyboards, tv screens, monitors, and bookshelves. It seems to effortlessly clean dust off everything without disrupting its placement or surface whatsoever. Minus one star for the high price.
- Source row 535360: true=0, predicted=1, P(positive)=0.9897. Review excerpt: Giant. The movie was great it brought back old memories of the first viewing years ago. The only problem I found was that the disc that the movie was on did not have any label stating that it was the movie and that it was part 1 and part 2 was on the other side. Other than that I was well pleased. Alton

Full records: `evaluation/textcnn/predictions.csv`, `misclassified.csv`, `failure_examples.csv`, `classification_report.json`, and `confusion_matrix.csv`.

### BiLSTM

Misclassified 278 of 2000 test reviews.

| category | reviews | errors | error_rate |
| --- | --- | --- | --- |
| negation_cue | 1352 | 204 | 0.1509 |
| mixed_sentiment_cue | 220 | 34 | 0.1545 |
| sarcasm_cue | 18 | 3 | 0.1667 |

- Source row 3258943: true=1, predicted=0, P(positive)=0.0063. Review excerpt: Not bad after rebate. Good but video is kind of slow. It's USB 1.0 not USB 2.0. If you have broadband spend a little bit more and get the Live! Pro USB 2.0.
- Source row 143935: true=1, predicted=0, P(positive)=0.0138. Review excerpt: NOT POISON. My initial impression was that Sayers gave far too much information on the trial of Harriet Vane. However, the witty characters of Lord Peter Wimsey and Miss Climpson caused me to become hooked.
- Source row 756579: true=0, predicted=1, P(positive)=0.9836. Review excerpt: Qué es esto?'???. Demasiado comercial, y con el productor terminaron de desenfocarse, el album es una pena, nada comparado con ningún otro albúm de la banda, incluso que everyday. Definitvamente trabajar con Mark Batson no funciona, los cambios son buenos pero cuando son para bien y en éste albúm creo que a varios y en mi opinión a mi, me ha dejado decepcionado. Y creo que la gente que hace una crítica buena debería de reaccionar y comparar....No habrá opción mas que esperar otros 2-3 años para 

Full records: `evaluation/bilstm/predictions.csv`, `misclassified.csv`, `failure_examples.csv`, `classification_report.json`, and `confusion_matrix.csv`.

## Memory use and reproducibility

The loader scans parquet in batches instead of converting the multi-million-row corpus to pandas. Float32 TF-IDF matrices remain CSR sparse; only one 32-row batch is densified at a time (about 2.44 MiB at 20,000 features). TensorFlow stages run in separate processes, releasing allocations between candidates. Sequence arrays are loaded with memory mapping. `data_audit.json` records sparse storage sizes, and `resource_usage.json` records measured stage memory and duration.

The dataset revision, sampled source IDs, split CSVs, source hashes, package versions, tuning histories, checkpoints, and predictions are saved. `--resume` verifies stage signatures and output hashes before reuse. Results can vary across library versions or hardware; repeated seeds and confidence intervals remain future work.

# Qualitative review of real test failures

This post-evaluation review examines nine real source reviews selected from the exported failures. It covers negation, mixed sentiment, ironic praise, target/context changes, domain wording, language limitations, and a suspected label/text conflict. It does not change labels, tune models, or revise any test score. Explanations describe the observed text and plausible difficulties; they are not causal feature attributions.

All three models misclassify 103 of the same test reviews. The union of their errors contains 470 reviews. The original positive/negative dataset labels are the evaluation ground truth.

The 2,000-row test set contains 1,352 negation-cue rows, 220 mixed-word-polarity rows, and 18 possible-sarcasm-cue rows. These groups overlap and are heuristic. They do not measure the prevalence or difficulty of genuine sarcasm, negation-scope errors, or mixed sentiment. In particular, an ordinary use of "as if" can be flagged while the ironic toy-ball review below is missed.

## Source row 3258943 Negation and qualified praise

The title says 'Not bad after rebate' and the review begins 'Good', then criticizes video speed and USB 1.0. All three predictions are negative despite the positive source label. This is a concrete negation-scope and qualified-praise challenge.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 1 | 0 | 0.1371 |
| textcnn | 1 | 0 | 0.0243 |
| bilstm | 1 | 0 | 0.0063 |

Review excerpt: Not bad after rebate. Good but video is kind of slow. It's USB 1.0 not USB 2.0. If you have broadband spend a little bit more and get the Live! Pro USB 2.0.

## Source row 191582 Past failures versus current product

The reviewer describes earlier remedies that did not work, then says this product reduced staining and recommends trying it. All three models predict negative. The relevant distinction is earlier failed treatments versus the successful reviewed product, rather than the presence of a negative word alone.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 1 | 0 | 0.0389 |
| textcnn | 1 | 0 | 0.0110 |
| bilstm | 1 | 0 | 0.3049 |

Review excerpt: It "Works"!. I'd tried pretty much everything on my pomapoo's staining problem. The cleaning pads, solutions and had even tried a homemade potion suggested on the internet, nothing worked (and my poor baby almost went BLIND!). Then I purchased this product and after about a week I'd noticed her staining was almost gone. Her eyes were clearer and even the daily cleaning was easier. Try this!

## Source row 317569 Mixed sentiment and final verdict

Initial praise for automatic clock setting is followed by an excessively bright display and a failed alarm. The negative source label matches the final product verdict. MLP and BiLSTM predict positive, while TextCNN predicts negative, showing that this failure is not universal across architectures.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 0 | 1 | 0.9310 |
| textcnn | 0 | 0 | 0.1287 |
| bilstm | 0 | 1 | 0.6509 |

Review excerpt: great promise, terrible delivery. I too liked the automatic clock setting feature. I like of all my timepieces having the "correct" time especially if there in areas of my home where changing the time is troublesome. Additionally, if you in in a building where the power was a little sketchy (power constantly flickering or going out), this clock and its price seemed too good as it would automatically set itself when the power was restored.Now for drawbacks... The LED is very bright! I have trouble sleeping with clock near my bed. (Hint: keep it a good five feet away from you.) You can literally see it across the street. It is that bright - eve

## Source row 58355 Different opinion targets

The negative phrase about a horrible sequel is followed by praise for the reviewed classic in HD. TextCNN predicts negative; MLP and BiLSTM predict positive. The contrasting targets are the sequel and the actual reviewed release.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 1 | 1 | 0.5749 |
| textcnn | 1 | 0 | 0.0219 |
| bilstm | 1 | 1 | 0.6904 |

Review excerpt: A neverending classic. Such a great movie didnt deserve such a horrible sequel!This release lacks some extra features, but viewing this classic in HD is just a treat.

## Source row 1876833 Ironic praise

The expensive toy ball is called an 'excellent deal', but the reviewer ends by saying they will pass. The praise is plausibly ironic in context. TextCNN and BiLSTM predict positive; MLP predicts negative. This clear review candidate is NOT caught by the simple sarcasm cue list, demonstrating that cue counts are not an estimate of sarcasm prevalence.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 0 | 0 | 0.3274 |
| textcnn | 0 | 1 | 0.6088 |
| bilstm | 0 | 1 | 0.9726 |

Review excerpt: The $43.20 amazing PINKY HI-Bounce. I find it hard to believe that everyone don't give this product a high rating. A toy ball with a list price of $43.20 must be one hell of a ball. I was thinking of buying this for my dogs to play but after what I have read, I think I will just pass on this excellent deal.

## Source row 2059653 Irony and mixed track opinions

The reviewer calls most songs unlistenable and jokes that they improve when the skip button is used, while praising two tracks and assigning two stars. All three models predict positive. Interpreting the ironic praise and separating a few good tracks from the overall album verdict is necessary.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 0 | 1 | 0.5563 |
| textcnn | 0 | 1 | 0.9794 |
| bilstm | 0 | 1 | 0.8276 |

Review excerpt: La la. The album title should be warning enough. Follow the chain of association:"City of Angels" = LA = la-la land = la laJon gets together with a good chunk of Toto to make a late-80's new-agey kind of deal, with the dreamy synths and world-music percussion and all. This CD should come with a freakin' magic crystal.Most of the songs are unlistenable. And yet, they do get better if you hit the skip button. "Top of the World" is a solid piece -- I think one of his best solo songs -- and "Hurry Home" ain't bad either.And that's the two stars right there.

## Source row 1934203 Domain meaning of a negative-looking word

The product 'sucks' dust up successfully and cleans without disrupting surfaces; only the high price loses a star. All three models predict negative despite the favorable review and positive source label. This is consistent with difficulty interpreting domain-specific wording, but no causal feature-attribution experiment was performed.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 1 | 0 | 0.0271 |
| textcnn | 1 | 0 | 0.0050 |
| bilstm | 1 | 0 | 0.1887 |

Review excerpt: Much better than a feather duster. Whereas a standard feather duster (or even a wet rag) can kick up lots of dust into the air, this thing just sucks it up gone completely. I've run it over delicate electronics, picture frames, keyboards, tv screens, monitors, and bookshelves. It seems to effortlessly clean dust off everything without disrupting its placement or surface whatsoever. Minus one star for the high price.

## Source row 756579 Language and preprocessing limitation

This Spanish review is negative about the album and says it left the writer disappointed. All three models predict positive. The shared ASCII-oriented cleaning removes accented characters, and the learned representations are not a multilingual language model. This is a limitation to study in a separate experiment, not a reason to change this test set after evaluation.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 0 | 1 | 0.9315 |
| textcnn | 0 | 1 | 0.9809 |
| bilstm | 0 | 1 | 0.9836 |

Review excerpt: Qué es esto?'???. Demasiado comercial, y con el productor terminaron de desenfocarse, el album es una pena, nada comparado con ningún otro albúm de la banda, incluso que everyday. Definitvamente trabajar con Mark Batson no funciona, los cambios son buenos pero cuando son para bien y en éste albúm creo que a varios y en mi opinión a mi, me ha dejado decepcionado. Y creo que la gente que hace una crítica buena debería de reaccionar y comparar....No habrá opción mas que esperar otros 2-3 años para ver que sucede con esta banda (otra producción)...espero se nivelen otra vez y definitivamente que ya no trabajen con Mark Batson.Muy Fan de DMB pero 

## Source row 2005780 Suspected source label and text conflict

The source label is positive, but the review says it would not recommend the game and would award zero stars. All models predict negative. This appears to be a label/text conflict rather than an unambiguous semantic model error. The original label was retained and the row remains counted as an error in every reported score.

| Model | Source label | Prediction | P positive |
| --- | --- | --- | --- |
| mlp | 1 | 0 | 0.0110 |
| textcnn | 1 | 0 | 0.0045 |
| bilstm | 1 | 0 | 0.0298 |

Review excerpt: Repetitive action, so-so graphics, linear plot. i would not recommend this game at all. i think the creators of The game put no work into this game. The environment is not interactive and gameplay is not very flexible. The plot is too linear, with only one way to solve a certain puzzle. Because of this, many of the puzzles are almost unsolvable, with only a single solution. I dropped the game at the third world solely for this reason. Battle sequences occur superfluously and with too many of the same enemies. The gummi ship graphics are embarassingly juvenile. Overall, if games could be awarded with zero stars, this would be one of them.

## Limitations and next experiments

This targeted review is not a random annotated sample and cannot estimate category-level error rates. Dataset labels may be noisy or disagree with literal review sentiment. Keep the current labels and fixed test set unchanged. Future experiments could use Unicode-preserving cleaning, longer sequences, explicit target/context analysis, and an independently annotated sarcasm/error sample; settings should be chosen using training/validation data or a new test protocol.

The complete test predictions and source texts are under evaluation/<model>/predictions.csv. The nine reviewed cases with all three predictions and these interpretations are in reviewed_failure_cases.csv.

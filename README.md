# Sound classifier
Using torch audio to try and understand how to build a simple classifier to predict whether a sound is a cat or dog

You can download the dataset [here](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs)

# TODO
- [x] EDA for sound - generate melspectogram
- [x] Train with simple cnn
- [x] Evaluate Model
- [x] Augment Data
- [x] Train with resnet50
- [x] Train CRNN Model
- [x] Add pytorch lighthing for sample usage
- [x] Add torch lighthing data module

# Model Results
| Model Name  | Accuracy | MCC  |
|-------------|----------|------|
| CNN Vanilla | 0.69     | 0.39 |
| CRNN        | 0.66     | 0.32 |
| Resnet 50   | 0.93     | 0.86 |

# References
- https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
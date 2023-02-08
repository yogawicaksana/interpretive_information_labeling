# Interpretive Information Labeling (Taiwan AICUP2022)

## Mission statement
- Given an input `(q, r, s)`, where `q` is a long article, `r` is short article responding to `q`, and `s` is the discussion relationship between `r` and `q`, which may be an aggrement or diasgreement (agree or disagree). 

- The output data is a double tuple (`q'`, `r'`). `q'` and `r'` are the subsequences of `q` and `r`, respectively.

- `q'` and `r'` provide key information to judge `q`. `r'` `r` presentes the relation of `s`.

- Our task is to predict the `q'` and `r'` given an input `q`, `r`, and `s`.

## Data Pre-processing

- The first data processing step simply removes any leading and trailing
character in the `q`, `r`, `q’`, and `r’` features. This step is important in order to make
the model really fed by clean data.
- We do the 50% upsampling for the training data in order to provide more data
to our model. We have tried another augmentation strategy such as using
synonym words from the existing data, but 50% upsampling works best for us.
This upsampling method is done by sampling 50% of the data with stated
random state for reproducibility. 
- We treat this Interpretive Information Labeling Project as a question answering
task. Hence, we need to design a system that can do the extraction in order to
make target labels. There is a slight modification from the traditional question
answering, since there is s feature, the discussion relationship between `r` and `q`.
In order to do that, we just place the `s` feature in front of the `r` feature with
format “`s:r`” to become a new `r` feature.
- The next step is to do the index extraction both for `q` and `r` with respect to `q’`
and `r’`.
- Our final features to train the model are `q`, `r`, `q’`, `r’`, `q_start`, `r_start`, `q_end`, and
`r_end`. The `*_start` and `*_end` features are the start and end index
as mentioned in the previous point.
- We split the final data with the percentage of 90:10 for the training and
validation data, respectively. - The next step is tokenizing the data with the help of BertTokenizer. To add the
tokenized version of start and end positions to the encoded features, we just
simply use the char_to_token function. If the `q` or `r` start position is nothing,
we just simply treat the `q` or `r` start and end position with 0. Finally, we got the
encoded features with keys of `input_ids`, `token_type_ids`, `attention_mask`,
`q_start`, `r_start`, `q_end`, and `r_end`. The final step is just simply transforming
the encoded features to tensor.

## Model Overview
![image](https://user-images.githubusercontent.com/62101089/217675952-0e33b3b4-e443-4e89-a402-d97c6d67672e.png)

- Our model architecture consists of BERT as a backbone, followed by
Bi-LSTM and Linear layer as a classifier. - First, BERT (bert-base-cased) works as a backbone for our model architecture
and then 2 layers of Bi-LSTM processes the encoded representation from
BERT. We argue that Bi-LSTM can better model the encoded representations
by BERT. This Bi-LSTM is outputting a dimension of 256. This output from
the Bi-LSTM layer is then fed to the dense layer with the dimension of 512.
The final output is the dense layer with 4 outputs, which are q_start, r_start,
q_end, and r_end.
- We use Cross Entropy Loss function with AdamW optimizer. We finetune the
model for 2 epochs with learning rate 3e-5. We just finetune with that number
of epochs since the model is starting to overfit if the number of epochs is
greater than 2. We use batch size equal to 8 for both training and validation
data.

## Result

| Public Leaderboard | Private Leaderboard |
|:------------------:|:-------------------:|
|0.803938            |0.854274             |

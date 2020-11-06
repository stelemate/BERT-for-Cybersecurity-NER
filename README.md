# BERT-for-Cybersecurity-NER
An implementation of BERT for cybersecurity named entity recognition

<strong>***** New November 5th, 2020: Named Entity Recognition Models *****</strong>
## Introduction
We design a few joint BERT models for cybersecurity named entity recognition. The BERT pretraining model is described in [google-research](https://github.com/google-research/bert).
These models are suitable for Chinese and English combined data, but other languages need to modify the DataProcessor code

### Named Entity Recognition(NER) task
A task to identify entities with specific meanings in the text, including names of people, locations, organizations, specific nouns, etc.

## Results
<table>
  <tbody>
    <tr>
      <td rowspan="1">
        <p>
          <strong>Metric</strong>
          <br></p>
      </td>
      <td colspan="1">
        <strong>accuracy</strong>
        <br></td>
      <td colspan="1">
        <strong>precision</strong>
        <br></td>
      <td colspan="1">
        <strong>recall</strong>
        <strong></strong>
        <br></td>
      <td colspan="1">
        <strong>f1-score</strong>
        <br></td>
    </tr>
    <tr>
      <td><strong>BERT finetuning</strong></td>
      <td>97.78</td>
      <td>88.73</td>
      <td>92.22</td>
      <td>90.44</td>
    </tr>
    <tr>
      <td><strong>BERT-CRF</strong></td>
      <td>97.53 <span>(<strong>-0.25</strong>)</span></td>
      <td>91.46 <span>(<strong>+2.73</strong>)</span></td>
      <td>87.67 <span>(<strong>-4.55</strong>)</span></td>
      <td>89.53 <span>(<strong>-0.91</strong>)</span></td>
    </tr>
    <tr>
      <td><strong>BERT-LSTM-CRF</strong></td>
      <td>98.13 <span>(<strong>+0.35</strong>)</span></td>
      <td>93.00 <span>(<strong>+4.27</strong>)</span></td>
      <td>93.09 <span>(<strong>+0.87</strong>)</span></td>
      <td>93.05 <span>(<strong>+2.61</strong>)</span></td>
    </tr>
    <tr>
      <td><strong>BERT-Bi-LSTM-CRF</strong></td>
      <td>98.23 <span>(<strong>+0.45</strong>)</span></td>
      <td>94.77 <span>(<strong>+6.04</strong>)</span></td>
      <td>92.97 <span>(<strong>+0.75</strong>)</span></td>
      <td>93.11 <span>(<strong>+2.67</strong>)</span></td>
    </tr>
    <tr>
      <td><strong>BERT-ID-CNN-CRF</strong></td>
      <td>98.18 <span>(<strong>+0.4</strong>)</span></td>
      <td>93.37 <span>(<strong>+4.64</strong>)</span></td>
      <td>93.07 <span>(<strong>+0.85</strong>)</span></td>
      <td>93.13 <span>(<strong>+2.69</strong>)</span></td>
    </tr>


  </tbody>
</table>


## Usage
First, download the BERT model from [google-search](https://github.com/google-research/bert). In this project, we choose the BERT-Base Chinese model as the pretraining model.
Clone this project and data_dir, bert_config_file, output_dir, init_checkpoint, vocab_file must be specified in bert_lstm_ner.py.

If you change the train/dev/test dataset, the structure of the dataset should be like this:


这 O
次 O
的 O
问 O
题 O
是 O
出 O
在 O
我 O
们 O
的 O
M B-SW
a I-SW
r I-SW
k I-SW
d I-SW
o I-SW
w I-SW
n I-SW
渲 O
染 O
中 O
。 O

And there must be a blank line between the two sentences, while the maximum length of a single sentence is [max_seq_length] which is defined in bert_lstm_ner.py.
These models can be trained in a few hours on a GPU, but maybe 1 day on CPU, starting from the exact same pre-training model.

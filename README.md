# README

Run the script ```compliance_assessment.py``` to produce Procedure 1 in the paper. This script, when given the path for the embeddings and the test queries pairs, calculates the ranks and produces a CSV file, scatter plot and histogram.

Run the script ```metrics.py``` to calculate the assessment metrics when the information of the ranks is given in form of previously produced CSV files.

## Usage of ```compliance_assessment.py```

```
python compliance_assessment.py 
--embedding_folder '/home/workspace/model_1/embeddings_pt/'
--output_folder '/home/workspace/model_1'
--pairs_file '/home/workspace/pairs/ctrl_hdlr.npy
--bin_width 10
--corpus_split 3
```

#### Input
- Embedding files of the test corpus.
- Program pairs from the test query corpus, either positive (benchmark set containing all known instances of the pattern) or negative (set containing pairs that do not implement the CH pattern)

#### Output
- **CSV file** - .csv that has the information of the rank for each individual pair
- **Histogram** - .png which illustrates a histogram of rank distribution of all the test query pairs. Such rank distribution depicts the frequency with which each rank in a set of programs appears. 
- **Scatter plot** - .png showing the rank of each pair during the assessment, provides a pair-specific view of the performance.

#### Parameters
- embedding_folder : folder where the embedding file stored in the form of '*.pt'

- output_folder : folder where to save the outputs

- pairs_file : npy file which contains the file name of pairs in form of dictionary where key is the controller file and value is the list of the handlers.

- bin_width : bin width for the hist plot

- corpus_split : split the corpus when calculating cos score for limited GPU memory


## Usage of ```metrics.py```

```
python metrics.py 
--positive_csv '/home/workspace/model_1/csvs/ctrl_hdlr/positive_query.csv' 
--negative_csv '/home/workspace/model_1/csvs/ctrl_hdlr/negative_query.csv' 
--threshold 500
```

#### Input
- CSV files containing information about the ranks for both positive and negative test sets.
- Threshold rank for the binary classification

#### Output
- Displays the assessments metrics

#### Parameters
- positive_csv : csv file with ranks collected using Positive input set (Q+)

- negative_csv : csv file with ranks collected using Negative input set (Q-)

- threshold : threshold rank for the binary classification

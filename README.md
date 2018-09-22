# TILES_Feature_Extraction

#### This is a repo for TILES project, topic is regarding extracting om feature for predicting ground truth label

### Prerequisites

Mainly used:

* Python3
* [Pandas](http://pandas.pydata.org/pandas-docs/version/0.15/index.html) -- `pip3 install pandas`
* [Numpy](http://www.numpy.org/) -- `pip3 install numpy`

### Recommended deploy file hierarchy

```
.
├── 
├── TILES_Feature_Extraction                    
│   ├── output
│   │
│   └── feat_extraction
|   │   ├── extract_feat.py        # Extract statitical feature of OMSignal with a defined hour window pior to survey
│   │
│   └── preprocessing
|   |   ├── preprocessing.py       # Extract recording start and end
|   |   ├── days_at_work.py        # Extract days at work
│   │
│   └── util
|       ├── files.py               # Get common files path
|       ├── load_data_basic.py     # Load basic information like participant id - user id mapping
| 
├── data                    
│   ├── keck_wave2
|       ├── 3_preprocessed_data
|       ├── ground_truth
|       ├── demographic
|       ├── participant_info
|       ├── id-mapping
```

### 1. Run preprocessing.py
### 2. Run extract_feat.py

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Tiantian Feng** 

**Feel free to contact me if you want to be a collaborator.**





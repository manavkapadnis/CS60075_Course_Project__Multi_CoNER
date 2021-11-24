# CS60075_Course_Project_Multi_CoNER


Group members:
- Manav Nitin Kapadnis (19EE30013)
- Sohan Patnaik (19ME30051)
- Kolla Ananta Raj (19EE30012)
- Mehul Gupta (19IE10020)

## Code Repository Organisation
```bash
├── Data/
|    ├── BN-Bangla
|    ├── EN-English
|    └── HI-Hindi
|    └── bn_dev_features.csv
|    └── bn_train_features.csv
|    └── en_dev_features.csv
|    └── en_train_features.csv
|    └── hn_dev_features.csv
|    └── hn_train_features.csv

├── Code With Outputs/
|    ├── Additional Features Implementation Code.ipynb
|    ├── Baseline Code.ipynb
|    └── Fine Tuning Dense Layers Code.ipynb
|    └── Transformers+BiLSTM-CRF Code.ipynb


├── src/
|    ├── data_reader.py
|    ├── global_var.py
|    └── main.py
|    └── metric.py
|    └── models.py
|    └── utils.py

├── README.md
```

## Obtaining the code and data

First, clone this repository
```
https://github.com/manavkapadnis/CS60075_Course_Project__Multi_CoNER.git

```

All the code and Data will get downloaded by the above command.
In order to reproduce our results,
- Open any choice of the IPython Notebook present in the 'Code with Outputs' folder
- In order to change the parameters for your running the notebook
- Just change the parameter settings in **Args Class** in the IPython Notebook and then do run all
- Provide correct path to the dataset and features.csv files for the language for which you are running the code
- The model will be saved after all the code cells get executed.


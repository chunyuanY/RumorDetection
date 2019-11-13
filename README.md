# Paper of the source codes released:
Chunyuan Yuan, Qianwen Ma, Wei Zhou, Jizhong Han, Songlin Hu. Jointly embedding the local and global relations of heterogeneous graph for rumor detection. In 19th IEEE International Conference on Data Mining, IEEE ICDM 2019.

# Dependencies:
Gensim==3.7.2

Jieba==0.39

Scikit-learn==0.21.2

Pytorch==1.1.0

# Datasets
The main directory contains the directories of Weibo dataset and two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- twitter15.train, twitter15.dev, and twitter15.test file: This files provide traing, development and test samples in a format like: 'source tweet ID \t source tweet content \t label'
  
- twitter15_graph.txt file: This file provides the source posts content of the trees in a format like: 'source tweet ID \t userID1:weight1 userID2:weight2 ...'  

These dastasets are preprocessed according to our requirement and original datasets can be available at https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0  (Twitter)  and http://alt.qcri.org/~wgao/data/rumdect.zip (Weibo).

If you want to preprocess the dataset by youself, you can use the word2vec used in our work. The pretrained word2vec can be available at https://drive.google.com/drive/folders/1IMOJCyolpYtoflEqQsj3jn5BYnaRhsiY?usp=sharing.


# Reproduce the experimental results:
1. create an empty directory: checkpoint/
2. run script run.py 


## Citation
If you find this code useful in your research, please cite our paper:
```
@inproceedings{rumor_yuan_2019,
  title={Jointly embedding the local and global relations of heterogeneous graph for rumor detection},
  author={Yuan, Chunyuan and Ma, Qianwen and Zhou, Wei and Han, Jizhong and Hu, Songlin},
  booktitle={The 19th IEEE International Conference on Data Mining},
  year={2019},
  organization={IEEE}
}
```


Datasets
========
The main directory contains the directories of two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- 'tree' sub-directory: This folder contains all the tree files, each of which corresponds to the tree structure given a source tweet whose file name is indicated by the source tweet ID. In the tree file, each line represents an edge given in the following format:
  ** parent node -> child node
  ** Each node is given as a tuple: ['uid', 'tweet ID', 'post time delay (in minutes)']
  
- label.txt file: This file provides the ground-truth labels of the trees in a format like:
  ** 'label:source tweet ID'
  
- source_tweets.txt file: This file provides the source posts content of the trees in a format like:
  ** 'source tweet ID \t source tweet content'  

Note that constrained by the terms of Twitter service, we cannot contain the content of the rest of the tweets. Data users can obtain the sepcifics based on the provided tweet IDs and uids by their own.

Feature description
===================
- Content features: uni-grams, bi-grams (presence/absence, binary)
- User features:
  ** # of followers
  ** # of friends
  ** ratio of followers and friends
  ** # of history tweets
  ** registration time (year)
  ** whether a verify account or not

References
==========
Substantial number of source tweets and their correspoding propagations trees were extracted based on two reference datasets described and released by the following works:

- twitter15: 
	@inproceedings{liu2015real,
	  title={Real-time Rumor Debunking on Twitter},
	  author={Liu, Xiaomo and Nourbakhsh, Armineh and Li, Quanzhi and Fang, Rui and Shah, Sameena},
	  booktitle={Proceedings of the 24th ACM International on Conference on Information and Knowledge Management},
	  pages={1867--1870},
	  year={2015}
	}

- twitter16:
	@inproceedings{ma2016detecting,
	  title={Detecting Rumors from Microblogs with Recurrent Neural Networks},
	  author={Ma, Jing and Gao, Wei and Mitra, Prasenjit and Kwon, Sejeong and Jansen, Bernard J. and Wong, Kam-Fai and Meeyoung, Cha},
	  booktitle={The 25th International Joint Conference on Artificial Intelligence},
	  year={2016},
	  organization={AAAI}
	}
	
If you use the datasets released hereby, please also cite:

	@inproceedings{ma2017detect,
	  title={Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning},
	  author={Ma, Jing and Gao, Wei and Wong, Kam-Fai},
	  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      volume={1},
      pages={708--717},
      year={2017}
	}



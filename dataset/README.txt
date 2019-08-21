Datasets
========
The main directory contains the directories of Weibo dataset and two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- twitter15.train, twitter15.dev, and twitter15.test file: This files provide traing, development and test samples in a format like:
  ** 'source tweet ID \t source tweet content \t label'
  
- twitter15_graph.txt file: This file provides the source posts content of the trees in a format like:
  ** 'source tweet ID \t userID1:weight1 userID2:weight2 ...'  
These dastasets are preprecessed according to our requirement and original datasets can be available at https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0  (Twitter)  and 


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



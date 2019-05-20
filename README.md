# Recommender-System
This is a course project for DS-GA-1004, Big Data, Master of Science in Data Science in New York University

The project target is to build a recommender system based on music tracks and users.

The project is completed by the team, whose members are Junge Zhang, Jing Qian and  Yuhong Zhu.

The original data is given by the class instructor Dr. Brian McFee.

The recommender system is build based upon the ASL implicit model within the Spark environment.

There are 8 total python files, notably for three parts of the project.

Part 1: basic model implementation, includes train.py, valiation.py and test.py.

Part 2: alternative model formulation by dropping low counts, log compression, combination of dropping low counts and log compression, includes train_count.py, train_log.py and train_logcount.py.

Part3: error analysis by finding over-represented items and under-represented items, includes error_analysis.py.

All the findings and summaries are included in the final report PDF file.

# EH2745-Assignment-2-Jorge-Ballesteros-
In this assignment we want to implement machine learning and the power system modeling techniques in order to create a system that can predict depending on the values of the nominal voltages and angles at the buses if we have a high load, low load, generator disconnected or line disconnected. In this way we can recognize much faster after running our load flow calculations in which situation our system is.

The first step in this code is to generate the system, which is comprised of:
- 9 Buses
  - 2 Generator Buses
  - 6 Load Buses
  - 1 Slack Bus
 - Lines of 10 km length each

Secondly a data source profile is created, for this we are considering 6 cases:
- High Load: Set the P and Q for each load to a value higher than the default, and add some noise with a standard deviation of about 5-10% of the nominal values.
- Low Load: Set the P and Q for each load to a value smaller than the default, and add some noise with a standard deviation of about 5-10% of the nominal values. 
- Generator Disconnected: Disconnect the generator at bus 3. 
  - High Load case
  - Low Load Case
- Line Disconnected: Disconnect the line between bus 5 and 6. 
  - High Load case
  - Low Load Case
  
After creating all the data profiles we merge them in a sigle file which will become our dataset. In order to have a clearer data the buses voltage values and angles, with the exception of the salck bus, are normalized. Doing this we will get a much clearer distinction between our 6 different cases in the dataset. We will use this dataset in the kmeans algorithm in order to cluster the data.

Once we have created this dataset, this last one have to be splitted into 2:
- Training dataset: with the 80% of the original dataset.
- Testing dataset: with the remainig 20% of the original dataset.

To finish, a knn algorithm is implemented in order to train our test set which will predict in which operating state (hihg or low load, generator or line disconnected, etc). A function that calculates the accuraccy of of the predicted results is also implemented, this last one basically takes the original dataset and compares with the test data to see if the predicted operating state mactches with the operating state in the original dataset.



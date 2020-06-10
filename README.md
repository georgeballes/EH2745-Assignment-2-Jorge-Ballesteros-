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
  
After creating all the data profiles we merge them in a sigle file which will become our dataset. In order to have a clearer data the voltage values and angles are normalize, doing this we will get a much clearer distinction between our 6 different cases in the dataset.

 


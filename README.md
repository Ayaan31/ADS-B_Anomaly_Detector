# ADS-B_Anomaly_Detector
This is my Master's Project for Rensselaer Polytechnic Institute.

The primary goal of this project is to research, design, implement, and evaluate a system capable of identifying any anomalous data within real-time ADS-B signals streams. This system will serve as a proof-of-concept for enhancing the integrity and reliability of ADS-B surveillance data. The process of achieving these goals will be through:
1.	ADS-B Data Acquisition and Parsing
      * Publicly available ADS-B data can be gathered through many resources like OpenSky Network. This data will be gathered from specific regions of the world
      * Data will be parsed and cleaned in a format suitable for analysis

2.	Anomaly Detection Model(s)
      * Research and create initial models using methods like KNN and then move forward with Machine Learning algorithms (Isolation Forest, Spatial Clustering, etc.)

      * Historical ADS-B data can be used to train the data on what normal flight behavior looks like

3.	Implementation and Evaluation
      * Create a pipeline to feed the model real time data to measure performance using a set of metrics
      * Develop visualizations to display aircrafts tracks and flag detected anomalies

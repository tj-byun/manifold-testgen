# aahaa-testgen

Abhi's document. Outdated.

Design
------

* Test Input Sampler
  * Initializes parameter variables with bound constraints
  * Samples combination of parameters with a given strategy:
    * Grid sampling: divide the range of parameters into equal portions and 
      sample n from each.
    * Random sampling: sample a combination randomly
    * Bayesian optimization: optimize a Gaussian Process Kernel to approximate 
      the sentiment measure over the (test input) parameter space. Iteratively 
      optimize towards a set of parameter combinations that are likely to 
      maximize the sentiment measure.
* Image Collector
  * given a set of parameter combinations, query XPlane to obtain corresponding
    images
* Sentiment Calculator
  * Runs images on neural network and get the sentiments (uncertainty)



Structure
---------
* common_utils
    * compute_metadata: a tool to compute metadata values from raw labels recorded in XPlane
    * df_computations: A set of tools for computing and storing extra information about image sets via dataframes
    * dataset: classes to abstract dataset-related tasks
    * models: classes that add extra functionality on top of the keras Model class
    * taxinet: a TaxiNet dataset class
* error_model
    * a neural network that takes metadata as input and predicts the error in CTE and HE
* metadata_model
    * a neural network that predicts the metadata of an XPlane image in the TaxiNet scenario
* taxinet_utils
    * autorecorder: automatic grid-based collection of image sets for TaxiNet
    * run_traj: a tool to run trajectories generated by traj_builder
    * traj_builder: a tool to generate chirp trajectories along a runway
    * xplanerecorder: a tool to record XPlane data from the current run in real-time
* testgen
    * bugfinder: a bulky class with methods for implementing particle swarm optimization, finding high uncertainty
                    points, capturing those points in XPlane, and computing predictions and errors on them
    * parameters: abstraction for implementing metadata
    * testgenerator: a looped sampler and evaluator of the TaxiNet scenario
* trajectory
    * mov_avg: a library for calculating moving averages over datasets
    * traj_generation: runs neural network controlled TaxiNet runs from selected starting points
    * convergence_times: computes the distribution of the convergence of trajectories to the centerline
* uncertainty_model
    * a neural network that takes in metadata as input and predicts the uncertainty of a TaxiNet prediction
    * uncertainty_training_data: creates training data for the metadata-uncertainty model
* xplane_utils: a set of utilities that interface with XPlane for data collection and plane control
    * AptParser: a tool that retrieves JSON-formatted information about runways
    * neural_net_controller: a tool to steer the a taxiing plane via neural net predictions
    * xplane: a set of classes for collecting and converting XPlane data
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include "..\..\C++_Library\Eigen\Dense"

using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data, 
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

  vector<string> possible_labels = {"left","keep","right"};

   vector<double> means_left;
   vector<double> means_keep;
   vector<double> means_right;
   vector<double> stds_left;
   vector<double> stds_keep;
   vector<double> stds_right;
       
};

#endif  // CLASSIFIER_H
#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using std::string;
using std::vector;

double gaussian_prob(double obs, double mu, double sig){
    double num = pow((obs - mu),2);
    double denum = 2 * sig * 2.0;
    double norm = 1.0 / sqrt(pow(2.0 * M_PI * sig,2));
    return norm * exp(-num/denum);
}

double mean_cal(vector<double> &v){
    double sum = 0;
    for (int i = 0; i < v.size(); i++){
        sum += v[i];
    }

    return sum/v.size();
}

double std_cal(vector<double> &v){
    double mu = mean_cal(v);
    double val = 0;

    for (int i = 0; i < v.size(); i++){
        val += pow(v[i] - mu, 2);
    }

    val /= v.size();

    return sqrt(val);
}

// Initializes GNB
GNB::GNB() {
  /**
   * TODO: Initialize GNB, if necessary. May depend on your implementation.
   */
    //vector<std::string> classes = {"left", "keep", "right"};
  
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * TODO: Implement the training function for your classifier.
   */



  int num_vars = 4;

  vector< vector<double >> total_left;
  vector< vector<double >> total_keep;
  vector< vector<double >> total_right;

  //Debug
  std::ofstream outputfile1("debug\\data.txt");

  for (int i = 0; i < labels.size(); i++){
      if(labels[i] == "left"){
        total_left.push_back(data[i]);
        outputfile1 << "left: " << data[i][0] << ", " << data[i][1] << ", " << data[i][2] << ", " << data[i][3] << std::endl;
      } else if(labels[i] == "keep"){
        total_keep.push_back(data[i]);
        outputfile1 << "keep: " << data[i][0] << ", " << data[i][1] << ", " << data[i][2] << ", " << data[i][3] << std::endl;
      }else {
        total_right.push_back(data[i]);
        outputfile1 << "right:" << data[i][0] << ", " << data[i][1] << ", " << data[i][2] << ", " << data[i][3] << std::endl;
      }
  }
  
  //Debug
  outputfile1.close();     

    //Debug
    std::ofstream outputfile2("debug\\test_train.txt");

    vector<double >mean_left = {0, 0, 0, 0};
    vector<double >mean_keep = {0, 0, 0, 0};
    vector<double >mean_right = {0, 0, 0, 0};
    vector<double >std_left = {0, 0, 0, 0};
    vector<double >std_keep = {0, 0, 0, 0};
    vector<double >std_right = {0, 0, 0, 0};
     
    for (int i = 0; i < num_vars; i++){
        for (int j = 0; j < total_left.size(); j++){
            mean_left[i] += total_left[j][i];
        }
        for (int j = 0; j < total_keep.size(); j++){
            mean_keep[i] += total_keep[j][i];
        }
        for (int j = 0; j < total_right.size(); j++){
            mean_right[i] += total_right[j][i];
        }
        mean_left[i] /= total_left.size();
        mean_keep[i] /= total_keep.size();
        mean_right[i] /= total_right.size();

        means_left.push_back(mean_left[i]);
        means_keep.push_back(mean_keep[i]);
        means_right.push_back(mean_right[i]);
        
        for (int j = 0; j < total_left.size(); j++){
            std_left[i] += pow(total_left[j][i] - mean_left[i], 2);
        }
        for (int j = 0; j < total_left.size(); j++){
            std_keep[i] += pow(total_keep[j][i] - mean_keep[i], 2);
        }
        for (int j = 0; j < total_left.size(); j++){
            std_right[i] += pow(total_right[j][i] - mean_right[i], 2);
        }
        std_left[i] /= total_left.size();
        std_left[i] = sqrt(std_left[i]);
        std_keep[i] /= total_keep.size();
        std_keep[i] = sqrt(std_keep[i]);
        std_right[i] /= total_right.size();
        std_right[i] = sqrt(std_right[i]);  

        stds_left.push_back(std_left[i]);
        stds_keep.push_back(std_keep[i]);
        stds_right.push_back(std_right[i]);

        //int lenght = labels.size();
        //int left_len = total_left.size();
        //int keep_len = total_keep.size();
        //int right_len = total_right.size();
        //double test_left_0 = total_left[0][0];
        //double test_left_1 = total_left[0][1];
        //double test_left_2 = total_left[0][2];
        //double test_left_3 = total_left[0][3];


        //double test_mean_left = mean_left[0];
        //double test_mean_keep = mean_keep[0];
        //double test_mean_right = mean_right[0];
        //double std_mean_left = std_left[0];
        //double std_mean_keep = std_keep[0];
        //double std_mean_right = std_right[0];
 


        //Debug
        outputfile2 << "mean_left: " << mean_left[i] << std::endl;
        outputfile2 << "std_left:  " << std_left[i] << std::endl;
        outputfile2 << "mean_keep: " << mean_keep[i] << std::endl;
        outputfile2 << "std_keep:  " << std_keep[i] << std::endl;
        outputfile2 << "mean_right:" << mean_right[i] << std::endl;
        outputfile2 << "std_right: " << std_right[i] << std::endl << std::endl;
              
    }

    //Debug 
    outputfile2.close();   

}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * TODO: Complete this function to return your classifier's prediction
   */
    //Debug

    //Debug: Check whether the values of means, stds are correct. It looks OK. 
    //for (int i = 0; i < 4; i++){
    //    outputfile3 << means_left[i] << std::endl;
    //    outputfile3 << stds_left[i] << std::endl;
    //}
    //outputfile3.close(); 

   vector<vector<double >> probability = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    for (int j = 0; j < sample.size(); j++){
        //left
        //probability[0][j] = 1/(sqrt(2*M_PI*pow(stds_left[j],2)))
        //                    *exp(-(sample[j]-means_left[j]/(2*pow(stds_left[j],2))));
        probability[0][j] = gaussian_prob(sample[j], means_left[j], stds_left[j]);
        //keep
        //probability[1][j] = 1/(sqrt(2*M_PI*pow(stds_keep[j],2)))
        //                    *exp(-(sample[j]-means_keep[j]/(2*pow(stds_keep[j],2))));
        probability[1][j] = gaussian_prob(sample[j], means_keep[j], stds_keep[j]);
        //right
        //probability[2][j] = 1/(sqrt(2*M_PI*pow(stds_right[j],2)))
        //                    *exp(-(sample[j]-means_right[j]/(2*pow(stds_right[j],2))));
        probability[2][j] = gaussian_prob(sample[j], means_right[j], stds_right[j]);
   }

   vector<double > multipled = {1., 1., 1.} ;

    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 4; j++){
            multipled[i] *= probability[i][j];  

        }
    }

    int index = 0;
    double max = multipled[0];

    //Debug: max selection logic is OK. the calculation has some problem because always left is chosen..
    double multipled_0 = multipled[0];
    double multipled_1 = multipled[1];
    double multipled_2 = multipled[2];
       

    for (int i = 1; i<3; i++){
        if(multipled[i] > max){
            index = i;
            max = multipled[i];
        }
    }
  
  return this -> possible_labels[index];
}